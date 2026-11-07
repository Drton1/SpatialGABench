"""Build peak and fixed-bin matrices from one frozen fragment input."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .fragments import validate_filtered_fragment_audit
from .guard import assert_development_study


def _normalize_barcode(value: str) -> str:
    value = value.strip().strip('"')
    return value[:-2] if value.endswith("-1") else value


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_positions(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    has_header = first_line.lower().startswith("barcode,")
    columns = ["barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"]
    table = pd.read_csv(
        path,
        compression="gzip" if path.suffix == ".gz" else None,
        header=0 if has_header else None,
        names=None if has_header else columns,
    )
    if has_header:
        table = table.iloc[:, :6].copy()
        table.columns = columns
    table["barcode"] = table["barcode"].map(lambda value: _normalize_barcode(str(value)))
    for column in columns[1:]:
        table[column] = pd.to_numeric(table[column], errors="raise")
    table = table.loc[table["in_tissue"] != 0].copy()
    if table["barcode"].duplicated().any():
        raise ValueError("Spatial positions contain duplicate normalized barcodes")
    table = table.set_index("barcode")
    return table


def _read_standard_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    allowed = {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY"}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            chromosome, size, *_ = line.rstrip("\n").split("\t")
            if chromosome in allowed:
                sizes[chromosome] = int(size)
    if not sizes:
        raise ValueError(f"No standard chromosomes found in {path}")
    return sizes


def _make_bin_features(chrom_sizes: dict[str, int], bin_size: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for chromosome, chromosome_size in chrom_sizes.items():
        for start in range(0, chromosome_size, bin_size):
            end = min(start + bin_size, chromosome_size)
            records.append(
                {
                    "feature_id": f"{chromosome}:{start}-{end}",
                    "chromosome": chromosome,
                    "start": start,
                    "end": end,
                    "width": end - start,
                }
            )
    return pd.DataFrame(records).set_index("feature_id")


def _read_peak_features(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", header=None, comment="#")
    if table.shape[1] < 3:
        raise ValueError(f"Peak BED must have at least three columns: {path}")
    table = table.iloc[:, :7].copy()
    table.columns = ["chromosome", "start", "end", "name", "qscore", "signal", "summit"][: table.shape[1]]
    table["start"] = pd.to_numeric(table["start"], errors="raise").astype(int)
    table["end"] = pd.to_numeric(table["end"], errors="raise").astype(int)
    table["width"] = table["end"] - table["start"]
    if (table["width"] <= 0).any():
        raise ValueError("Peak BED contains non-positive intervals")
    table["feature_id"] = [
        f"{chromosome}:{start}-{end}"
        for chromosome, start, end in table[["chromosome", "start", "end"]].itertuples(index=False)
    ]
    if table["feature_id"].duplicated().any():
        raise ValueError("Peak BED contains duplicate intervals")
    return table.set_index("feature_id")


@dataclass
class FeatureLookup:
    representation_id: str
    table: pd.DataFrame
    bin_size: int | None = None

    def __post_init__(self) -> None:
        self._chromosome_data: dict[str, dict[str, np.ndarray | int]] = {}
        for chromosome, chromosome_table in self.table.groupby("chromosome", sort=False):
            positions = self.table.index.get_indexer(chromosome_table.index)
            self._chromosome_data[str(chromosome)] = {
                "starts": chromosome_table["start"].to_numpy(dtype=np.int64),
                "ends": chromosome_table["end"].to_numpy(dtype=np.int64),
                "columns": positions.astype(np.int64),
            }

    def assign(self, chromosome: str, midpoints: np.ndarray) -> np.ndarray:
        result = np.full(len(midpoints), -1, dtype=np.int64)
        if chromosome not in self._chromosome_data:
            return result
        data = self._chromosome_data[chromosome]
        starts = np.asarray(data["starts"])
        ends = np.asarray(data["ends"])
        columns = np.asarray(data["columns"])
        indexes = np.searchsorted(starts, midpoints, side="right") - 1
        valid_index = indexes >= 0
        candidate_indexes = indexes[valid_index]
        valid_interval = midpoints[valid_index] < ends[candidate_indexes]
        valid_positions = np.flatnonzero(valid_index)[valid_interval]
        result[valid_positions] = columns[indexes[valid_positions]]
        return result


def _flush_chunk(
    row_values: list[int],
    chromosome_values: list[str],
    midpoint_values: list[int],
    matrices: dict[str, sparse.csr_matrix],
    lookups: dict[str, FeatureLookup],
    assigned_counts: dict[str, int],
) -> None:
    if not row_values:
        return
    rows = np.asarray(row_values, dtype=np.int32)
    chromosomes = np.asarray(chromosome_values, dtype=object)
    midpoints = np.asarray(midpoint_values, dtype=np.int64)
    for representation_id, lookup in lookups.items():
        assigned_rows: list[np.ndarray] = []
        assigned_columns: list[np.ndarray] = []
        for chromosome in np.unique(chromosomes):
            positions = np.flatnonzero(chromosomes == chromosome)
            columns = lookup.assign(str(chromosome), midpoints[positions])
            valid = columns >= 0
            if valid.any():
                assigned_rows.append(rows[positions][valid])
                assigned_columns.append(columns[valid].astype(np.int32))
        if not assigned_rows:
            continue
        chunk_rows = np.concatenate(assigned_rows)
        chunk_columns = np.concatenate(assigned_columns)
        chunk = sparse.coo_matrix(
            (np.ones(len(chunk_rows), dtype=np.int32), (chunk_rows, chunk_columns)),
            shape=matrices[representation_id].shape,
            dtype=np.int32,
        ).tocsr()
        matrices[representation_id] = matrices[representation_id] + chunk
        assigned_counts[representation_id] += len(chunk_rows)


def build_matched_representations(
    project_config: str | Path,
    study_id: str,
    dataset_id: str,
    sample_id: str,
    fragments_path: str | Path,
    positions_path: str | Path,
    chrom_sizes_path: str | Path,
    peaks_path: str | Path,
    genome_build: str,
    bin_sizes: list[int],
    output_dir: str | Path,
    chunk_size: int = 500_000,
    fragment_filter_audit_path: str | Path | None = None,
) -> dict[str, Path]:
    """Build all matched matrices in one fragment pass and validate count contracts."""

    authorization = assert_development_study(project_config, study_id)
    fragments_path = Path(fragments_path)
    positions_path = Path(positions_path)
    chrom_sizes_path = Path(chrom_sizes_path)
    peaks_path = Path(peaks_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fragment_sha256 = _sha256(fragments_path)
    fragment_filter_audit = None
    fragment_filter_audit_sha256 = None
    if fragment_filter_audit_path is not None:
        fragment_filter_audit_path = Path(fragment_filter_audit_path)
        fragment_filter_audit = validate_filtered_fragment_audit(
            fragment_filter_audit_path,
            fragments_path,
            observed_sha256=fragment_sha256,
        )
        fragment_filter_audit_sha256 = _sha256(fragment_filter_audit_path)
    counting_unit = (
        "one_filtered_fragment_row_midpoint"
        if fragment_filter_audit is not None
        else "one_eligible_source_fragment_row_midpoint"
    )
    read_support_policy = (
        "fifth_column_forced_to_one_by_verified_filter"
        if fragment_filter_audit is not None
        else "fifth_column_ignored_each_eligible_source_row_counts_once"
    )
    positions = _read_positions(positions_path)
    barcodes = positions.index.astype(str).tolist()
    barcode_to_row = {barcode: index for index, barcode in enumerate(barcodes)}
    chrom_sizes = _read_standard_chrom_sizes(chrom_sizes_path)

    feature_tables: dict[str, pd.DataFrame] = {}
    for bin_size in bin_sizes:
        if bin_size <= 0:
            raise ValueError("Bin sizes must be positive")
        representation_id = f"fixed_bin_{bin_size // 1000}kb"
        feature_tables[representation_id] = _make_bin_features(chrom_sizes, bin_size)
    peak_representation_id = "macs3_summit_501bp"
    feature_tables[peak_representation_id] = _read_peak_features(peaks_path)
    lookups = {
        representation_id: FeatureLookup(representation_id, table)
        for representation_id, table in feature_tables.items()
    }
    matrices = {
        representation_id: sparse.csr_matrix((len(barcodes), len(table)), dtype=np.int32)
        for representation_id, table in feature_tables.items()
    }
    assigned_counts = {representation_id: 0 for representation_id in feature_tables}

    fragment_rows = 0
    eligible_fragment_rows = 0
    invalid_fragment_rows = 0
    excluded_barcode_rows = 0
    excluded_chromosome_rows = 0
    row_values: list[int] = []
    chromosome_values: list[str] = []
    midpoint_values: list[int] = []
    opener = gzip.open if fragments_path.suffix == ".gz" else open
    with opener(fragments_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fragment_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                invalid_fragment_rows += 1
                continue
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                invalid_fragment_rows += 1
                continue
            if start < 0 or end <= start:
                invalid_fragment_rows += 1
                continue
            chromosome = fields[0]
            if chromosome not in chrom_sizes:
                excluded_chromosome_rows += 1
                continue
            barcode = _normalize_barcode(fields[3])
            if barcode not in barcode_to_row:
                excluded_barcode_rows += 1
                continue
            midpoint = (start + end) // 2
            if midpoint >= chrom_sizes[chromosome]:
                invalid_fragment_rows += 1
                continue
            eligible_fragment_rows += 1
            row_values.append(barcode_to_row[barcode])
            chromosome_values.append(chromosome)
            midpoint_values.append(midpoint)
            if len(row_values) >= chunk_size:
                _flush_chunk(row_values, chromosome_values, midpoint_values, matrices, lookups, assigned_counts)
                row_values.clear()
                chromosome_values.clear()
                midpoint_values.clear()
    _flush_chunk(row_values, chromosome_values, midpoint_values, matrices, lookups, assigned_counts)

    obs = positions.copy()
    obs.index.name = "barcode"
    spatial = obs[["pixel_col", "pixel_row"]].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    outputs: dict[str, Path] = {}
    validation_failures: list[str] = []
    if fragment_filter_audit is not None:
        if fragment_rows != int(fragment_filter_audit["included_fragment_rows"]):
            validation_failures.append("filtered fragment row count does not match its audit")
        if invalid_fragment_rows or excluded_barcode_rows or excluded_chromosome_rows:
            validation_failures.append("filtered fragment input still contains ineligible rows")
    for representation_id, matrix in matrices.items():
        matrix.sum_duplicates()
        matrix.sort_indices()
        matrix_sum = int(matrix.sum())
        if matrix_sum != assigned_counts[representation_id]:
            validation_failures.append(f"{representation_id}: matrix sum does not match assigned fragments")
        if matrix.data.size and (matrix.data < 0).any():
            validation_failures.append(f"{representation_id}: negative matrix entries")
        table = feature_tables[representation_id].copy()
        table["representation_id"] = representation_id
        input_track = "B" if representation_id.startswith("fixed_bin") else "P"
        adata = ad.AnnData(X=matrix, obs=obs.copy(), var=table)
        adata.obsm["spatial"] = spatial.copy()
        adata.uns.update(
            {
                "study_id": study_id,
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "representation_id": representation_id,
                "representation_source": "same_filtered_fragments",
                "input_track": input_track,
                "comparison_scope": "matched_fragment_derived",
                "genome_build": genome_build,
                "counting_unit": counting_unit,
                "fragment_read_support_policy": read_support_policy,
                "source_fragments": str(fragments_path),
                "source_fragments_sha256": fragment_sha256,
                "fragment_filter_audit": str(fragment_filter_audit_path or ""),
                "fragment_filter_audit_sha256": fragment_filter_audit_sha256 or "",
                "original_source_fragments": (
                    str(fragment_filter_audit["source_fragments"]) if fragment_filter_audit else str(fragments_path)
                ),
                "original_source_fragments_sha256": (
                    str(fragment_filter_audit["source_fragments_sha256"])
                    if fragment_filter_audit
                    else fragment_sha256
                ),
                "construction_leakage_inputs": "none",
            }
        )
        output_path = output_dir / f"{dataset_id}__{representation_id}.h5ad"
        adata.write_h5ad(output_path, compression="gzip")
        outputs[representation_id] = output_path
        rows.append(
            {
                "study_id": study_id,
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "representation_id": representation_id,
                "representation_source": "same_filtered_fragments",
                "input_track": input_track,
                "comparison_scope": "matched_fragment_derived",
                "counting_unit": counting_unit,
                "n_observations": matrix.shape[0],
                "n_features": matrix.shape[1],
                "matrix_nonzero_entries": matrix.nnz,
                "matrix_total_counts": matrix_sum,
                "assigned_fragment_rows": assigned_counts[representation_id],
                "eligible_fragment_fraction_assigned": _safe_fraction(
                    assigned_counts[representation_id], eligible_fragment_rows
                ),
                "output_path": str(output_path),
                "output_sha256": _sha256(output_path),
                "fragment_filter_audit": str(fragment_filter_audit_path or ""),
                "fragment_filter_audit_sha256": fragment_filter_audit_sha256 or "",
                "status": "pass" if matrix_sum == assigned_counts[representation_id] else "fail",
            }
        )

    manifest_path = output_dir / "representation_manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)
    validation = {
        "authorization": authorization,
        "study_id": study_id,
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "source_fragments": str(fragments_path),
        "source_fragments_sha256": fragment_sha256,
        "fragment_filter_audit": str(fragment_filter_audit_path or ""),
        "fragment_filter_audit_sha256": fragment_filter_audit_sha256 or "",
        "fragment_filter_contract_verified": fragment_filter_audit is not None,
        "original_source_fragments": (
            str(fragment_filter_audit["source_fragments"]) if fragment_filter_audit else str(fragments_path)
        ),
        "original_source_fragments_sha256": (
            str(fragment_filter_audit["source_fragments_sha256"])
            if fragment_filter_audit
            else fragment_sha256
        ),
        "fragment_rows": fragment_rows,
        "eligible_fragment_rows": eligible_fragment_rows,
        "invalid_fragment_rows": invalid_fragment_rows,
        "excluded_barcode_rows": excluded_barcode_rows,
        "excluded_chromosome_rows": excluded_chromosome_rows,
        "barcode_rows": len(barcodes),
        "representations": len(rows),
        "validation_failures": validation_failures,
        "status": "pass" if not validation_failures and invalid_fragment_rows == 0 else "fail",
    }
    validation_path = output_dir / "matched_representation_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path
    outputs["validation"] = validation_path
    return outputs


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
