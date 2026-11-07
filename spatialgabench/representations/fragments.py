"""Create an auditable fragment input shared by matched representations."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

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


def _read_in_tissue_barcodes(path: Path) -> set[str]:
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
    table["in_tissue"] = pd.to_numeric(table["in_tissue"], errors="raise")
    table = table.loc[table["in_tissue"] != 0]
    if table["barcode"].duplicated().any():
        raise ValueError("Spatial positions contain duplicate normalized in-tissue barcodes")
    return set(table["barcode"])


def _read_standard_chrom_sizes(path: Path) -> dict[str, int]:
    allowed = {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY"}
    sizes: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            chromosome, size, *_ = line.rstrip("\n").split("\t")
            if chromosome in allowed:
                sizes[chromosome] = int(size)
    if not sizes:
        raise ValueError(f"No standard chromosomes found in {path}")
    return sizes


def filter_fragments_for_representation(
    project_config: str | Path,
    study_id: str,
    fragments_path: str | Path,
    positions_path: str | Path,
    chrom_sizes_path: str | Path,
    output_path: str | Path,
    audit_path: str | Path,
) -> dict[str, object]:
    """Filter once so peak calling and matrix counting consume identical rows."""

    authorization = assert_development_study(project_config, study_id)
    fragments_path = Path(fragments_path)
    positions_path = Path(positions_path)
    chrom_sizes_path = Path(chrom_sizes_path)
    output_path = Path(output_path)
    audit_path = Path(audit_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    in_tissue_barcodes = _read_in_tissue_barcodes(positions_path)
    chrom_sizes = _read_standard_chrom_sizes(chrom_sizes_path)
    source_rows = 0
    included_rows = 0
    invalid_fragment_rows = 0
    excluded_barcode_rows = 0
    excluded_chromosome_rows = 0
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    opener = gzip.open if fragments_path.suffix == ".gz" else open

    with opener(fragments_path, "rt", encoding="utf-8") as source, temporary_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as target:
                for line in source:
                    if line.startswith("#"):
                        continue
                    source_rows += 1
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
                    chromosome = fields[0]
                    if chromosome not in chrom_sizes:
                        excluded_chromosome_rows += 1
                        continue
                    if start < 0 or end <= start or end > chrom_sizes[chromosome]:
                        invalid_fragment_rows += 1
                        continue
                    barcode = _normalize_barcode(fields[3])
                    if barcode not in in_tissue_barcodes:
                        excluded_barcode_rows += 1
                        continue
                    target.write(f"{chromosome}\t{start}\t{end}\t{barcode}\t1\n")
                    included_rows += 1
    temporary_path.replace(output_path)

    result: dict[str, object] = {
        "authorization": authorization,
        "study_id": study_id,
        "source_fragments": str(fragments_path),
        "source_fragments_sha256": _sha256(fragments_path),
        "output_fragments": str(output_path),
        "output_fragments_sha256": _sha256(output_path),
        "positions": str(positions_path),
        "chrom_sizes": str(chrom_sizes_path),
        "in_tissue_barcode_rows": len(in_tissue_barcodes),
        "source_fragment_rows": source_rows,
        "included_fragment_rows": included_rows,
        "invalid_fragment_rows": invalid_fragment_rows,
        "excluded_barcode_rows": excluded_barcode_rows,
        "excluded_chromosome_rows": excluded_chromosome_rows,
        "read_support_policy": "fifth_column_forced_to_one_per_included_source_row",
        "filter_contract": "valid_interval_and_in_tissue_barcode_and_standard_chromosome",
        "status": "pass" if included_rows > 0 and invalid_fragment_rows == 0 else "fail",
    }
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat_result = {key: value for key, value in result.items() if not isinstance(value, dict)}
    pd.DataFrame([flat_result]).to_csv(audit_path.with_suffix(".tsv"), sep="\t", index=False)
    return result


def validate_filtered_fragment_audit(
    audit_path: str | Path,
    fragments_path: str | Path,
    observed_sha256: str | None = None,
) -> dict[str, object]:
    """Validate that a filtered fragment artifact still matches its audit."""

    audit_path = Path(audit_path)
    fragments_path = Path(fragments_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError(f"Filtered fragment audit is not passing: {audit_path}")
    audited_output = Path(str(payload.get("output_fragments", "")))
    if audited_output.resolve() != fragments_path.resolve():
        raise ValueError("Filtered fragment audit output path does not match matrix input")
    observed_sha256 = observed_sha256 or _sha256(fragments_path)
    if payload.get("output_fragments_sha256") != observed_sha256:
        raise ValueError("Filtered fragment checksum does not match its audit")
    return payload
