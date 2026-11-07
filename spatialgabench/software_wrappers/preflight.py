"""Deterministic real-data fixtures for G4 software contract preflight."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import inspect
import json
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd
import pysam
from scipy import sparse

from spatialgabench.representations import assert_development_study


FORBIDDEN_OBSERVATION_TOKENS = {
    "cluster",
    "disease",
    "label",
    "marker",
    "pathology",
    "rna",
    "truth",
}

G4_PREFLIGHT_PANEL = {
    "mouse": {
        "fixture": "results/v1/g4/preflight/fixtures/mouse_gse278007_ffpe_s4_rep1",
        "genome_build": "mm10",
    },
    "human": {
        "fixture": "results/v1/g4/preflight/fixtures/human_gse278007_melanoma_50um",
        "genome_build": "hg38",
    },
}
G4_PREFLIGHT_METHODS = {
    "snapatac2": "snapatac2_geneactivity",
    "signac": "signac_geneactivity",
    "archr": "archr_genescorematrix",
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_barcode(value: object) -> str:
    barcode = str(value).strip().strip('"')
    return barcode[:-2] if barcode.endswith("-1") else barcode


def _read_positions(path: Path) -> pd.DataFrame:
    first_line = path.open(encoding="utf-8").readline().strip()
    has_header = first_line.lower().startswith("barcode,")
    columns = ["barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"]
    table = pd.read_csv(path, header=0 if has_header else None, names=None if has_header else columns)
    if has_header:
        table = table.iloc[:, :6].copy()
        table.columns = columns
    table["barcode"] = table["barcode"].map(_normalize_barcode)
    for column in columns[1:]:
        table[column] = pd.to_numeric(table[column], errors="raise")
    table = table.loc[table["in_tissue"] != 0].copy()
    if table["barcode"].duplicated().any():
        raise ValueError("Spatial positions contain duplicate normalized in-tissue barcodes")
    return table.set_index("barcode")


def _read_canonical_chrom_sizes(path: Path) -> dict[str, int]:
    canonical = {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY"}
    sizes: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2 and fields[0] in canonical:
                sizes[fields[0]] = int(fields[1])
    if not sizes:
        raise ValueError(f"No canonical chromosomes found in {path}")
    return sizes


def _select_barcodes(barcodes: list[str], n_observations: int, seed: int) -> list[str]:
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    if len(barcodes) < n_observations:
        raise ValueError(f"Requested {n_observations} observations but only {len(barcodes)} are available")
    return sorted(
        barcodes,
        key=lambda barcode: (
            hashlib.sha256(f"{seed}\0{barcode}".encode()).hexdigest(),
            barcode,
        ),
    )[:n_observations]


def _normalize_gtf(
    source_path: Path,
    output_path: Path,
    chrom_sizes: dict[str, int],
) -> dict[str, int]:
    opener = gzip.open if source_path.suffix == ".gz" else open
    records = 0
    seqname_prefix_additions = 0
    biotype_additions = 0
    gene_name_additions = 0
    with opener(source_path, "rt", encoding="utf-8") as source, output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as target:
                for line in source:
                    if line.startswith("#"):
                        target.write(line)
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 9:
                        continue
                    source_seqname = fields[0]
                    normalized_seqname = source_seqname
                    if normalized_seqname not in chrom_sizes and f"chr{source_seqname}" in chrom_sizes:
                        normalized_seqname = f"chr{source_seqname}"
                        seqname_prefix_additions += 1
                    if normalized_seqname not in chrom_sizes:
                        continue
                    fields[0] = normalized_seqname
                    attributes = fields[8]
                    if "gene_biotype " not in attributes:
                        match = re.search(r'gene_type "([^"]+)"', attributes)
                        if match is not None:
                            attributes = attributes.rstrip().rstrip(";") + f'; gene_biotype "{match.group(1)}";'
                            biotype_additions += 1
                    if "gene_name " not in attributes:
                        match = re.search(r'gene_id "([^"]+)"', attributes)
                        if match is not None:
                            attributes = attributes.rstrip().rstrip(";") + f'; gene_name "{match.group(1)}";'
                            gene_name_additions += 1
                    fields[8] = attributes
                    target.write("\t".join(fields) + "\n")
                    records += 1
    if records == 0:
        raise ValueError(f"Normalized GTF contains no canonical records: {source_path}")
    return {
        "annotation_records": records,
        "seqname_prefix_additions": seqname_prefix_additions,
        "gene_biotype_additions": biotype_additions,
        "gene_name_additions": gene_name_additions,
    }


def build_real_data_contract_fixture(
    *,
    project_config: str | Path,
    study_id: str,
    dataset_id: str,
    sample_id: str,
    genome_build: str,
    fragments_path: str | Path,
    positions_path: str | Path,
    gene_annotation_path: str | Path,
    chrom_sizes_path: str | Path,
    output_dir: str | Path,
    n_observations: int = 64,
    seed: int = 4160,
    bin_size: int = 10_000,
) -> dict[str, Path]:
    """Build a small accessibility-only fixture from an authorized development sample."""

    if genome_build not in {"mm10", "hg38"}:
        raise ValueError(f"Unsupported preflight genome build: {genome_build}")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    authorization = assert_development_study(project_config, study_id)
    fragments_path = Path(fragments_path)
    positions_path = Path(positions_path)
    gene_annotation_path = Path(gene_annotation_path)
    chrom_sizes_path = Path(chrom_sizes_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = _read_positions(positions_path)
    selected_barcodes = _select_barcodes(positions.index.astype(str).tolist(), n_observations, seed)
    selected_set = set(selected_barcodes)
    selected_positions = positions.loc[selected_barcodes].copy()
    chrom_sizes = _read_canonical_chrom_sizes(chrom_sizes_path)
    chromosome_order = {chromosome: index for index, chromosome in enumerate(chrom_sizes)}

    plain_fragments = output_dir / "preflight.fragments.tsv"
    fragments_gz = output_dir / "preflight.fragments.tsv.gz"
    fragments_index = Path(str(fragments_gz) + ".tbi")
    active_bin_counts: Counter[tuple[str, str, int]] = Counter()
    fragment_rows_by_barcode: Counter[str] = Counter()
    source_fragment_rows = 0
    selected_fragment_rows = 0
    invalid_selected_rows = 0
    excluded_noncanonical_rows = 0
    opener = gzip.open if fragments_path.suffix == ".gz" else open
    with opener(fragments_path, "rt", encoding="utf-8") as source, plain_fragments.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if line.startswith("#"):
                continue
            source_fragment_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            barcode = _normalize_barcode(fields[3])
            if barcode not in selected_set:
                continue
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                invalid_selected_rows += 1
                continue
            chromosome = fields[0]
            if chromosome not in chrom_sizes:
                excluded_noncanonical_rows += 1
                continue
            if start < 0 or end <= start or end > chrom_sizes[chromosome]:
                invalid_selected_rows += 1
                continue
            midpoint = (start + end) // 2
            bin_start = (midpoint // bin_size) * bin_size
            active_bin_counts[(barcode, chromosome, bin_start)] += 1
            fragment_rows_by_barcode[barcode] += 1
            target.write(f"{chromosome}\t{start}\t{end}\t{barcode}\t1\n")
            selected_fragment_rows += 1
    missing_barcodes = sorted(selected_set - set(fragment_rows_by_barcode))
    if missing_barcodes:
        raise ValueError(f"Selected observations lack canonical fragments: {missing_barcodes}")
    if selected_fragment_rows == 0 or invalid_selected_rows:
        raise ValueError(
            f"Invalid selected fragment contract: rows={selected_fragment_rows}, invalid={invalid_selected_rows}"
        )
    pysam.tabix_compress(str(plain_fragments), str(fragments_gz), force=True)
    pysam.tabix_index(str(fragments_gz), preset="bed", force=True)
    plain_fragments.unlink()

    active_features = sorted(
        {(chromosome, start) for _, chromosome, start in active_bin_counts},
        key=lambda value: (chromosome_order[value[0]], value[1]),
    )
    feature_to_column = {feature: index for index, feature in enumerate(active_features)}
    barcode_to_row = {barcode: index for index, barcode in enumerate(selected_barcodes)}
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[int] = []
    for (barcode, chromosome, start), count in sorted(active_bin_counts.items()):
        matrix_rows.append(barcode_to_row[barcode])
        matrix_columns.append(feature_to_column[(chromosome, start)])
        matrix_values.append(count)
    matrix = sparse.coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(selected_barcodes), len(active_features)),
        dtype=np.int32,
    ).tocsr()
    feature_table = pd.DataFrame(
        {
            "chromosome": [chromosome for chromosome, _ in active_features],
            "start": [start for _, start in active_features],
            "end": [min(start + bin_size, chrom_sizes[chromosome]) for chromosome, start in active_features],
        },
        index=pd.Index(
            [
                f"{chromosome}:{start}-{min(start + bin_size, chrom_sizes[chromosome])}"
                for chromosome, start in active_features
            ],
            name="feature_id",
        ),
    )
    feature_table["width"] = feature_table["end"] - feature_table["start"]
    feature_table["representation_id"] = f"preflight_active_fixed_bin_{bin_size // 1000}kb"
    selected_positions["fragment_rows"] = [fragment_rows_by_barcode[value] for value in selected_barcodes]
    selected_positions.index.name = "barcode"
    adata = ad.AnnData(X=matrix, obs=selected_positions, var=feature_table)
    adata.obsm["spatial"] = selected_positions[["pixel_col", "pixel_row"]].to_numpy(dtype=float)
    adata.uns.update(
        {
            "study_id": study_id,
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "dataset_role": "development",
            "genome_build": genome_build,
            "modality": "spatial_atac",
            "representation_id": f"preflight_active_fixed_bin_{bin_size // 1000}kb",
            "representation_source": "deterministic_real_fragment_subset",
            "input_track": "F",
            "comparison_scope": "contract_preflight_only_no_performance_inference",
            "counting_unit": "one_selected_source_fragment_row_midpoint",
            "construction_leakage_inputs": "none",
            "seed": seed,
        }
    )
    h5ad_path = output_dir / "preflight_input.h5ad"
    adata.write_h5ad(h5ad_path, compression="gzip")

    fixture_chrom_sizes = output_dir / "preflight.chrom.sizes.tsv"
    fixture_chrom_sizes.write_text(
        "".join(f"{chromosome}\t{size}\n" for chromosome, size in chrom_sizes.items()),
        encoding="utf-8",
    )
    fixture_annotation = output_dir / "preflight.annotation.gtf.gz"
    annotation_audit = _normalize_gtf(gene_annotation_path, fixture_annotation, chrom_sizes)
    forbidden_obs_columns = sorted(
        column
        for column in adata.obs.columns
        if any(token in column.lower() for token in FORBIDDEN_OBSERVATION_TOKENS)
    )
    if forbidden_obs_columns:
        raise ValueError(f"Forbidden evaluation fields entered fixture observations: {forbidden_obs_columns}")

    artifacts = {
        "input_h5ad": h5ad_path,
        "fragments": fragments_gz,
        "fragments_index": fragments_index,
        "gene_annotation": fixture_annotation,
        "chrom_sizes": fixture_chrom_sizes,
    }
    selected_barcode_sha256 = hashlib.sha256(
        ("\n".join(selected_barcodes) + "\n").encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "fixture_role": "g4_contract_preflight_only",
        "authorization": authorization,
        "study_id": study_id,
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "genome_build": genome_build,
        "selection_policy": "lowest_sha256(seed_null_barcode)",
        "seed": seed,
        "selected_barcode_sha256": selected_barcode_sha256,
        "n_observations": adata.n_obs,
        "n_active_bins": adata.n_vars,
        "bin_size": bin_size,
        "source_fragment_rows": source_fragment_rows,
        "selected_fragment_rows": selected_fragment_rows,
        "excluded_noncanonical_fragment_rows": excluded_noncanonical_rows,
        "invalid_selected_fragment_rows": invalid_selected_rows,
        "fragment_counting_policy": "fifth_column_forced_to_one_each_source_row_counts_once",
        "annotation_normalization": annotation_audit,
        "source_inputs": {
            "fragments": {"path": str(fragments_path), "sha256": _sha256(fragments_path)},
            "positions": {"path": str(positions_path), "sha256": _sha256(positions_path)},
            "gene_annotation": {
                "path": str(gene_annotation_path),
                "sha256": _sha256(gene_annotation_path),
            },
            "chrom_sizes": {"path": str(chrom_sizes_path), "sha256": _sha256(chrom_sizes_path)},
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for name, path in artifacts.items()
        },
        "construction_input_classes": [
            "accessibility_fragments",
            "spatial_coordinates",
            "gene_annotation",
            "chrom_sizes",
        ],
        "forbidden_evaluation_inputs_present": False,
        "forbidden_observation_columns": forbidden_obs_columns,
        "holdout_accessed": False,
        "development_benchmark_run": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "status": "pass",
    }
    manifest_path = output_dir / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["manifest"] = manifest_path
    return artifacts


def write_g4_contract_preflight_evidence(
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Validate and summarize the frozen two-species, three-method G4 preflight."""

    root = Path(root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_attestation_path = root / "results/v1/g3/environment/method_runtime_container_attestation.json"
    runtime_attestation = json.loads(runtime_attestation_path.read_text(encoding="utf-8"))
    runtime_images = {
        row["method_id"]: row
        for row in runtime_attestation.get("methods", [])
        if isinstance(row, dict) and "method_id" in row
    }
    fixture_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for species, panel in G4_PREFLIGHT_PANEL.items():
        fixture_dir = root / panel["fixture"]
        fixture_manifest_path = fixture_dir / "fixture_manifest.json"
        fixture = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
        fixture_status = (
            fixture.get("status") == "pass"
            and fixture.get("fixture_role") == "g4_contract_preflight_only"
            and fixture.get("genome_build") == panel["genome_build"]
            and fixture.get("n_observations") == 64
            and fixture.get("invalid_selected_fragment_rows") == 0
            and fixture.get("forbidden_evaluation_inputs_present") is False
            and fixture.get("holdout_accessed") is False
            and fixture.get("development_benchmark_run") is False
            and fixture.get("full_benchmark_run") is False
            and fixture.get("final_ranking_run") is False
        )
        fixture_rows.append(
            {
                "species": species,
                "genome_build": panel["genome_build"],
                "n_observations": fixture.get("n_observations"),
                "n_active_bins": fixture.get("n_active_bins"),
                "selected_fragment_rows": fixture.get("selected_fragment_rows"),
                "excluded_noncanonical_fragment_rows": fixture.get(
                    "excluded_noncanonical_fragment_rows"
                ),
                "annotation_records": fixture.get("annotation_normalization", {}).get(
                    "annotation_records"
                ),
                "gene_name_additions": fixture.get("annotation_normalization", {}).get(
                    "gene_name_additions"
                ),
                "gene_biotype_additions": fixture.get("annotation_normalization", {}).get(
                    "gene_biotype_additions"
                ),
                "manifest_sha256": _sha256(fixture_manifest_path),
                "status": "pass" if fixture_status else "fail",
            }
        )

        for method_dir, method_id in G4_PREFLIGHT_METHODS.items():
            run_dir = root / "results/v1/g4/preflight/runs" / species / method_dir
            input_validation_path = run_dir / "input_validation.json"
            output_manifest_path = run_dir / "output_manifest.json"
            contract_path = run_dir / "contract_validation.tsv"
            runtime_path = run_dir / "runtime.tsv"
            output_h5ad_path = run_dir / "gene_activity.h5ad"
            input_validation = json.loads(input_validation_path.read_text(encoding="utf-8"))
            output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
            contract = pd.read_csv(contract_path, sep="\t")
            runtime = pd.read_csv(runtime_path, sep="\t").iloc[0]
            output = ad.read_h5ad(output_h5ad_path, backed="r")
            image = runtime_images.get(method_id, {})
            run_status = (
                input_validation.get("status") == "pass"
                and output_manifest.get("status") == "pass"
                and len(contract) == 18
                and bool((contract["status"] == "pass").all())
                and runtime.get("status") == "pass"
                and output.n_obs == 64
                and output.n_vars > 0
                and output.var_names.is_unique
                and output.uns.get("method_id") == method_id
                and output.uns.get("genome_build") == panel["genome_build"]
                and output.uns.get("construction_leakage_inputs") == "none"
                and output.uns.get("comparison_scope")
                == "fragment_portability_contract_preflight_no_performance_inference"
                and image.get("status") == "pass"
                and image.get("contract_status") == "pass"
                and image.get("contract_checks_passed") == image.get("contract_checks_total") == 18
            )
            run_rows.append(
                {
                    "species": species,
                    "genome_build": panel["genome_build"],
                    "method_id": method_id,
                    "software_name": runtime.get("software_name"),
                    "software_version": runtime.get("software_version"),
                    "n_observations": output.n_obs,
                    "n_genes": output.n_vars,
                    "contract_checks_passed": int((contract["status"] == "pass").sum()),
                    "contract_checks_total": len(contract),
                    "runtime_seconds": float(runtime.get("runtime_seconds")),
                    "driver_observed_peak_memory_mb": float(runtime.get("peak_memory_mb")),
                    "image_digest": image.get("image_digest"),
                    "input_validation_sha256": _sha256(input_validation_path),
                    "output_h5ad_sha256": _sha256(output_h5ad_path),
                    "resolved_attempt_failures": len(
                        list(run_dir.glob("attempt_*_schema_failure.tsv"))
                    ),
                    "status": "pass" if run_status else "fail",
                }
            )
            output.file.close()

    fixture_table = pd.DataFrame(fixture_rows)
    run_table = pd.DataFrame(run_rows)
    fixture_table_path = output_dir / "g4_preflight_fixture_summary.tsv"
    run_table_path = output_dir / "g4_preflight_run_summary.tsv"
    fixture_table.to_csv(fixture_table_path, sep="\t", index=False)
    run_table.to_csv(run_table_path, sep="\t", index=False)
    status = (
        "pass"
        if len(fixture_table) == 2
        and len(run_table) == 6
        and bool((fixture_table["status"] == "pass").all())
        and bool((run_table["status"] == "pass").all())
        else "fail"
    )
    evidence = {
        "schema_version": 1,
        "milestone": "G4-M0",
        "scope": "mouse_human_fragment_method_contract_preflight_only",
        "status": status,
        "fixture_summary": {
            "path": str(fixture_table_path.relative_to(root)),
            "sha256": _sha256(fixture_table_path),
            "rows": len(fixture_table),
        },
        "run_summary": {
            "path": str(run_table_path.relative_to(root)),
            "sha256": _sha256(run_table_path),
            "rows": len(run_table),
        },
        "resolved_attempt_failures": int(run_table["resolved_attempt_failures"].sum()),
        "fixture_generator_function_sha256": hashlib.sha256(
            inspect.getsource(build_real_data_contract_fixture).encode()
        ).hexdigest(),
        "memory_measurement_scope": "driver_observed_only_not_admissible_for_performance_comparison",
        "cicero_scope": "excluded_from_64_observation_preflight_retained_for_separate_scalability_ladder",
        "holdout_accessed": False,
        "construction_evaluation_inputs_used": False,
        "development_benchmark_run": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
    }
    evidence_path = output_dir / "g4_contract_preflight_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "evidence": evidence_path,
        "fixtures": fixture_table_path,
        "runs": run_table_path,
    }
