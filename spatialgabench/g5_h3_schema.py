"""Authorized accessibility-only schema preflight for GSE307620."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


FORBIDDEN_FALSE_FIELDS = (
    "representation_construction_run",
    "peak_calling_run",
    "fixed_bin_construction_run",
    "gene_activity_construction_run",
    "method_execution_run",
    "rna_access_run",
    "label_or_pathology_access_run",
    "policy_leaf_outcome_access_run",
    "regret_evaluation_run",
    "policy_refit_run",
    "original_holdout_accessed",
    "full_benchmark_run",
    "results_writing_run",
    "universal_ranking_run",
    "final_ranking_run",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_contract(config_path: Path, root: Path) -> dict[str, Any]:
    config = _yaml(config_path)
    for key in ("authorization", "previous_gate", "sample_metadata"):
        path = root / str(config[key])
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"G5-H3 frozen artifact mismatch: {key}")
    for key in ("interpretation_amendment", "initial_blocked_evidence"):
        if key in config:
            path = root / str(config[key])
            if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
                raise ValueError(f"G5-H3 frozen artifact mismatch: {key}")
    authorization = _json(root / config["authorization"])
    previous = _json(root / config["previous_gate"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone")
        != config.get("authorization_milestone", "H3_GSE307620_accessibility_schema_preflight")
        or authorization.get("authorized_fragment_sha256") != config["fragment_sha256"]
        or authorization.get("fragment_row_streaming_authorized") is not True
        or authorization.get("spatial_barcode_overlap_authorized") is not True
        or authorization.get("method_execution_authorized") is not False
    ):
        raise PermissionError("G5-H3 accessibility-schema authorization is invalid")
    if "interpretation_amendment" in config and (
        authorization.get("interpretation_amendment_sha256") != config["interpretation_amendment_sha256"]
        or authorization.get("initial_blocked_evidence_sha256") != config["initial_blocked_evidence_sha256"]
        or authorization.get("header_prefix") != config.get("allowed_header_prefix")
        or authorization.get("exact_fragment_barcode_suffix") != config.get("fragment_barcode_suffix")
        or authorization.get("fallback_barcode_rewrite_authorized") is not False
    ):
        raise PermissionError("G5-H3A interpretation amendment is not authorization-bound")
    if (
        previous.get("status") != "pass"
        or previous.get("milestone") != "H2_GSE307620_minimal_fragment_download"
        or previous.get("fragment_sha256") != config["fragment_sha256"]
        or previous.get("observed_bytes") != config["fragment_expected_bytes"]
        or previous.get("fragment_content_analysis_run") is not False
    ):
        raise ValueError("G5-H2 transfer boundary is invalid")
    if config.get("fragment_content_analysis_run") is not True:
        raise PermissionError("G5-H3 fragment schema access is not enabled")
    invalid = [field for field in FORBIDDEN_FALSE_FIELDS if config.get(field) is not False]
    if invalid:
        raise PermissionError(f"G5-H3 forbidden flags are not false: {invalid}")

    fragment_path = root / config["fragment_path"]
    positions_path = root / config["positions_path"]
    if not fragment_path.is_file() or fragment_path.stat().st_size != config["fragment_expected_bytes"]:
        raise ValueError("G5-H3 fragment size no longer matches the H2 identity")
    if not positions_path.is_file() or sha256(positions_path) != config["positions_sha256"]:
        raise ValueError("G5-H3 positions artifact mismatch")
    return config


def _load_positions(path: Path) -> dict[str, tuple[bool, int, int, int, int]]:
    positions: dict[str, tuple[bool, int, int, int, int]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 6:
                raise ValueError("G5-H3 positions row does not have six columns")
            barcode = row[0]
            if barcode in positions:
                raise ValueError(f"G5-H3 duplicate spatial barcode: {barcode}")
            positions[barcode] = (bool(int(row[1])), int(row[2]), int(row[3]), int(row[4]), int(row[5]))
    return positions


def _scan_fragments(
    path: Path,
    position_barcodes: set[bytes],
    primary_chromosomes: list[bytes],
    mitochondrial_chromosome: bytes,
    header_prefix: bytes = b"",
    barcode_suffix: bytes = b"",
) -> dict[str, Any]:
    chromosome_rows: Counter[bytes] = Counter()
    barcode_rows: Counter[bytes] = Counter()
    barcode_weighted_counts: Counter[bytes] = Counter()
    invalid_column_rows = 0
    invalid_coordinate_rows = 0
    invalid_duplicate_count_rows = 0
    empty_barcode_rows = 0
    coordinate_order_violations = 0
    source_lines = 0
    header_rows = 0
    barcode_suffix_mismatch_rows = 0
    fragment_rows = 0
    weighted_fragment_count = 0
    min_span: int | None = None
    max_span = 0
    previous_chromosome: bytes | None = None
    previous_start = -1
    closed_chromosomes: set[bytes] = set()
    header_primary_contigs: set[bytes] = set()

    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
            source_lines += 1
            if header_prefix and raw_line.startswith(header_prefix):
                header_rows += 1
                stripped = raw_line.strip()
                declaration = header_prefix + b" primary_contig="
                if stripped.startswith(declaration):
                    header_primary_contigs.add(stripped[len(declaration) :])
                continue
            fragment_rows += 1
            fields = raw_line.rstrip(b"\r\n").split(b"\t")
            if len(fields) != 5:
                invalid_column_rows += 1
                continue
            chromosome, start_raw, end_raw, barcode, count_raw = fields
            chromosome_rows[chromosome] += 1
            if not barcode:
                empty_barcode_rows += 1
            else:
                if barcode_suffix:
                    if barcode.endswith(barcode_suffix):
                        barcode = barcode[: -len(barcode_suffix)]
                    else:
                        barcode_suffix_mismatch_rows += 1
                barcode_rows[barcode] += 1
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError:
                invalid_coordinate_rows += 1
                continue
            if start < 0 or end <= start:
                invalid_coordinate_rows += 1
                continue
            try:
                duplicate_count = int(count_raw)
            except ValueError:
                invalid_duplicate_count_rows += 1
                continue
            if duplicate_count <= 0:
                invalid_duplicate_count_rows += 1
                continue

            if chromosome != previous_chromosome:
                if chromosome in closed_chromosomes:
                    coordinate_order_violations += 1
                if previous_chromosome is not None:
                    closed_chromosomes.add(previous_chromosome)
                previous_chromosome = chromosome
                previous_start = -1
            if start < previous_start:
                coordinate_order_violations += 1
            previous_start = start

            span = end - start
            min_span = span if min_span is None else min(min_span, span)
            max_span = max(max_span, span)
            weighted_fragment_count += duplicate_count
            if barcode:
                barcode_weighted_counts[barcode] += duplicate_count

    primary = set(primary_chromosomes)
    allowed = primary | {mitochondrial_chromosome}
    observed_barcodes = set(barcode_rows)
    undeclared_chromosome_rows = sum(
        count for chrom, count in chromosome_rows.items() if header_primary_contigs and chrom not in header_primary_contigs
    )
    return {
        "source_lines": source_lines,
        "header_rows": header_rows,
        "fragment_rows": fragment_rows,
        "weighted_fragment_count": weighted_fragment_count,
        "invalid_column_rows": invalid_column_rows,
        "invalid_coordinate_rows": invalid_coordinate_rows,
        "invalid_duplicate_count_rows": invalid_duplicate_count_rows,
        "empty_barcode_rows": empty_barcode_rows,
        "coordinate_order_violations": coordinate_order_violations,
        "barcode_suffix_mismatch_rows": barcode_suffix_mismatch_rows,
        "minimum_fragment_span": min_span,
        "maximum_fragment_span": max_span,
        "unique_fragment_barcodes": len(observed_barcodes),
        "fragment_barcodes_in_positions": len(observed_barcodes & position_barcodes),
        "fragment_barcodes_not_in_positions": len(observed_barcodes - position_barcodes),
        "position_barcodes_with_fragments": len(position_barcodes & observed_barcodes),
        "primary_chromosome_rows": sum(chromosome_rows[chrom] for chrom in primary),
        "mitochondrial_rows": chromosome_rows[mitochondrial_chromosome],
        "unexpected_chromosome_rows": sum(count for chrom, count in chromosome_rows.items() if chrom not in allowed),
        "header_primary_contigs": header_primary_contigs,
        "undeclared_chromosome_rows": undeclared_chromosome_rows,
        "chromosome_rows": chromosome_rows,
        "barcode_rows": barcode_rows,
        "barcode_weighted_counts": barcode_weighted_counts,
    }


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_contract(config_path, root)
    positions = _load_positions(root / config["positions_path"])
    in_tissue = {barcode for barcode, row in positions.items() if row[0]}
    if len(positions) != config["positions_expected_rows"] or len(in_tissue) != config["in_tissue_expected_rows"]:
        raise ValueError("G5-H3 frozen spatial observation contract mismatch")

    primary = [chrom.encode("ascii") for chrom in config["primary_chromosomes"]]
    mitochondrial = config["mitochondrial_chromosome"].encode("ascii")
    header_prefix = str(config.get("allowed_header_prefix", "")).encode("ascii")
    barcode_suffix = str(config.get("fragment_barcode_suffix", "")).encode("ascii")
    position_bytes = {barcode.encode("ascii") for barcode in positions}
    scan = _scan_fragments(
        root / config["fragment_path"],
        position_bytes,
        primary,
        mitochondrial,
        header_prefix,
        barcode_suffix,
    )
    observed_barcodes = set(scan["barcode_rows"])
    in_tissue_bytes = {barcode.encode("ascii") for barcode in in_tissue}
    in_tissue_overlap = len(in_tissue_bytes & observed_barcodes)
    overlap_fraction = in_tissue_overlap / len(in_tissue_bytes)

    checks = {
        "source_fragment_rows_positive": scan["fragment_rows"] > 0,
        "exact_five_column_schema": scan["invalid_column_rows"] == 0,
        "valid_zero_based_half_open_spans": scan["invalid_coordinate_rows"] == 0,
        "positive_duplicate_counts": scan["invalid_duplicate_count_rows"] == 0,
        "nonempty_fragment_barcodes": scan["empty_barcode_rows"] == 0,
        "coordinate_sorted_and_chromosome_grouped": scan["coordinate_order_violations"] == 0,
        "cellranger_header_present": scan["header_rows"] > 0 if header_prefix else scan["header_rows"] == 0,
        "barcode_suffix_contract_exact": scan["barcode_suffix_mismatch_rows"] == 0,
        "all_fragment_barcodes_declared_spatially": scan["fragment_barcodes_not_in_positions"] == 0,
        "all_spatial_barcodes_have_fragments": scan["position_barcodes_with_fragments"] == len(positions),
        "all_in_tissue_barcodes_have_fragments": in_tissue_overlap == len(in_tissue),
        "in_tissue_overlap_threshold": overlap_fraction >= config["minimum_in_tissue_barcode_overlap_fraction"],
        "primary_mm10_rows_present": scan["primary_chromosome_rows"] > 0,
        "all_observed_chromosomes_declared_by_source_reference": scan["undeclared_chromosome_rows"] == 0,
        "weighted_count_not_below_rows": scan["weighted_fragment_count"] >= scan["fragment_rows"],
    }

    output = root / config["output_root"]
    chromosome_path = output / "chromosome_inventory.tsv"
    barcode_path = output / "spatial_barcode_overlap.tsv"
    summary_path = output / "fragment_schema_summary.json"
    evidence_path = output / "completion_evidence.json"
    _write_tsv(
        chromosome_path,
        [
            {
                "chromosome": chromosome.decode("ascii", errors="replace"),
                "fragment_rows": count,
                "classification": (
                    "primary" if chromosome in set(primary) else "mitochondrial" if chromosome == mitochondrial else "unexpected"
                ),
                "construction_inclusion": chromosome in set(primary),
            }
            for chromosome, count in sorted(scan["chromosome_rows"].items())
        ],
        ["chromosome", "fragment_rows", "classification", "construction_inclusion"],
    )
    _write_tsv(
        barcode_path,
        [
            {
                "barcode": barcode,
                "in_tissue": positions[barcode][0],
                "array_row": positions[barcode][1],
                "array_col": positions[barcode][2],
                "fragment_rows": scan["barcode_rows"].get(barcode.encode("ascii"), 0),
                "weighted_fragment_count": scan["barcode_weighted_counts"].get(barcode.encode("ascii"), 0),
                "overlap_status": "present" if barcode.encode("ascii") in observed_barcodes else "absent",
            }
            for barcode in sorted(positions)
        ],
        ["barcode", "in_tissue", "array_row", "array_col", "fragment_rows", "weighted_fragment_count", "overlap_status"],
    )
    summary = {
        key: value
        for key, value in scan.items()
        if key not in {"chromosome_rows", "barcode_rows", "barcode_weighted_counts"}
    }
    summary["header_primary_contigs"] = sorted(
        chrom.decode("ascii", errors="replace") for chrom in scan["header_primary_contigs"]
    )
    summary.update(
        {
            "schema_version": 1,
            "status": "pass" if all(checks.values()) else "blocked",
            "accession": config["accession"],
            "sample_accession": config["sample_accession"],
            "species": config["species"],
            "genome_build": config["genome_build"],
            "positions_rows": len(positions),
            "in_tissue_rows": len(in_tissue),
            "in_tissue_barcodes_with_fragments": in_tissue_overlap,
            "in_tissue_barcode_overlap_fraction": overlap_fraction,
            "checks": checks,
        }
    )
    _write_json(summary_path, summary)
    artifacts = {
        "fragment_schema_summary": summary_path,
        "chromosome_inventory": chromosome_path,
        "spatial_barcode_overlap": barcode_path,
    }
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G5",
            "milestone": "H3_GSE307620_accessibility_schema_preflight",
            "status": summary["status"],
            "scientific_assessment": "standard_mm10_spatial_fragment_contract_verified" if summary["status"] == "pass" else "schema_preflight_blocked",
            "fragment_sha256_inherited_from_h2": config["fragment_sha256"],
            "fragment_sha256_recomputed_in_h3": False,
            "fragment_rows": scan["fragment_rows"],
            "weighted_fragment_count": scan["weighted_fragment_count"],
            "positions_rows": len(positions),
            "in_tissue_rows": len(in_tissue),
            "in_tissue_barcodes_with_fragments": in_tissue_overlap,
            "in_tissue_barcode_overlap_fraction": overlap_fraction,
            "primary_chromosome_rows": scan["primary_chromosome_rows"],
            "mitochondrial_rows": scan["mitochondrial_rows"],
            "unexpected_chromosome_rows": scan["unexpected_chromosome_rows"],
            "checks": checks,
            "artifacts": {
                key: {"path": str(path.relative_to(root)), "sha256": sha256(path)} for key, path in artifacts.items()
            },
            "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
            "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
            "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
            "fragment_content_analysis_run": True,
            "representation_construction_run": False,
            "peak_calling_run": False,
            "fixed_bin_construction_run": False,
            "gene_activity_construction_run": False,
            "method_execution_run": False,
            "rna_access_run": False,
            "label_or_pathology_access_run": False,
            "policy_leaf_outcome_access_run": False,
            "regret_evaluation_run": False,
            "policy_refit_run": False,
            "original_holdout_accessed": False,
            "full_benchmark_run": False,
            "results_writing_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "next_milestone_authorized": False,
        },
    )
    if summary["status"] != "pass":
        raise ValueError(f"G5-H3 schema preflight failed: {[key for key, value in checks.items() if not value]}")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g5_h3_gse307620_accessibility_schema_preflight.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    output = run(args.config, args.root)
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
