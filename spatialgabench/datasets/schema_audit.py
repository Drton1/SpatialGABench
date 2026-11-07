"""Streaming schema audits for candidate spatial ATAC/RNA inputs."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import TextIO

import pandas as pd


BARCODE_COVERAGE_THRESHOLD = 0.95


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _normalize_barcode(value: str) -> str:
    value = value.strip().strip('"')
    return value[:-2] if value.endswith("-1") else value


def _audit_fragments(path: Path, allow_zero_span: bool = False) -> tuple[dict[str, object], set[str]]:
    barcodes: set[str] = set()
    chromosomes: set[str] = set()
    rows = 0
    invalid_rows = 0
    header_rows = 0
    span_total = 0
    span_min: int | None = None
    span_max = 0
    spans_over_2kb = 0
    spans_over_10kb = 0
    spans_over_100kb = 0
    zero_span_rows = 0
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                header_rows += 1
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                invalid_rows += 1
                continue
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                invalid_rows += 1
                continue
            barcode = _normalize_barcode(fields[3])
            if not fields[0] or not barcode or start < 0 or end < start or (end == start and not allow_zero_span):
                invalid_rows += 1
                continue
            rows += 1
            span = end - start
            zero_span_rows += int(span == 0)
            span_total += span
            span_min = span if span_min is None else min(span_min, span)
            span_max = max(span_max, span)
            spans_over_2kb += int(span > 2_000)
            spans_over_10kb += int(span > 10_000)
            spans_over_100kb += int(span > 100_000)
            chromosomes.add(fields[0])
            barcodes.add(barcode)
    return (
        {
            "fragment_rows": rows,
            "fragment_header_rows": header_rows,
            "fragment_invalid_rows": invalid_rows,
            "fragment_barcodes": len(barcodes),
            "fragment_chromosomes": sorted(chromosomes),
            "fragment_span_min": span_min or 0,
            "fragment_span_max": span_max,
            "fragment_span_mean": float(span_total / rows) if rows else 0.0,
            "fragment_spans_over_2kb": spans_over_2kb,
            "fragment_spans_over_10kb": spans_over_10kb,
            "fragment_spans_over_100kb": spans_over_100kb,
            "fragment_zero_span_rows": zero_span_rows,
            "fragment_spans_over_10kb_fraction": float(spans_over_10kb / rows) if rows else 0.0,
        },
        barcodes,
    )


def _audit_positions(path: Path) -> tuple[dict[str, int], set[str]]:
    barcodes: set[str] = set()
    rows = 0
    malformed_rows = 0
    header_rows = 0
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            if row_index == 0 and row and _normalize_barcode(row[0]).lower() in {"barcode", "barcodes"}:
                header_rows += 1
                continue
            if len(row) < 6:
                malformed_rows += 1
                continue
            barcode = _normalize_barcode(row[0])
            try:
                [float(value) for value in row[1:6]]
            except ValueError:
                malformed_rows += 1
                continue
            if not barcode:
                malformed_rows += 1
                continue
            rows += 1
            barcodes.add(barcode)
    return (
        {
            "position_rows": rows,
            "position_header_rows": header_rows,
            "position_malformed_rows": malformed_rows,
            "position_barcodes": len(barcodes),
        },
        barcodes,
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _audit_tsv_identifiers(path: Path, minimum_columns: int) -> tuple[dict[str, int], set[str]]:
    identifiers: set[str] = set()
    rows = 0
    malformed_rows = 0
    duplicate_rows = 0
    with _open_text(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < minimum_columns or not fields[0].strip():
                malformed_rows += 1
                continue
            identifier = _normalize_barcode(fields[0])
            if identifier in identifiers:
                duplicate_rows += 1
            identifiers.add(identifier)
            rows += 1
    return (
        {
            "rows": rows,
            "malformed_rows": malformed_rows,
            "duplicate_rows": duplicate_rows,
            "unique_identifiers": len(identifiers),
        },
        identifiers,
    )


def _audit_matrix_market(path: Path) -> dict[str, object]:
    header_valid = False
    dimensions: tuple[int, int, int] | None = None
    observed_entries = 0
    malformed_entries = 0
    out_of_bounds_entries = 0
    with _open_text(path) as handle:
        for line_index, line in enumerate(handle):
            stripped = line.strip()
            if line_index == 0:
                header_valid = stripped.lower().startswith("%%matrixmarket matrix coordinate")
                continue
            if not stripped or stripped.startswith("%"):
                continue
            if dimensions is None:
                fields = stripped.split()
                try:
                    dimensions = tuple(int(value) for value in fields)  # type: ignore[assignment]
                except ValueError:
                    dimensions = None
                if dimensions is None or len(dimensions) != 3 or any(value < 0 for value in dimensions):
                    raise ValueError(f"Invalid Matrix Market dimensions in {path}: {stripped}")
                continue
            fields = stripped.split()
            if len(fields) != 3:
                malformed_entries += 1
                continue
            try:
                row = int(fields[0])
                column = int(fields[1])
                float(fields[2])
            except ValueError:
                malformed_entries += 1
                continue
            observed_entries += 1
            if row < 1 or row > dimensions[0] or column < 1 or column > dimensions[1]:
                out_of_bounds_entries += 1

    matrix_rows, matrix_columns, declared_entries = dimensions or (0, 0, 0)
    return {
        "matrix_market_header_valid": header_valid,
        "matrix_rows": matrix_rows,
        "matrix_columns": matrix_columns,
        "matrix_declared_entries": declared_entries,
        "matrix_observed_entries": observed_entries,
        "matrix_malformed_entries": malformed_entries,
        "matrix_out_of_bounds_entries": out_of_bounds_entries,
    }


def audit_fragment_spatial_inputs(
    fragments_path: str | Path,
    positions_path: str | Path,
) -> dict[str, object]:
    """Audit a fragment file and its spatial coordinates without requiring RNA."""

    fragments_path = Path(fragments_path)
    positions_path = Path(positions_path)
    fragment_summary, fragment_barcodes = _audit_fragments(fragments_path)
    position_summary, position_barcodes = _audit_positions(positions_path)
    overlap = fragment_barcodes & position_barcodes
    fragment_coverage = _fraction(len(overlap), len(fragment_barcodes))

    schema_pass = (
        fragment_summary["fragment_rows"] > 0
        and fragment_summary["fragment_invalid_rows"] == 0
        and position_summary["position_rows"] > 0
        and position_summary["position_malformed_rows"] == 0
    )
    barcode_pass = bool(overlap) and fragment_coverage >= BARCODE_COVERAGE_THRESHOLD
    return {
        "audit_type": "fragment_spatial",
        "fragments_path": str(fragments_path),
        "positions_path": str(positions_path),
        **fragment_summary,
        **position_summary,
        "fragment_position_overlap": len(overlap),
        "fragment_barcodes_in_positions_fraction": fragment_coverage,
        "position_barcodes_with_fragments_fraction": _fraction(len(overlap), len(position_barcodes)),
        "barcode_coverage_threshold": BARCODE_COVERAGE_THRESHOLD,
        "schema_status": "pass" if schema_pass else "fail",
        "barcode_contract_status": "pass" if barcode_pass else "fail",
        "biological_pairing_status": "not_applicable_no_rna",
    }


def audit_paired_spatial_10x_inputs(
    fragments_path: str | Path,
    rna_barcodes_path: str | Path,
    rna_features_path: str | Path,
    rna_matrix_path: str | Path,
    atac_positions_path: str | Path,
    rna_positions_path: str | Path,
) -> dict[str, object]:
    """Audit fragments and a 10x RNA matrix with assay-specific coordinates."""

    fragments_path = Path(fragments_path)
    rna_barcodes_path = Path(rna_barcodes_path)
    rna_features_path = Path(rna_features_path)
    rna_matrix_path = Path(rna_matrix_path)
    atac_positions_path = Path(atac_positions_path)
    rna_positions_path = Path(rna_positions_path)

    fragment_summary, fragment_barcodes = _audit_fragments(fragments_path)
    atac_position_summary, atac_position_barcodes = _audit_positions(atac_positions_path)
    rna_position_summary, rna_position_barcodes = _audit_positions(rna_positions_path)
    barcode_summary, rna_barcodes = _audit_tsv_identifiers(rna_barcodes_path, minimum_columns=1)
    feature_summary, _ = _audit_tsv_identifiers(rna_features_path, minimum_columns=2)
    matrix_summary = _audit_matrix_market(rna_matrix_path)

    fragment_atac_positions = fragment_barcodes & atac_position_barcodes
    rna_rna_positions = rna_barcodes & rna_position_barcodes
    fragment_rna = fragment_barcodes & rna_barcodes
    cross_assay_coordinate_overlap = fragment_barcodes & rna_barcodes & atac_position_barcodes & rna_position_barcodes

    fragment_position_fraction = _fraction(len(fragment_atac_positions), len(fragment_barcodes))
    rna_position_fraction = _fraction(len(rna_rna_positions), len(rna_barcodes))
    dimensions_match = (
        matrix_summary["matrix_rows"] == feature_summary["rows"]
        and matrix_summary["matrix_columns"] == barcode_summary["rows"]
    )
    entries_match = matrix_summary["matrix_declared_entries"] == matrix_summary["matrix_observed_entries"]
    schema_pass = (
        fragment_summary["fragment_rows"] > 0
        and fragment_summary["fragment_invalid_rows"] == 0
        and atac_position_summary["position_rows"] > 0
        and atac_position_summary["position_malformed_rows"] == 0
        and rna_position_summary["position_rows"] > 0
        and rna_position_summary["position_malformed_rows"] == 0
        and barcode_summary["rows"] > 0
        and barcode_summary["malformed_rows"] == 0
        and barcode_summary["duplicate_rows"] == 0
        and feature_summary["rows"] > 0
        and feature_summary["malformed_rows"] == 0
        and matrix_summary["matrix_market_header_valid"]
        and matrix_summary["matrix_malformed_entries"] == 0
        and matrix_summary["matrix_out_of_bounds_entries"] == 0
        and dimensions_match
        and entries_match
    )
    barcode_pass = (
        fragment_position_fraction >= BARCODE_COVERAGE_THRESHOLD
        and rna_position_fraction >= BARCODE_COVERAGE_THRESHOLD
    )
    return {
        "audit_type": "paired_fragment_10x_rna_spatial",
        "fragments_path": str(fragments_path),
        "rna_barcodes_path": str(rna_barcodes_path),
        "rna_features_path": str(rna_features_path),
        "rna_matrix_path": str(rna_matrix_path),
        "atac_positions_path": str(atac_positions_path),
        "rna_positions_path": str(rna_positions_path),
        **fragment_summary,
        "atac_position_rows": atac_position_summary["position_rows"],
        "atac_position_header_rows": atac_position_summary["position_header_rows"],
        "atac_position_malformed_rows": atac_position_summary["position_malformed_rows"],
        "atac_position_barcodes": atac_position_summary["position_barcodes"],
        "rna_position_rows": rna_position_summary["position_rows"],
        "rna_position_header_rows": rna_position_summary["position_header_rows"],
        "rna_position_malformed_rows": rna_position_summary["position_malformed_rows"],
        "rna_position_barcodes": rna_position_summary["position_barcodes"],
        "rna_barcode_rows": barcode_summary["rows"],
        "rna_barcode_malformed_rows": barcode_summary["malformed_rows"],
        "rna_barcode_duplicate_rows": barcode_summary["duplicate_rows"],
        "rna_barcodes": barcode_summary["unique_identifiers"],
        "rna_feature_rows": feature_summary["rows"],
        "rna_feature_malformed_rows": feature_summary["malformed_rows"],
        "rna_feature_duplicate_ids": feature_summary["duplicate_rows"],
        **matrix_summary,
        "matrix_dimensions_match_metadata": dimensions_match,
        "matrix_entries_match_declaration": entries_match,
        "fragment_atac_position_overlap": len(fragment_atac_positions),
        "fragment_barcodes_in_atac_positions_fraction": fragment_position_fraction,
        "rna_rna_position_overlap": len(rna_rna_positions),
        "rna_barcodes_in_rna_positions_fraction": rna_position_fraction,
        "fragment_rna_barcode_overlap": len(fragment_rna),
        "cross_assay_coordinate_barcode_overlap": len(cross_assay_coordinate_overlap),
        "rna_barcodes_in_fragment_fraction": _fraction(len(fragment_rna), len(rna_barcodes)),
        "fragment_barcodes_in_rna_fraction": _fraction(len(fragment_rna), len(fragment_barcodes)),
        "barcode_coverage_threshold": BARCODE_COVERAGE_THRESHOLD,
        "schema_status": "pass" if schema_pass else "fail",
        "barcode_contract_status": "pass" if barcode_pass else "fail",
        "biological_pairing_status": "requires_source_design_verification",
    }


def audit_paired_spatial_inputs(
    fragments_path: str | Path,
    rna_matrix_path: str | Path,
    positions_path: str | Path,
) -> dict[str, object]:
    """Audit schemas and barcode compatibility without loading count matrices."""

    fragments_path = Path(fragments_path)
    rna_matrix_path = Path(rna_matrix_path)
    positions_path = Path(positions_path)

    fragment_summary, fragment_barcodes = _audit_fragments(fragments_path)

    with _open_text(rna_matrix_path) as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rna_barcodes = {_normalize_barcode(value) for value in header[1:] if value.strip()}
        expected_columns = len(header)
        rna_gene_rows = 0
        rna_malformed_rows = 0
        for row in reader:
            if len(row) != expected_columns:
                rna_malformed_rows += 1
            rna_gene_rows += 1

    position_summary, position_barcodes = _audit_positions(positions_path)

    fragment_rna = fragment_barcodes & rna_barcodes
    fragment_positions = fragment_barcodes & position_barcodes
    rna_positions = rna_barcodes & position_barcodes
    triple = fragment_barcodes & rna_barcodes & position_barcodes

    schema_pass = (
        fragment_summary["fragment_rows"] > 0
        and fragment_summary["fragment_invalid_rows"] == 0
        and rna_gene_rows > 0
        and rna_malformed_rows == 0
        and position_summary["position_rows"] > 0
        and position_summary["position_malformed_rows"] == 0
    )
    fragment_position_fraction = _fraction(len(fragment_positions), len(fragment_barcodes))
    rna_triple_fraction = _fraction(len(triple), len(rna_barcodes))
    barcode_pass = (
        bool(triple)
        and fragment_position_fraction >= BARCODE_COVERAGE_THRESHOLD
        and rna_triple_fraction >= BARCODE_COVERAGE_THRESHOLD
    )
    return {
        "audit_type": "paired_fragment_rna_spatial",
        "fragments_path": str(fragments_path),
        "rna_matrix_path": str(rna_matrix_path),
        "positions_path": str(positions_path),
        **fragment_summary,
        "rna_gene_rows": rna_gene_rows,
        "rna_malformed_rows": rna_malformed_rows,
        "rna_barcodes": len(rna_barcodes),
        **position_summary,
        "fragment_rna_overlap": len(fragment_rna),
        "fragment_position_overlap": len(fragment_positions),
        "rna_position_overlap": len(rna_positions),
        "triple_barcode_overlap": len(triple),
        "fragment_barcodes_in_positions_fraction": fragment_position_fraction,
        "fragment_barcodes_in_triple_fraction": _fraction(len(triple), len(fragment_barcodes)),
        "rna_barcodes_in_triple_fraction": rna_triple_fraction,
        "position_barcodes_in_triple_fraction": _fraction(len(triple), len(position_barcodes)),
        "barcode_coverage_threshold": BARCODE_COVERAGE_THRESHOLD,
        "schema_status": "pass" if schema_pass else "fail",
        "barcode_contract_status": "pass" if barcode_pass else "fail",
        "biological_pairing_status": "requires_source_design_verification",
    }


def write_paired_spatial_audit(
    fragments_path: str | Path,
    rna_matrix_path: str | Path,
    positions_path: str | Path,
    outdir: str | Path,
) -> dict[str, Path]:
    """Write JSON and one-row TSV audit outputs."""

    result = audit_paired_spatial_inputs(fragments_path, rna_matrix_path, positions_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "paired_spatial_schema_audit.json"
    tsv_path = outdir / "paired_spatial_schema_audit.tsv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_row = dict(result)
    table_row["fragment_chromosomes"] = ";".join(result["fragment_chromosomes"])
    pd.DataFrame([table_row]).to_csv(tsv_path, sep="\t", index=False)
    return {"json": json_path, "tsv": tsv_path}


def write_fragment_spatial_audit(
    fragments_path: str | Path,
    positions_path: str | Path,
    outdir: str | Path,
) -> dict[str, Path]:
    """Write JSON and one-row TSV outputs for a fragment/spatial audit."""

    result = audit_fragment_spatial_inputs(fragments_path, positions_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "fragment_spatial_schema_audit.json"
    tsv_path = outdir / "fragment_spatial_schema_audit.tsv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_row = dict(result)
    table_row["fragment_chromosomes"] = ";".join(result["fragment_chromosomes"])
    pd.DataFrame([table_row]).to_csv(tsv_path, sep="\t", index=False)
    return {"json": json_path, "tsv": tsv_path}


def write_paired_spatial_10x_audit(
    fragments_path: str | Path,
    rna_barcodes_path: str | Path,
    rna_features_path: str | Path,
    rna_matrix_path: str | Path,
    atac_positions_path: str | Path,
    rna_positions_path: str | Path,
    outdir: str | Path,
) -> dict[str, Path]:
    """Write JSON and one-row TSV outputs for a fragment/10x RNA audit."""

    result = audit_paired_spatial_10x_inputs(
        fragments_path,
        rna_barcodes_path,
        rna_features_path,
        rna_matrix_path,
        atac_positions_path,
        rna_positions_path,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "paired_spatial_10x_schema_audit.json"
    tsv_path = outdir / "paired_spatial_10x_schema_audit.tsv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_row = dict(result)
    table_row["fragment_chromosomes"] = ";".join(result["fragment_chromosomes"])
    pd.DataFrame([table_row]).to_csv(tsv_path, sep="\t", index=False)
    return {"json": json_path, "tsv": tsv_path}
