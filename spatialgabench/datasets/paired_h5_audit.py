"""Cross-modal schema audit for fragments and a 10x RNA H5 matrix."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .schema_audit import (
    BARCODE_COVERAGE_THRESHOLD,
    _audit_fragments,
    _audit_positions,
    _fraction,
    _normalize_barcode,
)


def _audit_10x_h5(path: Path, chunk_size: int = 1_000_000) -> tuple[dict[str, object], set[str]]:
    with h5py.File(path, "r") as handle:
        if "matrix" not in handle:
            raise ValueError(f"Missing /matrix group in {path}")
        matrix = handle["matrix"]
        required = {"barcodes", "data", "indices", "indptr", "shape", "features"}
        if not required <= set(matrix.keys()):
            raise ValueError(f"Incomplete 10x matrix group in {path}")
        shape = tuple(int(value) for value in matrix["shape"][:])
        if len(shape) != 2:
            raise ValueError(f"Invalid 10x matrix dimensions in {path}: {shape}")
        matrix_rows, matrix_columns = shape
        barcode_values = [_normalize_barcode(str(value)) for value in matrix["barcodes"].asstr()[:]]
        feature_ids = [str(value) for value in matrix["features/id"].asstr()[:]]
        feature_genomes = sorted({str(value) for value in matrix["features/genome"].asstr()[:]})
        data_count = int(matrix["data"].shape[0])
        index_count = int(matrix["indices"].shape[0])
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        out_of_bounds = 0
        negative_values = 0
        for start in range(0, index_count, chunk_size):
            stop = min(start + chunk_size, index_count)
            index_chunk = np.asarray(matrix["indices"][start:stop])
            value_chunk = np.asarray(matrix["data"][start:stop])
            out_of_bounds += int(((index_chunk < 0) | (index_chunk >= matrix_rows)).sum())
            negative_values += int((value_chunk < 0).sum())

    indptr_valid = (
        len(indptr) == matrix_columns + 1
        and len(indptr) > 0
        and int(indptr[0]) == 0
        and int(indptr[-1]) == data_count
        and bool(np.all(indptr[1:] >= indptr[:-1]))
    )
    schema_pass = (
        matrix_rows == len(feature_ids)
        and matrix_columns == len(barcode_values)
        and data_count == index_count
        and indptr_valid
        and out_of_bounds == 0
        and negative_values == 0
        and len(barcode_values) == len(set(barcode_values))
        and len(feature_ids) == len(set(feature_ids))
    )
    return (
        {
            "rna_matrix_rows": matrix_rows,
            "rna_matrix_columns": matrix_columns,
            "rna_matrix_entries": data_count,
            "rna_matrix_index_entries": index_count,
            "rna_matrix_indptr_entries": len(indptr),
            "rna_matrix_indptr_valid": indptr_valid,
            "rna_matrix_out_of_bounds_entries": out_of_bounds,
            "rna_matrix_negative_entries": negative_values,
            "rna_feature_rows": len(feature_ids),
            "rna_feature_genomes": feature_genomes,
            "rna_barcode_rows": len(barcode_values),
            "rna_matrix_schema_status": "pass" if schema_pass else "fail",
        },
        set(barcode_values),
    )


def audit_paired_spatial_h5_inputs(
    fragments_path: str | Path,
    rna_h5_path: str | Path,
    positions_path: str | Path,
    spot_metadata_path: str | Path,
    fragment_semantics: str,
) -> dict[str, object]:
    """Audit ATAC, 10x RNA, coordinates, and spot metadata barcode contracts."""

    fragments_path = Path(fragments_path)
    rna_h5_path = Path(rna_h5_path)
    positions_path = Path(positions_path)
    spot_metadata_path = Path(spot_metadata_path)
    allow_zero_span = fragment_semantics == "author_single_end_insertion_events_pseudo_paired_by_sorted_order"
    fragment_summary, fragment_barcodes = _audit_fragments(fragments_path, allow_zero_span=allow_zero_span)
    position_summary, position_barcodes = _audit_positions(positions_path)
    rna_summary, rna_barcodes = _audit_10x_h5(rna_h5_path)

    positions = pd.read_csv(positions_path)
    in_tissue = pd.to_numeric(positions["in_tissue"], errors="coerce")
    in_tissue_barcodes = {
        _normalize_barcode(str(value))
        for value in positions.loc[in_tissue == 1, "barcode"]
    }
    metadata = pd.read_csv(spot_metadata_path)
    index_values = [str(value) for value in metadata.index]
    metadata_values = index_values if any("#" in value for value in index_values) else metadata["Sample"].astype(str)
    metadata_barcodes = {
        _normalize_barcode(str(value).split("#")[-1])
        for value in metadata_values
        if str(value).strip()
    }
    intersection = fragment_barcodes & rna_barcodes & in_tissue_barcodes & metadata_barcodes
    fragment_contract_fraction = _fraction(len(intersection), len(fragment_barcodes))
    rna_contract_fraction = _fraction(len(intersection), len(rna_barcodes))
    schema_pass = (
        fragment_summary["fragment_rows"] > 0
        and fragment_summary["fragment_invalid_rows"] == 0
        and position_summary["position_malformed_rows"] == 0
        and rna_summary["rna_matrix_schema_status"] == "pass"
        and int(in_tissue.isna().sum()) == 0
        and len(metadata) == len(metadata_barcodes)
    )
    barcode_pass = (
        bool(intersection)
        and fragment_contract_fraction >= BARCODE_COVERAGE_THRESHOLD
        and rna_contract_fraction >= BARCODE_COVERAGE_THRESHOLD
    )
    posthoc_fields = [
        field
        for field in ["ATAC_cluster", "RNA_cluster", "COMBO_cluster", "ecDNA_CN", "EGFR_CN"]
        if field in metadata.columns
    ]
    return {
        "audit_type": "paired_fragment_10x_h5_spatial",
        "fragments_path": str(fragments_path),
        "rna_h5_path": str(rna_h5_path),
        "positions_path": str(positions_path),
        "spot_metadata_path": str(spot_metadata_path),
        "fragment_semantics": fragment_semantics,
        "standard_paired_fragment_semantics_status": "fail" if allow_zero_span else "not_assessed",
        "source_representation_schema_status": "pass" if schema_pass else "fail",
        **fragment_summary,
        **position_summary,
        **rna_summary,
        "position_in_tissue_barcodes": len(in_tissue_barcodes),
        "spot_metadata_rows": len(metadata),
        "spot_metadata_barcodes": len(metadata_barcodes),
        "cross_modal_barcode_overlap": len(intersection),
        "fragment_barcodes_in_cross_modal_contract_fraction": fragment_contract_fraction,
        "rna_barcodes_in_cross_modal_contract_fraction": rna_contract_fraction,
        "posthoc_only_annotation_fields": posthoc_fields,
        "barcode_coverage_threshold": BARCODE_COVERAGE_THRESHOLD,
        "schema_status": "pass" if schema_pass else "fail",
        "barcode_contract_status": "pass" if barcode_pass else "fail",
        "biological_pairing_status": "requires_source_design_verification",
        "annotation_use_policy": "posthoc_evaluation_only",
    }


def write_paired_spatial_h5_audit(
    fragments_path: str | Path,
    rna_h5_path: str | Path,
    positions_path: str | Path,
    spot_metadata_path: str | Path,
    fragment_semantics: str,
    outdir: str | Path,
) -> dict[str, Path]:
    """Write JSON and one-row TSV outputs for a paired 10x H5 audit."""

    result = audit_paired_spatial_h5_inputs(
        fragments_path,
        rna_h5_path,
        positions_path,
        spot_metadata_path,
        fragment_semantics,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "paired_spatial_h5_schema_audit.json"
    tsv_path = outdir / "paired_spatial_h5_schema_audit.tsv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_row = dict(result)
    for field in ["fragment_chromosomes", "rna_feature_genomes", "posthoc_only_annotation_fields"]:
        table_row[field] = ";".join(result[field])
    pd.DataFrame([table_row]).to_csv(tsv_path, sep="\t", index=False)
    return {"json": json_path, "tsv": tsv_path}
