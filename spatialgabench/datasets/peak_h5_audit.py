"""Schema audit for peak-by-barcode sparse H5 files with spatial metadata."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REQUIRED_SPARSE_KEYS = {"barcodes", "data", "indices", "indptr", "shape"}


def _decode(dataset: h5py.Dataset) -> list[str]:
    values = dataset.asstr()[:]
    return [str(value) for value in values]


def _find_sparse_group(handle: h5py.File) -> h5py.Group:
    candidates = [
        value
        for value in handle.values()
        if isinstance(value, h5py.Group)
        and REQUIRED_SPARSE_KEYS <= set(value.keys())
        and ({"genes", "gene_names"} & set(value.keys()))
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one sparse matrix group, found {len(candidates)}")
    return candidates[0]


def _scan_sparse_values(group: h5py.Group, matrix_rows: int, chunk_size: int) -> dict[str, int]:
    indices = group["indices"]
    data = group["data"]
    out_of_bounds = 0
    nonfinite = 0
    negative = 0
    noninteger = 0
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        index_chunk = np.asarray(indices[start:stop])
        value_chunk = np.asarray(data[start:stop])
        out_of_bounds += int(((index_chunk < 0) | (index_chunk >= matrix_rows)).sum())
        finite = np.isfinite(value_chunk)
        nonfinite += int((~finite).sum())
        negative += int((value_chunk[finite] < 0).sum())
        noninteger += int((value_chunk[finite] != np.floor(value_chunk[finite])).sum())
    return {
        "matrix_index_out_of_bounds_entries": out_of_bounds,
        "matrix_nonfinite_entries": nonfinite,
        "matrix_negative_entries": negative,
        "matrix_noninteger_entries": noninteger,
    }


def _parse_peak(value: str) -> tuple[str, int, int] | None:
    normalized = value.replace("_", "-")
    fields = normalized.rsplit("-", 2)
    if len(fields) != 3:
        return None
    try:
        start = int(fields[1])
        end = int(fields[2])
    except ValueError:
        return None
    if not fields[0] or start < 0 or end <= start:
        return None
    return fields[0], start, end


def audit_peak_h5_spatial_inputs(
    h5_path: str | Path,
    metadata_path: str | Path,
    chunk_size: int = 1_000_000,
) -> dict[str, object]:
    """Audit sparse matrix internals and spatial metadata without creating AnnData."""

    h5_path = Path(h5_path)
    metadata_path = Path(metadata_path)
    with h5py.File(h5_path, "r") as handle:
        group = _find_sparse_group(handle)
        group_name = group.name
        shape_values = tuple(int(value) for value in group["shape"][:])
        if len(shape_values) != 2:
            raise ValueError(f"Expected two matrix dimensions, found {shape_values}")
        matrix_rows, matrix_columns = shape_values
        barcodes = _decode(group["barcodes"])
        feature_key = "genes" if "genes" in group else "gene_names"
        features = _decode(group[feature_key])
        indptr = np.asarray(group["indptr"][:], dtype=np.int64)
        data_entries = int(len(group["data"]))
        index_entries = int(len(group["indices"]))
        sparse_values = _scan_sparse_values(group, matrix_rows, chunk_size)

    parsed_peaks = [_parse_peak(value) for value in features]
    valid_peaks = [value for value in parsed_peaks if value is not None]
    peak_chromosomes = sorted({value[0] for value in valid_peaks})
    metadata = pd.read_csv(metadata_path, sep=";", index_col=0)
    metadata.index = metadata.index.astype(str)
    required_metadata_columns = {"x", "y", "section", "sample", "age"}
    missing_metadata_columns = sorted(required_metadata_columns - set(metadata.columns))
    x = pd.to_numeric(metadata["x"], errors="coerce") if "x" in metadata else pd.Series(dtype=float)
    y = pd.to_numeric(metadata["y"], errors="coerce") if "y" in metadata else pd.Series(dtype=float)

    indptr_valid = (
        len(indptr) == matrix_columns + 1
        and len(indptr) > 0
        and int(indptr[0]) == 0
        and int(indptr[-1]) == data_entries
        and bool(np.all(indptr[1:] >= indptr[:-1]))
    )
    matrix_contract = (
        matrix_rows == len(features)
        and matrix_columns == len(barcodes)
        and data_entries == index_entries
        and indptr_valid
        and all(value == 0 for value in sparse_values.values())
    )
    barcode_contract = (
        len(barcodes) == len(set(barcodes))
        and len(metadata.index) == len(set(metadata.index))
        and set(barcodes) == set(metadata.index)
    )
    metadata_contract = (
        not missing_metadata_columns
        and len(metadata) == len(barcodes)
        and int(x.isna().sum()) == 0
        and int(y.isna().sum()) == 0
    )
    peak_contract = len(valid_peaks) == len(features) and len(features) == len(set(features))
    schema_pass = matrix_contract and barcode_contract and metadata_contract and peak_contract
    return {
        "audit_type": "peak_sparse_h5_spatial",
        "h5_path": str(h5_path),
        "metadata_path": str(metadata_path),
        "matrix_group": group_name,
        "matrix_orientation": "features_by_barcodes",
        "matrix_rows": matrix_rows,
        "matrix_columns": matrix_columns,
        "matrix_data_entries": data_entries,
        "matrix_index_entries": index_entries,
        "matrix_indptr_entries": len(indptr),
        "matrix_indptr_valid": indptr_valid,
        **sparse_values,
        "feature_key": feature_key,
        "feature_rows": len(features),
        "duplicate_features": len(features) - len(set(features)),
        "parseable_peak_rows": len(valid_peaks),
        "parseable_peak_fraction": float(len(valid_peaks) / len(features)) if features else 0.0,
        "peak_chromosomes": peak_chromosomes,
        "barcode_rows": len(barcodes),
        "duplicate_barcodes": len(barcodes) - len(set(barcodes)),
        "metadata_rows": len(metadata),
        "duplicate_metadata_barcodes": len(metadata.index) - len(set(metadata.index)),
        "metadata_missing_required_columns": missing_metadata_columns,
        "metadata_invalid_x_rows": int(x.isna().sum()),
        "metadata_invalid_y_rows": int(y.isna().sum()),
        "matrix_metadata_barcode_overlap": len(set(barcodes) & set(metadata.index)),
        "matrix_metadata_barcodes_equal": set(barcodes) == set(metadata.index),
        "schema_status": "pass" if schema_pass else "fail",
        "barcode_contract_status": "pass" if barcode_contract else "fail",
    }


def write_peak_h5_spatial_audit(
    h5_path: str | Path,
    metadata_path: str | Path,
    outdir: str | Path,
) -> dict[str, Path]:
    """Write JSON and one-row TSV outputs for a peak sparse-H5 audit."""

    result = audit_peak_h5_spatial_inputs(h5_path, metadata_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "peak_h5_spatial_schema_audit.json"
    tsv_path = outdir / "peak_h5_spatial_schema_audit.tsv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_row = dict(result)
    table_row["peak_chromosomes"] = ";".join(result["peak_chromosomes"])
    table_row["metadata_missing_required_columns"] = ";".join(result["metadata_missing_required_columns"])
    pd.DataFrame([table_row]).to_csv(tsv_path, sep="\t", index=False)
    return {"json": json_path, "tsv": tsv_path}
