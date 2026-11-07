"""I/O helpers for SpatialGABench."""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


DATASET_ID_GSE214986 = "gse214986_spatial_atac_mouse_embryo"


def _decode_array(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def _read_feature_table(features_group: h5py.Group) -> pd.DataFrame:
    columns: dict[str, list[str]] = {}
    for key, dataset in features_group.items():
        if not isinstance(dataset, h5py.Dataset) or dataset.shape == ():
            continue
        if len(dataset.shape) == 1:
            columns[key] = _decode_array(dataset[:])
    if not columns:
        raise ValueError("H5 features group does not contain one-dimensional feature ID/name datasets.")
    table = pd.DataFrame(columns)
    if "id" in table.columns:
        table.index = table["id"].astype(str)
    elif "name" in table.columns:
        table.index = table["name"].astype(str)
    else:
        first = next(iter(table.columns))
        table.index = table[first].astype(str)
    table.index.name = "peak_id"
    return table


def _feature_table_from_arrays(ids: list[str], names: list[str] | None = None) -> pd.DataFrame:
    table = pd.DataFrame({"id": ids})
    if names is not None:
        table["name"] = names
    table.index = table["id"].astype(str)
    table.index.name = "peak_id"
    return table


def _read_sparse_group(group: h5py.Group) -> tuple[sparse.csc_matrix, tuple[int, int]]:
    required = {"data", "indices", "indptr", "shape"}
    missing = sorted(required.difference(group.keys()))
    if missing:
        raise ValueError(f"Sparse matrix group is missing required keys: {missing}")
    shape = tuple(int(x) for x in group["shape"][:])
    csc = sparse.csc_matrix(
        (
            group["data"][:],
            group["indices"][:],
            group["indptr"][:],
        ),
        shape=shape,
    )
    return csc, shape


def _read_10x_like_h5(path: Path) -> tuple[sparse.csr_matrix, list[str], pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if "matrix" in handle:
            matrix_group = handle["matrix"]
            if "barcodes" not in matrix_group or "features" not in matrix_group:
                raise ValueError("H5 /matrix group lacks barcodes or features.")
            csc, _ = _read_sparse_group(matrix_group)
            barcodes = _decode_array(matrix_group["barcodes"][:])
            var = _read_feature_table(matrix_group["features"])
        else:
            sparse_groups = [
                key
                for key, value in handle.items()
                if isinstance(value, h5py.Group)
                and {"data", "indices", "indptr", "shape", "barcodes"}.issubset(value.keys())
                and ("genes" in value.keys() or "gene_names" in value.keys())
            ]
            if len(sparse_groups) != 1:
                raise ValueError(f"H5 does not contain a unique supported sparse count group: {sparse_groups}")
            matrix_group = handle[sparse_groups[0]]
            csc, _ = _read_sparse_group(matrix_group)
            barcodes = _decode_array(matrix_group["barcodes"][:])
            ids = _decode_array(matrix_group["genes"][:]) if "genes" in matrix_group else _decode_array(matrix_group["gene_names"][:])
            names = _decode_array(matrix_group["gene_names"][:]) if "gene_names" in matrix_group else None
            var = _feature_table_from_arrays(ids, names)

    if len(set(barcodes)) != len(barcodes):
        raise ValueError("H5 contains duplicate barcodes; refusing ambiguous import.")
    if var.index.duplicated().any():
        raise ValueError("H5 contains duplicate peak/feature IDs; refusing ambiguous import.")

    orientation_note: str
    if csc.shape == (var.shape[0], len(barcodes)):
        X = csc.T.tocsr()
        orientation_note = "H5 sparse matrix was features x barcodes and was transposed to cells/spots x peaks."
    elif csc.shape == (len(barcodes), var.shape[0]):
        X = csc.tocsr()
        orientation_note = "H5 sparse matrix was already cells/spots x peaks."
    else:
        raise ValueError(
            "H5 matrix shape does not align with feature and barcode counts: "
            f"shape={csc.shape}, n_features={var.shape[0]}, n_barcodes={len(barcodes)}"
        )
    return X, barcodes, var, orientation_note


def _parse_peak_coordinates(var: pd.DataFrame) -> pd.DataFrame:
    var = var.copy()
    if "peak_id" not in var.columns:
        var["peak_id"] = var.index.astype(str)
    source = var["peak_id"].astype(str)
    chroms: list[str | None] = []
    starts: list[int | None] = []
    ends: list[int | None] = []
    for peak_id in source:
        normalized = peak_id.replace(":", "-").replace("_", "-")
        match = re.match(r"^(chr[^-]+)-([0-9]+)-([0-9]+)$", normalized)
        if match:
            chroms.append(match.group(1))
            starts.append(int(match.group(2)))
            ends.append(int(match.group(3)))
        else:
            chroms.append(None)
            starts.append(None)
            ends.append(None)
    var["chr"] = chroms
    var["start"] = pd.Series(starts, index=var.index, dtype="Int64")
    var["end"] = pd.Series(ends, index=var.index, dtype="Int64")
    return var


def _load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    best: pd.DataFrame | None = None
    for sep in [",", ";", "\t"]:
        candidate = pd.read_csv(path, sep=sep, index_col=0)
        if best is None or candidate.shape[1] > best.shape[1]:
            best = candidate
    if best is None or best.shape[1] == 0:
        raise ValueError("Metadata CSV could not be parsed into columns.")
    metadata = best
    metadata.index = metadata.index.astype(str)
    if metadata.index.duplicated().any():
        raise ValueError("Metadata contains duplicate index values; refusing ambiguous import.")
    return metadata


def _metadata_with_barcode_index(metadata: pd.DataFrame, barcodes: list[str]) -> pd.DataFrame:
    barcode_set = set(barcodes)
    if set(metadata.index) == barcode_set:
        return metadata.loc[barcodes].copy()

    candidates: list[str] = []
    for column in metadata.columns:
        lower = str(column).lower()
        if lower in {"barcode", "barcodes", "cell", "cell_id", "spot", "spot_id"} or "barcode" in lower:
            candidates.append(str(column))
    for column in candidates:
        values = metadata[column].astype(str)
        if len(set(values)) != len(values):
            continue
        if set(values) == barcode_set:
            aligned = metadata.copy()
            aligned.index = values
            return aligned.loc[barcodes].copy()

    overlaps: dict[str, int] = {"__index__": len(set(metadata.index).intersection(barcode_set))}
    for column in candidates:
        overlaps[column] = len(set(metadata[column].astype(str)).intersection(barcode_set))
    raise ValueError(
        "Metadata cannot be exactly aligned to H5 barcodes. "
        f"n_h5_barcodes={len(barcodes)}, n_metadata_rows={metadata.shape[0]}, overlaps={overlaps}"
    )


def _choose_spatial_columns(obs: pd.DataFrame) -> tuple[str, str]:
    exact_pairs = [
        ("x", "y"),
        ("X", "Y"),
        ("spatial_x", "spatial_y"),
        ("array_col", "array_row"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("imagecol", "imagerow"),
    ]
    for x_col, y_col in exact_pairs:
        if x_col in obs.columns and y_col in obs.columns:
            return x_col, y_col

    numeric_columns = [column for column in obs.columns if pd.api.types.is_numeric_dtype(obs[column])]
    lowered = {str(column).lower(): column for column in numeric_columns}
    for x_token in ["xcoord", "x_coord", "coord_x", "spatialx", "spatial_x", "x"]:
        for y_token in ["ycoord", "y_coord", "coord_y", "spatialy", "spatial_y", "y"]:
            if x_token in lowered and y_token in lowered:
                return str(lowered[x_token]), str(lowered[y_token])

    x_candidates = [c for c in numeric_columns if str(c).lower() in {"x", "xpos", "x_position"} or "coord" in str(c).lower() and "x" in str(c).lower()]
    y_candidates = [c for c in numeric_columns if str(c).lower() in {"y", "ypos", "y_position"} or "coord" in str(c).lower() and "y" in str(c).lower()]
    if x_candidates and y_candidates:
        return str(x_candidates[0]), str(y_candidates[0])
    raise ValueError("Spatial coordinate columns are missing or not recognized in metadata.")


def convert_gse214986_pilot(h5_path: Path, metadata_path: Path) -> ad.AnnData:
    """Read GSE214986 processed peak counts and metadata into standard AnnData."""

    X, barcodes, var, orientation_note = _read_10x_like_h5(h5_path)
    metadata = _load_metadata(metadata_path)
    obs = _metadata_with_barcode_index(metadata, barcodes)
    x_col, y_col = _choose_spatial_columns(obs)
    spatial = obs[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(spatial).any():
        raise ValueError(f"Spatial coordinate columns contain missing/non-numeric values: {x_col}, {y_col}")

    var = _parse_peak_coordinates(var)
    parseable_peak_fraction = float(var[["chr", "start", "end"]].notna().all(axis=1).mean())
    if parseable_peak_fraction == 0:
        raise ValueError("Peak IDs are present but none could be parsed into chr/start/end coordinates.")

    adata = ad.AnnData(X=X, obs=obs.copy(), var=var.copy())
    adata.obs_names = barcodes
    adata.var_names = var.index.astype(str)
    adata.obsm["spatial"] = spatial
    adata.uns["dataset_id"] = DATASET_ID_GSE214986
    adata.uns["modality"] = "spatial_atac"
    adata.uns["genome_build"] = "unknown"
    adata.uns["source_accession"] = "GSE214986"
    adata.uns["processing_notes"] = (
        f"{orientation_note} Metadata aligned exactly to H5 barcodes. "
        f"Spatial coordinates read from metadata columns `{x_col}` and `{y_col}`. "
        f"Peak coordinate parseable fraction: {parseable_peak_fraction:.4f}."
    )
    adata.uns["spatial_coordinate_columns"] = [x_col, y_col]
    adata.uns["peak_coordinate_parseable_fraction"] = parseable_peak_fraction
    return adata
