"""Schema audit for SnapATAC2-style fragment-backed AnnData files."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _infer_build(reference_lengths: dict[str, int]) -> str:
    signatures = {
        "GRCh38": {"chr1": 248_956_422, "chr2": 242_193_529, "chrX": 156_040_895},
        "GRCm38_mm10": {"chr1": 195_471_971, "chr2": 182_113_224, "chrX": 171_031_299},
        "GRCm39_mm39": {"chr1": 195_154_279, "chr2": 181_755_017, "chrX": 169_476_592},
    }
    matches = [
        build
        for build, signature in signatures.items()
        if all(reference_lengths.get(chromosome) == length for chromosome, length in signature.items())
    ]
    return matches[0] if len(matches) == 1 else "unresolved"


def audit_fragment_h5ad_inputs(
    h5ad_path: str | Path,
    chunk_size: int = 1_000_000,
) -> dict[str, object]:
    """Audit a fragment-backed h5ad without materializing its sparse matrix."""

    h5ad_path = Path(h5ad_path)
    with h5py.File(h5ad_path, "r") as handle:
        encoding_type = str(handle.attrs.get("encoding-type", ""))
        required_paths = [
            "obs/_index",
            "obsm/fragment_paired/data",
            "obsm/fragment_paired/indices",
            "obsm/fragment_paired/indptr",
            "obsm/spatial",
            "uns/reference_sequences/reference_seq_name",
            "uns/reference_sequences/reference_seq_length",
        ]
        missing_paths = [path for path in required_paths if path not in handle]
        if missing_paths:
            return {
                "audit_type": "fragment_backed_h5ad",
                "h5ad_path": str(h5ad_path),
                "missing_required_paths": missing_paths,
                "schema_status": "fail",
            }

        observation_ids = [str(value) for value in handle["obs/_index"].asstr()[:]]
        spatial = np.asarray(handle["obsm/spatial"][:])
        fragment_group = handle["obsm/fragment_paired"]
        fragment_shape = tuple(int(value) for value in fragment_group.attrs.get("shape", ()))
        fragment_data = fragment_group["data"]
        fragment_indices = fragment_group["indices"]
        fragment_entry_count = int(fragment_data.shape[0])
        fragment_index_count = int(fragment_indices.shape[0])
        fragment_indptr = np.asarray(fragment_group["indptr"][:], dtype=np.int64)
        reference_names = [str(value) for value in handle["uns/reference_sequences/reference_seq_name"].asstr()[:]]
        reference_values = [int(value) for value in handle["uns/reference_sequences/reference_seq_length"][:]]

        out_of_bounds = 0
        zero_or_negative_lengths = 0
        for start in range(0, len(fragment_indices), chunk_size):
            stop = min(start + chunk_size, len(fragment_indices))
            index_chunk = np.asarray(fragment_indices[start:stop])
            length_chunk = np.asarray(fragment_data[start:stop])
            if len(fragment_shape) == 2:
                out_of_bounds += int(((index_chunk < 0) | (index_chunk >= fragment_shape[1])).sum())
            zero_or_negative_lengths += int((length_chunk <= 0).sum())

        use_spot_count: int | None = None
        if "obs/use_spot" in handle:
            use_spot_count = int((np.asarray(handle["obs/use_spot"][:]) != 0).sum())

    reference_lengths = dict(zip(reference_names, reference_values, strict=True))
    reference_total_length = sum(reference_values)
    inferred_build = _infer_build(reference_lengths)
    n_observations = len(observation_ids)
    n_entries = fragment_entry_count
    indptr_valid = (
        len(fragment_indptr) == n_observations + 1
        and len(fragment_indptr) > 0
        and int(fragment_indptr[0]) == 0
        and int(fragment_indptr[-1]) == n_entries
        and bool(np.all(fragment_indptr[1:] >= fragment_indptr[:-1]))
    )
    shape_valid = (
        len(fragment_shape) == 2
        and fragment_shape[0] == n_observations
        and fragment_shape[1] == reference_total_length
    )
    spatial_valid = spatial.shape == (n_observations, 2) and bool(np.isfinite(spatial).all())
    schema_pass = (
        encoding_type == "anndata"
        and not missing_paths
        and len(observation_ids) == len(set(observation_ids))
        and len(reference_names) == len(set(reference_names))
        and fragment_entry_count == fragment_index_count
        and indptr_valid
        and shape_valid
        and spatial_valid
        and out_of_bounds == 0
        and zero_or_negative_lengths == 0
        and inferred_build != "unresolved"
    )
    return {
        "audit_type": "fragment_backed_h5ad",
        "h5ad_path": str(h5ad_path),
        "anndata_encoding_type": encoding_type,
        "missing_required_paths": missing_paths,
        "observation_rows": n_observations,
        "duplicate_observation_ids": n_observations - len(set(observation_ids)),
        "spatial_rows": int(spatial.shape[0]) if spatial.ndim == 2 else 0,
        "spatial_columns": int(spatial.shape[1]) if spatial.ndim == 2 else 0,
        "spatial_nonfinite_values": int((~np.isfinite(spatial)).sum()),
        "duplicate_spatial_coordinates": int(len(spatial) - len(np.unique(spatial, axis=0))),
        "use_spot_count": use_spot_count,
        "fragment_matrix_rows": fragment_shape[0] if len(fragment_shape) == 2 else 0,
        "fragment_genome_axis_length": fragment_shape[1] if len(fragment_shape) == 2 else 0,
        "fragment_entries": n_entries,
        "fragment_index_entries": fragment_index_count,
        "fragment_indptr_entries": len(fragment_indptr),
        "fragment_indptr_valid": indptr_valid,
        "fragment_shape_valid": shape_valid,
        "fragment_index_out_of_bounds_entries": out_of_bounds,
        "fragment_zero_or_negative_lengths": zero_or_negative_lengths,
        "reference_sequence_count": len(reference_names),
        "reference_sequence_names": reference_names,
        "reference_total_length": reference_total_length,
        "inferred_genome_build": inferred_build,
        "schema_status": "pass" if schema_pass else "fail",
        "spatial_contract_status": "pass" if spatial_valid else "fail",
    }


def write_fragment_h5ad_audit(h5ad_path: str | Path, outdir: str | Path) -> dict[str, Path]:
    """Write JSON and one-row TSV outputs for a fragment-backed h5ad audit."""

    result = audit_fragment_h5ad_inputs(h5ad_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "fragment_h5ad_schema_audit.json"
    tsv_path = outdir / "fragment_h5ad_schema_audit.tsv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_row = dict(result)
    table_row["missing_required_paths"] = ";".join(result["missing_required_paths"])
    table_row["reference_sequence_names"] = ";".join(result.get("reference_sequence_names", []))
    pd.DataFrame([table_row]).to_csv(tsv_path, sep="\t", index=False)
    return {"json": json_path, "tsv": tsv_path}
