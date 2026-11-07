"""Lossless streaming adapter for SnapATAC2 fragment-backed AnnData."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_fragment_h5ad(
    h5ad_path: str | Path,
    fragments_path: str | Path,
    positions_path: str | Path,
    audit_path: str | Path,
) -> dict[str, object]:
    """Export encoded fragment starts/lengths and spatial coordinates without loading them in memory."""

    h5ad_path = Path(h5ad_path)
    fragments_path = Path(fragments_path)
    positions_path = Path(positions_path)
    audit_path = Path(audit_path)
    for path in (fragments_path, positions_path, audit_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    temporary_fragments = fragments_path.with_name(f".{fragments_path.name}.tmp")
    temporary_positions = positions_path.with_name(f".{positions_path.name}.tmp")
    fragment_rows = 0
    invalid_rows = 0
    chromosome_rows: dict[str, int] = {}

    with h5py.File(h5ad_path, "r") as handle:
        required = [
            "obs/_index",
            "obs/use_spot",
            "obsm/fragment_paired/data",
            "obsm/fragment_paired/indices",
            "obsm/fragment_paired/indptr",
            "obsm/spatial",
            "uns/reference_sequences/reference_seq_name",
            "uns/reference_sequences/reference_seq_length",
        ]
        missing = [path for path in required if path not in handle]
        if missing:
            raise ValueError(f"Fragment-backed h5ad is missing required paths: {missing}")

        observation_ids = handle["obs/_index"].asstr()[:]
        use_spot = np.asarray(handle["obs/use_spot"][:])
        spatial = np.asarray(handle["obsm/spatial"][:])
        fragment_group = handle["obsm/fragment_paired"]
        shape = tuple(int(value) for value in fragment_group.attrs.get("shape", ()))
        data = fragment_group["data"]
        indices = fragment_group["indices"]
        indptr = np.asarray(fragment_group["indptr"][:], dtype=np.int64)
        chromosome_names = handle["uns/reference_sequences/reference_seq_name"].asstr()[:]
        chromosome_lengths = np.asarray(
            handle["uns/reference_sequences/reference_seq_length"][:], dtype=np.int64
        )
        chromosome_ends = np.cumsum(chromosome_lengths, dtype=np.int64)
        chromosome_starts = np.concatenate((np.asarray([0], dtype=np.int64), chromosome_ends[:-1]))

        n_observations = len(observation_ids)
        if len(shape) != 2 or shape != (n_observations, int(chromosome_ends[-1])):
            raise ValueError("Fragment matrix shape does not match observations and reference lengths")
        if spatial.shape != (n_observations, 2) or not np.isfinite(spatial).all():
            raise ValueError("Spatial coordinates do not satisfy the n_observations x 2 finite contract")
        if len(indptr) != n_observations + 1 or indptr[0] != 0 or indptr[-1] != len(indices):
            raise ValueError("Fragment CSR indptr does not satisfy the observation contract")
        if len(indices) != len(data):
            raise ValueError("Fragment CSR indices and data lengths differ")
        if len(set(observation_ids)) != n_observations:
            raise ValueError("Fragment-backed h5ad contains duplicate observation IDs")

        positions = pd.DataFrame(
            {
                "barcode": observation_ids,
                "in_tissue": (use_spot != 0).astype(np.int8),
                "array_row": spatial[:, 1],
                "array_col": spatial[:, 0],
                "pixel_row": spatial[:, 1],
                "pixel_col": spatial[:, 0],
            }
        )
        positions.to_csv(temporary_positions, index=False, header=False)

        with temporary_fragments.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as target:
                    for observation_index, barcode in enumerate(observation_ids):
                        row_start = int(indptr[observation_index])
                        row_end = int(indptr[observation_index + 1])
                        starts_global = np.asarray(indices[row_start:row_end], dtype=np.int64)
                        lengths = np.asarray(data[row_start:row_end], dtype=np.int64)
                        chromosome_indexes = np.searchsorted(chromosome_ends, starts_global, side="right")
                        valid_chromosome = chromosome_indexes < len(chromosome_names)
                        local_starts = np.zeros_like(starts_global)
                        local_ends = np.zeros_like(starts_global)
                        local_starts[valid_chromosome] = (
                            starts_global[valid_chromosome]
                            - chromosome_starts[chromosome_indexes[valid_chromosome]]
                        )
                        local_ends[valid_chromosome] = local_starts[valid_chromosome] + lengths[valid_chromosome]
                        valid = (
                            valid_chromosome
                            & (lengths > 0)
                            & (local_starts >= 0)
                            & (local_ends > local_starts)
                        )
                        valid_indexes = np.flatnonzero(valid)
                        if len(valid_indexes):
                            valid[valid_indexes] &= (
                                local_ends[valid_indexes]
                                <= chromosome_lengths[chromosome_indexes[valid_indexes]]
                            )
                        invalid_rows += int((~valid).sum())
                        for entry_index in np.flatnonzero(valid):
                            chromosome = str(chromosome_names[chromosome_indexes[entry_index]])
                            target.write(
                                f"{chromosome}\t{local_starts[entry_index]}\t{local_ends[entry_index]}\t"
                                f"{barcode}\t1\n"
                            )
                            fragment_rows += 1
                            chromosome_rows[chromosome] = chromosome_rows.get(chromosome, 0) + 1

    temporary_fragments.replace(fragments_path)
    temporary_positions.replace(positions_path)
    result: dict[str, object] = {
        "adapter": "snapatac2_fragment_paired_csr_to_fragments_v1",
        "source_h5ad": str(h5ad_path),
        "source_h5ad_sha256": _sha256(h5ad_path),
        "output_fragments": str(fragments_path),
        "output_fragments_sha256": _sha256(fragments_path),
        "output_positions": str(positions_path),
        "output_positions_sha256": _sha256(positions_path),
        "observation_rows": len(observation_ids),
        "in_tissue_observation_rows": int((use_spot != 0).sum()),
        "fragment_rows": fragment_rows,
        "invalid_fragment_rows": invalid_rows,
        "chromosome_rows": chromosome_rows,
        "coordinate_decode": "global_start_to_reference_local_start_plus_stored_length",
        "spatial_decode": "obsm_spatial_col0_to_pixel_col_col1_to_pixel_row",
        "construction_leakage_inputs": "none",
        "status": "pass" if fragment_rows > 0 and invalid_rows == 0 else "fail",
    }
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat_result = {key: value for key, value in result.items() if not isinstance(value, dict)}
    pd.DataFrame([flat_result]).to_csv(audit_path.with_suffix(".tsv"), sep="\t", index=False)
    return result
