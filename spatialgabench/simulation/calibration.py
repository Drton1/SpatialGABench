"""Development-only accessibility calibration for semi-synthetic simulations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_authorization(calibration: dict[str, Any], project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization_path = project_root / str(calibration["authorization_json"])
    if not authorization_path.exists():
        raise FileNotFoundError(f"Development authorization does not exist: {authorization_path}")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("authorization") != "development_only":
        raise ValueError("Calibration authorization must be development_only")
    if authorization.get("holdout_overlap") is not False:
        raise ValueError("Calibration authorization must explicitly record holdout_overlap=false")
    if authorization.get("study_id") != calibration["source_study_id"]:
        raise ValueError("Calibration study_id does not match development authorization")

    contract_path = project_root / str(authorization["config_path"])
    if not contract_path.exists():
        raise FileNotFoundError(f"Authorization contract does not exist: {contract_path}")
    if sha256_file(contract_path) != authorization.get("config_sha256"):
        raise ValueError("Development authorization contract checksum mismatch")
    import yaml

    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    study_id = str(calibration["source_study_id"])
    if study_id not in set(contract.get("development_studies", [])):
        raise ValueError(f"Calibration study is not declared development-only: {study_id}")
    if study_id in set(contract.get("holdout_studies", [])):
        raise ValueError(f"Calibration study overlaps a frozen holdout: {study_id}")
    return authorization, contract


def _normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    lower = np.nanmin(coords, axis=0)
    span = np.nanmax(coords, axis=0) - lower
    span[span == 0] = 1.0
    return ((coords - lower) / span).astype(np.float32)


def load_empirical_background(
    calibration: dict[str, Any],
    *,
    project_root: str | Path,
    n_cells: int,
    n_features: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample accessibility and coordinates after verifying development authorization.

    Only `X`, `obsm['spatial']`, and provenance identifiers are consumed. RNA,
    labels, clusters, pathology, markers, and benchmark outcomes are not read.
    """

    root = Path(project_root).resolve()
    authorization, contract = _load_authorization(calibration, root)
    source_path = root / str(calibration["source_h5ad"])
    if not source_path.exists():
        raise FileNotFoundError(f"Calibration source does not exist: {source_path}")
    source = ad.read_h5ad(source_path)
    if str(source.uns.get("study_id", "")) != str(calibration["source_study_id"]):
        raise ValueError("Source h5ad study_id does not match calibration config")
    if str(source.uns.get("dataset_id", "")) != str(calibration["source_dataset_id"]):
        raise ValueError("Source h5ad dataset_id does not match calibration config")
    if n_cells > source.n_obs or n_features > source.n_vars:
        raise ValueError(
            f"Calibration source is too small: requested {n_cells}x{n_features}, source is {source.n_obs}x{source.n_vars}"
        )
    if "spatial" not in source.obsm:
        raise ValueError("Calibration source must contain obsm['spatial']")

    X = source.X.tocsr().astype(np.float64) if sparse.issparse(source.X) else sparse.csr_matrix(source.X, dtype=np.float64)
    if X.nnz == 0 or np.any(~np.isfinite(X.data)) or np.any(X.data < 0):
        raise ValueError("Calibration accessibility matrix must be finite, nonnegative, and nonempty")
    coordinates = np.asarray(source.obsm["spatial"], dtype=np.float64)
    anchor = int(rng.integers(0, source.n_obs))
    squared_distance = np.square(coordinates - coordinates[anchor]).sum(axis=1)
    cell_indices = np.argsort(squared_distance, kind="mergesort")[:n_cells]
    sampled = X[cell_indices, :]

    prevalence = np.asarray((sampled > 0).sum(axis=0)).ravel().astype(np.float64) / n_cells
    eligible = np.flatnonzero(prevalence > 0)
    if eligible.size < n_features:
        prevalence_all = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64) / source.n_obs
        eligible = np.flatnonzero(prevalence_all > 0)
        prevalence = prevalence_all
    if eligible.size < n_features:
        raise ValueError(f"Calibration source has only {eligible.size} detected features; {n_features} required")
    weights = np.sqrt(np.maximum(prevalence[eligible], 1.0 / max(1, n_cells)))
    weights = weights / weights.sum()
    feature_indices = np.sort(rng.choice(eligible, size=n_features, replace=False, p=weights))
    background = sampled[:, feature_indices].tocsr()
    library_sizes = np.asarray(X[cell_indices, :].sum(axis=1)).ravel().astype(np.float64)
    profile = {
        "source_h5ad": str(calibration["source_h5ad"]),
        "source_sha256": sha256_file(source_path),
        "authorization_json": str(calibration["authorization_json"]),
        "authorization_sha256": sha256_file(root / str(calibration["authorization_json"])),
        "authorization_contract_sha256": str(authorization["config_sha256"]),
        "source_dataset_id": str(calibration["source_dataset_id"]),
        "source_study_id": str(calibration["source_study_id"]),
        "source_role": "development_only",
        "source_shape": [int(source.n_obs), int(source.n_vars)],
        "sampled_shape": [int(n_cells), int(n_features)],
        "sampled_cell_indices_sha256": hashlib.sha256(np.asarray(cell_indices, dtype=np.int64).tobytes()).hexdigest(),
        "sampled_feature_indices_sha256": hashlib.sha256(np.asarray(feature_indices, dtype=np.int64).tobytes()).hexdigest(),
        "source_library_quantiles": np.quantile(library_sizes, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist(),
        "sampled_background_nnz": int(background.nnz),
        "sampled_background_sparsity": float(1.0 - background.nnz / (n_cells * n_features)),
        "construction_input_classes_used": ["accessibility", "spatial_coordinates"],
        "construction_evaluation_inputs_used": False,
        "forbidden_inputs_used": [],
        "holdout_accessed": False,
        "redistribution_policy": str(calibration["redistribution_policy"]),
        "frozen_contract_gate": str(contract.get("project", {}).get("current_gate", "")),
    }
    return {
        "background": background,
        "coordinates": _normalize_coordinates(coordinates[cell_indices]),
        "library_sizes": library_sizes,
        "profile": profile,
    }
