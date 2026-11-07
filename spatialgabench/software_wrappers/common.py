"""Shared helpers for software gene-activity wrappers."""

from __future__ import annotations

import hashlib
import platform
import resource
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse


REQUIRED_OUTPUT_UNS = [
    "run_id",
    "method_id",
    "software_name",
    "software_version",
    "wrapper_version",
    "input_dataset_id",
    "genome_build",
    "normalization_status",
    "study_id",
    "sample_id",
    "representation_id",
    "input_track",
    "comparison_scope",
    "subset_policy",
    "construction_input_classes",
    "construction_leakage_inputs",
    "input_validation_sha256",
    "environment_lock_sha256",
]

# Downstream variance and covariance calculations square activity values. Values
# above this threshold are finite in float64 but already unsafe for those tasks.
MAX_SAFE_ACTIVITY_VALUE = float(np.sqrt(np.finfo(np.float64).max))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(module_name: str, python_executable: str | None = None) -> str:
    code = (
        "import importlib.metadata as m\n"
        f"print(m.version('{module_name}'))\n"
    )
    try:
        if python_executable:
            result = subprocess.run([python_executable, "-c", code], check=True, capture_output=True, text=True)
            return result.stdout.strip()
        import importlib.metadata as metadata

        return metadata.version(module_name)
    except Exception:
        return "not_installed_or_unknown"


def ensure_standard_gene_activity_anndata(
    *,
    input_adata: ad.AnnData,
    X: sparse.spmatrix,
    var: pd.DataFrame,
    method_id: str,
    software_name: str,
    software_version: str,
    wrapper_version: str,
    genome_build: str,
    normalization_status: str,
    output_path: Path,
    extra_uns: dict[str, Any] | None = None,
) -> ad.AnnData:
    if not sparse.issparse(X):
        raise ValueError("Software wrapper output X must be sparse.")
    if X.shape[0] != input_adata.n_obs:
        raise ValueError(f"Output n_obs {X.shape[0]} does not match input n_obs {input_adata.n_obs}.")
    if X.shape[1] != var.shape[0]:
        raise ValueError(f"Output n_vars {X.shape[1]} does not match var rows {var.shape[0]}.")
    if var.index.has_duplicates:
        raise ValueError("Output var index must be unique.")
    if "gene_name" not in var.columns:
        var = var.copy()
        var["gene_name"] = var.index.astype(str)
    matrix = X.tocsr().copy()
    if matrix.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"Software wrapper output X must be numeric, observed {matrix.dtype}.")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.data.size:
        values = matrix.data.astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Software wrapper output X contains non-finite values.")
        if (values < 0).any():
            raise ValueError("Software wrapper output X contains negative values.")
        if float(np.max(values)) > MAX_SAFE_ACTIVITY_VALUE:
            raise ValueError(
                "Software wrapper output X exceeds the float64-safe magnitude for downstream second moments."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64)
            column_sums = np.asarray(matrix.sum(axis=0), dtype=np.float64)
            total = float(matrix.sum())
        if not (np.isfinite(row_sums).all() and np.isfinite(column_sums).all() and np.isfinite(total)):
            raise ValueError("Software wrapper output X has non-finite aggregate sums.")
    output = ad.AnnData(X=matrix, obs=input_adata.obs.copy(), var=var.copy())
    output.obs_names = input_adata.obs_names.copy()
    if "spatial" in input_adata.obsm:
        output.obsm["spatial"] = input_adata.obsm["spatial"].copy()
    input_dataset_id = str(input_adata.uns.get("dataset_id", "unknown"))
    output.uns["method_id"] = method_id
    output.uns["gene_activity_method"] = method_id
    output.uns["software_name"] = software_name
    output.uns["software_version"] = software_version
    output.uns["wrapper_version"] = wrapper_version
    output.uns["input_dataset_id"] = input_dataset_id
    output.uns["dataset_id"] = input_dataset_id
    output.uns["genome_build"] = genome_build
    output.uns["normalization_status"] = normalization_status
    output.uns["source_modality"] = str(input_adata.uns.get("modality", "spatial_atac"))
    output.uns["is_gene_activity"] = True
    for key, value in (extra_uns or {}).items():
        output.uns[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(output_path, compression="gzip")
    return output


def validate_standard_output(output: ad.AnnData, input_adata: ad.AnnData, method_id: str) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(check_name: str, passed: bool, expected: str, observed: Any, notes: str = "") -> None:
        checks.append(
            {
                "method_id": method_id,
                "check_name": check_name,
                "status": "pass" if passed else "fail",
                "expected": expected,
                "observed": str(observed),
                "notes": notes,
            }
        )

    add("x_is_sparse", sparse.issparse(output.X), "sparse matrix", type(output.X).__name__)
    add("n_obs_aligned", output.n_obs == input_adata.n_obs, str(input_adata.n_obs), output.n_obs)
    add("obs_names_aligned", output.obs_names.equals(input_adata.obs_names), "same obs_names", output.obs_names.equals(input_adata.obs_names))
    add("n_vars_positive", output.n_vars > 0, ">0 genes", output.n_vars)
    add("var_names_unique", output.var_names.is_unique, "unique gene IDs", output.var_names.is_unique)
    add("gene_name_available", "gene_name" in output.var.columns, "var['gene_name']", list(output.var.columns))
    gene_names_nonempty = (
        "gene_name" in output.var.columns
        and bool(output.var["gene_name"].astype(str).str.strip().ne("").all())
    )
    add("gene_names_nonempty", gene_names_nonempty, "non-empty gene names", gene_names_nonempty)
    spatial_preserved = "spatial" not in input_adata.obsm or (
        "spatial" in output.obsm
        and output.obsm["spatial"].shape == input_adata.obsm["spatial"].shape
        and bool(np.allclose(output.obsm["spatial"], input_adata.obsm["spatial"], equal_nan=True))
    )
    add("spatial_preserved", spatial_preserved, "identical spatial matrix if present", spatial_preserved)
    missing_uns = [key for key in REQUIRED_OUTPUT_UNS if key not in output.uns]
    add("required_uns_keys", not missing_uns, "all required uns keys", missing_uns)
    add("method_id_matches", output.uns.get("method_id") == method_id, method_id, output.uns.get("method_id"))
    input_dataset_id = str(input_adata.uns.get("dataset_id", "unknown"))
    add(
        "dataset_id_matches",
        str(output.uns.get("input_dataset_id")) == input_dataset_id,
        input_dataset_id,
        output.uns.get("input_dataset_id"),
    )
    add(
        "construction_leakage_absent",
        output.uns.get("construction_leakage_inputs") == "none",
        "none",
        output.uns.get("construction_leakage_inputs"),
    )
    if sparse.issparse(output.X):
        data = output.X.data
        finite = bool(np.isfinite(data).all()) if data.size else True
        nonnegative = bool((data >= 0).all()) if data.size else True
        nonzero_output = bool(data.size and np.any(data > 0))
        stored_zeros = int((data == 0).sum())
        magnitude_safe = bool(not data.size or float(np.max(data)) <= MAX_SAFE_ACTIVITY_VALUE)
        with np.errstate(over="ignore", invalid="ignore"):
            row_sums = np.asarray(output.X.sum(axis=1), dtype=np.float64)
            column_sums = np.asarray(output.X.sum(axis=0), dtype=np.float64)
            total = float(output.X.sum())
        aggregate_sums_finite = bool(
            np.isfinite(row_sums).all() and np.isfinite(column_sums).all() and np.isfinite(total)
        )
    else:
        finite = nonnegative = False
        nonzero_output = False
        stored_zeros = -1
        magnitude_safe = False
        aggregate_sums_finite = False
    add("finite_values", finite, "finite X.data", finite)
    add("nonnegative_values", nonnegative, "nonnegative X.data", nonnegative)
    add("nonzero_output", nonzero_output, "at least one positive score", nonzero_output)
    add("no_stored_zeros", stored_zeros == 0, "0 stored zeros", stored_zeros)
    add(
        "numeric_magnitude_safe",
        magnitude_safe,
        f"maximum <= {MAX_SAFE_ACTIVITY_VALUE:.6e}",
        magnitude_safe,
        "Required so downstream second moments remain representable in float64.",
    )
    add(
        "aggregate_sums_finite",
        aggregate_sums_finite,
        "finite row, column, and total sums",
        aggregate_sums_finite,
    )
    return pd.DataFrame(checks)


def write_runtime(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(path, sep="\t", index=False)


def write_failure(path: Path, record: dict[str, Any]) -> None:
    columns = [
        "dataset_id",
        "method_id",
        "software_name",
        "stage",
        "failure_type",
        "message",
        "traceback",
        "runtime_seconds",
        "peak_memory_mb",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record], columns=columns).to_csv(path, sep="\t", index=False)


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=False, width=120)


@dataclass
class WrapperPaths:
    output_dir: Path
    output_h5ad: Path
    metadata_yaml: Path
    runtime_tsv: Path
    log_txt: Path
    contract_tsv: Path
    failure_tsv: Path


def wrapper_paths(root: Path, dataset_id: str, method_id: str) -> WrapperPaths:
    output_dir = root / "data" / "gene_activity" / dataset_id / method_id
    return WrapperPaths(
        output_dir=output_dir,
        output_h5ad=output_dir / f"{dataset_id}_{method_id}_gene_activity.h5ad",
        metadata_yaml=output_dir / "method_metadata.yaml",
        runtime_tsv=output_dir / "wrapper_runtime.tsv",
        log_txt=output_dir / "wrapper_log.txt",
        contract_tsv=output_dir / "contract_validation.tsv",
        failure_tsv=output_dir / "wrapper_failure.tsv",
    )


def run_with_failure_capture(
    *,
    root: Path,
    input_h5ad: Path,
    method_id: str,
    software_name: str,
    function,
) -> int:
    start = time.time()
    input_adata = ad.read_h5ad(input_h5ad)
    dataset_id = str(input_adata.uns.get("dataset_id", input_h5ad.parent.name))
    paths = wrapper_paths(root, dataset_id, method_id)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = function(input_adata=input_adata, paths=paths, start_time=start)
        if result is None:
            result = {}
        runtime_record = {
            "dataset_id": dataset_id,
            "method_id": method_id,
            "software_name": software_name,
            "runtime_seconds": time.time() - start,
            "peak_memory_mb": peak_memory_mb(),
            "python_version": platform.python_version(),
            "status": "pass",
            **result,
        }
        write_runtime(paths.runtime_tsv, runtime_record)
        return 0
    except Exception as exc:
        message = str(exc)
        trace = traceback.format_exc(limit=12)
        paths.log_txt.write_text(message + "\n\n" + trace, encoding="utf-8")
        write_runtime(
            paths.runtime_tsv,
            {
                "dataset_id": dataset_id,
                "method_id": method_id,
                "software_name": software_name,
                "runtime_seconds": time.time() - start,
                "peak_memory_mb": peak_memory_mb(),
                "status": "fail",
            },
        )
        write_failure(
            paths.failure_tsv,
            {
                "dataset_id": dataset_id,
                "method_id": method_id,
                "software_name": software_name,
                "stage": "software_wrapper",
                "failure_type": type(exc).__name__,
                "message": message,
                "traceback": trace,
                "runtime_seconds": time.time() - start,
                "peak_memory_mb": peak_memory_mb(),
                "status": "fail",
            },
        )
        return 1
