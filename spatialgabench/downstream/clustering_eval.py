"""Downstream clustering evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


@dataclass(frozen=True)
class PreparedFeature:
    adata: ad.AnnData
    summary: dict[str, Any]


def load_feature_h5ad(path: str | Path) -> ad.AnnData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature h5ad does not exist: {path}")
    return ad.read_h5ad(path)


def validate_obs_alignment(feature_adata: ad.AnnData, raw_adata: ad.AnnData, method_id: str) -> None:
    if feature_adata.n_obs != raw_adata.n_obs:
        raise ValueError(f"{method_id}: n_obs mismatch: feature={feature_adata.n_obs}, raw={raw_adata.n_obs}")
    if not feature_adata.obs_names.equals(raw_adata.obs_names):
        raise ValueError(f"{method_id}: obs_names do not align with raw pilot AnnData")


def validate_spatial(feature_adata: ad.AnnData, raw_adata: ad.AnnData, method_id: str) -> None:
    if "spatial" not in feature_adata.obsm:
        raise ValueError(f"{method_id}: obsm['spatial'] is missing")
    if "spatial" not in raw_adata.obsm:
        raise ValueError("Raw pilot AnnData lacks obsm['spatial']")
    if feature_adata.obsm["spatial"].shape != raw_adata.obsm["spatial"].shape:
        raise ValueError(
            f"{method_id}: spatial shape mismatch: feature={feature_adata.obsm['spatial'].shape}, "
            f"raw={raw_adata.obsm['spatial'].shape}"
        )


def summarize_labels(labels: pd.Series) -> pd.DataFrame:
    counts = labels.astype(str).value_counts(dropna=False).sort_index()
    total = float(counts.sum())
    return pd.DataFrame(
        {
            "truth_label": counts.index.astype(str),
            "n_cells": counts.to_numpy(dtype=np.int64),
            "fraction": counts.to_numpy(dtype=np.float64) / total if total else 0.0,
        }
    )


def prepare_gene_activity_for_clustering(
    input_adata: ad.AnnData,
    *,
    method_id: str,
    preprocessing_config: dict[str, Any],
    random_seed: int,
) -> PreparedFeature:
    if input_adata.n_vars <= 0:
        raise ValueError(f"{method_id}: gene activity output has no genes")
    if not sparse.issparse(input_adata.X):
        raise ValueError(f"{method_id}: gene activity X must be sparse")
    eval_adata = ad.AnnData(X=input_adata.X.copy(), obs=pd.DataFrame(index=input_adata.obs_names), var=input_adata.var.copy())
    if "spatial" in input_adata.obsm:
        eval_adata.obsm["spatial"] = input_adata.obsm["spatial"].copy()

    summary: dict[str, Any] = {
        "method_id": method_id,
        "feature_type": "gene_activity",
        "input_n_cells": int(input_adata.n_obs),
        "input_n_features": int(input_adata.n_vars),
        "normalization_status": "not_started",
        "log1p_status": "not_started",
        "hvg_status": "not_started",
        "hvg_flavor": "",
        "n_hvg_selected": 0,
        "pca_status": "not_started",
        "pca_n_comps_requested": int(preprocessing_config["pca_n_comps"]),
        "pca_n_comps_used": 0,
    }

    target_sum = float(preprocessing_config["normalize_total_target_sum"])
    sc.pp.normalize_total(eval_adata, target_sum=target_sum)
    summary["normalization_status"] = f"success_normalize_total_target_sum_{target_sum:g}"
    if bool(preprocessing_config.get("log1p", True)):
        sc.pp.log1p(eval_adata)
        summary["log1p_status"] = "success"
    else:
        summary["log1p_status"] = "skipped_by_config"

    n_top = min(int(preprocessing_config["hvg_n_top_genes"]), eval_adata.n_vars)
    hvg_errors: list[str] = []
    selected = np.ones(eval_adata.n_vars, dtype=bool)
    for flavor in preprocessing_config.get("hvg_flavors", ["seurat_v3", "seurat"]):
        try:
            sc.pp.highly_variable_genes(eval_adata, n_top_genes=n_top, flavor=flavor)
            if "highly_variable" in eval_adata.var.columns and bool(eval_adata.var["highly_variable"].any()):
                selected = eval_adata.var["highly_variable"].to_numpy(dtype=bool)
                summary["hvg_status"] = "success"
                summary["hvg_flavor"] = flavor
                break
            hvg_errors.append(f"{flavor}: no genes marked highly_variable")
        except Exception as exc:  # pragma: no cover - exact scanpy dependency failures vary by environment.
            hvg_errors.append(f"{flavor}: {type(exc).__name__}: {exc}")
    if summary["hvg_status"] != "success":
        summary["hvg_status"] = "skipped_after_fallback_failure"
        summary["hvg_flavor"] = "none"
        summary["hvg_error"] = " | ".join(hvg_errors)
    else:
        eval_adata = eval_adata[:, selected].copy()
    summary["n_hvg_selected"] = int(eval_adata.n_vars)

    n_comps = min(int(preprocessing_config["pca_n_comps"]), eval_adata.n_obs - 1, eval_adata.n_vars - 1)
    if n_comps < 1:
        raise ValueError(f"{method_id}: not enough cells/features for PCA after HVG")
    sc.pp.pca(eval_adata, n_comps=n_comps, random_state=random_seed, svd_solver="arpack")
    summary["pca_status"] = "success"
    summary["pca_n_comps_used"] = int(n_comps)
    summary["output_n_features_for_clustering"] = int(eval_adata.n_vars)
    return PreparedFeature(eval_adata, summary)


def prepare_lsi_for_clustering(
    input_adata: ad.AnnData,
    *,
    method_id: str,
    expected_n_components: int,
) -> PreparedFeature:
    if "X_lsi" not in input_adata.obsm:
        raise ValueError(f"{method_id}: obsm['X_lsi'] is missing")
    X_lsi = np.asarray(input_adata.obsm["X_lsi"])
    if X_lsi.ndim != 2 or X_lsi.shape[0] != input_adata.n_obs:
        raise ValueError(f"{method_id}: invalid X_lsi shape {X_lsi.shape}")
    if X_lsi.shape[1] < expected_n_components:
        raise ValueError(f"{method_id}: X_lsi has {X_lsi.shape[1]} components, expected at least {expected_n_components}")
    if not np.isfinite(X_lsi[:, :expected_n_components]).all():
        raise ValueError(f"{method_id}: X_lsi contains non-finite values")
    eval_adata = ad.AnnData(X=sparse.csr_matrix((input_adata.n_obs, 0), dtype=np.float32), obs=pd.DataFrame(index=input_adata.obs_names))
    eval_adata.obsm["X_lsi"] = X_lsi[:, :expected_n_components].copy()
    if "spatial" in input_adata.obsm:
        eval_adata.obsm["spatial"] = input_adata.obsm["spatial"].copy()
    return PreparedFeature(
        eval_adata,
        {
            "method_id": method_id,
            "feature_type": "lsi_embedding",
            "input_n_cells": int(input_adata.n_obs),
            "input_n_features": int(input_adata.n_vars),
            "normalization_status": "not_applicable_lsi_embedding",
            "log1p_status": "not_applicable_lsi_embedding",
            "hvg_status": "not_applicable_lsi_embedding",
            "hvg_flavor": "not_applicable",
            "n_hvg_selected": 0,
            "pca_status": "not_applicable_lsi_embedding",
            "pca_n_comps_requested": int(expected_n_components),
            "pca_n_comps_used": int(expected_n_components),
            "output_n_features_for_clustering": int(expected_n_components),
        },
    )


def run_pca_neighbors_leiden(
    eval_adata: ad.AnnData,
    *,
    n_neighbors: int,
    n_pcs: int,
    random_seed: int,
) -> None:
    if "X_pca" not in eval_adata.obsm:
        raise ValueError("X_pca is missing; run PCA before neighbors")
    n_pcs_used = min(int(n_pcs), eval_adata.obsm["X_pca"].shape[1])
    sc.pp.neighbors(eval_adata, n_neighbors=int(n_neighbors), n_pcs=n_pcs_used, random_state=random_seed)


def run_lsi_neighbors_leiden(
    eval_adata: ad.AnnData,
    *,
    n_neighbors: int,
    random_seed: int,
) -> None:
    if "X_lsi" not in eval_adata.obsm:
        raise ValueError("X_lsi is missing; cannot build LSI neighbors")
    sc.pp.neighbors(eval_adata, n_neighbors=int(n_neighbors), use_rep="X_lsi", random_state=random_seed)


def run_leiden(
    eval_adata: ad.AnnData,
    *,
    resolution: float,
    random_seed: int,
    key_added: str,
) -> pd.Series:
    sc.tl.leiden(eval_adata, resolution=float(resolution), random_state=random_seed, key_added=key_added)
    return eval_adata.obs[key_added].astype(str).copy()


def select_resolution_by_k(search_summary: pd.DataFrame, truth_k: int) -> pd.Series:
    required = {"resolution", "number_of_predicted_clusters"}
    missing = required.difference(search_summary.columns)
    if missing:
        raise ValueError(f"resolution search summary missing columns: {sorted(missing)}")
    candidates = search_summary.copy()
    candidates["cluster_count_abs_diff_to_truth_k"] = (
        candidates["number_of_predicted_clusters"].astype(int) - int(truth_k)
    ).abs()
    candidates = candidates.sort_values(["cluster_count_abs_diff_to_truth_k", "resolution"], ascending=[True, True])
    return candidates.iloc[0]


def search_resolution_by_k(
    eval_adata: ad.AnnData,
    *,
    resolution_grid: list[float],
    truth_k: int,
    random_seed: int,
    method_id: str,
) -> tuple[float, pd.Series, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    predictions_by_resolution: dict[float, pd.Series] = {}
    for resolution in resolution_grid:
        key = f"leiden_kaware_{str(resolution).replace('.', '_')}"
        predicted = run_leiden(eval_adata, resolution=float(resolution), random_seed=random_seed, key_added=key)
        n_clusters = int(predicted.nunique())
        predictions_by_resolution[float(resolution)] = predicted
        records.append(
            {
                "method_id": method_id,
                "setting": "k_aware",
                "resolution": float(resolution),
                "number_of_predicted_clusters": n_clusters,
                "number_of_truth_clusters": int(truth_k),
                "cluster_count_abs_diff_to_truth_k": abs(n_clusters - int(truth_k)),
                "selection_rule": "minimize_abs_predicted_cluster_count_minus_truth_k_tie_choose_lower_resolution",
            }
        )
    frame = pd.DataFrame(records)
    chosen_row = select_resolution_by_k(frame, truth_k=int(truth_k))
    chosen_resolution = float(chosen_row["resolution"])
    return chosen_resolution, predictions_by_resolution[chosen_resolution], frame


def compute_clustering_metrics(truth_labels: pd.Series | np.ndarray, predicted_labels: pd.Series | np.ndarray) -> dict[str, Any]:
    truth = pd.Series(truth_labels).astype(str).to_numpy()
    predicted = pd.Series(predicted_labels).astype(str).to_numpy()
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError(f"truth/predicted length mismatch: {truth.shape[0]} vs {predicted.shape[0]}")
    if pd.isna(predicted).any():
        raise ValueError("predicted labels contain NaN")
    cluster_sizes = pd.Series(predicted).value_counts().to_numpy(dtype=np.int64)
    return {
        "ARI": float(adjusted_rand_score(truth, predicted)),
        "NMI": float(normalized_mutual_info_score(truth, predicted)),
        "AMI": float(adjusted_mutual_info_score(truth, predicted)),
        "homogeneity": float(homogeneity_score(truth, predicted)),
        "completeness": float(completeness_score(truth, predicted)),
        "v_measure": float(v_measure_score(truth, predicted)),
        "number_of_predicted_clusters": int(pd.Series(predicted).nunique()),
        "number_of_truth_clusters": int(pd.Series(truth).nunique()),
        "cluster_size_min": int(cluster_sizes.min()) if cluster_sizes.size else 0,
        "cluster_size_median": float(np.median(cluster_sizes)) if cluster_sizes.size else 0.0,
        "cluster_size_max": int(cluster_sizes.max()) if cluster_sizes.size else 0,
    }


def predictions_frame(
    *,
    barcodes: pd.Index,
    truth_labels: pd.Series,
    predicted_labels: pd.Series,
    method_id: str,
    setting: str,
    resolution: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "barcode": barcodes.astype(str).to_numpy(),
            "truth_label": truth_labels.astype(str).to_numpy(),
            "predicted_cluster": predicted_labels.astype(str).to_numpy(),
            "method_id": method_id,
            "setting": setting,
            "resolution": float(resolution),
        }
    )


def write_clustering_outputs(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
