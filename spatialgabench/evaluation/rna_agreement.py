"""Pseudo-bulk RNA agreement utilities for adjacent RNA references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats


FORBIDDEN_COMPARISONS = {"spot_level", "coordinate_matched", "barcode_matched"}


@dataclass(frozen=True)
class CollapsedMatrix:
    X: sparse.csr_matrix
    gene_names: np.ndarray
    duplicated_gene_names_before_collapse: int
    n_genes_before_collapse: int
    n_genes_after_collapse: int


def load_rna_h5ad(path: str | Path) -> ad.AnnData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RNA h5ad does not exist: {path}")
    return ad.read_h5ad(path)


def load_ga_h5ad(path: str | Path) -> ad.AnnData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gene activity h5ad does not exist: {path}")
    return ad.read_h5ad(path)


def extract_gene_symbols(adata: ad.AnnData, *, gene_key: str = "gene_name") -> pd.Series:
    if gene_key in adata.var.columns:
        symbols = adata.var[gene_key].astype(str)
    elif gene_key == "gene_name":
        symbols = pd.Series(adata.var_names.astype(str), index=adata.var_names)
    else:
        raise KeyError(f"Gene symbol key `{gene_key}` is missing from adata.var")
    symbols = symbols.str.strip()
    if symbols.isna().any() or (symbols == "").any():
        raise ValueError(f"Gene symbol key `{gene_key}` contains missing or empty symbols")
    return symbols.reset_index(drop=True)


def collapse_duplicate_genes_by_sum(X: sparse.spmatrix, gene_symbols: pd.Series | np.ndarray) -> CollapsedMatrix:
    X_csr = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
    genes = pd.Series(gene_symbols, dtype="object").astype(str).str.strip()
    if len(genes) != X_csr.shape[1]:
        raise ValueError(f"gene symbol length {len(genes)} does not match matrix n_vars {X_csr.shape[1]}")
    duplicated = int(genes.duplicated().sum())
    categories = pd.Categorical(genes, categories=sorted(genes.unique()), ordered=False)
    codes = categories.codes
    n_unique = len(categories.categories)
    indicator = sparse.csr_matrix(
        (
            np.ones(len(codes), dtype=np.float64),
            (np.arange(len(codes), dtype=np.int64), codes.astype(np.int64)),
        ),
        shape=(len(codes), n_unique),
    )
    collapsed = (X_csr @ indicator).tocsr()
    return CollapsedMatrix(
        X=collapsed,
        gene_names=np.asarray(categories.categories, dtype=object),
        duplicated_gene_names_before_collapse=duplicated,
        n_genes_before_collapse=int(X_csr.shape[1]),
        n_genes_after_collapse=int(n_unique),
    )


def normalize_log1p_matrix(X: sparse.spmatrix, *, target_sum: float = 10000.0, log1p: bool = True) -> sparse.csr_matrix:
    X_csr = X.tocsr(copy=True).astype(np.float64)
    totals = np.asarray(X_csr.sum(axis=1)).ravel()
    scale = np.zeros_like(totals, dtype=np.float64)
    nonzero = totals > 0
    scale[nonzero] = float(target_sum) / totals[nonzero]
    X_norm = sparse.diags(scale).dot(X_csr).tocsr()
    if log1p:
        X_norm.data = np.log1p(X_norm.data)
    return X_norm


def make_pseudobulk_vector(X_norm: sparse.spmatrix) -> np.ndarray:
    if X_norm.shape[0] == 0:
        raise ValueError("Cannot make pseudo-bulk vector from zero rows")
    return np.asarray(X_norm.mean(axis=0)).ravel()


def make_grouped_pseudobulk_vectors(
    X_norm: sparse.spmatrix,
    groups: pd.Series | np.ndarray,
) -> dict[str, np.ndarray]:
    group_series = pd.Series(groups).astype(str).str.lower()
    if len(group_series) != X_norm.shape[0]:
        raise ValueError("Group labels length does not match matrix n_obs")
    vectors: dict[str, np.ndarray] = {}
    for group in sorted(group_series.dropna().unique()):
        mask = group_series.to_numpy() == group
        if mask.sum() == 0:
            continue
        vectors[str(group)] = make_pseudobulk_vector(X_norm[mask, :])
    return vectors


def align_gene_vectors(
    rna_genes: np.ndarray,
    rna_vector: np.ndarray,
    ga_genes: np.ndarray,
    ga_vector: np.ndarray,
) -> pd.DataFrame:
    rna = pd.DataFrame({"gene_name": pd.Series(rna_genes, dtype="object"), "rna_pseudobulk": rna_vector})
    ga = pd.DataFrame({"gene_name": pd.Series(ga_genes, dtype="object"), "ga_pseudobulk": ga_vector})
    aligned = rna.merge(ga, on="gene_name", how="inner", validate="one_to_one").sort_values("gene_name")
    if aligned.empty:
        raise ValueError("No shared genes after gene_name matching")
    return aligned.reset_index(drop=True)


def _safe_corr(func, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    result = func(x, y)
    if isinstance(result, tuple):
        return float(result[0]), float(result[1])
    return float(result.statistic), float(result.pvalue)


def compute_correlation_metrics(aligned: pd.DataFrame) -> dict[str, float | int]:
    x = aligned["rna_pseudobulk"].to_numpy(dtype=np.float64)
    y = aligned["ga_pseudobulk"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    pearson_r, pearson_p = _safe_corr(stats.pearsonr, x, y)
    spearman_r, spearman_p = _safe_corr(stats.spearmanr, x, y)
    kendall_tau, kendall_p = _safe_corr(stats.kendalltau, x, y)
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
        "n_genes_used_for_correlation": int(len(x)),
    }


def compute_topk_overlap(aligned: pd.DataFrame, *, topk_values: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available = int(aligned.shape[0])
    for requested_k in topk_values:
        k_used = min(int(requested_k), available)
        rna_top = set(aligned.nlargest(k_used, "rna_pseudobulk")["gene_name"].astype(str))
        ga_top = set(aligned.nlargest(k_used, "ga_pseudobulk")["gene_name"].astype(str))
        overlap = len(rna_top.intersection(ga_top))
        union = len(rna_top.union(ga_top))
        rows.append(
            {
                "topk_requested": int(requested_k),
                "topk_used": int(k_used),
                "topk_overlap_count": int(overlap),
                "topk_jaccard": float(overlap / union) if union else np.nan,
            }
        )
    return rows


def bootstrap_spearman_ci(
    aligned: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, float | int]:
    x = aligned["rna_pseudobulk"].to_numpy(dtype=np.float64)
    y = aligned["ga_pseudobulk"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = len(x)
    if n < 2:
        return {
            "spearman_bootstrap_mean": np.nan,
            "spearman_bootstrap_ci_low": np.nan,
            "spearman_bootstrap_ci_high": np.nan,
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_random_seed": int(random_seed),
            "bootstrap_successful_samples": 0,
        }
    rng = np.random.default_rng(int(random_seed))
    values: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        if np.nanstd(x[idx]) == 0 or np.nanstd(y[idx]) == 0:
            continue
        rho = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(rho):
            values.append(float(rho))
    if not values:
        mean = low = high = np.nan
    else:
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
    return {
        "spearman_bootstrap_mean": mean,
        "spearman_bootstrap_ci_low": low,
        "spearman_bootstrap_ci_high": high,
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_random_seed": int(random_seed),
        "bootstrap_successful_samples": int(len(values)),
    }


def validate_no_spot_level_matching(config: dict[str, Any]) -> None:
    forbidden = set(map(str, config.get("forbidden_comparisons", [])))
    missing = FORBIDDEN_COMPARISONS.difference(forbidden)
    if missing:
        raise ValueError(f"Config does not explicitly forbid comparisons: {sorted(missing)}")
    requested = set(map(str, config.get("comparison_levels", [])))
    illegal_requested = FORBIDDEN_COMPARISONS.intersection(requested)
    if illegal_requested:
        raise ValueError(f"Config requested forbidden comparison levels: {sorted(illegal_requested)}")


def count_nonzero_values(vector: np.ndarray) -> int:
    return int(np.sum(np.asarray(vector) > 0))


def gene_overlap_summary(rna: CollapsedMatrix, ga: CollapsedMatrix, *, method_id: str) -> dict[str, Any]:
    rna_genes = set(map(str, rna.gene_names))
    ga_genes = set(map(str, ga.gene_names))
    shared = rna_genes.intersection(ga_genes)
    return {
        "method_id": method_id,
        "gene_matching_key": "gene_name",
        "duplicate_collapse": "sum",
        "rna_genes_before_collapse": rna.n_genes_before_collapse,
        "rna_duplicated_gene_names_before_collapse": rna.duplicated_gene_names_before_collapse,
        "rna_genes_after_collapse": rna.n_genes_after_collapse,
        "ga_genes_before_collapse": ga.n_genes_before_collapse,
        "ga_duplicated_gene_names_before_collapse": ga.duplicated_gene_names_before_collapse,
        "ga_genes_after_collapse": ga.n_genes_after_collapse,
        "n_shared_genes": len(shared),
        "percent_rna_overlap": 100.0 * len(shared) / len(rna_genes) if rna_genes else 0.0,
        "percent_ga_overlap": 100.0 * len(shared) / len(ga_genes) if ga_genes else 0.0,
    }


def add_vector_metadata(
    aligned: pd.DataFrame,
    *,
    method_id: str,
    comparison_level: str,
    group: str,
) -> pd.DataFrame:
    output = aligned.copy()
    output["method_id"] = method_id
    output["comparison_level"] = comparison_level
    output["group"] = group
    return output[["gene_name", "rna_pseudobulk", "ga_pseudobulk", "method_id", "comparison_level", "group"]]
