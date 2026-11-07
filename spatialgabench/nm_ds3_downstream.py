"""Shared statistics and frozen-table synthesis for NM-DS3.

NM-DS3 executes the remaining LIT1 downstream and robustness experiments that
are not owned by NM-DS1, NM-DS2, NM-RISK1, or NM-RISK2.  This module contains
the pure statistics and synthesis helpers.  The config-driven driver in
``scripts/run_nm_ds3_downstream.py`` owns all file I/O and writes only below
``results/v1_ds3``.
"""

from __future__ import annotations

import json
import platform
import resource
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse, stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def software_versions() -> dict[str, str]:
    import scanpy as sc
    import yaml

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "anndata": ad.__version__,
        "scanpy": sc.__version__,
        "yaml": yaml.__version__,
    }


def peak_rss_mb() -> dict[str, float]:
    self_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    child_kb = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return {
        "self_peak_rss_mb": self_kb / 1024.0,
        "children_peak_rss_mb": child_kb / 1024.0,
        "max_observed_peak_rss_mb": max(self_kb, child_kb) / 1024.0,
    }


# ---------------------------------------------------------------------------
# Bootstrap and verdict rules (frozen DS1 implementation)


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvalues, dtype=np.float64)
    n = pvals.size
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n, dtype=np.float64) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = q
    return out


def summarize_bootstrap(diffs: Sequence[float], margin: float) -> dict[str, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "bootstrap_mean_diff": float("nan"),
            "bootstrap_median_diff": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci90_low": float("nan"),
            "ci90_high": float("nan"),
            "p_difference_raw": float("nan"),
            "p_tost": float("nan"),
            "n_bootstrap_successful": 0,
        }
    return {
        "bootstrap_mean_diff": float(np.mean(arr)),
        "bootstrap_median_diff": float(np.median(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "ci90_low": float(np.percentile(arr, 5.0)),
        "ci90_high": float(np.percentile(arr, 95.0)),
        "p_difference_raw": float(
            min(1.0, 2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
        ),
        "p_tost": float(max(np.mean(arr <= -float(margin)), np.mean(arr >= float(margin)))),
        "n_bootstrap_successful": int(arr.size),
    }


def classify_verdict(row: dict[str, Any], margin: float, alpha: float) -> str:
    p_tost = row.get("p_tost", float("nan"))
    p_bh = row.get("p_difference_bh", row.get("p_difference_raw", float("nan")))
    diff = row.get("bootstrap_mean_diff", float("nan"))
    ci90_low = row.get("ci90_low", float("nan"))
    ci90_high = row.get("ci90_high", float("nan"))
    if (
        np.isfinite(p_tost)
        and p_tost < alpha
        and ci90_low >= -margin
        and ci90_high <= margin
    ):
        return "EQUIVALENT_WITHIN_0.05"
    if (
        np.isfinite(p_bh)
        and p_bh < alpha
        and np.isfinite(diff)
        and abs(diff) >= margin
    ):
        return "MATERIAL_DIFFERENCE"
    if np.isfinite(p_bh) and p_bh < alpha and np.isfinite(diff):
        return "DETECTABLE_SUB_MATERIAL"
    return "INCONCLUSIVE"


def _encode_labels(values: Sequence[str]) -> tuple[np.ndarray, int]:
    _, encoded = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    codes = np.asarray(encoded, dtype=np.int64)
    return codes, int(codes.max()) + 1 if codes.size else 0


def _contingency(
    truth: np.ndarray, predicted: np.ndarray, n_truth: int, n_pred: int
) -> np.ndarray:
    counts = np.bincount(truth * n_pred + predicted, minlength=n_truth * n_pred)
    return counts.reshape(n_truth, n_pred).astype(np.float64)


def ari_from_codes(truth: np.ndarray, predicted: np.ndarray, n_truth: int, n_pred: int) -> float:
    n = int(truth.size)
    if n < 2:
        return float("nan")
    table = _contingency(truth, predicted, n_truth, n_pred)
    sum_comb = (float(np.sum(table * table)) - n) / 2.0
    row = table.sum(axis=1)
    col = table.sum(axis=0)
    sum_row = (float(np.sum(row * row)) - n) / 2.0
    sum_col = (float(np.sum(col * col)) - n) / 2.0
    total_pairs = n * (n - 1) / 2.0
    if total_pairs <= 0:
        return float("nan")
    expected = sum_row * sum_col / total_pairs
    max_index = 0.5 * (sum_row + sum_col)
    denominator = max_index - expected
    if abs(denominator) < 1e-15:
        return 1.0 if np.array_equal(row.nonzero()[0], col.nonzero()[0]) else 0.0
    return float((sum_comb - expected) / denominator)


def nmi_from_codes(truth: np.ndarray, predicted: np.ndarray, n_truth: int, n_pred: int) -> float:
    n = int(truth.size)
    if n == 0:
        return float("nan")
    table = _contingency(truth, predicted, n_truth, n_pred)
    row = table.sum(axis=1)
    col = table.sum(axis=0)
    nz = table > 0
    nij = table[nz]
    ai = row[:, None] * np.ones((1, n_pred))
    bj = np.ones((n_truth, 1)) * col[None, :]
    mutual_information = float(
        np.sum((nij / n) * np.log(nij * n / (ai[nz] * bj[nz])))
    )
    row_p = row[row > 0] / n
    col_p = col[col > 0] / n
    entropy_truth = float(-np.sum(row_p * np.log(row_p)))
    entropy_pred = float(-np.sum(col_p * np.log(col_p)))
    denominator = 0.5 * (entropy_truth + entropy_pred)
    if denominator <= 1e-15:
        return 1.0
    return float(mutual_information / denominator)


def bootstrap_domain_difference(
    *,
    truth: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: str,
    n_bootstrap: int,
    seed: int,
    margin: float = 0.05,
) -> dict[str, Any]:
    metric_fn = adjusted_rand_score if metric == "ARI" else normalized_mutual_info_score
    truth_arr = np.asarray(truth, dtype=str)
    a_arr = np.asarray(pred_a, dtype=str)
    b_arr = np.asarray(pred_b, dtype=str)
    if not (truth_arr.shape == a_arr.shape == b_arr.shape):
        raise ValueError("Domain vectors are not aligned")
    point_a = float(metric_fn(truth_arr, a_arr))
    point_b = float(metric_fn(truth_arr, b_arr))
    truth_codes, n_truth = _encode_labels(truth_arr)
    a_codes, n_a = _encode_labels(a_arr)
    b_codes, n_b = _encode_labels(b_arr)
    fast_metric = ari_from_codes if metric == "ARI" else nmi_from_codes
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_bootstrap), dtype=np.float64)
    n = truth_arr.shape[0]
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        t = truth_codes[idx]
        diffs[i] = fast_metric(t, b_codes[idx], n_truth, n_b) - fast_metric(
            t, a_codes[idx], n_truth, n_a
        )
    row = summarize_bootstrap(diffs, margin=margin)
    row.update(
        {
            "n_units": int(n),
            "point_estimate_a": point_a,
            "point_estimate_b": point_b,
            "point_difference_b_minus_a": point_b - point_a,
        }
    )
    return row


def bootstrap_unit_difference(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
    margin: float = 0.05,
) -> dict[str, Any]:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError("Paired unit vectors are not aligned")
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a = arr_a[mask]
    arr_b = arr_b[mask]
    if arr_a.size == 0:
        return summarize_bootstrap([], margin=margin) | {
            "n_units": 0,
            "point_estimate_a": float("nan"),
            "point_estimate_b": float("nan"),
            "point_difference_b_minus_a": float("nan"),
        }
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_bootstrap), dtype=np.float64)
    n = arr_a.size
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(arr_b[idx] - arr_a[idx]))
    row = summarize_bootstrap(diffs, margin=margin)
    row.update(
        {
            "n_units": int(n),
            "point_estimate_a": float(np.mean(arr_a)),
            "point_estimate_b": float(np.mean(arr_b)),
            "point_difference_b_minus_a": float(np.mean(arr_b) - np.mean(arr_a)),
        }
    )
    return row


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size != y_arr.size or x_arr.size < 2:
        return float("nan")
    rx = stats.rankdata(x_arr)
    ry = stats.rankdata(y_arr)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denominator)


def bootstrap_svg_rho_difference(
    rna_values: Sequence[float],
    rep_a: Sequence[float],
    rep_b: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
    margin: float,
) -> dict[str, Any]:
    rna_values = np.asarray(rna_values, dtype=np.float64)
    rep_a = np.asarray(rep_a, dtype=np.float64)
    rep_b = np.asarray(rep_b, dtype=np.float64)
    mask = np.isfinite(rna_values) & np.isfinite(rep_a) & np.isfinite(rep_b)
    rna_values, rep_a, rep_b = rna_values[mask], rep_a[mask], rep_b[mask]
    n = rna_values.size
    if n < 3:
        return summarize_bootstrap([], margin=margin) | {
            "n_units": int(n),
            "point_estimate_a": float("nan"),
            "point_estimate_b": float("nan"),
            "point_difference_b_minus_a": float("nan"),
        }
    point_a = spearman_rho(rna_values, rep_a)
    point_b = spearman_rho(rna_values, rep_b)
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        rho_a = spearman_rho(rna_values[idx], rep_a[idx])
        rho_b = spearman_rho(rna_values[idx], rep_b[idx])
        diffs[i] = np.nan if not (np.isfinite(rho_a) and np.isfinite(rho_b)) else rho_b - rho_a
    row = summarize_bootstrap(diffs, margin=margin)
    row.update(
        {
            "n_units": int(n),
            "point_estimate_a": point_a,
            "point_estimate_b": point_b,
            "point_difference_b_minus_a": point_b - point_a,
        }
    )
    return row


def control_separation(row: dict[str, Any], margin: float = 0.05) -> str:
    """Classify a method-minus-control bootstrap without claiming equivalence."""

    median = row.get("bootstrap_median_diff", float("nan"))
    low = row.get("ci95_low", float("nan"))
    high = row.get("ci95_high", float("nan"))
    if np.isfinite(median) and np.isfinite(low) and median >= margin and low > 0.0:
        return "SEPARATES_POSITIVE"
    if np.isfinite(median) and np.isfinite(high) and median <= -margin and high < 0.0:
        return "SEPARATES_NEGATIVE"
    return "NULL"


# ---------------------------------------------------------------------------
# Unsupervised evaluation primitives


def normalize_total_log1p(adata: ad.AnnData, target_sum: float = 10000.0) -> ad.AnnData:
    out = adata.copy()
    out.X = sparse.csr_matrix(out.X, dtype=np.float32)
    counts = np.asarray(out.X.sum(axis=1)).ravel()
    scale = np.divide(
        float(target_sum),
        counts,
        out=np.zeros_like(counts, dtype=np.float64),
        where=counts > 0,
    )
    out.X = sparse.diags(scale) @ out.X
    out.X.data = np.log1p(out.X.data)
    return out


def select_hvg_and_pca(
    adata: ad.AnnData, config: dict[str, Any], seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    import scanpy as sc
    import warnings

    cl = config["clustering"]
    work = adata.copy()
    work.X = sparse.csr_matrix(work.X, dtype=np.float32)
    n_top = min(int(cl["n_top_genes"]), max(1, work.n_vars - 1))
    hvg_mask = np.ones(work.n_vars, dtype=bool)
    hvg_status = "all_genes"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sc.pp.highly_variable_genes(work, n_top_genes=n_top, flavor=str(cl["hvg_flavor"]))
            hvg_mask = work.var["highly_variable"].to_numpy(dtype=bool)
            hvg_status = "success"
        except Exception:
            hvg_mask = np.ones(work.n_vars, dtype=bool)
            hvg_status = "fallback_all_genes"
    if hvg_mask.sum() < 2:
        hvg_mask = np.ones(work.n_vars, dtype=bool)
        hvg_status = "fallback_all_genes"
    sub = work[:, hvg_mask].copy()
    n_comps = min(int(cl["n_pcs"]), sub.n_obs - 1, sub.n_vars - 1)
    if n_comps < 1:
        raise ValueError("Not enough spots/genes for PCA")
    sc.pp.pca(
        sub,
        n_comps=n_comps,
        random_state=int(seed),
        svd_solver=str(cl["pca_solver"]),
    )
    pcs = np.asarray(sub.obsm["X_pca"], dtype=np.float64)
    return pcs, {
        "hvg_status": hvg_status,
        "n_hvg": int(hvg_mask.sum()),
        "n_pcs": int(n_comps),
    }


def cluster_pcs(
    adata: ad.AnnData,
    pcs: np.ndarray,
    config: dict[str, Any],
    use_spatial: bool,
    seeds: Sequence[int],
) -> dict[int, np.ndarray]:
    k = int(config["clustering"]["k"])
    n_init = int(config["clustering"]["kmeans_n_init"])
    if use_spatial:
        if "spatial" not in adata.obsm:
            raise ValueError("Spatial coordinates are required for ATAC domain clustering")
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        spatial = (spatial - spatial.mean(axis=0)) / np.where(
            spatial.std(axis=0) == 0, 1.0, spatial.std(axis=0)
        )
        features = np.hstack([pcs, float(config["clustering"]["spatial_weight"]) * spatial])
    else:
        features = pcs
    out: dict[int, np.ndarray] = {}
    for seed in seeds:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=int(seed))
        out[int(seed)] = km.fit_predict(features).astype(str)
    return out


def clustering_metrics(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    truth_arr = np.asarray(truth, dtype=str)
    pred_arr = np.asarray(predicted, dtype=str)
    return {
        "ARI": float(adjusted_rand_score(truth_arr, pred_arr)),
        "NMI": float(normalized_mutual_info_score(truth_arr, pred_arr)),
        "n_predicted_clusters": int(pd.Series(pred_arr).nunique()),
        "n_truth_clusters": int(pd.Series(truth_arr).nunique()),
    }


def spatial_label_agreement(
    adata: ad.AnnData, labels: Sequence[str], n_neighbors: int = 6
) -> float:
    if "spatial" not in adata.obsm:
        return float("nan")
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    n = min(int(n_neighbors) + 1, xy.shape[0])
    nn = NearestNeighbors(n_neighbors=n).fit(xy)
    _, idx = nn.kneighbors(xy)
    lab = np.asarray(labels, dtype=str)
    agree = [np.mean(lab[idx[i, 1:]] == lab[i]) for i in range(xy.shape[0])]
    return float(np.mean(agree))


def spatial_weight_matrix(coords: np.ndarray, n_neighbors: int) -> tuple[sparse.csr_matrix, float]:
    xy = np.asarray(coords, dtype=np.float64)
    n = xy.shape[0]
    k = min(int(n_neighbors) + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(xy)
    _, idx = nn.kneighbors(xy)
    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].ravel()
    data = np.ones(rows.shape[0], dtype=np.float64)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums, dtype=np.float64)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    W = sparse.diags(inv) @ W
    return W.tocsr(), float(W.sum())


def moran_i_matrix(
    X: sparse.spmatrix | np.ndarray,
    W: sparse.csr_matrix,
    sum_w: float,
    chunk_size: int = 256,
) -> np.ndarray:
    X = sparse.csr_matrix(X, dtype=np.float64)
    n, n_genes = X.shape
    out = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        block = X[:, start:stop].toarray()
        centered = block - block.mean(axis=0, keepdims=True)
        wx = W @ centered
        num = np.sum(centered * wx, axis=0)
        den = np.sum(centered * centered, axis=0)
        vals = np.zeros(block.shape[1], dtype=np.float64)
        nz = den > 0
        vals[nz] = (float(n) / sum_w) * num[nz] / den[nz]
        out[start:stop] = vals
    return out


def top_set_metrics(
    rna_values: np.ndarray,
    rep_values: np.ndarray,
    genes: Sequence[str],
    top_n: int,
) -> dict[str, float]:
    k = min(int(top_n), len(genes))
    rna_idx = np.argsort(-np.asarray(rna_values))[:k]
    rep_idx = np.argsort(-np.asarray(rep_values))[:k]
    rna_set = set(np.asarray(genes, dtype=str)[rna_idx])
    rep_set = set(np.asarray(genes, dtype=str)[rep_idx])
    overlap = len(rna_set.intersection(rep_set))
    return {
        "top_n_used": int(k),
        "top_overlap": int(overlap),
        "top_jaccard": float(overlap / len(rna_set.union(rep_set)))
        if rna_set.union(rep_set)
        else float("nan"),
        "top_recall": float(overlap / len(rna_set)) if rna_set else float("nan"),
        "top_precision": float(overlap / len(rep_set)) if rep_set else float("nan"),
    }


def rna_marker_table(
    config: dict[str, Any], ref: ad.AnnData, labels: Sequence[str], dataset_key: str
) -> pd.DataFrame:
    import scanpy as sc

    work = ref.copy()
    work.obs["cluster"] = pd.Categorical(np.asarray(labels, dtype=str))
    sc.tl.rank_genes_groups(
        work,
        groupby="cluster",
        method=str(config["markers"]["method"]),
        use_raw=False,
    )
    rows: list[dict[str, Any]] = []
    marker_cfg = config["markers"]
    X_ref = sparse.csr_matrix(ref.X)
    for group in work.obs["cluster"].cat.categories.astype(str):
        names = pd.Series(work.uns["rank_genes_groups"]["names"][group].astype(str))
        scores = pd.Series(work.uns["rank_genes_groups"]["scores"][group].astype(float))
        logfc = pd.Series(work.uns["rank_genes_groups"]["logfoldchanges"][group].astype(float))
        padj = pd.Series(work.uns["rank_genes_groups"]["pvals_adj"][group].astype(float))
        mask_group = np.asarray(labels, dtype=str) == str(group)
        pct_values = np.asarray((X_ref[mask_group, :] > 0).mean(axis=0)).ravel()
        pct = (
            pd.Series(pct_values, index=ref.var_names.astype(str))
            .loc[names.to_numpy()]
            .reset_index(drop=True)
        )
        frame = pd.DataFrame(
            {
                "dataset_key": dataset_key,
                "cluster": group,
                "gene_id": names,
                "score": scores,
                "log_fold_change": logfc,
                "p_adjusted": padj,
                "fraction_in_group": pct,
            }
        )
        frame = frame[
            (frame["p_adjusted"] < float(marker_cfg["max_adjusted_p"]))
            & (frame["log_fold_change"] >= float(marker_cfg["min_log_fold_change"]))
            & (frame["fraction_in_group"] >= float(marker_cfg["min_fraction_in_group"]))
        ].sort_values(["score", "gene_id"], ascending=[False, True])
        rows.extend(frame.head(int(marker_cfg["top_n_per_cluster"])).to_dict("records"))
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"{dataset_key}: no RNA markers pass the pre-registered rule")
    return out


def atac_cluster_scores(
    norm: ad.AnnData, labels: Sequence[str], cluster: str, pseudocount: float = 1.0e-6
) -> pd.Series:
    labels_arr = np.asarray(labels, dtype=str)
    mask = labels_arr == str(cluster)
    if not mask.any() or mask.all():
        raise ValueError(f"Cluster {cluster} has no valid contrast")
    X = sparse.csr_matrix(norm.X, dtype=np.float64)
    mean_in = np.asarray(X[mask, :].mean(axis=0)).ravel()
    mean_out = np.asarray(X[~mask, :].mean(axis=0)).ravel()
    score = np.log2((mean_in + float(pseudocount)) / (mean_out + float(pseudocount)))
    return pd.Series(score, index=norm.var_names.astype(str))


def marker_percentiles(
    norm: ad.AnnData,
    labels: Sequence[str],
    markers: pd.DataFrame,
    pseudocount: float = 1.0e-6,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster, group in markers.groupby("cluster", sort=True):
        scores = atac_cluster_scores(norm, labels, str(cluster), pseudocount).sort_values(
            ascending=False
        )
        rank_map = {gene: rank + 1 for rank, gene in enumerate(scores.index.astype(str))}
        n_genes = len(rank_map)
        for marker in group.itertuples(index=False):
            gene_id = str(marker.gene_id)
            rank = rank_map.get(gene_id)
            if rank is None:
                continue
            percentile = 1.0 - (float(rank) - 1.0) / max(1.0, float(n_genes - 1))
            rows.append(
                {
                    "cluster": str(cluster),
                    "gene_id": gene_id,
                    "rank": int(rank),
                    "n_ranked_genes": int(n_genes),
                    "marker_percentile": float(percentile),
                }
            )
    return pd.DataFrame(rows)


def hungarian_cluster_mapping(
    truth_labels: Sequence[str],
    atac_labels: Sequence[str],
) -> pd.DataFrame:
    """Evaluation-only maximum-overlap mapping from RNA truth to ATAC clusters.

    KMeans integer labels have no cross-fit correspondence.  This function
    aligns the two arbitrary label spaces by maximizing the number of spots in
    matched cells of the RNA-truth x ATAC-cluster contingency table.  The
    mapping uses evaluation labels only and never changes ATAC features,
    clusters, or representation construction.
    """

    truth = np.asarray(truth_labels, dtype=str)
    pred = np.asarray(atac_labels, dtype=str)
    if truth.shape != pred.shape:
        raise ValueError("truth_labels and atac_labels must have the same length")
    truth_values = sorted(pd.unique(truth).tolist())
    pred_values = sorted(pd.unique(pred).tolist())
    counts = np.zeros((len(truth_values), len(pred_values)), dtype=np.int64)
    truth_pos = {value: index for index, value in enumerate(truth_values)}
    pred_pos = {value: index for index, value in enumerate(pred_values)}
    for truth_value, pred_value in zip(truth, pred):
        counts[truth_pos[truth_value], pred_pos[pred_value]] += 1
    n = max(counts.shape)
    padded = np.zeros((n, n), dtype=np.int64)
    padded[: counts.shape[0], : counts.shape[1]] = counts
    row_index, col_index = linear_sum_assignment(-padded)
    assigned: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for row, col in zip(row_index, col_index):
        if row >= counts.shape[0] or col >= counts.shape[1]:
            continue
        truth_value = truth_values[row]
        pred_value = pred_values[col]
        overlap = int(counts[row, col])
        if overlap <= 0:
            continue
        assigned[truth_value] = pred_value
        rows.append(
            {
                "rna_cluster": truth_value,
                "atac_cluster": pred_value,
                "overlap": overlap,
                "rna_cluster_size": int(counts[row, :].sum()),
                "atac_cluster_size": int(counts[:, col].sum()),
                "mapping_method": "hungarian_max_overlap",
                "matched": True,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "rna_cluster",
                "atac_cluster",
                "overlap",
                "rna_cluster_size",
                "atac_cluster_size",
                "mapping_method",
                "matched",
            ]
        )
    frame = frame.sort_values("rna_cluster").reset_index(drop=True)
    if frame["atac_cluster"].duplicated().any():
        raise ValueError("Hungarian mapping assigned one ATAC cluster to multiple RNA clusters")
    return frame


def marker_percentiles_matched(
    norm: ad.AnnData,
    labels: Sequence[str],
    markers: pd.DataFrame,
    mapping: pd.DataFrame,
    pseudocount: float = 1.0e-6,
) -> pd.DataFrame:
    """Score RNA markers in their matched ATAC cluster.

    ``mapping`` is produced by :func:`hungarian_cluster_mapping`.  RNA marker
    clusters without a positive-overlap ATAC match are skipped and remain
    NOT_ESTIMABLE rather than being scored against an arbitrary label.
    """

    if mapping.empty:
        return pd.DataFrame(
            columns=[
                "cluster",
                "matched_atac_cluster",
                "gene_id",
                "rank",
                "n_ranked_genes",
                "marker_percentile",
                "mapping_method",
                "matched",
            ]
        )
    map_dict = {
        str(row.rna_cluster): str(row.atac_cluster)
        for row in mapping.itertuples(index=False)
        if bool(getattr(row, "matched", True))
    }
    rows: list[dict[str, Any]] = []
    for cluster, group in markers.groupby("cluster", sort=True):
        atac_cluster = map_dict.get(str(cluster))
        if atac_cluster is None:
            continue
        scores = atac_cluster_scores(
            norm, labels, atac_cluster, pseudocount
        ).sort_values(ascending=False)
        rank_map = {gene: rank + 1 for rank, gene in enumerate(scores.index.astype(str))}
        n_genes = len(rank_map)
        for marker in group.itertuples(index=False):
            gene_id = str(marker.gene_id)
            rank = rank_map.get(gene_id)
            if rank is None:
                continue
            percentile = 1.0 - (float(rank) - 1.0) / max(1.0, float(n_genes - 1))
            rows.append(
                {
                    "cluster": str(cluster),
                    "matched_atac_cluster": atac_cluster,
                    "gene_id": gene_id,
                    "rank": int(rank),
                    "n_ranked_genes": int(n_genes),
                    "marker_percentile": float(percentile),
                    "mapping_method": "hungarian_max_overlap",
                    "matched": True,
                }
            )
    return pd.DataFrame(rows)


def p_tost_pairwise(
    table: pd.DataFrame,
    *,
    unit_count_column: str,
    margin: float,
    alpha: float,
) -> pd.DataFrame:
    """BH-adjust a small contrast family and assign frozen verdicts in place."""

    out = table.copy()
    if out.empty:
        return out
    qvals = bh_adjust(out["p_difference_raw"].to_numpy(dtype=float))
    out["p_difference_bh"] = qvals
    out["verdict"] = [
        classify_verdict(row, margin, alpha) for row in out.to_dict("records")
    ]
    out["n_units"] = out[unit_count_column] if unit_count_column in out else out.get("n_units")
    return out


def sign(value: float) -> int:
    if not np.isfinite(value):
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def canonical_contrast(value: str) -> str:
    return (
        str(value)
        .replace("F_signac_geneactivity", "F")
        .replace("fixed_bin_25kb", "B")
        .replace("macs3_summit_501bp", "P")
    )


def contrast_pair(value: str) -> tuple[str, str]:
    parts = canonical_contrast(value).split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Unrecognized contrast: {value}")
    return parts[0], parts[1]


def _endpoint_key(experiment: str, metric: str) -> str:
    return f"{experiment}:{metric}"


def _lookup_delta(
    frame: pd.DataFrame,
    *,
    experiment: str,
    metric: str,
    contrast: str,
    dataset_key: str | None = None,
    depth: float | None = None,
) -> float:
    sub = frame[(frame["experiment"] == experiment) & (frame["metric"] == metric)]
    if dataset_key is not None and "dataset_key" in sub:
        sub = sub[sub["dataset_key"] == dataset_key]
    if depth is not None and "depth_fraction" in sub:
        sub = sub[np.isclose(sub["depth_fraction"].astype(float), float(depth))]
    sub = sub[sub["contrast"].map(canonical_contrast) == canonical_contrast(contrast)]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0]["point_difference_b_minus_a"])


def e2_sign_agreement(
    human_tables: dict[str, pd.DataFrame],
    mouse_tables: dict[str, pd.DataFrame],
    endpoints: dict[str, dict[str, str]],
    mouse_dataset_key: str,
    mouse_depth: float,
    contrasts: Sequence[str] | None = None,
    pass_min_sign_agreement: int = 9,
    partial_min_sign_agreement: int = 6,
    endpoint_min_pairs: int = 2,
    endpoint_required_fraction: float = 2.0 / 3.0,
    endpoint_pass_min: int | None = None,
    track: str = "representation",
    decision_scope: str = "primary",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contrasts = list(contrasts or ["B-P", "F-P", "F-B"])
    for endpoint_name, spec in endpoints.items():
        experiment = str(spec["experiment"])
        metric = str(spec["metric"])
        for contrast in contrasts:
            human = _lookup_delta(
                human_tables[experiment],
                experiment=experiment,
                metric=metric,
                contrast=contrast,
            )
            mouse = _lookup_delta(
                mouse_tables[experiment],
                experiment=experiment,
                metric=metric,
                contrast=contrast,
                dataset_key=mouse_dataset_key,
                depth=mouse_depth,
            )
            rows.append(
                {
                    "endpoint": endpoint_name,
                    "experiment": experiment,
                    "metric": metric,
                    "contrast": contrast,
                    "human_delta": human,
                    "mouse_delta": mouse,
                    "human_sign": sign(human),
                    "mouse_sign": sign(mouse),
                    "sign_available": bool(np.isfinite(human) and np.isfinite(mouse)),
                    "sign_agreement": bool(
                        np.isfinite(human) and np.isfinite(mouse) and sign(human) == sign(mouse)
                    ),
                    "abs_human_minus_mouse": abs(human - mouse)
                    if np.isfinite(human) and np.isfinite(mouse)
                    else float("nan"),
                    "track": track,
                    "decision_scope": decision_scope,
                }
            )
    frame = pd.DataFrame(rows)
    available = frame[frame["sign_available"]]
    n_agree = int(available["sign_agreement"].sum())
    n_pairs = int(len(available))
    per_endpoint = (
        available.groupby("endpoint")["sign_agreement"].agg(["sum", "count"]).reset_index()
    )
    per_endpoint["fraction"] = per_endpoint["sum"] / per_endpoint["count"]
    endpoint_pass = int(
        (
            (per_endpoint["count"] >= int(endpoint_min_pairs))
            & (per_endpoint["fraction"] >= float(endpoint_required_fraction))
        ).sum()
    )
    summary = {
        "n_pairs": n_pairs,
        "n_sign_agree": n_agree,
        "sign_agreement_fraction": n_agree / n_pairs if n_pairs else float("nan"),
        "n_endpoints": int(len(per_endpoint)),
        "n_endpoints_meeting_required_fraction": endpoint_pass,
        "pass_min_sign_agreement": int(pass_min_sign_agreement),
        "partial_min_sign_agreement": int(partial_min_sign_agreement),
        "track": track,
        "decision_scope": decision_scope,
        "median_abs_difference": float(available["abs_human_minus_mouse"].median())
        if n_pairs
        else float("nan"),
        "max_abs_difference": float(available["abs_human_minus_mouse"].max())
        if n_pairs
        else float("nan"),
    }
    if endpoint_pass_min is None:
        endpoint_pass_min = 3
    summary["endpoint_pass_min"] = int(endpoint_pass_min)
    if decision_scope == "descriptive":
        summary["decision"] = "DESCRIPTIVE"
    elif n_agree >= int(pass_min_sign_agreement) and endpoint_pass >= int(
        endpoint_pass_min
    ):
        summary["decision"] = "PASS"
    elif n_agree >= int(partial_min_sign_agreement):
        summary["decision"] = "PARTIAL"
    else:
        summary["decision"] = "FAIL"
    return frame, summary


def e2_risk2_agreement(
    human_tables: dict[str, pd.DataFrame],
    risk2_human: pd.DataFrame,
    method_id: str,
    endpoints: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reps = {"P": "macs3_summit_501bp", "B": "fixed_bin_25kb", "F": "signac_geneactivity"}
    available_risk = risk2_human[risk2_human["method_id"] == method_id].set_index(
        "representation_id"
    )
    for endpoint_name, spec in endpoints.items():
        experiment = str(spec["experiment"])
        metric = str(spec["metric"])
        human = _lookup_delta(
            human_tables[experiment], experiment=experiment, metric=metric, contrast="B-P"
        )
        if {"macs3_summit_501bp", "fixed_bin_25kb"}.issubset(available_risk.index):
            # Human downstream contrast is B-P, so the RISK2 endpoint must use
            # the same orientation: fixed-bin minus peak.
            risk_delta = float(
                available_risk.loc["fixed_bin_25kb", "primary_frozen"]
                - available_risk.loc["macs3_summit_501bp", "primary_frozen"]
            )
            unavailable_reason = ""
        else:
            risk_delta = float("nan")
            unavailable_reason = "RISK2 human_endpoints lacks the required method/representation pair"
        rows.append(
            {
                "endpoint": endpoint_name,
                "experiment": experiment,
                "metric": metric,
                "contrast": "B-P",
                "human_downstream_delta": human,
                "risk2_rna_delta": risk_delta,
                "human_sign": sign(human),
                "risk2_sign": sign(risk_delta),
                "sign_available": bool(np.isfinite(human) and np.isfinite(risk_delta)),
                "sign_agreement": bool(
                    np.isfinite(human)
                    and np.isfinite(risk_delta)
                    and sign(human) == sign(risk_delta)
                ),
                "unavailable_reason": unavailable_reason,
            }
        )
    frame = pd.DataFrame(rows)
    available = frame[frame["sign_available"]]
    summary = {
        "n_available": int(len(available)),
        "n_sign_agree": int(available["sign_agreement"].sum()),
        "note": "F-P and F-B are unavailable because RISK2 human_endpoints covers annotation methods only, not the Signac software representation",
    }
    return frame, summary


def e3_synthesis(
    tables: dict[str, pd.DataFrame],
    *,
    d1_dataset_key: str,
    depths: Sequence[float],
    primary_contrasts: Sequence[str],
    contrast_filter: Sequence[str] | None = None,
    h2_min_disagreement_fraction: float = 1.0 / 3.0,
    domain_required_for_h1: bool = True,
    track: str = "representation",
    decision_scope: str = "primary",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    verdict_rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    allowed = (
        {canonical_contrast(value) for value in contrast_filter}
        if contrast_filter is not None
        else None
    )
    filtered_tables: dict[str, pd.DataFrame] = {}
    for endpoint, frame in tables.items():
        if frame.empty:
            continue
        frame = frame.copy()
        if allowed is not None:
            frame = frame[
                frame["contrast"].map(canonical_contrast).isin(allowed)
            ].copy()
        if frame.empty:
            continue
        filtered_tables[endpoint] = frame
        frame["track"] = track
        for (dataset_key, depth), group in frame.groupby(
            ["dataset_key", "depth_fraction"], dropna=False
        ):
            counts = group["verdict"].value_counts().to_dict()
            total = int(len(group))
            material = int(counts.get("MATERIAL_DIFFERENCE", 0))
            verdict_rows.append(
                {
                    "endpoint": endpoint,
                    "dataset_key": dataset_key,
                    "depth_fraction": float(depth) if pd.notna(depth) else float("nan"),
                    "n_contrasts": total,
                    "n_material": material,
                    "material_fraction": material / total if total else float("nan"),
                    "track": track,
                    "decision_scope": decision_scope,
                    **{f"n_{key}": int(value) for key, value in counts.items()},
                }
            )
            material_rows.append(
                {
                    "endpoint": endpoint,
                    "dataset_key": dataset_key,
                    "depth_fraction": float(depth) if pd.notna(depth) else float("nan"),
                    "n_material": material,
                    "track": track,
                }
            )
    frames["verdict_counts"] = pd.DataFrame(verdict_rows)
    frames["material_counts"] = pd.DataFrame(material_rows)

    domain = filtered_tables.get("domain", pd.DataFrame())
    ari_nmi_rows: list[dict[str, Any]] = []
    if not domain.empty:
        for (dataset_key, depth, contrast), group in domain.groupby(
            ["dataset_key", "depth_fraction", "contrast"]
        ):
            ari = group[group["metric"] == "ARI"]
            nmi = group[group["metric"] == "NMI"]
            if ari.empty or nmi.empty:
                continue
            ari_row = ari.iloc[0]
            nmi_row = nmi.iloc[0]
            ari_nmi_rows.append(
                {
                    "dataset_key": dataset_key,
                    "depth_fraction": float(depth),
                    "contrast": canonical_contrast(contrast),
                    "ari_delta": float(ari_row["point_difference_b_minus_a"]),
                    "nmi_delta": float(nmi_row["point_difference_b_minus_a"]),
                    "ari_verdict": str(ari_row["verdict"]),
                    "nmi_verdict": str(nmi_row["verdict"]),
                    "sign_disagreement": sign(float(ari_row["point_difference_b_minus_a"]))
                    != sign(float(nmi_row["point_difference_b_minus_a"])),
                    "verdict_disagreement": str(ari_row["verdict"])
                    != str(nmi_row["verdict"]),
                }
            )
    frames["ari_nmi_disagreement"] = pd.DataFrame(ari_nmi_rows)

    depth_rows: list[dict[str, Any]] = []
    if not domain.empty:
        d1 = domain[domain["dataset_key"] == d1_dataset_key]
        for (metric, contrast), group in d1[d1["metric"].isin(["ARI", "NMI"])].groupby(
            ["metric", "contrast"]
        ):
            group = group.sort_values("depth_fraction")
            signs = [sign(float(v)) for v in group["point_difference_b_minus_a"]]
            changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
            depth_rows.append(
                {
                    "metric": metric,
                    "contrast": canonical_contrast(contrast),
                    "depths": ",".join(str(float(v)) for v in group["depth_fraction"]),
                    "deltas": ",".join(f"{float(v):.12g}" for v in group["point_difference_b_minus_a"]),
                    "signs": ",".join(str(v) for v in signs),
                    "n_sign_changes_between_adjacent_depths": int(changes),
                    "any_sign_change": bool(changes > 0),
                }
            )
    frames["depth_reversals"] = pd.DataFrame(depth_rows)

    endpoint_rows: list[dict[str, Any]] = []
    endpoint_map = {
        "domain_NMI": ("domain", "NMI"),
        "marker": ("markers", "mean_marker_percentile"),
        "svg": ("svg", "svg_spearman_to_rna"),
    }
    d1_frames: dict[str, pd.DataFrame] = {}
    for endpoint, (experiment, metric) in endpoint_map.items():
        frame = filtered_tables.get(experiment, pd.DataFrame())
        if frame.empty:
            continue
        sub = frame[
            (frame["dataset_key"] == d1_dataset_key)
            & np.isclose(frame["depth_fraction"].astype(float), 1.0)
            & (frame["metric"] == metric)
        ]
        d1_frames[endpoint] = sub
    for contrast in primary_contrasts:
        row: dict[str, Any] = {"contrast": canonical_contrast(contrast)}
        signs: list[int] = []
        for endpoint, frame in d1_frames.items():
            sub = frame[
                frame["contrast"].map(canonical_contrast) == canonical_contrast(contrast)
            ]
            delta = float(sub.iloc[0]["point_difference_b_minus_a"]) if not sub.empty else float("nan")
            row[f"{endpoint}_delta"] = delta
            row[f"{endpoint}_sign"] = sign(delta)
            if np.isfinite(delta):
                signs.append(sign(delta))
        row["n_available_endpoints"] = len(signs)
        row["sign_reversal_across_endpoints"] = len(set(signs)) > 1 if len(signs) > 1 else False
        endpoint_rows.append(row)
    frames["endpoint_reversals"] = pd.DataFrame(endpoint_rows)

    summary: dict[str, Any] = {}
    material_counts = frames["material_counts"]
    non_domain_material = int(
        material_counts[
            (material_counts["endpoint"] != "domain")
            & (material_counts["n_material"] > 0)
        ]["n_material"].sum()
    ) if not material_counts.empty else 0
    domain_material = int(
        material_counts[material_counts["endpoint"] == "domain"]["n_material"].sum()
    ) if not material_counts.empty else 0
    summary["domain_material_total"] = domain_material
    summary["non_domain_material_total"] = non_domain_material
    if decision_scope == "descriptive":
        summary["h1_material_concentration"] = "DESCRIPTIVE"
    elif domain_required_for_h1 and domain_material == 0:
        summary["h1_material_concentration"] = "FAIL"
    elif non_domain_material >= 3:
        summary["h1_material_concentration"] = "FAIL"
    elif domain_material > 0 and non_domain_material == 0:
        summary["h1_material_concentration"] = "PASS"
    else:
        summary["h1_material_concentration"] = "PARTIAL"

    disagreement = frames["ari_nmi_disagreement"]
    if not disagreement.empty:
        n_disagree = int(
            (disagreement["sign_disagreement"] | disagreement["verdict_disagreement"]).sum()
        )
        summary["ari_nmi_disagreement_count"] = n_disagree
        summary["ari_nmi_denominator"] = int(len(disagreement))
        threshold = float(h2_min_disagreement_fraction) * int(len(disagreement))
        summary["h2_min_disagreement_fraction"] = float(h2_min_disagreement_fraction)
        summary["h2_min_disagreement_count"] = float(threshold)
        summary["h2_metric_dependence"] = (
            "DESCRIPTIVE"
            if decision_scope == "descriptive"
            else ("PASS" if float(n_disagree) >= threshold else "FAIL")
        )
    else:
        summary["ari_nmi_disagreement_count"] = 0
        summary["ari_nmi_denominator"] = 0
        summary["h2_metric_dependence"] = (
            "DESCRIPTIVE" if decision_scope == "descriptive" else "FAIL"
        )

    depth = frames["depth_reversals"]
    summary["depth_reversal_count"] = int(depth["any_sign_change"].sum()) if not depth.empty else 0
    summary["h3_depth_reversal"] = (
        "DESCRIPTIVE"
        if decision_scope == "descriptive"
        else ("PASS" if summary["depth_reversal_count"] > 0 else "FAIL")
    )
    endpoint = frames["endpoint_reversals"]
    summary["endpoint_reversal_count"] = int(
        endpoint["sign_reversal_across_endpoints"].sum()
    ) if not endpoint.empty else 0
    summary["h4_endpoint_reversal"] = (
        "DESCRIPTIVE"
        if decision_scope == "descriptive"
        else ("PASS" if summary["endpoint_reversal_count"] > 0 else "FAIL")
    )
    summary["track"] = track
    summary["decision_scope"] = decision_scope
    return frames, summary


def e4_real_saturation(
    *,
    domain_seed_metrics: pd.DataFrame,
    marker_percentiles: pd.DataFrame,
    svg_summary: pd.DataFrame,
    primary_seed: int,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    domain = domain_seed_metrics[
        domain_seed_metrics["clustering_seed"].astype(int) == int(primary_seed)
    ]
    for (dataset_key, depth), group in domain.groupby(["dataset_key", "depth_fraction"]):
        for metric in ("ARI", "NMI"):
            values = group[metric].astype(float)
            median = float(values.median())
            value_range = float(values.max() - values.min())
            rows.append(
                {
                    "scope": "real_data",
                    "endpoint_family": "domain",
                    "endpoint": metric,
                    "dataset_key": dataset_key,
                    "depth_fraction": float(depth),
                    "n_representations": int(len(values)),
                    "median": median,
                    "between_representation_range": value_range,
                    "floor_saturated": bool(
                        median <= float(thresholds["floor_median"])
                        and value_range < float(thresholds["saturation_range"])
                    ),
                    "ceiling_saturated": bool(
                        median >= float(thresholds["ceiling_median"])
                        and value_range < float(thresholds["saturation_range"])
                    ),
                    "near_chance": bool(median <= float(thresholds["near_chance_median"])),
                }
            )
    if not marker_percentiles.empty:
        for (dataset_key, depth), group in marker_percentiles.groupby(
            ["dataset_key", "depth_fraction"]
        ):
            rep_medians = group.groupby("representation")["marker_percentile"].mean()
            median = float(rep_medians.median())
            value_range = float(rep_medians.max() - rep_medians.min())
            rows.append(
                {
                    "scope": "real_data",
                    "endpoint_family": "marker",
                    "endpoint": "mean_marker_percentile",
                    "dataset_key": dataset_key,
                    "depth_fraction": float(depth),
                    "n_representations": int(len(rep_medians)),
                    "median": median,
                    "between_representation_range": value_range,
                    "floor_saturated": bool(
                        median <= float(thresholds["floor_median"])
                        and value_range < float(thresholds["saturation_range"])
                    ),
                    "ceiling_saturated": bool(
                        median >= float(thresholds["ceiling_median"])
                        and value_range < float(thresholds["saturation_range"])
                    ),
                    "near_chance": bool(median <= float(thresholds["near_chance_median"])),
                }
            )
    if not svg_summary.empty:
        for (dataset_key, depth), group in svg_summary.groupby(
            ["dataset_key", "depth_fraction"]
        ):
            values = group["spearman_to_rna_moran_i"].astype(float)
            median = float(values.median())
            value_range = float(values.max() - values.min())
            rows.append(
                {
                    "scope": "real_data",
                    "endpoint_family": "svg",
                    "endpoint": "svg_spearman_to_rna",
                    "dataset_key": dataset_key,
                    "depth_fraction": float(depth),
                    "n_representations": int(len(values)),
                    "median": median,
                    "between_representation_range": value_range,
                    "floor_saturated": bool(
                        median <= float(thresholds["floor_median"])
                        and value_range < float(thresholds["saturation_range"])
                    ),
                    "ceiling_saturated": bool(
                        median >= float(thresholds["ceiling_median"])
                        and value_range < float(thresholds["saturation_range"])
                    ),
                    "near_chance": bool(median <= float(thresholds["near_chance_median"])),
                }
            )
    return pd.DataFrame(rows)


def e4_endpoint_family(endpoint: str) -> str:
    """Map an E4 endpoint name to its pre-registered endpoint family.

    Endpoint names and endpoint families are not interchangeable.  For example,
    ``svg_precision_at_10`` and ``svg_recall_at_10`` are two endpoint names in
    one SVG family.
    """

    text = str(endpoint).strip().lower()
    if text.startswith("svg"):
        return "svg"
    if text.startswith("domain"):
        return "domain"
    if text.startswith("marker"):
        return "marker"
    if text.startswith("activity"):
        return "activity"
    if text.startswith("peak_gene_link"):
        return "peak_gene_link"
    return text.split("_", 1)[0] if text else "unknown"


def e4_simulation_saturation(
    endpoint_summary: pd.DataFrame, thresholds: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    is_control = endpoint_summary["method_id"].astype(str).str.contains(
        "control", case=False, regex=False
    )
    method_rows = endpoint_summary[~is_control].copy()
    for (simulation_type, endpoint), group in endpoint_summary.groupby(
        ["simulation_type", "endpoint"]
    ):
        method_group = method_rows[
            (method_rows["simulation_type"] == simulation_type)
            & (method_rows["endpoint"] == endpoint)
        ]
        medians = method_group["median"].astype(float)
        control = group["control_floor_median"].astype(float)
        value_range = float(method_group["between_method_range"].max())
        median_of_medians = float(medians.median())
        rows.append(
            {
                "scope": "simulation",
                "simulation_type": simulation_type,
                "endpoint": endpoint,
                "n_methods": int(len(method_group)),
                "median_of_method_medians": median_of_medians,
                "max_between_method_range": value_range,
                "control_floor_median": float(control.median()),
                "high_control_floor": bool(
                    float(control.median()) > float(thresholds["high_control_floor"])
                ),
                "near_ceiling": bool(
                    median_of_medians >= float(thresholds["ceiling_median"])
                ),
                "n_methods_near_ceiling": int(
                    (medians >= float(thresholds["ceiling_median"])).sum()
                ),
                "ceiling_saturated": bool(
                    median_of_medians >= float(thresholds["ceiling_median"])
                    and value_range < float(thresholds["saturation_range"])
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["endpoint_family"] = frame["endpoint"].map(e4_endpoint_family)
    h2_rows = method_rows[
        (method_rows["median"].astype(float) >= float(thresholds["ceiling_median"]))
        & (
            method_rows["control_floor_median"].astype(float)
            <= float(thresholds["high_control_floor"])
        )
    ]
    summary = {
        "h2_near_ceiling_with_low_control_floor": int(len(h2_rows)),
        "h2_near_ceiling_method_endpoints": sorted(
            set(
                h2_rows["simulation_type"].astype(str)
                + ":"
                + h2_rows["endpoint"].astype(str)
                + ":"
                + h2_rows["method_id"].astype(str)
            )
        ),
        "h2_decision": "PASS" if not h2_rows.empty else "FAIL",
        "high_control_floor_endpoint_families": int(
            frame[frame["high_control_floor"]]["endpoint_family"].nunique()
        )
        if not frame.empty
        else 0,
        "high_control_floor_endpoint_names": int(
            frame[frame["high_control_floor"]]["endpoint"].nunique()
        )
        if not frame.empty
        else 0,
        "high_control_floor_families": sorted(
            frame[frame["high_control_floor"]]["endpoint_family"].unique().tolist()
        )
        if not frame.empty
        else [],
    }
    summary["h3_decision"] = (
        "PASS" if summary["high_control_floor_endpoint_families"] >= 2 else "FAIL"
    )
    return frame, summary


def e4_real_saturation_v2(
    *,
    domain_seed_metrics: pd.DataFrame,
    marker_percentiles: pd.DataFrame,
    svg_summary: pd.DataFrame,
    primary_seed: int,
    thresholds: dict[str, float],
    endpoint_nulls: dict[str, float],
    near_chance_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Real-data saturation audit with endpoint-specific chance levels.

    Marker percentiles have chance level 0.5, whereas ARI/NMI/Spearman have
    chance level 0.  The formal floor/ceiling flags are not applied to marker
    percentiles because a percentile floor is not a chance floor.
    """

    rows: list[dict[str, Any]] = []

    def add_row(
        *,
        endpoint_family: str,
        endpoint: str,
        dataset_key: str,
        depth: float,
        values: Sequence[float],
        track: str,
    ) -> None:
        values_arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        if values_arr.size == 0:
            return
        median = float(np.median(values_arr))
        value_range = float(values_arr.max() - values_arr.min())
        chance = float(endpoint_nulls[endpoint_family])
        if endpoint_family == "marker":
            near_chance = bool(abs(median - chance) <= float(near_chance_tolerance))
            chance_rule = "abs(median-0.5)<=tolerance"
            saturation_applicable = False
        elif endpoint_family == "svg":
            near_chance = bool(abs(median - chance) <= float(near_chance_tolerance))
            chance_rule = "abs(median-0.0)<=tolerance"
            saturation_applicable = True
        else:
            near_chance = bool(median <= float(near_chance_tolerance))
            chance_rule = "median<=tolerance"
            saturation_applicable = True
        rows.append(
            {
                "scope": "real_data",
                "endpoint_family": endpoint_family,
                "endpoint": endpoint,
                "dataset_key": dataset_key,
                "depth_fraction": float(depth),
                "track": track,
                "n_representations": int(values_arr.size),
                "median": median,
                "between_representation_range": value_range,
                "chance_level": chance,
                "chance_rule": chance_rule,
                "saturation_applicable": saturation_applicable,
                "floor_saturated": bool(
                    saturation_applicable
                    and median <= float(thresholds["floor_median"])
                    and value_range < float(thresholds["saturation_range"])
                ),
                "ceiling_saturated": bool(
                    saturation_applicable
                    and median >= float(thresholds["ceiling_median"])
                    and value_range < float(thresholds["saturation_range"])
                ),
                "near_chance": near_chance,
            }
        )

    if not domain_seed_metrics.empty:
        domain = domain_seed_metrics[
            domain_seed_metrics["clustering_seed"].astype(int) == int(primary_seed)
        ]
        for (dataset_key, depth, track), group in domain.groupby(
            ["dataset_key", "depth_fraction", "track"], dropna=False
        ):
            for metric in ("ARI", "NMI"):
                add_row(
                    endpoint_family="domain",
                    endpoint=metric,
                    dataset_key=str(dataset_key),
                    depth=float(depth),
                    values=group[metric].astype(float).tolist(),
                    track=str(track),
                )
    if not marker_percentiles.empty:
        for (dataset_key, depth, track), group in marker_percentiles.groupby(
            ["dataset_key", "depth_fraction", "track"], dropna=False
        ):
            rep_means = group.groupby("representation")["marker_percentile"].mean()
            add_row(
                endpoint_family="marker",
                endpoint="mean_marker_percentile",
                dataset_key=str(dataset_key),
                depth=float(depth),
                values=rep_means.astype(float).tolist(),
                track=str(track),
            )
    if not svg_summary.empty:
        for (dataset_key, depth, track), group in svg_summary.groupby(
            ["dataset_key", "depth_fraction", "track"], dropna=False
        ):
            add_row(
                endpoint_family="svg",
                endpoint="svg_spearman_to_rna",
                dataset_key=str(dataset_key),
                depth=float(depth),
                values=group["spearman_to_rna_moran_i"].astype(float).tolist(),
                track=str(track),
            )
    return pd.DataFrame(rows)


def e5_depth_synthesis(
    domain: pd.DataFrame,
    marker: pd.DataFrame,
    svg: pd.DataFrame,
    retention: pd.DataFrame,
    *,
    d1_dataset_key: str,
    depths: Sequence[float],
    risky_depths: Sequence[float],
    stable_depths: Sequence[float],
    primary_method_id: str,
    peak_representation: str,
    bin_representation: str,
    contrast_filter: Sequence[str] | None = None,
    track: str = "representation",
    decision_scope: str = "primary",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    material_rows: list[dict[str, Any]] = []
    allowed = (
        {canonical_contrast(value) for value in contrast_filter}
        if contrast_filter is not None
        else None
    )
    for endpoint, frame in (("domain", domain), ("marker", marker), ("svg", svg)):
        if frame.empty:
            continue
        frame = frame[frame["dataset_key"] == d1_dataset_key].copy()
        if allowed is not None:
            frame = frame[frame["contrast"].map(canonical_contrast).isin(allowed)].copy()
        sub = frame
        for depth in depths:
            rows = sub[np.isclose(sub["depth_fraction"].astype(float), float(depth))]
            material_rows.append(
                {
                    "endpoint": endpoint,
                    "depth_fraction": float(depth),
                    "n_contrasts": int(len(rows)),
                    "n_material": int((rows["verdict"] == "MATERIAL_DIFFERENCE").sum()),
                    "track": track,
                    "decision_scope": decision_scope,
                }
            )
    material = pd.DataFrame(material_rows)
    domain_material = material[material["endpoint"] == "domain"].set_index("depth_fraction")
    risky = int(
        domain_material.loc[
            [d for d in risky_depths if d in domain_material.index], "n_material"
        ].sum()
    )
    stable = int(
        domain_material.loc[
            [d for d in stable_depths if d in domain_material.index], "n_material"
        ].sum()
    )
    non_domain = material[material["endpoint"].isin(["marker", "svg"])]
    non_domain_max = int(non_domain["n_material"].max()) if not non_domain.empty else 0
    retention_rows: list[dict[str, Any]] = []
    retention_sub = retention[
        retention["representation_id"].isin([peak_representation, bin_representation])
        & (retention["method_id"] == primary_method_id)
    ].copy()
    for row in retention_sub.itertuples(index=False):
        if float(row.depth_fraction) not in set(float(d) for d in depths):
            continue
        retention_rows.append(
            {
                "study_family": row.study_family,
                "method_id": row.method_id,
                "representation_id": row.representation_id,
                "depth_fraction": float(row.depth_fraction),
                "n_success": int(row.n_success),
                "primary_frozen_retention": float(row.primary_frozen_retention),
            }
        )
    retention_frame = pd.DataFrame(retention_rows)
    h3_rows: list[dict[str, Any]] = []
    for study in sorted(retention_frame["study_family"].unique()) if not retention_frame.empty else []:
        for depth in risky_depths:
            p_rows = retention_frame[
                (retention_frame["study_family"] == study)
                & (retention_frame["representation_id"] == peak_representation)
                & np.isclose(retention_frame["depth_fraction"], float(depth))
            ]
            b_rows = retention_frame[
                (retention_frame["study_family"] == study)
                & (retention_frame["representation_id"] == bin_representation)
                & np.isclose(retention_frame["depth_fraction"], float(depth))
            ]
            p_value = float(p_rows.iloc[0]["primary_frozen_retention"]) if not p_rows.empty else float("nan")
            b_value = float(b_rows.iloc[0]["primary_frozen_retention"]) if not b_rows.empty else float("nan")
            h3_rows.append(
                {
                    "study_family": study,
                    "depth_fraction": float(depth),
                    "peak_retention": p_value,
                    "bin_retention": b_value,
                    "peak_is_zero": bool(np.isfinite(p_value) and abs(p_value) <= 1e-12),
                    "bin_is_positive": bool(np.isfinite(b_value) and b_value > 0.0),
                }
            )
    h3_frame = pd.DataFrame(h3_rows)
    summary = {
        "domain_material_risky_depths": risky,
        "domain_material_stable_depths": stable,
        "h1_depth_concentration": "DESCRIPTIVE"
        if decision_scope == "descriptive"
        else (
            "PASS"
            if risky > stable
            else ("PARTIAL" if risky == stable else "FAIL")
        ),
        "non_domain_material_max_per_depth": non_domain_max,
        "h2_non_domain_stability": "DESCRIPTIVE"
        if decision_scope == "descriptive"
        else (
            "PASS"
            if non_domain_max == 0
            else ("PARTIAL" if non_domain_max <= 2 else "FAIL")
        ),
        "h3_all_peak_zero_and_bin_positive": bool(
            not h3_frame.empty
            and bool(h3_frame["peak_is_zero"].all())
            and bool(h3_frame["bin_is_positive"].all())
        ),
    }
    summary["h3_paired_rna_consistency"] = (
        "DESCRIPTIVE"
        if decision_scope == "descriptive"
        else ("PASS" if summary["h3_all_peak_zero_and_bin_positive"] else "FAIL")
    )
    summary["track"] = track
    summary["decision_scope"] = decision_scope
    return {"material_by_depth": material, "retention": retention_frame, "retention_check": h3_frame}, summary


def e6_gap_table(items: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append(
            {
                "item_id": str(item["id"]),
                "status": str(item["status"]),
                "reason": str(item["reason"]),
                "evidence": str(item.get("evidence", "")),
                "experiment_id": str(item.get("experiment_id", "")),
                "claim_supported": str(item.get("claim_supported", "")),
                "outcome_invented": False,
            }
        )
    return pd.DataFrame(rows)


E7_ENDPOINT_SPECS = (
    ("domain_ARI", "domain", "ARI"),
    ("domain_NMI", "domain", "NMI"),
    ("marker", "marker", "mean_marker_percentile"),
    ("svg", "svg", "svg_spearman_to_rna"),
)


def _e7_selected_rows(
    frame: pd.DataFrame,
    metric: str,
    *,
    primary_contrast: str,
    depth: float,
) -> pd.DataFrame:
    """Select primary rows without adding or imputing any dataset."""

    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {
        "metric",
        "depth_fraction",
        "contrast",
        "dataset_key",
        "point_difference_b_minus_a",
    }
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame()
    sub = frame[frame["metric"].astype(str) == str(metric)].copy()
    numeric_depth = pd.to_numeric(sub["depth_fraction"], errors="coerce")
    sub = sub[np.isclose(numeric_depth, float(depth))]
    sub = sub[
        sub["contrast"].map(canonical_contrast)
        == canonical_contrast(primary_contrast)
    ]
    if "representation_claim_eligible" in sub.columns:
        eligible = (
            sub["representation_claim_eligible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "nan", "none", ""})
        )
        sub = sub[eligible]
    if "analysis_role" in sub.columns:
        sub = sub[
            sub["analysis_role"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"primary", "nan", "none", ""})
        ]
    if "track" in sub.columns:
        sub = sub[
            sub["track"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"representation", "nan", "none", ""})
        ]
    return sub


def e7_expected_dataset_endpoints(
    domain: pd.DataFrame,
    marker: pd.DataFrame,
    svg: pd.DataFrame,
    *,
    study_map: dict[str, str],
    primary_contrast: str = "B-P",
    depth: float = 1.0,
) -> set[tuple[str, str]]:
    """Return the frozen (endpoint, dataset) cells supported by input tables."""

    frames = {"domain": domain, "marker": marker, "svg": svg}
    expected: set[tuple[str, str]] = set()
    for endpoint, frame_key, metric in E7_ENDPOINT_SPECS:
        sub = _e7_selected_rows(
            frames[frame_key],
            metric,
            primary_contrast=primary_contrast,
            depth=depth,
        )
        if sub.empty:
            continue
        for row in sub.to_dict("records"):
            dataset_key = str(row["dataset_key"])
            effect = float(row["point_difference_b_minus_a"])
            if dataset_key in study_map and np.isfinite(effect):
                expected.add((endpoint, dataset_key))
    return expected


def e7_study_level_meta(
    domain: pd.DataFrame,
    marker: pd.DataFrame,
    svg: pd.DataFrame,
    *,
    study_map: dict[str, str],
    primary_contrast: str = "B-P",
    depth: float = 1.0,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Descriptive study-level dots and ranges for D09.

    With fewer than five independent study families, this function deliberately
    does not report a population-level confidence interval, I², or a
    hierarchical-model estimate.  It does verify, however, that every
    (endpoint, dataset) cell supported by the frozen input tables is present
    exactly once.  A missing table therefore cannot silently become an
    all-zero within-family range.
    """

    dots: list[dict[str, Any]] = []
    frames = {"domain": domain, "marker": marker, "svg": svg}
    for endpoint, frame_key, metric in E7_ENDPOINT_SPECS:
        sub = _e7_selected_rows(
            frames[frame_key],
            metric,
            primary_contrast=primary_contrast,
            depth=depth,
        )
        if sub.empty:
            continue
        for row in sub.to_dict("records"):
            dataset_key = str(row["dataset_key"])
            if dataset_key not in study_map:
                continue
            effect = float(row["point_difference_b_minus_a"])
            if not np.isfinite(effect):
                continue
            dots.append(
                {
                    "endpoint": endpoint,
                    "dataset_key": dataset_key,
                    "study_family": study_map[dataset_key],
                    "contrast": canonical_contrast(primary_contrast),
                    "depth_fraction": float(depth),
                    "effect_b_minus_a": effect,
                    "ci95_low": float(row.get("ci95_low", float("nan"))),
                    "ci95_high": float(row.get("ci95_high", float("nan"))),
                    "verdict": str(row.get("verdict", "")),
                }
            )

    dot_frame = pd.DataFrame(dots)
    expected = e7_expected_dataset_endpoints(
        domain,
        marker,
        svg,
        study_map=study_map,
        primary_contrast=primary_contrast,
        depth=depth,
    )
    observed = {
        (str(row["endpoint"]), str(row["dataset_key"]))
        for row in dot_frame.to_dict("records")
    } if not dot_frame.empty else set()
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if (
        dot_frame.empty and expected
        or missing
        or unexpected
        or len(dot_frame) != len(expected)
    ):
        raise ValueError(
            "E7 dataset-endpoint coverage mismatch: "
            f"expected={len(expected)} observed={len(dot_frame)} "
            f"missing={missing} unexpected={unexpected}"
        )
    if dot_frame.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "endpoint",
                "study_family",
                "n_datasets",
                "study_median_effect",
                "study_min_effect",
                "study_max_effect",
            ]
        )
        return {
            "study_dots": dot_frame,
            "study_summary": empty_summary,
            "heterogeneity": pd.DataFrame(),
        }, {"n_study_families": 0, "n_dataset_endpoints": 0}

    summary = (
        dot_frame.groupby(["endpoint", "study_family"])
        .agg(
            n_datasets=("dataset_key", "nunique"),
            study_median_effect=("effect_b_minus_a", "median"),
            study_min_effect=("effect_b_minus_a", "min"),
            study_max_effect=("effect_b_minus_a", "max"),
        )
        .reset_index()
    )
    summary["study_range"] = summary["study_max_effect"] - summary["study_min_effect"]
    heterogeneity = (
        summary.groupby("endpoint")
        .agg(
            n_study_families=("study_family", "nunique"),
            n_datasets=("n_datasets", "sum"),
            minimum_study_median=("study_median_effect", "min"),
            maximum_study_median=("study_median_effect", "max"),
            maximum_study_range=("study_range", "max"),
        )
        .reset_index()
    )
    heterogeneity["population_ci_reported"] = False
    heterogeneity["i_squared_reported"] = False
    heterogeneity["note"] = "descriptive only; fewer than five independent study families"
    summary["population_ci_reported"] = False
    summary["i_squared_reported"] = False
    result = {
        "n_study_families": int(summary["study_family"].nunique()),
        "n_dataset_endpoints": int(len(dot_frame)),
        "n_expected_dataset_endpoints": int(len(expected)),
        "dataset_keys": sorted(dot_frame["dataset_key"].unique().tolist()),
        "study_families": sorted(dot_frame["study_family"].unique().tolist()),
        "dataset_endpoints_by_endpoint": {
            endpoint: sorted(
                dot_frame[dot_frame["endpoint"] == endpoint]["dataset_key"].unique().tolist()
            )
            for endpoint in sorted(dot_frame["endpoint"].unique().tolist())
        },
        "endpoints": sorted(summary["endpoint"].unique().tolist()),
        "population_ci_reported": False,
        "i_squared_reported": False,
        "status": "DONE_DESCRIPTIVE",
    }
    return {
        "study_dots": dot_frame.sort_values(["endpoint", "study_family", "dataset_key"]),
        "study_summary": summary.sort_values(["endpoint", "study_family"]),
        "heterogeneity": heterogeneity.sort_values("endpoint"),
    }, result


def e8_policy_utility(
    leaves: pd.DataFrame,
    downstream_effects: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """No-refit downstream decision-utility audit for D12.

    ``downstream_effects`` must contain one value per dataset, representation,
    method, and endpoint.  Only methods with a downstream metric are rankable.
    Leaves whose recommended set has no downstream evidence abstain.
    """

    rep_scope_map = {
        "fixed_bin_10kb": "B10",
        "fixed_bin_25kb": "B",
        "fixed_bin_50kb": "B50",
        "macs3_summit_501bp": "P",
        "filtered_fragments_with_frozen_peak_observation_contract": "F",
    }
    rows: list[dict[str, Any]] = []
    for leaf in leaves.to_dict("records"):
        if str(leaf.get("status", "")) != "active_frozen":
            continue
        rep_key = rep_scope_map.get(str(leaf.get("representation_scope", "")), "")
        recommended = [
            value
            for value in str(leaf.get("recommended_methods", "")).split(";")
            if value and value != "nan"
        ]
        evidence = downstream_effects[
            downstream_effects["representation"].astype(str) == rep_key
        ]
        methods_with_evidence = sorted(evidence["method_id"].astype(str).unique().tolist())
        recommended_with_evidence = [
            method for method in recommended if method in set(methods_with_evidence)
        ]
        base = {
            "policy_id": str(leaf.get("policy_id", "")),
            "task_id": str(leaf.get("task_id", "")),
            "representation_scope": str(leaf.get("representation_scope", "")),
            "representation_key": rep_key,
            "recommended_methods": ";".join(recommended),
            "methods_with_downstream_evidence": ";".join(methods_with_evidence),
            "recommended_methods_with_downstream_evidence": ";".join(
                recommended_with_evidence
            ),
            "n_methods_with_downstream_evidence": int(len(methods_with_evidence)),
            "n_recommended_with_downstream_evidence": int(
                len(recommended_with_evidence)
            ),
            "policy_abstain": bool(not recommended_with_evidence),
            "abstention_reason": (
                "NO_DOWNSTREAM_EVIDENCE_FOR_RECOMMENDED_SET"
                if not recommended_with_evidence
                else ""
            ),
            "positive_utility_claim": False,
        }
        if not recommended_with_evidence:
            rows.append(
                {
                    **base,
                    "endpoint": "all",
                    "oracle_utility": float("nan"),
                    "policy_utility": float("nan"),
                    "downstream_regret": float("nan"),
                    "top1_coverage": float("nan"),
                    "top3_coverage": float("nan"),
                    "identifiability": "NO_DOWNSTREAM_EVIDENCE_FOR_RECOMMENDED_SET",
                }
            )
            continue
        for endpoint, group in evidence.groupby("endpoint"):
            values = group.groupby("method_id")["value"].mean().sort_values(ascending=False)
            if values.empty:
                continue
            oracle = float(values.max())
            policy_value = float(values.loc[recommended_with_evidence].max())
            top1 = set(values.index[:1].tolist())
            top3 = set(values.index[:3].tolist())
            rows.append(
                {
                    **base,
                    "endpoint": str(endpoint),
                    "oracle_utility": oracle,
                    "policy_utility": policy_value,
                    "downstream_regret": float(max(0.0, oracle - policy_value)),
                    "top1_coverage": float(len(top1 & set(recommended_with_evidence)) / len(top1)),
                    "top3_coverage": float(len(top3 & set(recommended_with_evidence)) / len(top3)),
                    "identifiability": (
                        "IDENTIFIABLE"
                        if len(values) >= 2
                        else "NOT_IDENTIFIABLE_SINGLE_METHOD"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"policy_utility": frame}, {"status": "DONE", "n_leaves": 0}
    identifiable = frame[~frame["policy_abstain"]]
    summary = {
        "status": "DONE",
        "n_active_leaves": int(frame["policy_id"].nunique()),
        "n_leaf_endpoint_rows": int(len(frame)),
        "n_abstain_leaves": int(frame[frame["policy_abstain"]]["policy_id"].nunique()),
        "n_leaves_with_any_downstream_evidence": int(
            (~frame["policy_abstain"]).groupby(frame["policy_id"]).any().sum()
        ),
        "abstention_rate": float(
            frame[frame["policy_abstain"]]["policy_id"].nunique()
            / max(1, frame["policy_id"].nunique())
        ),
        "n_rows_identifiable_single_method": int(
            (identifiable["identifiability"] == "NOT_IDENTIFIABLE_SINGLE_METHOD").sum()
        ),
        "n_rows_identifiable_multi_method": int(
            (identifiable["identifiability"] == "IDENTIFIABLE").sum()
        ),
        "max_downstream_regret": float(identifiable["downstream_regret"].max())
        if not identifiable.empty
        else float("nan"),
        "mean_top1_coverage": float(identifiable["top1_coverage"].mean())
        if not identifiable.empty
        else float("nan"),
        "mean_top3_coverage": float(identifiable["top3_coverage"].mean())
        if not identifiable.empty
        else float("nan"),
        "positive_utility_claim": False,
        "reason": (
            "No leaf has at least two evidence-backed methods; the frozen policy's "
            "recommended methods have no downstream evidence in the available tables."
        ),
    }
    return {"policy_utility": frame}, summary
