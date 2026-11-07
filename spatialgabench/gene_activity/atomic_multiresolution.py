"""Peak-free atomic-projection multiresolution gene activity prototype."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm


@dataclass(frozen=True)
class AtomicProjection:
    scale_kb: int
    expanded_weights: sparse.csr_matrix
    complete_atom_mask: np.ndarray


@dataclass(frozen=True)
class AtomicMultiresolutionResult:
    activity_rate: np.ndarray
    selected_scale_kb: np.ndarray
    selected_log_variance: np.ndarray
    maximum_normalized_discrepancy: np.ndarray
    distance_to_selection_boundary: np.ndarray
    adaptive_applicable: np.ndarray
    insufficient_count: np.ndarray
    boundary_not_applicable: np.ndarray
    scale_log_rates: dict[int, np.ndarray]
    pairwise_log_difference_variance: dict[tuple[int, int], np.ndarray]
    clipped_negative_variances: int


@dataclass(frozen=True)
class PreparedAtomicWeightPanel:
    weights_by_scale: dict[int, sparse.csr_matrix]
    boundary_not_applicable: np.ndarray


def _validate_weights(weights: sparse.spmatrix, atoms: int) -> sparse.csr_matrix:
    if not sparse.issparse(weights):
        raise TypeError("atomic weights must be sparse")
    result = weights.tocsr().astype(np.float64)
    if result.shape[0] != atoms:
        raise ValueError("atomic feature count does not match weight rows")
    if result.nnz and (not np.all(np.isfinite(result.data)) or np.any(result.data < 0)):
        raise ValueError("atomic weights must be finite and nonnegative")
    return result


def _atomic_table(features: pd.DataFrame, atomic_size: int) -> pd.DataFrame:
    required = {"chromosome", "start", "end"}
    if not required <= set(features.columns):
        raise ValueError(f"atomic features missing columns: {sorted(required - set(features.columns))}")
    table = features.reset_index(drop=True)[["chromosome", "start", "end"]].copy()
    table["chromosome"] = table.chromosome.astype(str)
    table["start"] = pd.to_numeric(table.start, errors="raise").astype(np.int64)
    table["end"] = pd.to_numeric(table.end, errors="raise").astype(np.int64)
    table["width"] = table.end - table.start
    if np.any(table.start < 0) or np.any(table.width <= 0) or np.any(table.width > atomic_size):
        raise ValueError("invalid atomic intervals")
    if table.duplicated(["chromosome", "start", "end"]).any():
        raise ValueError("atomic intervals are duplicated")
    return table


def project_atomic_weights(
    features: pd.DataFrame,
    atomic_weights: sparse.spmatrix,
    parent_size: int,
    *,
    atomic_size: int = 5_000,
) -> AtomicProjection:
    """Project atomic weights to complete parents and expand them back to atoms."""
    if parent_size <= atomic_size or parent_size % atomic_size:
        raise ValueError("parent_size must be an integer multiple larger than atomic_size")
    table = _atomic_table(features, atomic_size)
    weights = _validate_weights(atomic_weights, len(table))
    table["parent_start"] = (table.start // parent_size) * parent_size
    group_columns = ["chromosome", "parent_start"]
    expected_atoms = parent_size // atomic_size
    group = table.groupby(group_columns, sort=False, observed=True)
    counts = group.size().rename("atoms")
    widths = group.width.sum().rename("width")
    parent = table[group_columns].merge(
        pd.concat([counts, widths], axis=1).reset_index(), on=group_columns, how="left", validate="many_to_one",
    )
    complete = (parent.atoms.to_numpy(np.int64) == expected_atoms) & (
        parent.width.to_numpy(np.int64) == parent_size
    )

    complete_keys = (
        parent.loc[complete, group_columns].drop_duplicates().reset_index(drop=True)
    )
    key_to_parent = {
        (str(row.chromosome), int(row.parent_start)): index
        for index, row in enumerate(complete_keys.itertuples(index=False))
    }
    child_rows = np.flatnonzero(complete)
    parent_columns = np.asarray(
        [key_to_parent[(str(parent.iloc[row].chromosome), int(parent.iloc[row].parent_start))] for row in child_rows],
        dtype=np.int64,
    )
    child_widths = table.width.to_numpy(np.float64)[child_rows]
    aggregation = sparse.coo_matrix(
        (child_widths / float(parent_size), (parent_columns, child_rows)),
        shape=(len(complete_keys), len(table)),
    ).tocsr()
    expansion = sparse.coo_matrix(
        (np.ones(len(child_rows), dtype=np.float64), (child_rows, parent_columns)),
        shape=(len(table), len(complete_keys)),
    ).tocsr()
    projected = (expansion @ (aggregation @ weights)).tocsr()
    projected.eliminate_zeros()
    return AtomicProjection(
        scale_kb=parent_size // 1_000,
        expanded_weights=projected,
        complete_atom_mask=complete,
    )


def boundary_genes(
    atomic_weights: sparse.spmatrix,
    complete_atom_mask: np.ndarray,
) -> np.ndarray:
    weights = atomic_weights.tocsr()
    complete = np.asarray(complete_atom_mask, dtype=bool)
    if complete.shape != (weights.shape[0],):
        raise ValueError("complete_atom_mask has the wrong shape")
    incomplete_support = weights[~complete, :]
    return np.asarray(incomplete_support.getnnz(axis=0)).ravel() > 0


def prepare_atomic_weight_panel(
    features: pd.DataFrame,
    atomic_weights: sparse.spmatrix,
    *,
    scales_kb: tuple[int, int, int] = (5, 25, 100),
) -> PreparedAtomicWeightPanel:
    """Prepare frozen projections once for reuse across observation or gene chunks."""
    if scales_kb != (5, 25, 100):
        raise ValueError("prototype requires frozen scales (5, 25, 100)")
    weights5 = _validate_weights(atomic_weights, len(features))
    projection25 = project_atomic_weights(features, weights5, 25_000)
    projection100 = project_atomic_weights(features, weights5, 100_000)
    return PreparedAtomicWeightPanel(
        weights_by_scale={
            5: weights5,
            25: projection25.expanded_weights,
            100: projection100.expanded_weights,
        },
        boundary_not_applicable=boundary_genes(weights5, projection100.complete_atom_mask),
    )


def _dense_product(counts: sparse.csr_matrix, weights: sparse.csr_matrix) -> np.ndarray:
    return np.asarray((counts @ weights).toarray(), dtype=np.float64)


def _conditional_covariance_of_means(
    counts: sparse.csr_matrix,
    weights_i: sparse.csr_matrix,
    weights_j: sparse.csr_matrix,
    sum_i: np.ndarray,
    sum_j: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    cross = _dense_product(counts, weights_i.multiply(weights_j))
    result = np.full(sum_i.shape, np.nan, dtype=np.float64)
    valid = totals > 1
    t = totals[valid, None]
    centered_sum = cross[valid] - (sum_i[valid] * sum_j[valid] / t)
    result[valid] = centered_sum / (t * (t - 1.0))
    return result


def fit_preprojected_atomic_multiresolution(
    atomic_counts: sparse.spmatrix,
    weights_by_scale: dict[int, sparse.spmatrix],
    boundary_not_applicable: np.ndarray,
    *,
    scales_kb: tuple[int, int, int] = (5, 25, 100),
    scale_factor: float = 10_000.0,
    alpha: float = 0.05,
    negative_tolerance: float = 1e-12,
) -> AtomicMultiresolutionResult:
    if scales_kb != (5, 25, 100):
        raise ValueError("prototype requires frozen scales (5, 25, 100)")
    if not sparse.issparse(atomic_counts):
        raise TypeError("atomic_counts must be sparse")
    counts = atomic_counts.tocsr().astype(np.float64)
    if counts.nnz and (not np.all(np.isfinite(counts.data)) or np.any(counts.data < 0)):
        raise ValueError("atomic_counts must be finite and nonnegative")
    if set(weights_by_scale) != set(scales_kb):
        raise ValueError("preprojected panel must contain exactly the frozen scales")
    weights = {
        scale: _validate_weights(weights_by_scale[scale], counts.shape[1]) for scale in scales_kb
    }
    genes = {value.shape[1] for value in weights.values()}
    if len(genes) != 1:
        raise ValueError("preprojected weight matrices have different gene dimensions")
    boundary = np.asarray(boundary_not_applicable, dtype=bool)
    if boundary.shape != (next(iter(genes)),):
        raise ValueError("boundary_not_applicable has the wrong shape")
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    insufficient = totals <= 1

    sums = {scale: _dense_product(counts, value) for scale, value in weights.items()}
    rates: dict[int, np.ndarray] = {}
    logs: dict[int, np.ndarray] = {}
    for scale in scales_kb:
        rate = np.zeros_like(sums[scale])
        valid = totals > 0
        rate[valid] = scale_factor * sums[scale][valid] / totals[valid, None]
        rates[scale] = rate
        logs[scale] = np.log1p(rate)

    covariance_m: dict[tuple[int, int], np.ndarray] = {}
    covariance_log: dict[tuple[int, int], np.ndarray] = {}
    for first in scales_kb:
        for second in scales_kb:
            if second < first:
                continue
            key = (first, second)
            cov_m = _conditional_covariance_of_means(
                counts, weights[first], weights[second], sums[first], sums[second], totals,
            )
            covariance_m[key] = cov_m
            covariance_r = np.square(scale_factor) * cov_m
            covariance_log[key] = covariance_r / ((1.0 + rates[first]) * (1.0 + rates[second]))

    pair_variance: dict[tuple[int, int], np.ndarray] = {}
    clipped = 0
    for first, second in combinations(scales_kb, 2):
        variance = (
            covariance_log[(first, first)]
            + covariance_log[(second, second)]
            - 2.0 * covariance_log[(first, second)]
        )
        finite_negative = np.isfinite(variance) & (variance < 0)
        if np.any(variance[finite_negative] < -negative_tolerance):
            raise ValueError("pairwise log-difference variance is negative beyond tolerance")
        clipped += int(finite_negative.sum())
        variance[finite_negative] = 0.0
        pair_variance[(first, second)] = variance

    z = float(norm.ppf(1.0 - alpha / 6.0))

    def discrepancy(first: int, second: int) -> tuple[np.ndarray, np.ndarray]:
        difference = np.abs(logs[first] - logs[second])
        variance = pair_variance[(first, second)]
        threshold = z * np.sqrt(variance)
        consistent = difference <= threshold
        zero_variance = variance == 0
        ratio = np.divide(
            difference, np.sqrt(variance),
            out=np.where(difference == 0, 0.0, np.inf),
            where=~zero_variance,
        )
        return consistent, ratio

    c5_25, r5_25 = discrepancy(5, 25)
    c5_100, r5_100 = discrepancy(5, 100)
    c25_100, r25_100 = discrepancy(25, 100)
    shape = rates[5].shape
    selected = np.full(shape, 5, dtype=np.int16)
    coarse = c5_100 & c25_100
    medium = (~coarse) & c5_25
    selected[medium] = 25
    selected[coarse] = 100
    selected[totals == 1, :] = 100
    selected[totals == 0, :] = 0
    selected[:, boundary] = 5

    activity_rate = np.zeros(shape, dtype=np.float64)
    selected_variance = np.full(shape, np.nan, dtype=np.float64)
    for scale in scales_kb:
        mask = selected == scale
        activity_rate[mask] = (rates[scale] / scale_factor)[mask]
        selected_variance[mask] = covariance_log[(scale, scale)][mask]
    activity_rate[totals == 0, :] = 0.0
    adaptive = (~insufficient[:, None]) & (~boundary[None, :])
    maximum_discrepancy = np.maximum.reduce([r5_25, r5_100, r25_100])
    boundary_distance = np.minimum.reduce([np.abs(r5_25 - z), np.abs(r5_100 - z), np.abs(r25_100 - z)])
    maximum_discrepancy[~adaptive] = np.nan
    boundary_distance[~adaptive] = np.nan
    selected_variance[~adaptive] = np.nan
    return AtomicMultiresolutionResult(
        activity_rate=activity_rate,
        selected_scale_kb=selected,
        selected_log_variance=selected_variance,
        maximum_normalized_discrepancy=maximum_discrepancy,
        distance_to_selection_boundary=boundary_distance,
        adaptive_applicable=adaptive,
        insufficient_count=insufficient,
        boundary_not_applicable=boundary,
        scale_log_rates=logs,
        pairwise_log_difference_variance=pair_variance,
        clipped_negative_variances=clipped,
    )


def fit_atomic_multiresolution(
    atomic_counts: sparse.spmatrix,
    atomic_weights: sparse.spmatrix,
    features: pd.DataFrame,
    *,
    scales_kb: tuple[int, int, int] = (5, 25, 100),
    scale_factor: float = 10_000.0,
    alpha: float = 0.05,
    negative_tolerance: float = 1e-12,
) -> AtomicMultiresolutionResult:
    """Fit from atomic weights, including one-time preparation of projected weights."""
    panel = prepare_atomic_weight_panel(features, atomic_weights, scales_kb=scales_kb)
    return fit_preprojected_atomic_multiresolution(
        atomic_counts,
        panel.weights_by_scale,
        panel.boundary_not_applicable,
        scales_kb=scales_kb,
        scale_factor=scale_factor,
        alpha=alpha,
        negative_tolerance=negative_tolerance,
    )
