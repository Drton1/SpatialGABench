"""ATAC-only reliability-weighted multiscale gene activity prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MultiscaleResult:
    activity_rate: np.ndarray
    peak_weight: np.ndarray
    continuous_risk: np.ndarray
    conditional_delete_group_risk: np.ndarray
    count_risk_upper: np.ndarray
    representation_disagreement: np.ndarray
    peak_signal_variance: np.ndarray
    peak_structural_risk: np.ndarray
    valid_peak_perturbations: np.ndarray


def _as_nonnegative_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return result


def _as_exposure(value: np.ndarray, observations: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (observations,):
        raise ValueError("exposure must have one value per observation")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("exposure must contain finite nonnegative values")
    return np.maximum(result, 1.0)


def log_rate(activity: np.ndarray, exposure: np.ndarray, scale: float = 10_000.0) -> np.ndarray:
    values = _as_nonnegative_matrix(activity, "activity")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    denominator = _as_exposure(exposure, values.shape[0])[:, None]
    return np.log1p(scale * values / denominator)


def delta_count_variance(
    activity: np.ndarray,
    weighted_count_variance: np.ndarray,
    exposure: np.ndarray,
    scale: float = 10_000.0,
) -> np.ndarray:
    values = _as_nonnegative_matrix(activity, "activity")
    variance = _as_nonnegative_matrix(weighted_count_variance, "weighted_count_variance")
    if variance.shape != values.shape:
        raise ValueError("activity and weighted_count_variance must have identical shapes")
    denominator = _as_exposure(exposure, values.shape[0])[:, None]
    rate = scale * values / denominator
    derivative = (scale / denominator) / (1.0 + rate)
    return np.square(derivative) * variance


def weighted_activity_moments(
    counts: sparse.spmatrix,
    mapping: sparse.spmatrix,
    regularizer: float = 0.5,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Return weighted activity, event variance, and branch-specific weight floor.

    The Poisson working variance uses the same mapped insertion counts with squared
    mapper weights. The returned floor is added per gene by downstream chunked
    code, avoiding a dense spot-by-gene broadcast here.
    """
    if not sparse.issparse(counts) or not sparse.issparse(mapping):
        raise TypeError("counts and mapping must be sparse matrices")
    if counts.shape[1] != mapping.shape[0]:
        raise ValueError("counts features do not match mapping rows")
    if regularizer < 0 or not np.isfinite(regularizer):
        raise ValueError("regularizer must be finite and nonnegative")
    x = counts.tocsr().astype(np.float64)
    weights = mapping.tocsr().astype(np.float64)
    if x.nnz and (np.any(x.data < 0) or not np.all(np.isfinite(x.data))):
        raise ValueError("counts must contain finite nonnegative values")
    if weights.nnz and (np.any(weights.data < 0) or not np.all(np.isfinite(weights.data))):
        raise ValueError("mapping must contain finite nonnegative values")
    activity = (x @ weights).tocsr()
    event_variance = (x @ weights.multiply(weights)).tocsr()
    squared = weights.multiply(weights).tocsc()
    mapped = np.diff(squared.indptr)
    sums = np.asarray(squared.sum(axis=0)).ravel()
    mean_squared = np.divide(sums, mapped, out=np.zeros_like(sums), where=mapped > 0)
    return activity, event_variance, regularizer * mean_squared


def _validate_perturbations(
    values: Sequence[np.ndarray],
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    if len(values) < 2:
        raise ValueError(f"{name} requires at least two perturbations")
    arrays = [_as_nonnegative_matrix(value, name) for value in values]
    if any(value.shape != expected_shape for value in arrays):
        raise ValueError(f"{name} perturbations must match the full matrix shape")
    return np.stack(arrays, axis=0)


def _validate_perturbation_exposures(
    values: Sequence[np.ndarray] | None,
    perturbations: int,
    full_exposure: np.ndarray,
) -> np.ndarray:
    if values is None:
        return np.repeat((0.75 * full_exposure)[None, :], perturbations, axis=0)
    if len(values) != perturbations:
        raise ValueError("perturbation exposures must match the number of perturbations")
    return np.stack(
        [_as_exposure(value, full_exposure.shape[0]) for value in values],
        axis=0,
    )


def valid_delete_group_variance(
    values: np.ndarray,
    valid_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute delete-group perturbation summaries over valid feature replicates."""
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_features, dtype=bool)
    if array.ndim != 3:
        raise ValueError("values must have shape perturbation x observation x gene")
    if valid.shape != (array.shape[0], array.shape[2]):
        raise ValueError("valid_features must have shape perturbation x gene")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("perturbation values must be finite and nonnegative")

    observations, genes = array.shape[1:]
    means = np.zeros((observations, genes), dtype=np.float64)
    variances = np.full((observations, genes), np.nan, dtype=np.float64)
    counts = valid.sum(axis=0).astype(np.int64)
    for gene in range(genes):
        selected = array[valid[:, gene], :, gene]
        if selected.shape[0] == 0:
            continue
        means[:, gene] = selected.mean(axis=0)
        if selected.shape[0] >= 2:
            centered = selected - means[:, gene]
            factor = (selected.shape[0] - 1.0) / selected.shape[0]
            variances[:, gene] = factor * np.square(centered).sum(axis=0)
    return means, variances, counts


def _winsorized_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    if values.size == 0:
        return np.nan
    lower, upper = np.quantile(values, [proportion, 1.0 - proportion])
    return float(np.clip(values, lower, upper).mean())


def robust_peak_signal_variance(
    peak_log_rate: np.ndarray,
    peak_noise: np.ndarray,
    valid_gene: np.ndarray,
    minimum_spots: int,
) -> np.ndarray:
    values = np.asarray(peak_log_rate, dtype=np.float64)
    noise = np.asarray(peak_noise, dtype=np.float64)
    valid = np.asarray(valid_gene, dtype=bool)
    if values.ndim != 2 or noise.shape != values.shape or valid.shape != (values.shape[1],):
        raise ValueError("invalid robust variance input dimensions")
    if minimum_spots < 2:
        raise ValueError("minimum_spots must be at least two")

    result = np.zeros(values.shape[1], dtype=np.float64)
    for gene in range(values.shape[1]):
        if not valid[gene]:
            continue
        mask = np.isfinite(values[:, gene]) & np.isfinite(noise[:, gene])
        if int(mask.sum()) < minimum_spots:
            continue
        observed = values[mask, gene]
        median = np.median(observed)
        mad = np.median(np.abs(observed - median))
        robust_variance = np.square(mad / 0.67448975)
        mean_noise = _winsorized_mean(noise[mask, gene], 0.1)
        result[gene] = max(float(robust_variance - mean_noise), 0.0)
    return result


def fit_reliability_multiscale(
    peak_activity: np.ndarray,
    bin_activity: np.ndarray,
    peak_weighted_variance: np.ndarray,
    bin_weighted_variance: np.ndarray,
    exposure: np.ndarray,
    peak_perturbations: Sequence[np.ndarray],
    bin_perturbations: Sequence[np.ndarray],
    peak_perturbation_valid: np.ndarray,
    full_peak_valid: np.ndarray,
    *,
    perturbation_exposures: Sequence[np.ndarray] | None = None,
    scale: float = 10_000.0,
    minimum_spots: int | None = None,
    epsilon: float = 1e-12,
) -> MultiscaleResult:
    peak = _as_nonnegative_matrix(peak_activity, "peak_activity")
    bins = _as_nonnegative_matrix(bin_activity, "bin_activity")
    if bins.shape != peak.shape:
        raise ValueError("peak and bin activity must have identical shapes")
    observations, genes = peak.shape
    exp = _as_exposure(exposure, observations)
    peak_log = log_rate(peak, exp, scale)
    bin_log = log_rate(bins, exp, scale)
    peak_count_noise = delta_count_variance(peak, peak_weighted_variance, exp, scale)
    bin_count_noise = delta_count_variance(bins, bin_weighted_variance, exp, scale)

    peak_stack = _validate_perturbations(peak_perturbations, peak.shape, "peak_perturbations")
    bin_stack = _validate_perturbations(bin_perturbations, bins.shape, "bin_perturbations")
    if peak_stack.shape != bin_stack.shape:
        raise ValueError("peak and bin perturbation counts must match")
    perturbation_exposure = _validate_perturbation_exposures(
        perturbation_exposures, peak_stack.shape[0], exp,
    )
    valid = np.asarray(peak_perturbation_valid, dtype=bool)
    if valid.shape != (peak_stack.shape[0], genes):
        raise ValueError("peak_perturbation_valid has the wrong shape")
    full_valid = np.asarray(full_peak_valid, dtype=bool)
    if full_valid.shape != (genes,):
        raise ValueError("full_peak_valid must have one value per gene")

    peak_logs = np.stack(
        [log_rate(value, perturbation_exposure[index], scale) for index, value in enumerate(peak_stack)],
        axis=0,
    )
    bin_logs = np.stack(
        [log_rate(value, perturbation_exposure[index], scale) for index, value in enumerate(bin_stack)],
        axis=0,
    )
    _, peak_delete_variance, valid_counts = valid_delete_group_variance(peak_logs, valid)
    valid_reliability = full_valid & (valid_counts >= 2)
    peak_noise = peak_delete_variance + peak_count_noise
    peak_noise[:, ~valid_reliability] = np.inf

    n_min = minimum_spots if minimum_spots is not None else max(20, int(np.ceil(0.01 * observations)))
    signal_variance = robust_peak_signal_variance(peak_log, peak_noise, valid_reliability, n_min)
    peak_weight = np.zeros_like(peak_log)
    eligible = valid_reliability & (signal_variance > 0)
    if np.any(eligible):
        numerator = signal_variance[eligible][None, :]
        peak_weight[:, eligible] = numerator / (numerator + peak_noise[:, eligible] + epsilon)
    peak_weight = np.clip(peak_weight, 0.0, 1.0)

    combined_log = peak_weight * peak_log + (1.0 - peak_weight) * bin_log
    activity_rate = np.expm1(combined_log) / scale
    disagreement = np.abs(peak_log - bin_log)

    combined_perturbations = peak_weight[None, :, :] * peak_logs + (1.0 - peak_weight[None, :, :]) * bin_logs
    _, conditional_risk, _ = valid_delete_group_variance(combined_perturbations, valid)
    conditional_risk[:, ~valid_reliability] = 0.0
    conditional_risk = np.nan_to_num(conditional_risk, nan=0.0, posinf=0.0, neginf=0.0)
    count_risk = np.square(
        peak_weight * np.sqrt(peak_count_noise)
        + (1.0 - peak_weight) * np.sqrt(bin_count_noise)
    )
    continuous_risk = conditional_risk + count_risk
    structural_risk = 1.0 - valid_counts.astype(np.float64) / float(peak_stack.shape[0])

    return MultiscaleResult(
        activity_rate=activity_rate,
        peak_weight=peak_weight,
        continuous_risk=continuous_risk,
        conditional_delete_group_risk=conditional_risk,
        count_risk_upper=count_risk,
        representation_disagreement=disagreement,
        peak_signal_variance=signal_variance,
        peak_structural_risk=structural_risk,
        valid_peak_perturbations=valid_counts,
    )
