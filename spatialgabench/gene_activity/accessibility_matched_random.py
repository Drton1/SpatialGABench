"""Accessibility-matched random peak-to-gene negative control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .interval_aggregation import PeakGeneMapping, build_peak_gene_matrix


REQUIRED_PROMOTER_LINK_COLUMNS = [
    "peak_index",
    "peak_id",
    "gene_index",
    "gene_id",
    "gene_name",
]


@dataclass(frozen=True)
class AccessibilityMatchedResult:
    mapping: PeakGeneMapping
    matching_qc: pd.DataFrame
    peak_metrics: pd.DataFrame


def compute_peak_accessibility_metrics(X, peaks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute per-peak detection frequency and total counts from a sparse matrix."""

    if not sparse.issparse(X):
        raise ValueError("Input peak matrix must be sparse; refusing to densify.")
    X_csc = X.tocsc(copy=True)
    X_csc.eliminate_zeros()
    n_cells, n_peaks = X_csc.shape
    if X_csc.data.size and (X_csc.data < 0).any():
        raise ValueError("Input peak matrix contains negative values.")
    detected_cells = np.diff(X_csc.indptr).astype(np.int64)
    total_counts = np.asarray(X_csc.sum(axis=0)).ravel().astype(np.float64)
    metrics = pd.DataFrame(
        {
            "peak_index": np.arange(n_peaks, dtype=np.int64),
            "peak_detection_frequency": detected_cells.astype(np.float64) / float(n_cells),
            "peak_total_counts": total_counts,
        }
    )
    if peaks is not None:
        required = {"peak_index", "peak_id", "chr", "start", "end"}
        missing = sorted(required.difference(peaks.columns))
        if missing:
            raise ValueError(f"Peaks table missing required columns: {missing}")
        peak_info = peaks[["peak_index", "peak_id", "chr", "start", "end"]].copy()
        metrics = metrics.merge(peak_info, on="peak_index", how="left", validate="one_to_one")
    return metrics


def make_accessibility_bins(metrics: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """Assign quantile bins by peak detection frequency, robust to duplicate quantiles."""

    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if "peak_detection_frequency" not in metrics.columns:
        raise ValueError("metrics must contain peak_detection_frequency")
    result = metrics.copy()
    values = result["peak_detection_frequency"].to_numpy(dtype=np.float64)
    if values.size == 0:
        result["accessibility_bin"] = np.array([], dtype=np.int64)
        return result
    if not np.isfinite(values).all():
        raise ValueError("peak_detection_frequency contains non-finite values")
    if np.unique(values).size == 1:
        bins = np.zeros(values.shape[0], dtype=np.int64)
    else:
        binned = pd.qcut(values, q=min(n_bins, values.shape[0]), labels=False, duplicates="drop")
        bins = pd.Series(binned).fillna(0).astype(np.int64).to_numpy()
    result["accessibility_bin"] = bins
    return result


def load_reference_promoter_links(path: str | Path) -> pd.DataFrame:
    links = pd.read_csv(path, sep="\t")
    missing = [column for column in REQUIRED_PROMOTER_LINK_COLUMNS if column not in links.columns]
    if missing:
        raise ValueError(f"Reference promoter links missing required columns: {missing}")
    links = links.copy()
    links["peak_index"] = pd.to_numeric(links["peak_index"], errors="raise").astype(np.int64)
    links["gene_index"] = pd.to_numeric(links["gene_index"], errors="raise").astype(np.int64)
    links["gene_id"] = links["gene_id"].astype(str)
    links["gene_name"] = links["gene_name"].astype(str)
    links["peak_id"] = links["peak_id"].astype(str)
    return links.reset_index(drop=True)


def _candidate_indices_for_bins(bin_to_peak_indices: dict[int, np.ndarray], bins: list[int]) -> np.ndarray:
    pieces = [bin_to_peak_indices[int(bin_id)] for bin_id in bins if int(bin_id) in bin_to_peak_indices]
    if not pieces:
        return np.array([], dtype=np.int64)
    if len(pieces) == 1:
        return pieces[0]
    return np.concatenate(pieces).astype(np.int64, copy=False)


def _neighbor_bins(target_bin: int, radius: int, min_bin: int, max_bin: int) -> list[int]:
    if radius == 0:
        return [target_bin]
    bins: list[int] = []
    left = target_bin - radius
    right = target_bin + radius
    if left >= min_bin:
        bins.append(left)
    if right <= max_bin:
        bins.append(right)
    return bins


def sample_accessibility_matched_peaks(
    *,
    peak_metrics: pd.DataFrame,
    reference_promoter_links: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    seed: int,
    n_bins: int = 20,
    within_gene_replacement: bool = False,
    matched_from_method: str = "promoter_only_2kb_pc",
) -> AccessibilityMatchedResult:
    """Sample one matched random peak for each reference promoter peak."""

    if "accessibility_bin" not in peak_metrics.columns:
        peak_metrics = make_accessibility_bins(peak_metrics, n_bins=n_bins)
    required_metric_columns = {
        "peak_index",
        "peak_id",
        "chr",
        "start",
        "end",
        "accessibility_bin",
        "peak_detection_frequency",
        "peak_total_counts",
    }
    missing_metrics = sorted(required_metric_columns.difference(peak_metrics.columns))
    if missing_metrics:
        raise ValueError(f"peak_metrics missing required columns: {missing_metrics}")
    missing_gene = sorted({"gene_id", "gene_name"}.difference(gene_metadata.columns))
    if missing_gene:
        raise ValueError(f"gene_metadata missing required columns: {missing_gene}")
    missing_links = [column for column in REQUIRED_PROMOTER_LINK_COLUMNS if column not in reference_promoter_links.columns]
    if missing_links:
        raise ValueError(f"reference_promoter_links missing required columns: {missing_links}")

    metrics = peak_metrics.sort_values("peak_index").reset_index(drop=True)
    if not np.array_equal(metrics["peak_index"].to_numpy(np.int64), np.arange(metrics.shape[0], dtype=np.int64)):
        raise ValueError("peak_metrics peak_index must cover all peaks in ascending order")
    metric_by_peak = metrics.set_index("peak_index", drop=False)
    bin_values = metrics["accessibility_bin"].to_numpy(np.int64)
    min_bin = int(bin_values.min()) if bin_values.size else 0
    max_bin = int(bin_values.max()) if bin_values.size else 0
    bin_to_peak_indices = {
        int(bin_id): group["peak_index"].to_numpy(np.int64)
        for bin_id, group in metrics.groupby("accessibility_bin", sort=True)
    }

    rng = np.random.default_rng(seed)
    rows: list[int] = []
    cols: list[int] = []
    records: list[dict[str, object]] = []
    fallback_distances: list[int] = []
    original_detection: list[float] = []
    matched_detection: list[float] = []
    original_total: list[float] = []
    matched_total: list[float] = []

    links = reference_promoter_links.copy().reset_index(drop=True)
    links["gene_index"] = pd.to_numeric(links["gene_index"], errors="raise").astype(np.int64)
    links["peak_index"] = pd.to_numeric(links["peak_index"], errors="raise").astype(np.int64)

    n_peaks = metrics.shape[0]
    n_genes = gene_metadata.shape[0]
    for gene_index, gene_links in links.groupby("gene_index", sort=True):
        gene_index = int(gene_index)
        if gene_index < 0 or gene_index >= n_genes:
            raise ValueError(f"reference link has out-of-range gene_index={gene_index}")
        if (not within_gene_replacement) and gene_links.shape[0] > n_peaks:
            raise ValueError(f"Cannot sample {gene_links.shape[0]} peaks without replacement from {n_peaks} peaks")
        used_for_gene: set[int] = set()
        gene = gene_metadata.iloc[gene_index]
        for ref in gene_links.itertuples(index=False):
            ref_peak_index = int(ref.peak_index)
            if ref_peak_index not in metric_by_peak.index:
                raise ValueError(f"reference link has out-of-range peak_index={ref_peak_index}")
            ref_metric = metric_by_peak.loc[ref_peak_index]
            target_bin = int(ref_metric["accessibility_bin"])
            selected_peak_index: int | None = None
            selected_distance = 0
            max_radius = max(abs(target_bin - min_bin), abs(max_bin - target_bin))
            for radius in range(max_radius + 1):
                candidate_bins = _neighbor_bins(target_bin, radius, min_bin, max_bin)
                candidate_indices = _candidate_indices_for_bins(bin_to_peak_indices, candidate_bins)
                if candidate_indices.size == 0:
                    continue
                if not within_gene_replacement:
                    if candidate_indices.size <= len(used_for_gene):
                        candidate_indices = np.asarray(
                            [idx for idx in candidate_indices if int(idx) not in used_for_gene],
                            dtype=np.int64,
                        )
                        if candidate_indices.size == 0:
                            continue
                    else:
                        while True:
                            candidate = int(rng.choice(candidate_indices))
                            if candidate not in used_for_gene:
                                selected_peak_index = candidate
                                break
                if selected_peak_index is None:
                    selected_peak_index = int(rng.choice(candidate_indices))
                selected_distance = radius
                break
            if selected_peak_index is None:
                raise ValueError(
                    f"Unable to preserve per-gene peak count for gene_index={gene_index}; "
                    "all candidate peaks were exhausted."
                )

            used_for_gene.add(selected_peak_index)
            selected_metric = metric_by_peak.loc[selected_peak_index]
            rows.append(selected_peak_index)
            cols.append(gene_index)
            fallback_distances.append(selected_distance)
            original_detection.append(float(ref_metric["peak_detection_frequency"]))
            matched_detection.append(float(selected_metric["peak_detection_frequency"]))
            original_total.append(float(ref_metric["peak_total_counts"]))
            matched_total.append(float(selected_metric["peak_total_counts"]))
            records.append(
                {
                    "peak_index": selected_peak_index,
                    "peak_id": str(selected_metric["peak_id"]),
                    "peak_chr": str(selected_metric["chr"]),
                    "peak_start": int(selected_metric["start"]),
                    "peak_end": int(selected_metric["end"]),
                    "gene_index": gene_index,
                    "gene_id": str(gene["gene_id"]),
                    "gene_name": str(gene["gene_name"]),
                    "random_seed": int(seed),
                    "accessibility_bin": int(selected_metric["accessibility_bin"]),
                    "peak_detection_frequency": float(selected_metric["peak_detection_frequency"]),
                    "peak_total_counts": float(selected_metric["peak_total_counts"]),
                    "matched_from_method": matched_from_method,
                    "link_type": "accessibility_matched_random",
                    "target_promoter_peak_index": ref_peak_index,
                    "target_promoter_peak_id": str(ref_metric["peak_id"]),
                    "target_promoter_accessibility_bin": target_bin,
                    "target_promoter_detection_frequency": float(ref_metric["peak_detection_frequency"]),
                    "target_promoter_total_counts": float(ref_metric["peak_total_counts"]),
                    "bin_distance": int(selected_distance),
                    "used_adjacent_bin_fallback": bool(selected_distance > 0),
                }
            )

    link_columns = [
        "peak_index",
        "peak_id",
        "peak_chr",
        "peak_start",
        "peak_end",
        "gene_index",
        "gene_id",
        "gene_name",
        "random_seed",
        "accessibility_bin",
        "peak_detection_frequency",
        "peak_total_counts",
        "matched_from_method",
        "link_type",
        "target_promoter_peak_index",
        "target_promoter_peak_id",
        "target_promoter_accessibility_bin",
        "target_promoter_detection_frequency",
        "target_promoter_total_counts",
        "bin_distance",
        "used_adjacent_bin_fallback",
    ]
    matched_links = pd.DataFrame(records, columns=link_columns)
    mapping = build_peak_gene_matrix(matched_links, n_peaks=n_peaks, n_genes=n_genes)
    peak_counts = np.asarray(mapping.sum(axis=1)).ravel()
    gene_counts = np.asarray(mapping.sum(axis=0)).ravel()
    genes = gene_metadata.copy().reset_index(drop=True)
    genes["n_assigned_peaks"] = gene_counts.astype(np.int64)

    ref_counts = np.zeros(n_genes, dtype=np.int64)
    if not links.empty:
        ref_counts += links.groupby("gene_index").size().reindex(np.arange(n_genes), fill_value=0).to_numpy(dtype=np.int64)
    fallback_array = np.asarray(fallback_distances, dtype=np.int64)
    original_detection_array = np.asarray(original_detection, dtype=np.float64)
    matched_detection_array = np.asarray(matched_detection, dtype=np.float64)
    original_total_array = np.asarray(original_total, dtype=np.float64)
    matched_total_array = np.asarray(matched_total, dtype=np.float64)
    count_preserved = bool(np.array_equal(gene_counts.astype(np.int64), ref_counts))
    zero_preserved = bool(((ref_counts == 0) == (gene_counts == 0)).all())
    matching_qc = pd.DataFrame(
        [
            {
                "random_seed": int(seed),
                "n_accessibility_bins_requested": int(n_bins),
                "n_accessibility_bins_observed": int(metrics["accessibility_bin"].nunique()),
                "n_reference_promoter_links": int(links.shape[0]),
                "n_matched_random_links": int(matched_links.shape[0]),
                "n_reference_genes": int(n_genes),
                "n_genes_with_reference_peaks": int((ref_counts > 0).sum()),
                "n_zero_peak_genes_in_reference": int((ref_counts == 0).sum()),
                "n_zero_peak_genes_in_matched": int((gene_counts == 0).sum()),
                "per_gene_peak_count_preserved": count_preserved,
                "zero_peak_genes_preserved": zero_preserved,
                "same_bin_link_count": int((fallback_array == 0).sum()),
                "adjacent_fallback_link_count": int((fallback_array > 0).sum()),
                "max_bin_distance": int(fallback_array.max()) if fallback_array.size else 0,
                "same_bin_fraction": float((fallback_array == 0).mean()) if fallback_array.size else 1.0,
                "promoter_peak_detection_frequency_median": float(np.median(original_detection_array)) if original_detection_array.size else 0.0,
                "matched_peak_detection_frequency_median": float(np.median(matched_detection_array)) if matched_detection_array.size else 0.0,
                "detection_frequency_mean_abs_difference": float(np.mean(np.abs(matched_detection_array - original_detection_array))) if matched_detection_array.size else 0.0,
                "detection_frequency_median_abs_difference": float(np.median(np.abs(matched_detection_array - original_detection_array))) if matched_detection_array.size else 0.0,
                "promoter_peak_total_counts_median": float(np.median(original_total_array)) if original_total_array.size else 0.0,
                "matched_peak_total_counts_median": float(np.median(matched_total_array)) if matched_total_array.size else 0.0,
                "log1p_total_counts_mean_abs_difference": float(np.mean(np.abs(np.log1p(matched_total_array) - np.log1p(original_total_array)))) if matched_total_array.size else 0.0,
                "matched_unique_peak_count": int((peak_counts > 0).sum()),
                "matched_multi_assigned_peak_count": int((peak_counts > 1).sum()),
                "status": "pass" if count_preserved and zero_preserved else "fail",
            }
        ]
    )
    return AccessibilityMatchedResult(
        mapping=PeakGeneMapping(mapping, genes, peak_counts, gene_counts, matched_links),
        matching_qc=matching_qc,
        peak_metrics=metrics,
    )
