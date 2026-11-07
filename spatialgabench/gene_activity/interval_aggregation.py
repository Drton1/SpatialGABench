"""Generic interval aggregation utilities for gene activity baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


REQUIRED_PEAK_COLUMNS = ["chr", "start", "end"]
BED_COLUMNS = ["chr", "start", "end", "gene_id", "score", "strand", "gene_name", "gene_type"]
GENE_METADATA_COLUMNS = ["gene_id", "gene_name", "gene_type", "chr", "start", "end", "strand"]


@dataclass(frozen=True)
class PeakGeneMapping:
    peak_gene_matrix: sparse.csr_matrix
    genes: pd.DataFrame
    peak_assignment_counts: np.ndarray
    gene_assigned_peak_counts: np.ndarray
    links: pd.DataFrame

    @property
    def gene_promoter_peak_counts(self) -> np.ndarray:
        """Backward-compatible alias used by the original promoter-only tests."""
        return self.gene_assigned_peak_counts


def parse_peak_coordinates(adata: ad.AnnData) -> pd.DataFrame:
    chromosome_column = "chr" if "chr" in adata.var.columns else "chromosome"
    missing = [
        column
        for column in REQUIRED_PEAK_COLUMNS
        if column not in adata.var.columns and not (column == "chr" and chromosome_column == "chromosome")
    ]
    if missing:
        raise ValueError(f"Peak coordinate columns are missing from adata.var: {missing}")
    peaks = adata.var[[chromosome_column, "start", "end"]].copy()
    peaks.columns = REQUIRED_PEAK_COLUMNS
    peak_ids = adata.var["peak_id"].astype(str).to_numpy() if "peak_id" in adata.var.columns else adata.var_names.astype(str)
    peaks.insert(0, "peak_id", peak_ids)
    peaks["peak_index"] = np.arange(peaks.shape[0], dtype=np.int64)
    peaks["chr"] = peaks["chr"].astype(str)
    peaks["start"] = pd.to_numeric(peaks["start"], errors="raise").astype(np.int64)
    peaks["end"] = pd.to_numeric(peaks["end"], errors="raise").astype(np.int64)
    invalid = (peaks["start"] < 0) | (peaks["end"] <= peaks["start"])
    if invalid.any():
        examples = peaks.loc[invalid, ["peak_id", "chr", "start", "end"]].head().to_dict("records")
        raise ValueError(f"Invalid peak coordinates detected: {examples}")
    return peaks[["peak_index", "peak_id", "chr", "start", "end"]]


def load_bed_intervals(path: str | Path, interval_type: str) -> pd.DataFrame:
    intervals = pd.read_csv(path, sep="\t", header=None, names=BED_COLUMNS)
    intervals["gene_index"] = np.arange(intervals.shape[0], dtype=np.int64)
    intervals["interval_type"] = interval_type
    for column in ["chr", "gene_id", "gene_name", "gene_type", "strand"]:
        intervals[column] = intervals[column].astype(str)
    intervals["start"] = pd.to_numeric(intervals["start"], errors="raise").astype(np.int64)
    intervals["end"] = pd.to_numeric(intervals["end"], errors="raise").astype(np.int64)
    invalid = (intervals["start"] < 0) | (intervals["end"] <= intervals["start"])
    if invalid.any():
        examples = intervals.loc[invalid, ["gene_id", "chr", "start", "end", "interval_type"]].head().to_dict("records")
        raise ValueError(f"Invalid {interval_type} coordinates detected: {examples}")
    if intervals["gene_id"].duplicated().any():
        examples = intervals.loc[intervals["gene_id"].duplicated(), "gene_id"].head().tolist()
        raise ValueError(f"Duplicated gene_id values in {interval_type} BED: {examples}")
    return intervals[["gene_index", "gene_id", "gene_name", "gene_type", "chr", "start", "end", "strand", "interval_type"]]


def load_gene_metadata(path: str | Path) -> pd.DataFrame:
    genes = pd.read_csv(path, sep="\t", header=None, names=BED_COLUMNS)
    for column in ["chr", "gene_id", "gene_name", "gene_type", "strand"]:
        genes[column] = genes[column].astype(str)
    genes["start"] = pd.to_numeric(genes["start"], errors="raise").astype(np.int64)
    genes["end"] = pd.to_numeric(genes["end"], errors="raise").astype(np.int64)
    invalid = (genes["start"] < 0) | (genes["end"] <= genes["start"])
    if invalid.any():
        examples = genes.loc[invalid, ["gene_id", "chr", "start", "end"]].head().to_dict("records")
        raise ValueError(f"Invalid gene coordinates detected: {examples}")
    if genes["gene_id"].duplicated().any():
        examples = genes.loc[genes["gene_id"].duplicated(), "gene_id"].head().tolist()
        raise ValueError(f"Duplicated gene_id values in gene BED: {examples}")
    return genes[GENE_METADATA_COLUMNS].copy()


def _check_chr_compatibility(peaks: pd.DataFrame, intervals: pd.DataFrame) -> None:
    peak_chr = set(peaks["chr"].unique())
    interval_chr = set(intervals["chr"].unique())
    if not peak_chr.intersection(interval_chr):
        raise ValueError(
            "Peak and interval chromosome naming is incompatible: "
            f"peak examples={sorted(peak_chr)[:5]}, interval examples={sorted(interval_chr)[:5]}"
        )


def build_peak_feature_overlap(peaks: pd.DataFrame, intervals: pd.DataFrame, feature_id_col: str = "gene_id") -> pd.DataFrame:
    """Return peak-feature links using half-open interval overlap."""

    required_peak = {"peak_index", "peak_id", "chr", "start", "end"}
    required_interval = {"gene_index", feature_id_col, "gene_name", "gene_type", "chr", "start", "end", "strand", "interval_type"}
    if missing := sorted(required_peak.difference(peaks.columns)):
        raise ValueError(f"Peaks table missing required columns: {missing}")
    if missing := sorted(required_interval.difference(intervals.columns)):
        raise ValueError(f"Intervals table missing required columns: {missing}")
    _check_chr_compatibility(peaks, intervals)

    records: list[dict[str, object]] = []
    for chrom, chrom_peaks in peaks.groupby("chr", sort=False):
        chrom_intervals = intervals[intervals["chr"] == chrom].sort_values("start")
        if chrom_intervals.empty:
            continue
        starts = chrom_intervals["start"].to_numpy(np.int64)
        ends = chrom_intervals["end"].to_numpy(np.int64)
        gene_indices = chrom_intervals["gene_index"].to_numpy(np.int64)
        gene_ids = chrom_intervals[feature_id_col].to_numpy(str)
        gene_names = chrom_intervals["gene_name"].to_numpy(str)
        interval_types = chrom_intervals["interval_type"].to_numpy(str)
        for peak in chrom_peaks.itertuples(index=False):
            right = np.searchsorted(starts, int(peak.end), side="left")
            if right == 0:
                continue
            candidate = np.arange(right, dtype=np.int64)
            hit_positions = candidate[ends[candidate] > int(peak.start)]
            for pos in hit_positions:
                records.append(
                    {
                        "peak_index": int(peak.peak_index),
                        "peak_id": str(peak.peak_id),
                        "peak_chr": chrom,
                        "peak_start": int(peak.start),
                        "peak_end": int(peak.end),
                        "gene_index": int(gene_indices[pos]),
                        "gene_id": str(gene_ids[pos]),
                        "gene_name": str(gene_names[pos]),
                        "interval_type": str(interval_types[pos]),
                        "interval_start": int(starts[pos]),
                        "interval_end": int(ends[pos]),
                    }
                )
    if not records:
        return pd.DataFrame(
            columns=[
                "peak_index",
                "peak_id",
                "peak_chr",
                "peak_start",
                "peak_end",
                "gene_index",
                "gene_id",
                "gene_name",
                "interval_type",
                "interval_start",
                "interval_end",
            ]
        )
    return pd.DataFrame(records).sort_values(["peak_index", "gene_index", "interval_type"]).reset_index(drop=True)


def deduplicate_peak_gene_links(links: pd.DataFrame) -> pd.DataFrame:
    if links.empty:
        return links.copy()
    return (
        links.sort_values(["peak_index", "gene_index", "interval_type"])
        .drop_duplicates(subset=["peak_index", "gene_index"], keep="first")
        .reset_index(drop=True)
    )


def build_peak_gene_matrix(links: pd.DataFrame, n_peaks: int, n_genes: int) -> sparse.csr_matrix:
    if links.empty:
        return sparse.csr_matrix((n_peaks, n_genes), dtype=np.float32)
    data = np.ones(links.shape[0], dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, (links["peak_index"].to_numpy(np.int64), links["gene_index"].to_numpy(np.int64))),
        shape=(n_peaks, n_genes),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix


def summarize_peak_gene_mapping(mapping: sparse.csr_matrix) -> dict[str, object]:
    mapping = mapping.tocsr()
    peak_counts = np.asarray(mapping.sum(axis=1)).ravel()
    gene_counts = np.asarray(mapping.sum(axis=0)).ravel()
    return {
        "mapping_matrix_nnz": int(mapping.nnz),
        "n_genes_with_ge1_assigned_peak": int((gene_counts >= 1).sum()),
        "n_genes_with_0_assigned_peaks": int((gene_counts == 0).sum()),
        "n_assigned_peaks": int((peak_counts >= 1).sum()),
        "n_unassigned_peaks": int((peak_counts == 0).sum()),
        "n_multi_assigned_peaks": int((peak_counts > 1).sum()),
        "assigned_peaks_per_gene_min": int(gene_counts.min()) if gene_counts.size else 0,
        "assigned_peaks_per_gene_median": float(np.median(gene_counts)) if gene_counts.size else 0.0,
        "assigned_peaks_per_gene_max": int(gene_counts.max()) if gene_counts.size else 0,
    }


def build_mapping_from_intervals(peaks: pd.DataFrame, intervals: pd.DataFrame, gene_metadata: pd.DataFrame, *, deduplicate: bool = True) -> PeakGeneMapping:
    gene_order = {gene_id: idx for idx, gene_id in enumerate(gene_metadata["gene_id"].astype(str).tolist())}
    intervals = intervals.copy()
    intervals["gene_index"] = intervals["gene_id"].astype(str).map(gene_order)
    if intervals["gene_index"].isna().any():
        missing = intervals.loc[intervals["gene_index"].isna(), "gene_id"].drop_duplicates().head().tolist()
        raise ValueError(f"Interval genes are missing from gene metadata: {missing}")
    intervals["gene_index"] = intervals["gene_index"].astype(np.int64)
    links = build_peak_feature_overlap(peaks, intervals)
    if deduplicate:
        links = deduplicate_peak_gene_links(links)
    mapping = build_peak_gene_matrix(links, n_peaks=peaks.shape[0], n_genes=gene_metadata.shape[0])
    peak_counts = np.asarray(mapping.sum(axis=1)).ravel()
    gene_counts = np.asarray(mapping.sum(axis=0)).ravel()
    genes = gene_metadata.copy().reset_index(drop=True)
    genes["n_assigned_peaks"] = gene_counts.astype(np.int64)
    return PeakGeneMapping(mapping, genes, peak_counts, gene_counts, links)


def build_random_peak_gene_mapping(
    *,
    n_peaks: int,
    gene_metadata: pd.DataFrame,
    per_gene_peak_counts: np.ndarray,
    seed: int,
    peaks: pd.DataFrame | None = None,
) -> PeakGeneMapping:
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    cols: list[int] = []
    records: list[dict[str, object]] = []
    for gene_index, n_assign in enumerate(per_gene_peak_counts.astype(int)):
        if n_assign <= 0:
            continue
        if n_assign > n_peaks:
            raise ValueError(f"Cannot sample {n_assign} peaks without replacement from {n_peaks} peaks")
        sampled = np.sort(rng.choice(n_peaks, size=n_assign, replace=False))
        gene = gene_metadata.iloc[gene_index]
        rows.extend(sampled.tolist())
        cols.extend([gene_index] * n_assign)
        for peak_index in sampled:
            peak_record = peaks.iloc[int(peak_index)] if peaks is not None else None
            records.append(
                {
                    "peak_index": int(peak_index),
                    "peak_id": str(peak_record["peak_id"]) if peak_record is not None else "",
                    "peak_chr": str(peak_record["chr"]) if peak_record is not None else "",
                    "peak_start": int(peak_record["start"]) if peak_record is not None else "",
                    "peak_end": int(peak_record["end"]) if peak_record is not None else "",
                    "gene_index": int(gene_index),
                    "gene_id": str(gene["gene_id"]),
                    "gene_name": str(gene["gene_name"]),
                    "interval_type": "random_peak",
                    "interval_start": "",
                    "interval_end": "",
                }
            )
    links = pd.DataFrame(records)
    mapping = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_peaks, gene_metadata.shape[0]), dtype=np.float32)
    mapping.sum_duplicates()
    peak_counts = np.asarray(mapping.sum(axis=1)).ravel()
    gene_counts = np.asarray(mapping.sum(axis=0)).ravel()
    genes = gene_metadata.copy().reset_index(drop=True)
    genes["n_assigned_peaks"] = gene_counts.astype(np.int64)
    return PeakGeneMapping(mapping, genes, peak_counts, gene_counts, links)


def compute_gene_activity(X_peak, peak_gene_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    if not sparse.issparse(X_peak):
        raise ValueError("Input peak matrix must be sparse; refusing to densify.")
    if not sparse.issparse(peak_gene_matrix):
        raise ValueError("Peak-gene matrix must be sparse.")
    X_peak = X_peak.tocsr()
    mapping = peak_gene_matrix.tocsr()
    if X_peak.shape[1] != mapping.shape[0]:
        raise ValueError(f"Shape mismatch: X_peak has {X_peak.shape[1]} peaks but mapping has {mapping.shape[0]} rows")
    gene_activity = (X_peak @ mapping).tocsr()
    if np.isnan(gene_activity.data).any():
        raise ValueError("Gene activity matrix contains NaN values.")
    if gene_activity.data.size and gene_activity.data.min() < 0:
        raise ValueError("Gene activity matrix contains negative values.")
    return gene_activity


def write_gene_activity_h5ad(
    *,
    input_adata: ad.AnnData,
    gene_activity_X: sparse.csr_matrix,
    genes: pd.DataFrame,
    output_path: str | Path,
    dataset_id: str,
    method_id: str,
    genome_build: str,
    annotation_source: str,
    input_peak_h5ad: str,
    method_notes: str,
    extra_uns: dict[str, object] | None = None,
) -> ad.AnnData:
    var = genes[GENE_METADATA_COLUMNS + ["n_assigned_peaks"]].copy()
    var.index = var["gene_id"].astype(str)
    output = ad.AnnData(X=gene_activity_X.tocsr(), obs=input_adata.obs.copy(), var=var)
    output.obs_names = input_adata.obs_names.copy()
    if "spatial" in input_adata.obsm:
        output.obsm["spatial"] = input_adata.obsm["spatial"].copy()
    output.uns["dataset_id"] = dataset_id
    output.uns["source_modality"] = "spatial_atac"
    output.uns["gene_activity_method"] = method_id
    output.uns["genome_build"] = genome_build
    output.uns["annotation_source"] = annotation_source
    output.uns["input_peak_h5ad"] = input_peak_h5ad
    output.uns["method_notes"] = method_notes
    for key, value in (extra_uns or {}).items():
        output.uns[key] = value
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(output_path, compression="gzip")
    return output
