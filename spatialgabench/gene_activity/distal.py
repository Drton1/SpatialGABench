"""Distance-weighted distal gene activity utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .interval_aggregation import (
    BED_COLUMNS,
    GENE_METADATA_COLUMNS,
    build_mapping_from_intervals,
    compute_gene_activity,
)


LINK_COLUMNS = [
    "peak_index",
    "gene_index",
    "peak_id",
    "gene_id",
    "gene_name",
    "chr",
    "peak_start",
    "peak_end",
    "peak_midpoint",
    "tss",
    "distance_to_tss",
    "weight",
    "link_type",
]


@dataclass(frozen=True)
class DistalGeneMapping:
    peak_gene_matrix: sparse.csr_matrix
    genes: pd.DataFrame
    peak_assignment_counts: np.ndarray
    gene_assigned_peak_counts: np.ndarray
    links: pd.DataFrame
    preflight: dict[str, object]


def _check_chr_compatibility(peaks: pd.DataFrame, tss: pd.DataFrame) -> None:
    peak_chr = set(peaks["chr"].astype(str).unique())
    tss_chr = set(tss["chr"].astype(str).unique())
    if not peak_chr.intersection(tss_chr):
        raise ValueError(
            "Peak and TSS chromosome naming is incompatible: "
            f"peak examples={sorted(peak_chr)[:5]}, TSS examples={sorted(tss_chr)[:5]}"
        )


def load_tss_table(path: str | Path, gene_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    tss = pd.read_csv(path, sep="\t", header=None, names=BED_COLUMNS)
    for column in ["chr", "gene_id", "gene_name", "gene_type", "strand"]:
        tss[column] = tss[column].astype(str)
    tss["start"] = pd.to_numeric(tss["start"], errors="raise").astype(np.int64)
    tss["end"] = pd.to_numeric(tss["end"], errors="raise").astype(np.int64)
    invalid = (tss["start"] < 0) | (tss["end"] <= tss["start"])
    if invalid.any():
        examples = tss.loc[invalid, ["gene_id", "chr", "start", "end"]].head().to_dict("records")
        raise ValueError(f"Invalid TSS coordinates detected: {examples}")
    if tss["gene_id"].duplicated().any():
        examples = tss.loc[tss["gene_id"].duplicated(), "gene_id"].head().tolist()
        raise ValueError(f"Duplicated gene_id values in TSS BED: {examples}")
    tss["tss"] = tss["start"].astype(np.int64)
    if gene_metadata is None:
        tss["gene_index"] = np.arange(tss.shape[0], dtype=np.int64)
    else:
        gene_order = {gene_id: idx for idx, gene_id in enumerate(gene_metadata["gene_id"].astype(str).tolist())}
        tss["gene_index"] = tss["gene_id"].map(gene_order)
        if tss["gene_index"].isna().any():
            missing = tss.loc[tss["gene_index"].isna(), "gene_id"].drop_duplicates().head().tolist()
            raise ValueError(f"TSS genes are missing from gene metadata: {missing}")
        tss["gene_index"] = tss["gene_index"].astype(np.int64)
    return tss[["gene_index", "gene_id", "gene_name", "gene_type", "chr", "start", "end", "strand", "tss"]].copy()


def _with_peak_midpoint(peaks: pd.DataFrame) -> pd.DataFrame:
    peaks = peaks.copy()
    peaks["peak_midpoint"] = (peaks["start"].astype(np.float64) + peaks["end"].astype(np.float64)) / 2.0
    return peaks


def compute_distance_weights(distance_to_tss: np.ndarray, distance_decay_bp: int) -> np.ndarray:
    if distance_decay_bp <= 0:
        raise ValueError("distance_decay_bp must be positive.")
    weights = np.exp(-distance_to_tss.astype(np.float64) / float(distance_decay_bp))
    if (weights < 0).any() or (weights > 1).any() or np.isnan(weights).any():
        raise ValueError("Computed distal weights must be finite and in [0, 1].")
    return weights.astype(np.float32)


def _iter_distal_link_arrays(
    peaks: pd.DataFrame,
    tss: pd.DataFrame,
    *,
    distal_window_bp: int,
    promoter_exclusion_window_bp: int,
):
    peaks = _with_peak_midpoint(peaks)
    _check_chr_compatibility(peaks, tss)
    for chrom, chrom_tss in tss.groupby("chr", sort=False):
        chrom_peaks = peaks[peaks["chr"] == chrom].sort_values("peak_midpoint").reset_index(drop=True)
        if chrom_peaks.empty:
            continue
        midpoints = chrom_peaks["peak_midpoint"].to_numpy(np.float64)
        peak_starts = chrom_peaks["start"].to_numpy(np.int64)
        peak_ends = chrom_peaks["end"].to_numpy(np.int64)
        for gene in chrom_tss.itertuples(index=False):
            tss_pos = int(gene.tss)
            left = np.searchsorted(midpoints, tss_pos - distal_window_bp, side="left")
            right = np.searchsorted(midpoints, tss_pos + distal_window_bp, side="right")
            if right <= left:
                continue
            candidate = np.arange(left, right, dtype=np.int64)
            promoter_start = max(0, tss_pos - promoter_exclusion_window_bp)
            promoter_end = tss_pos + promoter_exclusion_window_bp
            outside_promoter = (peak_ends[candidate] <= promoter_start) | (peak_starts[candidate] >= promoter_end)
            hit = candidate[outside_promoter]
            if hit.size == 0:
                continue
            distances = np.abs(midpoints[hit] - float(tss_pos))
            yield chrom, gene, chrom_peaks.iloc[hit], distances


def estimate_distal_link_count(
    peaks: pd.DataFrame,
    tss: pd.DataFrame,
    *,
    distal_window_bp: int,
    promoter_exclusion_window_bp: int,
) -> dict[str, object]:
    n_links = 0
    per_chrom: dict[str, int] = {}
    for chrom, _gene, hit_peaks, _distances in _iter_distal_link_arrays(
        peaks,
        tss,
        distal_window_bp=distal_window_bp,
        promoter_exclusion_window_bp=promoter_exclusion_window_bp,
    ):
        count = int(hit_peaks.shape[0])
        n_links += count
        per_chrom[chrom] = per_chrom.get(chrom, 0) + count
    estimated_matrix_memory_mb = float(n_links * (4 + 4 + 4) / (1024 * 1024))
    estimated_links_table_memory_mb = float(n_links * 120 / (1024 * 1024))
    return {
        "estimated_distal_links": int(n_links),
        "estimated_mapping_memory_mb": estimated_matrix_memory_mb,
        "estimated_links_table_memory_mb": estimated_links_table_memory_mb,
        "estimated_links_by_chromosome": per_chrom,
    }


def build_distal_peak_gene_links(
    peaks: pd.DataFrame,
    tss: pd.DataFrame,
    *,
    distal_window_bp: int,
    promoter_exclusion_window_bp: int,
    distance_decay_bp: int,
) -> pd.DataFrame:
    records = {column: [] for column in LINK_COLUMNS}
    for chrom, gene, hit_peaks, distances in _iter_distal_link_arrays(
        peaks,
        tss,
        distal_window_bp=distal_window_bp,
        promoter_exclusion_window_bp=promoter_exclusion_window_bp,
    ):
        weights = compute_distance_weights(distances, distance_decay_bp)
        n_hits = hit_peaks.shape[0]
        records["peak_index"].extend(hit_peaks["peak_index"].astype(np.int64).tolist())
        records["gene_index"].extend([int(gene.gene_index)] * n_hits)
        records["peak_id"].extend(hit_peaks["peak_id"].astype(str).tolist())
        records["gene_id"].extend([str(gene.gene_id)] * n_hits)
        records["gene_name"].extend([str(gene.gene_name)] * n_hits)
        records["chr"].extend([str(chrom)] * n_hits)
        records["peak_start"].extend(hit_peaks["start"].astype(np.int64).tolist())
        records["peak_end"].extend(hit_peaks["end"].astype(np.int64).tolist())
        records["peak_midpoint"].extend(hit_peaks["peak_midpoint"].astype(np.float64).tolist())
        records["tss"].extend([int(gene.tss)] * n_hits)
        records["distance_to_tss"].extend(distances.astype(np.float64).tolist())
        records["weight"].extend(weights.astype(np.float32).tolist())
        records["link_type"].extend(["distal"] * n_hits)
    links = pd.DataFrame(records, columns=LINK_COLUMNS)
    if links.empty:
        return links
    return links.sort_values(["peak_index", "gene_index", "link_type"]).reset_index(drop=True)


def build_promoter_peak_gene_links(
    peaks: pd.DataFrame,
    promoters: pd.DataFrame,
    tss: pd.DataFrame,
    gene_metadata: pd.DataFrame,
) -> pd.DataFrame:
    promoter_mapping = build_mapping_from_intervals(peaks, promoters, gene_metadata, deduplicate=True)
    links = promoter_mapping.links.copy()
    if links.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)
    tss_lookup = tss[["gene_id", "tss"]].copy()
    links = links.merge(tss_lookup, on="gene_id", how="left", validate="many_to_one")
    if links["tss"].isna().any():
        missing = links.loc[links["tss"].isna(), "gene_id"].drop_duplicates().head().tolist()
        raise ValueError(f"Promoter links missing TSS metadata: {missing}")
    links["chr"] = links["peak_chr"].astype(str)
    links["peak_midpoint"] = (links["peak_start"].astype(np.float64) + links["peak_end"].astype(np.float64)) / 2.0
    links["tss"] = links["tss"].astype(np.int64)
    links["distance_to_tss"] = np.abs(links["peak_midpoint"].to_numpy(np.float64) - links["tss"].to_numpy(np.float64))
    links["weight"] = 1.0
    links["link_type"] = "promoter"
    return links[LINK_COLUMNS].sort_values(["peak_index", "gene_index", "link_type"]).reset_index(drop=True)


def exclude_promoter_overlaps(links: pd.DataFrame, promoter_exclusion_window_bp: int) -> pd.DataFrame:
    if links.empty:
        return links.copy()
    promoter_start = np.maximum(0, links["tss"].to_numpy(np.int64) - int(promoter_exclusion_window_bp))
    promoter_end = links["tss"].to_numpy(np.int64) + int(promoter_exclusion_window_bp)
    outside = (links["peak_end"].to_numpy(np.int64) <= promoter_start) | (links["peak_start"].to_numpy(np.int64) >= promoter_end)
    return links.loc[outside].reset_index(drop=True)


def build_weighted_peak_gene_matrix(links: pd.DataFrame, n_peaks: int, n_genes: int) -> sparse.csr_matrix:
    if links.empty:
        return sparse.csr_matrix((n_peaks, n_genes), dtype=np.float32)
    weights = links["weight"].to_numpy(np.float32)
    if np.isnan(weights).any() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("Link weights must be finite and in [0, 1].")
    matrix = sparse.csr_matrix(
        (weights, (links["peak_index"].to_numpy(np.int64), links["gene_index"].to_numpy(np.int64))),
        shape=(n_peaks, n_genes),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    if matrix.data.size and ((matrix.data < 0).any() or np.isnan(matrix.data).any()):
        raise ValueError("Weighted peak-gene matrix contains invalid values.")
    return matrix


def _build_genes_with_counts(gene_metadata: pd.DataFrame, links: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n_genes = gene_metadata.shape[0]
    if links.empty:
        peak_counts = np.zeros(0, dtype=np.int64)
        gene_counts = np.zeros(n_genes, dtype=np.int64)
        distal_counts = np.zeros(n_genes, dtype=np.int64)
        promoter_counts = np.zeros(n_genes, dtype=np.int64)
    else:
        n_peaks = int(links["peak_index"].max()) + 1
        peak_counts = np.bincount(links["peak_index"].to_numpy(np.int64), minlength=n_peaks).astype(np.int64)
        gene_counts = np.bincount(links["gene_index"].to_numpy(np.int64), minlength=n_genes).astype(np.int64)
        distal_links = links[links["link_type"] == "distal"]
        promoter_links = links[links["link_type"] == "promoter"]
        distal_counts = np.bincount(distal_links["gene_index"].to_numpy(np.int64), minlength=n_genes).astype(np.int64) if not distal_links.empty else np.zeros(n_genes, dtype=np.int64)
        promoter_counts = np.bincount(promoter_links["gene_index"].to_numpy(np.int64), minlength=n_genes).astype(np.int64) if not promoter_links.empty else np.zeros(n_genes, dtype=np.int64)
    genes = gene_metadata.copy().reset_index(drop=True)
    genes["n_assigned_peaks"] = gene_counts
    genes["n_distal_peaks"] = distal_counts
    genes["n_promoter_peaks"] = promoter_counts
    return genes, peak_counts, gene_counts


def build_distal_peak_gene_mapping(
    peaks: pd.DataFrame,
    tss: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    *,
    distal_window_bp: int,
    promoter_exclusion_window_bp: int,
    distance_decay_bp: int,
    preflight: dict[str, object] | None = None,
) -> DistalGeneMapping:
    links = build_distal_peak_gene_links(
        peaks,
        tss,
        distal_window_bp=distal_window_bp,
        promoter_exclusion_window_bp=promoter_exclusion_window_bp,
        distance_decay_bp=distance_decay_bp,
    )
    matrix = build_weighted_peak_gene_matrix(links, peaks.shape[0], gene_metadata.shape[0])
    genes, peak_counts_partial, gene_counts = _build_genes_with_counts(gene_metadata, links)
    peak_counts = np.zeros(peaks.shape[0], dtype=np.int64)
    peak_counts[: peak_counts_partial.shape[0]] = peak_counts_partial
    return DistalGeneMapping(matrix, genes, peak_counts, gene_counts, links, preflight or {})


def combine_promoter_and_distal_links(promoter_links: pd.DataFrame, distal_links: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([promoter_links, distal_links], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)
    combined["_rank"] = combined["link_type"].map({"promoter": 0, "distal": 1}).fillna(2).astype(int)
    combined = (
        combined.sort_values(["peak_index", "gene_index", "_rank"])
        .drop_duplicates(subset=["peak_index", "gene_index"], keep="first")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return combined[LINK_COLUMNS]


def build_promoter_plus_distal_mapping(
    peaks: pd.DataFrame,
    promoters: pd.DataFrame,
    tss: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    *,
    distal_window_bp: int,
    promoter_exclusion_window_bp: int,
    distance_decay_bp: int,
    preflight: dict[str, object] | None = None,
) -> DistalGeneMapping:
    promoter_links = build_promoter_peak_gene_links(peaks, promoters, tss, gene_metadata)
    distal_links = build_distal_peak_gene_links(
        peaks,
        tss,
        distal_window_bp=distal_window_bp,
        promoter_exclusion_window_bp=promoter_exclusion_window_bp,
        distance_decay_bp=distance_decay_bp,
    )
    links = combine_promoter_and_distal_links(promoter_links, distal_links)
    matrix = build_weighted_peak_gene_matrix(links, peaks.shape[0], gene_metadata.shape[0])
    genes, peak_counts_partial, gene_counts = _build_genes_with_counts(gene_metadata, links)
    peak_counts = np.zeros(peaks.shape[0], dtype=np.int64)
    peak_counts[: peak_counts_partial.shape[0]] = peak_counts_partial
    return DistalGeneMapping(matrix, genes, peak_counts, gene_counts, links, preflight or {})


def compute_weighted_gene_activity(X_peak, peak_gene_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    return compute_gene_activity(X_peak, peak_gene_matrix)


def summarize_distal_mapping(mapping: DistalGeneMapping, n_peaks: int) -> dict[str, object]:
    links = mapping.links
    gene_counts = mapping.gene_assigned_peak_counts
    peak_counts = mapping.peak_assignment_counts
    distal_links = links[links["link_type"] == "distal"] if not links.empty else links
    promoter_links = links[links["link_type"] == "promoter"] if not links.empty else links
    distal_gene_counts = mapping.genes["n_distal_peaks"].to_numpy(np.int64)
    return {
        "mapping_matrix_nnz": int(mapping.peak_gene_matrix.nnz),
        "n_genes_with_ge1_assigned_peak": int((gene_counts >= 1).sum()),
        "n_genes_with_0_assigned_peaks": int((gene_counts == 0).sum()),
        "n_assigned_peaks": int((peak_counts >= 1).sum()),
        "n_unassigned_peaks": int(n_peaks - (peak_counts >= 1).sum()),
        "n_multi_assigned_peaks": int((peak_counts > 1).sum()),
        "n_distal_links": int(distal_links.shape[0]),
        "n_promoter_links": int(promoter_links.shape[0]),
        "assigned_peaks_per_gene_min": int(gene_counts.min()) if gene_counts.size else 0,
        "assigned_peaks_per_gene_median": float(np.median(gene_counts)) if gene_counts.size else 0.0,
        "assigned_peaks_per_gene_max": int(gene_counts.max()) if gene_counts.size else 0,
        "distal_peaks_per_gene_median": float(np.median(distal_gene_counts)) if distal_gene_counts.size else 0.0,
        "median_link_distance": float(np.median(links["distance_to_tss"])) if not links.empty else 0.0,
        "median_distal_link_distance": float(np.median(distal_links["distance_to_tss"])) if not distal_links.empty else 0.0,
        "median_link_weight": float(np.median(links["weight"])) if not links.empty else 0.0,
        "median_distal_link_weight": float(np.median(distal_links["weight"])) if not distal_links.empty else 0.0,
        "min_link_weight": float(links["weight"].min()) if not links.empty else 0.0,
        "max_link_weight": float(links["weight"].max()) if not links.empty else 0.0,
    }


def write_distal_gene_activity_h5ad(
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
    ordered_columns = GENE_METADATA_COLUMNS + ["n_assigned_peaks"]
    optional_columns = [column for column in ["n_distal_peaks", "n_promoter_peaks"] if column in genes.columns]
    var = genes[ordered_columns + optional_columns].copy()
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
