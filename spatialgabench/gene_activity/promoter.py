"""Promoter-only gene activity wrappers built on interval aggregation."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import sparse

from .interval_aggregation import (
    PeakGeneMapping,
    build_mapping_from_intervals,
    build_peak_feature_overlap,
    build_peak_gene_matrix,
    compute_gene_activity,
    deduplicate_peak_gene_links,
    load_bed_intervals,
    load_gene_metadata,
    parse_peak_coordinates,
    write_gene_activity_h5ad,
)


def load_promoters_bed(path: str | Path) -> pd.DataFrame:
    return load_bed_intervals(path, "promoter")


def load_genes_bed(path: str | Path) -> pd.DataFrame:
    return load_gene_metadata(path)


def build_peak_gene_overlap(peaks: pd.DataFrame, promoters: pd.DataFrame) -> PeakGeneMapping:
    gene_metadata = promoters[["gene_id", "gene_name", "gene_type", "chr", "start", "end", "strand"]].copy()
    return build_mapping_from_intervals(peaks, promoters, gene_metadata, deduplicate=True)


__all__ = [
    "ad",
    "sparse",
    "parse_peak_coordinates",
    "load_promoters_bed",
    "load_genes_bed",
    "build_peak_feature_overlap",
    "deduplicate_peak_gene_links",
    "build_peak_gene_matrix",
    "build_peak_gene_overlap",
    "compute_gene_activity",
    "write_gene_activity_h5ad",
]
