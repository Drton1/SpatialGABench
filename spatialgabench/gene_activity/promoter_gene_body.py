"""Promoter plus gene-body gene activity baseline."""

from __future__ import annotations

import pandas as pd

from .interval_aggregation import PeakGeneMapping, build_mapping_from_intervals


def combine_promoter_and_gene_body(promoters: pd.DataFrame, gene_body: pd.DataFrame, gene_metadata: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([promoters, gene_body], ignore_index=True)
    gene_order = {gene_id: idx for idx, gene_id in enumerate(gene_metadata["gene_id"].astype(str).tolist())}
    combined["gene_index"] = combined["gene_id"].astype(str).map(gene_order).astype(int)
    return combined


def build_promoter_gene_body_mapping(peaks: pd.DataFrame, promoters: pd.DataFrame, gene_body: pd.DataFrame, gene_metadata: pd.DataFrame) -> PeakGeneMapping:
    combined = combine_promoter_and_gene_body(promoters, gene_body, gene_metadata)
    return build_mapping_from_intervals(peaks, combined, gene_metadata, deduplicate=True)
