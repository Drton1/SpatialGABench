"""Gene-body gene activity baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .interval_aggregation import PeakGeneMapping, build_mapping_from_intervals, load_bed_intervals


def load_gene_body_bed(path: str | Path) -> pd.DataFrame:
    return load_bed_intervals(path, "gene_body")


def build_gene_body_mapping(peaks: pd.DataFrame, gene_body: pd.DataFrame, gene_metadata: pd.DataFrame) -> PeakGeneMapping:
    return build_mapping_from_intervals(peaks, gene_body, gene_metadata, deduplicate=True)
