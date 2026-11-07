"""Random peak-to-gene negative control."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .interval_aggregation import PeakGeneMapping, build_random_peak_gene_mapping


def build_random_control_mapping(
    *,
    n_peaks: int,
    gene_metadata: pd.DataFrame,
    promoter_peak_counts: np.ndarray,
    seed: int,
    peaks: pd.DataFrame | None = None,
) -> PeakGeneMapping:
    return build_random_peak_gene_mapping(
        n_peaks=n_peaks,
        gene_metadata=gene_metadata,
        per_gene_peak_counts=promoter_peak_counts,
        seed=seed,
        peaks=peaks,
    )
