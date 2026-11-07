"""Utilities for deterministic ranking inside explicit comparison scopes."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def rank_within_scopes(
    frame: pd.DataFrame,
    *,
    score_column: str,
    scope_columns: Sequence[str],
    method_column: str = "method_id",
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank rows without allowing records to cross declared input scopes."""

    required = [*scope_columns, score_column, method_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Ranking input is missing required columns: {missing}")

    ranked = frame.copy()
    for column in scope_columns:
        invalid = ranked[column].isna() | ranked[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"Ranking scope column {column!r} contains missing values")

    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="coerce")
    ranked = ranked.dropna(subset=[score_column]).copy()
    sort_columns = [*scope_columns, score_column, method_column]
    ascending_flags = [True] * len(scope_columns) + [ascending, True]
    ranked = ranked.sort_values(sort_columns, ascending=ascending_flags, kind="mergesort")
    ranked["rank"] = ranked.groupby(list(scope_columns), sort=False).cumcount() + 1
    return ranked
