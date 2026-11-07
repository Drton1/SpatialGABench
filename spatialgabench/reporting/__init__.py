"""Reporting helpers for reusable SpatialGABench resources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_table(df: pd.DataFrame, path: str | Path, *, title: str) -> Path:
    """Write a small Markdown report containing a table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        table = "_No rows._"
    else:
        rows = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        for row in df.astype(str).itertuples(index=False):
            rows.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
        table = "\n".join(rows)
    path.write_text(f"# {title}\n\n{table}\n", encoding="utf-8")
    return path


__all__ = ["write_markdown_table"]
