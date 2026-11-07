"""Writers for recommendation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _stringify(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def recommendation_to_rows(recommendation: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a recommendation payload to a simple section/value table."""

    fields = [
        "recommendation_scope",
        "canonical_frozen_policy",
        "recommended_methods",
        "methods_to_use_with_caution",
        "not_applicable_methods",
        "required_negative_controls",
        "required_baselines",
        "evaluation_tasks_to_run",
        "comparison_scope",
        "required_caveats",
        "forbidden_interpretations",
        "evidence_sources",
        "confidence_level",
        "matched_rules",
    ]
    return [{"section": field, "value": _stringify(recommendation.get(field, ""))} for field in fields]


def recommendation_to_markdown(recommendation: dict[str, Any]) -> str:
    """Render a recommendation payload as a compact Markdown report."""

    lines = [
        "# SpatialGABench recommendation",
        "",
        f"Dataset: `{recommendation.get('dataset_id', 'unspecified_dataset')}`",
        f"Confidence: `{recommendation.get('confidence_level', 'unknown')}`",
        "",
        "This is coarse input-contract guidance, not the canonical frozen task-specific policy and not a final ranking.",
        "",
    ]
    for row in recommendation_to_rows(recommendation):
        lines.extend([f"## {row['section']}", "", row["value"] or "none", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_recommendation_report(recommendation: dict[str, Any], outdir: str | Path) -> dict[str, Path]:
    """Write JSON, TSV, and Markdown recommendation outputs."""

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "recommendation.json"
    tsv_path = out / "recommendation.tsv"
    md_path = out / "recommendation.md"
    json_path.write_text(json.dumps(recommendation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(recommendation_to_rows(recommendation)).to_csv(tsv_path, sep="\t", index=False)
    md_path.write_text(recommendation_to_markdown(recommendation), encoding="utf-8")
    return {"json": json_path, "tsv": tsv_path, "markdown": md_path}
