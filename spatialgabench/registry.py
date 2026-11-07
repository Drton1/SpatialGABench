"""Registry discovery helpers used by the public CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ACTIVE_METHOD_REGISTRY = Path("registries/v1/methods.yaml")
ACTIVE_TASK_REGISTRY = Path("registries/v1/tasks.yaml")

FALLBACK_METHODS = [
    {"method_id": "promoter_only_2kb_pc", "category": "atac_annotation_only"},
    {"method_id": "gene_body_pc", "category": "atac_annotation_only"},
    {"method_id": "raw_peak_lsi_baseline", "category": "controls"},
]
FALLBACK_TASKS = [
    {"task_id": "construction_qc", "category": "core"},
    {"task_id": "paired_rna_fidelity", "category": "core"},
    {"task_id": "simulation_truth_recovery", "category": "core"},
]


def find_repo_root(start: str | Path | None = None) -> Path | None:
    """Find the active repository root from the v1 registries."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ACTIVE_METHOD_REGISTRY).exists() and (candidate / ACTIVE_TASK_REGISTRY).exists():
            return candidate
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def list_methods(repo_root: str | Path | None = None) -> list[dict[str, str]]:
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    if root is None or not (root / ACTIVE_METHOD_REGISTRY).exists():
        return FALLBACK_METHODS
    data = _load_yaml(root / ACTIVE_METHOD_REGISTRY)
    rows: list[dict[str, str]] = []
    for track_id, track in data.get("tracks", {}).items():
        if not isinstance(track, dict):
            continue
        for method_id, metadata in track.get("methods", {}).items():
            metadata = metadata if isinstance(metadata, dict) else {}
            rows.append(
                {
                    "method_id": str(method_id),
                    "category": str(track_id),
                    "status": str(metadata.get("status", "")),
                }
            )
    return rows or FALLBACK_METHODS


def list_tasks(repo_root: str | Path | None = None) -> list[dict[str, str]]:
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    if root is None or not (root / ACTIVE_TASK_REGISTRY).exists():
        return FALLBACK_TASKS
    data = _load_yaml(root / ACTIVE_TASK_REGISTRY)
    rows: list[dict[str, str]] = []
    for task_id, metadata in data.get("tasks", {}).items():
        metadata = metadata if isinstance(metadata, dict) else {}
        rows.append(
            {
                "task_id": str(task_id),
                "category": str(metadata.get("tier", "")),
                "status": str(metadata.get("status", "active")),
            }
        )
    return rows or FALLBACK_TASKS
