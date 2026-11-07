"""Input schema validation for QC-driven method recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ALLOWED_INPUT_REPRESENTATIONS = {
    "author_peak_matrix",
    "fixed_bin_matrix",
    "fragments",
    "gene_activity_matrix",
    "unknown",
}
ALLOWED_GENOMES = {"mm10", "hg38", "other", "unknown"}
ALLOWED_SPARSE_DEPTH_LEVELS = {"low", "medium", "high", "unknown"}
ALLOWED_DISTAL_LEVELS = {"low", "medium", "high", "unknown"}
ALLOWED_MAIN_TASKS = {
    "construction_qc",
    "clustering_domain",
    "rna_agreement",
    "marker_recovery",
    "regulatory_interpretation",
    "software_portability",
    "simulation_truth_recovery",
    "cross_dataset_benchmark",
    "exploratory_analysis",
}
ALLOWED_RNA_AVAILABILITY = {"none", "adjacent", "paired", "unknown"}
ALLOWED_RUNTIME_CONSTRAINTS = {"low", "medium", "high", "unknown"}

DEFAULT_RECOMMENDATION_INPUT: dict[str, Any] = {
    "dataset_id": "unspecified_dataset",
    "input_representation": "unknown",
    "genome": "unknown",
    "fragments_available": False,
    "has_gene_annotation": False,
    "has_spatial_coordinates": False,
    "data_sparsity": "unknown",
    "sequencing_depth": "unknown",
    "expected_distal_regulation": "unknown",
    "main_task": "exploratory_analysis",
    "paired_or_adjacent_rna_available": "unknown",
    "need_software_standard_workflow": False,
    "runtime_constraint": "unknown",
}


ENUM_FIELDS = {
    "input_representation": ALLOWED_INPUT_REPRESENTATIONS,
    "genome": ALLOWED_GENOMES,
    "data_sparsity": ALLOWED_SPARSE_DEPTH_LEVELS,
    "sequencing_depth": ALLOWED_SPARSE_DEPTH_LEVELS,
    "expected_distal_regulation": ALLOWED_DISTAL_LEVELS,
    "main_task": ALLOWED_MAIN_TASKS,
    "paired_or_adjacent_rna_available": ALLOWED_RNA_AVAILABILITY,
    "runtime_constraint": ALLOWED_RUNTIME_CONSTRAINTS,
}
BOOLEAN_FIELDS = {
    "fragments_available",
    "has_gene_annotation",
    "has_spatial_coordinates",
    "need_software_standard_workflow",
}


def _coerce_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field} must be boolean, got {value!r}")


def load_recommendation_input(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML recommendation config."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Recommendation config does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Recommendation config must parse to a mapping.")
    return validate_recommendation_input(data)


def validate_recommendation_input(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a recommendation input mapping."""

    if not isinstance(config, dict):
        raise ValueError("Recommendation input must be a mapping.")

    normalized = dict(DEFAULT_RECOMMENDATION_INPUT)
    normalized.update(config)

    for field, allowed in ENUM_FIELDS.items():
        value = str(normalized.get(field, "unknown"))
        if value not in allowed:
            raise ValueError(f"{field} must be one of {sorted(allowed)}, got {value!r}")
        normalized[field] = value

    for field in BOOLEAN_FIELDS:
        normalized[field] = _coerce_bool(normalized.get(field, False), field)

    if normalized["input_representation"] == "fragments":
        normalized["fragments_available"] = True
    if normalized["input_representation"] == "gene_activity_matrix":
        normalized.setdefault(
            "input_note",
            "Input is already a gene activity matrix; construction recommendations are limited to evaluation/QC.",
        )

    return normalized
