"""Recommendation engine for SpatialGABench method selection guidance."""

from __future__ import annotations

from typing import Any

from spatialgabench.recommendation.rules import (
    CICERO_SUBSET,
    CORE_BASELINES,
    DISTAL_METHODS,
    NEGATIVE_CONTROLS,
    REPRESENTATION_BASELINES,
    RULES,
    SOFTWARE_FRAGMENT_METHODS,
)
from spatialgabench.recommendation.schema import validate_recommendation_input


OUTPUT_FIELDS = [
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
]
ALLOWED_CONFIDENCE = {"high", "moderate", "low", "insufficient_information"}
ANNOTATION_DEPENDENT_METHODS = set(
    CORE_BASELINES
    + DISTAL_METHODS
    + NEGATIVE_CONTROLS
    + SOFTWARE_FRAGMENT_METHODS
    + [CICERO_SUBSET]
)
ACCESSIBILITY_INPUT_METHODS = ANNOTATION_DEPENDENT_METHODS | set(REPRESENTATION_BASELINES)


def _append_unique(target: list[str], values: tuple[str, ...] | list[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _remove_values(target: list[str], values: set[str]) -> list[str]:
    return [value for value in target if value not in values]


def _confidence(config: dict[str, Any], matched_rule_ids: list[str]) -> str:
    if config["input_representation"] == "unknown" or config["genome"] == "unknown":
        return "insufficient_information"
    if not config["has_gene_annotation"]:
        return "low"
    if "R10_missing_spatial_for_spatial_task" in matched_rule_ids:
        return "low"
    unknown_qc = [config["data_sparsity"], config["sequencing_depth"], config["expected_distal_regulation"]].count(
        "unknown"
    )
    if unknown_qc >= 2:
        return "moderate"
    return "high"


def recommend_methods(config: dict[str, Any]) -> dict[str, Any]:
    """Return evidence-backed recommendation guidance for one input contract."""

    normalized = validate_recommendation_input(config)
    matched = [rule for rule in RULES if rule.trigger(normalized)]
    matched_rule_ids = [rule.rule_id for rule in matched]

    output: dict[str, Any] = {
        "dataset_id": normalized.get("dataset_id", "unspecified_dataset"),
        "recommendation_scope": "legacy_input_contract_guidance_not_canonical_frozen_policy",
        "canonical_frozen_policy": "results/v1/g4/m4_recommendation_policy/frozen_task_specific_recommendation_policy.json",
        "input_config": normalized,
        "matched_rules": matched_rule_ids,
        "recommended_methods": [],
        "methods_to_use_with_caution": [],
        "not_applicable_methods": [],
        "required_negative_controls": [],
        "required_baselines": [],
        "evaluation_tasks_to_run": [],
        "comparison_scope": [],
        "required_caveats": [],
        "forbidden_interpretations": [],
        "evidence_sources": [],
    }

    has_accessibility_input = normalized["input_representation"] != "gene_activity_matrix"
    has_raw_matrix_or_fragments = normalized["input_representation"] in {
        "author_peak_matrix",
        "fixed_bin_matrix",
        "fragments",
    }
    if normalized["has_gene_annotation"] and has_accessibility_input:
        _append_unique(output["recommended_methods"], CORE_BASELINES)
    if normalized["has_gene_annotation"] and has_accessibility_input:
        _append_unique(output["required_negative_controls"], NEGATIVE_CONTROLS)
    if has_raw_matrix_or_fragments:
        _append_unique(output["required_baselines"], REPRESENTATION_BASELINES)
    _append_unique(output["evaluation_tasks_to_run"], ["construction_qc"])
    _append_unique(output["forbidden_interpretations"], ["Do not report a universal method winner."])
    _append_unique(
        output["required_caveats"],
        [
            "This coarse input-contract guide is not the canonical G4 frozen policy; use the exact task, input-track, and representation contract for prospective selection."
        ],
    )

    for rule in matched:
        _append_unique(output["recommended_methods"], rule.recommended_methods)
        _append_unique(output["methods_to_use_with_caution"], rule.methods_to_use_with_caution)
        _append_unique(output["not_applicable_methods"], rule.not_applicable_methods)
        _append_unique(output["required_negative_controls"], rule.required_negative_controls)
        _append_unique(output["required_baselines"], rule.required_baselines)
        _append_unique(output["evaluation_tasks_to_run"], rule.evaluation_tasks_to_run)
        _append_unique(output["comparison_scope"], rule.comparison_scope)
        _append_unique(output["required_caveats"], rule.required_caveats)
        _append_unique(output["forbidden_interpretations"], rule.forbidden_interpretations)
        _append_unique(output["evidence_sources"], rule.evidence_source)

    if normalized["main_task"] == "clustering_domain":
        _append_unique(output["evaluation_tasks_to_run"], ["clustering_domain_mvp"])
    if normalized["paired_or_adjacent_rna_available"] == "adjacent":
        _append_unique(output["required_caveats"], ["Adjacent RNA is an adjacent reference and not direct paired truth."])
    if normalized["paired_or_adjacent_rna_available"] == "none" and normalized["main_task"] == "rna_agreement":
        _append_unique(output["required_caveats"], ["RNA agreement cannot be run until paired or adjacent RNA is available."])
    if not normalized["fragments_available"]:
        _append_unique(
            output["not_applicable_methods"],
            ["Signac fragment route", "ArchR fragment route", "SnapATAC2 fragment route"],
        )
    if normalized["genome"] == "other":
        _append_unique(output["required_caveats"], ["Genome-specific annotations and software wrappers require explicit validation."])

    if normalized["input_representation"] == "gene_activity_matrix":
        for field in ("recommended_methods", "methods_to_use_with_caution"):
            output[field] = _remove_values(output[field], ACCESSIBILITY_INPUT_METHODS)
        output["required_negative_controls"] = _remove_values(
            output["required_negative_controls"], ACCESSIBILITY_INPUT_METHODS
        )
        output["required_baselines"] = _remove_values(output["required_baselines"], ACCESSIBILITY_INPUT_METHODS)
        _append_unique(output["not_applicable_methods"], sorted(ACCESSIBILITY_INPUT_METHODS))
        _append_unique(
            output["required_caveats"],
            [
                "The input is already a gene activity matrix; accessibility-based construction, raw-peak baselines, and peak-to-gene controls cannot be rerun from this input."
            ],
        )
        _append_unique(output["comparison_scope"], ["gene activity evaluation and QC only"])

    if not normalized["has_gene_annotation"]:
        for field in ("recommended_methods", "methods_to_use_with_caution"):
            output[field] = _remove_values(output[field], ANNOTATION_DEPENDENT_METHODS)
        output["required_negative_controls"] = _remove_values(
            output["required_negative_controls"], ANNOTATION_DEPENDENT_METHODS
        )
        _append_unique(output["not_applicable_methods"], sorted(ANNOTATION_DEPENDENT_METHODS))

    not_applicable_tokens = set(output["not_applicable_methods"])
    output["recommended_methods"] = [
        method for method in output["recommended_methods"] if method not in not_applicable_tokens
    ]
    output["confidence_level"] = _confidence(normalized, matched_rule_ids)
    if output["confidence_level"] not in ALLOWED_CONFIDENCE:
        raise ValueError(f"Unexpected confidence level: {output['confidence_level']}")
    return output
