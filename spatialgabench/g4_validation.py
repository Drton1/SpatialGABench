"""Validation for the frozen G4 development benchmark preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_DEVELOPMENT_STUDIES = {
    "llorens_bobadilla_2023",
    "zhang_2023",
    "deng_2022",
    "li_2026",
    "guo_2025_spatial_mux",
    "guo_deng_2025_ffpe_atac",
}
EXPECTED_HOLDOUT_STUDIES = {"zhang_2025_brain_triomics", "huang_2025"}
EXPECTED_MATCHED_STUDIES = {
    "zhang_2023",
    "deng_2022",
    "li_2026",
    "guo_2025_spatial_mux",
    "guo_deng_2025_ffpe_atac",
}
EXPECTED_MATCHED_BRANCHES = {
    "macs3_summit_501bp",
    "fixed_bin_10kb",
    "fixed_bin_25kb",
    "fixed_bin_50kb",
}
EXPECTED_TASKS = {
    "construction_qc",
    "paired_rna_fidelity",
    "spatial_domain_and_pathology",
    "simulation_truth_recovery",
    "portability_runtime_failure",
}
EXPECTED_CROSSOVER_METHODS = {
    "promoter_only_2kb_pc",
    "gene_body_pc",
    "promoter_plus_gene_body_2kb_pc",
    "distance_weighted_distal_250kb_decay50kb_pc",
    "promoter_plus_distal_2kb_250kb_decay50kb_pc",
}
EXPECTED_SOFTWARE_METHODS = {
    "snapatac2_geneactivity",
    "signac_geneactivity",
    "archr_genescorematrix",
}
FORBIDDEN_PLACEHOLDERS = {"tbd", "todo", "to_be_frozen", "to be frozen", "placeholder"}
FORBIDDEN_RECOMMENDATION_INPUTS = {
    "rna_agreement_outcome",
    "labels",
    "clusters",
    "pathology_outcome",
    "simulation_truth",
    "holdout_method_outcome",
}


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _repo_root(path: Path) -> Path:
    for candidate in [path.parent, *path.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"Could not locate repository root from {path}")


def _registered_methods(registry: dict[str, Any]) -> set[str]:
    methods: set[str] = set()
    for track in registry.get("tracks", {}).values():
        if isinstance(track, dict) and isinstance(track.get("methods"), dict):
            methods.update(map(str, track["methods"]))
    return methods


def _placeholder_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_placeholder_paths(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_placeholder_paths(nested, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if any(token in normalized for token in FORBIDDEN_PLACEHOLDERS):
            paths.append(prefix)
    return paths


def validate_g4_preregistration(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    prereg_path = Path(path).resolve()
    root = Path(root).resolve() if root is not None else _repo_root(prereg_path)
    prereg = _load_mapping(prereg_path)
    config = _load_mapping(root / "configs/main.yaml")
    datasets = _load_mapping(root / "registries/v1/datasets.yaml")
    methods = _load_mapping(root / "registries/v1/methods.yaml")
    tasks_registry = _load_mapping(root / "registries/v1/tasks.yaml")
    checks: list[dict[str, str]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    scope = prereg.get("scope", {})
    dataset_panel = prereg.get("dataset_panel", {})
    representations = prereg.get("representations", {})
    method_panel = prereg.get("method_panel", {})
    task_panel = prereg.get("tasks", {})
    statistics = prereg.get("statistics", {})
    recommendation = prereg.get("recommendation_policy", {})
    execution = prereg.get("execution", {})
    project_execution = config.get("execution", {})

    check("schema_version", prereg.get("schema_version") == 1, prereg.get("schema_version"))
    check("status_frozen", prereg.get("status") == "frozen", prereg.get("status"))
    check("freeze_date", str(prereg.get("frozen_on")) == "2026-07-14", prereg.get("frozen_on"))
    check("current_gate_g4", config.get("project", {}).get("current_gate") == "G4", config.get("project", {}))
    check("no_placeholders", not _placeholder_paths(prereg), _placeholder_paths(prereg))

    boundary_fields = (
        "development_benchmark_run",
        "holdout_accessed",
        "full_benchmark_run",
        "final_ranking_run",
        "construction_evaluation_inputs_used",
        "execution_enabled",
    )
    for field in boundary_fields:
        check(f"scope_{field}_false", scope.get(field) is False, scope.get(field))
    for field in ("allow_full_benchmark", "allow_results_writing", "allow_final_ranking"):
        check(f"project_{field}_false", project_execution.get(field) is False, project_execution.get(field))

    development = set(map(str, dataset_panel.get("development_studies", [])))
    holdouts = set(map(str, dataset_panel.get("frozen_holdout_studies", [])))
    matched = set(map(str, dataset_panel.get("matched_fragment_studies", [])))
    check("development_panel_exact", development == EXPECTED_DEVELOPMENT_STUDIES, sorted(development))
    check("development_matches_main_config", development == set(config.get("development_studies", [])), sorted(development))
    check("holdout_panel_exact", holdouts == EXPECTED_HOLDOUT_STUDIES, sorted(holdouts))
    check("holdouts_match_main_config", holdouts == set(config.get("holdout_studies", [])), sorted(holdouts))
    check("development_holdout_disjoint", development.isdisjoint(holdouts), sorted(development & holdouts))
    check("matched_panel_exact", matched == EXPECTED_MATCHED_STUDIES, sorted(matched))
    check("minimum_matched_studies", dataset_panel.get("minimum_independent_matched_studies_for_C1") == 3, dataset_panel.get("minimum_independent_matched_studies_for_C1"))

    studies = datasets.get("studies", {})
    matched_capabilities_valid = all(
        study_id in studies
        and studies[study_id].get("role") == "development"
        and studies[study_id].get("capabilities", {}).get("standard_fragment_level") is True
        for study_id in matched
    )
    check("matched_study_capabilities", matched_capabilities_valid, sorted(matched))
    holdout_freeze_valid = all(
        any(
            slot.get("study_id") == study_id
            and slot.get("freeze_status") == "frozen"
            and slot.get("benchmark_outcomes_inspected") is False
            for slot in datasets.get("holdout_slots", {}).values()
            if isinstance(slot, dict)
        )
        for study_id in holdouts
    )
    check("holdout_freeze_contract", holdout_freeze_valid, sorted(holdouts))
    paired_primary = set(map(str, dataset_panel.get("paired_rna_primary_development_studies", [])))
    paired_primary_valid = paired_primary == {"guo_2025_spatial_mux"} and all(
        studies[study_id].get("capabilities", {}).get("verified_paired_spatial_rna") is True
        for study_id in paired_primary
    )
    check("paired_rna_primary_scope", paired_primary_valid, sorted(paired_primary))

    branches = set(map(str, representations.get("matched_branches", [])))
    check("matched_branches_exact", branches == EXPECTED_MATCHED_BRANCHES, sorted(branches))
    check("fixed_bin_sizes", representations.get("fixed_bin_sizes_bp") == [10000, 25000, 50000], representations.get("fixed_bin_sizes_bp"))
    check("peak_width", representations.get("peak_width_bp") == 501, representations.get("peak_width_bp"))
    check("low_capture_retained", representations.get("low_peak_capture_policy") == "retain_and_flag", representations.get("low_peak_capture_policy"))
    check("outcome_exclusion_forbidden", representations.get("outcome_based_branch_exclusion_forbidden") is True, representations.get("outcome_based_branch_exclusion_forbidden"))

    registered_methods = _registered_methods(methods)
    crossover = set(map(str, representations.get("crossover_methods", [])))
    full_software = set(map(str, method_panel.get("full_fragment_software", [])))
    subset_methods = set(map(str, method_panel.get("subset_scalability_only", [])))
    declared_methods = set()
    for value in method_panel.values():
        if isinstance(value, list):
            declared_methods.update(map(str, value))
    check("crossover_methods_exact", crossover == EXPECTED_CROSSOVER_METHODS, sorted(crossover))
    check("software_methods_exact", full_software == EXPECTED_SOFTWARE_METHODS, sorted(full_software))
    check("cicero_subset_only", subset_methods == {"cicero_geneactivity"}, sorted(subset_methods))
    check("all_methods_registered", declared_methods <= registered_methods, sorted(declared_methods - registered_methods))
    ladder = method_panel.get("cicero_scalability_ladder", {})
    check("cicero_ladder_frozen", ladder.get("cells") == [500, 1000, 2500, 5000] and ladder.get("performance_pooling_forbidden") is True, ladder)

    registered_tasks = tasks_registry.get("tasks", {})
    check("task_panel_exact", set(task_panel) == EXPECTED_TASKS, sorted(task_panel))
    endpoint_values: list[str] = []
    task_rows_valid = True
    for task_id in EXPECTED_TASKS:
        row = task_panel.get(task_id, {})
        registry_row = registered_tasks.get(task_id, {})
        endpoint = row.get("primary_endpoint")
        endpoint_values.append(str(endpoint))
        task_rows_valid = task_rows_valid and (
            isinstance(row, dict)
            and row.get("status") in {"frozen", "frozen_conditional_on_audited_labels"}
            and row.get("direction") == "higher"
            and isinstance(endpoint, str)
            and endpoint == registry_row.get("primary_endpoint")
            and registry_row.get("status") == row.get("status")
        )
    check("task_rows_frozen_and_registered", task_rows_valid, sorted(endpoint_values))
    check("primary_endpoints_unique", len(endpoint_values) == len(set(endpoint_values)), endpoint_values)
    domain = task_panel.get("spatial_domain_and_pathology", {})
    check("no_label_metric_policy", domain.get("no_label_policy") == "no_ari_nmi_or_accuracy", domain.get("no_label_policy"))
    check("label_selection_forbidden", domain.get("label_based_parameter_selection_forbidden") is True, domain.get("label_based_parameter_selection_forbidden"))

    seeds = config.get("execution", {}).get("seeds")
    check("statistics_seeds_match", statistics.get("seeds") == seeds == [1, 2, 3, 4, 5], statistics.get("seeds"))
    check("study_level_inference", statistics.get("biological_generalization_unit") == "study_family", statistics.get("biological_generalization_unit"))
    check("universal_rank_forbidden", statistics.get("universal_rank_forbidden") is True, statistics.get("universal_rank_forbidden"))
    check("study_bootstrap_frozen", statistics.get("study_level_bootstrap_replicates") == 2000, statistics.get("study_level_bootstrap_replicates"))

    allowed_context = set(map(str, recommendation.get("allowed_context_inputs", [])))
    forbidden_context = set(map(str, recommendation.get("forbidden_context_inputs", [])))
    check("recommendation_context_disjoint", allowed_context.isdisjoint(forbidden_context), sorted(allowed_context & forbidden_context))
    check("recommendation_forbidden_inputs_complete", forbidden_context == FORBIDDEN_RECOMMENDATION_INPUTS, sorted(forbidden_context))
    check("recommendation_set_bounded", recommendation.get("maximum_recommended_methods") == 3, recommendation.get("maximum_recommended_methods"))
    complexity = recommendation.get("complexity_cap", {})
    check("recommendation_complexity_cap", complexity.get("maximum_context_splits") == 3 and complexity.get("minimum_development_studies_per_leaf") == 2, complexity)
    holdout_evaluation = recommendation.get("holdout_evaluation", {})
    check("holdout_evaluate_once", holdout_evaluation.get("evaluate_once") is True, holdout_evaluation.get("evaluate_once"))
    check("regret_margin_frozen", holdout_evaluation.get("regret_noninferiority_margin") == 0.05, holdout_evaluation.get("regret_noninferiority_margin"))
    check("failure_margin_frozen", holdout_evaluation.get("failure_rate_noninferiority_margin") == 0.10, holdout_evaluation.get("failure_rate_noninferiority_margin"))
    check("no_cross_task_utility", recommendation.get("no_cross_task_utility_pooling") is True, recommendation.get("no_cross_task_utility_pooling"))

    forbidden_execution = set(map(str, execution.get("forbidden_until_separate_authorization", [])))
    check("holdout_mount_forbidden", "frozen_holdout_mount" in forbidden_execution, sorted(forbidden_execution))
    check("benchmark_execution_forbidden", "full_benchmark_execution" in forbidden_execution, sorted(forbidden_execution))
    check("ranking_forbidden", "final_ranking" in forbidden_execution, sorted(forbidden_execution))

    status = "pass" if checks and all(row["status"] == "pass" for row in checks) else "fail"
    preregistration_label = (
        str(prereg_path.relative_to(root)) if prereg_path.is_relative_to(root) else str(prereg_path)
    )
    return {
        "schema_version": 1,
        "gate": "G4",
        "status": status,
        "scope": "preregistration_validation_only",
        "preregistration": {
            "path": preregistration_label,
            "sha256": sha256_path(prereg_path),
        },
        "artifacts": {
            "main_config": {"path": "configs/main.yaml", "sha256": sha256_path(root / "configs/main.yaml")},
            "dataset_registry": {"path": "registries/v1/datasets.yaml", "sha256": sha256_path(root / "registries/v1/datasets.yaml")},
            "method_registry": {"path": "registries/v1/methods.yaml", "sha256": sha256_path(root / "registries/v1/methods.yaml")},
            "task_registry": {"path": "registries/v1/tasks.yaml", "sha256": sha256_path(root / "registries/v1/tasks.yaml")},
            "representation_protocol": {"path": "protocols/REPRESENTATION_FAIRNESS_PROTOCOL.md", "sha256": sha256_path(root / "protocols/REPRESENTATION_FAIRNESS_PROTOCOL.md")},
            "statistical_plan": {"path": "protocols/STATISTICAL_ANALYSIS_PLAN.md", "sha256": sha256_path(root / "protocols/STATISTICAL_ANALYSIS_PLAN.md")},
        },
        "checks": checks,
        "development_benchmark_run": False,
        "holdout_accessed": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
    }


def write_g4_preregistration_validation(path: str | Path, output_path: str | Path) -> dict[str, Any]:
    payload = validate_g4_preregistration(path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
