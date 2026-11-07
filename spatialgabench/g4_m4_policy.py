"""Freeze the G4 task-specific recommendation policy from development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from spatialgabench.software_wrappers.common import sha256_file


ACTIVE_TASKS = {"paired_rna_fidelity", "portability_runtime_failure"}
INSUFFICIENT_TASKS = {
    "spatial_domain_and_pathology",
    "simulation_truth_recovery",
    "independent_pathology_accuracy",
}
REQUIRED_TRACKS = {"F", "P", "B"}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_false(config: dict[str, Any], fields: Iterable[str]) -> None:
    enabled = [field for field in fields if config.get(field) is not False]
    if enabled:
        raise PermissionError(f"M4 forbidden execution flags are not false: {enabled}")


def _validate_contract(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for key in ("authorization", "preregistration", "previous_gate"):
        path = root / str(config[key])
        if not path.is_file() or sha256_file(path) != config[f"{key}_sha256"]:
            raise ValueError(f"M4 frozen artifact mismatch: {key}")
    for evidence_id, record in config.get("evidence", {}).items():
        path_text = str(record.get("path", ""))
        if any(accession.lower() in path_text.lower() for accession in config["forbidden_holdout_accessions"]):
            raise PermissionError(f"Holdout path declared as M4 evidence: {evidence_id}")
        path = root / path_text
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"M4 evidence checksum mismatch: {evidence_id}")

    authorization = _load_json(root / config["authorization"])
    preregistration = _load_yaml(root / config["preregistration"])
    previous_gate = _load_json(root / config["previous_gate"])
    policy = config["policy"]
    prereg_policy = preregistration["recommendation_policy"]
    holdout = prereg_policy["holdout_evaluation"]
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "M4_development_recommendation_policy_fit"
        or authorization.get("holdout_access_authorized") is not False
        or authorization.get("g5_execution_authorized") is not False
    ):
        raise PermissionError("M4 development-only authorization is invalid")
    if previous_gate.get("status") != "pass" or previous_gate.get("holdout_accessed") is not False:
        raise ValueError("M4 previous gate boundary failed")
    if (
        policy["maximum_recommended_methods"] != prereg_policy["maximum_recommended_methods"]
        or policy["minimum_development_studies_per_leaf"]
        != prereg_policy["complexity_cap"]["minimum_development_studies_per_leaf"]
        or policy["maximum_context_splits"] != prereg_policy["complexity_cap"]["maximum_context_splits"]
        or policy["reliability_minimum_observed_success_rate"]
        != prereg_policy["reliability_filter"]["minimum_observed_success_rate"]
        or policy["regret_noninferiority_margin"] != holdout["regret_noninferiority_margin"]
        or policy["failure_rate_noninferiority_margin"] != holdout["failure_rate_noninferiority_margin"]
        or policy["random_comparator_draws"] != holdout["random_comparator_draws"]
        or policy["random_seed"] != holdout["random_seed"]
    ):
        raise ValueError("M4 policy constants do not match preregistration")
    if set(config["forbidden_policy_inputs"]) != set(prereg_policy["forbidden_context_inputs"]):
        raise ValueError("M4 forbidden policy inputs do not match preregistration")
    _assert_false(
        config,
        (
            "cross_task_utility_pooling",
            "cross_representation_utility_pooling",
            "runtime_or_memory_method_comparison",
            "universal_representation_selection",
            "universal_ranking",
            "final_ranking",
            "holdout_accessed",
            "full_benchmark_run",
            "g5_execution",
        ),
    )
    return authorization, preregistration


def _pareto_methods(utilities: pd.DataFrame) -> list[str]:
    """Return methods not dominated across development-study utility columns."""
    methods = sorted(utilities.index.astype(str))
    values = utilities.loc[methods].to_numpy(dtype=float)
    keep: list[str] = []
    for index, method in enumerate(methods):
        dominated = False
        for other in range(len(methods)):
            if other == index:
                continue
            if np.all(values[other] >= values[index]) and np.any(values[other] > values[index]):
                dominated = True
                break
        if not dominated:
            keep.append(method)
    return keep


def _selection(
    utilities: pd.DataFrame,
    *,
    maximum: int,
    tie_default: str | None = None,
) -> tuple[list[str], list[str], str]:
    pareto = _pareto_methods(utilities)
    if len(pareto) <= maximum:
        return pareto, pareto, "complete_pareto_set"
    vectors = utilities.loc[pareto].to_numpy(dtype=float)
    all_equal = bool(np.allclose(vectors, vectors[0], rtol=0.0, atol=0.0))
    if all_equal and tie_default in pareto:
        return pareto, [str(tie_default)], "preregistered_default_resolves_nondiscriminating_tie"
    raise ValueError(f"Pareto set exceeds frozen cap without a preregistered tie rule: {pareto}")


def _utility_payload(frame: pd.DataFrame) -> str:
    nested = {
        str(method): {str(study): float(value) for study, value in row.items()}
        for method, row in frame.sort_index().iterrows()
    }
    return json.dumps(nested, sort_keys=True, separators=(",", ":"))


def _active_leaf(
    *,
    policy_id: str,
    task_id: str,
    input_track: str,
    representation_id: str,
    utilities: pd.DataFrame,
    eligible_methods: list[str],
    maximum: int,
    primary_endpoint: str,
    required_controls: list[str],
    support_species: list[str],
    support_genome_builds: list[str],
    tie_default: str | None,
    caveat: str,
) -> dict[str, Any]:
    utilities = utilities.loc[sorted(eligible_methods)].sort_index(axis=1)
    pareto, recommended, selection_rule = _selection(
        utilities, maximum=maximum, tie_default=tie_default
    )
    studies = sorted(map(str, utilities.columns))
    return {
        "policy_id": policy_id,
        "task_id": task_id,
        "input_track": input_track,
        "representation_scope": representation_id,
        "context_predicate": f"primary_task={task_id};input_track={input_track};representation_id={representation_id}",
        "status": "active_frozen",
        "support_studies": ";".join(studies),
        "n_study_families": len(studies),
        "support_species": ";".join(sorted(support_species)),
        "support_genome_builds": ";".join(sorted(support_genome_builds)),
        "eligible_methods": ";".join(sorted(eligible_methods)),
        "pareto_methods": ";".join(pareto),
        "recommended_methods": ";".join(recommended),
        "n_recommended_methods": len(recommended),
        "selection_rule": selection_rule,
        "required_controls": ";".join(required_controls),
        "primary_endpoint": primary_endpoint,
        "study_utility_json": _utility_payload(utilities),
        "required_caveat": caveat,
        "forbidden_interpretation": "No universal method or representation winner; do not pool this leaf with another task, track, or representation.",
    }


def _insufficient_leaf(task_id: str, reason: str, support: int) -> dict[str, Any]:
    return {
        "policy_id": f"{task_id}__insufficient_evidence",
        "task_id": task_id,
        "input_track": "not_fitted",
        "representation_scope": "not_fitted",
        "context_predicate": f"primary_task={task_id}",
        "status": "insufficient_evidence_no_recommendation",
        "support_studies": "",
        "n_study_families": support,
        "support_species": "",
        "support_genome_builds": "",
        "eligible_methods": "",
        "pareto_methods": "",
        "recommended_methods": "",
        "n_recommended_methods": 0,
        "selection_rule": "no_policy_leaf_fitted",
        "required_controls": "",
        "primary_endpoint": "not_evaluable_for_policy_fit",
        "study_utility_json": "{}",
        "required_caveat": reason,
        "forbidden_interpretation": "Do not substitute proxy outcomes or simulation replicates for independent biological study families.",
    }


def _rna_leaves(config: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    paths = [
        root / config["evidence"]["paired_rna_study_1"]["path"],
        root / config["evidence"]["paired_rna_study_2"]["path"],
    ]
    frame = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
    endpoint = config["policy"]["paired_rna_primary_endpoint"]
    active = set(config["annotation_methods"]) | set(config["software_methods"])
    frame = frame[frame.method_id.isin(active) & frame.status.eq("pass")].copy()
    frame["species"] = frame["genome_build"].map({"mm10": "mouse", "hg38": "human"})
    if frame.species.isna().any():
        raise ValueError("RNA policy evidence has an unsupported genome build")
    frame["utility"] = np.clip((frame[endpoint].astype(float) + 1.0) / 2.0, 0.0, 1.0)
    leaves: list[dict[str, Any]] = []
    for (track, representation), branch in frame.groupby(["input_track", "representation_id"], sort=True):
        utilities = branch.pivot(index="method_id", columns="study_id", values="utility")
        if utilities.isna().any().any() or utilities.shape[1] != 2:
            raise ValueError(f"RNA policy branch lacks two complete study families: {track}/{representation}")
        methods = sorted(utilities.index.astype(str))
        controls = [] if track == "F" else ["accessibility_matched_random_peak_to_gene_control_pc"]
        caveat = (
            "Fragment-software RNA utilities are descriptive because no same-track matched random control exists."
            if track == "F"
            else "RNA is evaluation-only; construction tuning by RNA remains forbidden."
        )
        leaves.append(
            _active_leaf(
                policy_id=f"paired_rna_fidelity__{track}__{representation}",
                task_id="paired_rna_fidelity",
                input_track=str(track),
                representation_id=str(representation),
                utilities=utilities,
                eligible_methods=methods,
                maximum=int(config["policy"]["maximum_recommended_methods"]),
                primary_endpoint=endpoint,
                required_controls=controls,
                support_species=sorted(branch.species.astype(str).unique()),
                support_genome_builds=sorted(branch.genome_build.astype(str).unique()),
                tie_default=config["policy"]["fragment_tie_default"] if track == "F" else config["policy"]["matrix_tie_default"],
                caveat=caveat,
            )
        )
    return leaves, frame


def _portability_leaves(config: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    frame = pd.read_csv(root / config["evidence"]["admissibility_runs"]["path"], sep="\t")
    active = set(config["annotation_methods"]) | set(config["software_methods"])
    frame = frame[
        frame.method_id.isin(active)
        & frame.eligible.eq(True)
        & frame.subset_only.eq(False)
        & frame.included_in_primary_denominator.eq(True)
    ].copy()
    frame["utility"] = frame["valid_output_indicator"].astype(float)
    leaves: list[dict[str, Any]] = []
    minimum_rate = float(config["policy"]["reliability_minimum_observed_success_rate"])
    for (track, representation), branch in frame.groupby(["input_track", "representation_id"], sort=True):
        success = branch.groupby("method_id")["utility"].mean()
        eligible = sorted(success[success >= minimum_rate].index.astype(str))
        utilities = branch[branch.method_id.isin(eligible)].pivot(
            index="method_id", columns="study_id", values="utility"
        )
        if utilities.isna().any().any() or utilities.shape[1] < 2:
            raise ValueError(f"Portability policy branch lacks complete multi-study support: {track}/{representation}")
        leaves.append(
            _active_leaf(
                policy_id=f"portability_runtime_failure__{track}__{representation}",
                task_id="portability_runtime_failure",
                input_track=str(track),
                representation_id=str(representation),
                utilities=utilities,
                eligible_methods=eligible,
                maximum=int(config["policy"]["maximum_recommended_methods"]),
                primary_endpoint=config["policy"]["portability_primary_endpoint"],
                required_controls=[],
                support_species=sorted(branch.species.astype(str).unique()),
                support_genome_builds=sorted(branch.genome_build.astype(str).unique()),
                tie_default=config["policy"]["fragment_tie_default"] if track == "F" else config["policy"]["matrix_tie_default"],
                caveat="Runtime and memory were collected noncomparatively and do not enter this policy; failures remain reportable outcomes.",
            )
        )
    return leaves, frame


def _evidence_inventory(config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows = []
    for evidence_id, record in config["evidence"].items():
        path = root / record["path"]
        rows.append(
            {
                "evidence_id": evidence_id,
                "path": record["path"],
                "sha256": sha256_file(path),
                "role": record["role"],
                "development_only": True,
                "holdout_path": False,
                "status": "verified",
            }
        )
    return pd.DataFrame(rows)


def _validate_policy(leaves: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: Any) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    active = leaves[leaves.status.eq("active_frozen")]
    insufficient = leaves[leaves.status.eq("insufficient_evidence_no_recommendation")]
    check("active_tasks_exact", set(active.task_id) == ACTIVE_TASKS, sorted(set(active.task_id)))
    for task in sorted(ACTIVE_TASKS):
        task_tracks = set(active.loc[active.task_id.eq(task), "input_track"])
        check(f"{task}_tracks_separate_and_complete", task_tracks == REQUIRED_TRACKS, sorted(task_tracks))
    check(
        "minimum_studies_per_active_leaf",
        bool((active.n_study_families >= config["policy"]["minimum_development_studies_per_leaf"]).all()),
        int(active.n_study_families.min()),
    )
    check(
        "recommendation_set_bounded",
        bool((active.n_recommended_methods.between(1, config["policy"]["maximum_recommended_methods"])).all()),
        int(active.n_recommended_methods.max()),
    )
    controls = set(config["negative_controls"]) | {config["representation_control"]}
    recommended = {
        method
        for values in active.recommended_methods
        for method in str(values).split(";")
        if method
    }
    check("controls_not_recommended", recommended.isdisjoint(controls), sorted(recommended & controls))
    check("insufficient_tasks_exact", set(insufficient.task_id) == INSUFFICIENT_TASKS, sorted(set(insufficient.task_id)))
    check("insufficient_tasks_return_nothing", bool((insufficient.n_recommended_methods == 0).all()), int(insufficient.n_recommended_methods.sum()))
    check("no_cross_track_leaf", bool(active.input_track.isin(REQUIRED_TRACKS).all()), sorted(set(active.input_track)))
    check("no_duplicate_policy_id", bool(leaves.policy_id.is_unique), len(leaves))
    check("holdout_not_accessed", config["holdout_accessed"] is False, config["holdout_accessed"])
    check("universal_ranking_disabled", config["universal_ranking"] is False, config["universal_ranking"])
    check("final_ranking_disabled", config["final_ranking"] is False, config["final_ranking"])
    return checks


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    authorization, preregistration = _validate_contract(config, root)
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)

    rna_leaves, rna_evidence = _rna_leaves(config, root)
    portability_leaves, portability_evidence = _portability_leaves(config, root)
    leaves = rna_leaves + portability_leaves
    leaves.extend(
        [
            _insufficient_leaf(
                "spatial_domain_and_pathology",
                "Two author-domain studies use different unmatched author representations; pooling them into one policy leaf is forbidden.",
                2,
            ),
            _insufficient_leaf(
                "simulation_truth_recovery",
                "Replicated simulation seeds are not independent biological study families and cannot satisfy the policy-leaf support rule.",
                0,
            ),
            _insufficient_leaf(
                "independent_pathology_accuracy",
                "No public development dataset currently provides evaluation-ready independent machine-readable pathology truth.",
                0,
            ),
        ]
    )
    leaf_table = pd.DataFrame(leaves).sort_values(["status", "task_id", "input_track", "representation_scope"])
    checks = _validate_policy(leaf_table, config)
    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    if status != "pass":
        failed = [row for row in checks if row["status"] == "fail"]
        raise ValueError(f"M4 policy validation failed: {failed}")

    inventory = _evidence_inventory(config, root)
    inventory_path = output / "development_evidence_inventory.tsv"
    leaf_path = output / "task_specific_policy_leaves.tsv"
    rna_path = output / "paired_rna_policy_evidence.tsv"
    portability_path = output / "portability_policy_evidence.tsv"
    inventory.to_csv(inventory_path, sep="\t", index=False)
    leaf_table.to_csv(leaf_path, sep="\t", index=False)
    rna_evidence.to_csv(rna_path, sep="\t", index=False)
    portability_evidence.to_csv(portability_path, sep="\t", index=False)

    policy_path = output / "frozen_task_specific_recommendation_policy.json"
    policy_payload = {
        "schema_version": 1,
        "policy_id": "spatialgabench_g4_development_task_specific_v1",
        "status": "frozen",
        "frozen_on": "2026-07-17",
        "policy_output": "admissible_pareto_set",
        "maximum_recommended_methods": config["policy"]["maximum_recommended_methods"],
        "minimum_development_studies_per_leaf": config["policy"]["minimum_development_studies_per_leaf"],
        "maximum_context_splits": config["policy"]["maximum_context_splits"],
        "active_context_splits": ["primary_task", "input_track", "representation_id"],
        "compatibility_filters_not_learned_splits": ["species", "genome_build", "fragments_available"],
        "unused_allowed_context_due_insufficient_support": [
            "n_observations",
            "median_fragments_per_observation",
            "accessibility_sparsity",
            "runtime_budget_class",
        ],
        "forbidden_context_inputs": config["forbidden_policy_inputs"],
        "reliability_minimum_observed_success_rate": config["policy"]["reliability_minimum_observed_success_rate"],
        "regret_noninferiority_margin": config["policy"]["regret_noninferiority_margin"],
        "failure_rate_noninferiority_margin": config["policy"]["failure_rate_noninferiority_margin"],
        "random_comparator_draws": config["policy"]["random_comparator_draws"],
        "random_seed": config["policy"]["random_seed"],
        "leaves": leaf_table.to_dict(orient="records"),
        "cross_task_utility_pooling": False,
        "cross_representation_utility_pooling": False,
        "runtime_or_memory_method_comparison": False,
        "universal_representation_selection": False,
        "universal_ranking": False,
        "holdout_accessed": False,
        "g5_execution_authorized": False,
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "preregistration": {"path": config["preregistration"], "sha256": config["preregistration_sha256"]},
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
    }
    _write_json(policy_path, policy_payload)

    validation_path = output / "policy_validation.json"
    validation = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M4_development_recommendation_policy_freeze",
        "status": status,
        "checks": checks,
        "active_policy_leaves": int(leaf_table.status.eq("active_frozen").sum()),
        "insufficient_evidence_tasks": int(leaf_table.status.eq("insufficient_evidence_no_recommendation").sum()),
        "holdout_accessed": False,
        "full_benchmark_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    _write_json(validation_path, validation)

    evidence_path = output / "completion_evidence.json"
    artifacts = [inventory_path, leaf_path, rna_path, portability_path, policy_path, validation_path]
    completion = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M4_development_recommendation_policy_freeze",
        "status": "pass",
        "policy_frozen": True,
        "active_policy_leaves": int(leaf_table.status.eq("active_frozen").sum()),
        "active_tasks": sorted(ACTIVE_TASKS),
        "input_tracks_separate": sorted(REQUIRED_TRACKS),
        "insufficient_evidence_tasks": sorted(INSUFFICIENT_TASKS),
        "minimum_studies_per_active_leaf": int(
            leaf_table.loc[leaf_table.status.eq("active_frozen"), "n_study_families"].min()
        ),
        "maximum_methods_returned": int(leaf_table.n_recommended_methods.max()),
        "artifacts": {
            path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts
        },
        "source_evidence": {
            key: {"path": value["path"], "sha256": value["sha256"], "role": value["role"]}
            for key, value in config["evidence"].items()
        },
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_file(config_path)},
        "preregistration_status": preregistration["status"],
        "authorization_status": authorization["status"],
        "rna_or_labels_used_for_construction": False,
        "holdout_accessed": False,
        "full_benchmark_run": False,
        "universal_representation_selected": False,
        "cross_task_utility_pooling_run": False,
        "cross_representation_utility_pooling_run": False,
        "runtime_or_memory_comparison_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "g5_execution_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(evidence_path, completion)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m4_recommendation_policy_freeze.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
