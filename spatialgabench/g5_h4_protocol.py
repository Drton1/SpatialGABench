"""Freeze the metadata-only G5 one-pass execution protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


FALSE_FIELDS = (
    "policy_refit_allowed",
    "method_panel_change_allowed",
    "representation_change_allowed",
    "threshold_change_allowed",
    "holdout_substitution_after_access_allowed",
    "raw_holdout_accessed",
    "representation_construction_run",
    "method_execution_run",
    "rna_value_access_run",
    "holdout_outcome_accessed",
    "regret_computation_run",
    "full_benchmark_run",
    "results_writing_run",
    "universal_ranking_run",
    "final_ranking_run",
    "g5_execution_authorized",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _yaml(config_path)
    for key in (
        "authorization",
        "previous_gate",
        "frozen_policy",
        "development_preregistration",
        "statistical_plan",
        "original_holdout_freeze",
        "h0_preaccess_evidence",
        "supplemental_registry",
    ):
        path = root / str(config[key])
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"G5-H4 frozen artifact mismatch: {key}")
    authorization = _json(root / config["authorization"])
    previous = _json(root / config["previous_gate"])
    policy = _json(root / config["frozen_policy"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "H4_one_pass_external_validation_protocol_freeze"
        or authorization.get("raw_holdout_access_authorized") is not False
        or authorization.get("holdout_outcome_access_authorized") is not False
        or authorization.get("execution_manifest_generation_authorized") is not True
    ):
        raise PermissionError("G5-H4 protocol-freeze authorization is invalid")
    if (
        previous.get("status") != "pass"
        or previous.get("one_pass_g5_execution_authorized") is not False
        or previous.get("method_execution_run") is not False
    ):
        raise ValueError("G5-H3 boundary is invalid")
    if (
        policy.get("status") != "frozen"
        or policy.get("holdout_accessed") is not False
        or policy.get("g5_execution_authorized") is not False
        or len([leaf for leaf in policy.get("leaves", []) if leaf.get("status") == "active_frozen"]) != 10
    ):
        raise ValueError("G5-H4 M4 policy contract is invalid")
    invalid = [field for field in FALSE_FIELDS if config.get(field) is not False]
    if invalid:
        raise PermissionError(f"G5-H4 forbidden flags are not false: {invalid}")
    constants = config["comparators"] | {
        "regret_noninferiority_margin": config["regret_and_failure"]["regret_noninferiority_margin"],
        "failure_rate_noninferiority_margin": config["regret_and_failure"]["failure_rate_noninferiority_margin"],
    }
    expected = {
        "random_comparator_draws": policy["random_comparator_draws"],
        "random_seed": policy["random_seed"],
        "regret_noninferiority_margin": policy["regret_noninferiority_margin"],
        "failure_rate_noninferiority_margin": policy["failure_rate_noninferiority_margin"],
    }
    if any(constants[key] != value for key, value in expected.items()):
        raise ValueError("G5-H4 comparator constants differ from M4")
    return config, policy


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config, policy = _load(config_path, root)
    active_leaves = [leaf for leaf in policy["leaves"] if leaf["status"] == "active_frozen"]
    representations = config["representations"]
    studies = config["studies"]

    leaf_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    comparator_rows: list[dict[str, Any]] = []
    execution_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for study_id, study in studies.items():
        compatible = [
            leaf
            for leaf in active_leaves
            if leaf["task_id"] in study["evaluable_tasks"]
            and study["species"] in leaf["support_species"].split(";")
            and study["genome_build"] in leaf["support_genome_builds"].split(";")
        ]
        if len(compatible) != study["exact_active_leaves"]:
            raise ValueError(f"G5-H4 exact leaf count mismatch: {study_id}")
        for leaf in compatible:
            representation = leaf["representation_scope"]
            if representation not in representations or representations[representation]["input_track"] != leaf["input_track"]:
                raise ValueError(f"G5-H4 representation mismatch: {leaf['policy_id']}")
            methods = leaf["eligible_methods"].split(";")
            recommended = leaf["recommended_methods"].split(";")
            if not set(recommended) <= set(methods) or len(recommended) != leaf["n_recommended_methods"]:
                raise ValueError(f"G5-H4 recommended-set mismatch: {leaf['policy_id']}")
            leaf_rows.append(
                {
                    "study_id": study_id,
                    "accession": study["accession"],
                    "sample_id": study["sample_id"],
                    "policy_id": leaf["policy_id"],
                    "task_id": leaf["task_id"],
                    "input_track": leaf["input_track"],
                    "representation_id": representation,
                    "frozen_observations": study["frozen_observations"],
                    "eligible_methods": len(methods),
                    "recommended_methods": ";".join(recommended),
                    "primary_endpoint": leaf["primary_endpoint"],
                    "evaluation_status": "planned_not_authorized",
                }
            )
            default_method = (
                config["comparators"]["fragment_software_default"]
                if leaf["input_track"] == "F"
                else config["comparators"]["annotation_matrix_default"]
            )
            comparator_rows.append(
                {
                    "study_id": study_id,
                    "policy_id": leaf["policy_id"],
                    "task_id": leaf["task_id"],
                    "representation_id": representation,
                    "admissible_methods": len(methods),
                    "recommended_set_size": len(recommended),
                    "default_method": default_method,
                    "random_rule": config["comparators"]["random_selection"],
                    "random_draws": config["comparators"]["random_comparator_draws"],
                    "random_seed": config["comparators"]["random_seed"],
                    "regret_margin": config["regret_and_failure"]["regret_noninferiority_margin"],
                    "failure_margin": config["regret_and_failure"]["failure_rate_noninferiority_margin"],
                }
            )
            for method in methods:
                evaluation_rows.append(
                    {
                        "study_id": study_id,
                        "policy_id": leaf["policy_id"],
                        "task_id": leaf["task_id"],
                        "representation_id": representation,
                        "method_id": method,
                        "recommended": method in recommended,
                        "admissible": True,
                        "role": "policy_or_oracle_candidate",
                    }
                )
                key = (study_id, representation, method)
                record = execution_records.setdefault(
                    key,
                    {
                        "study_id": study_id,
                        "accession": study["accession"],
                        "representation_id": representation,
                        "input_track": leaf["input_track"],
                        "method_id": method,
                        "role": "admissible_method",
                        "tasks_using_output": set(),
                        "attempts_authorized": 0,
                        "execution_status": "not_authorized",
                    },
                )
                record["tasks_using_output"].add(leaf["task_id"])

    for study_id, study in studies.items():
        if "paired_rna_fidelity" not in study["evaluable_tasks"]:
            continue
        for representation, contract in representations.items():
            if contract["input_track"] == "F":
                continue
            for control in config["rna_evaluation_controls"]["matrix"]:
                key = (study_id, representation, control)
                execution_records[key] = {
                    "study_id": study_id,
                    "accession": study["accession"],
                    "representation_id": representation,
                    "input_track": contract["input_track"],
                    "method_id": control,
                    "role": "rna_evaluation_control_not_policy_admissible",
                    "tasks_using_output": {"paired_rna_fidelity"},
                    "attempts_authorized": 0,
                    "execution_status": "not_authorized",
                }

    execution_rows = []
    for record in execution_records.values():
        row = dict(record)
        row["tasks_using_output"] = ";".join(sorted(row["tasks_using_output"]))
        execution_rows.append(row)
    leaf_rows.sort(key=lambda row: (row["study_id"], row["policy_id"]))
    evaluation_rows.sort(key=lambda row: (row["study_id"], row["policy_id"], row["method_id"]))
    comparator_rows.sort(key=lambda row: (row["study_id"], row["policy_id"]))
    execution_rows.sort(key=lambda row: (row["study_id"], row["representation_id"], row["method_id"]))

    if len(leaf_rows) != 15 or len(evaluation_rows) != 69 or len(execution_rows) != 50 or len(comparator_rows) != 15:
        raise ValueError(
            f"G5-H4 manifest dimensions mismatch: {len(leaf_rows)}, {len(evaluation_rows)}, "
            f"{len(execution_rows)}, {len(comparator_rows)}"
        )
    output = root / config["output_root"]
    leaf_path = output / "external_validation_leaf_manifest.tsv"
    evaluation_path = output / "evaluation_method_manifest.tsv"
    execution_path = output / "method_execution_manifest.tsv"
    comparator_path = output / "comparator_manifest.tsv"
    _write_tsv(leaf_path, leaf_rows, list(leaf_rows[0]))
    _write_tsv(evaluation_path, evaluation_rows, list(evaluation_rows[0]))
    _write_tsv(execution_path, execution_rows, list(execution_rows[0]))
    _write_tsv(comparator_path, comparator_rows, list(comparator_rows[0]))
    artifacts = {
        "leaf_manifest": leaf_path,
        "evaluation_manifest": evaluation_path,
        "execution_manifest": execution_path,
        "comparator_manifest": comparator_path,
    }
    evidence_path = output / "completion_evidence.json"
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G5",
            "milestone": "H4_one_pass_external_validation_protocol_freeze",
            "status": "pass",
            "studies": 2,
            "active_policy_leaves": 10,
            "study_leaf_units": len(leaf_rows),
            "portability_study_leaf_units": sum(row["task_id"] == "portability_runtime_failure" for row in leaf_rows),
            "paired_rna_study_leaf_units": sum(row["task_id"] == "paired_rna_fidelity" for row in leaf_rows),
            "admissible_evaluation_rows": len(evaluation_rows),
            "planned_method_and_control_outputs": len(execution_rows),
            "planned_outputs_authorized": 0,
            "comparator_rows": len(comparator_rows),
            "two_study_external_portability_possible": True,
            "two_study_external_paired_rna_possible": False,
            "random_comparator_draws": 10000,
            "random_seed": 41017,
            "regret_noninferiority_margin": 0.05,
            "failure_rate_noninferiority_margin": 0.10,
            "cross_task_pooling": False,
            "cross_track_pooling": False,
            "cross_representation_pooling": False,
            "policy_refit_allowed": False,
            "raw_holdout_accessed": False,
            "representation_construction_run": False,
            "method_execution_run": False,
            "rna_value_access_run": False,
            "holdout_outcome_accessed": False,
            "regret_computation_run": False,
            "full_benchmark_run": False,
            "results_writing_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "g5_execution_authorized": False,
            "artifacts": {
                key: {"path": str(path.relative_to(root)), "sha256": sha256(path), "rows": rows}
                for (key, path), rows in zip(artifacts.items(), (15, 69, 50, 15), strict=True)
            },
            "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
            "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
            "frozen_policy": {"path": config["frozen_policy"], "sha256": config["frozen_policy_sha256"]},
            "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        },
    )
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g5_h4_one_pass_external_validation_protocol.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    output = run(args.config, args.root)
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
