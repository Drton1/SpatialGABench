"""G5 pre-access feasibility audit without opening frozen holdout source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from spatialgabench.software_wrappers.common import sha256_file


FORBIDDEN_EXECUTION_FIELDS = (
    "raw_holdout_access",
    "holdout_mount",
    "holdout_checksum_revalidation",
    "representation_construction",
    "method_execution",
    "rna_value_access",
    "label_or_pathology_value_access",
    "regret_evaluation",
    "policy_refit",
    "full_benchmark_run",
    "results_writing",
    "universal_ranking",
    "final_ranking",
)
STANDARD_REPRESENTATIONS = {
    "F": "filtered_fragments_with_frozen_peak_observation_contract",
    "P": "macs3_summit_501bp",
    "B10": "fixed_bin_10kb",
    "B25": "fixed_bin_25kb",
    "B50": "fixed_bin_50kb",
}


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


def _validate_hash(root: Path, path_text: str, expected: str, label: str) -> Path:
    path = root / path_text
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"G5-H0 frozen metadata artifact mismatch: {label}")
    if "data/raw" in path.as_posix():
        raise PermissionError(f"G5-H0 cannot open a raw holdout path: {label}")
    return path


def _require_false(mapping: dict[str, Any], fields: Iterable[str], label: str) -> None:
    invalid = [field for field in fields if mapping.get(field) is not False]
    if invalid:
        raise PermissionError(f"{label} forbidden flags are not false: {invalid}")


def _load_contract(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_yaml(config_path)
    for key in ("authorization", "holdout_freeze", "dataset_registry", "m4_gate", "m4_policy"):
        _validate_hash(root, str(config[key]), str(config[f"{key}_sha256"]), key)
    authorization = _load_json(root / config["authorization"])
    m4_gate = _load_json(root / config["m4_gate"])
    policy = _load_json(root / config["m4_policy"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "H0_preaccess_feasibility_audit"
        or authorization.get("raw_holdout_file_access_authorized") is not False
        or authorization.get("g5_execution_authorized") is not False
    ):
        raise PermissionError("G5-H0 authorization is invalid")
    if (
        m4_gate.get("status") != "pass"
        or m4_gate.get("policy_frozen") is not True
        or m4_gate.get("g5_execution_authorized") is not False
        or m4_gate.get("holdout_accessed") is not False
    ):
        raise ValueError("M4 gate does not preserve the pre-G5 boundary")
    if (
        policy.get("status") != "frozen"
        or policy.get("holdout_accessed") is not False
        or policy.get("universal_ranking") is not False
    ):
        raise ValueError("M4 policy boundary is invalid")
    _require_false(config, FORBIDDEN_EXECUTION_FIELDS, "G5-H0 config")
    if config.get("exact_policy_match_required") is not True:
        raise ValueError("G5-H0 requires exact policy matching")
    if config.get("representation_alias_or_fallback_allowed") is not False:
        raise ValueError("G5-H0 representation fallback must be false")
    return config, authorization, policy


def _validate_holdout_metadata(config: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    registry = _load_yaml(root / config["dataset_registry"])
    studies = registry["studies"]
    slots = {
        slot["study_id"]: slot
        for slot in registry["holdout_slots"].values()
        if isinstance(slot, dict)
    }
    metadata: dict[str, dict[str, Any]] = {}
    for study_id, declared in config["holdouts"].items():
        if study_id not in studies or study_id not in slots:
            raise ValueError(f"G5-H0 holdout is not frozen in registry: {study_id}")
        study = studies[study_id]
        slot = slots[study_id]
        if (
            study.get("role", "").startswith("external_holdout") is False
            or slot.get("freeze_status") != "frozen"
            or slot.get("benchmark_outcomes_inspected") is not False
            or declared["accession"] not in study.get("accessions", [])
            or declared["selected_sample"] != slot.get("selected_sample")
        ):
            raise ValueError(f"G5-H0 registry freeze mismatch: {study_id}")
        audit_path = _validate_hash(
            root, declared["schema_audit"], declared["schema_audit_sha256"], f"{study_id}.schema_audit"
        )
        receipt_path = _validate_hash(
            root,
            declared["receipt_manifest"],
            declared["receipt_manifest_sha256"],
            f"{study_id}.receipt_manifest",
        )
        if declared.get("upstream_provenance"):
            _validate_hash(
                root,
                declared["upstream_provenance"],
                declared["upstream_provenance_sha256"],
                f"{study_id}.upstream_provenance",
            )
        audit = _load_json(audit_path)
        receipts = pd.read_csv(receipt_path, sep="\t")
        accession_rows = receipts[receipts["accession"].astype(str).eq(declared["accession"])]
        receipt_checks = ["size_status", "verification_status"]
        if (
            accession_rows.empty
            or any(column not in accession_rows.columns for column in receipt_checks)
            or not all(accession_rows[column].astype(str).eq("pass").all() for column in receipt_checks)
        ):
            raise ValueError(f"G5-H0 prefreeze receipts are incomplete: {study_id}")
        if audit.get("schema_status") != "pass" or audit.get("barcode_contract_status") != "pass":
            raise ValueError(f"G5-H0 prefreeze schema audit failed: {study_id}")
        observed_standard = audit.get("standard_paired_fragment_semantics_status") != "fail"
        if bool(declared["standard_fragment_level"]) != observed_standard:
            raise ValueError(f"G5-H0 source semantics mismatch: {study_id}")
        metadata[study_id] = {"registry": study, "slot": slot, "audit": audit, "receipt_rows": len(accession_rows)}
    return metadata


def _leaf_eligibility(
    study_id: str,
    declared: dict[str, Any],
    leaf: dict[str, Any],
) -> dict[str, Any]:
    support_species = {value for value in str(leaf.get("support_species", "")).split(";") if value}
    support_genomes = {value for value in str(leaf.get("support_genome_builds", "")).split(";") if value}
    species_supported = declared["species"] in support_species
    genome_supported = declared["genome_build"] in support_genomes
    standard_fragments = bool(declared["standard_fragment_level"])
    exact_representation_available = standard_fragments and leaf["representation_scope"] in set(
        STANDARD_REPRESENTATIONS.values()
    )
    task_input_available = not (
        leaf["task_id"] == "paired_rna_fidelity" and declared.get("paired_rna") is not True
    )
    eligible = species_supported and genome_supported and exact_representation_available and task_input_available
    reasons: list[str] = []
    if not species_supported or not genome_supported:
        reasons.append("species_or_genome_outside_leaf_development_support")
    if not exact_representation_available:
        reasons.append(
            "nonstandard_pseudo_pair_cannot_alias_standard_F_P_B"
            if not standard_fragments
            else "exact_representation_unavailable"
        )
    if not task_input_available:
        reasons.append("required_task_input_unavailable")
    return {
        "study_id": study_id,
        "accession": declared["accession"],
        "selected_sample": declared["selected_sample"],
        "species": declared["species"],
        "genome_build": declared["genome_build"],
        "source_semantics": declared["source_semantics"],
        "policy_id": leaf["policy_id"],
        "task_id": leaf["task_id"],
        "input_track": leaf["input_track"],
        "representation_scope": leaf["representation_scope"],
        "leaf_support_species": leaf.get("support_species", ""),
        "leaf_support_genome_builds": leaf.get("support_genome_builds", ""),
        "species_genome_supported": species_supported and genome_supported,
        "exact_representation_available": exact_representation_available,
        "task_input_available": task_input_available,
        "exact_active_leaf_evaluable": eligible,
        "reason": "eligible_exact_frozen_leaf" if eligible else ";".join(reasons),
        "representation_alias_used": False,
        "holdout_outcome_used": False,
    }


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config, authorization, policy = _load_contract(config_path, root)
    metadata = _validate_holdout_metadata(config, root)
    active_leaves = [leaf for leaf in policy["leaves"] if leaf.get("status") == "active_frozen"]
    if len(active_leaves) != 10:
        raise ValueError("G5-H0 expected exactly ten frozen active policy leaves")

    rows = [
        _leaf_eligibility(study_id, declared, leaf)
        for study_id, declared in config["holdouts"].items()
        for leaf in active_leaves
    ]
    eligibility = pd.DataFrame(rows).sort_values(["study_id", "task_id", "input_track", "representation_scope"])
    summary_rows = []
    for study_id, declared in config["holdouts"].items():
        frame = eligibility[eligibility.study_id.eq(study_id)]
        evaluable = frame[frame.exact_active_leaf_evaluable.eq(True)]
        summary_rows.append(
            {
                "study_id": study_id,
                "accession": declared["accession"],
                "selected_sample": declared["selected_sample"],
                "species": declared["species"],
                "genome_build": declared["genome_build"],
                "source_semantics": declared["source_semantics"],
                "standard_fragment_level": declared["standard_fragment_level"],
                "active_policy_leaves_total": len(frame),
                "exact_active_leaves_evaluable": len(evaluable),
                "evaluable_tasks": ";".join(sorted(evaluable.task_id.unique())),
                "prefreeze_receipt_rows": metadata[study_id]["receipt_rows"],
                "holdout_outcomes_inspected": False,
                "performance_validation_eligible": len(evaluable) > 0,
                "status": "evaluable" if len(evaluable) > 0 else "out_of_policy_support_abstention_only",
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("study_id")
    evaluable_studies = int(summary.performance_validation_eligible.sum())
    minimum = int(config["minimum_untouched_studies_for_c2"])
    c2_ready = evaluable_studies >= minimum
    decision = "authorize_one_pass_unseal" if c2_ready else "do_not_unseal_preserve_remaining_holdout"

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    eligibility_path = output / "holdout_policy_leaf_eligibility.tsv"
    summary_path = output / "holdout_feasibility_summary.tsv"
    decision_path = output / "g5_access_decision.json"
    eligibility.to_csv(eligibility_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    _write_json(
        decision_path,
        {
            "schema_version": 1,
            "gate": "G5",
            "milestone": "H0_preaccess_feasibility_audit",
            "decision": decision,
            "c2_external_validation_ready": c2_ready,
            "minimum_untouched_studies_required": minimum,
            "exact_leaf_evaluable_holdouts": evaluable_studies,
            "untouched_holdouts_preserved": 2,
            "recommended_resolution": "add_and_freeze_one_independent_standard_fragment_holdout_matching_active_policy_leaves_before_any_current_holdout_unseal",
            "allowed_alternative": "downgrade_C2_to_single_study_external_validation_plus_out_of_support_abstention_demonstration",
            "forbidden_resolution": "relabel_nonstandard_pseudo_pairs_as_standard_fragments_or_alias_insertion_derived_matrices_to_frozen_F_P_B",
            "raw_holdout_accessed": False,
            "holdout_outcomes_inspected": False,
            "g5_execution_authorized": False,
        },
    )
    artifacts = [eligibility_path, summary_path, decision_path]
    evidence_path = output / "completion_evidence.json"
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G5",
            "milestone": "H0_preaccess_feasibility_audit",
            "status": "blocked",
            "audit_completed": True,
            "access_decision": decision,
            "active_policy_leaves_audited_per_holdout": len(active_leaves),
            "holdouts_audited": len(config["holdouts"]),
            "exact_leaf_evaluable_holdouts": evaluable_studies,
            "minimum_untouched_studies_required": minimum,
            "c2_external_validation_ready": c2_ready,
            "artifacts": {
                path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts
            },
            "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
            "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_file(config_path)},
            "m4_policy": {"path": config["m4_policy"], "sha256": config["m4_policy_sha256"]},
            "authorization_status": authorization["status"],
            "raw_holdout_paths_opened": 0,
            "raw_holdout_accessed": False,
            "holdout_mount_run": False,
            "holdout_checksum_revalidation_run": False,
            "representation_construction_run": False,
            "method_execution_run": False,
            "rna_value_access_run": False,
            "label_or_pathology_value_access_run": False,
            "regret_evaluation_run": False,
            "policy_refit_run": False,
            "full_benchmark_run": False,
            "results_writing_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "g5_execution_authorized": False,
            "next_milestone_authorized": False,
        },
    )
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g5_h0_preaccess_feasibility.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
