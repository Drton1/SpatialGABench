"""Independent audit for the locked NU1/NU2 human construction panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spatialgabench.nm_human_method_setup import ANNOTATION_METHODS, NU1, NU2, REPRESENTATIONS, SOFTWARE_METHODS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _display(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def validate(evidence_path: str | Path, root: str | Path = ".") -> Path:
    root, evidence_path = Path(root).resolve(), Path(evidence_path).resolve()
    evidence = _json(evidence_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({"check_id": check_id, "observed": _display(observed), "expected": _display(expected), "pass": bool(observed == expected)})

    check("completion_status", evidence.get("status"), "pass")
    check("planned_contracts", evidence.get("contracts_planned"), 50)
    check("accounted_contracts", evidence.get("contracts_accounted"), 50)
    check("retry_attempts", evidence.get("retry_attempts"), 0)
    for name in ("setup", "plan", "execution_ledger", "run_summary", "failure_accounting", "implementation"):
        record = evidence[name]
        check(f"artifact_checksum__{name}", _sha256(root / record["path"]), record["sha256"])

    setup = _json(root / evidence["setup"]["path"])
    check("setup_status", setup.get("status"), "pass")
    check("setup_outputs", setup.get("planned_outputs"), 50)
    for name in ("authorization", "annotation_config", "software_config", "representation_summary", "representation_completion"):
        record = setup[name]
        check(f"setup_binding__{name}", _sha256(root / record["path"]), record["sha256"])

    authorization = _json(root / setup["authorization"]["path"])
    check("authorization_status", authorization.get("status"), "authorized")
    check("authorization_upgrade_milestone", authorization.get("upgrade_milestone"), "NU1_NU2_human_method_construction")
    check("authorization_studies", set(authorization.get("authorized_studies", [])), {"zhang_2023", "guo_deng_2025_ffpe_atac"})
    check("authorization_methods", set(authorization.get("authorized_methods", [])), set(ANNOTATION_METHODS + SOFTWARE_METHODS))
    for field in (
        "evaluation_tasks_authorized", "rna_value_access_authorized",
        "label_or_pathology_value_access_authorized", "holdout_utility_access_authorized",
        "regret_computation_authorized", "policy_refit_authorized",
        "full_benchmark_authorized", "results_writing_authorized",
        "universal_ranking_authorized", "final_ranking_authorized",
    ):
        check(f"authorization_boundary_false__{field}", authorization.get(field), False)

    plan = pd.read_csv(root / evidence["plan"]["path"], sep="\t")
    keys = ["dataset_id", "representation_id", "method_id"]
    check("plan_rows", len(plan), 50)
    check("plan_unique_outputs", int(plan.output_dir.duplicated().sum()), 0)
    check("plan_unique_contracts", int(plan.duplicated(keys).sum()), 0)
    check("plan_annotation_count", int(plan.run_type.eq("annotation").sum()), 44)
    check("plan_software_count", int(plan.run_type.eq("software").sum()), 6)
    check("plan_nu1_count", int(plan.dataset_id.eq(NU1).sum()), 27)
    check("plan_nu2_count", int(plan.dataset_id.eq(NU2).sum()), 23)
    check("plan_human_only", set(plan.species), {"human"})
    check("plan_hg38_only", set(plan.genome_build), {"hg38"})
    check("plan_representations", set(plan.loc[plan.run_type.eq("annotation"), "representation_id"]), set(REPRESENTATIONS))
    controls = plan.loc[plan.method_id.eq("accessibility_matched_random_peak_to_gene_control_pc")]
    check("control_nu1_only", set(controls.dataset_id), {NU1})
    check("control_all_representations", set(controls.representation_id), set(REPRESENTATIONS))
    check("one_primary_attempt_authorized", set(plan.attempts_authorized), {1})

    ledger = pd.read_csv(root / evidence["execution_ledger"]["path"], sep="\t")
    check("ledger_rows", len(ledger), 50)
    check("ledger_unique_contracts", int(ledger.duplicated(keys).sum()), 0)
    check("ledger_attempt_one", set(ledger.attempt_index), {1})
    check("ledger_no_retries", set(ledger.retry_attempts), {0})
    joined = plan[keys].merge(ledger[keys], on=keys, how="outer", validate="one_to_one", indicator=True)
    check("ledger_exact_plan", bool(joined._merge.eq("both").all()), True)

    summary = pd.read_csv(root / evidence["run_summary"]["path"], sep="\t")
    check("summary_rows", len(summary), 50)
    check("summary_unique_contracts", int(summary.duplicated(keys).sum()), 0)
    check("summary_all_accounted", bool(summary.status.eq("accounted").all()), True)
    check("summary_outcomes_known", set(summary.outcome).issubset({"success", "structured_failure"}), True)
    check("summary_success_count", int(summary.outcome.eq("success").sum()), int(evidence["successful_contracts"]))
    check("summary_failure_count", int(summary.outcome.eq("structured_failure").sum()), int(evidence["structured_failures"]))
    success = summary.loc[summary.outcome.eq("success")]
    for column in (
        "observation_ids_and_order_match", "manifest_artifact_hashes_valid",
        "live_input_hashes_valid", "construction_boundary_valid", "construction_leakage_absent",
    ):
        check(f"success_flag_true__{column}", bool(success[column].astype(bool).all()), True)
    check("success_contract_checks", bool(success.contract_checks_passed.eq(18).all()), True)
    check("success_runtime_positive", bool(success.runtime_seconds_noncomparative.gt(0).all()), True)
    check("success_pss_positive", bool(success.peak_aggregate_pss_mb_noncomparative.gt(0).all()), True)

    failures = pd.read_csv(root / evidence["failure_accounting"]["path"], sep="\t")
    check("failure_rows", len(failures), int(evidence["structured_failures"]))
    if len(failures):
        check("failure_unique_contracts", int(failures.duplicated(keys).sum()), 0)
        check("failure_status", bool(failures.status.eq("fail").all()), True)
        check("failure_stage_present", bool(failures.failure_stage.notna().all()), True)
        check("failure_type_present", bool(failures.failure_type.notna().all()), True)
        check("failure_message_present", bool(failures.message.notna().all()), True)

    ledger_outcomes = ledger[keys + ["exit_code"]].merge(summary[keys + ["outcome"]], on=keys, validate="one_to_one")
    check("ledger_success_matches_summary", bool(ledger_outcomes.exit_code.eq(0).eq(ledger_outcomes.outcome.eq("success")).all()), True)
    for field in (
        "rna_accessed", "labels_pathology_images_accessed", "evaluation_run",
        "policy_refit_run", "resource_ranking_run", "universal_ranking_run", "final_ranking_run",
    ):
        check(f"completion_boundary_false__{field}", evidence.get(field), False)

    table = pd.DataFrame(checks)
    output = evidence_path.parent
    table_path = output / "validation_checks.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    passed = int(table["pass"].sum())
    audit = {
        "schema_version": 1,
        "milestone": "NU1_NU2_human_method_construction",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "successful_contracts": int(evidence["successful_contracts"]),
        "structured_failures": int(evidence["structured_failures"]),
        "failures_retained_without_retry": True,
        "rna_evaluation_authorized": False,
        "universal_ranking_allowed": False,
        "final_ranking_allowed": False,
        "validation_checks": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path)},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        failed = table.loc[~table["pass"], "check_id"].tolist()
        raise ValueError(f"NU1/NU2 human method validation failed: {failed}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="results/v1/nm_upgrade/nu1_nu2_human_methods/evidence/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
