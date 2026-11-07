"""Independent completion validator for NM-U3 external depth stress."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def _display(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def validate(evidence_path: str | Path, root: str | Path = ".") -> Path:
    root, evidence_path = Path(root).resolve(), Path(evidence_path).resolve()
    evidence = _mapping(evidence_path)
    config_path = root / evidence["config"]["path"]
    config = _mapping(config_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({
            "check_id": check_id,
            "observed": _display(observed),
            "expected": _display(expected),
            "pass": bool(observed == expected),
        })

    check("config_milestone", config.get("milestone"), "NU3_external_controlled_depth_stress")
    check("completion_status", evidence.get("status"), "pass")
    check("config_checksum", _sha256(config_path), evidence["config"]["sha256"])
    for name in ("authorization", "implementation", "metrics", "summary"):
        record = evidence[name]
        check(f"artifact_checksum__{name}", _sha256(root / record["path"]), record["sha256"])

    authorization = _mapping(root / evidence["authorization"]["path"])
    check("authorization_status", authorization.get("status"), "authorized")
    check("authorization_upgrade_milestone", authorization.get("upgrade_milestone"), config["milestone"])
    check("authorization_new_contracts", authorization.get("new_reduced_depth_contracts_authorized"), int(config["new_reduced_depth_contracts"]))
    setup_record = authorization["full_anchor_setup"]
    check("full_anchor_setup_checksum", _sha256(root / setup_record["path"]), setup_record["sha256"])
    setup = _mapping(root / setup_record["path"])
    check("full_anchor_setup_status", setup.get("status"), "pass")
    check("full_anchor_count", setup.get("anchors"), int(config["reused_full_depth_contracts"]))
    check("full_anchor_new_executions", setup.get("new_scientific_executions"), 0)

    metrics = pd.read_csv(root / evidence["metrics"]["path"], sep="\t")
    expected_conditions = int(config["planned_conditions"])
    expected_contracts = int(config["planned_contracts"])
    expected_per_condition = int(config["planned_contracts_per_condition"])
    check("contract_rows", len(metrics), expected_contracts)
    check("condition_count", metrics.condition_id.nunique(), expected_conditions)
    check("contracts_per_condition", bool(metrics.groupby("condition_id").size().eq(expected_per_condition).all()), True)
    check("unique_contract_keys", int(metrics.duplicated(["condition_id", "representation_id", "method_id"]).sum()), 0)
    check("study_panel", set(metrics.study_id), {study["study_id"] for study in config["studies"].values()})
    check("species_panel", set(metrics.species), {study["species"] for study in config["studies"].values()})
    check("genome_panel", set(metrics.genome_build), {study["genome_build"] for study in config["studies"].values()})
    check("depth_panel", set(metrics.depth_fraction), set(config["depth_fractions"]))

    full = metrics.loc[metrics.depth_fraction.eq(1.0)]
    reduced = metrics.loc[metrics.depth_fraction.lt(1.0)]
    check("full_depth_seed", set(full.seed), {int(config["full_depth_seed"])})
    check("reused_full_depth_contracts", len(full), int(config["reused_full_depth_contracts"]))
    check("full_depth_all_success", bool(full.success.astype(bool).all()), True)
    check("new_reduced_depth_contracts", len(reduced), int(config["new_reduced_depth_contracts"]))
    reduced_seeds = {int(value) for value in config["thinning_seeds"]}
    check("reduced_depth_seeds", set(reduced.seed), reduced_seeds)
    reduced_replicates = reduced.groupby(["dataset_id", "depth_fraction", "method_id", "representation_id"]).size()
    check("reduced_depth_three_replicates", bool(reduced_replicates.eq(len(reduced_seeds)).all()), True)

    expected_pairs = {
        (method, representation)
        for representation in config["matrix_representations"]
        for method in config["annotation_methods"]
    } | {(method, "filtered_fragments_depth_stress_contract") for method in config["software_methods"]}
    check("exact_contract_panel", set(zip(metrics.method_id, metrics.representation_id)), expected_pairs)

    success = metrics.success.astype(bool)
    required = [
        "common_genes", "median_spot_cosine_vs_full", "zero_output_spots", "output_nnz",
        "output_total_counts", "runtime_seconds", "peak_aggregate_pss_mb",
        "runtime_ratio_vs_full", "pss_ratio_vs_full",
    ]
    check("success_metric_columns_present", set(required).issubset(metrics.columns), True)
    if set(required).issubset(metrics.columns):
        check("successful_rows_have_metrics", bool(metrics.loc[success, required].notna().all().all()), True)
        check("runtime_positive", bool(metrics.loc[success, "runtime_seconds"].gt(0).all()), True)
        check("pss_positive", bool(metrics.loc[success, "peak_aggregate_pss_mb"].gt(0).all()), True)
        check("resource_ratios_positive", bool(metrics.loc[success, ["runtime_ratio_vs_full", "pss_ratio_vs_full"]].gt(0).all().all()), True)
        check("full_runtime_ratio_one", bool(np.allclose(full.runtime_ratio_vs_full, 1.0)), True)
        check("full_pss_ratio_one", bool(np.allclose(full.pss_ratio_vs_full, 1.0)), True)
        correlations = metrics.loc[success, "pseudobulk_gene_spearman_vs_full"]
        check("correlations_bounded_or_undefined", bool((correlations.between(-1, 1) | correlations.isna()).all()), True)
        check("cosines_bounded", bool(metrics.loc[success, "median_spot_cosine_vs_full"].between(0, 1).all()), True)
    check("success_accounting", int(success.sum()) + int((~success).sum()), expected_contracts)
    check("evidence_successes", int(evidence["successful_contracts"]), int(success.sum()))
    check("evidence_failures", int(evidence["failed_contracts"]), int((~success).sum()))

    summary = pd.read_csv(root / evidence["summary"]["path"], sep="\t")
    check("summary_rows", len(summary), len(expected_pairs) * len(config["depth_fractions"]))
    check("summary_unique_keys", int(summary.duplicated(["study_id", "method_id", "representation_id", "depth_fraction"]).sum()), 0)
    check("summary_full_planned_runs", bool(summary.loc[summary.depth_fraction.eq(1.0), "planned_runs"].eq(1).all()), True)
    check("summary_reduced_planned_runs", bool(summary.loc[summary.depth_fraction.lt(1.0), "planned_runs"].eq(3).all()), True)
    check("summary_success_rate_coherent", bool(np.allclose(summary.success_rate, summary.successful_runs / summary.planned_runs)), True)

    synthesis_path = evidence_path.parent / "synthesis_evidence.json"
    check("synthesis_present", synthesis_path.is_file(), True)
    if synthesis_path.is_file():
        synthesis = _mapping(synthesis_path)
        check("synthesis_status", synthesis.get("status"), "pass")
        check("synthesis_completion_checksum", synthesis["completion_evidence"]["sha256"], _sha256(evidence_path))
        check("synthesis_implementation_checksum", _sha256(root / synthesis["implementation"]["path"]), synthesis["implementation"]["sha256"])
        for name, record in synthesis["artifacts"].items():
            check(f"synthesis_artifact_checksum__{name}", _sha256(root / record["path"]), record["sha256"])
        qc = pd.read_csv(root / synthesis["artifacts"]["condition_construction_qc.tsv"]["path"], sep="\t")
        check("condition_qc_rows", len(qc), expected_conditions)
        check("condition_qc_unique", qc.condition_id.nunique(), expected_conditions)
        check("condition_qc_one_reused_anchor", int(qc.anchor_reused.astype(bool).sum()), 1)
        check("condition_qc_new_conditions", int(qc.new_scientific_execution.astype(bool).sum()), expected_conditions - 1)
        check("thinning_row_count_verified", bool(qc.included_fragment_rows.eq(qc.output_fragment_rows_verified).all()), True)
        check("thinning_status_all_pass", bool(qc.thinning_status.eq("pass").all()), True)
        check("representation_status_all_pass", bool(qc.representation_status.eq("pass").all()), True)
        check("fragment_contract_all_verified", bool(qc.fragment_filter_contract_verified.astype(bool).all()), True)
        failures = pd.read_csv(root / synthesis["artifacts"]["structured_failure_accounting.tsv"]["path"], sep="\t")
        check("structured_failure_rows", len(failures), int((~success).sum()))
        if len(failures):
            check("structured_failure_unique", int(failures.duplicated(["condition_id", "method_id", "representation_id"]).sum()), 0)
            check("structured_failure_status", bool(failures.failure_status.eq("fail").all()), True)
            check("structured_failure_types", bool(failures.failure_type.notna().all()), True)
            check("structured_failure_messages", bool(failures.message.notna().all()), True)
        failure_onset = pd.read_csv(root / synthesis["artifacts"]["failure_onset_by_contract.tsv"]["path"], sep="\t")
        check("failure_onset_rows", len(failure_onset), len(expected_pairs))
        check("failure_onset_full_success", bool(failure_onset.success_rate_depth_1.eq(1.0).all()), True)
        resources = pd.read_csv(root / synthesis["artifacts"]["descriptive_resource_summary.tsv"]["path"], sep="\t")
        check("resource_runtime_positive", bool(resources.median_runtime_seconds.gt(0).all()), True)
        check("resource_pss_positive", bool(resources.median_peak_aggregate_pss_mb.gt(0).all()), True)
        authorization = pd.read_csv(
            root / synthesis["artifacts"]["authorization_version_accounting.tsv"]["path"], sep="\t"
        )
        check("authorization_accounting_rows", len(authorization), expected_contracts)
        check(
            "authorization_accounting_unique",
            int(authorization.duplicated(["condition_id", "method_id", "representation_id"]).sum()),
            0,
        )
        v1 = authorization.loc[authorization.authorization_version.eq("V1")]
        check("v1_retained_contract_count", len(v1), int(synthesis["v1_retained_software_contracts"]))
        check(
            "v1_only_successful_software",
            bool(
                v1.success.astype(bool).all()
                and v1.representation_id.eq("filtered_fragments_depth_stress_contract").all()
            ),
            True,
        )
        reduced_annotation = authorization.loc[
            authorization.depth_fraction.lt(1.0)
            & ~authorization.representation_id.eq("filtered_fragments_depth_stress_contract")
        ]
        check("all_reduced_annotation_uses_v2", set(reduced_annotation.authorization_version), {"V2"})
        amendment = root / "governance/amendments/NM_U3_EXTERNAL_PROJECT_GUARD_CORRECTION_2026-07-30.yaml"
        check("framework_amendment_present", amendment.is_file(), True)
        for field in (
            "rna_accessed", "labels_accessed", "pathology_accessed", "policy_refit_run",
            "resource_ranking_run", "universal_ranking_run", "final_ranking_run",
        ):
            check(f"synthesis_boundary_false__{field}", synthesis.get(field), False)

    ledger_path = root / config["output_root"] / "execution_ledger.tsv"
    planned_path = root / config["output_root"] / "planned_conditions.tsv"
    check("execution_ledger_present", ledger_path.is_file(), True)
    check("planned_conditions_present", planned_path.is_file(), True)
    if ledger_path.is_file():
        ledger = pd.read_csv(ledger_path, sep="\t")
        stages = ledger.loc[ledger.stage.isin(["prepare_fragments", "call_peaks", "build_representations"])]
        contracts = ledger.loc[ledger.stage.isin(["annotation", "software"])].copy()
        check("execution_ledger_rows", len(ledger), expected_conditions * (3 + expected_per_condition))
        check("execution_stage_rows", len(stages), expected_conditions * 3)
        check("execution_stage_exit_codes", bool(stages.exit_code.eq(0).all()), True)
        check("execution_contract_rows", len(contracts), expected_contracts)
        check("execution_contract_keys_unique", int(contracts.duplicated(["condition_id", "method_id", "representation_id"]).sum()), 0)
        comparison = metrics[["condition_id", "method_id", "representation_id", "success"]].merge(
            contracts[["condition_id", "method_id", "representation_id", "exit_code"]],
            on=["condition_id", "method_id", "representation_id"], how="outer", validate="one_to_one", indicator=True,
        )
        check("execution_contract_key_match", bool(comparison._merge.eq("both").all()), True)
        check("execution_contract_status_match", bool(comparison.success.astype(bool).eq(comparison.exit_code.eq(0)).all()), True)
    if planned_path.is_file():
        planned = pd.read_csv(planned_path, sep="\t")
        check("planned_condition_rows", len(planned), expected_conditions)
        check("planned_condition_ids_unique", planned.condition_id.nunique(), expected_conditions)

    for field in (
        "rna_accessed", "labels_accessed", "pathology_accessed", "policy_refit_run",
        "universal_ranking_run", "final_ranking_run",
    ):
        check(f"boundary_false__{field}", evidence.get(field), False)

    table = pd.DataFrame(checks)
    output = evidence_path.parent / "evidence"
    output.mkdir(exist_ok=True)
    table_path = output / "validation_checks.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    passed = int(table["pass"].sum())
    audit = {
        "schema_version": 1,
        "milestone": "NU3_external_controlled_depth_stress",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "planned_contracts": expected_contracts,
        "reused_full_depth_contracts": int(config["reused_full_depth_contracts"]),
        "new_reduced_depth_contracts": int(config["new_reduced_depth_contracts"]),
        "successful_contracts": int(success.sum()),
        "failed_contracts": int((~success).sum()),
        "failures_retained_as_robustness_outcomes": True,
        "resource_ranking_allowed": False,
        "universal_ranking_allowed": False,
        "final_ranking_allowed": False,
        "validation_checks": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path)},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        failed = table.loc[~table["pass"], "check_id"].tolist()
        raise ValueError(f"NU3 independent validation failed: {failed}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="results/v1/nm_upgrade/nu3_external_depth_stress/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
