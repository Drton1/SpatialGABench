"""Independent completion validator for FB3 depth and resource stress tests."""

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
    root = Path(root).resolve()
    evidence_path = Path(evidence_path).resolve()
    evidence = _mapping(evidence_path)
    config_record = evidence["config"]
    config_path = root / config_record["path"]
    config = _mapping(config_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({
            "check_id": check_id,
            "observed": _display(observed),
            "expected": _display(expected),
            "pass": bool(observed == expected),
        })

    check("evidence_status", evidence.get("status"), "pass")
    check("config_checksum", _sha256(config_path), config_record["sha256"])
    for name in ("authorization", "implementation", "metrics", "summary"):
        record = evidence[name]
        path = root / record["path"]
        check(f"artifact_checksum__{name}", _sha256(path), record["sha256"])

    metrics = pd.read_csv(root / evidence["metrics"]["path"], sep="\t")
    expected_conditions = int(config["planned_conditions"])
    expected_contracts = int(config["planned_contracts"])
    expected_per_condition = int(config["planned_contracts_per_condition"])
    check("contract_rows", len(metrics), expected_contracts)
    check("condition_count", metrics.condition_id.nunique(), expected_conditions)
    check("contracts_per_condition", bool(metrics.groupby("condition_id").size().eq(expected_per_condition).all()), True)
    check(
        "unique_contract_keys",
        int(metrics.duplicated(["condition_id", "representation_id", "method_id"]).sum()),
        0,
    )
    check("study_count", metrics.study_id.nunique(), len(config["studies"]))
    check("species_panel", set(metrics.species), {"mouse", "human"})
    check("genome_panel", set(metrics.genome_build), {"mm10", "hg38"})
    check("depth_panel", set(metrics.depth_fraction), set(config["depth_fractions"]))

    full = metrics.loc[metrics.depth_fraction.eq(1.0)]
    reduced = metrics.loc[metrics.depth_fraction.lt(1.0)]
    check("full_depth_seed", set(full.seed), {int(config["full_depth_seed"])})
    check("full_depth_contracts", len(full), len(config["studies"]) * expected_per_condition)
    check("full_depth_all_success", bool(full.success.astype(bool).all()), True)
    expected_reduced_seeds = set(int(value) for value in config["thinning_seeds"])
    check("reduced_depth_seeds", set(reduced.seed), expected_reduced_seeds)
    reduced_replicates = reduced.groupby(["dataset_id", "depth_fraction", "method_id", "representation_id"]).size()
    check("reduced_depth_three_replicates", bool(reduced_replicates.eq(len(expected_reduced_seeds)).all()), True)

    expected_pairs = {
        (method, representation)
        for representation in config["matrix_representations"]
        for method in config["annotation_methods"]
    } | {
        (method, "filtered_fragments_depth_stress_contract") for method in config["software_methods"]
    }
    check("exact_contract_panel", set(zip(metrics.method_id, metrics.representation_id)), expected_pairs)

    successes = metrics.success.astype(bool)
    required_success_metrics = [
        "common_genes", "median_spot_cosine_vs_full", "zero_output_spots", "output_nnz",
        "output_total_counts", "runtime_seconds",
        "peak_aggregate_pss_mb", "runtime_ratio_vs_full", "pss_ratio_vs_full",
    ]
    check("success_metric_columns_present", set(required_success_metrics).issubset(metrics.columns), True)
    if set(required_success_metrics).issubset(metrics.columns):
        check("successful_rows_have_metrics", bool(metrics.loc[successes, required_success_metrics].notna().all().all()), True)
        check("runtime_positive", bool(metrics.loc[successes, "runtime_seconds"].gt(0).all()), True)
        check("pss_positive", bool(metrics.loc[successes, "peak_aggregate_pss_mb"].gt(0).all()), True)
        check("resource_ratios_positive", bool(metrics.loc[successes, ["runtime_ratio_vs_full", "pss_ratio_vs_full"]].gt(0).all().all()), True)
        check("full_depth_runtime_ratio_one", bool(np.allclose(full.runtime_ratio_vs_full, 1.0)), True)
        check("full_depth_pss_ratio_one", bool(np.allclose(full.pss_ratio_vs_full, 1.0)), True)
        correlations = metrics.loc[successes, "pseudobulk_gene_spearman_vs_full"]
        correlation_valid = correlations.between(-1, 1) | correlations.isna()
        check("correlations_bounded_or_undefined", bool(correlation_valid.all()), True)
        undefined = metrics.loc[successes & metrics.pseudobulk_gene_spearman_vs_full.isna()]
        check(
            "undefined_correlations_are_retained_not_failures",
            bool(undefined.success.astype(bool).all()),
            True,
        )
        check("cosines_bounded", bool(metrics.loc[successes, "median_spot_cosine_vs_full"].between(0, 1).all()), True)

    check("success_accounting", int(successes.sum()) + int((~successes).sum()), expected_contracts)
    check("evidence_successes", int(evidence["successful_contracts"]), int(successes.sum()))
    check("evidence_failures", int(evidence["failed_contracts"]), int((~successes).sum()))

    peak_representation = config["matrix_representations"][0]
    bin_representation = config["matrix_representations"][1]
    peak_failures = metrics.loc[(metrics.representation_id == peak_representation) & ~successes]
    peak_failure_conditions = set(peak_failures.condition_id)
    branch_rows = metrics.loc[
        metrics.condition_id.isin(peak_failure_conditions)
        & metrics.representation_id.eq(bin_representation)
        & metrics.method_id.isin(config["annotation_methods"])
    ]
    check(
        "peak_failure_kept_branch_local",
        len(branch_rows),
        len(peak_failure_conditions) * len(config["annotation_methods"]),
    )

    summary = pd.read_csv(root / evidence["summary"]["path"], sep="\t")
    expected_summary_rows = len(config["studies"]) * len(expected_pairs) * len(config["depth_fractions"])
    check("summary_rows", len(summary), expected_summary_rows)
    check("summary_unique_keys", int(summary.duplicated(["study_id", "method_id", "representation_id", "depth_fraction"]).sum()), 0)
    check("summary_full_planned_runs", bool(summary.loc[summary.depth_fraction.eq(1.0), "planned_runs"].eq(1).all()), True)
    check("summary_reduced_planned_runs", bool(summary.loc[summary.depth_fraction.lt(1.0), "planned_runs"].eq(3).all()), True)
    check(
        "summary_success_rate_coherent",
        bool(np.allclose(summary.success_rate, summary.successful_runs / summary.planned_runs)),
        True,
    )

    synthesis_path = evidence_path.parent / "synthesis_evidence.json"
    check("synthesis_evidence_present", synthesis_path.is_file(), True)
    if synthesis_path.is_file():
        synthesis = _mapping(synthesis_path)
        check("synthesis_status", synthesis.get("status"), "pass")
        check("synthesis_completion_checksum", synthesis["completion_evidence"]["sha256"], _sha256(evidence_path))
        check(
            "synthesis_implementation_checksum",
            _sha256(root / synthesis["implementation"]["path"]),
            synthesis["implementation"]["sha256"],
        )
        for name, record in synthesis["artifacts"].items():
            check(f"synthesis_artifact_checksum__{name}", _sha256(root / record["path"]), record["sha256"])

        qc_record = synthesis["artifacts"]["condition_construction_qc.tsv"]
        condition_qc = pd.read_csv(root / qc_record["path"], sep="\t")
        check("condition_qc_rows", len(condition_qc), expected_conditions)
        check("condition_qc_unique", condition_qc.condition_id.nunique(), expected_conditions)
        check(
            "thinning_row_count_verified",
            bool(condition_qc.included_fragment_rows.eq(condition_qc.output_fragment_rows_verified).all()),
            True,
        )
        check("thinning_status_all_pass", bool(condition_qc.thinning_status.eq("pass").all()), True)
        check("representation_status_all_pass", bool(condition_qc.representation_status.eq("pass").all()), True)
        check("fragment_contract_all_verified", bool(condition_qc.fragment_filter_contract_verified.astype(bool).all()), True)
        unavailable = condition_qc.loc[~condition_qc.peak_representation_available.astype(bool)]
        check(
            "unavailable_peaks_do_not_block_other_branches",
            bool(unavailable.fixed_bin_and_fragment_branches_may_continue.astype(bool).all()),
            True,
        )
        check("synthesis_unavailable_peak_count", int(synthesis["conditions_without_peak_representation"]), len(unavailable))

        failure_record = synthesis["artifacts"]["failure_onset_by_contract.tsv"]
        failure = pd.read_csv(root / failure_record["path"], sep="\t")
        expected_families = len(config["studies"]) * len(expected_pairs)
        check("failure_onset_rows", len(failure), expected_families)
        check(
            "failure_onset_unique",
            int(failure.duplicated(["dataset_id", "method_id", "representation_id"]).sum()),
            0,
        )
        check("failure_onset_full_depth_success", bool(failure.success_rate_depth_1.eq(1.0).all()), True)
        expected_failure_families = int(
            metrics.groupby(["dataset_id", "method_id", "representation_id"])["success"]
            .apply(lambda values: not values.astype(bool).all())
            .sum()
        )
        check("failure_family_accounting", int(failure.failure_observed.astype(bool).sum()), expected_failure_families)
        check("synthesis_failure_family_accounting", int(synthesis["contract_families_with_any_failure"]), expected_failure_families)

        structured_record = synthesis["artifacts"]["structured_failure_accounting.tsv"]
        structured = pd.read_csv(root / structured_record["path"], sep="\t")
        check("structured_failure_rows", len(structured), int((~successes).sum()))
        check(
            "structured_failure_unique",
            int(structured.duplicated(["condition_id", "method_id", "representation_id"]).sum()),
            0,
        )
        check("structured_failure_status", bool(structured.failure_status.eq("fail").all()), True)
        check("structured_failure_types_present", bool(structured.failure_type.notna().all()), True)
        check("structured_failure_messages_present", bool(structured.message.notna().all()), True)
        check("synthesis_structured_failure_count", int(synthesis["structured_failure_records"]), len(structured))

        resource_record = synthesis["artifacts"]["descriptive_resource_summary.tsv"]
        resource = pd.read_csv(root / resource_record["path"], sep="\t")
        successful_groups = metrics.loc[successes].groupby(
            ["study_id", "method_id", "representation_id", "depth_fraction"]
        ).ngroups
        check("resource_summary_rows", len(resource), successful_groups)
        check("resource_summary_runtime_positive", bool(resource.median_runtime_seconds.gt(0).all()), True)
        check("resource_summary_pss_positive", bool(resource.median_peak_aggregate_pss_mb.gt(0).all()), True)
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
        stage_rows = ledger.loc[ledger.stage.isin(["prepare_fragments", "call_peaks", "build_representations"])]
        contract_rows = ledger.loc[ledger.stage.isin(["annotation", "software"])].copy()
        expected_ledger_rows = expected_conditions * (3 + expected_per_condition)
        check("execution_ledger_rows", len(ledger), expected_ledger_rows)
        check("execution_ledger_condition_count", ledger.condition_id.nunique(), expected_conditions)
        check("execution_stage_rows", len(stage_rows), expected_conditions * 3)
        check("execution_stage_exit_codes", bool(stage_rows.exit_code.eq(0).all()), True)
        check("execution_contract_rows", len(contract_rows), expected_contracts)
        check(
            "execution_contract_keys_unique",
            int(contract_rows.duplicated(["condition_id", "method_id", "representation_id"]).sum()),
            0,
        )
        ledger_success = contract_rows[
            ["condition_id", "method_id", "representation_id", "exit_code"]
        ].copy()
        ledger_success["ledger_success"] = ledger_success.exit_code.eq(0)
        metric_success = metrics[
            ["condition_id", "method_id", "representation_id", "success"]
        ].copy()
        execution_comparison = metric_success.merge(
            ledger_success,
            on=["condition_id", "method_id", "representation_id"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        check("execution_contract_key_match", bool(execution_comparison._merge.eq("both").all()), True)
        check(
            "execution_contract_status_match",
            bool(
                execution_comparison._merge.eq("both").all()
                and execution_comparison.success.astype(bool).eq(
                    execution_comparison.ledger_success.astype(bool)
                ).all()
            ),
            True,
        )
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
        "milestone": "FB3_controlled_depth_resource_stress",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "planned_contracts": expected_contracts,
        "successful_contracts": int(successes.sum()),
        "failed_contracts": int((~successes).sum()),
        "failures_retained_as_robustness_outcomes": True,
        "resource_ranking_allowed": False,
        "universal_ranking_allowed": False,
        "final_ranking_allowed": False,
        "validation_checks": {
            "path": str(table_path.relative_to(root)),
            "sha256": _sha256(table_path),
        },
        "completion_evidence": {
            "path": str(evidence_path.relative_to(root)),
            "sha256": _sha256(evidence_path),
        },
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        failed = table.loc[~table["pass"], "check_id"].tolist()
        raise ValueError(f"FB3 independent validation failed: {failed}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        default="results/v1/full_benchmark/fb3_depth_resource_stress/completion_evidence.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
