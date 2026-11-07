"""Independent completion audit for G5-H5C external evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY = "coverage_adjusted_median_gene_wise_spearman"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def summarize(
    completion_path: str | Path,
    memory_path: str | Path,
    root: str | Path = ".",
) -> dict[str, Path]:
    root = Path(root).resolve()
    completion_path = Path(completion_path).resolve()
    memory_path = Path(memory_path).resolve()
    completion = _json(completion_path)
    memory = _json(memory_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    check(
        "completion_dimensions",
        completion.get("status") == "pass"
        and completion.get("rna_method_and_control_outputs_expected") == 27
        and completion.get("rna_method_and_control_outputs_passed") == 27
        and completion.get("rna_method_evaluation_failures") == 0
        and completion.get("admissible_evaluation_rows") == 69
        and completion.get("study_leaf_units") == 15
        and completion.get("random_comparator_rows") == 150000
        and completion.get("rna_observations") == 7122
        and completion.get("rna_eligible_genes") == 6074,
        (
            completion.get("rna_method_and_control_outputs_passed"),
            completion.get("admissible_evaluation_rows"),
            completion.get("study_leaf_units"),
        ),
    )
    check(
        "runtime_evidence",
        memory.get("status") == "pass"
        and memory.get("exit_code") == 0
        and float(memory.get("runtime_seconds", 0)) > 0
        and float(memory.get("peak_aggregate_pss_mb", 0)) > 0,
        (memory.get("status"), memory.get("exit_code"), memory.get("runtime_seconds")),
    )
    artifacts_valid = True
    rows_valid = True
    frames: dict[str, pd.DataFrame] = {}
    for name, record in completion.get("artifacts", {}).items():
        path = root / record["path"]
        artifacts_valid = artifacts_valid and path.is_file() and sha256(path) == record["sha256"]
        if name.endswith(".tsv") or name.endswith(".tsv.gz"):
            frame = pd.read_csv(path, sep="\t")
            frames[name] = frame
            rows_valid = rows_valid and len(frame) == int(record["rows"])
    check("live_artifact_hashes", artifacts_valid, len(completion.get("artifacts", {})))
    check("artifact_row_counts", rows_valid, {key: len(value) for key, value in frames.items()})

    metrics = frames["rna_metrics.tsv"]
    failures = frames["rna_evaluation_failures.tsv"]
    utilities = frames["method_utility.tsv"]
    leaves = frames["leaf_results.tsv"]
    tasks = frames["task_summary.tsv"]
    controls = frames["rna_control_deltas.tsv"]
    random = frames["random_comparator_draws.tsv.gz"]
    gene_rows = frames["gene_level_correlations.tsv.gz"]
    check(
        "rna_primary_formula",
        len(metrics) == 27
        and len(failures) == 0
        and np.isfinite(metrics[PRIMARY]).all()
        and np.allclose(
            metrics[PRIMARY],
            metrics["median_gene_wise_spearman"] * metrics["evaluable_gene_fraction"],
            rtol=1e-12,
            atol=1e-12,
        )
        and len(gene_rows) == 27 * 6074,
        (len(metrics), len(failures), len(gene_rows)),
    )
    rna_utility = utilities[utilities.task_id == "paired_rna_fidelity"]
    portability = utilities[utilities.task_id == "portability_runtime_failure"]
    expected_rna_utility = np.clip((rna_utility["primary_endpoint_value"] + 1) / 2, 0, 1)
    check(
        "method_utility_contract",
        len(utilities) == 69
        and len(rna_utility) == 23
        and len(portability) == 46
        and np.allclose(rna_utility["utility"], expected_rna_utility, rtol=1e-12, atol=1e-12)
        and (portability["utility"] == 1.0).all()
        and (utilities["evaluation_status"] == "pass").all(),
        (len(rna_utility), len(portability)),
    )

    leaf_valid = len(leaves) == 15
    for row in leaves.itertuples(index=False):
        frame = utilities[(utilities.study_id == row.study_id) & (utilities.policy_id == row.policy_id)]
        successful = frame[frame.evaluation_status == "pass"]
        recommended = successful[successful.recommended]
        oracle = float(successful.utility.max())
        policy = float(recommended.utility.max())
        default = frame[frame.method_id == row.default_method]
        leaf_valid = leaf_valid and (
            len(default) == 1
            and math.isclose(float(row.oracle_utility), oracle, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(float(row.policy_set_utility), policy, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(float(row.policy_regret), oracle - policy, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(
                float(row.default_regret),
                oracle - float(default.iloc[0].utility),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and not bool(row.policy_set_failure)
        )
    check("leaf_regret_recalculation", leaf_valid, len(leaves))

    random_valid = len(random) == 150000
    for key, frame in random.groupby(["study_id", "policy_id"], sort=True):
        leaf = leaves[(leaves.study_id == key[0]) & (leaves.policy_id == key[1])].iloc[0]
        random_valid = random_valid and (
            len(frame) == 10000
            and math.isclose(float(leaf.random_mean_regret), frame.regret.mean(), rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(float(leaf.random_failure_rate), frame.set_failure.mean(), rel_tol=1e-12, abs_tol=1e-12)
        )
    check("random_comparator_recalculation", random_valid, len(random))

    portability_leaves = leaves[leaves.task_id == "portability_runtime_failure"]
    rna_leaves = leaves[leaves.task_id == "paired_rna_fidelity"]
    check(
        "two_study_portability_ceiling",
        len(portability_leaves) == 10
        and portability_leaves.policy_id.nunique() == 5
        and (portability_leaves.groupby("policy_id").study_id.nunique() == 2).all()
        and (portability_leaves.policy_regret == 0).all()
        and (portability_leaves.default_regret == 0).all()
        and (portability_leaves.random_mean_regret == 0).all()
        and (~portability_leaves.policy_set_failure).all(),
        len(portability_leaves),
    )
    check(
        "single_study_rna_policy_regret",
        len(rna_leaves) == 5
        and (rna_leaves.policy_regret == 0).all()
        and (rna_leaves.default_regret > 0).all()
        and (rna_leaves.random_mean_regret > 0).all(),
        rna_leaves[["representation_id", "policy_regret"]].to_dict("records"),
    )
    check(
        "matched_control_direction",
        len(controls) == 20
        and (controls["delta_vs_accessibility_matched_control"] > 0).all(),
        (
            len(controls),
            float(controls["delta_vs_accessibility_matched_control"].min()),
            float(controls["delta_vs_accessibility_matched_control"].max()),
        ),
    )
    noninferiority_columns = [
        "regret_noninferior_to_default",
        "regret_noninferior_to_random",
        "failure_noninferior_to_default",
        "failure_noninferior_to_random",
    ]
    check(
        "leaf_specific_noninferiority",
        len(tasks) == 10 and tasks[noninferiority_columns].all().all(),
        len(tasks),
    )
    check(
        "governance_boundary",
        completion.get("rna_values_accessed") is True
        and completion.get("holdout_outcomes_accessed") is True
        and completion.get("regret_computation_run") is True
        and completion.get("limited_preauthorization_access_deviation_disclosed") is True
        and completion.get("construction_outputs_mutated") is False
        and completion.get("method_rerun") is False
        and completion.get("policy_refit_run") is False
        and completion.get("full_benchmark_run") is False
        and completion.get("results_writing_run") is False
        and completion.get("universal_ranking_run") is False
        and completion.get("final_ranking_run") is False,
        "evaluation-only boundary",
    )

    output = completion_path.parent / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    checks_path = output / "validation_checks.tsv"
    pd.DataFrame(checks).to_csv(checks_path, sep="\t", index=False)
    claims = pd.DataFrame(
        [
            {
                "claim_id": "external_portability",
                "status": "supported_nondiscriminating_ceiling",
                "evidence": "46/46 admissible rows pass across two studies; policy/default/random regret all zero",
                "scope": "construction_portability_only",
            },
            {
                "claim_id": "external_rna_policy_regret",
                "status": "descriptive_support",
                "evidence": "five of five frozen leaves have zero policy regret",
                "scope": "one external study with disclosed access-order deviation",
            },
            {
                "claim_id": "annotation_vs_matched_control",
                "status": "descriptive_direction_support",
                "evidence": "20/20 annotation-representation deltas are positive",
                "scope": "one external RNA study",
            },
            {
                "claim_id": "universal_method_or_representation_winner",
                "status": "not_tested_and_prohibited",
                "evidence": "leaf-specific outputs only",
                "scope": "no universal or final ranking",
            },
        ]
    )
    claims_path = output / "claim_scope_summary.tsv"
    claims.to_csv(claims_path, sep="\t", index=False)
    all_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    evidence = {
        "schema_version": 1,
        "gate": "G5",
        "milestone": "H5C_external_evaluation_completion_audit",
        "status": "pass" if all_pass else "blocked",
        "checks_passed": sum(row["status"] == "pass" for row in checks),
        "checks_total": len(checks),
        "rna_methods_and_controls": len(metrics),
        "rna_evaluation_failures": len(failures),
        "rna_eligible_genes": 6074,
        "admissible_evaluation_rows": len(utilities),
        "study_leaf_units": len(leaves),
        "random_comparator_rows": len(random),
        "portability_rows_passed": int((portability.evaluation_status == "pass").sum()),
        "portability_rows_total": len(portability),
        "zero_policy_regret_leaves": int((leaves.policy_regret == 0).sum()),
        "matched_control_positive_rows": int((controls.delta_vs_accessibility_matched_control > 0).sum()),
        "matched_control_rows": len(controls),
        "limited_preauthorization_access_deviation_disclosed": True,
        "two_study_external_rna_claim_allowed": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "completion_evidence": {
            "path": str(completion_path.relative_to(root)),
            "sha256": sha256(completion_path),
        },
        "process_tree_memory": {
            "path": str(memory_path.relative_to(root)),
            "sha256": sha256(memory_path),
            "runtime_seconds": memory["runtime_seconds"],
            "peak_aggregate_pss_mb": memory["peak_aggregate_pss_mb"],
        },
        "validation_checks": {
            "path": str(checks_path.relative_to(root)),
            "sha256": sha256(checks_path),
        },
        "claim_scope_summary": {
            "path": str(claims_path.relative_to(root)),
            "sha256": sha256(claims_path),
        },
        "implementation": {
            "path": "spatialgabench/g5_h5c_summary.py",
            "sha256": sha256(Path(__file__)),
        },
    }
    evidence_path = output / "completion_audit.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"checks": checks_path, "claims": claims_path, "evidence": evidence_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--completion",
        default="results/v1/g5/h5c_external_evaluation/completion_evidence.json",
    )
    parser.add_argument(
        "--memory",
        default="results/v1/g5/h5c_external_evaluation/process_tree_memory.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    outputs = summarize(args.completion, args.memory, args.root)
    print(outputs["evidence"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
