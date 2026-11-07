"""Independent validation for the TA1 paired depth-to-accuracy experiment."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spatialgabench.nm_paired_depth_accuracy import sha256_path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _record(checks: list[dict[str, Any]], check_id: str, passed: bool, detail: Any) -> None:
    checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})


def validate(
    config_path: str | Path = "configs/nm_paired_depth_accuracy.yaml",
    evidence_path: str | Path = "results/v1/nm_truth_anchor/ta1_paired_depth_accuracy/completion_evidence.json",
    root: str | Path = ".",
) -> Path:
    root = Path(root).resolve()
    config_path, evidence_path = root / config_path, root / evidence_path
    config, evidence = _yaml(config_path), _json(evidence_path)
    checks: list[dict[str, Any]] = []
    _record(checks, "completion_status", evidence.get("status") == "pass", evidence.get("status"))
    _record(checks, "planned_contract_count", evidence.get("planned_contracts") == 273, evidence.get("planned_contracts"))
    _record(checks, "successful_contract_count", evidence.get("evaluated_successful_contracts") == 236, evidence.get("evaluated_successful_contracts"))
    _record(checks, "retained_failure_count", evidence.get("retained_structural_failures") == 37, evidence.get("retained_structural_failures"))

    artifacts: dict[str, pd.DataFrame] = {}
    for name, record in evidence.get("artifacts", {}).items():
        path = root / record["path"]
        valid = path.is_file() and sha256_path(path) == record["sha256"]
        _record(checks, f"artifact_checksum::{name}", valid, record["path"])
        if name.endswith(".tsv"):
            artifacts[name] = pd.read_csv(path, sep="\t")

    metrics = artifacts["contract_rna_accuracy_metrics.tsv"]
    _record(checks, "metrics_rows", len(metrics) == 273, len(metrics))
    success = metrics.evaluation_status.eq("pass")
    failed = metrics.evaluation_status.eq("not_evaluated_structural_failure")
    _record(checks, "metrics_success_failure_partition", int(success.sum()) == 236 and int(failed.sum()) == 37 and bool((success | failed).all()), f"{success.sum()}/{failed.sum()}")
    scored = metrics.loc[success]
    unscored = metrics.loc[failed]
    primary = scored.median_gene_wise_spearman * scored.evaluable_gene_fraction
    _record(checks, "primary_formula", bool(np.allclose(primary, scored.coverage_adjusted_median_gene_wise_spearman, rtol=1e-12, atol=1e-12)), float(np.max(np.abs(primary - scored.coverage_adjusted_median_gene_wise_spearman))))
    score_columns = [
        "median_gene_wise_spearman", "evaluable_gene_fraction",
        "coverage_adjusted_median_gene_wise_spearman", "global_pseudobulk_spearman",
        "top100_gene_jaccard",
    ]
    _record(checks, "failures_have_no_scores", bool(unscored[score_columns].isna().all().all()), int(unscored[score_columns].notna().sum().sum()))
    _record(checks, "successful_scores_finite", bool(np.isfinite(scored[score_columns].to_numpy(dtype=float)).all()), len(scored))

    full = scored.loc[scored.depth_fraction.eq(1.0)]
    finite_full = full.loc[full.full_depth_rna_primary.gt(0)]
    _record(checks, "full_depth_retention_identity", bool(np.allclose(finite_full.rna_primary_retention_vs_full, 1.0)), len(finite_full))
    recomputed = scored.coverage_adjusted_median_gene_wise_spearman / scored.full_depth_rna_primary
    eligible_ratio = scored.full_depth_rna_primary.gt(0)
    _record(checks, "retention_formula", bool(np.allclose(scored.loc[eligible_ratio, "rna_primary_retention_vs_full"], recomputed[eligible_ratio], rtol=1e-12, atol=1e-12)), int(eligible_ratio.sum()))
    delta = scored.coverage_adjusted_median_gene_wise_spearman - scored.full_depth_rna_primary
    _record(checks, "delta_formula", bool(np.allclose(scored.rna_primary_delta_vs_full, delta, rtol=1e-12, atol=1e-12)), len(scored))

    expected_depths = {0.01, 0.05, 0.25, 1.0}
    expected_reduced_seeds = {42001, 42002, 42003}
    depth_ok = True
    for dataset_id in config["depth_sources"]:
        frame = metrics.loc[metrics.dataset_id.eq(dataset_id)]
        depth_ok &= len(frame) == 130 and set(frame.depth_fraction.astype(float)) == expected_depths
        depth_ok &= set(frame.loc[frame.depth_fraction.lt(1), "seed"].astype(int)) == expected_reduced_seeds
        depth_ok &= set(frame.loc[frame.depth_fraction.eq(1), "seed"].astype(int)) == {0}
    _record(checks, "exact_depth_seed_design", depth_ok, sorted(config["depth_sources"]))
    human_id = config["human_full_depth_anchor"]["dataset_id"]
    human = metrics.loc[metrics.dataset_id.eq(human_id)]
    _record(checks, "human_anchor_exact", len(human) == 13 and human.depth_fraction.eq(1).all() and human.seed.eq(0).all(), len(human))

    seed_summary = artifacts["depth_seed_summary.tsv"]
    _record(checks, "seed_summary_counts_bounded", bool((seed_summary.successful_seeds.fillna(0) <= seed_summary.planned_seeds).all()), len(seed_summary))
    _record(checks, "no_track_pooling_in_seed_summary", "representation_id" in seed_summary.columns, seed_summary.representation_id.nunique())
    proxy = artifacts["construction_proxy_vs_rna_truth.tsv"]
    _record(checks, "proxy_truth_is_descriptive", bool((~proxy.independent_study_inference.astype(bool)).all()), len(proxy))

    gene_record = evidence["artifacts"]["gene_level_correlations.tsv.gz"]
    gene_path = root / gene_record["path"]
    with gzip.open(gene_path, "rt", encoding="utf-8") as handle:
        gene_rows = sum(1 for _ in handle) - 1
    expected_gene_rows = int(scored.rna_eligible_genes.sum())
    _record(checks, "gene_level_row_accounting", gene_rows == expected_gene_rows, f"{gene_rows}/{expected_gene_rows}")

    forbidden = {
        "rna_used_for_construction": False,
        "construction_outputs_mutated": False,
        "construction_or_method_rerun": False,
        "labels_clusters_pathology_accessed": False,
        "technical_seeds_treated_as_independent_studies": False,
        "policy_refit_run": False,
        "input_tracks_pooled": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    for field, expected in forbidden.items():
        _record(checks, f"boundary::{field}", evidence.get(field) is expected, evidence.get(field))

    check_frame = pd.DataFrame(checks)
    output = evidence_path.parent / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    checks_path = output / "validation_checks.tsv"
    check_frame.to_csv(checks_path, sep="\t", index=False)
    audit = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_independent_validation",
        "status": "pass" if check_frame.status.eq("pass").all() else "fail",
        "checks_passed": int(check_frame.status.eq("pass").sum()),
        "checks_total": len(check_frame),
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": sha256_path(evidence_path)},
        "validation_checks": {"path": str(checks_path.relative_to(root)), "sha256": sha256_path(checks_path)},
        "validator": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha256_path(Path(__file__).resolve())},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        failed_checks = check_frame.loc[check_frame.status.eq("fail"), "check_id"].tolist()
        raise ValueError(f"TA1 independent validation failed: {failed_checks}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_paired_depth_accuracy.yaml")
    parser.add_argument("--evidence", default="results/v1/nm_truth_anchor/ta1_paired_depth_accuracy/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.config, args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
