"""Independent completion audit for NU1 human paired-RNA evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    check("method_contracts", evidence.get("method_contracts"), 27)
    check("matched_control_deltas", evidence.get("matched_control_deltas"), 20)
    check("within_family_comparisons", evidence.get("within_family_comparisons"), 20)
    check("independent_study_increment", evidence.get("independent_study_increment"), 0)
    for name in ("config", "setup", "authorization", "implementation"):
        record = evidence[name]
        check(f"artifact_checksum__{name}", _sha256(root / record["path"]), record["sha256"])
    for name, record in evidence["artifacts"].items():
        check(f"result_checksum__{name}", _sha256(root / record["path"]), record["sha256"])

    setup = _json(root / evidence["setup"]["path"])
    check("setup_status", setup.get("status"), "pass")
    check("setup_blind_to_rna_values", setup.get("rna_expression_values_read"), False)
    check("setup_blind_to_ga_values", setup.get("gene_activity_values_read"), False)
    authorization = _json(root / evidence["authorization"]["path"])
    check("authorization_status", authorization.get("status"), "authorized")
    check("rna_evaluation_authorized", authorization.get("rna_expression_value_access_authorized"), True)
    for field in (
        "method_rerun_authorized", "construction_tuning_authorized",
        "labels_clusters_pathology_access_authorized", "policy_refit_authorized",
        "universal_ranking_authorized", "final_ranking_authorized",
    ):
        check(f"authorization_boundary_false__{field}", authorization.get(field), False)

    metrics = pd.read_csv(root / evidence["artifacts"]["rna_metrics.tsv"]["path"], sep="\t")
    keys = ["representation_id", "method_id"]
    check("metrics_rows", len(metrics), 27)
    check("metrics_unique", int(metrics.duplicated(keys).sum()), 0)
    check("metrics_status", bool(metrics.status.eq("pass").all()), True)
    check("metrics_observations", set(metrics.n_observations), {2500})
    check("primary_finite", bool(np.isfinite(metrics.coverage_adjusted_median_gene_wise_spearman).all()), True)
    check("global_pseudobulk_finite", bool(np.isfinite(metrics.global_pseudobulk_spearman).all()), True)
    gene_overlap_fraction = metrics.ga_matched_rna_eligible_genes / metrics.rna_eligible_genes
    check("rna_gene_overlap_at_least_95pct", bool(gene_overlap_fraction.ge(0.95).all()), True)
    check("rna_gene_variability_nonzero", bool(metrics.ga_variable_rna_eligible_genes.gt(0).all()), True)
    check("coverage_bounded", bool(metrics.evaluable_gene_fraction.between(0, 1).all()), True)

    deltas = pd.read_csv(root / evidence["artifacts"]["matched_control_deltas.tsv"]["path"], sep="\t")
    check("delta_rows", len(deltas), 20)
    check("delta_unique", int(deltas.duplicated(keys).sum()), 0)
    check("delta_controls_present", bool(deltas.matched_control_primary.notna().all()), True)
    check("delta_identity", bool(np.allclose(deltas.delta_vs_matched_control, deltas.coverage_adjusted_median_gene_wise_spearman - deltas.matched_control_primary)), True)
    check("evidence_positive_count", int(deltas.delta_vs_matched_control.gt(0).sum()), int(evidence["positive_annotation_control_deltas"]))

    family = pd.read_csv(root / evidence["artifacts"]["within_family_mouse_human.tsv"]["path"], sep="\t")
    check("family_rows", len(family), 20)
    check("family_unique", int(family.duplicated(keys).sum()), 0)
    check("family_direction_identity", bool(family.direction_consistent.astype(bool).eq(family.human_positive.astype(bool).eq(family.mouse_positive.astype(bool))).all()), True)
    check("evidence_direction_count", int(family.direction_consistent.astype(bool).sum()), int(evidence["within_family_direction_consistent"]))
    check("descriptive_spearman_bounded", bool(-1 <= float(evidence["within_family_delta_spearman_descriptive"]) <= 1), True)

    check("rna_evaluation_recorded", evidence.get("rna_expression_values_accessed_for_evaluation"), True)
    check("evaluation_attempt_index", evidence.get("evaluation_attempt_index"), 3)
    check("invalid_schema_attempt_excluded", evidence.get("invalid_schema_attempt_excluded"), True)
    for field in (
        "rna_used_for_construction", "construction_outputs_mutated",
        "labels_clusters_pathology_accessed", "method_rerun", "construction_tuning_run",
        "policy_refit_run", "input_tracks_pooled", "universal_ranking_run", "final_ranking_run",
    ):
        check(f"completion_boundary_false__{field}", evidence.get(field), False)

    table = pd.DataFrame(checks)
    output = evidence_path.parent / "evidence"
    output.mkdir(exist_ok=True)
    table_path = output / "validation_checks.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    passed = int(table["pass"].sum())
    audit = {
        "schema_version": 1,
        "milestone": "NU1_human_standard_fragment_paired_RNA",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "positive_annotation_control_deltas": int(evidence["positive_annotation_control_deltas"]),
        "within_family_direction_consistent": int(evidence["within_family_direction_consistent"]),
        "independent_study_increment": 0,
        "universal_ranking_allowed": False,
        "final_ranking_allowed": False,
        "validation_checks": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path)},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        failed = table.loc[~table["pass"], "check_id"].tolist()
        raise ValueError(f"NU1 RNA independent validation failed: {failed}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="results/v1/nm_upgrade/nu1_human_rna/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
