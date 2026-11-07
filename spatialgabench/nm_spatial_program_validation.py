"""Independent completion audit for NU4 held-out spatial programs."""

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
    config_path = root / evidence["config"]["path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({"check_id": check_id, "observed": _display(observed), "expected": _display(expected), "pass": bool(observed == expected)})

    check("completion_status", evidence.get("status"), "pass")
    check("samples", evidence.get("samples"), 5)
    check("study_families", evidence.get("study_families"), 4)
    check("method_contracts", evidence.get("method_contracts"), 114)
    check("fold_component_rows", evidence.get("fold_component_rows"), 5700)
    check("matched_control_deltas", evidence.get("matched_control_deltas"), 85)
    for name in ("config", "preflight_evidence", "authorization", "implementation"):
        record = evidence[name]
        check(f"artifact_checksum__{name}", _sha256(root / record["path"]), record["sha256"])
    for name, record in evidence["artifacts"].items():
        check(f"result_checksum__{name}", _sha256(root / record["path"]), record["sha256"])

    preflight = _json(root / evidence["preflight_evidence"]["path"])
    check("preflight_status", preflight.get("status"), "pass")
    check("preflight_no_rna_values", preflight.get("rna_expression_values_read"), False)
    check("preflight_no_ga_values", preflight.get("gene_activity_values_read"), False)
    check("preflight_no_evaluation", preflight.get("evaluation_run"), False)
    for name in ("method_input_manifest", "data_identity_manifest", "implementation_manifest"):
        record = preflight[name]
        check(f"preflight_checksum__{name}", _sha256(root / record["path"]), record["sha256"])

    authorization = _json(root / evidence["authorization"]["path"])
    check("authorization_status", authorization.get("status"), "authorized")
    check("authorization_milestone", authorization.get("milestone"), "NU4_heldout_RNA_spatial_program_preservation")
    check("rna_evaluation_authorized", authorization.get("rna_expression_value_access_authorized"), True)
    check("spatial_program_authorized", authorization.get("spatial_program_evaluation_authorized"), True)
    for field in (
        "method_rerun_authorized", "construction_tuning_authorized",
        "labels_clusters_pathology_access_authorized", "policy_refit_authorized",
        "results_writing_authorized", "universal_ranking_authorized", "final_ranking_authorized",
    ):
        check(f"authorization_boundary_false__{field}", authorization.get(field), False)

    methods = pd.read_csv(root / preflight["method_input_manifest"]["path"], sep="\t")
    keys = ["dataset_id", "representation_id", "method_id"]
    check("method_manifest_rows", len(methods), 114)
    check("method_manifest_unique", int(methods.duplicated(keys).sum()), 0)
    expected_counts = {
        "gse263333_5m_20um_atac": 27,
        "gse205052_mousebrain_20um": 27,
        "gse308623_p2s3": 27,
        "gse205055_humanbrain_50um": 27,
        "gse279771_gbm_insertion_track": 6,
    }
    check("method_counts_by_dataset", methods.groupby("dataset_id").size().to_dict(), expected_counts)
    check("method_input_tracks_separated", set(methods.input_track), {"P", "B", "F", "I"})

    for dataset_id, sample in config["samples"].items():
        record = preflight["fold_assignments"][dataset_id]
        check(f"fold_checksum__{dataset_id}", _sha256(root / record["path"]), record["sha256"])
        folds = pd.read_csv(root / record["path"], sep="\t")
        check(f"fold_rows__{dataset_id}", len(folds), int(sample["expected_evaluation_observations"]))
        check(f"fold_barcodes_unique__{dataset_id}", int(folds.barcode.duplicated().sum()), 0)
        check(f"fold_panel__{dataset_id}", set(folds.fold), set(range(5)))
        check(f"fold_minimum__{dataset_id}", bool(folds.fold.value_counts().ge(int(config["folds"]["minimum_test_observations"])).all()), True)

    detail = pd.read_csv(root / evidence["artifacts"]["fold_component_metrics.tsv.gz"]["path"], sep="\t")
    check("detail_rows", len(detail), 5700)
    check("detail_status", bool(detail.status.eq("pass").all()), True)
    check("detail_units_per_contract", bool(detail.groupby(keys).size().eq(50).all()), True)
    check("detail_folds", set(detail.fold), set(range(5)))
    check("detail_components", set(detail.component), set(range(1, 11)))
    check("spearman_bounded", bool(detail.heldout_component_spearman.between(-1, 1).all()), True)
    check("pearson_bounded", bool(detail.heldout_component_pearson.between(-1, 1).all()), True)
    check("morans_difference_nonnegative", bool(detail.absolute_morans_i_difference.ge(0).all()), True)
    check("gene_fraction_bounded", bool(detail.activity_present_gene_fraction.between(0, 1).all()), True)

    summary = pd.read_csv(root / evidence["artifacts"]["contract_summary.tsv"]["path"], sep="\t")
    check("summary_rows", len(summary), 114)
    check("summary_unique", int(summary.duplicated(keys).sum()), 0)
    check("summary_units", bool(summary.fold_component_units.eq(50).all()), True)
    check("summary_status", bool(summary.status.eq("pass").all()), True)
    check("summary_exact_method_keys", set(map(tuple, summary[keys].to_numpy())), set(map(tuple, methods[keys].to_numpy())))
    check("score_ci_order", bool((summary.descriptive_bootstrap_score_q025 <= summary.descriptive_bootstrap_score_q975).all()), True)
    check("moran_ci_order", bool((summary.descriptive_bootstrap_morans_difference_q025 <= summary.descriptive_bootstrap_morans_difference_q975).all()), True)

    deltas = pd.read_csv(root / evidence["artifacts"]["matched_control_deltas.tsv"]["path"], sep="\t")
    check("delta_rows", len(deltas), 85)
    check("delta_controls_present", bool(deltas.control_median_heldout_component_spearman.notna().all()), True)
    check("delta_annotation_only", set(deltas.method_category), {"annotation"})
    check("delta_no_fragment_software", bool(~deltas.input_track.eq("F").any()), True)
    check("delta_identity", bool(np.allclose(deltas.delta_heldout_component_spearman_vs_control, deltas.median_heldout_component_spearman - deltas.control_median_heldout_component_spearman)), True)
    check("delta_moran_identity", bool(np.allclose(deltas.delta_morans_difference_vs_control, deltas.control_median_absolute_morans_i_difference - deltas.median_absolute_morans_i_difference)), True)

    genes = pd.read_csv(root / evidence["artifacts"]["fold_gene_eligibility.tsv"]["path"], sep="\t")
    check("gene_fold_rows", len(genes), 25)
    check("gene_fold_unique", int(genes.duplicated(["dataset_id", "fold"]).sum()), 0)
    check("eligible_gene_minimum", bool(genes.eligible_training_genes.ge(int(config["rna_programs"]["minimum_eligible_training_genes"])).all()), True)

    check("rna_evaluation_access_recorded", evidence.get("rna_expression_values_accessed_for_evaluation"), True)
    for field in (
        "rna_used_for_construction", "construction_outputs_mutated",
        "labels_clusters_pathology_accessed", "method_rerun", "construction_tuning_run",
        "policy_refit_run", "input_tracks_pooled", "universal_ranking_run",
        "final_ranking_run", "fold_component_units_treated_as_independent_studies",
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
        "milestone": "NU4_heldout_RNA_spatial_program_preservation",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "method_contracts": 114,
        "matched_control_deltas": 85,
        "study_family_unit_preserved": True,
        "universal_ranking_allowed": False,
        "final_ranking_allowed": False,
        "validation_checks": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path)},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        failed = table.loc[~table["pass"], "check_id"].tolist()
        raise ValueError(f"NU4 independent validation failed: {failed}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="results/v1/nm_upgrade/nu4_spatial_programs/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
