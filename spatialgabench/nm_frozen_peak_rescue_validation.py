"""Independent completion audit for the TA2 frozen full-depth peak rescue."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spatialgabench.nm_frozen_peak_rescue import B_ROLE, PF_ROLE, P_ROLE, sha256_path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _record(checks: list[dict[str, Any]], check_id: str, passed: bool, detail: Any) -> None:
    checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})


def _equal(left: pd.Series, right: pd.Series) -> bool:
    return bool(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=1e-12, atol=1e-12, equal_nan=True))


def validate(
    config_path: str | Path = "configs/nm_frozen_peak_rescue.yaml",
    evidence_path: str | Path = "results/v1/nm_truth_anchor/ta2_frozen_peak_rescue/evaluation/completion_evidence.json",
    root: str | Path = ".",
) -> Path:
    root = Path(root).resolve()
    config_path, evidence_path = root / config_path, root / evidence_path
    config, evidence = yaml.safe_load(config_path.read_text(encoding="utf-8")), _json(evidence_path)
    checks: list[dict[str, Any]] = []
    _record(checks, "completion_status", evidence.get("status") == "pass", evidence.get("status"))
    expected_counts = {
        "new_frozen_peak_rna_contracts": 90,
        "spatial_program_contracts": 260,
        "spatial_program_detail_rows": 13000,
        "mechanism_comparisons": 90,
        "method_study_depth_summaries": 30,
        "study_depth_summaries": 6,
    }
    for field, expected in expected_counts.items():
        _record(checks, f"count::{field}", evidence.get(field) == expected, evidence.get(field))

    for field in ("config", "preflight", "authorization"):
        record = evidence[field]
        _record(checks, f"binding_checksum::{field}", sha256_path(root / record["path"]) == record["sha256"], record["path"])
    artifacts: dict[str, pd.DataFrame] = {}
    for name, record in evidence["artifacts"].items():
        path = root / record["path"]
        valid = path.is_file() and sha256_path(path) == record["sha256"]
        _record(checks, f"artifact_checksum::{name}", valid, record["path"])
        if name.endswith(".tsv"):
            artifacts[name] = pd.read_csv(path, sep="\t")

    preflight = _json(root / evidence["preflight"]["path"])
    authorization = _json(root / evidence["authorization"]["path"])
    _record(checks, "evaluation_preflight_status", preflight.get("status") == "pass", preflight.get("status"))
    _record(checks, "preflight_did_not_read_rna", preflight.get("rna_expression_values_read") is False, preflight.get("rna_expression_values_read"))
    _record(checks, "authorization_status", authorization.get("status") == "authorized", authorization.get("status"))
    _record(checks, "authorization_milestone", authorization.get("milestone") == "TA2_frozen_peak_rescue_evaluation", authorization.get("milestone"))
    for field in ("rna_expression_value_access_authorized", "paired_rna_evaluation_authorized", "spatial_program_evaluation_authorized"):
        _record(checks, f"authorization_true::{field}", authorization.get(field) is True, authorization.get(field))
    for field in ("construction_or_method_rerun_authorized", "parameter_tuning_authorized", "labels_clusters_pathology_access_authorized", "policy_refit_authorized", "universal_ranking_authorized", "final_ranking_authorized"):
        _record(checks, f"authorization_false::{field}", authorization.get(field) is False, authorization.get(field))

    representation_evidence = _json(root / config["output_root"] / "representation_completion_evidence.json")
    representation_summary = pd.read_csv(root / representation_evidence["representation_summary"]["path"], sep="\t")
    _record(checks, "representation_status", representation_evidence.get("status") == "pass", representation_evidence.get("status"))
    _record(checks, "representation_rows", len(representation_summary) == 18, len(representation_summary))
    _record(checks, "representation_feature_order", bool(representation_summary.feature_id_order_match_full_depth.all()), int(representation_summary.feature_id_order_match_full_depth.sum()))
    _record(checks, "representation_observation_order", bool(representation_summary.observation_id_order_match_existing_condition.all()), int(representation_summary.observation_id_order_match_existing_condition.sum()))
    _record(checks, "representation_studies", set(representation_summary.dataset_id) == set(config["studies"]), set(representation_summary.dataset_id))

    method_evidence = _json(root / config["output_root"] / "method_completion_evidence.json")
    method_summary = pd.read_csv(root / method_evidence["method_run_summary"]["path"], sep="\t")
    _record(checks, "method_accounting_status", method_evidence.get("status") == "pass", method_evidence.get("status"))
    _record(checks, "method_rows", len(method_summary) == 90, len(method_summary))
    _record(checks, "method_success", bool(method_summary.success.all()), int(method_summary.success.sum()))
    _record(checks, "method_plan_unique", int(method_summary.duplicated(["condition_id", "representation_id", "method_id"]).sum()) == 0, int(method_summary.duplicated(["condition_id", "representation_id", "method_id"]).sum()))

    manifest = pd.read_csv(root / preflight["method_manifest"]["path"], sep="\t")
    success = manifest.success.astype(str).str.lower().eq("true")
    _record(checks, "evaluation_manifest_rows", len(manifest) == 270, len(manifest))
    _record(checks, "evaluation_manifest_success", int(success.sum()) == 260, int(success.sum()))
    _record(checks, "evaluation_manifest_unique", int(manifest.duplicated(["condition_id", "representation_id", "method_id"]).sum()) == 0, int(manifest.duplicated(["condition_id", "representation_id", "method_id"]).sum()))
    representation_counts = manifest.groupby("representation_role").size().to_dict()
    _record(checks, "representation_panel_separated", representation_counts == {B_ROLE: 90, P_ROLE: 90, PF_ROLE: 90}, representation_counts)

    rna = artifacts["frozen_peak_rna_metrics.tsv"]
    _record(checks, "rna_rows", len(rna) == 90, len(rna))
    _record(checks, "rna_unique", int(rna.duplicated(["dataset_id", "depth_fraction", "seed", "method_id"]).sum()) == 0, int(rna.duplicated(["dataset_id", "depth_fraction", "seed", "method_id"]).sum()))
    primary = rna.median_gene_wise_spearman * rna.evaluable_gene_fraction
    _record(checks, "rna_primary_formula", _equal(primary, rna.coverage_adjusted_median_gene_wise_spearman), float(np.max(np.abs(primary - rna.coverage_adjusted_median_gene_wise_spearman))))
    retention = rna.coverage_adjusted_median_gene_wise_spearman / rna.full_depth_P_primary
    _record(checks, "rna_retention_formula", _equal(retention, rna.frozen_peak_retention_vs_full_depth_P), len(rna))
    _record(checks, "rna_scores_finite", bool(np.isfinite(rna[["coverage_adjusted_median_gene_wise_spearman", "global_pseudobulk_spearman", "top100_gene_jaccard"]].to_numpy(dtype=float)).all()), len(rna))

    mechanism = artifacts["mechanism_comparison.tsv"]
    _record(checks, "mechanism_rows", len(mechanism) == 90, len(mechanism))
    _record(checks, "primary_rescue_formula", _equal(mechanism.frozen_P_minus_de_novo_P_primary, mechanism.frozen_P_primary - mechanism.de_novo_P_primary), int(mechanism.de_novo_P_primary.notna().sum()))
    _record(checks, "primary_residual_formula", _equal(mechanism.B25_minus_frozen_P_primary, mechanism.B25_primary - mechanism.frozen_P_primary), len(mechanism))
    _record(checks, "spatial_rescue_formula", _equal(mechanism.frozen_P_minus_de_novo_P_spatial_program, mechanism.frozen_P_spatial_program - mechanism.de_novo_P_spatial_program), int(mechanism.de_novo_P_spatial_program.notna().sum()))
    _record(checks, "spatial_residual_formula", _equal(mechanism.B25_minus_frozen_P_spatial_program, mechanism.B25_spatial_program - mechanism.frozen_P_spatial_program), len(mechanism))

    spatial = artifacts["spatial_program_contract_summary.tsv"]
    detail_record = evidence["artifacts"]["spatial_program_fold_component_metrics.tsv.gz"]
    detail = pd.read_csv(root / detail_record["path"], sep="\t")
    _record(checks, "spatial_summary_rows", len(spatial) == 260, len(spatial))
    _record(checks, "spatial_units_per_contract", bool(spatial.fold_component_units.eq(50).all()), spatial.fold_component_units.value_counts().to_dict())
    _record(checks, "spatial_detail_rows", len(detail) == 13000, len(detail))
    _record(checks, "spatial_detail_units", bool(detail.groupby(["condition_id", "representation_id", "method_id"]).size().eq(50).all()), detail.groupby(["condition_id", "representation_id", "method_id"]).size().value_counts().to_dict())
    _record(checks, "spatial_fold_panel", set(detail.fold) == set(range(5)), set(detail.fold))
    _record(checks, "spatial_component_panel", set(detail.component) == set(range(1, 11)), set(detail.component))
    _record(checks, "spatial_correlations_bounded", bool(detail.heldout_component_spearman.between(-1, 1).all() and detail.heldout_component_pearson.between(-1, 1).all()), len(detail))

    method_depth = artifacts["method_study_depth_summary.tsv"]
    study_depth = artifacts["study_depth_mechanism_summary.tsv"]
    _record(checks, "method_depth_rows", len(method_depth) == 30, len(method_depth))
    _record(checks, "method_depth_three_seeds", bool(method_depth.technical_seeds.eq(3).all()), method_depth.technical_seeds.value_counts().to_dict())
    _record(checks, "study_depth_rows", len(study_depth) == 6, len(study_depth))
    _record(checks, "study_depth_five_methods", bool(study_depth.methods.eq(5).all()), study_depth.methods.value_counts().to_dict())
    _record(checks, "study_is_inference_unit", set(study_depth.independent_inference_unit) == {"study_family"}, set(study_depth.independent_inference_unit))
    _record(checks, "seeds_not_independent", bool((~study_depth.technical_seeds_independent.astype(bool)).all()), study_depth.technical_seeds_independent.unique())
    _record(checks, "methods_not_independent", bool((~study_depth.methods_independent_replicates.astype(bool)).all()), study_depth.methods_independent_replicates.unique())

    gene_path = root / evidence["artifacts"]["frozen_peak_gene_level_correlations.tsv.gz"]["path"]
    with gzip.open(gene_path, "rt", encoding="utf-8") as handle:
        gene_rows = sum(1 for _ in handle) - 1
    _record(checks, "gene_row_accounting", gene_rows == int(rna.rna_eligible_genes.sum()), f"{gene_rows}/{int(rna.rna_eligible_genes.sum())}")

    _record(checks, "rna_evaluation_recorded", evidence.get("rna_expression_values_accessed_for_evaluation") is True, evidence.get("rna_expression_values_accessed_for_evaluation"))
    for field in ("rna_used_for_construction", "construction_outputs_mutated", "construction_or_method_rerun", "parameter_tuning_run", "labels_clusters_pathology_accessed", "technical_seeds_treated_as_independent_studies", "fold_component_units_treated_as_independent_studies", "input_tracks_pooled", "policy_refit_run", "universal_ranking_run", "final_ranking_run"):
        _record(checks, f"completion_boundary_false::{field}", evidence.get(field) is False, evidence.get(field))

    table = pd.DataFrame(checks)
    output = evidence_path.parent / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    checks_path = output / "validation_checks.tsv"
    table.to_csv(checks_path, sep="\t", index=False)
    audit = {
        "schema_version": 1,
        "milestone": "TA2_frozen_peak_rescue_independent_validation",
        "status": "pass" if table.status.eq("pass").all() else "fail",
        "checks_passed": int(table.status.eq("pass").sum()),
        "checks_total": len(table),
        "independent_study_families": 2,
        "universal_ranking_allowed": False,
        "final_ranking_allowed": False,
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": sha256_path(evidence_path)},
        "validation_checks": {"path": str(checks_path.relative_to(root)), "sha256": sha256_path(checks_path)},
        "validator": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha256_path(Path(__file__).resolve())},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        failed = table.loc[table.status.eq("fail"), "check_id"].tolist()
        raise ValueError(f"TA2 independent validation failed: {failed}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_frozen_peak_rescue.yaml")
    parser.add_argument("--evidence", default="results/v1/nm_truth_anchor/ta2_frozen_peak_rescue/evaluation/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.config, args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
