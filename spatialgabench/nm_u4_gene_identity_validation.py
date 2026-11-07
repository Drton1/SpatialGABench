"""Independent validation of the narrow NU4C gene-identity correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spatialgabench.nm_paired_depth_accuracy import sha256_path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(config_path: str | Path = "configs/nm_u4_human_snap_gene_identity_correction.yaml", evidence_path: str | Path = "results/v1/nm_upgrade/nu4_spatial_programs_v2/completion_evidence.json", root: str | Path = ".") -> Path:
    root, config_path, evidence_path = Path(root).resolve(), Path(config_path), Path(evidence_path)
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    evidence = _json(root / evidence_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check_id": name, "status": "pass" if passed else "fail", "detail": str(detail)})

    check("completion_status", evidence.get("status") == "pass", evidence.get("status"))
    tables: dict[str, pd.DataFrame] = {}
    for name, record in evidence["artifacts"].items():
        path = root / record["path"]
        check(f"checksum::{name}", path.is_file() and sha256_path(path) == record["sha256"], record["path"])
        if name.endswith(".tsv"):
            tables[name] = pd.read_csv(path, sep="\t")
    summary = tables["contract_summary.tsv"]
    detail = pd.read_csv(root / evidence["artifacts"]["fold_component_metrics.tsv.gz"]["path"], sep="\t")
    deltas = tables["matched_control_deltas.tsv"]
    check("corrected_dimensions", len(summary) == 114 and len(detail) == 5700 and len(deltas) == 85, f"{len(summary)}/{len(detail)}/{len(deltas)}")
    target = config["target"]
    target_mask = summary.dataset_id.eq(target["dataset_id"]) & summary.method_id.eq(target["method_id"]) & summary.representation_id.eq(target["representation_id"])
    corrected = summary.loc[target_mask]
    check("target_unique", len(corrected) == 1, len(corrected))
    check("target_gene_overlap_restored", len(corrected) == 1 and float(corrected.iloc[0].median_activity_present_gene_fraction) > 0.95, corrected.iloc[0].median_activity_present_gene_fraction if len(corrected) else "missing")
    check("target_score_finite", len(corrected) == 1 and np.isfinite(corrected[["median_heldout_component_spearman", "median_heldout_component_pearson", "median_absolute_morans_i_difference"]].to_numpy(dtype=float)).all(), len(corrected))

    old_summary = pd.read_csv(root / config["source_summary"], sep="\t")
    old_target_mask = old_summary.dataset_id.eq(target["dataset_id"]) & old_summary.method_id.eq(target["method_id"]) & old_summary.representation_id.eq(target["representation_id"])
    check("old_target_was_invalid_zero_overlap", int(old_target_mask.sum()) == 1 and float(old_summary.loc[old_target_mask, "median_activity_present_gene_fraction"].iloc[0]) == 0.0, int(old_target_mask.sum()))
    keys = ["dataset_id", "representation_id", "method_id"]
    old_unaffected = old_summary.loc[~old_target_mask].sort_values(keys, kind="mergesort").reset_index(drop=True)
    new_unaffected = summary.loc[~target_mask].sort_values(keys, kind="mergesort").reset_index(drop=True)
    check("unaffected_113_summaries_identical", old_unaffected.equals(new_unaffected), len(new_unaffected))
    old_detail = pd.read_csv(root / config["source_detail"], sep="\t")
    old_detail_target = old_detail.dataset_id.eq(target["dataset_id"]) & old_detail.method_id.eq(target["method_id"]) & old_detail.representation_id.eq(target["representation_id"])
    new_detail_target = detail.dataset_id.eq(target["dataset_id"]) & detail.method_id.eq(target["method_id"]) & detail.representation_id.eq(target["representation_id"])
    detail_keys = keys + ["fold", "component"]
    old_unaffected_detail = old_detail.loc[~old_detail_target].sort_values(detail_keys, kind="mergesort").reset_index(drop=True)
    new_unaffected_detail = detail.loc[~new_detail_target].sort_values(detail_keys, kind="mergesort").reset_index(drop=True)
    check("unaffected_5650_detail_rows_identical", old_unaffected_detail.equals(new_unaffected_detail), len(new_unaffected_detail))
    old_deltas = pd.read_csv(root / config["source_deltas"], sep="\t")
    check("annotation_control_deltas_identical", old_deltas.equals(deltas), len(deltas))
    for field in ("rna_used_for_construction", "construction_or_method_rerun", "annotation_or_control_rerun", "labels_clusters_pathology_accessed", "policy_refit_run", "input_tracks_pooled", "universal_ranking_run", "final_ranking_run"):
        check(f"boundary::{field}", evidence.get(field) is False, evidence.get(field))
    frame = pd.DataFrame(checks)
    output = root / config["output_root"] / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    checks_path = output / "validation_checks.tsv"
    frame.to_csv(checks_path, sep="\t", index=False)
    audit = {
        "schema_version": 1, "milestone": f"{config['milestone']}_independent_validation",
        "status": "pass" if frame.status.eq("pass").all() else "fail",
        "checks_passed": int(frame.status.eq("pass").sum()), "checks_total": len(frame),
        "completion_evidence": {"path": str(evidence_path), "sha256": sha256_path(root / evidence_path)},
        "validation_checks": {"path": str(checks_path.relative_to(root)), "sha256": sha256_path(checks_path)},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        raise ValueError(f"NU4C validation failed: {frame.loc[frame.status.eq('fail'), 'check_id'].tolist()}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_u4_human_snap_gene_identity_correction.yaml")
    parser.add_argument("--evidence", default="results/v1/nm_upgrade/nu4_spatial_programs_v2/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.config, args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
