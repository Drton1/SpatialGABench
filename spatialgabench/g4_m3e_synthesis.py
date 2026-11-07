"""Descriptive cross-study synthesis of author-domain concordance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import spearmanr

from spatialgabench.software_wrappers.common import sha256_file


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if sha256_file(root / config["authorization"]) != config["authorization_sha256"]:
        raise ValueError("Cross-study synthesis authorization checksum mismatch")
    authorization = _json(root / config["authorization"])
    if authorization.get("status") != "authorized" or authorization.get("authorized_task") != "descriptive_cross_study_direction_replication_only":
        raise PermissionError("Cross-study author-domain synthesis is not authorized")
    for study in config["studies"]:
        for field in ("evidence", "contrasts"):
            if sha256_file(root / study[field]) != study[f"{field}_sha256"]:
                raise ValueError(f"Frozen study artifact checksum mismatch: {study['study_id']}/{field}")
        evidence = _json(root / study["evidence"])
        if (
            evidence.get("status") != "pass"
            or evidence.get("methods_passed") != 8
            or evidence.get("seed_runs_passed") != 40
            or evidence.get("labels_used_for_parameter_selection") is not False
            or evidence.get("resolution_search_run") is not False
            or evidence.get("holdout_accessed") is not False
            or evidence.get("universal_ranking_run") is not False
        ):
            raise ValueError(f"Study evidence boundary failed: {study['study_id']}")
    if len({study["study_id"] for study in config["studies"]}) != 2:
        raise ValueError("Cross-study synthesis requires two independent study families")
    if any(
        bool(config.get(field))
        for field in (
            "holdout_accessed",
            "full_real_data_benchmark_run",
            "recommendation_policy_fit_run",
            "universal_ranking_run",
            "final_ranking_run",
        )
    ):
        raise PermissionError("A forbidden cross-study synthesis flag is enabled")
    config["_path"] = path
    return config


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    rows: list[pd.DataFrame] = []
    for study in config["studies"]:
        frame = pd.read_csv(root / study["contrasts"], sep="\t")
        expected_methods = set(config["annotation_methods"]) | {
            config["control_method"],
            config["negative_control_method"],
            config["raw_lsi_method"],
        }
        if set(frame.method_id) != expected_methods:
            raise ValueError(f"Method panel mismatch: {study['study_id']}")
        frame.insert(0, "representation_id", study["representation_id"])
        frame.insert(0, "dataset_id", study["dataset_id"])
        frame.insert(0, "study_id", study["study_id"])
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    annotation = combined[combined.method_id.isin(config["annotation_methods"])].copy()

    replication_rows = []
    for method_id, frame in annotation.groupby("method_id", sort=True):
        by_study = frame.set_index("study_id")
        replication_rows.append(
            {
                "method_id": method_id,
                "independent_studies": int(frame.study_id.nunique()),
                "studies_above_accessibility_matched_control_all_seeds": int(
                    frame.all_seeds_above_accessibility_matched_control.astype(bool).sum()
                ),
                "studies_above_raw_lsi_all_seeds": int(frame.all_seeds_above_raw_peak_lsi.astype(bool).sum()),
                "gse214986_delta_vs_matched_control": float(
                    by_study.loc["llorens_bobadilla_2023", "median_ARI_delta_vs_accessibility_matched_control"]
                ),
                "gse288178_delta_vs_matched_control": float(
                    by_study.loc["li_2026", "median_ARI_delta_vs_accessibility_matched_control"]
                ),
                "gse214986_delta_vs_raw_lsi": float(
                    by_study.loc["llorens_bobadilla_2023", "median_ARI_delta_vs_raw_peak_lsi"]
                ),
                "gse288178_delta_vs_raw_lsi": float(by_study.loc["li_2026", "median_ARI_delta_vs_raw_peak_lsi"]),
                "direction_vs_matched_control_replicates": bool(
                    frame.all_seeds_above_accessibility_matched_control.astype(bool).all()
                ),
                "direction_vs_raw_lsi_replicates": bool(frame.all_seeds_above_raw_peak_lsi.astype(bool).all()),
            }
        )
    replication = pd.DataFrame(replication_rows)

    scope_rows = []
    for study in config["studies"]:
        frame = combined[combined.study_id == study["study_id"]].set_index("method_id")
        methods = frame.loc[config["annotation_methods"]]
        scope_rows.append(
            {
                "study_id": study["study_id"],
                "dataset_id": study["dataset_id"],
                "representation_id": study["representation_id"],
                "annotation_methods_above_matched_control_all_seeds": int(
                    methods.all_seeds_above_accessibility_matched_control.astype(bool).sum()
                ),
                "annotation_methods_above_raw_lsi_all_seeds": int(
                    methods.all_seeds_above_raw_peak_lsi.astype(bool).sum()
                ),
                "raw_lsi_median_ARI": float(frame.loc[config["raw_lsi_method"], "median_ARI"]),
                "matched_control_median_ARI": float(frame.loc[config["control_method"], "median_ARI"]),
                "author_label_scope": "author_defined_cluster_not_independent_truth",
            }
        )
    scope = pd.DataFrame(scope_rows)
    wide = annotation.pivot(index="method_id", columns="study_id", values="median_ARI").loc[config["annotation_methods"]]
    order_spearman = float(spearmanr(wide.iloc[:, 0], wide.iloc[:, 1]).statistic)

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    combined_path = output / "cross_study_method_contrasts.tsv"
    replication_path = output / "method_direction_replication.tsv"
    scope_path = output / "representation_scope_summary.tsv"
    combined.to_csv(combined_path, sep="\t", index=False)
    replication.to_csv(replication_path, sep="\t", index=False)
    scope.to_csv(scope_path, sep="\t", index=False)
    evidence_path = output / "g4_m3e_cross_study_author_domain_synthesis_evidence.json"
    artifacts = [combined_path, replication_path, scope_path]
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3E_cross_study_author_domain_synthesis",
        "status": "pass",
        "independent_study_families": 2,
        "representations_reported_separately": True,
        "annotation_methods": len(config["annotation_methods"]),
        "method_study_rows": len(annotation),
        "methods_replicating_positive_direction_vs_matched_control": int(
            replication.direction_vs_matched_control_replicates.sum()
        ),
        "methods_replicating_positive_direction_vs_raw_lsi": int(replication.direction_vs_raw_lsi_replicates.sum()),
        "descriptive_annotation_method_order_spearman": order_spearman,
        "spearman_interpretation": "exploratory_five_methods_not_population_inference",
        "artifacts": {
            path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts
        },
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "study_evidence": [
            {"study_id": study["study_id"], "path": study["evidence"], "sha256": study["evidence_sha256"]}
            for study in config["studies"]
        ],
        "study_level_population_inference_run": False,
        "representation_pooling_run": False,
        "holdout_accessed": False,
        "full_real_data_benchmark_run": False,
        "recommendation_policy_fit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(evidence_path, evidence)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m3e_cross_study_author_domain_synthesis.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
