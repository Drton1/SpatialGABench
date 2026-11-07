"""Representation-stratified descriptive analysis of MALT morphology concordance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spatialgabench.software_wrappers.common import sha256_file


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def analyze(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for field in ("evaluation_evidence", "fold_metrics", "method_summary"):
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen morphology analysis input changed: {field}")
    if config.get("representation_pooling_run") is not False or config.get("universal_ranking_run") is not False:
        raise PermissionError("Morphology analysis configuration permits forbidden pooling/ranking")
    evidence = json.loads((root / config["evaluation_evidence"]).read_text(encoding="utf-8"))
    if evidence.get("status") != "pass" or evidence.get("representation_pooling_run") is not False:
        raise ValueError("Morphology evaluation evidence does not pass")
    metrics = pd.read_csv(root / config["fold_metrics"], sep="\t")
    primary = metrics[metrics["subset_id"] == config["primary_subset"]].copy()
    if len(primary) != 160 or not (primary["status"] == "pass").all():
        raise ValueError("Primary morphology fold panel is incomplete")

    absolute = (
        primary.groupby(["representation_id", "method_id"], sort=True)
        .agg(
            spatial_folds=("spatial_fold", "count"),
            median_coordinate_only_R2_image=("coordinate_only_R2_image", "median"),
            median_coordinate_plus_method_R2_image=("coordinate_plus_method_R2_image", "median"),
            median_delta_R2_image=("delta_R2_image", "median"),
            minimum_delta_R2_image=("delta_R2_image", "min"),
            maximum_delta_R2_image=("delta_R2_image", "max"),
            positive_absolute_R2_folds=("coordinate_plus_method_R2_image", lambda x: int((x > 0).sum())),
            positive_delta_R2_folds=("delta_R2_image", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    absolute["median_absolute_R2_positive"] = absolute[
        "median_coordinate_plus_method_R2_image"
    ] > 0

    contrasts: list[dict[str, Any]] = []
    for representation_id, frame in primary.groupby("representation_id", sort=True):
        pivot = frame.pivot(index="spatial_fold", columns="method_id", values="delta_R2_image")
        for method_id in config["annotation_methods"]:
            row: dict[str, Any] = {
                "representation_id": representation_id,
                "method_id": method_id,
                "spatial_folds": len(pivot),
            }
            for label, control in (
                ("matched_control", config["matched_control"]),
                ("random_control", config["random_control"]),
                ("raw_lsi", config["raw_lsi_control"]),
            ):
                delta = pivot[method_id] - pivot[control]
                row[f"median_paired_delta_vs_{label}"] = float(np.median(delta))
                row[f"minimum_paired_delta_vs_{label}"] = float(np.min(delta))
                row[f"maximum_paired_delta_vs_{label}"] = float(np.max(delta))
                row[f"all_folds_above_{label}"] = bool((delta > 0).all())
            contrasts.append(row)
    contrast_frame = pd.DataFrame(contrasts).sort_values(
        ["representation_id", "method_id"], kind="mergesort"
    )

    representation_rows: list[dict[str, Any]] = []
    for representation_id, frame in contrast_frame.groupby("representation_id", sort=True):
        absolute_rep = absolute[absolute["representation_id"] == representation_id]
        primary_rep = primary[primary["representation_id"] == representation_id]
        matched = primary_rep[
            primary_rep["method_id"] == config["matched_control"]
        ]["delta_R2_image"].median()
        random = primary_rep[
            primary_rep["method_id"] == config["random_control"]
        ]["delta_R2_image"].median()
        raw_lsi = primary_rep[
            primary_rep["method_id"] == config["raw_lsi_control"]
        ]["delta_R2_image"].median()
        representation_rows.append(
            {
                "representation_id": representation_id,
                "annotation_methods": len(frame),
                "annotation_median_above_matched_control": int(
                    (frame["median_paired_delta_vs_matched_control"] > 0).sum()
                ),
                "annotation_all_folds_above_matched_control": int(
                    frame["all_folds_above_matched_control"].sum()
                ),
                "annotation_median_above_raw_lsi": int(
                    (frame["median_paired_delta_vs_raw_lsi"] > 0).sum()
                ),
                "annotation_all_folds_above_raw_lsi": int(frame["all_folds_above_raw_lsi"].sum()),
                "methods_with_positive_median_absolute_R2": int(
                    absolute_rep["median_absolute_R2_positive"].sum()
                ),
                "matched_control_median_delta_R2_image": float(matched),
                "random_control_median_delta_R2_image": float(random),
                "raw_lsi_median_delta_R2_image": float(raw_lsi),
                "interpretation": "aggregation_signal_not_mapping_specific"
                if matched > 0 and random > 0 and raw_lsi < 0
                else "mixed_control_pattern",
            }
        )
    representation = pd.DataFrame(representation_rows)
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    absolute_path = output / "absolute_predictive_fit_summary.tsv"
    contrast_path = output / "within_representation_control_contrasts.tsv"
    representation_path = output / "representation_descriptive_summary.tsv"
    absolute.to_csv(absolute_path, sep="\t", index=False)
    contrast_frame.to_csv(contrast_path, sep="\t", index=False)
    representation.to_csv(representation_path, sep="\t", index=False)
    evidence_path = output / "morphology_analysis_evidence.json"
    artifacts = (absolute_path, contrast_path, representation_path)
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "status": "pass",
            "claim_boundary": "image_only_morphology_concordance_not_pathology_accuracy",
            "primary_fold_rows": len(primary),
            "absolute_summary_rows": len(absolute),
            "control_contrast_rows": len(contrast_frame),
            "representation_rows": len(representation),
            "annotation_methods_with_positive_median_vs_raw_lsi": int(
                (contrast_frame["median_paired_delta_vs_raw_lsi"] > 0).sum()
            ),
            "annotation_methods_total_across_representations": len(contrast_frame),
            "annotation_methods_all_folds_above_matched_control": int(
                contrast_frame["all_folds_above_matched_control"].sum()
            ),
            "methods_with_positive_median_absolute_R2": int(
                absolute["median_absolute_R2_positive"].sum()
            ),
            "aggregation_controls_positive_in_all_representations": bool(
                (representation["matched_control_median_delta_R2_image"] > 0).all()
                and (representation["random_control_median_delta_R2_image"] > 0).all()
            ),
            "raw_lsi_negative_in_all_representations": bool(
                (representation["raw_lsi_median_delta_R2_image"] < 0).all()
            ),
            "artifacts": {
                path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
                for path in artifacts
            },
            "evaluation_evidence": {
                "path": config["evaluation_evidence"],
                "sha256": config["evaluation_evidence_sha256"],
            },
            "representation_pooling_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
        },
    )
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/g4_m3f_gse288178_malt_morphology_analysis.yaml",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    path = analyze(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
