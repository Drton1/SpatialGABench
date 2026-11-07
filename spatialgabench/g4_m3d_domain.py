"""Fixed-pipeline author-domain concordance evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import normalized_mutual_info_score

from spatialgabench.downstream.clustering_eval import (
    compute_clustering_metrics,
    prepare_gene_activity_for_clustering,
    prepare_lsi_for_clustering,
    run_leiden,
    run_lsi_neighbors_leiden,
    run_pca_neighbors_leiden,
)
from spatialgabench.software_wrappers.common import sha256_file


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in ("authorization", "construction_evidence", "environment_amendment", "input_h5ad", "labels"):
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {field}")
    authorization = _json(root / config["authorization"])
    construction = _json(root / config["construction_evidence"])
    authorized_task = config.get("authorized_task", "fixed_pipeline_author_cluster_concordance")
    if authorization.get("status") != "authorized" or authorization.get("authorized_task") != authorized_task:
        raise PermissionError(f"{config['dataset_id']} domain-evaluation authorization is not active")
    if authorization.get("authorized_method_panel") != config["methods"]:
        raise ValueError(f"Authorized {config['dataset_id']} domain method panel differs from config")
    if construction.get("status") != "pass" or construction.get("methods_passed") != len(config["methods"]) or construction.get("labels_accessed") is not False:
        raise ValueError(f"{config['dataset_id']} construction evidence does not pass")
    if config.get("label_based_parameter_selection") is not False or config.get("resolution_search_run") is not False:
        raise PermissionError(f"{config['dataset_id']} domain config permits label-based parameter selection")
    for amendment in config.get("additional_amendments", []):
        if sha256_file(root / amendment["path"]) != amendment["sha256"]:
            raise ValueError(f"Frozen amendment checksum mismatch: {amendment['path']}")
    if any(bool(config.get(field)) for field in ("holdout_accessed", "full_real_data_benchmark_run", "recommendation_policy_fit_run", "universal_ranking_run", "final_ranking_run")):
        raise PermissionError("A forbidden domain-evaluation flag is enabled")
    config["_path"] = path
    return config


def _feature_path(config: dict[str, Any], root: Path, method_id: str) -> Path:
    name = "gene_activity.h5ad" if method_id in config["gene_activity_methods"] else "raw_peak_lsi.h5ad"
    run_dir = root / config["construction_root"] / method_id
    manifest = _json(run_dir / "output_manifest.json")
    path = run_dir / name
    if manifest.get("status") != "pass" or manifest.get("labels_accessed") is not False or sha256_file(path) != manifest.get("artifacts", {}).get(name):
        raise ValueError(f"{config['dataset_id']} construction artifact contract failed: {method_id}")
    return path


def run(config_path: str | Path, method_id: str, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    if method_id not in config["methods"]:
        raise ValueError(f"Method outside {config['dataset_id']} domain panel: {method_id}")
    feature_path = _feature_path(config, root, method_id)
    feature = ad.read_h5ad(feature_path)
    raw = ad.read_h5ad(root / config["input_h5ad"], backed="r")
    if not feature.obs_names.equals(raw.obs_names):
        raise ValueError(f"{config['dataset_id']} observation mismatch: {method_id}")
    raw.file.close()
    prep_seed = int(config["preprocessing"]["preprocessing_seed"])
    if method_id in config["gene_activity_methods"]:
        prepared = prepare_gene_activity_for_clustering(
            feature, method_id=method_id,
            preprocessing_config=config["preprocessing"], random_seed=prep_seed,
        )
        run_pca_neighbors_leiden(
            prepared.adata, n_neighbors=int(config["clustering"]["n_neighbors"]),
            n_pcs=int(config["clustering"]["n_pcs"]), random_seed=prep_seed,
        )
    else:
        prepared = prepare_lsi_for_clustering(
            feature, method_id=method_id,
            expected_n_components=int(config["clustering"]["raw_lsi_components"]),
        )
        run_lsi_neighbors_leiden(
            prepared.adata, n_neighbors=int(config["clustering"]["n_neighbors"]),
            random_seed=prep_seed,
        )
    prediction_frames = []
    for seed in config["clustering"]["seeds"]:
        predicted = run_leiden(
            prepared.adata, resolution=float(config["clustering"]["leiden_resolution"]),
            random_seed=int(seed), key_added=f"leiden_seed_{seed}",
        )
        prediction_frames.append(pd.DataFrame({
            "barcode": feature.obs_names.astype(str), "method_id": method_id,
            "seed": int(seed), "predicted_cluster": predicted.to_numpy(dtype=str),
        }))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    labels = pd.read_csv(root / config["labels"], sep="\t", index_col=0)
    labels.index = labels.index.astype(str)
    if (
        labels.index.tolist() != feature.obs_names.astype(str).tolist()
        or labels[config["label_column"]].nunique() != int(config.get("expected_clusters", 11))
    ):
        raise ValueError(f"{config['dataset_id']} evaluation labels do not align with frozen observations")
    metrics = []
    for seed, frame in predictions.groupby("seed", sort=True):
        truth = labels.loc[frame.barcode, config["label_column"]]
        values = compute_clustering_metrics(truth, frame.predicted_cluster)
        metrics.append({
            "dataset_id": config["dataset_id"], "study_id": config["study_id"],
            "representation_id": config["representation_id"], "method_id": method_id,
            "seed": int(seed), "resolution": float(config["clustering"]["leiden_resolution"]),
            **values, "label_interpretation": config["label_interpretation"],
            "labels_used_for_parameter_selection": False, "status": "pass",
        })
    metrics_frame = pd.DataFrame(metrics)
    output = root / config["output_root"] / method_id
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "cluster_predictions.tsv.gz"
    metrics_path = output / "domain_metrics.tsv"
    prep_path = output / "preprocessing_summary.json"
    predictions.to_csv(predictions_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    metrics_frame.to_csv(metrics_path, sep="\t", index=False)
    _write_json(prep_path, {
        **prepared.summary, "preprocessing_seed": prep_seed,
        "n_neighbors": int(config["clustering"]["n_neighbors"]),
        "leiden_resolution": float(config["clustering"]["leiden_resolution"]),
        "clustering_seeds": config["clustering"]["seeds"],
        "labels_loaded_after_predictions": True,
        "labels_used_for_parameter_selection": False,
        "resolution_search_run": False,
    })
    manifest_path = output / "evaluation_manifest.json"
    artifacts = [predictions_path, metrics_path, prep_path]
    _write_json(manifest_path, {
        "schema_version": 1, "status": "pass", "dataset_id": config["dataset_id"],
        "method_id": method_id, "seeds_expected": len(config["clustering"]["seeds"]),
        "seeds_passed": len(metrics_frame),
        "artifacts": {path.name: sha256_file(path) for path in artifacts},
        "labels_loaded_after_predictions": True,
        "labels_used_for_parameter_selection": False, "resolution_search_run": False,
        "construction_outputs_mutated": False, "holdout_accessed": False,
        "full_real_data_benchmark_run": False, "ranking_run": False,
    })
    return manifest_path


def summarize(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    metric_frames = []
    resource_rows = []
    for method_id in config["methods"]:
        run_dir = root / config["output_root"] / method_id
        manifest, memory = _json(run_dir / "evaluation_manifest.json"), _json(run_dir / "process_tree_memory.json")
        valid = manifest.get("status") == "pass" and memory.get("status") == "pass" and all(
            (run_dir / name).is_file() and sha256_file(run_dir / name) == digest
            for name, digest in manifest.get("artifacts", {}).items()
        )
        if not valid or manifest.get("labels_used_for_parameter_selection") is not False:
            raise ValueError(f"{config['dataset_id']} domain evaluation evidence failed: {method_id}")
        frame = pd.read_csv(run_dir / "domain_metrics.tsv", sep="\t")
        if len(frame) != len(config["clustering"]["seeds"]) or not (frame.status == "pass").all():
            raise ValueError(f"{config['dataset_id']} domain seed panel failed: {method_id}")
        metric_frames.append(frame)
        resource_rows.append({
            "method_id": method_id, "runtime_seconds": float(memory["runtime_seconds"]),
            "peak_aggregate_pss_mb": float(memory["peak_aggregate_pss_mb"]), "status": "pass",
        })
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = metrics.groupby("method_id", sort=True).agg(
        seeds=("seed", "count"), median_ARI=("ARI", "median"), min_ARI=("ARI", "min"), max_ARI=("ARI", "max"),
        median_NMI=("NMI", "median"), median_AMI=("AMI", "median"),
        median_homogeneity=("homogeneity", "median"), median_completeness=("completeness", "median"),
        median_v_measure=("v_measure", "median"), median_predicted_clusters=("number_of_predicted_clusters", "median"),
    ).reset_index()
    control = summary[summary.method_id == config["control_method"]][["median_ARI"]].iloc[0, 0]
    summary["median_ARI_delta_vs_accessibility_matched_control"] = summary.median_ARI - float(control)
    ari_by_seed = metrics.pivot(index="seed", columns="method_id", values="ARI")
    matched_control = ari_by_seed[config["control_method"]]
    raw_lsi = ari_by_seed[config["raw_lsi_method"]]
    contrast_rows = []
    for method_id in config["methods"]:
        values = ari_by_seed[method_id]
        delta_control = values - matched_control
        delta_raw = values - raw_lsi
        contrast_rows.append({
            "method_id": method_id,
            "seed_runs": int(values.count()),
            "mean_ARI": float(values.mean()),
            "sd_ARI_across_algorithm_seeds": float(values.std(ddof=1)),
            "median_ARI": float(values.median()),
            "median_ARI_delta_vs_accessibility_matched_control": float(delta_control.median()),
            "min_ARI_delta_vs_accessibility_matched_control": float(delta_control.min()),
            "max_ARI_delta_vs_accessibility_matched_control": float(delta_control.max()),
            "all_seeds_above_accessibility_matched_control": bool((delta_control > 0).all()),
            "median_ARI_delta_vs_raw_peak_lsi": float(delta_raw.median()),
            "all_seeds_above_raw_peak_lsi": bool((delta_raw > 0).all()),
            "seed_role": "algorithmic_repeats_not_biological_replicates",
        })
    contrasts = pd.DataFrame(contrast_rows)
    labels = pd.read_csv(root / config["labels"], sep="\t", index_col=0)
    context_rows = []
    for context in config.get("context_columns", ["section", "sample", "age"]):
        context_rows.append({
            "context_variable": context, "unique_context_values": int(labels[context].nunique()),
            "normalized_mutual_information_with_author_cluster": float(normalized_mutual_info_score(labels.author_cluster.astype(str), labels[context].astype(str))),
        })
    context = pd.DataFrame(
        context_rows,
        columns=["context_variable", "unique_context_values", "normalized_mutual_information_with_author_cluster"],
    )
    output = root / config["output_root"]
    metrics_path = output / config.get("domain_seed_metrics_name", "gse214986_domain_seed_metrics.tsv")
    summary_path = output / config.get("domain_method_summary_name", "gse214986_domain_method_summary.tsv")
    resource_path = output / config.get("domain_resource_summary_name", "gse214986_domain_resource_summary.tsv")
    contrast_path = output / config.get("domain_contrasts_name", "gse214986_domain_descriptive_contrasts.tsv")
    context_path = output / config.get("domain_context_audit_name", "gse214986_author_label_context_audit.tsv")
    metrics.to_csv(metrics_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    pd.DataFrame(resource_rows).to_csv(resource_path, sep="\t", index=False)
    contrasts.to_csv(contrast_path, sep="\t", index=False)
    context.to_csv(context_path, sep="\t", index=False)
    artifacts = [metrics_path, summary_path, contrast_path, resource_path, context_path]
    contingency_context = config.get("contingency_context_column", "section")
    if contingency_context:
        contingency = pd.crosstab(labels.author_cluster, labels[contingency_context]).reset_index()
        contingency_path = output / config.get("domain_contingency_name", "gse214986_author_cluster_by_section.tsv")
        contingency.to_csv(contingency_path, sep="\t", index=False)
        artifacts.append(contingency_path)
    evidence_path = output / config.get("domain_evidence_name", "gse214986_domain_evaluation_evidence.json")
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": config.get("domain_milestone", "M3D_GSE214986_author_domain_concordance"),
        "status": "pass", "dataset_id": config["dataset_id"],
        "methods_expected": len(config["methods"]), "methods_passed": summary.method_id.nunique(),
        "seed_runs_expected": len(config["methods"]) * len(config["clustering"]["seeds"]),
        "seed_runs_passed": len(metrics), "primary_endpoint": config["primary_endpoint"],
        "label_interpretation": config["label_interpretation"],
        "artifacts": {path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts},
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "construction_evidence": {"path": config["construction_evidence"], "sha256": config["construction_evidence_sha256"]},
        "environment_amendment": {"path": config["environment_amendment"], "sha256": config["environment_amendment_sha256"]},
        "additional_amendments": config.get("additional_amendments", []),
        "seed_role": "algorithmic_repeats_not_biological_replicates",
        "labels_loaded_after_predictions": True, "labels_used_for_parameter_selection": False,
        "resolution_search_run": False, "construction_outputs_mutated": False,
        "holdout_accessed": False, "full_real_data_benchmark_run": False,
        "recommendation_policy_fit_run": False, "universal_ranking_run": False,
        "final_ranking_run": False, "next_milestone_authorized": False,
    }
    _write_json(evidence_path, evidence)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--config", default="configs/g4_m3d_gse214986_domain_evaluation.yaml")
    parser.add_argument("--method")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "run":
        if not args.method:
            parser.error("--method is required for run")
        path = run(args.config, args.method, args.root)
    else:
        path = summarize(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
