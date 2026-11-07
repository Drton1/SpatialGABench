"""Correct one NU4 human SnapATAC2 gene-identity evaluation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.decomposition import PCA

from spatialgabench.nm_paired_depth_accuracy import canonicalize_unique_gene_names, sha256_path
from spatialgabench.nm_spatial_program_evaluation import (
    _align_rows,
    _canonicalize_ga_columns,
    _descriptive_bootstrap,
    _load_rna,
    morans_i,
    normalize_total_log1p,
    safe_correlation,
    symmetric_knn_graph,
)


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


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _yaml(config_path)
    source_config_path = root / config["source_config"]
    source_preflight_path = root / config["source_preflight"]
    source_completion_path = root / config["source_completion"]
    source_validation_path = root / config["source_validation"]
    source_config = _yaml(source_config_path)
    source_preflight = _json(source_preflight_path)
    source_completion = _json(source_completion_path)
    source_validation = _json(source_validation_path)
    if source_preflight.get("status") != "pass" or source_completion.get("status") != "pass" or source_validation.get("status") != "pass":
        raise ValueError("NU4C requires the accepted NU4 source evidence")
    target = config["target"]
    method_manifest_path = root / source_preflight["method_input_manifest"]["path"]
    methods = pd.read_csv(method_manifest_path, sep="\t")
    selected = methods.loc[
        methods.dataset_id.eq(target["dataset_id"])
        & methods.method_id.eq(target["method_id"])
        & methods.representation_id.eq(target["representation_id"])
    ]
    if len(selected) != 1:
        raise ValueError("NU4C target contract is not unique")
    row = selected.iloc[0]
    activity_path = root / row.gene_activity_h5ad
    if sha256_path(activity_path) != row.gene_activity_sha256:
        raise ValueError("NU4C target gene-activity output changed")
    fold_record = source_preflight["fold_assignments"][target["dataset_id"]]
    fold_path = root / fold_record["path"]
    if sha256_path(fold_path) != fold_record["sha256"]:
        raise ValueError("NU4C target fold assignment changed")
    sample = source_config["samples"][target["dataset_id"]]
    identity_rows = []
    for field in ("rna_matrix", "positions", "gene_annotation"):
        record = sample[field]
        path = root / record["path"]
        if sha256_path(path) != record["sha256"]:
            raise ValueError(f"NU4C source identity changed: {field}")
        identity_rows.append({"artifact_class": field, "path": record["path"], "sha256": record["sha256"], "rna_expression_values_read": False})
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    target_manifest_path = output / "target_contract_manifest.tsv"
    identity_path = output / "data_identity_manifest.tsv"
    selected.to_csv(target_manifest_path, sep="\t", index=False)
    pd.DataFrame(identity_rows).to_csv(identity_path, sep="\t", index=False)
    dependencies = [
        "spatialgabench/nm_u4_gene_identity_correction.py",
        "spatialgabench/nm_paired_depth_accuracy.py",
        "spatialgabench/nm_spatial_program_evaluation.py",
        "spatialgabench/g4_m3b_rna.py",
        "spatialgabench/g4_m3h_rna_replication.py",
        "spatialgabench/g5_h5c_evaluation.py",
    ]
    authorization = {
        "schema_version": 1,
        "authorization": "NM_U4C_HUMAN_SNAP_GENE_IDENTITY_CORRECTION",
        "milestone": config["milestone"],
        "authorization_date": "2026-07-30",
        "status": "authorized",
        "rna_expression_value_access_authorized": True,
        "target_contract_evaluation_authorized": True,
        "construction_or_method_rerun_authorized": False,
        "annotation_or_control_rerun_authorized": False,
        "labels_clusters_pathology_access_authorized": False,
        "policy_refit_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "source_config": {"path": config["source_config"], "sha256": sha256_path(source_config_path)},
        "source_preflight": {"path": config["source_preflight"], "sha256": sha256_path(source_preflight_path)},
        "source_completion": {"path": config["source_completion"], "sha256": sha256_path(source_completion_path)},
        "source_validation": {"path": config["source_validation"], "sha256": sha256_path(source_validation_path)},
        "target_contract_manifest": {"path": str(target_manifest_path.relative_to(root)), "sha256": sha256_path(target_manifest_path)},
        "data_identity_manifest": {"path": str(identity_path.relative_to(root)), "sha256": sha256_path(identity_path)},
        "fold_assignment": fold_record,
        "dependencies": [{"path": path, "sha256": sha256_path(root / path)} for path in dependencies],
    }
    authorization_path = root / "governance/authorizations/NM_U4C_HUMAN_SNAP_GENE_IDENTITY_CORRECTION_AUTHORIZATION_2026-07-30.json"
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_preflight",
        "status": "pass",
        "target_contracts": 1,
        "rna_expression_values_read": False,
        "gene_activity_values_read": False,
        "construction_or_method_rerun": False,
        "annotation_or_control_rerun": False,
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "target_contract_manifest": authorization["target_contract_manifest"],
        "data_identity_manifest": authorization["data_identity_manifest"],
    }
    evidence_path = output / "preflight_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def _validate(config_path: Path, preflight_path: Path, authorization_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config, preflight, authorization = _yaml(config_path), _json(preflight_path), _json(authorization_path)
    if preflight.get("status") != "pass" or preflight.get("rna_expression_values_read") is not False:
        raise ValueError("NU4C blind preflight is invalid")
    required = {
        "status": "authorized",
        "milestone": config["milestone"],
        "rna_expression_value_access_authorized": True,
        "target_contract_evaluation_authorized": True,
        "construction_or_method_rerun_authorized": False,
        "annotation_or_control_rerun_authorized": False,
        "labels_clusters_pathology_access_authorized": False,
        "policy_refit_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise PermissionError(f"NU4C authorization boundary failed: {field}")
    paths = {
        "config": config_path,
        "source_config": root / config["source_config"],
        "source_preflight": root / config["source_preflight"],
        "source_completion": root / config["source_completion"],
        "source_validation": root / config["source_validation"],
        "target_contract_manifest": root / preflight["target_contract_manifest"]["path"],
        "data_identity_manifest": root / preflight["data_identity_manifest"]["path"],
    }
    for name, path in paths.items():
        record = authorization[name]
        if record["path"] != str(path.relative_to(root)) or record["sha256"] != sha256_path(path):
            raise ValueError(f"NU4C authorization binding mismatch: {name}")
    for record in authorization["dependencies"]:
        if sha256_path(root / record["path"]) != record["sha256"]:
            raise ValueError(f"NU4C dependency changed: {record['path']}")
    return config, _yaml(paths["source_config"]), _json(paths["source_preflight"])


def evaluate(config_path: str | Path, preflight_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    preflight_path, authorization_path = Path(preflight_path).resolve(), Path(authorization_path).resolve()
    config, source_config, source_preflight = _validate(config_path, preflight_path, authorization_path, root)
    target = config["target"]
    sample = source_config["samples"][target["dataset_id"]]
    method_manifest = pd.read_csv(root / source_preflight["method_input_manifest"]["path"], sep="\t")
    method = method_manifest.loc[
        method_manifest.dataset_id.eq(target["dataset_id"])
        & method_manifest.method_id.eq(target["method_id"])
        & method_manifest.representation_id.eq(target["representation_id"])
    ].iloc[0]
    fold_table = pd.read_csv(root / source_preflight["fold_assignments"][target["dataset_id"]]["path"], sep="\t")
    barcodes = fold_table.barcode.astype(str).tolist()
    rna_counts, rna_barcodes, rna_gene_ids, protein_coding, unique_names = _load_rna(sample, root)
    unique_names = canonicalize_unique_gene_names(unique_names)
    rna_counts = _align_rows(rna_counts, rna_barcodes, barcodes)
    settings = source_config["rna_programs"]
    rna_normalized = normalize_total_log1p(rna_counts, float(settings["target_sum"]))
    activity = ad.read_h5ad(root / method.gene_activity_h5ad)
    selected = activity.obs_names.astype(str).get_indexer(barcodes)
    if np.any(selected < 0):
        raise ValueError("NU4C activity observations do not match")
    raw = activity.X.tocsr() if sparse.issparse(activity.X) else sparse.csr_matrix(activity.X)
    normalized = normalize_total_log1p(raw[selected, :], float(settings["target_sum"]))
    canonical, activity_gene_ids, _ = _canonicalize_ga_columns(normalized, activity, unique_names)
    activity_index = {gene_id: index for index, gene_id in enumerate(activity_gene_ids)}
    detail_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    for fold in range(int(source_config["folds"]["count"])):
        test = np.flatnonzero(fold_table.fold.to_numpy() == fold)
        train = np.flatnonzero(fold_table.fold.to_numpy() != fold)
        detection = np.asarray((rna_counts[train, :] > 0).sum(axis=0)).ravel()
        threshold = int(np.ceil(float(settings["minimum_detected_training_fraction"]) * len(train)))
        eligible = np.flatnonzero(protein_coding & (detection >= threshold))
        train_dense = rna_normalized[train, :][:, eligible].toarray().astype(np.float32, copy=False)
        test_dense = rna_normalized[test, :][:, eligible].toarray().astype(np.float32, copy=False)
        pca = PCA(n_components=int(settings["components"]), svd_solver=str(settings["svd_solver"]), random_state=int(settings["random_seed"])).fit(train_dense)
        rna_scores = pca.transform(test_dense)
        graph = symmetric_knn_graph(fold_table.loc[test, ["row_coordinate", "column_coordinate"]].to_numpy(dtype=float), int(source_config["spatial_metric"]["neighbors"]))
        mapped = np.asarray([activity_index.get(str(gene_id), -1) for gene_id in rna_gene_ids[eligible]], dtype=np.int64)
        present = np.flatnonzero(mapped >= 0)
        ga_scores = np.zeros((len(test), int(settings["components"])), dtype=np.float64)
        if len(present):
            ga_scores = canonical[test, :][:, mapped[present]] @ pca.components_[:, present].transpose()
            ga_scores = np.asarray(ga_scores)
        ga_scores -= pca.mean_ @ pca.components_.transpose()
        for component in range(int(settings["components"])):
            rna_score, ga_score = rna_scores[:, component], ga_scores[:, component]
            detail_rows.append({
                "dataset_id": target["dataset_id"], "study_id": method.study_id,
                "study_family": method.study_family, "species": method.species,
                "genome_build": method.genome_build, "representation_id": method.representation_id,
                "input_track": method.input_track, "method_id": method.method_id,
                "method_category": method.method_category, "fold": fold,
                "component": component + 1, "test_observations": len(test),
                "eligible_training_genes": len(eligible),
                "activity_present_gene_fraction": len(present) / len(eligible),
                "heldout_component_spearman": safe_correlation(rna_score, ga_score, "spearman"),
                "heldout_component_pearson": safe_correlation(rna_score, ga_score, "pearson"),
                "rna_morans_i": morans_i(rna_score, graph),
                "activity_morans_i": morans_i(ga_score, graph),
                "absolute_morans_i_difference": abs(morans_i(rna_score, graph) - morans_i(ga_score, graph)),
                "status": "pass",
            })
        gene_rows.append({"dataset_id": target["dataset_id"], "study_id": method.study_id, "fold": fold, "training_observations": len(train), "test_observations": len(test), "eligible_training_genes": len(eligible), "components": int(settings["components"])})
    detail = pd.DataFrame(detail_rows)
    if len(detail) != int(config["expected_target_detail_rows"]):
        raise ValueError("NU4C target detail dimensions changed")
    score = detail.heldout_component_spearman.to_numpy(dtype=float)
    pearson = detail.heldout_component_pearson.to_numpy(dtype=float)
    moran = detail.absolute_morans_i_difference.to_numpy(dtype=float)
    aggregation = source_config["aggregation"]
    score_low, score_high = _descriptive_bootstrap(score, int(aggregation["bootstrap_draws_descriptive_only"]), int(aggregation["bootstrap_seed"]))
    moran_low, moran_high = _descriptive_bootstrap(moran, int(aggregation["bootstrap_draws_descriptive_only"]), int(aggregation["bootstrap_seed"]) + 1000)
    summary = pd.DataFrame([{
        "dataset_id": target["dataset_id"], "study_id": method.study_id,
        "study_family": method.study_family, "species": method.species,
        "genome_build": method.genome_build, "representation_id": method.representation_id,
        "input_track": method.input_track, "method_id": method.method_id,
        "method_category": method.method_category, "fold_component_units": len(detail),
        "median_heldout_component_spearman": float(np.median(score)),
        "median_heldout_component_pearson": float(np.median(pearson)),
        "median_absolute_morans_i_difference": float(np.median(moran)),
        "descriptive_bootstrap_score_q025": score_low, "descriptive_bootstrap_score_q975": score_high,
        "descriptive_bootstrap_morans_difference_q025": moran_low,
        "descriptive_bootstrap_morans_difference_q975": moran_high,
        "median_activity_present_gene_fraction": float(detail.activity_present_gene_fraction.median()),
        "status": "pass",
    }])
    old_detail = pd.read_csv(root / config["source_detail"], sep="\t")
    old_summary = pd.read_csv(root / config["source_summary"], sep="\t")
    target_mask_detail = old_detail.dataset_id.eq(target["dataset_id"]) & old_detail.method_id.eq(target["method_id"]) & old_detail.representation_id.eq(target["representation_id"])
    target_mask_summary = old_summary.dataset_id.eq(target["dataset_id"]) & old_summary.method_id.eq(target["method_id"]) & old_summary.representation_id.eq(target["representation_id"])
    if int(target_mask_detail.sum()) != 50 or int(target_mask_summary.sum()) != 1:
        raise ValueError("NU4C old target dimensions changed")
    corrected_detail = pd.concat([old_detail.loc[~target_mask_detail], detail], ignore_index=True).sort_values(["dataset_id", "input_track", "representation_id", "method_id", "fold", "component"], kind="mergesort")
    corrected_summary = pd.concat([old_summary.loc[~target_mask_summary], summary], ignore_index=True).sort_values(["dataset_id", "input_track", "representation_id", "method_id"], kind="mergesort")
    deltas = pd.read_csv(root / config["source_deltas"], sep="\t")
    genes = pd.read_csv(root / config["source_gene_eligibility"], sep="\t")
    output = root / config["output_root"]
    detail_path, summary_path = output / "fold_component_metrics.tsv.gz", output / "contract_summary.tsv"
    delta_path, genes_path = output / "matched_control_deltas.tsv", output / "fold_gene_eligibility.tsv"
    target_detail_path, target_summary_path = output / "corrected_target_fold_component_metrics.tsv", output / "corrected_target_contract_summary.tsv"
    corrected_detail.to_csv(detail_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    corrected_summary.to_csv(summary_path, sep="\t", index=False)
    deltas.to_csv(delta_path, sep="\t", index=False)
    genes.to_csv(genes_path, sep="\t", index=False)
    detail.to_csv(target_detail_path, sep="\t", index=False)
    summary.to_csv(target_summary_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "milestone": config["milestone"], "status": "pass",
        "method_contracts": len(corrected_summary), "fold_component_rows": len(corrected_detail),
        "matched_control_deltas": len(deltas), "corrected_contracts": 1,
        "unaffected_contracts_preserved": len(corrected_summary) - 1,
        "rna_used_for_construction": False, "construction_or_method_rerun": False,
        "annotation_or_control_rerun": False, "labels_clusters_pathology_accessed": False,
        "policy_refit_run": False, "input_tracks_pooled": False,
        "universal_ranking_run": False, "final_ranking_run": False,
        "artifacts": {
            "fold_component_metrics.tsv.gz": {"path": str(detail_path.relative_to(root)), "sha256": sha256_path(detail_path)},
            "contract_summary.tsv": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
            "matched_control_deltas.tsv": {"path": str(delta_path.relative_to(root)), "sha256": sha256_path(delta_path)},
            "fold_gene_eligibility.tsv": {"path": str(genes_path.relative_to(root)), "sha256": sha256_path(genes_path)},
            "corrected_target_fold_component_metrics.tsv": {"path": str(target_detail_path.relative_to(root)), "sha256": sha256_path(target_detail_path)},
            "corrected_target_contract_summary.tsv": {"path": str(target_summary_path.relative_to(root)), "sha256": sha256_path(target_summary_path)},
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "preflight": {"path": str(preflight_path.relative_to(root)), "sha256": sha256_path(preflight_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "evaluate"])
    parser.add_argument("--config", default="configs/nm_u4_human_snap_gene_identity_correction.yaml")
    parser.add_argument("--preflight", default="results/v1/nm_upgrade/nu4_spatial_programs_v2/preflight_evidence.json")
    parser.add_argument("--authorization", default="governance/authorizations/NM_U4C_HUMAN_SNAP_GENE_IDENTITY_CORRECTION_AUTHORIZATION_2026-07-30.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = preflight(args.config, args.root) if args.command == "preflight" else evaluate(args.config, args.preflight, args.authorization, args.root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
