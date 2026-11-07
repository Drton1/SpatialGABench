"""Prepare leakage-separated GSE288178 mouse-brain fixed-bin inputs."""

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

from spatialgabench.software_wrappers.common import sha256_file


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in (
        "authorization",
        "previous_gate",
        "source_audit",
        "download_receipt",
        "source_metadata",
        "source_h5ad",
        "source_labels",
    ):
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {field}")
    authorization = _json(root / config["authorization"])
    if authorization.get("status") != "authorized" or authorization.get("authorized_task") != "fixed_bin_input_and_evaluation_label_split":
        raise PermissionError("GSE288178 mouse-brain input-preparation authorization is not active")
    if authorization.get("construction_methods_authorized") is not False or authorization.get("evaluation_authorized") is not False:
        raise PermissionError("GSE288178 input-preparation authorization boundary is invalid")
    if any(
        bool(config.get(field))
        for field in (
            "method_construction_run",
            "domain_metrics_run",
            "full_real_data_benchmark_run",
            "recommendation_policy_fit_run",
            "final_ranking_run",
        )
    ):
        raise PermissionError("A forbidden GSE288178 input-preparation flag is enabled")
    config["_path"] = path
    return config


def prepare(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    source = ad.read_h5ad(root / config["source_h5ad"], backed="r")
    expected_shape = (int(config["expected_observations"]), int(config["expected_features"]))
    if source.shape != expected_shape or not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError(f"GSE288178 source identity contract failed: {source.shape}")
    source_cluster_columns = [column for column in config["expected_source_cluster_columns"] if column in source.obs.columns]
    source_embedding_keys = [key for key in config["expected_source_embedding_keys"] if key in source.obsm]
    if source_cluster_columns != config["expected_source_cluster_columns"] or source_embedding_keys != config["expected_source_embedding_keys"]:
        raise ValueError("GSE288178 expected source leakage fields differ from the frozen audit")
    if "fragment_paired" not in source.obsm or "spatial" not in source.obsm:
        raise ValueError("GSE288178 source fragment/spatial backing is missing")

    labels = pd.read_csv(root / config["source_labels"], index_col=0)
    labels.index = labels.index.astype(str)
    if labels.index.tolist() != source.obs_names.astype(str).tolist() or not labels.index.is_unique:
        raise ValueError("GSE288178 source label barcodes do not exactly align")
    for column in ("coord", "x_coord", "y_coord", "cluster_use"):
        if not np.array_equal(source.obs[column].astype(str).to_numpy(), labels[column].astype(str).to_numpy()):
            raise ValueError(f"GSE288178 source label field does not align: {column}")
    if labels.cluster_use.nunique() != int(config["expected_clusters"]) or labels.isna().any().any():
        raise ValueError("GSE288178 source label dimensions differ from the frozen audit")

    feature_parts = source.var_names.to_series(index=source.var_names).str.extract(
        r"^(chr(?:[1-9]|1[0-9]|X)):(\d+)-(\d+)$"
    )
    if feature_parts.isna().any().any():
        raise ValueError("GSE288178 contains unparseable fixed-bin coordinates")
    var = pd.DataFrame(index=source.var_names.copy())
    var["chr"] = feature_parts[0].to_numpy(dtype=str)
    var["start"] = feature_parts[1].to_numpy(dtype=np.int64)
    var["end"] = feature_parts[2].to_numpy(dtype=np.int64)
    if not (var.end - var.start == int(config["expected_feature_width_bp"])).all():
        raise ValueError("GSE288178 feature widths differ from 500 bp")
    expected_chromosomes = {f"chr{value}" for value in range(1, 20)} | {"chrX"}
    if set(var.chr) != expected_chromosomes:
        raise ValueError("GSE288178 chromosome domain differs from standard mm10")

    matrix = source.X.to_memory().tocsr().astype(np.float32)
    if matrix.nnz != int(config["expected_matrix_nnz"]) or float(matrix.sum()) != float(config["expected_matrix_total"]):
        raise ValueError("GSE288178 source matrix totals differ from the frozen audit")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any() or not np.allclose(matrix.data, np.round(matrix.data)):
        raise ValueError("GSE288178 source matrix value contract failed")
    spatial = np.asarray(source.obsm["spatial"], dtype=np.float32)
    if not np.array_equal(spatial.astype(np.int64), labels[["x_coord", "y_coord"]].to_numpy(dtype=np.int64)):
        raise ValueError("GSE288178 source spatial coordinates do not align")
    source.file.close()

    obs = pd.DataFrame(
        {
            "x": spatial[:, 0],
            "y": spatial[:, 1],
            "input_role": "accessibility_and_spatial_coordinates_only",
        },
        index=pd.Index(labels.index, name="barcode"),
    )
    forbidden_obs = [
        column
        for column in obs.columns
        if any(token in str(column).lower() for token in config["forbidden_construction_obs_tokens"])
    ]
    if forbidden_obs or list(obs.columns) != config["allowed_construction_obs_columns"]:
        raise ValueError(f"Sanitized GSE288178 observation schema failed: {forbidden_obs}")

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    adata.uns.update(
        {
            "dataset_id": config["dataset_id"],
            "study_id": config["study_id"],
            "sample_id": config["sample_id"],
            "accession": config["accession"],
            "species": config["species"],
            "genome_build": config["genome_build"],
            "modality": "spatial_atac",
            "representation_id": config["representation_id"],
            "representation_source": "author_selected_fixed_500bp_bin_matrix",
            "input_track": config["input_track"],
            "comparison_scope": "unmatched_author_selected_fixed_bin_external_branch",
            "construction_evaluation_inputs_used": False,
            "evaluation_labels_present": False,
            "source_fragment_backing_preserved": False,
            "source_author_embeddings_preserved": False,
        }
    )
    evaluation_labels = labels[["cluster_use", "coord", "x_coord", "y_coord"]].rename(
        columns={"cluster_use": "author_cluster"}
    )
    evaluation_labels["author_cluster"] = evaluation_labels.author_cluster.astype(str)
    evaluation_labels.index.name = "barcode"

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    h5ad_path = output / "gse288178_mouse_brain_500bp_accessibility_only.h5ad"
    labels_path = output / "gse288178_mouse_brain_author_clusters_evaluation_only.tsv.gz"
    counts_path = output / "gse288178_mouse_brain_author_cluster_counts.tsv"
    validation_path = output / "input_preparation_validation.json"
    adata.write_h5ad(h5ad_path, compression="gzip")
    evaluation_labels.to_csv(labels_path, sep="\t", compression={"method": "gzip", "mtime": 0})
    (
        evaluation_labels.author_cluster.value_counts()
        .sort_index()
        .rename_axis("author_cluster")
        .reset_index(name="n_observations")
        .to_csv(counts_path, sep="\t", index=False)
    )
    validation = {
        "schema_version": 1,
        "status": "pass",
        "dataset_id": config["dataset_id"],
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "matrix_total": float(matrix.sum()),
        "fixed_bin_width_bp": int(config["expected_feature_width_bp"]),
        "barcode_order_matches_source_and_labels": True,
        "spatial_coordinates_match_source_and_labels": True,
        "author_cluster_rows": int(len(evaluation_labels)),
        "author_clusters": int(evaluation_labels.author_cluster.nunique()),
        "source_cluster_columns_observed_only_during_split": source_cluster_columns,
        "source_embedding_keys_observed_only_during_split": source_embedding_keys,
        "construction_obs_columns": list(obs.columns),
        "construction_obsm_keys": list(adata.obsm.keys()),
        "forbidden_construction_obs_columns": forbidden_obs,
        "source_fragment_backing_written_to_construction_input": False,
        "source_author_embeddings_written_to_construction_input": False,
        "author_clusters_written_to_construction_input": False,
        "label_split_performed": True,
        "holdout_accessed": False,
        "method_construction_run": False,
        "domain_metrics_run": False,
        "full_real_data_benchmark_run": False,
        "ranking_run": False,
    }
    _write_json(validation_path, validation)
    evidence_path = output / "gse288178_mouse_brain_input_preparation_evidence.json"
    artifacts = [h5ad_path, labels_path, counts_path, validation_path]
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3E_GSE288178_mouse_brain_input_preparation",
        "status": "pass",
        "dataset_id": config["dataset_id"],
        "study_id": config["study_id"],
        "n_observations": adata.n_obs,
        "n_fixed_500bp_bins": adata.n_vars,
        "matrix_nnz": int(matrix.nnz),
        "author_clusters": int(evaluation_labels.author_cluster.nunique()),
        "artifacts": {
            path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts
        },
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
        "source_h5ad": {"path": config["source_h5ad"], "sha256": config["source_h5ad_sha256"]},
        "source_labels": {"path": config["source_labels"], "sha256": config["source_labels_sha256"]},
        "label_interpretation": "author_defined_atac_cluster_not_independent_anatomy_histology_or_pathology_truth",
        "representation_scope": "unmatched_author_selected_fixed_500bp_bins",
        "construction_evaluation_inputs_used": False,
        "source_fragment_backing_preserved": False,
        "source_author_embeddings_preserved": False,
        "holdout_accessed": False,
        "method_construction_run": False,
        "domain_metrics_run": False,
        "full_real_data_benchmark_run": False,
        "final_ranking_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(evidence_path, evidence)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m3e_gse288178_mouse_brain_input_preparation.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(prepare(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
