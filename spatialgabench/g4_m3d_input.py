"""Prepare a leakage-separated GSE214986 author-peak development input."""

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

from spatialgabench.io import (
    _load_metadata,
    _metadata_with_barcode_index,
    _parse_peak_coordinates,
    _read_10x_like_h5,
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
    for field in ("authorization", "previous_gate", "schema_audit", "source_h5", "source_metadata"):
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {field}")
    authorization = _json(root / config["authorization"])
    audit = _json(root / config["schema_audit"])
    if authorization.get("status") != "authorized" or authorization.get("authorized_task") != "author_peak_input_and_evaluation_label_split":
        raise PermissionError("GSE214986 input-preparation authorization is not active")
    if authorization.get("construction_methods_authorized") is not False or authorization.get("evaluation_authorized") is not False:
        raise PermissionError("Input-preparation authorization boundary is invalid")
    if audit.get("schema_status") != "pass" or audit.get("matrix_metadata_barcodes_equal") is not True:
        raise ValueError("GSE214986 source schema audit does not pass")
    if any(bool(config.get(field)) for field in ("method_construction_run", "domain_metrics_run", "full_real_data_benchmark_run", "recommendation_policy_fit_run", "final_ranking_run")):
        raise PermissionError("A forbidden input-preparation flag is enabled")
    config["_path"] = path
    return config


def prepare(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    X, barcodes, var, orientation = _read_10x_like_h5(root / config["source_h5"])
    metadata = _metadata_with_barcode_index(_load_metadata(root / config["source_metadata"]), barcodes)
    if X.shape != (int(config["expected_observations"]), int(config["expected_features"])) or X.nnz != int(config["expected_matrix_nnz"]):
        raise ValueError(f"GSE214986 matrix dimensions differ from frozen contract: {X.shape}, nnz={X.nnz}")
    if not sparse.isspmatrix_csr(X) or X.data.size == 0 or not np.isfinite(X.data).all() or (X.data < 0).any() or not np.allclose(X.data, np.round(X.data)):
        raise ValueError("GSE214986 matrix value contract failed")
    var = _parse_peak_coordinates(var)
    if not var[["chr", "start", "end"]].notna().all().all():
        raise ValueError("GSE214986 contains unparseable author peaks")
    var["start"] = var["start"].astype(np.int64)
    var["end"] = var["end"].astype(np.int64)
    if (var.start < 0).any() or (var.end <= var.start).any():
        raise ValueError("GSE214986 contains invalid author peak intervals")
    if set(var.chr) != {f"chr{value}" for value in range(1, 20)} | {"chrX", "chrY"}:
        raise ValueError("GSE214986 chromosome domain differs from frozen mm10 standard chromosomes")
    labels = metadata[[config["evaluation_label_column"], *config["evaluation_context_columns"]]].copy()
    labels.index.name = "barcode"
    labels = labels.rename(columns={config["evaluation_label_column"]: "author_cluster"})
    if labels.author_cluster.isna().any() or labels.author_cluster.astype(str).str.strip().eq("").any():
        raise ValueError("GSE214986 author clusters contain missing values")
    if labels.author_cluster.nunique() != int(config["expected_clusters"]) or labels.section.nunique() != int(config["expected_sections"]):
        raise ValueError("GSE214986 label/context dimensions differ from frozen contract")
    obs = metadata[["x", "y", "pixel_x", "pixel_y"]].copy()
    obs["input_role"] = "accessibility_and_spatial_coordinates_only"
    obs.index = pd.Index(barcodes, name="barcode")
    forbidden = [
        column for column in obs.columns
        if any(token in str(column).lower() for token in config["forbidden_construction_obs_tokens"])
    ]
    if forbidden or list(obs.columns) != config["allowed_construction_obs_columns"]:
        raise ValueError(f"Sanitized observation schema failed: forbidden={forbidden}, columns={list(obs.columns)}")
    matrix = X.astype(np.float32)
    matrix.eliminate_zeros()
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obs_names = pd.Index(barcodes)
    adata.var_names = var.index.astype(str)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns.update({
        "dataset_id": config["dataset_id"], "study_id": config["study_id"],
        "sample_id": config["sample_id"], "accession": config["accession"],
        "species": config["species"], "genome_build": config["genome_build"],
        "modality": "spatial_atac", "representation_id": config["representation_id"],
        "representation_source": "author_processed_peak_matrix", "input_track": config["input_track"],
        "comparison_scope": "unmatched_author_peak_external_branch",
        "construction_evaluation_inputs_used": False,
        "evaluation_labels_present": False,
        "source_orientation": orientation,
    })
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    h5ad_path = output / "gse214986_author_peak_accessibility_only.h5ad"
    labels_path = output / "gse214986_author_clusters_evaluation_only.tsv.gz"
    counts_path = output / "gse214986_author_cluster_counts.tsv"
    validation_path = output / "input_preparation_validation.json"
    adata.write_h5ad(h5ad_path, compression="gzip")
    labels.to_csv(labels_path, sep="\t", compression={"method": "gzip", "mtime": 0})
    labels.author_cluster.value_counts().sort_index().rename_axis("author_cluster").reset_index(name="n_observations").to_csv(counts_path, sep="\t", index=False)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    metadata_counts = metadata["nCount_peaks"].to_numpy(dtype=np.float64)
    validation = {
        "schema_version": 1, "status": "pass", "dataset_id": config["dataset_id"],
        "matrix_shape": list(matrix.shape), "matrix_nnz": int(matrix.nnz),
        "matrix_total": float(matrix.sum()), "barcode_order_matches_metadata": True,
        "per_observation_peak_counts_match_metadata": bool(np.array_equal(row_sums, metadata_counts)),
        "peak_coordinates_parseable": int(var.shape[0]), "peak_coordinates_total": int(var.shape[0]),
        "author_cluster_rows": int(len(labels)), "author_clusters": int(labels.author_cluster.nunique()),
        "sections": int(labels.section.nunique()), "construction_obs_columns": list(obs.columns),
        "forbidden_construction_obs_columns": forbidden,
        "rna_derived_metadata_written_to_construction_input": False,
        "author_clusters_written_to_construction_input": False,
        "label_split_performed": True, "holdout_accessed": False,
        "method_construction_run": False, "domain_metrics_run": False,
        "full_real_data_benchmark_run": False, "ranking_run": False,
    }
    if validation["per_observation_peak_counts_match_metadata"] is not True:
        raise ValueError("GSE214986 matrix row sums do not match metadata nCount_peaks")
    _write_json(validation_path, validation)
    manifest_path = output / "gse214986_input_preparation_evidence.json"
    artifacts = [h5ad_path, labels_path, counts_path, validation_path]
    evidence = {
        "schema_version": 1, "gate": "G4", "milestone": "M3D_GSE214986_input_preparation",
        "status": "pass", "dataset_id": config["dataset_id"], "study_id": config["study_id"],
        "n_observations": adata.n_obs, "n_author_peaks": adata.n_vars, "matrix_nnz": int(matrix.nnz),
        "author_clusters": int(labels.author_cluster.nunique()), "sections": int(labels.section.nunique()),
        "artifacts": {path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts},
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
        "source_h5": {"path": config["source_h5"], "sha256": config["source_h5_sha256"]},
        "source_metadata": {"path": config["source_metadata"], "sha256": config["source_metadata_sha256"]},
        "construction_evaluation_inputs_used": False, "holdout_accessed": False,
        "method_construction_run": False, "domain_metrics_run": False,
        "full_real_data_benchmark_run": False, "final_ranking_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(manifest_path, evidence)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m3d_gse214986_input_preparation.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(prepare(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

