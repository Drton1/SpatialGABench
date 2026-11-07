"""Construct a governed unmatched author representation method panel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from spatialgabench.g4_m3c_simulation import _mapping
from spatialgabench.gene_activity.distal import estimate_distal_link_count, load_tss_table
from spatialgabench.gene_activity.interval_aggregation import (
    compute_gene_activity,
    load_bed_intervals,
    load_gene_metadata,
    parse_peak_coordinates,
)
from spatialgabench.software_wrappers.common import (
    ensure_standard_gene_activity_anndata,
    sha256_file,
    validate_standard_output,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in ("authorization", "input_evidence", "input_h5ad"):
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {field}")
    for artifact in config["references"].values():
        if sha256_file(root / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {artifact['path']}")
    authorization = _json(root / config["authorization"])
    evidence = _json(root / config["input_evidence"])
    authorized_task = config.get("authorized_task", "author_peak_method_construction_only")
    if authorization.get("status") != "authorized" or authorization.get("authorized_task") != authorized_task:
        raise PermissionError(f"{config['dataset_id']} construction authorization is not active")
    if authorization.get("authorized_method_panel") != config["methods"] or authorization.get("evaluation_labels_allowed") is not False:
        raise PermissionError(f"{config['dataset_id']} construction authorization boundary is invalid")
    if evidence.get("status") != "pass" or evidence.get("construction_evaluation_inputs_used") is not False:
        raise ValueError(f"{config['dataset_id']} prepared input evidence does not pass")
    if any(bool(config.get(field)) for field in ("domain_evaluation_run", "full_real_data_benchmark_run", "recommendation_policy_fit_run", "final_ranking_run")):
        raise PermissionError("A forbidden construction flag is enabled")
    config["_path"] = path
    return config


def _input(config: dict[str, Any], root: Path) -> ad.AnnData:
    adata = ad.read_h5ad(root / config["input_h5ad"])
    forbidden = [
        column for column in adata.obs.columns
        if any(token in str(column).lower() for token in config["forbidden_construction_inputs"])
    ]
    if forbidden or adata.uns.get("construction_evaluation_inputs_used") is not False or adata.uns.get("evaluation_labels_present") is not False:
        raise ValueError(f"Prepared input leaks evaluation fields: {forbidden}")
    if adata.uns.get("dataset_id") != config["dataset_id"] or adata.uns.get("genome_build") != config.get("genome_build", "mm10"):
        raise ValueError("Prepared input identity/build mismatch")
    if not sparse.issparse(adata.X) or not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Prepared input sparse/identity contract failed")
    return adata


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    adata = _input(config, root)
    peaks = parse_peak_coordinates(adata)
    genes = load_gene_metadata(root / config["references"]["genes"]["path"])
    tss = load_tss_table(root / config["references"]["tss"]["path"], genes)
    estimate = estimate_distal_link_count(
        peaks, tss,
        distal_window_bp=int(config["parameters"]["distal_window_bp"]),
        promoter_exclusion_window_bp=int(config["parameters"]["promoter_window_bp"]),
    )
    limits = config["resource_preflight"]
    passed = (
        estimate["estimated_distal_links"] <= int(limits["maximum_estimated_distal_links"])
        and estimate["estimated_mapping_memory_mb"] <= float(limits["maximum_estimated_mapping_memory_mb"])
        and estimate["estimated_links_table_memory_mb"] <= float(limits["maximum_estimated_links_table_memory_mb"])
    )
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "mapping_resource_preflight.json"
    payload = {
        "schema_version": 1, "status": "pass" if passed else "blocked",
        "dataset_id": config["dataset_id"], "n_observations": adata.n_obs,
        "n_reference_genes": len(genes),
        **estimate, "limits": limits, "labels_accessed": False,
        "holdout_accessed": False, "domain_evaluation_run": False,
    }
    payload[config.get("input_feature_count_field", "n_author_peaks")] = adata.n_vars
    _write_json(path, payload)
    if not passed:
        raise RuntimeError(f"{config['dataset_id']} mapping resource preflight exceeded frozen limits")
    return path


def construct(config_path: str | Path, method_id: str, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    if method_id not in config["methods"]:
        raise ValueError(f"Method outside authorized {config['dataset_id']} panel: {method_id}")
    preflight_path = root / config["output_root"] / "mapping_resource_preflight.json"
    if _json(preflight_path).get("status") != "pass":
        raise ValueError(f"{config['dataset_id']} mapping resource preflight does not pass")
    adata = _input(config, root)
    output = root / config["output_root"] / method_id
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "input_validation.json"
    _write_json(validation_path, {
        "schema_version": 1, "status": "pass", "dataset_id": config["dataset_id"],
        "method_id": method_id, "input_path": config["input_h5ad"],
        "input_sha256": config["input_h5ad_sha256"],
        "construction_inputs_used": ["accessibility_matrix", "gene_annotation", "spatial_coordinates"] if method_id != "raw_peak_lsi_baseline" else ["accessibility_matrix", "spatial_coordinates"],
        "forbidden_inputs_used": [], "labels_accessed": False, "clusters_accessed": False,
        "rna_accessed": False, "holdout_accessed": False,
    })
    started = time.perf_counter()
    if method_id == "raw_peak_lsi_baseline":
        from spatialgabench.gene_activity.raw_peak_lsi import (
            binarize_peak_matrix,
            compute_tfidf_binary,
            run_truncated_svd,
            write_lsi_h5ad,
        )
        binary = binarize_peak_matrix(adata.X)
        tfidf, idf, accessible = compute_tfidf_binary(binary)
        components = min(int(config["raw_lsi"]["n_components"]), min(tfidf.shape) - 1)
        embedding, variance, singular = run_truncated_svd(tfidf, components, int(config["raw_lsi"]["seed"]))
        output_path = output / "raw_peak_lsi.h5ad"
        result = write_lsi_h5ad(
            input_adata=adata, embedding=embedding, output_path=output_path,
            dataset_id=config["dataset_id"], method_id=method_id, genome_build=config.get("genome_build", "mm10"),
            input_peak_h5ad=config["input_h5ad"], n_components=components,
            random_seed=int(config["raw_lsi"]["seed"]), explained_variance_ratio=variance, singular_values=singular,
        )
        result.uns.update({
            "run_id": f"{config.get('run_id_prefix', 'g4_m3d')}__{method_id}", "study_id": config["study_id"],
            "sample_id": config["sample_id"], "representation_id": config["representation_id"],
            "input_track": config.get("input_track", "P"),
            "comparison_scope": config.get("comparison_scope", "unmatched_author_peak_external_branch"),
            "construction_leakage_inputs": "none", "authorization_sha256": config["authorization_sha256"],
        })
        result.write_h5ad(output_path, compression="gzip")
        checks = {
            "output_type_embedding": result.uns["output_type"] == "cell_embedding",
            "not_gene_activity": result.uns["is_gene_activity"] is False,
            "observations_match": result.obs_names.equals(adata.obs_names),
            "observation_count": result.n_obs == adata.n_obs,
            "observation_ids_unique": result.obs_names.is_unique,
            "spatial_preserved": np.allclose(result.obsm["spatial"], adata.obsm["spatial"]),
            "lsi_present": "X_lsi" in result.obsm,
            "lsi_shape": result.obsm["X_lsi"].shape == (adata.n_obs, components),
            "lsi_finite": np.isfinite(result.obsm["X_lsi"]).all(),
            "variance_finite": np.isfinite(variance).all(),
            "singular_values_finite": np.isfinite(singular).all(),
            "binary_sparse": sparse.issparse(binary), "tfidf_sparse": sparse.issparse(tfidf),
            "tfidf_finite": np.isfinite(tfidf.data).all(),
            "accessible_counts_nonnegative": bool((accessible >= 0).all()),
            "idf_finite": np.isfinite(idf).all(),
            "input_hash_matches": sha256_file(root / config["input_h5ad"]) == config["input_h5ad_sha256"],
            "leakage_inputs_absent": result.uns["construction_leakage_inputs"] == "none",
        }
        contract = pd.DataFrame([{"method_id": method_id, "check_name": key, "status": "pass" if value else "fail"} for key, value in checks.items()])
        extra_artifacts: list[Path] = []
        n_output_features = components
        output_nnz = int(np.count_nonzero(embedding))
        mapping_nnz = 0
    else:
        peaks = parse_peak_coordinates(adata)
        genes = load_gene_metadata(root / config["references"]["genes"]["path"])
        promoters = load_bed_intervals(root / config["references"]["promoters"]["path"], "promoter")
        gene_body = load_bed_intervals(root / config["references"]["gene_body"]["path"], "gene_body")
        tss = load_tss_table(root / config["references"]["tss"]["path"], genes)
        mapping, matching_qc = _mapping(method_id, adata, peaks, genes, promoters, gene_body, tss, config["parameters"])
        activity = compute_gene_activity(adata.X, mapping.peak_gene_matrix)
        var = mapping.genes.copy()
        var.index = var.gene_id.astype(str)
        output_path = output / "gene_activity.h5ad"
        result = ensure_standard_gene_activity_anndata(
            input_adata=adata, X=activity, var=var, method_id=method_id,
            software_name="SpatialGABench", software_version="1.0.0.dev0",
            wrapper_version=config.get("wrapper_version", "g4_m3d_author_peak_v1"),
            genome_build=config.get("genome_build", "mm10"),
            normalization_status="raw_accessibility_aggregation", output_path=output_path,
            extra_uns={
                "run_id": f"{config.get('run_id_prefix', 'g4_m3d')}__{method_id}", "study_id": config["study_id"],
                "sample_id": config["sample_id"], "representation_id": config["representation_id"],
                "input_track": config.get("input_track", "P"),
                "comparison_scope": config.get("comparison_scope", "unmatched_author_peak_external_branch"),
                "subset_policy": config.get("subset_policy", "all_author_peak_observations"),
                "construction_input_classes": ["accessibility_matrix", "gene_annotation"],
                "construction_leakage_inputs": "none", "input_validation_sha256": sha256_file(validation_path),
                "environment_lock_sha256": sha256_file(root / "envs/core-resolved.yaml"),
                "authorization_sha256": config["authorization_sha256"],
            },
        )
        contract = validate_standard_output(result, adata, method_id)
        mapping_path = output / "peak_gene_mapping.npz"
        links_path = output / "predicted_peak_gene_links.tsv.gz"
        sparse.save_npz(mapping_path, mapping.peak_gene_matrix.tocsr(), compressed=True)
        mapping.links.to_csv(links_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
        extra_artifacts = [mapping_path, links_path]
        if matching_qc is not None:
            matching_path = output / "accessibility_matching_qc.tsv"
            matching_qc.to_csv(matching_path, sep="\t", index=False)
            extra_artifacts.append(matching_path)
        n_output_features = result.n_vars
        output_nnz = result.X.nnz
        mapping_nnz = mapping.peak_gene_matrix.nnz
    contract_path = output / "contract_validation.tsv"
    contract.to_csv(contract_path, sep="\t", index=False)
    if not (contract.status == "pass").all() or len(contract) != 18:
        raise ValueError(f"{config['dataset_id']} construction contract failed: {method_id}")
    runtime = time.perf_counter() - started
    qc_path = output / "construction_qc.tsv"
    pd.DataFrame([{
        "dataset_id": config["dataset_id"], "method_id": method_id,
        "n_observations": adata.n_obs, "n_input_features": adata.n_vars,
        "n_output_features": n_output_features, "output_nnz": output_nnz,
        "mapping_nnz": mapping_nnz, "runtime_seconds": runtime, "status": "pass",
    }]).to_csv(qc_path, sep="\t", index=False)
    artifacts = [validation_path, output_path, contract_path, qc_path, *extra_artifacts]
    manifest_path = output / "output_manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1, "status": "pass", "dataset_id": config["dataset_id"],
        "method_id": method_id, "artifacts": {path.name: sha256_file(path) for path in artifacts},
        "contract_checks_passed": 18, "contract_checks_total": 18,
        "runtime_seconds": runtime, "forbidden_inputs_used": [],
        "labels_accessed": False, "clusters_accessed": False, "rna_accessed": False,
        "holdout_accessed": False, "domain_evaluation_run": False,
        "full_real_data_benchmark_run": False, "ranking_run": False,
    })
    return manifest_path


def summarize(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    preflight_path = root / config["output_root"] / "mapping_resource_preflight.json"
    if _json(preflight_path).get("status") != "pass":
        raise ValueError(f"{config['dataset_id']} mapping preflight does not pass")
    rows = []
    for method_id in config["methods"]:
        run_dir = root / config["output_root"] / method_id
        manifest_path = run_dir / "output_manifest.json"
        memory_path = run_dir / "process_tree_memory.json"
        manifest, memory = _json(manifest_path), _json(memory_path)
        artifacts_valid = all((run_dir / name).is_file() and sha256_file(run_dir / name) == digest for name, digest in manifest.get("artifacts", {}).items())
        status = "pass" if manifest.get("status") == "pass" and memory.get("status") == "pass" and artifacts_valid else "fail"
        qc = pd.read_csv(run_dir / "construction_qc.tsv", sep="\t").iloc[0]
        rows.append({
            "method_id": method_id, "n_observations": int(qc.n_observations),
            "n_input_features": int(qc.n_input_features), "n_output_features": int(qc.n_output_features),
            "output_nnz": int(qc.output_nnz), "mapping_nnz": int(qc.mapping_nnz),
            "runtime_seconds": float(memory["runtime_seconds"]),
            "peak_aggregate_pss_mb": float(memory["peak_aggregate_pss_mb"]),
            "contract_checks_passed": int(manifest["contract_checks_passed"]),
            "contract_checks_total": int(manifest["contract_checks_total"]),
            "artifacts_valid": artifacts_valid, "labels_accessed": manifest.get("labels_accessed"),
            "status": status,
        })
    frame = pd.DataFrame(rows)
    output = root / config["output_root"]
    summary_path = output / config.get("construction_summary_name", "gse214986_construction_summary.tsv")
    frame.to_csv(summary_path, sep="\t", index=False)
    evidence_path = output / config.get("construction_evidence_name", "gse214986_construction_evidence.json")
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": config.get("construction_milestone", "M3D_GSE214986_author_peak_construction"),
        "status": "pass" if len(frame) == len(config["methods"]) and (frame.status == "pass").all() else "fail",
        "dataset_id": config["dataset_id"], "methods_expected": len(config["methods"]),
        "methods_passed": int((frame.status == "pass").sum()),
        "contract_checks_passed": int(frame.contract_checks_passed.sum()),
        "contract_checks_total": int(frame.contract_checks_total.sum()),
        "summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_file(summary_path)},
        "resource_preflight": {"path": str(preflight_path.relative_to(root)), "sha256": sha256_file(preflight_path)},
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "input_evidence": {"path": config["input_evidence"], "sha256": config["input_evidence_sha256"]},
        "forbidden_inputs_used": [], "labels_accessed": False, "clusters_accessed": False,
        "rna_accessed": False, "holdout_accessed": False, "domain_evaluation_run": False,
        "full_real_data_benchmark_run": False, "ranking_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(evidence_path, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError(f"{config['dataset_id']} construction panel failed")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "construct", "summarize"))
    parser.add_argument("--config", default="configs/g4_m3d_gse214986_construction.yaml")
    parser.add_argument("--method")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "preflight":
        path = preflight(args.config, args.root)
    elif args.command == "construct":
        if not args.method:
            parser.error("--method is required for construct")
        path = construct(args.config, args.method, args.root)
    else:
        path = summarize(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
