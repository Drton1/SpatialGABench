"""Leakage-separated G4-M3C simulation truth-recovery canary."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats

from spatialgabench.g4_m3b_rna import column_spearman_from_ranks, normalize_total_log1p
from spatialgabench.gene_activity.accessibility_matched_random import (
    compute_peak_accessibility_metrics,
    sample_accessibility_matched_peaks,
)
from spatialgabench.gene_activity.distal import (
    build_distal_peak_gene_mapping,
    build_promoter_plus_distal_mapping,
)
from spatialgabench.gene_activity.gene_body import build_gene_body_mapping
from spatialgabench.gene_activity.interval_aggregation import (
    compute_gene_activity,
    load_bed_intervals,
    load_gene_metadata,
    parse_peak_coordinates,
)
from spatialgabench.gene_activity.promoter_gene_body import build_promoter_gene_body_mapping
from spatialgabench.gene_activity.random_control import build_random_control_mapping
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


def _load_config(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    mode = config.get("mode")
    if mode not in {"canary_only", "replicated_factorial"}:
        raise PermissionError("M3C module requires an explicit canary or replicated-factorial mode")
    frozen_fields = ["authorization", "preregistration", "previous_gate", "simulation_design", "simulation_runs", "simulation_manifest"]
    if "amendment" in config:
        frozen_fields.append("amendment")
    if "generation_evidence" in config:
        frozen_fields.append("generation_evidence")
    for field in frozen_fields:
        artifact = root / config[field]
        if sha256_file(artifact) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {field}")
    authorization = _json(root / config["authorization"])
    expected_milestone = "M3C_SIMULATION_CANARY" if mode == "canary_only" else "M3C_REPLICATED_SIMULATION_EVALUATION"
    if authorization.get("status") != "authorized" or authorization.get("milestone") != expected_milestone:
        raise PermissionError("M3C authorization is not active for the configured mode")
    if authorization.get("factorial_640_generation_authorized") is not False:
        raise PermissionError("Method/evaluation authorization incorrectly permits factorial generation")
    if mode == "replicated_factorial" and (
        authorization.get("construction_methods_authorized") is not True
        or authorization.get("evaluation_truth_access_authorized") is not True
    ):
        raise PermissionError("Replicated construction/evaluation authorization boundary is invalid")
    if authorization.get("authorized_method_panel") != config["methods"]:
        raise ValueError("Authorized method panel differs from config")
    manifest = _json(root / config["simulation_manifest"])
    if mode == "canary_only":
        manifest_valid = manifest.get("status") == "pass" and manifest.get("scope") == "design_expansion_and_first_seed_OAT_smoke_only"
    else:
        manifest_valid = (
            manifest.get("status") == "pass"
            and manifest.get("milestone") == "M3C_replicated_simulation_generation"
            and manifest.get("simulation_rows_passed") == int(config["expected_simulations"])
        )
    if not manifest_valid:
        raise ValueError("Simulation source evidence is not valid for configured mode")
    if any(bool(config.get(key)) for key in ("software_methods_run", "full_benchmark_run", "final_ranking_run", "universal_ranking_run", "recommendation_policy_fit_run")):
        raise PermissionError("A forbidden execution flag is enabled")
    config["_config_path"] = path
    return config


def _simulation_rows(config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows = pd.read_csv(root / config["simulation_runs"], sep="\t")
    expected = int(config["expected_simulations"])
    if len(rows) != expected or rows["simulation_id"].nunique() != expected or not (rows["status"] == "pass").all():
        raise ValueError("Simulation canary run panel is incomplete")
    counts = rows.groupby("simulation_type").size().to_dict()
    if set(counts) != set(config["expected_tracks"]) or set(counts.values()) != {int(config["expected_simulations_per_track"])}:
        raise ValueError("Simulation canary strata are unbalanced")
    return rows.sort_values("simulation_id", kind="mergesort").reset_index(drop=True)


def _tss_from_genes(genes: pd.DataFrame) -> pd.DataFrame:
    tss = genes.copy().reset_index(drop=True)
    tss["gene_index"] = np.arange(len(tss), dtype=np.int64)
    tss["tss"] = np.where(tss["strand"].astype(str) == "+", tss["start"], tss["end"]).astype(np.int64)
    tss["score"] = 0
    return tss[["gene_index", "gene_id", "gene_name", "gene_type", "chr", "start", "end", "strand", "tss"]]


def _mapping(method_id: str, adata: ad.AnnData, peaks: pd.DataFrame, genes: pd.DataFrame, promoters: pd.DataFrame, gene_body: pd.DataFrame, tss: pd.DataFrame, params: dict[str, Any]):
    from spatialgabench.gene_activity.interval_aggregation import build_mapping_from_intervals

    matching_qc = None
    if method_id == "promoter_only_2kb_pc":
        mapping = build_mapping_from_intervals(peaks, promoters, genes, deduplicate=True)
    elif method_id == "gene_body_pc":
        mapping = build_gene_body_mapping(peaks, gene_body, genes)
    elif method_id == "promoter_plus_gene_body_2kb_pc":
        mapping = build_promoter_gene_body_mapping(peaks, promoters, gene_body, genes)
    elif method_id == "distance_weighted_distal_250kb_decay50kb_pc":
        mapping = build_distal_peak_gene_mapping(
            peaks, tss, genes,
            distal_window_bp=int(params["distal_window_bp"]),
            promoter_exclusion_window_bp=int(params["promoter_window_bp"]),
            distance_decay_bp=int(params["distance_decay_bp"]),
        )
    elif method_id == "promoter_plus_distal_2kb_250kb_decay50kb_pc":
        mapping = build_promoter_plus_distal_mapping(
            peaks, promoters, tss, genes,
            distal_window_bp=int(params["distal_window_bp"]),
            promoter_exclusion_window_bp=int(params["promoter_window_bp"]),
            distance_decay_bp=int(params["distance_decay_bp"]),
        )
    elif method_id in {"random_peak_to_gene_negative_control_pc", "accessibility_matched_random_peak_to_gene_control_pc"}:
        promoter_mapping = build_mapping_from_intervals(peaks, promoters, genes, deduplicate=True)
        if method_id == "random_peak_to_gene_negative_control_pc":
            mapping = build_random_control_mapping(
                n_peaks=len(peaks), gene_metadata=genes,
                promoter_peak_counts=promoter_mapping.gene_assigned_peak_counts,
                seed=int(params["control_seed"]), peaks=peaks,
            )
        else:
            matched = sample_accessibility_matched_peaks(
                peak_metrics=compute_peak_accessibility_metrics(adata.X, peaks),
                reference_promoter_links=promoter_mapping.links,
                gene_metadata=genes, seed=int(params["control_seed"]),
                n_bins=int(params["accessibility_bins"]), within_gene_replacement=False,
            )
            mapping, matching_qc = matched.mapping, matched.matching_qc
    else:
        raise ValueError(f"Unsupported M3C method: {method_id}")
    return mapping, matching_qc


def construct(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load_config(config_path, root)
    rows = _simulation_rows(config, root)
    output_root = root / config["output_root"]
    run_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for simulation in rows.itertuples(index=False):
        simulation_dir = root / config["simulation_root"] / simulation.simulation_id
        input_path = simulation_dir / "simulated_atac_peak_matrix.h5ad"
        if any(value.lower() in str(input_path).lower() for value in config["forbidden_holdout_accessions"]):
            raise PermissionError("Frozen holdout path rejected")
        adata = ad.read_h5ad(input_path)
        forbidden_obs = [column for column in adata.obs.columns if any(token in column.lower() for token in ("true", "label", "cluster", "pathology"))]
        if forbidden_obs or adata.uns.get("evaluation_fields_present_in_obs") is not False or adata.uns.get("construction_evaluation_inputs_used") is not False:
            raise ValueError(f"Simulation accessibility input leaks evaluation fields: {simulation.simulation_id}")
        adata.uns["dataset_id"] = simulation.simulation_id
        peaks = parse_peak_coordinates(adata)
        reference = simulation_dir / "reference"
        genes = load_gene_metadata(reference / "gene_body.bed")
        promoters = load_bed_intervals(reference / "promoters_2kb.bed", "promoter")
        gene_body = load_bed_intervals(reference / "gene_body.bed", "gene_body")
        tss = _tss_from_genes(genes)
        for method_id in config["methods"]:
            started = time.perf_counter()
            run_dir = output_root / "construction" / simulation.simulation_id / method_id
            run_dir.mkdir(parents=True, exist_ok=True)
            input_validation = {
                "schema_version": 1, "status": "pass", "scope": "G4_M3C_simulation_construction_only",
                "simulation_id": simulation.simulation_id, "simulation_type": simulation.simulation_type,
                "method_id": method_id, "input_path": str(input_path.relative_to(root)),
                "input_sha256": sha256_file(input_path),
                "files_opened": [str(input_path.relative_to(root)), str((reference / "gene_body.bed").relative_to(root)), str((reference / "promoters_2kb.bed").relative_to(root))],
                "construction_inputs_used": config["construction_inputs_allowed"],
                "forbidden_inputs_used": [], "simulation_truth_accessed": False,
                "simulated_rna_accessed": False, "labels_accessed": False,
                "factor_values_accessed": False, "holdout_accessed": False,
            }
            validation_path = run_dir / "input_validation.json"
            _write_json(validation_path, input_validation)
            mapping, matching_qc = _mapping(method_id, adata, peaks, genes, promoters, gene_body, tss, config["parameters"])
            activity = compute_gene_activity(adata.X, mapping.peak_gene_matrix)
            var = mapping.genes.copy()
            var.index = var["gene_id"].astype(str)
            output_path = run_dir / "gene_activity.h5ad"
            output = ensure_standard_gene_activity_anndata(
                input_adata=adata, X=activity, var=var, method_id=method_id,
                software_name="SpatialGABench", software_version="1.0.0.dev0",
                wrapper_version="g4_m3c_simulation_canary_v1", genome_build="synthetic_chrSim_v2",
                normalization_status="raw_accessibility_aggregation", output_path=output_path,
                extra_uns={
                    "run_id": f"g4_m3c__{simulation.simulation_id}__{method_id}",
                    "study_id": "synthetic_simulation", "sample_id": simulation.simulation_id,
                    "representation_id": "simulated_author_peak_matrix", "input_track": "simulation_peak_matrix",
                    "comparison_scope": "simulation_canary_within_representation",
                    "subset_policy": "all_simulated_observations", "construction_input_classes": config["construction_inputs_allowed"],
                    "construction_leakage_inputs": "none", "input_validation_sha256": sha256_file(validation_path),
                    "environment_lock_sha256": sha256_file(root / "envs/core-resolved.yaml"),
                    "authorization_sha256": config["authorization_sha256"],
                },
            )
            contract = validate_standard_output(output, adata, method_id)
            contract_path = run_dir / "contract_validation.tsv"
            contract.to_csv(contract_path, sep="\t", index=False)
            if not (contract["status"] == "pass").all():
                raise ValueError(f"Output contract failed: {simulation.simulation_id}/{method_id}")
            mapping_path = run_dir / "peak_gene_mapping.npz"
            sparse.save_npz(mapping_path, mapping.peak_gene_matrix.tocsr(), compressed=True)
            links_path = run_dir / "predicted_peak_gene_links.tsv.gz"
            mapping.links.to_csv(links_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
            artifacts = [validation_path, output_path, contract_path, mapping_path, links_path]
            if matching_qc is not None:
                matching_path = run_dir / "accessibility_matching_qc.tsv"
                matching_qc.to_csv(matching_path, sep="\t", index=False)
                artifacts.append(matching_path)
            runtime = time.perf_counter() - started
            manifest_path = run_dir / "construction_manifest.json"
            manifest = {
                "schema_version": 1, "status": "pass", "simulation_id": simulation.simulation_id,
                "simulation_type": simulation.simulation_type, "method_id": method_id,
                "artifacts": {path.name: sha256_file(path) for path in artifacts},
                "runtime_seconds": runtime, "forbidden_inputs_used": [],
                "simulation_truth_accessed": False, "simulated_rna_accessed": False,
                "labels_accessed": False, "factor_values_accessed": False, "holdout_accessed": False,
                "full_benchmark_run": False, "ranking_run": False,
            }
            _write_json(manifest_path, manifest)
            run_rows.append({
                "simulation_id": simulation.simulation_id, "simulation_type": simulation.simulation_type,
                "method_id": method_id, "n_observations": adata.n_obs, "n_peaks": adata.n_vars,
                "n_genes": output.n_vars, "mapping_nnz": mapping.peak_gene_matrix.nnz,
                "output_nnz": output.X.nnz, "contract_checks_passed": int((contract.status == "pass").sum()),
                "contract_checks_total": len(contract), "runtime_seconds": runtime,
                "construction_manifest": str(manifest_path.relative_to(root)),
                "construction_manifest_sha256": sha256_file(manifest_path), "status": "pass",
            })
            if len(run_rows) % 100 == 0 or len(run_rows) == int(config["expected_simulations"]) * len(config["methods"]):
                print(f"M3C simulation construction: {len(run_rows)}/{int(config['expected_simulations']) * len(config['methods'])}", flush=True)
    summary = pd.DataFrame(run_rows)
    summary_path = output_root / "construction_run_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    expected = int(config["expected_simulations"]) * len(config["methods"])
    manifest_path = output_root / "construction_manifest.json"
    scope = "simulation_canary_construction_only" if config["mode"] == "canary_only" else "replicated_simulation_construction_only"
    manifest = {
        "schema_version": 1, "status": "pass" if len(summary) == expected and (summary.status == "pass").all() else "fail",
        "scope": scope, "runs_expected": expected, "runs_passed": len(summary),
        "contract_checks_passed": int(summary.contract_checks_passed.sum()),
        "contract_checks_total": int(summary.contract_checks_total.sum()),
        "runtime_seconds": time.perf_counter() - started_all,
        "run_summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_file(summary_path)},
        "construction_inputs_used": config["construction_inputs_allowed"], "forbidden_inputs_used": [],
        "simulation_truth_accessed": False, "simulated_rna_accessed": False, "labels_accessed": False,
        "factor_values_accessed": False, "holdout_accessed": False, "full_benchmark_run": False,
        "ranking_run": False,
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def _average_precision_over_prevalence(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return math.nan, positives / labels.size, math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0) + 1, len(sorted_scores)]
    tp = 0
    ap = 0.0
    start = 0
    for end in boundaries:
        group_tp = int(sorted_labels[start:end].sum())
        tp += group_tp
        ap += (group_tp / positives) * (tp / end)
        start = int(end)
    prevalence = positives / labels.size
    return float(ap), float(prevalence), float(ap / prevalence)


def _mean_marker_domain_auc(values: np.ndarray, truth: np.ndarray, labels: np.ndarray) -> float:
    aucs: list[float] = []
    domains = np.unique(labels)
    for gene_index in range(values.shape[1]):
        means = [truth[labels == domain, gene_index].mean() for domain in domains]
        positive = labels == domains[int(np.argmax(means))]
        n_pos, n_neg = int(positive.sum()), int((~positive).sum())
        ranks = stats.rankdata(values[:, gene_index])
        u = float(ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0)
        aucs.append(u / (n_pos * n_neg))
    return float(np.mean(aucs))


def _component_metric(predicted: np.ndarray, component: np.ndarray) -> tuple[float, int, bool]:
    pred_variable = np.var(predicted, axis=0) > 0
    truth_variable = np.var(component, axis=0) > 0
    eligible = pred_variable & truth_variable
    if not eligible.any():
        return math.nan, 0, False
    correlations = column_spearman_from_ranks(stats.rankdata(component[:, eligible], axis=0), predicted[:, eligible])
    return float(np.mean(correlations)), int(eligible.sum()), True


def evaluate(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load_config(config_path, root)
    simulations = _simulation_rows(config, root)
    output_root = root / config["output_root"]
    construction_manifest = _json(output_root / "construction_manifest.json")
    expected = int(config["expected_simulations"]) * len(config["methods"])
    if construction_manifest.get("status") != "pass" or construction_manifest.get("runs_passed") != expected:
        raise ValueError("M3C construction evidence does not pass")
    metric_rows: list[dict[str, Any]] = []
    gene_frames: list[pd.DataFrame] = []
    structural_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for simulation in simulations.itertuples(index=False):
        simulation_dir = root / config["simulation_root"] / simulation.simulation_id
        factors = pd.read_csv(simulation_dir / "simulation_factor_values.tsv", sep="\t").iloc[0]
        truth_obj = ad.read_h5ad(simulation_dir / "true_gene_activity.h5ad")
        promoter_obj = ad.read_h5ad(simulation_dir / "true_promoter_component.h5ad")
        distal_obj = ad.read_h5ad(simulation_dir / "true_distal_component.h5ad")
        labels_frame = pd.read_csv(simulation_dir / "true_domain_labels.tsv", sep="\t", index_col=0)
        truth = truth_obj.X.toarray() if sparse.issparse(truth_obj.X) else np.asarray(truth_obj.X)
        promoter = promoter_obj.X.toarray() if sparse.issparse(promoter_obj.X) else np.asarray(promoter_obj.X)
        distal = distal_obj.X.toarray() if sparse.issparse(distal_obj.X) else np.asarray(distal_obj.X)
        if not truth_obj.obs_names.equals(labels_frame.index) or not np.allclose(promoter + distal, truth, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Simulation truth alignment/decomposition failed: {simulation.simulation_id}")
        promoter_scalar = bool(np.allclose(promoter, float(1.0 - factors.distal_fraction) * truth))
        distal_scalar = bool(np.allclose(distal, float(factors.distal_fraction) * truth))
        structural_rows.append({
            "simulation_id": simulation.simulation_id, "simulation_type": simulation.simulation_type,
            "promoter_plus_distal_equals_true_ga": True,
            "promoter_is_scalar_multiple_of_true_ga": promoter_scalar,
            "distal_is_scalar_multiple_of_true_ga": distal_scalar,
            "component_endpoints_structurally_redundant": promoter_scalar and distal_scalar,
            "status": "diagnostic",
        })
        true_links = pd.read_csv(simulation_dir / "true_peak_gene_links.tsv", sep="\t")
        input_obj = ad.read_h5ad(simulation_dir / "simulated_atac_peak_matrix.h5ad", backed="r")
        peak_ids = input_obj.var_names.astype(str).tolist()
        input_obj.file.close()
        gene_ids = truth_obj.var_names.astype(str).tolist()
        peak_index = {value: index for index, value in enumerate(peak_ids)}
        gene_index = {value: index for index, value in enumerate(gene_ids)}
        true_binary = np.zeros((len(peak_ids), len(gene_ids)), dtype=bool)
        active = true_links[true_links["is_active_link"].astype(bool)]
        for link in active.itertuples(index=False):
            true_binary[peak_index[str(link.peak_id)], gene_index[str(link.gene_id)]] = True
        labels = labels_frame["true_domain_index"].to_numpy(dtype=int)
        truth_ranks = stats.rankdata(truth, axis=0)
        for method_id in config["methods"]:
            started = time.perf_counter()
            run_dir = output_root / "construction" / simulation.simulation_id / method_id
            run_manifest_path = run_dir / "construction_manifest.json"
            run_manifest = _json(run_manifest_path)
            if run_manifest.get("status") != "pass" or run_manifest.get("simulation_truth_accessed") is not False or run_manifest.get("forbidden_inputs_used") != []:
                raise ValueError(f"Construction leakage contract failed: {simulation.simulation_id}/{method_id}")
            for name, digest in run_manifest["artifacts"].items():
                if sha256_file(run_dir / name) != digest:
                    raise ValueError(f"Construction artifact hash mismatch: {simulation.simulation_id}/{method_id}/{name}")
            prediction_obj = ad.read_h5ad(run_dir / "gene_activity.h5ad")
            if not prediction_obj.obs_names.equals(truth_obj.obs_names) or prediction_obj.var_names.astype(str).tolist() != gene_ids:
                raise ValueError(f"Prediction/truth identity mismatch: {simulation.simulation_id}/{method_id}")
            prediction_sparse = prediction_obj.X.tocsr() if sparse.issparse(prediction_obj.X) else sparse.csr_matrix(prediction_obj.X)
            normalized = normalize_total_log1p(prediction_sparse, float(config["normalization"]["target_sum"])).toarray()
            variable = np.var(normalized, axis=0) > 0
            correlations = column_spearman_from_ranks(truth_ranks, normalized)
            correlations[~variable] = 0.0
            coverage = float(variable.mean())
            mean_variable = float(correlations[variable].mean()) if variable.any() else 0.0
            primary = mean_variable * coverage
            promoter_metric, promoter_genes, promoter_eligible = _component_metric(normalized, promoter)
            distal_metric, distal_genes, distal_eligible = _component_metric(normalized, distal)
            predicted_mapping = sparse.load_npz(run_dir / "peak_gene_mapping.npz").toarray()
            ap, prevalence, ap_over_prevalence = _average_precision_over_prevalence(predicted_mapping, true_binary)
            metric_rows.append({
                "simulation_id": simulation.simulation_id, "simulation_type": simulation.simulation_type,
                "physical_random_seed": int(simulation.random_seed),
                "replicate_seed_index": int(config["seed_aliases"][int(simulation.random_seed)]),
                "method_id": method_id,
                **{factor: float(factors[factor]) for factor in ("distal_fraction", "depth_multiplier", "zero_inflation_rate", "domain_log_effect", "decoy_fraction", "rna_latent_correlation")},
                "n_observations": truth.shape[0], "n_truth_genes": truth.shape[1],
                "variable_predicted_genes": int(variable.sum()), "evaluable_gene_fraction": coverage,
                "mean_gene_wise_spearman_variable_genes": mean_variable,
                config["primary_endpoint"]["id"]: primary,
                "peak_gene_link_auprc": ap, "active_link_prevalence": prevalence,
                "peak_gene_link_auprc_over_prevalence": ap_over_prevalence,
                "promoter_component_spearman": promoter_metric,
                "promoter_component_evaluable_genes": promoter_genes,
                "promoter_component_eligible": promoter_eligible,
                "distal_component_spearman": distal_metric,
                "distal_component_evaluable_genes": distal_genes,
                "distal_component_eligible": distal_eligible,
                "mean_gene_marker_domain_auroc": _mean_marker_domain_auc(normalized, truth, labels),
                "construction_manifest_sha256": sha256_file(run_manifest_path),
                "evaluation_runtime_seconds": time.perf_counter() - started,
                "truth_used_for_construction": False, "rna_used_for_construction": False,
                "labels_used_for_construction": False, "holdout_accessed": False,
                "status": "pass",
            })
            genes = pd.DataFrame({
                "simulation_id": simulation.simulation_id, "simulation_type": simulation.simulation_type,
                "method_id": method_id, "gene_id": gene_ids, "predicted_gene_variable": variable,
                "spearman_to_true_gene_activity": correlations,
            })
            gene_frames.append(genes)
            if len(metric_rows) % 100 == 0 or len(metric_rows) == expected:
                print(f"M3C simulation evaluation: {len(metric_rows)}/{expected}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    genes = pd.concat(gene_frames, ignore_index=True)
    structural = pd.DataFrame(structural_rows)
    metrics_path = output_root / "simulation_truth_recovery_metrics.tsv"
    genes_path = output_root / "simulation_gene_level_correlations.tsv.gz"
    structural_path = output_root / "simulation_truth_structure_diagnostics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    genes.to_csv(genes_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    structural.to_csv(structural_path, sep="\t", index=False)
    manifest_path = output_root / "evaluation_manifest.json"
    scope = "simulation_truth_recovery_canary_not_inferential_benchmark" if config["mode"] == "canary_only" else "replicated_simulation_truth_recovery_by_stratum"
    manifest = {
        "schema_version": 1, "status": "pass" if len(metrics) == expected and (metrics.status == "pass").all() else "fail",
        "scope": scope,
        "runs_expected": expected, "runs_passed": len(metrics), "gene_level_rows": len(genes),
        "runtime_seconds": time.perf_counter() - started_all,
        "primary_endpoint": config["primary_endpoint"],
        "artifacts": {path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in (metrics_path, genes_path, structural_path)},
        "truth_access_stage": "evaluation_only_after_frozen_construction",
        "truth_used_for_construction": False, "simulated_rna_used": False,
        "labels_used_for_construction": False, "holdout_accessed": False,
        "full_benchmark_run": False, "ranking_run": False,
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def summarize(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load_config(config_path, root)
    output = root / config["output_root"]
    construction = _json(output / "construction_manifest.json")
    evaluation = _json(output / "evaluation_manifest.json")
    construction_memory = _json(output / "construction_process_tree_memory.json")
    evaluation_memory = _json(output / "evaluation_process_tree_memory.json")
    expected = int(config["expected_simulations"]) * len(config["methods"])
    if any(item.get("status") != "pass" for item in (construction, evaluation, construction_memory, evaluation_memory)):
        raise ValueError("M3C canary manifests or process-tree memory evidence do not pass")
    if construction.get("runs_passed") != expected or evaluation.get("runs_passed") != expected:
        raise ValueError("M3C canary run count mismatch")
    metrics_path = output / "simulation_truth_recovery_metrics.tsv"
    metrics = pd.read_csv(metrics_path, sep="\t")
    primary = config["primary_endpoint"]["id"]
    if len(metrics) != expected or not np.isfinite(metrics[primary]).all() or not metrics[primary].between(-1, 1).all():
        raise ValueError("M3C primary endpoint contract failed")
    expected_primary = metrics["mean_gene_wise_spearman_variable_genes"] * metrics["evaluable_gene_fraction"]
    if not np.allclose(metrics[primary], expected_primary, rtol=1e-12, atol=1e-12):
        raise ValueError("M3C primary endpoint formula mismatch")
    factor_names = ["distal_fraction", "depth_multiplier", "zero_inflation_rate", "domain_log_effect", "decoy_fraction", "rna_latent_correlation"]
    low = {name: float(metrics[name].min()) for name in factor_names}
    invariance_rows = []
    for track in config["expected_tracks"]:
        frame = metrics[metrics.simulation_type == track]
        baseline = frame[np.logical_and.reduce([np.isclose(frame[name], value) for name, value in low.items()])]
        rna_changed = frame[
            np.logical_and.reduce([
                np.isclose(frame[name], float(frame[name].max()) if name == "rna_latent_correlation" else value)
                for name, value in low.items()
            ])
        ]
        paired_keys = ["method_id", "physical_random_seed"]
        merged = baseline.merge(rna_changed, on=paired_keys, suffixes=("_baseline", "_rna_changed"), validate="one_to_one")
        for row in merged.itertuples(index=False):
            differences = [
                abs(getattr(row, f"{column}_baseline") - getattr(row, f"{column}_rna_changed"))
                for column in (primary, "peak_gene_link_auprc_over_prevalence", "mean_gene_marker_domain_auroc")
            ]
            invariance_rows.append({
                "simulation_type": track, "method_id": row.method_id,
                "physical_random_seed": int(row.physical_random_seed),
                "maximum_absolute_metric_difference": max(differences),
                "rna_factor_construction_invariance_pass": max(differences) <= 1e-12,
            })
    invariance = pd.DataFrame(invariance_rows)
    invariance_path = output / "rna_factor_construction_invariance.tsv"
    invariance.to_csv(invariance_path, sep="\t", index=False)
    expected_invariance = len(config["expected_tracks"]) * len(config["methods"]) * metrics.physical_random_seed.nunique()
    if len(invariance) != expected_invariance or not invariance.rna_factor_construction_invariance_pass.all():
        raise ValueError("RNA-factor leakage sentinel failed")
    component = metrics[metrics.promoter_component_eligible & metrics.distal_component_eligible].copy()
    component["absolute_difference"] = (component.promoter_component_spearman - component.distal_component_spearman).abs()
    component_path = output / "component_endpoint_redundancy_audit.tsv"
    component[["simulation_id", "simulation_type", "method_id", "promoter_component_spearman", "distal_component_spearman", "absolute_difference"]].to_csv(component_path, sep="\t", index=False)
    component_policy = config["secondary_endpoints"]["promoter_component_spearman"]["status"]
    structure = pd.read_csv(output / "simulation_truth_structure_diagnostics.tsv", sep="\t")
    if component_policy == "diagnostic_structurally_redundant_in_generator_v2":
        component_contract_pass = bool(not component.empty and component.absolute_difference.max() <= 1e-12)
        component_evidence = "structural_redundancy_confirmed"
    elif component_policy == "supportive_component_identifiable_generator_v2_1":
        nonzero_distal = structure[~structure.component_endpoints_structurally_redundant]
        component_contract_pass = bool(
            not component.empty
            and not nonzero_distal.empty
            and component.absolute_difference.max() > 1e-4
        )
        component_evidence = "component_identifiability_confirmed"
    else:
        raise ValueError(f"Unknown component endpoint policy: {component_policy}")
    if not component_contract_pass:
        raise ValueError("Generator component endpoint policy contract failed")
    source_bytes = sum(path.stat().st_size for path in (root / config["simulation_root"]).rglob("*") if path.is_file())
    construction_bytes = sum(path.stat().st_size for path in (output / "construction").rglob("*") if path.is_file())
    smoke_runs = pd.read_csv(root / config["simulation_runs"], sep="\t")
    full_simulations = 640
    full_method_runs = full_simulations * len(config["methods"])
    resource = pd.DataFrame([
        {
            "scope": "observed_canary", "simulation_rows": int(config["expected_simulations"]),
            "method_runs": expected, "generator_cpu_seconds": float(smoke_runs.runtime_seconds.sum()),
            "construction_wall_seconds": float(construction_memory["runtime_seconds"]),
            "evaluation_wall_seconds": float(evaluation_memory["runtime_seconds"]),
            "generator_artifact_bytes": source_bytes, "construction_artifact_bytes": construction_bytes,
        },
        {
            "scope": "linear_projection_full_2x64x5", "simulation_rows": full_simulations,
            "method_runs": full_method_runs,
            "generator_cpu_seconds": float(smoke_runs.runtime_seconds.mean() * full_simulations),
            "construction_wall_seconds": float(construction_memory["runtime_seconds"] / expected * full_method_runs),
            "evaluation_wall_seconds": float(evaluation_memory["runtime_seconds"] / expected * full_method_runs),
            "generator_artifact_bytes": int(source_bytes / int(config["expected_simulations"]) * full_simulations),
            "construction_artifact_bytes": int(construction_bytes / expected * full_method_runs),
        },
    ])
    resource_path = output / "full_factorial_linear_resource_projection.tsv"
    resource.to_csv(resource_path, sep="\t", index=False)
    scope = metrics.groupby(["simulation_type", "method_id"], sort=True)[primary].agg(["count", "min", "median", "max"]).reset_index()
    scope["interpretation"] = "engineering_canary_only_no_inference_no_ranking"
    scope_path = output / "canary_scope_summary.tsv"
    scope.to_csv(scope_path, sep="\t", index=False)
    artifacts = [metrics_path, invariance_path, component_path, resource_path, scope_path]
    replicated = config["mode"] == "replicated_factorial"
    evidence = {
        "schema_version": 1, "gate": "G4",
        "milestone": "M3C_replicated_simulation_truth_recovery" if replicated else "M3C_simulation_truth_recovery_canary",
        "status": "pass", "scope": "replicated_2x64x5_factorial_by_stratum" if replicated else "14_run_OAT_engineering_canary_only",
        "simulation_rows": int(config["expected_simulations"]), "method_runs": expected,
        "tracks": config["expected_tracks"], "methods": config["methods"],
        "contract_checks_passed": construction["contract_checks_passed"],
        "contract_checks_total": construction["contract_checks_total"],
        "primary_endpoint": config["primary_endpoint"],
        "rna_factor_leakage_sentinel_passed": True,
        "component_endpoint_contract_passed": True,
        "component_endpoint_evidence": component_evidence,
        "component_endpoint_policy": component_policy,
        "artifacts": {path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts},
        "construction_manifest": {"path": str((output / "construction_manifest.json").relative_to(root)), "sha256": sha256_file(output / "construction_manifest.json")},
        "evaluation_manifest": {"path": str((output / "evaluation_manifest.json").relative_to(root)), "sha256": sha256_file(output / "evaluation_manifest.json")},
        "construction_memory": {"path": str((output / "construction_process_tree_memory.json").relative_to(root)), "sha256": sha256_file(output / "construction_process_tree_memory.json")},
        "evaluation_memory": {"path": str((output / "evaluation_process_tree_memory.json").relative_to(root)), "sha256": sha256_file(output / "evaluation_process_tree_memory.json")},
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
        "truth_used_for_construction": False, "simulated_rna_used": False,
        "labels_used_for_construction": False, "holdout_accessed": False,
        "full_benchmark_run": False, "method_ranking_run": False, "universal_ranking_run": False,
        "recommendation_policy_fit_run": False, "final_ranking_run": False,
        "inferential_simulation_claim_made": False, "next_milestone_authorized": False,
    }
    evidence_name = "g4_m3c_replicated_simulation_truth_recovery_evidence.json" if replicated else "g4_m3c_simulation_canary_completion_evidence.json"
    evidence_path = output / evidence_name
    _write_json(evidence_path, evidence)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("construct", "evaluate", "summarize"))
    parser.add_argument("--config", default="configs/g4_m3c_simulation_canary.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    action = {"construct": construct, "evaluate": evaluate, "summarize": summarize}[args.command]
    path = action(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
