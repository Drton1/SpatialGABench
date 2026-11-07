"""Paired-multiome evaluation of gene-activity accuracy under ATAC depth loss."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats

from spatialgabench.g4_m3b_rna import column_spearman_from_ranks
from spatialgabench.nm_spatial_program_evaluation import (
    _align_rows,
    _canonicalize_ga_columns,
    _load_rna,
    normalize_total_log1p,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_hash(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def canonicalize_unique_gene_names(unique_names: dict[str, str]) -> dict[str, str]:
    return {name: re.sub(r"\.\d+$", "", str(gene_id)) for name, gene_id in unique_names.items()}


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def _depth_manifest(config: dict[str, Any], root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    annotation = set(config["methods"]["annotation"])
    software = set(config["methods"]["software"])
    representations = set(config["methods"]["matrix_representations"])
    fragment_representation = config["methods"]["fragment_representation"]
    allowed_pairs = {(method, representation) for method in annotation for representation in representations}
    allowed_pairs |= {(method, fragment_representation) for method in software}

    for dataset_id, source in config["depth_sources"].items():
        metrics_path = root / source["metrics"]
        failures_path = root / source["failures"]
        metrics = pd.read_csv(metrics_path, sep="\t")
        metrics = metrics.loc[metrics.dataset_id.eq(dataset_id)].copy()
        metrics = metrics.loc[
            [(str(method), str(representation)) in allowed_pairs for method, representation in zip(metrics.method_id, metrics.representation_id)]
        ].copy()
        if len(metrics) != int(config["expected_depth_contracts_per_study"]):
            raise ValueError(f"TA1 depth panel changed: {dataset_id}/{len(metrics)}")
        expected_depths = set(float(value) for value in config["depth_fractions"])
        if set(metrics.depth_fraction.astype(float)) != expected_depths:
            raise ValueError(f"TA1 depth levels changed: {dataset_id}")
        failures = pd.read_csv(failures_path, sep="\t")
        failures = failures.loc[failures.dataset_id.eq(dataset_id)].copy()
        failure_fields = failures[[
            "condition_id", "method_id", "representation_id", "failure_stage",
            "failure_type", "message", "failure_record",
        ]]
        metrics = metrics.merge(
            failure_fields,
            on=["condition_id", "method_id", "representation_id"],
            how="left",
            validate="one_to_one",
        )
        metrics["structural_success"] = _as_bool(metrics["success"])
        metrics["study_family"] = source["study_family"]
        metrics["source_scope"] = "complete_depth_series"
        metrics["source_metrics"] = source["metrics"]
        frames.append(metrics)

    human = config["human_full_depth_anchor"]
    summary = pd.read_csv(root / human["run_summary"], sep="\t")
    summary = summary.loc[summary.dataset_id.eq(human["dataset_id"]) & summary.outcome.eq("success")].copy()
    annotation_mask = summary.method_id.isin(annotation) & summary.representation_id.isin(representations)
    software_mask = summary.method_id.isin(software) & summary.input_track.eq("F")
    summary = summary.loc[annotation_mask | software_mask].copy()
    if len(summary) != int(config["expected_human_anchor_contracts"]):
        raise ValueError(f"TA1 human anchor changed: {len(summary)}")
    summary["condition_id"] = human["dataset_id"] + "__d100__s0"
    summary["depth_fraction"] = 1.0
    summary["seed"] = 0
    summary["structural_success"] = True
    summary["study_family"] = human["study_family"]
    summary["source_scope"] = "human_full_depth_anchor_only"
    summary["source_metrics"] = human["run_summary"]
    summary["output_path"] = summary.output_dir.astype(str) + "/gene_activity.h5ad"
    summary["memory_path"] = summary.output_dir.astype(str) + "/process_tree_memory.json"
    summary["failure_stage"] = ""
    summary["failure_type"] = ""
    summary["message"] = ""
    summary["failure_record"] = ""
    summary["pseudobulk_gene_spearman_vs_full"] = 1.0
    summary["median_spot_cosine_vs_full"] = 1.0
    summary["zero_output_spots"] = np.nan
    frames.append(summary)

    manifest = pd.concat(frames, ignore_index=True, sort=False)
    if len(manifest) != int(config["expected_planned_contracts"]):
        raise ValueError(f"TA1 planned contract count changed: {len(manifest)}")
    if manifest[["condition_id", "method_id", "representation_id"]].duplicated().any():
        raise ValueError("TA1 manifest contains duplicate contracts")
    output_hashes: list[str] = []
    observation_hashes: list[str] = []
    output_observations: list[float] = []
    for row in manifest.itertuples(index=False):
        if not bool(row.structural_success):
            output_hashes.append("")
            observation_hashes.append("")
            output_observations.append(math.nan)
            continue
        path = root / str(row.output_path)
        if not path.is_file():
            raise ValueError(f"TA1 successful output is missing: {path}")
        output_hashes.append(sha256_path(path))
        data = ad.read_h5ad(path, backed="r")
        observation_hashes.append(id_hash(data.obs_names))
        output_observations.append(float(data.n_obs))
        data.file.close()
    manifest["output_sha256"] = output_hashes
    manifest["observation_id_order_sha256"] = observation_hashes
    manifest["output_observations"] = output_observations
    columns = [
        "condition_id", "dataset_id", "study_id", "study_family", "species",
        "genome_build", "depth_fraction", "seed", "method_id", "representation_id",
        "structural_success", "output_path", "output_sha256", "observation_id_order_sha256",
        "output_observations", "pseudobulk_gene_spearman_vs_full",
        "median_spot_cosine_vs_full", "zero_output_spots", "failure_stage",
        "failure_type", "message", "failure_record", "source_scope", "source_metrics",
    ]
    return manifest[columns].sort_values(
        ["dataset_id", "depth_fraction", "seed", "representation_id", "method_id"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _yaml(config_path)
    protocol_path = root / config["protocol"]
    rna_config_path = root / config["rna_source_config"]
    for path in (protocol_path, rna_config_path):
        if not path.is_file():
            raise ValueError(f"TA1 frozen input missing: {path}")
    rna_config = _yaml(rna_config_path)
    identity_rows: list[dict[str, Any]] = []
    for dataset_id in config["rna_samples"]:
        sample = rna_config["samples"][dataset_id]
        for field in ("rna_matrix", "positions", "gene_annotation"):
            record = sample[field]
            path = root / record["path"]
            observed = sha256_path(path)
            if observed != record["sha256"]:
                raise ValueError(f"TA1 source checksum mismatch: {dataset_id}/{field}")
            identity_rows.append({
                "dataset_id": dataset_id,
                "artifact_class": field,
                "path": record["path"],
                "sha256": observed,
                "rna_expression_values_read": False,
            })

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    manifest = _depth_manifest(config, root)
    manifest_path = output / "method_input_manifest.tsv"
    identity_path = output / "data_identity_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    pd.DataFrame(identity_rows).to_csv(identity_path, sep="\t", index=False)
    implementation_path = Path(__file__).resolve()
    authorization_path = root / "governance/authorizations/NM_TA1_PAIRED_DEPTH_ACCURACY_AUTHORIZATION_2026-07-30.json"
    authorization = {
        "schema_version": 1,
        "authorization": "NM_TA1_PAIRED_DEPTH_ACCURACY",
        "milestone": config["milestone"],
        "authorization_date": "2026-07-30",
        "status": "authorized",
        "paired_rna_value_access_authorized": True,
        "construction_or_method_rerun_authorized": False,
        "labels_clusters_pathology_access_authorized": False,
        "policy_refit_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "protocol": {"path": str(protocol_path.relative_to(root)), "sha256": sha256_path(protocol_path)},
        "rna_source_config": {"path": str(rna_config_path.relative_to(root)), "sha256": sha256_path(rna_config_path)},
        "method_input_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_path(manifest_path)},
        "data_identity_manifest": {"path": str(identity_path.relative_to(root)), "sha256": sha256_path(identity_path)},
        "implementation": {"path": str(implementation_path.relative_to(root)), "sha256": sha256_path(implementation_path)},
        "implementation_dependencies": [
            {"path": path, "sha256": sha256_path(root / path)}
            for path in config["implementation_dependencies"]
        ],
    }
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_preflight",
        "status": "pass",
        "planned_contracts": len(manifest),
        "structural_successes": int(manifest.structural_success.sum()),
        "structured_failures": int((~manifest.structural_success).sum()),
        "complete_depth_studies": int(manifest.source_scope.eq("complete_depth_series").groupby(manifest.dataset_id).any().sum()),
        "human_full_depth_anchor_contracts": int(manifest.source_scope.eq("human_full_depth_anchor_only").sum()),
        "rna_expression_values_read": False,
        "construction_or_method_rerun": False,
        "labels_clusters_pathology_accessed": False,
        "policy_refit_run": False,
        "input_tracks_pooled": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "method_input_manifest": authorization["method_input_manifest"],
        "data_identity_manifest": authorization["data_identity_manifest"],
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
    }
    evidence_path = output / "preflight_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def _validate_authorization(
    root: Path, config_path: Path, preflight_path: Path, authorization_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    config, preflight, authorization = _yaml(config_path), _json(preflight_path), _json(authorization_path)
    if preflight.get("status") != "pass" or preflight.get("rna_expression_values_read") is not False:
        raise ValueError("TA1 blind preflight is not valid")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != config["milestone"]
        or authorization.get("paired_rna_value_access_authorized") is not True
        or authorization.get("construction_or_method_rerun_authorized") is not False
        or authorization.get("labels_clusters_pathology_access_authorized") is not False
        or authorization.get("policy_refit_authorized") is not False
        or authorization.get("universal_ranking_authorized") is not False
        or authorization.get("final_ranking_authorized") is not False
    ):
        raise PermissionError("TA1 evaluation authorization boundary is invalid")
    bindings = {
        "config": config_path,
        "protocol": root / config["protocol"],
        "rna_source_config": root / config["rna_source_config"],
        "method_input_manifest": root / preflight["method_input_manifest"]["path"],
        "data_identity_manifest": root / preflight["data_identity_manifest"]["path"],
        "implementation": Path(__file__).resolve(),
    }
    for name, path in bindings.items():
        record = authorization[name]
        if record["path"] != str(path.relative_to(root)) or record["sha256"] != sha256_path(path):
            raise ValueError(f"TA1 authorization binding mismatch: {name}")
    dependency_records = authorization.get("implementation_dependencies", [])
    if [record.get("path") for record in dependency_records] != config["implementation_dependencies"]:
        raise ValueError("TA1 implementation dependency manifest changed")
    for record in dependency_records:
        path = root / record["path"]
        if not path.is_file() or sha256_path(path) != record.get("sha256"):
            raise ValueError(f"TA1 implementation dependency changed: {record.get('path')}")
    manifest = pd.read_csv(bindings["method_input_manifest"], sep="\t")
    if len(manifest) != int(config["expected_planned_contracts"]):
        raise ValueError("TA1 frozen manifest dimensions changed")
    for row in manifest.loc[_as_bool(manifest.structural_success)].itertuples(index=False):
        path = root / row.output_path
        if sha256_path(path) != row.output_sha256:
            raise ValueError(f"TA1 method output changed after preflight: {path}")
    return config, preflight, manifest


def _evaluate_matrix(
    activity: ad.AnnData,
    rna_normalized: sparse.csr_matrix,
    eligible_indices: np.ndarray,
    eligible_gene_ids: np.ndarray,
    rna_ranks: np.ndarray,
    unique_names: dict[str, str],
    target_sum: float,
    chunk_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = activity.X.tocsr() if sparse.issparse(activity.X) else sparse.csr_matrix(activity.X)
    normalized_original = normalize_total_log1p(raw, target_sum)
    normalized, gene_ids, _ = _canonicalize_ga_columns(normalized_original, activity, unique_names)
    activity_index = {gene_id: index for index, gene_id in enumerate(gene_ids)}
    mapped = np.asarray([activity_index.get(str(gene_id), -1) for gene_id in eligible_gene_ids], dtype=np.int64)
    present = np.flatnonzero(mapped >= 0)
    correlations = np.zeros(len(eligible_gene_ids), dtype=np.float64)
    variable = np.zeros(len(eligible_gene_ids), dtype=bool)
    for start in range(0, len(present), chunk_size):
        positions = present[start : start + chunk_size]
        dense = normalized[:, mapped[positions]].toarray()
        variances = np.var(dense, axis=0)
        variable[positions] = variances > 0
        correlations[positions] = column_spearman_from_ranks(rna_ranks[:, positions], dense)
        correlations[positions[variances == 0]] = 0.0
    median = float(np.median(correlations[present])) if len(present) else 0.0
    coverage = float(variable.sum() / len(eligible_gene_ids)) if len(eligible_gene_ids) else 0.0
    primary = median * coverage
    rna_bulk = np.asarray(rna_normalized[:, eligible_indices].mean(axis=0)).ravel()
    activity_bulk = np.zeros(len(eligible_gene_ids), dtype=np.float64)
    if len(present):
        activity_bulk[present] = np.asarray(normalized[:, mapped[present]].mean(axis=0)).ravel()
    bulk = float(stats.spearmanr(rna_bulk, activity_bulk).statistic)
    if not np.isfinite(bulk):
        bulk = 0.0
    top_k = min(100, len(eligible_gene_ids))
    rna_top = set(eligible_gene_ids[np.argsort(rna_bulk, kind="mergesort")[-top_k:]])
    activity_top = set(eligible_gene_ids[np.argsort(activity_bulk, kind="mergesort")[-top_k:]])
    top_jaccard = len(rna_top & activity_top) / len(rna_top | activity_top) if rna_top | activity_top else math.nan
    zero_spots = int(np.sum(np.asarray(raw.sum(axis=1)).ravel() == 0))
    gene_frame = pd.DataFrame({
        "gene_id": eligible_gene_ids,
        "activity_gene_present": mapped >= 0,
        "activity_gene_variable": variable,
        "spearman": correlations,
    })
    return {
        "rna_eligible_genes": len(eligible_gene_ids),
        "activity_matched_rna_eligible_genes": len(present),
        "activity_variable_rna_eligible_genes": int(variable.sum()),
        "evaluable_gene_fraction": coverage,
        "median_gene_wise_spearman": median,
        "coverage_adjusted_median_gene_wise_spearman": primary,
        "global_pseudobulk_spearman": bulk,
        "top100_gene_jaccard": top_jaccard,
        "zero_output_observations_direct": zero_spots,
        "undefined_or_absent_gene_correlations_set_to_zero": int((~variable).sum()),
    }, gene_frame


def add_full_depth_retention(metrics: pd.DataFrame) -> pd.DataFrame:
    successful = metrics.loc[metrics.evaluation_status.eq("pass")].copy()
    keys = ["dataset_id", "method_id", "representation_id"]
    anchors = successful.loc[successful.depth_fraction.eq(1.0), keys + ["coverage_adjusted_median_gene_wise_spearman"]].rename(
        columns={"coverage_adjusted_median_gene_wise_spearman": "full_depth_rna_primary"}
    )
    if anchors.duplicated(keys).any():
        raise ValueError("TA1 full-depth RNA anchors are not unique")
    merged = metrics.merge(anchors, on=keys, how="left", validate="many_to_one")
    current = merged["coverage_adjusted_median_gene_wise_spearman"]
    full = merged["full_depth_rna_primary"]
    merged["rna_primary_delta_vs_full"] = current - full
    merged["rna_primary_retention_vs_full"] = np.where(full > 0, current / full, np.nan)
    return merged


def evaluate(
    config_path: str | Path,
    preflight_path: str | Path,
    authorization_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    preflight_path, authorization_path = Path(preflight_path).resolve(), Path(authorization_path).resolve()
    config, preflight, manifest = _validate_authorization(root, config_path, preflight_path, authorization_path)
    rna_config = _yaml(root / config["rna_source_config"])
    metric_rows: list[dict[str, Any]] = []
    gene_frames: list[pd.DataFrame] = []
    target_sum = float(config["rna_evaluation"]["target_sum"])
    chunk_size = int(config["rna_evaluation"]["correlation_chunk_genes"])

    for dataset_id in config["rna_samples"]:
        contracts = manifest.loc[manifest.dataset_id.eq(dataset_id)].copy()
        sample = rna_config["samples"][dataset_id]
        success = contracts.loc[_as_bool(contracts.structural_success)]
        observation_hashes = set(success.observation_id_order_sha256.astype(str))
        if len(observation_hashes) != 1:
            raise ValueError(f"TA1 successful outputs do not share observations: {dataset_id}")
        first = ad.read_h5ad(root / success.iloc[0].output_path, backed="r")
        observations = first.obs_names.astype(str).tolist()
        first.file.close()
        rna_counts, rna_barcodes, rna_gene_ids, protein_coding, unique_names = _load_rna(sample, root)
        unique_names = canonicalize_unique_gene_names(unique_names)
        rna_counts = _align_rows(rna_counts, rna_barcodes, observations)
        detection = np.asarray((rna_counts > 0).sum(axis=0)).ravel()
        threshold = math.ceil(float(config["rna_evaluation"]["minimum_detected_observation_fraction"]) * len(observations))
        eligible_indices = np.flatnonzero(protein_coding & (detection >= threshold))
        eligible_gene_ids = rna_gene_ids[eligible_indices]
        rna_normalized = normalize_total_log1p(rna_counts, target_sum)
        rna_ranks = np.empty((len(observations), len(eligible_indices)), dtype=np.float32)
        for start in range(0, len(eligible_indices), chunk_size):
            stop = min(start + chunk_size, len(eligible_indices))
            dense = rna_normalized[:, eligible_indices[start:stop]].toarray()
            rna_ranks[:, start:stop] = stats.rankdata(dense, axis=0).astype(np.float32)

        def evaluate_contract(row: Any) -> tuple[dict[str, Any], pd.DataFrame | None]:
            base = {field: getattr(row, field) for field in [
                "condition_id", "dataset_id", "study_id", "study_family", "species",
                "genome_build", "depth_fraction", "seed", "method_id", "representation_id",
                "structural_success", "output_path", "output_sha256",
                "pseudobulk_gene_spearman_vs_full", "median_spot_cosine_vs_full",
                "failure_stage", "failure_type", "message", "failure_record", "source_scope",
            ]}
            if not bool(row.structural_success):
                return {**base, "evaluation_status": "not_evaluated_structural_failure"}, None
            activity = ad.read_h5ad(root / row.output_path)
            if id_hash(activity.obs_names) != row.observation_id_order_sha256:
                raise ValueError(f"TA1 observation order changed: {row.output_path}")
            values, genes = _evaluate_matrix(
                activity, rna_normalized, eligible_indices, eligible_gene_ids,
                rna_ranks, unique_names, target_sum, chunk_size,
            )
            metric = {**base, **values, "evaluation_status": "pass"}
            genes.insert(0, "representation_id", row.representation_id)
            genes.insert(0, "method_id", row.method_id)
            genes.insert(0, "seed", row.seed)
            genes.insert(0, "depth_fraction", row.depth_fraction)
            genes.insert(0, "dataset_id", row.dataset_id)
            del activity
            return metric, genes

        contract_rows = list(contracts.itertuples(index=False))
        with ThreadPoolExecutor(max_workers=int(config["rna_evaluation"]["workers_per_dataset"])) as executor:
            for metric, genes in executor.map(evaluate_contract, contract_rows):
                metric_rows.append(metric)
                if genes is not None:
                    gene_frames.append(genes)
        del rna_counts, rna_normalized, rna_ranks

    metrics = add_full_depth_retention(pd.DataFrame(metric_rows))
    successful = metrics.loc[metrics.evaluation_status.eq("pass")].copy()
    group_keys = ["dataset_id", "study_id", "study_family", "species", "genome_build", "depth_fraction", "method_id", "representation_id", "source_scope"]
    seed_summary = successful.groupby(group_keys, dropna=False, sort=True).agg(
        successful_seeds=("seed", "count"),
        median_rna_primary=("coverage_adjusted_median_gene_wise_spearman", "median"),
        min_rna_primary=("coverage_adjusted_median_gene_wise_spearman", "min"),
        max_rna_primary=("coverage_adjusted_median_gene_wise_spearman", "max"),
        median_rna_retention_vs_full=("rna_primary_retention_vs_full", "median"),
        median_rna_delta_vs_full=("rna_primary_delta_vs_full", "median"),
        median_construction_pseudobulk_vs_full=("pseudobulk_gene_spearman_vs_full", "median"),
        median_construction_spot_cosine_vs_full=("median_spot_cosine_vs_full", "median"),
        median_zero_output_observations=("zero_output_observations_direct", "median"),
    ).reset_index()
    planned_counts = metrics.groupby(group_keys, dropna=False, sort=True).size().rename("planned_seeds").reset_index()
    seed_summary = planned_counts.merge(seed_summary, on=group_keys, how="left", validate="one_to_one")
    seed_summary["structural_success_fraction"] = seed_summary.successful_seeds.fillna(0) / seed_summary.planned_seeds

    proxy_rows: list[dict[str, Any]] = []
    reduced = successful.loc[successful.depth_fraction.lt(1.0) & successful.rna_primary_retention_vs_full.notna()].copy()
    for dataset_id, frame in reduced.groupby("dataset_id", sort=True):
        for proxy in ("pseudobulk_gene_spearman_vs_full", "median_spot_cosine_vs_full"):
            valid = frame[[proxy, "rna_primary_retention_vs_full"]].dropna()
            rho = float(stats.spearmanr(valid[proxy], valid.rna_primary_retention_vs_full).statistic) if len(valid) >= 3 else math.nan
            proxy_rows.append({
                "scope": dataset_id,
                "proxy_metric": proxy,
                "contracts": len(valid),
                "spearman_proxy_vs_rna_retention": rho,
                "independent_study_inference": False,
            })
    for proxy in ("pseudobulk_gene_spearman_vs_full", "median_spot_cosine_vs_full"):
        valid = reduced[[proxy, "rna_primary_retention_vs_full"]].dropna()
        rho = float(stats.spearmanr(valid[proxy], valid.rna_primary_retention_vs_full).statistic) if len(valid) >= 3 else math.nan
        proxy_rows.append({
            "scope": "pooled_descriptive_contracts",
            "proxy_metric": proxy,
            "contracts": len(valid),
            "spearman_proxy_vs_rna_retention": rho,
            "independent_study_inference": False,
        })

    output = root / config["output_root"]
    metrics_path = output / "contract_rna_accuracy_metrics.tsv"
    genes_path = output / "gene_level_correlations.tsv.gz"
    seed_path = output / "depth_seed_summary.tsv"
    proxy_path = output / "construction_proxy_vs_rna_truth.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    pd.concat(gene_frames, ignore_index=True).to_csv(
        genes_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0},
    )
    seed_summary.to_csv(seed_path, sep="\t", index=False)
    pd.DataFrame(proxy_rows).to_csv(proxy_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "milestone": config["milestone"],
        "status": "pass",
        "planned_contracts": len(metrics),
        "evaluated_successful_contracts": int(metrics.evaluation_status.eq("pass").sum()),
        "retained_structural_failures": int(metrics.evaluation_status.eq("not_evaluated_structural_failure").sum()),
        "complete_depth_studies": 2,
        "human_full_depth_anchor_contracts": int(metrics.source_scope.eq("human_full_depth_anchor_only").sum()),
        "rna_expression_values_accessed_for_evaluation": True,
        "rna_used_for_construction": False,
        "construction_outputs_mutated": False,
        "construction_or_method_rerun": False,
        "labels_clusters_pathology_accessed": False,
        "technical_seeds_treated_as_independent_studies": False,
        "policy_refit_run": False,
        "input_tracks_pooled": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "interpretation_boundary": "paired_RNA_accuracy_under_depth_loss_with_two_complete_study_families_and_one_human_full_depth_anchor",
        "artifacts": {
            "contract_rna_accuracy_metrics.tsv": {"path": str(metrics_path.relative_to(root)), "sha256": sha256_path(metrics_path)},
            "gene_level_correlations.tsv.gz": {"path": str(genes_path.relative_to(root)), "sha256": sha256_path(genes_path)},
            "depth_seed_summary.tsv": {"path": str(seed_path.relative_to(root)), "sha256": sha256_path(seed_path)},
            "construction_proxy_vs_rna_truth.tsv": {"path": str(proxy_path.relative_to(root)), "sha256": sha256_path(proxy_path)},
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "preflight": {"path": str(preflight_path.relative_to(root)), "sha256": sha256_path(preflight_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "implementation": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha256_path(Path(__file__).resolve())},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "evaluate"])
    parser.add_argument("--config", default="configs/nm_paired_depth_accuracy.yaml")
    parser.add_argument("--preflight", default="results/v1/nm_truth_anchor/ta1_paired_depth_accuracy/preflight_evidence.json")
    parser.add_argument("--authorization", default="governance/authorizations/NM_TA1_PAIRED_DEPTH_ACCURACY_AUTHORIZATION_2026-07-30.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.config, args.root)
    else:
        result = evaluate(args.config, args.preflight, args.authorization, args.root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
