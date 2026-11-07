"""Run the frozen G4-M3B paired-RNA fidelity endpoint."""

from __future__ import annotations

import argparse
import gzip
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
from scipy.io import mmread


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def normalize_total_log1p(matrix: sparse.spmatrix, target_sum: float) -> sparse.csr_matrix:
    output = matrix.tocsr(copy=True).astype(np.float64)
    totals = np.asarray(output.sum(axis=1)).ravel()
    scale = np.zeros_like(totals)
    positive = totals > 0
    scale[positive] = float(target_sum) / totals[positive]
    output = sparse.diags(scale).dot(output).tocsr()
    output.data = np.log1p(output.data)
    return output


def column_spearman_from_ranks(reference_ranks: np.ndarray, values: np.ndarray) -> np.ndarray:
    if reference_ranks.shape != values.shape:
        raise ValueError("Reference ranks and values must have the same shape")
    candidate_ranks = stats.rankdata(values, axis=0).astype(np.float64, copy=False)
    reference = reference_ranks.astype(np.float64, copy=True)
    reference -= reference.mean(axis=0)
    candidate_ranks -= candidate_ranks.mean(axis=0)
    numerator = np.sum(reference * candidate_ranks, axis=0)
    denominator = np.sqrt(np.sum(reference * reference, axis=0) * np.sum(candidate_ranks * candidate_ranks, axis=0))
    result = np.zeros(values.shape[1], dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    result[~np.isfinite(result)] = 0.0
    return result


def _gtf_annotations(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    gene_pattern = re.compile(r'gene_id "([^"]+)"')
    name_pattern = re.compile(r'gene_name "([^"]+)"')
    type_pattern = re.compile(r'gene_(?:bio)?type "([^"]+)"')
    biotypes: dict[str, str] = {}
    names: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            gene = gene_pattern.search(fields[8])
            name = name_pattern.search(fields[8])
            biotype = type_pattern.search(fields[8])
            if gene and biotype:
                gene_id = gene.group(1)
                biotypes[gene_id] = biotype.group(1)
                if name:
                    names.setdefault(name.group(1), []).append(gene_id)
    unique_names = {name: values[0] for name, values in names.items() if len(set(values)) == 1}
    return biotypes, unique_names


def _features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, names=["gene_id_versioned", "gene_name", "feature_type"], dtype=str)
    frame["gene_id"] = frame["gene_id_versioned"].str.replace(r"\.\d+$", "", regex=True)
    if not frame["gene_id"].is_unique:
        raise ValueError("RNA gene IDs are not unique after version stripping")
    return frame


def _barcodes(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.strip().split("\t", 1)[0] for line in handle if line.strip()]


def _validate_frozen(config: dict[str, Any], root: Path) -> None:
    for key in ("authorization", "previous_gate", "preflight_evidence", "m2a_evidence", "m2b_evidence", "rna_matrix", "rna_features", "rna_barcodes", "gene_annotation"):
        path = root / config[key]
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"M3B RNA frozen artifact mismatch: {key}")
    auth = _load_json(root / config["authorization"])
    if auth.get("status") != "authorized" or auth.get("milestone") != "M3B_RNA":
        raise PermissionError("M3B RNA authorization is not active")
    if auth.get("construction_parameter_tuning_authorized") is not False or auth.get("holdout_accessed") is not False:
        raise PermissionError("M3B RNA authorization boundary is invalid")
    if config.get("labels_or_clusters_allowed") is not False or config.get("ranking_allowed") is not False:
        raise PermissionError("M3B RNA cannot use labels or ranking")


def _method_inputs(config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    m2a = _load_json(root / config["m2a_evidence"])
    m2a_summary = root / m2a["run_summary"]["path"]
    if sha256(m2a_summary) != m2a["run_summary"]["sha256"]:
        raise ValueError("M2A summary hash mismatch")
    frame = pd.read_csv(m2a_summary, sep="\t")
    selected = frame[frame.dataset_id == config["dataset_id"]]
    if len(selected) != int(config["expected_annotation_runs"]):
        raise ValueError("M3B RNA annotation panel is incomplete")
    for row in selected.itertuples(index=False):
        manifest_path = root / str(row.output_manifest)
        manifest = _load_json(manifest_path)
        output_path = manifest_path.parent / "gene_activity.h5ad"
        digest = manifest["artifacts"]["gene_activity.h5ad"]
        if not output_path.is_file() or sha256(output_path) != digest:
            raise ValueError(f"M3B RNA M2A output hash mismatch: {output_path}")
        rows.append({
            "study_id": row.study_id, "dataset_id": row.dataset_id, "sample_id": row.sample_id,
            "genome_build": row.genome_build, "representation_id": row.representation_id,
            "input_track": "B" if str(row.representation_id).startswith("fixed_bin") else "P",
            "method_id": row.method_id, "method_category": "negative_control" if "control" in str(row.method_id) else "hand_written_annotation",
            "gene_activity_h5ad": str(output_path.relative_to(root)), "gene_activity_sha256": digest,
        })
    m2b = _load_json(root / config["m2b_evidence"])
    m2b_summary = root / m2b["run_summary"]["path"]
    if sha256(m2b_summary) != m2b["run_summary"]["sha256"]:
        raise ValueError("M2B summary hash mismatch")
    frame = pd.read_csv(m2b_summary, sep="\t")
    selected = frame[frame.dataset_id == config["dataset_id"]]
    if len(selected) != int(config["expected_software_runs"]):
        raise ValueError("M3B RNA software panel is incomplete")
    for row in selected.itertuples(index=False):
        output_path = root / "results/v1/g4/m2/software" / row.dataset_id / row.method_id / "gene_activity.h5ad"
        manifest = _load_json(output_path.parent / "output_manifest.json")
        if sha256(output_path) != manifest["output_h5ad_sha256"]:
            raise ValueError(f"M3B RNA M2B output hash mismatch: {output_path}")
        rows.append({
            "study_id": row.study_id, "dataset_id": row.dataset_id, "sample_id": row.sample_id,
            "genome_build": row.genome_build, "representation_id": "filtered_fragments_with_frozen_peak_observation_contract",
            "input_track": "F", "method_id": row.method_id, "method_category": "fragment_software",
            "gene_activity_h5ad": str(output_path.relative_to(root)), "gene_activity_sha256": manifest["output_h5ad_sha256"],
        })
    return pd.DataFrame(rows).sort_values(["method_category", "method_id", "representation_id"], kind="mergesort").reset_index(drop=True)


def _prepare_rna(
    config: dict[str, Any],
    root: Path,
    obs_names: pd.Index,
) -> tuple[sparse.csr_matrix, pd.DataFrame, np.ndarray, dict[str, str]]:
    features = _features(root / config["rna_features"])
    biotypes, unique_names = _gtf_annotations(root / config["gene_annotation"])
    features["gene_biotype"] = features["gene_id"].map(biotypes)
    barcodes = _barcodes(root / config["rna_barcodes"])
    index = {barcode: position for position, barcode in enumerate(barcodes)}
    if any(barcode not in index for barcode in obs_names):
        raise ValueError("Final ATAC observations are not fully represented in RNA")
    with gzip.open(root / config["rna_matrix"], "rb") as handle:
        genes_by_spots = mmread(handle).tocsr()
    if genes_by_spots.shape != (len(features), len(barcodes)):
        raise ValueError("RNA matrix dimensions do not match features/barcodes")
    columns = np.asarray([index[str(barcode)] for barcode in obs_names], dtype=np.int64)
    spots_by_genes = genes_by_spots[:, columns].transpose().tocsr()
    detection = np.asarray((spots_by_genes > 0).sum(axis=0)).ravel()
    threshold = math.ceil(float(config["rna_gene_eligibility"]["minimum_detected_spot_fraction"]) * len(obs_names))
    eligible = (features["gene_biotype"].to_numpy() == config["rna_gene_eligibility"]["gene_biotype"]) & (detection >= threshold)
    eligible_indices = np.flatnonzero(eligible)
    eligible_features = features.iloc[eligible_indices].copy().reset_index(drop=True)
    eligible_features["detected_spots"] = detection[eligible_indices]
    eligible_features["detected_spot_fraction"] = detection[eligible_indices] / len(obs_names)
    normalized = normalize_total_log1p(spots_by_genes, float(config["normalization"]["target_sum"]))
    return normalized, eligible_features, eligible_indices, unique_names


def _canonicalize_ga_columns(
    matrix: sparse.csr_matrix,
    ga: ad.AnnData,
    unique_gene_names: dict[str, str],
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    if "gene_id" in ga.var:
        raw = ga.var["gene_id"].astype(str).to_numpy()
        mapped = np.asarray([re.sub(r"\.\d+$", "", value) for value in raw], dtype=object)
        mapping_rule = "direct_gene_id"
    elif np.mean([bool(re.match(r"^ENS[A-Z]*G\d+(?:\.\d+)?$", value)) for value in ga.var_names.astype(str)]) > 0.5:
        raw = ga.var_names.astype(str).to_numpy()
        mapped = np.asarray([re.sub(r"\.\d+$", "", value) for value in raw], dtype=object)
        mapping_rule = "ensembl_var_names"
    else:
        raw = ga.var_names.astype(str).to_numpy()
        mapped = np.asarray([unique_gene_names.get(value) for value in raw], dtype=object)
        mapping_rule = "unique_gtf_gene_name_to_gene_id"
    valid = np.asarray([value is not None and str(value) != "" for value in mapped], dtype=bool)
    valid_ids = mapped[valid].astype(str)
    categories = np.asarray(sorted(set(valid_ids)), dtype=object)
    category_index = {value: index for index, value in enumerate(categories)}
    valid_columns = np.flatnonzero(valid)
    indicator = sparse.csr_matrix(
        (
            np.ones(len(valid_columns), dtype=np.float64),
            (valid_columns, np.asarray([category_index[value] for value in valid_ids], dtype=np.int64)),
        ),
        shape=(matrix.shape[1], len(categories)),
    )
    collapsed = (matrix @ indicator).tocsr()
    metadata = {
        "gene_identity_mapping_rule": mapping_rule,
        "ga_input_columns": matrix.shape[1],
        "ga_mapped_columns": int(valid.sum()),
        "ga_unmapped_or_ambiguous_columns": int((~valid).sum()),
        "ga_canonical_gene_ids": len(categories),
        "ga_columns_collapsed_after_mapping": int(valid.sum() - len(categories)),
    }
    return collapsed, categories, metadata


def _rna_ranks(rna: sparse.csr_matrix, indices: np.ndarray, chunk_size: int) -> np.ndarray:
    ranks = np.empty((rna.shape[0], len(indices)), dtype=np.float32)
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        dense = rna[:, indices[start:stop]].toarray()
        ranks[:, start:stop] = stats.rankdata(dense, axis=0).astype(np.float32)
    return ranks


def _evaluate_method(
    record: pd.Series,
    root: Path,
    obs_names: pd.Index,
    eligible_features: pd.DataFrame,
    rna_normalized: sparse.csr_matrix,
    eligible_indices: np.ndarray,
    rna_ranks: np.ndarray,
    unique_gene_names: dict[str, str],
    target_sum: float,
    chunk_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    path = root / record.gene_activity_h5ad
    ga = ad.read_h5ad(path)
    if not ga.obs_names.equals(obs_names) or ga.uns.get("construction_leakage_inputs") != "none":
        raise ValueError(f"Observation/leakage contract failed: {record.method_id}/{record.representation_id}")
    matrix = ga.X.tocsr() if sparse.issparse(ga.X) else sparse.csr_matrix(ga.X)
    if np.any(matrix.data < 0) or not np.isfinite(matrix.data).all():
        raise ValueError("Gene activity matrix must be finite and nonnegative")
    normalized_original = normalize_total_log1p(matrix, target_sum)
    normalized, gene_ids, mapping_metadata = _canonicalize_ga_columns(normalized_original, ga, unique_gene_names)
    ga_index = {gene_id: index for index, gene_id in enumerate(gene_ids)}
    eligible_ids = eligible_features["gene_id"].astype(str).to_numpy()
    mapped = np.asarray([ga_index.get(gene_id, -1) for gene_id in eligible_ids], dtype=np.int64)
    present_positions = np.flatnonzero(mapped >= 0)
    correlations = np.zeros(len(eligible_ids), dtype=np.float64)
    variable = np.zeros(len(eligible_ids), dtype=bool)
    for start in range(0, len(present_positions), chunk_size):
        positions = present_positions[start : start + chunk_size]
        dense = normalized[:, mapped[positions]].toarray()
        variances = np.var(dense, axis=0)
        variable[positions] = variances > 0
        correlations[positions] = column_spearman_from_ranks(rna_ranks[:, positions], dense)
        correlations[positions[variances == 0]] = 0.0
    matched_correlations = correlations[present_positions]
    median_spearman = float(np.median(matched_correlations)) if len(matched_correlations) else 0.0
    coverage = float(variable.sum() / len(eligible_ids)) if len(eligible_ids) else 0.0
    primary = median_spearman * coverage
    rna_bulk = np.asarray(rna_normalized[:, eligible_indices].mean(axis=0)).ravel()
    ga_bulk = np.zeros(len(eligible_ids), dtype=np.float64)
    if len(present_positions):
        ga_bulk[present_positions] = np.asarray(normalized[:, mapped[present_positions]].mean(axis=0)).ravel()
    global_rho = float(stats.spearmanr(rna_bulk, ga_bulk).statistic)
    top_k = min(100, len(eligible_ids))
    rna_top = set(eligible_ids[np.argsort(rna_bulk, kind="mergesort")[-top_k:]])
    ga_top = set(eligible_ids[np.argsort(ga_bulk, kind="mergesort")[-top_k:]])
    jaccard = len(rna_top & ga_top) / len(rna_top | ga_top) if rna_top | ga_top else math.nan
    metric = {
        **record.to_dict(),
        **mapping_metadata,
        "n_observations": len(obs_names),
        "rna_eligible_genes": len(eligible_ids),
        "ga_matched_rna_eligible_genes": len(present_positions),
        "ga_variable_rna_eligible_genes": int(variable.sum()),
        "evaluable_gene_fraction": coverage,
        "median_gene_wise_spearman": median_spearman,
        "coverage_adjusted_median_gene_wise_spearman": primary,
        "global_pseudobulk_spearman": global_rho,
        "top100_gene_jaccard": jaccard,
        "undefined_or_absent_gene_correlations_set_to_zero": int((~variable).sum()),
        "status": "pass",
    }
    genes = eligible_features[["gene_id", "gene_name", "detected_spots", "detected_spot_fraction"]].copy()
    genes.insert(0, "representation_id", record.representation_id)
    genes.insert(0, "method_id", record.method_id)
    genes["ga_gene_present"] = mapped >= 0
    genes["ga_gene_variable"] = variable
    genes["spearman"] = correlations
    return metric, genes


def run(config_path: str | Path, root: str | Path = ".") -> dict[str, Path]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    _validate_frozen(config, root)
    methods = _method_inputs(config, root)
    first = ad.read_h5ad(root / methods.iloc[0].gene_activity_h5ad, backed="r")
    obs_names = first.obs_names.copy()
    first.file.close()
    rna, eligible_features, eligible_indices, unique_gene_names = _prepare_rna(config, root, obs_names)
    ranks = _rna_ranks(rna, eligible_indices, int(config["correlation_chunk_genes"]))
    metric_rows: list[dict[str, Any]] = []
    gene_frames: list[pd.DataFrame] = []
    for _, method in methods.iterrows():
        metric, genes = _evaluate_method(
            method, root, obs_names, eligible_features, rna, eligible_indices, ranks,
            unique_gene_names,
            float(config["normalization"]["target_sum"]), int(config["correlation_chunk_genes"]),
        )
        metric_rows.append(metric)
        gene_frames.append(genes)
    metrics = pd.DataFrame(metric_rows)
    gene_metrics = pd.concat(gene_frames, ignore_index=True)
    not_applicable = pd.DataFrame([
        {"method_id": method, "status": "not_applicable", "reason": reason, "included_in_primary_denominator": False}
        for method, reason in config["not_applicable_methods"].items()
    ])
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    methods_path = output / "g4_m3b_rna_method_input_manifest.tsv"
    eligible_path = output / "g4_m3b_rna_eligible_genes.tsv"
    metrics_path = output / "g4_m3b_rna_metrics.tsv"
    genes_path = output / "g4_m3b_rna_gene_level_correlations.tsv.gz"
    not_applicable_path = output / "g4_m3b_rna_not_applicable.tsv"
    manifest_path = output / "g4_m3b_rna_evaluation_manifest.json"
    methods.to_csv(methods_path, sep="\t", index=False)
    eligible_features.to_csv(eligible_path, sep="\t", index=False)
    metrics.to_csv(metrics_path, sep="\t", index=False)
    gene_metrics.to_csv(genes_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    not_applicable.to_csv(not_applicable_path, sep="\t", index=False)
    artifacts = [methods_path, eligible_path, metrics_path, genes_path, not_applicable_path]
    manifest = {
        "schema_version": 1, "status": "pass", "task_id": "paired_rna_fidelity",
        "methods_expected": 31, "methods_passed": int((metrics.status == "pass").sum()),
        "rna_eligible_genes": len(eligible_features), "n_observations": len(obs_names),
        "rna_used_for_construction": False, "construction_outputs_mutated": False,
        "labels_or_clusters_used": False, "holdout_accessed": False,
        "full_benchmark_run": False, "ranking_run": False,
        "artifacts": {path.name: {"path": str(path.relative_to(root)), "sha256": sha256(path)} for path in artifacts},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": manifest_path, "metrics": metrics_path, "genes": genes_path}


def summarize(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    _validate_frozen(config, root)
    output = root / config["output_root"]
    manifest_path = output / "g4_m3b_rna_evaluation_manifest.json"
    memory_path = output / "process_tree_memory.json"
    manifest = _load_json(manifest_path)
    memory = _load_json(memory_path)
    if (
        manifest.get("status") != "pass"
        or manifest.get("methods_passed") != 31
        or manifest.get("methods_expected") != 31
    ):
        raise ValueError("M3B RNA evaluation manifest does not pass")
    if memory.get("status") != "pass" or memory.get("exit_code") != 0:
        raise ValueError("M3B RNA process-tree memory evidence does not pass")
    artifacts_valid = all(
        (root / record["path"]).is_file() and sha256(root / record["path"]) == record["sha256"]
        for record in manifest["artifacts"].values()
    )
    if not artifacts_valid:
        raise ValueError("M3B RNA output artifact hash mismatch")
    metrics_path = output / "g4_m3b_rna_metrics.tsv"
    metrics = pd.read_csv(metrics_path, sep="\t")
    primary = config["primary_endpoint"]["id"]
    required_columns = {
        "method_id", "method_category", "representation_id", "input_track",
        "rna_eligible_genes", "evaluable_gene_fraction", "median_gene_wise_spearman", primary,
    }
    if len(metrics) != 31 or not required_columns <= set(metrics):
        raise ValueError("M3B RNA metric panel is incomplete")
    if not np.isfinite(metrics[list(required_columns - {"method_id", "method_category", "representation_id", "input_track"})].to_numpy(dtype=float)).all():
        raise ValueError("M3B RNA metrics contain non-finite values")
    expected_primary = metrics["median_gene_wise_spearman"] * metrics["evaluable_gene_fraction"]
    if not np.allclose(metrics[primary], expected_primary, rtol=1e-12, atol=1e-12):
        raise ValueError("M3B RNA primary endpoint formula mismatch")
    if not ((metrics[primary] >= -1) & (metrics[primary] <= 1)).all():
        raise ValueError("M3B RNA primary endpoint is outside [-1, 1]")
    annotation = metrics[metrics["method_category"] == "hand_written_annotation"]
    controls = metrics[metrics["method_id"] == "accessibility_matched_random_peak_to_gene_control_pc"][
        ["representation_id", primary]
    ].rename(columns={primary: "accessibility_matched_control"})
    deltas = annotation.merge(controls, on="representation_id", how="left", validate="many_to_one")
    deltas["delta_vs_accessibility_matched_control"] = deltas[primary] - deltas["accessibility_matched_control"]
    deltas = deltas[["representation_id", "input_track", "method_id", primary, "accessibility_matched_control", "delta_vs_accessibility_matched_control"]]
    sensitivity = annotation.groupby("method_id", sort=True)[primary].agg(["min", "max", "median"]).reset_index()
    sensitivity["representation_range"] = sensitivity["max"] - sensitivity["min"]
    sensitivity["representations"] = annotation.groupby("method_id")["representation_id"].nunique().reindex(sensitivity.method_id).to_numpy()
    scope = metrics.groupby(["method_category", "input_track"], sort=True)[primary].agg(["count", "min", "median", "max"]).reset_index()
    scope["comparison_scope"] = np.where(scope["input_track"] == "F", "fragment_native_separate_track", "matched_representation_track")
    delta_path = output / "g4_m3b_rna_matched_control_deltas.tsv"
    sensitivity_path = output / "g4_m3b_rna_representation_sensitivity.tsv"
    scope_path = output / "g4_m3b_rna_scope_summary.tsv"
    deltas.to_csv(delta_path, sep="\t", index=False)
    sensitivity.to_csv(sensitivity_path, sep="\t", index=False)
    scope.to_csv(scope_path, sep="\t", index=False)
    gene_rows = sum(1 for _ in gzip.open(output / "g4_m3b_rna_gene_level_correlations.tsv.gz", "rt", encoding="utf-8")) - 1
    if gene_rows != 31 * int(manifest["rna_eligible_genes"]):
        raise ValueError("M3B RNA gene-level table row count mismatch")
    retained_failure = output / "attempt_1_snapatac_gene_identity_failure" / "failure.json"
    if not retained_failure.is_file():
        raise ValueError("M3B RNA resolved gene-identity failure record is missing")
    summary_artifacts = {
        "matched_control_deltas": delta_path,
        "representation_sensitivity": sensitivity_path,
        "scope_summary": scope_path,
    }
    evidence = {
        "schema_version": 1, "gate": "G4", "milestone": "M3B_paired_rna_fidelity_evaluation",
        "status": "pass", "dataset_id": config["dataset_id"], "methods_expected": 31,
        "methods_passed": len(metrics), "rna_eligible_genes": int(manifest["rna_eligible_genes"]),
        "n_observations": int(manifest["n_observations"]), "not_applicable_methods": 2,
        "primary_endpoint": config["primary_endpoint"], "normalization": config["normalization"],
        "evaluation_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256(manifest_path)},
        "process_tree_memory": {"path": str(memory_path.relative_to(root)), "sha256": sha256(memory_path),
            "runtime_seconds": memory["runtime_seconds"], "peak_aggregate_pss_mb": memory["peak_aggregate_pss_mb"]},
        "artifacts": manifest["artifacts"], "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "summary_artifacts": {
            key: {"path": str(path.relative_to(root)), "sha256": sha256(path), "rows": len(pd.read_csv(path, sep="\t"))}
            for key, path in summary_artifacts.items()
        },
        "gene_level_rows": gene_rows,
        "retained_resolved_failure": {"path": str(retained_failure.relative_to(root)), "sha256": sha256(retained_failure)},
        "development_outcome_summary_generated": True,
        "generalization_scope": "single_paired_rna_development_study_no_cross_study_inference",
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "implementation": {"path": "spatialgabench/g4_m3b_rna.py", "sha256": sha256(Path(__file__))},
        "rna_used_for_construction": False, "construction_outputs_mutated": False,
        "labels_or_clusters_used": False, "holdout_accessed": False, "full_benchmark_run": False,
        "method_ranking_run": False, "universal_ranking_run": False, "recommendation_policy_fit_run": False,
        "final_ranking_run": False, "manuscript_results_written": False, "next_milestone_authorized": False,
    }
    evidence_path = output / "g4_m3b_rna_completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--config", default="configs/g4_m3b_rna_fidelity.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = run(args.config, args.root) if args.command == "run" else {"evidence": summarize(args.config, args.root)}
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
