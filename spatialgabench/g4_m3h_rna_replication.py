"""Second-study paired-RNA replication on GSE205052/GSE205054."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats

from spatialgabench.g4_m3b_rna import (
    _canonicalize_ga_columns,
    _gtf_annotations,
    _method_inputs,
    _rna_ranks,
    column_spearman_from_ranks,
    normalize_total_log1p,
)


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wide_tsv_identity(path: str | Path) -> tuple[list[str], list[str]]:
    """Read only barcode headers and gene identifiers, not expression values."""
    barcodes: list[str]
    genes: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        if not header:
            raise ValueError("RNA wide TSV header is empty")
        barcodes = header.split("\t")
        for line in handle:
            if not line.strip():
                continue
            gene, separator, _ = line.partition("\t")
            if not separator or not gene:
                raise ValueError("RNA wide TSV row lacks a gene and count columns")
            genes.append(gene)
    return barcodes, genes


def load_wide_tsv_sparse(path: str | Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """Load a gene-by-spot wide count table into sparse CSR without dense expansion."""
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    genes: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        if not header:
            raise ValueError("RNA wide TSV header is empty")
        barcodes = header.split("\t")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            row_index = len(genes)
            gene, separator, counts = line.partition("\t")
            if not separator or not gene:
                raise ValueError("RNA wide TSV row lacks a gene and count columns")
            values = np.fromstring(counts, sep="\t", dtype=np.int64)
            if len(values) != len(barcodes):
                raise ValueError(f"RNA wide TSV row width mismatch for {gene}")
            if np.any(values < 0):
                raise ValueError("RNA counts must be nonnegative")
            nonzero = np.flatnonzero(values)
            if len(nonzero):
                row_parts.append(np.full(len(nonzero), row_index, dtype=np.int64))
                column_parts.append(nonzero.astype(np.int64, copy=False))
                data_parts.append(values[nonzero])
            genes.append(gene)
    if len(set(barcodes)) != len(barcodes) or len(set(genes)) != len(genes):
        raise ValueError("RNA wide TSV barcodes and genes must be unique")
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
    columns = np.concatenate(column_parts) if column_parts else np.empty(0, dtype=np.int64)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.int64)
    matrix = sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(genes), len(barcodes)), dtype=np.int64
    ).tocsr()
    return matrix, genes, barcodes


def _spatial_barcodes(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row[0] for row in csv.reader(handle) if row]


def _validate_hashes(config: dict[str, Any], root: Path, keys: tuple[str, ...]) -> None:
    for key in keys:
        path = root / config[key]
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"M3H frozen artifact mismatch: {key}")


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    _validate_hashes(
        config,
        root,
        (
            "authorization",
            "previous_gate",
            "rna_matrix",
            "rna_spatial_positions",
            "atac_spatial_positions",
            "final_atac_observations",
            "gene_annotation",
            "source_atac_metadata",
            "source_rna_metadata",
        ),
    )
    authorization = _load_json(root / config["authorization"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "M3H_GSE205052_RNA_input_preflight"
        or authorization.get("rna_expression_values_authorized") is not False
        or authorization.get("holdout_access_authorized") is not False
    ):
        raise PermissionError("M3H input-preflight authorization is invalid")
    if config.get("rna_expression_values_allowed") is not False:
        raise PermissionError("M3H preflight cannot read RNA expression values")

    rna_barcodes, genes = wide_tsv_identity(root / config["rna_matrix"])
    rna_spatial = _spatial_barcodes(root / config["rna_spatial_positions"])
    atac_spatial = _spatial_barcodes(root / config["atac_spatial_positions"])
    atac = ad.read_h5ad(root / config["final_atac_observations"], backed="r")
    atac_barcodes = atac.obs_names.astype(str).tolist()
    atac_shape = atac.shape
    atac_study = str(atac.uns.get("study_id", ""))
    atac_dataset = str(atac.uns.get("dataset_id", ""))
    atac_leakage = str(atac.uns.get("construction_leakage_inputs", ""))
    atac.file.close()

    biotypes, unique_names = _gtf_annotations(root / config["gene_annotation"])
    mapped_ids = [unique_names.get(gene) for gene in genes]
    mapped_biotypes = [biotypes.get(gene_id, "") if gene_id else "" for gene_id in mapped_ids]
    rna_set = set(rna_barcodes)
    atac_set = set(atac_barcodes)
    rna_spatial_set = set(rna_spatial)
    atac_spatial_set = set(atac_spatial)
    selected_barcodes = [barcode for barcode in atac_barcodes if barcode in rna_set]
    selected_set = set(selected_barcodes)

    checks = {
        "rna_barcodes_unique": len(rna_set) == len(rna_barcodes),
        "rna_genes_unique": len(set(genes)) == len(genes),
        "rna_spatial_barcodes_unique": len(rna_spatial_set) == len(rna_spatial),
        "atac_spatial_barcodes_unique": len(atac_spatial_set) == len(atac_spatial),
        "spatial_files_byte_identical": config["rna_spatial_positions_sha256"]
        == config["atac_spatial_positions_sha256"],
        "rna_header_is_spatial_subset": rna_set <= rna_spatial_set,
        "rna_header_is_final_atac_subset": rna_set <= atac_set,
        "final_atac_matches_spatial_grid": atac_set == atac_spatial_set,
        "selected_observations_match_expected": len(selected_barcodes)
        == int(config["expected_rna_observations"]),
        "atac_only_observations_match_expected": len(atac_set - rna_set)
        == int(config["expected_atac_only_observations"]),
        "gene_names_map_to_gtf": any(value is not None for value in mapped_ids),
        "study_identity_matches": atac_study == config["study_id"],
        "dataset_identity_matches": atac_dataset == config["dataset_id"],
        "construction_leakage_none": atac_leakage == "none",
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"M3H pairing preflight failed: {failed}")

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    observation_path = output / "observation_contract.tsv"
    gene_path = output / "gene_identity_preflight.tsv"
    evidence_path = output / "pairing_preflight_evidence.json"
    pd.DataFrame(
        {
            "barcode": atac_barcodes,
            "atac_order": np.arange(len(atac_barcodes)),
            "in_rna_matrix_header": [barcode in rna_set for barcode in atac_barcodes],
            "in_shared_spatial_grid": [
                barcode in rna_spatial_set and barcode in atac_spatial_set for barcode in atac_barcodes
            ],
            "evaluation_included": [barcode in selected_set for barcode in atac_barcodes],
            "exclusion_reason": [
                "" if barcode in selected_set else "absent_from_author_RNA_matrix_header"
                for barcode in atac_barcodes
            ],
        }
    ).to_csv(observation_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "rna_gene_name": genes,
            "canonical_gene_id": mapped_ids,
            "canonical_gene_biotype": mapped_biotypes,
            "unique_gtf_name_mapping": [value is not None for value in mapped_ids],
        }
    ).to_csv(gene_path, sep="\t", index=False)
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G4",
            "milestone": "M3H_GSE205052_RNA_input_preflight",
            "status": "pass",
            "study_id": config["study_id"],
            "dataset_id": config["dataset_id"],
            "atac_sample_id": config["atac_sample_id"],
            "rna_sample_id": config["rna_sample_id"],
            "genome_build": config["genome_build"],
            "same_section_source_evidence": True,
            "same_spatial_grid_checksum": config["rna_spatial_positions_sha256"],
            "rna_matrix_format": "gene_by_spot_wide_tsv_gzip",
            "rna_matrix_barcodes": len(rna_barcodes),
            "rna_matrix_gene_rows": len(genes),
            "rna_spatial_grid_barcodes": len(rna_spatial),
            "final_atac_observations": atac_shape[0],
            "final_atac_features_reference_only": atac_shape[1],
            "evaluation_observations_frozen": len(selected_barcodes),
            "atac_only_observations_excluded": len(atac_set - rna_set),
            "observation_policy": "ATAC_order_intersection_with_author_RNA_matrix_header_no_imputation",
            "mapped_gtf_gene_names": sum(value is not None for value in mapped_ids),
            "mapped_protein_coding_gene_names": sum(
                value == "protein_coding" for value in mapped_biotypes
            ),
            "checks": checks,
            "artifacts": {
                "observation_contract.tsv": {
                    "path": str(observation_path.relative_to(root)),
                    "sha256": sha256(observation_path),
                },
                "gene_identity_preflight.tsv": {
                    "path": str(gene_path.relative_to(root)),
                    "sha256": sha256(gene_path),
                },
            },
            "rna_expression_values_read": False,
            "gene_activity_outcomes_read": False,
            "rna_agreement_computed": False,
            "construction_outputs_mutated": False,
            "labels_or_clusters_accessed": False,
            "holdout_accessed": False,
            "recommendation_policy_fit_run": False,
            "full_benchmark_run": False,
            "universal_ranking_run": False,
            "next_milestone_authorized": False,
        },
    )
    return evidence_path


def _prepare_rna(
    config: dict[str, Any], root: Path, obs_names: pd.Index
) -> tuple[sparse.csr_matrix, pd.DataFrame, np.ndarray, dict[str, str]]:
    genes_by_spots, gene_names, barcodes = load_wide_tsv_sparse(root / config["rna_matrix"])
    biotypes, unique_names = _gtf_annotations(root / config["gene_annotation"])
    gene_ids = [unique_names.get(name) for name in gene_names]
    features = pd.DataFrame({"gene_name": gene_names, "gene_id": gene_ids})
    features["gene_biotype"] = features["gene_id"].map(biotypes)
    barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
    if any(str(barcode) not in barcode_index for barcode in obs_names):
        raise ValueError("Frozen evaluation observations are missing from RNA")
    columns = np.asarray([barcode_index[str(barcode)] for barcode in obs_names], dtype=np.int64)
    spots_by_genes = genes_by_spots[:, columns].transpose().tocsr()
    detection = np.asarray((spots_by_genes > 0).sum(axis=0)).ravel()
    threshold = math.ceil(
        float(config["rna_gene_eligibility"]["minimum_detected_spot_fraction"])
        * len(obs_names)
    )
    eligible = (
        features["gene_biotype"].to_numpy()
        == config["rna_gene_eligibility"]["gene_biotype"]
    ) & (detection >= threshold)
    eligible_indices = np.flatnonzero(eligible)
    eligible_features = features.iloc[eligible_indices].copy().reset_index(drop=True)
    eligible_features["detected_spots"] = detection[eligible_indices]
    eligible_features["detected_spot_fraction"] = detection[eligible_indices] / len(obs_names)
    normalized = normalize_total_log1p(
        spots_by_genes, float(config["normalization"]["target_sum"])
    )
    return normalized, eligible_features, eligible_indices, unique_names


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
    ga = ad.read_h5ad(root / record.gene_activity_h5ad)
    if ga.uns.get("construction_leakage_inputs") != "none":
        raise ValueError("Gene activity construction leakage contract failed")
    selected = ga.obs_names.get_indexer(obs_names)
    if np.any(selected < 0) or len(set(selected)) != len(selected):
        raise ValueError("Frozen RNA observations do not map uniquely to gene activity")
    source = ga.X.tocsr() if sparse.issparse(ga.X) else sparse.csr_matrix(ga.X)
    matrix = source[selected, :].tocsr()
    if np.any(matrix.data < 0) or not np.isfinite(matrix.data).all():
        raise ValueError("Gene activity matrix must be finite and nonnegative")
    normalized_original = normalize_total_log1p(matrix, target_sum)
    normalized, gene_ids, mapping_metadata = _canonicalize_ga_columns(
        normalized_original, ga, unique_gene_names
    )
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
    matched = correlations[present_positions]
    median_spearman = float(np.median(matched)) if len(matched) else 0.0
    coverage = float(variable.sum() / len(eligible_ids)) if len(eligible_ids) else 0.0
    rna_bulk = np.asarray(rna_normalized[:, eligible_indices].mean(axis=0)).ravel()
    ga_bulk = np.zeros(len(eligible_ids), dtype=np.float64)
    if len(present_positions):
        ga_bulk[present_positions] = np.asarray(
            normalized[:, mapped[present_positions]].mean(axis=0)
        ).ravel()
    global_rho = float(stats.spearmanr(rna_bulk, ga_bulk).statistic)
    top_k = min(100, len(eligible_ids))
    rna_top = set(eligible_ids[np.argsort(rna_bulk, kind="mergesort")[-top_k:]])
    ga_top = set(eligible_ids[np.argsort(ga_bulk, kind="mergesort")[-top_k:]])
    jaccard = len(rna_top & ga_top) / len(rna_top | ga_top) if rna_top | ga_top else math.nan
    metric = {
        **record.to_dict(),
        **mapping_metadata,
        "n_observations": len(obs_names),
        "source_gene_activity_observations": ga.n_obs,
        "rna_eligible_genes": len(eligible_ids),
        "ga_matched_rna_eligible_genes": len(present_positions),
        "ga_variable_rna_eligible_genes": int(variable.sum()),
        "evaluable_gene_fraction": coverage,
        "median_gene_wise_spearman": median_spearman,
        "coverage_adjusted_median_gene_wise_spearman": median_spearman * coverage,
        "global_pseudobulk_spearman": global_rho,
        "top100_gene_jaccard": jaccard,
        "undefined_or_absent_gene_correlations_set_to_zero": int((~variable).sum()),
        "status": "pass",
    }
    genes = eligible_features[
        ["gene_id", "gene_name", "detected_spots", "detected_spot_fraction"]
    ].copy()
    genes.insert(0, "representation_id", record.representation_id)
    genes.insert(0, "method_id", record.method_id)
    genes["ga_gene_present"] = mapped >= 0
    genes["ga_gene_variable"] = variable
    genes["spearman"] = correlations
    return metric, genes


def _validate_evaluation(config: dict[str, Any], root: Path) -> None:
    _validate_hashes(
        config,
        root,
        (
            "authorization",
            "previous_gate",
            "preflight_evidence",
            "observation_contract",
            "rna_matrix",
            "gene_annotation",
            "m2a_evidence",
            "m2b_evidence",
        ),
    )
    authorization = _load_json(root / config["authorization"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "M3H_GSE205052_RNA_fidelity_replication"
        or authorization.get("rna_evaluation_authorized") is not True
        or authorization.get("construction_parameter_tuning_authorized") is not False
        or authorization.get("holdout_access_authorized") is not False
        or authorization.get("recommendation_policy_fit_authorized") is not False
    ):
        raise PermissionError("M3H RNA evaluation authorization is invalid")
    preflight_evidence = _load_json(root / config["preflight_evidence"])
    if (
        preflight_evidence.get("status") != "pass"
        or preflight_evidence.get("rna_expression_values_read") is not False
        or preflight_evidence.get("evaluation_observations_frozen")
        != int(config["expected_evaluation_observations"])
    ):
        raise ValueError("M3H RNA preflight evidence is invalid")
    if (
        config.get("labels_or_clusters_allowed") is not False
        or config.get("ranking_allowed") is not False
        or config.get("policy_fit_allowed") is not False
    ):
        raise PermissionError("M3H cannot use labels, ranking, or policy fitting")


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    _validate_evaluation(config, root)
    methods = _method_inputs(config, root)
    observation_contract = pd.read_csv(root / config["observation_contract"], sep="\t")
    obs_names = pd.Index(
        observation_contract.loc[observation_contract["evaluation_included"], "barcode"].astype(str)
    )
    if len(obs_names) != int(config["expected_evaluation_observations"]):
        raise ValueError("M3H frozen observation contract has the wrong size")
    rna, eligible_features, eligible_indices, unique_gene_names = _prepare_rna(
        config, root, obs_names
    )
    ranks = _rna_ranks(rna, eligible_indices, int(config["correlation_chunk_genes"]))
    metric_rows: list[dict[str, Any]] = []
    gene_frames: list[pd.DataFrame] = []
    for _, method in methods.iterrows():
        metric, genes = _evaluate_method(
            method,
            root,
            obs_names,
            eligible_features,
            rna,
            eligible_indices,
            ranks,
            unique_gene_names,
            float(config["normalization"]["target_sum"]),
            int(config["correlation_chunk_genes"]),
        )
        metric_rows.append(metric)
        gene_frames.append(genes)
    metrics = pd.DataFrame(metric_rows)
    gene_metrics = pd.concat(gene_frames, ignore_index=True)
    not_applicable = pd.DataFrame(
        [
            {
                "method_id": method,
                "status": "not_applicable",
                "reason": reason,
                "included_in_primary_denominator": False,
            }
            for method, reason in config["not_applicable_methods"].items()
        ]
    )
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    methods_path = output / "method_input_manifest.tsv"
    eligible_path = output / "rna_eligible_genes.tsv"
    metrics_path = output / "rna_metrics.tsv"
    genes_path = output / "gene_level_correlations.tsv.gz"
    not_applicable_path = output / "not_applicable.tsv"
    manifest_path = output / "evaluation_manifest.json"
    methods.to_csv(methods_path, sep="\t", index=False)
    eligible_features.to_csv(eligible_path, sep="\t", index=False)
    metrics.to_csv(metrics_path, sep="\t", index=False)
    gene_metrics.to_csv(
        genes_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0}
    )
    not_applicable.to_csv(not_applicable_path, sep="\t", index=False)
    artifacts = [methods_path, eligible_path, metrics_path, genes_path, not_applicable_path]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "pass",
            "task_id": "paired_rna_fidelity",
            "study_id": config["study_id"],
            "dataset_id": config["dataset_id"],
            "methods_expected": 31,
            "methods_passed": int((metrics.status == "pass").sum()),
            "rna_eligible_genes": len(eligible_features),
            "n_observations": len(obs_names),
            "source_atac_observations": int(metrics["source_gene_activity_observations"].iloc[0]),
            "observation_policy": "ATAC_order_intersection_with_author_RNA_matrix_header_no_imputation",
            "rna_used_for_construction": False,
            "construction_outputs_mutated": False,
            "labels_or_clusters_used": False,
            "holdout_accessed": False,
            "recommendation_policy_fit_run": False,
            "full_benchmark_run": False,
            "ranking_run": False,
            "artifacts": {
                path.name: {"path": str(path.relative_to(root)), "sha256": sha256(path)}
                for path in artifacts
            },
        },
    )
    return manifest_path


def summarize(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    _validate_evaluation(config, root)
    output = root / config["output_root"]
    manifest_path = output / "evaluation_manifest.json"
    memory_path = output / "process_tree_memory.json"
    manifest = _load_json(manifest_path)
    memory = _load_json(memory_path)
    if (
        manifest.get("status") != "pass"
        or manifest.get("methods_passed") != 31
        or manifest.get("methods_expected") != 31
        or manifest.get("n_observations") != int(config["expected_evaluation_observations"])
    ):
        raise ValueError("M3H evaluation manifest does not pass")
    if memory.get("status") != "pass" or memory.get("exit_code") != 0:
        raise ValueError("M3H process-tree memory evidence does not pass")
    if not all(
        (root / record["path"]).is_file()
        and sha256(root / record["path"]) == record["sha256"]
        for record in manifest["artifacts"].values()
    ):
        raise ValueError("M3H output artifact hash mismatch")
    metrics = pd.read_csv(output / "rna_metrics.tsv", sep="\t")
    primary = config["primary_endpoint"]["id"]
    required = {
        "method_id",
        "method_category",
        "representation_id",
        "input_track",
        "rna_eligible_genes",
        "evaluable_gene_fraction",
        "median_gene_wise_spearman",
        primary,
    }
    if len(metrics) != 31 or not required <= set(metrics):
        raise ValueError("M3H RNA metric panel is incomplete")
    expected_primary = metrics["median_gene_wise_spearman"] * metrics["evaluable_gene_fraction"]
    if not np.allclose(metrics[primary], expected_primary, rtol=1e-12, atol=1e-12):
        raise ValueError("M3H primary endpoint formula mismatch")
    annotation = metrics[metrics["method_category"] == "hand_written_annotation"]
    controls = metrics[
        metrics["method_id"] == "accessibility_matched_random_peak_to_gene_control_pc"
    ][["representation_id", primary]].rename(columns={primary: "accessibility_matched_control"})
    deltas = annotation.merge(controls, on="representation_id", how="left", validate="many_to_one")
    deltas["delta_vs_accessibility_matched_control"] = (
        deltas[primary] - deltas["accessibility_matched_control"]
    )
    deltas = deltas[
        [
            "representation_id",
            "input_track",
            "method_id",
            primary,
            "accessibility_matched_control",
            "delta_vs_accessibility_matched_control",
        ]
    ]
    sensitivity = (
        annotation.groupby("method_id", sort=True)[primary]
        .agg(["min", "max", "median"])
        .reset_index()
    )
    sensitivity["representation_range"] = sensitivity["max"] - sensitivity["min"]
    sensitivity["representations"] = (
        annotation.groupby("method_id")["representation_id"]
        .nunique()
        .reindex(sensitivity.method_id)
        .to_numpy()
    )
    scope = (
        metrics.groupby(["method_category", "input_track"], sort=True)[primary]
        .agg(["count", "min", "median", "max"])
        .reset_index()
    )
    scope["comparison_scope"] = np.where(
        scope["input_track"] == "F",
        "fragment_native_separate_track",
        "matched_representation_track",
    )
    delta_path = output / "matched_control_deltas.tsv"
    sensitivity_path = output / "representation_sensitivity.tsv"
    scope_path = output / "scope_summary.tsv"
    deltas.to_csv(delta_path, sep="\t", index=False)
    sensitivity.to_csv(sensitivity_path, sep="\t", index=False)
    scope.to_csv(scope_path, sep="\t", index=False)
    gene_rows = sum(
        1 for _ in gzip.open(output / "gene_level_correlations.tsv.gz", "rt", encoding="utf-8")
    ) - 1
    if gene_rows != 31 * int(manifest["rna_eligible_genes"]):
        raise ValueError("M3H RNA gene-level row count mismatch")
    evidence_path = output / "completion_evidence.json"
    summary_artifacts = {
        "matched_control_deltas": delta_path,
        "representation_sensitivity": sensitivity_path,
        "scope_summary": scope_path,
    }
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G4",
            "milestone": "M3H_GSE205052_RNA_fidelity_replication",
            "status": "pass",
            "study_id": config["study_id"],
            "dataset_id": config["dataset_id"],
            "methods_expected": 31,
            "methods_passed": len(metrics),
            "rna_eligible_genes": int(manifest["rna_eligible_genes"]),
            "n_observations": int(manifest["n_observations"]),
            "source_atac_observations": int(manifest["source_atac_observations"]),
            "not_applicable_methods": 2,
            "primary_endpoint": config["primary_endpoint"],
            "normalization": config["normalization"],
            "evaluation_manifest": {
                "path": str(manifest_path.relative_to(root)),
                "sha256": sha256(manifest_path),
            },
            "process_tree_memory": {
                "path": str(memory_path.relative_to(root)),
                "sha256": sha256(memory_path),
                "runtime_seconds": memory["runtime_seconds"],
                "peak_aggregate_pss_mb": memory["peak_aggregate_pss_mb"],
            },
            "artifacts": manifest["artifacts"],
            "summary_artifacts": {
                key: {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256(path),
                    "rows": len(pd.read_csv(path, sep="\t")),
                }
                for key, path in summary_artifacts.items()
            },
            "gene_level_rows": gene_rows,
            "generalization_scope": "second_development_study_family_replication_no_holdout_inference",
            "observation_policy": manifest["observation_policy"],
            "rna_used_for_construction": False,
            "construction_outputs_mutated": False,
            "labels_or_clusters_used": False,
            "holdout_accessed": False,
            "recommendation_policy_fit_run": False,
            "full_benchmark_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "next_milestone_authorized": False,
        },
    )
    return evidence_path


def synthesize(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    _validate_hashes(
        config,
        root,
        (
            "authorization",
            "m3b_gate",
            "m3b_metrics",
            "m3h_completion",
            "m3h_metrics",
        ),
    )
    authorization = _load_json(root / config["authorization"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "M3H_cross_study_RNA_synthesis"
        or authorization.get("holdout_access_authorized") is not False
        or authorization.get("recommendation_policy_fit_authorized") is not False
    ):
        raise PermissionError("M3H synthesis authorization is invalid")
    frames = []
    for study_id, path_key in config["study_metrics"].items():
        frame = pd.read_csv(root / config[path_key], sep="\t")
        frame["study_family"] = study_id
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    primary = config["primary_endpoint"]
    matrix = metrics[metrics["input_track"].isin(["P", "B"])].copy()
    annotation = matrix[matrix["method_category"] == "hand_written_annotation"].copy()
    controls = matrix[
        matrix["method_id"] == "accessibility_matched_random_peak_to_gene_control_pc"
    ][["study_family", "representation_id", primary]].rename(
        columns={primary: "accessibility_matched_control"}
    )
    deltas = annotation.merge(
        controls, on=["study_family", "representation_id"], how="left", validate="many_to_one"
    )
    deltas["delta_vs_accessibility_matched_control"] = (
        deltas[primary] - deltas["accessibility_matched_control"]
    )
    direction = (
        deltas.groupby(["method_id", "representation_id"], sort=True)
        .agg(
            study_families=("study_family", "nunique"),
            studies_positive=("delta_vs_accessibility_matched_control", lambda x: int((x > 0).sum())),
            minimum_delta=("delta_vs_accessibility_matched_control", "min"),
            median_delta=("delta_vs_accessibility_matched_control", "median"),
            maximum_delta=("delta_vs_accessibility_matched_control", "max"),
        )
        .reset_index()
    )
    direction["positive_direction_replicates"] = (
        (direction["study_families"] == int(config["expected_study_families"]))
        & (direction["studies_positive"] == int(config["expected_study_families"]))
    )
    study_values = deltas[
        [
            "study_family",
            "dataset_id",
            "representation_id",
            "input_track",
            "method_id",
            primary,
            "accessibility_matched_control",
            "delta_vs_accessibility_matched_control",
        ]
    ].sort_values(["representation_id", "method_id", "study_family"], kind="mergesort")
    representation = (
        annotation.groupby(["study_family", "method_id"], sort=True)[primary]
        .agg(["min", "max", "median"])
        .reset_index()
    )
    representation["representation_range"] = representation["max"] - representation["min"]
    software = metrics[metrics["input_track"] == "F"][
        ["study_family", "dataset_id", "method_id", primary, "status"]
    ].sort_values(["method_id", "study_family"], kind="mergesort")
    if (
        metrics["study_family"].nunique() != int(config["expected_study_families"])
        or direction["study_families"].min() != int(config["expected_study_families"])
    ):
        raise ValueError("M3H cross-study synthesis lacks complete study coverage")
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    study_path = output / "study_level_matched_control_deltas.tsv"
    direction_path = output / "cross_study_direction_replication.tsv"
    representation_path = output / "study_level_representation_sensitivity.tsv"
    software_path = output / "fragment_software_descriptive.tsv"
    evidence_path = output / "cross_study_synthesis_evidence.json"
    study_values.to_csv(study_path, sep="\t", index=False)
    direction.to_csv(direction_path, sep="\t", index=False)
    representation.to_csv(representation_path, sep="\t", index=False)
    software.to_csv(software_path, sep="\t", index=False)
    artifacts = [study_path, direction_path, representation_path, software_path]
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G4",
            "milestone": "M3H_cross_study_paired_RNA_synthesis",
            "status": "pass",
            "study_families": int(metrics["study_family"].nunique()),
            "method_runs": len(metrics),
            "matrix_track_runs": len(matrix),
            "fragment_track_runs": len(software),
            "direction_rows": len(direction),
            "positive_direction_replication_rows": int(
                direction["positive_direction_replicates"].sum()
            ),
            "primary_endpoint": primary,
            "artifacts": {
                path.name: {"path": str(path.relative_to(root)), "sha256": sha256(path)}
                for path in artifacts
            },
            "study_is_generalization_unit": True,
            "two_studies_are_directional_development_evidence_not_population_inference": True,
            "input_tracks_pooled": False,
            "representation_universal_rank_run": False,
            "recommendation_policy_fit_run": False,
            "holdout_accessed": False,
            "full_benchmark_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "next_milestone_authorized": False,
        },
    )
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "summarize", "synthesize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "preflight":
        path = preflight(args.config, args.root)
    elif args.command == "run":
        path = run(args.config, args.root)
    elif args.command == "summarize":
        path = summarize(args.config, args.root)
    else:
        path = synthesize(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
