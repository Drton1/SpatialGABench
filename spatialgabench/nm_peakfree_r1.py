"""Governed peak-free R1 construction and paired-RNA stop/go evaluation."""

from __future__ import annotations

import argparse
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

from spatialgabench.gene_activity.atomic_multiresolution import (
    fit_preprojected_atomic_multiresolution,
    prepare_atomic_weight_panel,
)
from spatialgabench.nm_paired_depth_accuracy import (
    _align_rows,
    _evaluate_matrix,
    _load_rna,
    canonicalize_unique_gene_names,
    normalize_total_log1p,
)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    loader = json.loads if path.suffix == ".json" else yaml.safe_load
    value = loader(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def fit_monotone_count_switch(
    totals: np.ndarray,
    selected_counts: np.ndarray,
) -> dict[str, Any]:
    """Fit ATAC-only 100/25/5 kb count breakpoints with coarse-favoring ties."""
    totals = np.asarray(totals, dtype=np.int64)
    selected_counts = np.asarray(selected_counts, dtype=np.int64)
    if selected_counts.shape != (len(totals), 3) or np.any(selected_counts < 0):
        raise ValueError("selected-count table has the wrong shape or negative values")
    unique = np.unique(totals)
    group_costs = np.zeros((len(unique), 3), dtype=np.int64)
    # Columns are assignment costs for 100, 25, and 5 kb.
    for index, total in enumerate(unique):
        rows = totals == total
        entries = selected_counts[rows].sum(axis=1)
        group_costs[index] = entries.sum() - selected_counts[rows][:, [2, 1, 0]].sum(axis=0)
    prefix = np.vstack([np.zeros(3, dtype=np.int64), np.cumsum(group_costs, axis=0)])
    best: tuple[int, int, int] | None = None
    for low_end in range(len(unique) + 1):
        for middle_end in range(low_end, len(unique) + 1):
            cost = int(
                prefix[low_end, 0]
                + prefix[middle_end, 1] - prefix[low_end, 1]
                + prefix[-1, 2] - prefix[middle_end, 2]
            )
            candidate = (cost, -low_end, -middle_end)
            if best is None or candidate < best:
                best = candidate
    low_end, middle_end = -best[1], -best[2]
    low_max = int(unique[low_end - 1]) if low_end else -1
    middle_max = int(unique[middle_end - 1]) if middle_end else low_max
    assigned = np.where(totals <= low_max, 100, np.where(totals <= middle_max, 25, 5)).astype(np.int16)
    return {
        "low_count_max_for_100kb": low_max,
        "middle_count_max_for_25kb": middle_max,
        "minimum_disagreement_entries": int(best[0]),
        "assigned_scale_kb": assigned,
        "unique_total_counts": int(len(unique)),
    }


def construction_preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _load(config_path)
    required = [
        config["protocol"], config["atomic_representation"], config["atomic_gene_kernel"],
        config["genes"], config["v1_1_completion"], config["v1_1_validation"],
        config["method_core"], "spatialgabench/nm_peakfree_r1.py",
    ]
    for relative in required:
        if not (root / relative).is_file():
            raise ValueError(f"R1 input missing: {relative}")
    completion = _load(root / config["v1_1_completion"])
    validation = _load(root / config["v1_1_validation"])
    if completion.get("status") != "pass" or validation.get("status") != "pass":
        raise ValueError("R1 requires passing v1.1 construction and validation")
    if validation.get("validated_decision") != "authorize_rna_protocol_freeze":
        raise PermissionError("v1.1 did not authorize the R1 protocol")
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    bindings_path = output / "construction_bindings.tsv"
    pd.DataFrame([{"path": value, "sha256": sha256_path(root / value)} for value in required]).to_csv(
        bindings_path, sep="\t", index=False
    )
    authorization = {
        "schema_version": 1,
        "authorization": "NM_PEAKFREE_R1_CONSTRUCTION",
        "milestone": config["milestone"],
        "status": "authorized",
        "atac_construction_authorized": True,
        "rna_value_access_authorized": False,
        "labels_clusters_pathology_authorized": False,
        "parameter_tuning_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "bindings": {"path": str(bindings_path.relative_to(root)), "sha256": sha256_path(bindings_path)},
    }
    auth_path = root / "governance/authorizations/NM_PEAKFREE_R1_CONSTRUCTION_2026-08-02.json"
    auth_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    evidence = {
        "schema_version": 1, "milestone": f"{config['milestone']}_CONSTRUCTION_PREFLIGHT",
        "status": "pass", "rna_expression_values_read": False,
        "labels_clusters_pathology_read": False, "parameter_tuning_run": False,
        "authorization": {"path": str(auth_path.relative_to(root)), "sha256": sha256_path(auth_path)},
    }
    path = output / "construction_preflight_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


def _validate_construction_auth(config_path: Path, auth_path: Path, root: Path) -> dict[str, Any]:
    config, auth = _load(config_path), _load(auth_path)
    if (
        auth.get("status") != "authorized"
        or auth.get("milestone") != config["milestone"]
        or auth.get("atac_construction_authorized") is not True
        or auth.get("rna_value_access_authorized") is not False
        or auth["config"]["sha256"] != sha256_path(config_path)
    ):
        raise PermissionError("R1 construction authorization is invalid")
    bindings = pd.read_csv(root / auth["bindings"]["path"], sep="\t")
    if auth["bindings"]["sha256"] != sha256_path(root / auth["bindings"]["path"]):
        raise ValueError("R1 construction bindings changed")
    for row in bindings.itertuples(index=False):
        if row.sha256 != sha256_path(root / row.path):
            raise ValueError(f"R1 construction input changed: {row.path}")
    return config


def construct(config_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _validate_construction_auth(config_path, Path(authorization_path).resolve(), root)
    output = root / config["output_root"]
    data = ad.read_h5ad(root / config["atomic_representation"])
    weights = sparse.load_npz(root / config["atomic_gene_kernel"]).tocsr()
    genes = pd.read_csv(root / config["genes"], sep="\t")
    if data.shape[0] != int(config["expected_observations"]):
        raise ValueError("R1 observation count changed")
    if int(data.X.sum()) != int(config["expected_fragment_events"]):
        raise ValueError("R1 fragment count changed")
    if weights.shape != (data.n_vars, int(config["expected_genes"])) or len(genes) != weights.shape[1]:
        raise ValueError("R1 atomic kernel dimensions changed")
    scales = tuple(int(value) for value in config["scales_kb"])
    prepared = prepare_atomic_weight_panel(data.var, weights, scales_kb=scales)
    branch_chunks = {name: [] for name in ("adaptive_v1_1", "fixed_5kb", "fixed_25kb", "fixed_100kb")}
    variance_chunks = []
    selected_counts = np.zeros((data.n_obs, 3), dtype=np.int64)
    chunk_size = int(config["gene_chunk_size"])
    for start in range(0, weights.shape[1], chunk_size):
        stop = min(start + chunk_size, weights.shape[1])
        result = fit_preprojected_atomic_multiresolution(
            data.X,
            {scale: value[:, start:stop] for scale, value in prepared.weights_by_scale.items()},
            prepared.boundary_not_applicable[start:stop],
            scales_kb=scales,
            scale_factor=float(config["scale_factor"]),
            alpha=float(config["alpha"]),
        )
        branch_chunks["adaptive_v1_1"].append(sparse.csr_matrix(result.activity_rate))
        for scale in scales:
            fixed = np.expm1(result.scale_log_rates[scale]) / float(config["scale_factor"])
            branch_chunks[f"fixed_{scale}kb"].append(sparse.csr_matrix(fixed))
            selected_counts[:, scales.index(scale)] += np.sum(
                (result.selected_scale_kb == scale) & result.adaptive_applicable, axis=1
            )
        variance = np.nan_to_num(result.selected_log_variance, nan=0.0, posinf=0.0, neginf=0.0)
        variance_chunks.append(sparse.csr_matrix(variance))

    matrices = {name: sparse.hstack(chunks, format="csr") for name, chunks in branch_chunks.items()}
    totals = np.asarray(data.X.sum(axis=1)).ravel().astype(np.int64)
    switch = fit_monotone_count_switch(totals, selected_counts)
    hard = sparse.csr_matrix(matrices["fixed_100kb"].shape, dtype=np.float64)
    for scale in scales:
        mask = switch["assigned_scale_kb"] == scale
        hard = hard + sparse.diags(mask.astype(np.float64)) @ matrices[f"fixed_{scale}kb"]
    matrices["count_hard_switch"] = hard.tocsr()
    matrix_dir = output / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for branch in config["branches"]:
        matrix = matrices[branch]
        if matrix.shape != (data.n_obs, len(genes)) or matrix.data.size and (
            not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
        ):
            raise ValueError(f"R1 branch matrix contract failed: {branch}")
        path = matrix_dir / f"{branch}.npz"
        sparse.save_npz(path, matrix, compressed=True)
        manifest_rows.append({
            "branch_id": branch, "observations": matrix.shape[0], "genes": matrix.shape[1],
            "nnz": matrix.nnz, "matrix_sum": float(matrix.sum()),
            "path": str(path.relative_to(root)), "sha256": sha256_path(path), "status": "pass",
        })
    variance_path = output / "selected_log_variance.npz"
    sparse.save_npz(variance_path, sparse.hstack(variance_chunks, format="csr"), compressed=True)
    observations_path = output / "observations.tsv"
    pd.DataFrame({"observation_id": data.obs_names.astype(str), "total_fragment_events": totals,
                  "hard_switch_scale_kb": switch["assigned_scale_kb"]}).to_csv(
        observations_path, sep="\t", index=False
    )
    genes_path = output / "genes.tsv"
    genes.to_csv(genes_path, sep="\t", index=False)
    manifest_path = output / "matrix_manifest.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
    switch_path = output / "hard_switch_contract.json"
    switch_payload = {key: value for key, value in switch.items() if key != "assigned_scale_kb"}
    switch_payload.update({
        "status": "pass", "fit_input": "ATAC_total_counts_and_adaptive_scale_calls_only",
        "rna_expression_values_read": False,
        "assigned_100kb_spots": int(np.sum(switch["assigned_scale_kb"] == 100)),
        "assigned_25kb_spots": int(np.sum(switch["assigned_scale_kb"] == 25)),
        "assigned_5kb_spots": int(np.sum(switch["assigned_scale_kb"] == 5)),
    })
    switch_path.write_text(json.dumps(switch_payload, indent=2, sort_keys=True) + "\n")
    evidence = {
        "schema_version": 1, "milestone": f"{config['milestone']}_CONSTRUCTION",
        "status": "pass", "branches": len(manifest_rows), "observations": data.n_obs,
        "genes": len(genes), "fragment_events": int(data.X.sum()),
        "rna_expression_values_read": False, "labels_clusters_pathology_read": False,
        "parameter_tuning_run": False, "universal_ranking_run": False, "final_ranking_run": False,
        "artifacts": {
            "matrix_manifest.tsv": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_path(manifest_path)},
            "observations.tsv": {"path": str(observations_path.relative_to(root)), "sha256": sha256_path(observations_path)},
            "genes.tsv": {"path": str(genes_path.relative_to(root)), "sha256": sha256_path(genes_path)},
            "selected_log_variance.npz": {"path": str(variance_path.relative_to(root)), "sha256": sha256_path(variance_path)},
            "hard_switch_contract.json": {"path": str(switch_path.relative_to(root)), "sha256": sha256_path(switch_path)},
        },
    }
    path = output / "construction_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


def rna_preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _load(config_path)
    output = root / config["output_root"]
    construction = _load(output / "construction_evidence.json")
    manifest = pd.read_csv(output / "matrix_manifest.tsv", sep="\t")
    if construction.get("status") != "pass" or set(manifest.branch_id) != set(config["branches"]):
        raise ValueError("R1 construction is incomplete")
    for row in manifest.itertuples(index=False):
        if row.sha256 != sha256_path(root / row.path):
            raise ValueError(f"R1 matrix changed before RNA preflight: {row.branch_id}")
    rna_config = _load(root / config["rna_source_config"])
    sample = rna_config["samples"][config["rna_sample_id"]]
    identities = []
    for name, record in sample.items():
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            observed = sha256_path(root / record["path"])
            if observed != record["sha256"]:
                raise ValueError(f"R1 RNA source checksum changed: {name}")
            identities.append({"artifact": name, "path": record["path"], "sha256": observed})
    identity_path = output / "rna_identity_manifest.tsv"
    pd.DataFrame(identities).to_csv(identity_path, sep="\t", index=False)
    auth = {
        "schema_version": 1, "authorization": "NM_PEAKFREE_R1_PAIRED_RNA",
        "milestone": config["milestone"], "status": "authorized",
        "paired_rna_evaluation_authorized": True, "construction_mutation_authorized": False,
        "parameter_tuning_authorized": False, "labels_clusters_pathology_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "protocol": {"path": config["protocol"], "sha256": sha256_path(root / config["protocol"])},
        "construction": {"path": str((output / "construction_evidence.json").relative_to(root)),
                         "sha256": sha256_path(output / "construction_evidence.json")},
        "matrix_manifest": {"path": str((output / "matrix_manifest.tsv").relative_to(root)),
                            "sha256": sha256_path(output / "matrix_manifest.tsv")},
        "rna_identities": {"path": str(identity_path.relative_to(root)), "sha256": sha256_path(identity_path)},
        "implementation": {"path": "spatialgabench/nm_peakfree_r1.py",
                           "sha256": sha256_path(root / "spatialgabench/nm_peakfree_r1.py")},
    }
    auth_path = root / "governance/authorizations/NM_PEAKFREE_R1_PAIRED_RNA_2026-08-02.json"
    auth_path.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n")
    evidence = {
        "schema_version": 1, "milestone": f"{config['milestone']}_RNA_PREFLIGHT",
        "status": "pass", "branches": len(manifest), "rna_expression_values_read": False,
        "construction_mutated": False,
        "authorization": {"path": str(auth_path.relative_to(root)), "sha256": sha256_path(auth_path)},
    }
    path = output / "rna_preflight_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


def evaluate(config_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config, auth = _load(config_path), _load(Path(authorization_path).resolve())
    output = root / config["output_root"]
    if (
        auth.get("status") != "authorized"
        or auth.get("paired_rna_evaluation_authorized") is not True
        or auth.get("construction_mutation_authorized") is not False
        or auth.get("parameter_tuning_authorized") is not False
        or auth["config"]["sha256"] != sha256_path(config_path)
    ):
        raise PermissionError("R1 RNA authorization is invalid")
    for key in ("protocol", "construction", "matrix_manifest", "rna_identities", "implementation"):
        if auth[key]["sha256"] != sha256_path(root / auth[key]["path"]):
            raise ValueError(f"R1 RNA bound artifact changed: {key}")
    manifest = pd.read_csv(output / "matrix_manifest.tsv", sep="\t")
    for row in manifest.itertuples(index=False):
        if row.sha256 != sha256_path(root / row.path):
            raise ValueError(f"R1 branch changed before evaluation: {row.branch_id}")
    observations = pd.read_csv(output / "observations.tsv", sep="\t").observation_id.astype(str).tolist()
    genes = pd.read_csv(output / "genes.tsv", sep="\t")
    rna_config = _load(root / config["rna_source_config"])
    sample = rna_config["samples"][config["rna_sample_id"]]
    rna_counts, rna_barcodes, rna_gene_ids, protein_coding, unique_names = _load_rna(sample, root)
    unique_names = canonicalize_unique_gene_names(unique_names)
    rna_counts = _align_rows(rna_counts, rna_barcodes, observations)
    detection = np.asarray((rna_counts > 0).sum(axis=0)).ravel()
    threshold = math.ceil(
        float(config["rna_evaluation"]["minimum_detected_observation_fraction"]) * len(observations)
    )
    eligible_indices = np.flatnonzero(protein_coding & (detection >= threshold))
    eligible_gene_ids = rna_gene_ids[eligible_indices]
    target_sum = float(config["rna_evaluation"]["target_sum"])
    rna_normalized = normalize_total_log1p(rna_counts, target_sum)
    chunk_size = int(config["rna_evaluation"]["correlation_chunk_genes"])
    rna_ranks = np.empty((len(observations), len(eligible_indices)), dtype=np.float32)
    for start in range(0, len(eligible_indices), chunk_size):
        stop = min(start + chunk_size, len(eligible_indices))
        rna_ranks[:, start:stop] = stats.rankdata(
            rna_normalized[:, eligible_indices[start:stop]].toarray(), axis=0
        ).astype(np.float32)
    metrics, gene_frames = [], []
    var = genes.set_index("gene_id", drop=False)
    for row in manifest.itertuples(index=False):
        matrix = sparse.load_npz(root / row.path).tocsr()
        activity = ad.AnnData(X=matrix, obs=pd.DataFrame(index=observations), var=var.copy())
        values, gene_frame = _evaluate_matrix(
            activity, rna_normalized, eligible_indices, eligible_gene_ids,
            rna_ranks, unique_names, target_sum, chunk_size,
        )
        metrics.append({"branch_id": row.branch_id, **values})
        gene_frame.insert(0, "branch_id", row.branch_id)
        gene_frames.append(gene_frame)
    metrics_frame = pd.DataFrame(metrics)
    primary = metrics_frame.set_index("branch_id").coverage_adjusted_median_gene_wise_spearman
    best_fixed = float(primary[["fixed_5kb", "fixed_25kb", "fixed_100kb"]].max())
    adaptive = float(primary["adaptive_v1_1"])
    regret = best_fixed - adaptive
    b25_deficit = float(primary["fixed_25kb"] - adaptive)
    gates = {
        "regret_to_best_fixed": regret <= float(config["gates"]["maximum_regret_to_best_fixed"]),
        "noninferior_to_fixed_25kb": b25_deficit <= float(config["gates"]["maximum_deficit_to_fixed_25kb"]),
    }
    metrics_frame["is_primary_adaptive"] = metrics_frame.branch_id.eq("adaptive_v1_1")
    metrics_path = output / "rna_branch_metrics.tsv"
    gene_path = output / "rna_gene_level_correlations.tsv.gz"
    metrics_frame.to_csv(metrics_path, sep="\t", index=False)
    pd.concat(gene_frames, ignore_index=True).to_csv(
        gene_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0}
    )
    decision_path = output / "rna_stopgo_decision.tsv"
    pd.DataFrame([{
        "adaptive_primary": adaptive, "best_fixed_primary": best_fixed,
        "fixed_25kb_primary": float(primary["fixed_25kb"]),
        "count_hard_switch_primary": float(primary["count_hard_switch"]),
        "regret_to_best_fixed": regret, "deficit_to_fixed_25kb": b25_deficit,
        "status": "pass" if all(gates.values()) else "fail",
    }]).to_csv(decision_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "milestone": config["milestone"],
        "status": "pass" if all(gates.values()) else "fail", "gates": gates,
        "rna_expression_values_accessed_for_evaluation": True,
        "rna_used_for_construction": False, "construction_mutated": False,
        "labels_clusters_pathology_read": False, "parameter_tuning_run": False,
        "universal_ranking_run": False, "final_ranking_run": False,
        "artifacts": {
            "rna_branch_metrics.tsv": {"path": str(metrics_path.relative_to(root)), "sha256": sha256_path(metrics_path)},
            "rna_gene_level_correlations.tsv.gz": {"path": str(gene_path.relative_to(root)), "sha256": sha256_path(gene_path)},
            "rna_stopgo_decision.tsv": {"path": str(decision_path.relative_to(root)), "sha256": sha256_path(decision_path)},
        },
    }
    evidence_path = output / "rna_completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if evidence["status"] != "pass":
        raise ValueError(f"R1 paired-RNA stop/go failed: {[name for name, value in gates.items() if not value]}")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["construction-preflight", "construct", "rna-preflight", "evaluate"])
    parser.add_argument("--config", default="configs/nm_peakfree_r1_paired_rna.yaml")
    parser.add_argument("--authorization", default="")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "construction-preflight":
        result = construction_preflight(args.config, args.root)
    elif args.command == "construct":
        result = construct(args.config, args.authorization, args.root)
    elif args.command == "rna-preflight":
        result = rna_preflight(args.config, args.root)
    else:
        result = evaluate(args.config, args.authorization, args.root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
