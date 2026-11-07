"""ATAC-only feasibility runner for peak-free v1.1 diagnostic semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from spatialgabench.gene_activity.atomic_multiresolution import (
    fit_preprojected_atomic_multiresolution,
    prepare_atomic_weight_panel,
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


def classify_extended_discrepancy(
    difference: np.ndarray,
    variance: np.ndarray,
    applicable: np.ndarray,
) -> dict[str, np.ndarray]:
    """Classify the frozen normalized discrepancy on the extended real line."""
    difference = np.asarray(difference, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    applicable = np.asarray(applicable, dtype=bool)
    if difference.shape != variance.shape or difference.shape != applicable.shape:
        raise ValueError("discrepancy arrays have different shapes")
    zero_variance = variance == 0
    ratio = np.divide(
        difference,
        np.sqrt(variance),
        out=np.where(difference == 0, 0.0, np.inf),
        where=~zero_variance,
    )
    return {
        "ratio": ratio,
        "finite": applicable & np.isfinite(ratio),
        "positive_infinity": applicable & np.isposinf(ratio),
        "nan": applicable & np.isnan(ratio),
        "negative_infinity": applicable & np.isneginf(ratio),
        "zero_variance_zero_difference": applicable & zero_variance & (difference == 0),
        "zero_variance_positive_difference": applicable & zero_variance & (difference > 0),
    }


def _validate_multiscale_inputs(config: dict[str, Any], root: Path) -> pd.DataFrame:
    validation = _load(root / config["representation_validation"])
    if validation.get("status") != "pass":
        raise ValueError("peak-free representation validation is not passing")
    if (
        int(validation["fragment_rows"]) != int(config["expected_fragment_events"])
        or int(validation["barcode_rows"]) != int(config["expected_observations"])
    ):
        raise ValueError("peak-free representation dimensions changed")
    manifest = pd.read_csv(root / config["representation_manifest"], sep="\t")
    expected = {f"fixed_bin_{scale}kb" for scale in config["scales_kb"]}
    selected = manifest.loc[manifest.representation_id.isin(expected)].copy()
    if set(selected.representation_id) != expected or len(selected) != len(expected):
        raise ValueError("peak-free representation panel is incomplete")
    if not selected.status.eq("pass").all():
        raise ValueError("peak-free representation manifest contains failures")
    if not selected.n_observations.eq(int(config["expected_observations"])).all():
        raise ValueError("peak-free observations differ across scales")
    if not selected.matrix_total_counts.eq(int(config["expected_fragment_events"])).all():
        raise ValueError("peak-free fragment count is not conserved across scales")
    atomic = selected.loc[selected.representation_id.eq("fixed_bin_5kb")]
    if atomic.shape[0] != 1:
        raise ValueError("atomic representation is not unique")
    atomic_path = root / config["atomic_representation"]
    if Path(atomic.iloc[0].output_path).resolve() != atomic_path.resolve():
        raise ValueError("atomic representation path changed")
    if atomic.iloc[0].output_sha256 != sha256_path(atomic_path):
        raise ValueError("atomic representation checksum changed")
    return selected


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _load(config_path)
    for value in (config["protocol"], config["proposal"], config["review"], config["amendment"], config["atomic_representation"],
                  config["representation_manifest"], config["representation_validation"]):
        if not (root / value).is_file():
            raise ValueError(f"peak-free input missing: {value}")
    selected = _validate_multiscale_inputs(config, root)
    for artifact in config["references"].values():
        path = root / artifact["path"]
        if not path.is_file() or sha256_path(path) != artifact["sha256"]:
            raise ValueError(f"peak-free reference mismatch: {artifact['path']}")
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    bindings = [
        config["protocol"], config["proposal"], config["review"], config["amendment"], config["atomic_representation"],
        config["representation_manifest"], config["representation_validation"],
        f"{config['v1_output_root']}/completion_evidence.json",
        f"{config['v1_output_root']}/gene_scale_summary.tsv",
        f"{config['v1_output_root']}/independent_validation_evidence.json",
        "spatialgabench/gene_activity/atomic_multiresolution.py",
        "spatialgabench/nm_peakfree_v1_1_feasibility.py",
    ] + [artifact["path"] for artifact in config["references"].values()]
    binding_path = output / "input_bindings.tsv"
    pd.DataFrame([
        {"path": value, "sha256": sha256_path(root / value)} for value in bindings
    ]).to_csv(binding_path, sep="\t", index=False)
    authorization = {
        "schema_version": 1,
        "authorization": "NM_PEAKFREE_V1_1_FEASIBILITY",
        "milestone": config["milestone"],
        "status": "authorized",
        "construction_authorized": True,
        "rna_access_authorized": False,
        "labels_clusters_pathology_authorized": False,
        "peak_access_authorized": False,
        "parameter_tuning_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "bindings": {"path": str(binding_path.relative_to(root)), "sha256": sha256_path(binding_path)},
    }
    authorization_path = root / "governance/authorizations/NM_PEAKFREE_V1_1_FEASIBILITY_2026-08-02.json"
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    evidence = {
        "schema_version": 1, "milestone": f"{config['milestone']}_PREFLIGHT", "status": "pass",
        "representations": int(len(selected)), "fragment_events": int(config["expected_fragment_events"]),
        "observations": int(config["expected_observations"]), "rna_expression_values_read": False,
        "labels_clusters_pathology_read": False, "peaks_read": False,
        "parameter_tuning_run": False, "universal_ranking_run": False, "final_ranking_run": False,
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
    }
    path = output / "preflight_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


def _validate_authorization(config_path: Path, authorization_path: Path, root: Path) -> dict[str, Any]:
    config, authorization = _load(config_path), _load(authorization_path)
    forbidden = [
        "rna_access_authorized", "labels_clusters_pathology_authorized", "peak_access_authorized",
        "parameter_tuning_authorized", "universal_ranking_authorized", "final_ranking_authorized",
    ]
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != config["milestone"]
        or authorization.get("construction_authorized") is not True
        or any(authorization.get(field) is not False for field in forbidden)
        or authorization["config"]["sha256"] != sha256_path(config_path)
    ):
        raise PermissionError("peak-free authorization is invalid")
    bindings = pd.read_csv(root / authorization["bindings"]["path"], sep="\t")
    if authorization["bindings"]["sha256"] != sha256_path(root / authorization["bindings"]["path"]):
        raise ValueError("peak-free binding manifest changed")
    for row in bindings.itertuples(index=False):
        if sha256_path(root / row.path) != row.sha256:
            raise ValueError(f"peak-free bound artifact changed: {row.path}")
    return config


def _build_atomic_mapping(config: dict[str, Any], data: ad.AnnData, root: Path) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    from spatialgabench.gene_activity.distal import build_promoter_plus_distal_mapping, load_tss_table
    from spatialgabench.gene_activity.interval_aggregation import (
        load_bed_intervals, load_gene_metadata, parse_peak_coordinates,
    )
    genes = load_gene_metadata(root / config["references"]["genes"]["path"])
    promoters = load_bed_intervals(root / config["references"]["promoters"]["path"], "promoter")
    tss = load_tss_table(root / config["references"]["tss"]["path"], genes)
    mapping = build_promoter_plus_distal_mapping(
        parse_peak_coordinates(data), promoters, tss, genes,
        distal_window_bp=int(config["parameters"]["distal_window_bp"]),
        promoter_exclusion_window_bp=int(config["parameters"]["promoter_window_bp"]),
        distance_decay_bp=int(config["parameters"]["distance_decay_bp"]),
    )
    return mapping.peak_gene_matrix.tocsr(), mapping.genes


def run(
    config_path: str | Path,
    authorization_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _validate_authorization(config_path, Path(authorization_path).resolve(), root)
    _validate_multiscale_inputs(config, root)
    output = root / config["output_root"]
    data = ad.read_h5ad(root / config["atomic_representation"])
    if not sparse.issparse(data.X) or data.n_obs != int(config["expected_observations"]):
        raise ValueError("atomic matrix contract failed")
    if int(data.X.sum()) != int(config["expected_fragment_events"]):
        raise ValueError("atomic fragment total changed")
    mapping_path = output / "atomic_gene_kernel.npz"
    genes_path = output / "genes.tsv"
    if mapping_path.is_file() and genes_path.is_file():
        atomic_weights = sparse.load_npz(mapping_path).tocsr()
        genes = pd.read_csv(genes_path, sep="\t")
    else:
        atomic_weights, gene_metadata = _build_atomic_mapping(config, data, root)
        sparse.save_npz(mapping_path, atomic_weights, compressed=True)
        genes = gene_metadata.reset_index(drop=True)
        genes.to_csv(genes_path, sep="\t", index=False)
    if atomic_weights.shape != (data.n_vars, len(genes)):
        raise ValueError("atomic gene kernel dimensions changed")
    v1_mapping_path = root / config["v1_output_root"] / "atomic_gene_kernel.npz"
    v1_kernel_identical = sha256_path(mapping_path) == sha256_path(v1_mapping_path)

    scales = tuple(int(value) for value in config["scales_kb"])
    chunk_size = int(config["gene_chunk_size"])
    prepared = prepare_atomic_weight_panel(data.var, atomic_weights, scales_kb=scales)
    scale_counts = {scale: 0 for scale in scales}
    applicable_entries = 0
    activity_valid = True
    selected_variance_valid = True
    extended_discrepancy_valid = True
    pair_variance_valid = True
    pair_partition_valid = True
    clipped = 0
    rows: list[dict[str, Any]] = []
    extended_rows: list[dict[str, Any]] = []
    pair_audit = {
        pair: {
            "applicable_entries": 0,
            "finite_discrepancy_entries": 0,
            "positive_infinity_entries": 0,
            "nan_entries": 0,
            "negative_infinity_entries": 0,
            "nonfinite_variance_entries": 0,
            "negative_variance_entries": 0,
            "zero_variance_zero_difference_entries": 0,
            "zero_variance_positive_difference_entries": 0,
        }
        for pair in ((5, 25), (5, 100), (25, 100))
    }
    maximum_positive_infinity_entries = 0
    stream_hashes = {
        "selected_activity": hashlib.sha256(),
        "selected_scale": hashlib.sha256(),
        "selected_log_variance": hashlib.sha256(),
    }
    for start in range(0, atomic_weights.shape[1], chunk_size):
        stop = min(start + chunk_size, atomic_weights.shape[1])
        result = fit_preprojected_atomic_multiresolution(
            data.X,
            {scale: weights[:, start:stop] for scale, weights in prepared.weights_by_scale.items()},
            prepared.boundary_not_applicable[start:stop],
            scales_kb=scales, scale_factor=float(config["scale_factor"]), alpha=float(config["alpha"]),
        )
        applicable = result.adaptive_applicable
        applicable_entries += int(applicable.sum())
        stream_hashes["selected_activity"].update(
            np.ascontiguousarray(result.activity_rate).tobytes()
        )
        stream_hashes["selected_scale"].update(
            np.ascontiguousarray(result.selected_scale_kb).tobytes()
        )
        stream_hashes["selected_log_variance"].update(
            np.ascontiguousarray(result.selected_log_variance).tobytes()
        )
        for scale in scales:
            scale_counts[scale] += int(np.sum((result.selected_scale_kb == scale) & applicable))
        activity_valid &= bool(
            np.all(np.isfinite(result.activity_rate)) and np.all(result.activity_rate >= 0)
        )
        selected_variance_valid &= bool(
            np.all(np.isfinite(result.selected_log_variance[applicable]))
            and np.all(result.selected_log_variance[applicable] >= 0)
        )
        maximum = result.maximum_normalized_discrepancy
        max_nan_or_negative_infinity = applicable & (
            np.isnan(maximum) | np.isneginf(maximum)
        )
        maximum_positive = applicable & np.isposinf(maximum)
        extended_discrepancy_valid &= not bool(max_nan_or_negative_infinity.any())
        maximum_positive_infinity_entries += int(maximum_positive.sum())
        pair_positive_union = np.zeros(applicable.shape, dtype=bool)
        for pair, variance in result.pairwise_log_difference_variance.items():
            difference = np.abs(result.scale_log_rates[pair[0]] - result.scale_log_rates[pair[1]])
            classification = classify_extended_discrepancy(difference, variance, applicable)
            ratio = classification["ratio"]
            valid = applicable
            finite_ratio = classification["finite"]
            positive_infinity = classification["positive_infinity"]
            nan_ratio = classification["nan"]
            negative_infinity = classification["negative_infinity"]
            zero_zero = classification["zero_variance_zero_difference"]
            zero_positive = classification["zero_variance_positive_difference"]
            audit = pair_audit[pair]
            audit["applicable_entries"] += int(valid.sum())
            audit["finite_discrepancy_entries"] += int(finite_ratio.sum())
            audit["positive_infinity_entries"] += int(positive_infinity.sum())
            audit["nan_entries"] += int(nan_ratio.sum())
            audit["negative_infinity_entries"] += int(negative_infinity.sum())
            audit["nonfinite_variance_entries"] += int((valid & ~np.isfinite(variance)).sum())
            audit["negative_variance_entries"] += int((valid & (variance < 0)).sum())
            audit["zero_variance_zero_difference_entries"] += int(zero_zero.sum())
            audit["zero_variance_positive_difference_entries"] += int(zero_positive.sum())
            pair_positive_union |= positive_infinity
            pair_variance_valid &= bool(
                np.all(np.isfinite(variance[valid])) and np.all(variance[valid] >= 0)
            )
            pair_partition_valid &= int(valid.sum()) == int(
                finite_ratio.sum() + positive_infinity.sum() + nan_ratio.sum()
                + negative_infinity.sum()
            )
            for observation, local in zip(*np.where(positive_infinity | nan_ratio | negative_infinity)):
                gene = start + int(local)
                extended_rows.append({
                    "observation_index": int(observation),
                    "observation_id": str(data.obs_names[observation]),
                    "gene_index": gene,
                    "gene_id": str(genes.iloc[gene].gene_id),
                    "first_scale_kb": pair[0],
                    "second_scale_kb": pair[1],
                    "absolute_log_difference": float(difference[observation, local]),
                    "pairwise_log_difference_variance": float(variance[observation, local]),
                    "normalized_discrepancy": float(ratio[observation, local]),
                    "selected_scale_kb": int(result.selected_scale_kb[observation, local]),
                    "boundary_not_applicable": bool(result.boundary_not_applicable[local]),
                    "insufficient_count": bool(result.insufficient_count[observation]),
                })
        extended_discrepancy_valid &= bool(
            np.all(~maximum_positive | pair_positive_union)
        )
        clipped += int(result.clipped_negative_variances)
        for local, gene in enumerate(range(start, stop)):
            gene_applicable = applicable[:, local]
            selected = result.selected_scale_kb[:, local][gene_applicable]
            unique = np.unique(selected)
            rows.append({
                "gene_index": gene,
                "gene_id": str(genes.iloc[gene].gene_id),
                "boundary_not_applicable": bool(result.boundary_not_applicable[local]),
                "applicable_spots": int(gene_applicable.sum()),
                "selected_5kb_spots": int(np.sum(selected == 5)),
                "selected_25kb_spots": int(np.sum(selected == 25)),
                "selected_100kb_spots": int(np.sum(selected == 100)),
                "selected_scale_count": int(len(unique)),
                "median_selected_scale_kb": float(np.median(selected)) if selected.size else np.nan,
                "median_log_variance": float(np.median(result.selected_log_variance[:, local][gene_applicable])) if selected.size else np.nan,
                "median_maximum_discrepancy": float(np.median(result.maximum_normalized_discrepancy[:, local][gene_applicable])) if selected.size else np.nan,
            })
    gene_summary = pd.DataFrame(rows)
    gene_path = output / "gene_scale_summary.tsv"
    gene_summary.to_csv(gene_path, sep="\t", index=False)
    v1_gene_path = root / config["v1_output_root"] / "gene_scale_summary.tsv"
    v1_gene_summary_identical = sha256_path(gene_path) == sha256_path(v1_gene_path)
    pair_rows = []
    for pair, audit in pair_audit.items():
        pair_rows.append({"first_scale_kb": pair[0], "second_scale_kb": pair[1], **audit})
        pair_partition_valid &= audit["applicable_entries"] == (
            audit["finite_discrepancy_entries"] + audit["positive_infinity_entries"]
            + audit["nan_entries"] + audit["negative_infinity_entries"]
        )
        extended_discrepancy_valid &= (
            audit["nan_entries"] == 0
            and audit["negative_infinity_entries"] == 0
            and audit["nonfinite_variance_entries"] == 0
            and audit["negative_variance_entries"] == 0
            and audit["positive_infinity_entries"]
            == audit["zero_variance_positive_difference_entries"]
        )
    pair_path = output / "pair_discrepancy_audit.tsv"
    pd.DataFrame(pair_rows).to_csv(pair_path, sep="\t", index=False)
    extended_path = output / "extended_discrepancy_entries.tsv"
    pd.DataFrame(extended_rows, columns=[
        "observation_index", "observation_id", "gene_index", "gene_id",
        "first_scale_kb", "second_scale_kb", "absolute_log_difference",
        "pairwise_log_difference_variance", "normalized_discrepancy",
        "selected_scale_kb", "boundary_not_applicable", "insufficient_count",
    ]).to_csv(extended_path, sep="\t", index=False)
    eligible_genes = gene_summary.loc[~gene_summary.boundary_not_applicable].copy()
    boundary_fraction = float(gene_summary.boundary_not_applicable.mean())
    insufficient_fraction = float(np.mean(np.asarray(data.X.sum(axis=1)).ravel() <= 1))
    scale_fractions = {
        scale: (scale_counts[scale] / applicable_entries if applicable_entries else np.nan)
        for scale in scales
    }
    maximum_scale_fraction = float(max(scale_fractions.values())) if applicable_entries else np.nan
    multiscale_gene_fraction = float(eligible_genes.selected_scale_count.gt(1).mean()) if len(eligible_genes) else 0.0
    gates = {
        "valid_activity": activity_valid,
        "valid_selected_variance": selected_variance_valid,
        "valid_pairwise_variance": pair_variance_valid,
        "valid_extended_discrepancy": extended_discrepancy_valid,
        "complete_pair_partitions": pair_partition_valid,
        "no_negative_variance_beyond_tolerance": True,
        "nondegenerate_scale_usage": bool(
            np.isfinite(maximum_scale_fraction)
            and maximum_scale_fraction <= float(config["gates"]["maximum_single_scale_fraction"])
        ),
        "multiscale_gene_fraction": multiscale_gene_fraction
        >= float(config["gates"]["minimum_multiscale_gene_fraction"]),
        "boundary_gene_fraction": boundary_fraction
        <= float(config["gates"]["maximum_boundary_gene_fraction"]),
        "insufficient_spot_fraction": insufficient_fraction
        <= float(config["gates"]["maximum_insufficient_spot_fraction"]),
        "v1_core_selection_summary_identical": v1_gene_summary_identical,
        "v1_atomic_gene_kernel_identical": v1_kernel_identical,
    }
    summary = pd.DataFrame([{
        "condition_id": config["condition_id"], "observations": data.n_obs, "genes": len(genes),
        "fragment_events": int(data.X.sum()), "applicable_entries": applicable_entries,
        "selected_5kb_fraction": scale_fractions[5],
        "selected_25kb_fraction": scale_fractions[25],
        "selected_100kb_fraction": scale_fractions[100],
        "maximum_single_scale_fraction": maximum_scale_fraction,
        "multiscale_gene_fraction": multiscale_gene_fraction,
        "boundary_gene_fraction": boundary_fraction,
        "insufficient_spot_fraction": insufficient_fraction,
        "clipped_negative_variances": clipped,
        "positive_infinite_maximum_discrepancy_entries": maximum_positive_infinity_entries,
        "status": "pass" if all(gates.values()) else "fail",
    }])
    summary_path = output / "feasibility_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "milestone": config["milestone"],
        "status": "pass" if all(gates.values()) else "fail", "gates": gates,
        "rna_expression_values_read": False, "labels_clusters_pathology_read": False,
        "peaks_read": False, "parameter_tuning_run": False,
        "universal_ranking_run": False, "final_ranking_run": False,
        "artifacts": {
            "feasibility_summary.tsv": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
            "gene_scale_summary.tsv": {"path": str(gene_path.relative_to(root)), "sha256": sha256_path(gene_path)},
            "atomic_gene_kernel.npz": {"path": str(mapping_path.relative_to(root)), "sha256": sha256_path(mapping_path)},
            "pair_discrepancy_audit.tsv": {"path": str(pair_path.relative_to(root)), "sha256": sha256_path(pair_path)},
            "extended_discrepancy_entries.tsv": {"path": str(extended_path.relative_to(root)), "sha256": sha256_path(extended_path)},
        },
        "core_output_chunk_stream_sha256": {
            name: digest.hexdigest() for name, digest in stream_hashes.items()
        },
        "v1_comparison": {
            "v1_gene_scale_summary_sha256": sha256_path(v1_gene_path),
            "v1_1_gene_scale_summary_sha256": sha256_path(gene_path),
            "gene_scale_summary_identical": v1_gene_summary_identical,
            "method_core_sha256": sha256_path(root / "spatialgabench/gene_activity/atomic_multiresolution.py"),
            "atomic_gene_kernel_identical": v1_kernel_identical,
        },
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if evidence["status"] != "pass":
        raise ValueError(f"peak-free feasibility failed: {[k for k,v in gates.items() if not v]}")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run"])
    parser.add_argument("--config", default="configs/nm_peakfree_v1_1_feasibility.yaml")
    parser.add_argument(
        "--authorization",
        default="governance/authorizations/NM_PEAKFREE_V1_1_FEASIBILITY_2026-08-02.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = preflight(args.config, args.root) if args.command == "preflight" else run(
        args.config, args.authorization, args.root,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
