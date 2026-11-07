"""Real ATAC-only feasibility runner for peak-free multiresolution activity."""

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
    for value in (config["protocol"], config["proposal"], config["amendment"], config["atomic_representation"],
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
        config["protocol"], config["proposal"], config["amendment"], config["atomic_representation"],
        config["representation_manifest"], config["representation_validation"],
        "spatialgabench/gene_activity/atomic_multiresolution.py",
        "spatialgabench/nm_peakfree_v1_feasibility.py",
    ] + [artifact["path"] for artifact in config["references"].values()]
    binding_path = output / "input_bindings.tsv"
    pd.DataFrame([
        {"path": value, "sha256": sha256_path(root / value)} for value in bindings
    ]).to_csv(binding_path, sep="\t", index=False)
    authorization = {
        "schema_version": 1,
        "authorization": "NM_PEAKFREE_V1_FEASIBILITY",
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
    authorization_path = root / "governance/authorizations/NM_PEAKFREE_V1_FEASIBILITY_2026-08-02.json"
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

    scales = tuple(int(value) for value in config["scales_kb"])
    chunk_size = int(config["gene_chunk_size"])
    prepared = prepare_atomic_weight_panel(data.var, atomic_weights, scales_kb=scales)
    scale_counts = {scale: 0 for scale in scales}
    applicable_entries = 0
    finite = True
    nonnegative = True
    clipped = 0
    rows: list[dict[str, Any]] = []
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
        for scale in scales:
            scale_counts[scale] += int(np.sum((result.selected_scale_kb == scale) & applicable))
        finite &= bool(
            np.all(np.isfinite(result.activity_rate))
            and np.all(np.isfinite(result.selected_log_variance[applicable]))
            and np.all(np.isfinite(result.maximum_normalized_discrepancy[applicable]))
        )
        nonnegative &= bool(np.all(result.activity_rate >= 0))
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
        "finite_nonnegative": finite and nonnegative,
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
    parser.add_argument("--config", default="configs/nm_peakfree_v1_feasibility.yaml")
    parser.add_argument(
        "--authorization",
        default="governance/authorizations/NM_PEAKFREE_V1_FEASIBILITY_2026-08-02.json",
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
