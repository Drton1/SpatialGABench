"""Audited toy depth ladder for reliability-weighted multiscale activity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spatialgabench.gene_activity.reliability_multiscale import (
    fit_reliability_multiscale,
    log_rate,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("toy config must contain a mapping")
    return value


def toy_inputs(depth: float) -> tuple[Any, ...]:
    peak = depth * np.array(
        [
            [80, 2, 0], [65, 4, 1], [50, 8, 2], [35, 14, 4],
            [20, 22, 8], [10, 35, 15], [4, 50, 25], [1, 70, 40],
        ],
        dtype=np.float64,
    )
    bins = depth * np.array(
        [
            [70, 10, 4], [60, 12, 5], [50, 15, 7], [40, 20, 9],
            [30, 27, 12], [22, 38, 18], [17, 50, 27], [14, 65, 38],
        ],
        dtype=np.float64,
    )
    exposure = np.full(peak.shape[0], 1000.0)
    factors = [0.92, 0.96, 1.02, 1.08]
    peak_perturbations = [0.75 * factor * peak for factor in factors]
    bin_perturbations = [0.75 * (1.0 + (factor - 1.0) / 4.0) * bins for factor in factors]
    valid = np.ones((4, peak.shape[1]), dtype=bool)
    full_valid = np.ones(peak.shape[1], dtype=bool)
    return (
        peak, bins, peak + 0.5, bins + 0.5, exposure,
        peak_perturbations, bin_perturbations, valid, full_valid,
    )


def build_toy_ladder(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    genes: list[dict[str, Any]] = []
    for depth in config["depth_fractions"]:
        values = toy_inputs(float(depth))
        result = fit_reliability_multiscale(
            *values,
            scale=float(config["scale"]),
            minimum_spots=int(config["minimum_spots"]),
        )
        peak_log = log_rate(values[0], values[4], float(config["scale"]))
        bin_log = log_rate(values[1], values[4], float(config["scale"]))
        output_log = np.log1p(float(config["scale"]) * result.activity_rate)
        lower = np.minimum(peak_log, bin_log)
        upper = np.maximum(peak_log, bin_log)
        rows.append({
            "depth_fraction": float(depth),
            "observations": result.activity_rate.shape[0],
            "genes": result.activity_rate.shape[1],
            "median_peak_weight": float(np.median(result.peak_weight)),
            "mean_peak_weight": float(np.mean(result.peak_weight)),
            "median_continuous_risk": float(np.median(result.continuous_risk)),
            "max_continuous_risk": float(np.max(result.continuous_risk)),
            "nonzero_signal_variance_genes": int(np.sum(result.peak_signal_variance > 0)),
            "minimum_output": float(result.activity_rate.min()),
            "finite_nonnegative": bool(
                np.all(np.isfinite(result.activity_rate)) and np.all(result.activity_rate >= 0)
            ),
            "within_branch_log_bounds": bool(
                np.all(output_log >= lower - 1e-12) and np.all(output_log <= upper + 1e-12)
            ),
        })
        for gene in range(result.activity_rate.shape[1]):
            genes.append({
                "depth_fraction": float(depth),
                "gene_index": gene,
                "signal_variance": float(result.peak_signal_variance[gene]),
                "structural_risk": float(result.peak_structural_risk[gene]),
                "median_peak_weight": float(np.median(result.peak_weight[:, gene])),
                "median_continuous_risk": float(np.median(result.continuous_risk[:, gene])),
            })
    return pd.DataFrame(rows), pd.DataFrame(genes)


def validate_ladder(summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, bool]:
    ordered = summary.sort_values("depth_fraction", ascending=False, kind="mergesort")
    weights = ordered.median_peak_weight.to_numpy(float)
    risks = ordered.median_continuous_risk.to_numpy(float)
    gates = {
        "expected_depths": set(summary.depth_fraction.astype(float))
        == set(float(value) for value in config["depth_fractions"]),
        "expected_dimensions": bool(
            summary.observations.eq(int(config["expected_observations"])).all()
            and summary.genes.eq(int(config["expected_genes"])).all()
        ),
        "peak_weight_nonincreasing_with_depth_loss": bool(np.all(np.diff(weights) <= 1e-12)),
        "continuous_risk_nondecreasing_with_depth_loss": bool(np.all(np.diff(risks) >= -1e-12)),
        "full_depth_peak_weight": bool(
            ordered.iloc[0].median_peak_weight
            >= float(config["gates"]["full_depth_median_peak_weight_min"])
        ),
        "one_percent_peak_weight": bool(
            ordered.iloc[-1].median_peak_weight
            <= float(config["gates"]["one_percent_median_peak_weight_max"])
        ),
        "finite_nonnegative": bool(summary.finite_nonnegative.all()),
        "within_branch_log_bounds": bool(summary.within_branch_log_bounds.all()),
    }
    return gates


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    summary, genes = build_toy_ladder(config)
    gates = validate_ladder(summary, config)
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "toy_depth_ladder.tsv"
    genes_path = output / "toy_gene_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    genes.to_csv(genes_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "milestone": config["milestone"],
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "rna_used_for_construction": False,
        "rna_used_for_evaluation": False,
        "labels_clusters_pathology_used": False,
        "parameter_tuning_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "artifacts": {
            "toy_depth_ladder.tsv": {
                "path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path),
            },
            "toy_gene_summary.tsv": {
                "path": str(genes_path.relative_to(root)), "sha256": sha256_path(genes_path),
            },
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": sha256_path(Path(__file__).resolve()),
        },
        "core": {
            "path": "spatialgabench/gene_activity/reliability_multiscale.py",
            "sha256": sha256_path(root / "spatialgabench/gene_activity/reliability_multiscale.py"),
        },
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if evidence["status"] != "pass":
        raise ValueError("RWGA toy ladder failed")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_reliability_multiscale_toy.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(run(args.config, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

