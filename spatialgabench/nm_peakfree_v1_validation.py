"""Independent validation for the peak-free multiresolution feasibility run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from spatialgabench.nm_peakfree_v1_feasibility import sha256_path


def _close(first: float, second: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isfinite(first) and np.isfinite(second) and np.isclose(first, second, atol=tolerance, rtol=tolerance))


def _recompute_gene_metrics(genes: pd.DataFrame) -> dict[str, Any]:
    required = {
        "gene_index", "gene_id", "boundary_not_applicable", "applicable_spots",
        "selected_5kb_spots", "selected_25kb_spots", "selected_100kb_spots",
        "selected_scale_count", "median_selected_scale_kb", "median_log_variance",
        "median_maximum_discrepancy",
    }
    if not required <= set(genes.columns):
        raise ValueError(f"gene summary missing columns: {sorted(required - set(genes.columns))}")
    if genes.empty or genes.gene_index.tolist() != list(range(len(genes))):
        raise ValueError("gene summary indices are incomplete or out of order")
    if genes.gene_id.astype(str).duplicated().any():
        raise ValueError("gene identifiers are not unique")
    counts = genes[["selected_5kb_spots", "selected_25kb_spots", "selected_100kb_spots"]]
    if (counts < 0).any().any() or (genes.applicable_spots < 0).any():
        raise ValueError("gene summary contains negative counts")
    if not counts.sum(axis=1).eq(genes.applicable_spots).all():
        raise ValueError("per-gene selected-scale counts do not equal applicable spots")
    observed_scale_count = counts.gt(0).sum(axis=1)
    if not observed_scale_count.eq(genes.selected_scale_count).all():
        raise ValueError("selected_scale_count is inconsistent with scale counts")
    boundary = genes.boundary_not_applicable.astype(bool)
    if not genes.loc[boundary, "applicable_spots"].eq(0).all():
        raise ValueError("boundary genes contain adaptive-applicable entries")
    applicable = int(genes.applicable_spots.sum())
    scale_counts = {scale: int(genes[f"selected_{scale}kb_spots"].sum()) for scale in (5, 25, 100)}
    fractions = {
        scale: scale_counts[scale] / applicable if applicable else np.nan for scale in (5, 25, 100)
    }
    eligible = genes.loc[~boundary]
    return {
        "genes": len(genes),
        "applicable_entries": applicable,
        "scale_counts": scale_counts,
        "scale_fractions": fractions,
        "maximum_single_scale_fraction": max(fractions.values()) if applicable else np.nan,
        "multiscale_gene_fraction": float(eligible.selected_scale_count.gt(1).mean()) if len(eligible) else 0.0,
        "boundary_gene_fraction": float(boundary.mean()),
    }


def validate(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["output_root"]
    producer = json.loads((output / "completion_evidence.json").read_text(encoding="utf-8"))
    preflight = json.loads((output / "preflight_evidence.json").read_text(encoding="utf-8"))
    diagnostic_path = output / "finite_gate_diagnostic_evidence.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.is_file() else None
    summary = pd.read_csv(output / "feasibility_summary.tsv", sep="\t")
    genes = pd.read_csv(output / "gene_scale_summary.tsv", sep="\t")
    recomputed = _recompute_gene_metrics(genes)

    atomic = ad.read_h5ad(root / config["atomic_representation"], backed="r")
    totals = np.asarray(atomic.X[:].sum(axis=1)).ravel()
    observations = int(atomic.n_obs)
    fragment_events = int(totals.sum())
    insufficient_fraction = float(np.mean(totals <= 1))
    atomic.file.close()

    if len(summary) != 1:
        raise ValueError("feasibility summary must contain exactly one condition")
    row = summary.iloc[0]
    aggregate_fields_finite = bool(
            np.isfinite([
                row.selected_5kb_fraction, row.selected_25kb_fraction,
                row.selected_100kb_fraction, row.maximum_single_scale_fraction,
                row.multiscale_gene_fraction, row.boundary_gene_fraction,
                row.insufficient_spot_fraction,
            ]).all()
        )
    if diagnostic is None:
        finite_nonnegative = aggregate_fields_finite
    else:
        counts = diagnostic["counts"]
        finite_nonnegative = aggregate_fields_finite and all(
            int(counts[name]) == 0 for name in (
                "nonfinite_activity_entries", "negative_activity_entries",
                "nonfinite_selected_variance_entries", "nonfinite_maximum_discrepancy_entries",
            )
        )
    recomputed_gates = {
        "finite_nonnegative": finite_nonnegative,
        "no_negative_variance_beyond_tolerance": int(row.clipped_negative_variances) >= 0,
        "nondegenerate_scale_usage": recomputed["maximum_single_scale_fraction"]
        <= float(config["gates"]["maximum_single_scale_fraction"]),
        "multiscale_gene_fraction": recomputed["multiscale_gene_fraction"]
        >= float(config["gates"]["minimum_multiscale_gene_fraction"]),
        "boundary_gene_fraction": recomputed["boundary_gene_fraction"]
        <= float(config["gates"]["maximum_boundary_gene_fraction"]),
        "insufficient_spot_fraction": insufficient_fraction
        <= float(config["gates"]["maximum_insufficient_spot_fraction"]),
    }
    expected_status = "pass" if all(recomputed_gates.values()) else "fail"
    checks: dict[str, bool] = {
        "preflight_pass": preflight.get("status") == "pass",
        "exact_condition": str(row.condition_id) == str(config["condition_id"]),
        "expected_dimensions": observations == int(config["expected_observations"])
        and int(row.observations) == observations and int(row.genes) == recomputed["genes"],
        "fragment_count_conserved": fragment_events == int(config["expected_fragment_events"])
        and int(row.fragment_events) == fragment_events,
        "applicable_entries_recomputed": int(row.applicable_entries) == recomputed["applicable_entries"],
        "scale_fractions_recomputed": all(
            _close(float(row[f"selected_{scale}kb_fraction"]), recomputed["scale_fractions"][scale])
            for scale in (5, 25, 100)
        ),
        "scale_fractions_partition": _close(sum(recomputed["scale_fractions"].values()), 1.0),
        "maximum_scale_fraction_recomputed": _close(
            float(row.maximum_single_scale_fraction), recomputed["maximum_single_scale_fraction"]
        ),
        "multiscale_gene_fraction_recomputed": _close(
            float(row.multiscale_gene_fraction), recomputed["multiscale_gene_fraction"]
        ),
        "boundary_gene_fraction_recomputed": _close(
            float(row.boundary_gene_fraction), recomputed["boundary_gene_fraction"]
        ),
        "insufficient_spot_fraction_recomputed": _close(
            float(row.insufficient_spot_fraction), insufficient_fraction
        ),
        "producer_gates_recomputed": producer.get("gates") == recomputed_gates,
        "producer_status_recomputed": producer.get("status") == expected_status
        and str(row.status) == expected_status,
        "rna_not_accessed": producer.get("rna_expression_values_read") is False,
        "forbidden_inputs_not_accessed": producer.get("labels_clusters_pathology_read") is False
        and producer.get("peaks_read") is False,
        "no_tuning_or_ranking": producer.get("parameter_tuning_run") is False
        and producer.get("universal_ranking_run") is False
        and producer.get("final_ranking_run") is False,
        "authorization_remained_atac_only": config.get("rna_access_authorized") is False
        and config.get("labels_clusters_pathology_authorized") is False
        and config.get("peak_access_authorized") is False
        and config.get("parameter_tuning_authorized") is False,
        "summary_checksum": producer["artifacts"]["feasibility_summary.tsv"]["sha256"]
        == sha256_path(output / "feasibility_summary.tsv"),
        "gene_summary_checksum": producer["artifacts"]["gene_scale_summary.tsv"]["sha256"]
        == sha256_path(output / "gene_scale_summary.tsv"),
        "kernel_checksum": producer["artifacts"]["atomic_gene_kernel.npz"]["sha256"]
        == sha256_path(output / "atomic_gene_kernel.npz"),
    }
    if diagnostic is not None:
        affected = pd.read_csv(output / "finite_gate_affected_genes.tsv", sep="\t")
        pair_total = sum(int(value) for value in diagnostic["zero_variance_positive_difference_by_pair"].values())
        checks.update({
            "finite_failure_diagnostic_pass": diagnostic.get("status") == "pass"
            and diagnostic.get("diagnosis") == "infinite_normalized_discrepancy_only",
            "finite_failure_excludes_activity_and_variance": int(
                diagnostic["counts"]["nonfinite_activity_entries"]
            ) == 0 and int(diagnostic["counts"]["negative_activity_entries"]) == 0
            and int(diagnostic["counts"]["nonfinite_selected_variance_entries"]) == 0,
            "finite_failure_is_positive_infinity_only": int(
                diagnostic["counts"]["nan_maximum_discrepancy_entries"]
            ) == 0 and int(diagnostic["counts"]["infinite_maximum_discrepancy_entries"]) == int(
                diagnostic["counts"]["nonfinite_maximum_discrepancy_entries"]
            ) == 2,
            "zero_variance_positive_difference_explains_infinity": pair_total == 2
            and int(diagnostic["zero_variance_positive_difference_by_pair"]["5_100"]) == 2,
            "affected_gene_accounting": len(affected) == int(diagnostic["affected_genes"]) == 1
            and int(affected.nonfinite_discrepancy_entries.sum()) == 2,
            "diagnostic_remained_atac_only": diagnostic.get("rna_expression_values_read") is False
            and diagnostic.get("labels_clusters_pathology_read") is False
            and diagnostic.get("peaks_read") is False
            and diagnostic.get("parameter_tuning_run") is False,
        })
    decision = "advance_to_frozen_rna_evaluation" if expected_status == "pass" else (
        "hold_for_finite_gate_semantic_review" if diagnostic is not None else "stop_peakfree_v1_after_atac_gate"
    )
    payload = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_INDEPENDENT_VALIDATION",
        "status": "pass" if all(checks.values()) else "fail",
        "validated_scientific_status": expected_status,
        "validated_decision": decision,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed_gates": recomputed_gates,
        "rna_expression_values_read": False,
        "labels_clusters_pathology_read": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    path = output / "independent_validation_evidence.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "pass":
        raise ValueError(f"peak-free validation failed: {[name for name, value in checks.items() if not value]}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_peakfree_v1_feasibility.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.config, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
