"""Read-only attribution of the peak-free v1 finite-gate failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spatialgabench.gene_activity.atomic_multiresolution import (
    fit_preprojected_atomic_multiresolution,
    prepare_atomic_weight_panel,
)
from spatialgabench.nm_peakfree_v1_feasibility import _load, _validate_authorization


def diagnose(
    config_path: str | Path,
    authorization_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _validate_authorization(config_path, Path(authorization_path).resolve(), root)
    output = root / config["output_root"]
    producer = _load(output / "completion_evidence.json")
    if producer.get("status") != "fail" or {
        name for name, value in producer.get("gates", {}).items() if not value
    } != {"finite_nonnegative"}:
        raise ValueError("diagnostic is authorized only for the observed finite-only failure")
    data = ad.read_h5ad(root / config["atomic_representation"])
    atomic_weights = sparse.load_npz(output / "atomic_gene_kernel.npz").tocsr()
    genes = pd.read_csv(output / "genes.tsv", sep="\t")
    scales = tuple(int(value) for value in config["scales_kb"])
    prepared = prepare_atomic_weight_panel(data.var, atomic_weights, scales_kb=scales)

    totals = {
        "applicable_entries": 0,
        "nonfinite_activity_entries": 0,
        "negative_activity_entries": 0,
        "nonfinite_selected_variance_entries": 0,
        "nonfinite_maximum_discrepancy_entries": 0,
        "nan_maximum_discrepancy_entries": 0,
        "infinite_maximum_discrepancy_entries": 0,
    }
    pair_counts = {f"{first}_{second}": 0 for first, second in ((5, 25), (5, 100), (25, 100))}
    affected_rows: list[dict[str, object]] = []
    chunk_size = int(config["gene_chunk_size"])
    for start in range(0, atomic_weights.shape[1], chunk_size):
        stop = min(start + chunk_size, atomic_weights.shape[1])
        result = fit_preprojected_atomic_multiresolution(
            data.X,
            {scale: weights[:, start:stop] for scale, weights in prepared.weights_by_scale.items()},
            prepared.boundary_not_applicable[start:stop],
            scales_kb=scales,
            scale_factor=float(config["scale_factor"]),
            alpha=float(config["alpha"]),
        )
        applicable = result.adaptive_applicable
        totals["applicable_entries"] += int(applicable.sum())
        totals["nonfinite_activity_entries"] += int((~np.isfinite(result.activity_rate)).sum())
        totals["negative_activity_entries"] += int((result.activity_rate < 0).sum())
        selected_nonfinite = applicable & ~np.isfinite(result.selected_log_variance)
        discrepancy_nonfinite = applicable & ~np.isfinite(result.maximum_normalized_discrepancy)
        totals["nonfinite_selected_variance_entries"] += int(selected_nonfinite.sum())
        totals["nonfinite_maximum_discrepancy_entries"] += int(discrepancy_nonfinite.sum())
        totals["nan_maximum_discrepancy_entries"] += int(
            (applicable & np.isnan(result.maximum_normalized_discrepancy)).sum()
        )
        totals["infinite_maximum_discrepancy_entries"] += int(
            (applicable & np.isinf(result.maximum_normalized_discrepancy)).sum()
        )
        pair_local: dict[str, np.ndarray] = {}
        for pair, variance in result.pairwise_log_difference_variance.items():
            difference = np.abs(result.scale_log_rates[pair[0]] - result.scale_log_rates[pair[1]])
            mask = applicable & (variance == 0) & (difference > 0)
            key = f"{pair[0]}_{pair[1]}"
            pair_counts[key] += int(mask.sum())
            pair_local[key] = mask.sum(axis=0)
        affected_by_gene = discrepancy_nonfinite.sum(axis=0)
        for local in np.flatnonzero(affected_by_gene):
            gene = start + int(local)
            affected_rows.append({
                "gene_index": gene,
                "gene_id": str(genes.iloc[gene].gene_id),
                "nonfinite_discrepancy_entries": int(affected_by_gene[local]),
                "zero_variance_positive_difference_5_25": int(pair_local["5_25"][local]),
                "zero_variance_positive_difference_5_100": int(pair_local["5_100"][local]),
                "zero_variance_positive_difference_25_100": int(pair_local["25_100"][local]),
            })

    affected = pd.DataFrame(affected_rows)
    affected_path = output / "finite_gate_affected_genes.tsv"
    affected.to_csv(affected_path, sep="\t", index=False)
    explained = (
        totals["nonfinite_activity_entries"] == 0
        and totals["negative_activity_entries"] == 0
        and totals["nonfinite_selected_variance_entries"] == 0
        and totals["nan_maximum_discrepancy_entries"] == 0
        and totals["infinite_maximum_discrepancy_entries"]
        == totals["nonfinite_maximum_discrepancy_entries"]
        and sum(pair_counts.values()) >= totals["infinite_maximum_discrepancy_entries"]
    )
    payload = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_FINITE_GATE_DIAGNOSTIC",
        "status": "pass" if explained else "fail",
        "diagnosis": "infinite_normalized_discrepancy_only" if explained else "unresolved",
        "counts": totals,
        "zero_variance_positive_difference_by_pair": pair_counts,
        "affected_genes": int(len(affected)),
        "rna_expression_values_read": False,
        "labels_clusters_pathology_read": False,
        "peaks_read": False,
        "parameter_tuning_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    path = output / "finite_gate_diagnostic_evidence.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not explained:
        raise ValueError("finite-gate failure was not fully attributed")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_peakfree_v1_feasibility.yaml")
    parser.add_argument(
        "--authorization",
        default="governance/authorizations/NM_PEAKFREE_V1_FEASIBILITY_2026-08-02.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(diagnose(args.config, args.authorization, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
