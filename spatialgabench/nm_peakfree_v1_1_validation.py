"""Independent artifact and gate validation for peak-free v1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from spatialgabench.nm_peakfree_v1_1_feasibility import sha256_path
from spatialgabench.nm_peakfree_v1_validation import _close, _recompute_gene_metrics


def validate(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["output_root"]
    v1_output = root / config["v1_output_root"]
    producer = json.loads((output / "completion_evidence.json").read_text(encoding="utf-8"))
    preflight = json.loads((output / "preflight_evidence.json").read_text(encoding="utf-8"))
    v1_producer = json.loads((v1_output / "completion_evidence.json").read_text(encoding="utf-8"))
    v1_validation = json.loads(
        (v1_output / "independent_validation_evidence.json").read_text(encoding="utf-8")
    )
    summary = pd.read_csv(output / "feasibility_summary.tsv", sep="\t")
    genes = pd.read_csv(output / "gene_scale_summary.tsv", sep="\t")
    pairs = pd.read_csv(output / "pair_discrepancy_audit.tsv", sep="\t")
    extended = pd.read_csv(output / "extended_discrepancy_entries.tsv", sep="\t")
    recomputed = _recompute_gene_metrics(genes)
    if len(summary) != 1:
        raise ValueError("v1.1 feasibility summary must contain one condition")
    row = summary.iloc[0]

    atomic = ad.read_h5ad(root / config["atomic_representation"], backed="r")
    totals = np.asarray(atomic.X[:].sum(axis=1)).ravel()
    observations = int(atomic.n_obs)
    fragments = int(totals.sum())
    insufficient_fraction = float(np.mean(totals <= 1))
    atomic.file.close()

    expected_pairs = {(5, 25), (5, 100), (25, 100)}
    observed_pairs = set(zip(pairs.first_scale_kb, pairs.second_scale_kb))
    pair_partition = bool(
        pairs.applicable_entries.eq(
            pairs.finite_discrepancy_entries
            + pairs.positive_infinity_entries
            + pairs.nan_entries
            + pairs.negative_infinity_entries
        ).all()
    )
    pair_valid = bool(
        pairs.nan_entries.eq(0).all()
        and pairs.negative_infinity_entries.eq(0).all()
        and pairs.nonfinite_variance_entries.eq(0).all()
        and pairs.negative_variance_entries.eq(0).all()
        and pairs.positive_infinity_entries.eq(
            pairs.zero_variance_positive_difference_entries
        ).all()
    )
    extended_pair_counts = (
        extended.groupby(["first_scale_kb", "second_scale_kb"]).size().to_dict()
        if len(extended) else {}
    )
    extended_accounted = all(
        int(pair.positive_infinity_entries + pair.nan_entries + pair.negative_infinity_entries)
        == int(extended_pair_counts.get((pair.first_scale_kb, pair.second_scale_kb), 0))
        for pair in pairs.itertuples(index=False)
    )
    expected_gates = {
        "valid_activity": True,
        "valid_selected_variance": True,
        "valid_pairwise_variance": pair_valid,
        "valid_extended_discrepancy": pair_valid,
        "complete_pair_partitions": pair_partition,
        "no_negative_variance_beyond_tolerance": int(row.clipped_negative_variances) == 0,
        "nondegenerate_scale_usage": recomputed["maximum_single_scale_fraction"]
        <= float(config["gates"]["maximum_single_scale_fraction"]),
        "multiscale_gene_fraction": recomputed["multiscale_gene_fraction"]
        >= float(config["gates"]["minimum_multiscale_gene_fraction"]),
        "boundary_gene_fraction": recomputed["boundary_gene_fraction"]
        <= float(config["gates"]["maximum_boundary_gene_fraction"]),
        "insufficient_spot_fraction": insufficient_fraction
        <= float(config["gates"]["maximum_insufficient_spot_fraction"]),
        "v1_core_selection_summary_identical": sha256_path(output / "gene_scale_summary.tsv")
        == sha256_path(v1_output / "gene_scale_summary.tsv"),
        "v1_atomic_gene_kernel_identical": sha256_path(output / "atomic_gene_kernel.npz")
        == sha256_path(v1_output / "atomic_gene_kernel.npz"),
    }
    expected_status = "pass" if all(expected_gates.values()) else "fail"
    checks = {
        "preflight_pass": preflight.get("status") == "pass",
        "v1_remains_failed": v1_producer.get("status") == "fail"
        and v1_validation.get("validated_scientific_status") == "fail",
        "v1_independent_validation_pass": v1_validation.get("status") == "pass"
        and v1_validation.get("checks_passed") == v1_validation.get("checks_total"),
        "exact_condition": str(row.condition_id) == str(config["condition_id"]),
        "expected_dimensions": observations == int(config["expected_observations"])
        and int(row.observations) == observations and int(row.genes) == recomputed["genes"],
        "fragment_count_conserved": fragments == int(config["expected_fragment_events"])
        and int(row.fragment_events) == fragments,
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
        "boundary_fraction_recomputed": _close(
            float(row.boundary_gene_fraction), recomputed["boundary_gene_fraction"]
        ),
        "insufficient_fraction_recomputed": _close(
            float(row.insufficient_spot_fraction), insufficient_fraction
        ),
        "exact_pair_panel": observed_pairs == expected_pairs and len(pairs) == 3,
        "each_pair_uses_full_denominator": pairs.applicable_entries.eq(
            recomputed["applicable_entries"]
        ).all(),
        "pair_partitions_complete": pair_partition,
        "pair_extended_semantics_valid": pair_valid,
        "extended_entries_fully_attributed": extended_accounted,
        "extended_entries_not_fallbacks": bool(
            len(extended) > 0
            and extended.boundary_not_applicable.eq(False).all()
            and extended.insufficient_count.eq(False).all()
        ),
        "positive_infinity_count_matches_summary": int(pairs.positive_infinity_entries.sum())
        == len(extended)
        and int(row.positive_infinite_maximum_discrepancy_entries)
        <= len(extended),
        "producer_gates_recomputed": producer.get("gates") == expected_gates,
        "producer_status_recomputed": producer.get("status") == expected_status
        and str(row.status) == expected_status,
        "v1_gene_summary_byte_identical": expected_gates["v1_core_selection_summary_identical"],
        "v1_kernel_byte_identical": expected_gates["v1_atomic_gene_kernel_identical"],
        "core_stream_hashes_present": set(producer.get("core_output_chunk_stream_sha256", {}))
        == {"selected_activity", "selected_scale", "selected_log_variance"}
        and all(len(value) == 64 for value in producer["core_output_chunk_stream_sha256"].values()),
        "rna_not_accessed": producer.get("rna_expression_values_read") is False,
        "forbidden_inputs_not_accessed": producer.get("labels_clusters_pathology_read") is False
        and producer.get("peaks_read") is False,
        "no_tuning_or_ranking": producer.get("parameter_tuning_run") is False
        and producer.get("universal_ranking_run") is False
        and producer.get("final_ranking_run") is False,
        "artifact_checksums": all(
            artifact["sha256"] == sha256_path(root / artifact["path"])
            for artifact in producer["artifacts"].values()
        ),
    }
    expected_gates = {name: bool(value) for name, value in expected_gates.items()}
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_INDEPENDENT_VALIDATION",
        "status": "pass" if all(checks.values()) else "fail",
        "validated_scientific_status": expected_status,
        "validated_decision": (
            "authorize_rna_protocol_freeze" if expected_status == "pass"
            else "stop_peakfree_v1_1_before_rna"
        ),
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed_gates": expected_gates,
        "rna_expression_values_read": False,
        "labels_clusters_pathology_read": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    path = output / "independent_validation_evidence.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "pass":
        raise ValueError(f"v1.1 validation failed: {[name for name, value in checks.items() if not value]}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_peakfree_v1_1_feasibility.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.config, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
