"""Independent validation of the RWGA low-depth feasibility decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from spatialgabench.nm_rwga_feasibility import sha256_path


def validate(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["output_root"]
    producer = json.loads((output / "completion_evidence.json").read_text())
    summary = pd.read_csv(output / "feasibility_summary.tsv", sep="\t")
    genes = pd.read_csv(output / "gene_feasibility_summary.tsv", sep="\t")
    preflight = json.loads((output / "preflight_evidence.json").read_text())
    checks: dict[str, bool] = {
        "preflight_pass": preflight.get("status") == "pass",
        "producer_scientific_gate_fail": producer.get("status") == "fail",
        "exact_condition": len(summary) == 1 and summary.iloc[0].condition_id == config["condition_id"],
        "expected_dimensions": len(genes) > 0
        and int(summary.iloc[0].observations) == int(config["expected_observations"])
        and int(summary.iloc[0].genes) == len(genes),
        "all_four_folds_present": all(
            (output / "folds" / f"delete_group_{fold}" / "construction_evidence.json").is_file()
            for fold in range(int(config["delete_groups"]))
        ),
        "all_four_folds_pass": all(
            json.loads(
                (output / "folds" / f"delete_group_{fold}" / "construction_evidence.json").read_text()
            ).get("status") == "pass"
            for fold in range(int(config["delete_groups"]))
        ),
        "finite_bounded_outputs": bool(summary.iloc[0].finite_bounded_outputs),
        "all_q2_zero": float(summary.iloc[0].nonzero_q2_gene_fraction) == 0.0
        and bool(genes.peak_signal_variance.eq(0).all()),
        "all_peak_weights_zero": float(summary.iloc[0].zero_peak_weight_fraction) == 1.0
        and bool(genes.mean_peak_weight.eq(0).all()),
        "structural_coverage_below_gate": float(summary.iloc[0].structurally_valid_gene_fraction)
        < float(config["gates"]["minimum_structurally_valid_gene_fraction"]),
        "expected_failed_gates": {
            name for name, value in producer["gates"].items() if not value
        } == {
            "nonzero_q2_gene_fraction", "structurally_valid_gene_fraction",
            "nonzero_weight_lower", "nonzero_weight_upper",
        },
        "rna_not_accessed": producer.get("rna_expression_values_read") is False,
        "forbidden_inputs_not_accessed": producer.get("labels_clusters_pathology_read") is False,
        "no_tuning_or_ranking": producer.get("parameter_tuning_run") is False
        and producer.get("universal_ranking_run") is False
        and producer.get("final_ranking_run") is False,
        "summary_checksum": producer["artifacts"]["feasibility_summary.tsv"]["sha256"]
        == sha256_path(output / "feasibility_summary.tsv"),
        "gene_summary_checksum": producer["artifacts"]["gene_feasibility_summary.tsv"]["sha256"]
        == sha256_path(output / "gene_feasibility_summary.tsv"),
    }
    payload = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_INDEPENDENT_VALIDATION",
        "status": "pass" if all(checks.values()) else "fail",
        "validated_decision": "stop_rwga_v0_after_low_depth_collapse",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "checks": checks,
        "producer_status": producer["status"],
        "rna_expression_values_read": False,
        "labels_clusters_pathology_read": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    path = output / "independent_validation_evidence.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "pass":
        raise ValueError(f"RWGA negative-result validation failed: {[k for k,v in checks.items() if not v]}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_rwga_low_depth_feasibility.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.config, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

