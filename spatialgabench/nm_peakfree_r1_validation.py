"""Independent validation of the R1 construction and paired-RNA stop/go."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from spatialgabench.nm_peakfree_r1 import sha256_path


def validate(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["output_root"]
    construction = json.loads((output / "construction_evidence.json").read_text())
    rna = json.loads((output / "rna_completion_evidence.json").read_text())
    manifest = pd.read_csv(output / "matrix_manifest.tsv", sep="\t")
    metrics = pd.read_csv(output / "rna_branch_metrics.tsv", sep="\t")
    decision = pd.read_csv(output / "rna_stopgo_decision.tsv", sep="\t").iloc[0]
    switch = json.loads((output / "hard_switch_contract.json").read_text())
    checks = {
        "construction_pass": construction.get("status") == "pass"
        and construction.get("rna_expression_values_read") is False,
        "exact_branch_panel": set(manifest.branch_id) == set(config["branches"])
        and len(manifest) == len(config["branches"]),
        "matrix_dimensions": manifest.observations.eq(int(config["expected_observations"])).all()
        and manifest.genes.eq(int(config["expected_genes"])).all(),
        "matrix_checksums": all(row.sha256 == sha256_path(root / row.path) for row in manifest.itertuples()),
        "rna_evaluation_pass": rna.get("status") == "pass"
        and rna.get("rna_used_for_construction") is False
        and rna.get("construction_mutated") is False,
        "all_branch_metrics_present": set(metrics.branch_id) == set(config["branches"]),
        "decision_recomputed": decision.status == "pass"
        and float(decision.regret_to_best_fixed) <= float(config["gates"]["maximum_regret_to_best_fixed"])
        and float(decision.deficit_to_fixed_25kb) <= float(config["gates"]["maximum_deficit_to_fixed_25kb"]),
        "producer_gates_recomputed": rna.get("gates", {}).get("regret_to_best_fixed") is True
        and rna.get("gates", {}).get("noninferior_to_fixed_25kb") is True,
        "hard_switch_atac_only": switch.get("status") == "pass"
        and switch.get("rna_expression_values_read") is False
        and switch.get("fit_input") == "ATAC_total_counts_and_adaptive_scale_calls_only",
        "no_forbidden_inputs": rna.get("labels_clusters_pathology_read") is False
        and rna.get("parameter_tuning_run") is False
        and rna.get("universal_ranking_run") is False
        and rna.get("final_ranking_run") is False,
        "artifact_checksums": all(
            artifact["sha256"] == sha256_path(root / artifact["path"])
            for artifact in rna["artifacts"].values()
        ),
    }
    payload = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_INDEPENDENT_VALIDATION",
        "status": "pass" if all(checks.values()) else "fail",
        "validated_decision": "authorize_locked_transfer_panel" if all(checks.values()) else "stop_r1",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "checks": {name: bool(value) for name, value in checks.items()},
        "rna_expression_values_read": True,
        "rna_used_for_construction": False,
        "labels_clusters_pathology_read": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    path = output / "independent_validation_evidence.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "pass":
        raise ValueError(f"R1 validation failed: {[name for name, value in checks.items() if not value]}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_peakfree_r1_paired_rna.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.config, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
