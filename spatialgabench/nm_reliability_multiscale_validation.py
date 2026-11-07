"""Independent validation for the RWGA toy depth ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from spatialgabench.nm_reliability_multiscale_toy import sha256_path, validate_ladder


def validate(
    config_path: str | Path,
    evidence_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    evidence_path = Path(evidence_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "producer_status_pass": evidence.get("status") == "pass",
        "config_checksum": evidence["config"]["sha256"] == sha256_path(config_path),
        "implementation_checksum": evidence["implementation"]["sha256"]
        == sha256_path(root / evidence["implementation"]["path"]),
        "core_checksum": evidence["core"]["sha256"]
        == sha256_path(root / evidence["core"]["path"]),
        "rna_not_used": evidence.get("rna_used_for_construction") is False
        and evidence.get("rna_used_for_evaluation") is False,
        "forbidden_inputs_not_used": evidence.get("labels_clusters_pathology_used") is False,
        "no_parameter_tuning": evidence.get("parameter_tuning_run") is False,
        "no_ranking": evidence.get("universal_ranking_run") is False
        and evidence.get("final_ranking_run") is False,
    }
    summary_record = evidence["artifacts"]["toy_depth_ladder.tsv"]
    genes_record = evidence["artifacts"]["toy_gene_summary.tsv"]
    summary_path = root / summary_record["path"]
    genes_path = root / genes_record["path"]
    checks["summary_checksum"] = summary_record["sha256"] == sha256_path(summary_path)
    checks["gene_summary_checksum"] = genes_record["sha256"] == sha256_path(genes_path)
    summary = pd.read_csv(summary_path, sep="\t")
    genes = pd.read_csv(genes_path, sep="\t")
    for name, value in validate_ladder(summary, config).items():
        checks[f"recomputed_{name}"] = bool(value)
    checks["exact_summary_rows"] = len(summary) == len(config["depth_fractions"])
    checks["exact_gene_rows"] = len(genes) == len(config["depth_fractions"]) * int(config["expected_genes"])
    checks["all_gene_risk_finite"] = bool(
        genes[["signal_variance", "structural_risk", "median_peak_weight", "median_continuous_risk"]]
        .notna().all().all()
    )
    payload = {
        "schema_version": 1,
        "milestone": f"{config['milestone']}_INDEPENDENT_VALIDATION",
        "status": "pass" if all(checks.values()) else "fail",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "checks": checks,
        "producer_evidence": {
            "path": str(evidence_path.relative_to(root)), "sha256": sha256_path(evidence_path),
        },
        "rna_expression_values_read": False,
        "labels_clusters_pathology_read": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    output = evidence_path.parent / "independent_validation_evidence.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "pass":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"RWGA toy validation failed: {failed}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_reliability_multiscale_toy.yaml")
    parser.add_argument(
        "--evidence",
        default="results/v1/nm_method_upgrade/rwga_m1_toy/completion_evidence.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.config, args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

