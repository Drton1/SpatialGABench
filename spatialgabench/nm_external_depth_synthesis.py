"""Synthesize NM-U3 external depth-stress outputs without ranking methods."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from spatialgabench.full_benchmark_depth_stress import planned_conditions
from spatialgabench.full_benchmark_depth_synthesis import _failure_onset, _structured_failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def _condition_qc(config: dict[str, Any], root: Path) -> pd.DataFrame:
    anchor_id = config["full_anchor"]["condition_id"]
    rows: list[dict[str, Any]] = []
    for condition in planned_conditions(config):
        base = root / config["output_root"] / "conditions" / condition["condition_id"]
        thinning = _mapping(base / "fragments/thinning_audit.json")
        peaks = _mapping(base / "peaks/peak_stage_audit.json")
        representation = _mapping(base / "representations/matched_representation_validation.json")
        reused = condition["condition_id"] == anchor_id
        row = {
            **condition,
            "anchor_reused": reused,
            "new_scientific_execution": not reused,
            "source_fragment_rows": thinning.get("source_fragment_rows", config["studies"][condition["dataset_id"]]["fragment_rows"]),
            "included_fragment_rows": thinning.get("included_fragment_rows", config["studies"][condition["dataset_id"]]["fragment_rows"]),
            "output_fragment_rows_verified": thinning.get("output_fragment_rows_verified", config["studies"][condition["dataset_id"]]["fragment_rows"]),
            "observed_retained_fraction": thinning.get("observed_retained_fraction", 1.0),
            "binomial_z_residual": thinning.get("binomial_z_residual", 0.0),
            "thinning_status": thinning["status"],
            "macs3_completed": peaks.get("macs3_completed", True),
            "input_peak_rows": peaks.get("input_peak_rows"),
            "frozen_peak_rows": peaks.get("frozen_peak_rows"),
            "peak_representation_available": peaks.get("peak_representation_available", True),
            "peak_representation_failure_reason": peaks.get("peak_representation_failure_reason", ""),
            "fixed_bin_and_fragment_branches_may_continue": peaks.get("fixed_bin_and_fragment_branches_may_continue", True),
            "representation_status": representation["status"],
            "fragment_filter_contract_verified": representation.get("fragment_filter_contract_verified", True),
        }
        for stage in ("prepare_fragments", "call_peaks", "build_representations"):
            resource_path = base / "resources" / f"{stage}.json"
            resource = _mapping(resource_path) if resource_path.is_file() else {}
            row[f"{stage}_runtime_seconds"] = resource.get("runtime_seconds")
            row[f"{stage}_peak_aggregate_pss_mb"] = resource.get("peak_aggregate_pss_mb")
            row[f"{stage}_exit_code"] = resource.get("exit_code", 0 if reused else None)
        rows.append(row)
    return pd.DataFrame(rows)


def _authorization_accounting(metrics: pd.DataFrame, config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in metrics.to_dict("records"):
        base = root / config["output_root"] / "conditions" / record["condition_id"]
        if float(record["depth_fraction"]) == 1.0:
            authorization_path = config["full_anchor"]["h5b_evidence"]["path"]
            authorization_sha = config["full_anchor"]["h5b_evidence"]["sha256"]
            version = "reused_frozen_H5B_anchor"
        elif record["representation_id"] == "filtered_fragments_depth_stress_contract":
            wrapper = _mapping(base / "methods/software" / record["method_id"] / "wrapper_config.yaml")
            authorization_path = wrapper["execution_authorization"]
            authorization_sha = wrapper["execution_authorization_sha256"]
            version = "V2" if "_V2_" in authorization_path else "V1"
        else:
            annotation = _mapping(base / "execution/annotation_config.yaml")
            authorization_path = annotation["authorization"]
            authorization_sha = annotation["authorization_sha256"]
            version = "V2" if "_V2_" in authorization_path else "V1"
        rows.append({
            "condition_id": record["condition_id"],
            "depth_fraction": record["depth_fraction"],
            "seed": record["seed"],
            "representation_id": record["representation_id"],
            "method_id": record["method_id"],
            "success": record["success"],
            "authorization_version": version,
            "authorization_path": authorization_path,
            "authorization_sha256_at_execution": authorization_sha,
            "framework_correction_only": version == "V1",
        })
    return pd.DataFrame(rows)


def synthesize(evidence_path: str | Path, config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    evidence_path = Path(evidence_path).resolve()
    config_path = Path(config_path).resolve()
    evidence, config = _mapping(evidence_path), _mapping(config_path)
    if evidence.get("status") != "pass" or _sha256(config_path) != evidence["config"]["sha256"]:
        raise ValueError("NU3 completion evidence/config is not frozen and complete")

    metrics = pd.read_csv(root / evidence["metrics"]["path"], sep="\t")
    if len(metrics) != int(config["planned_contracts"]):
        raise ValueError("NU3 metrics do not contain the preregistered contract count")
    output = root / config["output_root"]

    condition_qc = _condition_qc(config, root)
    failure_onset = _failure_onset(metrics)
    structured_failures = _structured_failures(metrics, config, root)
    if structured_failures.empty:
        structured_failures = pd.DataFrame(columns=[
            "condition_id", "dataset_id", "study_id", "species", "genome_build",
            "depth_fraction", "seed", "representation_id", "method_id",
            "failure_stage", "failure_type", "message", "failure_status", "failure_record",
        ])
    success = metrics.loc[metrics.success.astype(bool)].copy()
    resource_summary = (
        success.groupby(
            ["study_id", "species", "genome_build", "method_id", "representation_id", "depth_fraction"],
            dropna=False,
        )
        .agg(
            successful_runs=("success", "size"),
            median_runtime_seconds=("runtime_seconds", "median"),
            min_runtime_seconds=("runtime_seconds", "min"),
            max_runtime_seconds=("runtime_seconds", "max"),
            median_peak_aggregate_pss_mb=("peak_aggregate_pss_mb", "median"),
            min_peak_aggregate_pss_mb=("peak_aggregate_pss_mb", "min"),
            max_peak_aggregate_pss_mb=("peak_aggregate_pss_mb", "max"),
            median_runtime_ratio_vs_full=("runtime_ratio_vs_full", "median"),
            median_pss_ratio_vs_full=("pss_ratio_vs_full", "median"),
        )
        .reset_index()
    )
    authorization_accounting = _authorization_accounting(metrics, config, root)
    artifacts = {
        "condition_construction_qc.tsv": condition_qc,
        "failure_onset_by_contract.tsv": failure_onset,
        "structured_failure_accounting.tsv": structured_failures,
        "descriptive_resource_summary.tsv": resource_summary,
        "authorization_version_accounting.tsv": authorization_accounting,
    }
    records: dict[str, dict[str, str]] = {}
    for name, table in artifacts.items():
        path = output / name
        table.to_csv(path, sep="\t", index=False)
        records[name] = {"path": str(path.relative_to(root)), "sha256": _sha256(path)}

    synthesis = {
        "schema_version": 1,
        "milestone": "NU3_external_controlled_depth_stress_synthesis",
        "status": "pass",
        "conditions": len(condition_qc),
        "contracts": len(metrics),
        "reused_full_depth_contracts": int(config["reused_full_depth_contracts"]),
        "new_reduced_depth_contracts": int(config["new_reduced_depth_contracts"]),
        "successful_contracts": int(metrics.success.astype(bool).sum()),
        "failed_contracts": int((~metrics.success.astype(bool)).sum()),
        "structured_failure_records": len(structured_failures),
        "v1_retained_software_contracts": int(authorization_accounting.authorization_version.eq("V1").sum()),
        "contract_families_with_any_failure": int(failure_onset.failure_observed.sum()),
        "conditions_without_peak_representation": int((~condition_qc.peak_representation_available.astype(bool)).sum()),
        "rna_accessed": False,
        "labels_accessed": False,
        "pathology_accessed": False,
        "policy_refit_run": False,
        "resource_ranking_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "interpretation_boundary": "within_contract_external_depth_robustness_and_descriptive_resources_only",
        "artifacts": records,
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path)},
        "config": {"path": str(config_path.relative_to(root)), "sha256": _sha256(config_path)},
        "implementation": {"path": "spatialgabench/nm_external_depth_synthesis.py", "sha256": _sha256(Path(__file__))},
    }
    path = output / "synthesis_evidence.json"
    path.write_text(json.dumps(synthesis, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="results/v1/nm_upgrade/nu3_external_depth_stress/completion_evidence.json")
    parser.add_argument("--config", default="configs/nm_upgrade_external_depth_stress.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(synthesize(args.evidence, args.config, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
