"""Post hoc synthesis of frozen FB3 depth and resource stress outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from spatialgabench.full_benchmark_depth_stress import planned_conditions


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


def _first_depth(depths: list[float], values: list[bool]) -> float | None:
    for depth, value in sorted(zip(depths, values), reverse=True):
        if value:
            return float(depth)
    return None


def _condition_qc(config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in planned_conditions(config):
        base = root / config["output_root"] / "conditions" / condition["condition_id"]
        thinning = _mapping(base / "fragments/thinning_audit.json")
        peaks = _mapping(base / "peaks/peak_stage_audit.json")
        representation = _mapping(base / "representations/matched_representation_validation.json")
        resource_values: dict[str, Any] = {}
        for stage in ("prepare_fragments", "call_peaks", "build_representations"):
            resource = _mapping(base / "resources" / f"{stage}.json")
            resource_values[f"{stage}_runtime_seconds"] = resource.get("runtime_seconds")
            resource_values[f"{stage}_peak_aggregate_pss_mb"] = resource.get("peak_aggregate_pss_mb")
            resource_values[f"{stage}_exit_code"] = resource.get("exit_code")
        rows.append({
            **condition,
            "source_fragment_rows": thinning["source_fragment_rows"],
            "included_fragment_rows": thinning["included_fragment_rows"],
            "output_fragment_rows_verified": thinning["output_fragment_rows_verified"],
            "observed_retained_fraction": thinning["observed_retained_fraction"],
            "binomial_z_residual": thinning["binomial_z_residual"],
            "thinning_status": thinning["status"],
            "macs3_completed": peaks["macs3_completed"],
            "input_peak_rows": peaks["input_peak_rows"],
            "frozen_peak_rows": peaks["frozen_peak_rows"],
            "peak_representation_available": peaks["peak_representation_available"],
            "peak_representation_failure_reason": peaks["peak_representation_failure_reason"],
            "fixed_bin_and_fragment_branches_may_continue": peaks["fixed_bin_and_fragment_branches_may_continue"],
            "representation_status": representation["status"],
            "fragment_filter_contract_verified": representation["fragment_filter_contract_verified"],
            **resource_values,
        })
    return pd.DataFrame(rows)


def _failure_onset(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["dataset_id", "study_id", "species", "genome_build", "method_id", "representation_id"]
    for values, frame in metrics.groupby(keys, sort=True, dropna=False):
        depth = (
            frame.groupby("depth_fraction", sort=False)
            .agg(planned_runs=("success", "size"), successful_runs=("success", "sum"))
            .reset_index()
        )
        depth["any_failure"] = depth.successful_runs.lt(depth.planned_runs)
        depth["complete_failure"] = depth.successful_runs.eq(0)
        first_any = _first_depth(depth.depth_fraction.tolist(), depth.any_failure.tolist())
        first_complete = _first_depth(depth.depth_fraction.tolist(), depth.complete_failure.tolist())
        all_success = depth.loc[~depth.any_failure, "depth_fraction"]
        success_rates = dict(zip(depth.depth_fraction, depth.successful_runs / depth.planned_runs))
        rows.append({
            **dict(zip(keys, values)),
            "success_rate_depth_1": success_rates.get(1.0),
            "success_rate_depth_0_25": success_rates.get(0.25),
            "success_rate_depth_0_05": success_rates.get(0.05),
            "success_rate_depth_0_01": success_rates.get(0.01),
            "first_any_failure_depth": first_any,
            "first_complete_failure_depth": first_complete,
            "lowest_depth_all_replicates_successful": float(all_success.min()) if len(all_success) else None,
            "failure_observed": first_any is not None,
            "complete_failure_observed": first_complete is not None,
        })
    return pd.DataFrame(rows)


def _structured_failures(metrics: pd.DataFrame, config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    failures = metrics.loc[~metrics.success.astype(bool)]
    for record in failures.to_dict("records"):
        base = root / config["output_root"] / "conditions" / record["condition_id"]
        if record["representation_id"] == "filtered_fragments_depth_stress_contract":
            failure_path = base / "methods" / "software" / record["method_id"] / "failure.tsv"
        else:
            failure_path = (
                base / "methods" / "annotation" / record["condition_id"]
                / record["representation_id"] / record["method_id"] / "failure.tsv"
            )
        if not failure_path.is_file():
            raise ValueError(f"FB3 failed contract lacks structured failure record: {failure_path}")
        failure = pd.read_csv(failure_path, sep="\t")
        if len(failure) != 1:
            raise ValueError(f"FB3 failure record must have exactly one row: {failure_path}")
        detail = failure.iloc[0].to_dict()
        rows.append({
            "condition_id": record["condition_id"],
            "dataset_id": record["dataset_id"],
            "study_id": record["study_id"],
            "species": record["species"],
            "genome_build": record["genome_build"],
            "depth_fraction": record["depth_fraction"],
            "seed": record["seed"],
            "representation_id": record["representation_id"],
            "method_id": record["method_id"],
            "failure_stage": detail.get("stage"),
            "failure_type": detail.get("failure_type"),
            "message": detail.get("message"),
            "failure_status": detail.get("status"),
            "failure_record": str(failure_path.relative_to(root)),
        })
    return pd.DataFrame(rows)


def synthesize(
    evidence_path: str | Path,
    config_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root = Path(root).resolve()
    evidence_path = Path(evidence_path).resolve()
    config_path = Path(config_path).resolve()
    evidence = _mapping(evidence_path)
    config = _mapping(config_path)
    if evidence.get("status") != "pass":
        raise ValueError("FB3 completion evidence must pass before synthesis")
    if _sha256(config_path) != evidence["config"]["sha256"]:
        raise ValueError("FB3 config checksum differs from completion evidence")

    metrics_path = root / evidence["metrics"]["path"]
    metrics = pd.read_csv(metrics_path, sep="\t")
    if len(metrics) != int(config["planned_contracts"]):
        raise ValueError("FB3 metrics do not contain the preregistered contract count")

    output = root / config["output_root"]
    condition_qc = _condition_qc(config, root)
    condition_qc_path = output / "condition_construction_qc.tsv"
    condition_qc.to_csv(condition_qc_path, sep="\t", index=False)

    failure_onset = _failure_onset(metrics)
    failure_onset_path = output / "failure_onset_by_contract.tsv"
    failure_onset.to_csv(failure_onset_path, sep="\t", index=False)

    structured_failures = _structured_failures(metrics, config, root)
    structured_failures_path = output / "structured_failure_accounting.tsv"
    structured_failures.to_csv(structured_failures_path, sep="\t", index=False)

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
    resource_summary_path = output / "descriptive_resource_summary.tsv"
    resource_summary.to_csv(resource_summary_path, sep="\t", index=False)

    synthesis = {
        "schema_version": 1,
        "milestone": "FB3_controlled_depth_resource_stress_synthesis",
        "status": "pass",
        "conditions": len(condition_qc),
        "contracts": len(metrics),
        "contract_families": len(failure_onset),
        "successful_contracts": int(metrics.success.astype(bool).sum()),
        "failed_contracts": int((~metrics.success.astype(bool)).sum()),
        "structured_failure_records": len(structured_failures),
        "contract_families_with_any_failure": int(failure_onset.failure_observed.sum()),
        "conditions_without_peak_representation": int((~condition_qc.peak_representation_available.astype(bool)).sum()),
        "rna_accessed": False,
        "labels_accessed": False,
        "pathology_accessed": False,
        "policy_refit_run": False,
        "resource_ranking_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "interpretation_boundary": "within_contract_depth_robustness_and_descriptive_resources_only",
        "artifacts": {
            "condition_construction_qc.tsv": {
                "path": str(condition_qc_path.relative_to(root)), "sha256": _sha256(condition_qc_path),
            },
            "failure_onset_by_contract.tsv": {
                "path": str(failure_onset_path.relative_to(root)), "sha256": _sha256(failure_onset_path),
            },
            "structured_failure_accounting.tsv": {
                "path": str(structured_failures_path.relative_to(root)),
                "sha256": _sha256(structured_failures_path),
            },
            "descriptive_resource_summary.tsv": {
                "path": str(resource_summary_path.relative_to(root)), "sha256": _sha256(resource_summary_path),
            },
        },
        "completion_evidence": {
            "path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path),
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": _sha256(config_path)},
        "implementation": {
            "path": "spatialgabench/full_benchmark_depth_synthesis.py",
            "sha256": _sha256(Path(__file__)),
        },
    }
    synthesis_path = output / "synthesis_evidence.json"
    synthesis_path.write_text(json.dumps(synthesis, indent=2, sort_keys=True) + "\n")
    return synthesis_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        default="results/v1/full_benchmark/fb3_depth_resource_stress/completion_evidence.json",
    )
    parser.add_argument("--config", default="configs/full_benchmark_fb3_depth_resource_stress.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(synthesize(args.evidence, args.config, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
