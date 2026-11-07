"""G4-M3A development-only admissibility and portability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def wilson_interval(successes: int, total: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return math.nan, math.nan
    if not math.isclose(confidence_level, 0.95):
        raise ValueError("M3A freezes a 0.95 confidence level")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _species(genome_build: object) -> str:
    return "mouse" if str(genome_build) == "mm10" else "human" if str(genome_build) == "hg38" else "unresolved"


def _track(representation: object) -> str:
    value = str(representation)
    if value.startswith("fixed_bin"):
        return "B"
    if "fragment" in value:
        return "F"
    return "P"


def _resource_value(row: pd.Series, name: str) -> float:
    value = row.get(name, math.nan)
    return float(value) if pd.notna(value) else math.nan


def _base_record(
    row: pd.Series,
    *,
    component: str,
    category: str,
    representation_id: str,
    subset_only: bool,
    eligible: bool,
    status: str,
) -> dict[str, Any]:
    contract_passed = row.get("contract_checks_passed", pd.NA)
    contract_total = row.get("contract_checks_total", pd.NA)
    return {
        "study_id": str(row.get("study_id", "")),
        "dataset_id": str(row.get("dataset_id", "")),
        "sample_id": str(row.get("sample_id", "")),
        "species": str(row.get("species", _species(row.get("genome_build")))),
        "genome_build": str(row.get("genome_build", "")),
        "representation_id": representation_id,
        "input_track": _track(representation_id),
        "method_id": str(row.get("method_id", "")),
        "method_category": category,
        "source_component": component,
        "subset_only": subset_only,
        "eligible": eligible,
        "planned_status": "eligible" if eligible else "not_applicable",
        "run_status": status,
        "valid_output_indicator": status == "pass" if eligible else pd.NA,
        "included_in_primary_denominator": eligible,
        "not_applicable_reason": str(row.get("not_applicable_reason", "")) if not eligible else "",
        "n_observations": row.get("n_observations", row.get("n_subset_observations", pd.NA)),
        "requested_cells": row.get("requested_cells", pd.NA),
        "n_output_features": row.get("n_output_genes", row.get("n_components", pd.NA)),
        "output_nnz": row.get("output_nnz", pd.NA),
        "contract_checks_passed": contract_passed,
        "contract_checks_total": contract_total,
        "output_contract_pass": (
            int(contract_passed) == int(contract_total) and int(contract_total) > 0
            if pd.notna(contract_passed) and pd.notna(contract_total)
            else pd.NA
        ),
        "runtime_seconds_observed": _resource_value(row, "runtime_seconds_noncomparative"),
        "peak_aggregate_pss_mb_observed": _resource_value(row, "peak_aggregate_pss_mb_noncomparative"),
        "peak_aggregate_rss_mb_observed": _resource_value(row, "peak_aggregate_rss_mb_noncomparative"),
        "resource_comparison_allowed": False,
        "resource_scope": "descriptive_construction_observation_only",
    }


def _read_source_rows(root: Path, evidence: dict[str, Any], component: str, negative_controls: set[str]) -> list[dict[str, Any]]:
    summary_record = evidence.get("run_summary", {})
    summary_path = root / str(summary_record.get("path", ""))
    if not summary_path.is_file() or sha256(summary_path) != summary_record.get("sha256"):
        raise ValueError(f"{component}: run-summary checksum mismatch")
    frame = pd.read_csv(summary_path, sep="\t")
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        method = str(row.get("method_id", ""))
        if component == "m2a":
            category = "negative_control" if method in negative_controls else "hand_written_annotation"
            representation = str(row["representation_id"])
            subset_only = False
            eligible = True
        elif component == "m2b":
            category = "fragment_software"
            representation = "filtered_fragments_with_frozen_peak_observation_contract"
            subset_only = False
            eligible = True
        elif component == "m2c_raw_lsi":
            category = "representation_control"
            representation = str(row["representation_id"])
            subset_only = False
            eligible = True
        elif component == "m2c_cicero":
            category = "subset_scalability"
            representation = str(row["representation_id"])
            subset_only = True
            eligible = bool(row["eligible"])
        else:
            raise ValueError(f"Unknown M3A source component: {component}")
        records.append(
            _base_record(
                row,
                component=component,
                category=category,
                representation_id=representation,
                subset_only=subset_only,
                eligible=eligible,
                status=str(row["status"]),
            )
        )
    return records


def _summary_row(task_id: str, scope_type: str, scope_id: str, frame: pd.DataFrame, confidence: float) -> dict[str, Any]:
    eligible = frame[frame["included_in_primary_denominator"]]
    successes = int((eligible["run_status"] == "pass").sum())
    total = len(eligible)
    low, high = wilson_interval(successes, total, confidence)
    contract_rows = eligible[eligible["output_contract_pass"].notna()]
    contract_successes = int((contract_rows["output_contract_pass"] == True).sum())  # noqa: E712
    return {
        "task_id": task_id,
        "scope_type": scope_type,
        "scope_id": scope_id,
        "eligible_runs": total,
        "successful_runs": successes,
        "failed_runs": total - successes,
        "success_rate": successes / total if total else math.nan,
        "wilson_95ci_low": low,
        "wilson_95ci_high": high,
        "contract_evaluable_runs": len(contract_rows),
        "contract_passes": contract_successes,
        "output_contract_pass_rate": contract_successes / len(contract_rows) if len(contract_rows) else math.nan,
        "not_applicable_runs": int((frame["run_status"] == "not_applicable").sum()),
        "subset_only": bool(frame["subset_only"].all()),
        "status": "pass" if total > 0 and successes == total else "partial",
    }


def _summaries(run_table: pd.DataFrame, confidence: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(_summary_row("construction_qc", "overall", "all_eligible_runs", run_table, confidence))
    full_data = run_table[~run_table["subset_only"]]
    rows.append(_summary_row("portability_runtime_failure", "track", "full_data", full_data, confidence))
    subset = run_table[run_table["subset_only"]]
    rows.append(_summary_row("portability_runtime_failure", "track", "subset_scalability", subset, confidence))
    for category, frame in run_table.groupby("method_category", sort=True):
        rows.append(_summary_row("construction_qc", "method_category", str(category), frame, confidence))
    for method, frame in run_table.groupby("method_id", sort=True):
        rows.append(_summary_row("portability_runtime_failure", "method", str(method), frame, confidence))
    return pd.DataFrame(rows)


def _historical_attempts(root: Path, evidence_by_component: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, evidence in evidence_by_component.items():
        paths = list(evidence.get("retained_resolved_failures", [])) + list(evidence.get("retained_attempts", []))
        for relative in paths:
            path = root / str(relative)
            failure_path = path if path.name == "failure.tsv" else path / "failure.tsv"
            recovery = "pre_resume_artifacts" in str(path)
            rows.append({
                "source_component": component,
                "artifact_path": str(relative),
                "artifact_exists": path.exists(),
                "failure_record_exists": failure_path.is_file(),
                "attempt_class": "interrupted_scheduler_recovery_artifacts" if recovery else "resolved_engineering_attempt",
                "included_in_primary_denominator": False,
                "exclusion_reason": "superseded engineering attempt before the accepted final artifact; not a distinct frozen method-input contract",
            })
    return pd.DataFrame(rows)


def _hardware_snapshot() -> dict[str, Any]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    memory_bytes = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    return {
        "schema_version": 1,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_bytes,
        "resource_interpretation": "descriptive_noncomparative_only",
        "noncomparative_reasons": [
            "source construction workflows did not all use the same concurrency schedule",
            "system load was not isolated across source construction windows",
            "Cicero is subset-only and cannot be pooled with full-data methods",
        ],
    }


def evaluate(config_path: str | Path, root: str | Path = ".") -> dict[str, Path]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    for key in ("authorization", "preregistration", "previous_gate"):
        path = root / config[key]
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"Frozen M3A artifact checksum mismatch: {key}")
    authorization = _load_json(root / config["authorization"])
    if authorization.get("status") != "authorized" or authorization.get("milestone") != "M3A":
        raise PermissionError("M3A authorization is not active")
    if authorization.get("holdout_accessed") is not False or authorization.get("biological_task_evaluation_authorized") is not False:
        raise PermissionError("M3A authorization boundary is invalid")
    if set(config["primary_tasks"]) != {"construction_qc", "portability_runtime_failure"}:
        raise ValueError("M3A task panel differs from the authorized primary endpoints")
    if config.get("resource_comparison_allowed") is not False or config.get("universal_ranking_allowed") is not False:
        raise PermissionError("M3A cannot compare resources or rank methods")

    evidence_by_component: dict[str, dict[str, Any]] = {}
    for component, record in config["input_evidence"].items():
        evidence_path = root / record["path"]
        if not evidence_path.is_file() or sha256(evidence_path) != record["sha256"]:
            raise ValueError(f"M3A input evidence checksum mismatch: {component}")
        evidence = _load_json(evidence_path)
        if evidence.get("status") != "pass" or evidence.get("holdout_accessed") is not False:
            raise ValueError(f"M3A source evidence is inadmissible: {component}")
        evidence_by_component[component] = evidence

    records: list[dict[str, Any]] = []
    negative_controls = set(config["negative_controls"])
    for component, evidence in evidence_by_component.items():
        records.extend(_read_source_rows(root, evidence, component, negative_controls))
    run_table = pd.DataFrame(records).sort_values(
        ["method_category", "method_id", "dataset_id", "representation_id", "requested_cells"],
        kind="mergesort",
        na_position="last",
    )
    if len(run_table) != 195:
        raise ValueError(f"M3A planned run panel must contain 195 rows, observed {len(run_table)}")
    if int(run_table["eligible"].sum()) != 189:
        raise ValueError("M3A eligible denominator must contain 189 runs")
    if int((run_table["run_status"] == "not_applicable").sum()) != 6:
        raise ValueError("M3A requires exactly six preregistered not-applicable Cicero combinations")
    if not (run_table.loc[run_table["eligible"], "run_status"] == "pass").all():
        raise ValueError("M3A contains unresolved eligible final-run failures")
    contract_total = int(run_table["contract_checks_total"].fillna(0).sum())
    contract_passed = int(run_table["contract_checks_passed"].fillna(0).sum())
    if contract_total != contract_passed or contract_total != 3402:
        raise ValueError(f"M3A contract total mismatch: {contract_passed}/{contract_total}")

    output_root = root / config["output_root"]
    table_dir = output_root / "tables"
    evidence_dir = output_root / "evidence"
    table_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_table_path = table_dir / "g4_m3a_admissibility_run_table.tsv"
    summary_path = table_dir / "g4_m3a_success_rate_summary.tsv"
    not_applicable_path = table_dir / "g4_m3a_not_applicable.tsv"
    resources_path = table_dir / "g4_m3a_resource_observations_noncomparative.tsv"
    attempts_path = table_dir / "g4_m3a_historical_engineering_attempts.tsv"
    hardware_path = evidence_dir / "g4_m3a_hardware_snapshot.json"
    run_table.to_csv(run_table_path, sep="\t", index=False)
    summary = _summaries(run_table, float(config["confidence_level"]))
    summary.to_csv(summary_path, sep="\t", index=False)
    run_table[run_table["run_status"] == "not_applicable"].to_csv(not_applicable_path, sep="\t", index=False)
    resource_columns = [
        "study_id", "dataset_id", "sample_id", "species", "genome_build", "representation_id",
        "input_track", "method_id", "method_category", "subset_only", "n_observations",
        "requested_cells", "runtime_seconds_observed", "peak_aggregate_pss_mb_observed",
        "peak_aggregate_rss_mb_observed", "resource_comparison_allowed", "resource_scope",
    ]
    run_table[run_table["eligible"]][resource_columns].to_csv(resources_path, sep="\t", index=False)
    attempts = _historical_attempts(root, evidence_by_component)
    attempts.to_csv(attempts_path, sep="\t", index=False)
    hardware_path.write_text(json.dumps(_hardware_snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = {
        "run_table": run_table_path,
        "success_rate_summary": summary_path,
        "not_applicable": not_applicable_path,
        "resource_observations": resources_path,
        "historical_engineering_attempts": attempts_path,
        "hardware_snapshot": hardware_path,
    }
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3A_admissibility_portability_evaluation",
        "status": "pass",
        "tasks_evaluated": ["construction_qc", "portability_runtime_failure"],
        "planned_runs": len(run_table),
        "eligible_runs": int(run_table["eligible"].sum()),
        "eligible_runs_passed": int((run_table.loc[run_table["eligible"], "run_status"] == "pass").sum()),
        "eligible_runs_failed": int((run_table.loc[run_table["eligible"], "run_status"] == "fail").sum()),
        "full_data_eligible_runs": int((run_table["eligible"] & ~run_table["subset_only"]).sum()),
        "subset_scalability_eligible_runs": int((run_table["eligible"] & run_table["subset_only"]).sum()),
        "not_applicable_runs": int((run_table["run_status"] == "not_applicable").sum()),
        "not_applicable_in_primary_denominator": False,
        "contract_checks_passed": contract_passed,
        "contract_checks_total": contract_total,
        "historical_engineering_attempts_reported": len(attempts),
        "historical_engineering_attempts_in_primary_denominator": False,
        "resource_measurements_comparative": False,
        "resource_interpretation": "descriptive_noncomparative_only",
        "universal_ranking_run": False,
        "method_ranking_run": False,
        "biological_task_evaluation_run": False,
        "rna_evaluation_run": False,
        "label_evaluation_run": False,
        "simulation_truth_evaluation_run": False,
        "holdout_accessed": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "manuscript_results_written": False,
        "next_milestone_authorized": False,
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "preregistration": {"path": config["preregistration"], "sha256": config["preregistration_sha256"]},
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
        "input_evidence": config["input_evidence"],
        "config": {"path": _relative(config_path, root), "sha256": sha256(config_path)},
        "implementation": {"path": "spatialgabench/g4_m3a.py", "sha256": sha256(Path(__file__))},
        "artifacts": {
            key: {"path": _relative(path, root), "sha256": sha256(path), "rows": len(pd.read_csv(path, sep="\t")) if path.suffix == ".tsv" else None}
            for key, path in artifacts.items()
        },
    }
    evidence_path = evidence_dir / "g4_m3a_completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**artifacts, "evidence": evidence_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m3a_admissibility.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    outputs = evaluate(args.config, args.root)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
