"""Authorized replicated simulation generation for G4-M3C."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from spatialgabench.simulation.calibration import sha256_file
from spatialgabench.simulation.design import expand_factorial_design, load_design_config
from spatialgabench.simulation.g3 import _runtime_config
from spatialgabench.simulation.generator import (
    generate_empirical_semisynthetic_simulation,
    generate_mechanistic_simulation,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in ("authorization", "preregistration", "amendment", "canary_evidence", "simulation_design"):
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {field}")
    authorization = _json(root / config["authorization"])
    canary = _json(root / config["canary_evidence"])
    if authorization.get("status") != "authorized" or authorization.get("milestone") != "M3C_REPLICATED_SIMULATION_GENERATION":
        raise PermissionError("Replicated simulation generation authorization is not active")
    if authorization.get("factorial_640_generation_authorized") is not True or authorization.get("construction_methods_authorized") is not False or authorization.get("evaluation_authorized") is not False:
        raise PermissionError("Generation authorization boundary is invalid")
    if canary.get("status") != "pass" or canary.get("component_endpoint_evidence") != "component_identifiability_confirmed":
        raise ValueError("Component-identifiable canary evidence does not pass")
    forbidden_flags = ("holdout_access_allowed", "method_construction_run", "evaluation_run", "full_real_data_benchmark_run", "recommendation_policy_fit_run", "final_ranking_run")
    if any(bool(config.get(field)) for field in forbidden_flags):
        raise PermissionError("A forbidden generation config flag is enabled")
    config["_path"] = path
    return config


def _existing_run_valid(run_dir: Path, row: pd.Series) -> bool:
    required = [run_dir / "simulation_provenance.json", run_dir / "validation_summary.tsv", run_dir / "simulation_summary.tsv"]
    if not all(path.is_file() for path in required):
        return False
    provenance = _json(required[0])
    validation = pd.read_csv(required[1], sep="\t")
    return bool(
        provenance.get("simulation_id") == row["simulation_id"]
        and provenance.get("simulation_type") == row["simulation_type"]
        and int(provenance.get("random_seed", -1)) == int(row["random_seed"])
        and provenance.get("generator_version") == "2.1.0"
        and provenance.get("holdout_accessed") is False
        and provenance.get("construction_evaluation_inputs_used") is False
        and not validation.empty
        and (validation["validation_status"] == "pass").all()
    )


def generate(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _config(config_path, root)
    design_config = load_design_config(root / config["simulation_design"])
    design, audit = expand_factorial_design(design_config)
    expected = int(config["expected_simulations"])
    if len(design) != expected or audit.get("status") != "pass":
        raise ValueError("Replicated factorial design does not pass")
    counts = design.groupby(["simulation_type", "random_seed"]).size()
    if set(design.simulation_type) != set(config["expected_tracks"]) or set(design.random_seed) != set(config["expected_seeds"]) or set(counts.to_numpy()) != {int(config["expected_conditions_per_seed_per_track"])}:
        raise ValueError("Replicated factorial strata/seed balance mismatch")
    output = root / config["output_root"]
    runs_dir = output / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    design_path = output / "simulation_factorial_design.tsv"
    audit_path = output / "simulation_factorial_design_audit.json"
    design.to_csv(design_path, sep="\t", index=False)
    _write_json(audit_path, audit)
    run_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for index, (_, row) in enumerate(design.iterrows(), start=1):
        run_dir = runs_dir / str(row["simulation_id"])
        reused = _existing_run_valid(run_dir, row)
        started = time.perf_counter()
        if not reused:
            runtime_config = _runtime_config(design_config, row, run_dir)
            if row["simulation_type"] == "mechanistic":
                outputs = generate_mechanistic_simulation(runtime_config, run_dir)
            else:
                outputs = generate_empirical_semisynthetic_simulation(runtime_config, run_dir, project_root=root)
            validation_path = outputs["validation"]
            provenance_path = outputs["provenance"]
        else:
            validation_path = run_dir / "validation_summary.tsv"
            provenance_path = run_dir / "simulation_provenance.json"
        validation = pd.read_csv(validation_path, sep="\t")
        if validation.empty or not (validation.validation_status == "pass").all():
            raise ValueError(f"Simulation validation failed: {row['simulation_id']}")
        run_rows.append({
            "simulation_id": row["simulation_id"], "simulation_type": row["simulation_type"],
            "random_seed": int(row["random_seed"]), "replicate_id": row["replicate_id"],
            "condition_index": int(row["condition_index"]), "runtime_seconds": time.perf_counter() - started,
            "reused_valid_output": reused, "validation_checks_passed": int((validation.validation_status == "pass").sum()),
            "validation_checks_total": len(validation), "provenance_sha256": sha256_file(provenance_path),
            "status": "pass",
        })
        if index % 20 == 0 or index == expected:
            print(f"M3C factorial generation: {index}/{expected}", flush=True)
    runs = pd.DataFrame(run_rows)
    runs_path = output / "simulation_generation_runs.tsv"
    runs.to_csv(runs_path, sep="\t", index=False)
    manifest_path = output / "simulation_generation_manifest.json"
    manifest = {
        "schema_version": 1, "gate": "G4", "milestone": "M3C_replicated_simulation_generation",
        "status": "pass" if len(runs) == expected and (runs.status == "pass").all() else "fail",
        "design_id": design_config["design_id"], "simulation_rows_expected": expected,
        "simulation_rows_passed": len(runs), "tracks": sorted(runs.simulation_type.unique()),
        "physical_random_seeds": sorted(int(value) for value in runs.random_seed.unique()),
        "conditions_per_seed_per_track": int(config["expected_conditions_per_seed_per_track"]),
        "validation_checks_passed": int(runs.validation_checks_passed.sum()),
        "validation_checks_total": int(runs.validation_checks_total.sum()),
        "reused_valid_outputs": int(runs.reused_valid_output.sum()),
        "runtime_seconds": time.perf_counter() - started_all,
        "artifacts": {
            "design": {"path": str(design_path.relative_to(root)), "sha256": sha256_file(design_path)},
            "design_audit": {"path": str(audit_path.relative_to(root)), "sha256": sha256_file(audit_path)},
            "runs": {"path": str(runs_path.relative_to(root)), "sha256": sha256_file(runs_path)},
        },
        "generator_version": "2.1.0", "component_identifiable_design": True,
        "construction_methods_run": False, "evaluation_run": False,
        "holdout_accessed": False, "full_real_data_benchmark_run": False,
        "recommendation_policy_fit_run": False, "ranking_run": False,
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def summarize(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _config(config_path, root)
    output = root / config["output_root"]
    manifest_path = output / "simulation_generation_manifest.json"
    memory_path = output / "generation_process_tree_memory.json"
    manifest, memory = _json(manifest_path), _json(memory_path)
    expected = int(config["expected_simulations"])
    if manifest.get("status") != "pass" or manifest.get("simulation_rows_passed") != expected:
        raise ValueError("Replicated simulation generation manifest does not pass")
    if memory.get("status") != "pass" or memory.get("exit_code") != 0:
        raise ValueError("Replicated simulation generation memory evidence does not pass")
    if not all((root / item["path"]).is_file() and sha256_file(root / item["path"]) == item["sha256"] for item in manifest["artifacts"].values()):
        raise ValueError("Replicated simulation generation artifact hash mismatch")
    evidence_path = output / "g4_m3c_replicated_simulation_generation_evidence.json"
    evidence = {
        "schema_version": 1, "gate": "G4", "milestone": "M3C_replicated_simulation_generation",
        "status": "pass", "simulation_rows": expected, "generator_version": "2.1.0",
        "design_id": manifest["design_id"], "tracks": manifest["tracks"],
        "physical_random_seeds": manifest["physical_random_seeds"],
        "conditions_per_seed_per_track": manifest["conditions_per_seed_per_track"],
        "validation_checks_passed": manifest["validation_checks_passed"],
        "validation_checks_total": manifest["validation_checks_total"],
        "generation_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_file(manifest_path)},
        "process_tree_memory": {
            "path": str(memory_path.relative_to(root)), "sha256": sha256_file(memory_path),
            "runtime_seconds": memory["runtime_seconds"], "peak_aggregate_pss_mb": memory["peak_aggregate_pss_mb"],
        },
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "canary_evidence": {"path": config["canary_evidence"], "sha256": config["canary_evidence_sha256"]},
        "construction_methods_run": False, "evaluation_run": False, "holdout_accessed": False,
        "full_real_data_benchmark_run": False, "recommendation_policy_fit_run": False,
        "universal_ranking_run": False, "final_ranking_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(evidence_path, evidence)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "summarize"))
    parser.add_argument("--config", default="configs/g4_m3c_replicated_simulation_generation_v2.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    path = generate(args.config, args.root) if args.command == "generate" else summarize(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
