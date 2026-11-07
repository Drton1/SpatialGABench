"""Execute and account for the frozen NU1/NU2 human method panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from spatialgabench.g5_h5b_summary import _artifact_hashes_valid, _id_hash, _input_hashes_valid


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _validate_setup(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    path = root / "results/v1/nm_upgrade/nu1_nu2_human_methods/evidence/setup_evidence.json"
    setup = _json(path)
    if setup.get("status") != "pass":
        raise PermissionError("NU1/NU2 method setup is not complete")
    for field in ("authorization", "annotation_config", "software_config", "planned_execution_manifest", "representation_summary", "representation_completion"):
        record = setup[field]
        artifact = root / record["path"]
        if not artifact.is_file() or sha256_path(artifact) != record["sha256"]:
            raise ValueError(f"NU1/NU2 setup binding mismatch: {field}")
    plan = pd.read_csv(root / setup["planned_execution_manifest"]["path"], sep="\t")
    if len(plan) != 50 or plan.output_dir.duplicated().any():
        raise ValueError("NU1/NU2 planned execution manifest dimensions changed")
    return setup, plan


def _valid_manifest(path: Path) -> bool:
    try:
        return path.is_file() and _json(path).get("status") == "pass"
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def _write_orchestration_failure(run_dir: Path, row: Any, code: int) -> Path:
    path = run_dir / "failure.tsv"
    if path.is_file():
        return path
    pd.DataFrame([{
        "dataset_id": row.dataset_id,
        "representation_id": row.representation_id,
        "method_id": row.method_id,
        "stage": "method_process",
        "failure_type": "ProcessExit",
        "message": f"frozen primary attempt exited with code {code}",
        "status": "fail",
    }]).to_csv(path, sep="\t", index=False)
    return path


def run(root: Path) -> Path:
    setup, plan = _validate_setup(root)
    annotation_config = root / setup["annotation_config"]["path"]
    software_config = root / setup["software_config"]["path"]
    ledger_path = root / "results/v1/nm_upgrade/nu1_nu2_human_methods/execution_ledger.tsv"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    environment = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)

    for row in plan.itertuples(index=False):
        run_dir = root / str(row.output_dir)
        manifest = run_dir / "output_manifest.json"
        failure = run_dir / "failure.tsv"
        if _valid_manifest(manifest):
            code, disposition = 0, "checksum_pending_existing_success"
        elif failure.is_file():
            failure_frame = pd.read_csv(failure, sep="\t")
            if len(failure_frame) != 1 or str(failure_frame.iloc[0].get("status")) != "fail":
                raise ValueError(f"Malformed retained failure record: {failure}")
            code, disposition = 1, "retained_primary_failure_no_retry"
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            if row.run_type == "annotation":
                command = [
                    sys.executable,
                    str(root / "containers/tools/run_with_memory_tracking.py"),
                    "--output", str(run_dir / "process_tree_memory.json"),
                    "--sample-interval", "0.05", "--",
                    sys.executable,
                    str(root / "adapters/g4_m2_annotation_driver.py"),
                    "--config", str(annotation_config),
                    "--dataset", row.dataset_id,
                    "--representation", row.representation_id,
                    "--method", row.method_id,
                    "--root", str(root),
                ]
            else:
                command = [
                    sys.executable,
                    str(root / "adapters/g4_m2_software_driver.py"),
                    "--config", str(software_config),
                    "--dataset", row.dataset_id,
                    "--method", row.method_id,
                    "--root", str(root),
                ]
            log_path = run_dir / "execution.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
            code = int(process.returncode)
            disposition = "primary_success" if code == 0 else "primary_structured_failure"
            if code != 0:
                _write_orchestration_failure(run_dir, row, code)
        rows.append({
            "block": row.block,
            "study_id": row.study_id,
            "dataset_id": row.dataset_id,
            "representation_id": row.representation_id,
            "method_id": row.method_id,
            "run_type": row.run_type,
            "attempt_index": 1,
            "retry_attempts": 0,
            "exit_code": code,
            "disposition": disposition,
            "output_dir": row.output_dir,
        })
        pd.DataFrame(rows).to_csv(ledger_path, sep="\t", index=False)
    return ledger_path


def summarize(root: Path) -> Path:
    setup, plan = _validate_setup(root)
    ledger_path = root / "results/v1/nm_upgrade/nu1_nu2_human_methods/execution_ledger.tsv"
    ledger = pd.read_csv(ledger_path, sep="\t")
    if len(ledger) != len(plan):
        raise ValueError("NU1/NU2 execution ledger is incomplete")
    hash_cache: dict[Path, str] = {}

    def cached_sha(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in hash_cache:
            hash_cache[resolved] = sha256_path(resolved)
        return hash_cache[resolved]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source in plan.itertuples(index=False):
        run_dir = root / str(source.output_dir)
        base = {
            "block": source.block,
            "study_id": source.study_id,
            "accession": source.accession,
            "dataset_id": source.dataset_id,
            "sample_id": source.sample_id,
            "species": source.species,
            "genome_build": source.genome_build,
            "representation_id": source.representation_id,
            "input_track": source.input_track,
            "method_id": source.method_id,
            "role": source.role,
            "run_type": source.run_type,
            "output_dir": source.output_dir,
        }
        manifest_path = run_dir / "output_manifest.json"
        if not _valid_manifest(manifest_path):
            failure_path = run_dir / "failure.tsv"
            if not failure_path.is_file():
                raise ValueError(f"Unaccounted NU1/NU2 contract: {source.output_dir}")
            failure = pd.read_csv(failure_path, sep="\t")
            if len(failure) != 1 or str(failure.iloc[0].get("status")) != "fail":
                raise ValueError(f"Malformed NU1/NU2 failure: {failure_path}")
            detail = failure.iloc[0].to_dict()
            rows.append({**base, "outcome": "structured_failure", "status": "accounted"})
            failures.append({
                **base,
                "failure_stage": detail.get("stage"),
                "failure_type": detail.get("failure_type"),
                "message": detail.get("message"),
                "failure_record": str(failure_path.relative_to(root)),
                "status": "fail",
            })
            continue

        required = {
            name: run_dir / name
            for name in ("input_validation.json", "gene_activity.h5ad", "contract_validation.tsv", "runtime.tsv", "process_tree_memory.json")
        }
        if not all(path.is_file() for path in required.values()):
            raise ValueError(f"Successful NU1/NU2 contract lacks required artifacts: {source.output_dir}")
        manifest = _json(manifest_path)
        validation = _json(required["input_validation.json"])
        memory = _json(required["process_tree_memory.json"])
        contract = pd.read_csv(required["contract_validation.tsv"], sep="\t")
        input_data = ad.read_h5ad(root / str(source.input_path), backed="r")
        output_data = ad.read_h5ad(required["gene_activity.h5ad"], backed="r")
        observation_match = _id_hash(input_data.obs_names) == _id_hash(output_data.obs_names)
        n_observations, n_genes = output_data.shape
        leakage_absent = output_data.uns.get("construction_leakage_inputs") == "none"
        input_data.file.close()
        output_data.file.close()
        artifacts_valid = _artifact_hashes_valid(run_dir, manifest, root, cached_sha)
        inputs_valid = _input_hashes_valid(validation, root, cached_sha)
        boundary_valid = (
            validation.get("status") == "pass"
            and validation.get("forbidden_inputs_used", []) == []
            and all(validation.get(field, False) is False for field in ("rna_accessed", "labels_accessed", "clusters_accessed", "pathology_accessed", "simulation_truth_accessed"))
            and manifest.get("evaluation_run", False) is False
            and manifest.get("final_ranking_run", False) is False
        )
        contract_pass = bool(len(contract) == 18 and contract.status.eq("pass").all())
        success = all((
            memory.get("status") == "pass",
            memory.get("exit_code") == 0,
            observation_match,
            n_observations > 0,
            n_genes > 0,
            artifacts_valid,
            inputs_valid,
            boundary_valid,
            leakage_absent,
            contract_pass,
        ))
        if not success:
            raise ValueError(f"NU1/NU2 successful output failed independent contract audit: {source.output_dir}")
        rows.append({
            **base,
            "outcome": "success",
            "n_observations": n_observations,
            "n_output_genes": n_genes,
            "contract_checks_passed": 18,
            "observation_ids_and_order_match": observation_match,
            "manifest_artifact_hashes_valid": artifacts_valid,
            "live_input_hashes_valid": inputs_valid,
            "construction_boundary_valid": boundary_valid,
            "construction_leakage_absent": leakage_absent,
            "runtime_seconds_noncomparative": float(memory["runtime_seconds"]),
            "peak_aggregate_pss_mb_noncomparative": float(memory["peak_aggregate_pss_mb"]),
            "output_h5ad_sha256": cached_sha(required["gene_activity.h5ad"]),
            "status": "accounted",
        })

    summary = pd.DataFrame(rows)
    failure_columns = [
        "block", "study_id", "accession", "dataset_id", "sample_id", "species",
        "genome_build", "representation_id", "input_track", "method_id", "role",
        "run_type", "output_dir", "failure_stage", "failure_type", "message",
        "failure_record", "status",
    ]
    failure_table = pd.DataFrame(failures, columns=failure_columns)
    evidence_dir = root / "results/v1/nm_upgrade/nu1_nu2_human_methods/evidence"
    summary_path = evidence_dir / "run_summary.tsv"
    failure_path = evidence_dir / "structured_failure_accounting.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    failure_table.to_csv(failure_path, sep="\t", index=False)
    success_count = int(summary.outcome.eq("success").sum())
    evidence = {
        "schema_version": 1,
        "milestone": "NU1_NU2_human_method_construction",
        "status": "pass" if len(summary) == 50 and summary.status.eq("accounted").all() else "blocked",
        "contracts_planned": 50,
        "contracts_accounted": len(summary),
        "successful_contracts": success_count,
        "structured_failures": len(failure_table),
        "annotation_contracts": int(summary.run_type.eq("annotation").sum()),
        "software_contracts": int(summary.run_type.eq("software").sum()),
        "primary_attempts": len(ledger),
        "retry_attempts": int(ledger.retry_attempts.sum()),
        "rna_accessed": False,
        "labels_pathology_images_accessed": False,
        "evaluation_run": False,
        "policy_refit_run": False,
        "resource_ranking_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "interpretation_boundary": "construction_portability_and_contract_qc_only",
        "plan": setup["planned_execution_manifest"],
        "setup": {"path": "results/v1/nm_upgrade/nu1_nu2_human_methods/evidence/setup_evidence.json", "sha256": sha256_path(root / "results/v1/nm_upgrade/nu1_nu2_human_methods/evidence/setup_evidence.json")},
        "execution_ledger": {"path": str(ledger_path.relative_to(root)), "sha256": sha256_path(ledger_path)},
        "run_summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
        "failure_accounting": {"path": str(failure_path.relative_to(root)), "sha256": sha256_path(failure_path)},
        "implementation": {"path": "spatialgabench/nm_human_method_construction.py", "sha256": sha256_path(Path(__file__))},
    }
    evidence_path = evidence_dir / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(run(root) if args.command == "run" else summarize(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
