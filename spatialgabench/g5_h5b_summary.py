"""Evaluation-free completion audit for G5-H5B external method construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


FORBIDDEN_ACCESS_FIELDS = (
    "rna_accessed",
    "labels_accessed",
    "clusters_accessed",
    "pathology_accessed",
    "simulation_truth_accessed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_hash(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _artifact_hashes_valid(
    run_dir: Path,
    manifest: dict[str, Any],
    root: Path,
    hash_file: Any,
) -> bool:
    if isinstance(manifest.get("artifacts"), dict):
        return all(
            (run_dir / name).is_file() and hash_file(run_dir / name) == digest
            for name, digest in manifest["artifacts"].items()
        )
    bindings = (
        ("output_h5ad", "output_h5ad_sha256"),
        ("contract_tsv", "contract_tsv_sha256"),
        ("runtime_tsv", "runtime_tsv_sha256"),
    )
    return all(
        (root / str(manifest.get(path_key, ""))).is_file()
        and hash_file(root / str(manifest[path_key])) == manifest.get(hash_key)
        for path_key, hash_key in bindings
    )


def _input_hashes_valid(validation: dict[str, Any], root: Path, hash_file: Any) -> bool:
    input_hash = validation.get("input_sha256")
    if isinstance(input_hash, str):
        path = root / str(validation.get("input_path", ""))
        return path.is_file() and hash_file(path) == input_hash
    if not isinstance(input_hash, dict) or not isinstance(validation.get("resolved_paths"), dict):
        return False
    resolved = validation["resolved_paths"]
    return all(
        key in resolved and Path(resolved[key]).is_file() and hash_file(Path(resolved[key])) == digest
        for key, digest in input_hash.items()
    )


def summarize(
    plan_path: str | Path,
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(root).resolve()
    plan_path = Path(plan_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(plan_path, sep="\t").sort_values(
        ["dataset_id", "run_type", "representation_id", "method_id"]
    )
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}

    def cached_sha(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in hash_cache:
            hash_cache[resolved] = sha256(resolved)
        return hash_cache[resolved]

    for source in plan.itertuples(index=False):
        run_dir = root / str(source.output_dir)
        base = {
            "study_id": source.study_id,
            "accession": source.accession,
            "dataset_id": source.dataset_id,
            "sample_id": source.sample_id,
            "representation_id": source.representation_id,
            "input_track": source.input_track,
            "method_id": source.method_id,
            "role": source.role,
            "run_type": source.run_type,
            "output_dir": source.output_dir,
        }
        required = {
            name: run_dir / name
            for name in (
                "output_manifest.json",
                "input_validation.json",
                "gene_activity.h5ad",
                "contract_validation.tsv",
                "runtime.tsv",
                "process_tree_memory.json",
            )
        }
        if not all(path.is_file() for path in required.values()):
            rows.append({**base, "status": "missing"})
            attempts.append(
                {
                    **base,
                    "attempt_index": 1,
                    "attempt_type": "primary",
                    "attempt_authorized": True,
                    "attempt_executed": False,
                    "retry_attempts": 0,
                    "status": "missing",
                }
            )
            continue
        try:
            manifest = _load_json(required["output_manifest.json"])
            validation = _load_json(required["input_validation.json"])
            memory = _load_json(required["process_tree_memory.json"])
            contract = pd.read_csv(required["contract_validation.tsv"], sep="\t")
            runtime = pd.read_csv(required["runtime.tsv"], sep="\t").iloc[0]
            input_path = root / str(source.input_path)
            input_data = ad.read_h5ad(input_path, backed="r")
            output_data = ad.read_h5ad(required["gene_activity.h5ad"], backed="r")
            observation_match = _id_hash(input_data.obs_names) == _id_hash(output_data.obs_names)
            n_observations, n_genes = output_data.shape
            leakage_absent = output_data.uns.get("construction_leakage_inputs") == "none"
            input_data.file.close()
            output_data.file.close()
            contract_passed = int((contract["status"] == "pass").sum())
            artifacts_valid = _artifact_hashes_valid(run_dir, manifest, root, cached_sha)
            inputs_valid = _input_hashes_valid(validation, root, cached_sha)
            boundary_valid = (
                validation.get("status") == "pass"
                and validation.get("dataset_role") == "external_holdout"
                and validation.get("forbidden_inputs_used", []) == []
                and all(validation.get(field) is False for field in FORBIDDEN_ACCESS_FIELDS if field in validation)
                and manifest.get("evaluation_run", False) is False
                and manifest.get("full_benchmark_run", False) is False
                and manifest.get("final_ranking_run", False) is False
            )
            identity_valid = (
                manifest.get("dataset_id", source.dataset_id) == source.dataset_id
                and manifest.get("method_id") == source.method_id
                and validation.get("dataset_id") == source.dataset_id
                and validation.get("method_id") == source.method_id
            )
            run_pass = all(
                (
                    manifest.get("status") == "pass",
                    str(runtime["status"]) == "pass",
                    memory.get("status") == "pass",
                    memory.get("exit_code") == 0,
                    contract_passed == len(contract) == 18,
                    observation_match,
                    n_observations > 0,
                    n_genes > 0,
                    artifacts_valid,
                    inputs_valid,
                    boundary_valid,
                    leakage_absent,
                    identity_valid,
                )
            )
            rows.append(
                {
                    **base,
                    "n_observations": n_observations,
                    "n_output_genes": n_genes,
                    "contract_checks_passed": contract_passed,
                    "contract_checks_total": len(contract),
                    "observation_ids_and_order_match": observation_match,
                    "manifest_artifact_hashes_valid": artifacts_valid,
                    "live_input_hashes_valid": inputs_valid,
                    "construction_boundary_valid": boundary_valid,
                    "construction_leakage_absent": leakage_absent,
                    "holdout_accessed": validation.get("holdout_accessed", True),
                    "runtime_seconds_noncomparative": float(memory["runtime_seconds"]),
                    "peak_aggregate_pss_mb_noncomparative": float(memory["peak_aggregate_pss_mb"]),
                    "peak_aggregate_rss_mb_noncomparative": float(memory["peak_aggregate_rss_mb"]),
                    "process_tree_exit_code": int(memory["exit_code"]),
                    "output_h5ad_sha256": cached_sha(required["gene_activity.h5ad"]),
                    "output_manifest_sha256": cached_sha(required["output_manifest.json"]),
                    "contract_validation_sha256": cached_sha(required["contract_validation.tsv"]),
                    "process_tree_memory_sha256": cached_sha(required["process_tree_memory.json"]),
                    "status": "pass" if run_pass else "fail",
                }
            )
            attempts.append(
                {
                    **base,
                    "attempt_index": 1,
                    "attempt_type": "primary",
                    "attempt_authorized": True,
                    "attempt_executed": True,
                    "started_at_utc": memory.get("started_at_utc"),
                    "finished_at_utc": memory.get("finished_at_utc"),
                    "process_tree_exit_code": memory.get("exit_code"),
                    "retry_attempts": 0,
                    "retained_retry_directories": 0,
                    "outcome_based_retry": False,
                    "output_manifest_sha256": cached_sha(required["output_manifest.json"]),
                    "status": "pass" if run_pass else "fail",
                }
            )
        except Exception as exc:
            rows.append({**base, "aggregation_error": f"{type(exc).__name__}: {exc}", "status": "fail"})
            attempts.append(
                {
                    **base,
                    "attempt_index": 1,
                    "attempt_type": "primary",
                    "attempt_authorized": True,
                    "attempt_executed": True,
                    "retry_attempts": 0,
                    "aggregation_error": f"{type(exc).__name__}: {exc}",
                    "status": "fail",
                }
            )

    summary = pd.DataFrame(rows)
    attempt_manifest = pd.DataFrame(attempts)
    summary_path = output_dir / "run_summary.tsv"
    attempt_path = output_dir / "attempt_manifest.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    attempt_manifest.to_csv(attempt_path, sep="\t", index=False)
    expected = len(plan)
    passed = int((summary["status"] == "pass").sum())
    retry_dirs = [
        path
        for path in (root / "results/v1/g5/h5b_external_method_construction").rglob("attempt_*")
        if path.is_dir()
    ]
    evidence = {
        "schema_version": 1,
        "gate": "G5",
        "milestone": "H5B_external_method_construction",
        "status": "pass" if passed == expected and not retry_dirs else "partial",
        "studies": int(plan["study_id"].nunique()),
        "runs_expected": expected,
        "runs_passed": passed,
        "runs_failed": int((summary["status"] == "fail").sum()),
        "runs_missing": int((summary["status"] == "missing").sum()),
        "annotation_runs": int((summary["run_type"] == "annotation").sum()),
        "software_runs": int((summary["run_type"] == "software").sum()),
        "contract_checks_passed": int(summary["contract_checks_passed"].fillna(0).sum()),
        "contract_checks_total": int(summary["contract_checks_total"].fillna(0).sum()),
        "primary_attempts_executed": int(attempt_manifest["attempt_executed"].fillna(False).sum()),
        "retry_attempts_executed": 0,
        "retained_retry_directories": len(retry_dirs),
        "holdout_inputs_accessed": True,
        "method_execution_run": True,
        "construction_contract_qc_run": True,
        "rna_value_access_run": False,
        "label_or_pathology_access_run": False,
        "holdout_utility_access_run": False,
        "regret_computation_run": False,
        "policy_refit_run": False,
        "GSE279771_accessed": False,
        "full_benchmark_run": False,
        "results_writing_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "performance_outcomes_interpreted": False,
        "resource_measurements_comparative": False,
        "next_milestone_authorized": False,
        "planned_execution_manifest": {"path": _relative(plan_path, root), "sha256": sha256(plan_path)},
        "attempt_manifest": {"path": _relative(attempt_path, root), "sha256": sha256(attempt_path)},
        "run_summary": {"path": _relative(summary_path, root), "sha256": sha256(summary_path)},
        "implementation": {"path": "spatialgabench/g5_h5b_summary.py", "sha256": sha256(Path(__file__))},
    }
    evidence_path = output_dir / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"attempt_manifest": attempt_path, "run_summary": summary_path, "evidence": evidence_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="results/v1/g5/h5b_method_construction_preflight/planned_method_execution_manifest.tsv",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--outdir",
        default="results/v1/g5/h5b_external_method_construction/evidence",
    )
    args = parser.parse_args()
    outputs = summarize(args.plan, args.root, args.outdir)
    print(outputs["evidence"].read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
