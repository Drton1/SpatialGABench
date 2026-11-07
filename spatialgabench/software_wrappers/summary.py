"""Aggregate and verify software-wrapper smoke evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from .common import sha256_file


FAILURE_COLUMNS = [
    "run_id",
    "dataset_id",
    "method_id",
    "software_name",
    "stage",
    "failure_type",
    "message",
    "runtime_seconds",
    "peak_memory_mb",
    "status",
]
FORBIDDEN_FILENAME_TOKENS = {
    "rna",
    "label",
    "cluster",
    "marker",
    "pathology",
    "truth",
}


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _resolve_inside_root(root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes project root: {path}") from exc
    return path


def _verify_output_manifest(root: Path, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for field in ("output_h5ad", "contract_tsv", "runtime_tsv"):
        hash_field = f"{field}_sha256"
        try:
            artifact = _resolve_inside_root(root, manifest.get(field, ""))
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if not artifact.is_file():
            failures.append(f"missing:{field}")
        elif sha256_file(artifact) != manifest.get(hash_field):
            failures.append(f"hash_mismatch:{field}")
    return not failures, failures


def _scan_forbidden_work_files(output_dir: Path) -> list[str]:
    forbidden: list[str] = []
    work_dir = output_dir / "work"
    if not work_dir.is_dir():
        return forbidden
    for path in work_dir.rglob("*"):
        if not path.is_file():
            continue
        tokens = set(path.name.lower().replace("-", "_").replace(".", "_").split("_"))
        if tokens & FORBIDDEN_FILENAME_TOKENS:
            forbidden.append(str(path.relative_to(output_dir)))
    return sorted(forbidden)


def _summarize_run(root: Path, method_key: str, run_config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir = _resolve_inside_root(root, run_config["output_dir"])
    wrapper_config_path = _resolve_inside_root(root, run_config["config"])
    wrapper_config = _load_mapping(wrapper_config_path)
    input_validation_path = output_dir / "input_validation.json"
    output_manifest_path = output_dir / "output_manifest.json"
    metadata_path = output_dir / "method_metadata.yaml"
    failure_path = output_dir / "failure.tsv"
    failures: list[dict[str, Any]] = []

    required = [input_validation_path, output_manifest_path, metadata_path]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        failures.append(
            {
                "run_id": wrapper_config.get("run_id", ""),
                "dataset_id": wrapper_config.get("dataset_id", ""),
                "method_id": wrapper_config.get("method_id", method_key),
                "software_name": wrapper_config.get("software_name", ""),
                "stage": "summary",
                "failure_type": "MissingArtifact",
                "message": ";".join(missing),
                "runtime_seconds": np.nan,
                "peak_memory_mb": np.nan,
                "status": "fail",
            }
        )
        return {"method_id": wrapper_config.get("method_id", method_key), "status": "fail"}, failures

    input_validation = json.loads(input_validation_path.read_text(encoding="utf-8"))
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    metadata = _load_mapping(metadata_path)
    manifest_verified, manifest_failures = _verify_output_manifest(root, output_manifest)
    output_h5ad = _resolve_inside_root(root, output_manifest["output_h5ad"])
    contract_path = _resolve_inside_root(root, output_manifest["contract_tsv"])
    runtime_path = _resolve_inside_root(root, output_manifest["runtime_tsv"])
    output = ad.read_h5ad(output_h5ad)
    contract = pd.read_csv(contract_path, sep="\t")
    runtime = pd.read_csv(runtime_path, sep="\t").iloc[0]

    matrix = output.X.tocsr() if sparse.issparse(output.X) else sparse.csr_matrix(output.X)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    values = matrix.data.astype(np.float64, copy=False)
    classes = set(wrapper_config.get("construction_input_classes", []))
    peak_matrix_authorized = bool({"peak_matrix", "frozen_peak_matrix"} & classes)
    exported_peak_files = {
        "input_peak_matrix_features_by_cells.mtx",
        "input_features.tsv",
    }
    work_dir = output_dir / "work"
    work_files = {path.name for path in work_dir.iterdir() if path.is_file()} if work_dir.is_dir() else set()
    peak_matrix_present = bool(exported_peak_files & work_files)
    forbidden_work_files = _scan_forbidden_work_files(output_dir)
    contract_failures = int((contract["status"] != "pass").sum())
    run_pass = all(
        [
            input_validation.get("status") == "pass",
            output_manifest.get("status") == "pass",
            metadata.get("status") == "pass",
            str(runtime.get("status")) == "pass",
            manifest_verified,
            contract_failures == 0,
            sparse.issparse(output.X),
            bool(values.size),
            bool(np.isfinite(values).all()),
            bool((values >= 0).all()),
            not forbidden_work_files,
            peak_matrix_authorized or not peak_matrix_present,
        ]
    )
    if failure_path.is_file():
        recorded = pd.read_csv(failure_path, sep="\t")
        failures.extend(recorded.to_dict(orient="records"))
        run_pass = False
    if manifest_failures or forbidden_work_files or (peak_matrix_present and not peak_matrix_authorized):
        message = manifest_failures + [f"forbidden_work_file:{value}" for value in forbidden_work_files]
        if peak_matrix_present and not peak_matrix_authorized:
            message.append("unauthorized_peak_matrix_export")
        failures.append(
            {
                "run_id": wrapper_config["run_id"],
                "dataset_id": wrapper_config["dataset_id"],
                "method_id": wrapper_config["method_id"],
                "software_name": wrapper_config["software_name"],
                "stage": "summary",
                "failure_type": "EvidenceValidationFailure",
                "message": ";".join(message),
                "runtime_seconds": runtime.get("runtime_seconds", np.nan),
                "peak_memory_mb": runtime.get("peak_memory_mb", np.nan),
                "status": "fail",
            }
        )
        run_pass = False

    total_values = matrix.shape[0] * matrix.shape[1]
    row = {
        "run_id": wrapper_config["run_id"],
        "dataset_id": wrapper_config["dataset_id"],
        "method_id": wrapper_config["method_id"],
        "software_name": wrapper_config["software_name"],
        "software_version": metadata.get("software_version", "unknown"),
        "input_track": wrapper_config["input_track"],
        "subset_policy": wrapper_config["subset_policy"],
        "comparison_scope": wrapper_config["comparison_scope"],
        "construction_input_classes": ";".join(wrapper_config["construction_input_classes"]),
        "n_obs": output.n_obs,
        "n_genes": output.n_vars,
        "nnz": matrix.nnz,
        "sparsity": 1.0 - (matrix.nnz / total_values),
        "total_activity": float(matrix.sum()),
        "min_nonzero": float(values.min()),
        "max_activity": float(values.max()),
        "runtime_seconds": float(runtime["runtime_seconds"]),
        "peak_memory_mb": float(runtime["peak_memory_mb"]),
        "contract_checks_passed": int((contract["status"] == "pass").sum()),
        "contract_checks_failed": contract_failures,
        "input_validation_status": input_validation.get("status", "missing"),
        "output_manifest_status": output_manifest.get("status", "missing"),
        "manifest_hash_verified": manifest_verified,
        "peak_matrix_present_in_work": peak_matrix_present,
        "peak_matrix_authorized": peak_matrix_authorized,
        "forbidden_work_artifacts": ";".join(forbidden_work_files),
        "status": "pass" if run_pass else "fail",
    }
    return row, failures


def write_software_smoke_summary(
    config_path: str | Path,
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Verify all configured wrapper runs and write aggregate evidence."""

    root = Path(root).resolve()
    config_path = _resolve_inside_root(root, config_path)
    config = _load_mapping(config_path)
    method_order = config.get("method_order", [])
    if not isinstance(method_order, list) or not method_order or len(set(method_order)) != len(method_order):
        raise ValueError("method_order must be a non-empty list of unique config keys")
    output_dir = _resolve_inside_root(root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_manifests: dict[str, dict[str, str]] = {}
    for method_key in method_order:
        run_config = config.get(method_key)
        if not isinstance(run_config, dict):
            raise ValueError(f"Missing method config block: {method_key}")
        row, run_failures = _summarize_run(root, method_key, run_config)
        rows.append(row)
        failures.extend(run_failures)
        manifest_path = _resolve_inside_root(root, run_config["output_dir"]) / "output_manifest.json"
        if manifest_path.is_file():
            run_manifests[row["method_id"]] = {
                "path": str(manifest_path.relative_to(root)),
                "sha256": sha256_file(manifest_path),
            }

    summary_path = output_dir / "software_smoke_summary.tsv"
    failures_path = output_dir / "software_smoke_failures.tsv"
    manifest_path = output_dir / "software_smoke_manifest.json"
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, sep="\t", index=False)
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failures_path, sep="\t", index=False)
    status = "pass" if not failures and bool((summary["status"] == "pass").all()) else "fail"
    manifest = {
        "schema_version": 1,
        "gate": "G3",
        "required_methods": [row["method_id"] for row in rows],
        "run_manifests": run_manifests,
        "artifacts": {
            "summary": {
                "path": str(summary_path.relative_to(root)),
                "sha256": sha256_file(summary_path),
            },
            "failures": {
                "path": str(failures_path.relative_to(root)),
                "sha256": sha256_file(failures_path),
            },
        },
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": sha256_file(config_path),
        "failure_count": len(failures),
        "holdout_accessed": False,
        "construction_evaluation_inputs_used": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "status": status,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "pass":
        raise ValueError(f"Software smoke summary failed; see {failures_path}")
    return {"summary": summary_path, "failures": failures_path, "manifest": manifest_path}
