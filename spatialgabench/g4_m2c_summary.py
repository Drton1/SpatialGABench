"""Independent evidence aggregation for governed G4-M2C construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml


RAW_LSI_FILES = (
    "raw_peak_lsi.h5ad",
    "contract_validation.tsv",
    "input_validation.json",
    "construction_qc.tsv",
    "output_manifest.json",
    "process_tree_memory.json",
)
CICERO_FILES = (
    "wrapper_config.yaml",
    "cicero_subset.h5ad",
    "input_validation.json",
    "gene_activity.h5ad",
    "contract_validation.tsv",
    "runtime.tsv",
    "output_manifest.json",
    "process_tree_memory.json",
)


def _sha256(path: Path) -> str:
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


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def summarize_raw_lsi(
    config_path: str | Path,
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    representation_path = root / config["representation_summary"]
    representations = pd.read_csv(representation_path, sep="\t")
    rows: list[dict[str, Any]] = []

    for source in representations.itertuples(index=False):
        run_dir = root / config["output_root"] / str(source.dataset_id) / str(source.representation_id)
        paths = {name: run_dir / name for name in RAW_LSI_FILES}
        base = {
            "dataset_id": source.dataset_id,
            "study_id": source.study_id,
            "sample_id": source.sample_id,
            "genome_build": source.genome_build,
            "representation_id": source.representation_id,
            "method_id": config["method_id"],
        }
        if not all(path.is_file() for path in paths.values()):
            rows.append({**base, "status": "missing"})
            continue
        try:
            manifest = _load_json(paths["output_manifest.json"])
            validation = _load_json(paths["input_validation.json"])
            memory = _load_json(paths["process_tree_memory.json"])
            contract = pd.read_csv(paths["contract_validation.tsv"], sep="\t")
            qc = pd.read_csv(paths["construction_qc.tsv"], sep="\t").iloc[0]
            input_path = root / str(source.output_path)
            output = ad.read_h5ad(paths["raw_peak_lsi.h5ad"], backed="r")
            input_data = ad.read_h5ad(input_path, backed="r")
            output_obs_hash = _id_hash(output.obs_names)
            input_obs_hash = _id_hash(input_data.obs_names)
            lsi = np.asarray(output.obsm["X_lsi"])
            output_shape = output.shape
            output.file.close()
            input_data.file.close()
            artifact_hashes_valid = all(
                paths[name].is_file() and _sha256(paths[name]) == digest
                for name, digest in manifest.get("artifacts", {}).items()
            ) and set(manifest.get("artifacts", {})) == {
                "raw_peak_lsi.h5ad", "contract_validation.tsv",
                "input_validation.json", "construction_qc.tsv",
            }
            checks_passed = int((contract["status"] == "pass").sum())
            input_hash_valid = (
                input_path.is_file()
                and _sha256(input_path) == str(source.output_sha256)
                and validation.get("input_sha256") == str(source.output_sha256)
            )
            boundaries_valid = (
                validation.get("holdout_accessed") is False
                and validation.get("forbidden_inputs_used") == []
                and manifest.get("evaluation_run") is False
                and manifest.get("full_benchmark_run") is False
                and manifest.get("final_ranking_run") is False
            )
            run_pass = all((
                manifest.get("status") == "pass",
                validation.get("status") == "pass",
                memory.get("status") == "pass",
                memory.get("exit_code") == 0,
                str(qc["status"]) == "pass",
                checks_passed == len(contract) == 18,
                output_shape[0] == int(source.n_observations),
                output_obs_hash == input_obs_hash,
                lsi.shape[0] == int(source.n_observations),
                np.isfinite(lsi).all(),
                artifact_hashes_valid,
                input_hash_valid,
                boundaries_valid,
            ))
            rows.append({
                **base,
                "n_observations": output_shape[0],
                "n_input_features": int(source.n_features),
                "n_components": lsi.shape[1],
                "contract_checks_passed": checks_passed,
                "contract_checks_total": len(contract),
                "observation_ids_and_order_match": output_obs_hash == input_obs_hash,
                "manifest_hashes_valid": artifact_hashes_valid,
                "current_input_hash_valid": input_hash_valid,
                "runtime_seconds_noncomparative": float(memory["runtime_seconds"]),
                "peak_aggregate_pss_mb_noncomparative": float(memory["peak_aggregate_pss_mb"]),
                "peak_aggregate_rss_mb_noncomparative": float(memory["peak_aggregate_rss_mb"]),
                "memory_samples": int(memory["sample_count"]),
                "input_sha256": str(source.output_sha256),
                "output_manifest_sha256": _sha256(paths["output_manifest.json"]),
                "memory_evidence_sha256": _sha256(paths["process_tree_memory.json"]),
                "status": "pass" if run_pass else "fail",
            })
        except Exception as exc:  # Preserve aggregation failure as evidence, not an exception-only outcome.
            rows.append({**base, "aggregation_error": f"{type(exc).__name__}: {exc}", "status": "fail"})

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "g4_m2c_raw_lsi_run_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    expected = len(representations)
    passed = int((summary["status"] == "pass").sum())
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M2C_raw_lsi_representation_control_construction",
        "status": "pass" if passed == expected else "partial",
        "runs_expected": expected,
        "runs_passed": passed,
        "runs_failed": int((summary["status"] == "fail").sum()),
        "runs_missing": int((summary["status"] == "missing").sum()),
        "datasets_expected": int(representations["dataset_id"].nunique()),
        "representations_expected": int(representations["representation_id"].nunique()),
        "contract_checks_passed": int(summary.get("contract_checks_passed", pd.Series(dtype=float)).fillna(0).sum()),
        "contract_checks_total": int(summary.get("contract_checks_total", pd.Series(dtype=float)).fillna(0).sum()),
        "output_type": "cell_embedding",
        "is_gene_activity": False,
        "benchmark_role": "representation_control",
        "run_summary": {"path": _relative(summary_path, root), "sha256": _sha256(summary_path)},
        "config": {"path": _relative(config_path, root), "sha256": _sha256(config_path)},
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "representation_summary": {"path": config["representation_summary"], "sha256": config["representation_summary_sha256"]},
        "implementation": {"path": "spatialgabench/g4_m2c_summary.py", "sha256": _sha256(Path(__file__))},
        "resource_measurements_comparative": False,
        "resource_noncomparative_reason": "construction evidence only; interpretation requires a separately authorized evaluation stage",
        "holdout_accessed": False,
        "construction_evaluation_inputs_used": False,
        "rna_evaluation_run": False,
        "label_evaluation_run": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "performance_outcomes_interpreted": False,
        "next_milestone_authorized": False,
    }
    evidence_path = output_dir / "g4_m2c_raw_lsi_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary_path, "evidence": evidence_path}


def summarize_cicero(
    config_path: str | Path,
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    representation_path = root / config["representation_summary"]
    representations = pd.read_csv(representation_path, sep="\t")
    peaks = representations[representations["representation_id"] == "macs3_summit_501bp"]
    rows: list[dict[str, Any]] = []

    for source in peaks.itertuples(index=False):
        for requested_cells in config["cell_ladder"]:
            eligible = int(requested_cells) <= int(source.n_observations)
            base = {
                "dataset_id": source.dataset_id,
                "study_id": source.study_id,
                "sample_id": source.sample_id,
                "genome_build": source.genome_build,
                "representation_id": "macs3_summit_501bp_cicero_subset",
                "method_id": "cicero_geneactivity",
                "requested_cells": int(requested_cells),
                "available_observations": int(source.n_observations),
                "max_peaks": int(config["max_peaks"]),
                "subset_only": True,
                "performance_pooling_allowed": False,
                "eligible": eligible,
            }
            if not eligible:
                rows.append({
                    **base,
                    "not_applicable_reason": "requested_cells_exceed_available_observations",
                    "included_in_failure_denominator": False,
                    "status": "not_applicable",
                })
                continue
            run_dir = root / config["output_root"] / str(source.dataset_id) / f"cells_{requested_cells}"
            paths = {name: run_dir / name for name in CICERO_FILES}
            if not all(path.is_file() for path in paths.values()):
                rows.append({**base, "included_in_failure_denominator": True, "status": "missing"})
                continue
            try:
                wrapper = _load_yaml(paths["wrapper_config.yaml"])
                validation = _load_json(paths["input_validation.json"])
                manifest = _load_json(paths["output_manifest.json"])
                memory = _load_json(paths["process_tree_memory.json"])
                contract = pd.read_csv(paths["contract_validation.tsv"], sep="\t")
                runtime = pd.read_csv(paths["runtime.tsv"], sep="\t").iloc[0]
                subset = ad.read_h5ad(paths["cicero_subset.h5ad"], backed="r")
                output = ad.read_h5ad(paths["gene_activity.h5ad"], backed="r")
                subset_obs_hash = _id_hash(subset.obs_names)
                output_obs_hash = _id_hash(output.obs_names)
                subset_shape = subset.shape
                output_shape = output.shape
                subset_source_sha = subset.uns.get("source_sha256")
                subset_selection_policy = subset.uns.get("selection_policy")
                output_matrix = output.X.to_memory() if hasattr(output.X, "to_memory") else output.X
                output_values = output_matrix.data if hasattr(output_matrix, "data") else np.asarray(output_matrix)
                output_values_finite = np.isfinite(output_values).all()
                subset.file.close()
                output.file.close()
                manifest_hashes_valid = all(
                    (root / str(manifest[key])).is_file()
                    and _sha256(root / str(manifest[key])) == manifest[f"{key}_sha256"]
                    for key in ("output_h5ad", "contract_tsv", "runtime_tsv")
                )
                current_inputs_valid = all(
                    Path(validation["resolved_paths"][key]).is_file()
                    and _sha256(Path(validation["resolved_paths"][key])) == digest
                    for key, digest in validation["input_sha256"].items()
                )
                source_path = root / str(source.output_path)
                source_hash_valid = (
                    source_path.is_file()
                    and _sha256(source_path) == str(source.output_sha256)
                    and subset_source_sha == str(source.output_sha256)
                )
                checks_passed = int((contract["status"] == "pass").sum())
                input_checks = {
                    row.get("check_id"): row.get("status")
                    for row in validation.get("checks", [])
                    if isinstance(row, dict)
                }
                boundaries_valid = (
                    wrapper.get("subset_policy") == "scalability_subset"
                    and wrapper.get("comparison_scope") == "subset_scalability_construction_only_no_pooling"
                    and validation.get("status") == "pass"
                    and input_checks.get("forbidden_construction_keys_absent") == "pass"
                )
                run_pass = all((
                    manifest.get("status") == "pass",
                    memory.get("status") == "pass",
                    memory.get("exit_code") == 0,
                    str(runtime["status"]) == "pass",
                    checks_passed == len(contract) == 18,
                    subset_shape[0] == output_shape[0] == int(requested_cells),
                    subset_obs_hash == output_obs_hash,
                    output_shape[1] > 0,
                    output_values_finite,
                    manifest_hashes_valid,
                    current_inputs_valid,
                    source_hash_valid,
                    boundaries_valid,
                ))
                rows.append({
                    **base,
                    "n_subset_observations": subset_shape[0],
                    "n_subset_peaks": subset_shape[1],
                    "n_output_genes": output_shape[1],
                    "contract_checks_passed": checks_passed,
                    "contract_checks_total": len(contract),
                    "observation_ids_and_order_match": subset_obs_hash == output_obs_hash,
                    "selection_policy": subset_selection_policy,
                    "manifest_hashes_valid": manifest_hashes_valid,
                    "current_input_hashes_valid": current_inputs_valid,
                    "source_hash_valid": source_hash_valid,
                    "runtime_seconds_noncomparative": float(memory["runtime_seconds"]),
                    "peak_aggregate_pss_mb_noncomparative": float(memory["peak_aggregate_pss_mb"]),
                    "peak_aggregate_rss_mb_noncomparative": float(memory["peak_aggregate_rss_mb"]),
                    "memory_samples": int(memory["sample_count"]),
                    "output_manifest_sha256": _sha256(paths["output_manifest.json"]),
                    "memory_evidence_sha256": _sha256(paths["process_tree_memory.json"]),
                    "included_in_failure_denominator": True,
                    "status": "pass" if run_pass else "fail",
                })
            except Exception as exc:  # Preserve aggregation failure as evidence, not an exception-only outcome.
                rows.append({
                    **base,
                    "aggregation_error": f"{type(exc).__name__}: {exc}",
                    "included_in_failure_denominator": True,
                    "status": "fail",
                })

    summary = pd.DataFrame(rows)
    integer_columns = (
        "n_subset_observations",
        "n_subset_peaks",
        "n_output_genes",
        "contract_checks_passed",
        "contract_checks_total",
        "memory_samples",
    )
    for column in integer_columns:
        if column in summary:
            summary[column] = summary[column].astype("Int64")
    summary_path = output_dir / "g4_m2c_cicero_scalability_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    eligible = summary[summary["eligible"]]
    passed = int((eligible["status"] == "pass").sum())
    expected = len(eligible)
    not_applicable = int((summary["status"] == "not_applicable").sum())
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M2C_cicero_subset_scalability_construction",
        "status": "pass" if passed == expected and not_applicable == len(summary) - expected else "partial",
        "planned_combinations": len(summary),
        "eligible_runs_expected": expected,
        "eligible_runs_passed": passed,
        "eligible_runs_failed": int((eligible["status"] == "fail").sum()),
        "eligible_runs_missing": int((eligible["status"] == "missing").sum()),
        "not_applicable_combinations": not_applicable,
        "not_applicable_in_failure_denominator": False,
        "datasets_expected": len(peaks),
        "cell_ladder": [int(value) for value in config["cell_ladder"]],
        "max_peaks": int(config["max_peaks"]),
        "contract_checks_passed": int(eligible.get("contract_checks_passed", pd.Series(dtype=float)).fillna(0).sum()),
        "contract_checks_total": int(eligible.get("contract_checks_total", pd.Series(dtype=float)).fillna(0).sum()),
        "subset_only": True,
        "performance_pooling_allowed": False,
        "benchmark_role": "subset_scalability_method",
        "run_summary": {"path": _relative(summary_path, root), "sha256": _sha256(summary_path)},
        "config": {"path": _relative(config_path, root), "sha256": _sha256(config_path)},
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "representation_summary": {"path": config["representation_summary"], "sha256": config["representation_summary_sha256"]},
        "implementation": {"path": "spatialgabench/g4_m2c_summary.py", "sha256": _sha256(Path(__file__))},
        "retained_attempts": sorted(
            _relative(path, root)
            for path in (root / config["output_root"]).glob("*/cells_*/attempt_*")
        ),
        "resource_measurements_comparative": False,
        "resource_noncomparative_reason": "construction-only subset ladder; performance comparison and cross-scale pooling are not authorized",
        "holdout_accessed": False,
        "construction_evaluation_inputs_used": False,
        "rna_evaluation_run": False,
        "label_evaluation_run": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "performance_outcomes_interpreted": False,
        "next_milestone_authorized": False,
    }
    evidence_path = output_dir / "g4_m2c_cicero_scalability_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary_path, "evidence": evidence_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="results/v1/g4/m2/evidence")
    parser.add_argument("--raw-config", default="configs/g4_m2c_raw_lsi.yaml")
    parser.add_argument("--cicero-config", default="configs/g4_m2c_cicero.yaml")
    parser.add_argument("--component", choices=("all", "raw-lsi", "cicero"), default="all")
    args = parser.parse_args()
    outputs: dict[str, dict[str, str]] = {}
    if args.component in {"all", "raw-lsi"}:
        raw = summarize_raw_lsi(args.raw_config, args.root, args.outdir)
        outputs["raw_lsi"] = {key: str(value) for key, value in raw.items()}
    if args.component in {"all", "cicero"}:
        cicero = summarize_cicero(args.cicero_config, args.root, args.outdir)
        outputs["cicero"] = {key: str(value) for key, value in cicero.items()}
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
