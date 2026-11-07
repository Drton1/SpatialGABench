"""Frozen full-depth peak rescue experiment for low-depth spatial ATAC."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats
from sklearn.decomposition import PCA

from spatialgabench.nm_paired_depth_accuracy import (
    _evaluate_matrix,
    canonicalize_unique_gene_names,
    id_hash,
    sha256_path,
)
from spatialgabench.nm_spatial_program_evaluation import (
    _align_rows,
    _canonicalize_ga_columns,
    _descriptive_bootstrap,
    _load_rna,
    morans_i,
    normalize_total_log1p,
    safe_correlation,
    symmetric_knn_graph,
)
from spatialgabench.representations.matched import build_matched_representations


P_ROLE = "de_novo_recalled_peak"
PF_ROLE = "frozen_full_depth_peak"
B_ROLE = "fixed_bin_25kb"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _condition_id(dataset_id: str, fraction: float, seed: int) -> str:
    return f"{dataset_id}__d{int(round(100 * fraction)):03d}__s{seed}"


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "pass"})


def _condition_rows(config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, study in config["studies"].items():
        depth_root = root / study["depth_root"]
        full_peak = root / study["full_peak_bed"]
        full_matrix = root / study["full_peak_matrix"]
        if not full_peak.is_file() or not full_matrix.is_file():
            raise ValueError(f"TA2 full-depth peak anchor missing: {dataset_id}")
        full_data = ad.read_h5ad(full_matrix, backed="r")
        full_feature_hash = id_hash(full_data.var_names)
        full_features = full_data.n_vars
        full_data.file.close()
        for fraction in config["depth_fractions"]:
            for seed in config["thinning_seeds"]:
                condition_id = _condition_id(dataset_id, float(fraction), int(seed))
                base = depth_root / "conditions" / condition_id
                fragments = base / "fragments/eligible.fragments.tsv.gz"
                fragment_audit = base / "fragments/thinning_audit.json"
                de_novo = base / "representations" / f"{condition_id}__macs3_summit_501bp.h5ad"
                fixed_bin = base / "representations" / f"{condition_id}__fixed_bin_25kb.h5ad"
                for path in (fragments, fragment_audit, fixed_bin):
                    if not path.is_file():
                        raise ValueError(f"TA2 frozen condition input missing: {path}")
                audit = _json(fragment_audit)
                if audit.get("status") != "pass" or audit.get("output_fragments_sha256") != sha256_path(fragments):
                    raise ValueError(f"TA2 fragment audit mismatch: {condition_id}")
                fixed = ad.read_h5ad(fixed_bin, backed="r")
                observations = fixed.n_obs
                observation_hash = id_hash(fixed.obs_names)
                fixed.file.close()
                rows.append({
                    "condition_id": condition_id,
                    "dataset_id": dataset_id,
                    "study_id": study["study_id"],
                    "study_family": study["study_family"],
                    "sample_id": study["sample_id"],
                    "species": study["species"],
                    "genome_build": study["genome_build"],
                    "depth_fraction": float(fraction),
                    "seed": int(seed),
                    "fragments": str(fragments.relative_to(root)),
                    "fragments_sha256": audit["output_fragments_sha256"],
                    "fragment_audit": str(fragment_audit.relative_to(root)),
                    "fragment_audit_sha256": sha256_path(fragment_audit),
                    "positions": study["positions"],
                    "positions_sha256": sha256_path(root / study["positions"]),
                    "project_config": study["project_config"],
                    "full_depth_peak_bed": str(full_peak.relative_to(root)),
                    "full_depth_peak_bed_sha256": sha256_path(full_peak),
                    "full_depth_peak_matrix": str(full_matrix.relative_to(root)),
                    "full_depth_peak_matrix_sha256": sha256_path(full_matrix),
                    "full_depth_feature_id_order_sha256": full_feature_hash,
                    "full_depth_features": full_features,
                    "de_novo_peak_matrix": str(de_novo.relative_to(root)) if de_novo.is_file() else "",
                    "de_novo_peak_matrix_sha256": sha256_path(de_novo) if de_novo.is_file() else "",
                    "fixed_bin_matrix": str(fixed_bin.relative_to(root)),
                    "fixed_bin_matrix_sha256": sha256_path(fixed_bin),
                    "observations": observations,
                    "observation_id_order_sha256": observation_hash,
                })
    frame = pd.DataFrame(rows).sort_values(["dataset_id", "depth_fraction", "seed"], ascending=[True, False, True], kind="mergesort")
    if len(frame) != int(config["expected_conditions"]):
        raise ValueError("TA2 condition count changed")
    return frame.reset_index(drop=True)


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _yaml(config_path)
    for field in ("protocol", "amendment", "ta1_completion", "ta1_metrics", "ta1_manifest", "rna_source_config", "nu4_preflight"):
        if not (root / config[field]).is_file():
            raise ValueError(f"TA2 frozen source missing: {field}")
    for reference in config["references"]["mm10"].values():
        if sha256_path(root / reference["path"]) != reference["sha256"]:
            raise ValueError("TA2 mm10 reference checksum mismatch")
    conditions = _condition_rows(config, root)
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    condition_path = output / "condition_input_manifest.tsv"
    conditions.to_csv(condition_path, sep="\t", index=False)
    dependencies = [
        "spatialgabench/nm_frozen_peak_rescue.py",
        "spatialgabench/representations/matched.py",
        "adapters/g4_m2_annotation_driver.py",
        "spatialgabench/nm_paired_depth_accuracy.py",
        "spatialgabench/nm_spatial_program_evaluation.py",
        "spatialgabench/g4_m3b_rna.py",
        "spatialgabench/g4_m3h_rna_replication.py",
        "spatialgabench/g5_h5c_evaluation.py",
        "spatialgabench/full_benchmark_gse279771.py",
    ]
    authorization = {
        "schema_version": 1,
        "authorization": "NM_TA2_FROZEN_PEAK_REPRESENTATION_CONSTRUCTION",
        "milestone": "TA2_frozen_peak_representation_construction",
        "authorization_date": "2026-08-01",
        "status": "authorized",
        "representation_construction_authorized": True,
        "method_execution_authorized": False,
        "rna_expression_value_access_authorized": False,
        "labels_clusters_pathology_access_authorized": False,
        "parameter_tuning_authorized": False,
        "policy_refit_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "protocol": {"path": config["protocol"], "sha256": sha256_path(root / config["protocol"])},
        "amendment": {"path": config["amendment"], "sha256": sha256_path(root / config["amendment"])},
        "condition_input_manifest": {"path": str(condition_path.relative_to(root)), "sha256": sha256_path(condition_path)},
        "dependencies": [{"path": path, "sha256": sha256_path(root / path)} for path in dependencies],
    }
    authorization_path = root / "governance/authorizations/NM_TA2_FROZEN_PEAK_REPRESENTATION_CONSTRUCTION_AUTHORIZATION_2026-08-01.json"
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "milestone": "TA2_frozen_peak_rescue_preflight",
        "status": "pass",
        "conditions": len(conditions),
        "study_families": conditions.study_family.nunique(),
        "rna_expression_values_read": False,
        "method_outcomes_read_for_rescue_design": True,
        "post_TA1_mechanism_experiment": True,
        "construction_run": False,
        "method_run": False,
        "condition_input_manifest": authorization["condition_input_manifest"],
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
    }
    evidence_path = output / "preflight_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def _validate_authorization(config: dict[str, Any], config_path: Path, authorization_path: Path, root: Path, milestone: str) -> dict[str, Any]:
    authorization = _json(authorization_path)
    if authorization.get("status") != "authorized" or authorization.get("milestone") != milestone:
        raise PermissionError(f"TA2 authorization is invalid: {milestone}")
    if authorization["config"]["sha256"] != sha256_path(config_path):
        raise ValueError("TA2 config changed after authorization")
    for record in authorization.get("dependencies", []):
        if sha256_path(root / record["path"]) != record["sha256"]:
            raise ValueError(f"TA2 dependency changed: {record['path']}")
    return authorization


def _build_one(config: dict[str, Any], row: Any, root: Path) -> dict[str, Any]:
    output = root / config["output_root"] / "conditions" / row.condition_id
    final_path = output / f"{row.condition_id}__{config['representation_id']}.h5ad"
    contract_path = output / "representation_contract.json"
    if final_path.is_file() and contract_path.is_file():
        contract = _json(contract_path)
        if contract.get("status") == "pass" and contract.get("output_sha256") == sha256_path(final_path):
            return contract
        raise ValueError(f"TA2 existing representation is invalid: {row.condition_id}")
    raw_dir = output / "raw_builder"
    result = build_matched_representations(
        project_config=root / row.project_config,
        study_id=row.study_id,
        dataset_id=row.condition_id,
        sample_id=row.sample_id,
        fragments_path=root / row.fragments,
        positions_path=root / row.positions,
        chrom_sizes_path=root / config["references"][row.genome_build]["chrom_sizes"]["path"],
        peaks_path=root / row.full_depth_peak_bed,
        genome_build=row.genome_build,
        bin_sizes=[],
        output_dir=raw_dir,
        fragment_filter_audit_path=root / row.fragment_audit,
    )
    raw_path = result["macs3_summit_501bp"]
    data = ad.read_h5ad(raw_path)
    full = ad.read_h5ad(root / row.full_depth_peak_matrix, backed="r")
    fixed = ad.read_h5ad(root / row.fixed_bin_matrix, backed="r")
    feature_match = id_hash(data.var_names) == row.full_depth_feature_id_order_sha256 == id_hash(full.var_names)
    observation_match = id_hash(data.obs_names) == row.observation_id_order_sha256 == id_hash(fixed.obs_names)
    full.file.close(); fixed.file.close()
    if not feature_match or not observation_match or data.n_vars != int(row.full_depth_features):
        raise ValueError(f"TA2 frozen feature/observation contract failed: {row.condition_id}")
    data.var["representation_id"] = config["representation_id"]
    data.uns["representation_id"] = config["representation_id"]
    data.uns["representation_source"] = "full_depth_frozen_peak_set_counted_on_existing_thinned_fragments"
    data.uns["representation_role"] = PF_ROLE
    data.uns["input_track"] = "P"
    data.uns["comparison_scope"] = "within_study_depth_matched_frozen_peak_rescue"
    data.uns["full_depth_peak_bed"] = row.full_depth_peak_bed
    data.uns["full_depth_peak_bed_sha256"] = row.full_depth_peak_bed_sha256
    data.uns["construction_leakage_inputs"] = "none"
    output.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(final_path, compression="gzip")
    matrix = data.X.tocsr() if sparse.issparse(data.X) else sparse.csr_matrix(data.X)
    contract = {
        "schema_version": 1,
        "status": "pass",
        "condition_id": row.condition_id,
        "dataset_id": row.dataset_id,
        "study_id": row.study_id,
        "study_family": row.study_family,
        "sample_id": row.sample_id,
        "species": row.species,
        "genome_build": row.genome_build,
        "depth_fraction": float(row.depth_fraction),
        "seed": int(row.seed),
        "representation_id": config["representation_id"],
        "representation_role": PF_ROLE,
        "n_observations": data.n_obs,
        "n_features": data.n_vars,
        "matrix_nnz": int(matrix.nnz),
        "matrix_total_counts": float(matrix.sum()),
        "zero_output_observations": int(np.sum(np.asarray(matrix.sum(axis=1)).ravel() == 0)),
        "feature_id_order_match_full_depth": feature_match,
        "observation_id_order_match_existing_condition": observation_match,
        "source_fragments": row.fragments,
        "source_fragments_sha256": row.fragments_sha256,
        "full_depth_peak_bed": row.full_depth_peak_bed,
        "full_depth_peak_bed_sha256": row.full_depth_peak_bed_sha256,
        "output_path": str(final_path.relative_to(root)),
        "output_sha256": sha256_path(final_path),
        "rna_labels_pathology_accessed": False,
    }
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def build(config_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path, authorization_path = Path(root).resolve(), Path(config_path).resolve(), Path(authorization_path).resolve()
    config = _yaml(config_path)
    authorization = _validate_authorization(config, config_path, authorization_path, root, "TA2_frozen_peak_representation_construction")
    if authorization.get("representation_construction_authorized") is not True or authorization.get("rna_expression_value_access_authorized") is not False:
        raise PermissionError("TA2 representation construction boundary is invalid")
    conditions = pd.read_csv(root / authorization["condition_input_manifest"]["path"], sep="\t")
    with ThreadPoolExecutor(max_workers=2) as executor:
        contracts = list(executor.map(lambda row: _build_one(config, row, root), conditions.itertuples(index=False)))
    summary = pd.DataFrame(contracts).sort_values(["dataset_id", "depth_fraction", "seed"], ascending=[True, False, True], kind="mergesort")
    output = root / config["output_root"]
    summary_path = output / "representation_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "milestone": "TA2_frozen_peak_representation_construction",
        "status": "pass" if len(summary) == int(config["expected_conditions"]) and summary.status.eq("pass").all() else "fail",
        "representations": len(summary), "study_families": summary.study_family.nunique(),
        "exact_full_depth_feature_order_contracts": int(summary.feature_id_order_match_full_depth.sum()),
        "exact_observation_order_contracts": int(summary.observation_id_order_match_existing_condition.sum()),
        "rna_labels_pathology_accessed": False, "method_run": False,
        "representation_summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
    }
    evidence_path = output / "representation_completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if evidence["status"] != "pass":
        raise ValueError("TA2 frozen peak representation construction failed")
    return evidence_path


def freeze_methods(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _yaml(config_path)
    output = root / config["output_root"]
    representation_evidence_path = output / "representation_completion_evidence.json"
    representation_evidence = _json(representation_evidence_path)
    if representation_evidence.get("status") != "pass":
        raise ValueError("TA2 representations are incomplete")
    source_summary_path = root / representation_evidence["representation_summary"]["path"]
    if sha256_path(source_summary_path) != representation_evidence["representation_summary"]["sha256"]:
        raise ValueError("TA2 representation summary changed after construction")
    source_summary = pd.read_csv(source_summary_path, sep="\t")
    summary = source_summary.copy()
    summary.insert(summary.columns.get_loc("dataset_id") + 1, "source_dataset_id", summary["dataset_id"])
    summary["dataset_id"] = summary["condition_id"]
    summary_path = output / "driver_representation_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    rows = []
    for source in summary.itertuples(index=False):
        for method in config["methods"]:
            rows.append({
                "condition_id": source.condition_id, "dataset_id": source.condition_id,
                "source_dataset_id": source.source_dataset_id,
                "study_id": source.study_id, "study_family": source.study_family,
                "genome_build": source.genome_build, "depth_fraction": source.depth_fraction,
                "seed": source.seed, "representation_id": config["representation_id"],
                "method_id": method, "run_type": "annotation",
                "input_path": source.output_path, "input_sha256": source.output_sha256,
                "output_dir": f"{config['output_root']}/methods/{source.condition_id}/{config['representation_id']}/{method}",
            })
    plan = pd.DataFrame(rows)
    if len(plan) != int(config["expected_method_outputs"]):
        raise ValueError("TA2 method plan count changed")
    plan_path = output / "method_execution_manifest.tsv"
    plan.to_csv(plan_path, sep="\t", index=False)
    authorization = {
        "schema_version": 1, "authorization": "NM_TA2_FROZEN_PEAK_METHOD_CONSTRUCTION",
        "milestone": "TA2_frozen_peak_method_construction", "authorization_date": "2026-08-01",
        "status": "authorized", "method_execution_authorized": True,
        "evaluation_tasks_authorized": False, "rna_value_access_authorized": False,
        "label_or_pathology_value_access_authorized": False, "holdout_utility_access_authorized": False,
        "regret_computation_authorized": False, "policy_refit_authorized": False,
        "universal_ranking_authorized": False, "final_ranking_authorized": False,
        "authorized_methods": config["methods"], "authorized_method_panel": config["methods"],
        "authorized_representations": [config["representation_id"]],
        "authorized_studies": sorted(plan.study_id.unique()),
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "protocol": {"path": config["protocol"], "sha256": sha256_path(root / config["protocol"])},
        "representation_completion": {"path": str(representation_evidence_path.relative_to(root)), "sha256": sha256_path(representation_evidence_path)},
        "representation_summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
        "planned_execution_manifest": {"path": str(plan_path.relative_to(root)), "sha256": sha256_path(plan_path)},
        "dependencies": [{"path": path, "sha256": sha256_path(root / path)} for path in ["spatialgabench/nm_frozen_peak_rescue.py", "adapters/g4_m2_annotation_driver.py"]],
    }
    authorization_path = root / "governance/authorizations/NM_TA2_FROZEN_PEAK_METHOD_CONSTRUCTION_AUTHORIZATION_2026-08-01.json"
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    references = {"mm10": {name: value for name, value in config["references"]["mm10"].items() if name in {"genes", "promoters", "gene_body", "tss"}}}
    common = {
        "schema_version": 1, "gate": "NM_TRUTH_ANCHOR", "milestone": "TA2_frozen_peak_method_construction",
        "authorization": str(authorization_path.relative_to(root)), "authorization_sha256": sha256_path(authorization_path),
        "authorization_milestone": "TA2_frozen_peak_method_construction",
        "protocol": config["protocol"], "protocol_sha256": sha256_path(root / config["protocol"]),
        "representation_summary": str(summary_path.relative_to(root)), "representation_summary_sha256": sha256_path(summary_path),
        "planned_execution_manifest": str(plan_path.relative_to(root)), "planned_execution_manifest_sha256": sha256_path(plan_path),
        "frozen_artifacts": ["authorization", "protocol", "representation_summary", "planned_execution_manifest"],
        "output_root": f"{config['output_root']}/methods", "run_id_prefix": "nm_ta2",
        "input_validation_scope": "TA2_frozen_peak_accessibility_only", "driver_stage": "TA2_frozen_peak_annotation",
        "comparison_scope": "within_condition_full_depth_frozen_peak_rescue", "subset_policy": "full",
        "forbidden_holdout_accessions": [], "forbidden_construction_inputs": ["rna", "labels", "clusters", "pathology", "simulation_truth", "method_outcomes"],
        "methods": config["methods"], "parameters": config["parameters"], "references": references,
    }
    dev_config = {**common, "dataset_role": "development"}
    ext_config = {**common, "dataset_role": "external_holdout"}
    dev_path, ext_path = root / "configs/nm_frozen_peak_rescue_methods_development.yaml", root / "configs/nm_frozen_peak_rescue_methods_external.yaml"
    dev_path.write_text(yaml.safe_dump(dev_config, sort_keys=False), encoding="utf-8")
    ext_path.write_text(yaml.safe_dump(ext_config, sort_keys=False), encoding="utf-8")
    evidence = {
        "schema_version": 1, "milestone": "TA2_frozen_peak_method_setup", "status": "pass",
        "planned_outputs": len(plan), "rna_expression_values_read": False,
        "method_execution_run": False,
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "plan": authorization["planned_execution_manifest"],
        "development_config": {"path": str(dev_path.relative_to(root)), "sha256": sha256_path(dev_path)},
        "external_config": {"path": str(ext_path.relative_to(root)), "sha256": sha256_path(ext_path)},
    }
    evidence_path = output / "method_setup_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def _run_method_row(row: Any, setup: dict[str, Any], config: dict[str, Any], root: Path) -> dict[str, Any]:
    run_dir = root / row.output_dir
    manifest = run_dir / "output_manifest.json"
    failure = run_dir / "failure.tsv"
    if manifest.is_file() and _json(manifest).get("status") == "pass":
        return {**row._asdict(), "exit_code": 0, "outcome": "existing_success"}
    if failure.is_file():
        return {**row._asdict(), "exit_code": 1, "outcome": "retained_failure"}
    config_record = setup["external_config"] if row.source_dataset_id == "gse308623_p2s3" else setup["development_config"]
    command = [
        sys.executable, str(root / "containers/tools/run_with_memory_tracking.py"),
        "--output", str(run_dir / "process_tree_memory.json"), "--sample-interval", "0.05", "--",
        sys.executable, str(root / "adapters/g4_m2_annotation_driver.py"),
        "--config", str(root / config_record["path"]), "--dataset", row.condition_id,
        "--representation", row.representation_id, "--method", row.method_id, "--root", str(root),
    ]
    environment = dict(os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "execution.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    if code != 0 and not failure.is_file():
        pd.DataFrame([{"dataset_id": row.condition_id, "representation_id": row.representation_id, "method_id": row.method_id, "stage": "method_process", "failure_type": "ProcessExit", "message": f"exit code {code}", "status": "fail"}]).to_csv(failure, sep="\t", index=False)
    return {**row._asdict(), "exit_code": code, "outcome": "success" if code == 0 else "structured_failure"}


def run_methods(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _yaml(config_path)
    setup_path = root / config["output_root"] / "method_setup_evidence.json"
    setup = _json(setup_path)
    if setup.get("status") != "pass":
        raise ValueError("TA2 method setup is invalid")
    for field in ("authorization", "plan", "development_config", "external_config"):
        record = setup[field]
        if sha256_path(root / record["path"]) != record["sha256"]:
            raise ValueError(f"TA2 method setup changed: {field}")
    plan = pd.read_csv(root / setup["plan"]["path"], sep="\t")
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda row: _run_method_row(row, setup, config, root), plan.itertuples(index=False)))
    ledger = pd.DataFrame(rows)
    output = root / config["output_root"]
    ledger_path = output / "method_execution_ledger.tsv"
    ledger.to_csv(ledger_path, sep="\t", index=False)
    summaries = []
    for row in plan.itertuples(index=False):
        run_dir = root / row.output_dir
        manifest_path = run_dir / "output_manifest.json"
        success = manifest_path.is_file() and _json(manifest_path).get("status") == "pass"
        summaries.append({**row._asdict(), "success": success, "output_path": str((run_dir / "gene_activity.h5ad").relative_to(root)) if success else "", "output_sha256": sha256_path(run_dir / "gene_activity.h5ad") if success else "", "failure_record": str((run_dir / "failure.tsv").relative_to(root)) if not success and (run_dir / "failure.tsv").is_file() else ""})
    summary = pd.DataFrame(summaries)
    summary_path = output / "method_run_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "milestone": "TA2_frozen_peak_method_construction",
        "status": "pass" if len(summary) == int(config["expected_method_outputs"]) else "fail",
        "planned_outputs": len(summary), "successful_outputs": int(summary.success.sum()),
        "structured_failures": int((~summary.success).sum()), "retry_attempts": 0,
        "rna_labels_pathology_accessed": False, "parameter_tuning_run": False,
        "method_run_summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
        "execution_ledger": {"path": str(ledger_path.relative_to(root)), "sha256": sha256_path(ledger_path)},
        "setup": {"path": str(setup_path.relative_to(root)), "sha256": sha256_path(setup_path)},
    }
    evidence_path = output / "method_completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def freeze_evaluation(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _yaml(config_path)
    output = root / config["output_root"]
    method_evidence_path = output / "method_completion_evidence.json"
    method_evidence = _json(method_evidence_path)
    if method_evidence.get("status") != "pass" or method_evidence.get("successful_outputs") != int(config["expected_method_outputs"]):
        raise ValueError("TA2 requires all frozen-peak annotation outputs")
    new_methods = pd.read_csv(root / method_evidence["method_run_summary"]["path"], sep="\t")
    new_methods["representation_role"] = PF_ROLE
    new_methods["driver_dataset_id"] = new_methods["dataset_id"]
    new_methods["dataset_id"] = new_methods["source_dataset_id"]
    ta1_manifest = pd.read_csv(root / config["ta1_manifest"], sep="\t")
    existing = ta1_manifest.loc[
        ta1_manifest.dataset_id.isin(config["studies"])
        & ta1_manifest.depth_fraction.lt(1.0)
        & ta1_manifest.method_id.isin(config["methods"])
        & ta1_manifest.representation_id.isin(["macs3_summit_501bp", "fixed_bin_25kb"])
    ].copy()
    existing["representation_role"] = np.where(existing.representation_id.eq("fixed_bin_25kb"), B_ROLE, P_ROLE)
    existing = existing.rename(columns={"structural_success": "success"})
    common_columns = ["condition_id", "dataset_id", "study_id", "study_family", "genome_build", "depth_fraction", "seed", "representation_id", "representation_role", "method_id", "success", "output_path", "output_sha256"]
    manifest = pd.concat([existing[common_columns], new_methods[common_columns]], ignore_index=True)
    manifest["success"] = _as_bool(manifest["success"])
    success_manifest = manifest.loc[manifest.success].copy()
    if len(manifest) != 270 or len(success_manifest) != int(config["expected_spatial_contracts"]):
        raise ValueError(f"TA2 evaluation panel changed: {len(manifest)}/{len(success_manifest)}")
    for row in success_manifest.itertuples(index=False):
        if sha256_path(root / row.output_path) != row.output_sha256:
            raise ValueError(f"TA2 evaluation output changed: {row.output_path}")
    manifest_path = output / "evaluation_method_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    rna_config = _yaml(root / config["rna_source_config"])
    identity_rows = []
    nu4 = _json(root / config["nu4_preflight"])
    for dataset_id in config["studies"]:
        sample = rna_config["samples"][dataset_id]
        for field in ("rna_matrix", "positions", "gene_annotation"):
            record = sample[field]
            if sha256_path(root / record["path"]) != record["sha256"]:
                raise ValueError(f"TA2 evaluation identity changed: {dataset_id}/{field}")
            identity_rows.append({"dataset_id": dataset_id, "artifact_class": field, "path": record["path"], "sha256": record["sha256"], "rna_expression_values_read": False})
        fold = nu4["fold_assignments"][dataset_id]
        if sha256_path(root / fold["path"]) != fold["sha256"]:
            raise ValueError("TA2 spatial folds changed")
        identity_rows.append({"dataset_id": dataset_id, "artifact_class": "spatial_folds", "path": fold["path"], "sha256": fold["sha256"], "rna_expression_values_read": False})
    identity_path = output / "evaluation_data_identity_manifest.tsv"
    pd.DataFrame(identity_rows).to_csv(identity_path, sep="\t", index=False)
    dependencies = [
        "spatialgabench/nm_frozen_peak_rescue.py",
        "spatialgabench/nm_paired_depth_accuracy.py",
        "spatialgabench/nm_spatial_program_evaluation.py",
        "spatialgabench/g4_m3b_rna.py",
        "spatialgabench/g4_m3h_rna_replication.py",
        "spatialgabench/g5_h5c_evaluation.py",
        "spatialgabench/full_benchmark_gse279771.py",
    ]
    authorization = {
        "schema_version": 1, "authorization": "NM_TA2_FROZEN_PEAK_RESCUE_EVALUATION",
        "milestone": "TA2_frozen_peak_rescue_evaluation", "authorization_date": "2026-08-01",
        "status": "authorized", "rna_expression_value_access_authorized": True,
        "paired_rna_evaluation_authorized": True, "spatial_program_evaluation_authorized": True,
        "construction_or_method_rerun_authorized": False, "parameter_tuning_authorized": False,
        "labels_clusters_pathology_access_authorized": False, "policy_refit_authorized": False,
        "universal_ranking_authorized": False, "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "protocol": {"path": config["protocol"], "sha256": sha256_path(root / config["protocol"])},
        "method_completion": {"path": str(method_evidence_path.relative_to(root)), "sha256": sha256_path(method_evidence_path)},
        "evaluation_method_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_path(manifest_path)},
        "data_identity_manifest": {"path": str(identity_path.relative_to(root)), "sha256": sha256_path(identity_path)},
        "ta1_completion": {"path": config["ta1_completion"], "sha256": sha256_path(root / config["ta1_completion"])},
        "ta1_metrics": {"path": config["ta1_metrics"], "sha256": sha256_path(root / config["ta1_metrics"])},
        "nu4_preflight": {"path": config["nu4_preflight"], "sha256": sha256_path(root / config["nu4_preflight"])},
        "dependencies": [{"path": path, "sha256": sha256_path(root / path)} for path in dependencies],
    }
    authorization_path = root / "governance/authorizations/NM_TA2_FROZEN_PEAK_RESCUE_EVALUATION_AUTHORIZATION_2026-08-01.json"
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": 1, "milestone": "TA2_frozen_peak_rescue_evaluation_preflight", "status": "pass",
        "planned_contracts": len(manifest), "structurally_successful_contracts": len(success_manifest),
        "new_frozen_peak_contracts": len(new_methods), "rna_expression_values_read": False,
        "evaluation_run": False,
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "method_manifest": authorization["evaluation_method_manifest"], "data_identity_manifest": authorization["data_identity_manifest"],
    }
    evidence_path = output / "evaluation_preflight_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def _validate_evaluation(config: dict[str, Any], config_path: Path, preflight_path: Path, authorization_path: Path, root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    preflight, authorization = _json(preflight_path), _json(authorization_path)
    if preflight.get("status") != "pass" or preflight.get("rna_expression_values_read") is not False:
        raise ValueError("TA2 evaluation preflight is invalid")
    required = {"status": "authorized", "milestone": "TA2_frozen_peak_rescue_evaluation", "rna_expression_value_access_authorized": True, "paired_rna_evaluation_authorized": True, "spatial_program_evaluation_authorized": True, "construction_or_method_rerun_authorized": False, "parameter_tuning_authorized": False, "labels_clusters_pathology_access_authorized": False, "policy_refit_authorized": False, "universal_ranking_authorized": False, "final_ranking_authorized": False}
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise PermissionError(f"TA2 evaluation boundary failed: {field}")
    bindings = {"config": config_path, "protocol": root / config["protocol"], "method_completion": root / authorization["method_completion"]["path"], "evaluation_method_manifest": root / authorization["evaluation_method_manifest"]["path"], "data_identity_manifest": root / authorization["data_identity_manifest"]["path"], "ta1_completion": root / config["ta1_completion"], "ta1_metrics": root / config["ta1_metrics"], "nu4_preflight": root / config["nu4_preflight"]}
    for name, path in bindings.items():
        if authorization[name]["sha256"] != sha256_path(path):
            raise ValueError(f"TA2 evaluation binding changed: {name}")
    for record in authorization["dependencies"]:
        if sha256_path(root / record["path"]) != record["sha256"]:
            raise ValueError(f"TA2 evaluation dependency changed: {record['path']}")
    manifest = pd.read_csv(bindings["evaluation_method_manifest"], sep="\t")
    return authorization, manifest


def _fit_spatial_models(sample: dict[str, Any], fold_table: pd.DataFrame, rna_counts: sparse.csr_matrix, rna_gene_ids: np.ndarray, protein_coding: np.ndarray, rna_normalized: sparse.csr_matrix, settings: dict[str, Any], spatial: dict[str, Any]) -> list[dict[str, Any]]:
    models = []
    for fold in range(5):
        test = np.flatnonzero(fold_table.fold.to_numpy() == fold)
        train = np.flatnonzero(fold_table.fold.to_numpy() != fold)
        detection = np.asarray((rna_counts[train, :] > 0).sum(axis=0)).ravel()
        threshold = math.ceil(float(settings["minimum_detected_training_fraction"]) * len(train))
        eligible = np.flatnonzero(protein_coding & (detection >= threshold))
        pca = PCA(n_components=int(settings["components"]), svd_solver=str(settings["svd_solver"]), random_state=int(settings["random_seed"])).fit(rna_normalized[train, :][:, eligible].toarray().astype(np.float32, copy=False))
        rna_scores = pca.transform(rna_normalized[test, :][:, eligible].toarray().astype(np.float32, copy=False))
        graph = symmetric_knn_graph(fold_table.loc[test, ["row_coordinate", "column_coordinate"]].to_numpy(dtype=float), int(spatial["neighbors"]))
        models.append({"fold": fold, "test": test, "eligible_ids": rna_gene_ids[eligible], "components": pca.components_.astype(float), "mean": pca.mean_.astype(float), "rna_scores": rna_scores.astype(float), "rna_moran": np.asarray([morans_i(rna_scores[:, c], graph) for c in range(rna_scores.shape[1])]), "graph": graph})
    return models


def _spatial_contract(row: Any, models: list[dict[str, Any]], barcodes: list[str], unique_names: dict[str, str], target_sum: float, root: Path) -> pd.DataFrame:
    activity = ad.read_h5ad(root / row.output_path)
    selected = activity.obs_names.astype(str).get_indexer(barcodes)
    if np.any(selected < 0):
        raise ValueError(f"TA2 spatial observations mismatch: {row.output_path}")
    raw = activity.X.tocsr() if sparse.issparse(activity.X) else sparse.csr_matrix(activity.X)
    normalized = normalize_total_log1p(raw[selected, :], target_sum)
    canonical, gene_ids, _ = _canonicalize_ga_columns(normalized, activity, unique_names)
    index = {gene_id: i for i, gene_id in enumerate(gene_ids)}
    rows = []
    for model in models:
        mapped = np.asarray([index.get(str(gene_id), -1) for gene_id in model["eligible_ids"]], dtype=int)
        present = np.flatnonzero(mapped >= 0)
        test = model["test"]
        scores = np.zeros((len(test), model["components"].shape[0]), dtype=float)
        if len(present):
            scores = np.asarray(canonical[test, :][:, mapped[present]] @ model["components"][:, present].T)
        scores -= model["mean"] @ model["components"].T
        for component in range(scores.shape[1]):
            rna_score, ga_score = model["rna_scores"][:, component], scores[:, component]
            ga_moran, rna_moran = morans_i(ga_score, model["graph"]), float(model["rna_moran"][component])
            rows.append({"condition_id": row.condition_id, "dataset_id": row.dataset_id, "study_id": row.study_id, "study_family": row.study_family, "genome_build": row.genome_build, "depth_fraction": row.depth_fraction, "seed": row.seed, "representation_id": row.representation_id, "representation_role": row.representation_role, "method_id": row.method_id, "fold": model["fold"], "component": component + 1, "activity_present_gene_fraction": len(present) / len(mapped), "heldout_component_spearman": safe_correlation(rna_score, ga_score, "spearman"), "heldout_component_pearson": safe_correlation(rna_score, ga_score, "pearson"), "rna_morans_i": rna_moran, "activity_morans_i": ga_moran, "absolute_morans_i_difference": abs(rna_moran - ga_moran), "status": "pass"})
    return pd.DataFrame(rows)


def evaluate(config_path: str | Path, preflight_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    preflight_path, authorization_path = Path(preflight_path).resolve(), Path(authorization_path).resolve()
    config = _yaml(config_path)
    _, manifest = _validate_evaluation(config, config_path, preflight_path, authorization_path, root)
    rna_config = _yaml(root / config["rna_source_config"])
    nu4 = _json(root / config["nu4_preflight"])
    ta1 = pd.read_csv(root / config["ta1_metrics"], sep="\t")
    new_rna_rows, gene_frames, spatial_frames = [], [], []
    for dataset_id in config["studies"]:
        sample = rna_config["samples"][dataset_id]
        fold_table = pd.read_csv(root / nu4["fold_assignments"][dataset_id]["path"], sep="\t")
        barcodes = fold_table.barcode.astype(str).tolist()
        rna_counts, rna_barcodes, rna_gene_ids, protein_coding, unique_names = _load_rna(sample, root)
        unique_names = canonicalize_unique_gene_names(unique_names)
        rna_counts = _align_rows(rna_counts, rna_barcodes, barcodes)
        target_sum = float(rna_config["rna_programs"]["target_sum"])
        rna_normalized = normalize_total_log1p(rna_counts, target_sum)
        detection = np.asarray((rna_counts > 0).sum(axis=0)).ravel()
        eligible = np.flatnonzero(protein_coding & (detection >= math.ceil(0.05 * len(barcodes))))
        eligible_ids = rna_gene_ids[eligible]
        ranks = np.empty((len(barcodes), len(eligible)), dtype=np.float32)
        for start in range(0, len(eligible), 128):
            stop = min(start + 128, len(eligible))
            ranks[:, start:stop] = stats.rankdata(rna_normalized[:, eligible[start:stop]].toarray(), axis=0).astype(np.float32)
        new = manifest.loc[manifest.dataset_id.eq(dataset_id) & manifest.representation_role.eq(PF_ROLE) & _as_bool(manifest.success)]

        def rna_one(row: Any) -> tuple[dict[str, Any], pd.DataFrame]:
            activity = ad.read_h5ad(root / row.output_path)
            if activity.obs_names.astype(str).tolist() != barcodes:
                raise ValueError(f"TA2 paired-RNA observation order mismatch: {row.output_path}")
            values, genes = _evaluate_matrix(activity, rna_normalized, eligible, eligible_ids, ranks, unique_names, target_sum, 128)
            base = {"condition_id": row.condition_id, "dataset_id": row.dataset_id, "study_id": row.study_id, "study_family": row.study_family, "genome_build": row.genome_build, "depth_fraction": row.depth_fraction, "seed": row.seed, "representation_id": row.representation_id, "representation_role": row.representation_role, "method_id": row.method_id, "output_path": row.output_path, "output_sha256": row.output_sha256}
            genes.insert(0, "method_id", row.method_id); genes.insert(0, "seed", row.seed); genes.insert(0, "depth_fraction", row.depth_fraction); genes.insert(0, "dataset_id", row.dataset_id)
            return {**base, **values, "status": "pass"}, genes
        with ThreadPoolExecutor(max_workers=4) as executor:
            for metric, genes in executor.map(rna_one, new.itertuples(index=False)):
                new_rna_rows.append(metric); gene_frames.append(genes)
        models = _fit_spatial_models(sample, fold_table, rna_counts, rna_gene_ids, protein_coding, rna_normalized, rna_config["rna_programs"], rna_config["spatial_metric"])
        spatial_methods = manifest.loc[manifest.dataset_id.eq(dataset_id) & _as_bool(manifest.success)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            spatial_frames.extend(executor.map(lambda row: _spatial_contract(row, models, barcodes, unique_names, target_sum, root), spatial_methods.itertuples(index=False)))
    new_rna = pd.DataFrame(new_rna_rows)
    anchors = ta1.loc[ta1.depth_fraction.eq(1.0) & ta1.representation_id.eq("macs3_summit_501bp") & ta1.method_id.isin(config["methods"]), ["dataset_id", "method_id", "coverage_adjusted_median_gene_wise_spearman"]].rename(columns={"coverage_adjusted_median_gene_wise_spearman": "full_depth_P_primary"})
    new_rna = new_rna.merge(anchors, on=["dataset_id", "method_id"], how="left", validate="many_to_one")
    new_rna["frozen_peak_retention_vs_full_depth_P"] = new_rna.coverage_adjusted_median_gene_wise_spearman / new_rna.full_depth_P_primary
    spatial_detail = pd.concat(spatial_frames, ignore_index=True)
    group = ["condition_id", "dataset_id", "study_id", "study_family", "genome_build", "depth_fraction", "seed", "representation_id", "representation_role", "method_id"]
    spatial_summary = spatial_detail.groupby(group, sort=True).agg(fold_component_units=("component", "size"), median_heldout_component_spearman=("heldout_component_spearman", "median"), median_heldout_component_pearson=("heldout_component_pearson", "median"), median_absolute_morans_i_difference=("absolute_morans_i_difference", "median"), median_activity_present_gene_fraction=("activity_present_gene_fraction", "median")).reset_index()
    keys = ["dataset_id", "depth_fraction", "seed", "method_id"]
    existing = ta1.loc[ta1.dataset_id.isin(config["studies"]) & ta1.depth_fraction.lt(1.0) & ta1.method_id.isin(config["methods"]) & ta1.representation_id.isin(["macs3_summit_501bp", "fixed_bin_25kb"]), keys + ["representation_id", "structural_success", "coverage_adjusted_median_gene_wise_spearman", "rna_primary_retention_vs_full"]].copy()
    existing["representation_role"] = np.where(existing.representation_id.eq("fixed_bin_25kb"), B_ROLE, P_ROLE)
    primary_pivot = existing.pivot_table(index=keys, columns="representation_role", values="coverage_adjusted_median_gene_wise_spearman", aggfunc="first").reset_index()
    retention_pivot = existing.pivot_table(index=keys, columns="representation_role", values="rna_primary_retention_vs_full", aggfunc="first").reset_index().rename(columns={P_ROLE: "de_novo_P_retention", B_ROLE: "B25_retention"})
    mechanism = new_rna.merge(primary_pivot, on=keys, how="left", validate="one_to_one").merge(retention_pivot[keys + ["de_novo_P_retention", "B25_retention"]], on=keys, how="left", validate="one_to_one")
    mechanism = mechanism.rename(columns={"coverage_adjusted_median_gene_wise_spearman": "frozen_P_primary", P_ROLE: "de_novo_P_primary", B_ROLE: "B25_primary"})
    mechanism["frozen_P_minus_de_novo_P_primary"] = mechanism.frozen_P_primary - mechanism.de_novo_P_primary
    mechanism["B25_minus_frozen_P_primary"] = mechanism.B25_primary - mechanism.frozen_P_primary
    spatial_pivot = spatial_summary.pivot_table(index=keys, columns="representation_role", values="median_heldout_component_spearman", aggfunc="first").reset_index().rename(columns={P_ROLE: "de_novo_P_spatial_program", PF_ROLE: "frozen_P_spatial_program", B_ROLE: "B25_spatial_program"})
    mechanism = mechanism.merge(spatial_pivot, on=keys, how="left", validate="one_to_one")
    mechanism["frozen_P_minus_de_novo_P_spatial_program"] = mechanism.frozen_P_spatial_program - mechanism.de_novo_P_spatial_program
    mechanism["B25_minus_frozen_P_spatial_program"] = mechanism.B25_spatial_program - mechanism.frozen_P_spatial_program
    method_group = ["dataset_id", "study_id", "study_family", "genome_build", "depth_fraction", "method_id"]
    method_summary = mechanism.groupby(method_group, sort=True, dropna=False).agg(
        technical_seeds=("seed", "size"),
        frozen_P_primary_median=("frozen_P_primary", "median"),
        de_novo_P_primary_median=("de_novo_P_primary", "median"),
        B25_primary_median=("B25_primary", "median"),
        frozen_P_minus_de_novo_P_primary_median=("frozen_P_minus_de_novo_P_primary", "median"),
        B25_minus_frozen_P_primary_median=("B25_minus_frozen_P_primary", "median"),
        frozen_P_spatial_program_median=("frozen_P_spatial_program", "median"),
        de_novo_P_spatial_program_median=("de_novo_P_spatial_program", "median"),
        B25_spatial_program_median=("B25_spatial_program", "median"),
        frozen_P_minus_de_novo_P_spatial_program_median=("frozen_P_minus_de_novo_P_spatial_program", "median"),
        B25_minus_frozen_P_spatial_program_median=("B25_minus_frozen_P_spatial_program", "median"),
    ).reset_index()
    study_group = ["dataset_id", "study_id", "study_family", "genome_build", "depth_fraction"]
    study_summary = method_summary.groupby(study_group, sort=True, dropna=False).agg(
        methods=("method_id", "size"),
        median_frozen_P_primary=("frozen_P_primary_median", "median"),
        median_de_novo_P_primary=("de_novo_P_primary_median", "median"),
        median_B25_primary=("B25_primary_median", "median"),
        median_frozen_P_minus_de_novo_P_primary=("frozen_P_minus_de_novo_P_primary_median", "median"),
        methods_with_positive_primary_rescue=("frozen_P_minus_de_novo_P_primary_median", lambda values: int(np.sum(np.asarray(values) > 0))),
        methods_with_evaluable_de_novo_P=("de_novo_P_primary_median", "count"),
        median_B25_minus_frozen_P_primary=("B25_minus_frozen_P_primary_median", "median"),
        median_frozen_P_spatial_program=("frozen_P_spatial_program_median", "median"),
        median_de_novo_P_spatial_program=("de_novo_P_spatial_program_median", "median"),
        median_B25_spatial_program=("B25_spatial_program_median", "median"),
        median_frozen_P_minus_de_novo_P_spatial_program=("frozen_P_minus_de_novo_P_spatial_program_median", "median"),
        methods_with_positive_spatial_program_rescue=("frozen_P_minus_de_novo_P_spatial_program_median", lambda values: int(np.sum(np.asarray(values) > 0))),
        median_B25_minus_frozen_P_spatial_program=("B25_minus_frozen_P_spatial_program_median", "median"),
    ).reset_index()
    study_summary["independent_inference_unit"] = "study_family"
    study_summary["technical_seeds_independent"] = False
    study_summary["methods_independent_replicates"] = False
    output = root / config["output_root"] / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "frozen_peak_rna_metrics.tsv": new_rna,
        "spatial_program_contract_summary.tsv": spatial_summary,
        "mechanism_comparison.tsv": mechanism,
        "method_study_depth_summary.tsv": method_summary,
        "study_depth_mechanism_summary.tsv": study_summary,
    }
    artifacts = {}
    for name, frame in paths.items():
        path = output / name; frame.to_csv(path, sep="\t", index=False); artifacts[name] = {"path": str(path.relative_to(root)), "sha256": sha256_path(path)}
    detail_path = output / "spatial_program_fold_component_metrics.tsv.gz"
    spatial_detail.to_csv(detail_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    artifacts[detail_path.name] = {"path": str(detail_path.relative_to(root)), "sha256": sha256_path(detail_path)}
    genes_path = output / "frozen_peak_gene_level_correlations.tsv.gz"
    pd.concat(gene_frames, ignore_index=True).to_csv(genes_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    artifacts[genes_path.name] = {"path": str(genes_path.relative_to(root)), "sha256": sha256_path(genes_path)}
    evidence = {
        "schema_version": 1, "milestone": "TA2_frozen_peak_rescue_evaluation", "status": "pass",
        "new_frozen_peak_rna_contracts": len(new_rna), "spatial_program_contracts": len(spatial_summary),
        "spatial_program_detail_rows": len(spatial_detail), "mechanism_comparisons": len(mechanism),
        "method_study_depth_summaries": len(method_summary), "study_depth_summaries": len(study_summary),
        "rna_expression_values_accessed_for_evaluation": True,
        "rna_used_for_construction": False, "construction_outputs_mutated": False,
        "construction_or_method_rerun": False,
        "parameter_tuning_run": False, "labels_clusters_pathology_accessed": False,
        "technical_seeds_treated_as_independent_studies": False, "input_tracks_pooled": False,
        "fold_component_units_treated_as_independent_studies": False,
        "policy_refit_run": False, "universal_ranking_run": False, "final_ranking_run": False,
        "artifacts": artifacts,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "preflight": {"path": str(preflight_path.relative_to(root)), "sha256": sha256_path(preflight_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "build", "freeze-methods", "run-methods", "freeze-evaluation", "evaluate"])
    parser.add_argument("--config", default="configs/nm_frozen_peak_rescue.yaml")
    parser.add_argument("--authorization", default="")
    parser.add_argument("--preflight", default="results/v1/nm_truth_anchor/ta2_frozen_peak_rescue/evaluation_preflight_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "preflight": result = preflight(args.config, args.root)
    elif args.command == "build": result = build(args.config, args.authorization or "governance/authorizations/NM_TA2_FROZEN_PEAK_REPRESENTATION_CONSTRUCTION_AUTHORIZATION_2026-08-01.json", args.root)
    elif args.command == "freeze-methods": result = freeze_methods(args.config, args.root)
    elif args.command == "run-methods": result = run_methods(args.config, args.root)
    elif args.command == "freeze-evaluation": result = freeze_evaluation(args.config, args.root)
    else: result = evaluate(args.config, args.preflight, args.authorization or "governance/authorizations/NM_TA2_FROZEN_PEAK_RESCUE_EVALUATION_AUTHORIZATION_2026-08-01.json", args.root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
