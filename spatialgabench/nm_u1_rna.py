"""Governed NU1 human paired-RNA evaluation and within-family synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from spatialgabench.g4_m3b_rna import _gtf_annotations, _rna_ranks
from spatialgabench.g4_m3h_rna_replication import _evaluate_method, _prepare_rna


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _identity_preflight_is_expression_blind(identity: dict[str, Any]) -> bool:
    checks = identity.get("checks")
    return (
        identity.get("status") == "pass"
        and identity.get("rna_expression_accessed") is False
        and isinstance(checks, dict)
        and checks.get("rna_expression_rows_read") is False
        and checks.get("rna_used_for_construction") is False
    )


def _canonical_gene_id(value: str) -> str:
    return re.sub(r"\.\d+$", "", str(value))


def _canonicalize_unique_gene_names(unique_names: dict[str, str]) -> dict[str, str]:
    return {name: _canonical_gene_id(gene_id) for name, gene_id in unique_names.items()}


def setup(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _yaml(config_path)
    for key in ("rna_matrix", "gene_annotation", "identity_preflight", "mouse_within_family_metrics", "mouse_within_family_completion"):
        record = config[key]
        if sha256_path(root / record["path"]) != record["sha256"]:
            raise ValueError(f"NU1 RNA frozen artifact mismatch: {key}")
    identity = _json(root / config["identity_preflight"]["path"])
    if not _identity_preflight_is_expression_blind(identity):
        raise PermissionError("NU1 identity preflight does not preserve RNA blindness")
    method_completion_path = root / config["method_completion"]
    method_completion = _json(method_completion_path)
    if method_completion.get("status") != "pass" or method_completion.get("successful_contracts") != 50:
        raise ValueError("NU1 RNA waits for all human construction contracts")
    summary = pd.read_csv(root / method_completion["run_summary"]["path"], sep="\t")
    methods = summary.loc[
        summary.dataset_id.eq(config["dataset_id"]) & summary.outcome.eq("success")
    ].copy()
    if len(methods) != int(config["expected_method_contracts"]):
        raise ValueError("NU1 RNA exact method panel is incomplete")
    methods["method_category"] = np.where(
        methods.method_id.eq(config["matched_control"]),
        "matched_control",
        np.where(methods.run_type.eq("software"), "fragment_software", "hand_written_annotation"),
    )
    methods["gene_activity_h5ad"] = methods.output_dir.map(lambda value: f"{value}/gene_activity.h5ad")
    methods["gene_activity_sha256"] = methods.output_h5ad_sha256
    columns = [
        "study_id", "dataset_id", "sample_id", "genome_build", "representation_id",
        "input_track", "method_id", "method_category", "gene_activity_h5ad", "gene_activity_sha256",
    ]
    methods = methods[columns].sort_values(["method_category", "method_id", "representation_id"], kind="mergesort")
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    method_path = output / "method_input_manifest.tsv"
    methods.to_csv(method_path, sep="\t", index=False)
    normalization_amendment_path = root / "governance/amendments/NM_U1_RNA_CANONICAL_GENE_ID_ALIGNMENT_V3_2026-07-30.yaml"
    authorization = {
        "schema_version": 1,
        "authorization": "NM_U1_HUMAN_PAIRED_RNA",
        "milestone": "NU1_human_standard_fragment_paired_RNA",
        "authorization_date": "2026-07-30",
        "status": "authorized",
        "scope": "frozen_human_paired_RNA_evaluation_only",
        "rna_expression_value_access_authorized": True,
        "paired_rna_evaluation_authorized": True,
        "within_family_synthesis_authorized": True,
        "method_rerun_authorized": False,
        "evaluation_reexecution_after_invalid_schema_run_authorized": True,
        "construction_tuning_authorized": False,
        "labels_clusters_pathology_access_authorized": False,
        "policy_refit_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "identity_preflight": config["identity_preflight"],
        "method_completion": {"path": str(method_completion_path.relative_to(root)), "sha256": sha256_path(method_completion_path)},
        "method_input_manifest": {"path": str(method_path.relative_to(root)), "sha256": sha256_path(method_path)},
        "implementation": {"path": "spatialgabench/nm_u1_rna.py", "sha256": sha256_path(Path(__file__))},
        "normalization_amendment": {
            "path": str(normalization_amendment_path.relative_to(root)),
            "sha256": sha256_path(normalization_amendment_path),
        },
        "upgrade_protocol": {"path": config["upgrade_protocol"], "sha256": sha256_path(root / config["upgrade_protocol"])},
    }
    authorization_path = root / "governance/authorizations/NM_U1_HUMAN_PAIRED_RNA_AUTHORIZATION_V3_2026-07-30.json"
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "milestone": "NU1_human_paired_RNA_setup",
        "status": "pass",
        "method_contracts": len(methods),
        "rna_expression_values_read": False,
        "gene_activity_values_read": False,
        "evaluation_run": False,
        "method_rerun": False,
        "construction_tuning_run": False,
        "policy_refit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "normalization_amendment": authorization["normalization_amendment"],
        "method_input_manifest": authorization["method_input_manifest"],
        "method_completion": authorization["method_completion"],
        "config": authorization["config"],
    }
    path = output / "setup_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_authorization(config_path: Path, setup_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config, setup = _yaml(config_path), _json(setup_path)
    authorization_path = root / setup["authorization"]["path"]
    authorization = _json(authorization_path)
    if setup.get("status") != "pass" or setup.get("rna_expression_values_read") is not False:
        raise ValueError("NU1 RNA setup does not pass")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != config["milestone"]
        or authorization.get("rna_expression_value_access_authorized") is not True
        or authorization.get("paired_rna_evaluation_authorized") is not True
        or authorization.get("within_family_synthesis_authorized") is not True
        or authorization.get("evaluation_reexecution_after_invalid_schema_run_authorized") is not True
        or authorization.get("method_rerun_authorized") is not False
        or authorization.get("construction_tuning_authorized") is not False
        or authorization.get("labels_clusters_pathology_access_authorized") is not False
        or authorization.get("policy_refit_authorized") is not False
        or authorization.get("universal_ranking_authorized") is not False
        or authorization.get("final_ranking_authorized") is not False
    ):
        raise PermissionError("NU1 RNA evaluation authorization is invalid")
    bindings = {
        "config": config_path,
        "identity_preflight": root / authorization["identity_preflight"]["path"],
        "method_completion": root / authorization["method_completion"]["path"],
        "method_input_manifest": root / authorization["method_input_manifest"]["path"],
        "implementation": Path(__file__),
        "normalization_amendment": root / authorization["normalization_amendment"]["path"],
        "upgrade_protocol": root / config["upgrade_protocol"],
    }
    for name, path in bindings.items():
        record = authorization[name]
        if record.get("path") != str(path.resolve().relative_to(root)) or record.get("sha256") != sha256_path(path):
            raise ValueError(f"NU1 RNA authorization binding mismatch: {name}")
    return config, setup


def run(config_path: str | Path, setup_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path, setup_path = Path(root).resolve(), Path(config_path).resolve(), Path(setup_path).resolve()
    config, setup = _validate_authorization(config_path, setup_path, root)
    methods = pd.read_csv(root / setup["method_input_manifest"]["path"], sep="\t")
    for key in ("rna_matrix", "gene_annotation", "mouse_within_family_metrics", "mouse_within_family_completion"):
        record = config[key]
        if sha256_path(root / record["path"]) != record["sha256"]:
            raise ValueError(f"NU1 RNA live frozen artifact mismatch: {key}")
    for method in methods.itertuples(index=False):
        if sha256_path(root / method.gene_activity_h5ad) != method.gene_activity_sha256:
            raise ValueError("NU1 method output changed before RNA value access")
    first = ad.read_h5ad(root / methods.iloc[0].gene_activity_h5ad, backed="r")
    obs_names = first.obs_names.copy()
    first.file.close()
    if len(obs_names) != int(config["expected_observations"]):
        raise ValueError("NU1 RNA observation count changed")
    legacy_config = {
        **config,
        "rna_matrix": config["rna_matrix"]["path"],
        "gene_annotation": config["gene_annotation"]["path"],
    }
    rna, eligible_features, eligible_indices, unique_names = _prepare_rna(legacy_config, root, obs_names)
    eligible_features["source_gene_id"] = eligible_features["gene_id"].astype(str)
    eligible_features["gene_id"] = eligible_features["source_gene_id"].map(_canonical_gene_id)
    if not eligible_features["gene_id"].is_unique:
        raise ValueError("NU1 RNA gene IDs are not unique after Ensembl version normalization")
    unique_names = _canonicalize_unique_gene_names(unique_names)
    ranks = _rna_ranks(rna, eligible_indices, int(config["correlation_chunk_genes"]))
    metric_rows: list[dict[str, Any]] = []
    gene_frames: list[pd.DataFrame] = []
    for _, method in methods.iterrows():
        metric, genes = _evaluate_method(
            method,
            root,
            obs_names,
            eligible_features,
            rna,
            eligible_indices,
            ranks,
            unique_names,
            float(config["normalization"]["target_sum"]),
            int(config["correlation_chunk_genes"]),
        )
        metric_rows.append(metric)
        gene_frames.append(genes)
    metrics = pd.DataFrame(metric_rows)
    primary = config["primary_endpoint"]
    control = metrics.loc[metrics.method_id.eq(config["matched_control"]), ["representation_id", primary]].rename(columns={primary: "matched_control_primary"})
    deltas = metrics.loc[metrics.method_id.isin(config["annotation_methods"])].merge(control, on="representation_id", how="left", validate="many_to_one")
    deltas["delta_vs_matched_control"] = deltas[primary] - deltas.matched_control_primary
    if len(deltas) != int(config["expected_matched_control_deltas"]):
        raise ValueError("NU1 matched-control panel changed")

    mouse = pd.read_csv(root / config["mouse_within_family_metrics"]["path"], sep="\t")
    mouse = mouse.loc[mouse.method_id.isin(config["annotation_methods"] + [config["matched_control"]]) & mouse.representation_id.isin(config["representations"])]
    mouse_control = mouse.loc[mouse.method_id.eq(config["matched_control"]), ["representation_id", primary]].rename(columns={primary: "mouse_control_primary"})
    mouse_delta = mouse.loc[mouse.method_id.isin(config["annotation_methods"])].merge(mouse_control, on="representation_id", validate="many_to_one")
    mouse_delta["mouse_delta_vs_matched_control"] = mouse_delta[primary] - mouse_delta.mouse_control_primary
    within_family = deltas[["method_id", "representation_id", primary, "delta_vs_matched_control"]].merge(
        mouse_delta[["method_id", "representation_id", primary, "mouse_delta_vs_matched_control"]],
        on=["method_id", "representation_id"], suffixes=("_human", "_mouse"), validate="one_to_one",
    )
    within_family["human_positive"] = within_family.delta_vs_matched_control > 0
    within_family["mouse_positive"] = within_family.mouse_delta_vs_matched_control > 0
    within_family["direction_consistent"] = within_family.human_positive.eq(within_family.mouse_positive)
    delta_spearman = float(stats.spearmanr(within_family.delta_vs_matched_control, within_family.mouse_delta_vs_matched_control).statistic)
    if not np.isfinite(delta_spearman):
        delta_spearman = 0.0

    output = root / config["output_root"]
    metrics_path = output / "rna_metrics.tsv"
    genes_path = output / "gene_level_correlations.tsv.gz"
    eligible_path = output / "rna_eligible_genes.tsv"
    delta_path = output / "matched_control_deltas.tsv"
    family_path = output / "within_family_mouse_human.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    pd.concat(gene_frames, ignore_index=True).to_csv(genes_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    eligible_features.to_csv(eligible_path, sep="\t", index=False)
    deltas.to_csv(delta_path, sep="\t", index=False)
    within_family.to_csv(family_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "milestone": config["milestone"],
        "status": "pass",
        "method_contracts": len(metrics),
        "matched_control_deltas": len(deltas),
        "positive_annotation_control_deltas": int(deltas.delta_vs_matched_control.gt(0).sum()),
        "within_family_direction_consistent": int(within_family.direction_consistent.sum()),
        "within_family_comparisons": len(within_family),
        "within_family_delta_spearman_descriptive": delta_spearman,
        "independent_study_increment": 0,
        "evaluation_attempt_index": 3,
        "invalid_schema_attempt_excluded": True,
        "rna_expression_values_accessed_for_evaluation": True,
        "rna_used_for_construction": False,
        "construction_outputs_mutated": False,
        "labels_clusters_pathology_accessed": False,
        "method_rerun": False,
        "construction_tuning_run": False,
        "policy_refit_run": False,
        "input_tracks_pooled": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "artifacts": {
            "rna_metrics.tsv": {"path": str(metrics_path.relative_to(root)), "sha256": sha256_path(metrics_path)},
            "gene_level_correlations.tsv.gz": {"path": str(genes_path.relative_to(root)), "sha256": sha256_path(genes_path)},
            "rna_eligible_genes.tsv": {"path": str(eligible_path.relative_to(root)), "sha256": sha256_path(eligible_path)},
            "matched_control_deltas.tsv": {"path": str(delta_path.relative_to(root)), "sha256": sha256_path(delta_path)},
            "within_family_mouse_human.tsv": {"path": str(family_path.relative_to(root)), "sha256": sha256_path(family_path)},
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "setup": {"path": str(setup_path.relative_to(root)), "sha256": sha256_path(setup_path)},
        "authorization": setup["authorization"],
        "implementation": {"path": "spatialgabench/nm_u1_rna.py", "sha256": sha256_path(Path(__file__))},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["setup", "run"])
    parser.add_argument("--config", default="configs/nm_upgrade_u1_human_rna.yaml")
    parser.add_argument("--setup", default="results/v1/nm_upgrade/nu1_human_rna/setup_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(setup(args.config, args.root) if args.command == "setup" else run(args.config, args.setup, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
