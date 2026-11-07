"""Freeze the NU1/NU2 human method-construction plan after representation QC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ANNOTATION_METHODS = [
    "promoter_only_2kb_pc",
    "gene_body_pc",
    "promoter_plus_gene_body_2kb_pc",
    "distance_weighted_distal_250kb_decay50kb_pc",
    "promoter_plus_distal_2kb_250kb_decay50kb_pc",
    "accessibility_matched_random_peak_to_gene_control_pc",
]
SOFTWARE_METHODS = ["snapatac2_geneactivity", "signac_geneactivity", "archr_genescorematrix"]
REPRESENTATIONS = ["macs3_summit_501bp", "fixed_bin_10kb", "fixed_bin_25kb", "fixed_bin_50kb"]
NU1 = "gse205055_humanbrain_50um"
NU2 = "gse278007_human_melanoma_50um"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def build_plan(summary: pd.DataFrame, samples: dict[str, Any], output_root: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id in (NU1, NU2):
        sample = samples[dataset_id]
        methods = ANNOTATION_METHODS if dataset_id == NU1 else ANNOTATION_METHODS[:-1]
        for representation in REPRESENTATIONS:
            selected = summary.loc[
                summary.dataset_id.eq(dataset_id) & summary.representation_id.eq(representation)
            ]
            if len(selected) != 1:
                raise ValueError(f"Representation is not unique: {dataset_id}/{representation}")
            source = selected.iloc[0]
            for method in methods:
                rows.append({
                    "block": sample["block"],
                    "study_id": sample["study_id"],
                    "accession": sample["accession"],
                    "dataset_id": dataset_id,
                    "sample_id": sample["sample_id"],
                    "species": sample["species"],
                    "genome_build": sample["genome_build"],
                    "representation_id": representation,
                    "input_track": "P" if representation.startswith("macs3") else "B",
                    "method_id": method,
                    "role": "negative_control" if method.endswith("control_pc") else "primary_annotation",
                    "tasks_using_output": "NU1_RNA_and_portability" if dataset_id == NU1 else "NU2_portability_only",
                    "run_type": "annotation",
                    "input_path": source.output_path,
                    "input_sha256": source.output_sha256,
                    "fragments": "",
                    "output_dir": f"{output_root}/annotation/{dataset_id}/{representation}/{method}",
                    "attempts_authorized": 1,
                    "execution_status": "authorized_not_started",
                })
        peak = summary.loc[
            summary.dataset_id.eq(dataset_id) & summary.representation_id.eq("macs3_summit_501bp")
        ].iloc[0]
        for method in SOFTWARE_METHODS:
            rows.append({
                "block": sample["block"],
                "study_id": sample["study_id"],
                "accession": sample["accession"],
                "dataset_id": dataset_id,
                "sample_id": sample["sample_id"],
                "species": sample["species"],
                "genome_build": sample["genome_build"],
                "representation_id": "filtered_fragments_with_frozen_macs3_observation_contract",
                "input_track": "F",
                "method_id": method,
                "role": "primary_software",
                "tasks_using_output": "NU1_RNA_and_portability" if dataset_id == NU1 else "NU2_portability_only",
                "run_type": "software",
                "input_path": peak.output_path,
                "input_sha256": peak.output_sha256,
                "fragments": sample["filtered_fragments"],
                "output_dir": f"{output_root}/software/{dataset_id}/{method}",
                "attempts_authorized": 1,
                "execution_status": "authorized_not_started",
            })
    plan = pd.DataFrame(rows)
    if len(plan) != 50 or int(plan.run_type.eq("annotation").sum()) != 44 or int(plan.run_type.eq("software").sum()) != 6:
        raise ValueError("NU1/NU2 exact method plan dimensions changed")
    if plan.output_dir.duplicated().any():
        raise ValueError("NU1/NU2 method plan has duplicate output directories")
    return plan


def setup(root: Path) -> Path:
    representation_config_path = root / "configs/nm_upgrade_human_representations.yaml"
    representation_config = _load_yaml(representation_config_path)
    completion_path = root / representation_config["evidence_dir"] / "completion_evidence.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "pass" or completion.get("samples") != 2 or completion.get("representations") != 8:
        raise ValueError("NU1/NU2 representation construction is not complete")
    if completion["config"]["sha256"] != sha256_path(representation_config_path):
        raise ValueError("NU1/NU2 representation config changed after construction")

    summary_rows: list[pd.DataFrame] = []
    samples = representation_config["samples"]
    for dataset_id, sample in samples.items():
        manifest_path = root / representation_config["output_root"] / dataset_id / "matched/representation_manifest.tsv"
        manifest = pd.read_csv(manifest_path, sep="\t")
        if set(manifest.representation_id) != set(REPRESENTATIONS) or not manifest.status.eq("pass").all():
            raise ValueError(f"NU human representation manifest is incomplete: {dataset_id}")
        for row in manifest.itertuples(index=False):
            path = root / str(row.output_path)
            if not path.is_file() or sha256_path(path) != str(row.output_sha256):
                raise ValueError(f"NU human representation hash mismatch: {row.output_path}")
        manifest["block"] = sample["block"]
        manifest["species"] = sample["species"]
        manifest["genome_build"] = sample["genome_build"]
        summary_rows.append(manifest)
        fragments = root / representation_config["output_root"] / dataset_id / "filtered_fragments" / f"{dataset_id}__eligible.fragments.tsv.gz"
        sample["filtered_fragments"] = str(fragments.relative_to(root))
        sample["filtered_fragments_sha256"] = sha256_path(fragments)

    evidence_dir = root / "results/v1/nm_upgrade/nu1_nu2_human_methods/evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    representation_summary = pd.concat(summary_rows, ignore_index=True)
    summary_path = evidence_dir / "representation_summary.tsv"
    representation_summary.to_csv(summary_path, sep="\t", index=False)

    output_root = "results/v1/g5/h5b_external_method_construction/nm_upgrade_human"
    plan = build_plan(representation_summary, samples, output_root)
    plan_path = evidence_dir / "planned_method_execution_manifest.tsv"
    plan.to_csv(plan_path, sep="\t", index=False)

    protocol_path = root / representation_config["upgrade_protocol"]
    authorization_path = root / "governance/authorizations/NM_U1_U2_HUMAN_METHOD_CONSTRUCTION_AUTHORIZATION_2026-07-30.json"
    authorization = {
        "schema_version": 1,
        "authorization": "NM_U1_U2_HUMAN_METHOD_CONSTRUCTION",
        "milestone": "H5B_external_method_construction",
        "upgrade_milestone": "NU1_NU2_human_method_construction",
        "authorization_date": "2026-07-30",
        "status": "authorized",
        "scope": "locked_human_extension_accessibility_only_construction",
        "authorized_studies": sorted({sample["study_id"] for sample in samples.values()}),
        "authorized_datasets": sorted(samples),
        "authorized_methods": ANNOTATION_METHODS + SOFTWARE_METHODS,
        "authorized_method_panel": ANNOTATION_METHODS + SOFTWARE_METHODS,
        "authorized_representations": REPRESENTATIONS,
        "planned_outputs": 50,
        "planned_annotation_outputs": 44,
        "planned_software_outputs": 6,
        "method_execution_authorized": True,
        "evaluation_tasks_authorized": False,
        "rna_value_access_authorized": False,
        "label_or_pathology_value_access_authorized": False,
        "holdout_utility_access_authorized": False,
        "regret_computation_authorized": False,
        "policy_refit_authorized": False,
        "full_benchmark_authorized": False,
        "results_writing_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
        "technical_dataset_role": "external_holdout_contract_used_for_locked_extension_outputs",
        "independence_note": "both samples extend existing study families and do not increase independent-study denominators",
        "planned_execution_manifest": {"path": str(plan_path.relative_to(root)), "sha256": sha256_path(plan_path)},
        "representation_summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
        "representation_completion": {"path": str(completion_path.relative_to(root)), "sha256": sha256_path(completion_path)},
        "upgrade_protocol": {"path": str(protocol_path.relative_to(root)), "sha256": sha256_path(protocol_path)},
        "setup_implementation": {"path": "spatialgabench/nm_human_method_setup.py", "sha256": sha256_path(Path(__file__))},
    }
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    authorization_sha = sha256_path(authorization_path)

    reference_files = {
        "genes": "data/reference/hg38/protein_coding_genes.bed",
        "promoters": "data/reference/hg38/protein_coding_promoters_2kb.bed",
        "gene_body": "data/reference/hg38/protein_coding_gene_body.bed",
        "tss": "data/reference/hg38/protein_coding_tss.bed",
    }
    references = {name: {"path": path, "sha256": sha256_path(root / path)} for name, path in reference_files.items()}
    frozen = {
        "authorization": str(authorization_path.relative_to(root)),
        "authorization_sha256": authorization_sha,
        "planned_execution_manifest": str(plan_path.relative_to(root)),
        "planned_execution_manifest_sha256": sha256_path(plan_path),
        "representation_summary": str(summary_path.relative_to(root)),
        "representation_summary_sha256": sha256_path(summary_path),
        "representation_completion": str(completion_path.relative_to(root)),
        "representation_completion_sha256": sha256_path(completion_path),
        "upgrade_protocol": str(protocol_path.relative_to(root)),
        "upgrade_protocol_sha256": sha256_path(protocol_path),
        "frozen_artifacts": ["authorization", "planned_execution_manifest", "representation_summary", "representation_completion", "upgrade_protocol"],
    }
    annotation_config = {
        "schema_version": 1,
        "gate": "NM_UPGRADE",
        "milestone": "NU1_NU2_human_method_construction",
        "dataset_role": "external_holdout",
        "authorization_milestone": "H5B_external_method_construction",
        **frozen,
        "output_root": f"{output_root}/annotation",
        "run_id_prefix": "nm_u1_u2",
        "input_validation_scope": "NU1_NU2_human_accessibility_method_construction_only",
        "driver_stage": "nm_u1_u2_annotation_driver",
        "comparison_scope": "within_representation_human_extension_construction_only",
        "subset_policy": "full",
        "forbidden_holdout_accessions": [],
        "forbidden_construction_inputs": ["rna", "labels", "clusters", "pathology", "images", "simulation_truth", "method_outcomes"],
        "methods": ANNOTATION_METHODS,
        "parameters": {"promoter_window_bp": 2000, "distal_window_bp": 250000, "distance_decay_bp": 50000, "control_seed": 4160, "accessibility_bins": 20},
        "references": {"hg38": references},
    }
    annotation_config_path = root / "configs/nm_upgrade_human_annotation_methods.yaml"
    annotation_config_path.write_text(yaml.safe_dump(annotation_config, sort_keys=False), encoding="utf-8")

    base_software = _load_yaml(root / "configs/g4_m2_software_methods.yaml")
    software_samples = {}
    for dataset_id, sample in samples.items():
        peak = representation_summary.loc[
            representation_summary.dataset_id.eq(dataset_id)
            & representation_summary.representation_id.eq("macs3_summit_501bp")
        ].iloc[0]
        software_samples[dataset_id] = {
            "study_id": sample["study_id"],
            "sample_id": sample["sample_id"],
            "genome_build": "hg38",
            "input_h5ad": peak.output_path,
            "source_fragments": sample["filtered_fragments"],
            "source_fragments_sha256": sample["filtered_fragments_sha256"],
        }
    software_config = {
        "schema_version": 1,
        "project_config": "configs/main.yaml",
        "dataset_role": "external_holdout",
        **frozen,
        "output_root": f"{output_root}/software",
        "run_id_prefix": "nm_u1_u2",
        "comparison_scope": "locked_human_fragment_native_construction_only",
        "representation_id": "filtered_fragments_with_frozen_macs3_observation_contract",
        "subset_policy": "full",
        "threads": 8,
        "seed": 4160,
        "forbidden_holdout_accessions": [],
        "forbidden_construction_inputs": ["rna", "labels", "clusters", "pathology", "images", "simulation_truth", "method_outcomes"],
        "oci": base_software["oci"],
        "methods": base_software["methods"],
        "samples": software_samples,
        "references": {"hg38": base_software["references"]["hg38"]},
    }
    software_config_path = root / "configs/nm_upgrade_human_software_methods.yaml"
    software_config_path.write_text(yaml.safe_dump(software_config, sort_keys=False), encoding="utf-8")

    evidence = {
        "schema_version": 1,
        "milestone": "NU1_NU2_human_method_construction_setup",
        "status": "pass",
        "planned_outputs": len(plan),
        "planned_annotation_outputs": int(plan.run_type.eq("annotation").sum()),
        "planned_software_outputs": int(plan.run_type.eq("software").sum()),
        "rna_accessed": False,
        "labels_pathology_images_accessed": False,
        "method_execution_run": False,
        "policy_refit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": authorization_sha},
        "annotation_config": {"path": str(annotation_config_path.relative_to(root)), "sha256": sha256_path(annotation_config_path)},
        "software_config": {"path": str(software_config_path.relative_to(root)), "sha256": sha256_path(software_config_path)},
        "planned_execution_manifest": authorization["planned_execution_manifest"],
        "representation_summary": authorization["representation_summary"],
        "representation_completion": authorization["representation_completion"],
    }
    evidence_path = evidence_dir / "setup_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(setup(Path(args.root).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
