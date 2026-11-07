"""Audit G4-M3B development evaluation inputs without computing outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import anndata as ad
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


def _matrix_market_shape(path: Path) -> tuple[int, int, int]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if not header.startswith("%%MatrixMarket matrix coordinate"):
            raise ValueError("RNA matrix lacks a coordinate Matrix Market header")
        for line in handle:
            if not line.startswith("%"):
                fields = line.split()
                if len(fields) != 3:
                    raise ValueError("RNA Matrix Market dimension row is malformed")
                return tuple(int(value) for value in fields)
    raise ValueError("RNA Matrix Market dimensions are missing")


def _read_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["gene_id_versioned", "gene_name", "feature_type"],
        dtype=str,
        compression="gzip",
    )
    if frame.shape[1] != 3 or frame.isna().any().any():
        raise ValueError("RNA feature table must contain three complete columns")
    frame["gene_id"] = frame["gene_id_versioned"].str.replace(r"\.\d+$", "", regex=True)
    return frame


def _gtf_gene_biotypes(path: Path) -> dict[str, str]:
    gene_id_pattern = re.compile(r'gene_id "([^"]+)"')
    biotype_pattern = re.compile(r'gene_(?:bio)?type "([^"]+)"')
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            gene_match = gene_id_pattern.search(fields[8])
            biotype_match = biotype_pattern.search(fields[8])
            if gene_match and biotype_match:
                mapping[gene_match.group(1)] = biotype_match.group(1)
    return mapping


def _read_single_column_gzip(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t", 1)[0] for line in handle if line.strip()]


def _read_spatial_barcodes(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row[0] for row in csv.reader(handle) if row]


def audit(config_path: str | Path, root: str | Path = ".") -> dict[str, Path]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    for key in ("authorization", "previous_gate"):
        path = root / config[key]
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"M3B preflight checksum mismatch: {key}")
    authorization = _load_json(root / config["authorization"])
    if authorization.get("status") != "authorized" or authorization.get("milestone") != "M3B_input_preflight":
        raise PermissionError("M3B input-preflight authorization is not active")
    if authorization.get("biological_outcome_computation_authorized") is not False:
        raise PermissionError("M3B preflight cannot compute biological outcomes")
    if config.get("outcome_computation_allowed") is not False:
        raise PermissionError("M3B preflight config cannot enable outcomes")

    candidate = config["rna_candidate"]
    file_keys = ("matrix", "features", "barcodes", "spatial_positions", "final_atac_observations", "gene_annotation")
    for key in file_keys:
        path = root / candidate[key]
        if not path.is_file() or sha256(path) != candidate[f"{key}_sha256"]:
            raise ValueError(f"RNA candidate checksum mismatch: {key}")
    paired_audit_path = root / candidate["paired_schema_audit"]
    paired_audit = _load_json(paired_audit_path)
    if paired_audit.get("schema_status") != "pass" or paired_audit.get("barcode_contract_status") != "pass":
        raise ValueError("Existing paired RNA schema audit does not pass")

    matrix_rows, matrix_columns, matrix_nnz = _matrix_market_shape(root / candidate["matrix"])
    features = _read_features(root / candidate["features"])
    rna_barcodes = _read_single_column_gzip(root / candidate["barcodes"])
    spatial_barcodes = _read_spatial_barcodes(root / candidate["spatial_positions"])
    atac = ad.read_h5ad(root / candidate["final_atac_observations"], backed="r")
    atac_barcodes = atac.obs_names.astype(str).tolist()
    atac_shape = atac.shape
    atac_study = str(atac.uns.get("study_id", ""))
    atac_leakage = str(atac.uns.get("construction_leakage_inputs", ""))
    atac.file.close()
    gtf_biotypes = _gtf_gene_biotypes(root / candidate["gene_annotation"])
    mapped_biotypes = features["gene_id"].map(gtf_biotypes)
    rna_set = set(rna_barcodes)
    spatial_set = set(spatial_barcodes)
    atac_set = set(atac_barcodes)

    rna_audit = {
        "schema_version": 1,
        "status": "pass",
        "study_id": candidate["study_id"],
        "dataset_id": candidate["dataset_id"],
        "atac_sample_id": candidate["atac_sample_id"],
        "rna_sample_id": candidate["rna_sample_id"],
        "genome_build": candidate["genome_build"],
        "same_section_source_evidence": True,
        "matrix_rows": matrix_rows,
        "matrix_columns": matrix_columns,
        "matrix_declared_nonzero_entries": matrix_nnz,
        "feature_rows": len(features),
        "feature_ids_unique_versioned": bool(features["gene_id_versioned"].is_unique),
        "feature_ids_unique_after_version_strip": bool(features["gene_id"].is_unique),
        "feature_names_nonempty": bool(features["gene_name"].str.strip().ne("").all()),
        "feature_type_values": sorted(features["feature_type"].unique().tolist()),
        "gtf_gene_ids": len(gtf_biotypes),
        "features_mapped_to_canonical_gtf": int(mapped_biotypes.notna().sum()),
        "features_mapped_to_canonical_gtf_fraction": float(mapped_biotypes.notna().mean()),
        "features_mapped_protein_coding": int((mapped_biotypes == "protein_coding").sum()),
        "rna_barcode_rows": len(rna_barcodes),
        "rna_barcodes_unique": len(rna_set) == len(rna_barcodes),
        "rna_spatial_barcode_rows": len(spatial_barcodes),
        "rna_spatial_barcodes_unique": len(spatial_set) == len(spatial_barcodes),
        "final_atac_observations": atac_shape[0],
        "final_atac_features_reference_only": atac_shape[1],
        "final_atac_observation_ids_unique": len(atac_set) == len(atac_barcodes),
        "final_atac_observations_in_rna": len(atac_set & rna_set),
        "final_atac_observations_in_rna_fraction": len(atac_set & rna_set) / len(atac_set),
        "final_atac_observations_in_rna_spatial": len(atac_set & spatial_set),
        "final_atac_observations_in_rna_spatial_fraction": len(atac_set & spatial_set) / len(atac_set),
        "final_atac_order_sha256": hashlib.sha256("\0".join(atac_barcodes).encode()).hexdigest(),
        "rna_order_sha256": hashlib.sha256("\0".join(rna_barcodes).encode()).hexdigest(),
        "atac_study_id_matches": atac_study == candidate["study_id"],
        "atac_construction_leakage_inputs": atac_leakage,
        "protein_coding_detection_threshold_computed": False,
        "rna_expression_values_read": False,
        "gene_activity_outcomes_read": False,
        "rna_agreement_computed": False,
        "holdout_accessed": False,
        "recommendation": "ready_for_separate_m3b_rna_execution_authorization",
    }
    required = (
        matrix_rows == len(features), matrix_columns == len(rna_barcodes), matrix_columns == len(spatial_barcodes),
        rna_audit["feature_ids_unique_versioned"], rna_audit["rna_barcodes_unique"],
        rna_audit["rna_spatial_barcodes_unique"], atac_shape[0] == 9686,
        len(atac_set & rna_set) == len(atac_set), len(atac_set & spatial_set) == len(atac_set),
        atac_study == candidate["study_id"], atac_leakage == "none", mapped_biotypes.notna().any(),
    )
    if not all(required):
        rna_audit["status"] = "blocked"
        raise ValueError("M3B RNA candidate preflight contract failed")

    label_rows: list[dict[str, Any]] = []
    for record in config["label_candidates"]:
        row = dict(record)
        row.update({"label_file_exists": False, "label_rows": pd.NA, "unique_labels": pd.NA, "label_values_nonempty": pd.NA})
        if "path" in record:
            path = root / record["path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"Label candidate checksum mismatch: {record['dataset_id']}")
            frame = pd.read_csv(path, sep=record["delimiter"], compression="gzip", index_col=0)
            labels = frame[record["label_column"]].astype(str).str.strip()
            row.update({
                "label_file_exists": True,
                "label_rows": len(frame),
                "unique_labels": int(labels.nunique()),
                "label_values_nonempty": bool(labels.ne("").all()),
            })
        row["domain_metric_authorized_now"] = False
        row["method_predictions_read"] = False
        label_rows.append(row)
    labels = pd.DataFrame(label_rows)

    tasks = pd.DataFrame([
        {
            "task_id": "paired_rna_fidelity",
            "dataset_scope": candidate["dataset_id"],
            "preflight_status": "pass",
            "execution_status": "requires_separate_authorization",
            "reason": "same-section RNA and all final ATAC observations pass identity, schema, and GTF mapping checks",
        },
        {
            "task_id": "spatial_domain_and_pathology",
            "dataset_scope": "current_five_dataset_m2_panel",
            "preflight_status": "blocked",
            "execution_status": "not_authorized",
            "reason": "no independently audited region-label file exists for a current M2 construction dataset",
        },
        {
            "task_id": "spatial_domain_author_peak_extension",
            "dataset_scope": "gse214986_author_peak",
            "preflight_status": "partial",
            "execution_status": "requires_separate_construction_extension",
            "reason": "author clusters exist but active v1 method construction has not been run on this unmatched author-peak branch",
        },
        {
            "task_id": "simulation_truth_recovery",
            "dataset_scope": "mechanistic_and_empirical_semisynthetic",
            "preflight_status": "not_assessed_here",
            "execution_status": "requires_separate_authorization",
            "reason": "replicated simulation execution is a separate frozen task block",
        },
    ])

    output_root = root / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    rna_path = output_root / "g4_m3b_rna_input_audit.json"
    labels_path = output_root / "g4_m3b_label_candidate_audit.tsv"
    tasks_path = output_root / "g4_m3b_task_eligibility.tsv"
    evidence_path = output_root / "g4_m3b_input_preflight_evidence.json"
    rna_path.write_text(json.dumps(rna_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    labels.to_csv(labels_path, sep="\t", index=False)
    tasks.to_csv(tasks_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3B_input_preflight",
        "status": "pass",
        "rna_task_preflight": "pass",
        "domain_task_preflight": "blocked_no_labels_in_current_m2_panel",
        "author_peak_domain_extension": "partial_requires_separate_construction",
        "simulation_task_preflight": "not_assessed_here",
        "biological_outcome_computation_run": False,
        "rna_expression_values_read": False,
        "gene_activity_outcomes_read": False,
        "method_predictions_read": False,
        "rna_agreement_computed": False,
        "clustering_metrics_computed": False,
        "holdout_accessed": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "next_milestone_authorized": False,
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "implementation": {"path": "spatialgabench/g4_m3b_preflight.py", "sha256": sha256(Path(__file__))},
        "artifacts": {
            "rna_audit": {"path": str(rna_path.relative_to(root)), "sha256": sha256(rna_path)},
            "label_audit": {"path": str(labels_path.relative_to(root)), "sha256": sha256(labels_path), "rows": len(labels)},
            "task_eligibility": {"path": str(tasks_path.relative_to(root)), "sha256": sha256(tasks_path), "rows": len(tasks)},
        },
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rna_audit": rna_path, "label_audit": labels_path, "task_eligibility": tasks_path, "evidence": evidence_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m3b_input_preflight.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    outputs = audit(args.config, args.root)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

