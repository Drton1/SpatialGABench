"""Identity-only preflight for NU4 held-out RNA spatial programs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml

from spatialgabench.nm_human_method_setup import ANNOTATION_METHODS, SOFTWARE_METHODS


CONTROL = "accessibility_matched_random_peak_to_gene_control_pc"
STANDARD_REPRESENTATIONS = ["macs3_summit_501bp", "fixed_bin_10kb", "fixed_bin_25kb", "fixed_bin_50kb"]


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


def spatial_folds(coordinates: pd.DataFrame, bins: int = 5, folds: int = 5) -> pd.DataFrame:
    if not {"barcode", "row_coordinate", "column_coordinate"} <= set(coordinates.columns):
        raise ValueError("Coordinate table lacks required columns")
    output = coordinates.copy()
    output["row_bin"] = pd.qcut(
        output.row_coordinate.rank(method="first"), q=bins, labels=False
    ).astype(int)
    output["column_bin"] = pd.qcut(
        output.column_coordinate.rank(method="first"), q=bins, labels=False
    ).astype(int)
    output["fold"] = (output.row_bin + 2 * output.column_bin) % folds
    return output


def _positions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, compression="infer")
    if str(frame.iloc[0, 0]).strip().lower() == "barcode":
        frame = pd.read_csv(path, compression="infer")
        return pd.DataFrame({
            "barcode": frame.iloc[:, 0].astype(str),
            "row_coordinate": pd.to_numeric(frame.iloc[:, 4]),
            "column_coordinate": pd.to_numeric(frame.iloc[:, 5]),
        })
    return pd.DataFrame({
        "barcode": frame.iloc[:, 0].astype(str),
        "row_coordinate": pd.to_numeric(frame.iloc[:, 4]),
        "column_coordinate": pd.to_numeric(frame.iloc[:, 5]),
    })


def _rna_identities(sample: dict[str, Any], root: Path) -> tuple[list[str], list[str], tuple[int, int]]:
    fmt = sample["rna_format"]
    matrix_path = root / sample["rna_matrix"]["path"]
    if fmt == "tenx_mtx":
        with gzip.open(root / sample["rna_barcodes"]["path"], "rt", encoding="utf-8") as handle:
            barcodes = [line.strip().split("\t", 1)[0] for line in handle if line.strip()]
        features = pd.read_csv(
            root / sample["rna_features"]["path"], sep="\t", header=None, compression="gzip", dtype=str
        )
        genes = features.iloc[:, 0].astype(str).tolist()
        with gzip.open(matrix_path, "rt", encoding="utf-8") as handle:
            dimensions = None
            for line in handle:
                if not line.startswith("%"):
                    dimensions = tuple(int(value) for value in line.split())
                    break
        if dimensions is None:
            raise ValueError("MatrixMarket dimensions are absent")
        shape = (dimensions[0], dimensions[1])
    elif fmt == "wide_tsv_gzip":
        genes = []
        with gzip.open(matrix_path, "rt", encoding="utf-8") as handle:
            barcodes = handle.readline().rstrip("\n").split("\t")
            for line in handle:
                if line.strip():
                    gene, separator, _ = line.partition("\t")
                    if not separator:
                        raise ValueError("RNA TSV row lacks count fields")
                    genes.append(gene)
        shape = (len(genes), len(barcodes))
    elif fmt == "wide_csv_gzip":
        genes = []
        with gzip.open(matrix_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            barcodes = next(reader)
            if barcodes and barcodes[0] == "":
                barcodes = barcodes[1:]
            for row in reader:
                if row:
                    genes.append(row[0])
        shape = (len(genes), len(barcodes))
    elif fmt == "tenx_h5":
        with h5py.File(matrix_path, "r") as handle:
            matrix = handle["matrix"]
            barcodes = [value.decode() if isinstance(value, bytes) else str(value) for value in matrix["barcodes"][:]]
            feature_group = matrix["features"]
            key = "id" if "id" in feature_group else "name"
            genes = [value.decode() if isinstance(value, bytes) else str(value) for value in feature_group[key][:]]
            shape_value = tuple(int(value) for value in matrix["shape"][:])
        shape = (shape_value[0], shape_value[1])
    else:
        raise ValueError(f"Unsupported NU4 RNA format: {fmt}")
    if len(set(barcodes)) != len(barcodes) or len(set(genes)) != len(genes):
        raise ValueError("NU4 RNA identifiers are not unique")
    if shape != (len(genes), len(barcodes)):
        raise ValueError("NU4 RNA identity dimensions do not match")
    return barcodes, genes, shape


def _manifest_gene_activity(manifest_path: Path, root: Path) -> tuple[str, str]:
    manifest = _json(manifest_path)
    if isinstance(manifest.get("artifacts"), dict):
        path = manifest_path.parent / "gene_activity.h5ad"
        digest = manifest["artifacts"]["gene_activity.h5ad"]
    else:
        path = root / manifest["output_h5ad"]
        digest = manifest["output_h5ad_sha256"]
    if not path.is_file() or sha256_path(path) != digest:
        raise ValueError(f"NU4 method output hash mismatch: {path}")
    return str(path.relative_to(root)), digest


def _method_manifest(config: dict[str, Any], root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    common_annotation = ANNOTATION_METHODS
    g4_annotation = pd.read_csv(root / config["method_sources"]["g4_annotation_summary"], sep="\t")
    g4_software = pd.read_csv(root / config["method_sources"]["g4_software_summary"], sep="\t")
    for dataset_id in ("gse263333_5m_20um_atac", "gse205052_mousebrain_20um"):
        sample = config["samples"][dataset_id]
        selected = g4_annotation.loc[
            g4_annotation.dataset_id.eq(dataset_id)
            & g4_annotation.representation_id.isin(STANDARD_REPRESENTATIONS)
            & g4_annotation.method_id.isin(common_annotation)
            & g4_annotation.status.eq("pass")
        ]
        if len(selected) != 24:
            raise ValueError(f"NU4 G4 annotation panel incomplete: {dataset_id}")
        for source in selected.itertuples(index=False):
            path, digest = _manifest_gene_activity(root / source.output_manifest, root)
            rows.append({"dataset_id": dataset_id, "study_id": sample["study_id"], "study_family": sample["study_family"], "species": sample["species"], "genome_build": sample["genome_build"], "representation_id": source.representation_id, "input_track": "P" if str(source.representation_id).startswith("macs3") else "B", "method_id": source.method_id, "method_category": "matched_control" if source.method_id == CONTROL else "annotation", "gene_activity_h5ad": path, "gene_activity_sha256": digest})
        software = g4_software.loc[
            g4_software.dataset_id.eq(dataset_id)
            & g4_software.method_id.isin(SOFTWARE_METHODS)
            & g4_software.status.eq("pass")
        ]
        if len(software) != 3:
            raise ValueError(f"NU4 G4 software panel incomplete: {dataset_id}")
        for source in software.itertuples(index=False):
            manifest_path = root / "results/v1/g4/m2/software" / dataset_id / source.method_id / "output_manifest.json"
            path, digest = _manifest_gene_activity(manifest_path, root)
            rows.append({"dataset_id": dataset_id, "study_id": sample["study_id"], "study_family": sample["study_family"], "species": sample["species"], "genome_build": sample["genome_build"], "representation_id": "filtered_fragments_with_frozen_peak_observation_contract", "input_track": "F", "method_id": source.method_id, "method_category": "software", "gene_activity_h5ad": path, "gene_activity_sha256": digest})

    sample = config["samples"]["gse308623_p2s3"]
    external = pd.read_csv(root / config["method_sources"]["g5_external_summary"], sep="\t")
    external = external.loc[
        external.dataset_id.eq("gse308623_p2s3")
        & external.method_id.isin(common_annotation + SOFTWARE_METHODS)
        & external.status.eq("pass")
    ]
    if len(external) != 27:
        raise ValueError("NU4 GSE308623 panel incomplete")
    for source in external.itertuples(index=False):
        path = root / source.output_dir / "gene_activity.h5ad"
        digest = str(source.output_h5ad_sha256)
        if sha256_path(path) != digest:
            raise ValueError(f"NU4 external output hash mismatch: {path}")
        rows.append({"dataset_id": "gse308623_p2s3", "study_id": sample["study_id"], "study_family": sample["study_family"], "species": sample["species"], "genome_build": sample["genome_build"], "representation_id": source.representation_id, "input_track": source.input_track, "method_id": source.method_id, "method_category": "matched_control" if source.method_id == CONTROL else ("software" if source.run_type == "software" else "annotation"), "gene_activity_h5ad": str(path.relative_to(root)), "gene_activity_sha256": digest})

    human_evidence = _json(root / config["method_sources"]["nu1_human_completion"])
    if human_evidence.get("status") != "pass" or human_evidence.get("successful_contracts") != 50:
        raise ValueError("NU4 waits for all NU1/NU2 human construction contracts to pass")
    human = pd.read_csv(root / human_evidence["run_summary"]["path"], sep="\t")
    human = human.loc[human.dataset_id.eq("gse205055_humanbrain_50um") & human.outcome.eq("success")]
    if len(human) != 27:
        raise ValueError("NU4 NU1 human panel incomplete")
    sample = config["samples"]["gse205055_humanbrain_50um"]
    for source in human.itertuples(index=False):
        path = root / source.output_dir / "gene_activity.h5ad"
        digest = str(source.output_h5ad_sha256)
        if sha256_path(path) != digest:
            raise ValueError(f"NU4 NU1 output hash mismatch: {path}")
        rows.append({"dataset_id": "gse205055_humanbrain_50um", "study_id": sample["study_id"], "study_family": sample["study_family"], "species": sample["species"], "genome_build": sample["genome_build"], "representation_id": source.representation_id, "input_track": source.input_track, "method_id": source.method_id, "method_category": "matched_control" if source.method_id == CONTROL else ("software" if source.run_type == "software" else "annotation"), "gene_activity_h5ad": str(path.relative_to(root)), "gene_activity_sha256": digest})

    insertion = pd.read_csv(root / config["method_sources"]["fb4_insertion_manifest"], sep="\t")
    if len(insertion) != 6 or not insertion.status.eq("pass").all():
        raise ValueError("NU4 insertion panel incomplete")
    sample = config["samples"]["gse279771_gbm_insertion_track"]
    for source in insertion.itertuples(index=False):
        path = root / source.output_path
        if sha256_path(path) != source.output_sha256:
            raise ValueError(f"NU4 insertion output hash mismatch: {path}")
        rows.append({"dataset_id": "gse279771_gbm_insertion_track", "study_id": sample["study_id"], "study_family": sample["study_family"], "species": sample["species"], "genome_build": sample["genome_build"], "representation_id": source.representation_id, "input_track": "I", "method_id": source.method_id, "method_category": "matched_control" if source.method_id == CONTROL else "annotation", "gene_activity_h5ad": source.output_path, "gene_activity_sha256": source.output_sha256})

    frame = pd.DataFrame(rows).sort_values(["dataset_id", "input_track", "representation_id", "method_id"], kind="mergesort").reset_index(drop=True)
    if len(frame) != int(config["expected_method_contracts"]) or frame.duplicated(["dataset_id", "representation_id", "method_id"]).any():
        raise ValueError("NU4 exact method contract panel changed")
    return frame


def run(config_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path, authorization_path = Path(root).resolve(), Path(config_path).resolve(), Path(authorization_path).resolve()
    config, authorization = _yaml(config_path), _json(authorization_path)
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "NU4_spatial_program_identity_preflight"
        or authorization.get("rna_expression_value_access_authorized") is not False
        or authorization.get("gene_activity_value_access_authorized") is not False
        or authorization.get("labels_clusters_pathology_access_authorized") is not False
        or authorization.get("evaluation_authorized") is not False
        or authorization.get("method_rerun_authorized") is not False
        or authorization.get("construction_tuning_authorized") is not False
        or authorization.get("policy_refit_authorized") is not False
        or authorization.get("universal_ranking_authorized") is not False
        or authorization.get("final_ranking_authorized") is not False
    ):
        raise PermissionError("NU4 identity-only preflight authorization is invalid")
    bindings = {
        "config": config_path,
        "implementation": Path(__file__),
        "upgrade_protocol": root / config["upgrade_protocol"],
    }
    for name, path in bindings.items():
        record = authorization[name]
        if record.get("path") != str(path.resolve().relative_to(root)) or record.get("sha256") != sha256_path(path):
            raise ValueError(f"NU4 preflight frozen artifact mismatch: {name}")
    if config.get("rna_used_for_construction") is not False or config.get("labels_or_clusters_allowed") is not False or config.get("pathology_allowed") is not False:
        raise PermissionError("NU4 preflight information boundary is invalid")

    method_manifest = _method_manifest(config, root)
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    method_path = output / "method_input_manifest.tsv"
    method_manifest.to_csv(method_path, sep="\t", index=False)
    data_rows: list[dict[str, Any]] = []
    fold_paths: dict[str, dict[str, str]] = {}
    for dataset_id, sample in config["samples"].items():
        for key in ("rna_matrix", "positions", "gene_annotation"):
            record = sample[key]
            if sha256_path(root / record["path"]) != record["sha256"]:
                raise ValueError(f"NU4 frozen data hash mismatch: {dataset_id}/{key}")
        for key in ("rna_features", "rna_barcodes"):
            if key in sample and sha256_path(root / sample[key]["path"]) != sample[key]["sha256"]:
                raise ValueError(f"NU4 frozen data hash mismatch: {dataset_id}/{key}")
        rna_barcodes, genes, shape = _rna_identities(sample, root)
        positions = _positions(root / sample["positions"]["path"])
        if positions.barcode.duplicated().any():
            raise ValueError(f"NU4 positions contain duplicate barcodes: {dataset_id}")
        first = method_manifest.loc[method_manifest.dataset_id.eq(dataset_id)].iloc[0]
        activity = ad.read_h5ad(root / first.gene_activity_h5ad, backed="r")
        activity_barcodes = activity.obs_names.astype(str).tolist()
        activity.file.close()
        rna_set, position_set = set(rna_barcodes), set(positions.barcode)
        evaluation_barcodes = [barcode for barcode in activity_barcodes if barcode in rna_set and barcode in position_set]
        if len(evaluation_barcodes) != int(sample["expected_evaluation_observations"]):
            raise ValueError(f"NU4 evaluation observation count changed: {dataset_id}")
        selected = positions.set_index("barcode").loc[evaluation_barcodes].reset_index()
        folded = spatial_folds(selected, int(config["folds"]["coordinate_quantile_bins_per_axis"]), int(config["folds"]["count"]))
        counts = folded.fold.value_counts()
        if set(counts.index) != set(range(int(config["folds"]["count"]))) or counts.min() < int(config["folds"]["minimum_test_observations"]):
            raise ValueError(f"NU4 spatial fold sizes are inadmissible: {dataset_id}")
        fold_path = output / f"folds/{dataset_id}.tsv"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        folded.to_csv(fold_path, sep="\t", index=False)
        fold_paths[dataset_id] = {"path": str(fold_path.relative_to(root)), "sha256": sha256_path(fold_path)}
        data_rows.append({
            "dataset_id": dataset_id,
            "study_id": sample["study_id"],
            "study_family": sample["study_family"],
            "species": sample["species"],
            "genome_build": sample["genome_build"],
            "rna_format": sample["rna_format"],
            "rna_gene_identities": len(genes),
            "rna_barcode_identities": len(rna_barcodes),
            "rna_matrix_rows": shape[0],
            "rna_matrix_columns": shape[1],
            "activity_observations": len(activity_barcodes),
            "evaluation_observations": len(evaluation_barcodes),
            "fold_minimum_size": int(counts.min()),
            "fold_maximum_size": int(counts.max()),
            "status": "pass",
        })

    data_path = output / "data_identity_manifest.tsv"
    pd.DataFrame(data_rows).to_csv(data_path, sep="\t", index=False)
    implementation_files = [
        "spatialgabench/nm_spatial_program_evaluation.py",
        "spatialgabench/nm_spatial_program_preflight.py",
        "spatialgabench/g4_m3b_rna.py",
        "spatialgabench/g4_m3h_rna_replication.py",
        "spatialgabench/g5_h5c_evaluation.py",
        "spatialgabench/full_benchmark_gse279771.py",
    ]
    implementation = pd.DataFrame([
        {"path": path, "sha256": sha256_path(root / path)} for path in implementation_files
    ])
    implementation_path = output / "implementation_manifest.tsv"
    implementation.to_csv(implementation_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "milestone": "NU4_spatial_program_identity_preflight",
        "status": "pass",
        "samples": len(data_rows),
        "study_families": len({row["study_family"] for row in data_rows}),
        "method_contracts": len(method_manifest),
        "folds_per_sample": int(config["folds"]["count"]),
        "rna_expression_values_read": False,
        "rna_identity_fields_read": True,
        "spatial_coordinates_read": True,
        "gene_activity_values_read": False,
        "labels_clusters_pathology_read": False,
        "evaluation_run": False,
        "method_rerun": False,
        "policy_refit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "preflight_authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "method_input_manifest": {"path": str(method_path.relative_to(root)), "sha256": sha256_path(method_path)},
        "data_identity_manifest": {"path": str(data_path.relative_to(root)), "sha256": sha256_path(data_path)},
        "fold_assignments": fold_paths,
        "implementation_manifest": {"path": str(implementation_path.relative_to(root)), "sha256": sha256_path(implementation_path)},
        "implementation": {"path": "spatialgabench/nm_spatial_program_preflight.py", "sha256": sha256_path(Path(__file__))},
    }
    evidence_path = output / "preflight_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def authorize(preflight_path: str | Path, config_path: str | Path, root: str | Path = ".") -> Path:
    root, preflight_path, config_path = Path(root).resolve(), Path(preflight_path).resolve(), Path(config_path).resolve()
    preflight = _json(preflight_path)
    if preflight.get("status") != "pass" or preflight.get("rna_expression_values_read") is not False or preflight.get("evaluation_run") is not False:
        raise PermissionError("NU4 evaluation cannot be authorized from this preflight")
    implementation = pd.read_csv(root / preflight["implementation_manifest"]["path"], sep="\t")
    for row in implementation.itertuples(index=False):
        if sha256_path(root / row.path) != row.sha256:
            raise ValueError(f"NU4 implementation changed before authorization: {row.path}")
    authorization = {
        "schema_version": 1,
        "authorization": "NM_U4_SPATIAL_PROGRAM_EVALUATION",
        "milestone": "NU4_heldout_RNA_spatial_program_preservation",
        "authorization_date": "2026-07-30",
        "status": "authorized",
        "scope": "frozen_heldout_spatial_program_evaluation_only",
        "rna_expression_value_access_authorized": True,
        "spatial_program_evaluation_authorized": True,
        "method_rerun_authorized": False,
        "construction_tuning_authorized": False,
        "labels_clusters_pathology_access_authorized": False,
        "policy_refit_authorized": False,
        "results_writing_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "preflight_evidence": {"path": str(preflight_path.relative_to(root)), "sha256": sha256_path(preflight_path)},
        "method_input_manifest": preflight["method_input_manifest"],
        "data_identity_manifest": preflight["data_identity_manifest"],
        "implementation_manifest": preflight["implementation_manifest"],
        "fold_assignments": preflight["fold_assignments"],
    }
    path = root / "governance/authorizations/NM_U4_SPATIAL_PROGRAM_EVALUATION_AUTHORIZATION_2026-07-30.json"
    path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "authorize"], nargs="?", default="run")
    parser.add_argument("--config", default="configs/nm_upgrade_spatial_programs.yaml")
    parser.add_argument("--authorization", default="governance/authorizations/NM_U4_SPATIAL_PROGRAM_PREFLIGHT_AUTHORIZATION_2026-07-30.json")
    parser.add_argument("--preflight", default="results/v1/nm_upgrade/nu4_spatial_programs/preflight_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "run":
        print(run(args.config, args.authorization, args.root))
    else:
        print(authorize(args.preflight, args.config, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
