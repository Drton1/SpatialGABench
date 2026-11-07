"""Peak-free R2 input construction and locked two-study ATAC transfer."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from spatialgabench.gene_activity.atomic_multiresolution import (
    fit_preprojected_atomic_multiresolution,
    prepare_atomic_weight_panel,
)
from spatialgabench.nm_peakfree_r1 import sha256_path
from spatialgabench.representations.matched import (
    FeatureLookup,
    _flush_chunk,
    _make_bin_features,
    _read_positions,
    _read_standard_chrom_sizes,
    _normalize_barcode,
)


def _load(path: Path) -> dict[str, Any]:
    loader = json.loads if path.suffix == ".json" else yaml.safe_load
    value = loader(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def depth_fraction_from_condition_id(condition_id: str) -> float:
    try:
        code = condition_id.split("__d", 1)[1].split("__", 1)[0]
    except IndexError as error:
        raise ValueError(f"condition has no depth code: {condition_id}") from error
    if len(code) != 3 or not code.isdigit():
        raise ValueError(f"invalid depth code: {condition_id}")
    return int(code) / 100.0


def _completed_manifest(condition: dict[str, Any], condition_root: Path, root: Path) -> pd.DataFrame | None:
    path = condition_root / "matrix_manifest.tsv"
    if not path.is_file():
        return None
    manifest = pd.read_csv(path, sep="\t")
    expected = {"adaptive_v1_1", "fixed_5kb", "fixed_25kb", "fixed_100kb", "count_hard_switch"}
    if set(manifest.branch_id) != expected or len(manifest) != len(expected):
        return None
    if not all((root / row.path).is_file() and row.sha256 == sha256_path(root / row.path) for row in manifest.itertuples(index=False)):
        return None
    if not manifest.observations.eq(int(condition["observations"])).all():
        return None
    return manifest


def _condition_summary(condition: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": condition["condition_id"], "study_id": condition["study_id"],
        "dataset_id": condition["dataset_id"],
        "depth_fraction": depth_fraction_from_condition_id(condition["condition_id"]),
        "seed": int(0 if condition["condition_id"].endswith("__s0") else config["seed"]),
        "observations": int(condition["observations"]),
        "fragment_events": int(condition["expected_fragment_events"]),
        "branches": 5, "status": "pass",
    }


def _write_atomic_input(condition: dict[str, Any], output: Path, root: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "atomic_5kb.h5ad"
    if path.is_file():
        return path
    positions = _read_positions(root / condition["positions"])
    chrom_sizes = _read_standard_chrom_sizes(root / condition["chrom_sizes"])
    features = _make_bin_features(chrom_sizes, 5_000)
    lookup = FeatureLookup("fixed_bin_5kb", features)
    matrices = {"fixed_bin_5kb": sparse.csr_matrix((len(positions), len(features)), dtype=np.int32)}
    assigned = {"fixed_bin_5kb": 0}
    barcode_to_row = {barcode: index for index, barcode in enumerate(positions.index.astype(str))}
    rows: list[int] = []
    chromosomes: list[str] = []
    midpoints: list[int] = []
    opener = gzip.open if str(condition["fragments"]).endswith(".gz") else open
    with opener(root / condition["fragments"], "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError:
                continue
            chromosome = fields[0]
            barcode = _normalize_barcode(fields[3])
            if (
                end <= start or start < 0 or barcode not in barcode_to_row
                or chromosome not in chrom_sizes or end > chrom_sizes[chromosome]
            ):
                continue
            rows.append(barcode_to_row[barcode])
            chromosomes.append(chromosome)
            midpoints.append((start + end) // 2)
            if len(rows) >= 500_000:
                _flush_chunk(
                    rows, chromosomes, midpoints,
                    matrices, {"fixed_bin_5kb": lookup}, assigned,
                )
                rows.clear(); chromosomes.clear(); midpoints.clear()
    if rows:
        _flush_chunk(rows, chromosomes, midpoints, matrices, {"fixed_bin_5kb": lookup}, assigned)
    matrix = matrices["fixed_bin_5kb"]
    if int(matrix.sum()) != int(condition["expected_fragment_events"]):
        raise ValueError(f"atomic fragment count mismatch: {condition['condition_id']}")
    obs = positions.copy(); obs.index.name = "barcode"
    adata = ad.AnnData(X=matrix, obs=obs, var=features)
    adata.obsm["spatial"] = obs[["pixel_col", "pixel_row"]].to_numpy(dtype=float)
    adata.uns.update({
        "study_id": condition["study_id"], "dataset_id": condition["condition_id"],
        "representation_id": "fixed_bin_5kb", "input_track": "B",
        "representation_source": "same_filtered_fragments_peak_free",
        "genome_build": condition["genome_build"],
        "counting_unit": "one_filtered_fragment_row_midpoint",
        "source_fragments": condition["fragments"],
        "source_fragments_sha256": sha256_path(root / condition["fragments"]),
        "construction_leakage_inputs": "none",
    })
    adata.write_h5ad(path, compression="gzip")
    return path


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _load(config_path)
    required = [config["protocol"], config["amendment"], config["r1_validation"], config["atomic_gene_kernel"], config["genes"], config["hard_switch_contract"], config["method_core"], config["transfer_runner"]]
    for relative in required:
        if not (root / relative).is_file():
            raise ValueError(f"R2 input missing: {relative}")
    r1 = _load(root / config["r1_validation"])
    if r1.get("status") != "pass" or r1.get("validated_decision") != "authorize_locked_transfer_panel":
        raise PermissionError("R1 did not authorize R2")
    for condition in config["conditions"]:
        for key in ("fragments", "positions", "chrom_sizes", "fragment_audit"):
            if not (root / condition[key]).is_file():
                raise ValueError(f"R2 input missing: {condition['condition_id']}/{key}")
    output = root / config["output_root"]; output.mkdir(parents=True, exist_ok=True)
    bindings = required + [
        value for condition in config["conditions"] for value in (
            condition["fragments"], condition["positions"], condition["chrom_sizes"], condition["fragment_audit"]
        )
    ]
    binding_path = output / "transfer_bindings.tsv"
    pd.DataFrame([{"path": value, "sha256": sha256_path(root / value)} for value in bindings]).to_csv(binding_path, sep="\t", index=False)
    auth = {
        "schema_version": 1, "authorization": "NM_PEAKFREE_R2_TRANSFER_CONSTRUCTION",
        "milestone": config["milestone"], "status": "authorized",
        "atac_construction_authorized": True, "rna_value_access_authorized": False,
        "parameter_tuning_authorized": False, "labels_clusters_pathology_authorized": False,
        "universal_ranking_authorized": False, "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "bindings": {"path": str(binding_path.relative_to(root)), "sha256": sha256_path(binding_path)},
    }
    auth_path = root / "governance/authorizations/NM_PEAKFREE_R2_TRANSFER_CONSTRUCTION_2026-08-02.json"
    auth_path.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n")
    evidence = {"schema_version": 1, "milestone": f"{config['milestone']}_PREFLIGHT", "status": "pass", "conditions": len(config["conditions"]), "rna_expression_values_read": False, "authorization": str(auth_path.relative_to(root))}
    path = output / "preflight_evidence.json"; path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


def _validate_auth(config_path: Path, auth_path: Path, root: Path) -> dict[str, Any]:
    config, auth = _load(config_path), _load(auth_path)
    if auth.get("status") != "authorized" or auth.get("milestone") != config["milestone"] or auth.get("rna_value_access_authorized") is not False or auth["config"]["sha256"] != sha256_path(config_path):
        raise PermissionError("R2 authorization invalid")
    bindings = pd.read_csv(root / auth["bindings"]["path"], sep="\t")
    if auth["bindings"]["sha256"] != sha256_path(root / auth["bindings"]["path"]):
        raise ValueError("R2 bindings changed")
    for row in bindings.itertuples(index=False):
        if row.sha256 != sha256_path(root / row.path):
            raise ValueError(f"R2 bound input changed: {row.path}")
    return config


def run(config_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _validate_auth(config_path, Path(authorization_path).resolve(), root)
    output = root / config["output_root"]
    weights = sparse.load_npz(root / config["atomic_gene_kernel"]).tocsr()
    genes = pd.read_csv(root / config["genes"], sep="\t")
    switch = _load(root / config["hard_switch_contract"])
    condition_rows = []
    for condition in config["conditions"]:
        condition_root = output / "conditions" / condition["condition_id"]
        completed = _completed_manifest(condition, condition_root, root)
        if completed is not None:
            condition_rows.append(_condition_summary(condition, config))
            continue
        atomic_path = _write_atomic_input(condition, condition_root / "input", root)
        data = ad.read_h5ad(atomic_path)
        if data.n_obs != int(condition["observations"]) or int(data.X.sum()) != int(condition["expected_fragment_events"]):
            raise ValueError(f"R2 atomic input contract failed: {condition['condition_id']}")
        panel = prepare_atomic_weight_panel(data.var, weights, scales_kb=(5, 25, 100))
        chunks = {name: [] for name in ("adaptive_v1_1", "fixed_5kb", "fixed_25kb", "fixed_100kb", "count_hard_switch")}
        for start in range(0, weights.shape[1], int(config["gene_chunk_size"])):
            stop = min(start + int(config["gene_chunk_size"]), weights.shape[1])
            result = fit_preprojected_atomic_multiresolution(
                data.X, {scale: value[:, start:stop] for scale, value in panel.weights_by_scale.items()},
                panel.boundary_not_applicable[start:stop], scales_kb=(5, 25, 100),
                scale_factor=float(config["scale_factor"]), alpha=float(config["alpha"]),
            )
            chunks["adaptive_v1_1"].append(sparse.csr_matrix(result.activity_rate))
            for scale in (5, 25, 100):
                chunks[f"fixed_{scale}kb"].append(sparse.csr_matrix(np.expm1(result.scale_log_rates[scale]) / float(config["scale_factor"])))
        matrices = {name: sparse.hstack(values, format="csr") for name, values in chunks.items() if name != "count_hard_switch"}
        totals = np.asarray(data.X.sum(axis=1)).ravel().astype(int)
        assigned = np.where(totals <= int(switch["low_count_max_for_100kb"]), 100, np.where(totals <= int(switch["middle_count_max_for_25kb"]), 25, 5))
        hard = sparse.csr_matrix(matrices["fixed_100kb"].shape, dtype=float)
        for scale in (5, 25, 100):
            hard = hard + sparse.diags((assigned == scale).astype(float)) @ matrices[f"fixed_{scale}kb"]
        matrices["count_hard_switch"] = hard.tocsr()
        matrix_dir = condition_root / "matrices"; matrix_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, matrix in matrices.items():
            path = matrix_dir / f"{name}.npz"; sparse.save_npz(path, matrix, compressed=True)
            rows.append({"condition_id": condition["condition_id"], "branch_id": name, "observations": matrix.shape[0], "genes": matrix.shape[1], "nnz": matrix.nnz, "path": str(path.relative_to(root)), "sha256": sha256_path(path)})
        pd.DataFrame(rows).to_csv(condition_root / "matrix_manifest.tsv", sep="\t", index=False)
        condition_rows.append(_condition_summary(condition, config))
    manifest = pd.concat([pd.read_csv(output / "conditions" / row["condition_id"] / "matrix_manifest.tsv", sep="\t") for row in condition_rows], ignore_index=True)
    manifest.to_csv(output / "transfer_matrix_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(condition_rows).to_csv(output / "transfer_condition_summary.tsv", sep="\t", index=False)
    evidence = {"schema_version": 1, "milestone": f"{config['milestone']}_CONSTRUCTION", "status": "pass", "conditions": len(condition_rows), "branches": len(manifest), "rna_expression_values_read": False, "labels_clusters_pathology_read": False, "parameter_tuning_run": False, "universal_ranking_run": False, "final_ranking_run": False, "artifacts": {"transfer_matrix_manifest.tsv": {"path": str((output / "transfer_matrix_manifest.tsv").relative_to(root)), "sha256": sha256_path(output / "transfer_matrix_manifest.tsv")}, "transfer_condition_summary.tsv": {"path": str((output / "transfer_condition_summary.tsv").relative_to(root)), "sha256": sha256_path(output / "transfer_condition_summary.tsv")}}}
    path = output / "completion_evidence.json"; path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["preflight", "run"]); parser.add_argument("--config", default="configs/nm_peakfree_r2_transfer.yaml"); parser.add_argument("--authorization", default="governance/authorizations/NM_PEAKFREE_R2_TRANSFER_CONSTRUCTION_2026-08-02.json"); parser.add_argument("--root", default="."); args = parser.parse_args()
    result = preflight(args.config, args.root) if args.command == "preflight" else run(args.config, args.authorization, args.root)
    print(result); return 0


if __name__ == "__main__":
    raise SystemExit(main())
