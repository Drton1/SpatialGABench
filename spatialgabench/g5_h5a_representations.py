"""Preflight and summarize authorized G5-H5A external representations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import yaml

from spatialgabench.representations.fragments import _read_in_tissue_barcodes
from spatialgabench.representations.guard import assert_development_study


FALSE_FIELDS = (
    "method_execution_run",
    "rna_value_access_run",
    "label_or_pathology_access_run",
    "holdout_utility_access_run",
    "regret_computation_run",
    "policy_refit_run",
    "GSE279771_accessed",
    "full_benchmark_run",
    "results_writing_run",
    "universal_ranking_run",
    "final_ranking_run",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def _load(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("H5A config must be a mapping")
    for key in ("execution_authorization", "previous_gate"):
        path = root / config[key]
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"H5A frozen artifact mismatch: {key}")
    authorization = _json(root / config["execution_authorization"])
    previous = _json(root / config["previous_gate"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "H5A_external_representation_construction"
        or authorization.get("previous_gate_sha256") != config["previous_gate_sha256"]
        or authorization.get("representation_construction_authorized") is not True
        or authorization.get("method_execution_authorized") is not False
        or authorization.get("rna_value_access_authorized") is not False
    ):
        raise PermissionError("H5A authorization is invalid")
    if (
        previous.get("status") != "pass"
        or previous.get("g5_execution_authorized") is not False
        or previous.get("planned_outputs_authorized") != 0
    ):
        raise ValueError("H4 disabled protocol boundary is invalid")
    invalid = [field for field in FALSE_FIELDS if config.get(field) is not False]
    if invalid:
        raise PermissionError(f"H5A forbidden flags are not false: {invalid}")
    return config, authorization


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config, authorization = _load(config_path, root)
    rows: list[dict[str, Any]] = []
    for dataset_id, sample in config["samples"].items():
        guard = assert_development_study(config_path, sample["study_id"])
        fragment_path = root / sample["fragments"]
        positions_path = root / sample["positions"]
        chrom_sizes = root / sample["chrom_sizes"]
        blacklist = root / sample["blacklist"]
        checks = {
            "fragment_sha256": fragment_path.is_file() and sha256(fragment_path) == sample["source_sha256"],
            "positions_sha256": positions_path.is_file() and sha256(positions_path) == sample["positions_sha256"],
            "chrom_sizes_sha256": chrom_sizes.is_file() and sha256(chrom_sizes) == sample["chrom_sizes_sha256"],
            "blacklist_sha256": blacklist.is_file() and sha256(blacklist) == sample["blacklist_sha256"],
            "study_authorized": guard.get("authorization") == "g5_external_representation_construction",
            "observation_contract": len(_read_in_tissue_barcodes(positions_path)) == sample["expected_observations"],
            "accession_authorized": sample["accession"] in authorization["authorized_accessions"],
            "fragment_authorized": authorization["authorized_fragment_sha256"][sample["study_id"]]
            == sample["source_sha256"],
        }
        rows.append(
            {
                "dataset_id": dataset_id,
                "study_id": sample["study_id"],
                "accession": sample["accession"],
                "sample_id": sample["sample_id"],
                "expected_observations": sample["expected_observations"],
                "source_size_bytes": fragment_path.stat().st_size if fragment_path.is_file() else 0,
                "checks_passed": sum(checks.values()),
                "checks_total": len(checks),
                "failed_checks": ";".join(key for key, value in checks.items() if not value),
                "status": "pass" if all(checks.values()) else "fail",
            }
        )
    output = root / config["evidence_dir"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "input_preflight_manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)
    disk = os.statvfs(root)
    evidence = {
        "schema_version": 1,
        "gate": "G5",
        "milestone": "H5A_external_representation_preflight",
        "status": "pass" if len(rows) == 2 and all(row["status"] == "pass" for row in rows) else "blocked",
        "studies": len(rows),
        "preflight_checks_passed": sum(row["checks_passed"] for row in rows),
        "preflight_checks_total": sum(row["checks_total"] for row in rows),
        "disk_available_bytes": disk.f_bavail * disk.f_frsize,
        "minimum_disk_required_bytes": 200_000_000_000,
        "disk_ready": disk.f_bavail * disk.f_frsize >= 200_000_000_000,
        "input_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256(manifest_path)},
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "authorization": {"path": config["execution_authorization"], "sha256": config["execution_authorization_sha256"]},
        "raw_holdout_checksum_revalidation_run": True,
        "representation_construction_run": False,
        "method_execution_run": False,
        "rna_value_access_run": False,
        "holdout_utility_access_run": False,
        "regret_computation_run": False,
        "policy_refit_run": False,
        "GSE279771_accessed": False,
        "full_benchmark_run": False,
        "results_writing_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    evidence_path = output / "input_preflight_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if evidence["status"] != "pass" or not evidence["disk_ready"]:
        raise ValueError("H5A input/resource preflight failed")
    return evidence_path


def summarize(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config, _ = _load(config_path, root)
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset_id, sample in config["samples"].items():
        base = root / config["output_root"] / dataset_id
        filter_path = base / "filtered_fragments/fragment_filter_audit.json"
        peak_path = base / "frozen_peaks/peak_freeze_audit.json"
        validation_path = base / "matched/matched_representation_validation.json"
        manifest_path = base / "matched/representation_manifest.tsv"
        if not all(path.is_file() for path in (filter_path, peak_path, validation_path, manifest_path)):
            failures.append(f"{dataset_id}: missing construction artifacts")
            continue
        fragment_filter = _json(filter_path)
        peak = _json(peak_path)
        validation = _json(validation_path)
        manifest = pd.read_csv(manifest_path, sep="\t")
        source_rows = fragment_filter.get("source_fragment_rows")
        included_rows = fragment_filter.get("included_fragment_rows")
        excluded_barcode_rows = fragment_filter.get("excluded_barcode_rows")
        excluded_chromosome_rows = fragment_filter.get("excluded_chromosome_rows")
        invalid_rows = fragment_filter.get("invalid_fragment_rows")
        filter_partition_conserved = (
            all(
                isinstance(value, int)
                for value in (
                    source_rows,
                    included_rows,
                    excluded_barcode_rows,
                    excluded_chromosome_rows,
                    invalid_rows,
                )
            )
            and source_rows == included_rows + excluded_barcode_rows + excluded_chromosome_rows + invalid_rows
        )
        fixed_bin_rows = manifest[manifest["representation_id"].astype(str).str.startswith("fixed_bin_")]
        fixed_bin_counts_conserved = (
            len(fixed_bin_rows) == 3
            and set(fixed_bin_rows["matrix_total_counts"].astype(int)) == {included_rows}
            and set(fixed_bin_rows["assigned_fragment_rows"].astype(int)) == {included_rows}
        )
        representation_tracks_exact = set(
            zip(manifest["representation_id"].astype(str), manifest["input_track"].astype(str), strict=True)
        ) == {
            ("fixed_bin_10kb", "B"),
            ("fixed_bin_25kb", "B"),
            ("fixed_bin_50kb", "B"),
            ("macs3_summit_501bp", "P"),
        }
        sample_pass = (
            fragment_filter.get("status") == "pass"
            and fragment_filter.get("in_tissue_barcode_rows") == sample["expected_observations"]
            and invalid_rows == 0
            and filter_partition_conserved
            and peak.get("status") == "pass"
            and validation.get("status") == "pass"
            and validation.get("barcode_rows") == sample["expected_observations"]
            and len(manifest) == 4
            and set(manifest["representation_id"]) == set(config["construction_contract"]["representations"])
            and set(manifest["representation_source"]) == {"same_filtered_fragments"}
            and set(manifest["comparison_scope"]) == {"matched_fragment_derived"}
            and set(manifest["counting_unit"]) == {config["construction_contract"]["counting_unit"]}
            and fixed_bin_counts_conserved
            and representation_tracks_exact
        )
        samples.append(
            {
                "dataset_id": dataset_id,
                "study_id": sample["study_id"],
                "sample_id": sample["sample_id"],
                "n_observations": validation.get("barcode_rows"),
                "source_fragment_rows": fragment_filter.get("source_fragment_rows"),
                "included_fragment_rows": fragment_filter.get("included_fragment_rows"),
                "excluded_barcode_rows": fragment_filter.get("excluded_barcode_rows"),
                "excluded_chromosome_rows": fragment_filter.get("excluded_chromosome_rows"),
                "invalid_fragment_rows": fragment_filter.get("invalid_fragment_rows"),
                "frozen_peak_rows": peak.get("frozen_peak_rows"),
                "filter_partition_conserved": filter_partition_conserved,
                "fixed_bin_counts_conserved": fixed_bin_counts_conserved,
                "representation_tracks_exact": representation_tracks_exact,
                "status": "pass" if sample_pass else "fail",
            }
        )
        observation_order: tuple[str, ...] | None = None
        for record in manifest.to_dict(orient="records"):
            output_path = root / record["output_path"]
            data = ad.read_h5ad(output_path, backed="r") if output_path.is_file() else None
            current_order = tuple(data.obs_names.astype(str)) if data is not None else None
            if observation_order is None and current_order is not None:
                observation_order = current_order
            observation_order_matched = current_order is not None and current_order == observation_order
            valid = (
                output_path.is_file()
                and sha256(output_path) == record["output_sha256"]
                and data is not None
                and data.n_obs == sample["expected_observations"]
                and data.obs_names.is_unique
                and data.uns.get("construction_leakage_inputs") == "none"
                and observation_order_matched
            )
            if data is not None:
                data.file.close()
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "study_id": sample["study_id"],
                    "sample_id": sample["sample_id"],
                    "species": sample["species"],
                    "genome_build": sample["genome_build"],
                    "representation_id": record["representation_id"],
                    "representation_source": record["representation_source"],
                    "input_track": record["input_track"],
                    "comparison_scope": record["comparison_scope"],
                    "counting_unit": record["counting_unit"],
                    "n_observations": record["n_observations"],
                    "n_features": record["n_features"],
                    "matrix_total_counts": record["matrix_total_counts"],
                    "output_path": record["output_path"],
                    "output_sha256": record["output_sha256"],
                    "observation_order_matched": observation_order_matched,
                    "construction_leakage_inputs": "none" if valid else "contract_failed",
                    "status": "pass" if valid and record["status"] == "pass" else "fail",
                }
            )
        if not sample_pass:
            failures.append(f"{dataset_id}: construction contract failed")
    evidence_dir = root / config["evidence_dir"]
    sample_path = evidence_dir / "sample_summary.tsv"
    representation_path = evidence_dir / "representation_summary.tsv"
    pd.DataFrame(samples).to_csv(sample_path, sep="\t", index=False)
    pd.DataFrame(rows).to_csv(representation_path, sep="\t", index=False)
    passed = len(samples) == 2 and len(rows) == 8 and not failures and all(row["status"] == "pass" for row in rows)
    evidence = {
        "schema_version": 1,
        "gate": "G5",
        "milestone": "H5A_external_representation_construction",
        "status": "pass" if passed else "blocked",
        "studies": len(samples),
        "representations": len(rows),
        "holdout_inputs_accessed": True,
        "raw_holdout_checksum_revalidation_run": True,
        "representation_construction_run": True,
        "method_outputs_authorized": 0,
        "method_execution_run": False,
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
        "failures": failures,
        "sample_summary": {"path": str(sample_path.relative_to(root)), "sha256": sha256(sample_path)},
        "representation_summary": {
            "path": str(representation_path.relative_to(root)),
            "sha256": sha256(representation_path),
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
    }
    evidence_path = evidence_dir / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError(f"H5A representation construction failed: {failures}")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "summarize"):
        child = sub.add_parser(name)
        child.add_argument("--config", default="configs/g5_h5a_external_representations.yaml")
        child.add_argument("--root", default=".")
    args = parser.parse_args()
    output = preflight(args.config, args.root) if args.command == "preflight" else summarize(args.config, args.root)
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
