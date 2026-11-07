"""Authorized transfer preflight and byte-integrity verification for GSE307620."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


FALSE_FIELDS = (
    "hic_download_run",
    "fastq_or_bam_download_run",
    "image_download_run",
    "existing_holdout_accessed",
    "fragment_content_analysis_run",
    "representation_construction_run",
    "method_execution_run",
    "rna_or_label_access_run",
    "policy_refit_run",
    "regret_evaluation_run",
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


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _yaml(config_path)
    for key in ("authorization", "previous_gate", "sample_metadata"):
        path = root / config[key]
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"G5-H2 frozen artifact mismatch: {key}")
    authorization = _json(root / config["authorization"])
    gate = _json(root / config["previous_gate"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "H2_GSE307620_minimal_fragment_download"
        or authorization.get("authorized_file") != config["file_name"]
        or authorization.get("authorized_expected_bytes") != config["expected_bytes"]
        or authorization.get("network_transfer_authorized") is not True
        or authorization.get("fragment_content_analysis_authorized") is not False
    ):
        raise PermissionError("G5-H2 minimal-download authorization is invalid")
    if (
        gate.get("status") != "pass"
        or gate.get("supplemental_holdout_frozen") is not True
        or gate.get("selected_accession") != config["accession"]
        or gate.get("selected_sample_accession") != config["sample_accession"]
        or gate.get("large_fragment_download_run") is not False
    ):
        raise ValueError("G5-H1 supplemental freeze boundary is invalid")
    invalid = [field for field in FALSE_FIELDS if config.get(field) is not False]
    if invalid:
        raise PermissionError(f"G5-H2 forbidden execution flags are not false: {invalid}")
    metadata = (root / config["sample_metadata"]).read_text(encoding="utf-8")
    if config["url"].replace("https://", "ftp://") not in metadata:
        raise ValueError("Authorized URL is not declared by frozen GEO sample metadata")
    return config, authorization


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config, authorization = _load(config_path, root)
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    target = root / config["output_path"]
    payload = {
        "schema_version": 1,
        "gate": "G5",
        "milestone": "H2_GSE307620_minimal_fragment_download_preflight",
        "status": "pass",
        "authorized_url": config["url"],
        "authorized_output_path": config["output_path"],
        "authorized_expected_bytes": config["expected_bytes"],
        "partial_or_complete_target_bytes_before_transfer": target.stat().st_size if target.exists() else 0,
        "resume_allowed": authorization["resume_partial_transfer_authorized"],
        "large_fragment_download_started_by_preflight": False,
        "existing_holdout_accessed": False,
        "representation_construction_run": False,
        "method_execution_run": False,
        "policy_refit_run": False,
    }
    path = output / "download_preflight.json"
    _write(path, payload)
    return path


def _gzip_integrity(path: Path) -> tuple[bool, int]:
    uncompressed_bytes = 0
    try:
        with gzip.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                uncompressed_bytes += len(chunk)
    except (OSError, EOFError):
        return False, uncompressed_bytes
    return True, uncompressed_bytes


def verify(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config, _ = _load(config_path, root)
    target = root / config["output_path"]
    if not target.is_file():
        raise FileNotFoundError(f"Authorized GSE307620 transfer is absent: {target}")
    observed_bytes = target.stat().st_size
    size_pass = observed_bytes == config["expected_bytes"]
    observed_sha256 = sha256(target) if size_pass else ""
    gzip_pass, uncompressed_bytes = _gzip_integrity(target) if size_pass else (False, 0)
    status = "pass" if size_pass and len(observed_sha256) == 64 and gzip_pass else "fail"
    receipt_path = root / config["receipt_path"]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "study_id": config["study_id"],
                "accession": config["accession"],
                "sample_accession": config["sample_accession"],
                "file_name": config["file_name"],
                "source_url": config["url"],
                "local_path": config["output_path"],
                "expected_size_bytes": config["expected_bytes"],
                "observed_size_bytes": observed_bytes,
                "size_status": "pass" if size_pass else "fail",
                "gzip_status": "pass" if gzip_pass else "fail",
                "sha256": observed_sha256,
                "license_note": "public_GEO_analysis_only_no_source_redistribution",
                "verification_status": status,
            }
        ]
    ).to_csv(receipt_path, sep="\t", index=False)
    output = root / config["output_root"]
    evidence_path = output / "completion_evidence.json"
    _write(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G5",
            "milestone": "H2_GSE307620_minimal_fragment_download",
            "status": status,
            "observed_bytes": observed_bytes,
            "expected_bytes": config["expected_bytes"],
            "fragment_sha256": observed_sha256,
            "gzip_integrity": gzip_pass,
            "uncompressed_bytes_integrity_scan": uncompressed_bytes,
            "receipt": {"path": config["receipt_path"], "sha256": sha256(receipt_path)},
            "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
            "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
            "large_fragment_download_run": True,
            "downloaded_files": 1,
            "hic_download_run": False,
            "fastq_or_bam_download_run": False,
            "image_download_run": False,
            "existing_holdout_accessed": False,
            "fragment_content_analysis_run": False,
            "representation_construction_run": False,
            "method_execution_run": False,
            "rna_or_label_access_run": False,
            "policy_refit_run": False,
            "regret_evaluation_run": False,
            "full_benchmark_run": False,
            "results_writing_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "next_milestone_authorized": False,
        },
    )
    if status != "pass":
        raise ValueError("GSE307620 minimal fragment download verification failed")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "verify"):
        child = sub.add_parser(name)
        child.add_argument("--config", default="configs/g5_h2_gse307620_minimal_download.yaml")
        child.add_argument("--root", default=".")
    args = parser.parse_args()
    output = preflight(args.config, args.root) if args.command == "preflight" else verify(args.config, args.root)
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
