"""Validation and evidence synthesis for NM-U1/U2 human representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("NM human representation config must be a mapping")
    return payload


def validate_inputs(config_path: Path, authorization_path: Path, root: Path, outdir: Path) -> Path:
    config = _load_config(config_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    check("authorization_active", authorization.get("status") == "authorized", authorization.get("status"))
    check("config_checksum", authorization.get("config_sha256") == sha256_path(config_path), authorization.get("config_sha256"))
    check("representation_authorized", authorization.get("representation_construction_authorized") is True, authorization.get("representation_construction_authorized"))
    for field in ("rna_access_authorized", "label_access_authorized", "pathology_access_authorized", "image_access_authorized", "method_execution_authorized"):
        check(f"{field}_false", config.get(field) is False, config.get(field))
    for field in ("policy_refit_run", "universal_ranking_run", "final_ranking_run"):
        check(f"{field}_false", config.get(field) is False, config.get(field))
    check("sample_panel_exact", set(config.get("samples", {})) == {"gse205055_humanbrain_50um", "gse278007_human_melanoma_50um"}, sorted(config.get("samples", {})))

    input_rows: list[dict[str, Any]] = []
    for dataset_id, sample in config.get("samples", {}).items():
        row: dict[str, Any] = {"dataset_id": dataset_id, "block": sample.get("block"), "study_id": sample.get("study_id")}
        for key, hash_key in (("fragments", "source_sha256"), ("positions", "positions_sha256"), ("chrom_sizes", "chrom_sizes_sha256"), ("blacklist", "blacklist_sha256")):
            path = root / sample[key]
            exists = path.is_file()
            digest = sha256_path(path) if exists else ""
            valid = exists and digest == sample[hash_key]
            row[f"{key}_path"] = sample[key]
            row[f"{key}_sha256"] = digest
            row[f"{key}_pass"] = valid
            check(f"{dataset_id}_{key}_hash", valid, digest)
        preflight_path = root / sample["input_preflight"]
        preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
        preflight_pass = preflight.get("status") == "pass"
        if dataset_id == "gse278007_human_melanoma_50um":
            preflight_pass = preflight.get("barcode_contract_status") == "pass"
        row["input_preflight_status"] = "pass" if preflight_pass else "fail"
        check(f"{dataset_id}_input_preflight", preflight_pass, sample["input_preflight"])
        input_rows.append(row)

    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "input_manifest.tsv"
    pd.DataFrame(input_rows).to_csv(manifest_path, sep="\t", index=False)
    passed = bool(checks) and all(row["status"] == "pass" for row in checks)
    evidence = {
        "schema_version": 1,
        "milestone": "NM-U1-U2_human_representation_input_preflight",
        "status": "pass" if passed else "blocked",
        "checks_passed": sum(row["status"] == "pass" for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "input_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_path(manifest_path)},
        "rna_accessed": False,
        "labels_pathology_images_accessed": False,
        "method_execution_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    output = outdir / "input_preflight_evidence.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def summarize(config_path: Path, authorization_path: Path, root: Path) -> Path:
    config = _load_config(config_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("config_sha256") != sha256_path(config_path):
        raise ValueError("NM human representation config checksum mismatch")
    rows: list[dict[str, Any]] = []
    for dataset_id, sample in config["samples"].items():
        base = root / config["output_root"] / dataset_id
        validation_path = base / "matched/matched_representation_validation.json"
        audit_path = base / "filtered_fragments/fragment_filter_audit.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        rows.append({
            "dataset_id": dataset_id,
            "block": sample["block"],
            "study_id": sample["study_id"],
            "species": sample["species"],
            "genome_build": sample["genome_build"],
            "observations": int(validation["barcode_rows"]),
            "source_fragment_rows": audit["source_fragment_rows"],
            "included_fragment_rows": audit["included_fragment_rows"],
            "excluded_chromosome_rows": audit["excluded_chromosome_rows"],
            "invalid_fragment_rows": audit["invalid_fragment_rows"],
            "representations": int(validation["representations"]),
            "status": "pass" if audit["status"] == validation["status"] == "pass" else "fail",
            "validation_sha256": sha256_path(validation_path),
        })
    evidence_dir = root / config["evidence_dir"]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary_path = evidence_dir / "sample_summary.tsv"
    pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)
    passed = len(rows) == 2 and all(row["status"] == "pass" and row["representations"] == 4 for row in rows)
    evidence = {
        "schema_version": 1,
        "milestone": "NM-U1-U2_human_matched_representation_construction",
        "status": "pass" if passed else "blocked",
        "samples": len(rows),
        "representations": sum(row["representations"] for row in rows),
        "sample_summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "rna_accessed": False,
        "labels_pathology_images_accessed": False,
        "method_execution_run": False,
        "policy_refit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    output = evidence_dir / "completion_evidence.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "summarize"])
    parser.add_argument("--config", default="configs/nm_upgrade_human_representations.yaml")
    parser.add_argument("--authorization", default="governance/authorizations/NM_U1_U2_HUMAN_REPRESENTATION_CONSTRUCTION_AUTHORIZATION_2026-07-30.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config, authorization = root / args.config, root / args.authorization
    if args.command == "validate":
        output = validate_inputs(config, authorization, root, root / _load_config(config)["evidence_dir"])
    else:
        output = summarize(config, authorization, root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
