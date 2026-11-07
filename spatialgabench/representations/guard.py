"""Access guard preventing development workflows from touching holdouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def assert_development_study(config_path: str | Path, study_id: str) -> dict[str, object]:
    """Return authorization metadata or reject candidate/holdout access."""

    config_path = Path(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    external = set(payload.get("external_validation_studies", []))
    if study_id in external:
        authorization_path = Path(payload["execution_authorization"])
        observed_hash = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
        if observed_hash != payload.get("execution_authorization_sha256"):
            raise PermissionError("External-validation authorization checksum mismatch")
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        if (
            payload.get("project", {}).get("current_gate") != "G5"
            or authorization.get("status") != "authorized"
            or authorization.get("milestone") != payload.get("execution_authorization_milestone")
            or study_id not in authorization.get("authorized_studies", [])
            or authorization.get("representation_construction_authorized") is not True
            or authorization.get("rna_value_access_authorized") is not False
            or authorization.get("policy_refit_authorized") is not False
        ):
            raise PermissionError(f"Study {study_id} lacks a valid G5 representation authorization")
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        return {
            "study_id": study_id,
            "authorization": "g5_external_representation_construction",
            "authorization_path": str(authorization_path),
            "authorization_sha256": observed_hash,
            "current_gate": "G5",
            "project_version": payload.get("project", {}).get("version"),
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "holdout_overlap": True,
            "rna_value_accessed": False,
            "method_execution_run": False,
        }
    development = set(payload.get("development_studies", []))
    holdouts = set(payload.get("holdout_studies", []))
    if study_id in holdouts:
        raise PermissionError(f"Study {study_id} is a frozen holdout and cannot run in G2-G4")
    if study_id not in development:
        raise PermissionError(f"Study {study_id} is not registered for development")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return {
        "study_id": study_id,
        "authorization": "development_only",
        "current_gate": payload.get("project", {}).get("current_gate"),
        "project_version": payload.get("project", {}).get("version"),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "holdout_overlap": False,
    }


def write_development_authorization(
    config_path: str | Path,
    study_id: str,
    output_path: str | Path,
) -> Path:
    """Write a machine-readable development access authorization."""

    result = assert_development_study(config_path, study_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
