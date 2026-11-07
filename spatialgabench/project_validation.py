"""Validation for the active SpatialGABench project contract."""

from __future__ import annotations

import hashlib
import json
import csv
import gzip
import math
from pathlib import Path
from typing import Any

import yaml


REQUIRED_DISABLED_FLAGS = (
    "allow_full_benchmark",
    "allow_results_writing",
    "allow_final_ranking",
)

M3A_FROZEN_GATE_RELATIVE = "governance/gates/G4_M3A_ADMISSIBILITY_EVALUATION_2026-07-16.json"
M3A_FROZEN_COMPONENT_RELATIVE = "results/v1/g4/m3a/evidence/g4_m3a_completion_evidence.json"
M3A_SUPERSEDING_GATE_RELATIVE = (
    "governance/authorizations/G4_M3A_RECERTIFICATION_SUPERSEDING_GATE_2026-10-17.json"
)
M3A_SUPERSEDING_COMPONENT_RELATIVE = (
    "results/v1_recert/G4_M3A_2026-10-17/evidence/g4_m3a_completion_evidence.json"
)
M3A_INCIDENT_EVIDENCE_SHA256 = (
    "c4833f49aff1509739729cc5adfd58200db0fb76e3c0da7f4a53740eea03b840"
)
M3A_SUPERSEDING_APPROVER = "yang yan (user)"
M3A_SUPERSEDING_APPROVED_AT_UTC = "2026-10-17T14:31Z"
M3A_SUPERSESSION_REASON = "test-suite contamination incident 2026-10-17"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise ValueError(f"Could not locate repository root from {config_path}")


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping in {path}")
    return value


def _validate_g2_attestation(
    root: Path,
    relative_path: object,
    development_studies: set[str],
    holdout_studies: set[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail})

    if not isinstance(relative_path, str) or not relative_path:
        check("attestation_declared", False, repr(relative_path))
        return {"status": "blocked", "path": None, "checks": checks}
    path = root / relative_path
    check("attestation_exists", path.is_file(), relative_path)
    if not path.is_file():
        return {"status": "blocked", "path": relative_path, "checks": checks}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check("attestation_parses", False, str(exc))
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("attestation_parses", isinstance(payload, dict), "JSON mapping")
    if not isinstance(payload, dict):
        return {"status": "blocked", "path": relative_path, "checks": checks}

    study_id = payload.get("development_study")
    check("gate_is_g2", payload.get("gate") == "G2", repr(payload.get("gate")))
    check("attestation_passes", payload.get("status") == "pass", repr(payload.get("status")))
    check("development_scope", study_id in development_studies, repr(study_id))
    check(
        "holdout_excluded",
        study_id not in holdout_studies and payload.get("holdout_overlap") is False,
        repr(study_id),
    )
    check(
        "full_benchmark_not_run",
        payload.get("full_benchmark_run") is False,
        repr(payload.get("full_benchmark_run")),
    )
    check(
        "gene_activity_not_run",
        payload.get("gene_activity_methods_run") is False,
        repr(payload.get("gene_activity_methods_run")),
    )
    check(
        "performance_not_interpreted",
        payload.get("performance_outcomes_interpreted") is False,
        repr(payload.get("performance_outcomes_interpreted")),
    )
    check(
        "filter_contract_verified",
        payload.get("fragment_filter_contract_verified") is True,
        repr(payload.get("fragment_filter_contract_verified")),
    )

    fragment_filter = payload.get("filter", {})
    row_keys = (
        "source_fragment_rows",
        "included_fragment_rows",
        "invalid_fragment_rows",
        "excluded_barcode_rows",
        "excluded_chromosome_rows",
    )
    rows_are_integers = isinstance(fragment_filter, dict) and all(
        isinstance(fragment_filter.get(key), int) and fragment_filter[key] >= 0 for key in row_keys
    )
    check("filter_rows_valid", rows_are_integers, repr(fragment_filter))
    if rows_are_integers:
        partition_total = (
            fragment_filter["included_fragment_rows"]
            + fragment_filter["invalid_fragment_rows"]
            + fragment_filter["excluded_barcode_rows"]
            + fragment_filter["excluded_chromosome_rows"]
        )
        check(
            "filter_partition_conserved",
            partition_total == fragment_filter["source_fragment_rows"],
            f"{partition_total}/{fragment_filter['source_fragment_rows']}",
        )
        check(
            "filter_has_no_invalid_rows",
            fragment_filter["invalid_fragment_rows"] == 0,
            str(fragment_filter["invalid_fragment_rows"]),
        )

    matrices = payload.get("matrices", [])
    required_representations = {
        "fixed_bin_10kb",
        "fixed_bin_25kb",
        "fixed_bin_50kb",
        "macs3_summit_501bp",
    }
    representation_ids = {
        row.get("representation_id") for row in matrices if isinstance(row, dict)
    } if isinstance(matrices, list) else set()
    check(
        "required_representations_present",
        representation_ids == required_representations,
        repr(sorted(representation_ids)),
    )
    matrix_rows_valid = isinstance(matrices, list) and len(matrices) == 4 and all(
        isinstance(row, dict)
        and row.get("status") == "pass"
        and isinstance(row.get("n_observations"), int)
        and row["n_observations"] > 0
        and isinstance(row.get("n_features"), int)
        and row["n_features"] > 0
        and isinstance(row.get("matrix_total_counts"), int)
        and row["matrix_total_counts"] >= 0
        for row in matrices
    )
    matrix_row_count = len(matrices) if isinstance(matrices, list) else "invalid"
    check("matrix_contracts_pass", matrix_rows_valid, f"rows={matrix_row_count}")
    if matrix_rows_valid and rows_are_integers:
        observation_counts = {row["n_observations"] for row in matrices}
        fixed_bin_counts = {
            row["matrix_total_counts"]
            for row in matrices
            if str(row["representation_id"]).startswith("fixed_bin_")
        }
        check("observation_sets_matched", len(observation_counts) == 1, repr(sorted(observation_counts)))
        check(
            "fixed_bin_counts_conserved",
            fixed_bin_counts == {fragment_filter["included_fragment_rows"]},
            repr(sorted(fixed_bin_counts)),
        )

    artifacts = payload.get("artifacts", {})
    hexdigits = set("0123456789abcdef")
    artifact_hashes_valid = isinstance(artifacts, dict) and len(artifacts) >= 5 and all(
        isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and isinstance(record.get("sha256"), str)
        and len(record["sha256"]) == 64
        and set(record["sha256"]) <= hexdigits
        for record in artifacts.values()
    )
    artifact_count = len(artifacts) if isinstance(artifacts, dict) else "invalid"
    check("artifact_hashes_recorded", artifact_hashes_valid, f"artifacts={artifact_count}")
    status = "pass" if checks and all(row["status"] == "pass" for row in checks) else "blocked"
    return {"status": status, "path": relative_path, "checks": checks}


def _validate_g3_attestation(root: Path, relative_path: object) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail})

    if not isinstance(relative_path, str) or not relative_path:
        check("attestation_declared", False, repr(relative_path))
        return {"status": "blocked", "path": None, "checks": checks}
    path = root / relative_path
    check("attestation_exists", path.is_file(), relative_path)
    if not path.is_file():
        return {"status": "blocked", "path": relative_path, "checks": checks}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check("attestation_parses", False, str(exc))
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("attestation_parses", isinstance(payload, dict), "JSON mapping")
    if not isinstance(payload, dict):
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("gate_is_g3", payload.get("gate") == "G3", repr(payload.get("gate")))
    check("attestation_status_declared", payload.get("status") in {"partial", "pass"}, repr(payload.get("status")))
    check("holdout_excluded", payload.get("holdout_accessed") is False, repr(payload.get("holdout_accessed")))
    check("full_benchmark_not_run", payload.get("full_benchmark_run") is False, repr(payload.get("full_benchmark_run")))
    check("final_ranking_not_run", payload.get("final_ranking_run") is False, repr(payload.get("final_ranking_run")))
    check(
        "construction_evaluation_inputs_excluded",
        payload.get("construction_evaluation_inputs_used") is False,
        repr(payload.get("construction_evaluation_inputs_used")),
    )

    components = payload.get("components", {})
    required_pass_components = {"software_wrapper_contracts", "simulation_design_smoke", "core_contract_container"}
    check("required_components_declared", required_pass_components <= set(components), repr(sorted(components)))
    for component_name in sorted(required_pass_components):
        component = components.get(component_name, {})
        component_path = root / str(component.get("path", ""))
        path_valid = component_path.is_file()
        hash_valid = path_valid and sha256_path(component_path) == component.get("sha256")
        payload_valid = False
        if path_valid and component_path.suffix == ".json":
            try:
                component_payload = json.loads(component_path.read_text(encoding="utf-8"))
                payload_valid = component_payload.get("status") == "pass"
            except (OSError, json.JSONDecodeError):
                payload_valid = False
        check(f"{component_name}_path", path_valid, str(component.get("path", "")))
        check(f"{component_name}_hash", hash_valid, repr(component.get("sha256")))
        check(f"{component_name}_passes", component.get("status") == "pass" and payload_valid, repr(component.get("status")))

    runtime_component = components.get("method_runtime_containers", {})
    runtime_path = root / str(runtime_component.get("path", ""))
    runtime_path_valid = runtime_path.is_file()
    runtime_hash_valid = runtime_path_valid and sha256_path(runtime_path) == runtime_component.get("sha256")
    runtime_payload: dict[str, Any] = {}
    if runtime_path_valid and runtime_path.suffix == ".json":
        try:
            loaded_runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            if isinstance(loaded_runtime_payload, dict):
                runtime_payload = loaded_runtime_payload
        except (OSError, json.JSONDecodeError):
            runtime_payload = {}
    required_runtime_methods = {
        "snapatac2_geneactivity",
        "signac_geneactivity",
        "archr_genescorematrix",
        "cicero_geneactivity",
    }
    runtime_methods = runtime_payload.get("methods", [])
    runtime_method_ids = {
        row.get("method_id") for row in runtime_methods if isinstance(row, dict)
    } if isinstance(runtime_methods, list) else set()
    runtime_boundaries_valid = (
        runtime_payload.get("holdout_accessed") is False
        and runtime_payload.get("full_benchmark_run") is False
        and runtime_payload.get("final_ranking_run") is False
        and runtime_payload.get("construction_evaluation_inputs_used") is False
    )
    runtime_rows_valid = isinstance(runtime_methods, list) and len(runtime_methods) == 4
    for row in runtime_methods if isinstance(runtime_methods, list) else []:
        if not isinstance(row, dict):
            runtime_rows_valid = False
            continue
        containerfile = row.get("containerfile", {})
        lock = row.get("environment_lock", {})
        containerfile_path = root / str(containerfile.get("path", ""))
        lock_path = root / str(lock.get("path", ""))
        provenance_valid = (
            containerfile_path.is_file()
            and sha256_path(containerfile_path) == containerfile.get("sha256")
            and lock_path.is_file()
            and sha256_path(lock_path) == lock.get("sha256")
        )
        image_id = str(row.get("image_id", ""))
        image_digest = str(row.get("image_digest", ""))
        row_valid = (
            row.get("status") == "pass"
            and row.get("build_network") == "none"
            and row.get("test_network") == "none"
            and row.get("run_exit_code") == 0
            and row.get("contract_status") == "pass"
            and row.get("contract_checks_passed") == row.get("contract_checks_total") == 18
            and row.get("image_created") == "1970-01-01T00:00:00Z"
            and len(image_id) == 64
            and set(image_id) <= set("0123456789abcdef")
            and image_digest.startswith("sha256:")
            and len(image_digest) == 71
            and set(image_digest.removeprefix("sha256:")) <= set("0123456789abcdef")
            and provenance_valid
        )
        runtime_rows_valid = runtime_rows_valid and row_valid
    runtime_payload_valid = (
        runtime_payload.get("status") == "pass"
        and runtime_payload.get("method_runtime_containers_verified") is True
        and runtime_method_ids == required_runtime_methods
        and runtime_rows_valid
        and runtime_boundaries_valid
    )
    runtime_verified = (
        runtime_component.get("verified") is True
        and runtime_component.get("status") == "pass"
        and runtime_payload_valid
    )
    check("method_runtime_container_status_path", runtime_path_valid, str(runtime_component.get("path", "")))
    check("method_runtime_container_status_hash", runtime_hash_valid, repr(runtime_component.get("sha256")))
    check("method_runtime_container_attestation", runtime_payload_valid, repr(sorted(runtime_method_ids)))
    expected_status = "pass" if runtime_verified else "partial"
    check("attestation_status_consistent", payload.get("status") == expected_status, f"expected={expected_status}")
    check("gate_exit_consistent", payload.get("gate_exit_approved") is runtime_verified, repr(payload.get("gate_exit_approved")))

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": expected_status if checks_pass else "blocked",
        "path": relative_path,
        "checks": checks,
        "method_runtime_containers_verified": runtime_verified,
    }


def _validate_g4_attestation(root: Path, relative_path: object) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail})

    if not isinstance(relative_path, str) or not relative_path:
        check("attestation_declared", False, repr(relative_path))
        return {"status": "blocked", "path": None, "checks": checks}
    path = root / relative_path
    check("attestation_exists", path.is_file(), relative_path)
    if not path.is_file():
        return {"status": "blocked", "path": relative_path, "checks": checks}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check("attestation_parses", False, str(exc))
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("attestation_parses", isinstance(payload, dict), "JSON mapping")
    if not isinstance(payload, dict):
        return {"status": "blocked", "path": relative_path, "checks": checks}

    check("gate_is_g4", payload.get("gate") == "G4", repr(payload.get("gate")))
    check("status_ready", payload.get("status") == "ready_to_execute", repr(payload.get("status")))
    check("gate_not_exited", payload.get("gate_exit_approved") is False, repr(payload.get("gate_exit_approved")))
    for field in (
        "development_benchmark_run",
        "holdout_accessed",
        "full_benchmark_run",
        "final_ranking_run",
        "execution_enabled",
    ):
        check(f"{field}_false", payload.get(field) is False, repr(payload.get(field)))

    component = payload.get("component", {})
    component_path = root / str(component.get("path", ""))
    component_exists = component_path.is_file()
    component_hash_valid = component_exists and sha256_path(component_path) == component.get("sha256")
    component_payload: dict[str, Any] = {}
    if component_exists:
        try:
            loaded_component = json.loads(component_path.read_text(encoding="utf-8"))
            if isinstance(loaded_component, dict):
                component_payload = loaded_component
        except (OSError, json.JSONDecodeError):
            component_payload = {}
    component_boundaries_valid = (
        component_payload.get("status") == "pass"
        and component_payload.get("development_benchmark_run") is False
        and component_payload.get("holdout_accessed") is False
        and component_payload.get("full_benchmark_run") is False
        and component_payload.get("final_ranking_run") is False
    )
    check("validation_component_exists", component_exists, str(component.get("path", "")))
    check("validation_component_hash", component_hash_valid, repr(component.get("sha256")))
    check("validation_component_passes", component.get("status") == "pass" and component_boundaries_valid, repr(component_payload.get("status")))

    preregistration = payload.get("preregistration", {})
    prereg_path = root / str(preregistration.get("path", ""))
    prereg_hash_valid = prereg_path.is_file() and sha256_path(prereg_path) == preregistration.get("sha256")
    component_prereg = component_payload.get("preregistration", {})
    prereg_cross_hash_valid = (
        component_prereg.get("path") == preregistration.get("path")
        and component_prereg.get("sha256") == preregistration.get("sha256")
    )
    check("preregistration_hash", prereg_hash_valid, repr(preregistration.get("sha256")))
    check("preregistration_cross_hash", prereg_cross_hash_valid, repr(component_prereg))

    artifacts_valid = True
    for artifact in component_payload.get("artifacts", {}).values():
        if not isinstance(artifact, dict):
            artifacts_valid = False
            continue
        artifact_path = root / str(artifact.get("path", ""))
        artifacts_valid = artifacts_valid and (
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256")
        )
    check("preregistration_artifact_hashes", artifacts_valid, repr(sorted(component_payload.get("artifacts", {}))))
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "ready_to_execute" if checks_pass else "blocked",
        "path": relative_path,
        "checks": checks,
        "execution_enabled": False,
    }


def _validate_g4_preflight(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/preflight/evidence/g4_contract_preflight_evidence.json"
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append(
            {"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)}
        )

    check("evidence_exists", path.is_file(), relative_path)
    if not path.is_file():
        return {"status": "not_run", "path": relative_path, "checks": checks}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check("evidence_parses", False, exc)
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("evidence_parses", isinstance(payload, dict), "JSON mapping")
    if not isinstance(payload, dict):
        return {"status": "blocked", "path": relative_path, "checks": checks}

    check("milestone", payload.get("milestone") == "G4-M0", payload.get("milestone"))
    check("status_pass", payload.get("status") == "pass", payload.get("status"))
    for field in (
        "construction_evaluation_inputs_used",
        "development_benchmark_run",
        "holdout_accessed",
        "full_benchmark_run",
        "final_ranking_run",
    ):
        check(f"{field}_false", payload.get(field) is False, payload.get(field))
    check(
        "memory_scope_noncomparative",
        payload.get("memory_measurement_scope")
        == "driver_observed_only_not_admissible_for_performance_comparison",
        payload.get("memory_measurement_scope"),
    )
    check("resolved_failures_retained", payload.get("resolved_attempt_failures") == 2, payload.get("resolved_attempt_failures"))

    summary_rows: dict[str, list[dict[str, str]]] = {}
    for summary_name, expected_rows in (("fixture_summary", 2), ("run_summary", 6)):
        summary = payload.get(summary_name, {})
        summary_path = root / str(summary.get("path", ""))
        exists = summary_path.is_file()
        hash_valid = exists and sha256_path(summary_path) == summary.get("sha256")
        rows: list[dict[str, str]] = []
        if exists:
            with summary_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        summary_rows[summary_name] = rows
        check(f"{summary_name}_exists", exists, summary.get("path"))
        check(f"{summary_name}_hash", hash_valid, summary.get("sha256"))
        check(
            f"{summary_name}_rows",
            summary.get("rows") == len(rows) == expected_rows,
            len(rows),
        )
        check(
            f"{summary_name}_passes",
            bool(rows) and all(row.get("status") == "pass" for row in rows),
            [row.get("status") for row in rows],
        )

    run_rows = summary_rows.get("run_summary", [])
    expected_pairs = {
        (species, method)
        for species in ("mouse", "human")
        for method in (
            "snapatac2_geneactivity",
            "signac_geneactivity",
            "archr_genescorematrix",
        )
    }
    observed_pairs = {(row.get("species"), row.get("method_id")) for row in run_rows}
    contracts_valid = all(
        row.get("contract_checks_passed") == row.get("contract_checks_total") == "18"
        for row in run_rows
    )
    check("run_panel_exact", observed_pairs == expected_pairs, sorted(observed_pairs))
    check("all_output_contracts_18_of_18", contracts_valid, len(run_rows) * 18)

    runtime_path = root / "results/v1/g3/environment/method_runtime_container_attestation.json"
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    runtime_digests = {
        row.get("method_id"): row.get("image_digest")
        for row in runtime_payload.get("methods", [])
        if isinstance(row, dict)
    }
    digests_match = all(
        runtime_digests.get(row.get("method_id")) == row.get("image_digest") for row in run_rows
    )
    check("runtime_image_digests_current", digests_match, sorted(runtime_digests))
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "pass" if checks_pass else "blocked",
        "path": relative_path,
        "checks": checks,
    }


def _validate_g4_m1(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/m1/evidence/g4_m1_completion_evidence.json"
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append(
            {"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)}
        )

    check("evidence_exists", path.is_file(), relative_path)
    if not path.is_file():
        return {"status": "not_run", "path": relative_path, "checks": checks}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check("evidence_parses", False, exc)
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("evidence_parses", isinstance(payload, dict), "JSON mapping")
    if not isinstance(payload, dict):
        return {"status": "blocked", "path": relative_path, "checks": checks}

    check("gate", payload.get("gate") == "G4", payload.get("gate"))
    check("milestone", payload.get("milestone") == "M1", payload.get("milestone"))
    check("status_pass", payload.get("status") == "pass", payload.get("status"))
    check("study_count", payload.get("studies") == 5, payload.get("studies"))
    check("representation_count", payload.get("representations") == 20, payload.get("representations"))
    check(
        "construction_reuse_partition",
        payload.get("constructed_studies") == 4 and payload.get("reused_g2_studies") == 1,
        (payload.get("constructed_studies"), payload.get("reused_g2_studies")),
    )
    check("no_recorded_failures", payload.get("failures") == [], payload.get("failures"))
    check(
        "forbidden_inputs_excluded",
        payload.get("forbidden_inputs_used") == [],
        payload.get("forbidden_inputs_used"),
    )
    for field in (
        "holdout_accessed",
        "m2_run",
        "full_benchmark_run",
        "final_ranking_run",
        "performance_outcomes_interpreted",
    ):
        check(f"{field}_false", payload.get(field) is False, payload.get(field))

    tables: dict[str, list[dict[str, str]]] = {}
    expected_rows = {"sample_summary": 5, "representation_summary": 20, "resource_summary": 17}
    for key, row_count in expected_rows.items():
        table_path = root / str(payload.get(key, ""))
        exists = table_path.is_file()
        hash_valid = exists and sha256_path(table_path) == payload.get(f"{key}_sha256")
        rows: list[dict[str, str]] = []
        if exists:
            with table_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        tables[key] = rows
        check(f"{key}_exists", exists, payload.get(key))
        check(f"{key}_hash", hash_valid, payload.get(f"{key}_sha256"))
        check(f"{key}_rows", len(rows) == row_count, len(rows))
        check(
            f"{key}_passes",
            bool(rows) and all(row.get("status") == "pass" for row in rows),
            [row.get("status") for row in rows],
        )

    expected_studies = {
        "zhang_2023",
        "deng_2022",
        "li_2026",
        "guo_2025_spatial_mux",
        "guo_deng_2025_ffpe_atac",
    }
    sample_rows = tables.get("sample_summary", [])
    observed_studies = {row.get("study_id") for row in sample_rows}
    check("matched_study_panel_exact", observed_studies == expected_studies, sorted(observed_studies))
    check(
        "sample_invariants_pass",
        bool(sample_rows)
        and all(
            row.get("identical_observation_ids_and_order") == "True"
            and row.get("one_filtered_fragment_sha256") == "True"
            and row.get("source_checksum_pass") == "True"
            and row.get("invalid_fragment_rows") == "0"
            for row in sample_rows
        ),
        len(sample_rows),
    )
    representation_rows = tables.get("representation_summary", [])
    required_representations = {
        "fixed_bin_10kb",
        "fixed_bin_25kb",
        "fixed_bin_50kb",
        "macs3_summit_501bp",
    }
    per_dataset_representation_sets: dict[str, set[str]] = {}
    for row in representation_rows:
        per_dataset_representation_sets.setdefault(str(row.get("dataset_id")), set()).add(
            str(row.get("representation_id"))
        )
    check(
        "four_representations_per_dataset",
        len(per_dataset_representation_sets) == 5
        and all(values == required_representations for values in per_dataset_representation_sets.values()),
        {key: sorted(value) for key, value in per_dataset_representation_sets.items()},
    )
    check(
        "representation_checksums_and_counts_pass",
        bool(representation_rows)
        and all(
            row.get("output_checksum_pass") == "True"
            and row.get("exact_fixed_bin_count_conservation") == "True"
            for row in representation_rows
        ),
        len(representation_rows),
    )
    restart_path = root / "results/v1/g4/m1/failures/G4_M1_INTENTIONAL_RESTART_2026-07-15.json"
    check("intentional_restart_retained", restart_path.is_file(), str(restart_path.relative_to(root)))
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m2_memory_readiness(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/memory_preflight/g4_m2_memory_readiness_evidence.json"
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append(
            {"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)}
        )

    check("evidence_exists", path.is_file(), relative_path)
    if not path.is_file():
        return {"status": "not_run", "path": relative_path, "checks": checks}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check("evidence_parses", False, exc)
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("evidence_parses", isinstance(payload, dict), "JSON mapping")
    if not isinstance(payload, dict):
        return {"status": "blocked", "path": relative_path, "checks": checks}

    check("gate", payload.get("gate") == "G4", payload.get("gate"))
    check("milestone", payload.get("milestone") == "M2_memory_readiness", payload.get("milestone"))
    check("status_pass", payload.get("status") == "pass", payload.get("status"))
    for field in (
        "holdout_accessed",
        "construction_evaluation_inputs_used",
        "development_benchmark_run",
        "rna_evaluation_run",
        "full_benchmark_run",
        "final_ranking_run",
        "performance_outcomes_interpreted",
        "m2_authorized_by_this_evidence",
    ):
        check(f"{field}_false", payload.get(field) is False, payload.get(field))
    check(
        "contract_preflight_method_run",
        payload.get("contract_preflight_method_run") is True,
        payload.get("contract_preflight_method_run"),
    )

    policy = payload.get("measurement_policy", {})
    check("pss_primary", policy.get("primary_memory_metric") == "peak_aggregate_pss_mb", policy)
    check("rss_supplemental", policy.get("supplemental_memory_metric") == "peak_aggregate_rss_mb", policy)
    check("sample_interval", policy.get("sample_interval_seconds") == 0.05, policy)
    check(
        "driver_ru_maxrss_rejected",
        policy.get("driver_ru_maxrss_admissible_for_comparison") is False,
        policy,
    )

    expected_profiles = {"synthetic", "signac_oci", "archr_oci", "mouse_signac_m0"}
    profiles = payload.get("profiles", [])
    observed_profiles = {
        row.get("profile") for row in profiles if isinstance(row, dict)
    }
    profile_checks_pass = all(
        row.get("status") == "pass"
        and isinstance(row.get("checks"), dict)
        and bool(row["checks"])
        and all(value is True for value in row["checks"].values())
        for row in profiles
        if isinstance(row, dict)
    )
    check("profile_panel_exact", observed_profiles == expected_profiles, sorted(observed_profiles))
    check("all_profile_checks_pass", len(profiles) == 4 and profile_checks_pass, len(profiles))
    method_profile = next(
        (row for row in profiles if isinstance(row, dict) and row.get("profile") == "mouse_signac_m0"),
        {},
    )
    check(
        "driver_underestimate_demonstrated",
        float(method_profile.get("peak_aggregate_pss_mb", 0))
        > 5 * float(method_profile.get("driver_observed_peak_memory_mb", float("inf"))),
        (
            method_profile.get("peak_aggregate_pss_mb"),
            method_profile.get("driver_observed_peak_memory_mb"),
        ),
    )

    summary = payload.get("summary_table", {})
    summary_path = root / str(summary.get("path", ""))
    rows: list[dict[str, str]] = []
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    check("summary_exists", summary_path.is_file(), summary.get("path"))
    check(
        "summary_hash",
        summary_path.is_file() and sha256_path(summary_path) == summary.get("sha256"),
        summary.get("sha256"),
    )
    check("summary_rows", summary.get("rows") == len(rows) == 4, len(rows))
    check("summary_passes", bool(rows) and all(row.get("status") == "pass" for row in rows), len(rows))

    implementation = payload.get("implementation", {})
    check(
        "implementation_hashes_recorded",
        bool(implementation)
        and all(
            (root / relative).is_file()
            and isinstance(digest, str)
            and len(digest) == 64
            for relative, digest in implementation.items()
        ),
        len(implementation),
    )
    completion_gate_path = root / "governance/gates/G4_M2_MEMORY_READINESS_2026-07-15.json"
    completion_gate: dict[str, Any] = {}
    if completion_gate_path.is_file():
        try:
            loaded_gate = json.loads(completion_gate_path.read_text(encoding="utf-8"))
            if isinstance(loaded_gate, dict):
                completion_gate = loaded_gate
        except (OSError, json.JSONDecodeError):
            pass
    component = completion_gate.get("component", {})
    check(
        "completion_gate_binds_evidence",
        completion_gate.get("status") == "pass"
        and component.get("path") == relative_path
        and component.get("sha256") == sha256_path(path),
        completion_gate_path.relative_to(root),
    )
    method_artifacts_valid = all(
        (root / str(artifact.get("path", ""))).is_file()
        and sha256_path(root / str(artifact["path"])) == artifact.get("sha256")
        for key, artifact in payload.get("method_contract_preflight", {}).items()
        if key != "scope" and isinstance(artifact, dict)
    )
    check("method_artifact_hashes", method_artifacts_valid, payload.get("method_contract_preflight", {}))
    cgroup = payload.get("cgroup_capability", {})
    check("cgroup_v2_audited", cgroup.get("kernel_interface") == "cgroup_v2", cgroup)
    check(
        "procfs_fallback_declared",
        cgroup.get("decision") == "procfs_process_tree_fallback"
        and cgroup.get("per_run_container_cgroup_verified") is False,
        cgroup,
    )

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m2_annotation(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/m2/evidence/g4_m2_annotation_completion_evidence.json"
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append(
            {"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)}
        )

    check("evidence_exists", path.is_file(), relative_path)
    if not path.is_file():
        return {"status": "not_run", "path": relative_path, "checks": checks}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check("evidence_parses", False, exc)
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("evidence_parses", isinstance(payload, dict), "JSON mapping")
    if not isinstance(payload, dict):
        return {"status": "blocked", "path": relative_path, "checks": checks}

    check("gate", payload.get("gate") == "G4", payload.get("gate"))
    check("milestone", payload.get("milestone") == "M2A_annotation_construction", payload.get("milestone"))
    check("status_pass", payload.get("status") == "pass", payload.get("status"))
    check(
        "panel_dimensions",
        payload.get("runs") == 140
        and payload.get("datasets") == 5
        and payload.get("representations") == 20
        and payload.get("methods") == 7,
        (payload.get("runs"), payload.get("datasets"), payload.get("representations"), payload.get("methods")),
    )
    check(
        "all_contracts_pass",
        payload.get("contract_checks_passed") == payload.get("contract_checks_total") == 2520,
        (payload.get("contract_checks_passed"), payload.get("contract_checks_total")),
    )
    for field in (
        "holdout_accessed",
        "construction_evaluation_inputs_used",
        "rna_evaluation_run",
        "label_evaluation_run",
        "software_methods_run",
        "full_benchmark_run",
        "final_ranking_run",
        "performance_outcomes_interpreted",
        "resource_measurements_comparative",
        "next_milestone_authorized",
    ):
        check(f"{field}_false", payload.get(field) is False, payload.get(field))
    check("no_unresolved_failures", payload.get("unresolved_failures") == [], payload.get("unresolved_failures"))
    retained = payload.get("retained_resolved_failures", [])
    check(
        "resolved_schema_failure_retained",
        len(retained) == 1 and "attempt_1_schema_failure/failure.tsv" in retained[0],
        retained,
    )

    tables: dict[str, list[dict[str, str]]] = {}
    for key, expected_rows in (("run_summary", 140), ("representation_summary", 20)):
        artifact = payload.get(key, {})
        table_path = root / str(artifact.get("path", ""))
        rows: list[dict[str, str]] = []
        if table_path.is_file():
            with table_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        tables[key] = rows
        check(f"{key}_exists", table_path.is_file(), artifact.get("path"))
        check(
            f"{key}_hash",
            table_path.is_file() and sha256_path(table_path) == artifact.get("sha256"),
            artifact.get("sha256"),
        )
        check(f"{key}_rows", len(rows) == expected_rows, len(rows))
        check(f"{key}_passes", bool(rows) and all(row.get("status") == "pass" for row in rows), len(rows))

    run_rows = tables.get("run_summary", [])
    expected_methods = {
        "promoter_only_2kb_pc",
        "gene_body_pc",
        "promoter_plus_gene_body_2kb_pc",
        "distance_weighted_distal_250kb_decay50kb_pc",
        "promoter_plus_distal_2kb_250kb_decay50kb_pc",
        "random_peak_to_gene_negative_control_pc",
        "accessibility_matched_random_peak_to_gene_control_pc",
    }
    check("method_panel_exact", {row.get("method_id") for row in run_rows} == expected_methods, expected_methods)
    check(
        "run_invariants",
        bool(run_rows)
        and all(
            row.get("manifest_hashes_valid") == "True"
            and row.get("observation_ids_and_order_match_input") == "True"
            and row.get("contract_checks_passed") == row.get("contract_checks_total") == "18"
            and row.get("holdout_accessed") == "False"
            for row in run_rows
        ),
        len(run_rows),
    )
    representation_rows = tables.get("representation_summary", [])
    check(
        "representation_invariants",
        bool(representation_rows)
        and all(
            row.get("methods_expected") == row.get("methods_observed") == "7"
            and row.get("all_methods_pass") == "True"
            and row.get("all_observation_ids_and_order_match") == "True"
            and row.get("one_gene_universe_across_methods") == "True"
            and row.get("negative_control_link_counts_preserved") == "True"
            for row in representation_rows
        ),
        len(representation_rows),
    )
    implementation = payload.get("implementation", {})
    check(
        "implementation_hashes_recorded",
        bool(implementation)
        and all(
            (root / relative).is_file()
            and isinstance(digest, str)
            and len(digest) == 64
            for relative, digest in implementation.items()
        ),
        len(implementation),
    )
    completion_gate_path = root / "governance/gates/G4_M2A_ANNOTATION_CONSTRUCTION_2026-07-15.json"
    completion_gate: dict[str, Any] = {}
    if completion_gate_path.is_file():
        try:
            loaded_gate = json.loads(completion_gate_path.read_text(encoding="utf-8"))
            if isinstance(loaded_gate, dict):
                completion_gate = loaded_gate
        except (OSError, json.JSONDecodeError):
            pass
    component = completion_gate.get("component", {})
    check(
        "completion_gate_binds_evidence",
        completion_gate.get("status") == "pass"
        and component.get("path") == relative_path
        and component.get("sha256") == sha256_path(path),
        completion_gate_path.relative_to(root),
    )
    for key in ("config", "authorization"):
        artifact = payload.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m2_software(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/m2/evidence/g4_m2_software_completion_evidence.json"
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    check("evidence_exists", path.is_file(), relative_path)
    if not path.is_file():
        return {"status": "not_run", "path": relative_path, "checks": checks}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check("evidence_parses", False, exc)
        return {"status": "blocked", "path": relative_path, "checks": checks}
    check("evidence_parses", isinstance(payload, dict), "JSON mapping")
    check("status_pass", payload.get("status") == "pass", payload.get("status"))
    check("panel_dimensions", payload.get("runs_expected") == payload.get("runs_passed") == 15 and payload.get("runs_failed") == payload.get("runs_missing") == 0 and payload.get("datasets_expected") == 5 and payload.get("methods_expected") == 3, (payload.get("runs_passed"), payload.get("runs_failed"), payload.get("runs_missing")))
    for field in ("holdout_accessed", "construction_evaluation_inputs_used", "rna_evaluation_run", "label_evaluation_run", "full_benchmark_run", "final_ranking_run", "performance_outcomes_interpreted", "resource_measurements_comparative", "next_milestone_authorized"):
        check(f"{field}_false", payload.get(field) is False, payload.get(field))
    summary = payload.get("run_summary", {})
    summary_path = root / str(summary.get("path", ""))
    rows: list[dict[str, str]] = []
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    check("summary_hash", summary_path.is_file() and sha256_path(summary_path) == summary.get("sha256"), summary)
    check("summary_rows_pass", len(rows) == 15 and all(row.get("status") == "pass" for row in rows), len(rows))
    check("method_panel_exact", {row.get("method_id") for row in rows} == {"snapatac2_geneactivity", "signac_geneactivity", "archr_genescorematrix"}, len(rows))
    check("run_invariants", bool(rows) and all(row.get("contract_checks_passed") == row.get("contract_checks_total") == "18" and row.get("observation_ids_and_order_match") == "True" and row.get("manifest_hashes_valid") == "True" and row.get("current_input_hashes_valid") == "True" for row in rows), len(rows))
    for key in ("config", "authorization", "memory_gate", "implementation"):
        artifact = payload.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(f"{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)
    gate_path = root / "governance/gates/G4_M2B_SOFTWARE_CONSTRUCTION_2026-07-15.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    component = gate.get("component", {})
    check("completion_gate_binds_evidence", gate.get("status") == "pass" and component.get("path") == relative_path and component.get("sha256") == sha256_path(path), gate_path.relative_to(root))
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m2c(root: Path) -> dict[str, Any]:
    evidence_paths = {
        "raw_lsi": "results/v1/g4/m2/evidence/g4_m2c_raw_lsi_evidence.json",
        "cicero": "results/v1/g4/m2/evidence/g4_m2c_cicero_scalability_evidence.json",
    }
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    payloads: dict[str, dict[str, Any]] = {}
    for component, relative_path in evidence_paths.items():
        path = root / relative_path
        check(f"{component}_evidence_exists", path.is_file(), relative_path)
        payload: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                pass
        payloads[component] = payload
        check(f"{component}_evidence_parses", bool(payload), relative_path)

    raw = payloads["raw_lsi"]
    check("raw_status_pass", raw.get("status") == "pass", raw.get("status"))
    check(
        "raw_panel_dimensions",
        raw.get("runs_expected") == raw.get("runs_passed") == 20
        and raw.get("runs_failed") == raw.get("runs_missing") == 0
        and raw.get("datasets_expected") == 5
        and raw.get("representations_expected") == 4,
        (raw.get("runs_passed"), raw.get("runs_failed"), raw.get("runs_missing")),
    )
    check(
        "raw_contracts",
        raw.get("contract_checks_passed") == raw.get("contract_checks_total") == 360,
        (raw.get("contract_checks_passed"), raw.get("contract_checks_total")),
    )
    check(
        "raw_role",
        raw.get("output_type") == "cell_embedding"
        and raw.get("is_gene_activity") is False
        and raw.get("benchmark_role") == "representation_control",
        (raw.get("output_type"), raw.get("is_gene_activity"), raw.get("benchmark_role")),
    )

    cicero = payloads["cicero"]
    check("cicero_status_pass", cicero.get("status") == "pass", cicero.get("status"))
    check(
        "cicero_panel_dimensions",
        cicero.get("planned_combinations") == 20
        and cicero.get("eligible_runs_expected") == cicero.get("eligible_runs_passed") == 14
        and cicero.get("eligible_runs_failed") == cicero.get("eligible_runs_missing") == 0
        and cicero.get("not_applicable_combinations") == 6
        and cicero.get("datasets_expected") == 5,
        (
            cicero.get("eligible_runs_passed"),
            cicero.get("eligible_runs_failed"),
            cicero.get("eligible_runs_missing"),
            cicero.get("not_applicable_combinations"),
        ),
    )
    check(
        "cicero_contracts",
        cicero.get("contract_checks_passed") == cicero.get("contract_checks_total") == 252,
        (cicero.get("contract_checks_passed"), cicero.get("contract_checks_total")),
    )
    check(
        "cicero_subset_policy",
        cicero.get("subset_only") is True
        and cicero.get("performance_pooling_allowed") is False
        and cicero.get("not_applicable_in_failure_denominator") is False
        and cicero.get("benchmark_role") == "subset_scalability_method",
        cicero.get("benchmark_role"),
    )

    boundary_fields = (
        "holdout_accessed",
        "construction_evaluation_inputs_used",
        "rna_evaluation_run",
        "label_evaluation_run",
        "full_benchmark_run",
        "final_ranking_run",
        "performance_outcomes_interpreted",
        "resource_measurements_comparative",
        "next_milestone_authorized",
    )
    for component, payload in payloads.items():
        for field in boundary_fields:
            check(f"{component}_{field}_false", payload.get(field) is False, payload.get(field))
        for artifact_name in ("config", "authorization", "representation_summary", "implementation"):
            artifact = payload.get(artifact_name, {})
            artifact_path = root / str(artifact.get("path", ""))
            check(
                f"{component}_{artifact_name}_hash",
                artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
                artifact,
            )

    raw_summary = raw.get("run_summary", {})
    raw_summary_path = root / str(raw_summary.get("path", ""))
    raw_rows: list[dict[str, str]] = []
    if raw_summary_path.is_file():
        with raw_summary_path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "raw_summary_hash",
        raw_summary_path.is_file() and sha256_path(raw_summary_path) == raw_summary.get("sha256"),
        raw_summary,
    )
    check("raw_summary_rows", len(raw_rows) == 20, len(raw_rows))
    check(
        "raw_summary_invariants",
        bool(raw_rows)
        and all(
            row.get("status") == "pass"
            and row.get("method_id") == "raw_peak_lsi_baseline"
            and row.get("contract_checks_passed") == row.get("contract_checks_total") == "18"
            and row.get("observation_ids_and_order_match") == "True"
            and row.get("manifest_hashes_valid") == "True"
            and row.get("current_input_hash_valid") == "True"
            for row in raw_rows
        ),
        len(raw_rows),
    )

    cicero_summary = cicero.get("run_summary", {})
    cicero_summary_path = root / str(cicero_summary.get("path", ""))
    cicero_rows: list[dict[str, str]] = []
    if cicero_summary_path.is_file():
        with cicero_summary_path.open(encoding="utf-8", newline="") as handle:
            cicero_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "cicero_summary_hash",
        cicero_summary_path.is_file() and sha256_path(cicero_summary_path) == cicero_summary.get("sha256"),
        cicero_summary,
    )
    eligible_rows = [row for row in cicero_rows if row.get("eligible") == "True"]
    not_applicable_rows = [row for row in cicero_rows if row.get("status") == "not_applicable"]
    check("cicero_summary_rows", len(cicero_rows) == 20, len(cicero_rows))
    check(
        "cicero_eligible_invariants",
        len(eligible_rows) == 14
        and all(
            row.get("status") == "pass"
            and row.get("subset_only") == "True"
            and row.get("performance_pooling_allowed") == "False"
            and row.get("included_in_failure_denominator") == "True"
            and row.get("contract_checks_passed") == row.get("contract_checks_total") == "18"
            and row.get("observation_ids_and_order_match") == "True"
            and row.get("manifest_hashes_valid") == "True"
            and row.get("current_input_hashes_valid") == "True"
            and row.get("source_hash_valid") == "True"
            for row in eligible_rows
        ),
        len(eligible_rows),
    )
    check(
        "cicero_not_applicable_invariants",
        len(not_applicable_rows) == 6
        and all(
            row.get("eligible") == "False"
            and row.get("included_in_failure_denominator") == "False"
            and row.get("not_applicable_reason") == "requested_cells_exceed_available_observations"
            for row in not_applicable_rows
        ),
        len(not_applicable_rows),
    )

    authorization_hashes = {
        payload.get("authorization", {}).get("sha256") for payload in payloads.values()
    }
    check("one_m2c_authorization", len(authorization_hashes) == 1, authorization_hashes)
    gate_relative = "governance/gates/G4_M2C_CONTROLS_SCALABILITY_2026-07-16.json"
    gate_path = root / gate_relative
    gate: dict[str, Any] = {}
    if gate_path.is_file():
        try:
            loaded_gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if isinstance(loaded_gate, dict):
                gate = loaded_gate
        except (OSError, json.JSONDecodeError):
            pass
    components = gate.get("components", {})
    components_valid = all(
        components.get(name, {}).get("path") == evidence_paths[name]
        and (root / evidence_paths[name]).is_file()
        and components.get(name, {}).get("sha256") == sha256_path(root / evidence_paths[name])
        and components.get(name, {}).get("status") == "pass"
        for name in evidence_paths
    )
    check(
        "completion_gate_binds_evidence",
        gate.get("status") == "pass"
        and gate.get("milestone") == "M2C_controls_scalability_construction"
        and gate.get("gate_exit_approved") is False
        and components_valid,
        gate_relative,
    )
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "pass" if checks_pass else "blocked",
        "paths": evidence_paths,
        "completion_gate": gate_relative,
        "checks": checks,
    }


def _m3a_completion_gate_binding(root: Path, path: Path) -> tuple[bool, str]:
    """Validate the M3A evidence binding, preferring an approved superseding gate."""

    frozen_path = root / M3A_FROZEN_GATE_RELATIVE
    superseding_path = root / M3A_SUPERSEDING_GATE_RELATIVE
    if superseding_path.is_file():
        try:
            superseding = _load_json(superseding_path)
        except (OSError, json.JSONDecodeError, ValueError):
            superseding = {}
        component = superseding.get("component", {})
        supersedes = superseding.get("supersedes", {})
        frozen_hash = sha256_path(frozen_path) if frozen_path.is_file() else None
        evidence_hash = sha256_path(path) if path.is_file() else None
        valid = (
            superseding.get("status") == "approved"
            and superseding.get("effective") is True
            and superseding.get("formal_user_signoff") is True
            and superseding.get("approver") == M3A_SUPERSEDING_APPROVER
            and superseding.get("approved_at_utc") == M3A_SUPERSEDING_APPROVED_AT_UTC
            and superseding.get("supersession_reason") == M3A_SUPERSESSION_REASON
            and superseding.get("gate") == "G4"
            and superseding.get("milestone") == "M3A_admissibility_portability_evaluation"
            and superseding.get("gate_exit_approved") is False
            and isinstance(supersedes, dict)
            and supersedes.get("path") == M3A_FROZEN_GATE_RELATIVE
            and supersedes.get("sha256") == frozen_hash
            and supersedes.get("status") == "pass"
            and supersedes.get("gate_exit_approved") is False
            and supersedes.get("edited") is False
            and isinstance(component, dict)
            and component.get("path") == M3A_SUPERSEDING_COMPONENT_RELATIVE
            and path == root / M3A_SUPERSEDING_COMPONENT_RELATIVE
            and component.get("status") == "pass"
            and isinstance(evidence_hash, str)
            and isinstance(frozen_hash, str)
            and component.get("sha256") == evidence_hash
            and component.get("sha256") != M3A_INCIDENT_EVIDENCE_SHA256
            and evidence_hash != M3A_INCIDENT_EVIDENCE_SHA256
        )
        return valid, M3A_SUPERSEDING_GATE_RELATIVE

    gate: dict[str, Any] = {}
    if frozen_path.is_file():
        try:
            loaded_gate = _load_json(frozen_path)
            if isinstance(loaded_gate, dict):
                gate = loaded_gate
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    component = gate.get("component", {})
    valid = (
        gate.get("status") == "pass"
        and gate.get("milestone") == "M3A_admissibility_portability_evaluation"
        and gate.get("gate_exit_approved") is False
        and component.get("path") == M3A_FROZEN_COMPONENT_RELATIVE
        and path == root / M3A_FROZEN_COMPONENT_RELATIVE
        and path.is_file()
        and component.get("sha256") == sha256_path(path)
        and component.get("status") == "pass"
    )
    return valid, M3A_FROZEN_GATE_RELATIVE


def _validate_g4_m3a(root: Path) -> dict[str, Any]:
    if (root / M3A_SUPERSEDING_GATE_RELATIVE).is_file():
        relative_path = M3A_SUPERSEDING_COMPONENT_RELATIVE
    else:
        relative_path = M3A_FROZEN_COMPONENT_RELATIVE
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    check("evidence_exists", path.is_file(), relative_path)
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            pass
    check("evidence_parses", bool(payload), relative_path)
    check("status_pass", payload.get("status") == "pass", payload.get("status"))
    check("gate", payload.get("gate") == "G4", payload.get("gate"))
    check("milestone", payload.get("milestone") == "M3A_admissibility_portability_evaluation", payload.get("milestone"))
    check(
        "task_panel_exact",
        set(payload.get("tasks_evaluated", [])) == {"construction_qc", "portability_runtime_failure"},
        payload.get("tasks_evaluated"),
    )
    check(
        "panel_partition",
        payload.get("planned_runs") == 195
        and payload.get("eligible_runs") == payload.get("eligible_runs_passed") == 189
        and payload.get("eligible_runs_failed") == 0
        and payload.get("full_data_eligible_runs") == 175
        and payload.get("subset_scalability_eligible_runs") == 14
        and payload.get("not_applicable_runs") == 6,
        (
            payload.get("planned_runs"), payload.get("eligible_runs"),
            payload.get("full_data_eligible_runs"), payload.get("subset_scalability_eligible_runs"),
            payload.get("not_applicable_runs"),
        ),
    )
    check(
        "contracts",
        payload.get("contract_checks_passed") == payload.get("contract_checks_total") == 3402,
        (payload.get("contract_checks_passed"), payload.get("contract_checks_total")),
    )
    check(
        "attempt_policy",
        payload.get("historical_engineering_attempts_reported") == 8
        and payload.get("historical_engineering_attempts_in_primary_denominator") is False,
        payload.get("historical_engineering_attempts_reported"),
    )
    check("not_applicable_denominator", payload.get("not_applicable_in_primary_denominator") is False, payload.get("not_applicable_in_primary_denominator"))
    check(
        "resource_scope",
        payload.get("resource_measurements_comparative") is False
        and payload.get("resource_interpretation") == "descriptive_noncomparative_only",
        payload.get("resource_interpretation"),
    )
    for field in (
        "universal_ranking_run", "method_ranking_run", "biological_task_evaluation_run",
        "rna_evaluation_run", "label_evaluation_run", "simulation_truth_evaluation_run",
        "holdout_accessed", "full_benchmark_run", "final_ranking_run",
        "manuscript_results_written", "next_milestone_authorized",
    ):
        check(f"{field}_false", payload.get(field) is False, payload.get(field))

    for key in ("authorization", "preregistration", "previous_gate", "config", "implementation"):
        artifact = payload.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )
    for key, artifact in payload.get("input_evidence", {}).items():
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"input_{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )

    expected_artifact_rows = {
        "run_table": 195,
        "success_rate_summary": 20,
        "not_applicable": 6,
        "resource_observations": 189,
        "historical_engineering_attempts": 8,
        "hardware_snapshot": None,
    }
    tables: dict[str, list[dict[str, str]]] = {}
    for key, expected_rows in expected_artifact_rows.items():
        artifact = payload.get("artifacts", {}).get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        exists = artifact_path.is_file()
        hash_valid = exists and sha256_path(artifact_path) == artifact.get("sha256")
        check(f"{key}_exists", exists, artifact.get("path"))
        check(f"{key}_hash", hash_valid, artifact.get("sha256"))
        if expected_rows is not None:
            rows: list[dict[str, str]] = []
            if exists:
                with artifact_path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
            tables[key] = rows
            check(f"{key}_rows", len(rows) == expected_rows == artifact.get("rows"), len(rows))

    run_rows = tables.get("run_table", [])
    eligible_rows = [row for row in run_rows if row.get("eligible") == "True"]
    not_applicable_rows = [row for row in run_rows if row.get("run_status") == "not_applicable"]
    subset_rows = [row for row in eligible_rows if row.get("subset_only") == "True"]
    check(
        "eligible_run_invariants",
        len(eligible_rows) == 189
        and all(
            row.get("run_status") == "pass"
            and row.get("valid_output_indicator") == "True"
            and row.get("included_in_primary_denominator") == "True"
            and row.get("output_contract_pass") == "True"
            and row.get("resource_comparison_allowed") == "False"
            for row in eligible_rows
        ),
        len(eligible_rows),
    )
    check("subset_separate", len(subset_rows) == 14 and {row.get("method_id") for row in subset_rows} == {"cicero_geneactivity"}, len(subset_rows))
    check(
        "not_applicable_invariants",
        len(not_applicable_rows) == 6
        and all(
            row.get("eligible") == "False"
            and row.get("included_in_primary_denominator") == "False"
            and row.get("method_id") == "cicero_geneactivity"
            and row.get("not_applicable_reason") == "requested_cells_exceed_available_observations"
            for row in not_applicable_rows
        ),
        len(not_applicable_rows),
    )
    summary_rows = tables.get("success_rate_summary", [])
    overall = [row for row in summary_rows if row.get("task_id") == "construction_qc" and row.get("scope_type") == "overall"]
    check(
        "overall_wilson_summary",
        len(overall) == 1
        and overall[0].get("eligible_runs") == overall[0].get("successful_runs") == "189"
        and overall[0].get("success_rate") == "1.0"
        and 0.97 < float(overall[0].get("wilson_95ci_low", 0)) < 1.0
        and overall[0].get("wilson_95ci_high") == "1.0",
        overall,
    )
    attempt_rows = tables.get("historical_engineering_attempts", [])
    check(
        "historical_attempts_excluded",
        len(attempt_rows) == 8
        and all(row.get("artifact_exists") == "True" and row.get("included_in_primary_denominator") == "False" for row in attempt_rows),
        len(attempt_rows),
    )
    hardware_artifact = payload.get("artifacts", {}).get("hardware_snapshot", {})
    hardware_path = root / str(hardware_artifact.get("path", ""))
    hardware = json.loads(hardware_path.read_text(encoding="utf-8")) if hardware_path.is_file() else {}
    check(
        "hardware_scope_noncomparative",
        hardware.get("resource_interpretation") == "descriptive_noncomparative_only"
        and isinstance(hardware.get("logical_cpu_count"), int)
        and hardware.get("logical_cpu_count", 0) > 0
        and hardware.get("memory_total_bytes", 0) > 0,
        hardware.get("cpu_model"),
    )

    binding_valid, binding_source = _m3a_completion_gate_binding(root, path)
    check("completion_gate_binds_evidence", binding_valid, binding_source)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m3b_preflight(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/m3b_preflight/g4_m3b_input_preflight_evidence.json"
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            pass
    check("evidence_exists_and_parses", path.is_file() and bool(payload), relative_path)
    check("status_pass", payload.get("status") == "pass", payload.get("status"))
    check("milestone", payload.get("milestone") == "M3B_input_preflight", payload.get("milestone"))
    check("rna_preflight", payload.get("rna_task_preflight") == "pass", payload.get("rna_task_preflight"))
    check(
        "domain_preflight_blocked",
        payload.get("domain_task_preflight") == "blocked_no_labels_in_current_m2_panel",
        payload.get("domain_task_preflight"),
    )
    check(
        "author_peak_extension_partial",
        payload.get("author_peak_domain_extension") == "partial_requires_separate_construction",
        payload.get("author_peak_domain_extension"),
    )
    for field in (
        "biological_outcome_computation_run", "rna_expression_values_read", "gene_activity_outcomes_read",
        "method_predictions_read", "rna_agreement_computed", "clustering_metrics_computed",
        "holdout_accessed", "full_benchmark_run", "final_ranking_run", "next_milestone_authorized",
    ):
        check(f"{field}_false", payload.get(field) is False, payload.get(field))
    for key in ("authorization", "previous_gate", "config", "implementation"):
        artifact = payload.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(f"{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)

    artifact_rows = {"rna_audit": None, "label_audit": 6, "task_eligibility": 4}
    tables: dict[str, list[dict[str, str]]] = {}
    for key, row_count in artifact_rows.items():
        artifact = payload.get("artifacts", {}).get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(f"{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)
        if row_count is not None:
            rows: list[dict[str, str]] = []
            if artifact_path.is_file():
                with artifact_path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
            tables[key] = rows
            check(f"{key}_rows", len(rows) == row_count == artifact.get("rows"), len(rows))

    rna_artifact = payload.get("artifacts", {}).get("rna_audit", {})
    rna_path = root / str(rna_artifact.get("path", ""))
    rna = json.loads(rna_path.read_text(encoding="utf-8")) if rna_path.is_file() else {}
    check(
        "rna_identity_contract",
        rna.get("status") == "pass"
        and rna.get("study_id") == "guo_2025_spatial_mux"
        and rna.get("matrix_rows") == rna.get("feature_rows") == 48440
        and rna.get("matrix_columns") == rna.get("rna_barcode_rows") == 10000
        and rna.get("final_atac_observations") == rna.get("final_atac_observations_in_rna") == 9686
        and rna.get("final_atac_observations_in_rna_spatial") == 9686
        and rna.get("features_mapped_to_canonical_gtf") == 42755
        and rna.get("features_mapped_protein_coding") == 21905,
        (rna.get("final_atac_observations"), rna.get("features_mapped_to_canonical_gtf")),
    )
    check(
        "rna_outcomes_absent",
        rna.get("rna_expression_values_read") is False
        and rna.get("gene_activity_outcomes_read") is False
        and rna.get("rna_agreement_computed") is False
        and rna.get("protein_coding_detection_threshold_computed") is False,
        rna.get("recommendation"),
    )
    label_rows = tables.get("label_audit", [])
    m2_rows = [row for row in label_rows if row.get("m2_panel") == "True"]
    author_rows = [row for row in label_rows if row.get("dataset_id") == "gse214986_author_peak"]
    check(
        "m2_labels_absent",
        len(m2_rows) == 5
        and all(row.get("status") == "no_independent_label_file" and row.get("label_file_exists") == "False" for row in m2_rows),
        len(m2_rows),
    )
    check(
        "author_labels_extension_only",
        len(author_rows) == 1
        and author_rows[0].get("label_file_exists") == "True"
        and author_rows[0].get("label_rows") == "17787"
        and author_rows[0].get("unique_labels") == "11"
        and author_rows[0].get("domain_metric_authorized_now") == "False"
        and author_rows[0].get("method_predictions_read") == "False",
        author_rows,
    )
    task_rows = {row.get("task_id"): row for row in tables.get("task_eligibility", [])}
    check(
        "task_decisions_exact",
        task_rows.get("paired_rna_fidelity", {}).get("preflight_status") == "pass"
        and task_rows.get("paired_rna_fidelity", {}).get("execution_status") == "requires_separate_authorization"
        and task_rows.get("spatial_domain_and_pathology", {}).get("preflight_status") == "blocked"
        and task_rows.get("spatial_domain_author_peak_extension", {}).get("preflight_status") == "partial"
        and task_rows.get("simulation_truth_recovery", {}).get("preflight_status") == "not_assessed_here",
        sorted(task_rows),
    )

    gate_relative = "governance/gates/G4_M3B_INPUT_PREFLIGHT_2026-07-16.json"
    gate_path = root / gate_relative
    gate: dict[str, Any] = {}
    if gate_path.is_file():
        try:
            loaded_gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if isinstance(loaded_gate, dict):
                gate = loaded_gate
        except (OSError, json.JSONDecodeError):
            pass
    component = gate.get("component", {})
    check(
        "completion_gate_binds_evidence",
        gate.get("status") == "pass"
        and gate.get("gate_exit_approved") is False
        and component.get("path") == relative_path
        and path.is_file()
        and component.get("sha256") == sha256_path(path)
        and component.get("status") == "pass",
        gate_relative,
    )
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m3b_rna(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/m3b_rna/g4_m3b_rna_completion_evidence.json"
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            pass
    check("evidence_exists_and_parses", path.is_file() and bool(payload), relative_path)
    check("status_pass", payload.get("status") == "pass", payload.get("status"))
    check("milestone", payload.get("milestone") == "M3B_paired_rna_fidelity_evaluation", payload.get("milestone"))
    check(
        "panel_dimensions",
        payload.get("methods_expected") == payload.get("methods_passed") == 31
        and payload.get("n_observations") == 9686
        and payload.get("rna_eligible_genes") == 5751
        and payload.get("gene_level_rows") == 178281
        and payload.get("not_applicable_methods") == 2,
        (payload.get("methods_passed"), payload.get("rna_eligible_genes"), payload.get("gene_level_rows")),
    )
    check(
        "generalization_scope",
        payload.get("generalization_scope") == "single_paired_rna_development_study_no_cross_study_inference",
        payload.get("generalization_scope"),
    )
    for field in (
        "rna_used_for_construction", "construction_outputs_mutated", "labels_or_clusters_used",
        "holdout_accessed", "full_benchmark_run", "method_ranking_run", "universal_ranking_run",
        "recommendation_policy_fit_run", "final_ranking_run", "manuscript_results_written", "next_milestone_authorized",
    ):
        check(f"{field}_false", payload.get(field) is False, payload.get(field))
    for key in ("authorization", "previous_gate", "config", "implementation", "evaluation_manifest", "process_tree_memory", "retained_resolved_failure"):
        artifact = payload.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(f"{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)
    for group in ("artifacts", "summary_artifacts"):
        for key, artifact in payload.get(group, {}).items():
            artifact_path = root / str(artifact.get("path", ""))
            check(f"{group}_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)

    metrics_artifact = payload.get("artifacts", {}).get("g4_m3b_rna_metrics.tsv", {})
    metrics_path = root / str(metrics_artifact.get("path", ""))
    metric_rows: list[dict[str, str]] = []
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle, delimiter="\t"))
    primary = "coverage_adjusted_median_gene_wise_spearman"
    annotation_rows = [row for row in metric_rows if row.get("method_category") == "hand_written_annotation"]
    software_rows = [row for row in metric_rows if row.get("method_category") == "fragment_software"]
    control_rows = [row for row in metric_rows if row.get("method_category") == "negative_control"]
    metrics_valid = len(metric_rows) == 31
    for row in metric_rows:
        try:
            median = float(row["median_gene_wise_spearman"])
            coverage = float(row["evaluable_gene_fraction"])
            observed = float(row[primary])
            row_valid = (
                row.get("status") == "pass"
                and row.get("rna_eligible_genes") == "5751"
                and -1 <= observed <= 1
                and abs(observed - median * coverage) <= 1e-12
            )
        except (KeyError, TypeError, ValueError):
            row_valid = False
        metrics_valid = metrics_valid and row_valid
    check(
        "metric_panel_and_formula",
        metrics_valid and len(annotation_rows) == 20 and len(software_rows) == 3 and len(control_rows) == 8,
        (len(annotation_rows), len(software_rows), len(control_rows)),
    )
    check(
        "software_track_separate",
        len(software_rows) == 3
        and all(row.get("input_track") == "F" for row in software_rows)
        and {row.get("method_id") for row in software_rows}
        == {"snapatac2_geneactivity", "signac_geneactivity", "archr_genescorematrix"},
        len(software_rows),
    )
    not_applicable_artifact = payload.get("artifacts", {}).get("g4_m3b_rna_not_applicable.tsv", {})
    not_applicable_path = root / str(not_applicable_artifact.get("path", ""))
    not_applicable_rows: list[dict[str, str]] = []
    if not_applicable_path.is_file():
        with not_applicable_path.open(encoding="utf-8", newline="") as handle:
            not_applicable_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "not_applicable_exact",
        len(not_applicable_rows) == 2
        and {row.get("method_id") for row in not_applicable_rows} == {"raw_peak_lsi_baseline", "cicero_geneactivity"}
        and all(row.get("included_in_primary_denominator") == "False" for row in not_applicable_rows),
        not_applicable_rows,
    )
    summary_expected = {"matched_control_deltas": 20, "representation_sensitivity": 5, "scope_summary": 5}
    summary_tables: dict[str, list[dict[str, str]]] = {}
    for key, count in summary_expected.items():
        artifact = payload.get("summary_artifacts", {}).get(key, {})
        table_path = root / str(artifact.get("path", ""))
        rows: list[dict[str, str]] = []
        if table_path.is_file():
            with table_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        summary_tables[key] = rows
        check(f"{key}_rows", len(rows) == count == artifact.get("rows"), len(rows))
    delta_rows = summary_tables.get("matched_control_deltas", [])
    check(
        "matched_control_deltas_finite",
        len(delta_rows) == 20
        and all(math.isfinite(float(row.get("delta_vs_accessibility_matched_control", "nan"))) for row in delta_rows),
        len(delta_rows),
    )
    sensitivity_rows = summary_tables.get("representation_sensitivity", [])
    check(
        "representation_sensitivity_exact",
        len(sensitivity_rows) == 5
        and all(row.get("representations") == "4" and float(row.get("representation_range", 0)) >= 0 for row in sensitivity_rows),
        len(sensitivity_rows),
    )
    gate_relative = "governance/gates/G4_M3B_RNA_FIDELITY_2026-07-16.json"
    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    component = gate.get("component", {})
    check(
        "completion_gate_binds_evidence",
        gate.get("status") == "pass"
        and gate.get("gate_exit_approved") is False
        and component.get("path") == relative_path
        and path.is_file()
        and component.get("sha256") == sha256_path(path)
        and component.get("status") == "pass",
        gate_relative,
    )
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m3c_simulation(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/m3c_simulation_truth_recovery_v2/g4_m3c_replicated_simulation_truth_recovery_evidence.json"
    analysis_relative = "results/v1/g4/m3c_simulation_truth_recovery_v2/analysis/g4_m3c_simulation_factorial_analysis_evidence.json"
    path = root / relative_path
    analysis_path = root / analysis_relative
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.is_file() else {}
    check("truth_recovery_evidence", path.is_file() and evidence.get("status") == "pass", relative_path)
    check(
        "truth_recovery_dimensions",
        evidence.get("simulation_rows") == 640
        and evidence.get("method_runs") == 4480
        and evidence.get("contract_checks_passed") == evidence.get("contract_checks_total") == 80640,
        (evidence.get("simulation_rows"), evidence.get("method_runs"), evidence.get("contract_checks_passed")),
    )
    check(
        "truth_recovery_boundaries",
        evidence.get("truth_used_for_construction") is False
        and evidence.get("simulated_rna_used") is False
        and evidence.get("labels_used_for_construction") is False
        and evidence.get("holdout_accessed") is False
        and evidence.get("full_benchmark_run") is False
        and evidence.get("method_ranking_run") is False
        and evidence.get("universal_ranking_run") is False
        and evidence.get("recommendation_policy_fit_run") is False
        and evidence.get("final_ranking_run") is False,
        evidence.get("scope"),
    )
    check(
        "component_and_rna_sentinels",
        evidence.get("component_endpoint_contract_passed") is True
        and evidence.get("component_endpoint_evidence") == "component_identifiability_confirmed"
        and evidence.get("rna_factor_leakage_sentinel_passed") is True,
        (evidence.get("component_endpoint_evidence"), evidence.get("rna_factor_leakage_sentinel_passed")),
    )
    for key in ("authorization", "previous_gate", "construction_manifest", "evaluation_manifest", "construction_memory", "evaluation_memory"):
        artifact = evidence.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(f"truth_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)
    for key, artifact in evidence.get("artifacts", {}).items():
        artifact_path = root / str(artifact.get("path", ""))
        check(f"truth_artifact_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)

    metrics_path = root / str(evidence.get("artifacts", {}).get("simulation_truth_recovery_metrics.tsv", {}).get("path", ""))
    metric_rows: list[dict[str, str]] = []
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle, delimiter="\t"))
    primary = "coverage_adjusted_mean_gene_wise_spearman_to_true_gene_activity"
    metrics_valid = len(metric_rows) == 4480
    balance: dict[tuple[str, str, str], int] = {}
    for row in metric_rows:
        try:
            mean = float(row["mean_gene_wise_spearman_variable_genes"])
            coverage = float(row["evaluable_gene_fraction"])
            observed = float(row[primary])
            row_valid = row.get("status") == "pass" and -1 <= observed <= 1 and abs(observed - mean * coverage) <= 1e-12
            key = (row["simulation_type"], row["method_id"], row["physical_random_seed"])
            balance[key] = balance.get(key, 0) + 1
        except (KeyError, TypeError, ValueError):
            row_valid = False
        metrics_valid = metrics_valid and row_valid
    check("metric_formula_and_rows", metrics_valid and len(balance) == 70 and set(balance.values()) == {64}, (len(metric_rows), len(balance)))

    check(
        "factorial_analysis_evidence",
        analysis_path.is_file()
        and analysis.get("status") == "pass"
        and analysis.get("method_runs") == 4480
        and analysis.get("seed_factor_coefficients") == 6930
        and analysis.get("aggregated_factor_effects") == 1386
        and analysis.get("rna_factor_regression_leakage_sentinel_passed") is True,
        analysis_relative,
    )
    check(
        "factorial_analysis_boundaries",
        analysis.get("higher_order_interactions_run") is False
        and analysis.get("holdout_accessed") is False
        and analysis.get("full_real_data_benchmark_run") is False
        and analysis.get("recommendation_policy_fit_run") is False
        and analysis.get("universal_ranking_run") is False
        and analysis.get("final_ranking_run") is False,
        analysis.get("seed_role"),
    )
    for key in ("authorization", "truth_recovery_evidence"):
        artifact = analysis.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(f"analysis_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)
    for key, artifact in analysis.get("artifacts", {}).items():
        artifact_path = root / str(artifact.get("path", ""))
        check(f"analysis_artifact_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)

    gate_relative = "governance/gates/G4_M3C_REPLICATED_SIMULATION_TRUTH_RECOVERY_2026-07-16.json"
    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    truth_component = gate.get("truth_recovery", {})
    analysis_component = gate.get("analysis", {})
    check(
        "completion_gate_binds_evidence",
        gate.get("status") == "pass"
        and gate.get("gate_exit_approved") is False
        and truth_component.get("path") == relative_path
        and truth_component.get("sha256") == sha256_path(path)
        and analysis_component.get("path") == analysis_relative
        and analysis_component.get("sha256") == sha256_path(analysis_path),
        gate_relative,
    )
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m3d_domain(root: Path) -> dict[str, Any]:
    relative_path = "results/v1/g4/m3d_gse214986/domain_evaluation/gse214986_domain_evaluation_evidence.json"
    path = root / relative_path
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    check("domain_evidence", path.is_file() and evidence.get("status") == "pass", relative_path)
    check(
        "domain_dimensions",
        evidence.get("methods_expected") == evidence.get("methods_passed") == 8
        and evidence.get("seed_runs_expected") == evidence.get("seed_runs_passed") == 40,
        (evidence.get("methods_passed"), evidence.get("seed_runs_passed")),
    )
    check(
        "domain_boundaries",
        evidence.get("labels_loaded_after_predictions") is True
        and evidence.get("labels_used_for_parameter_selection") is False
        and evidence.get("resolution_search_run") is False
        and evidence.get("construction_outputs_mutated") is False
        and evidence.get("holdout_accessed") is False
        and evidence.get("full_real_data_benchmark_run") is False
        and evidence.get("recommendation_policy_fit_run") is False
        and evidence.get("universal_ranking_run") is False
        and evidence.get("final_ranking_run") is False,
        evidence.get("label_interpretation"),
    )
    check(
        "label_and_seed_interpretation",
        evidence.get("label_interpretation") == "author_defined_domain_concordance_not_independent_histology_ground_truth"
        and evidence.get("seed_role") == "algorithmic_repeats_not_biological_replicates",
        (evidence.get("label_interpretation"), evidence.get("seed_role")),
    )
    for key in ("authorization", "construction_evidence", "environment_amendment"):
        artifact = evidence.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(f"domain_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)
    for key, artifact in evidence.get("artifacts", {}).items():
        artifact_path = root / str(artifact.get("path", ""))
        check(f"domain_artifact_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)

    metrics_artifact = evidence.get("artifacts", {}).get("gse214986_domain_seed_metrics.tsv", {})
    metrics_path = root / str(metrics_artifact.get("path", ""))
    metric_rows: list[dict[str, str]] = []
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle, delimiter="\t"))
    method_seed_counts: dict[str, set[int]] = {}
    metrics_valid = len(metric_rows) == 40
    for row in metric_rows:
        try:
            ari = float(row["ARI"])
            seed = int(row["seed"])
            method_seed_counts.setdefault(row["method_id"], set()).add(seed)
            row_valid = row.get("status") == "pass" and -1 <= ari <= 1 and row.get("labels_used_for_parameter_selection") == "False"
        except (KeyError, TypeError, ValueError):
            row_valid = False
        metrics_valid = metrics_valid and row_valid
    check(
        "domain_metric_panel",
        metrics_valid and len(method_seed_counts) == 8 and all(seeds == {1, 2, 3, 4, 5} for seeds in method_seed_counts.values()),
        (len(metric_rows), len(method_seed_counts)),
    )

    contrast_artifact = evidence.get("artifacts", {}).get("gse214986_domain_descriptive_contrasts.tsv", {})
    contrast_path = root / str(contrast_artifact.get("path", ""))
    contrast_rows: list[dict[str, str]] = []
    if contrast_path.is_file():
        with contrast_path.open(encoding="utf-8", newline="") as handle:
            contrast_rows = list(csv.DictReader(handle, delimiter="\t"))
    annotation_methods = {
        "promoter_only_2kb_pc",
        "gene_body_pc",
        "promoter_plus_gene_body_2kb_pc",
        "distance_weighted_distal_250kb_decay50kb_pc",
        "promoter_plus_distal_2kb_250kb_decay50kb_pc",
    }
    contrast_by_method = {row.get("method_id", ""): row for row in contrast_rows}
    directional_pass = all(
        contrast_by_method.get(method_id, {}).get("all_seeds_above_accessibility_matched_control") == "True"
        and contrast_by_method.get(method_id, {}).get("all_seeds_above_raw_peak_lsi") == "True"
        for method_id in annotation_methods
    )
    check("domain_registered_directional_contrasts", len(contrast_rows) == 8 and directional_pass, sorted(annotation_methods))

    amendment = evidence.get("environment_amendment", {})
    amendment_path = root / str(amendment.get("path", ""))
    amendment_payload = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.is_file() else {}
    environment_checks = []
    for key in ("base_environment_lock", "m3d_environment_lock"):
        artifact = amendment_payload.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        environment_checks.append(artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"))
    failure = amendment_payload.get("failed_canary_retained", {})
    for prefix in ("failure", "memory"):
        artifact_path = root / str(failure.get(f"{prefix}_path", ""))
        environment_checks.append(artifact_path.is_file() and sha256_path(artifact_path) == failure.get(f"{prefix}_sha256"))
    check("domain_environment_amendment", amendment_payload.get("status") == "active" and all(environment_checks), amendment.get("path"))

    gate_relative = "governance/gates/G4_M3D_GSE214986_AUTHOR_DOMAIN_CONCORDANCE_2026-07-16.json"
    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    evaluation = gate.get("evaluation", {})
    check(
        "domain_completion_gate_binds_evidence",
        gate.get("status") == "pass"
        and gate.get("gate_exit_approved") is False
        and gate.get("next_milestone_authorized") is False
        and evaluation.get("path") == relative_path
        and evaluation.get("sha256") == sha256_path(path),
        gate_relative,
    )
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": relative_path, "checks": checks}


def _validate_g4_m3e_replication(root: Path) -> dict[str, Any]:
    synthesis_relative = "results/v1/g4/m3e_cross_study_author_domain/g4_m3e_cross_study_author_domain_synthesis_evidence.json"
    synthesis_path = root / synthesis_relative
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    components = {
        "input": "results/v1/g4/m3e_gse288178_mouse_brain/input/gse288178_mouse_brain_input_preparation_evidence.json",
        "construction": "results/v1/g4/m3e_gse288178_mouse_brain/construction/gse288178_mouse_brain_construction_evidence.json",
        "domain": "results/v1/g4/m3e_gse288178_mouse_brain/domain_evaluation/gse288178_mouse_brain_domain_evaluation_evidence.json",
        "synthesis": synthesis_relative,
    }
    payloads = {
        key: json.loads((root / relative).read_text(encoding="utf-8")) if (root / relative).is_file() else {}
        for key, relative in components.items()
    }
    for key, relative in components.items():
        check(f"m3e_{key}_evidence", (root / relative).is_file() and payloads[key].get("status") == "pass", relative)
    input_evidence = payloads["input"]
    check(
        "m3e_input_leakage_boundary",
        input_evidence.get("n_observations") == 2161
        and input_evidence.get("n_fixed_500bp_bins") == 500000
        and input_evidence.get("author_clusters") == 10
        and input_evidence.get("construction_evaluation_inputs_used") is False
        and input_evidence.get("source_fragment_backing_preserved") is False
        and input_evidence.get("source_author_embeddings_preserved") is False
        and input_evidence.get("holdout_accessed") is False,
        input_evidence.get("label_interpretation"),
    )
    construction = payloads["construction"]
    check(
        "m3e_construction_panel",
        construction.get("methods_expected") == construction.get("methods_passed") == 8
        and construction.get("contract_checks_passed") == construction.get("contract_checks_total") == 144
        and construction.get("labels_accessed") is False
        and construction.get("clusters_accessed") is False
        and construction.get("rna_accessed") is False
        and construction.get("holdout_accessed") is False,
        (construction.get("methods_passed"), construction.get("contract_checks_passed")),
    )
    domain = payloads["domain"]
    check(
        "m3e_domain_panel_and_boundaries",
        domain.get("methods_expected") == domain.get("methods_passed") == 8
        and domain.get("seed_runs_expected") == domain.get("seed_runs_passed") == 40
        and domain.get("labels_loaded_after_predictions") is True
        and domain.get("labels_used_for_parameter_selection") is False
        and domain.get("resolution_search_run") is False
        and domain.get("holdout_accessed") is False
        and domain.get("universal_ranking_run") is False,
        domain.get("label_interpretation"),
    )
    for key, artifact in domain.get("artifacts", {}).items():
        artifact_path = root / str(artifact.get("path", ""))
        check(f"m3e_domain_artifact_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)

    synthesis = payloads["synthesis"]
    check(
        "m3e_synthesis_dimensions",
        synthesis.get("independent_study_families") == 2
        and synthesis.get("annotation_methods") == 5
        and synthesis.get("method_study_rows") == 10
        and synthesis.get("methods_replicating_positive_direction_vs_matched_control") == 4
        and synthesis.get("methods_replicating_positive_direction_vs_raw_lsi") == 0
        and synthesis.get("representations_reported_separately") is True,
        (
            synthesis.get("independent_study_families"),
            synthesis.get("methods_replicating_positive_direction_vs_matched_control"),
            synthesis.get("methods_replicating_positive_direction_vs_raw_lsi"),
        ),
    )
    check(
        "m3e_synthesis_boundaries",
        synthesis.get("study_level_population_inference_run") is False
        and synthesis.get("representation_pooling_run") is False
        and synthesis.get("holdout_accessed") is False
        and synthesis.get("full_real_data_benchmark_run") is False
        and synthesis.get("recommendation_policy_fit_run") is False
        and synthesis.get("universal_ranking_run") is False
        and synthesis.get("final_ranking_run") is False,
        synthesis.get("spearman_interpretation"),
    )
    for key, artifact in synthesis.get("artifacts", {}).items():
        artifact_path = root / str(artifact.get("path", ""))
        check(f"m3e_synthesis_artifact_{key}_hash", artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"), artifact)

    replication_artifact = synthesis.get("artifacts", {}).get("method_direction_replication.tsv", {})
    replication_path = root / str(replication_artifact.get("path", ""))
    replication_rows: list[dict[str, str]] = []
    if replication_path.is_file():
        with replication_path.open(encoding="utf-8", newline="") as handle:
            replication_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "m3e_direction_replication_table",
        len(replication_rows) == 5
        and sum(row.get("direction_vs_matched_control_replicates") == "True" for row in replication_rows) == 4
        and sum(row.get("direction_vs_raw_lsi_replicates") == "True" for row in replication_rows) == 0,
        len(replication_rows),
    )

    amendment_relative = "governance/authorizations/G4_M3E_ACCESSIBILITY_MATCHED_CONTROL_SCALABILITY_AMENDMENT_2026-07-16.json"
    amendment_path = root / amendment_relative
    amendment = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.is_file() else {}
    retained = amendment.get("retained_attempt", {})
    implementation = amendment.get("amended_implementation", {})
    check(
        "m3e_scalability_amendment",
        amendment.get("status") == "active_before_domain_outcomes"
        and amendment.get("domain_outcomes_inspected_before_amendment") is False
        and (root / str(retained.get("path", ""))).is_file()
        and sha256_path(root / str(retained.get("path", ""))) == retained.get("sha256")
        and (root / str(implementation.get("path", ""))).is_file()
        and sha256_path(root / str(implementation.get("path", ""))) == implementation.get("sha256"),
        amendment_relative,
    )

    gate_relative = "governance/gates/G4_M3E_CROSS_STUDY_AUTHOR_DOMAIN_REPLICATION_2026-07-16.json"
    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    synthesis_component = gate.get("cross_study_synthesis", {})
    check(
        "m3e_completion_gate_binds_evidence",
        gate.get("status") == "pass"
        and gate.get("gate_exit_approved") is False
        and gate.get("next_milestone_authorized") is False
        and synthesis_component.get("path") == synthesis_relative
        and synthesis_component.get("sha256") == sha256_path(synthesis_path),
        gate_relative,
    )
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": synthesis_relative, "checks": checks}


def _validate_g4_m3f_source_audit(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g4/m3f_histology_pathology_source_audit/evidence.json"
    manifest_relative = "data/manifests/v1_m3f_histology_source_audit.tsv"
    audit_relative = "results/v1/g4/m3f_histology_pathology_source_audit/source_asset_audit.tsv"
    protocol_relative = "protocols/IMAGE_ONLY_MORPHOLOGY_CONCORDANCE_PROTOCOL.md"
    report_relative = "reports/G4_M3F_HISTOLOGY_PATHOLOGY_SOURCE_AUDIT_2026-07-16.md"
    gate_relative = "governance/gates/G4_M3F_HISTOLOGY_PATHOLOGY_SOURCE_AUDIT_2026-07-16.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    check(
        "m3f_evidence_dimensions",
        evidence.get("status") == "partial"
        and evidence.get("zenodo_records_audited") == 5
        and evidence.get("archive_files_audited") == 134
        and evidence.get("nature_supplement_containers_audited") == 3
        and evidence.get("machine_readable_pathology_region_label_sources") == 0
        and evidence.get("aligned_histology_candidates") == 1
        and evidence.get("aligned_spots") == 7825,
        evidence.get("status"),
    )
    check(
        "m3f_claim_boundary",
        evidence.get("independent_pathology_truth_available") is False
        and evidence.get("image_only_morphology_candidate_available") is True
        and evidence.get("mask_role") == "binary_tissue_support_only",
        evidence.get("mask_role"),
    )
    for field in (
        "holdout_accessed",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
    ):
        check(f"m3f_{field}_false", evidence.get(field) is False, evidence.get(field))

    manifest_path = root / manifest_relative
    manifest_rows: list[dict[str, str]] = []
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_sources = {
        "zenodo_13964086", "zenodo_15425611", "zenodo_18798929", "zenodo_6565118",
        "zenodo_7395313", "nature_moesm2", "nature_moesm3", "nature_moesm4",
    }
    manifest_valid = len(manifest_rows) == 8 and {row.get("source_id") for row in manifest_rows} == expected_sources
    for row in manifest_rows:
        try:
            manifest_valid = manifest_valid and int(row["size_bytes"]) > 0 and len(row["sha256"]) == 64
        except (KeyError, TypeError, ValueError):
            manifest_valid = False
    check("m3f_source_manifest", manifest_valid, len(manifest_rows))

    audit_path = root / audit_relative
    audit_rows: list[dict[str, str]] = []
    if audit_path.is_file():
        with audit_path.open(encoding="utf-8", newline="") as handle:
            audit_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "m3f_audit_table",
        len(audit_rows) == 6
        and all(row.get("pathology_region_labels") == "no" for row in audit_rows)
        and sum(row.get("aligned_histology", "").startswith("yes") for row in audit_rows) == 1,
        len(audit_rows),
    )

    protocol_path = root / protocol_relative
    protocol = protocol_path.read_text(encoding="utf-8") if protocol_path.is_file() else ""
    check(
        "m3f_protocol_boundary",
        all(
            phrase in protocol
            for phrase in (
                "image-only morphology concordance",
                "does not measure pathology accuracy",
                "coordinate-only subtraction is mandatory",
                "GSE279771",
                "GSE308623",
                "do not fit a universal ranking",
            )
        ),
        protocol_relative,
    )

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    binding_valid = gate.get("status") == "partial" and gate.get("gate_exit_approved") is False
    for key, relative in (
        ("evidence", evidence_relative),
        ("source_manifest", manifest_relative),
        ("audit_table", audit_relative),
        ("protocol", protocol_relative),
        ("report", report_relative),
    ):
        artifact = gate.get(key, {})
        artifact_path = root / relative
        binding_valid = (
            binding_valid
            and artifact.get("path") == relative
            and artifact_path.is_file()
            and artifact.get("sha256") == sha256_path(artifact_path)
        )
    check(
        "m3f_gate_binds_audit",
        binding_valid
        and gate.get("independent_pathology_truth_available") is False
        and gate.get("image_only_morphology_protocol_frozen") is True
        and gate.get("image_only_morphology_execution_authorized") is False
        and gate.get("next_milestone_authorized") is False,
        gate_relative,
    )

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "partial" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g4_m3f_he_preflight(root: Path) -> dict[str, Any]:
    evidence_relative = (
        "results/v1/g4/m3f_gse278007_melanoma_he_preflight/"
        "gse278007_melanoma_he_preflight_evidence.json"
    )
    gate_relative = "governance/gates/G4_M3F_GSE278007_MELANOMA_HE_PREFLIGHT_2026-07-16.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    check(
        "m3f_he_preflight_decision",
        evidence.get("status") == "pass"
        and evidence.get("registration_status") == "not_evaluable_spot_level_registration"
        and evidence.get("section_relationship") == "adjacent_tissue_section"
        and evidence.get("author_roi_localization_passed") is True
        and evidence.get("direct_local_registration_passed") is False
        and evidence.get("spot_level_morphology_evaluation_authorized") is False
        and evidence.get("pathology_region_labels_available") is False,
        evidence.get("registration_status"),
    )
    for field in (
        "image_feature_extraction_run",
        "method_outputs_accessed",
        "atac_fragments_accessed",
        "rna_accessed",
        "clusters_accessed",
        "holdout_accessed",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3f_he_{field}_false", evidence.get(field) is False, evidence.get(field))
    for group in ("artifacts",):
        for key, artifact in evidence.get(group, {}).items():
            artifact_path = root / str(artifact.get("path", ""))
            check(
                f"m3f_he_{key}_hash",
                artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
                artifact,
            )
    for key in ("authorization", "protocol", "download_receipt", "runtime_lock", "source_he", "author_figure"):
        artifact = evidence.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"m3f_he_{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )

    qc_artifact = evidence.get("artifacts", {}).get("registration_qc.tsv", {})
    qc_path = root / str(qc_artifact.get("path", ""))
    qc_rows: list[dict[str, str]] = []
    if qc_path.is_file():
        with qc_path.open(encoding="utf-8", newline="") as handle:
            qc_rows = list(csv.DictReader(handle, delimiter="\t"))
    qc_by_stage = {row.get("stage"): row for row in qc_rows}
    try:
        he_localization = qc_by_stage["author_he_panel_to_source_he"]
        spatial_localization = qc_by_stage["source_spatial_to_author_spatial_panel"]
        direct = qc_by_stage["adjacent_section_direct_local_registration"]
        qc_valid = (
            len(qc_rows) == 3
            and int(he_localization["inliers"]) >= 100
            and float(he_localization["inlier_fraction"]) >= 0.90
            and int(spatial_localization["inliers"]) >= 20
            and float(spatial_localization["inlier_fraction"]) >= 0.60
            and (
                int(direct["good_matches"]) < 20
                or int(direct["inliers"]) < 15
                or float(direct["inlier_fraction"]) < 0.60
            )
        )
    except (KeyError, TypeError, ValueError):
        qc_valid = False
    check("m3f_he_registration_qc", qc_valid, len(qc_rows))

    transform_artifact = evidence.get("artifacts", {}).get("registration_transform.json", {})
    transform_path = root / str(transform_artifact.get("path", ""))
    transform = json.loads(transform_path.read_text(encoding="utf-8")) if transform_path.is_file() else {}
    check(
        "m3f_he_transform_not_evaluation_eligible",
        transform.get("status") == "not_evaluable_spot_level_registration"
        and transform.get("direct_local_registration_passed") is False
        and transform.get("approximate_transform_evaluation_eligible") is False
        and transform.get("mapped_centers") == 2500
        and transform.get("mapped_centers_in_he_bounds") is True,
        transform.get("status"),
    )

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("registration_status") == "not_evaluable_spot_level_registration"
        and gate.get("gate_exit_approved") is False
        and gate.get("spot_level_morphology_evaluation_authorized") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in ("evidence", "registration_transform", "registration_qc", "image_schema"):
        artifact = gate.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        gate_valid = (
            gate_valid
            and artifact_path.is_file()
            and artifact.get("sha256") == sha256_path(artifact_path)
        )
    check("m3f_he_gate_binds_evidence", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g4_m3f_malt_preflight(root: Path) -> dict[str, Any]:
    evidence_relative = (
        "results/v1/g4/m3f_gse288178_malt_input_preflight/"
        "gse288178_malt_input_preflight_evidence.json"
    )
    gate_relative = "governance/gates/G4_M3F_GSE288178_MALT_INPUT_PREFLIGHT_2026-07-16.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    check(
        "m3f_malt_alignment_decision",
        evidence.get("status") == "pass_with_source_subset_discrepancy"
        and evidence.get("coordinate_alignment_status") == "pass_released_7825_spot_subset"
        and evidence.get("released_locs") == 7825
        and evidence.get("published_filter_expected_locs") == 8097
        and evidence.get("unexplained_source_exclusions") == 272
        and evidence.get("mask_foreground_at_locs") == 7515
        and abs(float(evidence.get("mask_foreground_fraction", 0)) - 7515 / 7825) < 1e-15
        and evidence.get("mask_role") == "binary_tissue_support_not_pathology_label"
        and evidence.get("independent_pathology_truth_available") is False
        and evidence.get("image_feature_extraction_authorized") is False
        and evidence.get("spot_level_method_comparison_authorized") is False,
        evidence.get("coordinate_alignment_status"),
    )
    for field in (
        "h5ad_X_read",
        "fragment_values_read",
        "source_cluster_values_read",
        "image_feature_extraction_run",
        "method_outputs_accessed",
        "rna_accessed",
        "pathology_labels_accessed",
        "holdout_accessed",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3f_malt_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key, artifact in evidence.get("artifacts", {}).items():
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"m3f_malt_{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )
    for key in ("authorization", "protocol", "download_receipt", "source_h5ad"):
        artifact = evidence.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"m3f_malt_{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )

    validation_artifact = evidence.get("artifacts", {}).get("input_preflight_validation.json", {})
    validation_path = root / str(validation_artifact.get("path", ""))
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    models = validation.get("coordinate_models", {})
    check(
        "m3f_malt_input_validation",
        validation.get("status") == "pass_with_source_subset_discrepancy"
        and validation.get("h5ad_shape") == [8444, 0]
        and validation.get("h5ad_X_absent") is True
        and validation.get("h5ad_X_read") is False
        and validation.get("fragment_backing_present_but_values_not_read") is True
        and validation.get("unexplained_exclusions_after_published_filter") == 272
        and all(float(models[axis]["r_squared"]) >= 0.9999999 for axis in ("x", "y"))
        and all(float(models[axis]["maximum_absolute_residual_px"]) <= 0.51 for axis in ("x", "y")),
        validation.get("status"),
    )

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass_with_source_subset_discrepancy"
        and gate.get("gate_exit_approved") is False
        and gate.get("released_locs") == 7825
        and gate.get("unexplained_source_exclusions") == 272
        and gate.get("image_feature_extraction_authorized") is False
        and gate.get("spot_level_method_comparison_authorized") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in ("evidence", "alignment_summary", "input_validation"):
        artifact = gate.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        gate_valid = (
            gate_valid
            and artifact_path.is_file()
            and artifact.get("sha256") == sha256_path(artifact_path)
        )
    check("m3f_malt_gate_binds_evidence", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g4_m3f_image_features(root: Path) -> dict[str, Any]:
    evidence_relative = (
        "results/v1/g4/m3f_gse288178_malt_image_features/"
        "gse288178_malt_image_feature_evidence.json"
    )
    gate_relative = "governance/gates/G4_M3F_GSE288178_MALT_IMAGE_FEATURES_2026-07-16.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    check(
        "m3f_image_feature_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("feature_branch") == "handcrafted_he_v1"
        and evidence.get("observations") == 7825
        and evidence.get("foreground_observations") == 7515
        and evidence.get("raw_image_features") == 68
        and evidence.get("median_nearest_neighbor_px") == 79.0
        and evidence.get("patch_side_px") == 119
        and evidence.get("feature_matrix_finite") is True
        and evidence.get("constant_features") == 0,
        (evidence.get("observations"), evidence.get("raw_image_features")),
    )
    for field in (
        "h5ad_accessed",
        "atac_fragments_or_counts_accessed",
        "method_outputs_accessed",
        "method_identifiers_accessed",
        "rna_accessed",
        "source_clusters_accessed",
        "pathology_labels_accessed",
        "pca_or_model_fit_run",
        "method_comparison_run",
        "holdout_accessed",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3f_image_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key, artifact in evidence.get("artifacts", {}).items():
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"m3f_image_{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )
    for key in ("authorization", "previous_gate", "protocol", "source_he", "source_mask", "source_locs"):
        artifact = evidence.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"m3f_image_{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )

    schema_artifact = evidence.get("artifacts", {}).get("image_feature_schema.tsv", {})
    schema_path = root / str(schema_artifact.get("path", ""))
    schema_rows: list[dict[str, str]] = []
    if schema_path.is_file():
        with schema_path.open(encoding="utf-8", newline="") as handle:
            schema_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "m3f_image_schema",
        len(schema_rows) == 68
        and len({row.get("feature_id") for row in schema_rows}) == 68
        and all(row.get("feature_branch") == "handcrafted_he_v1" for row in schema_rows)
        and all(row.get("uses_molecular_input") == "False" for row in schema_rows),
        len(schema_rows),
    )
    qc_artifact = evidence.get("artifacts", {}).get("image_feature_qc.tsv", {})
    qc_path = root / str(qc_artifact.get("path", ""))
    qc_rows: list[dict[str, str]] = []
    if qc_path.is_file():
        with qc_path.open(encoding="utf-8", newline="") as handle:
            qc_rows = list(csv.DictReader(handle, delimiter="\t"))
    try:
        qc_valid = (
            len(qc_rows) == 68
            and all(float(row["finite_fraction"]) == 1.0 for row in qc_rows)
            and all(float(row["standard_deviation"]) > 0 for row in qc_rows)
        )
    except (KeyError, TypeError, ValueError):
        qc_valid = False
    check("m3f_image_feature_qc", qc_valid, len(qc_rows))

    features_artifact = evidence.get("artifacts", {}).get("handcrafted_he_v1_raw_features.tsv.gz", {})
    features_path = root / str(features_artifact.get("path", ""))
    feature_rows = 0
    foreground_rows = 0
    feature_header: list[str] = []
    if features_path.is_file():
        with gzip.open(features_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            feature_header = list(reader.fieldnames or [])
            for row in reader:
                feature_rows += 1
                foreground_rows += row.get("mask_foreground") == "True"
    expected_features = [row.get("feature_id") for row in schema_rows]
    check(
        "m3f_image_raw_matrix",
        feature_rows == 7825
        and foreground_rows == 7515
        and feature_header[:4] == ["spot", "image_x", "image_y", "mask_foreground"]
        and feature_header[4:] == expected_features,
        (feature_rows, foreground_rows, len(feature_header)),
    )

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("gate_exit_approved") is False
        and gate.get("image_target_frozen_before_method_comparison") is True
        and gate.get("method_comparison_authorized") is False
        and gate.get("pathology_accuracy_claim_allowed") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in ("evidence", "raw_features", "feature_schema", "feature_qc"):
        artifact = gate.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        gate_valid = (
            gate_valid
            and artifact_path.is_file()
            and artifact.get("sha256") == sha256_path(artifact_path)
        )
    check("m3f_image_gate_binds_features", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g4_m3f_malt_matched_representations(root: Path) -> dict[str, Any]:
    evidence_relative = (
        "results/v1/g4/m3f_gse288178_malt_matched_representations/evidence/"
        "gse288178_malt_matched_representation_evidence.json"
    )
    gate_relative = (
        "governance/gates/"
        "G4_M3F_GSE288178_MALT_MATCHED_REPRESENTATIONS_2026-07-16.json"
    )
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    check(
        "m3f_malt_matched_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("source_observations") == 8444
        and evidence.get("released_subset_observations") == 7825
        and evidence.get("excluded_source_observations") == 619
        and evidence.get("unexplained_source_exclusions_retained") == 272
        and evidence.get("representations_expected") == 4
        and evidence.get("representations_passed") == 4,
        (
            evidence.get("source_observations"),
            evidence.get("released_subset_observations"),
            evidence.get("representations_passed"),
        ),
    )
    check(
        "m3f_malt_fragment_semantics",
        evidence.get("source_encoded_fragment_rows") == 38875070
        and evidence.get("mitochondrial_fragment_rows_excluded") == 635821
        and evidence.get("nuclear_fragment_rows_constructed") == 38239249
        and evidence.get("obs_n_fragment_equals_standard_chromosome_rows") is True,
        (
            evidence.get("source_encoded_fragment_rows"),
            evidence.get("mitochondrial_fragment_rows_excluded"),
            evidence.get("nuclear_fragment_rows_constructed"),
        ),
    )
    for field in (
        "image_target_accessed",
        "he_pixels_accessed",
        "mask_values_accessed",
        "rna_accessed",
        "labels_or_clusters_accessed",
        "method_outputs_accessed",
        "holdout_accessed",
        "morphology_evaluation_run",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3f_malt_matched_{field}_false", evidence.get(field) is False, evidence.get(field))

    for relative, digest in evidence.get("construction_artifacts", {}).items():
        artifact = root / str(relative)
        check(
            f"m3f_malt_matched_artifact_{Path(relative).name}",
            artifact.is_file() and sha256_path(artifact) == digest,
            relative,
        )
    for key in ("authorization", "config", "summary", "checks"):
        artifact = evidence.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        check(
            f"m3f_malt_matched_{key}_hash",
            artifact_path.is_file() and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )

    summary_artifact = evidence.get("summary", {})
    summary_path = root / str(summary_artifact.get("path", ""))
    summary_rows: list[dict[str, str]] = []
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_representations = {
        "macs3_summit_501bp",
        "fixed_bin_10kb",
        "fixed_bin_25kb",
        "fixed_bin_50kb",
    }
    try:
        summary_valid = (
            len(summary_rows) == 4
            and {row["representation_id"] for row in summary_rows} == expected_representations
            and all(row["status"] == "pass" for row in summary_rows)
            and all(int(row["n_observations"]) == 7825 for row in summary_rows)
            and all(int(row["checks_passed"]) == int(row["checks_total"]) for row in summary_rows)
        )
    except (KeyError, TypeError, ValueError):
        summary_valid = False
    check("m3f_malt_matched_summary", summary_valid, len(summary_rows))

    checks_artifact = evidence.get("checks", {})
    checks_path = root / str(checks_artifact.get("path", ""))
    construction_checks: list[dict[str, str]] = []
    if checks_path.is_file():
        with checks_path.open(encoding="utf-8", newline="") as handle:
            construction_checks = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "m3f_malt_matched_shared_checks",
        bool(construction_checks)
        and all(row.get("status") == "pass" for row in construction_checks)
        and evidence.get("shared_checks_passed") == evidence.get("shared_checks_total")
        == len(construction_checks),
        len(construction_checks),
    )

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("gate_exit_approved") is False
        and gate.get("method_construction_authorized") is False
        and gate.get("morphology_evaluation_authorized") is False
        and gate.get("holdout_accessed") is False
        and gate.get("full_real_data_benchmark_run") is False
        and gate.get("final_ranking_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in ("evidence", "summary", "checks"):
        artifact = gate.get(key, {})
        artifact_path = root / str(artifact.get("path", ""))
        gate_valid = (
            gate_valid
            and artifact_path.is_file()
            and sha256_path(artifact_path) == artifact.get("sha256")
        )
    check("m3f_malt_matched_gate_binds_evidence", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g4_m3f_malt_method_construction(root: Path) -> dict[str, Any]:
    evidence_relative = (
        "results/v1/g4/m3f_gse288178_malt_method_construction/evidence/"
        "gse288178_malt_method_construction_evidence.json"
    )
    gate_relative = "governance/gates/G4_M3F_GSE288178_MALT_METHOD_CONSTRUCTION_2026-07-16.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    check(
        "m3f_malt_methods_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("representations") == 4
        and evidence.get("methods") == 8
        and evidence.get("runs_expected") == evidence.get("runs_passed") == 32
        and evidence.get("annotation_control_runs_passed") == 28
        and evidence.get("raw_lsi_runs_passed") == 4,
        (evidence.get("representations"), evidence.get("methods"), evidence.get("runs_passed")),
    )
    check(
        "m3f_malt_methods_checks",
        evidence.get("contract_checks_passed") == evidence.get("contract_checks_total") == 576
        and evidence.get("validation_checks_passed") == evidence.get("validation_checks_total") == 416,
        (
            evidence.get("contract_checks_passed"),
            evidence.get("validation_checks_passed"),
        ),
    )
    for field in (
        "performance_outcomes_interpreted",
        "image_target_accessed",
        "he_pixels_accessed",
        "mask_values_accessed",
        "rna_accessed",
        "labels_or_clusters_accessed",
        "morphology_evaluation_run",
        "holdout_accessed",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3f_malt_methods_{field}_false", evidence.get(field) is False, evidence.get(field))
    check(
        "m3f_malt_methods_construction_qc_only",
        evidence.get("construction_qc_only") is True,
        evidence.get("construction_qc_only"),
    )
    for key in ("summary", "authorization", "previous_gate", "representation_summary"):
        artifact = evidence.get(key, {})
        path = root / str(artifact.get("path", ""))
        check(
            f"m3f_malt_methods_{key}_hash",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )
    for relative, digest in evidence.get("configs", {}).items():
        path = root / relative
        check(
            f"m3f_malt_methods_config_{Path(relative).name}",
            path.is_file() and sha256_path(path) == digest,
            relative,
        )
    sidecars = evidence.get("bound_run_sidecars", {})
    sidecars_valid = len(sidecars) == 64 and all(
        (root / relative).is_file() and sha256_path(root / relative) == digest
        for relative, digest in sidecars.items()
    )
    check("m3f_malt_methods_64_sidecars", sidecars_valid, len(sidecars))

    summary_artifact = evidence.get("summary", {})
    summary_path = root / str(summary_artifact.get("path", ""))
    rows: list[dict[str, str]] = []
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    try:
        summary_valid = (
            len(rows) == 32
            and len({row["representation_id"] for row in rows}) == 4
            and len({row["method_id"] for row in rows}) == 8
            and all(row["status"] == "pass" for row in rows)
            and all(int(row["n_observations"]) == 7825 for row in rows)
            and all(
                int(row["contract_checks_passed"]) == int(row["contract_checks_total"]) == 18
                for row in rows
            )
            and all(
                int(row["validation_checks_passed"]) == int(row["validation_checks_total"])
                for row in rows
            )
        )
    except (KeyError, TypeError, ValueError):
        summary_valid = False
    check("m3f_malt_methods_summary", summary_valid, len(rows))

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("gate_exit_approved") is False
        and gate.get("runs_passed") == gate.get("runs_expected") == 32
        and gate.get("morphology_evaluation_authorized") is False
        and gate.get("performance_outcomes_interpreted") is False
        and gate.get("holdout_accessed") is False
        and gate.get("full_real_data_benchmark_run") is False
        and gate.get("final_ranking_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in ("evidence", "summary", "authorization"):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("m3f_malt_methods_gate_binds_evidence", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g4_m3f_malt_morphology(root: Path) -> dict[str, Any]:
    evidence_relative = (
        "results/v1/g4/m3f_gse288178_malt_morphology/evaluation/"
        "gse288178_malt_morphology_concordance_evidence.json"
    )
    analysis_relative = (
        "results/v1/g4/m3f_gse288178_malt_morphology/analysis/"
        "morphology_analysis_evidence.json"
    )
    gate_relative = "governance/gates/G4_M3F_GSE288178_MALT_MORPHOLOGY_CONCORDANCE_2026-07-16.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    analysis_path = root / analysis_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.is_file() else {}
    check(
        "m3f_morphology_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("runs") == 32
        and evidence.get("representations") == 4
        and evidence.get("methods") == 8
        and evidence.get("primary_foreground_observations") == 7515
        and evidence.get("all_registered_observations") == 7825
        and evidence.get("fold_metric_rows") == 320
        and evidence.get("summary_rows") == 64,
        (evidence.get("runs"), evidence.get("fold_metric_rows"), evidence.get("summary_rows")),
    )
    check(
        "m3f_morphology_primary_contract",
        evidence.get("primary_endpoint") == "spatial_block_coordinate_adjusted_delta_R2_image"
        and evidence.get("claim_boundary")
        == "image_only_morphology_concordance_not_pathology_accuracy"
        and evidence.get("image_scaler_and_pca_fit_training_folds_only") is True
        and evidence.get("coordinate_adjustment_applied") is True
        and evidence.get("negative_delta_values_retained") is True,
        evidence.get("primary_endpoint"),
    )
    for field in (
        "representation_pooling_run",
        "pathology_accuracy_claimed",
        "rna_accessed",
        "labels_or_clusters_accessed",
        "pathology_labels_accessed",
        "holdout_accessed",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3f_morphology_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key in ("authorization", "protocol", "previous_gate", "image_evidence", "method_evidence", "fold_evidence"):
        artifact = evidence.get(key, {})
        path = root / str(artifact.get("path", ""))
        check(
            f"m3f_morphology_{key}_hash",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )
    for key, artifact in evidence.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m3f_morphology_artifact_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    fold_artifact = evidence.get("fold_evidence", {})
    fold_path = root / str(fold_artifact.get("path", ""))
    fold_evidence = json.loads(fold_path.read_text(encoding="utf-8")) if fold_path.is_file() else {}
    check(
        "m3f_morphology_folds_molecularly_blind",
        fold_evidence.get("status") == "pass"
        and fold_evidence.get("folds") == 5
        and fold_evidence.get("minimum_foreground_per_fold", 0) >= 500
        and fold_evidence.get("method_outputs_accessed") is False
        and fold_evidence.get("method_identifiers_accessed") is False
        and fold_evidence.get("atac_accessed") is False
        and fold_evidence.get("model_fit_run") is False,
        fold_evidence.get("minimum_foreground_per_fold"),
    )
    check(
        "m3f_morphology_analysis_boundary",
        analysis.get("status") == "pass"
        and analysis.get("primary_fold_rows") == 160
        and analysis.get("absolute_summary_rows") == 32
        and analysis.get("control_contrast_rows") == 20
        and analysis.get("representation_rows") == 4
        and analysis.get("annotation_methods_with_positive_median_vs_raw_lsi") == 20
        and analysis.get("annotation_methods_total_across_representations") == 20
        and analysis.get("annotation_methods_all_folds_above_matched_control") == 3
        and analysis.get("methods_with_positive_median_absolute_R2") == 0
        and analysis.get("aggregation_controls_positive_in_all_representations") is True
        and analysis.get("raw_lsi_negative_in_all_representations") is True
        and analysis.get("representation_pooling_run") is False
        and analysis.get("universal_ranking_run") is False
        and analysis.get("final_ranking_run") is False,
        (
            analysis.get("annotation_methods_all_folds_above_matched_control"),
            analysis.get("methods_with_positive_median_absolute_R2"),
        ),
    )
    for key, artifact in analysis.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m3f_morphology_analysis_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("scientific_assessment") == "weak_absolute_fit_and_not_mapping_specific"
        and gate.get("gate_exit_approved") is False
        and gate.get("methods_with_positive_median_absolute_R2") == 0
        and gate.get("pathology_accuracy_claim_allowed") is False
        and gate.get("representation_pooling_run") is False
        and gate.get("holdout_accessed") is False
        and gate.get("full_real_data_benchmark_run") is False
        and gate.get("recommendation_policy_fit_run") is False
        and gate.get("universal_ranking_run") is False
        and gate.get("final_ranking_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "evaluation_evidence",
        "analysis_evidence",
        "fold_evidence",
        "fold_metrics",
        "method_summary",
        "representation_summary",
        "authorization",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("m3f_morphology_gate_binds_evidence", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g4_m3f_malt_nonlinear_sensitivity(root: Path) -> dict[str, Any]:
    evidence_relative = (
        "results/v1/g4/m3f_gse288178_malt_morphology/nonlinear_spatial_sensitivity/"
        "nonlinear_spatial_sensitivity_evidence.json"
    )
    gate_relative = (
        "governance/gates/"
        "G4_M3F_GSE288178_MALT_NONLINEAR_SPATIAL_SENSITIVITY_2026-07-17.json"
    )
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    check(
        "m3f_nonlinear_sensitivity_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("runs") == 64
        and evidence.get("representations") == 4
        and evidence.get("methods") == 8
        and evidence.get("spatial_baselines") == 2
        and evidence.get("fold_metric_rows") == 640
        and evidence.get("summary_rows") == 128
        and evidence.get("baseline_fold_rows") == 20
        and evidence.get("control_diagnostic_rows") == 16,
        (
            evidence.get("runs"),
            evidence.get("fold_metric_rows"),
            evidence.get("summary_rows"),
        ),
    )
    check(
        "m3f_nonlinear_sensitivity_boundary",
        evidence.get("analysis_class") == "post_primary_exploratory_diagnostic"
        and evidence.get("claim_boundary")
        == "diagnostic_spatial_baseline_sensitivity_not_primary_endpoint_or_pathology_accuracy"
        and evidence.get("post_primary") is True
        and evidence.get("exploratory_diagnostic_only") is True
        and evidence.get("primary_endpoint_replaced") is False
        and evidence.get("negative_values_retained") is True
        and evidence.get("image_scaler_and_pca_fit_training_folds_only") is True
        and evidence.get("spatial_coordinate_scaler_fit_training_folds_only") is True
        and evidence.get("spatial_basis_fit_training_folds_only") is True
        and evidence.get("rbf_landmarks_fit_training_folds_only") is True,
        evidence.get("claim_boundary"),
    )
    check(
        "m3f_nonlinear_sensitivity_result_contract",
        evidence.get("baseline_median_absolute_R2_improved_vs_primary_linear")
        == {"polynomial_degree2": True, "rbf_kmeans25": True}
        and evidence.get("primary_control_patterns", {}).get("polynomial_degree2")
        == {
            "annotation_branches_all_folds_above_matched_control": 1,
            "methods_with_positive_median_absolute_R2_across_representations": 0,
            "representations_with_both_aggregation_controls_positive": 4,
        }
        and evidence.get("primary_control_patterns", {}).get("rbf_kmeans25")
        == {
            "annotation_branches_all_folds_above_matched_control": 0,
            "methods_with_positive_median_absolute_R2_across_representations": 0,
            "representations_with_both_aggregation_controls_positive": 1,
        },
        evidence.get("primary_control_patterns"),
    )
    for field in (
        "method_selection_run",
        "representation_pooling_run",
        "pathology_accuracy_claimed",
        "rna_accessed",
        "labels_or_clusters_accessed",
        "pathology_labels_accessed",
        "holdout_accessed",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(
            f"m3f_nonlinear_sensitivity_{field}_false",
            evidence.get(field) is False,
            evidence.get(field),
        )
    for key, artifact in evidence.get("frozen_inputs", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m3f_nonlinear_sensitivity_input_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )
    for key, artifact in evidence.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m3f_nonlinear_sensitivity_artifact_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("analysis_class") == "post_primary_exploratory_diagnostic"
        and gate.get("scientific_assessment")
        == "nonlinear_baselines_improve_median_but_absolute_fit_remains_negative_and_delta_is_baseline_sensitive"
        and gate.get("gate_exit_approved") is False
        and gate.get("primary_endpoint_replaced") is False
        and gate.get("method_branches_with_positive_median_absolute_R2") == 0
        and gate.get("method_branches_total") == 64
        and gate.get("positive_median_delta_branches")
        == {"primary_linear": 28, "polynomial_degree2": 30, "rbf_kmeans25": 4}
        and gate.get("representations_with_both_random_controls_positive")
        == {"primary_linear": 4, "polynomial_degree2": 4, "rbf_kmeans25": 1}
        and gate.get("annotation_branches_all_folds_above_matched_control")
        == {"primary_linear": 3, "polynomial_degree2": 1, "rbf_kmeans25": 0}
        and gate.get("pathology_accuracy_claim_allowed") is False
        and gate.get("method_selection_run") is False
        and gate.get("representation_pooling_run") is False
        and gate.get("holdout_accessed") is False
        and gate.get("full_real_data_benchmark_run") is False
        and gate.get("recommendation_policy_fit_run") is False
        and gate.get("universal_ranking_run") is False
        and gate.get("final_ranking_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "authorization",
        "config",
        "evidence",
        "fold_metrics",
        "method_summary",
        "baseline_comparison",
        "control_diagnostics",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("m3f_nonlinear_sensitivity_gate_binds_evidence", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g4_m3g_pathology_truth_candidate(root: Path) -> dict[str, Any]:
    evidence_relative = (
        "results/v1/g4/m3g_pathology_truth_candidate_audit/preflight/"
        "pathology_truth_source_preflight_evidence.json"
    )
    gate_relative = (
        "governance/gates/"
        "G4_M3G_INDEPENDENT_PATHOLOGY_TRUTH_CANDIDATE_AUDIT_2026-07-17.json"
    )
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.is_file() else {}
    check(
        "m3g_pathology_source_preflight_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("candidate_study_id") == "wang_di_2025_nf_pannet"
        and evidence.get("accession") == "GSE304813"
        and evidence.get("panels") == 12
        and evidence.get("outline_line_masks") == 20
        and evidence.get("annotation_classes")
        == ["fibroblast", "normal_pancreas", "stroma", "tumor"]
        and evidence.get("unexpected_outline_masks") == 0,
        (evidence.get("panels"), evidence.get("outline_line_masks")),
    )
    check(
        "m3g_pathology_source_preflight_boundary",
        evidence.get("pathologist_annotations_available") is True
        and evidence.get("pathologist_annotations_machine_readable_at_source") is False
        and evidence.get("panel_crops_are_embedded_composite_not_raw_histology") is True
        and evidence.get("histology_is_serial_section") is True
        and evidence.get("gse_publicly_accessible") is False
        and evidence.get("outline_line_masks_only") is True
        and evidence.get("filled_region_masks_created") is False
        and evidence.get("spot_registration_run") is False
        and evidence.get("candidate_evaluation_ready") is False
        and evidence.get("redistribution_authorized") is False,
        evidence.get("candidate_recommendation"),
    )
    for field in (
        "atac_accessed",
        "gene_activity_accessed",
        "rna_accessed",
        "source_clusters_accessed",
        "holdout_accessed",
        "method_comparison_run",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3g_pathology_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key, artifact in evidence.get("frozen_inputs", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m3g_pathology_input_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )
    for key, artifact in evidence.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m3g_pathology_artifact_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    gate_path = root / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "partial"
        and gate.get("source_preflight_status") == "pass"
        and gate.get("scientific_assessment")
        == "public_replacement_search_complete_no_evaluation_ready_pathology_dataset"
        and gate.get("gate_exit_approved") is False
        and gate.get("independent_pathologist_annotation_candidate_found") is True
        and gate.get("public_replacement_search_complete") is True
        and gate.get("public_evaluation_ready_replacement_found") is False
        and gate.get("author_contact_critical_path") is False
        and gate.get("pathology_endpoint_core_blocker") is False
        and gate.get("candidate_accession") == "GSE304813"
        and gate.get("candidate_panels") == 12
        and gate.get("outline_line_masks") == 20
        and gate.get("annotation_classes") == 4
        and gate.get("unexpected_outline_masks") == 0
        and gate.get("candidate_evaluation_ready") is False
        and gate.get("gse_publicly_accessible") is False
        and gate.get("raw_histology_available") is False
        and gate.get("machine_readable_polygons_available") is False
        and gate.get("serial_section_registration_passed") is False
        and gate.get("redistribution_authorized") is False
        and gate.get("filled_region_masks_created") is False
        and gate.get("holdout_accessed") is False
        and gate.get("method_comparison_run") is False
        and gate.get("full_real_data_benchmark_run") is False
        and gate.get("recommendation_policy_fit_run") is False
        and gate.get("universal_ranking_run") is False
        and gate.get("final_ranking_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "authorization",
        "config",
        "evidence",
        "candidate_summary",
        "panel_qc",
        "outline_inventory",
        "source_manifest",
        "report",
        "author_request_draft",
        "public_replacement_candidates",
        "public_replacement_source_manifest",
        "public_replacement_report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("m3g_pathology_gate_binds_evidence", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "partial" if checks_pass else "blocked",
        "path": evidence_relative,
        "checks": checks,
    }


def _validate_g4_m3h_cross_study_rna(root: Path) -> dict[str, Any]:
    preflight_relative = (
        "results/v1/g4/m3h_gse205052_rna_replication/pairing_preflight_evidence.json"
    )
    evaluation_relative = (
        "results/v1/g4/m3h_gse205052_rna_replication/completion_evidence.json"
    )
    synthesis_relative = (
        "results/v1/g4/m3h_cross_study_rna_synthesis/cross_study_synthesis_evidence.json"
    )
    gate_relative = (
        "governance/gates/G4_M3H_CROSS_STUDY_PAIRED_RNA_REPLICATION_2026-07-17.json"
    )
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append(
            {"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)}
        )

    preflight_path = root / preflight_relative
    preflight = _load_json(preflight_path) if preflight_path.is_file() else {}
    check(
        "m3h_pairing_preflight_dimensions",
        preflight.get("status") == "pass"
        and preflight.get("study_id") == "zhang_2023"
        and preflight.get("dataset_id") == "gse205052_mousebrain_20um"
        and preflight.get("rna_matrix_barcodes") == 2373
        and preflight.get("rna_spatial_grid_barcodes") == 2500
        and preflight.get("final_atac_observations") == 2500
        and preflight.get("evaluation_observations_frozen") == 2373
        and preflight.get("atac_only_observations_excluded") == 127
        and preflight.get("mapped_gtf_gene_names") == 18181
        and preflight.get("mapped_protein_coding_gene_names") == 15453,
        (
            preflight.get("evaluation_observations_frozen"),
            preflight.get("atac_only_observations_excluded"),
        ),
    )
    preflight_checks = preflight.get("checks", {})
    check(
        "m3h_pairing_preflight_checks",
        isinstance(preflight_checks, dict)
        and len(preflight_checks) == 14
        and all(value is True for value in preflight_checks.values()),
        preflight_checks,
    )
    for field in (
        "rna_expression_values_read",
        "gene_activity_outcomes_read",
        "rna_agreement_computed",
        "construction_outputs_mutated",
        "labels_or_clusters_accessed",
        "holdout_accessed",
        "recommendation_policy_fit_run",
        "full_benchmark_run",
        "universal_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3h_preflight_{field}_false", preflight.get(field) is False, preflight.get(field))
    for key, artifact in preflight.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m3h_preflight_artifact_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    evaluation_path = root / evaluation_relative
    evaluation = _load_json(evaluation_path) if evaluation_path.is_file() else {}
    check(
        "m3h_rna_evaluation_dimensions",
        evaluation.get("status") == "pass"
        and evaluation.get("study_id") == "zhang_2023"
        and evaluation.get("dataset_id") == "gse205052_mousebrain_20um"
        and evaluation.get("methods_expected") == 31
        and evaluation.get("methods_passed") == 31
        and evaluation.get("rna_eligible_genes") == 5301
        and evaluation.get("n_observations") == 2373
        and evaluation.get("source_atac_observations") == 2500
        and evaluation.get("not_applicable_methods") == 2
        and evaluation.get("gene_level_rows") == 164331,
        (
            evaluation.get("methods_passed"),
            evaluation.get("rna_eligible_genes"),
            evaluation.get("gene_level_rows"),
        ),
    )
    for field in (
        "rna_used_for_construction",
        "construction_outputs_mutated",
        "labels_or_clusters_used",
        "holdout_accessed",
        "recommendation_policy_fit_run",
        "full_benchmark_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3h_evaluation_{field}_false", evaluation.get(field) is False, evaluation.get(field))
    for group in ("artifacts", "summary_artifacts"):
        for key, artifact in evaluation.get(group, {}).items():
            path = root / str(artifact.get("path", ""))
            check(
                f"m3h_evaluation_{group}_{key}",
                path.is_file() and sha256_path(path) == artifact.get("sha256"),
                artifact,
            )

    synthesis_path = root / synthesis_relative
    synthesis = _load_json(synthesis_path) if synthesis_path.is_file() else {}
    check(
        "m3h_cross_study_synthesis_dimensions",
        synthesis.get("status") == "pass"
        and synthesis.get("study_families") == 2
        and synthesis.get("method_runs") == 62
        and synthesis.get("matrix_track_runs") == 56
        and synthesis.get("fragment_track_runs") == 6
        and synthesis.get("direction_rows") == 20
        and synthesis.get("positive_direction_replication_rows") == 20,
        (
            synthesis.get("study_families"),
            synthesis.get("direction_rows"),
            synthesis.get("positive_direction_replication_rows"),
        ),
    )
    for field in (
        "input_tracks_pooled",
        "representation_universal_rank_run",
        "recommendation_policy_fit_run",
        "holdout_accessed",
        "full_benchmark_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"m3h_synthesis_{field}_false", synthesis.get(field) is False, synthesis.get(field))
    for key, artifact in synthesis.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m3h_synthesis_artifact_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("scientific_assessment")
        == "annotation_mapping_signal_replicates_while_representation_preference_is_study_dependent"
        and gate.get("gate_exit_approved") is False
        and gate.get("recommendation_policy_frozen") is False
        and gate.get("study_families") == 2
        and gate.get("method_runs") == 62
        and gate.get("direction_rows") == 20
        and gate.get("positive_direction_replication_rows") == 20
        and gate.get("second_study_methods_passed") == 31
        and gate.get("second_study_rna_eligible_genes") == 5301
        and gate.get("second_study_evaluation_observations") == 2373
        and gate.get("second_study_atac_only_observations_excluded") == 127
        and gate.get("rna_imputation_run") is False
        and gate.get("input_tracks_pooled") is False
        and gate.get("universal_representation_selected") is False
        and gate.get("construction_outputs_mutated") is False
        and gate.get("labels_or_clusters_used") is False
        and gate.get("holdout_accessed") is False
        and gate.get("recommendation_policy_fit_run") is False
        and gate.get("full_benchmark_run") is False
        and gate.get("universal_ranking_run") is False
        and gate.get("final_ranking_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "preflight_authorization",
        "evaluation_authorization",
        "synthesis_authorization",
        "preflight_config",
        "evaluation_config",
        "synthesis_config",
        "preflight_evidence",
        "evaluation_evidence",
        "synthesis_evidence",
        "direction_table",
        "implementation",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("m3h_gate_binds_evidence", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": synthesis_relative, "checks": checks}


def _validate_g4_m4_recommendation_policy(root: Path) -> dict[str, Any]:
    completion_relative = "results/v1/g4/m4_recommendation_policy/completion_evidence.json"
    policy_relative = "results/v1/g4/m4_recommendation_policy/frozen_task_specific_recommendation_policy.json"
    validation_relative = "results/v1/g4/m4_recommendation_policy/policy_validation.json"
    gate_relative = "governance/gates/G4_M4_TASK_SPECIFIC_RECOMMENDATION_POLICY_FREEZE_2026-07-17.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    completion_path = root / completion_relative
    completion = _load_json(completion_path) if completion_path.is_file() else {}
    check(
        "m4_completion_dimensions",
        completion.get("status") == "pass"
        and completion.get("policy_frozen") is True
        and completion.get("active_policy_leaves") == 10
        and set(completion.get("active_tasks", []))
        == {"paired_rna_fidelity", "portability_runtime_failure"}
        and set(completion.get("input_tracks_separate", [])) == {"F", "P", "B"}
        and completion.get("minimum_studies_per_active_leaf") == 2
        and completion.get("maximum_methods_returned") == 3,
        (
            completion.get("active_policy_leaves"),
            completion.get("minimum_studies_per_active_leaf"),
            completion.get("maximum_methods_returned"),
        ),
    )
    for field in (
        "rna_or_labels_used_for_construction",
        "holdout_accessed",
        "full_benchmark_run",
        "universal_representation_selected",
        "cross_task_utility_pooling_run",
        "cross_representation_utility_pooling_run",
        "runtime_or_memory_comparison_run",
        "universal_ranking_run",
        "final_ranking_run",
        "g5_execution_run",
        "next_milestone_authorized",
    ):
        check(f"m4_completion_{field}_false", completion.get(field) is False, completion.get(field))
    for key, artifact in completion.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"m4_completion_artifact_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    policy_path = root / policy_relative
    policy = _load_json(policy_path) if policy_path.is_file() else {}
    leaves = policy.get("leaves", [])
    active = [leaf for leaf in leaves if isinstance(leaf, dict) and leaf.get("status") == "active_frozen"]
    insufficient = [
        leaf
        for leaf in leaves
        if isinstance(leaf, dict) and leaf.get("status") == "insufficient_evidence_no_recommendation"
    ]
    active_leaf_contract = len(active) == 10 and all(
        isinstance(leaf.get("n_study_families"), int)
        and leaf["n_study_families"] >= 2
        and isinstance(leaf.get("n_recommended_methods"), int)
        and 1 <= leaf["n_recommended_methods"] <= 3
        and leaf.get("input_track") in {"F", "P", "B"}
        and bool(leaf.get("support_species"))
        and bool(leaf.get("support_genome_builds"))
        for leaf in active
    )
    check("m4_active_leaf_contract", active_leaf_contract, len(active))
    for task in ("paired_rna_fidelity", "portability_runtime_failure"):
        tracks = {leaf.get("input_track") for leaf in active if leaf.get("task_id") == task}
        check(f"m4_{task}_tracks_exact", tracks == {"F", "P", "B"}, sorted(tracks))
    rna_leaves = [leaf for leaf in active if leaf.get("task_id") == "paired_rna_fidelity"]
    portability_leaves = [leaf for leaf in active if leaf.get("task_id") == "portability_runtime_failure"]
    check(
        "m4_rna_species_support_mouse_only",
        all(leaf.get("support_species") == "mouse" and leaf.get("support_genome_builds") == "mm10" for leaf in rna_leaves),
        [(leaf.get("support_species"), leaf.get("support_genome_builds")) for leaf in rna_leaves],
    )
    check(
        "m4_portability_species_support_mouse_human",
        all(
            set(str(leaf.get("support_species", "")).split(";")) == {"mouse", "human"}
            and set(str(leaf.get("support_genome_builds", "")).split(";")) == {"mm10", "hg38"}
            for leaf in portability_leaves
        ),
        [(leaf.get("support_species"), leaf.get("support_genome_builds")) for leaf in portability_leaves],
    )
    check(
        "m4_required_abstentions",
        {leaf.get("task_id") for leaf in insufficient}
        == {"spatial_domain_and_pathology", "simulation_truth_recovery", "independent_pathology_accuracy"}
        and all(leaf.get("n_recommended_methods") == 0 for leaf in insufficient),
        [leaf.get("task_id") for leaf in insufficient],
    )
    check(
        "m4_frozen_constants",
        policy.get("status") == "frozen"
        and policy.get("maximum_recommended_methods") == 3
        and policy.get("minimum_development_studies_per_leaf") == 2
        and policy.get("maximum_context_splits") == 3
        and policy.get("regret_noninferiority_margin") == 0.05
        and policy.get("failure_rate_noninferiority_margin") == 0.10
        and policy.get("random_comparator_draws") == 10000
        and policy.get("random_seed") == 41017,
        policy.get("policy_id"),
    )
    for field in (
        "cross_task_utility_pooling",
        "cross_representation_utility_pooling",
        "runtime_or_memory_method_comparison",
        "universal_representation_selection",
        "universal_ranking",
        "holdout_accessed",
        "g5_execution_authorized",
    ):
        check(f"m4_policy_{field}_false", policy.get(field) is False, policy.get(field))

    validation_path = root / validation_relative
    validation = _load_json(validation_path) if validation_path.is_file() else {}
    validation_checks = validation.get("checks", [])
    check(
        "m4_policy_validation_pass",
        validation.get("status") == "pass"
        and len(validation_checks) >= 10
        and all(row.get("status") == "pass" for row in validation_checks if isinstance(row, dict)),
        len(validation_checks),
    )
    rules_path = root / "spatialgabench/recommendation/rules.py"
    rules_text = rules_path.read_text(encoding="utf-8") if rules_path.is_file() else ""
    legacy_markers = ("STAGE" + "23", "STAGE" + "24")
    check(
        "m4_legacy_stage_evidence_removed",
        all(marker not in rules_text for marker in legacy_markers),
        "recommendation/rules.py",
    )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("policy_frozen") is True
        and gate.get("g4_development_exit_criteria_met") is True
        and gate.get("g5_execution_authorized") is False
        and gate.get("active_policy_leaves") == 10
        and gate.get("minimum_studies_per_active_leaf") == 2
        and gate.get("maximum_methods_returned") == 3
        and gate.get("legacy_stage23_stage24_references_remaining") == 0
        and gate.get("holdout_accessed") is False
        and gate.get("full_benchmark_run") is False
        and gate.get("universal_ranking_run") is False
        and gate.get("final_ranking_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "authorization",
        "config",
        "completion_evidence",
        "policy",
        "validation",
        "policy_leaves",
        "implementation",
        "exact_contract_api",
        "workflow",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("m4_gate_binds_frozen_policy", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": policy_relative, "checks": checks}


def _validate_g5_h0_preaccess_feasibility(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h0_preaccess_feasibility/completion_evidence.json"
    decision_relative = "results/v1/g5/h0_preaccess_feasibility/g5_access_decision.json"
    summary_relative = "results/v1/g5/h0_preaccess_feasibility/holdout_feasibility_summary.tsv"
    eligibility_relative = "results/v1/g5/h0_preaccess_feasibility/holdout_policy_leaf_eligibility.tsv"
    gate_relative = "governance/gates/G5_H0_PREACCESS_FEASIBILITY_AUDIT_2026-07-17.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    check(
        "g5_h0_completion_decision",
        evidence.get("status") == "blocked"
        and evidence.get("audit_completed") is True
        and evidence.get("access_decision") == "do_not_unseal_preserve_remaining_holdout"
        and evidence.get("holdouts_audited") == 2
        and evidence.get("active_policy_leaves_audited_per_holdout") == 10
        and evidence.get("exact_leaf_evaluable_holdouts") == 1
        and evidence.get("minimum_untouched_studies_required") == 2
        and evidence.get("c2_external_validation_ready") is False,
        (
            evidence.get("access_decision"),
            evidence.get("exact_leaf_evaluable_holdouts"),
            evidence.get("minimum_untouched_studies_required"),
        ),
    )
    for field in (
        "raw_holdout_accessed",
        "holdout_mount_run",
        "holdout_checksum_revalidation_run",
        "representation_construction_run",
        "method_execution_run",
        "rna_value_access_run",
        "label_or_pathology_value_access_run",
        "regret_evaluation_run",
        "policy_refit_run",
        "full_benchmark_run",
        "results_writing_run",
        "universal_ranking_run",
        "final_ranking_run",
        "g5_execution_authorized",
        "next_milestone_authorized",
    ):
        check(f"g5_h0_{field}_false", evidence.get(field) is False, evidence.get(field))
    check("g5_h0_raw_paths_opened_zero", evidence.get("raw_holdout_paths_opened") == 0, evidence.get("raw_holdout_paths_opened"))
    for key, artifact in evidence.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h0_evidence_artifact_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    decision_path = root / decision_relative
    decision = _load_json(decision_path) if decision_path.is_file() else {}
    check(
        "g5_h0_access_decision_blocked",
        decision.get("decision") == "do_not_unseal_preserve_remaining_holdout"
        and decision.get("c2_external_validation_ready") is False
        and decision.get("exact_leaf_evaluable_holdouts") == 1
        and decision.get("raw_holdout_accessed") is False
        and decision.get("g5_execution_authorized") is False,
        decision.get("decision"),
    )

    summary_path = root / summary_relative
    summary_rows: list[dict[str, str]] = []
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle, delimiter="\t"))
    summary = {row.get("study_id"): row for row in summary_rows}
    check(
        "g5_h0_holdout_summary_exact",
        len(summary_rows) == 2
        and summary.get("zhang_2025_brain_triomics", {}).get("exact_active_leaves_evaluable") == "10"
        and summary.get("zhang_2025_brain_triomics", {}).get("status") == "evaluable"
        and summary.get("huang_2025", {}).get("exact_active_leaves_evaluable") == "0"
        and summary.get("huang_2025", {}).get("status") == "out_of_policy_support_abstention_only",
        [(row.get("study_id"), row.get("exact_active_leaves_evaluable"), row.get("status")) for row in summary_rows],
    )
    eligibility_path = root / eligibility_relative
    eligibility_rows: list[dict[str, str]] = []
    if eligibility_path.is_file():
        with eligibility_path.open(encoding="utf-8", newline="") as handle:
            eligibility_rows = list(csv.DictReader(handle, delimiter="\t"))
    h1_eligible = sum(
        row.get("exact_active_leaf_evaluable") == "True"
        for row in eligibility_rows
        if row.get("study_id") == "zhang_2025_brain_triomics"
    )
    h2_eligible = sum(
        row.get("exact_active_leaf_evaluable") == "True"
        for row in eligibility_rows
        if row.get("study_id") == "huang_2025"
    )
    check(
        "g5_h0_leaf_eligibility_exact",
        len(eligibility_rows) == 20
        and h1_eligible == 10
        and h2_eligible == 0
        and all(row.get("representation_alias_used") == "False" for row in eligibility_rows)
        and all(row.get("holdout_outcome_used") == "False" for row in eligibility_rows),
        (len(eligibility_rows), h1_eligible, h2_eligible),
    )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "blocked"
        and gate.get("audit_status") == "pass"
        and gate.get("access_decision") == "do_not_unseal_preserve_remaining_holdout"
        and gate.get("c2_external_validation_ready") is False
        and gate.get("exact_leaf_evaluable_holdouts") == 1
        and gate.get("gse308623_exact_active_leaves") == 10
        and gate.get("gse279771_exact_active_leaves") == 0
        and gate.get("raw_holdout_paths_opened") == 0
        and gate.get("raw_holdout_accessed") is False
        and gate.get("holdout_outcomes_inspected") is False
        and gate.get("g5_execution_authorized") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "authorization",
        "config",
        "completion_evidence",
        "access_decision_artifact",
        "leaf_eligibility",
        "holdout_summary",
        "implementation",
        "workflow",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h0_gate_binds_no_go_decision", gate_valid, gate_relative)

    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "blocked_second_evaluable_holdout_missing" if checks_pass else "blocked_invalid_audit",
        "path": evidence_relative,
        "checks": checks,
    }


def _validate_g5_h1_supplemental_holdout(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h1_supplemental_holdout_candidate/completion_evidence.json"
    gate_relative = "governance/gates/G5_H1_SUPPLEMENTAL_HOLDOUT_FREEZE_2026-07-17.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    source_checks = evidence.get("checks", {})
    check(
        "g5_h1_candidate_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("decision") == "freeze_GSE307620_GSM9228169_as_supplemental_portability_holdout"
        and evidence.get("selected_accession") == "GSE307620"
        and evidence.get("selected_sample_accession") == "GSM9228169"
        and evidence.get("positions_rows") == 2500
        and evidence.get("in_tissue_rows") == 1966
        and evidence.get("exact_active_policy_leaves") == 5
        and evidence.get("evaluable_task") == "portability_runtime_failure"
        and evidence.get("paired_rna_available") is False
        and evidence.get("biological_utility_holdout_added") is False
        and evidence.get("minimal_fragment_download_bytes") == 2219237716,
        (
            evidence.get("selected_accession"),
            evidence.get("positions_rows"),
            evidence.get("exact_active_policy_leaves"),
        ),
    )
    check(
        "g5_h1_small_file_checks_pass",
        isinstance(source_checks, dict) and len(source_checks) >= 20 and all(value is True for value in source_checks.values()),
        len(source_checks) if isinstance(source_checks, dict) else "invalid",
    )
    for field in (
        "large_fragment_download_run",
        "raw_fastq_or_bam_download_run",
        "method_or_task_outcome_used_for_selection",
        "method_execution_run",
        "existing_holdout_accessed",
        "policy_refit_run",
        "full_benchmark_run",
        "results_writing_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"g5_h1_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key, artifact in evidence.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h1_evidence_artifact_{key}",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("supplemental_holdout_frozen") is True
        and gate.get("selected_accession") == "GSE307620"
        and gate.get("selected_sample_accession") == "GSM9228169"
        and gate.get("positions_rows") == 2500
        and gate.get("in_tissue_rows") == 1966
        and gate.get("exact_active_policy_leaves") == 5
        and gate.get("two_study_external_portability_validation_possible") is True
        and gate.get("two_study_external_paired_rna_validation_possible") is False
        and gate.get("minimal_fragment_download_bytes") == 2219237716
        and gate.get("large_fragment_download_run") is False
        and gate.get("large_fragment_download_authorized") is False
        and gate.get("method_or_task_outcome_used_for_selection") is False
        and gate.get("existing_holdout_accessed") is False
        and gate.get("candidate_method_execution_run") is False
        and gate.get("policy_refit_run") is False
        and gate.get("g5_one_pass_evaluation_authorized") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "authorization",
        "config",
        "completion_evidence",
        "candidate_comparison",
        "exact_policy_leaves",
        "minimal_download",
        "supplemental_registry",
        "freeze_amendment",
        "implementation",
        "workflow",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h1_gate_binds_supplemental_freeze", gate_valid, gate_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "pass" if checks_pass else "blocked",
        "path": evidence_relative,
        "checks": checks,
    }


def _validate_g5_h2_minimal_download(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h2_gse307620_minimal_download/completion_evidence.json"
    receipt_relative = "data/manifests/v1_g5_h2_gse307620_download_receipt.tsv"
    gate_relative = "governance/gates/G5_H2_GSE307620_MINIMAL_DOWNLOAD_2026-07-17.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    check(
        "g5_h2_transfer_identity",
        evidence.get("status") == "pass"
        and evidence.get("milestone") == "H2_GSE307620_minimal_fragment_download"
        and evidence.get("downloaded_files") == 1
        and evidence.get("expected_bytes") == evidence.get("observed_bytes") == 2219237716
        and evidence.get("fragment_sha256") == "33a4ec5bdc3e9955206bdd37e37773ac56737d833b5f9da298ffdc0906f4c45c"
        and evidence.get("gzip_integrity") is True
        and evidence.get("uncompressed_bytes_integrity_scan") == 10543449615,
        (evidence.get("observed_bytes"), evidence.get("fragment_sha256"), evidence.get("gzip_integrity")),
    )
    for field in (
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
        "next_milestone_authorized",
    ):
        check(f"g5_h2_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key in ("receipt", "authorization", "config"):
        artifact = evidence.get(key, {})
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h2_evidence_{key}_hash",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    receipt_path = root / receipt_relative
    receipt_rows: list[dict[str, str]] = []
    if receipt_path.is_file():
        with receipt_path.open(encoding="utf-8", newline="") as handle:
            receipt_rows = list(csv.DictReader(handle, delimiter="\t"))
    receipt = receipt_rows[0] if len(receipt_rows) == 1 else {}
    check(
        "g5_h2_receipt_exact",
        receipt.get("accession") == "GSE307620"
        and receipt.get("sample_accession") == "GSM9228169"
        and receipt.get("file_name") == "GSM9228169_MBR6merge.fragments.tsv.gz"
        and receipt.get("expected_size_bytes") == receipt.get("observed_size_bytes") == "2219237716"
        and receipt.get("size_status") == receipt.get("gzip_status") == receipt.get("verification_status") == "pass"
        and receipt.get("sha256") == evidence.get("fragment_sha256")
        and receipt.get("license_note") == "public_GEO_analysis_only_no_source_redistribution",
        receipt,
    )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("milestone") == "H2_GSE307620_minimal_fragment_download"
        and gate.get("selected_accession") == "GSE307620"
        and gate.get("selected_sample_accession") == "GSM9228169"
        and gate.get("downloaded_files") == 1
        and gate.get("expected_bytes") == gate.get("observed_bytes") == 2219237716
        and gate.get("fragment_sha256") == evidence.get("fragment_sha256")
        and gate.get("gzip_integrity") is True
        and gate.get("existing_holdout_accessed") is False
        and gate.get("fragment_content_analysis_run") is False
        and gate.get("representation_construction_run") is False
        and gate.get("method_execution_run") is False
        and gate.get("policy_refit_run") is False
        and gate.get("g5_one_pass_evaluation_authorized") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "previous_gate",
        "authorization",
        "config",
        "completion_evidence",
        "download_receipt",
        "implementation",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h2_gate_binds_verified_transfer", gate_valid, gate_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "pass" if checks_pass else "blocked",
        "path": evidence_relative,
        "checks": checks,
    }


def _validate_g5_h3_schema_preflight(root: Path) -> dict[str, Any]:
    initial_relative = "results/v1/g5/h3_gse307620_accessibility_schema_preflight/completion_evidence.json"
    evidence_relative = "results/v1/g5/h3a_gse307620_accessibility_schema_preflight/completion_evidence.json"
    summary_relative = "results/v1/g5/h3a_gse307620_accessibility_schema_preflight/fragment_schema_summary.json"
    gate_relative = "governance/gates/G5_H3_GSE307620_ACCESSIBILITY_SCHEMA_PREFLIGHT_2026-07-17.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    initial_path = root / initial_relative
    initial = _load_json(initial_path) if initial_path.is_file() else {}
    check(
        "g5_h3_initial_block_retained",
        initial.get("status") == "blocked"
        and initial.get("fragment_rows") == 234955970
        and initial.get("in_tissue_barcode_overlap_fraction") == 0.0
        and initial.get("representation_construction_run") is False
        and initial.get("method_execution_run") is False,
        (initial.get("status"), initial.get("fragment_rows"), initial.get("in_tissue_barcode_overlap_fraction")),
    )

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    source_checks = evidence.get("checks", {})
    check(
        "g5_h3_amended_schema_pass",
        evidence.get("status") == "pass"
        and evidence.get("scientific_assessment") == "standard_mm10_spatial_fragment_contract_verified"
        and isinstance(source_checks, dict)
        and len(source_checks) == 15
        and all(value is True for value in source_checks.values()),
        (evidence.get("status"), len(source_checks) if isinstance(source_checks, dict) else "invalid"),
    )
    check(
        "g5_h3_dimensions_exact",
        evidence.get("fragment_rows") == 234955918
        and evidence.get("weighted_fragment_count") == 455931745
        and evidence.get("positions_rows") == 2500
        and evidence.get("in_tissue_rows") == 1966
        and evidence.get("in_tissue_barcodes_with_fragments") == 1966
        and evidence.get("in_tissue_barcode_overlap_fraction") == 1.0
        and evidence.get("primary_chromosome_rows") == 234807891
        and evidence.get("unexpected_chromosome_rows") == 148027,
        (
            evidence.get("fragment_rows"),
            evidence.get("weighted_fragment_count"),
            evidence.get("in_tissue_barcodes_with_fragments"),
        ),
    )
    for field in (
        "representation_construction_run",
        "peak_calling_run",
        "fixed_bin_construction_run",
        "gene_activity_construction_run",
        "method_execution_run",
        "rna_access_run",
        "label_or_pathology_access_run",
        "policy_leaf_outcome_access_run",
        "regret_evaluation_run",
        "policy_refit_run",
        "original_holdout_accessed",
        "full_benchmark_run",
        "results_writing_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"g5_h3_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key, artifact in evidence.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h3_evidence_{key}_hash",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )
    for key in ("authorization", "previous_gate", "config"):
        artifact = evidence.get(key, {})
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h3_evidence_{key}_hash",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    summary_path = root / summary_relative
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    check(
        "g5_h3_summary_identity",
        summary.get("status") == "pass"
        and summary.get("source_lines") == 234955970
        and summary.get("header_rows") == 52
        and summary.get("invalid_column_rows") == 0
        and summary.get("invalid_coordinate_rows") == 0
        and summary.get("invalid_duplicate_count_rows") == 0
        and summary.get("coordinate_order_violations") == 0
        and summary.get("barcode_suffix_mismatch_rows") == 0
        and summary.get("fragment_barcodes_not_in_positions") == 0
        and summary.get("position_barcodes_with_fragments") == 2500
        and summary.get("undeclared_chromosome_rows") == 0,
        (summary.get("source_lines"), summary.get("header_rows"), summary.get("position_barcodes_with_fragments")),
    )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("initial_preflight_status") == "blocked_retained"
        and gate.get("amended_preflight_status") == "pass"
        and gate.get("fragment_rows") == 234955918
        and gate.get("positions_rows") == gate.get("position_barcodes_with_fragments") == 2500
        and gate.get("in_tissue_rows") == gate.get("in_tissue_barcodes_with_fragments") == 1966
        and gate.get("schema_checks_passed") == gate.get("schema_checks_total") == 15
        and gate.get("representation_construction_run") is False
        and gate.get("method_execution_run") is False
        and gate.get("policy_refit_run") is False
        and gate.get("one_pass_g5_execution_authorized") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "previous_gate",
        "initial_authorization",
        "initial_config",
        "initial_blocked_evidence",
        "interpretation_amendment",
        "rerun_authorization",
        "rerun_config",
        "completion_evidence",
        "fragment_schema_summary",
        "chromosome_inventory",
        "spatial_barcode_overlap",
        "implementation",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h3_gate_binds_schema_and_retained_amendment", gate_valid, gate_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "pass" if checks_pass else "blocked",
        "path": evidence_relative,
        "checks": checks,
    }


def _validate_g5_h4_protocol_freeze(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h4_one_pass_protocol_freeze/completion_evidence.json"
    gate_relative = "governance/gates/G5_H4_ONE_PASS_EXTERNAL_VALIDATION_PROTOCOL_FREEZE_2026-07-17.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    check(
        "g5_h4_manifest_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("studies") == 2
        and evidence.get("active_policy_leaves") == 10
        and evidence.get("study_leaf_units") == 15
        and evidence.get("portability_study_leaf_units") == 10
        and evidence.get("paired_rna_study_leaf_units") == 5
        and evidence.get("admissible_evaluation_rows") == 69
        and evidence.get("planned_method_and_control_outputs") == 50
        and evidence.get("planned_outputs_authorized") == 0
        and evidence.get("comparator_rows") == 15,
        (
            evidence.get("study_leaf_units"),
            evidence.get("admissible_evaluation_rows"),
            evidence.get("planned_method_and_control_outputs"),
        ),
    )
    check(
        "g5_h4_frozen_statistics",
        evidence.get("random_comparator_draws") == 10000
        and evidence.get("random_seed") == 41017
        and evidence.get("regret_noninferiority_margin") == 0.05
        and evidence.get("failure_rate_noninferiority_margin") == 0.10
        and evidence.get("cross_task_pooling") is False
        and evidence.get("cross_track_pooling") is False
        and evidence.get("cross_representation_pooling") is False
        and evidence.get("two_study_external_portability_possible") is True
        and evidence.get("two_study_external_paired_rna_possible") is False,
        (evidence.get("random_comparator_draws"), evidence.get("random_seed")),
    )
    for field in (
        "policy_refit_allowed",
        "raw_holdout_accessed",
        "representation_construction_run",
        "method_execution_run",
        "rna_value_access_run",
        "holdout_outcome_accessed",
        "regret_computation_run",
        "full_benchmark_run",
        "results_writing_run",
        "universal_ranking_run",
        "final_ranking_run",
        "g5_execution_authorized",
    ):
        check(f"g5_h4_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key, artifact in evidence.get("artifacts", {}).items():
        path = root / str(artifact.get("path", ""))
        rows: list[dict[str, str]] = []
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        check(
            f"g5_h4_{key}_bound",
            path.is_file() and sha256_path(path) == artifact.get("sha256") and len(rows) == artifact.get("rows"),
            (artifact.get("path"), len(rows)),
        )
        if key == "execution_manifest":
            check(
                "g5_h4_execution_manifest_disabled",
                len(rows) == 50
                and all(row.get("attempts_authorized") == "0" for row in rows)
                and all(row.get("execution_status") == "not_authorized" for row in rows),
                len(rows),
            )
    for key in ("authorization", "previous_gate", "frozen_policy", "config"):
        artifact = evidence.get(key, {})
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h4_evidence_{key}_hash",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("study_leaf_units") == 15
        and gate.get("planned_method_and_control_outputs") == 50
        and gate.get("planned_outputs_authorized") == 0
        and gate.get("two_study_external_portability_possible") is True
        and gate.get("two_study_external_paired_rna_possible") is False
        and gate.get("policy_refit_allowed") is False
        and gate.get("raw_holdout_accessed") is False
        and gate.get("method_execution_run") is False
        and gate.get("g5_execution_authorized") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "authorization",
        "previous_gate",
        "config",
        "completion_evidence",
        "leaf_manifest",
        "evaluation_manifest",
        "execution_manifest",
        "comparator_manifest",
        "frozen_protocol",
        "implementation",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h4_gate_binds_disabled_one_pass_protocol", gate_valid, gate_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {
        "status": "pass" if checks_pass else "blocked",
        "path": evidence_relative,
        "checks": checks,
    }


def _validate_g5_h5a_external_representations(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h5a_external_representations/evidence/completion_evidence.json"
    sample_relative = "results/v1/g5/h5a_external_representations/evidence/sample_summary.tsv"
    representation_relative = "results/v1/g5/h5a_external_representations/evidence/representation_summary.tsv"
    gate_relative = "governance/gates/G5_H5A_EXTERNAL_REPRESENTATION_CONSTRUCTION_2026-07-18.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    check(
        "g5_h5a_completion_boundary",
        evidence.get("status") == "pass"
        and evidence.get("studies") == 2
        and evidence.get("representations") == 8
        and evidence.get("holdout_inputs_accessed") is True
        and evidence.get("raw_holdout_checksum_revalidation_run") is True
        and evidence.get("representation_construction_run") is True
        and evidence.get("method_outputs_authorized") == 0,
        (evidence.get("status"), evidence.get("studies"), evidence.get("representations")),
    )
    for field in (
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
    ):
        check(f"g5_h5a_{field}_false", evidence.get(field) is False, evidence.get(field))

    sample_path = root / sample_relative
    sample_rows: list[dict[str, str]] = []
    if sample_path.is_file():
        with sample_path.open(encoding="utf-8", newline="") as handle:
            sample_rows = list(csv.DictReader(handle, delimiter="\t"))
    samples = {row.get("dataset_id"): row for row in sample_rows}
    check(
        "g5_h5a_sample_contracts",
        len(sample_rows) == 2
        and samples.get("gse308623_p2s3", {}).get("n_observations") == "7122"
        and samples.get("gse308623_p2s3", {}).get("included_fragment_rows") == "16687952"
        and samples.get("gse308623_p2s3", {}).get("frozen_peak_rows") == "34199"
        and samples.get("gse307620_mbr6", {}).get("n_observations") == "1966"
        and samples.get("gse307620_mbr6", {}).get("included_fragment_rows") == "216587772"
        and samples.get("gse307620_mbr6", {}).get("frozen_peak_rows") == "148516"
        and all(row.get("filter_partition_conserved") == "True" for row in sample_rows)
        and all(row.get("fixed_bin_counts_conserved") == "True" for row in sample_rows)
        and all(row.get("representation_tracks_exact") == "True" for row in sample_rows)
        and all(row.get("status") == "pass" for row in sample_rows),
        [(row.get("dataset_id"), row.get("n_observations"), row.get("status")) for row in sample_rows],
    )

    representation_path = root / representation_relative
    representation_rows: list[dict[str, str]] = []
    if representation_path.is_file():
        with representation_path.open(encoding="utf-8", newline="") as handle:
            representation_rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_pairs = {
        (dataset_id, representation_id)
        for dataset_id in ("gse308623_p2s3", "gse307620_mbr6")
        for representation_id in ("fixed_bin_10kb", "fixed_bin_25kb", "fixed_bin_50kb", "macs3_summit_501bp")
    }
    observed_pairs = {(row.get("dataset_id"), row.get("representation_id")) for row in representation_rows}
    outputs_bound = True
    for row in representation_rows:
        output = root / str(row.get("output_path", ""))
        outputs_bound = outputs_bound and output.is_file() and sha256_path(output) == row.get("output_sha256")
    check(
        "g5_h5a_representation_contracts",
        len(representation_rows) == 8
        and observed_pairs == expected_pairs
        and all(row.get("status") == "pass" for row in representation_rows)
        and all(row.get("observation_order_matched") == "True" for row in representation_rows)
        and all(row.get("construction_leakage_inputs") == "none" for row in representation_rows)
        and outputs_bound,
        (len(representation_rows), len(observed_pairs), outputs_bound),
    )
    for key, relative in (("sample_summary", sample_relative), ("representation_summary", representation_relative)):
        artifact = evidence.get(key, {})
        artifact_path = root / relative
        check(
            f"g5_h5a_{key}_bound",
            artifact.get("path") == relative
            and artifact_path.is_file()
            and sha256_path(artifact_path) == artifact.get("sha256"),
            artifact,
        )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("studies") == 2
        and gate.get("representations") == 8
        and gate.get("holdout_inputs_accessed") is True
        and gate.get("representation_construction_run") is True
        and gate.get("method_execution_run") is False
        and gate.get("rna_value_access_run") is False
        and gate.get("holdout_utility_access_run") is False
        and gate.get("regret_computation_run") is False
        and gate.get("policy_refit_run") is False
        and gate.get("GSE279771_accessed") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "authorization",
        "previous_gate",
        "config",
        "completion_evidence",
        "sample_summary",
        "representation_summary",
        "implementation",
        "workflow",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h5a_gate_binds_construction_only", gate_valid, gate_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g5_h5b_method_preflight(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h5b_method_construction_preflight/completion_evidence.json"
    gate_relative = "governance/gates/G5_H5B_METHOD_CONSTRUCTION_PREFLIGHT_2026-07-18.json"
    authorization_relative = (
        "governance/authorizations/G5_H5B_EXTERNAL_METHOD_CONSTRUCTION_AUTHORIZATION_2026-07-18.json"
    )
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    check(
        "g5_h5b_preflight_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("planned_outputs") == 50
        and evidence.get("planned_annotation_outputs") == 44
        and evidence.get("planned_software_outputs") == 6
        and evidence.get("planned_outputs_authorized") == 0
        and evidence.get("studies") == 2
        and evidence.get("resources_ready") is True
        and evidence.get("runtime_images_ready") is True,
        (
            evidence.get("planned_outputs"),
            evidence.get("planned_annotation_outputs"),
            evidence.get("planned_software_outputs"),
        ),
    )
    for field in (
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
    ):
        check(f"g5_h5b_preflight_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key in (
        "planned_execution_manifest",
        "implementation_manifest",
        "runtime_image_manifest",
        "config",
        "previous_gate",
        "frozen_execution_manifest",
        "representation_summary",
    ):
        artifact = evidence.get(key, {})
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h5b_preflight_{key}_bound",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )
    plan_path = root / str(evidence.get("planned_execution_manifest", {}).get("path", ""))
    plan_rows: list[dict[str, str]] = []
    if plan_path.is_file():
        with plan_path.open(encoding="utf-8", newline="") as handle:
            plan_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "g5_h5b_preflight_plan_disabled",
        len(plan_rows) == 50
        and len({row.get("output_dir") for row in plan_rows}) == 50
        and sum(row.get("run_type") == "annotation" for row in plan_rows) == 44
        and sum(row.get("run_type") == "software" for row in plan_rows) == 6
        and all(row.get("attempts_authorized") == "0" for row in plan_rows)
        and all(row.get("execution_status") == "not_authorized" for row in plan_rows),
        len(plan_rows),
    )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("planned_outputs") == 50
        and gate.get("planned_outputs_authorized") == 0
        and gate.get("runtime_images_ready") is True
        and gate.get("resources_ready") is True
        and gate.get("method_execution_run") is False
        and gate.get("rna_value_access_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "previous_gate",
        "config",
        "completion_evidence",
        "planned_execution_manifest",
        "implementation_manifest",
        "runtime_image_manifest",
        "frozen_execution_manifest",
        "implementation",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h5b_preflight_gate_bound", gate_valid, gate_relative)

    authorization_path = root / authorization_relative
    authorization = _load_json(authorization_path) if authorization_path.is_file() else {}
    authorization_valid = (
        authorization.get("status") == "authorized"
        and authorization.get("milestone") == "H5B_external_method_construction"
        and authorization.get("authorized_output_manifest_rows") == 50
        and authorization.get("primary_attempts_authorized_per_row") == 1
        and authorization.get("method_execution_authorized") is True
        and authorization.get("evaluation_tasks_authorized") is False
        and authorization.get("rna_value_access_authorized") is False
        and authorization.get("label_or_pathology_value_access_authorized") is False
        and authorization.get("holdout_utility_access_authorized") is False
        and authorization.get("regret_computation_authorized") is False
        and authorization.get("policy_refit_authorized") is False
        and authorization.get("full_benchmark_authorized") is False
        and authorization.get("results_writing_authorized") is False
        and authorization.get("universal_ranking_authorized") is False
        and authorization.get("final_ranking_authorized") is False
        and authorization.get("outcome_based_retry_authorized") is False
        and authorization.get("post_attempt_code_or_environment_change_authorized") is False
    )
    for key in (
        "preflight_gate",
        "planned_execution_manifest",
        "implementation_manifest",
        "runtime_image_manifest",
    ):
        artifact = authorization.get(key, {})
        path = root / str(artifact.get("path", ""))
        authorization_valid = (
            authorization_valid
            and path.is_file()
            and sha256_path(path) == artifact.get("sha256")
        )
    check("g5_h5b_construction_authorization_bound", authorization_valid, authorization_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g5_h5b_external_method_construction(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h5b_external_method_construction/evidence/completion_evidence.json"
    attempt_relative = "results/v1/g5/h5b_external_method_construction/evidence/attempt_manifest.tsv"
    summary_relative = "results/v1/g5/h5b_external_method_construction/evidence/run_summary.tsv"
    gate_relative = "governance/gates/G5_H5B_EXTERNAL_METHOD_CONSTRUCTION_2026-07-18.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    check(
        "g5_h5b_completion_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("studies") == 2
        and evidence.get("runs_expected") == 50
        and evidence.get("runs_passed") == 50
        and evidence.get("runs_failed") == 0
        and evidence.get("runs_missing") == 0
        and evidence.get("annotation_runs") == 44
        and evidence.get("software_runs") == 6
        and evidence.get("contract_checks_passed") == 900
        and evidence.get("contract_checks_total") == 900
        and evidence.get("primary_attempts_executed") == 50
        and evidence.get("retry_attempts_executed") == 0
        and evidence.get("retained_retry_directories") == 0,
        (
            evidence.get("runs_passed"),
            evidence.get("annotation_runs"),
            evidence.get("software_runs"),
            evidence.get("contract_checks_passed"),
        ),
    )
    check(
        "g5_h5b_completion_construction_boundary",
        evidence.get("holdout_inputs_accessed") is True
        and evidence.get("method_execution_run") is True
        and evidence.get("construction_contract_qc_run") is True,
        (
            evidence.get("holdout_inputs_accessed"),
            evidence.get("method_execution_run"),
            evidence.get("construction_contract_qc_run"),
        ),
    )
    for field in (
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
        "performance_outcomes_interpreted",
        "resource_measurements_comparative",
        "next_milestone_authorized",
    ):
        check(f"g5_h5b_completion_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key, relative in (("attempt_manifest", attempt_relative), ("run_summary", summary_relative)):
        artifact = evidence.get(key, {})
        path = root / relative
        check(
            f"g5_h5b_completion_{key}_bound",
            artifact.get("path") == relative
            and path.is_file()
            and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )

    attempt_rows: list[dict[str, str]] = []
    attempt_path = root / attempt_relative
    if attempt_path.is_file():
        with attempt_path.open(encoding="utf-8", newline="") as handle:
            attempt_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "g5_h5b_exact_primary_attempt_contract",
        len(attempt_rows) == 50
        and len({row.get("output_dir") for row in attempt_rows}) == 50
        and all(row.get("attempt_index") == "1" for row in attempt_rows)
        and all(row.get("attempt_type") == "primary" for row in attempt_rows)
        and all(row.get("attempt_executed") == "True" for row in attempt_rows)
        and all(row.get("retry_attempts") == "0" for row in attempt_rows)
        and all(row.get("outcome_based_retry") == "False" for row in attempt_rows)
        and all(row.get("status") == "pass" for row in attempt_rows),
        len(attempt_rows),
    )

    summary_rows: list[dict[str, str]] = []
    summary_path = root / summary_relative
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        "g5_h5b_live_output_contracts",
        len(summary_rows) == 50
        and sum(row.get("run_type") == "annotation" for row in summary_rows) == 44
        and sum(row.get("run_type") == "software" for row in summary_rows) == 6
        and all(row.get("contract_checks_passed") == "18" for row in summary_rows)
        and all(row.get("contract_checks_total") == "18" for row in summary_rows)
        and all(row.get("observation_ids_and_order_match") == "True" for row in summary_rows)
        and all(row.get("manifest_artifact_hashes_valid") == "True" for row in summary_rows)
        and all(row.get("live_input_hashes_valid") == "True" for row in summary_rows)
        and all(row.get("construction_boundary_valid") == "True" for row in summary_rows)
        and all(row.get("construction_leakage_absent") == "True" for row in summary_rows)
        and all(row.get("holdout_accessed") == "True" for row in summary_rows)
        and all(row.get("process_tree_exit_code") == "0" for row in summary_rows)
        and all(row.get("status") == "pass" for row in summary_rows),
        len(summary_rows),
    )

    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("runs_expected") == 50
        and gate.get("runs_passed") == 50
        and gate.get("contract_checks_passed") == 900
        and gate.get("contract_checks_total") == 900
        and gate.get("primary_attempts_executed") == 50
        and gate.get("retry_attempts_executed") == 0
        and gate.get("holdout_inputs_accessed") is True
        and gate.get("method_execution_run") is True
        and gate.get("construction_contract_qc_run") is True
        and gate.get("rna_value_access_run") is False
        and gate.get("holdout_utility_access_run") is False
        and gate.get("regret_computation_run") is False
        and gate.get("policy_refit_run") is False
        and gate.get("GSE279771_accessed") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "authorization",
        "preflight_gate",
        "planned_execution_manifest",
        "annotation_config",
        "software_config",
        "completion_evidence",
        "attempt_manifest",
        "run_summary",
        "implementation",
        "workflow",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h5b_completion_gate_bound", gate_valid, gate_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g5_h5c_external_evaluation_preflight(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h5c_external_evaluation_preflight/completion_evidence.json"
    gate_relative = "governance/gates/G5_H5C_EXTERNAL_EVALUATION_PREFLIGHT_2026-07-18.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    check(
        "g5_h5c_preflight_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("study_leaf_units") == 15
        and evidence.get("admissible_evaluation_rows") == 69
        and evidence.get("comparator_rows") == 15
        and evidence.get("frozen_method_outputs") == 50
        and evidence.get("rna_method_and_control_outputs") == 27
        and evidence.get("rna_observations") == 7122
        and evidence.get("rna_gene_rows") == 32285
        and evidence.get("mapped_gtf_gene_names") == 25912
        and evidence.get("mapped_protein_coding_gene_names") == 20465,
        (
            evidence.get("frozen_method_outputs"),
            evidence.get("rna_observations"),
            evidence.get("mapped_protein_coding_gene_names"),
        ),
    )
    check(
        "g5_h5c_preflight_deviation_disclosed",
        evidence.get("limited_preauthorization_access_deviation_disclosed") is True
        and evidence.get("rna_values_parsed_by_preflight") is False,
        evidence.get("limited_preauthorization_access_deviation_disclosed"),
    )
    for field in (
        "evaluation_run",
        "rna_metric_computation_run",
        "regret_computation_run",
        "policy_refit_run",
        "method_rerun",
        "GSE279771_accessed",
        "full_benchmark_run",
        "results_writing_run",
        "universal_ranking_run",
        "final_ranking_run",
        "next_milestone_authorized",
    ):
        check(f"g5_h5c_preflight_{field}_false", evidence.get(field) is False, evidence.get(field))
    for key in (
        "method_input_manifest",
        "rna_observation_contract",
        "rna_gene_identity_preflight",
        "implementation_manifest",
        "evaluation_manifest",
        "comparator_manifest",
        "rna_matrix",
        "gene_annotation",
        "deviation_amendment",
        "implementation",
    ):
        artifact = evidence.get(key, {})
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h5c_preflight_{key}_bound",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )
    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("admissible_evaluation_rows") == 69
        and gate.get("frozen_method_outputs") == 50
        and gate.get("rna_method_and_control_outputs") == 27
        and gate.get("rna_values_parsed_by_preflight") is False
        and gate.get("limited_preauthorization_access_deviation_disclosed") is True
        and gate.get("evaluation_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "previous_gate",
        "h4_gate",
        "config",
        "completion_evidence",
        "method_input_manifest",
        "rna_observation_contract",
        "rna_gene_identity_preflight",
        "implementation_manifest",
        "implementation",
        "evaluator",
        "deviation_amendment",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h5c_preflight_gate_bound", gate_valid, gate_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def _validate_g5_h5c_external_evaluation(root: Path) -> dict[str, Any]:
    evidence_relative = "results/v1/g5/h5c_external_evaluation/evidence/completion_audit.json"
    gate_relative = "governance/gates/G5_H5C_EXTERNAL_EVALUATION_2026-07-18.json"
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": str(detail)})

    evidence_path = root / evidence_relative
    evidence = _load_json(evidence_path) if evidence_path.is_file() else {}
    check(
        "g5_h5c_completion_dimensions",
        evidence.get("status") == "pass"
        and evidence.get("checks_passed") == evidence.get("checks_total") == 13
        and evidence.get("rna_methods_and_controls") == 27
        and evidence.get("rna_evaluation_failures") == 0
        and evidence.get("rna_eligible_genes") == 6074
        and evidence.get("admissible_evaluation_rows") == 69
        and evidence.get("study_leaf_units") == 15
        and evidence.get("random_comparator_rows") == 150000
        and evidence.get("portability_rows_passed") == evidence.get("portability_rows_total") == 46
        and evidence.get("zero_policy_regret_leaves") == 15
        and evidence.get("matched_control_positive_rows") == evidence.get("matched_control_rows") == 20,
        (
            evidence.get("checks_passed"),
            evidence.get("portability_rows_passed"),
            evidence.get("zero_policy_regret_leaves"),
        ),
    )
    check(
        "g5_h5c_claim_boundary",
        evidence.get("limited_preauthorization_access_deviation_disclosed") is True
        and evidence.get("two_study_external_rna_claim_allowed") is False
        and evidence.get("universal_ranking_run") is False
        and evidence.get("final_ranking_run") is False,
        "single-study RNA scope retained",
    )
    for key in (
        "completion_evidence",
        "process_tree_memory",
        "validation_checks",
        "claim_scope_summary",
        "implementation",
    ):
        artifact = evidence.get(key, {})
        path = root / str(artifact.get("path", ""))
        check(
            f"g5_h5c_completion_{key}_bound",
            path.is_file() and sha256_path(path) == artifact.get("sha256"),
            artifact,
        )
    gate_path = root / gate_relative
    gate = _load_json(gate_path) if gate_path.is_file() else {}
    gate_valid = (
        gate.get("status") == "pass"
        and gate.get("completion_checks_passed") == gate.get("completion_checks_total") == 13
        and gate.get("portability_rows_passed") == gate.get("portability_rows_total") == 46
        and gate.get("zero_policy_regret_leaves") == 15
        and gate.get("matched_control_positive_rows") == gate.get("matched_control_rows") == 20
        and gate.get("two_study_external_rna_claim_allowed") is False
        and gate.get("universal_method_or_representation_winner_claim_allowed") is False
        and gate.get("limited_preauthorization_access_deviation_disclosed") is True
        and gate.get("method_rerun") is False
        and gate.get("policy_refit_run") is False
        and gate.get("GSE279771_accessed") is False
        and gate.get("full_benchmark_run") is False
        and gate.get("results_writing_run") is False
        and gate.get("universal_ranking_run") is False
        and gate.get("final_ranking_run") is False
        and gate.get("next_milestone_authorized") is False
    )
    for key in (
        "preflight_gate",
        "authorization",
        "completion_evidence",
        "process_tree_memory",
        "completion_audit",
        "validation_checks",
        "claim_scope_summary",
        "evaluator",
        "auditor",
        "deviation_amendment",
        "report",
    ):
        artifact = gate.get(key, {})
        path = root / str(artifact.get("path", ""))
        gate_valid = gate_valid and path.is_file() and sha256_path(path) == artifact.get("sha256")
    check("g5_h5c_completion_gate_bound", gate_valid, gate_relative)
    checks_pass = bool(checks) and all(row["status"] == "pass" for row in checks)
    return {"status": "pass" if checks_pass else "blocked", "path": evidence_relative, "checks": checks}


def validate_project_config(config_path: str | Path) -> dict[str, Any]:
    """Validate the structural G0 contract without claiming later gates are ready."""

    path = Path(config_path).resolve()
    root = _repo_root(path)
    config = _load_mapping(path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail})

    project = config.get("project", {})
    execution = config.get("execution", {})
    check("project_version_v1", str(project.get("version", "")).startswith("1."), str(project.get("version")))
    valid_gates = {f"G{index}" for index in range(7)}
    check("current_gate_valid", project.get("current_gate") in valid_gates, str(project.get("current_gate")))

    for flag in REQUIRED_DISABLED_FLAGS:
        check(f"{flag}_disabled", execution.get(flag) is False, repr(execution.get(flag)))

    seeds = execution.get("seeds", [])
    check(
        "five_unique_seeds_declared",
        isinstance(seeds, list) and len(seeds) >= 5 and len(set(seeds)) == len(seeds),
        repr(seeds),
    )

    loaded_registries: dict[str, dict[str, Any]] = {}
    for name, relative in config.get("registries", {}).items():
        registry_path = root / str(relative)
        exists = registry_path.is_file()
        check(f"registry_{name}_exists", exists, str(registry_path.relative_to(root)))
        if exists:
            try:
                loaded_registries[name] = _load_mapping(registry_path)
                check(f"registry_{name}_parses", True, "YAML mapping")
            except (OSError, ValueError, yaml.YAMLError) as exc:
                check(f"registry_{name}_parses", False, str(exc))

    for name, relative in config.get("protocols", {}).items():
        protocol_path = root / str(relative)
        exists = protocol_path.is_file() and protocol_path.stat().st_size > 0
        check(f"protocol_{name}_exists", exists, str(protocol_path.relative_to(root)))

    datasets = loaded_registries.get("datasets", {})
    studies = datasets.get("studies", {}) if isinstance(datasets, dict) else {}
    targets = datasets.get("panel_targets", {}) if isinstance(datasets, dict) else {}
    declared = set(studies) if isinstance(studies, dict) else set()
    development = set(config.get("development_studies", []))
    candidates = set(config.get("candidate_studies", []))
    configured_holdouts = set(config.get("holdout_studies", []))
    check("development_studies_registered", development <= declared, repr(sorted(development - declared)))
    check("candidate_studies_registered", candidates <= declared, repr(sorted(candidates - declared)))
    check("holdout_studies_registered", configured_holdouts <= declared, repr(sorted(configured_holdouts - declared)))
    check(
        "development_holdout_disjoint",
        development.isdisjoint(configured_holdouts),
        repr(sorted(development & configured_holdouts)),
    )

    minimum_studies = int(targets.get("minimum_independent_studies", 0) or 0)
    ready_studies = {
        study_id
        for study_id, metadata in studies.items()
        if isinstance(metadata, dict) and metadata.get("inclusion_status") == "included_ready"
    }
    holdout_slots = datasets.get("holdout_slots", {}) if isinstance(datasets, dict) else {}
    frozen_holdouts = {
        slot.get("study_id")
        for slot in holdout_slots.values()
        if isinstance(slot, dict)
        and slot.get("freeze_status") == "frozen"
        and slot.get("benchmark_outcomes_inspected") is False
    }

    def capability_count(capability: str) -> int:
        return sum(
            1
            for study_id in ready_studies
            if isinstance(studies[study_id].get("capabilities"), dict)
            and studies[study_id]["capabilities"].get(capability) is True
        )

    fragment_studies = capability_count("standard_fragment_level")
    human_pathology_studies = capability_count("human_pathology")
    paired_rna_studies = capability_count("verified_paired_spatial_rna")
    minimum_fragments = int(targets.get("minimum_fragment_studies", 0) or 0)
    minimum_pathology = int(targets.get("minimum_human_pathology_studies", 0) or 0)
    minimum_paired_rna = int(targets.get("minimum_verified_paired_rna_studies", 0) or 0)
    minimum_holdouts = int(targets.get("frozen_holdout_studies", 0) or 0)
    holdout_contract_pass = configured_holdouts == frozen_holdouts and len(frozen_holdouts) >= minimum_holdouts
    g1_pass = (
        len(ready_studies) >= minimum_studies
        and minimum_studies > 0
        and fragment_studies >= minimum_fragments
        and human_pathology_studies >= minimum_pathology
        and paired_rna_studies >= minimum_paired_rna
        and holdout_contract_pass
    )
    gate_evidence = config.get("gate_evidence", {})
    g2_evidence = _validate_g2_attestation(
        root,
        gate_evidence.get("G2") if isinstance(gate_evidence, dict) else None,
        development,
        configured_holdouts,
    )
    g3_evidence = _validate_g3_attestation(
        root,
        gate_evidence.get("G3") if isinstance(gate_evidence, dict) else None,
    )
    g4_evidence = _validate_g4_attestation(
        root,
        gate_evidence.get("G4") if isinstance(gate_evidence, dict) else None,
    )
    g4_preflight = _validate_g4_preflight(root)
    g4_m1 = _validate_g4_m1(root)
    g4_m2_memory = _validate_g4_m2_memory_readiness(root)
    g4_m2_annotation = _validate_g4_m2_annotation(root)
    g4_m2_software = _validate_g4_m2_software(root)
    g4_m2c = _validate_g4_m2c(root)
    g4_m3a = _validate_g4_m3a(root)
    g4_m3b_preflight = _validate_g4_m3b_preflight(root)
    g4_m3b_rna = _validate_g4_m3b_rna(root)
    g4_m3c_simulation = _validate_g4_m3c_simulation(root)
    g4_m3d_domain = _validate_g4_m3d_domain(root)
    g4_m3e_replication = _validate_g4_m3e_replication(root)
    g4_m3f_source_audit = _validate_g4_m3f_source_audit(root)
    g4_m3f_he_preflight = _validate_g4_m3f_he_preflight(root)
    g4_m3f_malt_preflight = _validate_g4_m3f_malt_preflight(root)
    g4_m3f_image_features = _validate_g4_m3f_image_features(root)
    g4_m3f_malt_matched = _validate_g4_m3f_malt_matched_representations(root)
    g4_m3f_malt_methods = _validate_g4_m3f_malt_method_construction(root)
    g4_m3f_malt_morphology = _validate_g4_m3f_malt_morphology(root)
    g4_m3f_malt_nonlinear = _validate_g4_m3f_malt_nonlinear_sensitivity(root)
    g4_m3g_pathology_truth = _validate_g4_m3g_pathology_truth_candidate(root)
    g4_m3h_cross_study_rna = _validate_g4_m3h_cross_study_rna(root)
    g4_m4_recommendation_policy = _validate_g4_m4_recommendation_policy(root)
    g5_h0_preaccess = _validate_g5_h0_preaccess_feasibility(root)
    g5_h1_supplemental = _validate_g5_h1_supplemental_holdout(root)
    g5_h2_minimal_download = _validate_g5_h2_minimal_download(root)
    g5_h3_schema_preflight = _validate_g5_h3_schema_preflight(root)
    g5_h4_protocol_freeze = _validate_g5_h4_protocol_freeze(root)
    g5_h5a_external_representations = _validate_g5_h5a_external_representations(root)
    g5_h5b_method_preflight = _validate_g5_h5b_method_preflight(root)
    g5_h5b_external_method_construction = _validate_g5_h5b_external_method_construction(root)
    g5_h5c_external_evaluation_preflight = _validate_g5_h5c_external_evaluation_preflight(root)
    g5_h5c_external_evaluation = _validate_g5_h5c_external_evaluation(root)
    g4_status = g4_evidence["status"]
    if g4_status == "ready_to_execute" and g4_preflight["status"] == "pass":
        g4_status = "preflight_passed"
    if g4_status == "preflight_passed" and g4_m1["status"] == "pass":
        g4_status = "m1_passed"
    if g4_status == "m1_passed" and g4_m2_memory["status"] == "pass":
        g4_status = "m2_ready"
    if g4_status == "m2_ready" and g4_m2_annotation["status"] == "pass":
        g4_status = "m2a_passed"
    if g4_status == "m2a_passed" and g4_m2_software["status"] == "pass":
        g4_status = "m2b_passed"
    if g4_status == "m2b_passed" and g4_m2c["status"] == "pass":
        g4_status = "m2c_passed"
    if g4_status == "m2c_passed" and g4_m3a["status"] == "pass":
        g4_status = "m3a_passed"
    if g4_status == "m3a_passed" and g4_m3b_preflight["status"] == "pass":
        g4_status = "m3b_preflight_passed"
    if g4_status == "m3b_preflight_passed" and g4_m3b_rna["status"] == "pass":
        g4_status = "m3b_rna_passed"
    if g4_status == "m3b_rna_passed" and g4_m3c_simulation["status"] == "pass":
        g4_status = "m3c_simulation_passed"
    if g4_status == "m3c_simulation_passed" and g4_m3d_domain["status"] == "pass":
        g4_status = "m3d_author_domain_passed"
    if g4_status == "m3d_author_domain_passed" and g4_m3e_replication["status"] == "pass":
        g4_status = "m3e_cross_study_author_domain_passed"
    if g4_status == "m3e_cross_study_author_domain_passed" and g4_m3f_source_audit["status"] == "partial":
        g4_status = "m3f_histology_pathology_source_audit_partial"
    if g4_status == "m3f_histology_pathology_source_audit_partial" and g4_m3f_malt_preflight["status"] == "pass":
        g4_status = "m3f_malt_input_preflight_passed_with_source_subset_discrepancy"
    if g4_status == "m3f_malt_input_preflight_passed_with_source_subset_discrepancy" and g4_m3f_image_features["status"] == "pass":
        g4_status = "m3f_malt_image_features_frozen"
    if g4_status == "m3f_malt_image_features_frozen" and g4_m3f_malt_matched["status"] == "pass":
        g4_status = "m3f_malt_matched_representations_passed"
    if g4_status == "m3f_malt_matched_representations_passed" and g4_m3f_malt_methods["status"] == "pass":
        g4_status = "m3f_malt_method_construction_passed"
    if g4_status == "m3f_malt_method_construction_passed" and g4_m3f_malt_morphology["status"] == "pass":
        g4_status = "m3f_malt_morphology_completed_weak_not_mapping_specific"
    if (
        g4_status == "m3f_malt_morphology_completed_weak_not_mapping_specific"
        and g4_m3f_malt_nonlinear["status"] == "pass"
    ):
        g4_status = "m3f_nonlinear_sensitivity_completed_weak_baseline_sensitive_not_mapping_specific"
    if (
        g4_status == "m3f_nonlinear_sensitivity_completed_weak_baseline_sensitive_not_mapping_specific"
        and g4_m3g_pathology_truth["status"] == "partial"
    ):
        g4_status = "m3g_public_replacement_search_complete_pathology_endpoint_deferred"
    if (
        g4_status == "m3g_public_replacement_search_complete_pathology_endpoint_deferred"
        and g4_m3h_cross_study_rna["status"] == "pass"
    ):
        g4_status = "m3h_cross_study_paired_rna_replication_passed_policy_not_frozen"
    if (
        g4_status == "m3h_cross_study_paired_rna_replication_passed_policy_not_frozen"
        and g4_m4_recommendation_policy["status"] == "pass"
    ):
        g4_status = "m4_task_specific_policy_frozen_g5_not_authorized"
    g5_status = g5_h0_preaccess["status"]
    if g5_status == "blocked_second_evaluable_holdout_missing" and g5_h1_supplemental["status"] == "pass":
        g5_status = "supplemental_portability_holdout_frozen_download_pending"
    if g5_status == "supplemental_portability_holdout_frozen_download_pending" and g5_h2_minimal_download["status"] == "pass":
        g5_status = "supplemental_portability_download_verified_schema_preflight_not_authorized"
    if (
        g5_status == "supplemental_portability_download_verified_schema_preflight_not_authorized"
        and g5_h3_schema_preflight["status"] == "pass"
    ):
        g5_status = "supplemental_portability_schema_preflight_passed_one_pass_protocol_not_frozen"
    if (
        g5_status == "supplemental_portability_schema_preflight_passed_one_pass_protocol_not_frozen"
        and g5_h4_protocol_freeze["status"] == "pass"
    ):
        g5_status = "one_pass_protocol_frozen_execution_not_authorized"
    if (
        g5_status == "one_pass_protocol_frozen_execution_not_authorized"
        and g5_h5a_external_representations["status"] == "pass"
    ):
        g5_status = "external_representations_frozen_methods_not_authorized"
    if (
        g5_status == "external_representations_frozen_methods_not_authorized"
        and g5_h5b_method_preflight["status"] == "pass"
    ):
        g5_status = "external_method_construction_authorized_not_started"
    if (
        g5_status == "external_method_construction_authorized_not_started"
        and g5_h5b_external_method_construction["status"] == "pass"
    ):
        g5_status = "external_method_outputs_frozen_rna_evaluation_not_authorized"
    if (
        g5_status == "external_method_outputs_frozen_rna_evaluation_not_authorized"
        and g5_h5c_external_evaluation_preflight["status"] == "pass"
    ):
        g5_status = "external_evaluation_authorized_not_started"
    if (
        g5_status == "external_evaluation_authorized_not_started"
        and g5_h5c_external_evaluation["status"] == "pass"
    ):
        g5_status = "external_evaluation_completed_scope_limited_release_not_ready"
    readiness = {
        "G0": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "G1": "pass" if g1_pass else "blocked",
        "G2": g2_evidence["status"],
        "G3": g3_evidence["status"],
        "G4": g4_status,
        "G5": g5_status,
        "G6": "blocked",
    }
    return {
        "project": project.get("name", "SpatialGABench"),
        "version": project.get("version", "unknown"),
        "config": str(path.relative_to(root)),
        "validation_status": readiness["G0"],
        "checks": checks,
        "readiness": readiness,
        "dataset_summary": {
            "registered_independent_studies": len(declared),
            "included_ready_studies": len(ready_studies),
            "minimum_independent_studies": minimum_studies,
            "standard_fragment_studies": fragment_studies,
            "minimum_fragment_studies": minimum_fragments,
            "human_pathology_studies": human_pathology_studies,
            "minimum_human_pathology_studies": minimum_pathology,
            "verified_paired_rna_studies": paired_rna_studies,
            "minimum_verified_paired_rna_studies": minimum_paired_rna,
            "frozen_holdout_studies": sorted(frozen_holdouts),
            "minimum_frozen_holdout_studies": minimum_holdouts,
            "development_studies": sorted(development),
            "candidate_studies": sorted(candidates),
            "configured_holdout_studies": sorted(configured_holdouts),
        },
        "gate_evidence": {
            "G2": g2_evidence,
            "G3": g3_evidence,
            "G4": g4_evidence,
            "G4-M0": g4_preflight,
            "G4-M1": g4_m1,
            "G4-M2-memory": g4_m2_memory,
            "G4-M2A-annotation": g4_m2_annotation,
            "G4-M2B-software": g4_m2_software,
            "G4-M2C-controls-scalability": g4_m2c,
            "G4-M3A-admissibility": g4_m3a,
            "G4-M3B-input-preflight": g4_m3b_preflight,
            "G4-M3B-RNA": g4_m3b_rna,
            "G4-M3C-simulation": g4_m3c_simulation,
            "G4-M3D-author-domain": g4_m3d_domain,
            "G4-M3E-cross-study-author-domain": g4_m3e_replication,
            "G4-M3F-histology-pathology-source-audit": g4_m3f_source_audit,
            "G4-M3F-GSE278007-melanoma-HE-preflight": g4_m3f_he_preflight,
            "G4-M3F-GSE288178-MALT-input-preflight": g4_m3f_malt_preflight,
            "G4-M3F-GSE288178-MALT-image-features": g4_m3f_image_features,
            "G4-M3F-GSE288178-MALT-matched-representations": g4_m3f_malt_matched,
            "G4-M3F-GSE288178-MALT-method-construction": g4_m3f_malt_methods,
            "G4-M3F-GSE288178-MALT-morphology-concordance": g4_m3f_malt_morphology,
            "G4-M3F-GSE288178-MALT-nonlinear-spatial-sensitivity": g4_m3f_malt_nonlinear,
            "G4-M3G-pathology-truth-candidate-audit": g4_m3g_pathology_truth,
            "G4-M3H-cross-study-paired-RNA": g4_m3h_cross_study_rna,
            "G4-M4-task-specific-recommendation-policy": g4_m4_recommendation_policy,
            "G5-H0-preaccess-feasibility": g5_h0_preaccess,
            "G5-H1-supplemental-holdout-freeze": g5_h1_supplemental,
            "G5-H2-minimal-fragment-download": g5_h2_minimal_download,
            "G5-H3-accessibility-schema-preflight": g5_h3_schema_preflight,
            "G5-H4-one-pass-protocol-freeze": g5_h4_protocol_freeze,
            "G5-H5A-external-representations": g5_h5a_external_representations,
            "G5-H5B-method-construction-preflight": g5_h5b_method_preflight,
            "G5-H5B-external-method-construction": g5_h5b_external_method_construction,
            "G5-H5C-external-evaluation-preflight": g5_h5c_external_evaluation_preflight,
            "G5-H5C-external-evaluation": g5_h5c_external_evaluation,
        },
    }


def write_project_validation(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Validate and write the machine-readable G0 report."""

    payload = validate_project_config(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
