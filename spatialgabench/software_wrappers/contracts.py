"""Leakage-safe input contracts for external software wrappers."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import yaml

from spatialgabench.representations import assert_development_study


SCHEMA_VERSION = 1
G4_M2_AUTHORIZATION = "governance/authorizations/G4_M2_METHOD_CONSTRUCTION_AUTHORIZATION_2026-07-15.json"
G4_M2C_AUTHORIZATION = "governance/authorizations/G4_M2C_CONTROLS_SCALABILITY_AUTHORIZATION_2026-07-15.json"
ALLOWED_ROLES = {"toy", "development", "external_holdout"}
ALLOWED_SUBSET_POLICIES = {"toy_smoke", "deterministic_smoke_subset", "full", "scalability_subset"}
FORBIDDEN_CONFIG_KEYS = {
    "rna",
    "rna_reference",
    "label",
    "labels",
    "cluster",
    "clusters",
    "marker",
    "markers",
    "pathology",
    "truth",
    "true_gene_activity",
}
GENOME_BUILD_ALIASES = {
    "mm10": "mm10",
    "grcm38": "mm10",
    "grcm38_mm10": "mm10",
    "hg38": "hg38",
    "grch38": "hg38",
}
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "run_id",
    "project_config",
    "study_id",
    "dataset_id",
    "sample_id",
    "dataset_role",
    "method_id",
    "software_name",
    "genome_build",
    "input_track",
    "representation_id",
    "comparison_scope",
    "input_h5ad",
    "fragments",
    "gene_annotation",
    "chrom_sizes",
    "environment_lock",
    "adapter",
    "method_script",
    "implementation_files",
    "construction_input_classes",
    "subset_policy",
    "threads",
    "seed",
    "output_dir",
}
OPTIONAL_CONFIG_KEYS = {"execution_authorization", "execution_authorization_sha256"}

METHOD_CONTRACTS: dict[str, dict[str, Any]] = {
    "signac_geneactivity": {
        "software_name": "Signac",
        "runtime": "R",
        "input_track": "F",
        "fragments_required": True,
        "construction_input_classes": {
            "accessibility_fragments",
            "frozen_peak_matrix",
            "gene_annotation",
        },
        "allowed_subset_policies": {"toy_smoke", "deterministic_smoke_subset", "full"},
    },
    "archr_genescorematrix": {
        "software_name": "ArchR",
        "runtime": "R",
        "input_track": "F",
        "fragments_required": True,
        "construction_input_classes": {"accessibility_fragments", "gene_annotation"},
        "allowed_subset_policies": {"toy_smoke", "deterministic_smoke_subset", "full"},
    },
    "snapatac2_geneactivity": {
        "software_name": "SnapATAC2",
        "runtime": "python",
        "input_track": "F",
        "fragments_required": True,
        "construction_input_classes": {"accessibility_fragments", "gene_annotation"},
        "allowed_subset_policies": {"toy_smoke", "deterministic_smoke_subset", "full"},
    },
    "cicero_geneactivity": {
        "software_name": "Cicero",
        "runtime": "R",
        "input_track": "P",
        "fragments_required": False,
        "construction_input_classes": {"peak_matrix", "gene_annotation"},
        "allowed_subset_policies": {"toy_smoke", "scalability_subset"},
    },
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_root(root: Path, value: object) -> tuple[Path, bool]:
    path = root / str(value)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return path, False
    return path, True


def _canonical_build(value: object) -> str:
    return GENOME_BUILD_ALIASES.get(str(value).lower(), "unresolved")


def _gzip_integrity(path: Path, chunk_size: int = 1024 * 1024) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(chunk_size):
                pass
    except (EOFError, OSError):
        return False
    return True


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _find_forbidden_keys(value: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_string = str(key)
            qualified = f"{prefix}.{key_string}" if prefix else key_string
            if key_string.lower() in FORBIDDEN_CONFIG_KEYS:
                found.append(qualified)
            found.extend(_find_forbidden_keys(nested, qualified))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_find_forbidden_keys(nested, f"{prefix}[{index}]"))
    return found


def load_wrapper_config(path: str | Path) -> dict[str, Any]:
    """Load a wrapper YAML config as a mapping."""

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Wrapper config must be a YAML mapping: {path}")
    return payload


def validate_wrapper_input_contract(
    config: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    """Validate method admissibility, leakage exclusions, paths, and input identity."""

    root = Path(root).resolve()
    checks: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}

    def check(check_id: str, passed: bool, detail: object) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "pass" if passed else "fail",
                "detail": str(detail),
            }
        )

    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    unknown = sorted(set(config) - REQUIRED_CONFIG_KEYS - OPTIONAL_CONFIG_KEYS)
    check("required_keys", not missing, missing)
    check("unknown_keys_absent", not unknown, unknown)
    forbidden = _find_forbidden_keys(config)
    check("forbidden_construction_keys_absent", not forbidden, forbidden)
    check("schema_version", config.get("schema_version") == SCHEMA_VERSION, config.get("schema_version"))

    method_id = str(config.get("method_id", ""))
    method_contract = METHOD_CONTRACTS.get(method_id)
    check("method_registered", method_contract is not None, method_id)
    if method_contract is not None:
        check(
            "software_name_matches",
            config.get("software_name") == method_contract["software_name"],
            config.get("software_name"),
        )
        check(
            "input_track_matches",
            config.get("input_track") == method_contract["input_track"],
            config.get("input_track"),
        )
        observed_classes = config.get("construction_input_classes", [])
        observed_class_set = set(observed_classes) if isinstance(observed_classes, list) else set()
        check(
            "construction_input_budget_matches",
            observed_class_set == method_contract["construction_input_classes"],
            sorted(observed_class_set),
        )
        check(
            "subset_policy_admissible",
            config.get("subset_policy") in method_contract["allowed_subset_policies"],
            config.get("subset_policy"),
        )

    role = config.get("dataset_role")
    check("dataset_role_admissible", role in ALLOWED_ROLES, role)
    authorization: dict[str, object] | None = None
    project_config_path, project_config_inside = _inside_root(root, config.get("project_config", ""))
    check("project_config_inside_root", project_config_inside, project_config_path)
    check("project_config_exists", project_config_path.is_file(), project_config_path)
    if role == "development" and project_config_inside and project_config_path.is_file():
        try:
            authorization = assert_development_study(project_config_path, str(config.get("study_id", "")))
            check("development_authorized", True, config.get("study_id"))
        except (PermissionError, ValueError, OSError, yaml.YAMLError) as exc:
            check("development_authorized", False, exc)
    elif role == "toy":
        check("toy_scope_declared", str(config.get("study_id", "")).startswith("toy_"), config.get("study_id"))
    elif role == "external_holdout":
        authorization_path, authorization_inside = _inside_root(
            root, config.get("execution_authorization", "")
        )
        check("external_authorization_inside_root", authorization_inside, authorization_path)
        check("external_authorization_exists", authorization_path.is_file(), authorization_path)
        observed_authorization_sha256 = _sha256(authorization_path) if authorization_path.is_file() else ""
        check(
            "external_authorization_hash",
            observed_authorization_sha256 == config.get("execution_authorization_sha256"),
            observed_authorization_sha256,
        )
        external: dict[str, Any] = {}
        if authorization_path.is_file():
            try:
                loaded = json.loads(authorization_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    external = loaded
            except (OSError, json.JSONDecodeError):
                pass
        check(
            "external_construction_authorized",
            external.get("status") == "authorized"
            and external.get("milestone") == "H5B_external_method_construction"
            and external.get("method_execution_authorized") is True
            and external.get("evaluation_tasks_authorized") is False
            and external.get("rna_value_access_authorized") is False
            and external.get("label_or_pathology_value_access_authorized") is False
            and external.get("holdout_utility_access_authorized") is False
            and external.get("regret_computation_authorized") is False
            and external.get("policy_refit_authorized") is False
            and external.get("full_benchmark_authorized") is False
            and external.get("results_writing_authorized") is False
            and external.get("universal_ranking_authorized") is False
            and external.get("final_ranking_authorized") is False,
            external.get("scope"),
        )
        check(
            "external_study_authorized",
            str(config.get("study_id", "")) in set(external.get("authorized_studies", [])),
            config.get("study_id"),
        )
        check(
            "external_method_authorized",
            method_id in set(external.get("authorized_methods", [])),
            method_id,
        )
        authorization = external
        if authorization_inside and authorization_path.is_file():
            checksums["execution_authorization"] = observed_authorization_sha256
    check(
        "role_subset_policy_matches",
        (role == "toy" and config.get("subset_policy") == "toy_smoke")
        or (role in {"development", "external_holdout"} and config.get("subset_policy") != "toy_smoke"),
        f"{role}/{config.get('subset_policy')}",
    )

    canonical_build = _canonical_build(config.get("genome_build"))
    check("genome_build_supported", canonical_build != "unresolved", config.get("genome_build"))

    try:
        threads = int(config.get("threads", 0))
        seed = int(config.get("seed", -1))
    except (TypeError, ValueError):
        threads = 0
        seed = -1
    check("threads_bounded", 1 <= threads <= 64, threads)
    check("seed_nonnegative", seed >= 0, seed)
    check("subset_policy_known", config.get("subset_policy") in ALLOWED_SUBSET_POLICIES, config.get("subset_policy"))

    file_fields = ["input_h5ad", "gene_annotation", "chrom_sizes", "environment_lock", "adapter"]
    fragments_value = config.get("fragments")
    fragments_required = bool(method_contract and method_contract["fragments_required"])
    check(
        "fragments_requirement",
        (fragments_required and isinstance(fragments_value, str) and bool(fragments_value))
        or (not fragments_required and (fragments_value is None or fragments_value == "")),
        fragments_value,
    )
    if fragments_required:
        file_fields.append("fragments")
    method_script_value = config.get("method_script")
    method_script_required = bool(method_contract and method_contract["runtime"] == "R")
    check(
        "method_script_requirement",
        (method_script_required and isinstance(method_script_value, str) and bool(method_script_value))
        or (not method_script_required and (method_script_value is None or method_script_value == "")),
        method_script_value,
    )
    if method_script_required:
        file_fields.append("method_script")

    resolved_paths: dict[str, str] = {}
    if role == "external_holdout":
        external_authorization_path, external_authorization_inside = _inside_root(
            root, config.get("execution_authorization", "")
        )
        if external_authorization_inside:
            resolved_paths["execution_authorization"] = str(external_authorization_path)
    for field in file_fields:
        path, inside = _inside_root(root, config.get(field, ""))
        exists = path.is_file()
        check(f"{field}_inside_root", inside, path)
        check(f"{field}_exists", exists, path)
        if inside:
            resolved_paths[field] = str(path)
        if inside and exists:
            checksums[field] = _sha256(path)
            if field == "fragments" and path.suffix == ".gz":
                check("fragments_gzip_integrity", _gzip_integrity(path), path)

    implementation_files = config.get("implementation_files", [])
    check(
        "implementation_files_declared",
        isinstance(implementation_files, list) and bool(implementation_files),
        implementation_files,
    )
    if isinstance(implementation_files, list):
        for index, value in enumerate(implementation_files):
            key = f"implementation_file_{index}"
            path, inside = _inside_root(root, value)
            exists = path.is_file()
            check(f"{key}_inside_root", inside, path)
            check(f"{key}_exists", exists, path)
            if inside:
                resolved_paths[key] = str(path)
            if inside and exists:
                checksums[key] = _sha256(path)

    if project_config_inside:
        resolved_paths["project_config"] = str(project_config_path)
    if project_config_inside and project_config_path.is_file():
        checksums["project_config"] = _sha256(project_config_path)

    output_dir, output_inside_root = _inside_root(root, config.get("output_dir", ""))
    toy_output_root = root / "results" / "v1" / "g3"
    preflight_output_root = root / "results" / "v1" / "g4" / "preflight"
    m2_output_root = root / "results" / "v1" / "g4" / "m2"
    m2c_output_root = m2_output_root / "cicero_scalability"
    h5b_output_root = root / "results" / "v1" / "g5" / "h5b_external_method_construction"
    output_is_m2c = role == "development" and _is_relative_to(output_dir, m2c_output_root)
    output_is_m2 = role == "development" and _is_relative_to(output_dir, m2_output_root) and not output_is_m2c
    output_inside_authorized_scope = (
        (role == "toy" and _is_relative_to(output_dir, toy_output_root))
        or (role == "development" and _is_relative_to(output_dir, preflight_output_root))
        or output_is_m2
        or output_is_m2c
        or (role == "external_holdout" and _is_relative_to(output_dir, h5b_output_root))
    )
    check("output_dir_inside_root", output_inside_root, output_dir)
    check("output_dir_inside_authorized_scope", output_inside_authorized_scope, output_dir)
    resolved_paths["output_dir"] = str(output_dir)

    if output_is_m2:
        authorization_path = root / G4_M2_AUTHORIZATION
        check("m2_authorization_exists", authorization_path.is_file(), authorization_path)
        m2_authorization: dict[str, Any] = {}
        if authorization_path.is_file():
            try:
                loaded = json.loads(authorization_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    m2_authorization = loaded
            except (OSError, json.JSONDecodeError):
                pass
        check("m2_authorization_parses", bool(m2_authorization), authorization_path)
        check("m2_authorization_status", m2_authorization.get("status") == "authorized", m2_authorization.get("status"))
        check("m2_authorization_milestone", m2_authorization.get("milestone") == "M2", m2_authorization.get("milestone"))
        check(
            "m2_construction_scope_only",
            m2_authorization.get("scope") == "development_only_method_construction_and_contract_qc"
            and m2_authorization.get("development_benchmark_authorized") is True
            and m2_authorization.get("evaluation_tasks_authorized") is False
            and m2_authorization.get("full_benchmark_authorized") is False
            and m2_authorization.get("final_ranking_authorized") is False
            and m2_authorization.get("holdout_accessed") is False,
            m2_authorization.get("scope"),
        )
        authorized_studies = set(m2_authorization.get("authorized_studies", []))
        authorized_methods = set(
            m2_authorization.get("authorized_method_groups", {}).get("fragment_software", [])
        )
        check("m2_study_authorized", str(config.get("study_id", "")) in authorized_studies, config.get("study_id"))
        check("m2_fragment_software_authorized", method_id in authorized_methods, method_id)
        authorized_output_roots = [root / str(value) for value in m2_authorization.get("authorized_output_roots", [])]
        check(
            "m2_output_root_authorized",
            any(_is_relative_to(output_dir, value) for value in authorized_output_roots),
            output_dir,
        )
        check(
            "m2_memory_evidence_required",
            m2_authorization.get("memory_evidence_required_per_attempt") is True,
            m2_authorization.get("memory_evidence_required_per_attempt"),
        )
        memory_gate = m2_authorization.get("memory_gate", {})
        memory_gate_path = root / str(memory_gate.get("path", ""))
        memory_gate_exists = memory_gate_path.is_file()
        check("m2_memory_gate_exists", memory_gate_exists, memory_gate_path)
        observed_memory_hash = _sha256(memory_gate_path) if memory_gate_exists else ""
        check("m2_memory_gate_hash", observed_memory_hash == memory_gate.get("sha256"), observed_memory_hash)
        if authorization_path.is_file():
            resolved_paths["m2_authorization"] = str(authorization_path)
            checksums["m2_authorization"] = _sha256(authorization_path)
        if memory_gate_exists:
            resolved_paths["m2_memory_gate"] = str(memory_gate_path)
            checksums["m2_memory_gate"] = observed_memory_hash

    if output_is_m2c:
        authorization_path = root / G4_M2C_AUTHORIZATION
        check("m2c_authorization_exists", authorization_path.is_file(), authorization_path)
        m2c: dict[str, Any] = {}
        if authorization_path.is_file():
            try:
                loaded = json.loads(authorization_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    m2c = loaded
            except (OSError, json.JSONDecodeError):
                pass
        methods = set(m2c.get("authorized_method_groups", {}).get("subset_scalability_only", []))
        studies = set(m2c.get("authorized_studies", []))
        check("m2c_authorization_status", m2c.get("status") == "authorized" and m2c.get("milestone") == "M2C", m2c.get("status"))
        check("m2c_method_authorized", method_id in methods, method_id)
        check("m2c_study_authorized", str(config.get("study_id", "")) in studies, config.get("study_id"))
        check("m2c_subset_policy", config.get("subset_policy") == "scalability_subset", config.get("subset_policy"))
        check("m2c_evaluation_forbidden", m2c.get("evaluation_tasks_authorized") is False and m2c.get("full_benchmark_authorized") is False and m2c.get("final_ranking_authorized") is False and m2c.get("holdout_accessed") is False, m2c.get("scope"))
        roots = [root / str(value) for value in m2c.get("authorized_output_roots", [])]
        check("m2c_output_root_authorized", any(_is_relative_to(output_dir, value) for value in roots), output_dir)
        if authorization_path.is_file():
            resolved_paths["m2c_authorization"] = str(authorization_path)
            checksums["m2c_authorization"] = _sha256(authorization_path)

    input_h5ad_path = Path(resolved_paths["input_h5ad"]) if "input_h5ad" in resolved_paths else None
    if input_h5ad_path is not None and input_h5ad_path.is_file():
        try:
            adata = ad.read_h5ad(input_h5ad_path, backed="r")
            observed_dataset_id = str(adata.uns.get("dataset_id", ""))
            observed_build = str(adata.uns.get("genome_build", ""))
            check("input_observations_positive", adata.n_obs > 0, adata.n_obs)
            check("input_observation_ids_unique", adata.obs_names.is_unique, adata.n_obs)
            check(
                "input_dataset_id_matches",
                observed_dataset_id in {"", str(config.get("dataset_id", ""))},
                observed_dataset_id,
            )
            check(
                "input_genome_build_matches",
                observed_build == "" or _canonical_build(observed_build) == canonical_build,
                observed_build,
            )
            if method_id in {"signac_geneactivity", "cicero_geneactivity"}:
                check("peak_features_available", adata.n_vars > 0, adata.n_vars)
            adata.file.close()
        except (OSError, ValueError, KeyError) as exc:
            check("input_h5ad_readable", False, exc)

    status = "pass" if checks and all(row["status"] == "pass" for row in checks) else "fail"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(config.get("run_id", "")),
        "method_id": method_id,
        "dataset_id": str(config.get("dataset_id", "")),
        "study_id": str(config.get("study_id", "")),
        "dataset_role": str(role or ""),
        "authorization": authorization,
        "resolved_paths": resolved_paths,
        "input_sha256": checksums,
        "checks": checks,
        "status": status,
    }


def write_wrapper_input_validation(
    config_path: str | Path,
    root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate and write a machine-readable wrapper input decision."""

    config_path = Path(config_path)
    payload = validate_wrapper_input_contract(load_wrapper_config(config_path), root)
    payload["config_path"] = str(config_path)
    payload["config_sha256"] = _sha256(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
