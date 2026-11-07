"""Versioned configuration contracts for active simulation generators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SIMULATION_SCHEMA_VERSION = 2
ACTIVE_SIMULATION_TYPES = {"mechanistic", "empirical_semisynthetic", "fragments_from_peak_matrix"}
RETIRED_SIMULATION_TYPES = {"controlled_synthetic", "scdesign3_calibrated_truth_injected"}
FACTOR_RANGES = {
    "distal_fraction": (0.0, 0.9),
    "depth_multiplier": (0.05, 10.0),
    "zero_inflation_rate": (0.0, 0.95),
    "domain_log_effect": (0.0, 3.0),
    "decoy_fraction": (0.0, 1.0),
    "rna_latent_correlation": (0.0, 0.99),
}
DOMAIN_GEOMETRIES = {"quadrants", "gradient", "nested"}
COMPONENT_ALLOCATION_MODELS = {"centered_bounded_cell_gene_v1"}

COMMON_KEYS = {
    "schema_version",
    "simulation_id",
    "simulation_type",
    "random_seed",
    "replicate_id",
    "n_cells",
    "n_peaks",
    "n_genes",
    "n_domains",
    "distal_peaks_per_gene",
    "domain_geometry",
    "causal_signal_fraction",
    "base_library_size",
    "rna_mean_count",
    "factors",
    "tf_program_enabled",
    "fragments_enabled",
    "output_dir",
    "truth_outputs",
    "validation_outputs",
}

EMPIRICAL_CALIBRATION_KEYS = {
    "source_h5ad",
    "authorization_json",
    "source_dataset_id",
    "source_study_id",
    "source_role",
    "redistribution_policy",
}


def load_simulation_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML simulation config without mutating it."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Simulation config does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Simulation config must parse to a mapping.")
    return data


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _validate_fragment_bridge(config: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "simulation_id", "simulation_type", "random_seed", "output_dir"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Simulation config missing required keys: {missing}")
    return {
        "status": "pass",
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "simulation_id": str(config["simulation_id"]),
        "simulation_type": "fragments_from_peak_matrix",
    }


def validate_simulation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate active simulation configs and reject retired claim-bearing paths."""

    if not isinstance(config, dict):
        raise ValueError("Simulation config must be a mapping")
    sim_type = str(config.get("simulation_type", ""))
    if sim_type in RETIRED_SIMULATION_TYPES:
        raise ValueError(
            f"simulation_type {sim_type!r} is retired because the v0.1 generator was not identifiable "
            "and its scDesign3-labelled path did not invoke scDesign3"
        )
    if sim_type not in ACTIVE_SIMULATION_TYPES:
        raise ValueError(f"simulation_type must be one of {sorted(ACTIVE_SIMULATION_TYPES)}, got {sim_type!r}")

    schema_version = int(config.get("schema_version", -1))
    if schema_version != SIMULATION_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SIMULATION_SCHEMA_VERSION}, got {schema_version}")
    if sim_type == "fragments_from_peak_matrix":
        return _validate_fragment_bridge(config)

    missing = sorted(COMMON_KEYS.difference(config))
    if missing:
        raise ValueError(f"Simulation config missing required keys: {missing}")

    integer_values = {key: int(config[key]) for key in ["n_cells", "n_peaks", "n_genes", "n_domains", "distal_peaks_per_gene"]}
    if any(value <= 0 for value in integer_values.values()):
        raise ValueError("n_cells, n_peaks, n_genes, n_domains, and distal_peaks_per_gene must be positive")
    if integer_values["n_domains"] < 2:
        raise ValueError("n_domains must be at least 2")
    if integer_values["n_cells"] > 200000 or integer_values["n_peaks"] > 2000000:
        raise ValueError("Reusable CLI refuses very large simulations; use an explicitly reviewed workflow")
    minimum_peaks = integer_values["n_genes"] * (2 + integer_values["distal_peaks_per_gene"])
    if integer_values["n_peaks"] <= minimum_peaks:
        raise ValueError(
            "n_peaks must exceed n_genes * (2 + distal_peaks_per_gene) so the design contains "
            "promoter, gene-body proxy, distal, and auxiliary peaks without truncation"
        )

    geometry = str(config["domain_geometry"])
    if geometry not in DOMAIN_GEOMETRIES:
        raise ValueError(f"domain_geometry must be one of {sorted(DOMAIN_GEOMETRIES)}")
    factors = _require_mapping(config, "factors")
    missing_factors = sorted(set(FACTOR_RANGES).difference(factors))
    extra_factors = sorted(set(factors).difference(FACTOR_RANGES))
    if missing_factors or extra_factors:
        raise ValueError(f"factors mismatch: missing={missing_factors}, extra={extra_factors}")
    for factor, (lower, upper) in FACTOR_RANGES.items():
        value = float(factors[factor])
        if not lower <= value <= upper:
            raise ValueError(f"{factor} must be within [{lower}, {upper}], got {value}")

    causal_fraction = float(config["causal_signal_fraction"])
    if not 0.0 < causal_fraction < 1.0:
        raise ValueError("causal_signal_fraction must be strictly between 0 and 1")
    if float(config["base_library_size"]) <= 0 or float(config["rna_mean_count"]) <= 0:
        raise ValueError("base_library_size and rna_mean_count must be positive")
    allocation_amplitude = float(config.get("component_allocation_amplitude_fraction", 0.5))
    if not 0.0 < allocation_amplitude <= 1.0:
        raise ValueError("component_allocation_amplitude_fraction must be within (0, 1]")
    allocation_model = str(config.get("component_allocation_model", "centered_bounded_cell_gene_v1"))
    if allocation_model not in COMPONENT_ALLOCATION_MODELS:
        raise ValueError(f"component_allocation_model must be one of {sorted(COMPONENT_ALLOCATION_MODELS)}")
    for key in ["truth_outputs", "validation_outputs"]:
        if not isinstance(config.get(key), list) or not config[key]:
            raise ValueError(f"{key} must be a non-empty list")

    if sim_type == "empirical_semisynthetic":
        calibration = _require_mapping(config, "calibration")
        missing_calibration = sorted(EMPIRICAL_CALIBRATION_KEYS.difference(calibration))
        if missing_calibration:
            raise ValueError(f"calibration missing required keys: {missing_calibration}")
        if calibration["source_role"] != "development_only":
            raise ValueError("empirical calibration source_role must be development_only")
        if calibration["redistribution_policy"] != "local_only_source_derived":
            raise ValueError("empirical outputs must use redistribution_policy=local_only_source_derived")

    return {
        "status": "pass",
        "schema_version": schema_version,
        "simulation_id": str(config["simulation_id"]),
        "simulation_type": sim_type,
        **integer_values,
        "minimum_causal_peaks": minimum_peaks,
        "n_auxiliary_peaks": integer_values["n_peaks"] - minimum_peaks,
    }


def load_and_validate_simulation_config(config_path: str | Path) -> dict[str, Any]:
    config = load_simulation_config(config_path)
    validate_simulation_config(config)
    return config
