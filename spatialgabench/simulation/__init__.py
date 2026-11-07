"""Simulation utilities for SpatialGABench."""

from .api import (
    expand_factorial_design,
    generate_empirical_semisynthetic_simulation,
    generate_fragments_from_peak_matrix,
    generate_mechanistic_simulation,
    load_design_config,
    load_simulation_config,
    run_g3_simulation_smoke,
    summarize_simulation,
    validate_fragments_for_simulation_dir,
    validate_simulation_config,
    validate_simulation_outputs,
    write_factorial_design,
)

__all__ = [
    "load_simulation_config",
    "validate_simulation_config",
    "load_design_config",
    "expand_factorial_design",
    "write_factorial_design",
    "generate_mechanistic_simulation",
    "generate_empirical_semisynthetic_simulation",
    "run_g3_simulation_smoke",
    "generate_fragments_from_peak_matrix",
    "validate_fragments_for_simulation_dir",
    "validate_simulation_outputs",
    "summarize_simulation",
]
