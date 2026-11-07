"""Public API for active SpatialGABench simulations."""

from __future__ import annotations

from .config import load_simulation_config, validate_simulation_config
from .design import expand_factorial_design, load_design_config, write_factorial_design
from .fragments import generate_fragments_from_peak_matrix, validate_fragments_for_simulation_dir
from .generator import generate_empirical_semisynthetic_simulation, generate_mechanistic_simulation
from .g3 import run_g3_simulation_smoke
from .validation import summarize_simulation, validate_simulation_outputs

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
