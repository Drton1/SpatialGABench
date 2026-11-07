"""Utilities for external software gene-activity wrappers."""

from .contracts import (
    METHOD_CONTRACTS,
    load_wrapper_config,
    validate_wrapper_input_contract,
    write_wrapper_input_validation,
)
from .preflight import build_real_data_contract_fixture, write_g4_contract_preflight_evidence

__all__ = [
    "METHOD_CONTRACTS",
    "load_wrapper_config",
    "validate_wrapper_input_contract",
    "write_wrapper_input_validation",
    "build_real_data_contract_fixture",
    "write_g4_contract_preflight_evidence",
]
