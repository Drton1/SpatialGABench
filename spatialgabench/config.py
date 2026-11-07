"""Config loading and lightweight validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PUBLIC_CONFIG_KEYS = {
    "name",
    "description",
    "inputs",
    "methods",
    "tasks",
    "outputs",
    "comparison_scope",
    "caveats",
    "runtime",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a dictionary."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must parse to a mapping, got {type(data).__name__}")
    return data


def validate_config(path: str | Path) -> dict[str, Any]:
    """Perform intentionally small public-template validation.

    This is not a benchmark execution validator. It checks parseability and
    records whether the template-level public keys are present.
    """

    data = load_config(path)
    present = sorted(PUBLIC_CONFIG_KEYS.intersection(data.keys()))
    missing = sorted(PUBLIC_CONFIG_KEYS.difference(data.keys()))
    return {
        "config": str(path),
        "status": "pass",
        "present_public_keys": present,
        "missing_public_keys": missing,
        "top_level_keys": sorted(map(str, data.keys())),
    }
