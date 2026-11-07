"""Balanced factorial design expansion and identifiability checks."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import FACTOR_RANGES, SIMULATION_SCHEMA_VERSION


def load_design_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Simulation design config must be a mapping")
    return data


def validate_design_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "design_id", "tracks", "seeds", "factors", "fixed_parameters", "execution"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Simulation design missing required keys: {missing}")
    if int(config["schema_version"]) != 1:
        raise ValueError("Simulation design schema_version must be 1")
    if set(config["tracks"]) != {"mechanistic", "empirical_semisynthetic"}:
        raise ValueError("Design tracks must contain mechanistic and empirical_semisynthetic")
    if len(set(map(int, config["seeds"]))) < 5:
        raise ValueError("At least five unique seeds are required")
    if set(config["factors"]) != set(FACTOR_RANGES):
        raise ValueError(f"Design factors must be exactly {sorted(FACTOR_RANGES)}")
    for factor, values in config["factors"].items():
        if not isinstance(values, list) or len(values) != 2 or len(set(map(float, values))) != 2:
            raise ValueError(f"G3 design requires exactly two distinct levels for {factor}")
        lower, upper = FACTOR_RANGES[factor]
        if any(not lower <= float(value) <= upper for value in values):
            raise ValueError(f"Design level outside allowed range for {factor}")
    execution = config["execution"]
    if execution.get("full_benchmark_enabled") is not False:
        raise ValueError("G3 simulation design must keep full_benchmark_enabled=false")
    if execution.get("holdout_access_allowed") is not False:
        raise ValueError("G3 simulation design must keep holdout_access_allowed=false")
    if int(config["fixed_parameters"].get("generator_schema_version", -1)) != SIMULATION_SCHEMA_VERSION:
        raise ValueError("fixed_parameters.generator_schema_version does not match active simulation schema")


def _design_audit(table: pd.DataFrame, factor_names: list[str], seeds: list[int]) -> dict[str, Any]:
    per_track = {}
    expected_conditions = 2 ** len(factor_names)
    for track, frame in table.groupby("simulation_type", sort=True):
        one_seed = frame.loc[frame["random_seed"] == seeds[0]].copy()
        coded = np.column_stack(
            [
                np.where(one_seed[factor].to_numpy(float) == one_seed[factor].min(), -1.0, 1.0)
                for factor in factor_names
            ]
        )
        columns = [np.ones(coded.shape[0])]
        columns.extend(coded[:, index] for index in range(coded.shape[1]))
        columns.extend(coded[:, i] * coded[:, j] for i, j in itertools.combinations(range(coded.shape[1]), 2))
        model_matrix = np.column_stack(columns)
        correlation = np.corrcoef(coded, rowvar=False)
        off_diagonal = correlation - np.eye(correlation.shape[0])
        level_counts = {
            factor: {str(level): int(count) for level, count in frame[factor].value_counts().sort_index().items()}
            for factor in factor_names
        }
        rank = int(np.linalg.matrix_rank(model_matrix))
        expected_rank = int(model_matrix.shape[1])
        per_track[str(track)] = {
            "n_rows": int(frame.shape[0]),
            "n_conditions_per_seed": int(one_seed.shape[0]),
            "expected_conditions_per_seed": expected_conditions,
            "n_seeds": int(frame["random_seed"].nunique()),
            "main_and_two_way_model_rank": rank,
            "expected_model_rank": expected_rank,
            "max_abs_main_effect_correlation": float(np.abs(off_diagonal).max()),
            "level_counts": level_counts,
            "status": "pass"
            if one_seed.shape[0] == expected_conditions
            and frame["random_seed"].nunique() == len(seeds)
            and rank == expected_rank
            and float(np.abs(off_diagonal).max()) < 1e-12
            else "fail",
        }
    return {
        "status": "pass" if all(item["status"] == "pass" for item in per_track.values()) else "fail",
        "factor_order": factor_names,
        "tracks": per_track,
        "full_benchmark_run": False,
        "holdout_accessed": False,
    }


def expand_factorial_design(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand a replicated full factorial and mark an OAT smoke subset."""

    validate_design_config(config)
    factor_names = list(config["factors"])
    level_product = list(itertools.product(*(config["factors"][name] for name in factor_names)))
    baseline = tuple(config["factors"][name][0] for name in factor_names)
    oat_conditions = {baseline}
    for index in range(len(factor_names)):
        condition = list(baseline)
        condition[index] = config["factors"][factor_names[index]][1]
        oat_conditions.add(tuple(condition))

    rows: list[dict[str, Any]] = []
    seeds = list(map(int, config["seeds"]))
    for track in config["tracks"]:
        short_track = "mech" if track == "mechanistic" else "semi"
        for seed_index, seed in enumerate(seeds, start=1):
            for condition_index, values in enumerate(level_product, start=1):
                row = {
                    "design_id": str(config["design_id"]),
                    "simulation_id": f"{config['design_id']}__{short_track}__r{seed_index:02d}__c{condition_index:03d}",
                    "simulation_type": track,
                    "replicate_id": f"replicate_{seed_index:02d}",
                    "random_seed": seed,
                    "condition_index": condition_index,
                    "smoke_member": seed_index == 1 and tuple(values) in oat_conditions,
                    "full_benchmark_run": False,
                    "holdout_accessed": False,
                }
                row.update({name: float(value) for name, value in zip(factor_names, values)})
                rows.append(row)
    table = pd.DataFrame(rows)
    return table, _design_audit(table, factor_names, seeds)


def write_factorial_design(config_path: str | Path, outdir: str | Path) -> dict[str, Path]:
    config = load_design_config(config_path)
    table, audit = expand_factorial_design(config)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table_path = outdir / "simulation_factorial_design.tsv"
    audit_path = outdir / "simulation_factorial_design_audit.json"
    table.to_csv(table_path, sep="\t", index=False)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"design": table_path, "audit": audit_path}
