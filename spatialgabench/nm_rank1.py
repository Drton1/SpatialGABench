"""NM-RANK1 evaluation-regime ranking stability audit."""

from __future__ import annotations

import hashlib
import gzip
import json
import logging
import math
import os
import platform
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.stats import kendalltau, rankdata


LOGGER = logging.getLogger("nm_rank1")
SIMULATION_RE = re.compile(r"__r(?P<replicate>\d+)__c(?P<factor_cell>\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must parse to a mapping")
    return config


def resolve_project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
    }


def stable_seed(base: int, key: str) -> int:
    return base + int(sha256_text(key)[:8], 16) % 100000


def verify_freeze(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_project_path(root, config["protocol"]["freeze_manifest"])
    with manifest_path.open("r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    checks: list[dict[str, Any]] = []

    config_path = root / "configs" / "nm_rank1_rank_stability_2026-10-18.yaml"
    expected_config = freeze["config"]
    actual_config = sha256_file(config_path)
    checks.append(
        {
            "path": str(config_path.relative_to(root)),
            "expected_sha256": expected_config["sha256"],
            "actual_sha256": actual_config,
            "match": actual_config == expected_config["sha256"],
            "role": "config",
        }
    )

    prereg_path = resolve_project_path(root, freeze["preregistration"]["path"])
    actual_prereg = sha256_file(prereg_path)
    checks.append(
        {
            "path": str(prereg_path.relative_to(root)),
            "expected_sha256": freeze["preregistration"]["sha256"],
            "actual_sha256": actual_prereg,
            "match": actual_prereg == freeze["preregistration"]["sha256"],
            "role": "preregistration",
        }
    )

    for item in freeze["inputs"]:
        input_path = resolve_project_path(root, item["path"])
        actual = sha256_file(input_path)
        checks.append(
            {
                "path": str(input_path.relative_to(root)),
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "match": actual == item["sha256"],
                "role": "analysis_input",
            }
        )

    failed = [item for item in checks if not item["match"]]
    if failed:
        raise ValueError(f"Freeze hash mismatch: {failed}")
    return checks


def parse_simulation_components(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["simulation_id"].astype(str).str.extract(
        r"__r(?P<replicate>\d+)__c(?P<factor_cell>\d+)$"
    )
    if parsed.isna().any().any():
        raise ValueError("simulation_id does not end in __rNN__cNNN for every row")
    result = frame.copy()
    result["replicate_index"] = parsed["replicate"].astype(int)
    result["factor_cell"] = parsed["factor_cell"].astype(int)
    return result


def load_inputs(config: dict[str, Any], root: Path) -> dict[str, Any]:
    metrics_path = resolve_project_path(root, config["inputs"]["simulation_metrics"])
    summary_path = resolve_project_path(root, config["inputs"]["endpoint_summary"])
    key_path = resolve_project_path(root, config["inputs"]["ds2_key_numbers"])
    manifest_path = resolve_project_path(root, config["inputs"]["ds2_run_manifest"])
    for path in (metrics_path, summary_path, key_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = parse_simulation_components(pd.read_csv(metrics_path, sep="\t"))
    summary = pd.read_csv(summary_path, sep="\t")
    with key_path.open("r", encoding="utf-8") as handle:
        key_numbers = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        run_manifest = json.load(handle)

    artifacts = run_manifest.get("artifacts", {})
    for path in (metrics_path, summary_path, key_path):
        relative = str(path.relative_to(root))
        expected = artifacts.get(relative)
        if expected is None:
            raise ValueError(f"DS2 run manifest lacks artifact hash for {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"DS2 artifact hash mismatch for {relative}: {actual} != {expected}")

    return {
        "metrics": metrics,
        "summary": summary,
        "key_numbers": key_numbers,
        "run_manifest": run_manifest,
        "input_paths": [metrics_path, summary_path, key_path, manifest_path],
    }


def validate_inputs(config: dict[str, Any], inputs: dict[str, Any]) -> None:
    metrics = inputs["metrics"]
    summary = inputs["summary"]
    key_numbers = inputs["key_numbers"]
    factors = list(config["factors"])
    endpoints = list(config["endpoints"])
    strata = list(config["strata"])
    annotation = list(config["methods"]["annotation"])
    controls = list(config["methods"]["controls"])
    all_methods = annotation + controls

    required = {
        "simulation_id",
        "simulation_type",
        "method_id",
        *endpoints,
        *factors,
    }
    if not required.issubset(metrics.columns):
        raise ValueError(f"Simulation metrics lack columns {sorted(required - set(metrics.columns))}")
    if len(metrics) != 4480:
        raise ValueError(f"Expected 4,480 simulation metrics rows, found {len(metrics)}")
    if sorted(metrics["simulation_type"].unique().tolist()) != sorted(strata):
        raise ValueError("Simulation strata do not match the config")
    if sorted(metrics["method_id"].unique().tolist()) != sorted(all_methods):
        raise ValueError("Simulation methods do not match the config")
    if "endpoint" in metrics.columns:
        raise ValueError("Simulation metrics must be wide, not long by endpoint")

    counts = metrics.groupby(["simulation_type", "factor_cell", "method_id"]).size()
    if not bool((counts == 5).all()):
        raise ValueError("Every stratum/factor cell/method must have exactly five seeds")
    if int(metrics["factor_cell"].nunique()) != 64:
        raise ValueError("Expected 64 factor cells")
    if int(metrics.groupby("simulation_type")["factor_cell"].nunique().min()) != 64:
        raise ValueError("Each stratum must contain all 64 factor cells")

    factor_nunique = metrics.groupby(["simulation_type", "factor_cell"])[factors].nunique()
    if bool((factor_nunique > 1).any().any()):
        raise ValueError("A factor varies within a simulation factor cell")
    if bool((metrics.groupby("simulation_type")[factors].nunique() != 2).any().any()):
        raise ValueError("Every factor must have exactly two levels within each stratum")

    for endpoint in endpoints:
        if not np.isfinite(metrics[endpoint].to_numpy(dtype=float)).all():
            raise ValueError(f"Endpoint {endpoint} contains non-finite values")

    if len(summary) != 2 * 7 * 9:
        raise ValueError(f"Expected 126 endpoint-summary rows, found {len(summary)}")
    if sorted(summary["method_id"].unique().tolist()) != sorted(all_methods):
        raise ValueError("Endpoint-summary methods do not match the config")
    if sorted(summary["endpoint"].unique().tolist()) != sorted(endpoints):
        raise ValueError("Endpoint-summary endpoints do not match the config")
    if sorted(summary["simulation_type"].unique().tolist()) != sorted(strata):
        raise ValueError("Endpoint-summary strata do not match the config")

    if int(key_numbers.get("simulation_rows", -1)) != 4480:
        raise ValueError("DS2 key numbers do not report 4,480 simulation rows")
    if int(key_numbers.get("simulations", -1)) != 640:
        raise ValueError("DS2 key numbers do not report 640 simulations")
    if int(key_numbers.get("methods", -1)) != 7:
        raise ValueError("DS2 key numbers do not report 7 methods")
    if sorted(key_numbers.get("strata", [])) != sorted(strata):
        raise ValueError("DS2 key-number strata do not match the config")
    if sorted(key_numbers.get("endpoint_columns", [])) != sorted(endpoints):
        raise ValueError("DS2 key-number endpoints do not match the config")


def build_long_cells(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    annotation = list(config["methods"]["annotation"])
    endpoints = list(config["endpoints"])
    factors = list(config["factors"])
    rows: list[dict[str, Any]] = []
    for stratum in config["strata"]:
        stratum_frame = metrics[metrics["simulation_type"].astype(str) == str(stratum)]
        for factor_cell, cell_frame in stratum_frame.groupby("factor_cell", sort=True):
            factor_values = {factor: cell_frame[factor].iloc[0] for factor in factors}
            for endpoint in endpoints:
                for method in annotation:
                    selected = cell_frame[cell_frame["method_id"].astype(str) == method]
                    if len(selected) != 5:
                        raise ValueError(f"Expected five seeds for {stratum}/{factor_cell}/{method}")
                    rows.append(
                        {
                            "simulation_type": str(stratum),
                            "factor_cell": int(factor_cell),
                            "endpoint": endpoint,
                            "method_id": method,
                            "cell_median": float(selected[endpoint].median()),
                            "n_seeds": int(len(selected)),
                            **{factor: factor_values[factor] for factor in factors},
                        }
                    )
    result = pd.DataFrame(rows)
    expected = len(config["strata"]) * 64 * len(endpoints) * len(annotation)
    if len(result) != expected:
        raise ValueError(f"Long cell table has {len(result)} rows, expected {expected}")
    return result


def top1_set(values: pd.Series, methods: pd.Series, tolerance: float) -> tuple[str, ...]:
    maximum = float(values.max())
    return tuple(sorted(str(method) for method in methods[values >= maximum - tolerance]))


def build_overall_scores(long_cells: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    rows: list[dict[str, Any]] = []
    for (stratum, endpoint), group in long_cells.groupby(["simulation_type", "endpoint"], sort=True):
        method_scores = (
            group.groupby("method_id", as_index=False)["cell_median"]
            .median()
            .rename(columns={"cell_median": "overall_median"})
        )
        method_scores["overall_rank"] = method_scores["overall_median"].rank(
            method="average", ascending=False
        )
        top_set = top1_set(
            method_scores["overall_median"],
            method_scores["method_id"],
            tolerance,
        )
        for _, row in method_scores.iterrows():
            rows.append(
                {
                    "simulation_type": stratum,
                    "endpoint": endpoint,
                    "method_id": str(row["method_id"]),
                    "overall_median": float(row["overall_median"]),
                    "overall_rank": float(row["overall_rank"]),
                    "overall_top1_methods": "|".join(top_set),
                }
            )
    return pd.DataFrame(rows)


def build_cell_tables(
    long_cells: pd.DataFrame,
    overall: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    overall_lookup = {
        (row.simulation_type, row.endpoint, row.method_id): row
        for row in overall.itertuples(index=False)
    }
    method_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for (stratum, endpoint, factor_cell), group in long_cells.groupby(
        ["simulation_type", "endpoint", "factor_cell"], sort=True
    ):
        values = group["cell_median"].to_numpy(dtype=float)
        methods = group["method_id"].to_numpy(dtype=str)
        cell_ranks = rankdata(-values, method="average")
        overall_ranks = np.array(
            [overall_lookup[(stratum, endpoint, method)].overall_rank for method in methods],
            dtype=float,
        )
        if float(np.ptp(values)) <= tolerance:
            tau = float("nan")
            degenerate = True
        else:
            tau = float(kendalltau(cell_ranks, overall_ranks).statistic)
            degenerate = False
        cell_top_set = top1_set(pd.Series(values), pd.Series(methods), tolerance)
        overall_top_set = tuple(
            sorted(str(overall_lookup[(stratum, endpoint, methods[0])].overall_top1_methods).split("|"))
        )
        cell_rows.append(
            {
                "simulation_type": stratum,
                "endpoint": endpoint,
                "factor_cell": int(factor_cell),
                "kendall_tau": tau,
                "degenerate_all_tied": bool(degenerate),
                "cell_top1_methods": "|".join(cell_top_set),
                "overall_top1_methods": "|".join(overall_top_set),
                "top1_match": bool(cell_top_set == overall_top_set),
                **{factor: group[factor].iloc[0] for factor in config["factors"]},
            }
        )
        for method, value, cell_rank, overall_rank in zip(methods, values, cell_ranks, overall_ranks):
            method_rows.append(
                {
                    "simulation_type": stratum,
                    "endpoint": endpoint,
                    "factor_cell": int(factor_cell),
                    "method_id": str(method),
                    "cell_median": float(value),
                    "cell_rank": float(cell_rank),
                    "overall_rank": float(overall_rank),
                    "kendall_tau": tau,
                    "degenerate_all_tied": bool(degenerate),
                    "cell_top1_methods": "|".join(cell_top_set),
                    "overall_top1_methods": "|".join(overall_top_set),
                    "top1_match": bool(cell_top_set == overall_top_set),
                }
            )
    methods_frame = pd.DataFrame(method_rows)
    cells_frame = pd.DataFrame(cell_rows)
    expected_methods = len(config["strata"]) * len(config["endpoints"]) * 64 * len(config["methods"]["annotation"])
    expected_cells = len(config["strata"]) * len(config["endpoints"]) * 64
    if len(methods_frame) != expected_methods:
        raise ValueError(f"Method ranking table has {len(methods_frame)} rows, expected {expected_methods}")
    if len(cells_frame) != expected_cells:
        raise ValueError(f"Cell summary table has {len(cells_frame)} rows, expected {expected_cells}")
    return methods_frame, cells_frame


def build_pair_reversals(
    long_cells: pd.DataFrame,
    overall: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    threshold = float(config["thresholds"]["sign_reversal_fraction"])
    annotation = list(config["methods"]["annotation"])
    rows: list[dict[str, Any]] = []
    for stratum in config["strata"]:
        for endpoint in config["endpoints"]:
            scores = overall[
                (overall["simulation_type"] == stratum) & (overall["endpoint"] == endpoint)
            ].set_index("method_id")["overall_median"]
            cells = long_cells[
                (long_cells["simulation_type"] == stratum) & (long_cells["endpoint"] == endpoint)
            ]
            for left_index, method_a in enumerate(annotation):
                for method_b in annotation[left_index + 1 :]:
                    overall_diff = float(scores[method_a] - scores[method_b])
                    overall_sign = 0 if abs(overall_diff) <= tolerance else int(np.sign(overall_diff))
                    n_match = n_reversal = n_zero = 0
                    for _, cell in cells.groupby("factor_cell", sort=True):
                        value_a = float(cell.loc[cell["method_id"] == method_a, "cell_median"].iloc[0])
                        value_b = float(cell.loc[cell["method_id"] == method_b, "cell_median"].iloc[0])
                        diff = value_a - value_b
                        if abs(diff) <= tolerance:
                            n_zero += 1
                        elif int(np.sign(diff)) == overall_sign:
                            n_match += 1
                        else:
                            n_reversal += 1
                    n_cells = n_match + n_reversal + n_zero
                    denominator = n_match + n_reversal
                    sign_stability = float(n_match / denominator) if denominator else float("nan")
                    reversal_fraction = float(n_reversal / n_cells) if n_cells else float("nan")
                    status = "ESTIMABLE" if overall_sign != 0 else "NOT_ESTIMABLE_OVERALL_TIE"
                    rows.append(
                        {
                            "simulation_type": stratum,
                            "endpoint": endpoint,
                            "method_a": method_a,
                            "method_b": method_b,
                            "overall_diff_a_minus_b": overall_diff,
                            "overall_sign": overall_sign,
                            "n_cells": int(n_cells),
                            "n_sign_match": int(n_match),
                            "n_sign_reversal": int(n_reversal),
                            "n_zero": int(n_zero),
                            "sign_stability": sign_stability,
                            "reversal_fraction": reversal_fraction,
                            "reversal_ge_20pct": bool(
                                status == "ESTIMABLE"
                                and math.isfinite(reversal_fraction)
                                and reversal_fraction >= threshold
                            ),
                            "status": status,
                        }
                    )
    result = pd.DataFrame(rows)
    expected = len(config["strata"]) * len(config["endpoints"]) * 10
    if len(result) != expected:
        raise ValueError(f"Pair-reversal table has {len(result)} rows, expected {expected}")
    return result


def bootstrap_mean_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_endpoint_stability(
    cells: pd.DataFrame,
    overall: pd.DataFrame,
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    draws = int(config["thresholds"]["bootstrap_draws"])
    base_seed = int(config["seeds"]["bootstrap"])
    low_threshold = float(config["thresholds"]["dynamic_range_low"])
    rows: list[dict[str, Any]] = []
    for stratum in config["strata"]:
        for endpoint in config["endpoints"]:
            cell_group = cells[
                (cells["simulation_type"] == stratum) & (cells["endpoint"] == endpoint)
            ]
            valid = cell_group.loc[np.isfinite(cell_group["kendall_tau"]), "kendall_tau"].to_numpy(dtype=float)
            if len(valid) == 0:
                mean_tau = float("nan")
                ci_low = ci_high = float("nan")
            else:
                mean_tau = float(valid.mean())
                ci_low, ci_high = bootstrap_mean_ci(
                    valid, draws, stable_seed(base_seed, f"{stratum}::{endpoint}")
                )
            method_scores = overall[
                (overall["simulation_type"] == stratum) & (overall["endpoint"] == endpoint)
            ]
            summary_scores = summary[
                (summary["simulation_type"].astype(str) == str(stratum))
                & (summary["endpoint"].astype(str) == str(endpoint))
                & (summary["method_id"].astype(str).isin(config["methods"]["annotation"]))
            ]
            if len(summary_scores) != len(config["methods"]["annotation"]):
                raise ValueError(
                    f"Frozen endpoint summary lacks annotation methods for {stratum}/{endpoint}"
                )
            method_min = float(summary_scores["median"].min())
            method_max = float(summary_scores["median"].max())
            method_range = method_max - method_min
            pair_group = pairs[
                (pairs["simulation_type"] == stratum) & (pairs["endpoint"] == endpoint)
            ]
            estimable_pairs = pair_group[pair_group["status"] == "ESTIMABLE"]
            rows.append(
                {
                    "simulation_type": stratum,
                    "endpoint": endpoint,
                    "n_cells": int(len(cell_group)),
                    "n_valid_tau_cells": int(len(valid)),
                    "mean_kendall_tau": mean_tau,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "top1_stability": float(cell_group["top1_match"].mean()),
                    "n_distinct_top1_sets": int(cell_group["cell_top1_methods"].nunique()),
                    "overall_top1_methods": str(method_scores["overall_top1_methods"].iloc[0]),
                    "method_median_min": method_min,
                    "method_median_max": method_max,
                    "method_median_range": method_range,
                    "dynamic_range_class": "low" if method_range <= low_threshold else "high",
                    "min_pair_sign_stability": float(pair_group["sign_stability"].min()),
                    "median_pair_sign_stability": float(pair_group["sign_stability"].median()),
                    "n_pairs_estimable": int(len(estimable_pairs)),
                    "n_pairs_not_estimable_overall_tie": int(len(pair_group) - len(estimable_pairs)),
                    "max_pair_reversal_fraction": float(estimable_pairs["reversal_fraction"].max()),
                    "n_pairs_reversal_ge_20pct": int(estimable_pairs["reversal_ge_20pct"].sum()),
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != len(config["strata"]) * len(config["endpoints"]):
        raise ValueError("Endpoint-stability row count is incorrect")
    return result


def build_method_summary(methods: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stratum, endpoint, method), group in methods.groupby(
        ["simulation_type", "endpoint", "method_id"], sort=True
    ):
        overall_row = overall[
            (overall["simulation_type"] == stratum)
            & (overall["endpoint"] == endpoint)
            & (overall["method_id"] == method)
        ].iloc[0]
        rows.append(
            {
                "simulation_type": stratum,
                "endpoint": endpoint,
                "method_id": method,
                "overall_median": float(overall_row["overall_median"]),
                "overall_rank": float(overall_row["overall_rank"]),
                "rank_min": float(group["cell_rank"].min()),
                "rank_max": float(group["cell_rank"].max()),
                "rank_range": float(group["cell_rank"].max() - group["cell_rank"].min()),
                "top1_count": int((group["cell_top1_methods"].str.split("|").apply(lambda items: method in items)).sum()),
                "top1_fraction": float(
                    group["cell_top1_methods"].str.split("|").apply(lambda items: method in items).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_factor_sensitivity(cells: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum in config["strata"]:
        for endpoint in config["endpoints"]:
            group = cells[
                (cells["simulation_type"] == stratum) & (cells["endpoint"] == endpoint)
            ]
            for factor in config["factors"]:
                for level, level_group in group.groupby(factor, sort=True):
                    rows.append(
                        {
                            "simulation_type": stratum,
                            "endpoint": endpoint,
                            "factor": factor,
                            "level": level,
                            "n_cells": int(len(level_group)),
                            "mean_kendall_tau": float(level_group["kendall_tau"].mean()),
                            "mean_top1_stability": float(level_group["top1_match"].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def build_control_check(
    long_cells: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    controls = list(config["methods"]["controls"])
    annotation = list(config["methods"]["annotation"])
    rows: list[dict[str, Any]] = []
    for stratum in config["strata"]:
        for endpoint in config["endpoints"]:
            n_cells = 0
            n_both_below = 0
            for factor_cell in range(1, 65):
                cell = metrics[
                    (metrics["simulation_type"] == stratum)
                    & (metrics["factor_cell"] == factor_cell)
                ]
                annotation_max = max(
                    float(cell.loc[cell["method_id"] == method, endpoint].median())
                    for method in annotation
                )
                annotation_min = min(
                    float(cell.loc[cell["method_id"] == method, endpoint].median())
                    for method in annotation
                )
                n_cells += 1
                both_below = True
                for control in controls:
                    control_value = float(cell.loc[cell["method_id"] == control, endpoint].median())
                    if not control_value < annotation_min - tolerance:
                        both_below = False
                if both_below:
                    n_both_below += 1
            rows.append(
                {
                    "simulation_type": stratum,
                    "endpoint": endpoint,
                    "n_cells": n_cells,
                    "n_cells_both_controls_below_all_annotation_methods": n_both_below,
                    "fraction_both_controls_below": float(n_both_below / n_cells),
                }
            )
    return pd.DataFrame(rows)


def hypothesis_verdicts(
    endpoint_stability: pd.DataFrame,
    pairs: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    stability_tau = float(config["thresholds"]["stability_tau"])
    material_gap = float(config["thresholds"]["material_tau_gap"])
    reversal_threshold = float(config["thresholds"]["sign_reversal_fraction"])

    mean_taus = endpoint_stability["mean_kendall_tau"].to_numpy(dtype=float)
    h1_value = float(np.nanmin(mean_taus))
    h1 = "PASS" if h1_value < stability_tau else "FAIL"

    low = endpoint_stability.loc[endpoint_stability["dynamic_range_class"] == "low", "mean_kendall_tau"]
    high = endpoint_stability.loc[endpoint_stability["dynamic_range_class"] == "high", "mean_kendall_tau"]
    if len(low) >= 2 and len(high) >= 2:
        mean_low = float(low.mean())
        mean_high = float(high.mean())
        gap = mean_low - mean_high
        h2 = "PASS" if gap <= -material_gap else "FAIL"
        h2_status = "ESTIMABLE"
    else:
        mean_low = mean_high = gap = float("nan")
        h2 = "NOT_ESTIMABLE"
        h2_status = "INSUFFICIENT_DYNAMIC_RANGE_CLASSES"

    valid_pairs = pairs[
        (pairs["status"] == "ESTIMABLE")
        & np.isfinite(pairs["reversal_fraction"].to_numpy(dtype=float))
    ]
    max_reversal = float(valid_pairs["reversal_fraction"].max()) if len(valid_pairs) else float("nan")
    h3 = "PASS" if math.isfinite(max_reversal) and max_reversal >= reversal_threshold else "FAIL"
    if math.isfinite(max_reversal):
        max_row = valid_pairs.loc[valid_pairs["reversal_fraction"].idxmax()]
        max_reversal_detail = {
            "simulation_type": str(max_row["simulation_type"]),
            "endpoint": str(max_row["endpoint"]),
            "method_a": str(max_row["method_a"]),
            "method_b": str(max_row["method_b"]),
            "reversal_fraction": float(max_row["reversal_fraction"]),
        }
    else:
        max_reversal_detail = None

    h4_endpoints: dict[str, Any] = {}
    h4_pass = False
    h4_not_estimable = False
    for endpoint in config["endpoints"]:
        mechanistic = endpoint_stability[
            (endpoint_stability["simulation_type"] == "mechanistic")
            & (endpoint_stability["endpoint"] == endpoint)
        ]["overall_top1_methods"].iloc[0]
        empirical = endpoint_stability[
            (endpoint_stability["simulation_type"] == "empirical_semisynthetic")
            & (endpoint_stability["endpoint"] == endpoint)
        ]["overall_top1_methods"].iloc[0]
        mech_set = str(mechanistic).split("|")
        emp_set = str(empirical).split("|")
        if len(mech_set) != 1 or len(emp_set) != 1:
            status = "NOT_ESTIMABLE_TIE"
            h4_not_estimable = True
        elif mech_set[0] != emp_set[0]:
            status = "DIFFERENT"
            h4_pass = True
        else:
            status = "SAME"
        h4_endpoints[endpoint] = {
            "mechanistic_top1_methods": mech_set,
            "empirical_top1_methods": emp_set,
            "status": status,
        }
    if h4_pass:
        h4 = "PASS"
    elif h4_not_estimable:
        h4 = "NOT_ESTIMABLE"
    else:
        h4 = "FAIL"

    h5_value = float(np.nanmax(mean_taus))
    h5 = "PASS" if h5_value >= stability_tau else "FAIL"

    return {
        "H1": {"verdict": h1, "min_mean_kendall_tau": h1_value, "threshold": stability_tau},
        "H2": {
            "verdict": h2,
            "status": h2_status,
            "mean_tau_low": mean_low,
            "mean_tau_high": mean_high,
            "gap_low_minus_high": gap,
            "material_gap": material_gap,
            "n_low_pairs": int(len(low)),
            "n_high_pairs": int(len(high)),
        },
        "H3": {
            "verdict": h3,
            "max_reversal_fraction": max_reversal,
            "threshold": reversal_threshold,
            "max_reversal_detail": max_reversal_detail,
        },
        "H4": {"verdict": h4, "endpoints": h4_endpoints},
        "H5": {"verdict": h5, "max_mean_kendall_tau": h5_value, "threshold": stability_tau},
    }


def key_numbers(
    endpoint_stability: pd.DataFrame,
    pairs: pd.DataFrame,
    methods: pd.DataFrame,
    factors: pd.DataFrame,
    controls: pd.DataFrame,
    verdicts: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    stability_tau = float(config["thresholds"]["stability_tau"])
    estimable_pairs = pairs[pairs["status"] == "ESTIMABLE"]
    return {
        "task_id": config["task_id"],
        "interpretation": {
            "claim_upgrade": False,
            "scope": "simulation factor-grid ranking stability; no method-superiority or biological claim",
            "tost": "NOT_APPLICABLE",
            "study_unit": "designed factor cell, not an independent study",
        },
        "inputs": {
            "simulation_rows": 4480,
            "strata": 2,
            "factor_cells_per_stratum": 64,
            "seeds_per_cell_method": 5,
            "annotation_methods": 5,
            "controls": 2,
            "endpoints": 9,
    },
        "outputs": {
            "endpoint_stability_rows": int(len(endpoint_stability)),
            "pair_reversal_rows": int(len(pairs)),
            "method_rank_rows": int(len(methods)),
            "factor_sensitivity_rows": int(len(factors)),
            "control_check_rows": int(len(controls)),
            "cell_ranking_rows": int(len(methods)),
        },
        "endpoint_stability_summary": {
            "min_mean_kendall_tau": float(endpoint_stability["mean_kendall_tau"].min()),
            "median_mean_kendall_tau": float(endpoint_stability["mean_kendall_tau"].median()),
            "max_mean_kendall_tau": float(endpoint_stability["mean_kendall_tau"].max()),
            "min_top1_stability": float(endpoint_stability["top1_stability"].min()),
            "median_top1_stability": float(endpoint_stability["top1_stability"].median()),
            "max_top1_stability": float(endpoint_stability["top1_stability"].max()),
            "n_pairs_below_stability_tau": int(
                (endpoint_stability["mean_kendall_tau"] < stability_tau).sum()
            ),
            "n_pairs_at_or_above_stability_tau": int(
                (endpoint_stability["mean_kendall_tau"] >= stability_tau).sum()
            ),
            "dynamic_range_low_pairs": int((endpoint_stability["dynamic_range_class"] == "low").sum()),
            "dynamic_range_high_pairs": int((endpoint_stability["dynamic_range_class"] == "high").sum()),
        },
        "pair_reversal_summary": {
            "max_reversal_fraction": float(estimable_pairs["reversal_fraction"].max()),
            "max_reversal_fraction_estimable": float(estimable_pairs["reversal_fraction"].max()),
            "max_reversal_fraction_all_rows_including_overall_ties": float(
                pairs["reversal_fraction"].max()
            ),
            "n_pairs_estimable": int(len(estimable_pairs)),
            "n_pairs_not_estimable_overall_tie": int(len(pairs) - len(estimable_pairs)),
            "n_pairs_reversal_ge_20pct": int(pairs["reversal_ge_20pct"].sum()),
        },
        "control_check_summary": {
            "min_fraction_both_controls_below": float(controls["fraction_both_controls_below"].min()),
            "median_fraction_both_controls_below": float(controls["fraction_both_controls_below"].median()),
            "max_fraction_both_controls_below": float(controls["fraction_both_controls_below"].max()),
        },
        "hypotheses": verdicts,
    }


def output_files(attempt_root: Path) -> list[Path]:
    return sorted(
        path
        for path in attempt_root.rglob("*")
        if path.is_file()
        and path.name not in {"output_checksums.tsv", "checksum_verification.json"}
    )


def write_checksum_table(attempt_root: Path) -> Path:
    path = attempt_root / "manifests" / "output_checksums.tsv"
    rows = [
        {
            "path": str(item.relative_to(attempt_root)),
            "sha256": sha256_file(item),
            "bytes": int(item.stat().st_size),
        }
        for item in output_files(attempt_root)
    ]
    write_tsv(pd.DataFrame(rows), path)
    return path


def verify_attempt(attempt_root: Path, write: bool = True) -> dict[str, Any]:
    checksum_path = attempt_root / "manifests" / "output_checksums.tsv"
    frame = pd.read_csv(checksum_path, sep="\t")
    checks: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        path = attempt_root / str(row["path"])
        actual = sha256_file(path)
        checks.append(
            {
                "path": str(row["path"]),
                "expected_sha256": str(row["sha256"]),
                "actual_sha256": actual,
                "match": actual == str(row["sha256"]),
            }
        )
    expected_paths = {str(row["path"]) for _, row in frame.iterrows()}
    actual_paths = {str(path.relative_to(attempt_root)) for path in output_files(attempt_root)}
    result = {
        "attempt_root": str(attempt_root),
        "verified_utc": utc_now(),
        "n_files_listed": int(len(frame)),
        "n_files_verified": int(len(checks)),
        "all_hashes_match": bool(all(item["match"] for item in checks)),
        "coverage_all_non_terminal_files": bool(expected_paths == actual_paths),
        "missing_files": sorted(expected_paths - actual_paths),
        "unlisted_files": sorted(actual_paths - expected_paths),
        "files": checks,
        "terminal_note": "checksum_verification.json is written after output_checksums.tsv and is not included in that table",
    }
    if write:
        write_json(result, attempt_root / "manifests" / "checksum_verification.json")
    return result


def self_check(
    config: dict[str, Any],
    root: Path,
    attempt_root: Path,
    freeze_checks: list[dict[str, Any]],
    metrics: pd.DataFrame,
    methods: pd.DataFrame,
    cells: pd.DataFrame,
    endpoint_stability: pd.DataFrame,
    pairs: pd.DataFrame,
    method_summary: pd.DataFrame,
    factor_sensitivity: pd.DataFrame,
    controls: pd.DataFrame,
    verdicts: dict[str, Any],
) -> dict[str, Any]:
    freeze_manifest = json.loads(
        resolve_project_path(root, config["protocol"]["freeze_manifest"]).read_text(encoding="utf-8")
    )
    checks = {
        "freeze_manifest_frozen_before_analysis": bool(freeze_manifest.get("frozen_before_analysis") is True),
        "freeze_hashes_match": all(item["match"] for item in freeze_checks),
        "input_rows_4480": len(metrics) == 4480,
        "strata_2": metrics["simulation_type"].nunique() == 2,
        "factor_cells_64": metrics["factor_cell"].nunique() == 64,
        "annotation_methods_5": methods["method_id"].nunique() == 5,
        "endpoints_9": methods["endpoint"].nunique() == 9,
        "cell_table_rows_5760": len(methods) == 5760,
        "cell_summary_rows_1152": len(cells) == 1152,
        "endpoint_stability_rows_18": len(endpoint_stability) == 18,
        "pair_reversal_rows_180": len(pairs) == 180,
        "method_summary_rows_90": len(method_summary) == 90,
        "factor_sensitivity_rows_216": len(factor_sensitivity) == 216,
        "control_check_rows_18": len(controls) == 18,
        "cell_ranks_in_1_to_5": bool(methods["cell_rank"].between(1, 5).all()),
        "overall_ranks_in_1_to_5": bool(methods["overall_rank"].between(1, 5).all()),
        "tau_in_minus1_to_1_or_nan": bool(
            ((methods["kendall_tau"].dropna() >= -1 - 1e-9) & (methods["kendall_tau"].dropna() <= 1 + 1e-9)).all()
        ),
        "no_control_in_primary_cell_table": bool(
            not set(methods["method_id"]).intersection(set(config["methods"]["controls"]))
        ),
        "hypothesis_verdicts_present": set(verdicts) == {"H1", "H2", "H3", "H4", "H5"},
        "outputs_contained": all(
            str(path.resolve()).startswith(str(attempt_root.resolve()) + os.sep)
            for path in output_files(attempt_root)
        ),
    }
    payload = {
        "task_id": config["task_id"],
        "checked_utc": utc_now(),
        "n_checks": int(len(checks)),
        "n_passed": int(sum(bool(value) for value in checks.values())),
        "all_passed": bool(all(bool(value) for value in checks.values())),
        "checks": checks,
    }
    write_json(payload, attempt_root / "manifests" / "self_check.json")
    if not payload["all_passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"Self-check failed: {failed}")
    return payload


def ensure_attempt_root(root: Path, config: dict[str, Any]) -> Path:
    base = resolve_project_path(root, config["output"]["root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = base / "attempts" / f"run_{stamp}"
    suffix = 1
    while attempt.exists():
        attempt = base / "attempts" / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def configure_logging(attempt_root: Path) -> logging.Handler:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(attempt_root / "logs" / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)
    return file_handler


def run(config_path: Path, root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    started_utc = utc_now()
    config = load_config(config_path)
    attempt_root = ensure_attempt_root(root, config)
    (attempt_root / "logs").mkdir(parents=True, exist_ok=True)
    (attempt_root / "manifests").mkdir(parents=True, exist_ok=True)
    (attempt_root / "tables").mkdir(parents=True, exist_ok=True)
    file_handler = configure_logging(attempt_root)
    try:
        LOGGER.info("NM-RANK1 start %s", started_utc)
        freeze_checks = verify_freeze(config, root)
        LOGGER.info("Freeze hashes verified: %s", len(freeze_checks))
        inputs = load_inputs(config, root)
        validate_inputs(config, inputs)
        LOGGER.info("Inputs validated")
        metrics = inputs["metrics"]
        long_cells = build_long_cells(metrics, config)
        overall = build_overall_scores(long_cells, config)
        methods, cells = build_cell_tables(long_cells, overall, config)
        pairs = build_pair_reversals(long_cells, overall, config)
        endpoint_stability = build_endpoint_stability(
            cells, overall, pairs, inputs["summary"], config
        )
        method_summary = build_method_summary(methods, overall)
        factor_sensitivity = build_factor_sensitivity(cells, config)
        controls = build_control_check(long_cells, metrics, config)
        verdicts = hypothesis_verdicts(endpoint_stability, pairs, config)
        LOGGER.info("Hypotheses: %s", json.dumps(verdicts, sort_keys=True))

        key = key_numbers(
            endpoint_stability, pairs, methods, factor_sensitivity, controls, verdicts, config
        )
        with gzip.open(attempt_root / "tables" / "rank1_cell_rankings.tsv.gz", "wt", encoding="utf-8") as handle:
            methods.to_csv(handle, sep="\t", index=False)
        write_tsv(endpoint_stability, attempt_root / "tables" / "rank1_endpoint_stability.tsv")
        write_tsv(pairs, attempt_root / "tables" / "rank1_pair_reversals.tsv")
        write_tsv(method_summary, attempt_root / "tables" / "rank1_method_rank_summary.tsv")
        write_tsv(factor_sensitivity, attempt_root / "tables" / "rank1_factor_sensitivity.tsv")
        write_tsv(controls, attempt_root / "tables" / "rank1_control_check.tsv")
        write_json(key, attempt_root / "key_numbers.json")
        write_tsv(pd.DataFrame(freeze_checks), attempt_root / "manifests" / "input_checksums.tsv")

        self_check(
            config,
            root,
            attempt_root,
            freeze_checks,
            metrics,
            methods,
            cells,
            endpoint_stability,
            pairs,
            method_summary,
            factor_sensitivity,
            controls,
            verdicts,
        )
        runtime = time.perf_counter() - started
        peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        manifest = {
            "task_id": config["task_id"],
            "status": "complete",
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "runtime_seconds": float(runtime),
            "peak_rss_kb": peak_rss_kb,
            "command": " ".join(sys.argv),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "preregistration_path": str(config["protocol"]["path"]),
            "preregistration_sha256": json.loads(
                resolve_project_path(root, config["protocol"]["freeze_manifest"]).read_text(encoding="utf-8")
            )["preregistration"]["sha256"],
            "freeze_manifest_path": str(config["protocol"]["freeze_manifest"]),
            "freeze_manifest_sha256": sha256_file(
                resolve_project_path(root, config["protocol"]["freeze_manifest"])
            ),
            "seed": int(config["seeds"]["bootstrap"]),
            "software_versions": software_versions(),
            "outputs": {
                "cell_rankings": "tables/rank1_cell_rankings.tsv.gz",
                "endpoint_stability": "tables/rank1_endpoint_stability.tsv",
                "pair_reversals": "tables/rank1_pair_reversals.tsv",
                "method_rank_summary": "tables/rank1_method_rank_summary.tsv",
                "factor_sensitivity": "tables/rank1_factor_sensitivity.tsv",
                "control_check": "tables/rank1_control_check.tsv",
                "key_numbers": "key_numbers.json",
            },
        }
        write_json(manifest, attempt_root / "manifests" / "run_manifest.json")
        LOGGER.info("Run complete: %.3f s; peak RSS %s KB", runtime, peak_rss_kb)
    finally:
        file_handler.close()
        LOGGER.removeHandler(file_handler)
    checksum_path = write_checksum_table(attempt_root)
    verification = verify_attempt(attempt_root, write=True)
    LOGGER.info("Checksums verified: %s", verification["all_hashes_match"])
    return {
        "attempt_root": str(attempt_root),
        "checksum_table": str(checksum_path),
        "n_files": verification["n_files_verified"],
        "all_hashes_match": verification["all_hashes_match"],
        "coverage_all_non_terminal_files": verification["coverage_all_non_terminal_files"],
        "hypotheses": verdicts,
    }
