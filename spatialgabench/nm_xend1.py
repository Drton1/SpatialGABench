"""NM-XEND1 cross-endpoint ranking-agreement audit.

The module reads only frozen evaluation tables. It never constructs a
representation, executes a method, or touches RNA expression values.
"""

from __future__ import annotations

import gzip
import itertools
import json
import logging
import math
import os
import platform
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

from spatialgabench.nm_rank1 import (
    bootstrap_mean_ci,
    ensure_attempt_root,
    load_config,
    output_files,
    resolve_project_path,
    sha256_file,
    stable_seed,
    utc_now,
    verify_attempt,
    write_checksum_table,
    write_json,
    write_tsv,
)


LOGGER = logging.getLogger("nm_xend1")
SIMULATION_RE = r"__r(?P<replicate>\d+)__c(?P<factor_cell>\d+)$"


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
    }


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def verify_freeze(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_project_path(root, config["protocol"]["freeze_manifest"])
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    for role in ("config", "preregistration"):
        item = freeze[role]
        path = resolve_project_path(root, str(item["path"]))
        actual = sha256_file(path)
        checks.append(
            {
                "path": str(path.relative_to(root)),
                "expected_sha256": str(item["sha256"]),
                "actual_sha256": actual,
                "match": actual == str(item["sha256"]),
                "role": role,
            }
        )
    for item in freeze["inputs"]:
        path = resolve_project_path(root, str(item["path"]))
        actual = sha256_file(path)
        checks.append(
            {
                "path": str(path.relative_to(root)),
                "expected_sha256": str(item["sha256"]),
                "actual_sha256": actual,
                "match": actual == str(item["sha256"]),
                "role": "analysis_input",
            }
        )
    if not _freeze_chronology_reviewable(freeze):
        raise ValueError("Freeze manifest chronology is not reviewable")
    failed = [item for item in checks if not item["match"]]
    if failed:
        raise ValueError(f"Freeze hash mismatch: {failed}")
    return checks


def _freeze_chronology_reviewable(freeze: dict[str, Any]) -> bool:
    """Accept either a pre-analysis freeze or an explicitly reconstructed one."""
    if freeze.get("frozen_before_analysis") is True:
        return True
    if freeze.get("freeze_chronology_status") != (
        "REPAIRED_BY_DISCLOSURE_AND_RECONSTRUCTION"
    ):
        return False
    first_attempt = freeze.get("first_outcome_attempt", {})
    original = freeze.get("original_freeze_record", {})
    superseded = freeze.get("superseded_rewritten_record", {})
    return bool(
        first_attempt.get("attempt_id")
        and first_attempt.get("started_utc")
        and first_attempt.get("completed_utc")
        and original.get("sha256")
        and superseded.get("sha256")
        and freeze.get("scientific_inputs_hash_equivalence") is True
    )


def load_inputs(config: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = {
        name: resolve_project_path(root, value)
        for name, value in config["inputs"].items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "simulation": pd.read_csv(paths["sat1_simulation_cell_audit"], sep="\t"),
        "ds1_domain": pd.read_csv(paths["ds1_domain_seed_metrics"], sep="\t"),
        "ds1_marker": pd.read_csv(paths["ds1_marker_summary"], sep="\t"),
        "ds1_svg": pd.read_csv(paths["ds1_svg_summary"], sep="\t"),
        "paths": paths,
    }


def validate_inputs(inputs: dict[str, Any], config: dict[str, Any]) -> None:
    expected = config["expected"]
    simulation = inputs["simulation"]
    required = {
        "simulation_type",
        "simulation_id",
        "method_id",
        "endpoint",
        "value",
    }
    if not required.issubset(simulation.columns):
        raise ValueError(
            f"SAT1 table lacks columns {sorted(required - set(simulation.columns))}"
        )
    if len(simulation) != int(expected["sat1_rows_total"]):
        raise ValueError(
            f"SAT1 table has {len(simulation)} rows, expected {expected['sat1_rows_total']}"
        )
    strata = list(config["simulation"]["strata"])
    methods = list(config["simulation"]["methods"]["annotation"])
    controls = list(config["simulation"]["methods"]["controls"])
    endpoints = list(config["simulation"]["endpoints"])
    if sorted(simulation["simulation_type"].astype(str).unique()) != sorted(strata):
        raise ValueError("SAT1 strata do not match the config")
    if sorted(simulation["method_id"].astype(str).unique()) != sorted(methods + controls):
        raise ValueError("SAT1 methods do not match the config")
    if not set(endpoints).issubset(set(simulation["endpoint"].astype(str).unique())):
        raise ValueError("SAT1 endpoints do not match the config")

    parsed = simulation["simulation_id"].astype(str).str.extract(SIMULATION_RE)
    if parsed.isna().any().any():
        raise ValueError("simulation_id does not end in __rNN__cNNN")
    simulation = simulation.copy()
    simulation["replicate_index"] = parsed["replicate"].astype(int)
    simulation["factor_cell"] = parsed["factor_cell"].astype(int)
    selected = simulation[
        simulation["simulation_type"].astype(str).isin(strata)
        & simulation["method_id"].astype(str).isin(methods)
        & simulation["endpoint"].astype(str).isin(endpoints)
    ]
    if len(selected) != int(expected["sat1_selected_rows"]):
        raise ValueError(
            f"SAT1 selected rows {len(selected)}, expected {expected['sat1_selected_rows']}"
        )
    if selected["factor_cell"].nunique() != int(expected["factor_cells_per_stratum"]):
        raise ValueError("SAT1 factor-cell count does not match the config")
    for stratum in strata:
        stratum_rows = selected[selected["simulation_type"].astype(str) == stratum]
        if stratum_rows["factor_cell"].nunique() != int(expected["factor_cells_per_stratum"]):
            raise ValueError(f"SAT1 {stratum} does not contain every factor cell")
        counts = stratum_rows.groupby(["factor_cell", "method_id", "endpoint"]).size()
        if not bool((counts == int(expected["replicates_per_cell"])).all()):
            raise ValueError(f"SAT1 {stratum} does not contain five replicates per cell/method/endpoint")
        if not np.isfinite(stratum_rows["value"].to_numpy(dtype=float)).all():
            raise ValueError(f"SAT1 {stratum} contains non-finite endpoint values")

    domain = inputs["ds1_domain"]
    expected_domain_rows = int(expected.get("ds1_domain_rows", 85))
    if len(domain) != expected_domain_rows:
        raise ValueError(
            f"DS1 domain table has {len(domain)} rows, expected {expected_domain_rows}"
        )
    if not {"dataset_key", "depth_fraction", "representation", "clustering_seed", "ARI", "NMI"}.issubset(domain.columns):
        raise ValueError("DS1 domain table lacks required columns")
    marker = inputs["ds1_marker"]
    expected_marker_rows = int(expected.get("ds1_marker_rows", 15))
    if len(marker) != expected_marker_rows:
        raise ValueError(
            f"DS1 marker table has {len(marker)} rows, expected {expected_marker_rows}"
        )
    if not {"dataset_key", "depth_fraction", "representation", "mean_precision", "mean_recall"}.issubset(marker.columns):
        raise ValueError("DS1 marker table lacks required columns")
    svg = inputs["ds1_svg"]
    expected_svg_rows = int(expected.get("ds1_svg_rows", 15))
    if len(svg) != expected_svg_rows:
        raise ValueError(
            f"DS1 SVG table has {len(svg)} rows, expected {expected_svg_rows}"
        )
    if not {"dataset_key", "depth_fraction", "representation", "spearman_to_rna_moran_i", "top_jaccard"}.issubset(svg.columns):
        raise ValueError("DS1 SVG table lacks required columns")


def _normalise_real_method(value: Any, config: dict[str, Any]) -> str:
    text = str(value)
    return str(config["real"]["method_aliases"].get(text, text))


def _real_contexts(config: dict[str, Any]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for role in ("primary", "sensitivity"):
        for context in config["real"]["contexts"][role]:
            contexts.append({**context, "role": role})
    return contexts


def build_real_unit_scores(inputs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    methods = list(config["real"]["methods"])
    rows: list[dict[str, Any]] = []
    for context in _real_contexts(config):
        dataset = str(context["dataset_key"])
        depth = float(context["depth_fraction"])
        role = str(context["role"])
        unit_id = f"real__{dataset}__d{depth:.2f}"
        domain = inputs["ds1_domain"]
        selected_domain = domain[
            (domain["dataset_key"].astype(str) == dataset)
            & np.isclose(domain["depth_fraction"].astype(float), depth)
        ].copy()
        selected_domain["method_id"] = selected_domain["representation"].map(
            lambda value: _normalise_real_method(value, config)
        )
        selected_domain = selected_domain[selected_domain["method_id"].isin(methods)]
        if len(selected_domain) != len(methods) * 5:
            raise ValueError(f"Real domain rows missing for {unit_id}")
        if selected_domain.groupby("method_id")["clustering_seed"].nunique().min() != 5:
            raise ValueError(f"Real domain seeds missing for {unit_id}")
        for method in methods:
            method_rows = selected_domain[selected_domain["method_id"] == method]
            if len(method_rows) != 5:
                raise ValueError(f"Real domain does not have five seeds for {unit_id}/{method}")
            for endpoint, column in (("domain_ari", "ARI"), ("domain_nmi", "NMI")):
                rows.append(
                    {
                        "stratum": f"real_{role}",
                        "unit_id": unit_id,
                        "unit_role": role,
                        "method_id": method,
                        "endpoint": endpoint,
                        "value": float(method_rows[column].median()),
                        "n_source_rows": int(len(method_rows)),
                    }
                )

        marker = inputs["ds1_marker"]
        selected_marker = marker[
            (marker["dataset_key"].astype(str) == dataset)
            & np.isclose(marker["depth_fraction"].astype(float), depth)
        ].copy()
        selected_marker["method_id"] = selected_marker["representation"].map(
            lambda value: _normalise_real_method(value, config)
        )
        selected_marker = selected_marker[selected_marker["method_id"].isin(methods)]
        if len(selected_marker) != len(methods):
            raise ValueError(f"Real marker rows missing for {unit_id}")
        for method in methods:
            method_rows = selected_marker[selected_marker["method_id"] == method]
            if len(method_rows) != 1:
                raise ValueError(f"Real marker row is not unique for {unit_id}/{method}")
            for endpoint, column in (
                ("marker_mean_precision", "mean_precision"),
                ("marker_mean_recall", "mean_recall"),
            ):
                rows.append(
                    {
                        "stratum": f"real_{role}",
                        "unit_id": unit_id,
                        "unit_role": role,
                        "method_id": method,
                        "endpoint": endpoint,
                        "value": _finite(method_rows[column].iloc[0]),
                        "n_source_rows": 1,
                    }
                )

        svg = inputs["ds1_svg"]
        selected_svg = svg[
            (svg["dataset_key"].astype(str) == dataset)
            & np.isclose(svg["depth_fraction"].astype(float), depth)
        ].copy()
        selected_svg["method_id"] = selected_svg["representation"].map(
            lambda value: _normalise_real_method(value, config)
        )
        selected_svg = selected_svg[selected_svg["method_id"].isin(methods)]
        if len(selected_svg) != len(methods):
            raise ValueError(f"Real SVG rows missing for {unit_id}")
        for method in methods:
            method_rows = selected_svg[selected_svg["method_id"] == method]
            if len(method_rows) != 1:
                raise ValueError(f"Real SVG row is not unique for {unit_id}/{method}")
            for endpoint, column in (
                ("svg_spearman", "spearman_to_rna_moran_i"),
                ("svg_jaccard", "top_jaccard"),
            ):
                rows.append(
                    {
                        "stratum": f"real_{role}",
                        "unit_id": unit_id,
                        "unit_role": role,
                        "method_id": method,
                        "endpoint": endpoint,
                        "value": _finite(method_rows[column].iloc[0]),
                        "n_source_rows": 1,
                    }
                )

    result = pd.DataFrame(rows)
    endpoints = list(config["real"]["endpoints"])
    families = config["real"]["endpoint_families"]
    result["endpoint_family"] = result["endpoint"].map(families)
    if set(result["endpoint"]) != set(endpoints):
        raise ValueError("Real unit scores do not cover the configured endpoints")
    if result["value"].isna().any():
        raise ValueError("Real unit scores contain missing values")
    return result


def build_simulation_unit_scores(inputs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    simulation = inputs["simulation"].copy()
    parsed = simulation["simulation_id"].astype(str).str.extract(SIMULATION_RE)
    simulation["replicate_index"] = parsed["replicate"].astype(int)
    simulation["factor_cell"] = parsed["factor_cell"].astype(int)
    methods = list(config["simulation"]["methods"]["annotation"])
    endpoints = list(config["simulation"]["endpoints"])
    simulation = simulation[
        simulation["method_id"].astype(str).isin(methods)
        & simulation["endpoint"].astype(str).isin(endpoints)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (stratum, cell, endpoint, method), group in simulation.groupby(
        ["simulation_type", "factor_cell", "endpoint", "method_id"], sort=True
    ):
        if len(group) != 5:
            raise ValueError(
                f"Expected five replicates for {stratum}/{cell}/{endpoint}/{method}"
            )
        rows.append(
            {
                "stratum": str(stratum),
                "unit_id": f"{stratum}__c{int(cell):03d}",
                "unit_role": "simulation",
                "method_id": str(method),
                "endpoint": str(endpoint),
                "value": float(group["value"].median()),
                "n_source_rows": int(len(group)),
            }
        )
    result = pd.DataFrame(rows)
    result["endpoint_family"] = result["endpoint"].map(
        config["simulation"]["endpoint_families"]
    )
    expected = (
        int(config["expected"]["simulation_units"])
        * int(config["expected"]["annotation_methods"])
        * int(config["expected"]["simulation_endpoints"])
    )
    if len(result) != expected:
        raise ValueError(f"Simulation unit scores have {len(result)} rows, expected {expected}")
    return result


def add_ranks(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    rows: list[pd.DataFrame] = []
    for (stratum, unit_id, endpoint), group in scores.groupby(
        ["stratum", "unit_id", "endpoint"], sort=True
    ):
        group = group.copy()
        values = group["value"].to_numpy(dtype=float)
        maximum = float(np.max(values))
        group["rank"] = rankdata(-values, method="average")
        group["top1_set"] = "|".join(
            sorted(
                str(method)
                for method, value in zip(group["method_id"], values)
                if value >= maximum - tolerance
            )
        )
        group["top1_unique"] = bool((values >= maximum - tolerance).sum() == 1)
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def _tau_b_components(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return upper-triangle pair products, upper indices, and tau-b denominator."""
    sign_x = np.sign(x[:, None] - x[None, :])
    sign_y = np.sign(y[:, None] - y[None, :])
    upper = np.triu_indices(len(x), k=1)
    sx = sign_x[upper]
    sy = sign_y[upper]
    n0 = float(len(sx))
    n1 = float(np.sum(sx == 0))
    n2 = float(np.sum(sy == 0))
    denominator = math.sqrt(max(0.0, (n0 - n1) * (n0 - n2)))
    return sx * sy, upper, denominator


def _permutation_tau_null(
    x: np.ndarray,
    y: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null taus from permuting y's method labels relative to fixed x."""
    if float(np.ptp(x)) <= 1e-12 or float(np.ptp(y)) <= 1e-12:
        return np.asarray([], dtype=float)
    rank_x = rankdata(x, method="average")
    rank_y = rankdata(y, method="average")
    product, upper, denominator = _tau_b_components(rank_x, rank_y)
    if denominator <= 1e-15:
        return np.asarray([], dtype=float)
    n_methods = len(rank_x)
    permutations = np.argsort(rng.random((int(draws), n_methods)), axis=1)
    permuted_y = rank_y[permutations]
    sign_y = np.sign(permuted_y[:, :, None] - permuted_y[:, None, :])
    sign_y_upper = sign_y[:, upper[0], upper[1]]
    taus = (sign_y_upper @ product) / denominator
    taus = taus[np.isfinite(taus)]
    return taus


def build_pair_taus(
    scores: pd.DataFrame,
    config: dict[str, Any],
    null_pool: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    perm_draws = int(config["thresholds"]["permutation_draws"])
    base_seed = int(config["seeds"]["permutation"])
    rows: list[dict[str, Any]] = []
    for (stratum, unit_id), unit in scores.groupby(["stratum", "unit_id"], sort=True):
        endpoints = list(dict.fromkeys(unit["endpoint"].tolist()))
        pivot = unit.pivot(index="endpoint", columns="method_id", values="value")
        top1 = unit.drop_duplicates("endpoint").set_index("endpoint")["top1_set"]
        unique_top1 = unit.drop_duplicates("endpoint").set_index("endpoint")["top1_unique"]
        for endpoint_a, endpoint_b in itertools.combinations(endpoints, 2):
            family_a = str(unit.loc[unit["endpoint"] == endpoint_a, "endpoint_family"].iloc[0])
            family_b = str(unit.loc[unit["endpoint"] == endpoint_b, "endpoint_family"].iloc[0])
            pair_type = "within_family" if family_a == family_b else "cross_family"
            values_a = pivot.loc[endpoint_a].to_numpy(dtype=float)
            values_b = pivot.loc[endpoint_b].to_numpy(dtype=float)
            if float(np.ptp(values_a)) <= tolerance or float(np.ptp(values_b)) <= tolerance:
                tau = float("nan")
                status = "NOT_ESTIMABLE_TIED"
            else:
                tau = float(kendalltau(values_a, values_b, variant="b").statistic)
                status = "ESTIMABLE" if math.isfinite(tau) else "NOT_ESTIMABLE"
            both_unique = bool(unique_top1.loc[endpoint_a] and unique_top1.loc[endpoint_b])
            top1_match = (
                str(top1.loc[endpoint_a]) == str(top1.loc[endpoint_b])
                if both_unique
                else pd.NA
            )
            null_values: np.ndarray = np.asarray([], dtype=float)
            if pair_type == "cross_family" and status == "ESTIMABLE":
                rng = np.random.default_rng(
                    stable_seed(base_seed, f"{stratum}::{unit_id}::{endpoint_a}::{endpoint_b}")
                )
                null_values = _permutation_tau_null(values_a, values_b, perm_draws, rng)
                if (
                    null_pool is not None
                    and stratum in set(config["simulation"]["strata"])
                    and len(null_values)
                ):
                    null_pool.append(null_values)
            rows.append(
                {
                    "stratum": stratum,
                    "unit_id": unit_id,
                    "endpoint_a": endpoint_a,
                    "endpoint_b": endpoint_b,
                    "family_a": family_a,
                    "family_b": family_b,
                    "pair_type": pair_type,
                    "tau_b": tau,
                    "status": status,
                    "top1_a": str(top1.loc[endpoint_a]),
                    "top1_b": str(top1.loc[endpoint_b]),
                    "top1_both_unique": both_unique,
                    "top1_match": top1_match,
                    "null_n": int(len(null_values)),
                    "null_q95": float(np.percentile(null_values, 95)) if len(null_values) else float("nan"),
                    "null_q99": float(np.percentile(null_values, 99)) if len(null_values) else float("nan"),
                }
            )
    result = pd.DataFrame(rows)
    expected = int(config["expected"]["simulation_pair_tau_rows"]) + int(
        config["expected"]["real_pair_tau_rows"]
    )
    if len(result) != expected:
        raise ValueError(f"Pair-tau table has {len(result)} rows, expected {expected}")
    return result


def build_unit_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stratum, unit_id), group in pairs.groupby(["stratum", "unit_id"], sort=True):
        within = group[(group["pair_type"] == "within_family") & (group["status"] == "ESTIMABLE")]
        cross = group[(group["pair_type"] == "cross_family") & (group["status"] == "ESTIMABLE")]
        cross_all = group[group["pair_type"] == "cross_family"]
        estimable_top1 = cross_all[cross_all["top1_both_unique"].astype(bool)]
        disagreement = float(
            (~estimable_top1["top1_match"].astype(bool)).sum() / len(estimable_top1)
        ) if len(estimable_top1) else float("nan")
        all_valid = group[group["status"] == "ESTIMABLE"]
        rows.append(
            {
                "stratum": stratum,
                "unit_id": unit_id,
                "within_tau_unit": float(within["tau_b"].mean()) if len(within) else float("nan"),
                "cross_tau_unit": float(cross["tau_b"].mean()) if len(cross) else float("nan"),
                "combined_tau_unit": float(all_valid["tau_b"].mean()) if len(all_valid) else float("nan"),
                "n_within_pairs": int(len(within)),
                "n_cross_pairs": int(len(cross)),
                "n_cross_pairs_all": int(len(cross_all)),
                "n_cross_top1_estimable": int(len(estimable_top1)),
                "n_cross_top1_disagreement": int((~estimable_top1["top1_match"].astype(bool)).sum()),
                "top1_disagreement_fraction": disagreement,
                "n_tied_endpoint_pairs": int((group["status"] != "ESTIMABLE").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_endpoint_pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in pairs.groupby(
        [
            "stratum",
            "endpoint_a",
            "endpoint_b",
            "family_a",
            "family_b",
            "pair_type",
        ],
        sort=True,
    ):
        stratum, endpoint_a, endpoint_b, family_a, family_b, pair_type = keys
        estimable = group[group["status"] == "ESTIMABLE"]
        top1 = group[group["top1_both_unique"].astype(bool)]
        rows.append(
            {
                "stratum": stratum,
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "family_a": family_a,
                "family_b": family_b,
                "pair_type": pair_type,
                "n_units": int(len(group)),
                "n_estimable": int(len(estimable)),
                "median_tau_b": float(estimable["tau_b"].median()) if len(estimable) else float("nan"),
                "q25_tau_b": float(estimable["tau_b"].quantile(0.25)) if len(estimable) else float("nan"),
                "q75_tau_b": float(estimable["tau_b"].quantile(0.75)) if len(estimable) else float("nan"),
                "n_top1_estimable": int(len(top1)),
                "n_top1_match": int(top1["top1_match"].astype(bool).sum()),
                "top1_match_fraction": float(top1["top1_match"].astype(bool).mean()) if len(top1) else float("nan"),
                "n_top1_disagreement": int((~top1["top1_match"].astype(bool)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_median_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def build_aggregate(unit_summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    draws = int(config["thresholds"]["bootstrap_draws"])
    base_seed = int(config["seeds"]["bootstrap"])
    strata = list(config["simulation"]["strata"])
    labels = strata + ["simulation_pooled", "real_primary", "real_sensitivity"]
    rows: list[dict[str, Any]] = []
    for label in labels:
        if label == "simulation_pooled":
            selected = unit_summary[unit_summary["stratum"].isin(strata)]
        elif label == "real_primary":
            selected = unit_summary[unit_summary["stratum"] == "real_primary"]
        elif label == "real_sensitivity":
            selected = unit_summary[unit_summary["stratum"] == "real_sensitivity"]
        else:
            selected = unit_summary[unit_summary["stratum"] == label]
        for scope, column in (
            ("within_family", "within_tau_unit"),
            ("cross_family", "cross_tau_unit"),
            ("combined", "combined_tau_unit"),
        ):
            values = selected[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                median = float(np.median(values))
                ci_low, ci_high = _bootstrap_median_ci(
                    values, draws, stable_seed(base_seed, f"{label}::{scope}")
                )
            else:
                median = ci_low = ci_high = float("nan")
            rows.append(
                {
                    "stratum": label,
                    "scope": scope,
                    "n_units": int(len(selected)),
                    "n_estimable_units": int(len(values)),
                    "median_tau": median,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "analysis_role": "primary_simulation"
                    if label in strata or label == "simulation_pooled"
                    else "descriptive_real",
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != int(config["expected"]["aggregate_rows"]):
        raise ValueError(f"Aggregate table has {len(result)} rows, expected {config['expected']['aggregate_rows']}")
    return result


def build_top1_transfer(pair_summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = pair_summary[
        [
            "stratum",
            "endpoint_a",
            "endpoint_b",
            "family_a",
            "family_b",
            "pair_type",
            "n_units",
            "n_top1_estimable",
            "n_top1_match",
            "n_top1_disagreement",
            "top1_match_fraction",
        ]
    ].copy()
    result["chance_match_fraction"] = result["stratum"].map(
        {
            "mechanistic": 0.20,
            "empirical_semisynthetic": 0.20,
            "real_primary": 1.0 / 3.0,
            "real_sensitivity": 1.0 / 3.0,
        }
    )
    expected = int(config["expected"]["simulation_top1_rows"]) + int(
        config["expected"]["real_top1_rows"]
    )
    if len(result) != expected:
        raise ValueError(f"Top-1 transfer table has {len(result)} rows, expected {expected}")
    return result


def build_null_calibration(
    pairs: pd.DataFrame,
    unit_summary: pd.DataFrame,
    config: dict[str, Any],
    pooled_null: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stratum, endpoint_a, endpoint_b), group in pairs.groupby(
        ["stratum", "endpoint_a", "endpoint_b"], sort=True
    ):
        if group["pair_type"].iloc[0] != "cross_family":
            continue
        primary_simulation = stratum in set(config["simulation"]["strata"])
        observed_values = group["tau_b"].to_numpy(dtype=float)
        observed_values = observed_values[np.isfinite(observed_values)]
        observed_median = float(np.median(observed_values)) if len(observed_values) else float("nan")
        null_q95 = float(np.nanmedian(group["null_q95"].to_numpy(dtype=float))) if group["null_n"].max() > 0 else float("nan")
        null_q99 = float(np.nanmedian(group["null_q99"].to_numpy(dtype=float))) if group["null_n"].max() > 0 else float("nan")
        rows.append(
            {
                "stratum": stratum,
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "n_units": int(len(group)),
                "n_estimable_units": int(len(observed_values)),
                "observed_median_tau": observed_median,
                "null_q95_median": null_q95,
                "null_q99_median": null_q99,
                "null_draws_per_unit": int(group["null_n"].max()),
                "null_aggregation": "per_unit_endpoint_pair",
                "pooled_null_q95": float(pooled_null.get("null_q95_pooled", float("nan")))
                if pooled_null and primary_simulation
                else float("nan"),
                "pooled_null_draws": int(pooled_null.get("n_null_draws_pooled", 0))
                if pooled_null and primary_simulation
                else 0,
                "analysis_role": "primary_simulation"
                if stratum in config["simulation"]["strata"]
                else "descriptive_real",
            }
        )
    result = pd.DataFrame(rows)
    expected = int(config["expected"]["simulation_null_rows"]) + int(
        config["expected"]["real_null_rows"]
    )
    if len(result) != expected:
        raise ValueError(f"Null-calibration table has {len(result)} rows, expected {expected}")
    return result


def summarize_pooled_null(null_pool: list[np.ndarray]) -> dict[str, Any]:
    """Return the preregistered pooled null statistic over unit/pair cells."""
    nonempty = [np.asarray(values, dtype=float) for values in null_pool if len(values)]
    if not nonempty:
        return {
            "null_q95_pooled": float("nan"),
            "n_null_draws_pooled": 0,
            "n_null_cells_pooled": 0,
            "null_aggregation": "pooled_across_simulation_units_and_endpoint_pairs",
        }
    pooled = np.concatenate(nonempty)
    pooled = pooled[np.isfinite(pooled)]
    if not len(pooled):
        q95 = float("nan")
    else:
        q95 = float(np.percentile(pooled, 95))
    return {
        "null_q95_pooled": q95,
        "n_null_draws_pooled": int(len(pooled)),
        "n_null_cells_pooled": int(len(nonempty)),
        "null_aggregation": "pooled_across_simulation_units_and_endpoint_pairs",
    }


def hypothesis_verdicts(
    aggregate: pd.DataFrame,
    unit_summary: pd.DataFrame,
    pairs: pd.DataFrame,
    config: dict[str, Any],
    null_pool: list[np.ndarray],
) -> dict[str, Any]:
    stability = float(config["thresholds"]["stability_tau"])
    gap_threshold = float(config["thresholds"]["material_tau_gap"])
    top1_threshold = float(config["thresholds"]["top1_disagreement_fraction"])
    min_top1_fraction = float(config["thresholds"]["min_estimable_top1_fraction"])

    pooled = aggregate[
        (aggregate["stratum"] == "simulation_pooled")
    ].set_index("scope")
    within = float(pooled.loc["within_family", "median_tau"])
    cross = float(pooled.loc["cross_family", "median_tau"])

    if math.isfinite(within):
        h1 = "PASS" if within >= stability else "FAIL"
    else:
        h1 = "NOT_ESTIMABLE"
    if math.isfinite(cross):
        h2 = "PASS" if cross < stability else "FAIL"
    else:
        h2 = "NOT_ESTIMABLE"
    gap = within - cross if math.isfinite(within) and math.isfinite(cross) else float("nan")
    h3 = "PASS" if math.isfinite(gap) and gap >= gap_threshold else (
        "FAIL" if math.isfinite(gap) else "NOT_ESTIMABLE"
    )

    sim_units = unit_summary[
        unit_summary["stratum"].isin(config["simulation"]["strata"])
    ]
    cross_pairs = pairs[
        pairs["stratum"].isin(config["simulation"]["strata"])
        & (pairs["pair_type"] == "cross_family")
    ]
    top1_estimable = cross_pairs[cross_pairs["top1_both_unique"].astype(bool)]
    top1_fraction = float(len(top1_estimable) / len(cross_pairs)) if len(cross_pairs) else 0.0
    if top1_fraction >= min_top1_fraction and len(top1_estimable):
        disagreement = float((~top1_estimable["top1_match"].astype(bool)).mean())
        h4 = "PASS" if disagreement >= top1_threshold else "FAIL"
    else:
        disagreement = float("nan")
        h4 = "NOT_ESTIMABLE"

    null_rows = pairs[
        pairs["stratum"].isin(config["simulation"]["strata"])
        & (pairs["pair_type"] == "cross_family")
        & (pairs["null_n"] > 0)
    ]
    pooled_null = summarize_pooled_null(null_pool)
    null_q95 = float(pooled_null["null_q95_pooled"])
    null_q95_median = (
        float(np.nanmedian(null_rows["null_q95"].to_numpy(dtype=float)))
        if len(null_rows)
        else float("nan")
    )
    if math.isfinite(cross) and math.isfinite(null_q95):
        h5 = "PASS" if cross > null_q95 else "FAIL"
    else:
        h5 = "NOT_ESTIMABLE"

    return {
        "H1": {
            "verdict": h1,
            "statement": "within-family endpoint rankings are stable",
            "metric": "simulation pooled median within-family Kendall tau-b",
            "value": within,
            "threshold": stability,
        },
        "H2": {
            "verdict": h2,
            "statement": "cross-family endpoint rankings are not stable",
            "metric": "simulation pooled median cross-family Kendall tau-b",
            "value": cross,
            "threshold": stability,
        },
        "H3": {
            "verdict": h3,
            "statement": "within-family agreement exceeds cross-family agreement",
            "metric": "within minus cross median tau-b",
            "value": gap,
            "threshold": gap_threshold,
        },
        "H4": {
            "verdict": h4,
            "statement": "the unique top-1 method depends on endpoint family",
            "metric": "cross-family top-1 disagreement fraction",
            "value": disagreement,
            "threshold": top1_threshold,
            "estimable_pair_fraction": top1_fraction,
        },
        "H5": {
            "verdict": h5,
            "statement": "cross-family agreement exceeds the matched permutation null",
            "metric": "observed median tau-b vs pooled null q95",
            "value": cross,
            "null_q95_pooled": null_q95,
            "null_q95_median_of_cell_q95": null_q95_median,
            "n_null_cells_pooled": int(pooled_null["n_null_cells_pooled"]),
            "n_null_draws_pooled": int(pooled_null["n_null_draws_pooled"]),
            "null_aggregation": str(pooled_null["null_aggregation"]),
        },
    }


def key_numbers(
    unit_scores: pd.DataFrame,
    pairs: pd.DataFrame,
    unit_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    pair_summary: pd.DataFrame,
    nulls: pd.DataFrame,
    verdicts: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    pooled = aggregate[aggregate["stratum"] == "simulation_pooled"].set_index("scope")
    cross_pairs = pairs[
        pairs["stratum"].isin(config["simulation"]["strata"])
        & (pairs["pair_type"] == "cross_family")
    ]
    top1 = cross_pairs[cross_pairs["top1_both_unique"].astype(bool)]
    real = aggregate[aggregate["stratum"] == "real_primary"].set_index("scope")
    return {
        "task_id": config["task_id"],
        "analysis_role": "cross-endpoint ranking agreement; descriptive audit",
        "claim_upgrade": False,
        "simulation": {
            "n_units": int(unit_summary["stratum"].isin(config["simulation"]["strata"]).sum()),
            "n_strata": len(config["simulation"]["strata"]),
            "n_methods": int(unit_scores[
                unit_scores["stratum"].isin(config["simulation"]["strata"])
            ]["method_id"].nunique()),
            "n_endpoints": int(unit_scores[
                unit_scores["stratum"].isin(config["simulation"]["strata"])
            ]["endpoint"].nunique()),
            "within_median_tau": float(pooled.loc["within_family", "median_tau"]),
            "cross_median_tau": float(pooled.loc["cross_family", "median_tau"]),
            "combined_median_tau": float(pooled.loc["combined", "median_tau"]),
            "n_cross_endpoint_pairs": int(len(cross_pairs)),
            "n_top1_estimable_pairs": int(len(top1)),
            "n_top1_disagreement_pairs": int((~top1["top1_match"].astype(bool)).sum()),
            "top1_disagreement_fraction": float(
                (~top1["top1_match"].astype(bool)).mean()
            ) if len(top1) else float("nan"),
        },
        "real_primary_descriptive": {
            "n_units": int(
                unit_summary.loc[
                    unit_summary["stratum"] == "real_primary", "unit_id"
                ].nunique()
            ),
            "within_median_tau": float(real.loc["within_family", "median_tau"]),
            "cross_median_tau": float(real.loc["cross_family", "median_tau"]),
            "combined_median_tau": float(real.loc["combined", "median_tau"]),
        },
        "n_pair_rows": int(len(pairs)),
        "n_endpoint_pair_summary_rows": int(len(pair_summary)),
        "n_null_rows": int(len(nulls)),
        "hypotheses": verdicts,
        "protocol_path": config["protocol"]["path"],
    }


def self_check(
    config: dict[str, Any],
    root: Path,
    attempt_root: Path,
    freeze_checks: list[dict[str, Any]],
    inputs: dict[str, Any],
    unit_scores: pd.DataFrame,
    pairs: pd.DataFrame,
    unit_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    top1_transfer: pd.DataFrame,
    nulls: pd.DataFrame,
    verdicts: dict[str, Any],
) -> dict[str, Any]:
    freeze_manifest = json.loads(
        resolve_project_path(root, config["protocol"]["freeze_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    expected = config["expected"]
    simulation_scores = unit_scores[unit_scores["unit_role"] == "simulation"]
    real_primary = unit_scores[unit_scores["stratum"] == "real_primary"]
    real_sensitivity = unit_scores[unit_scores["stratum"] == "real_sensitivity"]
    rank_max = unit_scores.groupby(["stratum", "unit_id", "endpoint"])["rank"].max()
    checks = {
        "freeze_manifest_chronology_reviewable": _freeze_chronology_reviewable(
            freeze_manifest
        ),
        "freeze_hashes_match": all(item["match"] for item in freeze_checks),
        "sat1_rows_total_expected": len(inputs["simulation"]) == int(expected["sat1_rows_total"]),
        "simulation_units_expected": simulation_scores["unit_id"].nunique() == int(expected["simulation_units"]),
        "simulation_methods_expected": simulation_scores["method_id"].nunique() == int(expected["annotation_methods"]),
        "simulation_endpoints_expected": simulation_scores["endpoint"].nunique() == int(expected["simulation_endpoints"]),
        "simulation_unit_scores_rows_expected": len(simulation_scores) == int(expected["simulation_unit_scores_rows"]),
        "real_primary_units_expected": real_primary["unit_id"].nunique() == int(expected["real_primary_units"]),
        "real_sensitivity_units_expected": real_sensitivity["unit_id"].nunique() == int(expected["real_sensitivity_units"]),
        "real_methods_expected": real_primary["method_id"].nunique() == int(expected["real_methods"]),
        "real_endpoints_expected": real_primary["endpoint"].nunique() == int(expected["real_endpoints"]),
        "real_unit_scores_rows_expected": len(real_primary) + len(real_sensitivity) == int(expected["real_unit_scores_rows"]),
        "pair_tau_rows_expected": len(pairs) == int(expected["simulation_pair_tau_rows"]) + int(expected["real_pair_tau_rows"]),
        "unit_summary_rows_expected": len(unit_summary) == int(expected["unit_summary_rows"]),
        "pair_summary_rows_expected": len(pair_summary) == int(expected["simulation_pair_summary_rows"]) + int(expected["real_pair_summary_rows"]),
        "aggregate_rows_expected": len(aggregate) == int(expected["aggregate_rows"]),
        "top1_rows_expected": len(top1_transfer) == int(expected["simulation_top1_rows"]) + int(expected["real_top1_rows"]),
        "null_rows_expected": len(nulls) == int(expected["simulation_null_rows"]) + int(expected["real_null_rows"]),
        "ranks_within_method_count": bool(
            (rank_max <= simulation_scores["method_id"].nunique()).all()
        ),
        "no_control_in_unit_scores": bool(
            not set(unit_scores["method_id"]).intersection(
                set(config["simulation"]["methods"]["controls"])
            )
        ),
        "no_missing_scores": bool(not unit_scores["value"].isna().any()),
        "tau_range_valid": bool(
            ((pairs["tau_b"].dropna() >= -1 - 1e-9) & (pairs["tau_b"].dropna() <= 1 + 1e-9)).all()
        ),
        "hypotheses_H1_to_H5": set(verdicts) == {"H1", "H2", "H3", "H4", "H5"},
        "h5_uses_pooled_null_aggregation": bool(
            verdicts["H5"].get("null_aggregation")
            == "pooled_across_simulation_units_and_endpoint_pairs"
        ),
        "h5_pooled_null_draws_match_pair_rows": bool(
            int(verdicts["H5"].get("n_null_draws_pooled", 0))
            == int(
                pairs.loc[
                    pairs["stratum"].isin(config["simulation"]["strata"])
                    & (pairs["pair_type"] == "cross_family"),
                    "null_n",
                ].sum()
            )
        ),
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


def configure_logging(attempt_root: Path) -> logging.Handler:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        attempt_root / "logs" / "run.log", mode="a", encoding="utf-8"
    )
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
        LOGGER.info("NM-XEND1 start %s", started_utc)
        freeze_checks = verify_freeze(config, root)
        LOGGER.info("Freeze hashes verified: %s", len(freeze_checks))
        inputs = load_inputs(config, root)
        validate_inputs(inputs, config)
        LOGGER.info("Inputs validated")

        simulation_scores = build_simulation_unit_scores(inputs, config)
        real_scores = build_real_unit_scores(inputs, config)
        unit_scores = pd.concat([simulation_scores, real_scores], ignore_index=True)
        unit_scores = add_ranks(unit_scores, config)
        null_pool: list[np.ndarray] = []
        pairs = build_pair_taus(unit_scores, config, null_pool=null_pool)
        unit_summary = build_unit_summary(pairs)
        pair_summary = build_endpoint_pair_summary(pairs)
        aggregate = build_aggregate(unit_summary, config)
        top1_transfer = build_top1_transfer(pair_summary, config)
        pooled_null = summarize_pooled_null(null_pool)
        nulls = build_null_calibration(
            pairs, unit_summary, config, pooled_null=pooled_null
        )
        verdicts = hypothesis_verdicts(
            aggregate, unit_summary, pairs, config, null_pool=null_pool
        )
        LOGGER.info("Hypotheses: %s", json.dumps(verdicts, sort_keys=True))

        key = key_numbers(
            unit_scores,
            pairs,
            unit_summary,
            aggregate,
            pair_summary,
            nulls,
            verdicts,
            config,
        )
        write_tsv(unit_scores, attempt_root / "tables" / "xend1_unit_scores.tsv")
        write_tsv(pairs, attempt_root / "tables" / "xend1_endpoint_pair_taus.tsv")
        write_tsv(unit_summary, attempt_root / "tables" / "xend1_unit_summary.tsv")
        write_tsv(pair_summary, attempt_root / "tables" / "xend1_endpoint_pair_summary.tsv")
        write_tsv(aggregate, attempt_root / "tables" / "xend1_aggregate.tsv")
        write_tsv(top1_transfer, attempt_root / "tables" / "xend1_top1_transfer.tsv")
        write_tsv(nulls, attempt_root / "tables" / "xend1_null_calibration.tsv")
        write_tsv(
            aggregate[aggregate["stratum"] == "real_sensitivity"],
            attempt_root / "tables" / "xend1_real_sensitivity.tsv",
        )
        write_json(key, attempt_root / "key_numbers.json")
        write_tsv(pd.DataFrame(freeze_checks), attempt_root / "manifests" / "input_checksums.tsv")

        self_check(
            config,
            root,
            attempt_root,
            freeze_checks,
            inputs,
            unit_scores,
            pairs,
            unit_summary,
            pair_summary,
            aggregate,
            top1_transfer,
            nulls,
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
                resolve_project_path(root, config["protocol"]["freeze_manifest"]).read_text(
                    encoding="utf-8"
                )
            )["preregistration"]["sha256"],
            "freeze_manifest_path": str(config["protocol"]["freeze_manifest"]),
            "freeze_manifest_sha256": sha256_file(
                resolve_project_path(root, config["protocol"]["freeze_manifest"])
            ),
            "seed_bootstrap": int(config["seeds"]["bootstrap"]),
            "seed_permutation": int(config["seeds"]["permutation"]),
            "software_versions": software_versions(),
            "resource_policy": {
                "workers": int(config["resources"]["workers"]),
                "nice": int(config["resources"]["nice"]),
                "memory_limit_mb": int(config["resources"]["memory_limit_mb"]),
            },
            "outputs": {
                "unit_scores": "tables/xend1_unit_scores.tsv",
                "endpoint_pair_taus": "tables/xend1_endpoint_pair_taus.tsv",
                "unit_summary": "tables/xend1_unit_summary.tsv",
                "endpoint_pair_summary": "tables/xend1_endpoint_pair_summary.tsv",
                "aggregate": "tables/xend1_aggregate.tsv",
                "top1_transfer": "tables/xend1_top1_transfer.tsv",
                "null_calibration": "tables/xend1_null_calibration.tsv",
                "real_sensitivity": "tables/xend1_real_sensitivity.tsv",
                "key_numbers": "key_numbers.json",
            },
        }
        write_json(manifest, attempt_root / "manifests" / "run_manifest.json")
        LOGGER.info("NM-XEND1 complete: %s", json.dumps(verdicts, sort_keys=True))
        write_checksum_table(attempt_root)
        verification = verify_attempt(attempt_root, write=True)
        if not verification["all_hashes_match"] or not verification["coverage_all_non_terminal_files"]:
            raise ValueError(f"Checksum verification failed: {verification}")
        return {
            "attempt_root": str(attempt_root),
            "runtime_seconds": float(runtime),
            "peak_rss_kb": peak_rss_kb,
            "all_hashes_match": bool(verification["all_hashes_match"]),
            "coverage_all_non_terminal_files": bool(
                verification["coverage_all_non_terminal_files"]
            ),
            "hypotheses": verdicts,
            "key_numbers": key,
        }
    finally:
        LOGGER.removeHandler(file_handler)
        file_handler.close()
