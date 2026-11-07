"""NM-NC1 negative-control and RNA-oracle ladder analysis."""

from __future__ import annotations

import hashlib
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
from scipy.stats import rankdata


LOGGER = logging.getLogger("nm_nc1")


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


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def normalized_utility(method_value: Any, control_value: Any, margin: float) -> tuple[float, str]:
    method = finite_float(method_value)
    control = finite_float(control_value)
    if not math.isfinite(method) or not math.isfinite(control):
        return float("nan"), "NOT_ESTIMABLE_MISSING_VALUE"
    if 1.0 - control < margin:
        return float("nan"), "NOT_ESTIMABLE_NEAR_CEILING_CONTROL"
    return (method - control) / (1.0 - control), "ESTIMABLE"


def control_separation(delta: Any, margin: float) -> str:
    value = finite_float(delta)
    if not math.isfinite(value):
        return "NOT_ESTIMABLE"
    if value >= margin:
        return "ABOVE_CONTROL_GE_0.05"
    if value <= -margin:
        return "BELOW_CONTROL_BY_GE_0.05"
    return "WITHIN_0.05_OF_CONTROL"


def parse_simulation_components(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["simulation_id"].astype(str).str.extract(r"__r(?P<replicate>\d+)__c(?P<factor_cell>\d+)$")
    if parsed.isna().any().any():
        raise ValueError("simulation_id does not end in __rNN__cNNN for every row")
    result = frame.copy()
    result["replicate_index"] = parsed["replicate"].astype(int)
    result["factor_cell"] = parsed["factor_cell"].astype(int)
    return result


def collapse_factor_cells(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["simulation_type", "method_id", "endpoint", "factor_cell"],
            as_index=False,
        )["value"]
        .median()
        .rename(columns={"value": "cell_median"})
    )


def bootstrap_median_ci(
    values: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float, float]:
    draws = np.median(values[indices], axis=1)
    return float(np.median(values)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_gap_ci(
    method_values: np.ndarray,
    control_values: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float, float]:
    method_draws = np.median(method_values[indices], axis=1)
    control_draws = np.median(control_values[indices], axis=1)
    gap_draws = method_draws - control_draws
    return (
        float(np.median(method_values) - np.median(control_values)),
        float(np.percentile(gap_draws, 2.5)),
        float(np.percentile(gap_draws, 97.5)),
    )


def bootstrap_utility_ci(
    method_values: np.ndarray,
    control_values: np.ndarray,
    indices: np.ndarray,
    margin: float,
) -> tuple[float, float, float, float]:
    point, status = normalized_utility(np.median(method_values), np.median(control_values), margin)
    if status != "ESTIMABLE":
        return point, float("nan"), float("nan"), 0.0
    method_draws = np.median(method_values[indices], axis=1)
    control_draws = np.median(control_values[indices], axis=1)
    valid = (1.0 - control_draws) >= margin
    if not valid.any():
        return point, float("nan"), float("nan"), 0.0
    utility_draws = (method_draws[valid] - control_draws[valid]) / (1.0 - control_draws[valid])
    return (
        point,
        float(np.percentile(utility_draws, 2.5)),
        float(np.percentile(utility_draws, 97.5)),
        float(valid.mean()),
    )


def load_verified_manifest(
    manifest_path: Path,
    root: Path,
    input_paths: list[Path],
) -> list[dict[str, Any]]:
    frame = pd.read_csv(manifest_path, sep="\t")
    required = {"path", "sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest lacks columns {sorted(required)}: {manifest_path}")
    checks: list[dict[str, Any]] = []
    for input_path in input_paths:
        input_abs = input_path.resolve()
        matches: list[pd.Series] = []
        for _, row in frame.iterrows():
            manifest_path_value = str(row["path"])
            candidate = Path(manifest_path_value)
            candidate_abs = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
            if candidate_abs == input_abs:
                matches.append(row)
                continue
            if str(input_abs).endswith(manifest_path_value):
                matches.append(row)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one manifest row for {input_path}, found {len(matches)}")
        expected = str(matches[0]["sha256"])
        actual = sha256_file(input_path)
        checks.append(
            {
                "path": str(input_path.relative_to(root)),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
        if actual != expected:
            raise ValueError(f"Input hash mismatch for {input_path}: {actual} != {expected}")
    return checks


def verify_inputs(
    config: dict[str, Any],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = config["inputs"]
    real_paths = [
        resolve_project_path(root, inputs["real_control_deltas"]),
        resolve_project_path(root, inputs["real_marker_percentiles"]),
        resolve_project_path(root, inputs["real_svg_moran_i"]),
    ]
    simulation_paths = [
        resolve_project_path(root, inputs["simulation_cell_audit"]),
        resolve_project_path(root, inputs["sat1_output_checksums"]),
        resolve_project_path(root, inputs["sat1_checksum_verification"]),
    ]
    all_paths = real_paths + simulation_paths
    for path in all_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    checks = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in all_paths]
    sat1_manifest = resolve_project_path(root, inputs["sat1_output_checksums"])
    sat1_checks = load_verified_manifest(sat1_manifest, root, [resolve_project_path(root, inputs["simulation_cell_audit"])])
    ds3_manifest = resolve_project_path(root, inputs["ds3_output_checksums"])
    ds3_checks = load_verified_manifest(ds3_manifest, root, real_paths)
    return checks, sat1_checks + ds3_checks


def build_real_ladder(config: dict[str, Any], root: Path) -> pd.DataFrame:
    source = resolve_project_path(root, config["inputs"]["real_control_deltas"])
    frame = pd.read_csv(source, sep="\t")
    required = {"representation_key", "representation_id", "endpoint", "method_value", "control_value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Real-data source lacks columns {sorted(required)}")
    margin = float(config["thresholds"]["materiality_margin"])
    rows: list[dict[str, Any]] = []
    for rep in config["real"]["representations"]:
        for endpoint in config["real"]["endpoints"]:
            selected = frame[
                (frame["representation_key"].astype(str) == str(rep["key"]))
                & (frame["endpoint"].astype(str) == str(endpoint))
            ]
            if len(selected) > 1:
                raise ValueError(f"Duplicate real-data rows for {rep['key']} / {endpoint}")
            if selected.empty:
                method_value = control_value = float("nan")
                frozen_decision = "ABSENT_FROM_SOURCE"
            else:
                row = selected.iloc[0]
                method_value = finite_float(row.get("method_value"))
                control_value = finite_float(row.get("control_value"))
                frozen_decision = str(row.get("decision", ""))
                source_representation_id = str(row.get("representation_id", ""))
                if source_representation_id != str(rep["representation_id"]):
                    raise ValueError(
                        "Real-data source representation_id does not match the configured "
                        f"representation for {rep['key']}: {source_representation_id!r} != "
                        f"{rep['representation_id']!r}"
                    )
            delta = method_value - control_value if math.isfinite(method_value) and math.isfinite(control_value) else float("nan")
            utility, utility_status = normalized_utility(method_value, control_value, float(config["thresholds"]["near_ceiling_control_margin"]))
            rows.append(
                {
                    "task_id": config["task_id"],
                    "dataset_key": config["real"]["dataset_key"],
                    "study_id": config["real"]["study_id"],
                    "representation_key": str(rep["key"]),
                    "representation_id": str(rep["representation_id"]),
                    "role": str(rep["role"]),
                    "endpoint": str(endpoint),
                    "method_value": method_value,
                    "control_id": config["real"]["primary_control_id"],
                    "control_binding": "config_declared_accessibility_matched_control",
                    "control_value": control_value,
                    "oracle_value": float(config["real"]["oracle_value"]),
                    "control_gap": delta,
                    "oracle_gap": 1.0 - method_value if math.isfinite(method_value) else float("nan"),
                    "normalized_utility": utility,
                    "utility_status": utility_status,
                    "control_separation": control_separation(delta, margin),
                    "frozen_decision": frozen_decision,
                    "null_status": (
                        "NOT_ESTIMABLE_LABELS_NOT_FROZEN"
                        if str(endpoint).startswith("domain_")
                        else "SEE_NULL_CALIBRATION"
                    ),
                    "source_file": str(source.relative_to(root)),
                }
            )
    result = pd.DataFrame(rows)
    expected = len(config["real"]["representations"]) * len(config["real"]["endpoints"])
    if len(result) != expected:
        raise ValueError(f"Real ladder row count {len(result)} != expected {expected}")
    return result


def load_simulation_cells(config: dict[str, Any], root: Path) -> pd.DataFrame:
    source = resolve_project_path(root, config["inputs"]["simulation_cell_audit"])
    frame = pd.read_csv(source, sep="\t")
    required = {"simulation_type", "simulation_id", "method_id", "endpoint", "endpoint_role", "value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Simulation source lacks columns {sorted(required)}")
    methods = list(config["simulation"]["methods"]) + list(config["simulation"]["controls"])
    endpoints = list(config["simulation"]["endpoints"])
    strata = list(config["simulation"]["strata"])
    selected = frame[
        frame["endpoint_role"].astype(str).eq("primary")
        & frame["simulation_type"].astype(str).isin(strata)
        & frame["endpoint"].astype(str).isin(endpoints)
        & frame["method_id"].astype(str).isin(methods)
    ].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    if selected["value"].isna().any():
        raise ValueError("Simulation primary endpoint values contain non-finite entries")
    selected = parse_simulation_components(selected)
    counts = selected.groupby(["simulation_type", "method_id", "endpoint"]).size()
    expected = 64 * 5
    if not counts.eq(expected).all():
        bad = counts[counts != expected]
        raise ValueError(f"Simulation cell counts are not {expected}: {bad.to_dict()}")
    if not selected.groupby(["simulation_type", "method_id", "endpoint", "factor_cell"]).size().eq(5).all():
        raise ValueError("Each factor cell must contain exactly five seeds")
    expected_rows = len(strata) * len(methods) * len(endpoints) * expected
    if len(selected) != expected_rows:
        raise ValueError(f"Simulation primary row count {len(selected)} != expected {expected_rows}")
    return selected


def build_simulation_ladder(config: dict[str, Any], cells: pd.DataFrame) -> pd.DataFrame:
    collapsed = collapse_factor_cells(cells)
    margin = float(config["thresholds"]["materiality_margin"])
    near_ceiling = float(config["thresholds"]["near_ceiling_control_margin"])
    draws = int(config["thresholds"]["bootstrap_draws"])
    base_seed = int(config["seeds"]["bootstrap"])
    offsets = {
        "mechanistic": int(config["seeds"]["mechanistic_offset"]),
        "empirical_semisynthetic": int(config["seeds"]["empirical_offset"]),
    }
    primary_control = str(config["simulation"]["primary_control_id"])
    secondary_control = str(config["simulation"]["secondary_control_id"])
    methods = list(config["simulation"]["methods"])
    controls = list(config["simulation"]["controls"])
    rows: list[dict[str, Any]] = []
    for stratum in config["simulation"]["strata"]:
        for endpoint in config["simulation"]["endpoints"]:
            block = collapsed[
                collapsed["simulation_type"].astype(str).eq(stratum)
                & collapsed["endpoint"].astype(str).eq(str(endpoint))
            ]
            pivot = block.pivot(index="factor_cell", columns="method_id", values="cell_median").sort_index()
            if len(pivot) != 64:
                raise ValueError(f"Expected 64 factor cells for {stratum}/{endpoint}, found {len(pivot)}")
            rng = np.random.default_rng(base_seed + offsets[str(stratum)])
            indices = rng.integers(0, len(pivot), size=(draws, len(pivot)))
            primary_values = pivot[primary_control].to_numpy(dtype=float)
            secondary_values = pivot[secondary_control].to_numpy(dtype=float)
            for method_id in methods + controls:
                method_values = pivot[method_id].to_numpy(dtype=float)
                method_median, method_low, method_high = bootstrap_median_ci(method_values, indices)
                primary_median, primary_low, primary_high = bootstrap_median_ci(primary_values, indices)
                secondary_median, secondary_low, secondary_high = bootstrap_median_ci(secondary_values, indices)
                gap, gap_low, gap_high = bootstrap_gap_ci(method_values, primary_values, indices)
                secondary_gap, secondary_gap_low, secondary_gap_high = bootstrap_gap_ci(method_values, secondary_values, indices)
                utility, utility_low, utility_high, utility_valid = bootstrap_utility_ci(
                    method_values, primary_values, indices, near_ceiling
                )
                rows.append(
                    {
                        "task_id": config["task_id"],
                        "simulation_type": str(stratum),
                        "method_id": str(method_id),
                        "method_role": "control" if str(method_id) in controls else "annotation_method",
                        "endpoint": str(endpoint),
                        "n_factor_cells": int(len(pivot)),
                        "n_seeds_per_cell": 5,
                        "method_median": method_median,
                        "method_ci95_low": method_low,
                        "method_ci95_high": method_high,
                        "primary_control_id": primary_control,
                        "primary_control_median": primary_median,
                        "primary_control_ci95_low": primary_low,
                        "primary_control_ci95_high": primary_high,
                        "secondary_control_id": secondary_control,
                        "secondary_control_median": secondary_median,
                        "secondary_control_ci95_low": secondary_low,
                        "secondary_control_ci95_high": secondary_high,
                        "oracle_value": float(config["simulation"]["oracle_value"]),
                        "control_gap": gap,
                        "control_gap_ci95_low": gap_low,
                        "control_gap_ci95_high": gap_high,
                        "secondary_control_gap": secondary_gap,
                        "secondary_control_gap_ci95_low": secondary_gap_low,
                        "secondary_control_gap_ci95_high": secondary_gap_high,
                        "oracle_gap": 1.0 - method_median,
                        "normalized_utility": utility,
                        "normalized_utility_ci95_low": utility_low,
                        "normalized_utility_ci95_high": utility_high,
                        "utility_bootstrap_valid_fraction": utility_valid,
                        "utility_status": (
                            "ESTIMABLE"
                            if math.isfinite(utility)
                            else "NOT_ESTIMABLE_NEAR_CEILING_CONTROL"
                        ),
                        "control_separation": control_separation(gap, margin),
                    }
                )
    result = pd.DataFrame(rows)
    expected = (
        len(config["simulation"]["strata"])
        * (len(config["simulation"]["methods"]) + len(config["simulation"]["controls"]))
        * len(config["simulation"]["endpoints"])
    )
    if len(result) != expected:
        raise ValueError(f"Simulation ladder row count {len(result)} != expected {expected}")
    return result


def marker_null_row(group: pd.DataFrame, draws: int, seed: int, margin: float) -> dict[str, Any]:
    spread = group.groupby("cluster")["n_ranked_genes"].nunique(dropna=False)
    if bool((spread > 1).any()):
        raise ValueError("n_ranked_genes varies within a marker cluster")
    stats = (
        group.groupby("cluster", as_index=False)
        .agg(n_markers=("gene_id", "size"), n_ranked_genes=("n_ranked_genes", "first"), observed_mean=("marker_percentile", "mean"))
        .sort_values("cluster")
    )
    counts = stats["n_markers"].to_numpy(dtype=int)
    n_ranked_values = np.rint(stats["n_ranked_genes"].to_numpy(dtype=float)).astype(int)
    if bool((n_ranked_values < 100).any()) or counts.sum() < 100:
        raise ValueError("Marker null calibration has fewer than 100 markers or ranked genes")
    observed = float(np.average(stats["observed_mean"].to_numpy(dtype=float), weights=counts))
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=float)
    total = int(counts.sum())
    for draw_index in range(draws):
        weighted = 0.0
        for count, n_ranked in zip(counts, n_ranked_values):
            ranks = rng.choice(int(n_ranked), size=int(count), replace=False) + 1
            weighted += int(count) * float(ranks.mean()) / float(n_ranked)
        null[draw_index] = weighted / float(total)
    null_mean = float(null.mean())
    extremum = np.abs(null - null_mean)
    observed_extremum = abs(observed - null_mean)
    p_value = float((1 + int((extremum >= observed_extremum).sum())) / (draws + 1))
    return {
        "null_type": "marker_shuffled_gene_rank",
        "endpoint": "marker_mean_percentile",
        "n_units": total,
        "observed_value": observed,
        "null_mean": null_mean,
        "null_ci95_low": float(np.percentile(null, 2.5)),
        "null_ci95_high": float(np.percentile(null, 97.5)),
        "empirical_p_two_sided": p_value,
        "draws": draws,
        "seed": seed,
        "status": "OK",
    }


def svg_null_row(group: pd.DataFrame, draws: int, seed: int) -> dict[str, Any]:
    genes = group["gene_id"].astype(str).to_numpy()
    if len(genes) < 1000:
        raise ValueError("SVG null calibration has fewer than 1,000 genes")
    rna = group["rna_moran_i"].to_numpy(dtype=float)
    atac = group["atac_moran_i"].to_numpy(dtype=float)
    if not np.isfinite(rna).all() or not np.isfinite(atac).all():
        raise ValueError("SVG null calibration contains non-finite Moran values")
    rna_centered = rankdata(rna)
    atac_centered = rankdata(atac)
    rna_centered = (rna_centered - rna_centered.mean()) / rna_centered.std(ddof=0)
    atac_centered = (atac_centered - atac_centered.mean()) / atac_centered.std(ddof=0)
    observed = float(np.dot(rna_centered, atac_centered) / len(rna))
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=float)
    for draw_index in range(draws):
        permutation = rng.permutation(len(atac_centered))
        null[draw_index] = float(np.dot(rna_centered, atac_centered[permutation]) / len(rna))
    null_mean = float(null.mean())
    extremum = np.abs(null - null_mean)
    observed_extremum = abs(observed - null_mean)
    p_value = float((1 + int((extremum >= observed_extremum).sum())) / (draws + 1))
    return {
        "null_type": "svg_gene_label_permutation",
        "endpoint": "svg_spearman",
        "n_units": int(len(genes)),
        "observed_value": observed,
        "null_mean": null_mean,
        "null_ci95_low": float(np.percentile(null, 2.5)),
        "null_ci95_high": float(np.percentile(null, 97.5)),
        "empirical_p_two_sided": p_value,
        "draws": draws,
        "seed": seed,
        "status": "OK",
    }


def stable_seed(base: int, key: str) -> int:
    return base + int(sha256_text(key)[:8], 16) % 100000


def build_null_calibration(config: dict[str, Any], root: Path) -> pd.DataFrame:
    draws = int(config["thresholds"]["null_draws"])
    base_seed = int(config["seeds"]["nulls"])
    marker_source = resolve_project_path(root, config["inputs"]["real_marker_percentiles"])
    svg_source = resolve_project_path(root, config["inputs"]["real_svg_moran_i"])
    markers = pd.read_csv(marker_source, sep="\t")
    svg = pd.read_csv(svg_source, sep="\t")
    marker_required = {"representation_key", "representation_id", "role", "cluster", "gene_id", "n_ranked_genes", "marker_percentile"}
    svg_required = {"representation_key", "representation_id", "role", "gene_id", "rna_moran_i", "atac_moran_i"}
    if not marker_required.issubset(markers.columns):
        raise ValueError(f"Marker calibration source lacks columns {sorted(marker_required)}")
    if not svg_required.issubset(svg.columns):
        raise ValueError(f"SVG calibration source lacks columns {sorted(svg_required)}")
    rows: list[dict[str, Any]] = []
    for (rep_key, rep_id, role), group in markers.groupby(["representation_key", "representation_id", "role"], sort=True):
        result = marker_null_row(group, draws, stable_seed(base_seed, f"marker::{rep_key}::{role}"), float(config["thresholds"]["materiality_margin"]))
        rows.append({**result, "representation_key": str(rep_key), "representation_id": str(rep_id), "role": str(role), "source_file": str(marker_source.relative_to(root))})
    for (rep_key, rep_id, role), group in svg.groupby(["representation_key", "representation_id", "role"], sort=True):
        result = svg_null_row(group, draws, stable_seed(base_seed, f"svg::{rep_key}::{role}"))
        rows.append({**result, "representation_key": str(rep_key), "representation_id": str(rep_id), "role": str(role), "source_file": str(svg_source.relative_to(root))})
    domain_keys = markers[["representation_key", "representation_id", "role"]].drop_duplicates().sort_values(["representation_key", "role"])
    for _, rep in domain_keys.iterrows():
        for endpoint in ("domain_ARI", "domain_NMI"):
            rows.append(
                {
                    "null_type": "domain_shuffled_label",
                    "representation_key": str(rep["representation_key"]),
                    "representation_id": str(rep["representation_id"]),
                    "role": str(rep["role"]),
                    "endpoint": endpoint,
                    "n_units": 0,
                    "observed_value": float("nan"),
                    "null_mean": float("nan"),
                    "null_ci95_low": float("nan"),
                    "null_ci95_high": float("nan"),
                    "empirical_p_two_sided": float("nan"),
                    "draws": 0,
                    "seed": base_seed,
                    "status": "NOT_ESTIMABLE_LABELS_NOT_FROZEN",
                    "source_file": str(marker_source.relative_to(root)),
                }
            )
    result = pd.DataFrame(rows)
    result["task_id"] = config["task_id"]
    return result


def key_numbers(
    real: pd.DataFrame,
    simulation: pd.DataFrame,
    nulls: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    real_estimable = real["utility_status"].eq("ESTIMABLE")
    sim_methods = simulation["method_role"].eq("annotation_method")
    return {
        "task_id": config["task_id"],
        "protocol_sha256": config["protocol"]["sha256"],
        "real_ladder": {
            "n_rows": int(len(real)),
            "n_estimable": int(real_estimable.sum()),
            "n_not_estimable": int((~real_estimable).sum()),
            "n_above_control_ge_0_05": int(real["control_separation"].eq("ABOVE_CONTROL_GE_0.05").sum()),
            "n_within_0_05_of_control": int(real["control_separation"].eq("WITHIN_0.05_OF_CONTROL").sum()),
            "n_below_control_by_ge_0_05": int(real["control_separation"].eq("BELOW_CONTROL_BY_GE_0.05").sum()),
            "median_oracle_gap": float(real.loc[real_estimable, "oracle_gap"].median()) if real_estimable.any() else None,
            "max_normalized_utility": float(real.loc[real_estimable, "normalized_utility"].max()) if real_estimable.any() else None,
        },
        "simulation_ladder": {
            "n_rows": int(len(simulation)),
            "n_annotation_method_rows": int(sim_methods.sum()),
            "n_control_rows": int((~sim_methods).sum()),
            "n_strata": int(simulation["simulation_type"].nunique()),
            "n_methods": int(simulation.loc[sim_methods, "method_id"].nunique()),
            "n_endpoints": int(simulation["endpoint"].nunique()),
            "n_factor_cells_per_stratum_endpoint": int(simulation["n_factor_cells"].max()),
            "n_cells_used": int(len(simulation["simulation_type"].unique()) * 7 * len(simulation["endpoint"].unique()) * 64 * 5),
            "n_method_rows_above_matched_control_ge_0_05": int(
                (sim_methods & simulation["control_separation"].eq("ABOVE_CONTROL_GE_0.05")).sum()
            ),
            "n_method_rows_below_matched_control_by_ge_0_05": int(
                (sim_methods & simulation["control_separation"].eq("BELOW_CONTROL_BY_GE_0.05")).sum()
            ),
            "max_normalized_utility_annotation_method": float(
                simulation.loc[sim_methods, "normalized_utility"].max()
            ),
        },
        "null_calibration": {
            "n_rows": int(len(nulls)),
            "n_marker_rows": int((nulls["null_type"] == "marker_shuffled_gene_rank").sum()),
            "n_svg_rows": int((nulls["null_type"] == "svg_gene_label_permutation").sum()),
            "n_domain_not_estimable_rows": int((nulls["status"] == "NOT_ESTIMABLE_LABELS_NOT_FROZEN").sum()),
            "n_marker_p_lt_0_05": int(((nulls["null_type"] == "marker_shuffled_gene_rank") & (nulls["empirical_p_two_sided"] < 0.05)).sum()),
            "n_svg_p_lt_0_05": int(((nulls["null_type"] == "svg_gene_label_permutation") & (nulls["empirical_p_two_sided"] < 0.05)).sum()),
        },
        "interpretation": {
            "oracle": "RNA/reference agreement upper bound (1.0), not a deployable method",
            "normalized_utility": "(method - matched_control) / (1 - matched_control)",
            "claim_upgrade": False,
            "domain_shuffled_null": "NOT_ESTIMABLE because per-spot labels were not frozen",
        },
    }


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
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
    input_hash_checks: list[dict[str, Any]],
    real: pd.DataFrame,
    simulation: pd.DataFrame,
    nulls: pd.DataFrame,
) -> dict[str, Any]:
    protocol_path = resolve_project_path(root, config["protocol"]["path"])
    checks = {
        "protocol_hash_matches": sha256_file(protocol_path) == str(config["protocol"]["sha256"]),
        "input_manifest_hashes_match": all(item["match"] for item in input_hash_checks),
        "real_row_count_28": len(real) == 28,
        "real_estimable_count_16": int(real["utility_status"].eq("ESTIMABLE").sum()) == 16,
        "simulation_row_count_84": len(simulation) == 84,
        "simulation_method_rows_60": int(simulation["method_role"].eq("annotation_method").sum()) == 60,
        "simulation_control_rows_24": int(simulation["method_role"].eq("control").sum()) == 24,
        "simulation_cells_26880": int(
            len(config["simulation"]["strata"])
            * (len(config["simulation"]["methods"]) + len(config["simulation"]["controls"]))
            * len(config["simulation"]["endpoints"])
            * 64
            * 5
        )
        == 26880,
        "marker_null_rows_9": int((nulls["null_type"] == "marker_shuffled_gene_rank").sum()) == 9,
        "svg_null_rows_9": int((nulls["null_type"] == "svg_gene_label_permutation").sum()) == 9,
        "domain_null_rows_18": int((nulls["status"] == "NOT_ESTIMABLE_LABELS_NOT_FROZEN").sum()) == 18,
        "outputs_contained": all(
            str(path.resolve()).startswith(str(attempt_root.resolve()) + os.sep)
            for path in output_files(attempt_root)
        ),
        "no_utility_without_status": bool(
            (
                real["normalized_utility"].isna() == ~real["utility_status"].eq("ESTIMABLE")
            ).all()
        ),
        "simulation_utility_ci_consistent": bool(
            (
                simulation["normalized_utility"].isna()
                == (simulation["utility_status"] != "ESTIMABLE")
            ).all()
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
        LOGGER.info("NM-NC1 start %s", started_utc)
        LOGGER.info("Protocol SHA-256 %s", config["protocol"]["sha256"])
        input_checks, manifest_checks = verify_inputs(config, root)
        LOGGER.info("Input hashes verified: %s files", len(input_checks))
        real = build_real_ladder(config, root)
        LOGGER.info("Real ladder rows: %s", len(real))
        cells = load_simulation_cells(config, root)
        LOGGER.info("Simulation primary cells: %s", len(cells))
        simulation = build_simulation_ladder(config, cells)
        LOGGER.info("Simulation ladder rows: %s", len(simulation))
        nulls = build_null_calibration(config, root)
        LOGGER.info("Null calibration rows: %s", len(nulls))
        key = key_numbers(real, simulation, nulls, config)
        write_tsv(real, attempt_root / "tables" / "nc1_real_ladder.tsv")
        write_tsv(simulation, attempt_root / "tables" / "nc1_simulation_ladder.tsv")
        write_tsv(nulls, attempt_root / "tables" / "nc1_null_calibration.tsv")
        write_json(key, attempt_root / "key_numbers.json")
        input_table = pd.DataFrame(input_checks)
        write_tsv(input_table, attempt_root / "manifests" / "input_checksums.tsv")
        manifest_table = pd.DataFrame(manifest_checks)
        write_tsv(manifest_table, attempt_root / "manifests" / "source_manifest_verification.tsv")
        self_check(config, root, attempt_root, manifest_checks, real, simulation, nulls)
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
            "protocol_path": config["protocol"]["path"],
            "protocol_sha256": config["protocol"]["sha256"],
            "seed": int(config["seeds"]["bootstrap"]),
            "software_versions": software_versions(),
            "outputs": {
                "real_ladder": "tables/nc1_real_ladder.tsv",
                "simulation_ladder": "tables/nc1_simulation_ladder.tsv",
                "null_calibration": "tables/nc1_null_calibration.tsv",
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
    }
