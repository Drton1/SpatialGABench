"""NM-NULL1 domain-label permutation null for frozen DS1 concordance.

The audit reads only frozen DS1 per-spot domain predictions and frozen seed
metrics. It does not construct a representation, execute a method, access RNA
values, or inspect new outcomes.
"""

from __future__ import annotations

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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from spatialgabench.nm_rank1 import (
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


LOGGER = logging.getLogger("nm_null1")


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
    }


def _unit_id(dataset_key: str, depth_fraction: float, representation: str) -> str:
    return f"{dataset_key}__d{float(depth_fraction):.2f}__{representation}"


def _as_codes(values: pd.Series) -> np.ndarray:
    codes, uniques = pd.factorize(values.astype(str), sort=True)
    if len(uniques) < 2:
        raise ValueError("A unit must have at least two observed label classes")
    return codes.astype(np.int64)


def _comb2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return values * (values - 1.0) / 2.0


def ari_from_counts(
    counts: np.ndarray,
    n_samples: int,
    truth_counts: np.ndarray,
    pred_counts: np.ndarray,
) -> float:
    """Adjusted Rand index from a flat contingency table."""
    sum_comb = float(_comb2(counts).sum())
    sum_truth = float(_comb2(truth_counts).sum())
    sum_pred = float(_comb2(pred_counts).sum())
    total = float(_comb2(np.asarray([n_samples]))[0])
    if total == 0.0:
        return 1.0
    expected = sum_truth * sum_pred / total
    denominator = 0.5 * (sum_truth + sum_pred) - expected
    if denominator == 0.0:
        return 1.0
    return (sum_comb - expected) / denominator


def nmi_from_counts(
    counts: np.ndarray,
    n_samples: int,
    truth_counts: np.ndarray,
    pred_counts: np.ndarray,
    n_pred: int,
) -> float:
    """Arithmetic-normalized mutual information from a flat contingency table."""
    nonzero = counts > 0
    if not bool(nonzero.any()) or n_samples <= 0:
        return 0.0
    flat = np.flatnonzero(nonzero)
    truth_index = flat // n_pred
    pred_index = flat % n_pred
    observed = counts[flat].astype(np.float64)
    mi = float(
        np.sum(
            (observed / n_samples)
            * np.log(
                (observed * n_samples)
                / (truth_counts[truth_index] * pred_counts[pred_index])
            )
        )
    )
    mi = max(mi, 0.0)
    truth_prob = truth_counts.astype(np.float64) / n_samples
    pred_prob = pred_counts.astype(np.float64) / n_samples
    entropy_truth = float(-np.sum(truth_prob[truth_prob > 0] * np.log(truth_prob[truth_prob > 0])))
    entropy_pred = float(-np.sum(pred_prob[pred_prob > 0] * np.log(pred_prob[pred_prob > 0])))
    normalizer = 0.5 * (entropy_truth + entropy_pred)
    if normalizer == 0.0:
        return 1.0
    return mi / normalizer


def observed_metrics(
    truth_labels: pd.Series,
    predicted_labels: pd.Series,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    truth = np.asarray(truth_labels.astype(str), dtype=object)
    predicted = np.asarray(predicted_labels.astype(str), dtype=object)
    ari = float(adjusted_rand_score(truth, predicted))
    nmi = float(normalized_mutual_info_score(truth, predicted, average_method="arithmetic"))
    return ari, nmi, truth, predicted


def permutation_null(
    truth_labels: pd.Series,
    predicted_labels: pd.Series,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-draw null ARI/NMI with fixed truth and predicted marginals."""
    truth_codes = _as_codes(truth_labels)
    pred_codes = _as_codes(predicted_labels)
    n_samples = int(len(truth_codes))
    n_truth = int(truth_codes.max()) + 1
    n_pred = int(pred_codes.max()) + 1
    truth_counts = np.bincount(truth_codes, minlength=n_truth).astype(np.float64)
    pred_counts = np.bincount(pred_codes, minlength=n_pred).astype(np.float64)
    rng = np.random.default_rng(seed)
    null_ari = np.empty(int(draws), dtype=np.float64)
    null_nmi = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        permuted = rng.permutation(pred_codes)
        counts = np.bincount(
            truth_codes * n_pred + permuted,
            minlength=n_truth * n_pred,
        ).astype(np.float64)
        null_ari[index] = ari_from_counts(
            counts, n_samples, truth_counts, pred_counts
        )
        null_nmi[index] = nmi_from_counts(
            counts, n_samples, truth_counts, pred_counts, n_pred
        )
        if (index + 1) % 2000 == 0:
            LOGGER.info("  permutation progress %d/%d", index + 1, draws)
    return null_ari, null_nmi, truth_codes, pred_codes


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    n_values = int(len(p_values))
    if n_values == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    adjusted = ranked * n_values / np.arange(1, n_values + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty(n_values, dtype=np.float64)
    result[order] = adjusted
    return result


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def build_unit_table(inputs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    predictions = inputs["predictions"]
    seed_metrics = inputs["seed_metrics"]
    draws = int(config["thresholds"]["permutation_draws"])
    base_seed = int(config["seeds"]["permutation"])
    primary_seed = int(config["scope"]["primary_seed"])
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    sensitivity = config["scope"]["sensitivity"]
    sensitivity_dataset = str(sensitivity["dataset_key"])
    sensitivity_depth = float(sensitivity["depth_fraction"])
    endpoints = list(config["endpoints"])

    primary_seed_metrics = seed_metrics[
        seed_metrics["clustering_seed"].astype(int) == primary_seed
    ].copy()
    if len(primary_seed_metrics) != int(config["expected"]["primary_seed_units"]):
        raise ValueError("Primary-seed metric row count does not match the freeze")
    keyed_metrics = {}
    for row in primary_seed_metrics.itertuples(index=False):
        key = (str(row.dataset_key), float(row.depth_fraction), str(row.representation))
        keyed_metrics[key] = (float(row.ARI), float(row.NMI))

    rows: list[dict[str, Any]] = []
    observed_seed_keys: set[tuple[str, float, str]] = set()
    for (dataset_key, depth_fraction, representation), unit in predictions.groupby(
        ["dataset_key", "depth_fraction", "representation"], sort=True
    ):
        dataset_key = str(dataset_key)
        depth_fraction = float(depth_fraction)
        representation = str(representation)
        unit_key = (dataset_key, depth_fraction, representation)
        observed_seed_keys.add(unit_key)
        unit_id = _unit_id(dataset_key, depth_fraction, representation)
        is_sensitivity = (
            dataset_key == sensitivity_dataset
            and math.isclose(depth_fraction, sensitivity_depth, abs_tol=tolerance)
        )
        role = str(sensitivity["role"]) if is_sensitivity else "primary"

        if unit["barcode"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate barcode in unit {unit_id}")
        if unit["truth_label"].isna().any() or unit["predicted_cluster"].isna().any():
            raise ValueError(f"Missing label in unit {unit_id}")
        unit = unit.sort_values("barcode").reset_index(drop=True)
        truth = unit["truth_label"].astype(str)
        predicted = unit["predicted_cluster"].astype(str)
        n_truth = int(truth.nunique())
        n_pred = int(predicted.nunique())
        if n_truth < 2 or n_pred < 2:
            raise ValueError(f"Unit {unit_id} has fewer than two truth or predicted clusters")

        ari, nmi, _, _ = observed_metrics(truth, predicted)
        if unit_key not in keyed_metrics:
            raise ValueError(f"Frozen seed metrics lack unit {unit_id}")
        expected_ari, expected_nmi = keyed_metrics[unit_key]
        if abs(ari - expected_ari) > tolerance or abs(nmi - expected_nmi) > tolerance:
            raise ValueError(
                f"Observed metric mismatch for {unit_id}: "
                f"ARI {ari} vs {expected_ari}, NMI {nmi} vs {expected_nmi}"
            )

        LOGGER.info(
            "Unit %s: n=%d, truth=%d, pred=%d, ARI=%.6f, NMI=%.6f, draws=%d",
            unit_id,
            len(unit),
            n_truth,
            n_pred,
            ari,
            nmi,
            draws,
        )
        null_ari, null_nmi, _, _ = permutation_null(
            truth,
            predicted,
            draws=draws,
            seed=stable_seed(base_seed, unit_id),
        )

        for endpoint, observed, null_values in (
            ("ARI", ari, null_ari),
            ("NMI", nmi, null_nmi),
        ):
            null_median = float(np.median(null_values))
            null_q05 = _quantile(null_values, 0.05)
            null_q95 = _quantile(null_values, 0.95)
            null_q99 = _quantile(null_values, 0.99)
            empirical_p = float(
                (1.0 + float(np.sum(null_values >= observed)))
                / (1.0 + float(len(null_values)))
            )
            rows.append(
                {
                    "unit_id": unit_id,
                    "dataset_key": dataset_key,
                    "depth_fraction": depth_fraction,
                    "representation": representation,
                    "role": role,
                    "analysis_role": (
                        "NONCOMPARABLE_SENSITIVITY_ONLY" if is_sensitivity else "primary"
                    ),
                    "n_spots": int(len(unit)),
                    "n_truth_clusters": n_truth,
                    "n_predicted_clusters": n_pred,
                    "endpoint": endpoint,
                    "observed": observed,
                    "null_median": null_median,
                    "null_q05": null_q05,
                    "null_q95": null_q95,
                    "null_q99": null_q99,
                    "null_min": float(np.min(null_values)),
                    "null_max": float(np.max(null_values)),
                    "observed_minus_median": observed - null_median,
                    "observed_minus_q95": observed - null_q95,
                    "empirical_p": empirical_p,
                    "permutation_draws": draws,
                    "permutation_seed": int(stable_seed(base_seed, unit_id)),
                    "above_q95": bool(observed > null_q95),
                }
            )

    if observed_seed_keys != set(keyed_metrics):
        raise ValueError("Prediction units and primary-seed metric units differ")
    result = pd.DataFrame(rows)
    primary_mask = result["role"] == "primary"
    result["bh_q"] = np.nan
    result.loc[primary_mask, "bh_q"] = bh_adjust(
        result.loc[primary_mask, "empirical_p"].to_numpy(dtype=float)
    )
    result["bh_significant"] = False
    result.loc[primary_mask, "bh_significant"] = (
        result.loc[primary_mask, "bh_q"]
        <= float(config["thresholds"]["bh_alpha"])
    )
    return result.sort_values(["role", "dataset_key", "depth_fraction", "representation", "endpoint"]).reset_index(drop=True)


def build_dataset_loo(unit_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = unit_table[unit_table["role"] == "primary"].copy()
    alpha = float(config["thresholds"]["bh_alpha"])
    coverage_bar = float(config["thresholds"]["chance_coverage"])
    rows: list[dict[str, Any]] = []
    for omitted in config["scope"]["dataset_loo"]:
        subset = primary[primary["dataset_key"] != str(omitted)].copy()
        if subset.empty:
            raise ValueError(f"No primary units remain after omitting {omitted}")
        subset["subset_bh_q"] = bh_adjust(subset["empirical_p"].to_numpy(dtype=float))
        subset["subset_significant"] = subset["subset_bh_q"] <= alpha
        row: dict[str, Any] = {
            "omitted_dataset_key": str(omitted),
            "n_units_remaining": int(subset["unit_id"].nunique()),
            "n_cells_remaining": int(len(subset)),
        }
        all_pass = True
        for endpoint in config["endpoints"]:
            endpoint_rows = subset[subset["endpoint"] == endpoint]
            n_cells = int(len(endpoint_rows))
            n_sig = int(endpoint_rows["subset_significant"].sum())
            coverage = float(n_sig / n_cells) if n_cells else float("nan")
            row[f"n_{endpoint.lower()}_cells"] = n_cells
            row[f"n_{endpoint.lower()}_significant"] = n_sig
            row[f"coverage_{endpoint.lower()}"] = coverage
            row[f"pass_{endpoint.lower()}"] = bool(
                n_cells > 0 and coverage >= coverage_bar
            )
            all_pass = all_pass and bool(row[f"pass_{endpoint.lower()}"])
        row["pass"] = bool(all_pass)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("omitted_dataset_key").reset_index(drop=True)


def build_seed_sensitivity(
    inputs: dict[str, Any],
    unit_table: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    seed_metrics = inputs["seed_metrics"].copy()
    all_units = unit_table.copy()
    rows: list[dict[str, Any]] = []
    for unit_id, group in all_units.groupby("unit_id", sort=True):
        first = group.iloc[0]
        dataset_key = str(first["dataset_key"])
        depth_fraction = float(first["depth_fraction"])
        representation = str(first["representation"])
        unit_seed = seed_metrics[
            (seed_metrics["dataset_key"].astype(str) == dataset_key)
            & np.isclose(
                seed_metrics["depth_fraction"].astype(float),
                depth_fraction,
                atol=float(config["thresholds"]["numeric_tolerance"]),
            )
            & (seed_metrics["representation"].astype(str) == representation)
        ].copy()
        if len(unit_seed) != int(config["expected"]["seed_rows_per_unit"]):
            raise ValueError(f"Seed metrics for {unit_id} do not have five rows")
        row: dict[str, Any] = {
            "unit_id": unit_id,
            "dataset_key": dataset_key,
            "depth_fraction": depth_fraction,
            "representation": representation,
            "n_seeds": int(len(unit_seed)),
            "reference_role": "DESCRIPTIVE_PRIMARY_SEED_NULL_REFERENCE",
        }
        for endpoint in config["endpoints"]:
            endpoint_row = group[group["endpoint"] == endpoint].iloc[0]
            values = unit_seed[endpoint].to_numpy(dtype=float)
            q95 = float(endpoint_row["null_q95"])
            row[f"{endpoint.lower()}_seed_min"] = float(np.min(values))
            row[f"{endpoint.lower()}_seed_median"] = float(np.median(values))
            row[f"{endpoint.lower()}_seed_max"] = float(np.max(values))
            row[f"{endpoint.lower()}_n_seeds_above_primary_null_q95"] = int(
                np.sum(values > q95)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


def hypothesis_verdicts(
    unit_table: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    primary = unit_table[unit_table["role"] == "primary"]
    sensitivity = unit_table[unit_table["role"] != "primary"]
    endpoints = list(config["endpoints"])
    chance_bar = float(config["thresholds"]["chance_coverage"])
    excess_bar = float(config["thresholds"]["positive_excess_coverage"])
    center_tol = float(config["thresholds"]["null_center_tolerance"])
    alpha = float(config["thresholds"]["bh_alpha"])

    h1_details: dict[str, Any] = {}
    h1_pass = True
    for endpoint in endpoints:
        endpoint_rows = primary[primary["endpoint"] == endpoint]
        n_units = int(len(endpoint_rows))
        n_separated = int(
            ((endpoint_rows["above_q95"]) & (endpoint_rows["bh_q"] <= alpha)).sum()
        )
        coverage = float(n_separated / n_units) if n_units else float("nan")
        h1_details[f"n_{endpoint.lower()}_separated"] = n_separated
        h1_details[f"n_{endpoint.lower()}_units"] = n_units
        h1_details[f"coverage_{endpoint.lower()}"] = coverage
        h1_pass = h1_pass and bool(n_units > 0 and coverage >= chance_bar)

    n_cells = int(len(primary))
    n_positive = int((primary["observed_minus_median"] > 0).sum())
    positive_coverage = float(n_positive / n_cells) if n_cells else float("nan")
    median_gap = {
        endpoint: float(
            primary.loc[primary["endpoint"] == endpoint, "observed_minus_median"].median()
        )
        for endpoint in endpoints
    }
    h2_pass = bool(
        n_cells > 0
        and positive_coverage >= excess_bar
        and all(value > 0 for value in median_gap.values())
    )

    null_center = primary["null_median"].abs()
    null_order = primary["null_q95"] - primary["null_median"]
    null_spread = primary["null_q95"] - primary["null_q05"]
    h3_pass = bool(
        (null_center <= center_tol).all()
        and (null_order > 0).all()
        and (null_spread > 0).all()
    )

    h4_pass = bool(len(dataset_loo) == int(config["expected"]["dataset_loo_rows"]) and dataset_loo["pass"].all())

    h5_cells = int(len(sensitivity))
    h5_above = int(sensitivity["above_q95"].sum()) if h5_cells else 0
    h5_pass = bool(h5_cells == int(config["expected"]["sensitivity_cells"]) and h5_above == h5_cells)

    return {
        "H1": {
            "hypothesis": "Observed domain concordance is separated from the label-permutation null",
            "verdict": "PASS" if h1_pass else "FAIL",
            "coverage_bar": chance_bar,
            "details": h1_details,
        },
        "H2": {
            "hypothesis": "The null-calibrated excess is positive",
            "verdict": "PASS" if h2_pass else "FAIL",
            "positive_excess_coverage": positive_coverage,
            "positive_excess_bar": excess_bar,
            "n_positive_cells": n_positive,
            "n_cells": n_cells,
            "median_observed_minus_null_median": median_gap,
        },
        "H3": {
            "hypothesis": "The permutation null is non-degenerate and centered near zero",
            "verdict": "PASS" if h3_pass else "FAIL",
            "null_center_tolerance": center_tol,
            "max_abs_null_median": float(null_center.max()) if len(null_center) else float("nan"),
            "min_null_q95_minus_median": float(null_order.min()) if len(null_order) else float("nan"),
            "min_null_q95_minus_q05": float(null_spread.min()) if len(null_spread) else float("nan"),
        },
        "H4": {
            "hypothesis": "The chance-separation conclusion is not driven by one dataset",
            "verdict": "PASS" if h4_pass else "FAIL",
            "n_loo_subsets": int(len(dataset_loo)),
            "n_passing_subsets": int(dataset_loo["pass"].sum()) if len(dataset_loo) else 0,
        },
        "H5": {
            "hypothesis": "The lowest-depth sensitivity-only units are directionally consistent",
            "verdict": "PASS" if h5_pass else "FAIL",
            "n_sensitivity_cells": h5_cells,
            "n_above_q95": h5_above,
            "role": "SENSITIVITY_ONLY_NOT_A_RANKING_OR_DEPTH_CLAIM",
        },
    }


def key_numbers(
    unit_table: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    seed_sensitivity: pd.DataFrame,
    verdicts: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    primary = unit_table[unit_table["role"] == "primary"]
    sensitivity = unit_table[unit_table["role"] != "primary"]
    return {
        "task_id": config["task_id"],
        "n_units_total": int(unit_table["unit_id"].nunique()),
        "n_primary_units": int(primary["unit_id"].nunique()),
        "n_sensitivity_units": int(sensitivity["unit_id"].nunique()),
        "n_primary_cells": int(len(primary)),
        "n_sensitivity_cells": int(len(sensitivity)),
        "primary_endpoints": list(config["endpoints"]),
        "permutation_draws_per_unit": int(config["thresholds"]["permutation_draws"]),
        "total_permutations": int(
            unit_table["unit_id"].nunique()
            * int(config["thresholds"]["permutation_draws"])
        ),
        "permutation_seed": int(config["seeds"]["permutation"]),
        "bh_alpha": float(config["thresholds"]["bh_alpha"]),
        "chance_coverage_bar": float(config["thresholds"]["chance_coverage"]),
        "positive_excess_coverage_bar": float(config["thresholds"]["positive_excess_coverage"]),
        "max_observed_ari": float(primary.loc[primary["endpoint"] == "ARI", "observed"].max()),
        "max_observed_nmi": float(primary.loc[primary["endpoint"] == "NMI", "observed"].max()),
        "median_observed_ari": float(primary.loc[primary["endpoint"] == "ARI", "observed"].median()),
        "median_observed_nmi": float(primary.loc[primary["endpoint"] == "NMI", "observed"].median()),
        "dataset_loo": dataset_loo.to_dict(orient="records"),
        "seed_sensitivity_rows": int(len(seed_sensitivity)),
        "hypotheses": verdicts,
        "claim_boundary": [
            "Label-permutation null, not a spatially constrained or independent-truth null.",
            "DS1 only; it does not provide the missing GSE205055 human domain-label null.",
            "No method, representation, biological, causal, depth-effect, or universal-ranking claim is authorized.",
        ],
    }


def verify_freeze(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_project_path(root, config["protocol"]["freeze_manifest"])
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    if freeze.get("frozen_before_analysis") is not True:
        raise ValueError("NM-NULL1 requires an explicit pre-analysis freeze")
    checks: list[dict[str, Any]] = []
    for role in ("config", "preregistration", "code", "tests"):
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
    failed = [item for item in checks if not item["match"]]
    if failed:
        raise ValueError(f"Freeze hash mismatch: {failed}")
    return checks


def load_inputs(config: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = {
        name: resolve_project_path(root, value)
        for name, value in config["inputs"].items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "predictions": pd.read_csv(paths["domain_predictions"], sep="\t"),
        "seed_metrics": pd.read_csv(paths["domain_seed_metrics"], sep="\t"),
        "paths": paths,
    }


def validate_inputs(inputs: dict[str, Any], config: dict[str, Any]) -> None:
    predictions = inputs["predictions"]
    seed_metrics = inputs["seed_metrics"]
    expected = config["expected"]
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    primary_seed = int(config["scope"]["primary_seed"])

    prediction_required = {
        "dataset_key",
        "depth_fraction",
        "representation",
        "barcode",
        "truth_label",
        "predicted_cluster",
        "clustering_seed",
    }
    if not prediction_required.issubset(predictions.columns):
        raise ValueError(
            f"Domain predictions lack columns {sorted(prediction_required - set(predictions.columns))}"
        )
    seed_required = {
        "ARI",
        "NMI",
        "dataset_key",
        "depth_fraction",
        "representation",
        "clustering_seed",
        "clustering_seed_role",
    }
    if not seed_required.issubset(seed_metrics.columns):
        raise ValueError(
            f"Seed metrics lack columns {sorted(seed_required - set(seed_metrics.columns))}"
        )
    if set(predictions["clustering_seed"].astype(int).unique().tolist()) != {primary_seed}:
        raise ValueError("Prediction table must contain only the primary clustering seed")
    unit_keys = (
        predictions[["dataset_key", "depth_fraction", "representation"]]
        .drop_duplicates()
        .shape[0]
    )
    if unit_keys != int(expected["units_total"]):
        raise ValueError(f"Prediction table has {unit_keys} units, expected {expected['units_total']}")
    if len(seed_metrics) != int(expected["seed_metrics_rows"]):
        raise ValueError(
            f"Seed metrics have {len(seed_metrics)} rows, expected {expected['seed_metrics_rows']}"
        )
    if seed_metrics.duplicated(
        ["dataset_key", "depth_fraction", "representation", "clustering_seed"]
    ).any():
        raise ValueError("Seed metrics contain duplicate unit/seed rows")
    if set(seed_metrics["dataset_key"].astype(str).unique().tolist()) != set(
        config["scope"]["datasets"]
    ):
        raise ValueError("Seed metrics datasets do not match the config")
    if not np.isfinite(seed_metrics[["ARI", "NMI"]].to_numpy(dtype=float)).all():
        raise ValueError("Seed metrics contain non-finite ARI/NMI")
    if not predictions["truth_label"].notna().all() or not predictions["predicted_cluster"].notna().all():
        raise ValueError("Prediction table contains missing labels")
    if predictions.duplicated(["dataset_key", "depth_fraction", "representation", "barcode"]).any():
        raise ValueError("Prediction table contains duplicate unit/barcode rows")
    if predictions.groupby(["dataset_key", "depth_fraction", "representation"]).size().min() < 2:
        raise ValueError("Every prediction unit needs at least two spots")

    # Bind the per-spot observed metric to the frozen primary-seed table.
    seed_primary = seed_metrics[
        seed_metrics["clustering_seed"].astype(int) == primary_seed
    ]
    if len(seed_primary) != int(expected["primary_seed_units"]):
        raise ValueError("Primary-seed metric row count does not match the freeze")
    seed_lookup = {
        (str(row.dataset_key), float(row.depth_fraction), str(row.representation)): (
            float(row.ARI),
            float(row.NMI),
        )
        for row in seed_primary.itertuples(index=False)
    }
    for (dataset_key, depth_fraction, representation), unit in predictions.groupby(
        ["dataset_key", "depth_fraction", "representation"], sort=True
    ):
        key = (str(dataset_key), float(depth_fraction), str(representation))
        if key not in seed_lookup:
            raise ValueError(f"Primary-seed metrics lack unit {key}")
        ari, nmi, _, _ = observed_metrics(unit["truth_label"], unit["predicted_cluster"])
        expected_ari, expected_nmi = seed_lookup[key]
        if abs(ari - expected_ari) > tolerance or abs(nmi - expected_nmi) > tolerance:
            raise ValueError(
                f"Observed metric mismatch for {key}: ARI {ari} vs {expected_ari}, NMI {nmi} vs {expected_nmi}"
            )


def self_check(
    config: dict[str, Any],
    attempt_root: Path,
    freeze_checks: list[dict[str, Any]],
    inputs: dict[str, Any],
    unit_table: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    seed_sensitivity: pd.DataFrame,
    verdicts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = config["expected"]
    primary = unit_table[unit_table["role"] == "primary"]
    sensitivity = unit_table[unit_table["role"] != "primary"]
    checks = {
        "freeze_hashes_match": all(item["match"] for item in freeze_checks),
        "units_total_match": int(unit_table["unit_id"].nunique()) == int(expected["units_total"]),
        "primary_units_match": int(primary["unit_id"].nunique()) == int(expected["primary_units"]),
        "sensitivity_units_match": int(sensitivity["unit_id"].nunique()) == int(expected["sensitivity_units"]),
        "primary_cells_match": len(primary) == int(expected["primary_cells"]),
        "sensitivity_cells_match": len(sensitivity) == int(expected["sensitivity_cells"]),
        "hypotheses_present": set(verdicts) == {"H1", "H2", "H3", "H4", "H5"},
        "permutation_draws_match": bool(
            (unit_table["permutation_draws"] == int(config["thresholds"]["permutation_draws"])).all()
        ),
        "empirical_p_in_range": bool(unit_table["empirical_p"].between(0.0, 1.0).all()),
        "primary_bh_q_in_range": bool(primary["bh_q"].between(0.0, 1.0).all()),
        "null_quantiles_ordered": bool(
            (unit_table["null_q05"] <= unit_table["null_median"]).all()
            and (unit_table["null_median"] <= unit_table["null_q95"]).all()
            and (unit_table["null_q95"] <= unit_table["null_q99"]).all()
        ),
        "dataset_loo_rows_match": len(dataset_loo) == int(expected["dataset_loo_rows"]),
        "seed_sensitivity_rows_match": len(seed_sensitivity) == int(expected["seed_sensitivity_rows"]),
        "no_duplicate_unit_endpoint": bool(not unit_table.duplicated(["unit_id", "endpoint"]).any()),
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
        LOGGER.info("NM-NULL1 start %s", started_utc)
        freeze_checks = verify_freeze(config, root)
        LOGGER.info("Freeze hashes verified: %s", len(freeze_checks))
        inputs = load_inputs(config, root)
        validate_inputs(inputs, config)
        LOGGER.info("Frozen DS1 inputs validated")
        unit_table = build_unit_table(inputs, config)
        dataset_loo = build_dataset_loo(unit_table, config)
        seed_sensitivity = build_seed_sensitivity(inputs, unit_table, config)
        verdicts = hypothesis_verdicts(unit_table, dataset_loo, config)
        key = key_numbers(unit_table, dataset_loo, seed_sensitivity, verdicts, config)

        write_tsv(unit_table, attempt_root / "tables" / "null1_unit_metrics.tsv")
        write_tsv(dataset_loo, attempt_root / "tables" / "null1_dataset_loo.tsv")
        write_tsv(seed_sensitivity, attempt_root / "tables" / "null1_seed_sensitivity.tsv")
        write_tsv(
            pd.DataFrame(
                [
                    {
                        "hypothesis": hypothesis,
                        "hypothesis_text": payload["hypothesis"],
                        "verdict": payload["verdict"],
                        "details_json": json.dumps(
                            {k: v for k, v in payload.items() if k not in {"hypothesis", "verdict"}},
                            sort_keys=True,
                        ),
                    }
                    for hypothesis, payload in verdicts.items()
                ]
            ),
            attempt_root / "tables" / "null1_hypotheses.tsv",
        )
        write_json(key, attempt_root / "key_numbers.json")
        write_tsv(
            pd.DataFrame(freeze_checks),
            attempt_root / "manifests" / "input_checksums.tsv",
        )

        self_check(
            config,
            attempt_root,
            freeze_checks,
            inputs,
            unit_table,
            dataset_loo,
            seed_sensitivity,
            verdicts,
        )
        runtime = time.perf_counter() - started
        peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        freeze_manifest = json.loads(
            resolve_project_path(root, config["protocol"]["freeze_manifest"]).read_text(
                encoding="utf-8"
            )
        )
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
            "preregistration_sha256": str(freeze_manifest["preregistration"]["sha256"]),
            "freeze_manifest_path": str(config["protocol"]["freeze_manifest"]),
            "freeze_manifest_sha256": sha256_file(
                resolve_project_path(root, config["protocol"]["freeze_manifest"])
            ),
            "seed_permutation": int(config["seeds"]["permutation"]),
            "parameters": {
                "permutation_draws": int(config["thresholds"]["permutation_draws"]),
                "alpha": float(config["thresholds"]["alpha"]),
                "bh_alpha": float(config["thresholds"]["bh_alpha"]),
                "chance_coverage": float(config["thresholds"]["chance_coverage"]),
                "positive_excess_coverage": float(config["thresholds"]["positive_excess_coverage"]),
                "null_center_tolerance": float(config["thresholds"]["null_center_tolerance"]),
                "numeric_tolerance": float(config["thresholds"]["numeric_tolerance"]),
                "primary_seed": int(config["scope"]["primary_seed"]),
            },
            "input_hashes": {
                str(item["path"]): str(item["sha256"]) for item in freeze_manifest["inputs"]
            },
            "software_versions": software_versions(),
            "resource_policy": {
                "workers": int(config["resources"]["workers"]),
                "nice": int(config["resources"]["nice"]),
                "memory_limit_mb": int(config["resources"]["memory_limit_mb"]),
            },
            "outputs": {
                "unit_metrics": "tables/null1_unit_metrics.tsv",
                "dataset_loo": "tables/null1_dataset_loo.tsv",
                "seed_sensitivity": "tables/null1_seed_sensitivity.tsv",
                "hypotheses": "tables/null1_hypotheses.tsv",
                "key_numbers": "key_numbers.json",
            },
        }
        write_json(manifest, attempt_root / "manifests" / "run_manifest.json")
        LOGGER.info("NM-NULL1 complete: %s", json.dumps(verdicts, sort_keys=True))
        write_checksum_table(attempt_root)
        verification = verify_attempt(attempt_root, write=True)
        if (
            not verification["all_hashes_match"]
            or not verification["coverage_all_non_terminal_files"]
        ):
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


__all__ = [
    "ari_from_counts",
    "bh_adjust",
    "build_dataset_loo",
    "build_seed_sensitivity",
    "build_unit_table",
    "hypothesis_verdicts",
    "key_numbers",
    "load_inputs",
    "nmi_from_counts",
    "observed_metrics",
    "permutation_null",
    "run",
    "self_check",
    "validate_inputs",
    "verify_freeze",
]
