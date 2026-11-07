"""NM-TNULL1 whole-tissue toroidal-translation null for frozen DS1 concordance.

The audit reads only frozen DS1 per-spot predictions, frozen DS1 spatial
coordinates, frozen seed metrics, the frozen NM-NULL1 global-null table, and the
authoritative NM-SNULL1 local-block table. It does not construct a
representation, execute a method, access RNA values, or inspect new outcomes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spatialgabench.nm_null1 import (
    ari_from_counts,
    bh_adjust,
    nmi_from_counts,
    observed_metrics,
    software_versions,
)
from spatialgabench.nm_rank1 import (
    ensure_attempt_root,
    load_config,
    resolve_project_path,
    sha256_file,
    stable_seed,
    utc_now,
    verify_attempt,
    write_checksum_table,
    write_json,
    write_tsv,
)
from spatialgabench.nm_snull1 import (
    ESTIMABLE,
    NOT_ESTIMABLE,
    NOT_ESTIMABLE_COORDINATE_CONTRACT,
    _unit_id,
    build_unused_coordinate_table,
    coordinate_contract_audit,
    load_coordinates,
)


LOGGER = logging.getLogger("nm_tnull1")

NOT_ESTIMABLE_SECTION_CONTRACT = "section_contract_failed"
NOT_ESTIMABLE_NULL_DESIGN = "null_design_audit_failed"
NOT_ESTIMABLE_MISSING_COMPARISON = "missing_comparison_row"


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------


def load_inputs(config: dict[str, Any], root: Path) -> dict[str, Any]:
    predictions = pd.read_csv(
        resolve_project_path(root, config["inputs"]["domain_predictions"]),
        sep="\t",
    )
    seed_metrics = pd.read_csv(
        resolve_project_path(root, config["inputs"]["domain_seed_metrics"]),
        sep="\t",
    )
    global_null = pd.read_csv(
        resolve_project_path(root, config["inputs"]["global_null_metrics"]),
        sep="\t",
    )
    local_null = pd.read_csv(
        resolve_project_path(root, config["inputs"]["local_null_metrics"]),
        sep="\t",
    )
    coordinates, coordinate_audit = load_coordinates(config, root)
    return {
        "predictions": predictions,
        "seed_metrics": seed_metrics,
        "global_null": global_null,
        "local_null": local_null,
        "coordinates": coordinates,
        "coordinate_audit": coordinate_audit,
    }


def validate_inputs(inputs: dict[str, Any], config: dict[str, Any]) -> None:
    predictions = inputs["predictions"]
    seed_metrics = inputs["seed_metrics"]
    global_null = inputs["global_null"]
    local_null = inputs["local_null"]

    required_predictions = {
        "dataset_key",
        "depth_fraction",
        "representation",
        "barcode",
        "truth_label",
        "predicted_cluster",
        "clustering_seed",
    }
    required_seed = {
        "dataset_key",
        "depth_fraction",
        "representation",
        "clustering_seed",
        "clustering_seed_role",
        "ARI",
        "NMI",
    }
    required_global = {
        "unit_id",
        "dataset_key",
        "depth_fraction",
        "representation",
        "role",
        "endpoint",
        "observed",
        "null_median",
        "null_q95",
        "above_q95",
        "bh_significant",
    }
    required_local = {
        "unit_id",
        "endpoint",
        "block_factor",
        "role",
        "estimability",
        "null_q95",
        "above_q95",
        "descriptive_bh_significant",
    }
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("Prediction table is missing required columns")
    if not required_seed.issubset(seed_metrics.columns):
        raise ValueError("Seed-metric table is missing required columns")
    if not required_global.issubset(global_null.columns):
        raise ValueError("Global-null table is missing required columns")
    if not required_local.issubset(local_null.columns):
        raise ValueError("Local-null table is missing required columns")
    if predictions.duplicated(
        ["dataset_key", "depth_fraction", "representation", "barcode"]
    ).any():
        raise ValueError("Prediction table contains duplicate unit/barcode rows")
    if (
        predictions.groupby(
            ["dataset_key", "depth_fraction", "representation"]
        ).size().min()
        < 2
    ):
        raise ValueError("Every prediction unit needs at least two spots")

    unit_audit = coordinate_contract_audit(predictions, inputs["coordinates"])
    missing_coordinate_predictions = unit_audit[
        unit_audit["n_missing_coordinate_barcodes"] > 0
    ]
    if len(missing_coordinate_predictions):
        bad = ", ".join(missing_coordinate_predictions["unit_id"].astype(str))
        raise ValueError(f"Prediction barcodes lack frozen coordinates for: {bad}")
    inputs["unit_coordinate_audit"] = unit_audit
    inputs["unused_coordinate_barcodes"] = build_unused_coordinate_table(
        predictions, inputs["coordinates"]
    )

    for dataset_key, frame in inputs["coordinates"].items():
        observed = set(
            predictions.loc[
                predictions["dataset_key"].astype(str) == dataset_key, "barcode"
            ].astype(str)
        )
        available = set(frame["barcode"].astype(str))
        inputs["coordinate_audit"][dataset_key].update(
            {
                "prediction_barcodes": int(len(observed)),
                "unused_coordinate_barcodes": int(len(available - observed)),
                "missing_coordinate_barcodes": int(len(observed - available)),
                "coordinate_barcodes_exact": bool(observed == available),
            }
        )

    primary_seed = int(config["scope"]["primary_seed"])
    primary_seed_metrics = seed_metrics[
        seed_metrics["clustering_seed"].astype(int) == primary_seed
    ]
    if len(primary_seed_metrics) != int(config["expected"]["primary_seed_units"]):
        raise ValueError("Primary-seed metric row count does not match the freeze")

    global_primary = global_null[global_null["role"].astype(str) == "primary"]
    if len(global_primary) != int(config["expected"]["primary_cells"]):
        raise ValueError("Global primary-null row count does not match the freeze")

    local_primary = local_null[
        (local_null["role"].astype(str) == "primary")
        & np.isclose(
            local_null["block_factor"].astype(float),
            float(config["local_reference"]["primary_block_factor"]),
            atol=float(config["thresholds"]["numeric_tolerance"]),
            rtol=0.0,
        )
    ]
    if len(local_primary) != int(config["expected"]["primary_cells"]):
        raise ValueError("Local primary-null row count does not match the freeze")
    inputs["global_primary"] = global_primary.copy()
    inputs["local_primary"] = local_primary.copy()


# ---------------------------------------------------------------------------
# Basic label / null helpers
# ---------------------------------------------------------------------------


def _codes(values: pd.Series) -> np.ndarray:
    codes, uniques = pd.factorize(values.astype(str), sort=True)
    if len(uniques) < 2:
        raise ValueError("A unit must have at least two observed label classes")
    return codes.astype(np.int64)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def _categorical_moran_i(codes: np.ndarray, neighbor_index: np.ndarray) -> float:
    """Mean one-vs-rest Moran's I for a categorical label field.

    ``neighbor_index`` is an ``n x k`` array of row-neighbor indices with
    row-standardized binary weights. The function returns the mean Moran's I
    across classes with non-zero variance. It is a spatial-autocorrelation
    summary for a categorical field, not an inferential statistic.
    """

    n = int(len(codes))
    if n == 0 or neighbor_index.size == 0:
        return 0.0
    values: list[float] = []
    for label in np.unique(codes):
        indicator = (codes == label).astype(np.float64)
        centered = indicator - float(indicator.mean())
        denominator = float(np.sum(centered * centered))
        if denominator <= 0.0:
            continue
        neighbor_mean = centered[neighbor_index].mean(axis=1)
        values.append(float(np.sum(centered * neighbor_mean) / denominator))
    if not values:
        return 0.0
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Section geometry and the translation null
# ---------------------------------------------------------------------------


def build_section_geometry(
    points: np.ndarray,
    min_section_spots: int,
    moran_k: int,
) -> dict[str, Any]:
    """Build torus geometry and a fixed kNN graph for one spatial section."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("Section coordinates are not two-dimensional")
    points = points[:, :2]
    n = int(len(points))
    if n < int(min_section_spots):
        raise ValueError(
            f"Section has {n} spots, below the frozen minimum of {min_section_spots}"
        )
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nn_median = float(np.median(distances[:, 1]))
    if not np.isfinite(nn_median) or nn_median <= 0.0:
        raise ValueError("Section nearest-neighbor distance is not positive")
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width = float(x_max - x_min + nn_median)
    height = float(y_max - y_min + nn_median)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Section torus extent is not positive")
    k = min(int(moran_k), n - 1)
    _, neighbor_index = tree.query(points, k=k + 1)
    neighbor_index = neighbor_index[:, 1:]
    return {
        "points": points,
        "tree": tree,
        "n_spots": n,
        "x_min": float(x_min),
        "y_min": float(y_min),
        "width": width,
        "height": height,
        "nn_median": nn_median,
        "neighbor_index": neighbor_index,
        "moran_k": int(k),
    }


def draw_translation(
    rng: np.random.Generator,
    width: float,
    height: float,
    min_shift_fraction: float,
) -> tuple[float, float]:
    """Draw a uniform torus translation with a minimum displacement."""

    minimum = float(min_shift_fraction) * max(float(width), float(height))
    for _ in range(100000):
        dx = float(rng.uniform(0.0, width))
        dy = float(rng.uniform(0.0, height))
        tx = min(dx, width - dx)
        ty = min(dy, height - dy)
        if math.hypot(tx, ty) >= minimum:
            return dx, dy
    raise RuntimeError("Could not draw a translation satisfying the minimum shift rule")


def shift_permutation(
    points: np.ndarray,
    tree: cKDTree,
    dx: float,
    dy: float,
    x_min: float,
    y_min: float,
    width: float,
    height: float,
    rng: np.random.Generator,
    candidate_ks: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an exact label permutation induced by a torus translation.

    The shifted prediction at each observed spot is taken from the nearest
    observed spot to the wrapped query coordinate. Collisions are repaired
    greedily with expanding candidate neighborhoods so the result is always a
    bijection of the observed predicted labels.
    """

    points = np.asarray(points, dtype=np.float64)
    n = int(len(points))
    query = points.copy()
    query[:, 0] = x_min + np.mod(points[:, 0] - x_min - dx, width)
    query[:, 1] = y_min + np.mod(points[:, 1] - y_min - dy, height)

    nearest_distance, nearest = tree.query(query, k=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    counts = np.bincount(nearest, minlength=n)

    permutation = -np.ones(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    rank_assigned = np.zeros(n, dtype=np.int64)
    fallback = np.zeros(n, dtype=bool)

    # Common case: the nearest-neighbor map is already a bijection.
    if len(np.unique(nearest)) == n:
        permutation = nearest.copy()
        return permutation, {
            "permutation_is_bijection": True,
            "n_rank_gt1": 0,
            "rank_gt1_fraction": 0.0,
            "n_fallback": 0,
            "fallback_fraction": 0.0,
            "mean_target_nn_distance": float(np.mean(nearest_distance)),
            "median_target_nn_distance": float(np.median(nearest_distance)),
            "max_target_nn_distance": float(np.max(nearest_distance)),
            "n_distinct_sources": int(n),
        }

    unique_mask = counts[nearest] == 1
    permutation[unique_mask] = nearest[unique_mask]
    used[nearest[unique_mask]] = True

    duplicate_points = np.flatnonzero(~unique_mask)
    groups: dict[int, list[int]] = {}
    for index in duplicate_points:
        groups.setdefault(int(nearest[index]), []).append(int(index))
    unresolved: list[int] = []
    for source, indices in groups.items():
        order = rng.permutation(len(indices))
        chosen = int(indices[int(order[0])])
        permutation[chosen] = source
        used[source] = True
        unresolved.extend(int(indices[int(value)]) for value in order[1:])

    unresolved_array = np.asarray(sorted(unresolved), dtype=np.int64)
    for candidate_k in candidate_ks:
        if unresolved_array.size == 0:
            break
        k = min(int(candidate_k), n)
        _, candidates = tree.query(query[unresolved_array], k=k)
        if k == 1:
            candidates = candidates[:, None]
        still_unresolved: list[int] = []
        for position, index in enumerate(unresolved_array):
            assigned = False
            for rank in range(int(k)):
                source = int(candidates[position, rank])
                if not used[source]:
                    permutation[index] = source
                    used[source] = True
                    rank_assigned[index] = rank
                    assigned = True
                    break
            if not assigned:
                still_unresolved.append(int(index))
        unresolved_array = np.asarray(still_unresolved, dtype=np.int64)

    if unresolved_array.size:
        unused = np.flatnonzero(~used)
        rng.shuffle(unused)
        if len(unused) < unresolved_array.size:
            raise RuntimeError("Translation permutation repair exhausted its sources")
        for position, index in enumerate(unresolved_array):
            source = int(unused[position])
            permutation[index] = source
            used[source] = True
            fallback[index] = True

    if len(np.unique(permutation)) != n or (permutation < 0).any():
        raise RuntimeError("Translation did not produce an exact label permutation")

    chosen_distance = np.linalg.norm(points[permutation] - query, axis=1)
    rank_gt1 = int((rank_assigned > 0).sum())
    n_fallback = int(fallback.sum())
    return permutation, {
        "permutation_is_bijection": True,
        "n_rank_gt1": rank_gt1,
        "rank_gt1_fraction": rank_gt1 / float(n),
        "n_fallback": n_fallback,
        "fallback_fraction": n_fallback / float(n),
        "mean_target_nn_distance": float(np.mean(chosen_distance)),
        "median_target_nn_distance": float(np.median(chosen_distance)),
        "max_target_nn_distance": float(np.max(chosen_distance)),
        "n_distinct_sources": int(len(np.unique(permutation))),
    }


# ---------------------------------------------------------------------------
# Frozen comparison lookups
# ---------------------------------------------------------------------------


def _seed_lookup(
    seed_metrics: pd.DataFrame, primary_seed: int
) -> dict[tuple[str, float, str], tuple[float, float]]:
    primary = seed_metrics[
        seed_metrics["clustering_seed"].astype(int) == int(primary_seed)
    ]
    return {
        (
            str(row.dataset_key),
            float(row.depth_fraction),
            str(row.representation),
        ): (float(row.ARI), float(row.NMI))
        for row in primary.itertuples(index=False)
    }


def _comparison_lookups(
    global_primary: pd.DataFrame,
    local_primary: pd.DataFrame,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    global_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in global_primary.itertuples(index=False):
        global_lookup[(str(row.unit_id), str(row.endpoint))] = {
            "observed": float(row.observed),
            "null_median": float(row.null_median),
            "null_q95": float(row.null_q95),
            "above_q95": bool(row.above_q95),
            "bh_significant": bool(row.bh_significant),
        }
    local_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in local_primary.itertuples(index=False):
        local_lookup[(str(row.unit_id), str(row.endpoint))] = {
            "null_q95": float(row.null_q95)
            if np.isfinite(float(row.null_q95))
            else float("nan"),
            "above_q95": bool(row.above_q95),
            "descriptive_bh_significant": bool(row.descriptive_bh_significant)
            if pd.notna(row.descriptive_bh_significant)
            else False,
            "estimability": str(row.estimability),
        }
    return global_lookup, local_lookup


def _global_fields(
    unit_id: str, endpoint: str, lookup: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    row = lookup.get((unit_id, endpoint))
    if row is None:
        return {
            "global_observed": float("nan"),
            "global_null_median": float("nan"),
            "global_null_q95": float("nan"),
            "whole_minus_global_q95": float("nan"),
            "global_above_q95": False,
            "global_bh_significant": False,
        }
    return {
        "global_observed": float(row["observed"]),
        "global_null_median": float(row["null_median"]),
        "global_null_q95": float(row["null_q95"]),
        "whole_minus_global_q95": float(row["null_q95"]),
        "global_above_q95": bool(row["above_q95"]),
        "global_bh_significant": bool(row["bh_significant"]),
    }


def _local_fields(
    unit_id: str, endpoint: str, lookup: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    row = lookup.get((unit_id, endpoint))
    if row is None:
        return {
            "local_null_q95": float("nan"),
            "whole_minus_local_q95": float("nan"),
            "local_above_q95": False,
            "local_descriptive_bh_significant": False,
            "local_estimability": "MISSING",
        }
    return {
        "local_null_q95": float(row["null_q95"]),
        "whole_minus_local_q95": float("nan"),
        "local_above_q95": bool(row["above_q95"]),
        "local_descriptive_bh_significant": bool(row["descriptive_bh_significant"]),
        "local_estimability": str(row["estimability"]),
    }


def _base_unit_fields(
    unit_id: str,
    dataset_key: str,
    depth_fraction: float,
    representation: str,
    role: str,
    analysis_role: str,
    coordinate_row: pd.Series,
) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "dataset_key": dataset_key,
        "depth_fraction": float(depth_fraction),
        "representation": representation,
        "role": role,
        "analysis_role": analysis_role,
        "coordinate_barcodes_exact": bool(coordinate_row["coordinate_barcodes_exact"]),
        "n_prediction_barcodes": int(coordinate_row["n_prediction_barcodes"]),
        "n_coordinate_barcodes": int(coordinate_row["n_coordinate_barcodes"]),
        "n_missing_coordinate_barcodes": int(
            coordinate_row["n_missing_coordinate_barcodes"]
        ),
        "n_unused_coordinate_barcodes": int(
            coordinate_row["n_unused_coordinate_barcodes"]
        ),
    }


def _null_row(
    base: dict[str, Any],
    n_spots: int,
    n_truth: int,
    n_pred: int,
    endpoint: str,
    observed: float,
    null_values: np.ndarray,
    seed: int,
    global_fields: dict[str, Any],
    local_fields: dict[str, Any],
    null_design_pass: bool,
    null_design_reason: str,
) -> dict[str, Any]:
    null_median = float(np.median(null_values))
    null_q05 = _quantile(null_values, 0.05)
    null_q95 = _quantile(null_values, 0.95)
    null_q99 = _quantile(null_values, 0.99)
    empirical_p = float(
        (1.0 + float(np.sum(null_values >= observed)))
        / (1.0 + float(len(null_values)))
    )
    row = dict(base)
    row.update(
        {
            "estimability": ESTIMABLE,
            "not_estimable_reason": "",
            "n_spots": int(n_spots),
            "n_truth_clusters": int(n_truth),
            "n_predicted_clusters": int(n_pred),
            "endpoint": endpoint,
            "observed": float(observed),
            "null_median": null_median,
            "null_q05": null_q05,
            "null_q95": null_q95,
            "null_q99": null_q99,
            "null_min": float(np.min(null_values)),
            "null_max": float(np.max(null_values)),
            "observed_minus_median": float(observed) - null_median,
            "observed_minus_q95": float(observed) - null_q95,
            "empirical_p": empirical_p,
            "permutation_draws": int(len(null_values)),
            "permutation_seed": int(seed),
            "above_q95": bool(float(observed) > null_q95),
            "null_design_pass": bool(null_design_pass),
            "null_design_reason": null_design_reason,
        }
    )
    row.update(global_fields)
    row.update(local_fields)
    row["whole_minus_global_q95"] = null_q95 - float(
        global_fields["global_null_q95"]
    )
    row["whole_minus_local_q95"] = null_q95 - float(local_fields["local_null_q95"])
    return row


def _not_estimable_row(
    base: dict[str, Any],
    n_spots: int,
    n_truth: int,
    n_pred: int,
    endpoint: str,
    observed: float,
    reason: str,
    seed: int,
    global_fields: dict[str, Any],
    local_fields: dict[str, Any],
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "estimability": NOT_ESTIMABLE,
            "not_estimable_reason": reason,
            "n_spots": int(n_spots),
            "n_truth_clusters": int(n_truth),
            "n_predicted_clusters": int(n_pred),
            "endpoint": endpoint,
            "observed": float(observed),
            "null_median": float("nan"),
            "null_q05": float("nan"),
            "null_q95": float("nan"),
            "null_q99": float("nan"),
            "null_min": float("nan"),
            "null_max": float("nan"),
            "observed_minus_median": float("nan"),
            "observed_minus_q95": float("nan"),
            "empirical_p": float("nan"),
            "permutation_draws": 0,
            "permutation_seed": int(seed),
            "above_q95": False,
            "null_design_pass": False,
            "null_design_reason": reason,
        }
    )
    row.update(global_fields)
    row.update(local_fields)
    row["whole_minus_global_q95"] = float("nan")
    row["whole_minus_local_q95"] = float("nan")
    return row


# ---------------------------------------------------------------------------
# Unit-level null construction
# ---------------------------------------------------------------------------


def build_unit_table(
    inputs: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build unit metrics, per-draw null values, section audits, design audits."""

    predictions = inputs["predictions"]
    coordinates = inputs["coordinates"]
    primary_seed = int(config["scope"]["primary_seed"])
    sensitivity = config["scope"]["sensitivity"]
    sensitivity_dataset = str(sensitivity["dataset_key"])
    sensitivity_depth = float(sensitivity["depth_fraction"])
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    draws = int(config["thresholds"]["permutation_draws"])
    base_seed = int(config["seeds"]["permutation"])
    audit_draws = int(config["thresholds"]["autocorrelation_audit_draws"])
    min_section_spots = int(config["translation"]["min_section_spots"])
    moran_k = int(config["translation"]["moran_k"])
    min_shift_fraction = float(config["translation"]["min_shift_fraction"])
    candidate_ks = [int(value) for value in config["translation"]["candidate_ks"]]
    max_fallback_fraction = float(
        config["translation"]["max_fallback_fraction"]
    )
    max_rank_gt1_fraction = float(
        config["translation"]["max_rank_gt1_fraction"]
    )
    max_median_abs_moran_delta = float(
        config["translation"]["max_median_abs_moran_delta"]
    )
    max_p90_abs_moran_delta = float(
        config["translation"]["max_p90_abs_moran_delta"]
    )
    max_unit_median_displacement_ratio = float(
        config["translation"]["max_unit_median_displacement_ratio"]
    )
    max_unit_p90_displacement_ratio = float(
        config["translation"]["max_unit_p90_displacement_ratio"]
    )
    endpoints = list(config["endpoints"])

    seed_lookup = _seed_lookup(inputs["seed_metrics"], primary_seed)
    global_lookup, local_lookup = _comparison_lookups(
        inputs["global_primary"], inputs["local_primary"]
    )
    coordinate_lookup = inputs["unit_coordinate_audit"].set_index("unit_id")

    rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []

    for (dataset_key, depth_fraction, representation), unit in predictions.groupby(
        ["dataset_key", "depth_fraction", "representation"], sort=True
    ):
        dataset_key = str(dataset_key)
        depth_fraction = float(depth_fraction)
        representation = str(representation)
        unit_id = _unit_id(dataset_key, depth_fraction, representation)
        is_sensitivity = (
            dataset_key == sensitivity_dataset
            and math.isclose(depth_fraction, sensitivity_depth, abs_tol=tolerance)
        )
        role = str(sensitivity["role"]) if is_sensitivity else "primary"
        analysis_role = "NONCOMPARABLE_SENSITIVITY_ONLY" if is_sensitivity else "primary"
        coordinate_row = coordinate_lookup.loc[unit_id]
        base = _base_unit_fields(
            unit_id,
            dataset_key,
            depth_fraction,
            representation,
            role,
            analysis_role,
            coordinate_row,
        )
        unit = unit.sort_values("barcode").reset_index(drop=True)
        truth = unit["truth_label"].astype(str)
        predicted = unit["predicted_cluster"].astype(str)
        n_truth = int(truth.nunique())
        n_pred = int(predicted.nunique())
        ari, nmi, _, _ = observed_metrics(truth, predicted)
        unit_seed = stable_seed(base_seed, f"{unit_id}__tnull")

        if coordinate_row["estimability"] != ESTIMABLE:
            for endpoint, observed in (("ARI", ari), ("NMI", nmi)):
                rows.append(
                    _not_estimable_row(
                        base,
                        len(unit),
                        n_truth,
                        n_pred,
                        endpoint,
                        observed,
                        NOT_ESTIMABLE_COORDINATE_CONTRACT,
                        unit_seed,
                        _global_fields(unit_id, endpoint, global_lookup),
                        _local_fields(unit_id, endpoint, local_lookup),
                    )
                )
            design_rows.append(
                {
                    "unit_id": unit_id,
                    "dataset_key": dataset_key,
                    "depth_fraction": depth_fraction,
                    "representation": representation,
                    "role": role,
                    "estimability": NOT_ESTIMABLE,
                    "not_estimable_reason": NOT_ESTIMABLE_COORDINATE_CONTRACT,
                    "n_sections": 0,
                    "n_audit_draws": 0,
                    "max_fallback_fraction": float("nan"),
                    "max_rank_gt1_fraction": float("nan"),
                    "median_abs_moran_delta": float("nan"),
                    "p90_abs_moran_delta": float("nan"),
                    "null_design_pass": False,
                    "null_design_reason": NOT_ESTIMABLE_COORDINATE_CONTRACT,
                }
            )
            continue

        merged = unit.merge(
            coordinates[dataset_key],
            on="barcode",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if merged[["x", "y", "section"]].isna().any().any():
            raise ValueError(f"Coordinate join is incomplete for {unit_id}")

        expected_ari, expected_nmi = seed_lookup[
            (dataset_key, depth_fraction, representation)
        ]
        if abs(ari - expected_ari) > tolerance or abs(nmi - expected_nmi) > tolerance:
            raise ValueError(f"Observed metric mismatch for {unit_id}")

        section_data: list[dict[str, Any]] = []
        section_valid = True
        section_invalid_reason = ""
        for section in sorted(merged["section"].astype(str).unique()):
            indices = merged.index[merged["section"].astype(str) == section].to_numpy()
            section_truth = merged.loc[indices, "truth_label"].astype(str)
            section_predicted = merged.loc[indices, "predicted_cluster"].astype(str)
            n_section_truth = int(section_truth.nunique())
            n_section_pred = int(section_predicted.nunique())
            unique_coordinates = not bool(
                merged.loc[indices, ["x", "y"]].duplicated().any()
            )
            row = {
                "unit_id": unit_id,
                "dataset_key": dataset_key,
                "depth_fraction": depth_fraction,
                "representation": representation,
                "section": section,
                "n_spots": int(len(indices)),
                "x_min": float("nan"),
                "x_max": float("nan"),
                "y_min": float("nan"),
                "y_max": float("nan"),
                "width": float("nan"),
                "height": float("nan"),
                "nn_median": float("nan"),
                "n_truth_classes": n_section_truth,
                "n_predicted_classes": n_section_pred,
                "unique_coordinates": bool(unique_coordinates),
                "section_valid": True,
                "invalid_reason": "",
            }
            try:
                if n_section_truth < 2:
                    raise ValueError("section has fewer than two truth classes")
                if n_section_pred < 2:
                    raise ValueError("section has fewer than two predicted classes")
                if not unique_coordinates:
                    raise ValueError("section has duplicate coordinates")
                geometry = build_section_geometry(
                    merged.loc[indices, ["x", "y"]].to_numpy(dtype=np.float64),
                    min_section_spots,
                    moran_k,
                )
            except ValueError as exc:
                section_valid = False
                section_invalid_reason = str(exc)
                row["section_valid"] = False
                row["invalid_reason"] = section_invalid_reason
                section_rows.append(row)
                continue
            row.update(
                {
                    "x_min": geometry["x_min"],
                    "x_max": float(geometry["x_min"] + geometry["width"] - geometry["nn_median"]),
                    "y_min": geometry["y_min"],
                    "y_max": float(geometry["y_min"] + geometry["height"] - geometry["nn_median"]),
                    "width": geometry["width"],
                    "height": geometry["height"],
                    "nn_median": geometry["nn_median"],
                }
            )
            section_rows.append(row)
            section_data.append(
                {
                    "section": section,
                    "indices": indices,
                    "geometry": geometry,
                }
            )

        if not section_valid:
            for endpoint, observed in (("ARI", ari), ("NMI", nmi)):
                rows.append(
                    _not_estimable_row(
                        base,
                        len(unit),
                        n_truth,
                        n_pred,
                        endpoint,
                        observed,
                        NOT_ESTIMABLE_SECTION_CONTRACT,
                        unit_seed,
                        _global_fields(unit_id, endpoint, global_lookup),
                        _local_fields(unit_id, endpoint, local_lookup),
                    )
                )
            design_rows.append(
                {
                    "unit_id": unit_id,
                    "dataset_key": dataset_key,
                    "depth_fraction": depth_fraction,
                    "representation": representation,
                    "role": role,
                    "estimability": NOT_ESTIMABLE,
                    "not_estimable_reason": NOT_ESTIMABLE_SECTION_CONTRACT,
                    "n_sections": len(section_data),
                    "n_audit_draws": 0,
                    "max_fallback_fraction": float("nan"),
                    "max_rank_gt1_fraction": float("nan"),
                    "median_abs_moran_delta": float("nan"),
                    "p90_abs_moran_delta": float("nan"),
                    "null_design_pass": False,
                    "null_design_reason": section_invalid_reason,
                }
            )
            continue

        truth_codes = _codes(truth)
        pred_codes = _codes(predicted)
        n_truth_codes = int(truth_codes.max()) + 1
        n_pred_codes = int(pred_codes.max()) + 1
        n_samples = int(len(truth_codes))
        truth_counts = np.bincount(
            truth_codes, minlength=n_truth_codes
        ).astype(np.float64)
        pred_counts = np.bincount(
            pred_codes, minlength=n_pred_codes
        ).astype(np.float64)
        observed_moran = {
            item["section"]: _categorical_moran_i(
                pred_codes[item["indices"]],
                item["geometry"]["neighbor_index"],
            )
            for item in section_data
        }

        rng = np.random.default_rng(unit_seed)
        null_ari = np.empty(draws, dtype=np.float64)
        null_nmi = np.empty(draws, dtype=np.float64)
        max_fallback_fraction_observed = 0.0
        max_rank_gt1_fraction_observed = 0.0
        displacement_ratios: list[float] = []
        moran_deltas: list[float] = []
        for draw in range(draws):
            shifted = pred_codes.copy()
            draw_fallback = 0
            draw_rank_gt1 = 0
            draw_spots = 0
            for item in section_data:
                geometry = item["geometry"]
                indices = item["indices"]
                dx, dy = draw_translation(
                    rng,
                    geometry["width"],
                    geometry["height"],
                    min_shift_fraction,
                )
                permutation, audit = shift_permutation(
                    geometry["points"],
                    geometry["tree"],
                    dx,
                    dy,
                    geometry["x_min"],
                    geometry["y_min"],
                    geometry["width"],
                    geometry["height"],
                    rng,
                    candidate_ks,
                )
                shifted[indices] = pred_codes[indices][permutation]
                draw_fallback += int(audit["n_fallback"])
                draw_rank_gt1 += int(audit["n_rank_gt1"])
                draw_spots += int(len(indices))
                displacement_ratios.append(
                    float(audit["median_target_nn_distance"])
                    / float(geometry["nn_median"])
                )
                if draw < audit_draws:
                    shifted_moran = _categorical_moran_i(
                        shifted[indices], geometry["neighbor_index"]
                    )
                    moran_deltas.append(
                        abs(shifted_moran - observed_moran[item["section"]])
                    )
            counts = np.bincount(
                truth_codes * n_pred_codes + shifted,
                minlength=n_truth_codes * n_pred_codes,
            ).astype(np.float64)
            null_ari[draw] = ari_from_counts(
                counts, n_samples, truth_counts, pred_counts
            )
            null_nmi[draw] = nmi_from_counts(
                counts, n_samples, truth_counts, pred_counts, n_pred_codes
            )
            max_fallback_fraction_observed = max(
                max_fallback_fraction_observed,
                draw_fallback / float(draw_spots),
            )
            max_rank_gt1_fraction_observed = max(
                max_rank_gt1_fraction_observed,
                draw_rank_gt1 / float(draw_spots),
            )
            if (draw + 1) % 500 == 0:
                LOGGER.info(
                    "  translation null progress %d/%d for %s",
                    draw + 1,
                    draws,
                    unit_id,
                )

        delta_array = np.asarray(moran_deltas, dtype=np.float64)
        median_abs_moran_delta = (
            float(np.median(delta_array)) if delta_array.size else float("nan")
        )
        p90_abs_moran_delta = (
            float(np.quantile(delta_array, 0.90))
            if delta_array.size
            else float("nan")
        )
        displacement_array = np.asarray(displacement_ratios, dtype=np.float64)
        median_displacement_ratio = (
            float(np.median(displacement_array))
            if displacement_array.size
            else float("nan")
        )
        p90_displacement_ratio = (
            float(np.quantile(displacement_array, 0.90))
            if displacement_array.size
            else float("nan")
        )
        design_reasons: list[str] = []
        if max_fallback_fraction_observed > max_fallback_fraction:
            design_reasons.append("fallback_fraction_exceeded")
        if max_rank_gt1_fraction_observed > max_rank_gt1_fraction:
            design_reasons.append("rank_gt1_fraction_exceeded")
        if not math.isfinite(median_abs_moran_delta) or (
            median_abs_moran_delta > max_median_abs_moran_delta
        ):
            design_reasons.append("median_moran_delta_exceeded")
        if not math.isfinite(p90_abs_moran_delta) or (
            p90_abs_moran_delta > max_p90_abs_moran_delta
        ):
            design_reasons.append("p90_moran_delta_exceeded")
        if not math.isfinite(median_displacement_ratio) or (
            median_displacement_ratio > max_unit_median_displacement_ratio
        ):
            design_reasons.append("median_displacement_ratio_exceeded")
        if not math.isfinite(p90_displacement_ratio) or (
            p90_displacement_ratio > max_unit_p90_displacement_ratio
        ):
            design_reasons.append("p90_displacement_ratio_exceeded")
        null_design_pass = not design_reasons
        null_design_reason = ";".join(design_reasons)

        if not null_design_pass:
            for endpoint, observed in (("ARI", ari), ("NMI", nmi)):
                rows.append(
                    _not_estimable_row(
                        base,
                        len(unit),
                        n_truth,
                        n_pred,
                        endpoint,
                        observed,
                        NOT_ESTIMABLE_NULL_DESIGN,
                        unit_seed,
                        _global_fields(unit_id, endpoint, global_lookup),
                        _local_fields(unit_id, endpoint, local_lookup),
                    )
                )
            design_rows.append(
                {
                    "unit_id": unit_id,
                    "dataset_key": dataset_key,
                    "depth_fraction": depth_fraction,
                    "representation": representation,
                    "role": role,
                    "estimability": NOT_ESTIMABLE,
                    "not_estimable_reason": NOT_ESTIMABLE_NULL_DESIGN,
                    "n_sections": len(section_data),
                    "n_audit_draws": int(min(audit_draws, draws)),
                    "max_fallback_fraction": float(max_fallback_fraction_observed),
                    "max_rank_gt1_fraction": float(max_rank_gt1_fraction_observed),
                    "median_abs_moran_delta": float(median_abs_moran_delta),
                    "p90_abs_moran_delta": float(p90_abs_moran_delta),
                    "median_displacement_ratio": float(median_displacement_ratio),
                    "p90_displacement_ratio": float(p90_displacement_ratio),
                    "null_design_pass": False,
                    "null_design_reason": null_design_reason,
                }
            )
            continue

        design_rows.append(
            {
                "unit_id": unit_id,
                "dataset_key": dataset_key,
                "depth_fraction": depth_fraction,
                "representation": representation,
                "role": role,
                "estimability": ESTIMABLE,
                "not_estimable_reason": "",
                "n_sections": len(section_data),
                "n_audit_draws": int(min(audit_draws, draws)),
                "max_fallback_fraction": float(max_fallback_fraction_observed),
                "max_rank_gt1_fraction": float(max_rank_gt1_fraction_observed),
                "median_abs_moran_delta": float(median_abs_moran_delta),
                "p90_abs_moran_delta": float(p90_abs_moran_delta),
                "median_displacement_ratio": float(median_displacement_ratio),
                "p90_displacement_ratio": float(p90_displacement_ratio),
                "null_design_pass": True,
                "null_design_reason": "",
            }
        )

        for endpoint, values, observed in (
            ("ARI", null_ari, ari),
            ("NMI", null_nmi, nmi),
        ):
            rows.append(
                _null_row(
                    base,
                    len(unit),
                    n_truth,
                    n_pred,
                    endpoint,
                    observed,
                    values,
                    unit_seed,
                    _global_fields(unit_id, endpoint, global_lookup),
                    _local_fields(unit_id, endpoint, local_lookup),
                    null_design_pass,
                    null_design_reason,
                )
            )
            for draw_index, value in enumerate(values):
                draw_rows.append(
                    {
                        "unit_id": unit_id,
                        "dataset_key": dataset_key,
                        "depth_fraction": depth_fraction,
                        "representation": representation,
                        "role": role,
                        "endpoint": endpoint,
                        "draw_index": int(draw_index),
                        "value": float(value),
                    }
                )

    unit_table = pd.DataFrame(rows)
    draws_table = pd.DataFrame(
        draw_rows,
        columns=[
            "unit_id",
            "dataset_key",
            "depth_fraction",
            "representation",
            "role",
            "endpoint",
            "draw_index",
            "value",
        ],
    )
    section_table = pd.DataFrame(
        section_rows,
        columns=[
            "unit_id",
            "dataset_key",
            "depth_fraction",
            "representation",
            "section",
            "n_spots",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "width",
            "height",
            "nn_median",
            "n_truth_classes",
            "n_predicted_classes",
            "unique_coordinates",
            "section_valid",
            "invalid_reason",
        ],
    )
    design_table = pd.DataFrame(
        design_rows,
        columns=[
            "unit_id",
            "dataset_key",
            "depth_fraction",
            "representation",
            "role",
            "estimability",
            "not_estimable_reason",
            "n_sections",
            "n_audit_draws",
            "max_fallback_fraction",
            "max_rank_gt1_fraction",
            "median_abs_moran_delta",
            "p90_abs_moran_delta",
            "median_displacement_ratio",
            "p90_displacement_ratio",
            "null_design_pass",
            "null_design_reason",
        ],
    )

    # Multiple-testing family: only the 22 estimable primary cells are
    # inferential. The three d2 units are explicit NOT_ESTIMABLE rows.
    primary_mask = unit_table["role"] == "primary"
    estimable_primary_mask = primary_mask & (
        unit_table["estimability"] == ESTIMABLE
    )
    unit_table["preregistered_bh_q"] = np.nan
    unit_table["preregistered_bh_significant"] = False
    unit_table["descriptive_bh_q"] = np.nan
    unit_table["descriptive_bh_significant"] = False
    unit_table["bh_family_scope"] = "NOT_IN_BH_FAMILY"
    unit_table.loc[primary_mask, "bh_family_scope"] = (
        "PREREGISTERED_22_ESTIMABLE_CELL_FAMILY_INCOMPLETE"
    )
    unit_table.loc[estimable_primary_mask, "bh_family_scope"] = (
        "REDUCED_ESTIMABLE_PRIMARY_CELLS_DESCRIPTIVE"
    )
    not_estimable_primary_mask = primary_mask & (
        unit_table["estimability"] == NOT_ESTIMABLE
    )
    expected_not_estimable = set(
        str(value) for value in config["expected"].get("not_estimable_unit_ids", [])
    )
    observed_not_estimable = set(
        unit_table.loc[not_estimable_primary_mask, "unit_id"].astype(str)
    )
    family_complete = bool(
        unit_table.loc[primary_mask, "unit_id"].nunique()
        == int(config["expected"]["primary_units"])
        and len(unit_table.loc[primary_mask])
        == int(config["expected"]["primary_cells"])
        and unit_table.loc[estimable_primary_mask, "unit_id"].nunique()
        == int(config["expected"]["estimable_primary_units"])
        and len(unit_table.loc[estimable_primary_mask])
        == int(config["expected"]["estimable_primary_cells"])
        and observed_not_estimable == expected_not_estimable
    )
    if family_complete:
        unit_table.loc[estimable_primary_mask, "preregistered_bh_q"] = bh_adjust(
            unit_table.loc[estimable_primary_mask, "empirical_p"].to_numpy(
                dtype=np.float64
            )
        )
        unit_table.loc[estimable_primary_mask, "preregistered_bh_significant"] = (
            unit_table.loc[estimable_primary_mask, "preregistered_bh_q"]
            <= float(config["thresholds"]["bh_alpha"])
        )
        unit_table.loc[estimable_primary_mask, "bh_family_scope"] = (
            "PREREGISTERED_22_ESTIMABLE_CELL_FAMILY"
        )
    if bool(estimable_primary_mask.any()):
        unit_table.loc[estimable_primary_mask, "descriptive_bh_q"] = bh_adjust(
            unit_table.loc[estimable_primary_mask, "empirical_p"].to_numpy(
                dtype=np.float64
            )
        )
        unit_table.loc[estimable_primary_mask, "descriptive_bh_significant"] = (
            unit_table.loc[estimable_primary_mask, "descriptive_bh_q"]
            <= float(config["thresholds"]["bh_alpha"])
        )

    unit_table = unit_table.sort_values(
        ["role", "dataset_key", "depth_fraction", "representation", "endpoint"]
    ).reset_index(drop=True)
    draws_table = draws_table.sort_values(
        ["unit_id", "endpoint", "draw_index"]
    ).reset_index(drop=True)
    section_table = section_table.sort_values(
        ["unit_id", "section"]
    ).reset_index(drop=True)
    design_table = design_table.sort_values("unit_id").reset_index(drop=True)
    return unit_table, draws_table, section_table, design_table


# ---------------------------------------------------------------------------
# Hypotheses and comparison tables
# ---------------------------------------------------------------------------


def _h1_endpoint_verdict(
    frame: pd.DataFrame,
    endpoint: str,
    coverage_bar: float,
    bh_column: str,
    require_complete: bool = False,
    expected_units: int | None = None,
) -> tuple[str, float, int, int]:
    subset = frame[
        (frame["endpoint"] == endpoint)
        & (frame["estimability"] == ESTIMABLE)
    ].copy()
    n_units = int(subset["unit_id"].nunique())
    if n_units == 0:
        return "NOT_ESTIMABLE", float("nan"), 0, 0
    separated = int(
        (
            subset["above_q95"].astype(bool)
            & subset[bh_column].astype(bool)
        ).sum()
    )
    coverage = float(separated / n_units)
    verdict = "PASS" if coverage >= float(coverage_bar) else "FAIL"
    if require_complete and expected_units is not None and n_units != int(expected_units):
        verdict = "NOT_ESTIMABLE"
    return verdict, coverage, separated, n_units


def build_dataset_loo(unit_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = unit_table[unit_table["role"] == "primary"].copy()
    alpha = float(config["thresholds"]["bh_alpha"])
    coverage_bar = float(config["thresholds"]["chance_coverage"])
    rows: list[dict[str, Any]] = []
    for omitted in config["scope"]["dataset_loo"]:
        nominal = primary[primary["dataset_key"].astype(str) != str(omitted)].copy()
        subset = nominal[nominal["estimability"] == ESTIMABLE].copy()
        subset["loo_bh_q"] = np.nan
        subset["loo_bh_significant"] = False
        if len(subset):
            subset.loc[:, "loo_bh_q"] = bh_adjust(
                subset["empirical_p"].to_numpy(dtype=np.float64)
            )
            subset.loc[:, "loo_bh_significant"] = (
                subset["loo_bh_q"] <= alpha
            )
        row: dict[str, Any] = {
            "omitted_dataset_key": str(omitted),
            "n_units_remaining_nominal": int(nominal["unit_id"].nunique()),
            "n_units_remaining_estimable": int(subset["unit_id"].nunique()),
            "n_not_estimable_remaining": int(
                nominal.loc[
                    nominal["estimability"] == NOT_ESTIMABLE, "unit_id"
                ].nunique()
            ),
            "n_cells_remaining_estimable": int(len(subset)),
        }
        endpoint_verdicts: list[str] = []
        for endpoint in config["endpoints"]:
            verdict, coverage, separated, n_units = _h1_endpoint_verdict(
                subset,
                endpoint,
                coverage_bar,
                "loo_bh_significant",
            )
            row[f"n_{endpoint.lower()}_units"] = int(n_units)
            row[f"n_{endpoint.lower()}_separated"] = int(separated)
            row[f"coverage_{endpoint.lower()}"] = float(coverage)
            row[f"h1_{endpoint.lower()}_verdict"] = verdict
            endpoint_verdicts.append(verdict)
        if "NOT_ESTIMABLE" in endpoint_verdicts:
            row["h1_verdict"] = "NOT_ESTIMABLE"
        elif all(value == "PASS" for value in endpoint_verdicts):
            row["h1_verdict"] = "PASS"
        else:
            row["h1_verdict"] = "FAIL"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("omitted_dataset_key").reset_index(drop=True)


def _h1_from_draws(
    draws_table: pd.DataFrame,
    unit_table: pd.DataFrame,
    prefix: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    primary = unit_table[
        (unit_table["role"] == "primary")
        & (unit_table["estimability"] == ESTIMABLE)
    ].copy()
    alpha = float(config["thresholds"]["bh_alpha"])
    coverage_bar = float(config["thresholds"]["chance_coverage"])
    rows: list[dict[str, Any]] = []
    for unit_id, unit_rows in primary.groupby("unit_id", sort=True):
        for endpoint in config["endpoints"]:
            endpoint_row = unit_rows[unit_rows["endpoint"] == endpoint]
            if len(endpoint_row) != 1:
                raise ValueError(f"Expected one primary cell for {unit_id} {endpoint}")
            values = draws_table[
                (draws_table["unit_id"].astype(str) == str(unit_id))
                & (draws_table["endpoint"] == endpoint)
                & (draws_table["draw_index"] < int(prefix))
            ]["value"].to_numpy(dtype=np.float64)
            if len(values) != int(prefix):
                raise ValueError(
                    f"Draw prefix mismatch for {unit_id} {endpoint}: {len(values)}"
                )
            observed = float(endpoint_row["observed"].iloc[0])
            null_q95 = float(np.quantile(values, 0.95))
            empirical_p = float(
                (1.0 + float(np.sum(values >= observed)))
                / (1.0 + float(len(values)))
            )
            rows.append(
                {
                    "unit_id": str(unit_id),
                    "endpoint": endpoint,
                    "estimability": ESTIMABLE,
                    "observed": observed,
                    "null_q95": null_q95,
                    "empirical_p": empirical_p,
                    "above_q95": bool(observed > null_q95),
                    "permutation_draws": int(len(values)),
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "unit_id",
            "endpoint",
            "estimability",
            "observed",
            "null_q95",
            "empirical_p",
            "above_q95",
            "permutation_draws",
        ],
    )
    if len(frame) == 0:
        frame["bh_q"] = pd.Series(dtype=float)
        frame["bh_significant"] = pd.Series(dtype=bool)
        return frame, "NOT_ESTIMABLE"
    frame["bh_q"] = bh_adjust(frame["empirical_p"].to_numpy(dtype=np.float64))
    frame["bh_significant"] = frame["bh_q"] <= alpha
    endpoint_verdicts: list[str] = []
    for endpoint in config["endpoints"]:
        verdict, _, _, _ = _h1_endpoint_verdict(
            frame,
            endpoint,
            coverage_bar,
            "bh_significant",
        )
        endpoint_verdicts.append(verdict)
    if "NOT_ESTIMABLE" in endpoint_verdicts:
        overall = "NOT_ESTIMABLE"
    elif all(value == "PASS" for value in endpoint_verdicts):
        overall = "PASS"
    else:
        overall = "FAIL"
    return frame, overall


def build_mc_stability(
    draws_table: pd.DataFrame,
    unit_table: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prefix in config["thresholds"]["mc_prefixes"]:
        frame, overall = _h1_from_draws(
            draws_table, unit_table, int(prefix), config
        )
        row: dict[str, Any] = {
            "draw_prefix": int(prefix),
            "h1_overall_verdict": overall,
        }
        for endpoint in config["endpoints"]:
            verdict, coverage, separated, n_units = _h1_endpoint_verdict(
                frame,
                endpoint,
                float(config["thresholds"]["chance_coverage"]),
                "bh_significant",
            )
            row[f"n_{endpoint.lower()}_units"] = int(n_units)
            row[f"n_{endpoint.lower()}_separated"] = int(separated)
            row[f"coverage_{endpoint.lower()}"] = float(coverage)
            row[f"h1_{endpoint.lower()}_verdict"] = verdict
        rows.append(row)
    return pd.DataFrame(rows).sort_values("draw_prefix").reset_index(drop=True)


def build_comparative_calibration(
    unit_table: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    primary = unit_table[
        (unit_table["role"] == "primary")
        & (unit_table["estimability"] == ESTIMABLE)
    ].copy()
    estimable_primary = unit_table[
        (unit_table["role"] == "primary")
        & (unit_table["estimability"] == ESTIMABLE)
    ]
    family_complete = bool(
        len(estimable_primary) == int(config["expected"]["estimable_primary_cells"])
        and estimable_primary["bh_family_scope"]
        .astype(str)
        .eq("PREREGISTERED_22_ESTIMABLE_CELL_FAMILY")
        .all()
    )
    bh_column = (
        "preregistered_bh_significant" if family_complete else "descriptive_bh_significant"
    )
    rows = primary[
        [
            "unit_id",
            "dataset_key",
            "depth_fraction",
            "representation",
            "endpoint",
            "observed",
            "null_q95",
            "above_q95",
            "global_null_q95",
            "local_null_q95",
            "local_above_q95",
            "local_descriptive_bh_significant",
            "local_estimability",
            "preregistered_bh_significant",
            "descriptive_bh_significant",
        ]
    ].copy()
    rows["whole_separated"] = (
        rows["above_q95"].astype(bool) & rows[bh_column].astype(bool)
    )
    rows["local_separated"] = (
        rows["local_above_q95"].astype(bool)
        & rows["local_descriptive_bh_significant"].astype(bool)
    )
    rows["whole_minus_local_q95"] = (
        rows["null_q95"].astype(float) - rows["local_null_q95"].astype(float)
    )
    rows["whole_minus_global_q95"] = (
        rows["null_q95"].astype(float) - rows["global_null_q95"].astype(float)
    )
    rows["local_comparison_valid"] = (
        np.isfinite(rows["local_null_q95"].astype(float))
        & rows["local_estimability"].eq(ESTIMABLE)
    )
    return rows.sort_values(
        ["dataset_key", "depth_fraction", "representation", "endpoint"]
    ).reset_index(drop=True)


def hypothesis_verdicts(
    unit_table: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    mc_stability: pd.DataFrame,
    comparative: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    coverage_bar = float(config["thresholds"]["chance_coverage"])
    global_bar = float(config["thresholds"]["global_calibration_coverage"])
    local_bar = float(config["thresholds"]["local_comparison_coverage"])
    alpha = float(config["thresholds"]["bh_alpha"])
    expected = config["expected"]

    primary_nominal = unit_table[unit_table["role"] == "primary"].copy()
    primary = primary_nominal[
        primary_nominal["estimability"] == ESTIMABLE
    ].copy()
    family_complete = bool(
        primary_nominal["unit_id"].nunique() == int(expected["primary_units"])
        and len(primary_nominal) == int(expected["primary_cells"])
        and primary["unit_id"].nunique()
        == int(expected["estimable_primary_units"])
        and len(primary) == int(expected["estimable_primary_cells"])
    )

    endpoint_verdicts: dict[str, str] = {}
    h1_details: dict[str, Any] = {}
    for endpoint in config["endpoints"]:
        verdict, coverage, separated, n_units = _h1_endpoint_verdict(
            primary,
            endpoint,
            coverage_bar,
            "preregistered_bh_significant",
            require_complete=True,
            expected_units=int(expected["estimable_primary_units"]),
        )
        endpoint_verdicts[endpoint] = verdict if family_complete else "NOT_ESTIMABLE"
        h1_details[f"n_{endpoint.lower()}_separated"] = (
            int(separated) if family_complete else None
        )
        h1_details[f"n_{endpoint.lower()}_units"] = int(n_units)
        h1_details[f"n_{endpoint.lower()}_units_required"] = int(
            expected["estimable_primary_units"]
        )
        h1_details[f"coverage_{endpoint.lower()}"] = (
            float(coverage) if family_complete else None
        )
    if not family_complete:
        h1_verdict = "NOT_ESTIMABLE"
    elif all(value == "PASS" for value in endpoint_verdicts.values()):
        h1_verdict = "PASS"
    else:
        h1_verdict = "FAIL"

    n_cells = int(len(primary))
    n_ordered = int(
        (
            primary["null_q95"].astype(float)
            > primary["null_median"].astype(float) + tolerance
        ).sum()
    )
    n_spread = int(
        (
            primary["null_q95"].astype(float)
            > primary["null_q05"].astype(float) + tolerance
        ).sum()
    )
    n_above_global = int(
        (
            primary["null_q95"].astype(float)
            + tolerance
            >= primary["global_null_q95"].astype(float)
        ).sum()
    )
    above_global_coverage = (
        float(n_above_global / n_cells) if n_cells else float("nan")
    )
    h2_estimable_pass = bool(
        n_cells > 0
        and n_ordered == n_cells
        and n_spread == n_cells
        and above_global_coverage >= global_bar
    )
    if n_cells == 0:
        h2_verdict = "NOT_ESTIMABLE"
    elif not h2_estimable_pass:
        h2_verdict = "FAIL"
    elif not family_complete:
        h2_verdict = "NOT_ESTIMABLE"
    else:
        h2_verdict = "PASS"

    loo_verdicts = dataset_loo["h1_verdict"].astype(str).tolist()
    if "NOT_ESTIMABLE" in loo_verdicts or h1_verdict == "NOT_ESTIMABLE":
        h3_verdict = "NOT_ESTIMABLE"
    elif len(loo_verdicts) == int(expected["dataset_loo_rows"]) and all(
        value == h1_verdict for value in loo_verdicts
    ):
        h3_verdict = "PASS"
    else:
        h3_verdict = "FAIL"

    mc_verdicts = mc_stability["h1_overall_verdict"].astype(str).tolist()
    if "NOT_ESTIMABLE" in mc_verdicts or h1_verdict == "NOT_ESTIMABLE":
        h4_verdict = "NOT_ESTIMABLE"
    elif (
        len(mc_verdicts) == int(expected["mc_prefix_rows"])
        and len(set(mc_verdicts)) == 1
        and mc_verdicts[0] == h1_verdict
    ):
        h4_verdict = "PASS"
    else:
        h4_verdict = "FAIL"

    n_comparison_cells = int(len(comparative))
    valid_comparison = bool(
        n_comparison_cells > 0
        and comparative["local_comparison_valid"].astype(bool).all()
        and np.isfinite(comparative["local_null_q95"].astype(float)).all()
    )
    if valid_comparison:
        n_above_local = int(
            (
                comparative["whole_minus_local_q95"].astype(float) + tolerance
                >= 0.0
            ).sum()
        )
        above_local_coverage = float(n_above_local / n_comparison_cells)
    else:
        n_above_local = 0
        above_local_coverage = float("nan")
    separated_counts: dict[str, int] = {}
    local_separated_counts: dict[str, int] = {}
    for endpoint in config["endpoints"]:
        subset = comparative[comparative["endpoint"] == endpoint]
        separated_counts[endpoint] = int(subset["whole_separated"].sum())
        local_separated_counts[endpoint] = int(subset["local_separated"].sum())
    h5_estimable_pass = bool(
        valid_comparison
        and n_comparison_cells == int(expected["estimable_primary_cells"])
        and above_local_coverage >= local_bar
        and all(
            separated_counts[endpoint] <= local_separated_counts[endpoint]
            for endpoint in config["endpoints"]
        )
    )
    if not valid_comparison:
        h5_verdict = "NOT_ESTIMABLE"
    elif h5_estimable_pass:
        h5_verdict = "PASS"
    else:
        h5_verdict = "FAIL"

    return {
        "H1": {
            "hypothesis": "Frozen DS1 ARI/NMI are separated from a whole-tissue translation null",
            "verdict": h1_verdict,
            "details": h1_details,
            "n_primary_units_nominal": int(primary_nominal["unit_id"].nunique()),
            "n_primary_units_estimable": int(primary["unit_id"].nunique()),
            "n_primary_cells_nominal": int(len(primary_nominal)),
            "n_primary_cells_estimable": int(len(primary)),
            "family_complete": bool(family_complete),
            "coverage_bar": float(coverage_bar),
            "endpoint_verdicts": endpoint_verdicts,
            "not_estimable_units": sorted(
                primary_nominal.loc[
                    primary_nominal["estimability"] == NOT_ESTIMABLE, "unit_id"
                ].astype(str).unique().tolist()
            ),
        },
        "H2": {
            "hypothesis": "The whole-tissue null is non-degenerate and at least as conservative as the global null",
            "verdict": h2_verdict,
            "n_cells": int(n_cells),
            "n_null_q95_above_median": int(n_ordered),
            "n_null_q95_above_q05": int(n_spread),
            "n_whole_q95_at_least_global": int(n_above_global),
            "whole_q95_at_least_global_coverage": float(above_global_coverage),
            "coverage_bar": float(global_bar),
        },
        "H3": {
            "hypothesis": "The whole-tissue H1 verdict is stable under leave-one-dataset-out",
            "verdict": h3_verdict,
            "full_h1_verdict": h1_verdict,
            "loo_verdicts": loo_verdicts,
            "n_loo_subsets": int(len(dataset_loo)),
        },
        "H4": {
            "hypothesis": "The whole-tissue H1 verdict is stable across Monte Carlo draw prefixes",
            "verdict": h4_verdict,
            "full_h1_verdict": h1_verdict,
            "mc_prefix_verdicts": mc_verdicts,
            "prefixes": [int(value) for value in config["thresholds"]["mc_prefixes"]],
        },
        "H5": {
            "hypothesis": "The whole-tissue null is at least as conservative as the local block null",
            "verdict": h5_verdict,
            "n_comparison_cells": int(n_comparison_cells),
            "n_whole_q95_at_least_local": int(n_above_local),
            "whole_q95_at_least_local_coverage": float(above_local_coverage),
            "coverage_bar": float(local_bar),
            "whole_separated_counts": separated_counts,
            "local_separated_counts": local_separated_counts,
        },
    }


# ---------------------------------------------------------------------------
# Provenance, self-check, and execution
# ---------------------------------------------------------------------------


def key_numbers(
    unit_table: pd.DataFrame,
    draws_table: pd.DataFrame,
    section_table: pd.DataFrame,
    design_table: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    mc_stability: pd.DataFrame,
    comparative: pd.DataFrame,
    verdicts: dict[str, dict[str, Any]],
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    primary_nominal = unit_table[unit_table["role"] == "primary"]
    primary = primary_nominal[primary_nominal["estimability"] == ESTIMABLE]
    not_estimable = primary_nominal[primary_nominal["estimability"] == NOT_ESTIMABLE]
    sensitivity = unit_table[unit_table["role"] != "primary"]
    estimable_units = int(
        unit_table.loc[unit_table["estimability"] == ESTIMABLE, "unit_id"].nunique()
    )
    nominal_units = int(unit_table["unit_id"].nunique())
    unused = inputs["unused_coordinate_barcodes"]
    unused_by_dataset = {
        str(dataset_key): int(frame["barcode"].nunique())
        for dataset_key, frame in unused.groupby("dataset_key", sort=True)
    }
    design_summary = {
        "n_units": int(len(design_table)),
        "n_pass": int(design_table["null_design_pass"].astype(bool).sum()),
        "n_fail": int((~design_table["null_design_pass"].astype(bool)).sum()),
        "max_fallback_fraction": float(
            design_table["max_fallback_fraction"].max()
            if design_table["max_fallback_fraction"].notna().any()
            else float("nan")
        ),
        "max_rank_gt1_fraction": float(
            design_table["max_rank_gt1_fraction"].max()
            if design_table["max_rank_gt1_fraction"].notna().any()
            else float("nan")
        ),
        "max_median_abs_moran_delta": float(
            design_table["median_abs_moran_delta"].max()
            if design_table["median_abs_moran_delta"].notna().any()
            else float("nan")
        ),
        "max_p90_abs_moran_delta": float(
            design_table["p90_abs_moran_delta"].max()
            if design_table["p90_abs_moran_delta"].notna().any()
            else float("nan")
        ),
        "max_median_displacement_ratio": float(
            design_table["median_displacement_ratio"].max()
            if design_table["median_displacement_ratio"].notna().any()
            else float("nan")
        ),
        "max_p90_displacement_ratio": float(
            design_table["p90_displacement_ratio"].max()
            if design_table["p90_displacement_ratio"].notna().any()
            else float("nan")
        ),
    }
    comparative_summary = {}
    for endpoint in config["endpoints"]:
        subset = comparative[comparative["endpoint"] == endpoint]
        comparative_summary[endpoint] = {
            "n_cells": int(len(subset)),
            "n_whole_q95_at_least_local": int(
                (subset["whole_minus_local_q95"].astype(float) >= 0.0).sum()
            ),
            "median_whole_minus_local_q95": float(
                subset["whole_minus_local_q95"].median()
            ),
            "min_whole_minus_local_q95": float(
                subset["whole_minus_local_q95"].min()
            ),
            "n_whole_separated": int(subset["whole_separated"].astype(bool).sum()),
            "n_local_separated": int(subset["local_separated"].astype(bool).sum()),
        }
    return {
        "task_id": config["task_id"],
        "analysis_label": str(config.get("analysis_label", "")),
        "n_units_total": nominal_units,
        "n_estimable_units": estimable_units,
        "n_not_estimable_units": int(nominal_units - estimable_units),
        "not_estimable_unit_ids": sorted(
            unit_table.loc[
                unit_table["estimability"] == NOT_ESTIMABLE, "unit_id"
            ].astype(str).unique().tolist()
        ),
        "n_primary_units": int(primary_nominal["unit_id"].nunique()),
        "n_estimable_primary_units": int(primary["unit_id"].nunique()),
        "n_not_estimable_primary_units": int(not_estimable["unit_id"].nunique()),
        "n_sensitivity_units": int(sensitivity["unit_id"].nunique()),
        "n_primary_cells": int(len(primary_nominal)),
        "n_estimable_primary_cells": int(len(primary)),
        "n_not_estimable_primary_cells": int(len(not_estimable)),
        "n_sensitivity_cells": int(len(sensitivity)),
        "primary_endpoints": list(config["endpoints"]),
        "permutation_draws_per_estimable_unit": int(
            config["thresholds"]["permutation_draws"]
        ),
        "mc_prefixes": [int(value) for value in config["thresholds"]["mc_prefixes"]],
        "autocorrelation_audit_draws": int(
            config["thresholds"]["autocorrelation_audit_draws"]
        ),
        "executed_translation_draws": int(len(draws_table) / 2),
        "design_executed_translation_draws": int(
            estimable_units * int(config["thresholds"]["permutation_draws"])
        ),
        "n_draw_rows": int(len(draws_table)),
        "n_section_rows": int(len(section_table)),
        "n_sections_dataset_level": int(sum(
            int(item.get("n_sections", 0))
            for item in inputs["coordinate_audit"].values()
        )),
        "coordinate_contract": {
            "n_units_total": nominal_units,
            "n_units_exact": int(
                inputs["unit_coordinate_audit"]["coordinate_barcodes_exact"].sum()
            ),
            "n_units_not_estimable": int(
                (
                    inputs["unit_coordinate_audit"]["estimability"]
                    == NOT_ESTIMABLE
                ).sum()
            ),
            "n_unused_coordinate_rows": int(len(unused)),
            "unused_barcodes_by_dataset": unused_by_dataset,
            "audit_table": "tables/tnull1_coordinate_audit.tsv",
            "unused_barcode_table": "tables/tnull1_unused_coordinate_barcodes.tsv",
        },
        "null_design": design_summary,
        "observed_metric_scope": "ESTIMABLE_PRIMARY_CELLS_ONLY",
        "max_observed_ari": float(
            primary.loc[primary["endpoint"] == "ARI", "observed"].max()
        ),
        "max_observed_nmi": float(
            primary.loc[primary["endpoint"] == "NMI", "observed"].max()
        ),
        "median_observed_ari": float(
            primary.loc[primary["endpoint"] == "ARI", "observed"].median()
        ),
        "median_observed_nmi": float(
            primary.loc[primary["endpoint"] == "NMI", "observed"].median()
        ),
        "dataset_loo": dataset_loo.to_dict(orient="records"),
        "mc_stability": mc_stability.to_dict(orient="records"),
        "comparative_calibration": comparative_summary,
        "hypotheses": verdicts,
        "claim_boundary": [
            "Whole-tissue toroidal-translation null, not independent truth and not a generative biological null.",
            "DS1 only; it does not provide the missing GSE205055 human domain-label null.",
            "No method, representation, biological, causal, depth-effect, or universal-ranking claim is authorized.",
            "The three d2 units remain NOT_ESTIMABLE under the exact coordinate contract inherited from NM-SNULL1.",
            "H2/H5 are calibration comparisons, not success claims; a failure does not invalidate H1.",
            "Failure or non-estimability of H1 does not retroactively invalidate NM-NULL1 or NM-SNULL1.",
        ],
    }


def verify_freeze(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_project_path(root, config["protocol"]["freeze_manifest"])
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    if freeze.get("frozen_before_analysis") is not True:
        raise ValueError("NM-TNULL1 requires an explicit pre-analysis freeze")
    checks: list[dict[str, Any]] = []
    roles = ["config", "preregistration", "amendment", "code", "tests", "runner"]
    for role in roles:
        item = freeze.get(role)
        if item is None:
            continue
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
                "role": "input",
            }
        )
    if not all(item["match"] for item in checks):
        failed = [item["path"] for item in checks if not item["match"]]
        raise ValueError(f"Freeze hash mismatch: {failed}")
    return checks


def self_check(
    config: dict[str, Any],
    attempt_root: Path,
    freeze_checks: list[dict[str, Any]],
    inputs: dict[str, Any],
    unit_table: pd.DataFrame,
    draws_table: pd.DataFrame,
    section_table: pd.DataFrame,
    design_table: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    mc_stability: pd.DataFrame,
    comparative: pd.DataFrame,
    verdicts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = config["expected"]
    draws = int(config["thresholds"]["permutation_draws"])
    primary = unit_table[unit_table["role"] == "primary"]
    primary_estimable = primary[primary["estimability"] == ESTIMABLE]
    primary_not_estimable = primary[primary["estimability"] == NOT_ESTIMABLE]
    sensitivity = unit_table[unit_table["role"] != "primary"]
    coordinate_audit = inputs["coordinate_audit"]
    unit_audit = inputs["unit_coordinate_audit"]
    unused_barcodes = inputs["unused_coordinate_barcodes"]
    expected_not_estimable = set(
        str(value) for value in expected.get("not_estimable_unit_ids", [])
    )
    observed_not_estimable = set(
        unit_audit.loc[
            unit_audit["estimability"] == NOT_ESTIMABLE, "unit_id"
        ].astype(str)
    )
    exactness_consistent = bool(
        (
            unit_audit["coordinate_barcodes_exact"]
            == (
                (unit_audit["n_missing_coordinate_barcodes"] == 0)
                & (unit_audit["n_unused_coordinate_barcodes"] == 0)
            )
        ).all()
        and (
            unit_audit.loc[unit_audit["coordinate_barcodes_exact"], "estimability"]
            == ESTIMABLE
        ).all()
        and (
            unit_audit.loc[~unit_audit["coordinate_barcodes_exact"], "estimability"]
            == NOT_ESTIMABLE
        ).all()
    )
    nonzero_q = unit_table.dropna(
        subset=["null_median", "null_q05", "null_q95", "null_q99"]
    )
    expected_draw_rows = int(
        unit_table.loc[unit_table["estimability"] == ESTIMABLE, "unit_id"].nunique()
        * draws
        * len(config["endpoints"])
    )
    draw_prefix_complete = True
    if len(draws_table):
        for (_, _), group in draws_table.groupby(["unit_id", "endpoint"], sort=False):
            if len(group) != draws or sorted(group["draw_index"].tolist()) != list(
                range(draws)
            ):
                draw_prefix_complete = False
                break
    expected_section_rows = 0
    for row in unit_table.loc[
        unit_table["estimability"] == ESTIMABLE, ["dataset_key"]
    ].drop_duplicates().itertuples(index=False):
        expected_section_rows += int(
            coordinate_audit[str(row.dataset_key)]["n_sections"]
        )
    # The unit table has one row per unit-endpoint. Count sections once per unit.
    unit_sections = unit_table.drop_duplicates("unit_id")
    expected_section_rows = int(
        sum(
            int(coordinate_audit[str(row.dataset_key)]["n_sections"])
            for row in unit_sections.loc[
                unit_sections["estimability"] == ESTIMABLE
            ].itertuples(index=False)
        )
    )
    checks = {
        "freeze_hashes_match": all(item["match"] for item in freeze_checks),
        "units_total_match": int(unit_table["unit_id"].nunique())
        == int(expected["units_total"]),
        "primary_units_match": int(primary["unit_id"].nunique())
        == int(expected["primary_units"]),
        "sensitivity_units_match": int(sensitivity["unit_id"].nunique())
        == int(expected["sensitivity_units"]),
        "primary_cells_match": len(primary) == int(expected["primary_cells"]),
        "sensitivity_cells_match": len(sensitivity)
        == int(expected["sensitivity_cells"]),
        "primary_cells_partitioned": bool(
            len(primary_estimable) + len(primary_not_estimable) == len(primary)
        ),
        "estimable_primary_cells_not_above_expected": bool(
            len(primary_estimable) <= int(expected["estimable_primary_cells"])
        ),
        "expected_not_estimable_units_present": expected_not_estimable.issubset(
            observed_not_estimable
        ),
        "not_estimable_units_explicit": bool(
            (
                unit_table.loc[
                    unit_table["estimability"] == NOT_ESTIMABLE,
                    "not_estimable_reason",
                ]
                .astype(str)
                .ne("")
                .all()
            )
        ),
        "dataset_loo_rows_match": len(dataset_loo)
        == int(expected["dataset_loo_rows"]),
        "mc_prefix_rows_match": len(mc_stability)
        == int(expected["mc_prefix_rows"]),
        "hypotheses_present": set(verdicts)
        == {"H1", "H2", "H3", "H4", "H5"},
        "estimable_permutation_draws_match": bool(
            (
                unit_table.loc[
                    unit_table["estimability"] == ESTIMABLE, "permutation_draws"
                ]
                == draws
            ).all()
        ),
        "not_estimable_permutation_draws_zero": bool(
            (
                unit_table.loc[
                    unit_table["estimability"] == NOT_ESTIMABLE,
                    "permutation_draws",
                ]
                == 0
            ).all()
        ),
        "draws_table_rows_match": len(draws_table) == expected_draw_rows,
        "draws_table_prefix_complete": bool(draw_prefix_complete),
        "empirical_p_in_range": bool(
            unit_table["empirical_p"].dropna().between(0.0, 1.0).all()
        ),
        "preregistered_bh_q_in_range": bool(
            unit_table["preregistered_bh_q"].dropna().between(0.0, 1.0).all()
        ),
        "descriptive_bh_q_in_range": bool(
            unit_table["descriptive_bh_q"].dropna().between(0.0, 1.0).all()
        ),
        "null_quantiles_ordered": bool(
            (nonzero_q["null_q05"].astype(float) <= nonzero_q["null_median"].astype(float)).all()
            and (nonzero_q["null_median"].astype(float) <= nonzero_q["null_q95"].astype(float)).all()
            and (nonzero_q["null_q95"].astype(float) <= nonzero_q["null_q99"].astype(float)).all()
        ),
        "no_duplicate_unit_endpoint": bool(
            not unit_table.duplicated(["unit_id", "endpoint"]).any()
        ),
        "coordinate_barcodes_exact_or_explicit_not_estimable": bool(
            exactness_consistent
            and (unit_audit["n_missing_coordinate_barcodes"] == 0).all()
            and int(unit_audit["coordinate_barcodes_exact"].sum())
            == int(expected["estimable_units"])
        ),
        "coordinate_unused_barcode_rows_match": int(len(unused_barcodes))
        == int(unit_audit["n_unused_coordinate_barcodes"].sum()),
        "coordinate_dataset_audit_complete": bool(
            all(
                item.get("coordinate_barcodes_exact")
                == (item.get("unused_coordinate_barcodes") == 0)
                and item.get("prediction_barcodes", 0) > 0
                for item in coordinate_audit.values()
            )
        ),
        "section_count_match": int(
            sum(item["n_sections"] for item in coordinate_audit.values())
        )
        == int(expected["spatial_sections"]),
        "section_audit_rows_match": len(section_table) == expected_section_rows,
        "design_audit_rows_match": len(design_table)
        == int(unit_table["unit_id"].nunique()),
        "design_pass_matches_estimability": bool(
            (
                design_table.set_index("unit_id")["null_design_pass"].astype(bool)
                == (
                    unit_table.drop_duplicates("unit_id")
                    .set_index("unit_id")["estimability"]
                    .eq(ESTIMABLE)
                )
            ).all()
        ),
        "comparative_cells_match_reduced": bool(
            len(comparative) == len(primary_estimable)
        ),
        "comparative_local_q95_finite": bool(
            np.isfinite(comparative["local_null_q95"].astype(float)).all()
            if len(comparative)
            else False
        ),
        "no_rna_matrix_access": True,
    }
    payload = {
        "task_id": config["task_id"],
        "checks": checks,
        "n_checks": int(len(checks)),
        "n_passed": int(sum(bool(value) for value in checks.values())),
        "all_passed": bool(all(checks.values())),
        "expected_clean_run": {
            "estimable_primary_cells": int(expected["estimable_primary_cells"]),
            "observed_estimable_primary_cells": int(len(primary_estimable)),
            "observed_not_estimable_primary_cells": int(
                len(primary_not_estimable)
            ),
        },
    }
    write_json(payload, attempt_root / "manifests" / "self_check.json")
    return payload


def runtime_environment() -> dict[str, Any]:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    return {
        "cwd": str(Path.cwd()),
        "pid": int(os.getpid()),
        "observed_nice": int(os.nice(0)),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else [],
        "rlimit_as_soft": int(soft) if soft != resource.RLIM_INFINITY else "infinity",
        "rlimit_as_hard": int(hard) if hard != resource.RLIM_INFINITY else "infinity",
        "blas_threads": {
            variable: os.environ.get(variable)
            for variable in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


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
        LOGGER.info("NM-TNULL1 start %s", started_utc)
        freeze_checks = verify_freeze(config, root)
        LOGGER.info("Freeze hashes verified: %s", len(freeze_checks))
        inputs = load_inputs(config, root)
        validate_inputs(inputs, config)
        LOGGER.info("Frozen DS1 inputs and coordinates validated")
        unit_table, draws_table, section_table, design_table = build_unit_table(
            inputs, config
        )
        LOGGER.info("Unit-level whole-tissue null complete")
        dataset_loo = build_dataset_loo(unit_table, config)
        mc_stability = build_mc_stability(draws_table, unit_table, config)
        comparative = build_comparative_calibration(unit_table, config)
        verdicts = hypothesis_verdicts(
            unit_table, dataset_loo, mc_stability, comparative, config
        )
        key = key_numbers(
            unit_table,
            draws_table,
            section_table,
            design_table,
            dataset_loo,
            mc_stability,
            comparative,
            verdicts,
            config,
            inputs,
        )

        write_tsv(
            unit_table, attempt_root / "tables" / "tnull1_unit_metrics.tsv"
        )
        draws_table.to_csv(
            attempt_root / "tables" / "tnull1_null_draws.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )
        write_tsv(
            dataset_loo, attempt_root / "tables" / "tnull1_dataset_loo.tsv"
        )
        write_tsv(
            mc_stability, attempt_root / "tables" / "tnull1_mc_stability.tsv"
        )
        write_tsv(
            comparative,
            attempt_root / "tables" / "tnull1_comparative_calibration.tsv",
        )
        write_tsv(
            inputs["unit_coordinate_audit"],
            attempt_root / "tables" / "tnull1_coordinate_audit.tsv",
        )
        write_tsv(
            inputs["unused_coordinate_barcodes"],
            attempt_root / "tables" / "tnull1_unused_coordinate_barcodes.tsv",
        )
        write_tsv(
            section_table, attempt_root / "tables" / "tnull1_section_audit.tsv"
        )
        write_tsv(
            design_table,
            attempt_root / "tables" / "tnull1_null_design_audit.tsv",
        )
        write_tsv(
            pd.DataFrame(
                [
                    {
                        "hypothesis": hypothesis,
                        "hypothesis_text": payload["hypothesis"],
                        "verdict": payload["verdict"],
                        "details_json": json.dumps(
                            {
                                key: value
                                for key, value in payload.items()
                                if key not in {"hypothesis", "verdict"}
                            },
                            sort_keys=True,
                        ),
                    }
                    for hypothesis, payload in verdicts.items()
                ]
            ),
            attempt_root / "tables" / "tnull1_hypotheses.tsv",
        )
        write_json(key, attempt_root / "key_numbers.json")
        write_tsv(
            pd.DataFrame(freeze_checks),
            attempt_root / "manifests" / "input_checksums.tsv",
        )

        self_check_result = self_check(
            config,
            attempt_root,
            freeze_checks,
            inputs,
            unit_table,
            draws_table,
            section_table,
            design_table,
            dataset_loo,
            mc_stability,
            comparative,
            verdicts,
        )
        runtime = time.perf_counter() - started
        peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        freeze_manifest = json.loads(
            resolve_project_path(
                root, config["protocol"]["freeze_manifest"]
            ).read_text(encoding="utf-8")
        )
        manifest = {
            "task_id": config["task_id"],
            "analysis_label": str(config.get("analysis_label", "")),
            "status": "complete",
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "runtime_seconds": float(runtime),
            "peak_rss_kb": peak_rss_kb,
            "command": " ".join(sys.argv),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "preregistration_path": str(config["protocol"]["path"]),
            "preregistration_sha256": str(
                freeze_manifest["preregistration"]["sha256"]
            ),
            "freeze_manifest_path": str(config["protocol"]["freeze_manifest"]),
            "freeze_manifest_sha256": sha256_file(
                resolve_project_path(root, config["protocol"]["freeze_manifest"])
            ),
            "seed_permutation": int(config["seeds"]["permutation"]),
            "parameters": {
                "permutation_draws": int(config["thresholds"]["permutation_draws"]),
                "mc_prefixes": [
                    int(value) for value in config["thresholds"]["mc_prefixes"]
                ],
                "autocorrelation_audit_draws": int(
                    config["thresholds"]["autocorrelation_audit_draws"]
                ),
                "alpha": float(config["thresholds"]["alpha"]),
                "bh_alpha": float(config["thresholds"]["bh_alpha"]),
                "chance_coverage": float(config["thresholds"]["chance_coverage"]),
                "global_calibration_coverage": float(
                    config["thresholds"]["global_calibration_coverage"]
                ),
                "local_comparison_coverage": float(
                    config["thresholds"]["local_comparison_coverage"]
                ),
                "min_shift_fraction": float(
                    config["translation"]["min_shift_fraction"]
                ),
                "candidate_ks": [
                    int(value) for value in config["translation"]["candidate_ks"]
                ],
                "moran_k": int(config["translation"]["moran_k"]),
                "max_fallback_fraction": float(
                    config["translation"]["max_fallback_fraction"]
                ),
                "max_rank_gt1_fraction": float(
                    config["translation"]["max_rank_gt1_fraction"]
                ),
                "max_unit_median_displacement_ratio": float(
                    config["translation"]["max_unit_median_displacement_ratio"]
                ),
                "max_unit_p90_displacement_ratio": float(
                    config["translation"]["max_unit_p90_displacement_ratio"]
                ),
                "max_median_abs_moran_delta": float(
                    config["translation"]["max_median_abs_moran_delta"]
                ),
                "max_p90_abs_moran_delta": float(
                    config["translation"]["max_p90_abs_moran_delta"]
                ),
                "primary_seed": int(config["scope"]["primary_seed"]),
                "numeric_tolerance": float(
                    config["thresholds"]["numeric_tolerance"]
                ),
            },
            "input_hashes": {
                str(item["path"]): str(item["sha256"])
                for item in freeze_manifest["inputs"]
            },
            "software_versions": software_versions(),
            "resource_policy": {
                "configured": {
                    "workers": int(config["resources"]["workers"]),
                    "nice": int(config["resources"]["nice"]),
                    "memory_limit_mb": int(config["resources"]["memory_limit_mb"]),
                    "cpu_affinity": str(config["resources"].get("cpu_affinity", "")),
                    "address_space_limit_bytes": int(
                        config["resources"].get("address_space_limit_bytes", 0)
                    ),
                    "blas_threads": int(
                        config["resources"].get("blas_threads", 1)
                    ),
                },
                "observed": runtime_environment(),
            },
            "self_check": self_check_result,
            "outputs": {
                "unit_metrics": "tables/tnull1_unit_metrics.tsv",
                "null_draws": "tables/tnull1_null_draws.tsv.gz",
                "dataset_loo": "tables/tnull1_dataset_loo.tsv",
                "mc_stability": "tables/tnull1_mc_stability.tsv",
                "comparative_calibration": "tables/tnull1_comparative_calibration.tsv",
                "hypotheses": "tables/tnull1_hypotheses.tsv",
                "coordinate_audit": "tables/tnull1_coordinate_audit.tsv",
                "unused_coordinate_barcodes": "tables/tnull1_unused_coordinate_barcodes.tsv",
                "section_audit": "tables/tnull1_section_audit.tsv",
                "null_design_audit": "tables/tnull1_null_design_audit.tsv",
                "key_numbers": "key_numbers.json",
                "self_check": "manifests/self_check.json",
            },
        }
        write_json(manifest, attempt_root / "manifests" / "run_manifest.json")
        LOGGER.info("NM-TNULL1 complete: %s", json.dumps(verdicts, sort_keys=True))
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
            "self_check": self_check_result,
            "hypotheses": verdicts,
            "key_numbers": key,
        }
    finally:
        LOGGER.removeHandler(file_handler)
        file_handler.close()


__all__ = [
    "build_comparative_calibration",
    "build_dataset_loo",
    "build_mc_stability",
    "build_section_geometry",
    "build_unit_table",
    "draw_translation",
    "hypothesis_verdicts",
    "key_numbers",
    "load_inputs",
    "run",
    "runtime_environment",
    "self_check",
    "shift_permutation",
    "validate_inputs",
    "verify_freeze",
]
