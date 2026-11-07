"""NM-SNULL1 spatially constrained domain null for frozen DS1 concordance.

The audit reads only frozen DS1 per-spot predictions, frozen DS1 spatial
coordinates, frozen seed metrics, and the frozen NM-NULL1 global-null table. It
does not construct a representation, execute a method, access RNA values, or
inspect new outcomes.
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

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

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


LOGGER = logging.getLogger("nm_snull1")

ESTIMABLE = "ESTIMABLE"
NOT_ESTIMABLE = "NOT_ESTIMABLE"
NOT_ESTIMABLE_COORDINATE_CONTRACT = "prediction_barcodes_do_not_exactly_match_coordinates"


def _unit_id(dataset_key: str, depth_fraction: float, representation: str) -> str:
    return f"{dataset_key}__d{float(depth_fraction):.2f}__{representation}"


def coordinate_contract_audit(
    predictions: pd.DataFrame,
    coordinates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Audit the preregistered exact barcode-set contract at unit level.

    A unit is estimable only when its prediction barcode set is exactly equal to
    the frozen coordinate barcode set. Unused coordinates are not silently
    discarded: they make the unit explicitly ``NOT_ESTIMABLE``.
    """

    rows: list[dict[str, Any]] = []
    for (dataset_key, depth_fraction, representation), unit in predictions.groupby(
        ["dataset_key", "depth_fraction", "representation"], sort=True
    ):
        dataset_key = str(dataset_key)
        if dataset_key not in coordinates:
            raise ValueError(f"Coordinates are absent for dataset {dataset_key}")
        predicted = set(unit["barcode"].astype(str))
        available = set(coordinates[dataset_key]["barcode"].astype(str))
        missing = predicted - available
        unused = available - predicted
        exact = not missing and not unused
        rows.append(
            {
                "unit_id": _unit_id(
                    dataset_key, float(depth_fraction), str(representation)
                ),
                "dataset_key": dataset_key,
                "depth_fraction": float(depth_fraction),
                "representation": str(representation),
                "n_prediction_barcodes": int(len(predicted)),
                "n_coordinate_barcodes": int(len(available)),
                "n_missing_coordinate_barcodes": int(len(missing)),
                "n_unused_coordinate_barcodes": int(len(unused)),
                "coordinate_barcodes_exact": bool(exact),
                "estimability": ESTIMABLE if exact else NOT_ESTIMABLE,
                "not_estimable_reason": "" if exact else NOT_ESTIMABLE_COORDINATE_CONTRACT,
            }
        )
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


def build_unused_coordinate_table(
    predictions: pd.DataFrame,
    coordinates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """List every coordinate barcode outside each unit's prediction set."""

    rows: list[dict[str, Any]] = []
    for (dataset_key, depth_fraction, representation), unit in predictions.groupby(
        ["dataset_key", "depth_fraction", "representation"], sort=True
    ):
        dataset_key = str(dataset_key)
        predicted = set(unit["barcode"].astype(str))
        available = set(coordinates[dataset_key]["barcode"].astype(str))
        unit_id = _unit_id(dataset_key, float(depth_fraction), str(representation))
        for barcode in sorted(available - predicted):
            rows.append(
                {
                    "unit_id": unit_id,
                    "dataset_key": dataset_key,
                    "depth_fraction": float(depth_fraction),
                    "representation": str(representation),
                    "barcode": barcode,
                    "status": "UNUSED_COORDINATE",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "unit_id",
            "dataset_key",
            "depth_fraction",
            "representation",
            "barcode",
            "status",
        ],
    )


def _codes(values: pd.Series) -> np.ndarray:
    codes, uniques = pd.factorize(values.astype(str), sort=True)
    if len(uniques) < 2:
        raise ValueError("A unit must have at least two observed label classes")
    return codes.astype(np.int64)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def _read_backed(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        barcodes = adata.obs_names.astype(str).to_numpy()
        if "spatial" not in adata.obsm:
            raise ValueError(f"{path} has no obsm['spatial']")
        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError(f"{path} spatial coordinates are not two-dimensional")
        sections = None
        if "section" in adata.obs.columns:
            sections = adata.obs["section"].astype(str).to_numpy()
    finally:
        adata.file.close()
    return barcodes, spatial[:, :2], sections


def load_coordinates(config: dict[str, Any], root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load frozen spatial coordinates and section labels by dataset."""

    coordinates: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}
    section_source = config["inputs"].get("d3_section_source")
    section_map: dict[str, str] | None = None
    if section_source:
        raw = ad.read_h5ad(resolve_project_path(root, section_source), backed="r")
        try:
            if "section" not in raw.obs.columns:
                raise ValueError("D3 section source lacks a section column")
            section_map = pd.Series(
                raw.obs["section"].astype(str).to_numpy(),
                index=raw.obs_names.astype(str),
            ).to_dict()
        finally:
            raw.file.close()

    for dataset_key, relative in config["inputs"]["coordinates"].items():
        path = resolve_project_path(root, relative)
        barcodes, spatial, embedded_sections = _read_backed(path)
        if not pd.Index(barcodes).is_unique:
            raise ValueError(f"{dataset_key} coordinate barcodes are not unique")
        if section_map is not None and dataset_key == str(config["scope"]["sectioned_dataset"]):
            sections = pd.Series(barcodes).map(section_map)
            if sections.isna().any():
                raise ValueError(f"{dataset_key} section join is incomplete")
            sections = sections.astype(str).to_numpy()
        elif embedded_sections is not None:
            sections = embedded_sections
        else:
            sections = np.full(len(barcodes), "single", dtype=object)

        frame = pd.DataFrame(
            {
                "barcode": barcodes,
                "x": spatial[:, 0],
                "y": spatial[:, 1],
                "section": sections,
            }
        )
        duplicated = frame.duplicated(["section", "x", "y"])
        if bool(duplicated.any()):
            raise ValueError(f"{dataset_key} has duplicate coordinates within a section")
        coordinates[dataset_key] = frame
        audit[dataset_key] = {
            "coordinate_path": str(relative),
            "coordinate_sha256": sha256_file(path),
            "n_spots": int(len(frame)),
            "n_sections": int(frame["section"].nunique()),
            "sections": sorted(frame["section"].unique().tolist()),
            "unique_coordinates_within_section": True,
        }
    return coordinates, audit


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
    coordinates, coordinate_audit = load_coordinates(config, root)
    return {
        "predictions": predictions,
        "seed_metrics": seed_metrics,
        "global_null": global_null,
        "coordinates": coordinates,
        "coordinate_audit": coordinate_audit,
    }


def validate_inputs(inputs: dict[str, Any], config: dict[str, Any]) -> None:
    predictions = inputs["predictions"]
    seed_metrics = inputs["seed_metrics"]
    global_null = inputs["global_null"]
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
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("Prediction table is missing required columns")
    if not required_seed.issubset(seed_metrics.columns):
        raise ValueError("Seed-metric table is missing required columns")
    if not required_global.issubset(global_null.columns):
        raise ValueError("Global-null table is missing required columns")
    if predictions.duplicated(["dataset_key", "depth_fraction", "representation", "barcode"]).any():
        raise ValueError("Prediction table contains duplicate unit/barcode rows")
    if predictions.groupby(["dataset_key", "depth_fraction", "representation"]).size().min() < 2:
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
        raise ValueError("Global-null primary cell count does not match the freeze")

    expected = config["expected"]
    not_estimable = unit_audit[unit_audit["estimability"] == NOT_ESTIMABLE]
    estimable = unit_audit[unit_audit["estimability"] == ESTIMABLE]
    if len(unit_audit) != int(expected["units_total"]):
        raise ValueError("Coordinate-audit unit count does not match the freeze")
    if len(not_estimable) != int(expected["not_estimable_units"]):
        raise ValueError("NOT_ESTIMABLE unit count does not match the freeze")
    if len(estimable) != int(expected["estimable_units"]):
        raise ValueError("ESTIMABLE unit count does not match the freeze")
    if set(not_estimable["unit_id"].astype(str)) != set(
        str(value) for value in expected.get("not_estimable_unit_ids", [])
    ):
        raise ValueError("NOT_ESTIMABLE unit identities do not match the freeze")


def block_assignment(
    coordinates: pd.DataFrame,
    block_factor: float,
    tolerance: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign each spot to a section-local square block."""

    if coordinates.duplicated(["section", "x", "y"]).any():
        raise ValueError("Duplicate coordinates within a section")
    blocks = np.full(len(coordinates), -1, dtype=np.int64)
    diagnostics: list[dict[str, Any]] = []
    barcode_order = np.arange(len(coordinates))
    for section in sorted(coordinates["section"].astype(str).unique()):
        mask = coordinates["section"].astype(str).to_numpy() == section
        row_idx = barcode_order[mask]
        xy = coordinates.loc[mask, ["x", "y"]].to_numpy(dtype=np.float64)
        if len(xy) < 2:
            raise ValueError(f"Section {section} has fewer than two spots")
        nn = NearestNeighbors(n_neighbors=2).fit(xy)
        distances, _ = nn.kneighbors(xy)
        median_nn = float(np.median(distances[:, 1]))
        if median_nn <= 0:
            raise ValueError(f"Section {section} has non-positive median nearest-neighbor distance")
        block_side = float(block_factor) * median_nn
        bx = np.floor((xy[:, 0] - xy[:, 0].min()) / block_side).astype(np.int64)
        by = np.floor((xy[:, 1] - xy[:, 1].min()) / block_side).astype(np.int64)
        keys = pd.Series(
            [f"{section}|{int(a)}|{int(b)}" for a, b in zip(bx, by, strict=True)]
        )
        local_codes, uniques = pd.factorize(keys, sort=True)
        blocks[row_idx] = local_codes.astype(np.int64) + int(blocks.max()) + 1
        counts = pd.Series(local_codes).value_counts()
        diagnostics.append(
            {
                "section": section,
                "block_factor": float(block_factor),
                "n_spots": int(len(xy)),
                "median_nearest_neighbor_distance": median_nn,
                "block_side": block_side,
                "n_blocks": int(len(uniques)),
                "min_spots_per_block": int(counts.min()),
                "median_spots_per_block": float(counts.median()),
                "max_spots_per_block": int(counts.max()),
            }
        )
    if bool((blocks < 0).any()):
        raise ValueError("Spatial block assignment is incomplete")
    _, global_codes = np.unique(blocks, return_inverse=True)
    del tolerance  # block assignment is exact and independent of numeric tolerance
    return global_codes.astype(np.int64), pd.DataFrame(diagnostics)


def _block_index_groups(block_codes: np.ndarray) -> list[np.ndarray]:
    """Return original row indices for each spatial block."""

    order = np.argsort(block_codes, kind="stable")
    sorted_codes = block_codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sorted_codes)]))
    return [order[start:end] for start, end in zip(starts, ends, strict=True)]


def _permute_codes_within_blocks(
    predicted_codes: np.ndarray,
    block_codes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute predicted codes within blocks without changing their marginals."""

    permuted = predicted_codes.copy()
    for indices in _block_index_groups(block_codes):
        if len(indices) > 1:
            permuted[indices] = predicted_codes[indices][
                rng.permutation(len(indices))
            ]
    return permuted


def permutation_null_spatial(
    truth_labels: pd.Series,
    predicted_labels: pd.Series,
    block_codes: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-draw spatial-block null ARI/NMI with fixed truth and marginals."""

    truth_codes = _codes(truth_labels.reset_index(drop=True))
    pred_codes = _codes(predicted_labels.reset_index(drop=True))
    if len(block_codes) != len(truth_codes):
        raise ValueError("Block assignment length does not match labels")
    n_samples = int(len(truth_codes))
    n_truth = int(truth_codes.max()) + 1
    n_pred = int(pred_codes.max()) + 1
    truth_counts = np.bincount(truth_codes, minlength=n_truth).astype(np.float64)
    pred_counts = np.bincount(pred_codes, minlength=n_pred).astype(np.float64)
    rng = np.random.default_rng(seed)
    null_ari = np.empty(int(draws), dtype=np.float64)
    null_nmi = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        permuted = _permute_codes_within_blocks(pred_codes, block_codes, rng)
        counts = np.bincount(
            truth_codes * n_pred + permuted,
            minlength=n_truth * n_pred,
        ).astype(np.float64)
        null_ari[index] = ari_from_counts(counts, n_samples, truth_counts, pred_counts)
        null_nmi[index] = nmi_from_counts(counts, n_samples, truth_counts, pred_counts, n_pred)
        if (index + 1) % 2000 == 0:
            LOGGER.info("  spatial permutation progress %d/%d", index + 1, draws)
    return null_ari, null_nmi


def _global_lookup(global_null: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in global_null.itertuples(index=False):
        lookup[(str(row.unit_id), str(row.endpoint))] = {
            "observed": float(row.observed),
            "null_median": float(row.null_median),
            "null_q95": float(row.null_q95),
            "above_q95": bool(row.above_q95),
            "bh_significant": bool(row.bh_significant),
        }
    return lookup


def _null_row(
    unit_id: str,
    dataset_key: str,
    depth_fraction: float,
    representation: str,
    role: str,
    analysis_role: str,
    n_spots: int,
    n_truth: int,
    n_pred: int,
    block_factor: float,
    endpoint: str,
    observed: float,
    null_values: np.ndarray,
    seed: int,
    global_row: dict[str, float],
    coordinate_row: pd.Series,
) -> dict[str, Any]:
    null_median = float(np.median(null_values))
    null_q05 = _quantile(null_values, 0.05)
    null_q95 = _quantile(null_values, 0.95)
    null_q99 = _quantile(null_values, 0.99)
    empirical_p = float(
        (1.0 + float(np.sum(null_values >= observed))) / (1.0 + float(len(null_values)))
    )
    return {
        "unit_id": unit_id,
        "dataset_key": dataset_key,
        "depth_fraction": float(depth_fraction),
        "representation": representation,
        "role": role,
        "analysis_role": analysis_role,
        "estimability": ESTIMABLE,
        "not_estimable_reason": "",
        "coordinate_barcodes_exact": bool(coordinate_row["coordinate_barcodes_exact"]),
        "n_prediction_barcodes": int(coordinate_row["n_prediction_barcodes"]),
        "n_coordinate_barcodes": int(coordinate_row["n_coordinate_barcodes"]),
        "n_missing_coordinate_barcodes": int(
            coordinate_row["n_missing_coordinate_barcodes"]
        ),
        "n_unused_coordinate_barcodes": int(
            coordinate_row["n_unused_coordinate_barcodes"]
        ),
        "n_spots": int(n_spots),
        "n_truth_clusters": int(n_truth),
        "n_predicted_clusters": int(n_pred),
        "block_factor": float(block_factor),
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
        "global_observed": float(global_row["observed"]),
        "global_null_median": float(global_row["null_median"]),
        "global_null_q95": float(global_row["null_q95"]),
        "spatial_minus_global_q95": null_q95 - float(global_row["null_q95"]),
        "global_above_q95": bool(global_row["above_q95"]),
        "global_bh_significant": bool(global_row["bh_significant"]),
    }


def _not_estimable_row(
    unit_id: str,
    dataset_key: str,
    depth_fraction: float,
    representation: str,
    role: str,
    analysis_role: str,
    n_spots: int,
    n_truth: int,
    n_pred: int,
    block_factor: float,
    endpoint: str,
    observed: float,
    seed: int,
    global_row: dict[str, float],
    coordinate_row: pd.Series,
) -> dict[str, Any]:
    """Represent an invalid unit explicitly instead of dropping it."""

    return {
        "unit_id": unit_id,
        "dataset_key": dataset_key,
        "depth_fraction": float(depth_fraction),
        "representation": representation,
        "role": role,
        "analysis_role": analysis_role,
        "estimability": NOT_ESTIMABLE,
        "not_estimable_reason": NOT_ESTIMABLE_COORDINATE_CONTRACT,
        "coordinate_barcodes_exact": False,
        "n_prediction_barcodes": int(coordinate_row["n_prediction_barcodes"]),
        "n_coordinate_barcodes": int(coordinate_row["n_coordinate_barcodes"]),
        "n_missing_coordinate_barcodes": int(
            coordinate_row["n_missing_coordinate_barcodes"]
        ),
        "n_unused_coordinate_barcodes": int(
            coordinate_row["n_unused_coordinate_barcodes"]
        ),
        "n_spots": int(n_spots),
        "n_truth_clusters": int(n_truth),
        "n_predicted_clusters": int(n_pred),
        "block_factor": float(block_factor),
        "endpoint": endpoint,
        "observed": float(observed),
        "null_median": np.nan,
        "null_q05": np.nan,
        "null_q95": np.nan,
        "null_q99": np.nan,
        "null_min": np.nan,
        "null_max": np.nan,
        "observed_minus_median": np.nan,
        "observed_minus_q95": np.nan,
        "empirical_p": np.nan,
        "permutation_draws": 0,
        "permutation_seed": int(seed),
        "above_q95": False,
        "global_observed": float(global_row["observed"]),
        "global_null_median": float(global_row["null_median"]),
        "global_null_q95": float(global_row["null_q95"]),
        "spatial_minus_global_q95": np.nan,
        "global_above_q95": bool(global_row["above_q95"]),
        "global_bh_significant": bool(global_row["bh_significant"]),
    }


def _seed_lookup(seed_metrics: pd.DataFrame, primary_seed: int) -> dict[tuple[str, float, str], tuple[float, float]]:
    primary = seed_metrics[seed_metrics["clustering_seed"].astype(int) == primary_seed]
    return {
        (str(row.dataset_key), float(row.depth_fraction), str(row.representation)): (
            float(row.ARI),
            float(row.NMI),
        )
        for row in primary.itertuples(index=False)
    }


def build_unit_table(inputs: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = inputs["predictions"]
    primary_seed = int(config["scope"]["primary_seed"])
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    draws = int(config["thresholds"]["permutation_draws"])
    base_seed = int(config["seeds"]["permutation"])
    primary_factor = float(config["spatial"]["primary_block_factor"])
    sensitivity = config["scope"]["sensitivity"]
    sensitivity_dataset = str(sensitivity["dataset_key"])
    sensitivity_depth = float(sensitivity["depth_fraction"])
    seed_lookup = _seed_lookup(inputs["seed_metrics"], primary_seed)
    global_lookup = _global_lookup(inputs["global_null"])
    coordinates = inputs["coordinates"]
    coordinate_lookup = inputs["unit_coordinate_audit"].set_index("unit_id")
    endpoints = list(config["endpoints"])

    rows: list[dict[str, Any]] = []
    diagnostics: list[pd.DataFrame] = []
    observed_units: set[tuple[str, float, str]] = set()
    for (dataset_key, depth_fraction, representation), unit in predictions.groupby(
        ["dataset_key", "depth_fraction", "representation"], sort=True
    ):
        dataset_key = str(dataset_key)
        depth_fraction = float(depth_fraction)
        representation = str(representation)
        unit_key = (dataset_key, depth_fraction, representation)
        observed_units.add(unit_key)
        unit_id = _unit_id(dataset_key, depth_fraction, representation)
        is_sensitivity = (
            dataset_key == sensitivity_dataset
            and math.isclose(depth_fraction, sensitivity_depth, abs_tol=tolerance)
        )
        role = str(sensitivity["role"]) if is_sensitivity else "primary"
        coordinate_row = coordinate_lookup.loc[unit_id]
        if coordinate_row["estimability"] != ESTIMABLE:
            analysis_role = "NOT_ESTIMABLE_COORDINATE_CONTRACT"
        elif is_sensitivity:
            analysis_role = "NONCOMPARABLE_SENSITIVITY_ONLY"
        else:
            analysis_role = "primary"
        if unit["barcode"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate barcode in unit {unit_id}")
        if unit["truth_label"].isna().any() or unit["predicted_cluster"].isna().any():
            raise ValueError(f"Missing label in unit {unit_id}")
        unit = unit.sort_values("barcode").reset_index(drop=True)
        coords = coordinates[dataset_key].copy()
        merged = unit.merge(coords, on="barcode", how="left", validate="one_to_one", sort=False)
        if merged[["x", "y", "section"]].isna().any().any():
            raise ValueError(f"Coordinate join is incomplete for {unit_id}")
        if not merged["barcode"].astype(str).equals(unit["barcode"].astype(str)):
            raise ValueError(f"Coordinate join changed row order for {unit_id}")

        truth = unit["truth_label"].astype(str)
        predicted = unit["predicted_cluster"].astype(str)
        n_truth = int(truth.nunique())
        n_pred = int(predicted.nunique())
        if n_truth < 2 or n_pred < 2:
            raise ValueError(f"Unit {unit_id} has fewer than two truth or predicted classes")
        ari, nmi, _, _ = observed_metrics(truth, predicted)
        if unit_key not in seed_lookup:
            raise ValueError(f"Frozen seed metrics lack unit {unit_id}")
        expected_ari, expected_nmi = seed_lookup[unit_key]
        if abs(ari - expected_ari) > tolerance or abs(nmi - expected_nmi) > tolerance:
            raise ValueError(
                f"Observed metric mismatch for {unit_id}: ARI {ari} vs {expected_ari}, "
                f"NMI {nmi} vs {expected_nmi}"
            )

        unit_seed = stable_seed(base_seed, f"{unit_id}__bf{primary_factor:g}")
        if coordinate_row["estimability"] != ESTIMABLE:
            LOGGER.warning(
                "Unit %s is NOT_ESTIMABLE: predictions=%d coordinates=%d unused=%d; "
                "no spatial permutations executed",
                unit_id,
                int(coordinate_row["n_prediction_barcodes"]),
                int(coordinate_row["n_coordinate_barcodes"]),
                int(coordinate_row["n_unused_coordinate_barcodes"]),
            )
            for endpoint, observed in (("ARI", ari), ("NMI", nmi)):
                key = (unit_id, endpoint)
                if key not in global_lookup:
                    raise ValueError(f"Global-null table lacks {key}")
                rows.append(
                    _not_estimable_row(
                        unit_id,
                        dataset_key,
                        depth_fraction,
                        representation,
                        role,
                        analysis_role,
                        len(unit),
                        n_truth,
                        n_pred,
                        primary_factor,
                        endpoint,
                        observed,
                        unit_seed,
                        global_lookup[key],
                        coordinate_row,
                    )
                )
            continue

        blocks, block_diag = block_assignment(merged, primary_factor, tolerance)
        block_diag.insert(0, "dataset_key", dataset_key)
        block_diag.insert(1, "role", analysis_role)
        diagnostics.append(block_diag)
        LOGGER.info(
            "Unit %s: n=%d, truth=%d, pred=%d, blocks=%d, ARI=%.6f, NMI=%.6f",
            unit_id,
            len(unit),
            n_truth,
            n_pred,
            int(len(np.unique(blocks))),
            ari,
            nmi,
        )
        null_ari, null_nmi = permutation_null_spatial(
            truth,
            predicted,
            blocks,
            draws=draws,
            seed=unit_seed,
        )
        null_by_endpoint = {"ARI": null_ari, "NMI": null_nmi}
        observed_by_endpoint = {"ARI": ari, "NMI": nmi}
        for endpoint in endpoints:
            key = (unit_id, endpoint)
            if key not in global_lookup:
                raise ValueError(f"Global-null table lacks {key}")
            rows.append(
                _null_row(
                    unit_id,
                    dataset_key,
                    depth_fraction,
                    representation,
                    role,
                    analysis_role,
                    len(unit),
                    n_truth,
                    n_pred,
                    primary_factor,
                    endpoint,
                    observed_by_endpoint[endpoint],
                    null_by_endpoint[endpoint],
                    unit_seed,
                    global_lookup[key],
                    coordinate_row,
                )
            )

    if observed_units != set(seed_lookup):
        raise ValueError("Prediction units and primary-seed metric units differ")
    result = pd.DataFrame(rows)
    primary_mask = result["role"] == "primary"
    estimable_primary_mask = primary_mask & (result["estimability"] == ESTIMABLE)
    result["preregistered_bh_q"] = np.nan
    result["preregistered_bh_significant"] = False
    result["descriptive_bh_q"] = np.nan
    result["descriptive_bh_significant"] = False
    result["bh_family_scope"] = "SENSITIVITY_ONLY_NOT_IN_BH_FAMILY"
    result.loc[primary_mask, "bh_family_scope"] = (
        "PREREGISTERED_28_CELL_FAMILY_INCOMPLETE"
    )
    result.loc[estimable_primary_mask, "bh_family_scope"] = (
        "REDUCED_ESTIMABLE_PRIMARY_CELLS_DESCRIPTIVE"
    )

    primary_complete = bool(
        len(result.loc[primary_mask]) == int(config["expected"]["primary_cells"])
        and bool(estimable_primary_mask.loc[primary_mask].all())
    )
    if primary_complete:
        result.loc[primary_mask, "preregistered_bh_q"] = bh_adjust(
            result.loc[primary_mask, "empirical_p"].to_numpy(dtype=float)
        )
        result.loc[primary_mask, "preregistered_bh_significant"] = (
            result.loc[primary_mask, "preregistered_bh_q"]
            <= float(config["thresholds"]["bh_alpha"])
        )
    if bool(estimable_primary_mask.any()):
        result.loc[estimable_primary_mask, "descriptive_bh_q"] = bh_adjust(
            result.loc[estimable_primary_mask, "empirical_p"].to_numpy(dtype=float)
        )
        result.loc[estimable_primary_mask, "descriptive_bh_significant"] = (
            result.loc[estimable_primary_mask, "descriptive_bh_q"]
            <= float(config["thresholds"]["bh_alpha"])
        )
    result = result.sort_values(
        ["role", "dataset_key", "depth_fraction", "representation", "endpoint"]
    ).reset_index(drop=True)
    block_table = pd.concat(diagnostics, ignore_index=True)
    return result, block_table


def build_block_sensitivity(
    inputs: dict[str, Any],
    config: dict[str, Any],
    primary_table: pd.DataFrame,
) -> pd.DataFrame:
    predictions = inputs["predictions"]
    primary_seed = int(config["scope"]["primary_seed"])
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    draws = int(config["thresholds"]["permutation_draws"])
    base_seed = int(config["seeds"]["permutation"])
    factors = [float(value) for value in config["spatial"]["block_factors"]]
    primary_factor = float(config["spatial"]["primary_block_factor"])
    sensitivity = config["scope"]["sensitivity"]
    sensitivity_dataset = str(sensitivity["dataset_key"])
    sensitivity_depth = float(sensitivity["depth_fraction"])
    seed_lookup = _seed_lookup(inputs["seed_metrics"], primary_seed)
    global_lookup = _global_lookup(inputs["global_null"])
    coordinates = inputs["coordinates"]
    coordinate_lookup = inputs["unit_coordinate_audit"].set_index("unit_id")
    endpoints = list(config["endpoints"])

    frames = [
        primary_table.assign(factor_role="PRIMARY_ANALYSIS").copy()
    ]
    for factor in factors:
        if math.isclose(factor, primary_factor, abs_tol=tolerance):
            continue
        rows: list[dict[str, Any]] = []
        for (dataset_key, depth_fraction, representation), unit in predictions.groupby(
            ["dataset_key", "depth_fraction", "representation"], sort=True
        ):
            dataset_key = str(dataset_key)
            depth_fraction = float(depth_fraction)
            representation = str(representation)
            unit_key = (dataset_key, depth_fraction, representation)
            unit_id = _unit_id(dataset_key, depth_fraction, representation)
            is_sensitivity = (
                dataset_key == sensitivity_dataset
                and math.isclose(depth_fraction, sensitivity_depth, abs_tol=tolerance)
            )
            role = str(sensitivity["role"]) if is_sensitivity else "primary"
            coordinate_row = coordinate_lookup.loc[unit_id]
            if coordinate_row["estimability"] != ESTIMABLE:
                analysis_role = "NOT_ESTIMABLE_COORDINATE_CONTRACT"
            elif is_sensitivity:
                analysis_role = "NONCOMPARABLE_SENSITIVITY_ONLY"
            else:
                analysis_role = "primary"
            unit = unit.sort_values("barcode").reset_index(drop=True)
            merged = unit.merge(
                coordinates[dataset_key],
                on="barcode",
                how="left",
                validate="one_to_one",
                sort=False,
            )
            truth = unit["truth_label"].astype(str)
            predicted = unit["predicted_cluster"].astype(str)
            ari, nmi, _, _ = observed_metrics(truth, predicted)
            expected_ari, expected_nmi = seed_lookup[unit_key]
            if abs(ari - expected_ari) > tolerance or abs(nmi - expected_nmi) > tolerance:
                raise ValueError(f"Observed metric mismatch for {unit_id} at factor {factor}")
            unit_seed = stable_seed(base_seed, f"{unit_id}__bf{factor:g}")
            if coordinate_row["estimability"] != ESTIMABLE:
                for endpoint, observed in (("ARI", ari), ("NMI", nmi)):
                    row = _not_estimable_row(
                        unit_id,
                        dataset_key,
                        depth_fraction,
                        representation,
                        role,
                        analysis_role,
                        len(unit),
                        int(truth.nunique()),
                        int(predicted.nunique()),
                        factor,
                        endpoint,
                        observed,
                        unit_seed,
                        global_lookup[(unit_id, endpoint)],
                        coordinate_row,
                    )
                    row["factor_role"] = "SENSITIVITY_BLOCK_SIZE"
                    rows.append(row)
                continue
            blocks, _ = block_assignment(merged, factor, tolerance)
            null_ari, null_nmi = permutation_null_spatial(
                truth,
                predicted,
                blocks,
                draws=draws,
                seed=unit_seed,
            )
            null_by_endpoint = {"ARI": null_ari, "NMI": null_nmi}
            observed_by_endpoint = {"ARI": ari, "NMI": nmi}
            for endpoint in endpoints:
                row = _null_row(
                    unit_id,
                    dataset_key,
                    depth_fraction,
                    representation,
                    role,
                    analysis_role,
                    len(unit),
                    int(truth.nunique()),
                    int(predicted.nunique()),
                    factor,
                    endpoint,
                    observed_by_endpoint[endpoint],
                    null_by_endpoint[endpoint],
                    unit_seed,
                    global_lookup[(unit_id, endpoint)],
                    coordinate_row,
                )
                row["factor_role"] = "SENSITIVITY_BLOCK_SIZE"
                rows.append(row)
        frame = pd.DataFrame(rows)
        primary_mask = frame["role"] == "primary"
        estimable_primary_mask = primary_mask & (frame["estimability"] == ESTIMABLE)
        frame["preregistered_bh_q"] = np.nan
        frame["preregistered_bh_significant"] = False
        frame["descriptive_bh_q"] = np.nan
        frame["descriptive_bh_significant"] = False
        frame["bh_family_scope"] = "SENSITIVITY_ONLY_NOT_IN_BH_FAMILY"
        frame.loc[primary_mask, "bh_family_scope"] = (
            "PREREGISTERED_28_CELL_FAMILY_INCOMPLETE"
        )
        frame.loc[estimable_primary_mask, "bh_family_scope"] = (
            "REDUCED_ESTIMABLE_PRIMARY_CELLS_DESCRIPTIVE"
        )
        primary_complete = bool(
            len(frame.loc[primary_mask]) == int(config["expected"]["primary_cells"])
            and bool(estimable_primary_mask.loc[primary_mask].all())
        )
        if primary_complete:
            frame.loc[primary_mask, "preregistered_bh_q"] = bh_adjust(
                frame.loc[primary_mask, "empirical_p"].to_numpy(dtype=float)
            )
            frame.loc[primary_mask, "preregistered_bh_significant"] = (
                frame.loc[primary_mask, "preregistered_bh_q"]
                <= float(config["thresholds"]["bh_alpha"])
            )
        if bool(estimable_primary_mask.any()):
            frame.loc[estimable_primary_mask, "descriptive_bh_q"] = bh_adjust(
                frame.loc[estimable_primary_mask, "empirical_p"].to_numpy(dtype=float)
            )
            frame.loc[estimable_primary_mask, "descriptive_bh_significant"] = (
                frame.loc[estimable_primary_mask, "descriptive_bh_q"]
                <= float(config["thresholds"]["bh_alpha"])
            )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["block_factor", "role", "dataset_key", "depth_fraction", "representation", "endpoint"]
    ).reset_index(drop=True)


def build_dataset_loo(unit_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = unit_table[unit_table["role"] == "primary"].copy()
    alpha = float(config["thresholds"]["bh_alpha"])
    coverage_bar = float(config["thresholds"]["chance_coverage"])
    rows: list[dict[str, Any]] = []
    for omitted in config["scope"]["dataset_loo"]:
        nominal_subset = primary[
            primary["dataset_key"].astype(str) != str(omitted)
        ].copy()
        subset = nominal_subset[
            nominal_subset["estimability"] == ESTIMABLE
        ].copy()
        incomplete = len(subset) != len(nominal_subset)
        subset["loo_descriptive_bh_q"] = np.nan
        subset["loo_descriptive_bh_significant"] = False
        if len(subset):
            subset.loc[:, "loo_descriptive_bh_q"] = bh_adjust(
                subset["empirical_p"].to_numpy(dtype=float)
            )
            subset.loc[:, "loo_descriptive_bh_significant"] = (
                subset["loo_descriptive_bh_q"] <= alpha
            )
        descriptive_endpoint_pass: dict[str, bool] = {}
        row: dict[str, Any] = {
            "omitted_dataset_key": str(omitted),
            "n_units_remaining": int(nominal_subset["unit_id"].nunique()),
            "n_cells_remaining": int(len(nominal_subset)),
            "n_units_remaining_estimable": int(subset["unit_id"].nunique()),
            "n_cells_remaining_estimable": int(len(subset)),
            "n_units_remaining_not_estimable": int(
                nominal_subset.loc[
                    nominal_subset["estimability"] == NOT_ESTIMABLE, "unit_id"
                ].nunique()
            ),
            "loo_family_complete": bool(not incomplete),
        }
        for endpoint in config["endpoints"]:
            endpoint_rows = subset[subset["endpoint"] == endpoint]
            n_cells = int(len(endpoint_rows))
            n_separated = int(
                (
                    endpoint_rows["above_q95"]
                    & endpoint_rows["loo_descriptive_bh_significant"]
                ).sum()
            )
            coverage = float(n_separated / n_cells) if n_cells else float("nan")
            descriptive_passed = bool(n_cells > 0 and coverage >= coverage_bar)
            descriptive_endpoint_pass[endpoint] = descriptive_passed
            row[f"n_{endpoint.lower()}_cells"] = n_cells
            row[f"n_{endpoint.lower()}_separated"] = n_separated
            row[f"coverage_{endpoint.lower()}"] = coverage
            row[f"descriptive_pass_{endpoint.lower()}"] = descriptive_passed
        row["descriptive_pass"] = bool(all(descriptive_endpoint_pass.values()))
        row["descriptive_h1_verdict"] = (
            "PASS" if row["descriptive_pass"] else "FAIL"
        )
        row["h1_verdict"] = (
            "NOT_ESTIMABLE"
            if incomplete
            else row["descriptive_h1_verdict"]
        )
        row["pass"] = bool(row["h1_verdict"] == "PASS")
        rows.append(row)
    return pd.DataFrame(rows)


def _endpoint_h1_verdict(
    frame: pd.DataFrame,
    endpoint: str,
    coverage_bar: float,
    *,
    bh_column: str = "descriptive_bh_significant",
    require_complete: bool = False,
    expected_cells: int | None = None,
) -> tuple[str, float, int, int]:
    nominal = frame[(frame["role"] == "primary") & (frame["endpoint"] == endpoint)]
    primary = nominal[nominal["estimability"] == ESTIMABLE]
    n_cells = int(len(primary))
    n_separated = int(((primary["above_q95"]) & (primary[bh_column])).sum())
    coverage = float(n_separated / n_cells) if n_cells else float("nan")
    if require_complete and (
        expected_cells is None
        or n_cells != int(expected_cells)
        or len(nominal) != int(expected_cells)
    ):
        return ("NOT_ESTIMABLE", coverage, n_separated, n_cells)
    return ("PASS" if n_cells > 0 and coverage >= coverage_bar else "FAIL", coverage, n_separated, n_cells)


def _h5_separated_counts(frame: pd.DataFrame) -> tuple[int, int]:
    """Return symmetric spatial/global separated counts for one endpoint."""

    spatial_count = int(
        (frame["above_q95"] & frame["descriptive_bh_significant"]).sum()
    )
    global_count = int(
        (frame["global_above_q95"] & frame["global_bh_significant"]).sum()
    )
    return spatial_count, global_count


def hypothesis_verdicts(
    unit_table: pd.DataFrame,
    block_sensitivity: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    endpoints = list(config["endpoints"])
    coverage_bar = float(config["thresholds"]["chance_coverage"])
    positivity_bar = float(config["thresholds"]["null_positivity_coverage"])
    alpha = float(config["thresholds"]["bh_alpha"])
    primary_nominal = unit_table[unit_table["role"] == "primary"].copy()
    primary = primary_nominal[
        primary_nominal["estimability"] == ESTIMABLE
    ].copy()
    expected_primary_units = int(config["expected"]["primary_units"])
    expected_primary_cells = int(config["expected"]["primary_cells"])
    n_estimable_units = int(primary["unit_id"].nunique())
    h1_complete = bool(
        primary_nominal["unit_id"].nunique() == expected_primary_units
        and n_estimable_units == expected_primary_units
        and len(primary) == expected_primary_cells
    )

    h1_details: dict[str, Any] = {}
    h1_descriptive: dict[str, Any] = {}
    endpoint_verdicts: dict[str, str] = {}
    for endpoint in endpoints:
        registered_verdict, registered_coverage, registered_separated, registered_cells = (
            _endpoint_h1_verdict(
                primary_nominal,
                endpoint,
                coverage_bar,
                bh_column="preregistered_bh_significant",
                require_complete=True,
                expected_cells=expected_primary_units,
            )
        )
        verdict = registered_verdict if h1_complete else "NOT_ESTIMABLE"
        endpoint_verdicts[endpoint] = verdict
        descriptive_verdict, descriptive_coverage, descriptive_separated, descriptive_cells = (
            _endpoint_h1_verdict(
                primary,
                endpoint,
                coverage_bar,
                bh_column="descriptive_bh_significant",
            )
        )
        h1_details[f"n_{endpoint.lower()}_separated"] = (
            registered_separated if h1_complete else None
        )
        h1_details[f"n_{endpoint.lower()}_units"] = registered_cells
        h1_details[f"n_{endpoint.lower()}_units_required"] = expected_primary_units
        h1_details[f"coverage_{endpoint.lower()}"] = (
            registered_coverage if h1_complete else None
        )
        h1_details[f"verdict_{endpoint.lower()}"] = verdict
        h1_descriptive[f"n_{endpoint.lower()}_separated"] = descriptive_separated
        h1_descriptive[f"n_{endpoint.lower()}_units"] = descriptive_cells
        h1_descriptive[f"coverage_{endpoint.lower()}"] = descriptive_coverage
        h1_descriptive[f"verdict_{endpoint.lower()}"] = descriptive_verdict
    if not h1_complete:
        h1_verdict = "NOT_ESTIMABLE"
    elif all(value == "PASS" for value in endpoint_verdicts.values()):
        h1_verdict = "PASS"
    else:
        h1_verdict = "FAIL"

    n_positive_median = int((primary["null_median"] > 0).sum())
    n_ordered = int((primary["null_q95"] > primary["null_median"]).sum())
    n_spread = int((primary["null_q95"] > primary["null_q05"]).sum())
    n_above_global = int((primary["spatial_minus_global_q95"] > 0).sum())
    n_cells = int(len(primary))
    above_global_coverage = float(n_above_global / n_cells) if n_cells else float("nan")
    h2_estimable_pass = bool(
        n_cells > 0
        and n_positive_median == n_cells
        and n_ordered == n_cells
        and n_spread == n_cells
        and above_global_coverage >= positivity_bar
    )
    if n_cells == 0:
        h2_verdict = "NOT_ESTIMABLE"
    elif not h2_estimable_pass:
        h2_verdict = "FAIL"
    elif len(primary_nominal) < expected_primary_cells:
        h2_verdict = "NOT_ESTIMABLE"
    else:
        h2_verdict = "PASS"

    loo_verdicts = dataset_loo["h1_verdict"].astype(str).tolist() if len(dataset_loo) else []
    descriptive_loo_verdicts = (
        dataset_loo["descriptive_h1_verdict"].astype(str).tolist()
        if len(dataset_loo)
        else []
    )
    if "NOT_ESTIMABLE" in loo_verdicts or h1_verdict == "NOT_ESTIMABLE":
        h3_verdict = "NOT_ESTIMABLE"
    elif (
        len(loo_verdicts) == int(config["expected"]["dataset_loo_rows"])
        and all(value == h1_verdict for value in loo_verdicts)
    ):
        h3_verdict = "PASS"
    else:
        h3_verdict = "FAIL"

    block_verdicts: dict[str, Any] = {}
    descriptive_block_verdicts: dict[str, Any] = {}
    block_verdict_values: list[str] = []
    for endpoint in endpoints:
        per_factor: dict[str, str] = {}
        descriptive_per_factor: dict[str, str] = {}
        for factor in sorted(block_sensitivity["block_factor"].unique()):
            subset = block_sensitivity[block_sensitivity["block_factor"] == factor]
            verdict, _, _, _ = _endpoint_h1_verdict(
                subset,
                endpoint,
                coverage_bar,
                bh_column="preregistered_bh_significant",
                require_complete=True,
                expected_cells=expected_primary_units,
            )
            per_factor[f"{float(factor):g}"] = verdict
            block_verdict_values.append(verdict)
            descriptive_verdict, _, _, _ = _endpoint_h1_verdict(
                subset,
                endpoint,
                coverage_bar,
                bh_column="descriptive_bh_significant",
            )
            descriptive_per_factor[f"{float(factor):g}"] = descriptive_verdict
        block_verdicts[endpoint] = per_factor
        descriptive_block_verdicts[endpoint] = descriptive_per_factor
    if "NOT_ESTIMABLE" in block_verdict_values or h1_verdict == "NOT_ESTIMABLE":
        h4_verdict = "NOT_ESTIMABLE"
    elif all(len(set(value.values())) == 1 for value in block_verdicts.values()):
        h4_verdict = "PASS"
    else:
        h4_verdict = "FAIL"

    h5_details: dict[str, Any] = {}
    h5_estimable_pass = True
    for endpoint in endpoints:
        endpoint_rows = primary[primary["endpoint"] == endpoint]
        spatial_count, global_count = _h5_separated_counts(endpoint_rows)
        h5_details[f"spatial_{endpoint.lower()}_separated"] = spatial_count
        h5_details[f"global_{endpoint.lower()}_separated"] = global_count
        h5_details[f"attenuated_{endpoint.lower()}"] = bool(spatial_count < global_count)
        h5_estimable_pass = h5_estimable_pass and bool(spatial_count < global_count)
    if n_cells == 0:
        h5_verdict = "NOT_ESTIMABLE"
    elif not h5_estimable_pass:
        h5_verdict = "FAIL"
    elif len(primary_nominal) < expected_primary_cells:
        h5_verdict = "NOT_ESTIMABLE"
    else:
        h5_verdict = "PASS"
    h5_details["bh_definition"] = (
        "spatial=above_q95 & descriptive_bh_significant; "
        "global=global_above_q95 & global_bh_significant"
    )
    h5_details["cell_scope"] = (
        f"{len(primary)}/{expected_primary_cells} estimable primary cells"
    )

    return {
        "H1": {
            "hypothesis": "Observed domain concordance is separated from the local spatial-block null",
            "verdict": h1_verdict,
            "coverage_bar": coverage_bar,
            "details": h1_details,
            "reduced_descriptive_set": h1_descriptive,
            "n_primary_units_nominal": int(primary_nominal["unit_id"].nunique()),
            "n_primary_units_estimable": n_estimable_units,
            "not_estimable_units": sorted(
                primary_nominal.loc[
                    primary_nominal["estimability"] == NOT_ESTIMABLE, "unit_id"
                ].astype(str).unique().tolist()
            ),
        },
        "H2": {
            "hypothesis": "The local spatial null is non-degenerate and raises the chance bar",
            "verdict": h2_verdict,
            "n_cells": n_cells,
            "n_cells_nominal": int(len(primary_nominal)),
            "cell_scope": "estimable primary cells only",
            "n_positive_null_median": n_positive_median,
            "n_null_q95_above_median": n_ordered,
            "n_null_q95_above_q05": n_spread,
            "n_spatial_q95_above_global": n_above_global,
            "spatial_q95_above_global_coverage": above_global_coverage,
            "spatial_q95_above_global_bar": positivity_bar,
        },
        "H3": {
            "hypothesis": "The spatial-null H1 verdict is stable under leave-one-dataset-out",
            "verdict": h3_verdict,
            "full_h1_verdict": h1_verdict,
            "loo_verdicts": loo_verdicts,
            "reduced_descriptive_loo_verdicts": descriptive_loo_verdicts,
            "n_loo_subsets": int(len(dataset_loo)),
        },
        "H4": {
            "hypothesis": "The spatial-null H1 verdict is stable across block factors 5, 10, and 20",
            "verdict": h4_verdict,
            "per_endpoint_factor_verdicts": block_verdicts,
            "reduced_descriptive_per_endpoint_factor_verdicts": descriptive_block_verdicts,
        },
        "H5": {
            "hypothesis": "Local spatial structure attenuates the global-null separation",
            "verdict": h5_verdict,
            "details": h5_details,
        },
    }


def key_numbers(
    unit_table: pd.DataFrame,
    block_sensitivity: pd.DataFrame,
    block_table: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    verdicts: dict[str, dict[str, Any]],
    config: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    primary_nominal = unit_table[unit_table["role"] == "primary"]
    primary = primary_nominal[primary_nominal["estimability"] == ESTIMABLE]
    not_estimable = primary_nominal[primary_nominal["estimability"] == NOT_ESTIMABLE]
    sensitivity = unit_table[unit_table["role"] != "primary"]
    factors = sorted(block_sensitivity["block_factor"].unique().tolist())
    estimable_units = int(unit_table.loc[unit_table["estimability"] == ESTIMABLE, "unit_id"].nunique())
    nominal_units = int(unit_table["unit_id"].nunique())
    unused = inputs["unused_coordinate_barcodes"]
    unused_by_dataset = {
        str(dataset_key): int(frame["barcode"].nunique())
        for dataset_key, frame in unused.groupby("dataset_key", sort=True)
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
        "block_factors": [float(value) for value in factors],
        "primary_block_factor": float(config["spatial"]["primary_block_factor"]),
        "permutation_draws_per_unit_factor": int(config["thresholds"]["permutation_draws"]),
        "design_primary_permutations": int(
            nominal_units * int(config["thresholds"]["permutation_draws"])
        ),
        "design_total_permutations_across_block_factors": int(
            nominal_units
            * len(factors)
            * int(config["thresholds"]["permutation_draws"])
        ),
        "executed_primary_permutations": int(
            estimable_units * int(config["thresholds"]["permutation_draws"])
        ),
        "executed_total_permutations_across_block_factors": int(
            estimable_units * len(factors) * int(config["thresholds"]["permutation_draws"])
        ),
        "permutation_seed": int(config["seeds"]["permutation"]),
        "bh_alpha": float(config["thresholds"]["bh_alpha"]),
        "chance_coverage_bar": float(config["thresholds"]["chance_coverage"]),
        "coordinate_contract": {
            "rule": "prediction_barcodes_must_exactly_equal_coordinate_barcodes",
            "n_units_exact": int((inputs["unit_coordinate_audit"]["coordinate_barcodes_exact"]).sum()),
            "n_units_not_estimable": int(not_estimable["unit_id"].nunique()),
            "not_estimable_units": sorted(not_estimable["unit_id"].astype(str).unique().tolist()),
            "unused_coordinate_barcodes_by_dataset": unused_by_dataset,
            "n_unused_coordinate_rows": int(len(unused)),
            "audit_table": "tables/snull1_coordinate_audit.tsv",
            "unused_barcode_table": "tables/snull1_unused_coordinate_barcodes.tsv",
        },
        "max_observed_ari": float(primary.loc[primary["endpoint"] == "ARI", "observed"].max()),
        "max_observed_nmi": float(primary.loc[primary["endpoint"] == "NMI", "observed"].max()),
        "median_observed_ari": float(primary.loc[primary["endpoint"] == "ARI", "observed"].median()),
        "median_observed_nmi": float(primary.loc[primary["endpoint"] == "NMI", "observed"].median()),
        "observed_metric_scope": "ESTIMABLE_PRIMARY_CELLS_ONLY",
        "dataset_loo": dataset_loo.to_dict(orient="records"),
        "block_diagnostics": block_table.to_dict(orient="records"),
        "hypotheses": verdicts,
        "claim_boundary": [
            "Local spatial block-permutation null, not independent truth and not a full autocorrelation-preserving null at all scales.",
            "DS1 only; it does not provide the missing GSE205055 human domain-label null.",
            "No method, representation, biological, causal, depth-effect, or universal-ranking claim is authorized.",
            "H1, H3, and H4 are NOT_ESTIMABLE on the frozen 14-unit set after exact coordinate-contract enforcement; reduced-set summaries are descriptive only.",
            "H2 and H5 are literal failures on the 22 estimable primary cells; they do not establish a spatially robust domain-recovery result.",
            "Failure or non-estimability of H1 does not retroactively invalidate NM-NULL1's pre-specified global-null verdict.",
        ],
    }


def verify_freeze(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_project_path(root, config["protocol"]["freeze_manifest"])
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    if freeze.get("frozen_before_analysis") is not True:
        raise ValueError("NM-SNULL1 requires an explicit pre-analysis freeze")
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
    block_sensitivity: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    block_table: pd.DataFrame,
    verdicts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = config["expected"]
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
    nonzero_q = unit_table.dropna(subset=["null_median", "null_q05", "null_q95", "null_q99"])
    nonzero_q_blocks = block_sensitivity.dropna(
        subset=["null_median", "null_q05", "null_q95", "null_q99"]
    )
    checks = {
        "freeze_hashes_match": all(item["match"] for item in freeze_checks),
        "units_total_match": int(unit_table["unit_id"].nunique()) == int(expected["units_total"]),
        "primary_units_match": int(primary["unit_id"].nunique()) == int(expected["primary_units"]),
        "estimable_primary_units_match": int(primary_estimable["unit_id"].nunique())
        == int(expected["estimable_primary_units"]),
        "not_estimable_units_match": int(primary_not_estimable["unit_id"].nunique())
        == int(expected["not_estimable_units"]),
        "not_estimable_unit_ids_match": observed_not_estimable == expected_not_estimable,
        "sensitivity_units_match": int(sensitivity["unit_id"].nunique()) == int(expected["sensitivity_units"]),
        "primary_cells_match": len(primary) == int(expected["primary_cells"]),
        "estimable_primary_cells_match": len(primary_estimable)
        == int(expected["estimable_primary_cells"]),
        "not_estimable_primary_cells_match": len(primary_not_estimable)
        == int(expected["not_estimable_primary_cells"]),
        "sensitivity_cells_match": len(sensitivity) == int(expected["sensitivity_cells"]),
        "block_sensitivity_rows_match": len(block_sensitivity) == int(expected["block_sensitivity_rows"]),
        "dataset_loo_rows_match": len(dataset_loo) == int(expected["dataset_loo_rows"]),
        "hypotheses_present": set(verdicts) == {"H1", "H2", "H3", "H4", "H5"},
        "estimable_permutation_draws_match": bool(
            (
                unit_table.loc[unit_table["estimability"] == ESTIMABLE, "permutation_draws"]
                == int(config["thresholds"]["permutation_draws"])
            ).all()
            and (
                block_sensitivity.loc[
                    block_sensitivity["estimability"] == ESTIMABLE, "permutation_draws"
                ]
                == int(config["thresholds"]["permutation_draws"])
            ).all()
        ),
        "not_estimable_permutation_draws_zero": bool(
            (unit_table.loc[unit_table["estimability"] == NOT_ESTIMABLE, "permutation_draws"] == 0).all()
            and (
                block_sensitivity.loc[
                    block_sensitivity["estimability"] == NOT_ESTIMABLE, "permutation_draws"
                ]
                == 0
            ).all()
        ),
        "empirical_p_in_range": bool(
            unit_table["empirical_p"].dropna().between(0.0, 1.0).all()
            and block_sensitivity["empirical_p"].dropna().between(0.0, 1.0).all()
        ),
        "descriptive_bh_q_in_range": bool(
            unit_table["descriptive_bh_q"].dropna().between(0.0, 1.0).all()
            and block_sensitivity["descriptive_bh_q"].dropna().between(0.0, 1.0).all()
        ),
        "descriptive_bh_scope_complete": bool(
            int(unit_table["descriptive_bh_q"].notna().sum()) == len(primary_estimable)
            and (
                unit_table.loc[
                    unit_table["descriptive_bh_q"].notna(), "estimability"
                ]
                == ESTIMABLE
            ).all()
            and (
                unit_table.loc[
                    unit_table["descriptive_bh_q"].notna(), "role"
                ]
                == "primary"
            ).all()
        ),
        "null_quantiles_ordered": bool(
            (nonzero_q["null_q05"] <= nonzero_q["null_median"]).all()
            and (nonzero_q["null_median"] <= nonzero_q["null_q95"]).all()
            and (nonzero_q["null_q95"] <= nonzero_q["null_q99"]).all()
            and (nonzero_q_blocks["null_q05"] <= nonzero_q_blocks["null_median"]).all()
            and (nonzero_q_blocks["null_median"] <= nonzero_q_blocks["null_q95"]).all()
            and (nonzero_q_blocks["null_q95"] <= nonzero_q_blocks["null_q99"]).all()
        ),
        "no_duplicate_unit_endpoint_factor": bool(
            not block_sensitivity.duplicated(["unit_id", "endpoint", "block_factor"]).any()
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
        ) == int(expected["spatial_sections"]),
        "block_partition_complete": bool(
            len(block_table) > 0
            and (block_table["n_blocks"] > 0).all()
            and (block_table["min_spots_per_block"] >= 1).all()
            and int(block_table["dataset_key"].nunique())
            == int(
                unit_audit.loc[
                    unit_audit["estimability"] == ESTIMABLE, "dataset_key"
                ].nunique()
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


def runtime_environment() -> dict[str, Any]:
    """Record the actual CPU, address-space, and BLAS execution policy."""

    affinity = sorted(int(value) for value in os.sched_getaffinity(0))
    address_space_soft, address_space_hard = resource.getrlimit(resource.RLIMIT_AS)
    return {
        "cpu_affinity": affinity,
        "cpu_affinity_count": int(len(affinity)),
        "nice": int(os.getpriority(os.PRIO_PROCESS, 0)),
        "address_space_limit_bytes": (
            None if address_space_soft == resource.RLIM_INFINITY else int(address_space_soft)
        ),
        "address_space_limit_hard_bytes": (
            None if address_space_hard == resource.RLIM_INFINITY else int(address_space_hard)
        ),
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
        LOGGER.info("NM-SNULL1 start %s", started_utc)
        freeze_checks = verify_freeze(config, root)
        LOGGER.info("Freeze hashes verified: %s", len(freeze_checks))
        inputs = load_inputs(config, root)
        validate_inputs(inputs, config)
        LOGGER.info("Frozen DS1 inputs and coordinates validated")
        unit_table, block_table = build_unit_table(inputs, config)
        block_sensitivity = build_block_sensitivity(inputs, config, unit_table)
        dataset_loo = build_dataset_loo(unit_table, config)
        verdicts = hypothesis_verdicts(unit_table, block_sensitivity, dataset_loo, config)
        key = key_numbers(
            unit_table,
            block_sensitivity,
            block_table,
            dataset_loo,
            verdicts,
            config,
            inputs,
        )

        write_tsv(unit_table, attempt_root / "tables" / "snull1_unit_metrics.tsv")
        write_tsv(block_sensitivity, attempt_root / "tables" / "snull1_block_sensitivity.tsv")
        write_tsv(dataset_loo, attempt_root / "tables" / "snull1_dataset_loo.tsv")
        write_tsv(block_table, attempt_root / "tables" / "snull1_spatial_blocks.tsv")
        write_tsv(
            inputs["unit_coordinate_audit"],
            attempt_root / "tables" / "snull1_coordinate_audit.tsv",
        )
        write_tsv(
            inputs["unused_coordinate_barcodes"],
            attempt_root / "tables" / "snull1_unused_coordinate_barcodes.tsv",
        )
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
            attempt_root / "tables" / "snull1_hypotheses.tsv",
        )
        write_json(key, attempt_root / "key_numbers.json")
        write_tsv(pd.DataFrame(freeze_checks), attempt_root / "manifests" / "input_checksums.tsv")

        self_check(
            config,
            attempt_root,
            freeze_checks,
            inputs,
            unit_table,
            block_sensitivity,
            dataset_loo,
            block_table,
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
                "null_positivity_coverage": float(config["thresholds"]["null_positivity_coverage"]),
                "primary_block_factor": float(config["spatial"]["primary_block_factor"]),
                "block_factors": [float(value) for value in config["spatial"]["block_factors"]],
                "primary_seed": int(config["scope"]["primary_seed"]),
                "numeric_tolerance": float(config["thresholds"]["numeric_tolerance"]),
            },
            "input_hashes": {
                str(item["path"]): str(item["sha256"]) for item in freeze_manifest["inputs"]
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
                    "blas_threads": int(config["resources"].get("blas_threads", 1)),
                },
                "observed": runtime_environment(),
            },
            "outputs": {
                "unit_metrics": "tables/snull1_unit_metrics.tsv",
                "block_sensitivity": "tables/snull1_block_sensitivity.tsv",
                "dataset_loo": "tables/snull1_dataset_loo.tsv",
                "spatial_blocks": "tables/snull1_spatial_blocks.tsv",
                "coordinate_audit": "tables/snull1_coordinate_audit.tsv",
                "unused_coordinate_barcodes": "tables/snull1_unused_coordinate_barcodes.tsv",
                "hypotheses": "tables/snull1_hypotheses.tsv",
                "key_numbers": "key_numbers.json",
            },
        }
        write_json(manifest, attempt_root / "manifests" / "run_manifest.json")
        LOGGER.info("NM-SNULL1 complete: %s", json.dumps(verdicts, sort_keys=True))
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
    "block_assignment",
    "build_block_sensitivity",
    "build_dataset_loo",
    "build_unit_table",
    "build_unused_coordinate_table",
    "coordinate_contract_audit",
    "hypothesis_verdicts",
    "key_numbers",
    "load_coordinates",
    "load_inputs",
    "permutation_null_spatial",
    "run",
    "runtime_environment",
    "self_check",
    "validate_inputs",
    "verify_freeze",
]
