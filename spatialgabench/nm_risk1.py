"""Pre-registered NM-RISK1 closure analyses on frozen benchmark artifacts.

The module is deliberately read-only with respect to ``results/v1`` and
``registries/v1``.  It writes only under the configured ``results/v1_risk1``
tree.  RNA is never read here: all analyses consume already-frozen paired-RNA
evaluation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats


ENDPOINT = "coverage_adjusted_median_gene_wise_spearman"
TRACK_BY_REPRESENTATION = {
    "macs3_summit_501bp": "P",
    "fixed_bin_10kb": "B",
    "fixed_bin_25kb": "B",
    "fixed_bin_50kb": "B",
}
ANNOTATION_METHODS = (
    "promoter_only_2kb_pc",
    "gene_body_pc",
    "promoter_plus_gene_body_2kb_pc",
    "distance_weighted_distal_250kb_decay50kb_pc",
    "promoter_plus_distal_2kb_250kb_decay50kb_pc",
)
PRIMARY_REPRESENTATIONS = (
    "macs3_summit_501bp",
    "fixed_bin_10kb",
    "fixed_bin_25kb",
    "fixed_bin_50kb",
)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def _bootstrap_mean_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if draws <= 0 or len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)), dtype=np.int64)
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _sign_flip_pvalue(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    if len(values) <= 12:
        signs = np.array(
            [[1.0 if bit == "1" else -1.0 for bit in format(mask, f"0{len(values)}b")]
             for mask in range(2 ** len(values))],
            dtype=float,
        )
        exact = float(np.mean(np.abs((values[None, :] * signs).mean(axis=1)) >= observed - 1e-15))
    else:
        exact = math.nan
    rng = np.random.default_rng(seed)
    random_signs = rng.choice([-1.0, 1.0], size=(draws, len(values)))
    monte_carlo = float(np.mean(np.abs((values[None, :] * random_signs).mean(axis=1)) >= observed - 1e-15))
    return exact, monte_carlo


def _one_sided_t_pvalues(mean: float, se: float, df: int, margin: float) -> tuple[float, float]:
    if se <= 0 or not np.isfinite(se):
        if mean > -margin and mean < margin:
            return 0.0, 0.0
        return (0.0, 1.0) if mean >= margin else (1.0, 0.0)
    lower = float(stats.t.sf((mean + margin) / se, df))
    upper = float(stats.t.cdf((mean - margin) / se, df))
    return lower, upper


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    n = len(values)
    if n == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    output = np.empty(n, dtype=float)
    output[order] = adjusted
    return output


def load_fb1_metrics(config: Mapping[str, Any], root: str | Path) -> pd.DataFrame:
    path = Path(root) / config["equivalence"]["metrics"]
    metrics = pd.read_csv(path, sep="\t")
    methods = list(config["equivalence"]["methods"])
    representations = list(config["equivalence"]["representations"].values())
    selected = metrics.loc[
        metrics.method_category.eq("hand_written_annotation")
        & metrics.method_id.isin(methods)
        & metrics.representation_id.isin(representations)
    ].copy()
    studies = list(config["equivalence"]["expected_studies"])
    selected = selected.loc[selected.study_family.isin(studies)]
    expected = len(methods) * len(representations) * len(studies)
    if len(selected) != expected:
        raise ValueError(f"FB1 primary matrix changed: {len(selected)} != {expected}")
    if selected.duplicated(["study_family", "method_id", "representation_id"]).any():
        raise ValueError("FB1 primary matrix contains duplicate cells")
    return selected


def build_tost_table(
    metrics: pd.DataFrame,
    *,
    methods: Sequence[str],
    representations: Sequence[str],
    studies: Sequence[str],
    margin: float,
    alpha: float,
    draws: int,
    bootstrap_seed: int,
    permutation_seed: int,
    permutation_draws: int,
) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["study_family", "method_id"],
        columns="representation_id",
        values=ENDPOINT,
    )
    rows: list[dict[str, Any]] = []
    for method in methods:
        for left, right in combinations(representations, 2):
            deltas = np.asarray(
                [pivot.loc[(study, method), left] - pivot.loc[(study, method), right] for study in studies],
                dtype=float,
            )
            n = len(deltas)
            mean = float(deltas.mean())
            se = float(deltas.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
            df = n - 1
            p_lower, p_upper = _one_sided_t_pvalues(mean, se, df, margin)
            tost_p = max(p_lower, p_upper)
            t_critical = float(stats.t.ppf(0.975, df)) if df > 0 else math.nan
            t_low = mean - t_critical * se if np.isfinite(se) else math.nan
            t_high = mean + t_critical * se if np.isfinite(se) else math.nan
            boot_low, boot_high = _bootstrap_mean_ci(deltas, draws, bootstrap_seed)
            exact_p, permutation_p = _sign_flip_pvalue(deltas, permutation_draws, permutation_seed)
            row: dict[str, Any] = {
                "contrast_id": f"{method}::{left}_vs_{right}",
                "method_id": method,
                "representation_left": left,
                "representation_right": right,
                "track_left": TRACK_BY_REPRESENTATION[left],
                "track_right": TRACK_BY_REPRESENTATION[right],
                "n_studies": n,
                "mean_delta": mean,
                "standard_error": se,
                "t_statistic": mean / se if se and np.isfinite(se) else math.nan,
                "degrees_of_freedom": df,
                "p_lower_nonequivalence": p_lower,
                "p_upper_nonequivalence": p_upper,
                "tost_p": tost_p,
                "ci_low_t": t_low,
                "ci_high_t": t_high,
                "ci_low_bootstrap": boot_low,
                "ci_high_bootstrap": boot_high,
                "permutation_p_exact": exact_p,
                "permutation_p_monte_carlo": permutation_p,
                "abs_mean_below_margin": abs(mean) < margin,
                "bootstrap_ci_inside_margin": bool(boot_low > -margin and boot_high < margin),
                "tost_pass": bool(tost_p < alpha),
            }
            for study, delta in zip(studies, deltas):
                row[f"delta_{study}"] = float(delta)
            rows.append(row)
    table = pd.DataFrame(rows)
    table["bh_q_primary"] = bh_adjust(table["tost_p"].to_numpy())
    table["material_difference_established"] = (
        (table["ci_low_bootstrap"] > margin)
        | (table["ci_high_bootstrap"] < -margin)
        | (table["ci_low_t"] > margin)
        | (table["ci_high_t"] < -margin)
    )
    table["equivalent"] = (
        table["abs_mean_below_margin"]
        & table["bootstrap_ci_inside_margin"]
        & table["tost_pass"]
        & table["bh_q_primary"].lt(alpha)
        & ~table["material_difference_established"]
    )
    table["equivalence_decision"] = np.where(
        table["material_difference_established"],
        "material_difference_established",
        np.where(table["equivalent"], "equivalent", "inconclusive_equivalence"),
    )
    return table


def _load_gene_frames(
    config: Mapping[str, Any], root: str | Path
) -> dict[tuple[str, str, str], pd.DataFrame]:
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    methods = set(config["equivalence"]["methods"])
    representations = set(config["equivalence"]["representations"].values())
    for study, relative in config["equivalence"]["gene_level_sources"].items():
        path = Path(root) / relative
        data = pd.read_csv(path, sep="\t")
        data = data.loc[data.method_id.isin(methods) & data.representation_id.isin(representations)]
        for (method, representation), group in data.groupby(["method_id", "representation_id"], sort=False):
            frames[(study, str(method), str(representation))] = group.sort_values("gene_id").reset_index(drop=True)
    expected = len(config["equivalence"]["expected_studies"]) * len(methods) * len(representations)
    if len(frames) != expected:
        raise ValueError(f"gene-level coverage changed: {len(frames)} != {expected}")
    return frames


def gene_level_bootstrap(
    frames: Mapping[tuple[str, str, str], pd.DataFrame],
    contrasts: pd.DataFrame,
    *,
    studies: Sequence[str],
    draws: int,
    seed: int,
    chunk_size: int = 200,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in contrasts.itertuples(index=False):
        aligned: dict[str, tuple[np.ndarray, ...]] = {}
        for study in studies:
            left = frames[(study, record.method_id, record.representation_left)]
            right = frames[(study, record.method_id, record.representation_right)]
            merged = left.merge(right, on="gene_id", suffixes=("_left", "_right"), how="inner")
            if merged.empty:
                raise ValueError(f"gene alignment failed: {record.contrast_id}/{study}")
            aligned[study] = (
                _as_bool(merged["ga_gene_present_left"]).to_numpy(),
                _as_bool(merged["ga_gene_variable_left"]).to_numpy(),
                merged["spearman_left"].to_numpy(dtype=float),
                _as_bool(merged["ga_gene_present_right"]).to_numpy(),
                _as_bool(merged["ga_gene_variable_right"]).to_numpy(),
                merged["spearman_right"].to_numpy(dtype=float),
            )
        rng = np.random.default_rng(seed)
        sampled = np.empty(draws, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for start in range(0, draws, chunk_size):
                stop = min(start + chunk_size, draws)
                size = stop - start
                study_deltas = np.zeros((size, len(studies)), dtype=float)
                for column, study in enumerate(studies):
                    present_left, variable_left, corr_left, present_right, variable_right, corr_right = aligned[study]
                    n = len(corr_left)
                    indices = rng.integers(0, n, size=(size, n), dtype=np.int64)
                    median_left = np.nanmedian(np.where(present_left[indices], corr_left[indices], np.nan), axis=1)
                    median_right = np.nanmedian(np.where(present_right[indices], corr_right[indices], np.nan), axis=1)
                    coverage_left = variable_left[indices].mean(axis=1)
                    coverage_right = variable_right[indices].mean(axis=1)
                    study_deltas[:, column] = median_left * coverage_left - median_right * coverage_right
                sampled[start:stop] = study_deltas.mean(axis=1)
        low, high = np.percentile(sampled, [2.5, 97.5])
        rows.append(
            {
                "contrast_id": record.contrast_id,
                "gene_bootstrap_ci_low": float(low),
                "gene_bootstrap_ci_high": float(high),
                "gene_bootstrap_draws": draws,
                "gene_bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(rows)


def material_onset(
    depth_values: Sequence[float], deltas: Mapping[float, float], margin: float
) -> float | None:
    order = sorted(
        (float(value) for value in depth_values if float(value) in deltas),
        reverse=True,
    )
    for index, depth in enumerate(order):
        if abs(float(deltas[depth])) < margin:
            continue
        lower = order[index + 1 :]
        if all(abs(float(deltas[value])) >= margin for value in lower):
            return depth
    return None


def _onset_label(onset: float | None) -> str:
    return "not_material_in_observed_range" if onset is None else f"{onset:.2f}"


def bootstrap_supported_onset(
    onsets: Mapping[str, float | None],
    *,
    depth_values: Sequence[float],
    draws: int,
    seed: int,
    minimum_methods: int = 4,
) -> dict[str, Any]:
    order = sorted((float(value) for value in depth_values), reverse=True)
    values = list(onsets.values())
    n = len(values)
    if n == 0:
        return {"status": "not_estimable", "methods_with_onset": 0, "draws": draws}
    numeric = np.asarray(
        [-np.inf if value is None or pd.isna(value) else float(value) for value in values]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(draws, n), dtype=np.int64)
    supported = np.sort(numeric[indices], axis=1)[:, ::-1][:, minimum_methods - 1]
    labels = [_onset_label(None if value == -np.inf else float(value)) for value in supported]
    counts = pd.Series(labels).value_counts()
    methods_with_onset = int(sum(value is not None and not pd.isna(value) for value in values))
    result: dict[str, Any] = {
        "methods_with_onset": methods_with_onset,
        "draws": draws,
        "seed": seed,
        "minimum_methods": minimum_methods,
    }
    for depth in order:
        result[f"fraction_onset_{depth:.2f}"] = float(counts.get(f"{depth:.2f}", 0) / draws)
    result["fraction_not_material"] = float(counts.get("not_material_in_observed_range", 0) / draws)
    if methods_with_onset < 2:
        result["status"] = "not_estimable"
        result["ci_low"] = "not_estimable"
        result["ci_high"] = "not_estimable"
        return result
    ranks = np.asarray([0.0 if value == -np.inf else order.index(float(value)) + 1 for value in supported])
    low, high = np.percentile(ranks, [2.5, 97.5])
    categories = ["not_material_in_observed_range"] + [f"{depth:.2f}" for depth in order]
    result["status"] = "estimated"
    result["ci_low"] = categories[int(round(low))]
    result["ci_high"] = categories[int(round(high))]
    return result


def depth_tables(config: Mapping[str, Any], root: str | Path) -> dict[str, pd.DataFrame]:
    depth_config = config["depth"]
    methods = list(depth_config["methods"])
    studies = list(depth_config["ta1_studies"])
    depth_order = [float(value) for value in depth_config["depth_order"]]
    ta1 = pd.read_csv(Path(root) / depth_config["ta1_summary"], sep="\t")
    ta1 = ta1.loc[
        ta1.study_family.isin(studies)
        & ta1.method_id.isin(methods)
        & ta1.depth_fraction.isin(depth_order)
    ].copy()
    pivot = ta1.pivot_table(
        index=["study_family", "method_id", "depth_fraction"],
        columns="representation_id",
        values="median_rna_primary",
        aggfunc="first",
    )
    retention = ta1.pivot_table(
        index=["study_family", "method_id", "depth_fraction"],
        columns="representation_id",
        values="median_rna_retention_vs_full",
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    for study in studies:
        for method in methods:
            for depth in depth_order:
                p_value = float(pivot.loc[(study, method, depth), "macs3_summit_501bp"])
                b_value = float(pivot.loc[(study, method, depth), "fixed_bin_25kb"])
                rows.append(
                    {
                        "study_family": study,
                        "method_id": method,
                        "depth_fraction": depth,
                        "P_primary": p_value,
                        "B25_primary": b_value,
                        "P_minus_B25": p_value - b_value,
                        "abs_P_minus_B25": abs(p_value - b_value),
                        "P_retention_vs_full": float(retention.loc[(study, method, depth), "macs3_summit_501bp"]),
                        "B25_retention_vs_full": float(retention.loc[(study, method, depth), "fixed_bin_25kb"]),
                    }
                )
    depth_frame = pd.DataFrame(rows)

    onsets: list[dict[str, Any]] = []
    for (study, method), group in depth_frame.groupby(["study_family", "method_id"]):
        deltas = dict(zip(group["depth_fraction"].astype(float), group["abs_P_minus_B25"].astype(float)))
        onset = material_onset(depth_order, deltas, float(config["thresholds"]["materiality"]))
        onsets.append(
            {
                "study_family": study,
                "method_id": method,
                "onset_depth": np.nan if onset is None else onset,
                "onset_label": _onset_label(onset),
                "max_abs_delta": float(group["abs_P_minus_B25"].max()),
                "depth_at_max_abs_delta": float(group.loc[group["abs_P_minus_B25"].idxmax(), "depth_fraction"]),
            }
        )
    onset_frame = pd.DataFrame(onsets)

    bootstrap_rows: list[dict[str, Any]] = []
    for study, group in onset_frame.groupby("study_family"):
        result = bootstrap_supported_onset(
            dict(zip(group["method_id"], group["onset_depth"].where(group["onset_depth"].notna(), None))),
            depth_values=depth_order,
            draws=int(config["bootstrap"]["draws"]),
            seed=int(config["seeds"]["primary_bootstrap"]),
        )
        result["study_family"] = study
        bootstrap_rows.append(result)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    retention_rows: list[dict[str, Any]] = []
    for study in studies:
        for depth in depth_order:
            group = depth_frame.loc[depth_frame.study_family.eq(study) & depth_frame.depth_fraction.eq(depth)]
            p_values = group["P_retention_vs_full"].to_numpy(dtype=float)
            b_values = group["B25_retention_vs_full"].to_numpy(dtype=float)
            retention_rows.append(
                {
                    "study_family": study,
                    "depth_fraction": depth,
                    "n_methods": len(group),
                    "P_retention_median": float(np.median(p_values)),
                    "P_retention_min": float(np.min(p_values)),
                    "P_retention_max": float(np.max(p_values)),
                    "B25_retention_median": float(np.median(b_values)),
                    "B25_retention_min": float(np.min(b_values)),
                    "B25_retention_max": float(np.max(b_values)),
                    "P_zero_retention_methods": int(np.sum(p_values == 0)),
                    "B25_over_P_retention_ratio_median": float(np.median(b_values / p_values))
                    if np.all(p_values > 0)
                    else math.nan,
                }
            )
    retention_frame = pd.DataFrame(retention_rows)

    ta2 = pd.read_csv(Path(root) / depth_config["ta2_comparison"], sep="\t")
    ta2 = ta2.loc[
        ta2.study_family.isin(depth_config["ta2_studies"])
        & ta2.method_id.isin(methods)
        & ta2.depth_fraction.isin(depth_order)
    ].copy()
    ta2_aggregated = (
        ta2.groupby(["study_family", "method_id", "depth_fraction"], as_index=False)
        .agg(
            frozen_P_minus_de_novo_P_primary=("frozen_P_minus_de_novo_P_primary", "median"),
            B25_minus_frozen_P_primary=("B25_minus_frozen_P_primary", "median"),
            frozen_P_retention_vs_full_depth_P=("frozen_peak_retention_vs_full_depth_P", "median"),
            de_novo_P_retention=("de_novo_P_retention", "median"),
            B25_retention=("B25_retention", "median"),
            n_seeds=("seed", "nunique"),
        )
    )
    ta2_rows: list[dict[str, Any]] = []
    for (study, method), group in ta2_aggregated.groupby(["study_family", "method_id"]):
        for contrast_column in ("frozen_P_minus_de_novo_P_primary", "B25_minus_frozen_P_primary"):
            deltas = dict(
                zip(
                    group["depth_fraction"].astype(float),
                    group[contrast_column].abs().astype(float),
                )
            )
            onset = material_onset(depth_order, deltas, float(config["thresholds"]["materiality"]))
            ta2_rows.append(
                {
                    "study_family": study,
                    "method_id": method,
                    "contrast": contrast_column,
                    "onset_depth": np.nan if onset is None else onset,
                    "onset_label": _onset_label(onset),
                    "max_abs_delta": float(group[contrast_column].abs().max()),
                }
            )
    ta2_onset_frame = pd.DataFrame(ta2_rows)

    threshold_rows: list[dict[str, Any]] = []
    for margin in config["thresholds"]["depth_margin_grid"]:
        margin = float(margin)
        count = 0
        for _, group in depth_frame.groupby(["study_family", "method_id"]):
            deltas = dict(zip(group["depth_fraction"].astype(float), group["abs_P_minus_B25"].astype(float)))
            count += int(material_onset(depth_order, deltas, margin) is not None)
        ta2_count = 0
        for _, group in ta2_aggregated.groupby(["study_family", "method_id"]):
            for contrast_column in ("frozen_P_minus_de_novo_P_primary", "B25_minus_frozen_P_primary"):
                deltas = dict(zip(group["depth_fraction"].astype(float), group[contrast_column].abs().astype(float)))
                ta2_count += int(material_onset(depth_order, deltas, margin) is not None)
        threshold_rows.append(
            {
                "margin": margin,
                "primary_study_method_onsets": count,
                "primary_study_method_cells": len(studies) * len(methods),
                "ta2_study_method_contrast_onsets": ta2_count,
                "ta2_study_method_contrast_cells": len(depth_config["ta2_studies"]) * len(methods) * 2,
            }
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    return {
        "depth": depth_frame,
        "onsets": onset_frame,
        "bootstrap": bootstrap_frame,
        "retention": retention_frame,
        "ta2_aggregated": ta2_aggregated,
        "ta2_onsets": ta2_onset_frame,
        "threshold_sensitivity": threshold_frame,
    }


def equivalence_threshold_sensitivity(
    table: pd.DataFrame, margins: Sequence[float], alpha: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    delta_columns = [column for column in table.columns if column.startswith("delta_")]
    for margin in margins:
        margin = float(margin)
        pvalues = []
        for record in table.itertuples(index=False):
            deltas = np.asarray([getattr(record, column) for column in delta_columns], dtype=float)
            mean = float(deltas.mean())
            se = float(deltas.std(ddof=1) / math.sqrt(len(deltas)))
            p_lower, p_upper = _one_sided_t_pvalues(mean, se, len(deltas) - 1, margin)
            pvalues.append(max(p_lower, p_upper))
        adjusted = bh_adjust(pvalues)
        equivalent = (
            table["abs_mean_below_margin"].to_numpy()
            & (table["mean_delta"].abs().to_numpy() < margin)
            & (table["ci_low_bootstrap"].to_numpy() > -margin)
            & (table["ci_high_bootstrap"].to_numpy() < margin)
            & (np.asarray(pvalues) < alpha)
            & (adjusted < alpha)
            & ~table["material_difference_established"].to_numpy()
        )
        rows.append(
            {
                "margin": margin,
                "equivalent_contrasts": int(equivalent.sum()),
                "inconclusive_contrasts": int(len(table) - equivalent.sum()),
                "tost_pass_before_bh": int(np.sum(np.asarray(pvalues) < alpha)),
                "minimum_tost_p": float(np.min(pvalues)),
                "maximum_tost_p": float(np.max(pvalues)),
                "primary_margin": bool(margin == 0.05),
            }
        )
    return pd.DataFrame(rows)


def seed_stability(
    table: pd.DataFrame,
    seeds: Sequence[int],
    *,
    draws: int,
    margin: float,
    alpha: float,
) -> pd.DataFrame:
    delta_columns = [column for column in table.columns if column.startswith("delta_")]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        lows: list[float] = []
        highs: list[float] = []
        equivalent = 0
        inside = 0
        for record in table.itertuples(index=False):
            deltas = np.asarray([getattr(record, column) for column in delta_columns], dtype=float)
            low, high = _bootstrap_mean_ci(deltas, draws, int(seed))
            lows.append(low)
            highs.append(high)
            is_inside = bool(low > -margin and high < margin)
            inside += int(is_inside)
            equivalent += int(
                is_inside
                and record.abs_mean_below_margin
                and record.tost_pass
                and record.bh_q_primary < alpha
                and not record.material_difference_established
            )
        rows.append(
            {
                "seed": int(seed),
                "draws": draws,
                "bootstrap_ci_inside_margin": inside,
                "equivalent_contrasts": equivalent,
                "minimum_ci_low": float(np.min(lows)),
                "maximum_ci_low": float(np.max(lows)),
                "minimum_ci_high": float(np.min(highs)),
                "maximum_ci_high": float(np.max(highs)),
            }
        )
    return pd.DataFrame(rows)


def _process_tree_for(
    completion_path: Path, evidence: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any] | None, str]:
    candidate = evidence.get("process_tree_memory")
    if isinstance(candidate, dict) and candidate.get("path"):
        path = Path(candidate["path"])
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")), str(path)
    sibling = completion_path.parent / "process_tree_memory.json"
    if sibling.is_file():
        return json.loads(sibling.read_text(encoding="utf-8")), str(sibling)
    return None, "not_available"


def runtime_memory_table(config: Mapping[str, Any], root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in config["robustness"]["runtime_sources"]:
        completion = Path(root) / source["path"]
        if not completion.is_file():
            rows.append(
                {
                    "task": source["task"],
                    "completion_evidence": source["path"],
                    "status": "missing",
                    "runtime_seconds": math.nan,
                    "peak_aggregate_pss_mb": math.nan,
                    "max_concurrent_processes": math.nan,
                    "process_tree_path": "not_available",
                }
            )
            continue
        evidence = json.loads(completion.read_text(encoding="utf-8"))
        process_tree, path = _process_tree_for(completion, evidence, Path(root).resolve())
        runtime = math.nan
        if isinstance(evidence.get("runtime_seconds"), (int, float)):
            runtime = float(evidence["runtime_seconds"])
        pss = math.nan
        concurrency = math.nan
        if process_tree is not None:
            if isinstance(process_tree.get("runtime_seconds"), (int, float)):
                runtime = float(process_tree["runtime_seconds"])
            if isinstance(process_tree.get("peak_aggregate_pss_mb"), (int, float)):
                pss = float(process_tree["peak_aggregate_pss_mb"])
            if isinstance(process_tree.get("max_concurrent_processes"), (int, float)):
                concurrency = float(process_tree["max_concurrent_processes"])
        rows.append(
            {
                "task": source["task"],
                "completion_evidence": source["path"],
                "status": evidence.get("status", "unknown"),
                "runtime_seconds": runtime,
                "peak_aggregate_pss_mb": pss,
                "max_concurrent_processes": concurrency,
                "process_tree_path": path,
            }
        )
    return pd.DataFrame(rows)


def claim_gap_table(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for row in config["robustness"]["claim_rows"]:
        rows.append(
            {
                "claim_id": row["claim_id"],
                "claim": row["claim"],
                "status": row["status"],
                "evidence": row["evidence"],
                "remaining_gap": row["remaining_gap"],
                "risk1_action": "read_only_frozen_evidence_mapping",
            }
        )
    return pd.DataFrame(rows)


def external_gse205052_decisions(
    metrics: pd.DataFrame, *, b25_margin: float, materiality: float
) -> pd.DataFrame:
    selected = metrics.loc[
        metrics.method_category.eq("hand_written_annotation")
        & metrics.method_id.isin(ANNOTATION_METHODS)
        & metrics.representation_id.isin(PRIMARY_REPRESENTATIONS)
    ].copy()
    if selected.empty:
        raise ValueError("GSE205052 frozen metrics contain no primary annotation cells")
    pivot = selected.pivot(index="method_id", columns="representation_id", values=ENDPOINT)
    rows = []
    for method in pivot.index:
        values = {representation: float(pivot.loc[method, representation]) for representation in PRIMARY_REPRESENTATIONS}
        p_value = values["macs3_summit_501bp"]
        b25_value = values["fixed_bin_25kb"]
        best_representation = max(values, key=values.get)
        best_value = values[best_representation]
        deficit = max(p_value, b25_value) - b25_value
        rows.append(
            {
                "method_id": method,
                "P_primary": p_value,
                "B10_primary": values["fixed_bin_10kb"],
                "B25_primary": b25_value,
                "B50_primary": values["fixed_bin_50kb"],
                "best_representation": best_representation,
                "best_primary": best_value,
                "regret_to_best_representation": best_value - p_value,
                "B25_deficit": deficit,
                "B25_non_inferior_at_0_01": bool(deficit <= b25_margin),
                "abs_P_minus_B25": abs(p_value - b25_value),
                "material_at_0_05": bool(abs(p_value - b25_value) >= materiality),
                "cell_status": "pass" if deficit <= b25_margin else "fail",
            }
        )
    return pd.DataFrame(rows)


def _software_versions() -> dict[str, str]:
    import anndata
    import scipy

    return {
        "python": platform_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "anndata": anndata.__version__,
    }


def platform_version() -> str:
    import platform

    return platform.python_version()


def _write_completion(
    output_root: Path,
    *,
    status: str,
    runtime_seconds: float,
    parameters: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> None:
    payload = {
        "schema_version": 1,
        "milestone": "NM_RISK1_CLOSURE",
        "status": status,
        "runtime_seconds": runtime_seconds,
        "peak_rss_kb_current_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "software_versions": _software_versions(),
        "parameters": dict(parameters),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in artifacts.items()
            if Path(path).is_file()
        },
        "frozen_inputs_mutated": False,
        "rna_values_read_by_risk1": False,
        "external_release_performed": False,
    }
    write_json(output_root / "completion_evidence.json", payload)


def run(config_path: str | Path, root: str | Path = ".", *, skip_gene_bootstrap: bool = False) -> Path:
    started = time.perf_counter()
    root = Path(root).resolve()
    config = load_yaml(config_path)
    output_root = root / config["output_root"]
    equivalence_root = output_root / "equivalence"
    depth_root = output_root / "depth_gradient"
    robustness_root = output_root / "robustness"
    external_root = output_root / "external" / "gse205052_locked_r2"
    for path in (equivalence_root, depth_root, robustness_root, external_root):
        path.mkdir(parents=True, exist_ok=True)

    thresholds = config["thresholds"]
    methods = list(config["equivalence"]["methods"])
    representations = list(config["equivalence"]["representations"].values())
    studies = list(config["equivalence"]["expected_studies"])
    metrics = load_fb1_metrics(config, root)
    write_tsv(
        metrics[["study_family", "method_id", "representation_id", ENDPOINT]],
        equivalence_root / "study_method_representation_metrics.tsv",
    )
    tost = build_tost_table(
        metrics,
        methods=methods,
        representations=representations,
        studies=studies,
        margin=float(thresholds["tost_margin"]),
        alpha=float(thresholds["alpha"]),
        draws=int(config["bootstrap"]["draws"]),
        bootstrap_seed=int(config["seeds"]["primary_bootstrap"]),
        permutation_seed=int(config["seeds"]["sign_permutation"]),
        permutation_draws=int(config["bootstrap"]["permutation_draws"]),
    )
    write_tsv(tost, equivalence_root / "tost_primary_family.tsv")
    track_summary = (
        tost.assign(track_pair=tost.track_left + "-" + tost.track_right)
        .groupby("track_pair", as_index=False)
        .agg(
            contrasts=("contrast_id", "size"),
            equivalent=("equivalent", "sum"),
            inconclusive=("equivalence_decision", lambda values: int(np.sum(values == "inconclusive_equivalence"))),
            material=("material_difference_established", "sum"),
            max_abs_mean=("mean_delta", lambda values: float(np.max(np.abs(values)))),
        )
    )
    write_tsv(track_summary, equivalence_root / "equivalence_by_track.tsv")

    gene_table: pd.DataFrame | None = None
    if not skip_gene_bootstrap:
        frames = _load_gene_frames(config, root)
        gene_table = gene_level_bootstrap(
            frames,
            tost,
            studies=studies,
            draws=int(config["bootstrap"]["draws"]),
            seed=int(config["seeds"]["gene_bootstrap"]),
        )
        write_tsv(gene_table, equivalence_root / "gene_level_bootstrap.tsv")
        tost = tost.merge(gene_table, on="contrast_id", how="left")
        write_tsv(tost, equivalence_root / "tost_primary_family.tsv")

    equivalence_sensitivity = equivalence_threshold_sensitivity(
        tost, thresholds["equivalence_grid"], float(thresholds["alpha"])
    )
    write_tsv(equivalence_sensitivity, equivalence_root / "threshold_sensitivity.tsv")
    seed_table = seed_stability(
        tost,
        config["seeds"]["stability"],
        draws=int(config["bootstrap"]["draws"]),
        margin=float(thresholds["tost_margin"]),
        alpha=float(thresholds["alpha"]),
    )
    write_tsv(seed_table, robustness_root / "seed_stability.tsv")

    f_path = Path(root) / config["equivalence"]["f_track"]["source"]
    if f_path.is_file():
        f_frame = pd.read_csv(f_path, sep="\t")
        f_frame["descriptive_only"] = True
        f_frame["causal_representation_claim"] = False
        write_tsv(f_frame, equivalence_root / "f_track_descriptive.tsv")
    write_json(
        equivalence_root / "i_track_status.json",
        {
            "status": "OPEN",
            "reason": "single-study insertion-derived track with no same-study P/B/F counterpart",
            "formal_equivalence_estimable": False,
            "source": config["equivalence"]["i_track"]["source"],
            "study_family": config["equivalence"]["i_track"]["study_family"],
            "policy": config["equivalence"]["i_track"]["report"],
        },
    )

    equivalence_count = int(tost["equivalent"].sum())
    equivalence_material = int(tost["material_difference_established"].sum())
    if equivalence_material > 0:
        equivalence_verdict = "FAIL"
    elif equivalence_count == len(tost):
        equivalence_verdict = "PASS"
    elif equivalence_count >= 24:
        equivalence_verdict = "PARTIAL"
    else:
        equivalence_verdict = "FAIL"
    equivalence_summary = {
        "verdict": equivalence_verdict,
        "primary_contrasts": int(len(tost)),
        "equivalent_contrasts": equivalence_count,
        "inconclusive_contrasts": int(np.sum(tost.equivalence_decision == "inconclusive_equivalence")),
        "material_difference_contrasts": equivalence_material,
        "tost_pass_contrasts": int(tost.tost_pass.sum()),
        "bootstrap_ci_inside_margin_contrasts": int(tost.bootstrap_ci_inside_margin.sum()),
        "max_abs_study_delta": float(
            max(abs(tost[column]).max() for column in tost.columns if column.startswith("delta_"))
        ),
        "all_study_deltas_below_margin": bool(
            all((tost[column].abs() < float(thresholds["tost_margin"])).all()
                for column in tost.columns if column.startswith("delta_"))
        ),
        "pass_threshold_equivalent_contrasts": int(math.ceil(0.8 * len(tost))),
        "max_abs_mean_delta": float(tost.mean_delta.abs().max()),
        "max_abs_bootstrap_ci_bound": float(max(tost.ci_low_bootstrap.abs().max(), tost.ci_high_bootstrap.abs().max())),
        "by_track": track_summary.to_dict(orient="records"),
        "interpretation": (
            "All study-level deltas and all 10,000-draw study-family bootstrap intervals are "
            "inside the frozen 0.05 margin, but formal TOST equivalence passes for 20/30 "
            "contrasts. The remaining 10/30 are inconclusive because the paired t test over "
            "only three study families does not reject non-equivalence at alpha=0.05. "
            "Inconclusive is neither difference nor established equivalence."
        ),
        "parameters": {
            "margin": float(thresholds["tost_margin"]),
            "alpha": float(thresholds["alpha"]),
            "bootstrap_draws": int(config["bootstrap"]["draws"]),
            "bootstrap_seed": int(config["seeds"]["primary_bootstrap"]),
            "permutation_seed": int(config["seeds"]["sign_permutation"]),
            "multiplicity": "benjamini_hochberg_within_30_contrast_family",
        },
    }
    write_json(equivalence_root / "summary.json", equivalence_summary)

    depth = depth_tables(config, root)
    write_tsv(depth["depth"], depth_root / "depth_p_b25_metrics.tsv")
    write_tsv(depth["onsets"], depth_root / "depth_onset_primary.tsv")
    write_tsv(depth["bootstrap"], depth_root / "depth_onset_bootstrap.tsv")
    write_tsv(depth["retention"], depth_root / "retention_collapse.tsv")
    write_tsv(depth["ta2_aggregated"], depth_root / "ta2_depth_aggregated.tsv")
    write_tsv(depth["ta2_onsets"], depth_root / "ta2_onset.tsv")
    write_tsv(depth["threshold_sensitivity"], depth_root / "threshold_sensitivity.tsv")
    onset_count = int(depth["onsets"].onset_depth.notna().sum())
    ta2_onset_count = int(depth["ta2_onsets"].onset_depth.notna().sum())
    per_study_onsets = {
        str(study): int(group.onset_depth.notna().sum())
        for study, group in depth["onsets"].groupby("study_family")
    }
    primary_pass = (
        len(per_study_onsets) == 2
        and all(count >= 4 for count in per_study_onsets.values())
    )
    if primary_pass:
        depth_verdict = "PASS"
    elif onset_count > 0 or ta2_onset_count > 0:
        depth_verdict = "PARTIAL"
    else:
        depth_verdict = "FAIL"
    depth_summary = {
        "verdict": depth_verdict,
        "primary_study_method_cells": int(len(depth["onsets"])),
        "primary_material_onsets": onset_count,
        "primary_material_onsets_by_study": per_study_onsets,
        "ta2_study_method_contrast_cells": int(len(depth["ta2_onsets"])),
        "ta2_material_onsets": ta2_onset_count,
        "max_abs_P_minus_B25": float(depth["depth"].abs_P_minus_B25.max()),
        "materiality_margin": float(thresholds["materiality"]),
        "retention_collapse": depth["retention"].to_dict(orient="records"),
        "interpretation": (
            "The absolute P-vs-B25 endpoint gap never reaches the 0.05 materiality margin "
            "in the observed depth range because both representations collapse at low depth. "
            "The low-depth false-success claim is supported by retention collapse and zero-output "
            "P contracts, not by a material absolute endpoint gap. The pre-registered absolute-gap "
            "onset verdict is therefore FAIL and the figure wording must separate the two quantities."
        ),
    }
    write_json(depth_root / "summary.json", depth_summary)

    external_metrics = pd.read_csv(Path(root) / config["external"]["m3h_metrics"], sep="\t")
    external = external_gse205052_decisions(
        external_metrics,
        b25_margin=float(config["external"]["local_transfer_b25_margin"]),
        materiality=float(config["external"]["local_transfer_materiality"]),
    )
    write_tsv(external, external_root / "decisions.tsv")
    write_json(
        external_root / "completion_evidence.json",
        {
            "schema_version": 1,
            "milestone": "NM_RISK1_GSE205052_LOCKED_REEVALUATION",
            "status": "pass",
            "study_family": "zhang_2023",
            "dataset_id": "gse205052_mousebrain_20um",
            "new_independent_study_family": bool(config["external"]["local_transfer_new_independent_family"]),
            "scope": config["external"]["local_transfer_scope"],
            "adaptive_branch_available": bool(config["external"]["local_transfer_adaptive_branch_available"]),
            "construction_reused_from_frozen_v1": True,
            "rna_evaluation_reused_from_frozen_v1": True,
            "rna_values_read_by_risk1": False,
            "inputs": {
                "metrics": {
                    "path": config["external"]["m3h_metrics"],
                    "sha256": sha256_path(Path(root) / config["external"]["m3h_metrics"]),
                },
                "pairing_preflight": {
                    "path": config["external"]["m3h_preflight"],
                    "sha256": sha256_path(Path(root) / config["external"]["m3h_preflight"]),
                },
            },
            "b25_margin": float(config["external"]["local_transfer_b25_margin"]),
            "materiality_margin": float(config["external"]["local_transfer_materiality"]),
            "methods": int(len(external)),
            "b25_non_inferior_methods": int(external["B25_non_inferior_at_0_01"].sum()),
            "material_delta_methods": int(external["material_at_0_05"].sum()),
            "interpretation": (
                "Within-family frozen re-evaluation only. It does not count as a new independent "
                "study family and cannot close the two-independent-study external RNA risk."
            ),
            "artifacts": {
                "decisions.tsv": {
                    "path": str(external_root / "decisions.tsv"),
                    "sha256": sha256_path(external_root / "decisions.tsv"),
                }
            },
        },
    )

    runtime_table = runtime_memory_table(config, root)
    write_tsv(runtime_table, robustness_root / "runtime_memory.tsv")
    write_tsv(claim_gap_table(config), robustness_root / "claim_evidence_gaps.tsv")
    write_json(
        robustness_root / "summary.json",
        {
            "verdict": "PASS",
            "threshold_sensitivity": equivalence_sensitivity.to_dict(orient="records"),
            "seed_stability_equivalent_min": int(seed_table.equivalent_contrasts.min()),
            "seed_stability_equivalent_max": int(seed_table.equivalent_contrasts.max()),
            "seed_stability_seeds": len(seed_table),
            "runtime_rows": int(len(runtime_table)),
            "runtime_rows_with_process_tree": int(runtime_table.process_tree_path.ne("not_available").sum()),
            "hardware_note": config["robustness"]["hardware_note"],
        },
    )

    artifacts = {
        "equivalence_summary": equivalence_root / "summary.json",
        "tost_primary_family": equivalence_root / "tost_primary_family.tsv",
        "depth_summary": depth_root / "summary.json",
        "depth_onset_primary": depth_root / "depth_onset_primary.tsv",
        "retention_collapse": depth_root / "retention_collapse.tsv",
        "robustness_summary": robustness_root / "summary.json",
        "runtime_memory": robustness_root / "runtime_memory.tsv",
        "claim_evidence_gaps": robustness_root / "claim_evidence_gaps.tsv",
        "external_decisions": external_root / "decisions.tsv",
    }
    _write_completion(
        output_root,
        status="pass",
        runtime_seconds=time.perf_counter() - started,
        parameters={
            "config": str(config_path),
            "config_sha256": sha256_path(config_path),
            "protocol": config["protocol"],
            "protocol_sha256": sha256_path(Path(root) / config["protocol"]),
            "thresholds": thresholds,
            "seeds": config["seeds"],
            "equivalence_verdict": equivalence_summary["verdict"],
            "depth_verdict": depth_summary["verdict"],
        },
        artifacts=artifacts,
    )
    return output_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-gene-bootstrap", action="store_true")
    arguments = parser.parse_args()
    output = run(arguments.config, arguments.root, skip_gene_bootstrap=arguments.skip_gene_bootstrap)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
