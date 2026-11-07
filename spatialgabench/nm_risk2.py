"""NM-RISK2 evaluation-method sensitivity on frozen paired-RNA artifacts.

The module is read-only with respect to ``results/v1`` and ``registries/v1``.
It writes only under the configured ``results/v1_risk2`` tree.  RNA is not
opened here; all analyses consume already-frozen gene-level correlation tables
and frozen endpoint summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
import traceback
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from spatialgabench.nm_risk1 import ENDPOINT, build_tost_table, material_onset


PRIMARY_ENDPOINT = "coverage_adjusted_median_gene_wise_spearman"
ENDPOINT_IDS = (
    "primary_frozen",
    "median_all_zero",
    "median_variable",
    "mean_variable",
    "median_variable_fisher",
    "median_detected_top50",
    "median_detected_bottom50",
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_tsv(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, sep="\t", index=False, na_rep="NA")


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def endpoint_values(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    detected_col: str = "detected_spot_fraction",
) -> pd.DataFrame:
    """Compute the pre-registered endpoint suite for every group.

    ``primary_frozen`` is the frozen endpoint: median over the RNA-eligible
    gene universe with undefined/absent correlations set to zero, multiplied
    by the variable-gene fraction.  ``median_variable`` and the Fisher/mean
    variants use only variable genes with finite correlations.
    """

    present_col = "activity_gene_present" if "activity_gene_present" in frame.columns else "ga_gene_present"
    variable_col = "activity_gene_variable" if "activity_gene_variable" in frame.columns else "ga_gene_variable"
    required = {"spearman", present_col, variable_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"gene table is missing columns: {sorted(missing)}")
    if not set(group_cols).issubset(frame.columns):
        raise ValueError(f"gene table is missing group columns: {sorted(set(group_cols) - set(frame.columns))}")

    work = frame.copy()
    work["_rho"] = pd.to_numeric(work["spearman"], errors="coerce")
    work["_present"] = _bool_series(work[present_col])
    work["_variable"] = work["_present"] & _bool_series(work[variable_col])
    work["_rho_zero"] = np.where(np.isfinite(work["_rho"].to_numpy(dtype=float)), work["_rho"].to_numpy(dtype=float), 0.0)

    has_detected = detected_col in work.columns
    if has_detected:
        work["_detected"] = pd.to_numeric(work[detected_col], errors="coerce")

    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(list(group_cols), dropna=False, sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        rho = group["_rho"].to_numpy(dtype=float)
        variable = group["_variable"].to_numpy(dtype=bool)
        rho_zero = group["_rho_zero"].to_numpy(dtype=float)
        valid = variable & np.isfinite(rho)
        coverage = float(np.mean(variable)) if len(variable) else math.nan
        median_all_zero = float(np.median(rho_zero)) if len(rho_zero) else math.nan
        record: dict[str, Any] = {column: value for column, value in zip(group_cols, keys)}
        record.update(
            {
                "n_genes": int(len(group)),
                "n_variable": int(np.sum(variable)),
                "coverage": coverage,
                "median_all_zero": median_all_zero,
                "primary_frozen": median_all_zero * coverage,
                "median_variable": float(np.median(rho[valid])) if valid.any() else math.nan,
                "mean_variable": float(np.mean(rho[valid])) if valid.any() else math.nan,
                "median_variable_fisher": float(
                    np.tanh(np.mean(np.arctanh(np.clip(rho[valid], -0.999999, 0.999999))))
                )
                if valid.any()
                else math.nan,
            }
        )
        if has_detected:
            detected = group["_detected"].to_numpy(dtype=float)
            finite_detected = np.isfinite(detected)
            if finite_detected.any():
                cut = float(np.nanmedian(detected[finite_detected]))
                top = finite_detected & (detected >= cut)
                bottom = finite_detected & (detected < cut)
                top_finite = top & np.isfinite(rho)
                bottom_finite = bottom & np.isfinite(rho)
                record["median_detected_top50"] = float(np.median(rho[top_finite])) if top_finite.any() else math.nan
                record["median_detected_bottom50"] = (
                    float(np.median(rho[bottom_finite])) if bottom_finite.any() else math.nan
                )
            else:
                record["median_detected_top50"] = math.nan
                record["median_detected_bottom50"] = math.nan
        rows.append(record)
    result = pd.DataFrame(rows)
    for endpoint in ENDPOINT_IDS:
        if endpoint not in result.columns:
            result[endpoint] = math.nan
    return result


def load_standard_gene_table(config: Mapping[str, Any], root: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "method_id",
        "representation_id",
        "spearman",
        "activity_gene_present",
        "activity_gene_variable",
        "ga_gene_present",
        "ga_gene_variable",
        "detected_spot_fraction",
    ]
    for study, spec in config["standard_depth"]["studies"].items():
        path = Path(root) / spec["gene_table"]
        frame = pd.read_csv(path, sep="\t", usecols=[column for column in columns if column in pd.read_csv(path, sep="\t", nrows=0).columns])
        frame["study_family"] = study
        frame["dataset_id"] = spec["dataset_id"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def standard_endpoints(config: Mapping[str, Any], root: str | Path) -> pd.DataFrame:
    frame = load_standard_gene_table(config, root)
    methods = list(config["methods"])
    representations = list(config["representations"].values())
    frame = frame.loc[frame.method_id.isin(methods) & frame.representation_id.isin(representations)].copy()
    return endpoint_values(frame, ["study_family", "dataset_id", "method_id", "representation_id"])


def reproduction_table(endpoints: pd.DataFrame, config: Mapping[str, Any], root: str | Path) -> pd.DataFrame:
    metrics = pd.read_csv(Path(root) / config["standard_depth"]["common_metrics"], sep="\t")
    metrics = metrics.loc[
        metrics.method_category.eq("hand_written_annotation")
        & metrics.method_id.isin(config["methods"])
        & metrics.representation_id.isin(list(config["representations"].values()))
    ].copy()
    keep = [
        "study_family",
        "dataset_id",
        "method_id",
        "representation_id",
        PRIMARY_ENDPOINT,
        "median_gene_wise_spearman",
        "evaluable_gene_fraction",
    ]
    merged = endpoints.merge(metrics[keep], on=["study_family", "dataset_id", "method_id", "representation_id"], how="inner")
    merged["primary_abs_diff"] = (merged["primary_frozen"] - merged[PRIMARY_ENDPOINT]).abs()
    merged["median_abs_diff"] = (merged["median_all_zero"] - merged["median_gene_wise_spearman"]).abs()
    merged["coverage_abs_diff"] = (merged["coverage"] - merged["evaluable_gene_fraction"]).abs()
    return merged.sort_values(["study_family", "method_id", "representation_id"]).reset_index(drop=True)


def tost_for_endpoint(
    endpoint_frame: pd.DataFrame,
    endpoint_col: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    methods = list(config["methods"])
    representations = list(config["representations"].values())
    studies = list(config["standard_depth"]["studies"])
    data = endpoint_frame[["study_family", "method_id", "representation_id", endpoint_col]].dropna().copy()
    complete_methods: list[str] = []
    for method in methods:
        subset = data.loc[data.method_id.eq(method)]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index="study_family", columns="representation_id", values=endpoint_col, aggfunc="first")
        if all(study in pivot.index for study in studies) and all(rep in pivot.columns for rep in representations):
            block = pivot.loc[studies, representations]
            if bool(block.notna().all().all()):
                complete_methods.append(method)
    metadata = {
        "endpoint": endpoint_col,
        "estimable_methods": len(complete_methods),
        "methods": complete_methods,
        "missing_methods": [method for method in methods if method not in complete_methods],
        "contrasts": 6 * len(complete_methods),
    }
    if not complete_methods:
        return None, metadata
    metrics = data.loc[data.method_id.isin(complete_methods)].rename(columns={endpoint_col: ENDPOINT})
    table = build_tost_table(
        metrics,
        methods=complete_methods,
        representations=representations,
        studies=studies,
        margin=float(config["thresholds"]["tost_margin"]),
        alpha=float(config["thresholds"]["alpha"]),
        draws=int(config["bootstrap"]["draws"]),
        bootstrap_seed=int(config["seeds"]["primary_bootstrap"]),
        permutation_seed=int(config["seeds"]["permutation"]),
        permutation_draws=int(config["bootstrap"]["permutation_draws"]),
    )
    return table, metadata


def compare_decisions(primary: pd.DataFrame, alternative: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = primary.copy()
    right = alternative.copy()
    merged = left.merge(right, on="contrast_id", suffixes=("_primary", "_alt"), how="inner")
    merged["decision_agreement"] = merged["equivalence_decision_primary"].eq(merged["equivalence_decision_alt"])
    merged["direction_agreement"] = np.sign(merged["mean_delta_primary"]).eq(np.sign(merged["mean_delta_alt"]))
    primary_inconclusive = set(left.loc[left.equivalence_decision.eq("inconclusive_equivalence"), "contrast_id"])
    alt_inconclusive = set(right.loc[right.equivalence_decision.eq("inconclusive_equivalence"), "contrast_id"])
    union = primary_inconclusive | alt_inconclusive
    jaccard = len(primary_inconclusive & alt_inconclusive) / len(union) if union else 1.0
    equivalent = int(right["equivalent"].sum())
    summary = {
        "contrasts_compared": int(len(merged)),
        "primary_contrasts": int(len(left)),
        "alternative_contrasts": int(len(right)),
        "decision_agreement": int(merged["decision_agreement"].sum()),
        "direction_agreement": int(merged["direction_agreement"].sum()),
        "equivalent_count": equivalent,
        "inconclusive_count": int((right["equivalence_decision"] == "inconclusive_equivalence").sum()),
        "material_count": int(right["material_difference_established"].sum()),
        "primary_inconclusive_jaccard": float(jaccard),
        "implied_risk1_protocol_verdict": "PASS" if equivalent == 30 else ("PARTIAL" if equivalent >= 24 else "FAIL"),
        "max_abs_mean_delta": float(right["mean_delta"].abs().max()) if len(right) else math.nan,
    }
    return merged, summary


def leave_one_study_out(endpoint_frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Descriptive sensitivity of each mean contrast to dropping one study."""

    methods = list(config["methods"])
    representations = list(config["representations"].values())
    studies = list(config["standard_depth"]["studies"])
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_IDS:
        if endpoint not in endpoint_frame.columns:
            continue
        data = endpoint_frame[["study_family", "method_id", "representation_id", endpoint]].dropna().copy()
        pivot = data.pivot_table(
            index=["study_family", "method_id"],
            columns="representation_id",
            values=endpoint,
            aggfunc="first",
        )
        for method in methods:
            for left, right in combinations(representations, 2):
                deltas = {}
                for study in studies:
                    key = (study, method)
                    if key in pivot.index and left in pivot.columns and right in pivot.columns:
                        value_left = pivot.loc[key, left]
                        value_right = pivot.loc[key, right]
                        if pd.notna(value_left) and pd.notna(value_right):
                            deltas[study] = float(value_left - value_right)
                if not deltas:
                    continue
                record: dict[str, Any] = {
                    "endpoint": endpoint,
                    "method_id": method,
                    "representation_left": left,
                    "representation_right": right,
                    "studies_available": int(len(deltas)),
                    "full_mean_delta": float(np.mean(list(deltas.values()))),
                }
                loo_values: list[float] = []
                for study in studies:
                    retained = {name: value for name, value in deltas.items() if name != study}
                    record[f"drop_{study}_mean_delta"] = (
                        float(np.mean(list(retained.values()))) if retained else math.nan
                    )
                    if retained:
                        loo_values.append(float(np.mean(list(retained.values()))))
                record["loo_min_mean_delta"] = min(loo_values) if loo_values else math.nan
                record["loo_max_mean_delta"] = max(loo_values) if loo_values else math.nan
                record["loo_range"] = (
                    record["loo_max_mean_delta"] - record["loo_min_mean_delta"] if loo_values else math.nan
                )
                record["loo_sign_stable"] = bool(
                    len({bool(np.sign(value) >= 0) for value in loo_values}) <= 1
                ) if loo_values else False
                rows.append(record)
    return pd.DataFrame(rows)


def e1_verdict(summaries: Mapping[str, Mapping[str, Any]], reproduction_ok: bool) -> dict[str, Any]:
    alternative_summaries = {name: value for name, value in summaries.items() if name != "primary_frozen"}
    material = {name: value for name, value in alternative_summaries.items() if int(value["material_count"]) > 0}
    crosses = {
        name: value
        for name, value in alternative_summaries.items()
        if int(value["equivalent_count"]) >= 24
    }
    low_direction = {
        name: value
        for name, value in alternative_summaries.items()
        if int(value["direction_agreement"]) < 27
    }
    if not reproduction_ok or material:
        verdict = "FAIL"
        reason = "primary reproduction failed" if not reproduction_ok else "an alternative endpoint established a material difference"
    elif crosses or low_direction:
        verdict = "PARTIAL"
        reason = "the formal verdict is endpoint-dependent: at least one alternative crosses 24/30 equivalent or has direction agreement <27/30"
    else:
        verdict = "PASS"
        reason = "every estimable alternative remains below 24/30 equivalent, has no material difference, and agrees in direction for at least 27/30 contrasts"
    return {
        "verdict": verdict,
        "reason": reason,
        "material_endpoints": sorted(material),
        "crosses_24_endpoints": sorted(crosses),
        "low_direction_endpoints": sorted(low_direction),
    }


def load_depth_gene_table(config: Mapping[str, Any], root: str | Path) -> pd.DataFrame:
    path = Path(root) / config["depth"]["gene_table"]
    columns = [
        "dataset_id",
        "depth_fraction",
        "seed",
        "method_id",
        "representation_id",
        "spearman",
        "activity_gene_present",
        "activity_gene_variable",
    ]
    frame = pd.read_csv(path, sep="\t", usecols=columns)
    frame["study_family"] = frame["dataset_id"].map(config["depth"]["datasets"])
    frame = frame.loc[
        frame.study_family.isin(config["depth"]["studies"])
        & frame.method_id.isin(config["methods"])
        & frame.representation_id.isin(
            [config["depth"]["p_representation"], config["depth"]["b25_representation"]]
        )
        & frame.depth_fraction.isin(config["depth"]["depths"])
    ].copy()
    return frame


def depth_tables(config: Mapping[str, Any], root: str | Path) -> dict[str, pd.DataFrame]:
    frame = load_depth_gene_table(config, root)
    group_cols = ["study_family", "method_id", "representation_id", "depth_fraction", "seed"]
    per_seed = endpoint_values(frame, group_cols)
    endpoint_cols = [endpoint for endpoint in ENDPOINT_IDS if endpoint in per_seed.columns]
    gene_agg = (
        per_seed.groupby(["study_family", "method_id", "representation_id", "depth_fraction"], as_index=False)
        .agg(
            {endpoint: "median" for endpoint in endpoint_cols}
            | {"coverage": "median", "n_variable": "median", "n_genes": "median"}
        )
    )

    contract = pd.read_csv(Path(root) / config["depth"]["contract_metrics"], sep="\t")
    contract = contract.loc[
        contract.study_family.isin(config["depth"]["studies"])
        & contract.method_id.isin(config["methods"])
        & contract.representation_id.isin(
            [config["depth"]["p_representation"], config["depth"]["b25_representation"]]
        )
        & contract.depth_fraction.isin(config["depth"]["depths"])
    ].copy()
    contract_agg = (
        contract.groupby(["study_family", "method_id", "representation_id", "depth_fraction"], as_index=False)
        .agg(
            coverage_contract=("evaluable_gene_fraction", "median"),
            primary_contract=(PRIMARY_ENDPOINT, "median"),
            median_all_zero_contract=("median_gene_wise_spearman", "median"),
            n_planned=("evaluation_status", "size"),
            n_success=("evaluation_status", lambda values: int((values == "pass").sum())),
        )
    )
    merged = gene_agg.merge(
        contract_agg,
        on=["study_family", "method_id", "representation_id", "depth_fraction"],
        how="outer",
    )
    merged["primary_abs_diff_vs_contract"] = (merged["primary_frozen"] - merged["primary_contract"]).abs()
    merged["median_abs_diff_vs_contract"] = (merged["median_all_zero"] - merged["median_all_zero_contract"]).abs()

    full = merged.loc[merged.depth_fraction.eq(1.0)].set_index(
        ["study_family", "method_id", "representation_id"]
    )
    retention_rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        key = (row.study_family, row.method_id, row.representation_id)
        if key not in full.index:
            continue
        base = full.loc[key]
        record = {
            "study_family": row.study_family,
            "method_id": row.method_id,
            "representation_id": row.representation_id,
            "depth_fraction": float(row.depth_fraction),
            "n_planned": int(row.n_planned) if pd.notna(row.n_planned) else math.nan,
            "n_success": int(row.n_success) if pd.notna(row.n_success) else math.nan,
        }
        for endpoint in endpoint_cols:
            current = getattr(row, endpoint)
            baseline = getattr(base, endpoint)
            record[endpoint] = current
            record[f"{endpoint}_full"] = baseline
            record[f"{endpoint}_retention"] = (
                float(current / baseline)
                if pd.notna(current) and pd.notna(baseline) and float(baseline) > 0
                else math.nan
            )
        retention_rows.append(record)
    retention = pd.DataFrame(retention_rows)

    gap_rows: list[dict[str, Any]] = []
    for (study, method, depth), group in merged.groupby(["study_family", "method_id", "depth_fraction"]):
        p_rows = group.loc[group.representation_id.eq(config["depth"]["p_representation"])]
        b_rows = group.loc[group.representation_id.eq(config["depth"]["b25_representation"])]
        if p_rows.empty or b_rows.empty:
            continue
        p_row = p_rows.iloc[0]
        b_row = b_rows.iloc[0]
        record: dict[str, Any] = {
            "study_family": study,
            "method_id": method,
            "depth_fraction": float(depth),
            "P_n_success": int(p_row["n_success"]) if pd.notna(p_row["n_success"]) else math.nan,
            "B25_n_success": int(b_row["n_success"]) if pd.notna(b_row["n_success"]) else math.nan,
            "P_n_planned": int(p_row["n_planned"]) if pd.notna(p_row["n_planned"]) else math.nan,
            "B25_n_planned": int(b_row["n_planned"]) if pd.notna(b_row["n_planned"]) else math.nan,
        }
        for endpoint in endpoint_cols:
            p_value = getattr(p_row, endpoint)
            b_value = getattr(b_row, endpoint)
            record[f"P_{endpoint}"] = p_value
            record[f"B25_{endpoint}"] = b_value
            record[f"gap_{endpoint}"] = (
                float(b_value - p_value)
                if pd.notna(p_value) and pd.notna(b_value)
                else math.nan
            )
        gap_rows.append(record)
    gaps = pd.DataFrame(gap_rows)

    onset_rows: list[dict[str, Any]] = []
    depth_order = [float(value) for value in config["depth"]["depths"]]
    for (study, method), group in gaps.groupby(["study_family", "method_id"]):
        for endpoint in endpoint_cols:
            column = f"gap_{endpoint}"
            deltas = dict(zip(group["depth_fraction"].astype(float), group[column].abs().astype(float)))
            finite_gaps = [abs(value) for value in deltas.values() if np.isfinite(value)]
            onset = (
                material_onset(depth_order, deltas, float(config["thresholds"]["materiality"]))
                if finite_gaps
                else None
            )
            onset_rows.append(
                {
                    "study_family": study,
                    "method_id": method,
                    "endpoint": endpoint,
                    "onset_depth": np.nan if onset is None else onset,
                    "onset_label": (
                        "not_estimable"
                        if not finite_gaps
                        else ("not_material_in_observed_range" if onset is None else f"{onset:.2f}")
                    ),
                    "max_abs_gap": float(max(finite_gaps)) if finite_gaps else math.nan,
                }
            )
    onsets = pd.DataFrame(onset_rows)

    decomposition_rows: list[dict[str, Any]] = []
    for row in merged.loc[merged.representation_id.eq(config["depth"]["p_representation"])].itertuples(index=False):
        if float(row.depth_fraction) not in {0.05, 0.01}:
            continue
        key = (row.study_family, row.method_id, config["depth"]["p_representation"])
        if key not in full.index:
            continue
        base = full.loc[key]
        coverage_full = getattr(base, "coverage")
        coverage_retention = (
            float(row.coverage / coverage_full)
            if pd.notna(row.coverage) and pd.notna(coverage_full) and float(coverage_full) > 0
            else math.nan
        )
        median_variable = getattr(row, "median_variable")
        if pd.isna(coverage_retention) or pd.isna(median_variable):
            category = "not_estimable"
        elif coverage_retention < float(config["thresholds"]["coverage_retention_boundary"]) and median_variable > 0:
            category = "coverage_limited"
        elif coverage_retention >= float(config["thresholds"]["coverage_retention_boundary"]) and median_variable <= float(config["thresholds"]["signal_margin"]):
            category = "quality_limited"
        elif coverage_retention < float(config["thresholds"]["coverage_retention_boundary"]) and median_variable <= float(config["thresholds"]["signal_margin"]):
            category = "both"
        else:
            category = "neither"
        decomposition_rows.append(
            {
                "study_family": row.study_family,
                "method_id": row.method_id,
                "depth_fraction": float(row.depth_fraction),
                "coverage": float(row.coverage) if pd.notna(row.coverage) else math.nan,
                "coverage_full": float(coverage_full) if pd.notna(coverage_full) else math.nan,
                "coverage_retention": coverage_retention,
                "median_variable": float(median_variable) if pd.notna(median_variable) else math.nan,
                "category": category,
            }
        )
    decomposition = pd.DataFrame(decomposition_rows)

    zero_rows: list[dict[str, Any]] = []
    for (study, depth), group in merged.loc[merged.representation_id.eq(config["depth"]["p_representation"])].groupby(
        ["study_family", "depth_fraction"]
    ):
        zero_rows.append(
            {
                "study_family": study,
                "depth_fraction": float(depth),
                "n_methods": int(len(group)),
                "P_zero_primary_methods": int((group["primary_frozen"] == 0).sum()),
                "P_zero_median_variable_methods": int(
                    (group["median_variable"].isna() | (group["median_variable"] == 0)).sum()
                ),
            }
        )
    zero_counts = pd.DataFrame(zero_rows)
    return {
        "endpoint_cells": merged,
        "retention": retention,
        "decomposition": decomposition,
        "gaps": gaps,
        "onsets": onsets,
        "zero_counts": zero_counts,
    }


def e2_verdict(decomposition: pd.DataFrame, gaps: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    studies = list(config["depth"]["studies"])
    details: dict[str, Any] = {}
    pass_studies = 0
    for study in studies:
        study_rows = decomposition.loc[decomposition.study_family.eq(study)]
        study_pass = True
        for depth in (0.05, 0.01):
            depth_rows = study_rows.loc[study_rows.depth_fraction.eq(depth)]
            coverage_like = int(depth_rows["category"].isin(["coverage_limited", "both"]).sum())
            details[f"{study}_d{depth}_coverage_like"] = coverage_like
            details[f"{study}_d{depth}_n"] = int(len(depth_rows))
            if coverage_like < 4:
                study_pass = False
        at5 = gaps.loc[gaps.study_family.eq(study) & gaps.depth_fraction.eq(0.05)]
        gap_ok = int((at5["gap_median_variable"] >= float(config["thresholds"]["signal_margin"])).sum())
        details[f"{study}_d0.05_variable_gap_ge_0.01"] = gap_ok
        details[f"{study}_d0.05_n"] = int(len(at5))
        if gap_ok < 4:
            study_pass = False
        if study_pass:
            pass_studies += 1
    fail_condition = False
    for study in studies:
        at5 = gaps.loc[gaps.study_family.eq(study) & gaps.depth_fraction.eq(0.05)]
        if bool((at5["P_median_variable"] >= at5["B25_median_variable"] + float(config["thresholds"]["signal_margin"])).any()):
            fail_condition = True
    if fail_condition:
        verdict = "FAIL"
        reason = "at least one P method has variable-gene signal at least 0.01 above B25 at 5%; the RISK1 'no per-gene signal' wording is not supported"
    elif pass_studies == 2:
        verdict = "PASS"
        reason = "both studies show coverage-limited or both-limited P at 5% and 1% for at least 4/5 methods, and B25 minus P variable-gene gap >=0.01 at 5% for at least 4/5 methods"
    elif pass_studies == 1:
        verdict = "PARTIAL"
        reason = "the coverage-limited pattern holds in only one study"
    else:
        verdict = "FAIL"
        reason = "the coverage-limited pattern does not hold in either study"
    return {"verdict": verdict, "reason": reason, "details": details}


def human_cross_platform(config: Mapping[str, Any], root: str | Path, standard: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, Any]]:
    human_path = Path(root) / config["human"]["gene_table"]
    human = pd.read_csv(human_path, sep="\t")
    human = human.loc[
        human.method_id.isin(config["methods"])
        & human.representation_id.isin(list(config["representations"].values()))
    ].copy()
    human_endpoints = endpoint_values(human, ["method_id", "representation_id"])
    mouse = standard.loc[standard.study_family.eq(config["cross_platform"]["mouse_study_family"])].copy()
    human_endpoints["study_family"] = config["human"]["study_family"]
    human_endpoints["dataset_id"] = config["human"]["dataset_id"]
    human_endpoints["species"] = "human"
    human_endpoints["within_family"] = True
    human_endpoints["independent_replication"] = False
    mouse["study_family"] = config["cross_platform"]["mouse_study_family"]
    mouse["dataset_id"] = config["cross_platform"]["mouse_dataset_id"]
    mouse["species"] = "mouse"
    mouse["within_family"] = True
    mouse["independent_replication"] = False

    representations = list(config["representations"].values())
    delta_rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_IDS:
        for method in config["methods"]:
            for left, right in combinations(representations, 2):
                h_left = human_endpoints.loc[
                    human_endpoints.method_id.eq(method) & human_endpoints.representation_id.eq(left), endpoint
                ]
                h_right = human_endpoints.loc[
                    human_endpoints.method_id.eq(method) & human_endpoints.representation_id.eq(right), endpoint
                ]
                m_left = mouse.loc[mouse.method_id.eq(method) & mouse.representation_id.eq(left), endpoint]
                m_right = mouse.loc[mouse.method_id.eq(method) & mouse.representation_id.eq(right), endpoint]
                if h_left.empty or h_right.empty or m_left.empty or m_right.empty:
                    continue
                h_delta = float(h_left.iloc[0] - h_right.iloc[0])
                m_delta = float(m_left.iloc[0] - m_right.iloc[0])
                delta_rows.append(
                    {
                        "endpoint": endpoint,
                        "method_id": method,
                        "representation_left": left,
                        "representation_right": right,
                        "human_delta": h_delta,
                        "mouse_delta": m_delta,
                        "same_sign": bool(np.sign(h_delta) == np.sign(m_delta)),
                        "abs_difference": abs(h_delta - m_delta),
                        "within_family": True,
                        "independent_replication": False,
                    }
                )
    deltas = pd.DataFrame(delta_rows)
    primary = deltas.loc[deltas.endpoint.eq("primary_frozen")]
    same_sign = int(primary["same_sign"].sum()) if len(primary) else 0
    by_method = (
        primary.groupby("method_id")["same_sign"].agg(["sum", "size"]).to_dict(orient="index")
        if len(primary)
        else {}
    )
    same_sign_by_method = {method: int(values["sum"]) for method, values in by_method.items()}
    pairs_per_method = {method: int(values["size"]) for method, values in by_method.items()}
    all_methods_at_least_4_of_6 = bool(
        len(same_sign_by_method) == len(config["methods"])
        and all(
            same_sign_by_method.get(method, 0) >= 4 and pairs_per_method.get(method, 0) == 6
            for method in config["methods"]
        )
    )
    if not len(primary):
        verdict = "INFEASIBLE"
        reason = "human or mouse primary endpoint table is not estimable"
    elif all_methods_at_least_4_of_6:
        verdict = "PASS"
        minimum = min(same_sign_by_method.values())
        reason = (
            "every hand-written method has at least 4/6 primary representation-pair deltas "
            f"with the same sign across human and mouse (minimum {minimum}/6; overall {same_sign}/{len(primary)})"
        )
    else:
        verdict = "PARTIAL"
        reason = (
            "at least one hand-written method has fewer than 4/6 primary representation-pair deltas "
            f"with the same sign across human and mouse (per-method counts {same_sign_by_method}; "
            f"overall {same_sign}/{len(primary)})"
        )
    summary = {
        "verdict": verdict,
        "reason": reason,
        "same_sign_primary": same_sign,
        "pairs_primary": int(len(primary)),
        "pairs_per_method": 6,
        "same_sign_by_method": same_sign_by_method,
        "pairs_primary_by_method": pairs_per_method,
        "within_family": True,
        "independent_replication": False,
    }
    return {"human_endpoints": human_endpoints, "mouse_endpoints": mouse, "deltas": deltas, "summary": summary}


def _distribution(values: Sequence[float], config: Mapping[str, Any]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if len(array) == 0:
        return {
            "n": 0,
            "estimable": False,
            "status": "NOT_ESTIMABLE",
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "min": math.nan,
            "max": math.nan,
            "frac_zero": math.nan,
            "frac_one": math.nan,
            "ceiling_saturated": False,
            "floor_saturated": False,
            "zero_inflated": False,
        }
    q1, q3 = np.percentile(array, [25, 75])
    median = float(np.median(array))
    span = float(array.max() - array.min())
    return {
        "n": int(len(array)),
        "estimable": True,
        "status": "ESTIMABLE",
        "median": median,
        "q1": float(q1),
        "q3": float(q3),
        "min": float(array.min()),
        "max": float(array.max()),
        "frac_zero": float(np.mean(array == 0)),
        "frac_one": float(np.mean(array == 1)),
        "ceiling_saturated": bool(median >= float(config["ceiling"]["ceiling_threshold"]) and span < float(config["ceiling"]["saturation_range"])),
        "floor_saturated": bool(median <= float(config["ceiling"]["floor_threshold"]) and span < float(config["ceiling"]["saturation_range"])),
        "zero_inflated": bool(float(np.mean(array == 0)) >= float(config["ceiling"]["zero_inflation_threshold"])),
    }


def ceiling_audit(standard: pd.DataFrame, depth: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_IDS:
        for scope, frame, group_cols, depth_col in (
            ("standard_depth", standard, ["study_family", "representation_id"], None),
            ("depth", depth, ["study_family", "representation_id", "depth_fraction"], "depth_fraction"),
        ):
            if endpoint not in frame.columns:
                continue
            for key, group in frame.groupby(group_cols, dropna=False):
                keys = key if isinstance(key, tuple) else (key,)
                record = {
                    "scope": scope,
                    "endpoint": endpoint,
                    "study_family": keys[0],
                    "representation_id": keys[1],
                    "depth_fraction": float(keys[2]) if depth_col else 1.0,
                    "n_methods": int(len(group)),
                }
                record.update(_distribution(group[endpoint].tolist(), config))
                rows.append(record)
    return pd.DataFrame(rows)


def _software_versions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {"python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"}
    for package in ("numpy", "pandas", "scipy", "PyYAML", "anndata"):
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = "unknown"
    return versions


def _input_paths(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for study, spec in config["standard_depth"]["studies"].items():
        paths.append((f"standard_gene_table:{study}", spec["gene_table"]))
    paths.extend(
        [
            ("standard_common_metrics", config["standard_depth"]["common_metrics"]),
            ("risk1_decisions", config["standard_depth"]["risk1_decisions"]),
            ("depth_gene_table", config["depth"]["gene_table"]),
            ("depth_contract_metrics", config["depth"]["contract_metrics"]),
            ("depth_summary", config["depth"]["depth_summary"]),
            ("human_gene_table", config["human"]["gene_table"]),
            ("human_metrics", config["human"]["metrics"]),
        ]
    )
    return paths


def write_input_manifest(config: Mapping[str, Any], root: str | Path, output_root: Path) -> pd.DataFrame:
    rows = []
    for role, relative in _input_paths(config):
        path = Path(root) / relative
        rows.append(
            {
                "role": role,
                "path": relative,
                "sha256": sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
        )
    frame = pd.DataFrame(rows)
    target = output_root / "manifests" / "input_manifest.tsv"
    if target.exists():
        previous = pd.read_csv(target, sep="\t")
        keys = ["role", "path", "sha256", "size_bytes"]
        left = frame[keys].astype(str).sort_values(["role", "path"]).reset_index(drop=True)
        right = previous[keys].astype(str).sort_values(["role", "path"]).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError("frozen input manifest mismatch; refusing to continue")
    write_tsv(frame, target)
    return frame


def write_output_checksums(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"output_checksums.tsv"}:
            continue
        rows.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(frame, output_root / "manifests" / "output_checksums.tsv")
    return frame


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    started = time.perf_counter()
    root = Path(root).resolve()
    config = load_yaml(config_path)
    output_root = root / config["output_root"]
    for path in (
        output_root / "manifests",
        output_root / "endpoints",
        output_root / "equivalence",
        output_root / "depth",
        output_root / "cross_platform",
        output_root / "ceiling",
        output_root / "failures",
    ):
        path.mkdir(parents=True, exist_ok=True)

    protocol_path = root / config["protocol"]
    config_path = Path(config_path).resolve()
    input_manifest = write_input_manifest(config, root, output_root)

    standard = standard_endpoints(config, root)
    write_tsv(standard, output_root / "endpoints" / "standard_depth_endpoints.tsv")

    reproduction = reproduction_table(standard, config, root)
    write_tsv(reproduction, output_root / "endpoints" / "primary_reproduction.tsv")
    max_primary_diff = float(reproduction["primary_abs_diff"].max())
    max_median_diff = float(reproduction["median_abs_diff"].max())
    max_coverage_diff = float(reproduction["coverage_abs_diff"].max())
    tolerance = float(config["thresholds"]["reproduction_tolerance"])
    expected_reproduction_cells = (
        len(config["standard_depth"]["studies"])
        * len(config["methods"])
        * len(config["representations"])
    )
    reproduction_ok = bool(
        len(reproduction) == expected_reproduction_cells
        and max_primary_diff <= tolerance
        and max_median_diff <= tolerance
        and max_coverage_diff <= tolerance
    )

    risk1_decisions = pd.read_csv(Path(root) / config["standard_depth"]["risk1_decisions"], sep="\t")
    e1_tables: dict[str, pd.DataFrame] = {}
    e1_summaries: dict[str, dict[str, Any]] = {}
    e1_metadata: dict[str, Any] = {}
    for endpoint in ENDPOINT_IDS:
        table, metadata = tost_for_endpoint(standard, endpoint, config)
        e1_metadata[endpoint] = metadata
        if table is None:
            e1_summaries[endpoint] = {
                "estimable": False,
                "material_count": 0,
                "equivalent_count": 0,
                "direction_agreement": 0,
            }
            continue
        e1_tables[endpoint] = table
        write_tsv(table, output_root / "equivalence" / f"{endpoint}_tost.tsv")
        _, summary = compare_decisions(risk1_decisions, table)
        summary["estimable"] = True
        e1_summaries[endpoint] = summary

    e1 = e1_verdict(e1_summaries, reproduction_ok)
    loo = leave_one_study_out(standard, config)
    write_tsv(loo, output_root / "equivalence" / "leave_one_study_out.tsv")
    write_json(
        output_root / "equivalence" / "summary.json",
        {
            "endpoint_summaries": e1_summaries,
            "endpoint_metadata": e1_metadata,
            "reproduction_ok": reproduction_ok,
            "max_primary_abs_diff": max_primary_diff,
            "max_median_abs_diff": max_median_diff,
            "max_coverage_abs_diff": max_coverage_diff,
            "expected_cells": expected_reproduction_cells,
            "risk1_primary_decisions": config["standard_depth"]["risk1_decisions"],
            "verdict": e1,
            "thresholds": config["thresholds"],
            "seeds": config["seeds"],
            "bootstrap_draws": config["bootstrap"],
            "leave_one_study_out_rows": int(len(loo)),
        },
    )

    depth = depth_tables(config, root)
    for name, frame in depth.items():
        write_tsv(frame, output_root / "depth" / f"{name}.tsv")
    e2 = e2_verdict(depth["decomposition"], depth["gaps"], config)
    write_json(
        output_root / "depth" / "summary.json",
        {
            "verdict": e2,
            "thresholds": config["thresholds"],
            "studies": config["depth"]["studies"],
            "depths": config["depth"]["depths"],
            "representations": {
                "P": config["depth"]["p_representation"],
                "B25": config["depth"]["b25_representation"],
            },
            "row_counts": {name: int(len(frame)) for name, frame in depth.items()},
        },
    )

    cross = human_cross_platform(config, root, standard)
    write_tsv(cross["human_endpoints"], output_root / "cross_platform" / "human_endpoints.tsv")
    write_tsv(cross["mouse_endpoints"], output_root / "cross_platform" / "mouse_endpoints.tsv")
    write_tsv(cross["deltas"], output_root / "cross_platform" / "endpoint_deltas.tsv")
    write_json(output_root / "cross_platform" / "summary.json", cross["summary"])

    ceiling = ceiling_audit(standard, depth["endpoint_cells"], config)
    write_tsv(ceiling, output_root / "ceiling" / "distribution.tsv")
    ceiling_summary = {
        "rows": int(len(ceiling)),
        "endpoints": sorted(ceiling["endpoint"].unique().tolist()) if len(ceiling) else [],
        "estimable_rows": int(ceiling["estimable"].sum()) if len(ceiling) else 0,
        "not_estimable_rows": int((~ceiling["estimable"]).sum()) if len(ceiling) else 0,
        "ceiling_saturated_rows": int(ceiling["ceiling_saturated"].sum()) if len(ceiling) else 0,
        "floor_saturated_rows": int(ceiling["floor_saturated"].sum()) if len(ceiling) else 0,
        "zero_inflated_rows": int(ceiling["zero_inflated"].sum()) if len(ceiling) else 0,
        "verdict": "PASS" if len(ceiling) and ceiling["status"].notna().all() else "PARTIAL",
    }
    write_json(output_root / "ceiling" / "summary.json", ceiling_summary)

    runtime = time.perf_counter() - started
    status = "pass"
    overall = "DONE"
    if not reproduction_ok:
        status = "fail"
        overall = "FAILED"
        write_json(
            output_root / "failures" / "primary_reproduction_failure.json",
            {
                "failure_type": "primary_endpoint_reproduction_failure",
                "max_primary_abs_diff": max_primary_diff,
                "max_median_abs_diff": max_median_diff,
                "max_coverage_abs_diff": max_coverage_diff,
                "tolerance": tolerance,
                "status": "fail",
            },
        )
    elif e1["verdict"] == "FAIL":
        overall = "FAILED"
    if (
        cross["summary"]["verdict"] == "INFEASIBLE"
        or ceiling_summary["not_estimable_rows"] > 0
        or any(not value.get("estimable", True) for value in e1_summaries.values())
    ) and overall == "DONE":
        overall = "PARTIAL"

    historical_primary_failure = output_root / "failures" / "primary_reproduction_failure.json"
    if reproduction_ok and historical_primary_failure.exists():
        historical = json.loads(historical_primary_failure.read_text(encoding="utf-8"))
        if historical.get("status") == "fail":
            historical["original_status"] = "fail"
            historical["status"] = "resolved_historical"
            historical["superseded"] = True
            historical["superseded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            historical["note"] = (
                "Retained implementation-attempt evidence; the final run reproduced all 60 endpoint cells."
            )
            write_json(historical_primary_failure, historical)

    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    software_versions = _software_versions()
    launcher_path = root / "logs" / "nm_risk2_launcher.txt"
    launcher = launcher_path.read_text(encoding="utf-8").strip() if launcher_path.exists() else "unknown"
    runtime_memory = pd.DataFrame(
        [
            {
                "config_sha256": sha256_path(config_path),
                "protocol_sha256": sha256_path(protocol_path),
                "primary_bootstrap_seed": config["seeds"]["primary_bootstrap"],
                "permutation_seed": config["seeds"]["permutation"],
                "runtime_seconds": runtime,
                "peak_rss_kb": peak_rss_kb,
                "software_versions": json.dumps(software_versions, sort_keys=True),
                "status": status,
                "overall_status": overall,
                "log_path": "logs/nm_risk2_analysis.log",
                "launcher": launcher.splitlines()[0] if launcher else "unknown",
            }
        ]
    )
    write_tsv(runtime_memory, output_root / "manifests" / "runtime_memory.tsv")
    completion = {
        "schema_version": 1,
        "milestone": config["milestone"],
        "status": status,
        "overall_status": overall,
        "runtime_seconds": runtime,
        "peak_rss_kb_current_process": peak_rss_kb,
        "software_versions": software_versions,
        "parameters": {
            "config": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),
            "config_sha256": sha256_path(config_path),
            "protocol": config["protocol"],
            "protocol_sha256": sha256_path(protocol_path),
            "thresholds": config["thresholds"],
            "seeds": config["seeds"],
            "bootstrap": config["bootstrap"],
            "compute": config["compute"],
        },
        "reproduction": {
            "ok": reproduction_ok,
            "rows": int(len(reproduction)),
            "max_primary_abs_diff": max_primary_diff,
            "max_median_abs_diff": max_median_diff,
            "max_coverage_abs_diff": max_coverage_diff,
            "expected_cells": expected_reproduction_cells,
        },
        "experiments": {
            "E1_endpoint_sensitivity": e1,
            "E2_low_depth_decomposition": e2,
            "E3_cross_platform": cross["summary"],
            "E4_ceiling_floor_audit": ceiling_summary,
        },
        "input_manifest_rows": int(len(input_manifest)),
        "retained_failure_records": int(len(list((output_root / "failures").glob("*.json")))),
        "output_checksum_rows": 0,
        "frozen_inputs_mutated": False,
        "rna_values_read_by_risk2": False,
        "external_release_performed": False,
        "log_path": "logs/nm_risk2_analysis.log",
    }
    write_json(output_root / "completion_evidence.json", completion)
    write_json(output_root / "manifests" / "run_manifest.json", completion)
    output_checksums = write_output_checksums(output_root)
    completion["output_checksum_rows"] = int(len(output_checksums))
    write_json(output_root / "completion_evidence.json", completion)
    write_json(output_root / "manifests" / "run_manifest.json", completion)
    output_checksums = write_output_checksums(output_root)
    return output_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    try:
        output_root = run(args.config, args.root)
        print(output_root)
        return 0
    except Exception:
        config = load_yaml(args.config)
        failure_root = Path(args.root).resolve() / config["output_root"] / "failures"
        failure_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "fail",
            "failure_type": "unhandled_exception",
            "traceback": traceback.format_exc(),
        }
        stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        failure_path = failure_root / f"unhandled_exception_{stamp}_{time.time_ns()}.json"
        write_json(failure_path, failure)
        print(json.dumps(failure, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
