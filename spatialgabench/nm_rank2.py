"""NM-RANK2 real-data domain ranking-stability audit."""

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
from scipy.stats import kendalltau, rankdata

from spatialgabench.nm_rank1 import (
    bootstrap_mean_ci,
    ensure_attempt_root,
    load_config,
    output_files,
    resolve_project_path,
    sha256_file,
    software_versions,
    stable_seed,
    utc_now,
    verify_attempt,
    write_checksum_table,
    write_json,
    write_tsv,
)


LOGGER = logging.getLogger("nm_rank2")


def _float(value: Any) -> float:
    return float(value)


def _context_id(dataset_key: str, depth_fraction: float) -> str:
    return f"{dataset_key}__d{float(depth_fraction):.2f}"


def verify_freeze(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_project_path(root, config["protocol"]["freeze_manifest"])
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    config_path = resolve_project_path(root, str(freeze["config"]["path"]))
    actual_config = sha256_file(config_path)
    checks.append(
        {
            "path": str(config_path.relative_to(root)),
            "expected_sha256": str(freeze["config"]["sha256"]),
            "actual_sha256": actual_config,
            "match": actual_config == str(freeze["config"]["sha256"]),
            "role": "config",
        }
    )

    prereg_path = resolve_project_path(root, str(freeze["preregistration"]["path"]))
    actual_prereg = sha256_file(prereg_path)
    checks.append(
        {
            "path": str(prereg_path.relative_to(root)),
            "expected_sha256": str(freeze["preregistration"]["sha256"]),
            "actual_sha256": actual_prereg,
            "match": actual_prereg == str(freeze["preregistration"]["sha256"]),
            "role": "preregistration",
        }
    )

    for item in freeze["inputs"]:
        input_path = resolve_project_path(root, str(item["path"]))
        actual = sha256_file(input_path)
        checks.append(
            {
                "path": str(input_path.relative_to(root)),
                "expected_sha256": str(item["sha256"]),
                "actual_sha256": actual,
                "match": actual == str(item["sha256"]),
                "role": "analysis_input",
            }
        )

    if freeze.get("frozen_before_analysis") is not True:
        raise ValueError("The freeze manifest does not declare frozen_before_analysis")
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
        "ds1_seed": pd.read_csv(paths["ds1_domain_seed_metrics"], sep="\t"),
        "ds1_pairwise": pd.read_csv(paths["ds1_domain_pairwise"], sep="\t"),
        "ds3_seed": pd.read_csv(paths["ds3_domain_seed_metrics"], sep="\t"),
        "ds3_pairwise": pd.read_csv(paths["ds3_domain_pairwise"], sep="\t"),
        "paths": paths,
    }


def _normalise_seed_frame(
    frame: pd.DataFrame,
    source: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    if source == "ds1":
        method = frame["representation"].astype(str).replace(config["methods"]["ds1_aliases"])
        representation_role = pd.Series("primary", index=frame.index, dtype=str)
    elif source == "ds3":
        method = frame["representation_key"].astype(str)
        representation_role = frame["role"].astype(str)
    else:
        raise ValueError(f"Unknown seed source: {source}")
    if "depth_seed" in frame.columns:
        depth_seed = frame["depth_seed"].astype(int)
    else:
        depth_seed = pd.Series(0, index=frame.index, dtype=int)
    if "depth_fallback_used" in frame.columns:
        depth_fallback_used = (
            frame["depth_fallback_used"].astype(str).str.lower().isin({"true", "1", "yes"})
        )
    else:
        depth_fallback_used = pd.Series(False, index=frame.index, dtype=bool)
    if "n_hvg" in frame.columns:
        n_hvg = frame["n_hvg"].astype(int)
    else:
        n_hvg = pd.Series(np.nan, index=frame.index, dtype=float)
    result = pd.DataFrame(
        {
            "source": source,
            "dataset_key": frame["dataset_key"].astype(str),
            "depth_fraction": frame["depth_fraction"].astype(float),
            "method_id": method,
            "seed": frame["clustering_seed"].astype(int),
            "clustering_seed_role": frame["clustering_seed_role"].astype(str),
            "representation_role": representation_role,
            "depth_seed": depth_seed.astype(int),
            "depth_fallback_used": depth_fallback_used.astype(bool),
            "n_hvg": n_hvg,
            "analysis_role": representation_role.astype(str),
            "ARI": frame["ARI"].astype(float),
            "NMI": frame["NMI"].astype(float),
            "AMI": frame["AMI"].astype(float) if "AMI" in frame.columns else np.nan,
        }
    )
    return result


def _ds1_method_analysis_roles(
    pairs: pd.DataFrame,
    config: dict[str, Any],
) -> dict[tuple[str, float, str], str]:
    """Derive the DS1 frozen contrast role for each dataset/depth/method."""
    if "analysis_role" not in pairs.columns:
        return {}
    aliases = dict(config["methods"]["ds1_aliases"])
    roles: dict[tuple[str, float, str], set[str]] = {}
    for row in pairs.itertuples(index=False):
        method_a = aliases.get(str(row.representation_a), str(row.representation_a))
        method_b = aliases.get(str(row.representation_b), str(row.representation_b))
        key_base = (str(row.dataset_key), round(float(row.depth_fraction), 6))
        role = str(row.analysis_role)
        for method in (method_a, method_b):
            roles.setdefault((*key_base, method), set()).add(role)
    result: dict[tuple[str, float, str], str] = {}
    for key, values in roles.items():
        if values == {"primary"}:
            result[key] = "primary"
        elif len(values) == 1:
            result[key] = sorted(values)[0]
        else:
            result[key] = "mixed"
    return result


def _attach_analysis_roles(
    frame: pd.DataFrame,
    inputs: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    roles = _ds1_method_analysis_roles(inputs["ds1_pairwise"], config)
    mapped: list[str] = []
    for row in result.itertuples(index=False):
        if row.source != "ds1":
            mapped.append(str(row.analysis_role))
            continue
        key = (str(row.dataset_key), round(float(row.depth_fraction), 6), str(row.method_id))
        mapped.append(roles.get(key, "unmapped"))
    result["analysis_role"] = mapped
    return result


def _candidate_contexts(config: dict[str, Any]) -> list[dict[str, Any]]:
    primary = [
        {**context, "configured_role": "primary"}
        for context in config["contexts"]["primary"]
    ]
    audit = [
        {**context, "configured_role": str(context.get("role", "comparability_audit"))}
        for context in config["contexts"].get("comparability_audit", [])
    ]
    return primary + audit


def build_context_comparability(
    inputs: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Audit whether each candidate context is a common-seed, common-feature comparison."""
    all_seed = pd.concat(
        [
            _normalise_seed_frame(inputs["ds1_seed"], "ds1", config),
            _normalise_seed_frame(inputs["ds3_seed"], "ds3", config),
        ],
        ignore_index=True,
    )
    all_seed = _attach_analysis_roles(all_seed, inputs, config)
    methods = list(config["methods"]["primary"])
    rows: list[dict[str, Any]] = []
    for context in _candidate_contexts(config):
        dataset = str(context["dataset_key"])
        depth = float(context["depth_fraction"])
        source = str(context["source"])
        selected = all_seed[
            (all_seed["source"] == source)
            & (all_seed["dataset_key"] == dataset)
            & np.isclose(all_seed["depth_fraction"].to_numpy(dtype=float), depth)
            & all_seed["method_id"].isin(methods)
        ].copy()
        if source == "ds3":
            selected = selected[selected["representation_role"] == "primary"]
        context_id = _context_id(dataset, depth)

        method_meta: dict[str, dict[str, Any]] = {}
        for method in methods:
            method_rows = selected[selected["method_id"] == method]
            depth_seeds = sorted({int(value) for value in method_rows["depth_seed"].unique()})
            fallbacks = sorted({bool(value) for value in method_rows["depth_fallback_used"].unique()})
            n_hvgs = sorted({int(value) for value in method_rows["n_hvg"].dropna().unique()})
            roles = sorted({str(value) for value in method_rows["analysis_role"].unique()})
            method_meta[method] = {
                "n_rows": int(len(method_rows)),
                "n_seeds": int(method_rows["seed"].nunique()),
                "depth_seeds": depth_seeds,
                "fallbacks": fallbacks,
                "n_hvgs": n_hvgs,
                "analysis_roles": roles,
            }

        all_depth_seeds = sorted({seed for meta in method_meta.values() for seed in meta["depth_seeds"]})
        all_fallbacks = sorted({value for meta in method_meta.values() for value in meta["fallbacks"]})
        all_n_hvgs = sorted({value for meta in method_meta.values() for value in meta["n_hvgs"]})
        single_depth_seed = all(len(meta["depth_seeds"]) == 1 for meta in method_meta.values())
        single_fallback = all(len(meta["fallbacks"]) == 1 for meta in method_meta.values())
        single_n_hvg = all(len(meta["n_hvgs"]) == 1 for meta in method_meta.values())
        all_methods_present = all(meta["n_rows"] > 0 for meta in method_meta.values())
        role_primary = all(meta["analysis_roles"] == ["primary"] for meta in method_meta.values())
        depth_seed_comparable = single_depth_seed and len(all_depth_seeds) == 1
        fallback_comparable = single_fallback and len(all_fallbacks) == 1
        n_hvg_comparable = single_n_hvg and len(all_n_hvgs) == 1
        reasons: list[str] = []
        if not all_methods_present:
            reasons.append("missing_primary_method_rows")
        if not depth_seed_comparable:
            reasons.append(
                "depth_seed_mismatch:"
                + ";".join(
                    f"{method}={','.join(str(value) for value in meta['depth_seeds']) or 'missing'}"
                    for method, meta in method_meta.items()
                )
            )
        if not fallback_comparable:
            reasons.append(
                "depth_fallback_mismatch:"
                + ";".join(
                    f"{method}={','.join(str(value) for value in meta['fallbacks']) or 'missing'}"
                    for method, meta in method_meta.items()
                )
            )
        if not n_hvg_comparable:
            reasons.append(
                "n_hvg_mismatch:"
                + ";".join(
                    f"{method}={','.join(str(value) for value in meta['n_hvgs']) or 'missing'}"
                    for method, meta in method_meta.items()
                )
            )
        if not role_primary:
            reasons.append(
                "analysis_role_not_primary:"
                + ";".join(
                    f"{method}={','.join(meta['analysis_roles']) or 'missing'}"
                    for method, meta in method_meta.items()
                )
            )
        comparable = not reasons
        configured_role = str(context["configured_role"])
        primary_eligible = bool(comparable and configured_role == "primary")
        exclusion_reason = "" if primary_eligible else ";".join(reasons) if reasons else "configured_sensitivity_only"

        for method in methods:
            meta = method_meta[method]
            rows.append(
                {
                    "context_id": context_id,
                    "source": source,
                    "dataset_key": dataset,
                    "depth_fraction": depth,
                    "method_id": method,
                    "n_rows": int(meta["n_rows"]),
                    "n_seeds": int(meta["n_seeds"]),
                    "depth_seed": meta["depth_seeds"][0] if len(meta["depth_seeds"]) == 1 else "|".join(
                        str(value) for value in meta["depth_seeds"]
                    ),
                    "depth_fallback_used": meta["fallbacks"][0]
                    if len(meta["fallbacks"]) == 1
                    else "|".join(str(value) for value in meta["fallbacks"]),
                    "n_hvg": meta["n_hvgs"][0]
                    if len(meta["n_hvgs"]) == 1
                    else "|".join(str(value) for value in meta["n_hvgs"]),
                    "analysis_role": meta["analysis_roles"][0]
                    if len(meta["analysis_roles"]) == 1
                    else "|".join(meta["analysis_roles"]),
                    "common_depth_seed": all_depth_seeds[0]
                    if depth_seed_comparable
                    else "|".join(str(value) for value in all_depth_seeds),
                    "common_depth_fallback": all_fallbacks[0]
                    if fallback_comparable
                    else "|".join(str(value) for value in all_fallbacks),
                    "common_n_hvg": all_n_hvgs[0]
                    if n_hvg_comparable
                    else "|".join(str(value) for value in all_n_hvgs),
                    "configured_role": configured_role,
                    "comparable": bool(comparable),
                    "primary_eligible": bool(primary_eligible),
                    "exclusion_reason": exclusion_reason,
                }
            )
    result = pd.DataFrame(rows)
    expected = int(config["expected"].get("comparability_rows", len(result)))
    if len(result) != expected:
        raise ValueError(f"Context-comparability table has {len(result)} rows, expected {expected}")
    return result


def validate_primary_contexts(
    comparability: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Reject any configured primary context that is not a common-seed comparison."""
    primary = comparability[comparability["configured_role"] == "primary"]
    if primary["context_id"].nunique() != int(config["expected"]["primary_contexts"]):
        raise ValueError("Configured primary context count does not match the comparability audit")
    invalid = primary[~primary["primary_eligible"].astype(bool)]
    if not invalid.empty:
        details = (
            invalid.groupby("context_id")["exclusion_reason"].first().to_dict()
        )
        raise ValueError(f"Primary contexts are not common-seed comparable: {details}")


def _build_seed_long(
    inputs: dict[str, Any],
    config: dict[str, Any],
    contexts: list[dict[str, Any]],
) -> pd.DataFrame:
    all_seed = pd.concat(
        [
            _normalise_seed_frame(inputs["ds1_seed"], "ds1", config),
            _normalise_seed_frame(inputs["ds3_seed"], "ds3", config),
        ],
        ignore_index=True,
    )
    all_seed = _attach_analysis_roles(all_seed, inputs, config)
    primary_methods = list(config["methods"]["primary"])
    rows: list[pd.DataFrame] = []
    for context in contexts:
        dataset = str(context["dataset_key"])
        depth = float(context["depth_fraction"])
        source = str(context["source"])
        selected = all_seed[
            (all_seed["source"] == source)
            & (all_seed["dataset_key"] == dataset)
            & np.isclose(all_seed["depth_fraction"].to_numpy(dtype=float), depth)
            & all_seed["method_id"].isin(primary_methods)
        ].copy()
        if source == "ds3":
            selected = selected[selected["representation_role"] == "primary"]
        selected["context_id"] = _context_id(dataset, depth)
        context_id = selected["context_id"].iloc[0] if len(selected) else _context_id(dataset, depth)
        selected["configured_role"] = str(context.get("configured_role", "primary"))
        rows.append(selected)
    if not rows:
        raise ValueError("No contexts selected")
    long = pd.concat(rows, ignore_index=True)
    metrics = list(config["metrics"]["primary"])
    long = long.melt(
        id_vars=[
            "source",
            "dataset_key",
            "depth_fraction",
            "context_id",
            "method_id",
            "seed",
            "clustering_seed_role",
            "representation_role",
            "depth_seed",
            "depth_fallback_used",
            "n_hvg",
            "analysis_role",
            "configured_role",
        ],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    if long["value"].isna().any():
        raise ValueError("Long table contains missing values")
    return long


def validate_inputs(config: dict[str, Any], inputs: dict[str, Any]) -> None:
    ds1 = inputs["ds1_seed"]
    ds3 = inputs["ds3_seed"]
    expected = config["expected"]
    if len(ds1) != int(expected["ds1_rows"]):
        raise ValueError(f"Expected {expected['ds1_rows']} DS1 seed rows, found {len(ds1)}")
    if len(ds3) != int(expected["ds3_rows"]):
        raise ValueError(f"Expected {expected['ds3_rows']} DS3 seed rows, found {len(ds3)}")
    for name, frame in (("ds1", ds1), ("ds3", ds3)):
        required = {"dataset_key", "depth_fraction", "clustering_seed", "ARI", "NMI"}
        if name == "ds1":
            required |= {
                "representation",
                "depth_seed",
                "depth_fallback_used",
                "n_hvg",
            }
        else:
            required |= {"representation_key", "role", "n_hvg"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} seed table lacks {sorted(required - set(frame.columns))}")
        if frame[["ARI", "NMI"]].isna().any().any():
            raise ValueError(f"{name} seed table contains missing ARI/NMI values")


def build_primary_long(inputs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    comparability = build_context_comparability(inputs, config)
    validate_primary_contexts(comparability, config)
    primary_methods = list(config["methods"]["primary"])
    primary_contexts = [
        {**context, "configured_role": "primary"}
        for context in config["contexts"]["primary"]
    ]
    long = _build_seed_long(inputs, config, primary_contexts)
    expected_rows = int(config["expected"]["primary_context_method_metric_rows"])
    if len(long) != expected_rows:
        raise ValueError(f"Primary long table has {len(long)} rows, expected {expected_rows}")

    counts = long.groupby(["context_id", "metric", "method_id"]).size()
    expected_seeds = int(config["expected"]["seeds_per_context_method"])
    if not bool((counts == expected_seeds).all()):
        raise ValueError("Each primary context/metric/method must have exactly five seeds")
    if int(long["context_id"].nunique()) != int(config["expected"]["primary_contexts"]):
        raise ValueError("Primary context count does not match the config")
    if sorted(long["method_id"].unique().tolist()) != sorted(primary_methods):
        raise ValueError("Primary method set does not match the config")
    if sorted(long["metric"].unique().tolist()) != sorted(config["metrics"]["primary"]):
        raise ValueError("Primary metric set does not match the config")
    for context_id, group in long.groupby("context_id"):
        if group["depth_seed"].nunique() != 1:
            raise ValueError(f"Primary context {context_id} mixes depth seeds")
        if group["depth_fallback_used"].nunique() != 1:
            raise ValueError(f"Primary context {context_id} mixes depth fallback flags")
        if group["n_hvg"].nunique() != 1:
            raise ValueError(f"Primary context {context_id} mixes HVG counts")
        if set(group["analysis_role"].astype(str)) != {"primary"}:
            raise ValueError(f"Primary context {context_id} has non-primary contrast roles")
    return long


def _top1_set(values: Any, methods: list[str], tolerance: float) -> tuple[str, ...]:
    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    return tuple(
        sorted(str(method) for method, value in zip(methods, array) if value >= maximum - tolerance)
    )


def build_context_rankings(
    long: pd.DataFrame,
    methods: list[str],
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (context_id, dataset, depth, metric), group in long.groupby(
        ["context_id", "dataset_key", "depth_fraction", "metric"], sort=True
    ):
        meta = group.iloc[0]
        scores = group.groupby("method_id")["value"].median().reindex(methods)
        means = group.groupby("method_id")["value"].mean().reindex(methods)
        counts = group.groupby("method_id")["value"].size().reindex(methods)
        if scores.isna().any():
            raise ValueError(f"Missing method scores for {context_id}/{metric}")
        values = scores.to_numpy(dtype=float)
        ranks = rankdata(-values, method="average")
        top_set = _top1_set(values, methods, tolerance)
        method_range = float(np.ptp(values))
        metadata = {}
        method_metadata: dict[str, dict[str, Any]] = {}
        if "depth_seed" in group.columns:
            metadata = {
                "depth_seed": int(meta["depth_seed"]),
                "depth_fallback_used": bool(meta["depth_fallback_used"]),
                "n_hvg": int(meta["n_hvg"]),
                "analysis_role": str(meta["analysis_role"]),
            }
            for method in methods:
                selected = group[group["method_id"] == method]
                if selected.empty:
                    continue
                method_meta = selected.iloc[0]
                method_metadata[method] = {
                    "depth_seed": int(method_meta["depth_seed"]),
                    "depth_fallback_used": bool(method_meta["depth_fallback_used"]),
                    "n_hvg": int(method_meta["n_hvg"]),
                    "analysis_role": str(method_meta["analysis_role"]),
                }
        for method, score, mean, count, rank in zip(methods, values, means.to_numpy(), counts.to_numpy(), ranks):
            method_rows.append(
                {
                    "context_id": context_id,
                    "dataset_key": dataset,
                    "depth_fraction": float(depth),
                    "metric": metric,
                    "method_id": method,
                    "seed_median": float(score),
                    "seed_mean": float(mean),
                    "n_seeds": int(count),
                    "overall_rank": float(rank),
                    "context_top1_set": "|".join(top_set),
                    "context_method_range": method_range,
                    **method_metadata.get(method, metadata),
                }
            )
        summary_rows.append(
            {
                "context_id": context_id,
                "dataset_key": dataset,
                "depth_fraction": float(depth),
                "metric": metric,
                "context_top1_set": "|".join(top_set),
                "context_method_range": method_range,
                "method_median_min": float(values.min()),
                "method_median_max": float(values.max()),
                "n_methods": int(len(methods)),
                "n_seeds": int(counts.max()),
                **metadata,
            }
        )
    rankings = pd.DataFrame(method_rows)
    summary = pd.DataFrame(summary_rows)
    return rankings, summary


def build_excluded_context_rankings(
    inputs: dict[str, Any],
    config: dict[str, Any],
    comparability: pd.DataFrame,
) -> pd.DataFrame:
    """Rank demoted contexts descriptively without admitting them to H1-H7."""
    excluded_contexts = [
        {
            **context,
            "configured_role": str(context.get("role", "comparability_audit")),
        }
        for context in config["contexts"].get("comparability_audit", [])
        if str(context.get("role", "comparability_audit")) != "primary"
    ]
    if not excluded_contexts:
        return pd.DataFrame(
            columns=[
                "context_id",
                "dataset_key",
                "depth_fraction",
                "metric",
                "method_id",
                "seed_median",
                "n_seeds",
                "overall_rank",
                "context_top1_set",
                "exclusion_reason",
                "analysis_status",
            ]
        )
    long = _build_seed_long(inputs, config, excluded_contexts)
    rankings, _ = build_context_rankings(
        long,
        list(config["methods"]["primary"]),
        float(config["thresholds"]["numeric_tolerance"]),
    )
    reasons = (
        comparability.groupby("context_id")["exclusion_reason"].first().to_dict()
    )
    rankings["exclusion_reason"] = rankings["context_id"].map(reasons).fillna("")
    rankings["analysis_status"] = "NONCOMPARABLE_SENSITIVITY_ONLY"
    expected = int(
        config["expected"].get(
            "excluded_context_rankings_rows",
            len(excluded_contexts) * len(config["metrics"]["primary"]) * len(config["methods"]["primary"]),
        )
    )
    if len(rankings) != expected:
        raise ValueError(
            f"Excluded-context rankings have {len(rankings)} rows, expected {expected}"
        )
    return rankings


def build_seed_stability(
    long: pd.DataFrame,
    rankings: pd.DataFrame,
    summary: pd.DataFrame,
    methods: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    draws = int(config["thresholds"]["bootstrap_draws"])
    base_seed = int(config["seeds"]["bootstrap"])
    rows: list[dict[str, Any]] = []
    for context in summary.itertuples(index=False):
        context_id = str(context.context_id)
        metric = str(context.metric)
        overall = rankings[
            (rankings["context_id"] == context_id) & (rankings["metric"] == metric)
        ].set_index("method_id")
        overall_ranks = overall.loc[methods, "overall_rank"].to_numpy(dtype=float)
        overall_top = tuple(str(context.context_top1_set).split("|"))
        seed_values = long[
            (long["context_id"] == context_id) & (long["metric"] == metric)
        ]
        taus: list[float] = []
        top_matches: list[bool] = []
        n_degenerate = 0
        for seed, seed_group in seed_values.groupby("seed", sort=True):
            values = seed_group.set_index("method_id").loc[methods, "value"].to_numpy(dtype=float)
            seed_top = _top1_set(values, methods, tolerance)
            top_matches.append(seed_top == overall_top)
            if float(np.ptp(values)) <= tolerance:
                n_degenerate += 1
                continue
            tau = float(kendalltau(rankdata(-values, method="average"), overall_ranks).statistic)
            if math.isfinite(tau):
                taus.append(tau)
        if taus:
            mean_tau = float(np.mean(taus))
            ci_low, ci_high = bootstrap_mean_ci(
                np.asarray(taus, dtype=float),
                draws,
                stable_seed(base_seed, f"seed::{context_id}::{metric}"),
            )
        else:
            mean_tau = float("nan")
            ci_low = ci_high = float("nan")
        row = {
                "context_id": context_id,
                "dataset_key": str(context.dataset_key),
                "depth_fraction": float(context.depth_fraction),
                "metric": metric,
                "n_seeds": int(seed_values["seed"].nunique()),
                "n_valid_tau_seeds": int(len(taus)),
                "n_degenerate_all_tied": int(n_degenerate),
                "mean_seed_kendall_tau": mean_tau,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "top1_stability": float(np.mean(top_matches)) if top_matches else float("nan"),
                "context_top1_set": "|".join(overall_top),
                "context_method_range": float(context.context_method_range),
            }
        if "depth_seed" in summary.columns:
            row.update(
                {
                    "depth_seed": int(context.depth_seed),
                    "depth_fallback_used": bool(context.depth_fallback_used),
                    "n_hvg": int(context.n_hvg),
                    "analysis_role": str(context.analysis_role),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_cross_context(
    rankings: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in config["metrics"]["primary"]:
        metric_summary = summary[summary["metric"] == metric].sort_values("context_id")
        contexts = metric_summary["context_id"].tolist()
        for left_index, context_a in enumerate(contexts):
            for context_b in contexts[left_index + 1 :]:
                ranks_a = rankings[
                    (rankings["context_id"] == context_a) & (rankings["metric"] == metric)
                ].sort_values("method_id")["overall_rank"].to_numpy(dtype=float)
                ranks_b = rankings[
                    (rankings["context_id"] == context_b) & (rankings["metric"] == metric)
                ].sort_values("method_id")["overall_rank"].to_numpy(dtype=float)
                top_a = str(
                    metric_summary.loc[metric_summary["context_id"] == context_a, "context_top1_set"].iloc[0]
                )
                top_b = str(
                    metric_summary.loc[metric_summary["context_id"] == context_b, "context_top1_set"].iloc[0]
                )
                if float(np.ptp(ranks_a)) <= 0 or float(np.ptp(ranks_b)) <= 0:
                    tau = float("nan")
                    status = "NOT_ESTIMABLE_TIED"
                else:
                    tau = float(kendalltau(ranks_a, ranks_b).statistic)
                    status = "ESTIMABLE" if math.isfinite(tau) else "NOT_ESTIMABLE_TIED"
                rows.append(
                    {
                        "metric": metric,
                        "context_a": context_a,
                        "context_b": context_b,
                        "kendall_tau": tau,
                        "top1_a": top_a,
                        "top1_b": top_b,
                        "top1_match": bool(top_a == top_b),
                        "status": status,
                    }
                )
    result = pd.DataFrame(rows)
    expected = (
        len(config["metrics"]["primary"])
        * int(config["expected"]["cross_context_pairs_per_metric"])
    )
    if len(result) != expected:
        raise ValueError(f"Cross-context table has {len(result)} rows, expected {expected}")
    return result


def build_metric_agreement(
    rankings: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contexts = summary.loc[
        summary["metric"] == config["metrics"]["primary"][0], "context_id"
    ].tolist()
    for context_id in contexts:
        ranks_by_metric: dict[str, np.ndarray] = {}
        top_by_metric: dict[str, str] = {}
        for metric in config["metrics"]["primary"]:
            selected = rankings[
                (rankings["context_id"] == context_id) & (rankings["metric"] == metric)
            ].sort_values("method_id")
            ranks_by_metric[metric] = selected["overall_rank"].to_numpy(dtype=float)
            top_by_metric[metric] = str(
                summary.loc[
                    (summary["context_id"] == context_id) & (summary["metric"] == metric),
                    "context_top1_set",
                ].iloc[0]
            )
        ari = ranks_by_metric["ARI"]
        nmi = ranks_by_metric["NMI"]
        if float(np.ptp(ari)) <= 0 or float(np.ptp(nmi)) <= 0:
            tau = float("nan")
            status = "NOT_ESTIMABLE_TIED"
        else:
            tau = float(kendalltau(ari, nmi).statistic)
            status = "ESTIMABLE" if math.isfinite(tau) else "NOT_ESTIMABLE_TIED"
        rows.append(
            {
                "context_id": context_id,
                "kendall_tau": tau,
                "ari_top1_set": top_by_metric["ARI"],
                "nmi_top1_set": top_by_metric["NMI"],
                "top1_match": bool(top_by_metric["ARI"] == top_by_metric["NMI"]),
                "status": status,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != int(config["expected"]["primary_contexts"]):
        raise ValueError("Metric-agreement table has the wrong number of contexts")
    return result


def build_pair_reversals(rankings: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    methods = list(config["methods"]["primary"])
    rows: list[dict[str, Any]] = []
    for metric in config["metrics"]["primary"]:
        metric_frame = rankings[rankings["metric"] == metric]
        contexts = sorted(metric_frame["context_id"].unique().tolist())
        for left_index, method_a in enumerate(methods):
            for method_b in methods[left_index + 1 :]:
                scores = {
                    context_id: float(
                        metric_frame.loc[
                            (metric_frame["context_id"] == context_id)
                            & (metric_frame["method_id"] == method_a),
                            "seed_median",
                        ].iloc[0]
                        - metric_frame.loc[
                            (metric_frame["context_id"] == context_id)
                            & (metric_frame["method_id"] == method_b),
                            "seed_median",
                        ].iloc[0]
                    )
                    for context_id in contexts
                }
                signs = {
                    context_id: 0 if abs(diff) <= tolerance else int(np.sign(diff))
                    for context_id, diff in scores.items()
                }
                n_nonzero = 0
                n_reversal = 0
                for context_a in contexts:
                    for context_b in contexts:
                        if context_a >= context_b:
                            continue
                        sign_a = signs[context_a]
                        sign_b = signs[context_b]
                        if sign_a == 0 or sign_b == 0:
                            continue
                        n_nonzero += 1
                        if sign_a != sign_b:
                            n_reversal += 1
                reversal_fraction = (
                    float(n_reversal / n_nonzero) if n_nonzero else float("nan")
                )
                rows.append(
                    {
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_contexts": int(len(contexts)),
                        "n_nonzero_context_pairs": int(n_nonzero),
                        "n_sign_reversals": int(n_reversal),
                        "reversal_fraction": reversal_fraction,
                        "status": "ESTIMABLE" if n_nonzero else "NOT_ESTIMABLE_TIED",
                    }
                )
    result = pd.DataFrame(rows)
    expected = len(config["metrics"]["primary"]) * (
        len(methods) * (len(methods) - 1) // 2
    )
    if len(result) != expected:
        raise ValueError(f"Pair-reversal table has {len(result)} rows, expected {expected}")
    return result


def _structural_crosscheck(
    long: pd.DataFrame,
    pairs: pd.DataFrame,
    source: str,
    methods: list[str],
    tolerance: float,
) -> dict[str, Any]:
    mapping: dict[tuple[str, float, str, str], float] = {}
    for row in pairs.itertuples(index=False):
        method_a = str(row.representation_a)
        method_b = str(row.representation_b)
        if method_a == "F_signac_geneactivity":
            method_a = "F"
        if method_b == "F_signac_geneactivity":
            method_b = "F"
        for method, value in ((method_a, row.point_estimate_a), (method_b, row.point_estimate_b)):
            if method not in methods:
                continue
            key = (
                str(row.dataset_key),
                round(float(row.depth_fraction), 6),
                str(row.metric),
                method,
            )
            previous = mapping.get(key)
            if previous is not None and abs(float(previous) - float(value)) > tolerance:
                raise ValueError(f"Inconsistent pairwise structural values for {key}")
            mapping[key] = float(value)

    selected = long[
        (long["source"] == source) & (long["clustering_seed_role"] == "primary")
    ]
    n_checked = 0
    mismatches: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        key = (
            str(row.dataset_key),
            round(float(row.depth_fraction), 6),
            str(row.metric),
            str(row.method_id),
        )
        expected = mapping.get(key)
        if expected is None:
            continue
        n_checked += 1
        if abs(float(row.value) - expected) > tolerance:
            mismatches.append(
                {
                    "key": list(key),
                    "observed": float(row.value),
                    "expected": expected,
                }
            )
    return {"n_checked": int(n_checked), "mismatches": mismatches}


def build_sensitivity(
    inputs: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tolerance = float(config["thresholds"]["numeric_tolerance"])

    ds1 = _normalise_seed_frame(inputs["ds1_seed"], "ds1", config)
    d3 = ds1[(ds1["dataset_key"] == "d3_gse214986") & (ds1["method_id"].isin(["P", "B"]))]
    for metric in config["metrics"]["primary"]:
        selected = d3[
            ["method_id", "seed", "clustering_seed_role", "representation_role", metric]
        ].rename(columns={metric: "value"})
        scores = selected.groupby("method_id")["value"].median()
        diff = float(scores["P"] - scores["B"])
        per_seed_signs = []
        for _, seed_group in selected.groupby("seed"):
            seed_scores = seed_group.set_index("method_id")["value"]
            seed_diff = float(seed_scores["P"] - seed_scores["B"])
            per_seed_signs.append(0 if abs(seed_diff) <= tolerance else int(np.sign(seed_diff)))
        overall_sign = 0 if abs(diff) <= tolerance else int(np.sign(diff))
        rows.append(
            {
                "check": "d3_two_method_sign",
                "dataset_key": "d3_gse214986",
                "depth_fraction": 1.0,
                "metric": metric,
                "method_a": "P",
                "method_b": "B",
                "value_a": float(scores["P"]),
                "value_b": float(scores["B"]),
                "difference_a_minus_b": diff,
                "overall_sign": overall_sign,
                "n_seeds_same_sign": int(sum(sign == overall_sign for sign in per_seed_signs)),
                "n_seeds": int(len(per_seed_signs)),
                "detail": "two-method sensitivity only; excluded from the three-method primary ranking",
            }
        )

    ds3 = _normalise_seed_frame(inputs["ds3_seed"], "ds3", config)
    human_methods = ["P", "B", "F", "B10", "B50"]
    human = ds3[
        (ds3["dataset_key"] == "d2_human_gse205055")
        & (ds3["method_id"].isin(human_methods))
    ].copy()
    human["context_id"] = _context_id("d2_human_gse205055", 1.0)
    for metric in config["metrics"]["primary"]:
        selected = human[
            [
                "context_id",
                "dataset_key",
                "depth_fraction",
                "method_id",
                "seed",
                "clustering_seed_role",
                "representation_role",
                metric,
            ]
        ].rename(columns={metric: "value"})
        selected["metric"] = metric
        rankings, summary = build_context_rankings(selected, human_methods, tolerance)
        stability = build_seed_stability(
            selected,
            rankings,
            summary,
            human_methods,
            config,
        )
        row = stability.iloc[0]
        rows.append(
            {
                "check": "human_b10_b50_within_context_stability",
                "dataset_key": "d2_human_gse205055",
                "depth_fraction": 1.0,
                "metric": metric,
                "method_a": "|".join(human_methods),
                "method_b": "",
                "value_a": float(row["mean_seed_kendall_tau"]),
                "value_b": float(row["top1_stability"]),
                "difference_a_minus_b": float(row["mean_seed_kendall_tau"] - row["top1_stability"]),
                "overall_sign": 0,
                "n_seeds_same_sign": int(row["n_valid_tau_seeds"]),
                "n_seeds": int(row["n_seeds"]),
                "detail": "P/B/F/B10/B50 within-context sensitivity; not a cross-context claim",
            }
        )

    # AMI is present only in the DS1 table; DS3 E1 does not release AMI.
    ds1_primary = pd.concat(
        [
            _normalise_seed_frame(inputs["ds1_seed"], "ds1", config)
        ],
        ignore_index=True,
    )
    ds1_primary = ds1_primary[
        ds1_primary["method_id"].isin(config["methods"]["primary"])
        & ds1_primary["AMI"].notna()
    ]
    ds1_contexts = [
        context
        for context in config["contexts"]["primary"]
        if context["source"] == "ds1"
    ]
    for context in ds1_contexts:
        dataset = str(context["dataset_key"])
        depth = float(context["depth_fraction"])
        selected = ds1_primary[
            (ds1_primary["dataset_key"] == dataset)
            & np.isclose(ds1_primary["depth_fraction"].to_numpy(dtype=float), depth)
        ].copy()
        selected["context_id"] = _context_id(dataset, depth)
        selected = selected[
            [
                "context_id",
                "dataset_key",
                "depth_fraction",
                "method_id",
                "seed",
                "clustering_seed_role",
                "representation_role",
                "AMI",
            ]
        ]
        selected = selected.rename(columns={"AMI": "value"})
        selected["metric"] = "AMI"
        rankings, summary = build_context_rankings(
            selected, list(config["methods"]["primary"]), tolerance
        )
        stability = build_seed_stability(
            selected,
            rankings,
            summary,
            list(config["methods"]["primary"]),
            config,
        )
        row = stability.iloc[0]
        rows.append(
            {
                "check": "ami_descriptive_stability",
                "dataset_key": dataset,
                "depth_fraction": depth,
                "metric": "AMI",
                "method_a": str(row["context_top1_set"]),
                "method_b": "",
                "value_a": float(row["mean_seed_kendall_tau"]),
                "value_b": float(row["top1_stability"]),
                "difference_a_minus_b": float(row["mean_seed_kendall_tau"] - row["top1_stability"]),
                "overall_sign": 0,
                "n_seeds_same_sign": int(row["n_valid_tau_seeds"]),
                "n_seeds": int(row["n_seeds"]),
                "detail": "AMI descriptive only; not a primary metric",
            }
        )
    rows.append(
        {
            "check": "ami_descriptive_stability",
            "dataset_key": "d2_human_gse205055",
            "depth_fraction": 1.0,
            "metric": "AMI",
            "method_a": "NOT_ESTIMABLE_MISSING_AMI",
            "method_b": "",
            "value_a": float("nan"),
            "value_b": float("nan"),
            "difference_a_minus_b": float("nan"),
            "overall_sign": 0,
            "n_seeds_same_sign": 0,
            "n_seeds": 5,
            "detail": "DS3 E1 seed table does not release AMI; retained as not estimable",
        }
    )

    frame = pd.DataFrame(rows)
    structural = {
        "ds1": _structural_crosscheck(
            long=pd.concat(
                [
                    _normalise_seed_frame(inputs["ds1_seed"], "ds1", config).melt(
                        id_vars=[
                            "source",
                            "dataset_key",
                            "depth_fraction",
                            "method_id",
                            "seed",
                            "clustering_seed_role",
                            "representation_role",
                        ],
                        value_vars=list(config["metrics"]["primary"]),
                        var_name="metric",
                        value_name="value",
                    )
                ],
                ignore_index=True,
            ),
            pairs=inputs["ds1_pairwise"],
            source="ds1",
            methods=list(config["methods"]["primary"]),
            tolerance=tolerance,
        ),
        "ds3": _structural_crosscheck(
            long=pd.concat(
                [
                    _normalise_seed_frame(inputs["ds3_seed"], "ds3", config).melt(
                        id_vars=[
                            "source",
                            "dataset_key",
                            "depth_fraction",
                            "method_id",
                            "seed",
                            "clustering_seed_role",
                            "representation_role",
                        ],
                        value_vars=list(config["metrics"]["primary"]),
                        var_name="metric",
                        value_name="value",
                    )
                ],
                ignore_index=True,
            ),
            pairs=inputs["ds3_pairwise"],
            source="ds3",
            methods=list(config["methods"]["primary"]),
            tolerance=tolerance,
        ),
    }
    return frame, structural


def hypothesis_verdicts(
    seed_stability: pd.DataFrame,
    cross_context: pd.DataFrame,
    metric_agreement: pd.DataFrame,
    pair_reversals: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    stability_tau = float(config["thresholds"]["stability_tau"])
    material_gap = float(config["thresholds"]["material_tau_gap"])
    reversal_threshold = float(config["thresholds"]["sign_reversal_fraction"])

    seed_values = seed_stability["mean_seed_kendall_tau"].to_numpy(dtype=float)
    estimable_seed = seed_values[np.isfinite(seed_values)]
    if len(estimable_seed):
        h1_value = float(np.min(estimable_seed))
        h1 = "PASS" if h1_value < stability_tau else "FAIL"
        h5_value = float(np.max(estimable_seed))
        h5 = "PASS" if h5_value >= stability_tau else "FAIL"
        mean_seed = float(np.mean(estimable_seed))
    else:
        h1_value = h5_value = mean_seed = float("nan")
        h1 = h5 = "NOT_ESTIMABLE"

    cross_values = cross_context.loc[
        cross_context["status"] == "ESTIMABLE", "kendall_tau"
    ].to_numpy(dtype=float)
    if len(cross_values):
        h2_value = float(np.min(cross_values))
        mean_cross = float(np.mean(cross_values))
        h2 = "PASS" if h2_value < stability_tau else "FAIL"
    else:
        h2_value = mean_cross = float("nan")
        h2 = "NOT_ESTIMABLE"

    metric_values = metric_agreement.loc[
        metric_agreement["status"] == "ESTIMABLE", "kendall_tau"
    ].to_numpy(dtype=float)
    if len(metric_values):
        h3_value = float(np.min(metric_values))
        mean_metric = float(np.mean(metric_values))
        h3 = "PASS" if h3_value < stability_tau else "FAIL"
    else:
        h3_value = mean_metric = float("nan")
        h3 = "NOT_ESTIMABLE"

    top_sets = summary["context_top1_set"].astype(str).tolist()
    tied = any(len(item.split("|")) != 1 for item in top_sets)
    distinct = sorted(set(top_sets))
    if tied:
        h4 = "NOT_ESTIMABLE"
    elif len(distinct) >= 2:
        h4 = "PASS"
    else:
        h4 = "FAIL"

    if math.isfinite(mean_seed) and math.isfinite(mean_cross):
        h6_value = mean_seed - mean_cross
        h6 = "PASS" if h6_value >= material_gap else "FAIL"
    else:
        h6_value = float("nan")
        h6 = "NOT_ESTIMABLE"

    reversal_values = pair_reversals.loc[
        pair_reversals["status"] == "ESTIMABLE", "reversal_fraction"
    ].to_numpy(dtype=float)
    if len(reversal_values):
        h7_value = float(np.max(reversal_values))
        h7 = "PASS" if h7_value >= reversal_threshold else "FAIL"
    else:
        h7_value = float("nan")
        h7 = "NOT_ESTIMABLE"

    return {
        "H1": {
            "verdict": h1,
            "min_mean_seed_kendall_tau": h1_value,
            "threshold": stability_tau,
        },
        "H2": {
            "verdict": h2,
            "min_cross_context_kendall_tau": h2_value,
            "mean_cross_context_kendall_tau": mean_cross,
            "threshold": stability_tau,
            "n_estimable_pairs": int(len(cross_values)),
        },
        "H3": {
            "verdict": h3,
            "min_metric_kendall_tau": h3_value,
            "mean_metric_kendall_tau": mean_metric,
            "threshold": stability_tau,
            "n_estimable_contexts": int(len(metric_values)),
        },
        "H4": {
            "verdict": h4,
            "distinct_top1_sets": distinct,
            "n_distinct_top1_sets": int(len(distinct)),
            "has_tied_top_set": bool(tied),
        },
        "H5": {
            "verdict": h5,
            "max_mean_seed_kendall_tau": h5_value,
            "threshold": stability_tau,
        },
        "H6": {
            "verdict": h6,
            "mean_seed_kendall_tau": mean_seed,
            "mean_cross_context_kendall_tau": mean_cross,
            "gap": h6_value,
            "material_tau_gap": material_gap,
        },
        "H7": {
            "verdict": h7,
            "max_reversal_fraction": h7_value,
            "threshold": reversal_threshold,
        },
    }


def summarise_uncertainty(
    seed_stability: pd.DataFrame,
    cross_context: pd.DataFrame,
    metric_agreement: pd.DataFrame,
    verdicts: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    draws = int(config["thresholds"]["bootstrap_draws"])
    base_seed = int(config["seeds"]["bootstrap"])

    seed_values = seed_stability["mean_seed_kendall_tau"].dropna().to_numpy(dtype=float)
    cross_values = cross_context.loc[
        cross_context["status"] == "ESTIMABLE", "kendall_tau"
    ].to_numpy(dtype=float)
    metric_values = metric_agreement.loc[
        metric_agreement["status"] == "ESTIMABLE", "kendall_tau"
    ].to_numpy(dtype=float)

    def interval(values: np.ndarray, key: str) -> list[float]:
        if len(values) == 0:
            return [float("nan"), float("nan")]
        low, high = bootstrap_mean_ci(values, draws, stable_seed(base_seed, key))
        return [float(low), float(high)]

    seed_ci = interval(seed_values, "summary::seed")
    cross_ci = interval(cross_values, "summary::cross")
    metric_ci = interval(metric_values, "summary::metric")
    if len(seed_values) and len(cross_values):
        rng = np.random.default_rng(stable_seed(base_seed, "summary::gap"))
        indices_seed = rng.integers(0, len(seed_values), size=(draws, len(seed_values)))
        indices_cross = rng.integers(0, len(cross_values), size=(draws, len(cross_values)))
        gaps = seed_values[indices_seed].mean(axis=1) - cross_values[indices_cross].mean(axis=1)
        gap_ci = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
    else:
        gap_ci = [float("nan"), float("nan")]
    return {
        "bootstrap_draws": draws,
        "bootstrap_seed": base_seed,
        "mean_seed_tau_ci95": seed_ci,
        "mean_cross_context_tau_ci95": cross_ci,
        "mean_metric_tau_ci95": metric_ci,
        "h6_gap_ci95": gap_ci,
        "note": "Descriptive percentile intervals over observed contexts or context pairs; not a population interval.",
    }


def key_numbers(
    seed_stability: pd.DataFrame,
    cross_context: pd.DataFrame,
    metric_agreement: pd.DataFrame,
    pair_reversals: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    comparability: pd.DataFrame,
    excluded_rankings: pd.DataFrame,
    verdicts: dict[str, Any],
    uncertainty: dict[str, Any],
    structural: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": config["task_id"],
        "interpretation": {
            "claim_upgrade": False,
            "scope": "real-data domain ranking stability; no method-superiority, representation-winner, biological, or causal claim",
            "tost": "NOT_APPLICABLE",
            "unit": "dataset-depth context, not an independent study family",
        },
        "inputs": {
            "ds1_seed_rows": int(config["expected"]["ds1_rows"]),
            "ds3_seed_rows": int(config["expected"]["ds3_rows"]),
            "primary_contexts": int(summary["context_id"].nunique()),
            "metrics": list(config["metrics"]["primary"]),
            "methods": list(config["methods"]["primary"]),
            "seeds_per_context_method": int(config["expected"]["seeds_per_context_method"]),
        },
        "outputs": {
            "context_rankings_rows": int(len(summary) * len(config["methods"]["primary"])),
            "context_summary_rows": int(len(summary)),
            "seed_stability_rows": int(len(seed_stability)),
            "cross_context_rows": int(len(cross_context)),
            "metric_agreement_rows": int(len(metric_agreement)),
            "pair_reversal_rows": int(len(pair_reversals)),
            "sensitivity_rows": int(len(sensitivity)),
            "context_comparability_rows": int(len(comparability)),
            "excluded_context_ranking_rows": int(len(excluded_rankings)),
        },
        "context_comparability_summary": {
            "candidate_contexts": int(comparability["context_id"].nunique()),
            "primary_eligible_contexts": int(
                comparability.loc[
                    comparability["primary_eligible"].astype(bool), "context_id"
                ].nunique()
            ),
            "excluded_contexts": sorted(
                comparability.loc[
                    ~comparability["primary_eligible"].astype(bool), "context_id"
                ].unique().tolist()
            ),
            "excluded_reasons": {
                str(context_id): str(group["exclusion_reason"].iloc[0])
                for context_id, group in comparability.groupby("context_id", sort=True)
                if not bool(group["primary_eligible"].astype(bool).any())
            },
            "all_primary_contexts_comparable": bool(
                comparability.loc[
                    comparability["configured_role"] == "primary", "primary_eligible"
                ].astype(bool).all()
            ),
        },
        "seed_stability_summary": {
            "min_mean_seed_kendall_tau": float(seed_stability["mean_seed_kendall_tau"].min()),
            "median_mean_seed_kendall_tau": float(seed_stability["mean_seed_kendall_tau"].median()),
            "max_mean_seed_kendall_tau": float(seed_stability["mean_seed_kendall_tau"].max()),
            "min_top1_stability": float(seed_stability["top1_stability"].min()),
            "median_top1_stability": float(seed_stability["top1_stability"].median()),
            "max_top1_stability": float(seed_stability["top1_stability"].max()),
            "n_contexts_below_stability_tau": int(
                (seed_stability["mean_seed_kendall_tau"] < float(config["thresholds"]["stability_tau"])).sum()
            ),
        },
        "cross_context_summary": {
            "min_kendall_tau": float(cross_context["kendall_tau"].min()),
            "median_kendall_tau": float(cross_context["kendall_tau"].median()),
            "max_kendall_tau": float(cross_context["kendall_tau"].max()),
            "n_estimable_pairs": int((cross_context["status"] == "ESTIMABLE").sum()),
            "n_pairs_below_stability_tau": int(
                (
                    cross_context.loc[cross_context["status"] == "ESTIMABLE", "kendall_tau"]
                    < float(config["thresholds"]["stability_tau"])
                ).sum()
            ),
            "n_negative_pairs": int(
                (
                    cross_context.loc[cross_context["status"] == "ESTIMABLE", "kendall_tau"]
                    < 0
                ).sum()
            ),
            "n_top1_matches": int(cross_context["top1_match"].sum()),
        },
        "metric_agreement_summary": {
            "min_kendall_tau": float(metric_agreement["kendall_tau"].min()),
            "median_kendall_tau": float(metric_agreement["kendall_tau"].median()),
            "max_kendall_tau": float(metric_agreement["kendall_tau"].max()),
            "n_top1_matches": int(metric_agreement["top1_match"].sum()),
        },
        "pair_reversal_summary": {
            "max_reversal_fraction": float(pair_reversals["reversal_fraction"].max()),
            "n_rows_reversal_ge_20pct": int(
                (pair_reversals["reversal_fraction"] >= float(config["thresholds"]["sign_reversal_fraction"])).sum()
            ),
        },
        "structural_crosscheck": structural,
        "uncertainty": uncertainty,
        "hypotheses": verdicts,
    }


def self_check(
    config: dict[str, Any],
    root: Path,
    attempt_root: Path,
    freeze_checks: list[dict[str, Any]],
    long: pd.DataFrame,
    rankings: pd.DataFrame,
    summary: pd.DataFrame,
    seed_stability: pd.DataFrame,
    cross_context: pd.DataFrame,
    metric_agreement: pd.DataFrame,
    pair_reversals: pd.DataFrame,
    sensitivity: pd.DataFrame,
    comparability: pd.DataFrame,
    excluded_rankings: pd.DataFrame,
    verdicts: dict[str, Any],
    structural: dict[str, Any],
) -> dict[str, Any]:
    freeze_manifest = json.loads(
        resolve_project_path(root, config["protocol"]["freeze_manifest"]).read_text(encoding="utf-8")
    )
    expected = config["expected"]
    n_methods = len(config["methods"]["primary"])
    n_metrics = len(config["metrics"]["primary"])
    n_contexts = int(expected["primary_contexts"])
    expected_rankings = n_contexts * n_metrics * n_methods
    expected_stability = n_contexts * n_metrics
    expected_cross = n_metrics * int(expected["cross_context_pairs_per_metric"])
    expected_pair_reversals = n_metrics * (n_methods * (n_methods - 1) // 2)
    expected_comparability = int(expected.get("comparability_rows", len(comparability)))
    expected_excluded = int(expected.get("excluded_context_rankings_rows", len(excluded_rankings)))
    checks = {
        "freeze_manifest_frozen_before_analysis": bool(
            freeze_manifest.get("frozen_before_analysis") is True
        ),
        "freeze_hashes_match": all(item["match"] for item in freeze_checks),
        "primary_long_rows_expected": len(long) == int(expected["primary_context_method_metric_rows"]),
        "primary_contexts_expected": long["context_id"].nunique() == n_contexts,
        "primary_methods_3": long["method_id"].nunique() == len(config["methods"]["primary"]),
        "primary_metrics_2": long["metric"].nunique() == len(config["metrics"]["primary"]),
        "context_rankings_rows_expected": len(rankings) == expected_rankings,
        "context_summary_rows_expected": len(summary) == expected_stability,
        "seed_stability_rows_expected": len(seed_stability) == expected_stability,
        "cross_context_rows_expected": len(cross_context) == expected_cross,
        "metric_agreement_rows_expected": len(metric_agreement) == n_contexts,
        "pair_reversal_rows_expected": len(pair_reversals) == expected_pair_reversals,
        "sensitivity_rows_positive": len(sensitivity) > 0,
        "comparability_rows_expected": len(comparability) == expected_comparability,
        "comparability_contexts_expected": comparability["context_id"].nunique()
        == int(expected.get("comparability_contexts", comparability["context_id"].nunique())),
        "excluded_context_rankings_rows_expected": len(excluded_rankings) == expected_excluded,
        "primary_contexts_all_comparable": bool(
            comparability.loc[
                comparability["configured_role"] == "primary", "primary_eligible"
            ].astype(bool).all()
        ),
        "excluded_context_count_expected": int(
            comparability.loc[
                ~comparability["primary_eligible"].astype(bool), "context_id"
            ].nunique()
        )
        == int(expected.get("excluded_noncomparable_contexts", 0)),
        "no_noncomparable_context_in_primary": bool(
            not set(long["context_id"].unique()).intersection(
                set(
                    comparability.loc[
                        ~comparability["primary_eligible"].astype(bool), "context_id"
                    ].unique()
                )
            )
        ),
        "primary_depth_metadata_constant": bool(
            (long.groupby("context_id")[["depth_seed", "depth_fallback_used", "n_hvg"]].nunique() <= 1)
            .all()
            .all()
        ),
        "primary_analysis_roles_all_primary": bool(
            set(long["analysis_role"].astype(str)) == {"primary"}
        ),
        "primary_metadata_columns_present": bool(
            {"depth_seed", "depth_fallback_used", "n_hvg", "analysis_role"}.issubset(
                rankings.columns
            )
        ),
        "ranks_in_1_to_3": bool(rankings["overall_rank"].between(1, 3).all()),
        "no_missing_primary_values": bool(not long["value"].isna().any()),
        "tau_range_valid": bool(
            ((seed_stability["mean_seed_kendall_tau"].dropna() >= -1 - 1e-9)
             & (seed_stability["mean_seed_kendall_tau"].dropna() <= 1 + 1e-9)).all()
        ),
        "hypotheses_H1_to_H7": set(verdicts) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"},
        "structural_crosschecks_no_mismatch": all(
            not item["mismatches"] for item in structural.values()
        ),
        "structural_crosschecks_have_rows": all(
            item["n_checked"] > 0 for item in structural.values()
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
        LOGGER.info("NM-RANK2 start %s", started_utc)
        freeze_checks = verify_freeze(config, root)
        LOGGER.info("Freeze hashes verified: %s", len(freeze_checks))
        inputs = load_inputs(config, root)
        validate_inputs(config, inputs)
        LOGGER.info("Inputs validated")

        comparability = build_context_comparability(inputs, config)
        validate_primary_contexts(comparability, config)
        LOGGER.info(
            "Primary contexts common-seed comparable: %s",
            int(comparability.loc[comparability["configured_role"] == "primary", "context_id"].nunique()),
        )
        long = build_primary_long(inputs, config)
        methods = list(config["methods"]["primary"])
        rankings, summary = build_context_rankings(
            long, methods, float(config["thresholds"]["numeric_tolerance"])
        )
        seed_stability = build_seed_stability(long, rankings, summary, methods, config)
        cross_context = build_cross_context(rankings, summary, config)
        metric_agreement = build_metric_agreement(rankings, summary, config)
        pair_reversals = build_pair_reversals(rankings, config)
        sensitivity, structural = build_sensitivity(inputs, config)
        excluded_rankings = build_excluded_context_rankings(
            inputs, config, comparability
        )
        verdicts = hypothesis_verdicts(
            seed_stability,
            cross_context,
            metric_agreement,
            pair_reversals,
            summary,
            config,
        )
        uncertainty = summarise_uncertainty(
            seed_stability, cross_context, metric_agreement, verdicts, config
        )
        LOGGER.info("Hypotheses: %s", json.dumps(verdicts, sort_keys=True))

        key = key_numbers(
            seed_stability,
            cross_context,
            metric_agreement,
            pair_reversals,
            summary,
            sensitivity,
            comparability,
            excluded_rankings,
            verdicts,
            uncertainty,
            structural,
            config,
        )
        write_tsv(rankings, attempt_root / "tables" / "rank2_context_rankings.tsv")
        write_tsv(seed_stability, attempt_root / "tables" / "rank2_seed_stability.tsv")
        write_tsv(cross_context, attempt_root / "tables" / "rank2_cross_context.tsv")
        write_tsv(metric_agreement, attempt_root / "tables" / "rank2_metric_agreement.tsv")
        write_tsv(pair_reversals, attempt_root / "tables" / "rank2_pair_reversals.tsv")
        write_tsv(sensitivity, attempt_root / "tables" / "rank2_sensitivity.tsv")
        write_tsv(
            comparability,
            attempt_root / "tables" / "rank2_context_comparability.tsv",
        )
        write_tsv(
            excluded_rankings,
            attempt_root / "tables" / "rank2_excluded_context_rankings.tsv",
        )
        write_json(key, attempt_root / "key_numbers.json")
        write_tsv(pd.DataFrame(freeze_checks), attempt_root / "manifests" / "input_checksums.tsv")

        self_check(
            config,
            root,
            attempt_root,
            freeze_checks,
            long,
            rankings,
            summary,
            seed_stability,
            cross_context,
            metric_agreement,
            pair_reversals,
            sensitivity,
            comparability,
            excluded_rankings,
            verdicts,
            structural,
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
            "seed": int(config["seeds"]["bootstrap"]),
            "software_versions": software_versions(),
            "outputs": {
                "context_rankings": "tables/rank2_context_rankings.tsv",
                "seed_stability": "tables/rank2_seed_stability.tsv",
                "cross_context": "tables/rank2_cross_context.tsv",
                "metric_agreement": "tables/rank2_metric_agreement.tsv",
                "pair_reversals": "tables/rank2_pair_reversals.tsv",
                "sensitivity": "tables/rank2_sensitivity.tsv",
                "context_comparability": "tables/rank2_context_comparability.tsv",
                "excluded_context_rankings": "tables/rank2_excluded_context_rankings.tsv",
                "key_numbers": "key_numbers.json",
            },
        }
        write_json(manifest, attempt_root / "manifests" / "run_manifest.json")
        LOGGER.info("NM-RANK2 complete: %s", json.dumps(verdicts, sort_keys=True))
        write_checksum_table(attempt_root)
        verification = verify_attempt(attempt_root, write=True)
        if not verification["all_hashes_match"] or not verification["coverage_all_non_terminal_files"]:
            raise ValueError(f"Checksum verification failed: {verification}")
        result = {
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
        return result
    finally:
        LOGGER.removeHandler(file_handler)
        file_handler.close()
