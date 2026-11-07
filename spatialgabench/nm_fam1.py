"""NM-FAM1 study-family uncertainty audit over frozen E7 summary effects.

The audit reads only frozen dataset-level B-P effects. It does not construct a
representation, execute a method, access RNA values, or inspect new outcomes.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml

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


LOGGER = logging.getLogger("nm_fam1")


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


def _sign(value: float, tolerance: float) -> int:
    value = _finite(value)
    if not math.isfinite(value):
        return 0
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def verify_freeze(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_project_path(root, config["protocol"]["freeze_manifest"])
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    if freeze.get("frozen_before_analysis") is not True:
        raise ValueError("NM-FAM1 requires an explicit pre-analysis freeze")

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
        "dots": pd.read_csv(paths["e7_study_dots"], sep="\t"),
        "summary": pd.read_csv(paths["e7_study_summary"], sep="\t"),
        "paths": paths,
    }


def validate_inputs(inputs: dict[str, Any], config: dict[str, Any]) -> None:
    dots = inputs["dots"]
    summary = inputs["summary"]
    expected = config["expected"]
    endpoints = list(config["endpoints"])
    families = list(config["study_families"])
    datasets = list(config["datasets"])
    required = {
        "endpoint",
        "dataset_key",
        "study_family",
        "contrast",
        "effect_b_minus_a",
        "ci95_low",
        "ci95_high",
        "verdict",
    }
    if not required.issubset(dots.columns):
        raise ValueError(f"E7 dots lack columns {sorted(required - set(dots.columns))}")
    if len(dots) != int(expected["dataset_endpoint_rows"]):
        raise ValueError(
            f"E7 dots have {len(dots)} rows, expected {expected['dataset_endpoint_rows']}"
        )
    if sorted(dots["endpoint"].unique().tolist()) != sorted(endpoints):
        raise ValueError("E7 endpoints do not match the config")
    if sorted(dots["study_family"].unique().tolist()) != sorted(families):
        raise ValueError("E7 study families do not match the config")
    if sorted(dots["dataset_key"].unique().tolist()) != sorted(datasets):
        raise ValueError("E7 datasets do not match the config")
    if set(dots["contrast"].astype(str)) != {str(config["contrast"])}:
        raise ValueError("E7 contains an unexpected contrast")
    if dots.duplicated(["endpoint", "dataset_key"]).any():
        raise ValueError("E7 contains duplicate endpoint/dataset rows")
    if not np.isfinite(dots["effect_b_minus_a"].astype(float)).all():
        raise ValueError("E7 contains a non-finite B-P effect")
    if not (dots["ci95_low"].astype(float) <= dots["effect_b_minus_a"].astype(float)).all():
        raise ValueError("E7 CI lower bound exceeds the point effect")
    if not (dots["effect_b_minus_a"].astype(float) <= dots["ci95_high"].astype(float)).all():
        raise ValueError("E7 point effect exceeds the CI upper bound")

    summary_required = {"endpoint", "study_family", "n_datasets", "study_median_effect"}
    if not summary_required.issubset(summary.columns):
        raise ValueError(
            f"E7 summary lacks columns {sorted(summary_required - set(summary.columns))}"
        )
    expected_summary = int(expected["study_family_endpoint_rows"])
    if len(summary) != expected_summary:
        raise ValueError(
            f"E7 summary has {len(summary)} rows, expected {expected_summary}"
        )
    observed_summary = (
        dots.groupby(["endpoint", "study_family"], as_index=False)
        .agg(
            observed_n_datasets=("dataset_key", "nunique"),
            observed_median=("effect_b_minus_a", "median"),
        )
        .merge(
            summary[["endpoint", "study_family", "n_datasets", "study_median_effect"]],
            on=["endpoint", "study_family"],
            how="outer",
            validate="one_to_one",
        )
    )
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    if observed_summary[["observed_n_datasets", "n_datasets"]].isna().any().any():
        raise ValueError("E7 dots and summary have different family/endpoint keys")
    if not (
        observed_summary["observed_n_datasets"].astype(int)
        == observed_summary["n_datasets"].astype(int)
    ).all():
        raise ValueError("E7 dots and summary disagree on family dataset counts")
    if not (
        observed_summary["observed_median"].astype(float)
        - observed_summary["study_median_effect"].astype(float)
    ).abs().le(tolerance).all():
        raise ValueError("E7 dots and summary disagree on family medians")


def build_family_effects(inputs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    dots = inputs["dots"]
    result = (
        dots.groupby(["endpoint", "study_family"], as_index=False)
        .agg(
            n_datasets=("dataset_key", "nunique"),
            family_median_effect=("effect_b_minus_a", "median"),
            family_min_effect=("effect_b_minus_a", "min"),
            family_max_effect=("effect_b_minus_a", "max"),
        )
        .sort_values(["endpoint", "study_family"])
        .reset_index(drop=True)
    )
    if len(result) != int(config["expected"]["study_family_endpoint_rows"]):
        raise ValueError("Family-effect row count does not match the freeze")
    return result


def build_baseline(
    inputs: dict[str, Any],
    family_effects: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    family_baseline = (
        family_effects.groupby("endpoint", as_index=False)
        .agg(
            n_families=("study_family", "nunique"),
            baseline_family_weighted_median=("family_median_effect", "median"),
            family_effect_min=("family_median_effect", "min"),
            family_effect_max=("family_median_effect", "max"),
        )
    )
    dataset_baseline = (
        inputs["dots"].groupby("endpoint", as_index=False)
        .agg(
            n_datasets=("dataset_key", "nunique"),
            baseline_dataset_median=("effect_b_minus_a", "median"),
            dataset_effect_min=("effect_b_minus_a", "min"),
            dataset_effect_max=("effect_b_minus_a", "max"),
        )
    )
    result = family_baseline.merge(dataset_baseline, on="endpoint", how="inner")
    result = result.sort_values("endpoint").reset_index(drop=True)
    result["baseline_family_sign"] = [
        _sign(value, tolerance) for value in result["baseline_family_weighted_median"]
    ]
    result["baseline_dataset_sign"] = [
        _sign(value, tolerance) for value in result["baseline_dataset_median"]
    ]
    result["family_effect_range"] = result["family_effect_max"] - result["family_effect_min"]
    result["dataset_effect_range"] = result["dataset_effect_max"] - result["dataset_effect_min"]
    if len(result) != int(config["expected"]["endpoints"]):
        raise ValueError("Baseline endpoint count does not match the freeze")
    return result


def build_family_loo(
    family_effects: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    material = float(config["thresholds"]["material_effect"])
    rows: list[dict[str, Any]] = []
    baseline_map = baseline.set_index("endpoint")
    for endpoint, group in family_effects.groupby("endpoint", sort=True):
        base = baseline_map.loc[endpoint]
        for omitted in sorted(group["study_family"].unique().tolist()):
            remaining = group[group["study_family"] != omitted]
            if remaining.empty:
                raise ValueError(f"No family remains after omitting {omitted} for {endpoint}")
            estimate = float(remaining["family_median_effect"].median())
            base_estimate = float(base["baseline_family_weighted_median"])
            base_sign = int(base["baseline_family_sign"])
            sign = _sign(estimate, tolerance)
            delta = estimate - base_estimate
            rows.append(
                {
                    "endpoint": endpoint,
                    "omitted_study_family": omitted,
                    "n_families_remaining": int(len(remaining)),
                    "family_weighted_median": estimate,
                    "baseline_family_weighted_median": base_estimate,
                    "delta_from_baseline": delta,
                    "sign_preserved": bool(base_sign != 0 and sign == base_sign),
                    "material_shift": bool(abs(delta) >= material - tolerance),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["endpoint", "omitted_study_family"]
    ).reset_index(drop=True)
    if len(result) != int(config["expected"]["family_loo_rows"]):
        raise ValueError("Family leave-one-out row count does not match the freeze")
    return result


def build_dataset_loo(
    inputs: dict[str, Any],
    baseline: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    dots = inputs["dots"]
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    material = float(config["thresholds"]["material_effect"])
    rows: list[dict[str, Any]] = []
    baseline_map = baseline.set_index("endpoint")
    for endpoint, group in dots.groupby("endpoint", sort=True):
        base = baseline_map.loc[endpoint]
        for omitted in sorted(group["dataset_key"].unique().tolist()):
            remaining = group[group["dataset_key"] != omitted]
            if remaining.empty:
                raise ValueError(f"No dataset remains after omitting {omitted} for {endpoint}")
            estimate = float(remaining["effect_b_minus_a"].median())
            base_estimate = float(base["baseline_dataset_median"])
            base_sign = int(base["baseline_dataset_sign"])
            sign = _sign(estimate, tolerance)
            delta = estimate - base_estimate
            omitted_family = str(
                group.loc[group["dataset_key"] == omitted, "study_family"].iloc[0]
            )
            rows.append(
                {
                    "endpoint": endpoint,
                    "omitted_dataset_key": omitted,
                    "omitted_study_family": omitted_family,
                    "n_datasets_remaining": int(len(remaining)),
                    "dataset_median": estimate,
                    "baseline_dataset_median": base_estimate,
                    "delta_from_baseline": delta,
                    "sign_preserved": bool(base_sign != 0 and sign == base_sign),
                    "material_shift": bool(abs(delta) >= material - tolerance),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["endpoint", "omitted_dataset_key"]
    ).reset_index(drop=True)
    if len(result) != int(config["expected"]["dataset_loo_rows"]):
        raise ValueError("Dataset leave-one-out row count does not match the freeze")
    return result


def build_family_bootstrap(
    family_effects: pd.DataFrame,
    baseline: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    tolerance = float(config["thresholds"]["numeric_tolerance"])
    draws = int(config["thresholds"]["bootstrap_draws"])
    seed = int(config["seeds"]["bootstrap"])
    probability_bar = float(config["thresholds"]["stability_probability"])
    baseline_map = baseline.set_index("endpoint")
    rows: list[dict[str, Any]] = []
    for endpoint, group in family_effects.groupby("endpoint", sort=True):
        values = np.sort(group["family_median_effect"].to_numpy(dtype=float))
        base = baseline_map.loc[endpoint]
        base_sign = int(base["baseline_family_sign"])
        rng = np.random.default_rng(stable_seed(seed, endpoint))
        sampled = rng.choice(values, size=(draws, len(values)), replace=True)
        medians = np.median(sampled, axis=1)
        if base_sign == 0:
            sign_probability = float("nan")
            ci_low = float(np.percentile(medians, 2.5))
            ci_high = float(np.percentile(medians, 97.5))
        else:
            sign_probability = float(np.mean(np.sign(medians) == base_sign))
            ci_low = float(np.percentile(medians, 2.5))
            ci_high = float(np.percentile(medians, 97.5))
        rows.append(
            {
                "endpoint": endpoint,
                "n_families": int(len(values)),
                "bootstrap_draws": draws,
                "observed_family_weighted_median": float(base["baseline_family_weighted_median"]),
                "bootstrap_median": float(np.median(medians)),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "sign_probability": sign_probability,
                "stability_probability_bar": probability_bar,
                "bootstrap_passes_sign_bar": bool(
                    math.isfinite(sign_probability) and sign_probability >= probability_bar
                ),
                "interval_role": "sensitivity_only_family_cluster_bootstrap",
            }
        )
    result = pd.DataFrame(rows).sort_values("endpoint").reset_index(drop=True)
    if len(result) != int(config["expected"]["bootstrap_rows"]):
        raise ValueError("Bootstrap row count does not match the freeze")
    return result


def hypothesis_verdicts(
    baseline: pd.DataFrame,
    family_loo: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    endpoints = list(config["endpoints"])

    def _stable(frame: pd.DataFrame, column: str) -> list[str]:
        return sorted(
            endpoint
            for endpoint in endpoints
            if bool(frame.loc[frame["endpoint"] == endpoint, column].all())
        )

    family_sign = _stable(family_loo, "sign_preserved")
    dataset_sign = _stable(dataset_loo, "sign_preserved")
    family_material = sorted(
        set(endpoints)
        - set(
            family_loo.loc[family_loo["material_shift"], "endpoint"].unique().tolist()
        )
    )
    dataset_material = sorted(
        set(endpoints)
        - set(
            dataset_loo.loc[dataset_loo["material_shift"], "endpoint"].unique().tolist()
        )
    )
    bootstrap_pass = sorted(
        bootstrap.loc[bootstrap["bootstrap_passes_sign_bar"], "endpoint"].unique().tolist()
    )

    def _entry(passing: list[str]) -> dict[str, Any]:
        return {
            "verdict": "PASS" if passing == sorted(endpoints) else "FAIL",
            "n_endpoints_passing": int(len(passing)),
            "n_endpoints": int(len(endpoints)),
            "passing_endpoints": passing,
            "failing_endpoints": sorted(set(endpoints) - set(passing)),
        }

    return {
        "H1": {
            "hypothesis": "Every endpoint keeps its family-weighted B-P sign under every leave-one-family-out deletion",
            **_entry(family_sign),
        },
        "H2": {
            "hypothesis": "Every endpoint keeps its dataset-median B-P sign under every leave-one-dataset-out deletion",
            **_entry(dataset_sign),
        },
        "H3": {
            "hypothesis": "No leave-one-family-out deletion shifts an endpoint median by at least the material margin",
            **_entry(family_material),
        },
        "H4": {
            "hypothesis": "No leave-one-dataset-out deletion shifts an endpoint median by at least the material margin",
            **_entry(dataset_material),
        },
        "H5": {
            "hypothesis": "Every endpoint has family-cluster bootstrap sign probability at or above the stability bar",
            **_entry(bootstrap_pass),
            "interval_role": "sensitivity_only_not_a_population_ci",
        },
    }


def key_numbers(
    family_effects: pd.DataFrame,
    baseline: pd.DataFrame,
    family_loo: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    bootstrap: pd.DataFrame,
    verdicts: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": config["task_id"],
        "contrast": config["contrast"],
        "endpoints": list(config["endpoints"]),
        "study_families": list(config["study_families"]),
        "datasets": list(config["datasets"]),
        "n_dataset_endpoint_rows": int(config["expected"]["dataset_endpoint_rows"]),
        "n_family_endpoint_rows": int(len(family_effects)),
        "n_family_loo_rows": int(len(family_loo)),
        "n_dataset_loo_rows": int(len(dataset_loo)),
        "n_bootstrap_rows": int(len(bootstrap)),
        "material_effect_margin": float(config["thresholds"]["material_effect"]),
        "stability_probability_bar": float(config["thresholds"]["stability_probability"]),
        "bootstrap_draws": int(config["thresholds"]["bootstrap_draws"]),
        "bootstrap_seed": int(config["seeds"]["bootstrap"]),
        "baseline": baseline.to_dict(orient="records"),
        "max_abs_family_loo_shift": float(
            family_loo["delta_from_baseline"].abs().max()
        ),
        "max_abs_dataset_loo_shift": float(
            dataset_loo["delta_from_baseline"].abs().max()
        ),
        "family_loo_material_shift_rows": int(family_loo["material_shift"].sum()),
        "dataset_loo_material_shift_rows": int(dataset_loo["material_shift"].sum()),
        "family_loo_sign_failure_rows": int((~family_loo["sign_preserved"]).sum()),
        "dataset_loo_sign_failure_rows": int((~dataset_loo["sign_preserved"]).sum()),
        "hypotheses": verdicts,
        "claim_boundary": [
            "Three study families are insufficient for a population CI or hierarchical random-effect estimate.",
            "The family-cluster bootstrap is a sensitivity check, not a population interval.",
            "No method, representation, biological, causal, or universal-ranking claim is authorized.",
        ],
    }


def self_check(
    config: dict[str, Any],
    attempt_root: Path,
    freeze_checks: list[dict[str, Any]],
    inputs: dict[str, Any],
    family_effects: pd.DataFrame,
    baseline: pd.DataFrame,
    family_loo: pd.DataFrame,
    dataset_loo: pd.DataFrame,
    bootstrap: pd.DataFrame,
    verdicts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = config["expected"]
    checks = {
        "freeze_hashes_match": all(item["match"] for item in freeze_checks),
        "dataset_endpoint_rows_match": len(inputs["dots"])
        == int(expected["dataset_endpoint_rows"]),
        "family_effect_rows_match": len(family_effects)
        == int(expected["study_family_endpoint_rows"]),
        "baseline_rows_match": len(baseline) == int(expected["endpoints"]),
        "family_loo_rows_match": len(family_loo) == int(expected["family_loo_rows"]),
        "dataset_loo_rows_match": len(dataset_loo)
        == int(expected["dataset_loo_rows"]),
        "bootstrap_rows_match": len(bootstrap) == int(expected["bootstrap_rows"]),
        "hypotheses_present": set(verdicts) == {"H1", "H2", "H3", "H4", "H5"},
        "bootstrap_draws_match": bool(
            (bootstrap["bootstrap_draws"] == int(config["thresholds"]["bootstrap_draws"])).all()
        ),
        "no_duplicate_family_loo": bool(
            not family_loo.duplicated(["endpoint", "omitted_study_family"]).any()
        ),
        "no_duplicate_dataset_loo": bool(
            not dataset_loo.duplicated(["endpoint", "omitted_dataset_key"]).any()
        ),
        "bootstrap_probabilities_in_range": bool(
            bootstrap["sign_probability"].dropna().between(0.0, 1.0).all()
        ),
        "interval_order_valid": bool(
            (bootstrap["bootstrap_ci95_low"] <= bootstrap["bootstrap_ci95_high"]).all()
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
        LOGGER.info("NM-FAM1 start %s", started_utc)
        freeze_checks = verify_freeze(config, root)
        LOGGER.info("Freeze hashes verified: %s", len(freeze_checks))
        inputs = load_inputs(config, root)
        validate_inputs(inputs, config)
        LOGGER.info("Frozen E7 inputs validated")
        family_effects = build_family_effects(inputs, config)
        baseline = build_baseline(inputs, family_effects, config)
        family_loo = build_family_loo(family_effects, baseline, config)
        dataset_loo = build_dataset_loo(inputs, baseline, config)
        bootstrap = build_family_bootstrap(family_effects, baseline, config)
        verdicts = hypothesis_verdicts(
            baseline, family_loo, dataset_loo, bootstrap, config
        )
        key = key_numbers(
            family_effects,
            baseline,
            family_loo,
            dataset_loo,
            bootstrap,
            verdicts,
            config,
        )

        write_tsv(family_effects, attempt_root / "tables" / "fam1_endpoint_family_effects.tsv")
        write_tsv(baseline, attempt_root / "tables" / "fam1_baseline.tsv")
        write_tsv(family_loo, attempt_root / "tables" / "fam1_family_loo.tsv")
        write_tsv(dataset_loo, attempt_root / "tables" / "fam1_dataset_loo.tsv")
        write_tsv(bootstrap, attempt_root / "tables" / "fam1_bootstrap.tsv")
        write_tsv(
            pd.DataFrame(
                [
                    {
                        "hypothesis": hypothesis,
                        **payload,
                        "passing_endpoints": "|".join(payload["passing_endpoints"]),
                        "failing_endpoints": "|".join(payload["failing_endpoints"]),
                    }
                    for hypothesis, payload in verdicts.items()
                ]
            ),
            attempt_root / "tables" / "fam1_hypotheses.tsv",
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
            family_effects,
            baseline,
            family_loo,
            dataset_loo,
            bootstrap,
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
            "software_versions": software_versions(),
            "resource_policy": {
                "workers": int(config["resources"]["workers"]),
                "nice": int(config["resources"]["nice"]),
                "memory_limit_mb": int(config["resources"]["memory_limit_mb"]),
            },
            "outputs": {
                "endpoint_family_effects": "tables/fam1_endpoint_family_effects.tsv",
                "baseline": "tables/fam1_baseline.tsv",
                "family_loo": "tables/fam1_family_loo.tsv",
                "dataset_loo": "tables/fam1_dataset_loo.tsv",
                "bootstrap": "tables/fam1_bootstrap.tsv",
                "hypotheses": "tables/fam1_hypotheses.tsv",
                "key_numbers": "key_numbers.json",
            },
        }
        write_json(manifest, attempt_root / "manifests" / "run_manifest.json")
        LOGGER.info("NM-FAM1 complete: %s", json.dumps(verdicts, sort_keys=True))
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
