"""Factorial analysis for replicated G4-M3C simulation outcomes."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from spatialgabench.software_wrappers.common import sha256_file


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in ("authorization", "preregistration", "amendment", "truth_recovery_evidence", "metrics"):
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {field}")
    authorization = _json(root / config["authorization"])
    evidence = _json(root / config["truth_recovery_evidence"])
    if authorization.get("status") != "authorized" or authorization.get("authorized_task") != "replicated_simulation_factorial_analysis_by_stratum":
        raise PermissionError("M3C factorial-analysis authorization is not active")
    if evidence.get("status") != "pass" or evidence.get("method_runs") != int(config["expected_rows"]):
        raise ValueError("Replicated truth-recovery evidence does not pass")
    if any(bool(config.get(field)) for field in ("higher_order_interactions_run", "full_real_data_benchmark_run", "recommendation_policy_fit_run", "universal_ranking_run", "final_ranking_run")):
        raise PermissionError("A forbidden factorial-analysis flag is enabled")
    config["_path"] = path
    return config


def _bh(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _terms(factors: list[str]) -> list[tuple[str, ...]]:
    return [(factor,) for factor in factors] + list(itertools.combinations(factors, 2))


def _seed_coefficients(frame: pd.DataFrame, factors: list[str], metric: str) -> list[dict[str, Any]]:
    expected = 2 ** len(factors)
    if len(frame) != expected:
        raise ValueError(f"Factorial cell count for {metric} is {len(frame)}, expected {expected}")
    codes: dict[str, np.ndarray] = {}
    for factor in factors:
        levels = np.sort(frame[factor].unique())
        if len(levels) != 2:
            raise ValueError(f"Factor {factor} does not have two levels")
        codes[factor] = np.where(np.isclose(frame[factor].to_numpy(float), levels[0]), -1.0, 1.0)
    y = frame[metric].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"Metric {metric} contains non-finite values in analysis subset")
    rows = []
    for term in _terms(factors):
        contrast = np.prod(np.column_stack([codes[factor] for factor in term]), axis=1)
        coefficient = float(np.mean(y * contrast))
        multiplier = 2.0 if len(term) == 1 else 4.0
        rows.append({
            "term": ":".join(term), "term_order": len(term),
            "coded_coefficient": coefficient, "effect_scale_multiplier": multiplier,
            "effect_contrast": coefficient * multiplier,
        })
    return rows


def _seed_summary(values: np.ndarray, confidence: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) if n > 1 else math.nan
    if n <= 1:
        p_value = math.nan
    elif sd == 0.0:
        p_value = 0.0 if mean != 0.0 else 1.0
    else:
        p_value = float(2.0 * stats.t.sf(abs(mean / se), n - 1))
    return {
        "n_seeds": n, "mean": mean, "seed_sd": sd, "seed_se": se,
        "ci_low": mean - tcrit * se if n > 1 else math.nan,
        "ci_high": mean + tcrit * se if n > 1 else math.nan,
        "p_value": p_value,
    }


def analyze(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    metrics = pd.read_csv(root / config["metrics"], sep="\t")
    if len(metrics) != int(config["expected_rows"]) or metrics.method_id.nunique() != int(config["expected_methods"]):
        raise ValueError("M3C metric panel dimensions are incomplete")
    if set(metrics.physical_random_seed) != set(config["expected_seeds"]) or set(metrics.simulation_type) != set(config["expected_tracks"]):
        raise ValueError("M3C metric panel seed/stratum contract failed")
    factor_rows: list[dict[str, Any]] = []
    factors = list(config["factors"])
    for (track, method, seed), frame in metrics.groupby(["simulation_type", "method_id", "physical_random_seed"], sort=True):
        for metric in config["metrics_panel"]:
            if metric == "distal_component_spearman":
                high = float(frame.distal_fraction.max())
                selected = frame[np.isclose(frame.distal_fraction, high)]
                model_factors = [factor for factor in factors if factor != "distal_fraction"]
                scope = "distal_fraction_high_only"
            else:
                selected, model_factors, scope = frame, factors, "complete_six_factorial"
            for record in _seed_coefficients(selected, model_factors, metric):
                factor_rows.append({
                    "simulation_type": track, "method_id": method,
                    "physical_random_seed": int(seed), "metric_id": metric,
                    "analysis_scope": scope, **record,
                })
    seed_effects = pd.DataFrame(factor_rows)
    aggregate_rows = []
    confidence = float(config["confidence_level"])
    for keys, frame in seed_effects.groupby(["simulation_type", "method_id", "metric_id", "analysis_scope", "term", "term_order", "effect_scale_multiplier"], sort=True):
        summary = _seed_summary(frame.effect_contrast.to_numpy(float), confidence)
        aggregate_rows.append(dict(zip(
            ["simulation_type", "method_id", "metric_id", "analysis_scope", "term", "term_order", "effect_scale_multiplier"], keys
        )) | summary)
    effects = pd.DataFrame(aggregate_rows)
    effects["p_adjust_bh_within_stratum_metric"] = np.nan
    for _, indices in effects.groupby(["simulation_type", "metric_id"], sort=True).groups.items():
        effects.loc[indices, "p_adjust_bh_within_stratum_metric"] = _bh(effects.loc[indices, "p_value"].to_numpy(float))

    control = str(config["control_method"])
    paired = metrics[metrics.method_id != control].merge(
        metrics[metrics.method_id == control][["simulation_id", *config["metrics_panel"]]],
        on="simulation_id", suffixes=("", "_control"), validate="many_to_one",
    )
    delta_seed_rows = []
    for metric in config["metrics_panel"]:
        selected = paired if metric != "distal_component_spearman" else paired[paired.distal_component_eligible]
        selected = selected.copy()
        selected["delta"] = selected[metric] - selected[f"{metric}_control"]
        for keys, frame in selected.groupby(["simulation_type", "method_id", "physical_random_seed"], sort=True):
            delta_seed_rows.append({
                "simulation_type": keys[0], "method_id": keys[1], "physical_random_seed": int(keys[2]),
                "metric_id": metric, "conditions_averaged": len(frame), "mean_paired_delta_vs_accessibility_matched_control": float(frame.delta.mean()),
            })
    delta_seeds = pd.DataFrame(delta_seed_rows)
    delta_rows = []
    for keys, frame in delta_seeds.groupby(["simulation_type", "method_id", "metric_id"], sort=True):
        summary = _seed_summary(frame.mean_paired_delta_vs_accessibility_matched_control.to_numpy(float), confidence)
        delta_rows.append({"simulation_type": keys[0], "method_id": keys[1], "metric_id": keys[2], **summary})
    deltas = pd.DataFrame(delta_rows)
    deltas["p_adjust_bh_within_stratum_metric"] = np.nan
    for _, indices in deltas.groupby(["simulation_type", "metric_id"], sort=True).groups.items():
        deltas.loc[indices, "p_adjust_bh_within_stratum_metric"] = _bh(deltas.loc[indices, "p_value"].to_numpy(float))

    profile_seed = metrics.groupby(["simulation_type", "method_id", "physical_random_seed", "distal_fraction", "depth_multiplier", "zero_inflation_rate", "domain_log_effect", "decoy_fraction", "rna_latent_correlation"], sort=True)[config["metrics_panel"][0]].mean().reset_index()
    profile_rows = []
    primary = config["metrics_panel"][0]
    for factor in factors:
        collapsed = metrics.groupby(["simulation_type", "method_id", "physical_random_seed", factor], sort=True)[primary].mean().reset_index()
        for keys, frame in collapsed.groupby(["simulation_type", "method_id", factor], sort=True):
            summary = _seed_summary(frame[primary].to_numpy(float), confidence)
            profile_rows.append({"simulation_type": keys[0], "method_id": keys[1], "factor": factor, "factor_level": keys[2], **summary})
    profiles = pd.DataFrame(profile_rows)

    duplicate_rows = []
    methods = sorted(metrics.method_id.unique())
    for track in config["expected_tracks"]:
        frame = metrics[metrics.simulation_type == track]
        arrays = {
            method: frame[frame.method_id == method].sort_values("simulation_id")[primary].to_numpy(float)
            for method in methods
        }
        for left, right in itertools.combinations(methods, 2):
            difference = np.abs(arrays[left] - arrays[right])
            duplicate_rows.append({
                "simulation_type": track, "left_method_id": left, "right_method_id": right,
                "maximum_absolute_primary_difference": float(difference.max()),
                "exact_primary_duplicate": bool(np.array_equal(arrays[left], arrays[right])),
            })
    duplicates = pd.DataFrame(duplicate_rows)

    rna_terms = effects[effects.term.str.contains("rna_latent_correlation")]
    leakage_pass = bool((rna_terms["mean"].abs() <= 1e-12).all())
    if not leakage_pass:
        raise ValueError("RNA-factor regression leakage sentinel failed")
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "seed_factor_coefficients": output / "seed_level_factor_coefficients.tsv.gz",
        "factor_effects": output / "factor_main_and_two_way_effects.tsv",
        "seed_control_deltas": output / "seed_level_paired_control_deltas.tsv",
        "control_deltas": output / "paired_accessibility_matched_control_deltas.tsv",
        "factor_profiles": output / "primary_factor_level_profiles.tsv",
        "duplicate_audit": output / "simulation_method_duplicate_audit.tsv",
    }
    seed_effects.to_csv(paths["seed_factor_coefficients"], sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    effects.to_csv(paths["factor_effects"], sep="\t", index=False)
    delta_seeds.to_csv(paths["seed_control_deltas"], sep="\t", index=False)
    deltas.to_csv(paths["control_deltas"], sep="\t", index=False)
    profiles.to_csv(paths["factor_profiles"], sep="\t", index=False)
    duplicates.to_csv(paths["duplicate_audit"], sep="\t", index=False)
    evidence_path = output / "g4_m3c_simulation_factorial_analysis_evidence.json"
    evidence = {
        "schema_version": 1, "gate": "G4", "milestone": "M3C_simulation_factorial_analysis",
        "status": "pass", "method_runs": len(metrics), "seed_factor_coefficients": len(seed_effects),
        "aggregated_factor_effects": len(effects), "paired_control_delta_seed_rows": len(delta_seeds),
        "paired_control_delta_summaries": len(deltas), "tracks_reported_separately": config["expected_tracks"],
        "physical_seeds": config["expected_seeds"], "seed_role": "repeated_execution_not_biological_replicate",
        "factor_model": "six_main_plus_fifteen_two_way_interactions",
        "distal_component_model": "distal_high_subset_five_main_plus_ten_two_way_interactions",
        "higher_order_interactions_run": False, "rna_factor_regression_leakage_sentinel_passed": leakage_pass,
        "multiplicity": config["multiplicity"],
        "artifacts": {name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for name, path in paths.items()},
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "truth_recovery_evidence": {"path": config["truth_recovery_evidence"], "sha256": config["truth_recovery_evidence_sha256"]},
        "holdout_accessed": False, "full_real_data_benchmark_run": False,
        "recommendation_policy_fit_run": False, "universal_ranking_run": False,
        "final_ranking_run": False, "manuscript_results_written": False,
    }
    _write_json(evidence_path, evidence)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m3c_simulation_factorial_analysis_v2.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(analyze(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

