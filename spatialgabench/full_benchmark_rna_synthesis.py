"""FB1 three-study paired-RNA representation synthesis."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PRIMARY = "coverage_adjusted_median_gene_wise_spearman"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def _sign_permutation_pvalue(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=float)
    observed = abs(float(deltas.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        values.append(abs(float(np.mean(deltas * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def _bootstrap_mean_ci(
    values: np.ndarray, draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _ordered_interaction_array(
    frame: pd.DataFrame, method_order: list[str], representation_order: list[str]
) -> np.ndarray:
    return frame.reindex(index=method_order, columns=representation_order).to_numpy()


def _validate(config: dict[str, Any], config_path: Path, authorization_path: Path, root: Path) -> None:
    authorization = _json(authorization_path)
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "FB1_three_study_RNA_synthesis"
        or authorization.get("policy_refit_authorized") is not False
        or authorization.get("construction_rerun_authorized") is not False
        or authorization.get("full_benchmark_expansion_authorized") is not True
        or authorization.get("universal_ranking_authorized") is not False
        or authorization.get("final_ranking_authorized") is not False
    ):
        raise PermissionError("FB1 authorization boundary is invalid")
    bindings = {
        "config": config_path,
        "implementation": Path(__file__).resolve(),
        "completion_protocol": root / config["completion_protocol"],
        "statistical_plan": root / config["statistical_plan"],
    }
    for study in config["studies"].values():
        bindings[f"metrics_{study['study_family']}"] = root / study["metrics"]
        bindings[f"gate_{study['study_family']}"] = root / study["gate"]
    for key, path in bindings.items():
        record = authorization.get(key, {})
        if not path.is_file() or record.get("path") != str(path.relative_to(root)) or sha256(path) != record.get("sha256"):
            raise ValueError(f"FB1 authorization binding mismatch: {key}")
    if config["primary_endpoint"] != PRIMARY:
        raise ValueError("FB1 primary endpoint changed")
    if float(config["materiality_threshold"]) != 0.05:
        raise ValueError("FB1 materiality threshold changed")


def run(
    config_path: str | Path,
    authorization_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    config = _yaml(config_path)
    _validate(config, config_path, authorization_path, root)
    frames = []
    for study in config["studies"].values():
        frame = pd.read_csv(root / study["metrics"], sep="\t")
        if len(frame) != int(study["expected_rows"]):
            raise ValueError(f"FB1 metric row count changed: {study['study_family']}")
        frame["study_family"] = study["study_family"]
        frame["study_role"] = study["study_role"]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    methods = config["common_methods"]
    representations = config["common_representations"]
    common = raw[
        raw.method_id.isin(methods)
        & raw.representation_id.isin(representations + [config["fragment_representation"]])
    ].copy()
    expected_per_study = (
        (len(config["annotation_methods"]) + 1) * len(representations)
        + len(config["software_methods"])
    )
    if (
        common.study_family.nunique() != 3
        or not (common.groupby("study_family").size() == expected_per_study).all()
        or not (common.status == "pass").all()
    ):
        raise ValueError("FB1 common panel is incomplete")

    annotation_methods = config["annotation_methods"]
    matrix = common[common.representation_id.isin(representations)].copy()
    annotation = matrix[matrix.method_id.isin(annotation_methods)].copy()
    controls = matrix[matrix.method_id == config["matched_control"]][
        ["study_family", "representation_id", PRIMARY]
    ].rename(columns={PRIMARY: "accessibility_matched_control"})
    deltas = annotation.merge(
        controls, on=["study_family", "representation_id"], how="left", validate="many_to_one"
    )
    deltas["delta_vs_accessibility_matched_control"] = (
        deltas[PRIMARY] - deltas["accessibility_matched_control"]
    )
    direction = (
        deltas.groupby(["method_id", "representation_id"], sort=True)
        .agg(
            study_families=("study_family", "nunique"),
            studies_positive=("delta_vs_accessibility_matched_control", lambda x: int((x > 0).sum())),
            minimum_delta=("delta_vs_accessibility_matched_control", "min"),
            median_delta=("delta_vs_accessibility_matched_control", "median"),
            maximum_delta=("delta_vs_accessibility_matched_control", "max"),
        )
        .reset_index()
    )
    direction["positive_direction_all_three_studies"] = (
        (direction.study_families == 3) & (direction.studies_positive == 3)
    )

    ranges = (
        annotation.groupby(["study_family", "method_id"], sort=True)[PRIMARY]
        .agg(["min", "max", "median"])
        .reset_index()
    )
    ranges["representation_range"] = ranges["max"] - ranges["min"]
    ranges["material_at_0_05"] = ranges["representation_range"] >= float(config["materiality_threshold"])

    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    pair_rows = []
    for method_id in annotation_methods:
        pivot = annotation[annotation.method_id == method_id].pivot(
            index="study_family", columns="representation_id", values=PRIMARY
        )
        pivot = pivot.loc[sorted(pivot.index), representations]
        for left, right in itertools.combinations(representations, 2):
            values = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_mean_ci(values, int(config["bootstrap_draws"]), rng)
            pair_rows.append(
                {
                    "method_id": method_id,
                    "representation_left": left,
                    "representation_right": right,
                    "study_families": len(values),
                    "mean_paired_delta": values.mean(),
                    "median_paired_delta": np.median(values),
                    "minimum_paired_delta": values.min(),
                    "maximum_paired_delta": values.max(),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "studies_positive": int((values > 0).sum()),
                    "exact_sign_permutation_p": _sign_permutation_pvalue(values),
                    "material_at_0_05": abs(values.mean()) >= float(config["materiality_threshold"]),
                }
            )
    pair_effects = pd.DataFrame(pair_rows)
    pair_effects["bh_q_within_30_pairs"] = _bh_adjust(
        pair_effects["exact_sign_permutation_p"].to_numpy()
    )

    cube = annotation.pivot_table(
        index="study_family", columns=["method_id", "representation_id"], values=PRIMARY
    ).loc[:, pd.MultiIndex.from_product([annotation_methods, representations])]

    def interaction_for(sample: pd.DataFrame) -> pd.DataFrame:
        cell = sample.mean(axis=0).unstack("representation_id")
        grand = float(cell.to_numpy().mean())
        method_mean = cell.mean(axis=1)
        representation_mean = cell.mean(axis=0)
        interaction = cell.copy()
        for method_id in cell.index:
            for representation_id in cell.columns:
                interaction.loc[method_id, representation_id] = (
                    cell.loc[method_id, representation_id]
                    - method_mean.loc[method_id]
                    - representation_mean.loc[representation_id]
                    + grand
                )
        return interaction

    observed_interaction = interaction_for(cube).reindex(
        index=annotation_methods, columns=representations
    )
    bootstrap_interactions = []
    for _ in range(int(config["bootstrap_draws"])):
        selected = rng.integers(0, len(cube), size=len(cube))
        bootstrap_interactions.append(
            _ordered_interaction_array(
                interaction_for(cube.iloc[selected]), annotation_methods, representations
            )
        )
    bootstrap_array = np.stack(bootstrap_interactions)
    interaction_rows = []
    for method_index, method_id in enumerate(annotation_methods):
        for representation_index, representation_id in enumerate(representations):
            value = float(observed_interaction.loc[method_id, representation_id])
            samples = bootstrap_array[:, method_index, representation_index]
            interaction_rows.append(
                {
                    "method_id": method_id,
                    "representation_id": representation_id,
                    "interaction_estimate": value,
                    "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                    "material_at_0_05": abs(value) >= float(config["materiality_threshold"]),
                }
            )
    interactions = pd.DataFrame(interaction_rows)

    values = cube.to_numpy().reshape(len(cube), len(annotation_methods), len(representations))
    grand = values.mean()
    study_mean = values.mean(axis=(1, 2), keepdims=True)
    method_mean = values.mean(axis=(0, 2), keepdims=True)
    representation_mean = values.mean(axis=(0, 1), keepdims=True)
    cell_mean = values.mean(axis=0, keepdims=True)
    interaction_array = cell_mean - method_mean - representation_mean + grand
    fitted = study_mean + method_mean + representation_mean - 2 * grand + interaction_array
    residual = values - fitted
    ss_total = float(((values - grand) ** 2).sum())
    variance = pd.DataFrame(
        [
            {"component": "study", "sum_squares": float(len(annotation_methods) * len(representations) * ((study_mean.ravel() - grand) ** 2).sum())},
            {"component": "method", "sum_squares": float(len(cube) * len(representations) * ((method_mean.ravel() - grand) ** 2).sum())},
            {"component": "representation", "sum_squares": float(len(cube) * len(annotation_methods) * ((representation_mean.ravel() - grand) ** 2).sum())},
            {"component": "method_by_representation", "sum_squares": float(len(cube) * (interaction_array ** 2).sum())},
            {"component": "residual_study_by_cell", "sum_squares": float((residual ** 2).sum())},
        ]
    )
    variance["fraction_total_sum_squares"] = variance.sum_squares / ss_total

    reversal_rows = []
    for study_family in sorted(annotation.study_family.unique()):
        pivot = annotation[annotation.study_family == study_family].pivot(
            index="method_id", columns="representation_id", values=PRIMARY
        )
        for left, right in itertools.combinations(annotation_methods, 2):
            differences = (pivot.loc[left, representations] - pivot.loc[right, representations]).to_numpy()
            reversal_rows.append(
                {
                    "study_family": study_family,
                    "method_left": left,
                    "method_right": right,
                    "representations_compared": len(representations),
                    "left_wins": int((differences > 0).sum()),
                    "right_wins": int((differences < 0).sum()),
                    "ties": int((differences == 0).sum()),
                    "ranking_reversal_across_representations": bool(
                        (differences > 0).any() and (differences < 0).any()
                    ),
                }
            )
    reversals = pd.DataFrame(reversal_rows)
    software = common[common.method_id.isin(config["software_methods"])][
        ["study_family", "study_role", "dataset_id", "method_id", PRIMARY, "status"]
    ].sort_values(["method_id", "study_family"], kind="mergesort")

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "common_study_metrics.tsv": common,
        "matched_control_deltas.tsv": deltas,
        "direction_replication.tsv": direction,
        "study_method_representation_ranges.tsv": ranges,
        "paired_representation_effects.tsv": pair_effects,
        "method_representation_interactions.tsv": interactions,
        "variance_decomposition.tsv": variance,
        "ranking_reversals.tsv": reversals,
        "fragment_software_descriptive.tsv": software,
    }
    artifacts = {}
    for name, frame in tables.items():
        path = output / name
        frame.to_csv(path, sep="\t", index=False)
        artifacts[name] = {"path": str(path.relative_to(root)), "sha256": sha256(path), "rows": len(frame)}
    material_ranges = int(ranges.material_at_0_05.sum())
    material_pairs = int(pair_effects.material_at_0_05.sum())
    material_interactions = int(interactions.material_at_0_05.sum())
    reversal_count = int(reversals.ranking_reversal_across_representations.sum())
    c1_status = (
        "material_representation_or_interaction_effect_detected"
        if material_ranges or material_pairs or material_interactions
        else "below_prespecified_materiality_despite_ranking_reversals"
        if reversal_count
        else "below_prespecified_materiality_without_ranking_reversal"
    )
    evidence = {
        "schema_version": 1,
        "gate": "FULL_BENCHMARK",
        "milestone": "FB1_three_study_paired_RNA_representation_synthesis",
        "status": "pass",
        "study_families": 3,
        "development_study_families": 2,
        "external_study_families": 1,
        "common_metric_rows": len(common),
        "matched_control_delta_rows": len(deltas),
        "direction_rows": len(direction),
        "positive_direction_all_three_studies_rows": int(direction.positive_direction_all_three_studies.sum()),
        "material_study_method_ranges": material_ranges,
        "material_paired_representation_effects": material_pairs,
        "material_method_representation_interactions": material_interactions,
        "ranking_reversal_rows": reversal_count,
        "ranking_reversal_rows_total": len(reversals),
        "maximum_study_method_representation_range": float(ranges.representation_range.max()),
        "maximum_absolute_mean_paired_representation_effect": float(pair_effects.mean_paired_delta.abs().max()),
        "maximum_absolute_interaction_estimate": float(interactions.interaction_estimate.abs().max()),
        "materiality_threshold": float(config["materiality_threshold"]),
        "bootstrap_draws": int(config["bootstrap_draws"]),
        "bootstrap_seed": int(config["bootstrap_seed"]),
        "c1_status": c1_status,
        "external_rna_access_order_deviation_carried_forward": True,
        "study_is_generalization_unit": True,
        "input_tracks_pooled": False,
        "policy_refit_run": False,
        "construction_rerun": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "artifacts": artifacts,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256(authorization_path)},
        "implementation": {"path": "spatialgabench/full_benchmark_rna_synthesis.py", "sha256": sha256(Path(__file__))},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_benchmark_fb1_rna_synthesis.yaml")
    parser.add_argument(
        "--authorization",
        default="governance/authorizations/FULL_BENCHMARK_FB1_RNA_SYNTHESIS_AUTHORIZATION_2026-07-18.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(run(args.config, args.authorization, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
