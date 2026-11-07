"""One-pass, leaf-specific G5-H5C external evaluation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spatialgabench.g4_m3b_rna import _gtf_annotations, _rna_ranks, normalize_total_log1p
from spatialgabench.g4_m3h_rna_replication import _evaluate_method


PRIMARY_ENDPOINT = "coverage_adjusted_median_gene_wise_spearman"


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_rna_gene_ids(
    identifiers: list[str], biotypes: dict[str, str], unique_names: dict[str, str]
) -> list[str | None]:
    mapped: list[str | None] = []
    for identifier in identifiers:
        direct = re.sub(r"\.\d+$", "", identifier)
        mapped.append(direct if direct in biotypes else unique_names.get(identifier))
    return mapped


def load_wide_csv_sparse(path: str | Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """Load the deposited gene-by-spot CSV as sparse counts."""
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    genes: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            barcodes = next(reader)
        except StopIteration as exc:
            raise ValueError("RNA CSV is empty") from exc
        if barcodes and barcodes[0] == "":
            barcodes = barcodes[1:]
        for row in reader:
            if not row:
                continue
            gene, values_text = row[0], row[1:]
            if not gene or len(values_text) != len(barcodes):
                raise ValueError(f"RNA CSV width mismatch for {gene!r}")
            values = np.asarray(values_text, dtype=np.int64)
            if np.any(values < 0):
                raise ValueError("RNA counts must be nonnegative")
            nonzero = np.flatnonzero(values)
            if len(nonzero):
                row_parts.append(np.full(len(nonzero), len(genes), dtype=np.int64))
                column_parts.append(nonzero.astype(np.int64, copy=False))
                data_parts.append(values[nonzero])
            genes.append(gene)
    if len(set(barcodes)) != len(barcodes) or len(set(genes)) != len(genes):
        raise ValueError("RNA barcodes and genes must be unique")
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
    columns = np.concatenate(column_parts) if column_parts else np.empty(0, dtype=np.int64)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.int64)
    matrix = sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(genes), len(barcodes)), dtype=np.int64
    ).tocsr()
    return matrix, genes, barcodes


def _prepare_rna(
    rna_path: Path,
    annotation_path: Path,
    obs_names: pd.Index,
    minimum_fraction: float,
    target_sum: float,
) -> tuple[sparse.csr_matrix, pd.DataFrame, np.ndarray, dict[str, str]]:
    genes_by_spots, gene_names, barcodes = load_wide_csv_sparse(rna_path)
    biotypes, unique_names = _gtf_annotations(annotation_path)
    gene_ids = _canonical_rna_gene_ids(gene_names, biotypes, unique_names)
    features = pd.DataFrame({"gene_name": gene_names, "gene_id": gene_ids})
    features["gene_biotype"] = features["gene_id"].map(biotypes)
    barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
    if any(str(barcode) not in barcode_index for barcode in obs_names):
        raise ValueError("Frozen H5C observations are missing from RNA")
    columns = np.asarray([barcode_index[str(barcode)] for barcode in obs_names], dtype=np.int64)
    spots_by_genes = genes_by_spots[:, columns].transpose().tocsr()
    detection = np.asarray((spots_by_genes > 0).sum(axis=0)).ravel()
    threshold = math.ceil(minimum_fraction * len(obs_names))
    eligible = (features["gene_biotype"].to_numpy() == "protein_coding") & (
        detection >= threshold
    )
    eligible_indices = np.flatnonzero(eligible)
    eligible_features = features.iloc[eligible_indices].copy().reset_index(drop=True)
    eligible_features["detected_spots"] = detection[eligible_indices]
    eligible_features["detected_spot_fraction"] = detection[eligible_indices] / len(obs_names)
    normalized = normalize_total_log1p(spots_by_genes, target_sum)
    return normalized, eligible_features, eligible_indices, unique_names


def _validate_authorization(
    root: Path, preflight_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight = _json(preflight_path)
    authorization = _json(authorization_path)
    if preflight.get("status") != "pass" or preflight.get("evaluation_run") is not False:
        raise ValueError("H5C preflight does not pass")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "H5C_external_evaluation"
        or authorization.get("rna_value_access_authorized") is not True
        or authorization.get("paired_rna_evaluation_authorized") is not True
        or authorization.get("portability_evaluation_authorized") is not True
        or authorization.get("regret_computation_authorized") is not True
        or authorization.get("policy_refit_authorized") is not False
        or authorization.get("method_rerun_authorized") is not False
        or authorization.get("outcome_based_retry_authorized") is not False
        or authorization.get("full_benchmark_authorized") is not False
        or authorization.get("results_writing_authorized") is not False
        or authorization.get("universal_ranking_authorized") is not False
        or authorization.get("final_ranking_authorized") is not False
    ):
        raise PermissionError("H5C authorization boundary is invalid")
    bindings = {
        "preflight_evidence": preflight_path,
        "method_input_manifest": root / preflight["method_input_manifest"]["path"],
        "evaluation_manifest": root / preflight["evaluation_manifest"]["path"],
        "comparator_manifest": root / preflight["comparator_manifest"]["path"],
        "implementation_manifest": root / preflight["implementation_manifest"]["path"],
    }
    for key, path in bindings.items():
        record = authorization.get(key, {})
        if not path.is_file() or record.get("path") != str(path.relative_to(root)) or sha256(path) != record.get("sha256"):
            raise ValueError(f"H5C authorization binding mismatch: {key}")
    implementation = preflight["implementation"]
    current = root / implementation["path"]
    if not current.is_file() or sha256(current) != implementation["sha256"]:
        raise ValueError("H5C evaluator changed after preflight")
    implementation_manifest = pd.read_csv(
        root / preflight["implementation_manifest"]["path"], sep="\t"
    )
    for row in implementation_manifest.itertuples(index=False):
        path = root / str(row.path)
        if not path.is_file() or sha256(path) != row.sha256:
            raise ValueError(f"H5C implementation dependency changed: {row.path}")
    for key in ("rna_matrix", "gene_annotation", "evaluation_manifest", "comparator_manifest"):
        record = preflight[key]
        path = root / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"H5C live frozen input changed: {key}")
    return preflight, authorization


def _random_comparators(
    leaf_key: tuple[str, str],
    leaf: pd.DataFrame,
    comparator: pd.Series,
    oracle_utility: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    methods = sorted(leaf["method_id"].astype(str))
    lookup = leaf.set_index("method_id")["utility"].to_dict()
    successful = leaf.set_index("method_id")["evaluation_status"].eq("pass").to_dict()
    rows: list[dict[str, Any]] = []
    regrets: list[float] = []
    failures: list[float] = []
    size = int(comparator["recommended_set_size"])
    for draw in range(int(comparator["random_draws"])):
        selected = sorted(rng.choice(methods, size=size, replace=False).tolist())
        valid = [method for method in selected if successful.get(method, False)]
        failed = not valid
        utility = max(float(lookup[method]) for method in valid) if valid else math.nan
        regret = 1.0 if failed else oracle_utility - utility
        rows.append(
            {
                "study_id": leaf_key[0],
                "policy_id": leaf_key[1],
                "draw": draw,
                "selected_methods": ";".join(selected),
                "set_utility": utility,
                "regret": regret,
                "set_failure": failed,
            }
        )
        regrets.append(regret)
        failures.append(float(failed))
    values = np.asarray(regrets, dtype=float)
    return rows, {
        "random_mean_regret": float(values.mean()),
        "random_median_regret": float(np.median(values)),
        "random_regret_q025": float(np.quantile(values, 0.025)),
        "random_regret_q975": float(np.quantile(values, 0.975)),
        "random_failure_rate": float(np.mean(failures)),
    }


def run(
    preflight_evidence: str | Path,
    authorization: str | Path,
    root: str | Path = ".",
) -> Path:
    root = Path(root).resolve()
    preflight_path = Path(preflight_evidence).resolve()
    authorization_path = Path(authorization).resolve()
    preflight, _ = _validate_authorization(root, preflight_path, authorization_path)
    method_inputs = pd.read_csv(root / preflight["method_input_manifest"]["path"], sep="\t")
    evaluation_plan = pd.read_csv(root / preflight["evaluation_manifest"]["path"], sep="\t")
    comparators = pd.read_csv(root / preflight["comparator_manifest"]["path"], sep="\t")
    output = root / preflight["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    if not pd.api.types.is_bool_dtype(evaluation_plan["recommended"]):
        raise ValueError("H5C recommended-set flags must be frozen booleans")
    for method in method_inputs.itertuples(index=False):
        path = root / str(method.gene_activity_h5ad)
        if not path.is_file() or sha256(path) != method.gene_activity_sha256:
            raise ValueError(f"H5C live method output changed: {method.gene_activity_h5ad}")

    rna_methods = method_inputs[method_inputs["dataset_id"] == "gse308623_p2s3"].copy()
    if len(rna_methods) != 27:
        raise ValueError("H5C RNA panel must contain 27 frozen outputs")
    first = ad.read_h5ad(root / rna_methods.iloc[0]["gene_activity_h5ad"], backed="r")
    obs_names = first.obs_names.copy()
    first.file.close()
    if len(obs_names) != int(preflight["rna_observations"]):
        raise ValueError("H5C RNA observation count changed")
    rna, eligible_features, eligible_indices, unique_names = _prepare_rna(
        root / preflight["rna_matrix"]["path"],
        root / preflight["gene_annotation"]["path"],
        obs_names,
        float(preflight["minimum_detected_spot_fraction"]),
        float(preflight["normalization_target_sum"]),
    )
    ranks = _rna_ranks(rna, eligible_indices, int(preflight["correlation_chunk_genes"]))
    metric_rows: list[dict[str, Any]] = []
    gene_frames: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []
    for _, method in rna_methods.iterrows():
        try:
            metric, genes = _evaluate_method(
                method,
                root,
                obs_names,
                eligible_features,
                rna,
                eligible_indices,
                ranks,
                unique_names,
                float(preflight["normalization_target_sum"]),
                int(preflight["correlation_chunk_genes"]),
            )
            metric_rows.append(metric)
            gene_frames.append(genes)
        except Exception as exc:
            failure_rows.append(
                {
                    "dataset_id": method.dataset_id,
                    "representation_id": method.representation_id,
                    "method_id": method.method_id,
                    "status": "fail",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    metrics = pd.DataFrame(metric_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["dataset_id", "representation_id", "method_id", "status", "reason"],
    )
    gene_metrics = pd.concat(gene_frames, ignore_index=True) if gene_frames else pd.DataFrame()

    utility_rows: list[dict[str, Any]] = []
    input_lookup = method_inputs.set_index(["study_id", "representation_id", "method_id"])
    metric_lookup = metrics.set_index(["representation_id", "method_id"]) if len(metrics) else pd.DataFrame()
    failed_rna = {(row["representation_id"], row["method_id"]) for row in failure_rows}
    for row in evaluation_plan.itertuples(index=False):
        key = (row.study_id, row.representation_id, row.method_id)
        construction = input_lookup.loc[key]
        status = "pass" if construction["construction_status"] == "pass" else "fail"
        primary = math.nan
        if row.task_id == "paired_rna_fidelity":
            metric_key = (row.representation_id, row.method_id)
            if metric_key in failed_rna or metric_key not in metric_lookup.index:
                status = "fail"
            else:
                primary = float(metric_lookup.loc[metric_key, PRIMARY_ENDPOINT])
            utility = float(np.clip((primary + 1.0) / 2.0, 0.0, 1.0)) if status == "pass" else math.nan
        else:
            utility = 1.0 if status == "pass" else 0.0
        utility_rows.append(
            {
                **row._asdict(),
                "construction_status": construction["construction_status"],
                "evaluation_status": status,
                "primary_endpoint_value": primary,
                "utility": utility,
            }
        )
    utilities = pd.DataFrame(utility_rows)

    leaf_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(preflight["random_seed"]))
    comparator_lookup = comparators.set_index(["study_id", "policy_id"])
    for leaf_key, leaf in utilities.groupby(["study_id", "policy_id"], sort=True):
        comparator = comparator_lookup.loc[leaf_key]
        successful = leaf[leaf["evaluation_status"] == "pass"]
        recommended = leaf[leaf["recommended"].astype(bool)]
        recommended_successful = recommended[recommended["evaluation_status"] == "pass"]
        evaluable = len(successful) > 0
        oracle = float(successful["utility"].max()) if evaluable else math.nan
        policy_failure = len(recommended_successful) == 0
        policy_utility = float(recommended_successful["utility"].max()) if not policy_failure else math.nan
        policy_regret = (1.0 if policy_failure else oracle - policy_utility) if evaluable else math.nan
        default = leaf[leaf["method_id"] == comparator["default_method"]]
        default_success = len(default) == 1 and default.iloc[0]["evaluation_status"] == "pass"
        default_utility = float(default.iloc[0]["utility"]) if default_success else math.nan
        default_regret = (oracle - default_utility) if default_success else (1.0 if evaluable else math.nan)
        draws, random_summary = _random_comparators(leaf_key, leaf, comparator, oracle, rng)
        random_rows.extend(draws)
        leaf_rows.append(
            {
                "study_id": leaf_key[0],
                "policy_id": leaf_key[1],
                "task_id": leaf.iloc[0]["task_id"],
                "representation_id": leaf.iloc[0]["representation_id"],
                "admissible_methods": len(leaf),
                "successful_methods": len(successful),
                "recommended_methods": int(leaf["recommended"].astype(bool).sum()),
                "evaluable": evaluable,
                "oracle_utility": oracle,
                "policy_set_utility": policy_utility,
                "policy_regret": policy_regret,
                "policy_set_failure": policy_failure,
                "default_method": comparator["default_method"],
                "default_utility": default_utility,
                "default_regret": default_regret,
                "default_failure": not default_success,
                **random_summary,
            }
        )
    leaves = pd.DataFrame(leaf_rows)
    random_draws = pd.DataFrame(random_rows)

    summary_rows: list[dict[str, Any]] = []
    for policy_id, frame in leaves.groupby("policy_id", sort=True):
        comparator = comparators[comparators["policy_id"] == policy_id].iloc[0]
        summary_rows.append(
            {
                "policy_id": policy_id,
                "task_id": frame.iloc[0]["task_id"],
                "representation_id": frame.iloc[0]["representation_id"],
                "external_studies": len(frame),
                "mean_policy_regret": frame["policy_regret"].mean(),
                "mean_default_regret": frame["default_regret"].mean(),
                "mean_random_regret": frame["random_mean_regret"].mean(),
                "policy_failure_rate": frame["policy_set_failure"].mean(),
                "default_failure_rate": frame["default_failure"].mean(),
                "random_failure_rate": frame["random_failure_rate"].mean(),
                "regret_noninferior_to_default": (
                    frame["policy_regret"].mean() - frame["default_regret"].mean()
                    <= float(comparator["regret_margin"])
                ),
                "regret_noninferior_to_random": (
                    frame["policy_regret"].mean() - frame["random_mean_regret"].mean()
                    <= float(comparator["regret_margin"])
                ),
                "failure_noninferior_to_default": (
                    frame["policy_set_failure"].mean() - frame["default_failure"].mean()
                    <= float(comparator["failure_margin"])
                ),
                "failure_noninferior_to_random": (
                    frame["policy_set_failure"].mean() - frame["random_failure_rate"].mean()
                    <= float(comparator["failure_margin"])
                ),
                "inference_scope": (
                    "two_study_external_portability"
                    if frame.iloc[0]["task_id"] == "portability_runtime_failure"
                    else "single_external_study_descriptive_with_access_order_deviation"
                ),
            }
        )
    task_summary = pd.DataFrame(summary_rows)

    controls = metrics[metrics["method_id"] == "accessibility_matched_random_peak_to_gene_control_pc"]
    annotations = metrics[metrics["method_category"] == "hand_written_annotation"]
    control_deltas = annotations.merge(
        controls[["representation_id", PRIMARY_ENDPOINT]].rename(
            columns={PRIMARY_ENDPOINT: "accessibility_matched_control"}
        ),
        on="representation_id",
        how="left",
        validate="many_to_one",
    )
    control_deltas["delta_vs_accessibility_matched_control"] = (
        control_deltas[PRIMARY_ENDPOINT] - control_deltas["accessibility_matched_control"]
    )

    artifacts = {
        "method_input_manifest.tsv": method_inputs,
        "rna_eligible_genes.tsv": eligible_features,
        "rna_metrics.tsv": metrics,
        "rna_evaluation_failures.tsv": failures,
        "method_utility.tsv": utilities,
        "leaf_results.tsv": leaves,
        "task_summary.tsv": task_summary,
        "rna_control_deltas.tsv": control_deltas,
    }
    artifact_records: dict[str, dict[str, Any]] = {}
    for name, frame in artifacts.items():
        path = output / name
        frame.to_csv(path, sep="\t", index=False)
        artifact_records[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "rows": len(frame),
        }
    gene_path = output / "gene_level_correlations.tsv.gz"
    gene_metrics.to_csv(gene_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    artifact_records[gene_path.name] = {
        "path": str(gene_path.relative_to(root)),
        "sha256": sha256(gene_path),
        "rows": len(gene_metrics),
    }
    random_path = output / "random_comparator_draws.tsv.gz"
    random_draws.to_csv(random_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    artifact_records[random_path.name] = {
        "path": str(random_path.relative_to(root)),
        "sha256": sha256(random_path),
        "rows": len(random_draws),
    }
    complete_dimensions = (
        len(metrics) + len(failures) == 27
        and len(utilities) == 69
        and len(leaves) == 15
        and len(random_draws) == 150000
    )
    completion = {
        "schema_version": 1,
        "gate": "G5",
        "milestone": "H5C_external_evaluation",
        "status": "pass" if complete_dimensions else "partial",
        "rna_method_and_control_outputs_expected": 27,
        "rna_method_and_control_outputs_passed": len(metrics),
        "rna_method_evaluation_failures": len(failures),
        "admissible_evaluation_rows": len(utilities),
        "study_leaf_units": len(leaves),
        "random_comparator_rows": len(random_draws),
        "rna_observations": len(obs_names),
        "rna_eligible_genes": len(eligible_features),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "rna_values_accessed": True,
        "holdout_outcomes_accessed": True,
        "regret_computation_run": True,
        "limited_preauthorization_access_deviation_disclosed": True,
        "construction_outputs_mutated": False,
        "method_rerun": False,
        "policy_refit_run": False,
        "full_benchmark_run": False,
        "results_writing_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "artifacts": artifact_records,
        "preflight_evidence": {
            "path": str(preflight_path.relative_to(root)),
            "sha256": sha256(preflight_path),
        },
        "authorization": {
            "path": str(authorization_path.relative_to(root)),
            "sha256": sha256(authorization_path),
        },
        "implementation": {
            "path": "spatialgabench/g5_h5c_evaluation.py",
            "sha256": sha256(Path(__file__)),
        },
    }
    completion_path = output / "completion_evidence.json"
    _write_json(completion_path, completion)
    return completion_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        default="results/v1/g5/h5c_external_evaluation_preflight/completion_evidence.json",
    )
    parser.add_argument(
        "--authorization",
        default="governance/authorizations/G5_H5C_EXTERNAL_EVALUATION_AUTHORIZATION_2026-07-18.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    path = run(args.preflight, args.authorization, args.root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
