"""Non-overlapping NM-DS2 downstream experiments.

E1: failure / zero-output / metric-protocol sensitivity on frozen paired-RNA
depth results.
E2: simulation-truth downstream recovery on the frozen M3C panel.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import platform
import resource
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy import sparse, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from spatialgabench.g4_m3b_rna import normalize_total_log1p
from spatialgabench.software_wrappers.common import sha256_file


LOGGER = logging.getLogger("nm_ds2")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _safe_sha(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "scikit_learn": sklearn.__version__,
        "anndata": ad.__version__,
        "h5py": h5py.__version__,
    }


def _peak_memory_mb() -> dict[str, float]:
    self_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    child_kb = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return {
        "self_peak_rss_mb": self_kb / 1024.0,
        "children_peak_rss_mb": child_kb / 1024.0,
        "max_observed_peak_rss_mb": max(self_kb, child_kb) / 1024.0,
    }


def _load_config(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("task_id") != "NM-DS2":
        raise ValueError("Config is not an NM-DS2 config")
    prereg = root / str(config["preregistration"])
    if not prereg.exists() or sha256_file(prereg) != str(config["preregistration_sha256"]):
        raise ValueError("NM-DS2 pre-registration checksum mismatch")
    output = (root / str(config["output_root"])).resolve()
    allowed_root = (root / "results/v1_ds2").resolve()
    forbidden = {
        (root / "results/v1").resolve(),
        (root / "results/v1_ds1").resolve(),
        (root / "results/v1_risk1").resolve(),
        (root / "registries/v1").resolve(),
    }
    if output in forbidden or not (output == allowed_root or str(output).startswith(str(allowed_root) + "/")):
        raise PermissionError(f"Unsafe NM-DS2 output root: {output}")
    workers = int(config["execution"]["workers"])
    if not 1 <= workers <= 20:
        raise ValueError("NM-DS2 worker cap must be between 1 and 20")
    config["_path"] = str(path)
    config["_sha256"] = sha256_file(path)
    config["_output_root"] = str(output)
    return config


def _bh(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    output = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    selected = values[finite]
    order = np.argsort(selected, kind="mergesort")
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    tmp = np.empty_like(adjusted)
    tmp[order] = np.clip(adjusted, 0.0, 1.0)
    output[finite] = tmp
    return output


def _bootstrap_median_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan, math.nan
    median = float(np.median(values))
    if len(values) == 1:
        return median, median, median, math.nan
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        estimates.append(np.median(values[indices], axis=1))
    boot = np.concatenate(estimates)
    p_value = min(1.0, 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0))))
    return median, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), p_value


def _median(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else math.nan


def _check_e1_inputs(config: dict[str, Any], root: Path) -> None:
    for name, relative in config["e1"]["inputs"].items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing E1 input {name}: {path}")


def _e1_contract_table(config: dict[str, Any], root: Path) -> pd.DataFrame:
    e1 = config["e1"]
    metrics = pd.read_csv(root / e1["inputs"]["ta1_metrics"], sep="\t")
    genes = pd.read_csv(root / e1["inputs"]["ta1_gene_correlations"], sep="\t")
    completion = _json(root / e1["inputs"]["ta1_completion_evidence"])
    if len(metrics) != 273 or int(metrics.structural_success.sum()) != 236:
        raise ValueError("TA1 planned/successful contract count changed")
    if (
        completion.get("status") != "pass"
        or completion.get("planned_contracts") != 273
        or completion.get("evaluated_successful_contracts") != 236
        or completion.get("retained_structural_failures") != 37
    ):
        raise ValueError("TA1 completion evidence is not accepted")
    included = set(map(str, e1["included_study_families"]))
    excluded = set(map(str, e1["excluded_study_families"]))
    if set(metrics.study_family) & excluded:
        LOGGER.info("Excluding RISK1-owned study families: %s", sorted(set(metrics.study_family) & excluded))
    metrics = metrics[metrics.study_family.isin(included)].copy()
    if metrics.empty:
        raise ValueError("E1 included-study filter removed every contract")
    metrics["structural_success"] = metrics["structural_success"].astype(bool)
    metrics["depth_fraction"] = pd.to_numeric(metrics["depth_fraction"], errors="coerce")
    metrics["seed"] = pd.to_numeric(metrics["seed"], errors="coerce").astype(int)
    metrics[e1["frozen_primary_column"]] = pd.to_numeric(metrics[e1["frozen_primary_column"]], errors="coerce")
    success_mask = metrics["structural_success"]
    if not np.isfinite(metrics.loc[success_mask, e1["frozen_primary_column"]]).all():
        raise ValueError("Successful TA1 contracts have a non-finite frozen endpoint")

    gene_keys = ["dataset_id", "depth_fraction", "seed", "method_id", "representation_id"]
    genes = genes[genes.dataset_id.isin(set(metrics.dataset_id))].copy()
    genes["depth_fraction"] = pd.to_numeric(genes["depth_fraction"], errors="coerce")
    genes["seed"] = pd.to_numeric(genes["seed"], errors="coerce").astype(int)
    if genes.empty:
        raise ValueError("E1 gene-level correlation table has no included-study rows")
    genes["activity_gene_variable"] = genes["activity_gene_variable"].astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if genes["activity_gene_variable"].isna().any():
        raise ValueError("E1 gene-level variable flag is not boolean")
    variable_genes = genes[genes["activity_gene_variable"]].copy()
    defined = variable_genes.groupby(gene_keys, sort=True).agg(
        defined_gene_count=("spearman", lambda series: int(np.isfinite(pd.to_numeric(series, errors="coerce")).sum())),
        defined_only_median=("spearman", lambda series: _median(pd.to_numeric(series, errors="coerce"))),
    ).reset_index()
    contract = metrics.merge(defined, on=gene_keys, how="left", validate="one_to_one")
    contract["defined_gene_count"] = contract["defined_gene_count"].fillna(0).astype(int)

    frozen = contract[e1["frozen_primary_column"]]
    success = contract["structural_success"].to_numpy(bool)
    contract["planned_failures_scored_zero"] = np.where(success, frozen, float(e1["failure_value"]))
    contract["successes_only"] = np.where(success, frozen, np.nan)
    contract["undefined_correlations_scored_zero"] = np.where(success, frozen, np.nan)
    contract["defined_correlations_only"] = np.where(success, contract["defined_only_median"], np.nan)
    contract["zero_output_observations_direct"] = pd.to_numeric(contract["zero_output_observations_direct"], errors="coerce")
    contract["defined_gene_fraction_of_variable"] = (
        contract["defined_gene_count"] / pd.to_numeric(contract["activity_variable_rna_eligible_genes"], errors="coerce")
    )
    keep = [
        "condition_id", "dataset_id", "study_id", "study_family", "species", "genome_build",
        "depth_fraction", "seed", "method_id", "representation_id", "structural_success",
        "failure_type", "evaluation_status", "zero_output_observations_direct",
        "activity_variable_rna_eligible_genes", "defined_gene_count",
        "defined_gene_fraction_of_variable", "defined_only_median",
        e1["frozen_primary_column"], "planned_failures_scored_zero", "successes_only",
        "undefined_correlations_scored_zero", "defined_correlations_only",
    ]
    return contract[keep].copy()


def _e1_group_summary(config: dict[str, Any], contract: pd.DataFrame) -> pd.DataFrame:
    e1 = config["e1"]
    materiality = float(config["execution"]["materiality"])
    rows: list[dict[str, Any]] = []
    for keys, frame in contract.groupby(e1["group_keys"], sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        med_a = _median(frame["planned_failures_scored_zero"])
        med_b = _median(frame["successes_only"])
        med_c = _median(frame["undefined_correlations_scored_zero"])
        med_d = _median(frame["defined_correlations_only"])
        delta_ab = med_a - med_b if np.isfinite(med_a) and np.isfinite(med_b) else math.nan
        delta_cd = med_c - med_d if np.isfinite(med_c) and np.isfinite(med_d) else math.nan
        row = dict(zip(e1["group_keys"], keys))
        row.update({
            "n_planned": int(len(frame)),
            "n_success": int(frame.structural_success.sum()),
            "n_failure": int((~frame.structural_success).sum()),
            "failure_rate": float((~frame.structural_success).mean()),
            "median_planned_failures_scored_zero": med_a,
            "median_successes_only": med_b,
            "median_undefined_correlations_scored_zero": med_c,
            "median_defined_correlations_only": med_d,
            "delta_A_minus_B": delta_ab,
            "delta_C_minus_D": delta_cd,
            "abs_delta_A_minus_B": abs(delta_ab) if np.isfinite(delta_ab) else math.nan,
            "abs_delta_C_minus_D": abs(delta_cd) if np.isfinite(delta_cd) else math.nan,
            "failure_handling_decision": "MATERIAL" if np.isfinite(delta_ab) and abs(delta_ab) >= materiality else "ROBUST",
            "metric_protocol_decision": "MATERIAL" if np.isfinite(delta_cd) and abs(delta_cd) >= materiality else "ROBUST",
            "median_zero_output_observations": _median(frame["zero_output_observations_direct"]),
            "median_defined_gene_fraction": _median(frame["defined_gene_fraction_of_variable"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _e1_reversals(config: dict[str, Any], contract: pd.DataFrame) -> pd.DataFrame:
    e1 = config["e1"]
    rows: list[dict[str, Any]] = []
    columns = ["planned_failures_scored_zero", "successes_only", "undefined_correlations_scored_zero", "defined_correlations_only"]
    rules = [
        ("A_planned_vs_B_success", "planned_failures_scored_zero", "successes_only"),
        ("C_zero_filled_vs_D_defined", "undefined_correlations_scored_zero", "defined_correlations_only"),
    ]
    for keys, frame in contract.groupby(e1["ordering_group_keys"], sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        method_medians: dict[str, dict[str, float]] = {}
        for method, method_frame in frame.groupby("method_id", sort=True):
            method_medians[str(method)] = {column: _median(method_frame[column]) for column in columns}
        methods = sorted(method_medians)
        for rule_name, left_column, right_column in rules:
            for left, right in itertools.combinations(methods, 2):
                lv = method_medians[left]
                rv = method_medians[right]
                if not all(np.isfinite([lv[left_column], rv[left_column], lv[right_column], rv[right_column]])):
                    continue
                left_high = lv[left_column] > rv[left_column]
                right_high = lv[right_column] > rv[left_column]
                if left_high == right_high:
                    continue
                row = dict(zip(e1["ordering_group_keys"], keys))
                row.update({
                    "rule": rule_name,
                    "left_method_id": left,
                    "right_method_id": right,
                    "left_minus_right_left_rule": lv[left_column] - rv[left_column],
                    "left_minus_right_right_rule": lv[right_column] - rv[right_column],
                    "direction_reversal": True,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def run_e1(config: dict[str, Any], root: Path, output: Path) -> list[Path]:
    _check_e1_inputs(config, root)
    e1_output = output / config["e1"]["output_subdir"]
    e1_output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("E1: reading TA1 contract and gene-level tables")
    contract = _e1_contract_table(config, root)
    groups = _e1_group_summary(config, contract)
    reversals = _e1_reversals(config, contract)
    paths = {
        "contracts": e1_output / "e1_contract_handling_values.tsv",
        "groups": e1_output / "e1_group_summary.tsv",
        "reversals": e1_output / "e1_method_pair_reversals.tsv",
    }
    contract.to_csv(paths["contracts"], sep="\t", index=False)
    groups.to_csv(paths["groups"], sep="\t", index=False)
    reversals.to_csv(paths["reversals"], sep="\t", index=False)

    materiality = float(config["execution"]["materiality"])
    ab_material = groups[groups["abs_delta_A_minus_B"] >= materiality]
    cd_material = groups[groups["abs_delta_C_minus_D"] >= materiality]
    study_rows = []
    for study, frame in contract.groupby("study_family", sort=True):
        study_rows.append({
            "study_family": study,
            "planned": int(len(frame)),
            "successful": int(frame.structural_success.sum()),
            "failures": int((~frame.structural_success).sum()),
            "failure_rate": float((~frame.structural_success).mean()),
            "depths": sorted(float(value) for value in frame.depth_fraction.unique()),
            "methods": int(frame.method_id.nunique()),
            "representations": int(frame.representation_id.nunique()),
        })
    identity_check = {
        "B_C_values_identical": bool(np.allclose(
            contract.loc[contract.structural_success, "successes_only"].to_numpy(float),
            contract.loc[contract.structural_success, "undefined_correlations_scored_zero"].to_numpy(float),
            equal_nan=True,
        )),
        "B_C_row_universe_identical": bool(contract["successes_only"].notna().equals(contract["undefined_correlations_scored_zero"].notna())),
    }
    key = {
        "status": "pass",
        "included_study_families": list(config["e1"]["included_study_families"]),
        "excluded_study_families": list(config["e1"]["excluded_study_families"]),
        "study_rows": study_rows,
        "groups": int(len(groups)),
        "materiality": materiality,
        "failure_handling_material_groups": int(len(ab_material)),
        "metric_protocol_material_groups": int(len(cd_material)),
        "failure_handling_status": "MATERIAL" if len(ab_material) else "ROBUST",
        "metric_protocol_status": "MATERIAL" if len(cd_material) else "ROBUST",
        "maximum_abs_delta_A_minus_B": float(groups["abs_delta_A_minus_B"].max()),
        "maximum_abs_delta_C_minus_D": float(groups["abs_delta_C_minus_D"].max()),
        "method_pair_reversals": int(len(reversals)),
        "ordering_status": "ORDERING_UNSTABLE" if len(reversals) else "ORDERING_STABLE",
        "B_C_identity_check": identity_check,
        "zero_output_metric": "zero_output_observations_direct retained as a diagnostic; no spot-level recomputation",
    }
    key_path = e1_output / "e1_key_numbers.json"
    _write_json(key_path, key)
    paths["key_numbers"] = key_path
    LOGGER.info("E1 complete: %s groups, %s reversals, A-B material=%s, C-D material=%s", len(groups), len(reversals), len(ab_material), len(cd_material))
    return list(paths.values())


def _knn_graph(coordinates: np.ndarray, neighbors: int) -> sparse.csr_matrix:
    coordinates = np.asarray(coordinates, dtype=float)
    n = coordinates.shape[0]
    if n < 2:
        raise ValueError("Cannot construct a spatial graph from fewer than two observations")
    effective = min(int(neighbors), n - 1)
    model = NearestNeighbors(n_neighbors=effective + 1).fit(coordinates)
    _, indices = model.kneighbors(coordinates)
    row = np.repeat(np.arange(n), effective)
    col = indices[:, 1:].reshape(-1)
    data = np.full(len(row), 1.0 / effective, dtype=float)
    graph = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    graph = (graph + graph.T) * 0.5
    return graph.tocsr()


def _moran_matrix(values: np.ndarray, graph: sparse.csr_matrix) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    denominator = np.sum(centered * centered, axis=0)
    numerator = np.sum(centered * (graph @ centered), axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=float)
    finite = denominator > 0
    result[finite] = numerator[finite] / denominator[finite]
    return result


def _rank_order(values: np.ndarray, gene_ids: list[str]) -> list[int]:
    def key(index: int) -> tuple[int, float, str]:
        value = float(values[index])
        return (0, -value, gene_ids[index]) if np.isfinite(value) else (1, 0.0, gene_ids[index])

    return sorted(range(len(values)), key=key)


def _domain_metrics(values: np.ndarray, labels: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float, str]:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape[0] < 3 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        return math.nan, math.nan, "invalid_matrix"
    if np.var(matrix, axis=0).max() <= 0:
        return math.nan, math.nan, "zero_output"
    n_components = min(int(cfg["pca_components"]), matrix.shape[0] - 1, matrix.shape[1] - 1)
    if n_components < 1:
        return math.nan, math.nan, "invalid_pca"
    try:
        embedded = PCA(n_components=n_components, random_state=int(cfg["seed"]), svd_solver="full").fit_transform(matrix)
        predicted = KMeans(
            n_clusters=int(cfg["n_clusters"]),
            n_init=int(cfg["kmeans_n_init"]),
            random_state=int(cfg["seed"]),
        ).fit_predict(embedded)
    except Exception:
        return math.nan, math.nan, "clustering_failure"
    return float(adjusted_rand_score(labels, predicted)), float(normalized_mutual_info_score(labels, predicted)), "pass"


def _svg_metrics(
    predicted: np.ndarray,
    truth_program: np.ndarray,
    graph: sparse.csr_matrix,
    gene_ids: list[str],
    top_fraction: float,
) -> dict[str, float | str]:
    pred_moran = _moran_matrix(predicted, graph)
    truth_moran = _moran_matrix(truth_program, graph)
    finite = np.isfinite(pred_moran) & np.isfinite(truth_moran)
    if finite.sum() < 2:
        return {
            "svg_precision_at_10": math.nan,
            "svg_recall_at_10": math.nan,
            "svg_jaccard_at_10": math.nan,
            "svg_moran_rank_spearman": math.nan,
            "svg_status": "insufficient_finite_moran",
        }
    k = max(1, int(round(float(top_fraction) * len(gene_ids))))
    pred_order = _rank_order(pred_moran, gene_ids)[:k]
    truth_order = _rank_order(truth_moran, gene_ids)[:k]
    pred_set, truth_set = set(pred_order), set(truth_order)
    overlap = len(pred_set & truth_set)
    precision = overlap / k
    recall = overlap / k
    jaccard = overlap / max(1, len(pred_set | truth_set))
    rho = float(stats.spearmanr(pred_moran[finite], truth_moran[finite]).statistic)
    return {
        "svg_precision_at_10": precision,
        "svg_recall_at_10": recall,
        "svg_jaccard_at_10": jaccard,
        "svg_moran_rank_spearman": rho,
        "svg_status": "pass",
    }


def _e2_task(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sim_id = str(payload["simulation_id"])
    simulation_root = Path(payload["simulation_root"])
    construction_root = Path(payload["construction_root"])
    simulation_dir = simulation_root / sim_id
    truth_obj = ad.read_h5ad(simulation_dir / "true_gene_activity.h5ad", backed="r")
    obs_names = truth_obj.obs_names.astype(str).tolist()
    gene_ids = truth_obj.var_names.astype(str).tolist()
    coordinates = np.asarray(truth_obj.obsm["spatial"], dtype=float)
    truth_obj.file.close()
    program_path = simulation_dir / "true_domain_program_component.h5ad"
    program_obj = ad.read_h5ad(program_path)
    if program_obj.obs_names.astype(str).tolist() != obs_names:
        raise ValueError(f"Truth program observation order mismatch: {sim_id}")
    if program_obj.var_names.astype(str).tolist() != gene_ids:
        raise ValueError(f"Truth program gene order mismatch: {sim_id}")
    truth_program = program_obj.X.toarray() if sparse.issparse(program_obj.X) else np.asarray(program_obj.X)
    truth_program = np.asarray(truth_program, dtype=float)
    label_path = simulation_dir / "true_domain_labels.tsv"
    labels_frame = pd.read_csv(label_path, sep="\t", index_col=0)
    labels_frame.index = labels_frame.index.astype(str)
    labels = labels_frame.reindex(obs_names)["true_domain_index"].to_numpy(int)
    graph = _knn_graph(coordinates, int(payload["svg_neighbors"]))
    truth_program_sha = sha256_file(program_path)
    label_sha = sha256_file(label_path)
    output_rows: list[dict[str, Any]] = []
    for method in payload["methods"]:
        prediction_path = construction_root / sim_id / method / "gene_activity.h5ad"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing E2 method output: {prediction_path}")
        prediction_obj = ad.read_h5ad(prediction_path)
        if prediction_obj.obs_names.astype(str).tolist() != obs_names:
            raise ValueError(f"Prediction observation order mismatch: {sim_id}/{method}")
        if prediction_obj.var_names.astype(str).tolist() != gene_ids:
            raise ValueError(f"Prediction gene order mismatch: {sim_id}/{method}")
        prediction_sparse = prediction_obj.X.tocsr() if sparse.issparse(prediction_obj.X) else sparse.csr_matrix(prediction_obj.X)
        normalized = normalize_total_log1p(prediction_sparse, float(payload["target_sum"])).toarray()
        normalized = np.asarray(normalized, dtype=float)
        ari, nmi, domain_status = _domain_metrics(normalized, labels, payload["domain"])
        svg = _svg_metrics(normalized, truth_program, graph, gene_ids, float(payload["svg_top_fraction"]))
        output_rows.append({
            "simulation_id": sim_id,
            "method_id": method,
            "n_observations": int(normalized.shape[0]),
            "n_truth_genes": int(normalized.shape[1]),
            "domain_ari": ari,
            "domain_nmi": nmi,
            "domain_status": domain_status,
            **svg,
            "prediction_sha256": sha256_file(prediction_path),
            "truth_program_sha256": truth_program_sha,
            "truth_label_sha256": label_sha,
        })
    return output_rows


def _e2_preflight(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    e2 = config["e2"]
    metrics = pd.read_csv(root / e2["inputs"]["metrics"], sep="\t")
    design = pd.read_csv(root / e2["inputs"]["design"], sep="\t")
    if len(metrics) != int(e2["expected"]["method_rows"]):
        raise ValueError(f"E2 metric row count {len(metrics)} != expected {e2['expected']['method_rows']}")
    if metrics.simulation_id.nunique() != int(e2["expected"]["simulations"]):
        raise ValueError("E2 simulation count mismatch")
    if metrics.method_id.nunique() != int(e2["expected"]["methods"]):
        raise ValueError("E2 method count mismatch")
    if set(metrics.simulation_type) != set(e2["strata"]):
        raise ValueError("E2 strata mismatch")
    if set(metrics.method_id) != set(e2["methods"]):
        raise ValueError("E2 configured methods do not match the frozen panel")
    if not (metrics["status"] == "pass").all():
        raise ValueError("E2 frozen metric table contains non-pass status rows")
    for flag in ("truth_used_for_construction", "rna_used_for_construction", "labels_used_for_construction", "holdout_accessed"):
        values = metrics[flag]
        if values.dtype == object:
            values = values.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if bool(values.fillna(True).astype(bool).any()):
            raise ValueError(f"E2 leakage sentinel failed for {flag}")
    expected_ids = set(design.simulation_id.astype(str))
    if expected_ids != set(metrics.simulation_id.astype(str)):
        raise ValueError("E2 design and metric simulation identifiers differ")
    simulation_root = root / e2["inputs"]["simulation_root"]
    construction_root = root / e2["inputs"]["construction_root"]
    for sim_id in sorted(expected_ids):
        for name in ("true_gene_activity.h5ad", "true_domain_program_component.h5ad", "true_domain_labels.tsv"):
            if not (simulation_root / sim_id / name).exists():
                raise FileNotFoundError(f"Missing E2 truth file: {simulation_root / sim_id / name}")
        for method in e2["methods"]:
            if not (construction_root / sim_id / method / "gene_activity.h5ad").exists():
                raise FileNotFoundError(f"Missing E2 method output: {construction_root / sim_id / method / 'gene_activity.h5ad'}")
    return metrics, design, sorted(expected_ids)


def _e2_new_metrics(config: dict[str, Any], root: Path, metric_table: pd.DataFrame, sim_ids: list[str]) -> pd.DataFrame:
    e2 = config["e2"]
    base = {
        "simulation_root": str(root / e2["inputs"]["simulation_root"]),
        "construction_root": str(root / e2["inputs"]["construction_root"]),
        "methods": list(e2["methods"]),
        "target_sum": float(e2["normalization"]["target_sum"]),
        "domain": e2["domain"],
        "svg_neighbors": int(e2["svg"]["neighbors"]),
        "svg_top_fraction": float(e2["svg"]["top_fraction"]),
    }
    rows: list[dict[str, Any]] = []
    workers = int(config["execution"]["workers"])
    LOGGER.info("E2: computing downstream metrics for %s simulations with %s workers", len(sim_ids), workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_e2_task, {**base, "simulation_id": sim_id}): sim_id for sim_id in sim_ids}
        completed = 0
        for future in as_completed(futures):
            sim_id = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"E2 task failed for {sim_id}") from exc
            completed += 1
            if completed % 50 == 0 or completed == len(sim_ids):
                LOGGER.info("E2 task progress: %s/%s", completed, len(sim_ids))
    computed = pd.DataFrame(rows)
    if len(computed) != int(e2["expected"]["method_rows"]):
        raise ValueError(f"E2 computed row count {len(computed)} != expected {e2['expected']['method_rows']}")
    metric_map = dict(e2["existing_metric_map"])
    frozen = metric_table[[
        "simulation_id", "simulation_type", "physical_random_seed", "replicate_seed_index", "method_id",
        "distal_fraction", "depth_multiplier", "zero_inflation_rate", "domain_log_effect",
        "decoy_fraction", "rna_latent_correlation", "active_link_prevalence",
        *metric_map.values(),
    ]].copy()
    frozen = frozen.rename(columns={source: target for target, source in metric_map.items()})
    merged = frozen.merge(computed, on=["simulation_id", "method_id"], how="inner", validate="one_to_one")
    if len(merged) != len(frozen) or len(merged) != len(computed):
        raise ValueError("E2 frozen/computed metric merge lost rows")
    merged["svg_top_k"] = int(round(float(e2["svg"]["top_fraction"]) * merged["n_truth_genes"].iloc[0]))
    return merged


def _factorial_terms(factors: list[str]) -> list[tuple[str, ...]]:
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
        raise ValueError(f"Metric {metric} contains non-finite values in factor subset")
    rows = []
    for term in _factorial_terms(factors):
        contrast = np.prod(np.column_stack([codes[factor] for factor in term]), axis=1)
        coefficient = float(np.mean(y * contrast))
        multiplier = 2.0 if len(term) == 1 else 4.0
        rows.append({
            "term": ":".join(term),
            "term_order": len(term),
            "coded_coefficient": coefficient,
            "effect_scale_multiplier": multiplier,
            "effect_contrast": coefficient * multiplier,
        })
    return rows


def _seed_summary(values: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
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
        "n_seeds": n,
        "mean": mean,
        "seed_sd": sd,
        "seed_se": se,
        "ci_low": mean - tcrit * se if n > 1 else math.nan,
        "ci_high": mean + tcrit * se if n > 1 else math.nan,
        "p_value": p_value,
    }


def _e2_factor_effects(config: dict[str, Any], metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    e2 = config["e2"]
    factors = list(e2["factors"])
    endpoint_columns = list(e2["endpoint_columns"])
    seed_rows: list[dict[str, Any]] = []
    for (sim_type, method, seed), frame in metrics.groupby(["simulation_type", "method_id", "physical_random_seed"], sort=True):
        for endpoint in endpoint_columns:
            for record in _seed_coefficients(frame, factors, endpoint):
                seed_rows.append({
                    "simulation_type": sim_type,
                    "method_id": method,
                    "physical_random_seed": int(seed),
                    "endpoint": endpoint,
                    **record,
                })
    seed_effects = pd.DataFrame(seed_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for keys, frame in seed_effects.groupby(
        ["simulation_type", "method_id", "endpoint", "term", "term_order", "effect_scale_multiplier"], sort=True
    ):
        summary = _seed_summary(frame.effect_contrast.to_numpy(float), 0.95)
        aggregate_rows.append(dict(zip(
            ["simulation_type", "method_id", "endpoint", "term", "term_order", "effect_scale_multiplier"], keys
        )) | summary)
    effects = pd.DataFrame(aggregate_rows)
    effects["effect_contrast"] = effects["mean"]
    effects["p_adjust_bh_within_stratum_endpoint"] = np.nan
    for _, indices in effects.groupby(["simulation_type", "endpoint"], sort=True).groups.items():
        effects.loc[indices, "p_adjust_bh_within_stratum_endpoint"] = _bh(effects.loc[indices, "p_value"].to_numpy(float))
    effects["material"] = (
        (effects["p_adjust_bh_within_stratum_endpoint"] < float(config["execution"]["bh_q"]))
        & (effects["effect_contrast"].abs() >= float(config["execution"]["materiality"]))
    )
    return effects, seed_effects


def _e2_control_deltas(config: dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    e2 = config["e2"]
    endpoints = list(e2["endpoint_columns"])
    controls = set(e2["controls"].values())
    methods = [method for method in e2["methods"] if method not in controls]
    rows: list[dict[str, Any]] = []
    draws = int(config["execution"]["bootstrap_draws"])
    base_seed = int(config["execution"]["bootstrap_seed"])
    for control_name in e2["controls"].values():
        control = metrics[metrics.method_id == control_name][["simulation_id", "simulation_type", *endpoints]].copy()
        control = control.rename(columns={endpoint: f"{endpoint}_control" for endpoint in endpoints})
        for method in methods:
            method_frame = metrics[metrics.method_id == method][["simulation_id", "simulation_type", *endpoints]].copy()
            paired_all = method_frame.merge(
                control, on=["simulation_id", "simulation_type"], how="inner", validate="one_to_one"
            )
            if paired_all.empty:
                continue
            for simulation_type, paired in paired_all.groupby("simulation_type", sort=True):
                for endpoint in endpoints:
                    delta = paired[endpoint].to_numpy(float) - paired[f"{endpoint}_control"].to_numpy(float)
                    median, low, high, p_value = _bootstrap_median_ci(delta, draws, base_seed + len(rows))
                    rows.append({
                        "simulation_type": str(simulation_type),
                        "method_id": method,
                        "control_id": control_name,
                        "endpoint": endpoint,
                        "n_pairs": int(np.isfinite(delta).sum()),
                        "median_delta": median,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "bootstrap_p_two_sided": p_value,
                    })
    output = pd.DataFrame(rows)
    output["p_adjust_bh_within_stratum_endpoint_control"] = np.nan
    for _, indices in output.groupby(["simulation_type", "endpoint", "control_id"], sort=True).groups.items():
        output.loc[indices, "p_adjust_bh_within_stratum_endpoint_control"] = _bh(output.loc[indices, "bootstrap_p_two_sided"].to_numpy(float))
    margin = float(config["execution"]["materiality"])
    output["decision"] = "NULL"
    significant = output["p_adjust_bh_within_stratum_endpoint_control"] < float(config["execution"]["bh_q"])
    positive = (output["median_delta"] >= margin) & (output["bootstrap_ci_low"] > 0) & significant
    negative = (output["median_delta"] <= -margin) & (output["bootstrap_ci_high"] < 0) & significant
    output.loc[positive, "decision"] = "POSITIVE"
    output.loc[negative, "decision"] = "NEGATIVE"
    return output


def _e2_endpoint_summary(config: dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    e2 = config["e2"]
    endpoints = list(e2["endpoint_columns"])
    methods = list(e2["methods"])
    controls = set(e2["controls"].values())
    rows: list[dict[str, Any]] = []
    for sim_type in e2["strata"]:
        stratum = metrics[metrics.simulation_type == sim_type]
        method_medians = {
            method: {endpoint: _median(stratum[stratum.method_id == method][endpoint]) for endpoint in endpoints}
            for method in methods
        }
        for method in methods:
            for endpoint in endpoints:
                values = stratum[stratum.method_id == method][endpoint].to_numpy(float)
                finite_values = values[np.isfinite(values)]
                med = method_medians[method][endpoint]
                control_medians = [method_medians[control][endpoint] for control in controls if np.isfinite(method_medians[control][endpoint])]
                control_floor = max(control_medians) if control_medians else math.nan
                oracle = 1.0
                headroom = oracle - med if np.isfinite(med) else math.nan
                denominator = oracle - control_floor
                normalized_utility = (med - control_floor) / denominator if np.isfinite(denominator) and denominator > 0.05 else math.nan
                finite_method_medians = [method_medians[item][endpoint] for item in methods if np.isfinite(method_medians[item][endpoint])]
                between_method_range = max(finite_method_medians) - min(finite_method_medians) if len(finite_method_medians) > 1 else math.nan
                rows.append({
                    "simulation_type": sim_type,
                    "method_id": method,
                    "endpoint": endpoint,
                    "n_simulations": int(len(stratum[stratum.method_id == method])),
                    "median": med,
                    "mean": float(finite_values.mean()) if len(finite_values) else math.nan,
                    "minimum": float(finite_values.min()) if len(finite_values) else math.nan,
                    "maximum": float(finite_values.max()) if len(finite_values) else math.nan,
                    "between_method_range": between_method_range,
                    "control_floor_median": control_floor,
                    "oracle": oracle,
                    "headroom_to_oracle": headroom,
                    "normalized_utility_vs_best_control": normalized_utility,
                    "ceiling_saturated": bool(np.isfinite(med) and med >= 0.90 and np.isfinite(between_method_range) and between_method_range < 0.02),
                    "high_control_floor": bool(np.isfinite(control_floor) and control_floor > 0.20),
                })
    return pd.DataFrame(rows)


def run_e2(config: dict[str, Any], root: Path, output: Path) -> list[Path]:
    e2_output = output / config["e2"]["output_subdir"]
    e2_output.mkdir(parents=True, exist_ok=True)
    metrics, design, sim_ids = _e2_preflight(config, root)
    computed = _e2_new_metrics(config, root, metrics, sim_ids)
    paths: dict[str, Path] = {
        "metrics": e2_output / "e2_simulation_downstream_metrics.tsv",
        "input_checksums": e2_output / "e2_input_checksums.tsv",
    }
    computed.to_csv(paths["metrics"], sep="\t", index=False)
    computed[["simulation_id", "method_id", "prediction_sha256", "truth_program_sha256", "truth_label_sha256"]].to_csv(paths["input_checksums"], sep="\t", index=False)
    summary = _e2_endpoint_summary(config, computed)
    paths["endpoint_summary"] = e2_output / "e2_endpoint_summary.tsv"
    summary.to_csv(paths["endpoint_summary"], sep="\t", index=False)
    deltas = _e2_control_deltas(config, computed)
    paths["control_deltas"] = e2_output / "e2_paired_control_deltas.tsv"
    deltas.to_csv(paths["control_deltas"], sep="\t", index=False)
    effects, seed_effects = _e2_factor_effects(config, computed)
    paths["factor_effects"] = e2_output / "e2_factor_main_and_two_way_effects.tsv"
    paths["seed_factor_effects"] = e2_output / "e2_seed_level_factor_coefficients.tsv.gz"
    effects.to_csv(paths["factor_effects"], sep="\t", index=False)
    seed_effects.to_csv(paths["seed_factor_effects"], sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    infeasible = pd.DataFrame([{
        "endpoint": "tf_motif_recovery",
        "status": "INFEASIBLE",
        "reason": "No method-accessible TF/motif prediction matrix was constructed; deriving one from true_tf_motif_links.tsv or truth matrices would inject evaluation truth into prediction.",
    }])
    paths["infeasible"] = e2_output / "e2_infeasible_endpoints.tsv"
    infeasible.to_csv(paths["infeasible"], sep="\t", index=False)

    depth_terms = effects[(effects.term == "depth_multiplier") & effects.material]
    domain_terms = effects[
        effects.term.isin(["domain_log_effect"]) & effects.material
        & effects.endpoint.isin(["domain_ari", "domain_nmi", "marker_domain_auroc"])
    ]
    positive_deltas = deltas[deltas.decision == "POSITIVE"]
    negative_deltas = deltas[deltas.decision == "NEGATIVE"]
    key = {
        "status": "pass",
        "simulation_rows": int(len(computed)),
        "simulations": int(computed.simulation_id.nunique()),
        "strata": list(config["e2"]["strata"]),
        "methods": int(computed.method_id.nunique()),
        "endpoint_columns": list(config["e2"]["endpoint_columns"]),
        "materiality": float(config["execution"]["materiality"]),
        "positive_matched_control_deltas": int(len(positive_deltas)),
        "negative_matched_control_deltas": int(len(negative_deltas)),
        "depth_material_effects": int(len(depth_terms)),
        "domain_material_effects": int(len(domain_terms)),
        "ceiling_saturated_rows": int(summary["ceiling_saturated"].sum()),
        "high_control_floor_rows": int(summary["high_control_floor"].sum()),
        "tf_motif_status": "INFEASIBLE",
        "domain_status_counts": computed.domain_status.value_counts().to_dict(),
        "svg_status_counts": computed.svg_status.value_counts().to_dict(),
        "pooling": False,
    }
    key_path = e2_output / "e2_key_numbers.json"
    _write_json(key_path, key)
    paths["key_numbers"] = key_path
    LOGGER.info("E2 complete: %s rows, positive deltas=%s, depth material effects=%s, saturated rows=%s", len(computed), len(positive_deltas), len(depth_terms), int(summary["ceiling_saturated"].sum()))
    return list(paths.values())


def _configure_logging(log_path: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NM-DS2 pre-registered downstream analyses")
    parser.add_argument("--config", default="configs/nm_ds2_downstream_2026-10-18.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--stage", choices=("all", "e1", "e2"), default="all")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    started = _utc()
    start_time = time.perf_counter()
    config = _load_config(args.config, root)
    output = Path(config["_output_root"])
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_id = started.replace(":", "").replace("-", "")
    log_path = logs / f"nm_ds2_{run_id}.log"
    _configure_logging(log_path)
    LOGGER.info("NM-DS2 start stage=%s config_sha256=%s", args.stage, config["_sha256"])
    artifacts: list[Path] = []
    status = "fail"
    try:
        if args.stage in ("all", "e1") and config["e1"].get("enabled", True):
            artifacts.extend(run_e1(config, root, output))
        if args.stage in ("all", "e2") and config["e2"].get("enabled", True):
            artifacts.extend(run_e2(config, root, output))
        status = "pass"
    except Exception as exc:
        failure_dir = output / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"nm_ds2_failure_{run_id}.json"
        _write_json(failure_path, {
            "timestamp_utc": _utc(),
            "status": "fail",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.perf_counter() - start_time,
            "memory": _peak_memory_mb(),
            "software_versions": _software_versions(),
            "log_file": str(log_path.relative_to(root)),
        })
        LOGGER.error("NM-DS2 failed: %s", exc)
        LOGGER.error(traceback.format_exc())
        raise
    finally:
        completed = _utc()
        runtime = time.perf_counter() - start_time
        artifact_hashes = {str(path.relative_to(root)): sha256_file(path) for path in artifacts if path.exists()}
        manifest = {
            "schema_version": 1,
            "task_id": "NM-DS2",
            "status": status,
            "stage": args.stage,
            "started_utc": started,
            "completed_utc": completed,
            "runtime_seconds": runtime,
            "memory": _peak_memory_mb(),
            "software_versions": _software_versions(),
            "config": {"path": str(Path(config["_path"]).resolve().relative_to(root)), "sha256": config["_sha256"]},
            "preregistration": {"path": config["preregistration"], "sha256": config["preregistration_sha256"]},
            "input_checksums": {
                name: {
                    "path": relative,
                    "kind": "directory" if (root / relative).is_dir() else "file",
                    "sha256": None if (root / relative).is_dir() else _safe_sha(root / relative),
                }
                for name, relative in {**config["e1"]["inputs"], **config["e2"]["inputs"]}.items()
            },
            "artifacts": artifact_hashes,
            "log_file": str(log_path.relative_to(root)),
            "failure_policy": "retained_under_results_v1_ds2/failures",
        }
        _write_json(output / "run_manifest.json", manifest)
        LOGGER.info("NM-DS2 status=%s runtime=%.1fs peak_memory=%sMB", manifest["status"], runtime, manifest["memory"]["max_observed_peak_rss_mb"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
