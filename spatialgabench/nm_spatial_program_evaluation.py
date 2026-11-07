"""Evaluate held-out RNA spatial-program preservation under the frozen NU4 plan."""

from __future__ import annotations

import argparse
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
import yaml
from scipy import sparse, stats
from scipy.io import mmread
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from spatialgabench.full_benchmark_gse279771 import _read_10x_rna_h5
from spatialgabench.g4_m3b_rna import _canonicalize_ga_columns, _gtf_annotations, normalize_total_log1p
from spatialgabench.g4_m3h_rna_replication import load_wide_tsv_sparse
from spatialgabench.g5_h5c_evaluation import _canonical_rna_gene_ids, load_wide_csv_sparse


CONTROL = "accessibility_matched_random_peak_to_gene_control_pc"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_biotypes(biotypes: dict[str, str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for gene_id, biotype in biotypes.items():
        key = re.sub(r"\.\d+$", "", gene_id)
        previous = canonical.get(key)
        if previous is not None and previous != biotype:
            raise ValueError(f"Conflicting biotypes after Ensembl version normalization: {key}")
        canonical[key] = biotype
    return canonical


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


def safe_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) != len(y) or len(x) < 2 or np.var(x) == 0 or np.var(y) == 0:
        return 0.0
    value = stats.spearmanr(x, y).statistic if kind == "spearman" else stats.pearsonr(x, y).statistic
    return float(value) if np.isfinite(value) else 0.0


def symmetric_knn_graph(coordinates: np.ndarray, neighbors: int) -> sparse.csr_matrix:
    count = len(coordinates)
    if count <= neighbors:
        raise ValueError("Not enough held-out observations for frozen kNN graph")
    indices = NearestNeighbors(n_neighbors=neighbors + 1).fit(coordinates).kneighbors(return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(count), neighbors)
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, indices.ravel())), shape=(count, count))
    graph = graph.maximum(graph.transpose()).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def morans_i(values: np.ndarray, graph: sparse.csr_matrix) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    denominator = float(centered @ centered)
    weight_sum = float(graph.sum())
    if denominator <= 0 or weight_sum <= 0:
        return 0.0
    return float(len(centered) / weight_sum * (centered @ (graph @ centered)) / denominator)


def _collapse_rna_genes(
    matrix: sparse.csr_matrix,
    identifiers: list[str | None],
    biotypes: dict[str, str],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    mapped = np.asarray([re.sub(r"\.\d+$", "", value) if value else None for value in identifiers], dtype=object)
    valid = np.asarray([value is not None and str(value) in biotypes for value in mapped], dtype=bool)
    canonical = np.asarray(sorted(set(mapped[valid].astype(str))), dtype=object)
    lookup = {value: index for index, value in enumerate(canonical)}
    source_columns = np.flatnonzero(valid)
    indicator = sparse.csr_matrix(
        (
            np.ones(len(source_columns), dtype=np.float64),
            (source_columns, np.asarray([lookup[str(mapped[index])] for index in source_columns], dtype=np.int64)),
        ),
        shape=(matrix.shape[1], len(canonical)),
    )
    collapsed = (matrix @ indicator).tocsr()
    protein_coding = np.asarray([biotypes.get(value) == "protein_coding" for value in canonical], dtype=bool)
    return collapsed, canonical, protein_coding


def _load_rna(sample: dict[str, Any], root: Path) -> tuple[sparse.csr_matrix, list[str], np.ndarray, np.ndarray, dict[str, str]]:
    annotation = root / sample["gene_annotation"]["path"]
    biotypes, unique_names = _gtf_annotations(annotation)
    biotypes = canonicalize_biotypes(biotypes)
    fmt = sample["rna_format"]
    path = root / sample["rna_matrix"]["path"]
    if fmt == "tenx_mtx":
        with gzip.open(path, "rb") as handle:
            genes_by_spots = mmread(handle).tocsr()
        features = pd.read_csv(root / sample["rna_features"]["path"], sep="\t", header=None, compression="gzip", dtype=str)
        identifiers = features.iloc[:, 0].astype(str).tolist()
        with gzip.open(root / sample["rna_barcodes"]["path"], "rt", encoding="utf-8") as handle:
            barcodes = [line.strip().split("\t", 1)[0] for line in handle if line.strip()]
        spots_by_genes = genes_by_spots.transpose().tocsr()
    elif fmt == "wide_tsv_gzip":
        genes_by_spots, names, barcodes = load_wide_tsv_sparse(path)
        identifiers = [unique_names.get(name) for name in names]
        spots_by_genes = genes_by_spots.transpose().tocsr()
    elif fmt == "wide_csv_gzip":
        genes_by_spots, names, barcodes = load_wide_csv_sparse(path)
        identifiers = _canonical_rna_gene_ids(names, biotypes, unique_names)
        spots_by_genes = genes_by_spots.transpose().tocsr()
    elif fmt == "tenx_h5":
        spots_by_genes, barcodes, var = _read_10x_rna_h5(path)
        identifiers = var["id"].astype(str).tolist()
    else:
        raise ValueError(f"Unsupported NU4 RNA format: {fmt}")
    collapsed, gene_ids, protein_coding = _collapse_rna_genes(spots_by_genes, identifiers, biotypes)
    if collapsed.shape[0] != len(barcodes):
        raise ValueError("NU4 RNA barcode dimension changed")
    return collapsed, barcodes, gene_ids, protein_coding, unique_names


def _align_rows(matrix: sparse.csr_matrix, source: list[str], target: list[str]) -> sparse.csr_matrix:
    lookup = {value: index for index, value in enumerate(source)}
    indices = np.asarray([lookup.get(value, -1) for value in target], dtype=np.int64)
    if np.any(indices < 0):
        raise ValueError("NU4 frozen evaluation barcodes are missing from RNA")
    return matrix[indices, :].tocsr()


def _descriptive_bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    medians = np.empty(draws, dtype=float)
    for draw in range(draws):
        medians[draw] = np.median(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _validate_authorization(root: Path, config_path: Path, preflight_path: Path, authorization_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config, preflight, authorization = _yaml(config_path), _json(preflight_path), _json(authorization_path)
    if preflight.get("status") != "pass" or preflight.get("evaluation_run") is not False:
        raise ValueError("NU4 preflight is not complete")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "NU4_heldout_RNA_spatial_program_preservation"
        or authorization.get("rna_expression_value_access_authorized") is not True
        or authorization.get("spatial_program_evaluation_authorized") is not True
        or authorization.get("method_rerun_authorized") is not False
        or authorization.get("construction_tuning_authorized") is not False
        or authorization.get("labels_clusters_pathology_access_authorized") is not False
        or authorization.get("policy_refit_authorized") is not False
        or authorization.get("universal_ranking_authorized") is not False
        or authorization.get("final_ranking_authorized") is not False
    ):
        raise PermissionError("NU4 evaluation authorization boundary is invalid")
    bindings = {
        "config": config_path,
        "preflight_evidence": preflight_path,
        "method_input_manifest": root / preflight["method_input_manifest"]["path"],
        "data_identity_manifest": root / preflight["data_identity_manifest"]["path"],
        "implementation_manifest": root / preflight["implementation_manifest"]["path"],
    }
    for name, path in bindings.items():
        record = authorization[name]
        if record.get("path") != str(path.relative_to(root)) or sha256_path(path) != record.get("sha256"):
            raise ValueError(f"NU4 evaluation authorization binding mismatch: {name}")
    for record in preflight["fold_assignments"].values():
        if sha256_path(root / record["path"]) != record["sha256"]:
            raise ValueError("NU4 fold assignment changed after preflight")
    implementation = pd.read_csv(root / preflight["implementation_manifest"]["path"], sep="\t")
    for row in implementation.itertuples(index=False):
        if sha256_path(root / row.path) != row.sha256:
            raise ValueError(f"NU4 implementation changed after preflight: {row.path}")
    return config, preflight


def run(config_path: str | Path, preflight_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    preflight_path, authorization_path = Path(preflight_path).resolve(), Path(authorization_path).resolve()
    config, preflight = _validate_authorization(root, config_path, preflight_path, authorization_path)
    methods = pd.read_csv(root / preflight["method_input_manifest"]["path"], sep="\t")
    detail_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []

    for dataset_index, (dataset_id, sample) in enumerate(config["samples"].items()):
        fold_table = pd.read_csv(root / preflight["fold_assignments"][dataset_id]["path"], sep="\t")
        evaluation_barcodes = fold_table.barcode.astype(str).tolist()
        rna_counts, rna_barcodes, rna_gene_ids, protein_coding, unique_names = _load_rna(sample, root)
        rna_counts = _align_rows(rna_counts, rna_barcodes, evaluation_barcodes)
        rna_normalized = normalize_total_log1p(rna_counts, float(config["rna_programs"]["target_sum"]))
        fold_models: list[dict[str, Any]] = []
        for fold in range(int(config["folds"]["count"])):
            test = np.flatnonzero(fold_table.fold.to_numpy() == fold)
            train = np.flatnonzero(fold_table.fold.to_numpy() != fold)
            detection = np.asarray((rna_counts[train, :] > 0).sum(axis=0)).ravel()
            threshold = math.ceil(float(config["rna_programs"]["minimum_detected_training_fraction"]) * len(train))
            eligible = np.flatnonzero(protein_coding & (detection >= threshold))
            if len(eligible) < int(config["rna_programs"]["minimum_eligible_training_genes"]):
                raise ValueError(f"NU4 has too few eligible training genes: {dataset_id}/fold{fold}")
            train_dense = rna_normalized[train, :][:, eligible].toarray().astype(np.float32, copy=False)
            test_dense = rna_normalized[test, :][:, eligible].toarray().astype(np.float32, copy=False)
            pca = PCA(
                n_components=int(config["rna_programs"]["components"]),
                svd_solver=str(config["rna_programs"]["svd_solver"]),
                random_state=int(config["rna_programs"]["random_seed"]),
            ).fit(train_dense)
            rna_scores = pca.transform(test_dense)
            graph = symmetric_knn_graph(
                fold_table.loc[test, ["row_coordinate", "column_coordinate"]].to_numpy(dtype=float),
                int(config["spatial_metric"]["neighbors"]),
            )
            rna_moran = np.asarray([morans_i(rna_scores[:, component], graph) for component in range(rna_scores.shape[1])])
            fold_models.append({
                "fold": fold,
                "test": test,
                "eligible": eligible,
                "eligible_gene_ids": rna_gene_ids[eligible],
                "components": pca.components_.astype(np.float64),
                "mean": pca.mean_.astype(np.float64),
                "rna_scores": rna_scores.astype(np.float64),
                "rna_moran": rna_moran,
                "graph": graph,
            })
            gene_rows.append({
                "dataset_id": dataset_id,
                "study_id": sample["study_id"],
                "fold": fold,
                "training_observations": len(train),
                "test_observations": len(test),
                "eligible_training_genes": len(eligible),
                "components": int(config["rna_programs"]["components"]),
            })

        dataset_methods = methods.loc[methods.dataset_id.eq(dataset_id)]
        for method_index, method in enumerate(dataset_methods.itertuples(index=False)):
            path = root / method.gene_activity_h5ad
            if sha256_path(path) != method.gene_activity_sha256:
                raise ValueError(f"NU4 method output changed after preflight: {path}")
            activity = ad.read_h5ad(path)
            selected = activity.obs_names.astype(str).get_indexer(evaluation_barcodes)
            if np.any(selected < 0) or len(set(selected)) != len(selected):
                raise ValueError(f"NU4 activity observations do not match: {dataset_id}/{method.method_id}")
            raw = activity.X.tocsr() if sparse.issparse(activity.X) else sparse.csr_matrix(activity.X)
            normalized = normalize_total_log1p(raw[selected, :], float(config["rna_programs"]["target_sum"]))
            canonical, activity_gene_ids, _ = _canonicalize_ga_columns(normalized, activity, unique_names)
            activity_index = {gene_id: index for index, gene_id in enumerate(activity_gene_ids)}
            for model in fold_models:
                mapped = np.asarray([activity_index.get(str(gene_id), -1) for gene_id in model["eligible_gene_ids"]], dtype=np.int64)
                present = np.flatnonzero(mapped >= 0)
                test = model["test"]
                ga_scores = np.zeros((len(test), int(config["rna_programs"]["components"])), dtype=np.float64)
                if len(present):
                    ga_scores = canonical[test, :][:, mapped[present]] @ model["components"][:, present].transpose()
                    ga_scores = np.asarray(ga_scores)
                ga_scores -= model["mean"] @ model["components"].transpose()
                for component in range(int(config["rna_programs"]["components"])):
                    rna_score = model["rna_scores"][:, component]
                    ga_score = ga_scores[:, component]
                    ga_moran = morans_i(ga_score, model["graph"])
                    rna_moran = float(model["rna_moran"][component])
                    detail_rows.append({
                        "dataset_id": dataset_id,
                        "study_id": method.study_id,
                        "study_family": method.study_family,
                        "species": method.species,
                        "genome_build": method.genome_build,
                        "representation_id": method.representation_id,
                        "input_track": method.input_track,
                        "method_id": method.method_id,
                        "method_category": method.method_category,
                        "fold": model["fold"],
                        "component": component + 1,
                        "test_observations": len(test),
                        "eligible_training_genes": len(model["eligible"]),
                        "activity_present_gene_fraction": len(present) / len(model["eligible"]),
                        "heldout_component_spearman": safe_correlation(rna_score, ga_score, "spearman"),
                        "heldout_component_pearson": safe_correlation(rna_score, ga_score, "pearson"),
                        "rna_morans_i": rna_moran,
                        "activity_morans_i": ga_moran,
                        "absolute_morans_i_difference": abs(rna_moran - ga_moran),
                        "status": "pass",
                    })
            del activity, raw, normalized, canonical

    detail = pd.DataFrame(detail_rows)
    expected_detail = int(config["expected_method_contracts"]) * int(config["folds"]["count"]) * int(config["rna_programs"]["components"])
    if len(detail) != expected_detail or not detail.status.eq("pass").all():
        raise ValueError("NU4 detailed result dimensions changed")
    group_columns = ["dataset_id", "study_id", "study_family", "species", "genome_build", "representation_id", "input_track", "method_id", "method_category"]
    summary_rows: list[dict[str, Any]] = []
    for contract_index, (keys, frame) in enumerate(detail.groupby(group_columns, sort=True, dropna=False)):
        score = frame.heldout_component_spearman.to_numpy(dtype=float)
        pearson = frame.heldout_component_pearson.to_numpy(dtype=float)
        moran = frame.absolute_morans_i_difference.to_numpy(dtype=float)
        score_low, score_high = _descriptive_bootstrap(score, int(config["aggregation"]["bootstrap_draws_descriptive_only"]), int(config["aggregation"]["bootstrap_seed"]) + contract_index)
        moran_low, moran_high = _descriptive_bootstrap(moran, int(config["aggregation"]["bootstrap_draws_descriptive_only"]), int(config["aggregation"]["bootstrap_seed"]) + 1000 + contract_index)
        summary_rows.append({
            **dict(zip(group_columns, keys)),
            "fold_component_units": len(frame),
            "median_heldout_component_spearman": float(np.median(score)),
            "median_heldout_component_pearson": float(np.median(pearson)),
            "median_absolute_morans_i_difference": float(np.median(moran)),
            "descriptive_bootstrap_score_q025": score_low,
            "descriptive_bootstrap_score_q975": score_high,
            "descriptive_bootstrap_morans_difference_q025": moran_low,
            "descriptive_bootstrap_morans_difference_q975": moran_high,
            "median_activity_present_gene_fraction": float(frame.activity_present_gene_fraction.median()),
            "status": "pass",
        })
    summary = pd.DataFrame(summary_rows)

    controls = summary.loc[summary.method_id.eq(CONTROL), [
        "dataset_id", "representation_id", "median_heldout_component_spearman",
        "median_heldout_component_pearson", "median_absolute_morans_i_difference",
    ]].rename(columns={
        "median_heldout_component_spearman": "control_median_heldout_component_spearman",
        "median_heldout_component_pearson": "control_median_heldout_component_pearson",
        "median_absolute_morans_i_difference": "control_median_absolute_morans_i_difference",
    })
    deltas = summary.loc[summary.method_category.eq("annotation")].merge(
        controls, on=["dataset_id", "representation_id"], how="left", validate="many_to_one"
    )
    deltas["delta_heldout_component_spearman_vs_control"] = deltas.median_heldout_component_spearman - deltas.control_median_heldout_component_spearman
    deltas["delta_heldout_component_pearson_vs_control"] = deltas.median_heldout_component_pearson - deltas.control_median_heldout_component_pearson
    deltas["delta_morans_difference_vs_control"] = deltas.control_median_absolute_morans_i_difference - deltas.median_absolute_morans_i_difference
    if len(deltas) != int(config["expected_matched_control_deltas"]) or deltas.filter(like="control_").isna().any().any():
        raise ValueError("NU4 representation-matched control delta panel changed")

    output = root / config["output_root"]
    detail_path = output / "fold_component_metrics.tsv.gz"
    summary_path = output / "contract_summary.tsv"
    delta_path = output / "matched_control_deltas.tsv"
    genes_path = output / "fold_gene_eligibility.tsv"
    detail.to_csv(detail_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    summary.to_csv(summary_path, sep="\t", index=False)
    deltas.to_csv(delta_path, sep="\t", index=False)
    pd.DataFrame(gene_rows).to_csv(genes_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "milestone": "NU4_heldout_RNA_spatial_program_preservation",
        "status": "pass",
        "samples": summary.dataset_id.nunique(),
        "study_families": summary.study_family.nunique(),
        "method_contracts": len(summary),
        "fold_component_rows": len(detail),
        "matched_control_deltas": len(deltas),
        "rna_expression_values_accessed_for_evaluation": True,
        "rna_used_for_construction": False,
        "construction_outputs_mutated": False,
        "labels_clusters_pathology_accessed": False,
        "method_rerun": False,
        "construction_tuning_run": False,
        "policy_refit_run": False,
        "input_tracks_pooled": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "fold_component_units_treated_as_independent_studies": False,
        "interpretation_boundary": "heldout_spatial_program_preservation_within_dataset_and_representation_only",
        "artifacts": {
            "fold_component_metrics.tsv.gz": {"path": str(detail_path.relative_to(root)), "sha256": sha256_path(detail_path)},
            "contract_summary.tsv": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
            "matched_control_deltas.tsv": {"path": str(delta_path.relative_to(root)), "sha256": sha256_path(delta_path)},
            "fold_gene_eligibility.tsv": {"path": str(genes_path.relative_to(root)), "sha256": sha256_path(genes_path)},
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "preflight_evidence": {"path": str(preflight_path.relative_to(root)), "sha256": sha256_path(preflight_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "implementation": {"path": "spatialgabench/nm_spatial_program_evaluation.py", "sha256": sha256_path(Path(__file__))},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_upgrade_spatial_programs.yaml")
    parser.add_argument("--preflight", default="results/v1/nm_upgrade/nu4_spatial_programs/preflight_evidence.json")
    parser.add_argument("--authorization", default="governance/authorizations/NM_U4_SPATIAL_PROGRAM_EVALUATION_AUTHORIZATION_2026-07-30.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(run(args.config, args.preflight, args.authorization, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
