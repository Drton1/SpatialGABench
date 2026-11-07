"""Leakage-separated image-only morphology concordance evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from spatialgabench.downstream.clustering_eval import prepare_gene_activity_for_clustering
from spatialgabench.software_wrappers.common import sha256_file


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def prepare_spatial_folds(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    frozen = ("authorization", "protocol", "image_evidence", "image_features", "mapping")
    for field in frozen:
        path = root / config[field]
        if sha256_file(path) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen morphology-fold input changed: {field}")
    authorization = _json(root / config["authorization"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("authorized_task")
        != "spatial_block_coordinate_adjusted_image_only_morphology_concordance"
        or authorization["frozen_parameters"]["spatial_fold_algorithm"]
        != config["spatial_fold_algorithm"]
        or authorization["frozen_parameters"]["spatial_fold_seed"]
        != config["spatial_fold_seed"]
        or authorization["frozen_parameters"]["spatial_folds"] != config["spatial_folds"]
    ):
        raise PermissionError("MALT morphology-fold authorization boundary is invalid")
    if config.get("method_outputs_accessed") is not False or config.get("atac_accessed") is not False:
        raise PermissionError("Morphology folds must be frozen without molecular access")

    mapping = pd.read_csv(root / config["mapping"], sep="\t", dtype={"barcode": str, "spot": str})
    image_index = pd.read_csv(
        root / config["image_features"],
        sep="\t",
        usecols=["spot", "image_x", "image_y", "mask_foreground"],
        dtype={"spot": str},
    )
    if (
        len(mapping) != int(config["expected_observations"])
        or len(image_index) != int(config["expected_observations"])
        or not mapping["barcode"].is_unique
        or not mapping["spot"].is_unique
        or mapping["spot"].tolist() != image_index["spot"].tolist()
        or not np.array_equal(
            mapping[["image_x", "image_y"]].to_numpy(),
            image_index[["image_x", "image_y"]].to_numpy(),
        )
    ):
        raise ValueError("MALT morphology fold inputs do not satisfy exact identity/order alignment")
    foreground = image_index["mask_foreground"].astype(bool).to_numpy()
    if int(foreground.sum()) != int(config["expected_foreground_observations"]):
        raise ValueError("MALT morphology foreground count changed")
    coordinates = image_index[["image_x", "image_y"]].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(coordinates)
    model = KMeans(
        n_clusters=int(config["spatial_folds"]),
        random_state=int(config["spatial_fold_seed"]),
        n_init=int(config["kmeans_n_init"]),
    ).fit(scaled)
    centroids = np.asarray(model.cluster_centers_)
    centroid_order = np.lexsort((centroids[:, 1], centroids[:, 0]))
    remap = {int(old): int(new) for new, old in enumerate(centroid_order)}
    folds = np.asarray([remap[int(value)] for value in model.labels_], dtype=np.int8)
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    assignment_path = output / "spatial_fold_assignments.tsv"
    assignment = pd.DataFrame(
        {
            "barcode": mapping["barcode"],
            "spot": mapping["spot"],
            "image_x": coordinates[:, 0],
            "image_y": coordinates[:, 1],
            "mask_foreground": foreground,
            "spatial_fold": folds,
        }
    )
    assignment.to_csv(assignment_path, sep="\t", index=False)
    qc = (
        assignment.groupby("spatial_fold", sort=True)
        .agg(
            observations=("barcode", "size"),
            foreground_observations=("mask_foreground", "sum"),
            image_x_min=("image_x", "min"),
            image_x_max=("image_x", "max"),
            image_y_min=("image_y", "min"),
            image_y_max=("image_y", "max"),
        )
        .reset_index()
    )
    qc["status"] = np.where(
        qc["foreground_observations"] >= int(config["minimum_foreground_per_fold"]),
        "pass",
        "fail",
    )
    qc_path = output / "spatial_fold_qc.tsv"
    qc.to_csv(qc_path, sep="\t", index=False)
    if len(qc) != int(config["spatial_folds"]) or not (qc["status"] == "pass").all():
        raise ValueError("MALT spatial folds fail the frozen fold-size contract")
    evidence_path = output / "spatial_fold_evidence.json"
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "status": "pass",
            "algorithm": config["spatial_fold_algorithm"],
            "seed": int(config["spatial_fold_seed"]),
            "kmeans_n_init": int(config["kmeans_n_init"]),
            "folds": int(config["spatial_folds"]),
            "observations": len(assignment),
            "foreground_observations": int(foreground.sum()),
            "minimum_foreground_per_fold": int(qc["foreground_observations"].min()),
            "artifacts": {
                "spatial_fold_assignments.tsv": {
                    "path": str(assignment_path.relative_to(root)),
                    "sha256": sha256_file(assignment_path),
                },
                "spatial_fold_qc.tsv": {
                    "path": str(qc_path.relative_to(root)),
                    "sha256": sha256_file(qc_path),
                },
            },
            "authorization": {
                "path": config["authorization"],
                "sha256": config["authorization_sha256"],
            },
            "protocol": {"path": config["protocol"], "sha256": config["protocol_sha256"]},
            "image_evidence": {
                "path": config["image_evidence"],
                "sha256": config["image_evidence_sha256"],
            },
            "mapping": {"path": config["mapping"], "sha256": config["mapping_sha256"]},
            "method_outputs_accessed": False,
            "method_identifiers_accessed": False,
            "atac_accessed": False,
            "rna_accessed": False,
            "labels_or_clusters_accessed": False,
            "pathology_labels_accessed": False,
            "holdout_accessed": False,
            "model_fit_run": False,
            "method_comparison_run": False,
            "full_real_data_benchmark_run": False,
            "ranking_run": False,
        },
    )
    return evidence_path


def _bootstrap_median_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def _fold_scores(
    image_features: np.ndarray,
    coordinates: np.ndarray,
    method_embedding: np.ndarray,
    folds: np.ndarray,
    subset: np.ndarray,
    config: dict[str, Any],
    representation_id: str,
    method_id: str,
    subset_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        test = subset & (folds == fold)
        train = subset & (folds != fold)
        image_scaler = StandardScaler().fit(image_features[train])
        image_train_scaled = image_scaler.transform(image_features[train])
        image_test_scaled = image_scaler.transform(image_features[test])
        n_image_pcs = min(
            int(config["image_pcs_fit_within_training_fold"]),
            image_train_scaled.shape[0] - 1,
            image_train_scaled.shape[1],
        )
        image_pca = PCA(n_components=n_image_pcs, svd_solver="full")
        image_train = image_pca.fit_transform(image_train_scaled)[
            :, : int(config["image_target_pcs"])
        ]
        image_test = image_pca.transform(image_test_scaled)[:, : int(config["image_target_pcs"])]
        coordinate_scaler = StandardScaler().fit(coordinates[train])
        coordinate_train = coordinate_scaler.transform(coordinates[train])
        coordinate_test = coordinate_scaler.transform(coordinates[test])
        embedding_scaler = StandardScaler().fit(method_embedding[train])
        embedding_train = embedding_scaler.transform(method_embedding[train])
        embedding_test = embedding_scaler.transform(method_embedding[test])
        coordinate_model = Ridge(alpha=float(config["ridge_alpha"])).fit(
            coordinate_train, image_train
        )
        full_model = Ridge(alpha=float(config["ridge_alpha"])).fit(
            np.column_stack((coordinate_train, embedding_train)), image_train
        )
        coordinate_r2 = float(
            r2_score(
                image_test,
                coordinate_model.predict(coordinate_test),
                multioutput="variance_weighted",
            )
        )
        full_r2 = float(
            r2_score(
                image_test,
                full_model.predict(np.column_stack((coordinate_test, embedding_test))),
                multioutput="variance_weighted",
            )
        )
        rows.append(
            {
                "representation_id": representation_id,
                "method_id": method_id,
                "subset_id": subset_id,
                "spatial_fold": int(fold),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "image_pcs_fit": n_image_pcs,
                "image_target_pcs": int(config["image_target_pcs"]),
                "method_embedding_dimensions": method_embedding.shape[1],
                "coordinate_only_R2_image": coordinate_r2,
                "coordinate_plus_method_R2_image": full_r2,
                "delta_R2_image": full_r2 - coordinate_r2,
                "status": "pass",
            }
        )
    return rows


def run_evaluation(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    frozen = (
        "authorization",
        "protocol",
        "previous_gate",
        "image_evidence",
        "image_features",
        "method_evidence",
        "method_summary",
        "mapping",
        "fold_evidence",
        "fold_assignments",
    )
    for field in frozen:
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen MALT morphology input changed: {field}")
    authorization = _json(root / config["authorization"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("authorized_method_panel") != config["methods"]
        or authorization.get("authorized_representations") != config["representations"]
        or authorization.get("primary_endpoint") != config["primary_endpoint"]
    ):
        raise PermissionError("MALT morphology evaluation authorization boundary is invalid")
    for key in (
        "spatial_folds",
        "spatial_fold_seed",
        "image_pcs_fit_within_training_fold",
        "image_target_pcs",
        "gene_activity_embedding_pcs",
        "raw_lsi_embedding_pcs",
        "ridge_alpha",
        "bootstrap_spatial_blocks",
        "bootstrap_seed",
    ):
        if authorization["frozen_parameters"][key] != config[key]:
            raise ValueError(f"MALT morphology frozen parameter changed: {key}")
    if any(
        bool(config.get(field))
        for field in (
            "rna_accessed",
            "labels_or_clusters_accessed",
            "pathology_labels_accessed",
            "holdout_accessed",
            "full_real_data_benchmark_run",
            "recommendation_policy_fit_run",
            "universal_ranking_run",
            "final_ranking_run",
        )
    ):
        raise PermissionError("A forbidden MALT morphology evaluation flag is enabled")

    image = pd.read_csv(root / config["image_features"], sep="\t", dtype={"spot": str})
    mapping = pd.read_csv(root / config["mapping"], sep="\t", dtype={"barcode": str, "spot": str})
    folds = pd.read_csv(
        root / config["fold_assignments"], sep="\t", dtype={"barcode": str, "spot": str}
    )
    if (
        image["spot"].tolist() != mapping["spot"].tolist()
        or folds["spot"].tolist() != mapping["spot"].tolist()
        or folds["barcode"].tolist() != mapping["barcode"].tolist()
    ):
        raise ValueError("MALT morphology image/method join identities do not align")
    feature_columns = list(image.columns[4:])
    if len(feature_columns) != 68:
        raise ValueError("MALT frozen image feature dimension changed")
    image_matrix = image[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(image_matrix).all():
        raise ValueError("MALT frozen image features contain non-finite values")
    coordinates = folds[["image_x", "image_y"]].to_numpy(dtype=float)
    fold_values = folds["spatial_fold"].to_numpy(dtype=int)
    foreground = folds["mask_foreground"].astype(bool).to_numpy()
    method_summary = pd.read_csv(root / config["method_summary"], sep="\t")
    expected_pairs = {(rep, method) for rep in config["representations"] for method in config["methods"]}
    observed_pairs = set(zip(method_summary["representation_id"], method_summary["method_id"]))
    if observed_pairs != expected_pairs or not (method_summary["status"] == "pass").all():
        raise ValueError("MALT morphology method panel differs from passed construction evidence")

    metric_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    for representation_id in config["representations"]:
        for method_id in config["methods"]:
            row = method_summary[
                (method_summary["representation_id"] == representation_id)
                & (method_summary["method_id"] == method_id)
            ].iloc[0]
            path = root / str(row["output_path"])
            if sha256_file(path) != row["output_sha256"]:
                raise ValueError(f"MALT morphology method output changed: {representation_id}/{method_id}")
            data = ad.read_h5ad(path)
            if data.obs_names.astype(str).tolist() != mapping["barcode"].tolist():
                raise ValueError(f"MALT morphology method observations changed: {representation_id}/{method_id}")
            if method_id == "raw_peak_lsi_baseline":
                embedding = np.asarray(data.obsm["X_lsi"])[
                    :, : int(config["raw_lsi_embedding_pcs"])
                ]
                preprocessing = {
                    "feature_type": "raw_lsi",
                    "embedding_dimensions": embedding.shape[1],
                    "preprocessing_seed": int(config["preprocessing_seed"]),
                }
            else:
                prepared = prepare_gene_activity_for_clustering(
                    data,
                    method_id=method_id,
                    preprocessing_config=config["method_preprocessing"],
                    random_seed=int(config["preprocessing_seed"]),
                )
                embedding = np.asarray(prepared.adata.obsm["X_pca"])[
                    :, : int(config["gene_activity_embedding_pcs"])
                ]
                preprocessing = prepared.summary
            if embedding.shape[0] != len(mapping) or not np.isfinite(embedding).all():
                raise ValueError(f"MALT morphology method embedding invalid: {representation_id}/{method_id}")
            preprocessing_rows.append(
                {
                    "representation_id": representation_id,
                    "method_id": method_id,
                    **preprocessing,
                    "status": "pass",
                }
            )
            metric_rows.extend(
                _fold_scores(
                    image_matrix,
                    coordinates,
                    embedding,
                    fold_values,
                    foreground,
                    config,
                    representation_id,
                    method_id,
                    "foreground_primary",
                )
            )
            metric_rows.extend(
                _fold_scores(
                    image_matrix,
                    coordinates,
                    embedding,
                    fold_values,
                    np.ones(len(mapping), dtype=bool),
                    config,
                    representation_id,
                    method_id,
                    "all_registered_sensitivity",
                )
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["representation_id", "method_id", "subset_id", "spatial_fold"], kind="mergesort"
    )
    summaries: list[dict[str, Any]] = []
    for (representation_id, method_id, subset_id), frame in metrics.groupby(
        ["representation_id", "method_id", "subset_id"], sort=True
    ):
        values = frame["delta_R2_image"].to_numpy(dtype=float)
        low, high = _bootstrap_median_interval(
            values,
            int(config["bootstrap_spatial_blocks"]),
            int(config["bootstrap_seed"]),
        )
        summaries.append(
            {
                "representation_id": representation_id,
                "method_id": method_id,
                "method_category": "representation_control"
                if method_id == "raw_peak_lsi_baseline"
                else "mapping_control"
                if "random_peak_to_gene" in method_id
                else "annotation_method",
                "subset_id": subset_id,
                "spatial_folds": len(values),
                "median_delta_R2_image": float(np.median(values)),
                "minimum_delta_R2_image": float(np.min(values)),
                "maximum_delta_R2_image": float(np.max(values)),
                "bootstrap_median_95ci_low": low,
                "bootstrap_median_95ci_high": high,
                "all_fold_values_retained": True,
                "status": "pass",
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["representation_id", "method_id", "subset_id"], kind="mergesort"
    )
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "morphology_fold_metrics.tsv"
    summary_path = output / "morphology_method_summary.tsv"
    preprocessing_path = output / "method_embedding_preprocessing.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    pd.DataFrame(preprocessing_rows).to_csv(preprocessing_path, sep="\t", index=False)
    evidence_path = output / "gse288178_malt_morphology_concordance_evidence.json"
    artifacts = (metrics_path, summary_path, preprocessing_path)
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G4",
            "milestone": "M3F_GSE288178_MALT_image_only_morphology_concordance",
            "status": "pass"
            if len(metrics) == 320 and len(summary) == 64 and len(preprocessing_rows) == 32
            else "fail",
            "primary_endpoint": config["primary_endpoint"],
            "claim_boundary": "image_only_morphology_concordance_not_pathology_accuracy",
            "runs": 32,
            "representations": 4,
            "methods": 8,
            "primary_foreground_observations": int(foreground.sum()),
            "all_registered_observations": len(foreground),
            "fold_metric_rows": len(metrics),
            "summary_rows": len(summary),
            "artifacts": {
                path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
                for path in artifacts
            },
            "authorization": {
                "path": config["authorization"],
                "sha256": config["authorization_sha256"],
            },
            "protocol": {"path": config["protocol"], "sha256": config["protocol_sha256"]},
            "previous_gate": {
                "path": config["previous_gate"],
                "sha256": config["previous_gate_sha256"],
            },
            "image_evidence": {
                "path": config["image_evidence"],
                "sha256": config["image_evidence_sha256"],
            },
            "method_evidence": {
                "path": config["method_evidence"],
                "sha256": config["method_evidence_sha256"],
            },
            "fold_evidence": {
                "path": config["fold_evidence"],
                "sha256": config["fold_evidence_sha256"],
            },
            "image_scaler_and_pca_fit_training_folds_only": True,
            "coordinate_adjustment_applied": True,
            "negative_delta_values_retained": True,
            "representation_pooling_run": False,
            "pathology_accuracy_claimed": False,
            "rna_accessed": False,
            "labels_or_clusters_accessed": False,
            "pathology_labels_accessed": False,
            "holdout_accessed": False,
            "full_real_data_benchmark_run": False,
            "recommendation_policy_fit_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "next_milestone_authorized": False,
        },
    )
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-folds", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "prepare-folds":
        path = prepare_spatial_folds(args.config, args.root)
    else:
        path = run_evaluation(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
