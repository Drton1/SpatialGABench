"""Post-primary nonlinear spatial-baseline sensitivity for MALT morphology."""

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
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from spatialgabench.downstream.clustering_eval import prepare_gene_activity_for_clustering
from spatialgabench.g4_m3f_morphology import _bootstrap_median_interval
from spatialgabench.software_wrappers.common import sha256_file


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rbf_design(coordinates: np.ndarray, landmarks: np.ndarray, bandwidth: float) -> np.ndarray:
    squared = pairwise_distances(coordinates, landmarks, metric="sqeuclidean")
    return np.exp(-squared / (2.0 * bandwidth**2))


def fit_spatial_basis(
    train_coordinates: np.ndarray,
    test_coordinates: np.ndarray,
    baseline_id: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one spatial basis using training coordinates only."""
    coordinate_scaler = StandardScaler().fit(train_coordinates)
    coordinate_train = coordinate_scaler.transform(train_coordinates)
    coordinate_test = coordinate_scaler.transform(test_coordinates)
    details: dict[str, Any] = {"coordinate_scaler_fit": "training_fold_only"}
    if baseline_id == "polynomial_degree2":
        polynomial = PolynomialFeatures(
            degree=int(config["polynomial_degree"]),
            include_bias=bool(config["polynomial_include_bias"]),
        )
        basis_train = polynomial.fit_transform(coordinate_train)
        basis_test = polynomial.transform(coordinate_test)
        details.update(
            {
                "basis": "polynomial",
                "degree": int(config["polynomial_degree"]),
                "landmarks": 0,
                "bandwidth": np.nan,
            }
        )
    elif baseline_id == "rbf_kmeans25":
        model = KMeans(
            n_clusters=int(config["rbf_landmarks"]),
            random_state=int(config["rbf_seed"]),
            n_init=int(config["rbf_kmeans_n_init"]),
        ).fit(coordinate_train)
        landmarks = np.asarray(model.cluster_centers_, dtype=float)
        distances = pairwise_distances(landmarks, landmarks)
        positive = distances[np.triu_indices_from(distances, k=1)]
        positive = positive[positive > 0]
        if len(positive) == 0:
            raise ValueError("RBF landmarks do not define a positive bandwidth")
        bandwidth = float(np.median(positive))
        basis_train = _rbf_design(coordinate_train, landmarks, bandwidth)
        basis_test = _rbf_design(coordinate_test, landmarks, bandwidth)
        details.update(
            {
                "basis": "rbf",
                "degree": 0,
                "landmarks": len(landmarks),
                "bandwidth": bandwidth,
                "landmark_fit": "training_fold_only",
            }
        )
    else:
        raise ValueError(f"Unknown spatial baseline: {baseline_id}")
    basis_scaler = StandardScaler().fit(basis_train)
    basis_train = basis_scaler.transform(basis_train)
    basis_test = basis_scaler.transform(basis_test)
    details["basis_dimensions"] = basis_train.shape[1]
    details["basis_scaler_fit"] = "training_fold_only"
    return basis_train, basis_test, details


def _fit_image_target(
    image_features: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    scaler = StandardScaler().fit(image_features[train])
    train_scaled = scaler.transform(image_features[train])
    test_scaled = scaler.transform(image_features[test])
    n_components = min(
        int(config["image_pcs_fit_within_training_fold"]),
        train_scaled.shape[0] - 1,
        train_scaled.shape[1],
    )
    pca = PCA(n_components=n_components, svd_solver="full")
    train_target = pca.fit_transform(train_scaled)[:, : int(config["image_target_pcs"])]
    test_target = pca.transform(test_scaled)[:, : int(config["image_target_pcs"])]
    return train_target, test_target, n_components


def _method_category(method_id: str) -> str:
    if method_id == "raw_peak_lsi_baseline":
        return "representation_control"
    if "random_peak_to_gene" in method_id:
        return "mapping_control"
    return "annotation_method"


def _validate_authorization(config: dict[str, Any], authorization: dict[str, Any]) -> None:
    if (
        authorization.get("status") != "authorized"
        or authorization.get("scope") != "post_primary_exploratory_diagnostic_sensitivity"
        or authorization.get("post_primary") is not True
        or authorization.get("exploratory_diagnostic_only") is not True
        or authorization.get("authorized_representations") != config["representations"]
        or authorization.get("authorized_method_panel") != config["methods"]
        or authorization.get("authorized_spatial_baselines") != config["spatial_baselines"]
    ):
        raise PermissionError("Nonlinear morphology sensitivity authorization boundary is invalid")
    parameter_map = {
        "spatial_folds": "spatial_folds",
        "image_pcs_fit_within_training_fold": "image_pcs_fit_within_training_fold",
        "image_target_pcs": "image_target_pcs",
        "gene_activity_embedding_pcs": "gene_activity_embedding_pcs",
        "raw_lsi_embedding_pcs": "raw_lsi_embedding_pcs",
        "ridge_alpha": "ridge_alpha",
        "polynomial_degree": "polynomial_degree",
        "polynomial_include_bias": "polynomial_include_bias",
        "rbf_landmarks": "rbf_landmarks",
        "rbf_kmeans_n_init": "rbf_kmeans_n_init",
        "rbf_seed": "rbf_seed",
        "rbf_bandwidth": "rbf_bandwidth",
        "bootstrap_spatial_blocks": "bootstrap_spatial_blocks",
        "bootstrap_seed": "bootstrap_seed",
    }
    for authorization_key, config_key in parameter_map.items():
        if authorization["frozen_parameters"][authorization_key] != config[config_key]:
            raise ValueError(f"Frozen sensitivity parameter changed: {config_key}")


def _control_diagnostics(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["spatial_baseline_id", "representation_id", "subset_id"]
    for (baseline_id, representation_id, subset_id), frame in metrics.groupby(keys, sort=True):
        pivot = frame.pivot(index="spatial_fold", columns="method_id", values="delta_R2_image")
        matched = pivot[config["matched_control"]]
        random = pivot[config["random_control"]]
        raw_lsi = pivot[config["raw_lsi_control"]]
        annotation_above_matched = []
        annotation_above_raw_lsi = []
        annotation_all_folds_above_matched = []
        for method_id in config["annotation_methods"]:
            annotation_above_matched.append(float(np.median(pivot[method_id] - matched)) > 0)
            annotation_above_raw_lsi.append(float(np.median(pivot[method_id] - raw_lsi)) > 0)
            annotation_all_folds_above_matched.append(bool(((pivot[method_id] - matched) > 0).all()))
        method_medians = frame.groupby("method_id")["spatial_plus_method_R2_image"].median()
        rows.append(
            {
                "spatial_baseline_id": baseline_id,
                "representation_id": representation_id,
                "subset_id": subset_id,
                "spatial_folds": len(pivot),
                "matched_control_median_delta_R2_image": float(np.median(matched)),
                "random_control_median_delta_R2_image": float(np.median(random)),
                "raw_lsi_median_delta_R2_image": float(np.median(raw_lsi)),
                "annotation_median_above_matched_control": int(sum(annotation_above_matched)),
                "annotation_all_folds_above_matched_control": int(
                    sum(annotation_all_folds_above_matched)
                ),
                "annotation_median_above_raw_lsi": int(sum(annotation_above_raw_lsi)),
                "methods_with_positive_median_absolute_R2": int((method_medians > 0).sum()),
                "aggregation_controls_both_positive": bool(
                    np.median(matched) > 0 and np.median(random) > 0
                ),
                "interpretation": "aggregation_signal_not_mapping_specific"
                if np.median(matched) > 0 and np.median(random) > 0
                else "aggregation_control_pattern_attenuated_or_mixed",
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="mergesort")


def run_sensitivity(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    frozen = (
        "authorization",
        "previous_gate",
        "primary_evaluation_evidence",
        "primary_fold_metrics",
        "image_features",
        "method_evidence",
        "method_summary",
        "mapping",
        "fold_assignments",
    )
    for field in frozen:
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen nonlinear sensitivity input changed: {field}")
    authorization = _json(root / config["authorization"])
    _validate_authorization(config, authorization)
    primary_evidence = _json(root / config["primary_evaluation_evidence"])
    if primary_evidence.get("status") != "pass" or primary_evidence.get("final_ranking_run") is not False:
        raise ValueError("Primary morphology evidence does not pass its frozen boundary")
    forbidden_flags = (
        "primary_endpoint_replacement_run",
        "method_selection_run",
        "rna_accessed",
        "labels_or_clusters_accessed",
        "pathology_labels_accessed",
        "holdout_accessed",
        "representation_pooling_run",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
    )
    if config.get("post_primary") is not True or config.get("exploratory_diagnostic_only") is not True:
        raise PermissionError("Sensitivity must remain explicitly post-primary and diagnostic")
    if any(bool(config.get(field)) for field in forbidden_flags):
        raise PermissionError("A forbidden nonlinear morphology sensitivity flag is enabled")

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
        raise ValueError("Sensitivity image/method/fold identities do not align")
    feature_columns = list(image.columns[4:])
    if len(feature_columns) != 68:
        raise ValueError("Frozen image feature dimension changed")
    image_matrix = image[feature_columns].to_numpy(dtype=float)
    coordinates = folds[["image_x", "image_y"]].to_numpy(dtype=float)
    fold_values = folds["spatial_fold"].to_numpy(dtype=int)
    foreground = folds["mask_foreground"].astype(bool).to_numpy()
    if not np.isfinite(image_matrix).all() or not np.isfinite(coordinates).all():
        raise ValueError("Sensitivity inputs contain non-finite values")

    primary = pd.read_csv(root / config["primary_fold_metrics"], sep="\t")
    if len(primary) != 320 or not (primary["status"] == "pass").all():
        raise ValueError("Primary morphology fold panel is incomplete")
    primary_keys = ["representation_id", "method_id", "subset_id", "spatial_fold"]
    if primary.duplicated(primary_keys).any():
        raise ValueError("Primary morphology fold panel has duplicate keys")
    primary_index = primary.set_index(primary_keys)

    method_summary = pd.read_csv(root / config["method_summary"], sep="\t")
    expected_pairs = {(rep, method) for rep in config["representations"] for method in config["methods"]}
    if (
        set(zip(method_summary["representation_id"], method_summary["method_id"])) != expected_pairs
        or not (method_summary["status"] == "pass").all()
    ):
        raise ValueError("Sensitivity method panel differs from passed construction evidence")

    subsets = {
        "foreground_primary": foreground,
        "all_registered_sensitivity": np.ones(len(mapping), dtype=bool),
    }
    fold_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    baseline_rows: list[dict[str, Any]] = []
    for subset_id, subset in subsets.items():
        for fold in sorted(np.unique(fold_values)):
            test = subset & (fold_values == fold)
            train = subset & (fold_values != fold)
            image_train, image_test, image_pcs = _fit_image_target(
                image_matrix, train, test, config
            )
            primary_baselines = primary[
                (primary["subset_id"] == subset_id) & (primary["spatial_fold"] == fold)
            ]["coordinate_only_R2_image"].to_numpy(dtype=float)
            if len(primary_baselines) != 32 or not np.allclose(
                primary_baselines, primary_baselines[0], rtol=0, atol=1e-12
            ):
                raise ValueError("Primary linear coordinate baseline is not method invariant")
            for baseline_id in config["spatial_baselines"]:
                basis_train, basis_test, details = fit_spatial_basis(
                    coordinates[train], coordinates[test], baseline_id, config
                )
                baseline_model = Ridge(alpha=float(config["ridge_alpha"])).fit(
                    basis_train, image_train
                )
                baseline_r2 = float(
                    r2_score(
                        image_test,
                        baseline_model.predict(basis_test),
                        multioutput="variance_weighted",
                    )
                )
                fold_cache[(subset_id, int(fold), baseline_id)] = {
                    "train": train,
                    "test": test,
                    "image_train": image_train,
                    "image_test": image_test,
                    "basis_train": basis_train,
                    "basis_test": basis_test,
                    "baseline_r2": baseline_r2,
                    "image_pcs": image_pcs,
                    **details,
                }
                baseline_rows.append(
                    {
                        "spatial_baseline_id": baseline_id,
                        "subset_id": subset_id,
                        "spatial_fold": int(fold),
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "image_pcs_fit": image_pcs,
                        "basis_dimensions": details["basis_dimensions"],
                        "rbf_landmarks": details["landmarks"],
                        "rbf_bandwidth": details["bandwidth"],
                        "primary_linear_coordinate_R2_image": float(primary_baselines[0]),
                        "nonlinear_spatial_baseline_R2_image": baseline_r2,
                        "baseline_R2_change_vs_primary_linear": baseline_r2
                        - float(primary_baselines[0]),
                        "status": "pass",
                    }
                )

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
                raise ValueError(f"Method output changed: {representation_id}/{method_id}")
            data = ad.read_h5ad(path)
            if data.obs_names.astype(str).tolist() != mapping["barcode"].tolist():
                raise ValueError(f"Method observations changed: {representation_id}/{method_id}")
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
                raise ValueError(f"Method embedding invalid: {representation_id}/{method_id}")
            preprocessing_rows.append(
                {
                    "representation_id": representation_id,
                    "method_id": method_id,
                    **preprocessing,
                    "status": "pass",
                }
            )
            for subset_id in subsets:
                for fold in sorted(np.unique(fold_values)):
                    for baseline_id in config["spatial_baselines"]:
                        cached = fold_cache[(subset_id, int(fold), baseline_id)]
                        train = cached["train"]
                        test = cached["test"]
                        embedding_scaler = StandardScaler().fit(embedding[train])
                        embedding_train = embedding_scaler.transform(embedding[train])
                        embedding_test = embedding_scaler.transform(embedding[test])
                        full_model = Ridge(alpha=float(config["ridge_alpha"])).fit(
                            np.column_stack((cached["basis_train"], embedding_train)),
                            cached["image_train"],
                        )
                        full_r2 = float(
                            r2_score(
                                cached["image_test"],
                                full_model.predict(
                                    np.column_stack((cached["basis_test"], embedding_test))
                                ),
                                multioutput="variance_weighted",
                            )
                        )
                        key = (representation_id, method_id, subset_id, int(fold))
                        primary_row = primary_index.loc[key]
                        baseline_r2 = float(cached["baseline_r2"])
                        metric_rows.append(
                            {
                                "spatial_baseline_id": baseline_id,
                                "representation_id": representation_id,
                                "method_id": method_id,
                                "method_category": _method_category(method_id),
                                "subset_id": subset_id,
                                "spatial_fold": int(fold),
                                "n_train": int(train.sum()),
                                "n_test": int(test.sum()),
                                "image_pcs_fit": int(cached["image_pcs"]),
                                "image_target_pcs": int(config["image_target_pcs"]),
                                "spatial_basis_dimensions": int(cached["basis_dimensions"]),
                                "method_embedding_dimensions": embedding.shape[1],
                                "spatial_baseline_R2_image": baseline_r2,
                                "spatial_plus_method_R2_image": full_r2,
                                "delta_R2_image": full_r2 - baseline_r2,
                                "primary_linear_coordinate_R2_image": float(
                                    primary_row["coordinate_only_R2_image"]
                                ),
                                "primary_linear_plus_method_R2_image": float(
                                    primary_row["coordinate_plus_method_R2_image"]
                                ),
                                "primary_delta_R2_image": float(primary_row["delta_R2_image"]),
                                "baseline_R2_change_vs_primary_linear": baseline_r2
                                - float(primary_row["coordinate_only_R2_image"]),
                                "full_R2_change_vs_primary_linear_model": full_r2
                                - float(primary_row["coordinate_plus_method_R2_image"]),
                                "delta_R2_change_vs_primary": (full_r2 - baseline_r2)
                                - float(primary_row["delta_R2_image"]),
                                "status": "pass",
                            }
                        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["spatial_baseline_id", "representation_id", "method_id", "subset_id", "spatial_fold"],
        kind="mergesort",
    )
    summaries: list[dict[str, Any]] = []
    group_keys = ["spatial_baseline_id", "representation_id", "method_id", "subset_id"]
    for keys, frame in metrics.groupby(group_keys, sort=True):
        values = frame["delta_R2_image"].to_numpy(dtype=float)
        low, high = _bootstrap_median_interval(
            values, int(config["bootstrap_spatial_blocks"]), int(config["bootstrap_seed"])
        )
        summaries.append(
            {
                **dict(zip(group_keys, keys)),
                "method_category": frame["method_category"].iloc[0],
                "spatial_folds": len(frame),
                "median_spatial_baseline_R2_image": float(
                    frame["spatial_baseline_R2_image"].median()
                ),
                "median_spatial_plus_method_R2_image": float(
                    frame["spatial_plus_method_R2_image"].median()
                ),
                "median_delta_R2_image": float(np.median(values)),
                "minimum_delta_R2_image": float(np.min(values)),
                "maximum_delta_R2_image": float(np.max(values)),
                "bootstrap_median_95ci_low": low,
                "bootstrap_median_95ci_high": high,
                "median_baseline_R2_change_vs_primary_linear": float(
                    frame["baseline_R2_change_vs_primary_linear"].median()
                ),
                "median_full_R2_change_vs_primary_linear_model": float(
                    frame["full_R2_change_vs_primary_linear_model"].median()
                ),
                "median_delta_R2_change_vs_primary": float(
                    frame["delta_R2_change_vs_primary"].median()
                ),
                "all_fold_values_retained": True,
                "status": "pass",
            }
        )
    summary = pd.DataFrame(summaries).sort_values(group_keys, kind="mergesort")
    baseline_frame = pd.DataFrame(baseline_rows).sort_values(
        ["spatial_baseline_id", "subset_id", "spatial_fold"], kind="mergesort"
    )
    diagnostics = _control_diagnostics(metrics, config)

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "nonlinear_spatial_sensitivity_fold_metrics.tsv"
    summary_path = output / "nonlinear_spatial_sensitivity_method_summary.tsv"
    baseline_path = output / "spatial_baseline_fold_comparison.tsv"
    diagnostic_path = output / "within_representation_control_diagnostics.tsv"
    preprocessing_path = output / "method_embedding_preprocessing.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    baseline_frame.to_csv(baseline_path, sep="\t", index=False)
    diagnostics.to_csv(diagnostic_path, sep="\t", index=False)
    pd.DataFrame(preprocessing_rows).to_csv(preprocessing_path, sep="\t", index=False)

    primary_diagnostics = diagnostics[diagnostics["subset_id"] == config["primary_subset"]]
    baseline_primary = baseline_frame[baseline_frame["subset_id"] == config["primary_subset"]]
    baseline_improvement = {
        baseline_id: bool(
            frame["nonlinear_spatial_baseline_R2_image"].median()
            > frame["primary_linear_coordinate_R2_image"].median()
        )
        for baseline_id, frame in baseline_primary.groupby("spatial_baseline_id", sort=True)
    }
    artifacts = (
        metrics_path,
        summary_path,
        baseline_path,
        diagnostic_path,
        preprocessing_path,
    )
    evidence_path = output / "nonlinear_spatial_sensitivity_evidence.json"
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G4",
            "milestone": "M3F_GSE288178_MALT_nonlinear_spatial_baseline_sensitivity",
            "status": "pass"
            if len(metrics) == 640
            and len(summary) == 128
            and len(baseline_frame) == 20
            and len(diagnostics) == 16
            and len(preprocessing_rows) == 32
            else "fail",
            "analysis_class": "post_primary_exploratory_diagnostic",
            "trigger": authorization["trigger"],
            "claim_boundary": "diagnostic_spatial_baseline_sensitivity_not_primary_endpoint_or_pathology_accuracy",
            "runs": 64,
            "representations": 4,
            "methods": 8,
            "spatial_baselines": 2,
            "fold_metric_rows": len(metrics),
            "summary_rows": len(summary),
            "baseline_fold_rows": len(baseline_frame),
            "control_diagnostic_rows": len(diagnostics),
            "baseline_median_absolute_R2_improved_vs_primary_linear": baseline_improvement,
            "primary_control_patterns": {
                baseline_id: {
                    "representations_with_both_aggregation_controls_positive": int(
                        frame["aggregation_controls_both_positive"].sum()
                    ),
                    "annotation_branches_all_folds_above_matched_control": int(
                        frame["annotation_all_folds_above_matched_control"].sum()
                    ),
                    "methods_with_positive_median_absolute_R2_across_representations": int(
                        frame["methods_with_positive_median_absolute_R2"].sum()
                    ),
                }
                for baseline_id, frame in primary_diagnostics.groupby(
                    "spatial_baseline_id", sort=True
                )
            },
            "artifacts": {
                path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
                for path in artifacts
            },
            "frozen_inputs": {
                field: {"path": config[field], "sha256": config[f"{field}_sha256"]}
                for field in frozen
            },
            "image_scaler_and_pca_fit_training_folds_only": True,
            "spatial_coordinate_scaler_fit_training_folds_only": True,
            "spatial_basis_fit_training_folds_only": True,
            "rbf_landmarks_fit_training_folds_only": True,
            "negative_values_retained": True,
            "post_primary": True,
            "exploratory_diagnostic_only": True,
            "primary_endpoint_replaced": False,
            "method_selection_run": False,
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
    parser.add_argument(
        "--config",
        default="configs/g4_m3f_gse288178_malt_nonlinear_spatial_sensitivity.yaml",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    path = run_sensitivity(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

