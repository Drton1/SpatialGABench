"""Extract frozen image-only handcrafted H&E features for GSE288178 MALT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


FROZEN_FIELDS = ("authorization", "previous_gate", "protocol", "source_he", "source_mask", "source_locs")
FORBIDDEN_FLAGS = (
    "h5ad_accessed",
    "atac_fragments_or_counts_accessed",
    "method_outputs_accessed",
    "method_identifiers_accessed",
    "rna_accessed",
    "source_clusters_accessed",
    "pathology_labels_accessed",
    "pca_or_model_fit_run",
    "method_comparison_run",
    "holdout_accessed",
    "full_real_data_benchmark_run",
    "recommendation_policy_fit_run",
    "universal_ranking_run",
    "final_ranking_run",
)
STATS = ("mean", "std", "p10", "p25", "p50", "p75", "p90")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_names() -> list[str]:
    names: list[str] = []
    for domain, channels in (("rgb", ("r", "g", "b")), ("od", ("r", "g", "b")), ("stain", ("h", "e"))):
        for channel in channels:
            names.extend(f"{domain}_{channel}_{stat}" for stat in STATS)
    names.extend(("sobel_edge_density", "grayscale_entropy"))
    names.extend(f"lbp_uniform_bin_{index}" for index in range(10))
    return names


def _load(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in FROZEN_FIELDS:
        artifact = root / config[field]
        if not artifact.is_file() or sha256_file(artifact) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen M3F image-feature artifact checksum mismatch: {field}")
    authorization = json.loads((root / config["authorization"]).read_text(encoding="utf-8"))
    if authorization.get("status") != "authorized" or authorization.get("scope") != "development_image_only_raw_feature_extraction":
        raise PermissionError("M3F image-feature authorization is not active")
    enabled = [field for field in FORBIDDEN_FLAGS if bool(config.get(field))]
    if enabled:
        raise PermissionError(f"Forbidden M3F image-feature flags enabled: {enabled}")
    if len(_feature_names()) != int(config["expected_feature_count"]):
        raise ValueError("M3F handcrafted feature schema differs from the frozen contract")
    config["_path"] = path
    return config


def _channel_stats(values: np.ndarray, quantiles: np.ndarray) -> list[float]:
    output: list[float] = []
    for channel in range(values.shape[2]):
        vector = values[:, :, channel]
        output.extend((float(vector.mean()), float(vector.std())))
        output.extend(float(value) for value in np.quantile(vector, quantiles))
    return output


def _uniform_lbp_histogram(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbors = (
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
        gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2],
    )
    bits = np.stack([neighbor >= center for neighbor in neighbors], axis=0)
    transitions = np.count_nonzero(bits != np.roll(bits, 1, axis=0), axis=0)
    ones = bits.sum(axis=0)
    bins = np.where(transitions <= 2, ones, 9)
    counts = np.bincount(bins.ravel(), minlength=10).astype(np.float64)
    return counts / counts.sum()


def _patch_features(patch_uint8: np.ndarray, config: dict[str, Any], stain_inverse: np.ndarray) -> np.ndarray:
    rgb = patch_uint8.astype(np.float32) / 255.0
    optical_density = -np.log((patch_uint8.astype(np.float32) + 1.0) / 256.0)
    concentration = optical_density.reshape(-1, 3) @ stain_inverse
    concentration = np.maximum(concentration[:, :2], 0).reshape(patch_uint8.shape[0], patch_uint8.shape[1], 2)
    quantiles = np.asarray(config["summary_quantiles"], dtype=np.float64)
    features = _channel_stats(rgb, quantiles)
    features.extend(_channel_stats(optical_density, quantiles))
    features.extend(_channel_stats(concentration, quantiles))
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gradient_x = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    gradient_y = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    gradient = np.hypot(gradient_x, gradient_y)
    features.append(float((gradient > float(config["sobel_edge_density_threshold"])).mean()))
    histogram = np.histogram(gray, bins=int(config["entropy_histogram_bins"]), range=(0, 1))[0].astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    features.append(float(-(probabilities * np.log2(probabilities)).sum()))
    features.extend(float(value) for value in _uniform_lbp_histogram(gray))
    return np.asarray(features, dtype=np.float64)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    locs = pd.read_csv(root / config["source_locs"], sep="\t")
    if len(locs) != int(config["expected_observations"]) or not locs.spot.is_unique:
        raise ValueError("M3F image-feature locations differ from the frozen subset")
    nearest = cKDTree(locs[["x", "y"]]).query(locs[["x", "y"]], k=2)[0][:, 1]
    median_nearest = float(np.median(nearest))
    target_side = median_nearest * float(config["patch_side_multiplier"])
    patch_side = int(np.floor(target_side))
    if patch_side % 2 == 0:
        patch_side += 1
    if median_nearest != float(config["expected_median_nearest_neighbor_px"]) or patch_side != int(config["expected_patch_side_px"]):
        raise ValueError("M3F image patch geometry differs from the frozen contract")

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(root / config["source_he"]) as image:
        he = np.asarray(image.convert("RGB"))
    with Image.open(root / config["source_mask"]) as image:
        mask = np.asarray(image)
    mask_x = np.clip(np.rint(locs.x.to_numpy() / 16).astype(int), 0, mask.shape[1] - 1)
    mask_y = np.clip(np.rint(locs.y.to_numpy() / 16).astype(int), 0, mask.shape[0] - 1)
    foreground = mask[mask_y, mask_x] == 1
    if int(foreground.sum()) != int(config["expected_foreground_observations"]):
        raise ValueError("M3F image-feature mask subset differs from preflight")
    radius = patch_side // 2
    if (
        locs.x.min() - radius < 0
        or locs.y.min() - radius < 0
        or locs.x.max() + radius >= he.shape[1]
        or locs.y.max() + radius >= he.shape[0]
    ):
        raise ValueError("M3F image patches would cross the H&E boundary")

    hematoxylin = np.array([0.650, 0.704, 0.286], dtype=np.float64)
    eosin = np.array([0.072, 0.990, 0.105], dtype=np.float64)
    residual = np.cross(hematoxylin, eosin)
    stain_matrix = np.stack((hematoxylin, eosin, residual), axis=1)
    stain_matrix /= np.linalg.norm(stain_matrix, axis=0, keepdims=True)
    stain_inverse = np.linalg.inv(stain_matrix.T)

    feature_names = _feature_names()
    matrix = np.empty((len(locs), len(feature_names)), dtype=np.float64)
    for index, (x_value, y_value) in enumerate(locs[["x", "y"]].itertuples(index=False, name=None)):
        x = int(round(x_value))
        y = int(round(y_value))
        patch = he[y - radius : y + radius + 1, x - radius : x + radius + 1]
        if patch.shape != (patch_side, patch_side, 3):
            raise ValueError(f"M3F image patch shape failed at row {index}: {patch.shape}")
        matrix[index] = _patch_features(patch, config, stain_inverse)
    if matrix.shape != (int(config["expected_observations"]), int(config["expected_feature_count"])):
        raise ValueError("M3F image feature matrix shape differs from the frozen contract")
    if not np.isfinite(matrix).all():
        raise ValueError("M3F image feature matrix contains non-finite values")
    feature_std = matrix.std(axis=0)
    if (feature_std <= 0).any():
        raise ValueError("M3F image feature matrix contains constant features")

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "handcrafted_he_v1_raw_features.tsv.gz"
    feature_frame = pd.DataFrame(matrix, columns=feature_names)
    feature_frame.insert(0, "mask_foreground", foreground)
    feature_frame.insert(0, "image_y", locs.y.to_numpy())
    feature_frame.insert(0, "image_x", locs.x.to_numpy())
    feature_frame.insert(0, "spot", locs.spot.astype(str).to_numpy())
    feature_frame.to_csv(
        features_path,
        sep="\t",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    schema_path = output / "image_feature_schema.tsv"
    pd.DataFrame(
        {
            "feature_id": feature_names,
            "feature_index": range(len(feature_names)),
            "feature_branch": config["feature_branch"],
            "uses_molecular_input": False,
        }
    ).to_csv(schema_path, sep="\t", index=False)
    qc_path = output / "image_feature_qc.tsv"
    pd.DataFrame(
        {
            "feature_id": feature_names,
            "minimum": matrix.min(axis=0),
            "maximum": matrix.max(axis=0),
            "mean": matrix.mean(axis=0),
            "standard_deviation": feature_std,
            "finite_fraction": np.isfinite(matrix).mean(axis=0),
        }
    ).to_csv(qc_path, sep="\t", index=False)
    artifacts = [features_path, schema_path, qc_path]
    evidence_path = output / "gse288178_malt_image_feature_evidence.json"
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3F_GSE288178_MALT_image_feature_extraction",
        "status": "pass",
        "feature_branch": config["feature_branch"],
        "observations": len(locs),
        "foreground_observations": int(foreground.sum()),
        "raw_image_features": len(feature_names),
        "median_nearest_neighbor_px": median_nearest,
        "patch_side_px": patch_side,
        "feature_matrix_finite": True,
        "constant_features": int((feature_std <= 0).sum()),
        "artifacts": {
            path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts
        },
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
        "protocol": {"path": config["protocol"], "sha256": config["protocol_sha256"]},
        "source_he": {"path": config["source_he"], "sha256": config["source_he_sha256"]},
        "source_mask": {"path": config["source_mask"], "sha256": config["source_mask_sha256"]},
        "source_locs": {"path": config["source_locs"], "sha256": config["source_locs_sha256"]},
        "h5ad_accessed": False,
        "atac_fragments_or_counts_accessed": False,
        "method_outputs_accessed": False,
        "method_identifiers_accessed": False,
        "rna_accessed": False,
        "source_clusters_accessed": False,
        "pathology_labels_accessed": False,
        "pca_or_model_fit_run": False,
        "method_comparison_run": False,
        "holdout_accessed": False,
        "full_real_data_benchmark_run": False,
        "recommendation_policy_fit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(evidence_path, evidence)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m3f_gse288178_malt_image_features.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(extract(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
