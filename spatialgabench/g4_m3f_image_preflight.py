"""Audit adjacent-section H&E registration without molecular inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


FROZEN_FIELDS = (
    "authorization",
    "protocol",
    "download_receipt",
    "runtime_lock",
    "source_metadata",
    "source_he",
    "source_spatial_image",
    "source_positions",
    "source_scalefactors",
    "author_figure",
)
FORBIDDEN_FLAGS = (
    "image_feature_extraction_run",
    "method_outputs_accessed",
    "atac_fragments_accessed",
    "rna_accessed",
    "clusters_accessed",
    "holdout_accessed",
    "full_real_data_benchmark_run",
    "recommendation_policy_fit_run",
    "universal_ranking_run",
    "final_ranking_run",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(config_path: str | Path, root: Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in FROZEN_FIELDS:
        artifact = root / config[field]
        if not artifact.is_file() or sha256_file(artifact) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen M3F artifact checksum mismatch: {field}")
    authorization = json.loads((root / config["authorization"]).read_text(encoding="utf-8"))
    if authorization.get("status") != "authorized" or authorization.get("scope") != "development_image_and_coordinate_preflight_only":
        raise PermissionError("M3F image-preflight authorization is not active")
    if config.get("section_relationship") != "adjacent_tissue_section":
        raise ValueError("GSE278007 H&E must remain adjacent-section evidence")
    enabled = [field for field in FORBIDDEN_FLAGS if bool(config.get(field))]
    if enabled:
        raise PermissionError(f"Forbidden M3F preflight flags enabled: {enabled}")
    config["_path"] = path
    return config


def _homography(cv2: Any, source: np.ndarray, target: np.ndarray, settings: dict[str, Any], ratio: float) -> tuple[np.ndarray | None, dict[str, Any]]:
    sift = cv2.SIFT_create(
        nfeatures=int(settings["nfeatures"]),
        contrastThreshold=float(settings["contrast_threshold"]),
        edgeThreshold=float(settings["edge_threshold"]),
    )
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    source_keypoints, source_descriptors = sift.detectAndCompute(source_gray, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_descriptors, target_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < ratio * second.distance]
    metrics: dict[str, Any] = {
        "source_keypoints": len(source_keypoints),
        "target_keypoints": len(target_keypoints),
        "good_matches": len(matches),
        "inliers": 0,
        "inlier_fraction": 0.0,
        "median_reprojection_error_px": None,
    }
    if len(matches) < 4:
        return None, metrics
    source_points = np.float32([source_keypoints[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
    target_points = np.float32([target_keypoints[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        float(settings["ransac_reprojection_threshold"]),
        maxIters=10000,
        confidence=0.999,
    )
    if matrix is None or mask is None:
        return None, metrics
    projected = cv2.perspectiveTransform(source_points, matrix)
    errors = np.linalg.norm(projected[:, 0, :] - target_points[:, 0, :], axis=1)
    inliers = mask.ravel().astype(bool)
    metrics.update(
        {
            "inliers": int(inliers.sum()),
            "inlier_fraction": float(inliers.mean()),
            "median_reprojection_error_px": float(np.median(errors[inliers])) if inliers.any() else None,
        }
    )
    return matrix, metrics


def _accepted(metrics: dict[str, Any], thresholds: dict[str, Any], prefix: str = "") -> bool:
    return (
        metrics["good_matches"] >= int(thresholds[f"{prefix}min_good_matches"])
        and metrics["inliers"] >= int(thresholds[f"{prefix}min_inliers"])
        and metrics["inlier_fraction"] >= float(thresholds[f"{prefix}min_inlier_fraction"])
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the hash-locked M3F OpenCV wheel before running image preflight") from exc

    root = Path(root).resolve()
    config = _load(config_path, root)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(4160)
    Image.MAX_IMAGE_PIXELS = None

    with gzip.open(root / config["source_he"], "rb") as handle:
        he_image = Image.open(handle)
        he_size = tuple(he_image.size)
        he_mode = he_image.mode
        he_frames = int(getattr(he_image, "n_frames", 1))
        if list(he_size) != config["expected_he_size"] or he_mode != config["expected_he_mode"] or he_frames != 1:
            raise ValueError(f"Unexpected GSE278007 H&E schema: {he_size}, {he_mode}, {he_frames}")
        he_thumbnail = he_image.convert("RGB")
        he_thumbnail.thumbnail(tuple(config["he_thumbnail_max_size"]), Image.Resampling.LANCZOS)
        he_bgr = cv2.cvtColor(np.asarray(he_thumbnail), cv2.COLOR_RGB2BGR)

    spatial_bgr = cv2.imread(str(root / config["source_spatial_image"]), cv2.IMREAD_COLOR)
    if spatial_bgr is None or [spatial_bgr.shape[1], spatial_bgr.shape[0]] != config["expected_spatial_image_size"]:
        raise ValueError("Unexpected GSE278007 spatial-image schema")
    spatial_thumbnail = cv2.resize(spatial_bgr, tuple(config["spatial_thumbnail_size"]), interpolation=cv2.INTER_AREA)
    figure = cv2.imread(str(root / config["author_figure"]), cv2.IMREAD_COLOR)
    if figure is None:
        raise ValueError("Author Figure 5 could not be decoded")

    hx0, hy0, hx1, hy1 = map(int, config["figure_he_panel_crop_xyxy"])
    sx0, sy0, sx1, sy1 = map(int, config["figure_spatial_panel_crop_xyxy"])
    he_panel = figure[hy0:hy1, hx0:hx1]
    spatial_panel = figure[sy0:sy1, sx0:sx1]
    settings = config["sift"]
    he_matrix, he_metrics = _homography(
        cv2, he_panel, he_bgr, settings, float(settings["localization_ratio_threshold"])
    )
    spatial_matrix, spatial_metrics = _homography(
        cv2, spatial_thumbnail, spatial_panel, settings, float(settings["localization_ratio_threshold"])
    )
    localization = config["localization_acceptance"]
    he_localized = he_matrix is not None and _accepted(he_metrics, localization, "he_")
    spatial_localized = spatial_matrix is not None and _accepted(spatial_metrics, localization, "spatial_")
    if not he_localized or not spatial_localized:
        raise ValueError("Author-figure ROI localization failed its frozen acceptance thresholds")

    roi_figure = np.float32(config["figure_he_author_roi_corners_xy"]).reshape(-1, 1, 2)
    roi_panel = roi_figure.copy()
    roi_panel[:, 0, 0] -= hx0
    roi_panel[:, 0, 1] -= hy0
    roi_he_thumbnail = cv2.perspectiveTransform(roi_panel, he_matrix)[:, 0, :]
    scale = np.array([he_size[0] / he_bgr.shape[1], he_size[1] / he_bgr.shape[0]], dtype=np.float64)
    roi_he_original = roi_he_thumbnail.astype(np.float64) * scale

    rectified_size = tuple(config["spatial_thumbnail_size"])
    rectified_corners = np.float32(
        [[0, 0], [rectified_size[0] - 1, 0], [rectified_size[0] - 1, rectified_size[1] - 1], [0, rectified_size[1] - 1]]
    )
    he_to_rectified = cv2.getPerspectiveTransform(roi_he_thumbnail.astype(np.float32), rectified_corners)
    rectified_he = cv2.warpPerspective(he_bgr, he_to_rectified, rectified_size, borderValue=(255, 255, 255))
    direct_matrix, direct_metrics = _homography(
        cv2,
        spatial_thumbnail,
        rectified_he,
        {**settings, "edge_threshold": 20},
        float(settings["direct_registration_ratio_threshold"]),
    )
    direct_accepted = direct_matrix is not None and _accepted(direct_metrics, config["direct_registration_acceptance"])
    registration_status = "spot_level_registration_pass" if direct_accepted else "not_evaluable_spot_level_registration"

    source_corners = np.float32(
        [
            [0, 0],
            [config["expected_spatial_image_size"][0] - 1, 0],
            [config["expected_spatial_image_size"][0] - 1, config["expected_spatial_image_size"][1] - 1],
            [0, config["expected_spatial_image_size"][1] - 1],
        ]
    )
    approximate_spatial_to_he = cv2.getPerspectiveTransform(source_corners, roi_he_original.astype(np.float32))
    positions: list[list[str]] = []
    with (root / config["source_positions"]).open(encoding="utf-8", newline="") as handle:
        positions = list(csv.reader(handle))
    if len(positions) != int(config["expected_positions"]):
        raise ValueError("Unexpected number of GSE278007 source positions")
    centers = np.float32([[float(row[4]), float(row[5])] for row in positions]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(centers, approximate_spatial_to_he)[:, 0, :]
    centers_in_bounds = bool(
        ((mapped[:, 0] >= 0) & (mapped[:, 0] < he_size[0]) & (mapped[:, 1] >= 0) & (mapped[:, 1] < he_size[1])).all()
    )

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    qc_path = output / "registration_qc.tsv"
    with qc_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", *he_metrics], delimiter="\t")
        writer.writeheader()
        for stage, metrics in (
            ("author_he_panel_to_source_he", he_metrics),
            ("source_spatial_to_author_spatial_panel", spatial_metrics),
            ("adjacent_section_direct_local_registration", direct_metrics),
        ):
            writer.writerow({"stage": stage, **metrics})
    transform_path = output / "registration_transform.json"
    transform_payload = {
        "schema_version": 1,
        "status": registration_status,
        "section_relationship": config["section_relationship"],
        "author_roi_localization_passed": True,
        "direct_local_registration_passed": direct_accepted,
        "he_original_size": list(he_size),
        "he_thumbnail_size": [he_bgr.shape[1], he_bgr.shape[0]],
        "spatial_image_size": config["expected_spatial_image_size"],
        "author_roi_corners_he_original_xy": roi_he_original.tolist(),
        "approximate_spatial_image_to_he_homography": approximate_spatial_to_he.tolist(),
        "approximate_transform_evaluation_eligible": direct_accepted,
        "mapped_centers_in_he_bounds": centers_in_bounds,
        "mapped_centers": len(mapped),
        "interpretation": "whole_ROI_localized_but_adjacent_section_spot_correspondence_requires_direct_registration_pass",
    }
    _write_json(transform_path, transform_payload)
    schema_path = output / "image_schema.json"
    _write_json(
        schema_path,
        {
            "schema_version": 1,
            "status": "pass",
            "he_format": "TIFF",
            "he_size": list(he_size),
            "he_mode": he_mode,
            "he_frames": he_frames,
            "he_compressed_size_bytes": (root / config["source_he"]).stat().st_size,
            "spatial_image_size": config["expected_spatial_image_size"],
            "positions": len(positions),
            "position_coordinates_unique": len({(row[4], row[5]) for row in positions}) == len(positions),
            "section_relationship": config["section_relationship"],
        },
    )
    artifacts = [qc_path, transform_path, schema_path]
    evidence_path = output / "gse278007_melanoma_he_preflight_evidence.json"
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3F_GSE278007_melanoma_HE_preflight",
        "status": "pass",
        "registration_status": registration_status,
        "section_relationship": config["section_relationship"],
        "author_roi_localization_passed": True,
        "direct_local_registration_passed": direct_accepted,
        "spot_level_morphology_evaluation_authorized": False,
        "pathology_region_labels_available": False,
        "pathology_accuracy_claim_allowed": False,
        "artifacts": {
            path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts
        },
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "protocol": {"path": config["protocol"], "sha256": config["protocol_sha256"]},
        "download_receipt": {"path": config["download_receipt"], "sha256": config["download_receipt_sha256"]},
        "runtime_lock": {"path": config["runtime_lock"], "sha256": config["runtime_lock_sha256"]},
        "source_he": {"path": config["source_he"], "sha256": config["source_he_sha256"]},
        "author_figure": {"path": config["author_figure"], "sha256": config["author_figure_sha256"]},
        "adapted_author_figure_redistributed": False,
        "image_feature_extraction_run": False,
        "method_outputs_accessed": False,
        "atac_fragments_accessed": False,
        "rna_accessed": False,
        "clusters_accessed": False,
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
    parser.add_argument("--config", default="configs/g4_m3f_gse278007_melanoma_he_preflight.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(run_preflight(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
