"""Validate source-aligned GSE288178 MALT image coordinates without reading counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml
from PIL import Image


FROZEN_FIELDS = (
    "authorization",
    "protocol",
    "download_receipt",
    "source_metadata",
    "source_h5ad",
    "source_he",
    "source_mask",
    "source_locs",
)
FORBIDDEN_FLAGS = (
    "h5ad_X_read",
    "fragment_values_read",
    "source_cluster_values_read",
    "image_feature_extraction_run",
    "method_outputs_accessed",
    "rna_accessed",
    "pathology_labels_accessed",
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
            raise ValueError(f"Frozen M3F MALT artifact checksum mismatch: {field}")
    authorization = json.loads((root / config["authorization"]).read_text(encoding="utf-8"))
    if (
        authorization.get("status") != "authorized"
        or authorization.get("scope") != "development_source_schema_and_image_coordinate_alignment_only"
    ):
        raise PermissionError("M3F MALT input-preflight authorization is not active")
    enabled = [field for field in FORBIDDEN_FLAGS if bool(config.get(field))]
    if enabled:
        raise PermissionError(f"Forbidden M3F MALT preflight flags enabled: {enabled}")
    config["_path"] = path
    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config = _load(config_path, root)
    h5ad_path = root / config["source_h5ad"]
    with h5py.File(h5ad_path, "r") as handle:
        x_absent = "X" not in handle
        fragment_backing_present = "obsm" in handle and "fragment_paired" in handle["obsm"]
    if not x_absent:
        raise ValueError("M3F MALT preflight expected a fragment-backed H5AD without X")

    source = ad.read_h5ad(h5ad_path, backed="r")
    expected_shape = (
        int(config["expected_h5ad_observations"]),
        int(config["expected_h5ad_variables"]),
    )
    if source.shape != expected_shape or not source.obs_names.is_unique:
        raise ValueError(f"Unexpected GSE288178 MALT H5AD identity contract: {source.shape}")
    required_obs = {"n_fragment", "x_coord", "y_coord"}
    if not required_obs <= set(source.obs.columns):
        raise ValueError("GSE288178 MALT source observation schema is incomplete")
    if "spatial" not in source.obsm or not fragment_backing_present:
        raise ValueError("GSE288178 MALT source backing schema is incomplete")
    obs = source.obs[["n_fragment", "x_coord", "y_coord"]].copy()
    obs["spot"] = obs.x_coord.astype(str) + "x" + obs.y_coord.astype(str)
    source.file.close()

    locs = pd.read_csv(root / config["source_locs"], sep="\t")
    if len(locs) != int(config["expected_source_locs"]) or not locs.spot.is_unique:
        raise ValueError("GSE288178 MALT released H&E locations differ from the frozen schema")
    loc_set = set(locs.spot.astype(str))
    h5ad_set = set(obs.spot.astype(str))
    if loc_set - h5ad_set:
        raise ValueError("Released MALT H&E locations are absent from the GEO H5AD")
    restricted = obs[obs.spot.isin(loc_set)]
    order_matches = restricted.spot.tolist() == locs.spot.astype(str).tolist()
    if not order_matches:
        raise ValueError("Released MALT H&E location order differs from the GEO H5AD")
    joined = locs.join(obs.set_index("spot")[["n_fragment", "x_coord", "y_coord"]], on="spot")
    under_15000 = int((obs.n_fragment < 15000).sum())
    locs_under_15000 = bool((joined.n_fragment < 15000).all())
    unexplained = under_15000 - len(locs)
    if (
        under_15000 != int(config["expected_n_fragment_lt_15000"])
        or unexplained != int(config["expected_unexplained_source_exclusions"])
        or not locs_under_15000
    ):
        raise ValueError("GSE288178 MALT source-subset discrepancy differs from the frozen audit")

    coordinate_models: dict[str, dict[str, float]] = {}
    for source_column, image_column in (("x_coord", "x"), ("y_coord", "y")):
        coefficient = np.polyfit(joined[source_column].to_numpy(), joined[image_column].to_numpy(), 1)
        predicted = np.polyval(coefficient, joined[source_column].to_numpy())
        residual = joined[image_column].to_numpy() - predicted
        r_squared = float(np.corrcoef(joined[source_column], joined[image_column])[0, 1] ** 2)
        max_residual = float(np.max(np.abs(residual)))
        if (
            r_squared < float(config["minimum_linear_coordinate_r_squared"])
            or max_residual > float(config["maximum_linear_coordinate_residual_px"])
        ):
            raise ValueError("GSE288178 MALT source coordinate transform is not sufficiently linear")
        coordinate_models[image_column] = {
            "slope": float(coefficient[0]),
            "intercept": float(coefficient[1]),
            "r_squared": r_squared,
            "maximum_absolute_residual_px": max_residual,
        }

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(root / config["source_he"]) as image:
        he_size = tuple(image.size)
        he_mode = image.mode
    with Image.open(root / config["source_mask"]) as image:
        mask_size = tuple(image.size)
        mask = np.asarray(image)
    if list(he_size) != config["expected_he_size"] or list(mask_size) != config["expected_mask_size"]:
        raise ValueError("GSE288178 MALT H&E/mask dimensions differ from the frozen audit")
    mask_values = np.unique(mask)
    if not np.array_equal(mask_values, np.array([0, 1], dtype=mask_values.dtype)):
        raise ValueError("GSE288178 MALT mask is not binary tissue support")
    scale = int(config["expected_image_to_mask_scale"])
    if he_size[0] != mask_size[0] * scale or he_size[1] != mask_size[1] * scale:
        raise ValueError("GSE288178 MALT image/mask scale differs from 16")
    mask_x = np.clip(np.rint(locs.x.to_numpy() / scale).astype(int), 0, mask_size[0] - 1)
    mask_y = np.clip(np.rint(locs.y.to_numpy() / scale).astype(int), 0, mask_size[1] - 1)
    foreground = int((mask[mask_y, mask_x] == 1).sum())
    foreground_fraction = foreground / len(locs)
    if (
        foreground != int(config["expected_mask_foreground_at_locs"])
        or foreground_fraction < float(config["minimum_foreground_fraction"])
    ):
        raise ValueError("GSE288178 MALT tissue-support coverage differs from the frozen audit")

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    alignment_path = output / "alignment_summary.tsv"
    pd.DataFrame(
        [
            {"check": "released_locs_in_h5ad", "value": len(locs), "status": "pass"},
            {"check": "released_locs_order_matches_h5ad", "value": order_matches, "status": "pass"},
            {"check": "h5ad_n_fragment_lt_15000", "value": under_15000, "status": "pass"},
            {"check": "unexplained_exclusions_after_published_filter", "value": unexplained, "status": "disclosed"},
            {"check": "mask_foreground_at_locs", "value": foreground, "status": "pass"},
            {"check": "mask_foreground_fraction", "value": foreground_fraction, "status": "pass"},
        ]
    ).to_csv(alignment_path, sep="\t", index=False)
    validation_path = output / "input_preflight_validation.json"
    _write_json(
        validation_path,
        {
            "schema_version": 1,
            "status": "pass_with_source_subset_discrepancy",
            "h5ad_shape": list(expected_shape),
            "h5ad_X_absent": x_absent,
            "h5ad_X_read": False,
            "fragment_backing_present_but_values_not_read": fragment_backing_present,
            "released_locs": len(locs),
            "released_locs_order_matches_h5ad": order_matches,
            "published_filter_n_fragment_lt_15000": under_15000,
            "unexplained_exclusions_after_published_filter": unexplained,
            "coordinate_models": coordinate_models,
            "he_size": list(he_size),
            "he_mode": he_mode,
            "mask_size": list(mask_size),
            "mask_role": "binary_tissue_support_not_pathology_label",
            "mask_foreground_at_locs": foreground,
            "mask_foreground_fraction": foreground_fraction,
        },
    )
    artifacts = [alignment_path, validation_path]
    evidence_path = output / "gse288178_malt_input_preflight_evidence.json"
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3F_GSE288178_MALT_input_preflight",
        "status": "pass_with_source_subset_discrepancy",
        "coordinate_alignment_status": "pass_released_7825_spot_subset",
        "released_locs": len(locs),
        "published_filter_expected_locs": under_15000,
        "unexplained_source_exclusions": unexplained,
        "mask_foreground_at_locs": foreground,
        "mask_foreground_fraction": foreground_fraction,
        "mask_role": "binary_tissue_support_not_pathology_label",
        "source_manual_alignment_accepted_for_image_only_development_preflight": True,
        "independent_pathology_truth_available": False,
        "image_feature_extraction_authorized": False,
        "spot_level_method_comparison_authorized": False,
        "artifacts": {
            path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts
        },
        "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
        "protocol": {"path": config["protocol"], "sha256": config["protocol_sha256"]},
        "download_receipt": {"path": config["download_receipt"], "sha256": config["download_receipt_sha256"]},
        "source_h5ad": {"path": config["source_h5ad"], "sha256": config["source_h5ad_sha256"]},
        "h5ad_X_read": False,
        "fragment_values_read": False,
        "source_cluster_values_read": False,
        "image_feature_extraction_run": False,
        "method_outputs_accessed": False,
        "rna_accessed": False,
        "pathology_labels_accessed": False,
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
    parser.add_argument("--config", default="configs/g4_m3f_gse288178_malt_input_preflight.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(run_preflight(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
