"""Molecularly blind preflight of embedded pathologist outline assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy import ndimage

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


def extract_outline(
    rgb: np.ndarray,
    target_rgb: tuple[int, int, int] | list[int],
    color_distance_threshold: float,
    minimum_connected_outline_pixels: int,
    closure_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract saturated annotation-line components near a frozen RGB color."""
    values = np.asarray(rgb, dtype=np.int32)
    target = np.asarray(target_rgb, dtype=np.int32)
    distance = np.sqrt(np.square(values - target).sum(axis=2))
    raw = distance < float(color_distance_threshold)
    labels, _ = ndimage.label(raw, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    qualifying = np.flatnonzero(sizes >= int(minimum_connected_outline_pixels))
    qualifying = qualifying[qualifying != 0]
    line_mask = np.isin(labels, qualifying)
    if line_mask.any():
        closed = ndimage.binary_closing(
            line_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(closure_iterations),
        )
        filled_diagnostic = ndimage.binary_fill_holes(closed)
        filled_pixels = int(filled_diagnostic.sum())
        ratio = float(filled_pixels / line_mask.sum())
    else:
        filled_pixels = 0
        ratio = 0.0
    return line_mask, {
        "raw_color_match_pixels": int(raw.sum()),
        "qualifying_components": len(qualifying),
        "outline_line_pixels": int(line_mask.sum()),
        "diagnostic_fill_pixels": filled_pixels,
        "diagnostic_fill_to_line_ratio": ratio,
    }


def run_preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    for field in ("authorization", "previous_gate", "source_docx", "source_figure"):
        if sha256_file(root / config[field]) != config[f"{field}_sha256"]:
            raise ValueError(f"Frozen pathology source-preflight input changed: {field}")
    authorization = _json(root / config["authorization"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("scope")
        != "source_only_pathologist_outline_inventory_and_digitization_preflight"
        or authorization.get("candidate_evaluation_authorized") is not False
        or authorization.get("frozen_annotation_classes") != config["annotation_classes"]
    ):
        raise PermissionError("Pathology source-preflight authorization boundary is invalid")
    for key in (
        "color_distance_threshold",
        "minimum_connected_outline_pixels",
        "closure_iterations",
        "source_image_width",
        "source_image_height",
    ):
        if authorization["frozen_parameters"][key] != config[key]:
            raise ValueError(f"Frozen pathology source-preflight parameter changed: {key}")
    if authorization["frozen_parameters"]["panels"] != len(config["panels"]):
        raise ValueError("Frozen pathology source-preflight panel count changed")
    required_true = (
        "source_is_composite_figure",
        "histology_is_serial_section",
        "outline_line_masks_only",
    )
    if any(config.get(field) is not True for field in required_true):
        raise PermissionError("Pathology source limitations must remain explicit")
    forbidden_flags = (
        "gse_publicly_accessible",
        "filled_region_masks_created",
        "atac_accessed",
        "gene_activity_accessed",
        "rna_accessed",
        "source_clusters_accessed",
        "holdout_accessed",
        "method_comparison_run",
        "full_real_data_benchmark_run",
        "recommendation_policy_fit_run",
        "universal_ranking_run",
        "final_ranking_run",
        "redistribution_authorized",
    )
    if any(bool(config.get(field)) for field in forbidden_flags):
        raise PermissionError("A forbidden pathology source-preflight flag is enabled")

    docx_path = root / config["source_docx"]
    with zipfile.ZipFile(docx_path) as archive:
        member = archive.read(config["source_docx_member"])
    member_sha256 = hashlib.sha256(member).hexdigest()
    if member_sha256 != config["source_figure_sha256"]:
        raise ValueError("Embedded pathology figure checksum changed")
    source_figure_path = root / config["source_figure"]
    if source_figure_path.read_bytes() != member:
        raise ValueError("Extracted pathology source figure differs from DOCX member")
    image = Image.open(io.BytesIO(member)).convert("RGB")
    if image.size != (int(config["source_image_width"]), int(config["source_image_height"])):
        raise ValueError("Embedded pathology source figure dimensions changed")

    output = root / config["output_root"]
    crop_root = output / "panel_crops"
    outline_root = output / "outline_line_masks"
    crop_root.mkdir(parents=True, exist_ok=True)
    outline_root.mkdir(parents=True, exist_ok=True)
    panel_rows: list[dict[str, Any]] = []
    outline_rows: list[dict[str, Any]] = []
    observed_masks = 0
    unexpected_masks = 0
    for panel in config["panels"]:
        sample_id = str(panel["sample_id"])
        crop_box = tuple(int(value) for value in panel["crop"])
        crop = image.crop(crop_box)
        crop_path = crop_root / f"{sample_id}.png"
        crop.save(crop_path)
        crop_array = np.asarray(crop, dtype=np.uint8)
        expected = set(panel["expected_classes"])
        observed: set[str] = set()
        for class_id, color in config["annotation_classes"].items():
            line_mask, qc = extract_outline(
                crop_array,
                color,
                float(config["color_distance_threshold"]),
                int(config["minimum_connected_outline_pixels"]),
                int(config["closure_iterations"]),
            )
            if not line_mask.any():
                continue
            observed.add(class_id)
            observed_masks += 1
            if class_id not in expected:
                unexpected_masks += 1
            mask_path = outline_root / f"{sample_id}__{class_id}.png"
            Image.fromarray(line_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            outline_rows.append(
                {
                    "sample_id": sample_id,
                    "annotation_class": class_id,
                    "caption_rgb": ",".join(str(value) for value in color),
                    "expected_from_frozen_visual_inventory": class_id in expected,
                    **qc,
                    "outline_mask_path": str(mask_path.relative_to(root)),
                    "outline_mask_sha256": sha256_file(mask_path),
                    "filled_region_mask_created": False,
                    "evaluation_ready": False,
                    "status": "pass" if class_id in expected else "unexpected",
                }
            )
        panel_status = "pass" if observed == expected else "fail"
        panel_rows.append(
            {
                "sample_id": sample_id,
                "crop_x0": crop_box[0],
                "crop_y0": crop_box[1],
                "crop_x1": crop_box[2],
                "crop_y1": crop_box[3],
                "crop_width": crop.width,
                "crop_height": crop.height,
                "expected_classes": ",".join(sorted(expected)),
                "observed_classes": ",".join(sorted(observed)),
                "crop_path": str(crop_path.relative_to(root)),
                "crop_sha256": sha256_file(crop_path),
                "status": panel_status,
            }
        )

    panel_frame = pd.DataFrame(panel_rows).sort_values("sample_id", kind="mergesort")
    outline_frame = pd.DataFrame(outline_rows).sort_values(
        ["sample_id", "annotation_class"], kind="mergesort"
    )
    panel_path = output / "pathologist_panel_qc.tsv"
    outline_path = output / "pathologist_outline_inventory.tsv"
    panel_frame.to_csv(panel_path, sep="\t", index=False)
    outline_frame.to_csv(outline_path, sep="\t", index=False)
    evidence_path = output / "pathology_truth_source_preflight_evidence.json"
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G4",
            "milestone": "M3G_independent_pathology_truth_source_preflight",
            "status": "pass"
            if len(panel_frame) == 12
            and (panel_frame["status"] == "pass").all()
            and len(outline_frame) == 20
            and (outline_frame["status"] == "pass").all()
            and unexpected_masks == 0
            else "fail",
            "candidate_study_id": "wang_di_2025_nf_pannet",
            "accession": "GSE304813",
            "source_publication": "bioRxiv_2025.09.11.675640v1",
            "source_figure": "Extended_Data_Figure_7",
            "source_figure_sha256": member_sha256,
            "source_figure_width": image.width,
            "source_figure_height": image.height,
            "panels": len(panel_frame),
            "outline_line_masks": len(outline_frame),
            "annotation_classes": sorted(outline_frame["annotation_class"].unique()),
            "unexpected_outline_masks": unexpected_masks,
            "panel_crops_are_embedded_composite_not_raw_histology": True,
            "histology_is_serial_section": True,
            "gse_publicly_accessible": False,
            "pathologist_annotations_available": True,
            "pathologist_annotations_machine_readable_at_source": False,
            "outline_line_masks_only": True,
            "filled_region_masks_created": False,
            "spot_registration_run": False,
            "candidate_evaluation_ready": False,
            "candidate_recommendation": "request_raw_histology_polygon_coordinates_registration_map_and_reuse_permission",
            "source_rights_note": authorization["source_rights_note"],
            "artifacts": {
                "pathologist_panel_qc.tsv": {
                    "path": str(panel_path.relative_to(root)),
                    "sha256": sha256_file(panel_path),
                },
                "pathologist_outline_inventory.tsv": {
                    "path": str(outline_path.relative_to(root)),
                    "sha256": sha256_file(outline_path),
                },
            },
            "frozen_inputs": {
                field: {"path": config[field], "sha256": config[f"{field}_sha256"]}
                for field in ("authorization", "previous_gate", "source_docx", "source_figure")
            },
            "atac_accessed": False,
            "gene_activity_accessed": False,
            "rna_accessed": False,
            "source_clusters_accessed": False,
            "holdout_accessed": False,
            "method_comparison_run": False,
            "full_real_data_benchmark_run": False,
            "recommendation_policy_fit_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "redistribution_authorized": False,
            "next_milestone_authorized": False,
        },
    )
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/g4_m3g_pathology_truth_source_preflight.yaml"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    path = run_preflight(args.config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

