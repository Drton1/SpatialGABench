"""Construction-only matched representations for the released GSE288178 MALT subset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import h5py
import anndata as ad
import numpy as np
import pandas as pd
import yaml


REQUIRED_INPUTS = (
    "authorization",
    "source_h5ad",
    "source_locs",
    "chrom_sizes",
    "blacklist",
)
FORBIDDEN_CONFIG_TOKENS = (
    "gse279771",
    "gse308623",
    "handcrafted_he",
    "image_feature",
    "source_he",
    "source_mask",
    "rna",
    "pathology",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_construction_config(config_path: str | Path, root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    lowered_keys = "\n".join(str(key).lower() for key in config)
    leaked = [token for token in FORBIDDEN_CONFIG_TOKENS if token in lowered_keys]
    if leaked:
        raise PermissionError(f"Forbidden evaluation/holdout keys in MALT construction config: {leaked}")

    for field in REQUIRED_INPUTS:
        artifact = root / config[field]
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing frozen MALT construction input: {artifact}")
        expected = str(config[f"{field}_sha256"])
        if sha256_file(artifact) != expected:
            raise ValueError(f"Frozen MALT construction checksum mismatch: {field}")

    authorization = json.loads((root / config["authorization"]).read_text(encoding="utf-8"))
    expected_representations = set(config["construction_contract"]["representations"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("scope")
        != "development_accessibility_only_matched_representation_construction"
        or authorization.get("study_id") != config["study_id"]
        or authorization.get("sample_id") != config["sample_id"]
        or set(authorization.get("representations", [])) != expected_representations
        or int(authorization["authorized_subset"]["released_coordinate_whitelist"])
        != int(config["expected_selected_observations"])
    ):
        raise PermissionError("M3F MALT matched-representation authorization is not valid")
    config["_path"] = path
    config["_root"] = root
    return config


def export_released_fragment_subset(
    config_path: str | Path,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Export only source rows named by the checksum-bound released coordinate whitelist."""

    config = load_construction_config(config_path, root)
    root = Path(config["_root"])
    h5ad_path = root / config["source_h5ad"]
    locs_path = root / config["source_locs"]
    output = root / config["output_root"] / "adapted"
    output.mkdir(parents=True, exist_ok=True)
    fragments_path = output / f"{config['dataset_id']}__released_7825.fragments.tsv.gz"
    positions_path = output / "tissue_positions_list.csv"
    mapping_path = output / "barcode_spot_image_coordinate_mapping.tsv"
    audit_path = output / "released_subset_fragment_adapter_audit.json"
    audit_tsv_path = output / "released_subset_fragment_adapter_audit.tsv"
    temporary_fragments = fragments_path.with_name(f".{fragments_path.name}.tmp")
    temporary_positions = positions_path.with_name(f".{positions_path.name}.tmp")
    temporary_mapping = mapping_path.with_name(f".{mapping_path.name}.tmp")

    locs = pd.read_csv(locs_path, sep="\t", dtype={"spot": str})
    if list(locs.columns) != ["spot", "x", "y"] or locs["spot"].duplicated().any():
        raise ValueError("Released MALT coordinate whitelist schema is invalid")
    if len(locs) != int(config["expected_selected_observations"]):
        raise ValueError("Released MALT coordinate whitelist row count changed")
    if not np.isfinite(locs[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("Released MALT image coordinates are not finite")

    fragment_rows = 0
    invalid_fragment_rows = 0
    chromosome_rows: dict[str, int] = {}
    with h5py.File(h5ad_path, "r") as handle:
        required = (
            "obs/_index",
            "obs/x_coord",
            "obs/y_coord",
            "obs/n_fragment",
            "obsm/fragment_paired/data",
            "obsm/fragment_paired/indices",
            "obsm/fragment_paired/indptr",
            "uns/reference_sequences/reference_seq_name",
            "uns/reference_sequences/reference_seq_length",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"Fragment-backed MALT H5AD is missing required paths: {missing}")
        if "X" in handle:
            raise ValueError("MALT construction expected a fragment-only H5AD without X")

        observation_ids = handle["obs/_index"].asstr()[:]
        source_x = np.asarray(handle["obs/x_coord"][:], dtype=np.int64)
        source_y = np.asarray(handle["obs/y_coord"][:], dtype=np.int64)
        n_fragment = np.asarray(handle["obs/n_fragment"][:], dtype=np.int64)
        n_observations = len(observation_ids)
        if n_observations != int(config["expected_source_observations"]):
            raise ValueError("MALT source observation count changed")
        if len(set(observation_ids)) != n_observations:
            raise ValueError("MALT source contains duplicate barcodes")
        source_spots = np.char.add(np.char.add(source_x.astype(str), "x"), source_y.astype(str))
        if len(set(source_spots)) != n_observations:
            raise ValueError("MALT source grid-coordinate spot IDs are not unique")
        source_lookup = {spot: index for index, spot in enumerate(source_spots)}
        unknown_spots = sorted(set(locs["spot"]) - set(source_lookup))
        if unknown_spots:
            raise ValueError(f"Released MALT spots are absent from H5AD: {unknown_spots[:5]}")
        selected_indexes = np.asarray([source_lookup[spot] for spot in locs["spot"]], dtype=np.int64)
        if source_spots[selected_indexes].tolist() != locs["spot"].tolist():
            raise ValueError("MALT whitelist-to-source spot mapping is not exact")
        if not np.all(np.diff(selected_indexes) > 0):
            raise ValueError("Released MALT whitelist no longer preserves source H5AD order")

        fragment_group = handle["obsm/fragment_paired"]
        data = fragment_group["data"]
        indices = fragment_group["indices"]
        indptr = np.asarray(fragment_group["indptr"][:], dtype=np.int64)
        chromosome_names = handle["uns/reference_sequences/reference_seq_name"].asstr()[:]
        chromosome_lengths = np.asarray(
            handle["uns/reference_sequences/reference_seq_length"][:], dtype=np.int64
        )
        chromosome_ends = np.cumsum(chromosome_lengths, dtype=np.int64)
        chromosome_starts = np.concatenate(([0], chromosome_ends[:-1]))
        shape = tuple(int(value) for value in fragment_group.attrs.get("shape", ()))
        if shape != (n_observations, int(chromosome_ends[-1])):
            raise ValueError("MALT fragment CSR shape differs from its reference contract")
        if len(indptr) != n_observations + 1 or indptr[0] != 0 or indptr[-1] != len(indices):
            raise ValueError("MALT fragment CSR indptr contract is invalid")
        if len(indices) != len(data):
            raise ValueError("MALT fragment CSR indices/data lengths differ")

        encoded_rows = indptr[selected_indexes + 1] - indptr[selected_indexes]
        expected_encoded_rows = int(encoded_rows.sum())
        if expected_encoded_rows != int(config["expected_selected_encoded_fragment_rows"]):
            raise ValueError("MALT selected encoded-fragment row count changed")
        selected_n_fragment_sum = int(n_fragment[selected_indexes].sum())
        if selected_n_fragment_sum != int(config["expected_selected_n_fragment_sum"]):
            raise ValueError("MALT selected obs/n_fragment sum changed")

        mapping = pd.DataFrame(
            {
                "barcode": observation_ids[selected_indexes],
                "spot": source_spots[selected_indexes],
                "source_h5ad_index": selected_indexes,
                "released_order": np.arange(len(selected_indexes), dtype=np.int64),
                "source_x_coord": source_x[selected_indexes],
                "source_y_coord": source_y[selected_indexes],
                "image_x": locs["x"].to_numpy(),
                "image_y": locs["y"].to_numpy(),
                "source_n_fragment": n_fragment[selected_indexes],
                "encoded_fragment_rows": encoded_rows,
            }
        )
        if mapping["barcode"].duplicated().any() or mapping["spot"].duplicated().any():
            raise ValueError("MALT released-subset mapping is not one-to-one")
        mapping.to_csv(temporary_mapping, sep="\t", index=False)
        positions = pd.DataFrame(
            {
                "barcode": mapping["barcode"],
                "in_tissue": 1,
                "array_row": mapping["source_y_coord"],
                "array_col": mapping["source_x_coord"],
                "pixel_row": mapping["image_y"],
                "pixel_col": mapping["image_x"],
            }
        )
        positions.to_csv(temporary_positions, index=False, header=False)

        with temporary_fragments.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as target:
                    for source_index, barcode in zip(selected_indexes, mapping["barcode"], strict=True):
                        row_start = int(indptr[source_index])
                        row_end = int(indptr[source_index + 1])
                        starts_global = np.asarray(indices[row_start:row_end], dtype=np.int64)
                        lengths = np.asarray(data[row_start:row_end], dtype=np.int64)
                        chromosome_indexes = np.searchsorted(chromosome_ends, starts_global, side="right")
                        valid_chromosome = chromosome_indexes < len(chromosome_names)
                        local_starts = np.zeros_like(starts_global)
                        local_starts[valid_chromosome] = (
                            starts_global[valid_chromosome]
                            - chromosome_starts[chromosome_indexes[valid_chromosome]]
                        )
                        local_ends = local_starts + lengths
                        valid = valid_chromosome & (lengths > 0) & (local_starts >= 0)
                        valid_indexes = np.flatnonzero(valid)
                        if len(valid_indexes):
                            valid[valid_indexes] &= local_ends[valid_indexes] <= chromosome_lengths[
                                chromosome_indexes[valid_indexes]
                            ]
                        invalid_fragment_rows += int((~valid).sum())
                        valid_indexes = np.flatnonzero(valid)
                        if len(valid_indexes):
                            names = chromosome_names[chromosome_indexes[valid_indexes]].astype(str)
                            target.write(
                                "".join(
                                    f"{chromosome}\t{start}\t{end}\t{barcode}\t1\n"
                                    for chromosome, start, end in zip(
                                        names,
                                        local_starts[valid_indexes],
                                        local_ends[valid_indexes],
                                        strict=True,
                                    )
                                )
                            )
                            fragment_rows += len(valid_indexes)
                            unique_names, counts = np.unique(names, return_counts=True)
                            for chromosome, count in zip(unique_names, counts, strict=True):
                                chromosome_rows[str(chromosome)] = (
                                    chromosome_rows.get(str(chromosome), 0) + int(count)
                                )

    if fragment_rows != int(config["expected_selected_encoded_fragment_rows"]):
        raise ValueError("Exported MALT fragment rows differ from selected CSR rows")
    if invalid_fragment_rows != 0:
        raise ValueError("MALT selected CSR contains invalid encoded fragments")
    temporary_fragments.replace(fragments_path)
    temporary_positions.replace(positions_path)
    temporary_mapping.replace(mapping_path)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "adapter": "gse288178_malt_released_coordinate_subset_fragment_paired_v1",
        "authorization": config["authorization"],
        "authorization_sha256": config["authorization_sha256"],
        "source_h5ad": config["source_h5ad"],
        "source_h5ad_sha256": config["source_h5ad_sha256"],
        "source_locs": config["source_locs"],
        "source_locs_sha256": config["source_locs_sha256"],
        "source_observations": int(config["expected_source_observations"]),
        "selected_observations": int(config["expected_selected_observations"]),
        "excluded_observations": int(config["expected_source_observations"])
        - int(config["expected_selected_observations"]),
        "unexplained_source_exclusions_retained": int(
            config["expected_unexplained_source_exclusions"]
        ),
        "selected_obs_n_fragment_sum": int(config["expected_selected_n_fragment_sum"]),
        "selected_encoded_fragment_rows": fragment_rows,
        "encoded_minus_obs_n_fragment": fragment_rows
        - int(config["expected_selected_n_fragment_sum"]),
        "invalid_fragment_rows": invalid_fragment_rows,
        "chromosome_rows": chromosome_rows,
        "output_fragments": str(fragments_path.relative_to(root)),
        "output_fragments_sha256": sha256_file(fragments_path),
        "output_positions": str(positions_path.relative_to(root)),
        "output_positions_sha256": sha256_file(positions_path),
        "output_mapping": str(mapping_path.relative_to(root)),
        "output_mapping_sha256": sha256_file(mapping_path),
        "observation_order": "released_locs_order_equals_source_h5ad_restricted_order",
        "barcode_role": "actual_source_h5ad_barcode",
        "spot_role": "source_grid_coordinate_whitelist_id",
        "pixel_coordinate_role": "released_serial_section_he_coordinates_evaluation_join_key_only",
        "coordinate_decode": "global_start_to_reference_local_start_plus_stored_length",
        "construction_leakage_inputs": "none",
        "he_pixels_accessed": False,
        "mask_values_accessed": False,
        "image_features_accessed": False,
        "rna_accessed": False,
        "labels_or_clusters_accessed": False,
        "holdout_accessed": False,
        "full_benchmark_run": False,
        "ranking_run": False,
    }
    _write_json(audit_path, result)
    flat = {key: value for key, value in result.items() if not isinstance(value, dict)}
    pd.DataFrame([flat]).to_csv(audit_tsv_path, sep="\t", index=False)
    return result


def summarize_matched_representations(
    config_path: str | Path,
    root: str | Path = ".",
) -> Path:
    """Independently validate the completed construction artifacts and observation contract."""

    config = load_construction_config(config_path, root)
    root = Path(config["_root"])
    output = root / config["output_root"]
    adapted = output / "adapted"
    filtered = output / "filtered_fragments"
    frozen_peaks = output / "frozen_peaks"
    matched = output / "matched"
    audit_path = adapted / "released_subset_fragment_adapter_audit.json"
    filter_path = filtered / "fragment_filter_audit.json"
    peak_path = frozen_peaks / "peak_freeze_audit.json"
    validation_path = matched / "matched_representation_validation.json"
    manifest_path = matched / "representation_manifest.tsv"
    mapping_path = adapted / "barcode_spot_image_coordinate_mapping.tsv"
    required = (audit_path, filter_path, peak_path, validation_path, manifest_path, mapping_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MALT matched-representation artifacts: {missing}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fragment_filter = json.loads(filter_path.read_text(encoding="utf-8"))
    peak_audit = json.loads(peak_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path, sep="\t")
    mapping = pd.read_csv(mapping_path, sep="\t", dtype={"barcode": str, "spot": str})
    expected_representations = set(config["construction_contract"]["representations"])
    nuclear_fragment_rows = int(config["expected_selected_n_fragment_sum"])
    mitochondrial_fragment_rows = int(audit.get("chromosome_rows", {}).get("chrM", -1))
    source_encoded_rows = int(config["expected_selected_encoded_fragment_rows"])

    filtered_fragment_artifact = Path(str(fragment_filter.get("output_fragments", "")))
    if not filtered_fragment_artifact.is_absolute():
        filtered_fragment_artifact = root / filtered_fragment_artifact

    shared_checks = {
        "subset_adapter_pass": audit.get("status") == "pass",
        "subset_rows_exact": audit.get("selected_observations")
        == int(config["expected_selected_observations"]),
        "subset_exclusions_exact": audit.get("excluded_observations")
        == int(config["expected_excluded_observations"]),
        "source_subset_mapping_current": audit.get("output_mapping_sha256")
        == sha256_file(mapping_path),
        "filter_pass": fragment_filter.get("status") == "pass",
        "filter_observations_exact": fragment_filter.get("in_tissue_barcode_rows")
        == int(config["expected_selected_observations"]),
        "filter_source_rows_exact": fragment_filter.get("source_fragment_rows")
        == source_encoded_rows,
        "filter_nuclear_rows_exact": fragment_filter.get("included_fragment_rows")
        == nuclear_fragment_rows,
        "filter_only_excludes_mitochondrial": fragment_filter.get("excluded_chromosome_rows")
        == mitochondrial_fragment_rows
        == source_encoded_rows - nuclear_fragment_rows,
        "filter_no_barcode_exclusions": fragment_filter.get("excluded_barcode_rows") == 0,
        "filter_no_invalid_rows": fragment_filter.get("invalid_fragment_rows") == 0,
        "filtered_fragments_current": fragment_filter.get("output_fragments_sha256")
        == sha256_file(filtered_fragment_artifact),
        "peak_freeze_pass": peak_audit.get("status") == "pass",
        "matched_validation_pass": validation.get("status") == "pass",
        "matched_fragment_filter_verified": validation.get("fragment_filter_contract_verified") is True,
        "matched_rows_exact": validation.get("fragment_rows") == nuclear_fragment_rows,
        "matched_observations_exact": validation.get("barcode_rows")
        == int(config["expected_selected_observations"]),
        "matched_no_invalid_or_excluded_rows": all(
            validation.get(field) == 0
            for field in (
                "invalid_fragment_rows",
                "excluded_barcode_rows",
                "excluded_chromosome_rows",
            )
        ),
        "manifest_representations_exact": set(manifest["representation_id"])
        == expected_representations,
        "manifest_rows_exact": len(manifest) == len(expected_representations),
        "mapping_rows_exact": len(mapping) == int(config["expected_selected_observations"]),
        "mapping_one_to_one": mapping["barcode"].is_unique and mapping["spot"].is_unique,
    }

    rows: list[dict[str, Any]] = []
    expected_barcodes = pd.Index(mapping["barcode"], dtype=str)
    expected_spatial = mapping[["image_x", "image_y"]].to_numpy(dtype=float)
    for record in manifest.itertuples(index=False):
        representation_id = str(record.representation_id)
        path = Path(str(record.output_path))
        if not path.is_absolute():
            path = root / path
        current_hash = sha256_file(path) if path.is_file() else ""
        checks: dict[str, bool] = {
            "manifest_status": str(record.status) == "pass",
            "output_exists": path.is_file(),
            "output_hash_current": current_hash == str(record.output_sha256),
            "manifest_observations_exact": int(record.n_observations)
            == int(config["expected_selected_observations"]),
            "manifest_count_contract": int(record.matrix_total_counts)
            == int(record.assigned_fragment_rows),
        }
        if path.is_file():
            data = ad.read_h5ad(path, backed="r")
            observed_spatial = np.asarray(data.obsm["spatial"])
            forbidden_obs = [
                str(column)
                for column in data.obs.columns
                if any(
                    token in str(column).lower()
                    for token in ("rna", "label", "cluster", "pathology", "image_feature")
                )
            ]
            checks.update(
                {
                    "h5ad_observations_exact": data.n_obs
                    == int(config["expected_selected_observations"]),
                    "h5ad_features_exact": data.n_vars == int(record.n_features),
                    "barcode_identity_and_order_exact": data.obs_names.equals(expected_barcodes),
                    "spatial_join_coordinates_exact": observed_spatial.shape == expected_spatial.shape
                    and np.array_equal(observed_spatial, expected_spatial),
                    "identity_unique": data.obs_names.is_unique and data.var_names.is_unique,
                    "representation_identity_exact": data.uns.get("representation_id")
                    == representation_id,
                    "genome_build_exact": data.uns.get("genome_build") == config["genome_build"],
                    "same_filtered_fragment_source": data.uns.get("representation_source")
                    == "same_filtered_fragments",
                    "construction_leakage_absent": data.uns.get("construction_leakage_inputs")
                    == "none",
                    "forbidden_observation_columns_absent": not forbidden_obs,
                }
            )
            data.file.close()
        rows.append(
            {
                "dataset_id": config["dataset_id"],
                "representation_id": representation_id,
                "input_track": record.input_track,
                "n_observations": int(record.n_observations),
                "n_features": int(record.n_features),
                "matrix_nonzero_entries": int(record.matrix_nonzero_entries),
                "matrix_total_counts": int(record.matrix_total_counts),
                "eligible_fragment_fraction_assigned": float(
                    record.eligible_fragment_fraction_assigned
                ),
                "output_path": str(path.relative_to(root)),
                "output_sha256": current_hash,
                "checks_passed": sum(checks.values()),
                "checks_total": len(checks),
                "failed_checks": ";".join(key for key, passed in checks.items() if not passed),
                "status": "pass" if all(checks.values()) else "fail",
            }
        )

    summary = pd.DataFrame(rows).sort_values("representation_id", kind="mergesort")
    evidence_dir = output / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary_path = evidence_dir / "matched_representation_summary.tsv"
    checks_path = evidence_dir / "construction_checks.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    pd.DataFrame(
        [
            {"check_id": name, "status": "pass" if passed else "fail"}
            for name, passed in shared_checks.items()
        ]
    ).to_csv(checks_path, sep="\t", index=False)
    status = "pass" if all(shared_checks.values()) and (summary["status"] == "pass").all() else "fail"
    adapter_fragments = root / str(audit["output_fragments"])
    adapter_positions = root / str(audit["output_positions"])
    frozen_peak_bed = (
        frozen_peaks / f"{config['dataset_id']}__macs3_summit_501bp.bed"
    )
    representation_outputs = [root / str(path) for path in manifest["output_path"]]
    bound_artifacts = [
        *required,
        adapter_fragments,
        adapter_positions,
        filtered_fragment_artifact,
        frozen_peak_bed,
        *representation_outputs,
    ]
    if not all(path.is_file() for path in bound_artifacts):
        raise FileNotFoundError("A bound MALT construction artifact disappeared before evidence freeze")
    evidence_path = evidence_dir / "gse288178_malt_matched_representation_evidence.json"
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3F_GSE288178_MALT_matched_representation_construction",
        "status": status,
        "dataset_id": config["dataset_id"],
        "source_observations": int(config["expected_source_observations"]),
        "released_subset_observations": int(config["expected_selected_observations"]),
        "excluded_source_observations": int(config["expected_excluded_observations"]),
        "unexplained_source_exclusions_retained": int(
            config["expected_unexplained_source_exclusions"]
        ),
        "source_encoded_fragment_rows": source_encoded_rows,
        "mitochondrial_fragment_rows_excluded": mitochondrial_fragment_rows,
        "nuclear_fragment_rows_constructed": nuclear_fragment_rows,
        "obs_n_fragment_equals_standard_chromosome_rows": nuclear_fragment_rows
        == source_encoded_rows - mitochondrial_fragment_rows,
        "representations_expected": len(expected_representations),
        "representations_passed": int((summary["status"] == "pass").sum()),
        "shared_checks_passed": sum(shared_checks.values()),
        "shared_checks_total": len(shared_checks),
        "representation_checks_passed": int(summary["checks_passed"].sum()),
        "representation_checks_total": int(summary["checks_total"].sum()),
        "summary": {
            "path": str(summary_path.relative_to(root)),
            "sha256": sha256_file(summary_path),
        },
        "checks": {"path": str(checks_path.relative_to(root)), "sha256": sha256_file(checks_path)},
        "construction_artifacts": {
            str(path.relative_to(root)): sha256_file(path) for path in bound_artifacts
        },
        "authorization": {
            "path": config["authorization"],
            "sha256": config["authorization_sha256"],
        },
        "config": {
            "path": str(Path(config["_path"]).relative_to(root)),
            "sha256": sha256_file(Path(config["_path"])),
        },
        "image_target_accessed": False,
        "he_pixels_accessed": False,
        "mask_values_accessed": False,
        "rna_accessed": False,
        "labels_or_clusters_accessed": False,
        "method_outputs_accessed": False,
        "holdout_accessed": False,
        "morphology_evaluation_run": False,
        "full_real_data_benchmark_run": False,
        "recommendation_policy_fit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(evidence_path, evidence)
    if status != "pass":
        raise RuntimeError("GSE288178 MALT matched-representation validation failed")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m3f_gse288178_malt_matched_representations.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        path = summarize_matched_representations(args.config, args.root)
        print(json.dumps({"output": str(path)}, indent=2, sort_keys=True))
        return 0
    result = export_released_fragment_subset(args.config, args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
