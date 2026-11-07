"""Input-only audit for an additional G5 external holdout study family."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from spatialgabench.software_wrappers.common import sha256_file


FALSE_FIELDS = (
    "biological_utility_holdout_added",
    "large_fragment_download_run",
    "raw_fastq_or_bam_download_run",
    "method_execution_run",
    "existing_holdout_accessed",
    "policy_refit_run",
    "full_benchmark_run",
    "results_writing_run",
    "universal_ranking_run",
    "final_ranking_run",
)


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check(root: Path, record: dict[str, Any], label: str) -> Path:
    path = root / str(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"G5-H1 candidate artifact mismatch: {label}")
    if "data/raw" in path.as_posix():
        raise PermissionError(f"G5-H1 cannot open a raw-data path: {label}")
    return path


def _load(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _yaml(config_path)
    for key in ("authorization", "previous_gate", "m4_policy"):
        path = root / config[key]
        if not path.is_file() or sha256_file(path) != config[f"{key}_sha256"]:
            raise ValueError(f"G5-H1 frozen contract mismatch: {key}")
    authorization = _json(root / config["authorization"])
    previous = _json(root / config["previous_gate"])
    policy = _json(root / config["m4_policy"])
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "H1_supplemental_holdout_candidate_audit"
        or authorization.get("large_fragment_download_authorized") is not False
        or authorization.get("holdout_method_execution_authorized") is not False
    ):
        raise PermissionError("G5-H1 candidate-audit authorization is invalid")
    if (
        previous.get("status") != "blocked"
        or previous.get("audit_status") != "pass"
        or previous.get("access_decision") != "do_not_unseal_preserve_remaining_holdout"
        or previous.get("raw_holdout_accessed") is not False
    ):
        raise ValueError("G5-H0 no-go boundary is invalid")
    if policy.get("status") != "frozen" or policy.get("holdout_accessed") is not False:
        raise ValueError("M4 policy is not frozen and untouched")
    invalid = [field for field in FALSE_FIELDS if config.get(field) is not False]
    if invalid:
        raise PermissionError(f"G5-H1 forbidden execution flags are not false: {invalid}")
    return config, policy


def _soft_values(text: str, field: str) -> list[str]:
    prefix = f"{field} = "
    return [line[len(prefix) :].strip() for line in text.splitlines() if line.startswith(prefix)]


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config, policy = _load(config_path, root)
    selected = config["selected_candidate"]
    series_path = _check(root, selected["series_metadata"], "series_metadata")
    sample_path = _check(root, selected["sample_metadata"], "sample_metadata")
    alternate_path = _check(root, selected["alternate_sample_metadata"], "alternate_sample_metadata")
    positions_path = _check(root, selected["positions"], "positions")
    metadata_path = _check(root, selected["metadata_json"], "metadata_json")
    for accession, record in config["candidate_sources"].items():
        _check(root, record, accession)

    series_text = series_path.read_text(encoding="utf-8")
    sample_text = sample_path.read_text(encoding="utf-8")
    alternate_text = alternate_path.read_text(encoding="utf-8")
    checks = {
        "series_accession_matches": selected["accession"] in series_text,
        "series_public_before_freeze": "!Series_status = Public on Jul 01 2026" in series_text,
        "series_spatial_atac_hic_design": "Spatial-ATAC-Hi-C" in series_text,
        "series_two_mouse_samples_declared": _soft_values(series_text, "!Series_sample_id")
        == ["GSM9228169", "GSM9228170"],
        "sample_identity_matches": "!Sample_title = Mouse Brain R6" in sample_text,
        "alternate_identity_matches": "!Sample_title = Mouse Brain R8" in alternate_text,
        "selected_sample_is_lexicographically_first": selected["sample_accession"]
        == min(_soft_values(series_text, "!Series_sample_id")),
        "mouse_species_declared": "!Sample_organism_ch1 = Mus musculus" in sample_text,
        "mm10_assembly_declared": "!Sample_data_processing = Assembly: mm10" in sample_text,
        "cellranger_atac_declared": "Cell Ranger ATAC v2.0" in sample_text,
        "standard_fragment_format_declared": "fragments.tsv.gz: a BED-like tabular file" in sample_text,
        "positions_file_declared": "tissue_positions_list.csv" in sample_text,
        "fragment_url_matches": selected["minimal_execution_download"]["url"].replace("https://", "ftp://")
        in sample_text,
    }

    rows: list[list[str]] = []
    with gzip.open(positions_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    barcodes = [row[0] for row in rows if len(row) == 6]
    in_tissue = sum(int(row[1]) for row in rows if len(row) == 6)
    metadata = _json_gzip(metadata_path)
    checks.update(
        {
            "positions_size_matches": positions_path.stat().st_size == selected["positions"]["expected_bytes"],
            "positions_rows_match": len(rows) == selected["positions"]["expected_rows"],
            "positions_six_columns": all(len(row) == 6 for row in rows),
            "positions_barcodes_unique": len(barcodes) == len(set(barcodes)) == len(rows),
            "positions_in_tissue_match": in_tissue == selected["positions"]["expected_in_tissue_rows"],
            "metadata_size_matches": metadata_path.stat().st_size == selected["metadata_json"]["expected_bytes"],
            "metadata_num_tixels_matches": metadata.get("numTixels")
            == selected["metadata_json"]["expected_num_tixels"],
            "metadata_run_matches": metadata.get("run") == "MBR6",
            "small_file_in_tissue_contract_matches": metadata.get("numTixels") == in_tissue,
        }
    )
    if not all(checks.values()):
        raise ValueError(f"G5-H1 selected candidate checks failed: {[key for key, value in checks.items() if not value]}")

    active = [
        leaf
        for leaf in policy["leaves"]
        if leaf.get("status") == "active_frozen"
        and leaf.get("task_id") == config["required_active_task"]
        and selected["species"] in str(leaf.get("support_species", "")).split(";")
        and selected["genome_build"] in str(leaf.get("support_genome_builds", "")).split(";")
    ]
    exact_representations = {leaf["representation_scope"] for leaf in active}
    expected_representations = {
        "filtered_fragments_with_frozen_peak_observation_contract",
        "macs3_summit_501bp",
        "fixed_bin_10kb",
        "fixed_bin_25kb",
        "fixed_bin_50kb",
    }
    if len(active) != config["required_exact_leaf_count"] or exact_representations != expected_representations:
        raise ValueError("G5-H1 selected candidate does not match all five portability leaves")

    candidate_rows = [
        {
            "accession": selected["accession"],
            "independent_study_family": True,
            "spatial_accessibility": True,
            "standard_processed_fragments": True,
            "spatial_coordinates": True,
            "species": selected["species"],
            "genome_build": selected["genome_build"],
            "paired_spatial_rna": selected["paired_rna"],
            "exact_active_policy_leaves": len(active),
            "minimum_download_bytes": selected["minimal_execution_download"]["expected_bytes"],
            "recommendation": "freeze_as_supplemental_portability_holdout_pending_separate_download_authorization",
        }
    ]
    excluded_properties = {
        "GSE260694": (True, False, True, "mouse", "mm10", True),
        "GSE237524": (False, False, False, "human", "hg38", True),
        "GSE218593": (True, True, True, "mouse", "mm10", True),
        "GSE305245": (False, False, True, "macaque", "unsupported", False),
        "GSE300809": (False, False, False, "human", "hg38", False),
    }
    for accession, values in excluded_properties.items():
        spatial, fragments, coordinates, species, genome, paired = values
        candidate_rows.append(
            {
                "accession": accession,
                "independent_study_family": accession != "GSE218593",
                "spatial_accessibility": spatial,
                "standard_processed_fragments": fragments,
                "spatial_coordinates": coordinates,
                "species": species,
                "genome_build": genome,
                "paired_spatial_rna": paired,
                "exact_active_policy_leaves": 0,
                "minimum_download_bytes": "not_applicable",
                "recommendation": config["candidate_sources"][accession]["recommendation"],
            }
        )

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "supplemental_holdout_candidate_comparison.tsv"
    leaf_path = output / "selected_candidate_exact_policy_leaves.tsv"
    download_path = output / "selected_candidate_minimal_download.tsv"
    pd.DataFrame(candidate_rows).to_csv(candidates_path, sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "policy_id": leaf["policy_id"],
                "task_id": leaf["task_id"],
                "input_track": leaf["input_track"],
                "representation_scope": leaf["representation_scope"],
                "support_species": leaf["support_species"],
                "support_genome_builds": leaf["support_genome_builds"],
                "candidate_exact_match": True,
            }
            for leaf in active
        ]
    ).sort_values(["input_track", "representation_scope"]).to_csv(leaf_path, sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "file_name": selected["minimal_execution_download"]["file_name"],
                "url": selected["minimal_execution_download"]["url"],
                "expected_bytes": selected["minimal_execution_download"]["expected_bytes"],
                "required_for": "standard_F_P_B_portability_holdout",
                "download_status": "not_downloaded_not_authorized_by_H1",
            },
            {
                "file_name": selected["excluded_large_hic_file"]["file_name"],
                "url": "not_required",
                "expected_bytes": selected["excluded_large_hic_file"]["expected_bytes"],
                "required_for": "not_required",
                "download_status": "excluded",
            },
        ]
    ).to_csv(download_path, sep="\t", index=False)
    evidence_path = output / "completion_evidence.json"
    artifacts = [candidates_path, leaf_path, download_path]
    _write_json(
        evidence_path,
        {
            "schema_version": 1,
            "gate": "G5",
            "milestone": "H1_supplemental_holdout_candidate_audit",
            "status": "pass",
            "decision": "freeze_GSE307620_GSM9228169_as_supplemental_portability_holdout",
            "selected_study_id": selected["study_id"],
            "selected_accession": selected["accession"],
            "selected_sample_accession": selected["sample_accession"],
            "positions_rows": len(rows),
            "in_tissue_rows": in_tissue,
            "exact_active_policy_leaves": len(active),
            "evaluable_task": config["required_active_task"],
            "paired_rna_available": False,
            "biological_utility_holdout_added": False,
            "minimal_fragment_download_bytes": selected["minimal_execution_download"]["expected_bytes"],
            "excluded_hic_download_bytes": selected["excluded_large_hic_file"]["expected_bytes"],
            "checks": checks,
            "artifacts": {
                path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in artifacts
            },
            "authorization": {"path": config["authorization"], "sha256": config["authorization_sha256"]},
            "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_file(config_path)},
            "large_fragment_download_run": False,
            "raw_fastq_or_bam_download_run": False,
            "method_or_task_outcome_used_for_selection": False,
            "method_execution_run": False,
            "existing_holdout_accessed": False,
            "policy_refit_run": False,
            "full_benchmark_run": False,
            "results_writing_run": False,
            "universal_ranking_run": False,
            "final_ranking_run": False,
            "next_milestone_authorized": False,
        },
    )
    return evidence_path


def _json_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected gzipped JSON mapping: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g5_h1_supplemental_holdout_candidate_audit.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.config, args.root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
