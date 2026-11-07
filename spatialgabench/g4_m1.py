"""Machine checks for development-only G4-M1 matched representation inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd
import yaml
import anndata as ad


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_g4_m1_input_manifest(config_path: str | Path, outdir: str | Path) -> dict[str, Path]:
    """Validate the frozen five-study M1 panel before any representation construction."""

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    authorization_path = Path(config["authorization"])
    preregistration_path = Path(config["preregistration"])
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    preregistration = yaml.safe_load(preregistration_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    required_authorization = {
        "status": "authorized",
        "scope": "development_only",
        "gate": "G4",
        "milestone": "M1",
        "holdout_accessed": False,
        "full_benchmark_authorized": False,
        "final_ranking_authorized": False,
    }
    for key, expected in required_authorization.items():
        if authorization.get(key) != expected:
            failures.append(f"authorization {key} must equal {expected!r}")
    if authorization.get("forbidden_inputs") != ["rna", "labels", "clusters", "pathology", "simulation_truth"]:
        failures.append("authorization forbidden_inputs contract differs from M1 freeze")
    expected_preregistration_sha256 = config["expected_preregistration_sha256"]
    observed_preregistration_sha256 = _sha256(preregistration_path)
    if observed_preregistration_sha256 != expected_preregistration_sha256:
        failures.append("frozen preregistration checksum mismatch")

    samples = config["samples"]
    expected_studies = set(preregistration["dataset_panel"]["matched_fragment_studies"])
    observed_studies = {sample["study_id"] for sample in samples.values()}
    if observed_studies != expected_studies or len(samples) != len(expected_studies):
        failures.append("M1 samples must map one-to-one to the frozen matched-fragment studies")
    holdouts = set(config["holdout_studies"])
    forbidden_accessions = set(config["forbidden_holdout_accessions"])
    if observed_studies & holdouts:
        failures.append("M1 sample panel overlaps a holdout study")

    rows: list[dict[str, object]] = []
    for dataset_id, sample in samples.items():
        source_key = "source_h5ad" if sample["source_type"] == "fragment_h5ad" else "fragments"
        source_path = Path(sample[source_key])
        positions_path = Path(sample["positions"]) if sample["source_type"] == "fragment_tsv" else None
        path_text = " ".join(str(value) for value in sample.values())
        holdout_path_overlap = sorted(accession for accession in forbidden_accessions if accession in path_text)
        source_exists = source_path.is_file()
        source_sha256 = _sha256(source_path) if source_exists else ""
        source_checksum_pass = source_exists and source_sha256 == sample["source_sha256"]
        positions_exists = positions_path is None or positions_path.is_file()
        positions_sha256 = _sha256(positions_path) if positions_path is not None and positions_exists else ""
        positions_checksum_pass = positions_path is None or (
            positions_exists and positions_sha256 == sample["positions_sha256"]
        )
        chrom_sizes_path = Path(sample["chrom_sizes"])
        blacklist_path = Path(sample["blacklist"])
        reference_pass = (
            chrom_sizes_path.is_file()
            and blacklist_path.is_file()
            and _sha256(chrom_sizes_path) == sample["chrom_sizes_sha256"]
            and _sha256(blacklist_path) == sample["blacklist_sha256"]
        )
        reuse_pass = True
        if sample.get("reuse"):
            reuse_pass = all(
                Path(artifact["path"]).is_file()
                and _sha256(Path(artifact["path"])) == artifact["sha256"]
                for artifact in sample["reuse_artifacts"]
            )
        status = "pass" if all(
            [source_checksum_pass, positions_checksum_pass, reference_pass, reuse_pass, not holdout_path_overlap]
        ) else "fail"
        if status == "fail":
            failures.append(f"{dataset_id} input or reuse contract failed")
        rows.append(
            {
                "study_id": sample["study_id"],
                "dataset_id": dataset_id,
                "sample_id": sample["sample_id"],
                "accession": sample["accession"],
                "species": sample["species"],
                "genome_build": sample["genome_build"],
                "source_type": sample["source_type"],
                "source_path": str(source_path),
                "source_size_bytes": source_path.stat().st_size if source_exists else 0,
                "source_sha256": source_sha256,
                "source_checksum_pass": source_checksum_pass,
                "positions_path": str(positions_path or "generated_from_h5ad"),
                "positions_sha256": positions_sha256,
                "positions_checksum_pass": positions_checksum_pass,
                "chrom_sizes": str(chrom_sizes_path),
                "blacklist": str(blacklist_path),
                "reference_checksum_pass": reference_pass,
                "reuse_existing_g2": bool(sample.get("reuse")),
                "reuse_checksum_pass": reuse_pass,
                "holdout_path_overlap": ";".join(holdout_path_overlap),
                "construction_inputs": "accessibility_fragments_and_spatial_coordinates_only",
                "status": status,
            }
        )

    table_path = outdir / "g4_m1_input_manifest.tsv"
    pd.DataFrame(rows).to_csv(table_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M1",
        "scope": "development_only",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "authorization": str(authorization_path),
        "authorization_sha256": _sha256(authorization_path),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": observed_preregistration_sha256,
        "studies": len(observed_studies),
        "samples": len(samples),
        "reused_samples": sum(bool(sample.get("reuse")) for sample in samples.values()),
        "construction_samples": sum(not bool(sample.get("reuse")) for sample in samples.values()),
        "forbidden_inputs_used": [],
        "holdout_accessed": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "failures": failures,
        "manifest": str(table_path),
        "manifest_sha256": _sha256(table_path),
        "status": "pass" if not failures else "fail",
    }
    evidence_path = outdir / "g4_m1_input_preflight.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": table_path, "evidence": evidence_path}


def write_g4_m1_completion_evidence(config_path: str | Path, outdir: str | Path) -> dict[str, Path]:
    """Aggregate and independently validate all five matched-representation studies."""

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, object]] = []
    representation_rows: list[dict[str, object]] = []
    resource_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for dataset_id, sample in config["samples"].items():
        reused = bool(sample.get("reuse"))
        base = (
            Path("results/v1/g2/representation_pilot/gse278007_ffpe_s4_rep1")
            if reused
            else Path(config["output_root"]) / dataset_id
        )
        filter_audit_path = base / "filtered_fragments/fragment_filter_audit.json"
        peak_audit_path = base / "frozen_peaks/peak_freeze_audit.json"
        validation_path = base / "matched/matched_representation_validation.json"
        manifest_path = base / "matched/representation_manifest.tsv"
        required_paths = [filter_audit_path, peak_audit_path, validation_path, manifest_path]
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            failures.append(f"{dataset_id} missing completion artifacts: {missing}")
            continue
        filter_audit = json.loads(filter_audit_path.read_text(encoding="utf-8"))
        peak_audit = json.loads(peak_audit_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        manifest = pd.read_csv(manifest_path, sep="\t")
        expected_representations = set(config["construction_contract"]["representations"])
        observed_representations = set(manifest["representation_id"])
        observation_contract_pass = int(validation["barcode_rows"]) == int(sample["expected_observations"])
        if sample["source_type"] == "fragment_h5ad":
            adapter_audit_path = base / "adapted/fragment_h5ad_adapter_audit.json"
            adapter_audit = json.loads(adapter_audit_path.read_text(encoding="utf-8"))
            source_checksum_pass = (
                adapter_audit["source_h5ad_sha256"] == sample["source_sha256"]
                and validation["original_source_fragments_sha256"]
                == adapter_audit["output_fragments_sha256"]
            )
        else:
            source_checksum_pass = validation["original_source_fragments_sha256"] == sample["source_sha256"]
        branch_observation_ids: list[list[str]] = []
        branch_contract_pass = True
        for row in manifest.to_dict(orient="records"):
            output_path = Path(row["output_path"])
            output_checksum_pass = output_path.is_file() and _sha256(output_path) == row["output_sha256"]
            if output_checksum_pass:
                adata = ad.read_h5ad(output_path, backed="r")
                branch_observation_ids.append(adata.obs_names.astype(str).tolist())
                branch_fragment_sha = str(adata.uns["source_fragments_sha256"])
                branch_contract_pass &= branch_fragment_sha == validation["source_fragments_sha256"]
                branch_contract_pass &= adata.n_obs == int(validation["barcode_rows"])
                adata.file.close()
            else:
                branch_contract_pass = False
            fixed_bin = str(row["representation_id"]).startswith("fixed_bin")
            exact_count_conservation = (
                not fixed_bin
                or int(row["matrix_total_counts"]) == int(filter_audit["included_fragment_rows"])
            )
            branch_contract_pass &= exact_count_conservation
            representation_rows.append(
                {
                    "study_id": sample["study_id"],
                    "dataset_id": dataset_id,
                    "sample_id": sample["sample_id"],
                    "species": sample["species"],
                    "genome_build": sample["genome_build"],
                    "representation_id": row["representation_id"],
                    "n_observations": int(row["n_observations"]),
                    "n_features": int(row["n_features"]),
                    "matrix_nonzero_entries": int(row["matrix_nonzero_entries"]),
                    "matrix_total_counts": int(row["matrix_total_counts"]),
                    "eligible_fragment_fraction_assigned": float(row["eligible_fragment_fraction_assigned"]),
                    "exact_fixed_bin_count_conservation": exact_count_conservation,
                    "filtered_fragment_sha256": validation["source_fragments_sha256"],
                    "output_path": str(output_path),
                    "output_sha256": row["output_sha256"],
                    "output_checksum_pass": output_checksum_pass,
                    "reused_from_g2": reused,
                    "status": "pass" if output_checksum_pass and exact_count_conservation else "fail",
                }
            )
        identical_observation_order = bool(branch_observation_ids) and all(
            ids == branch_observation_ids[0] for ids in branch_observation_ids[1:]
        )
        sample_pass = all(
            [
                filter_audit["status"] == "pass",
                peak_audit["status"] == "pass",
                validation["status"] == "pass",
                observed_representations == expected_representations,
                observation_contract_pass,
                source_checksum_pass,
                branch_contract_pass,
                identical_observation_order,
            ]
        )
        if not sample_pass:
            failures.append(f"{dataset_id} completion contract failed")
        sample_rows.append(
            {
                "study_id": sample["study_id"],
                "dataset_id": dataset_id,
                "sample_id": sample["sample_id"],
                "species": sample["species"],
                "genome_build": sample["genome_build"],
                "source_type": sample["source_type"],
                "source_fragment_rows": int(filter_audit["source_fragment_rows"]),
                "included_fragment_rows": int(filter_audit["included_fragment_rows"]),
                "excluded_barcode_rows": int(filter_audit["excluded_barcode_rows"]),
                "excluded_chromosome_rows": int(filter_audit["excluded_chromosome_rows"]),
                "invalid_fragment_rows": int(filter_audit["invalid_fragment_rows"]),
                "n_observations": int(validation["barcode_rows"]),
                "frozen_peak_rows": int(peak_audit["frozen_peak_rows"]),
                "peak_fragment_capture_fraction": float(
                    manifest.loc[
                        manifest["representation_id"] == "macs3_summit_501bp",
                        "eligible_fragment_fraction_assigned",
                    ].iloc[0]
                ),
                "representation_count": len(manifest),
                "identical_observation_ids_and_order": identical_observation_order,
                "one_filtered_fragment_sha256": branch_contract_pass,
                "source_checksum_pass": source_checksum_pass,
                "reused_from_g2": reused,
                "status": "pass" if sample_pass else "fail",
            }
        )

        if not reused:
            for stage in ["h5ad_adapter", "fragment_filter", "macs3", "peak_freeze", "matched_build"]:
                benchmark_path = base / f"benchmarks/{stage}.tsv"
                if not benchmark_path.is_file():
                    if stage != "h5ad_adapter" or sample["source_type"] == "fragment_h5ad":
                        failures.append(f"{dataset_id} missing {stage} benchmark")
                    continue
                benchmark = pd.read_csv(benchmark_path, sep="\t").iloc[0].to_dict()
                resource_rows.append(
                    {
                        "study_id": sample["study_id"],
                        "dataset_id": dataset_id,
                        "stage": stage,
                        "runtime_seconds": benchmark.get("s"),
                        "max_rss_mb": benchmark.get("max_rss"),
                        "max_vms_mb": benchmark.get("max_vms"),
                        "mean_load": benchmark.get("mean_load"),
                        "io_in_mb": benchmark.get("io_in"),
                        "io_out_mb": benchmark.get("io_out"),
                        "benchmark_path": str(benchmark_path),
                        "status": "pass",
                    }
                )

    sample_table = outdir / "g4_m1_sample_summary.tsv"
    representation_table = outdir / "g4_m1_representation_summary.tsv"
    resource_table = outdir / "g4_m1_resource_summary.tsv"
    pd.DataFrame(sample_rows).to_csv(sample_table, sep="\t", index=False)
    pd.DataFrame(representation_rows).to_csv(representation_table, sep="\t", index=False)
    pd.DataFrame(resource_rows).to_csv(resource_table, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M1",
        "scope": "development_only_matched_representation_construction",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "studies": len(sample_rows),
        "representations": len(representation_rows),
        "constructed_studies": sum(not row["reused_from_g2"] for row in sample_rows),
        "reused_g2_studies": sum(row["reused_from_g2"] for row in sample_rows),
        "forbidden_inputs_used": [],
        "holdout_accessed": False,
        "m2_run": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "performance_outcomes_interpreted": False,
        "construction_parameters": config["construction_contract"],
        "tool_versions": {
            "python": platform.python_version(),
            "spatialgabench": importlib.metadata.version("spatialgabench"),
            "anndata": importlib.metadata.version("anndata"),
            "h5py": importlib.metadata.version("h5py"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "macs3": importlib.metadata.version("macs3"),
            "snakemake": importlib.metadata.version("snakemake"),
        },
        "failures": failures,
        "sample_summary": str(sample_table),
        "sample_summary_sha256": _sha256(sample_table),
        "representation_summary": str(representation_table),
        "representation_summary_sha256": _sha256(representation_table),
        "resource_summary": str(resource_table),
        "resource_summary_sha256": _sha256(resource_table),
        "status": "pass" if not failures and len(sample_rows) == 5 and len(representation_rows) == 20 else "fail",
    }
    evidence_path = outdir / "g4_m1_completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "samples": sample_table,
        "representations": representation_table,
        "resources": resource_table,
        "evidence": evidence_path,
    }
