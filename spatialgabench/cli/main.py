"""SpatialGABench public command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import anndata as ad

from spatialgabench import __version__
from spatialgabench.config import validate_config
from spatialgabench.datasets import (
    write_download_receipt,
    write_fragment_h5ad_audit,
    write_fragment_spatial_audit,
    write_peak_h5_spatial_audit,
    write_paired_spatial_10x_audit,
    write_paired_spatial_audit,
    write_paired_spatial_h5_audit,
)
from spatialgabench.g4_validation import write_g4_preregistration_validation
from spatialgabench.g4_m1 import write_g4_m1_completion_evidence, write_g4_m1_input_manifest
from spatialgabench.project_validation import write_project_validation
from spatialgabench.recommendation import (
    load_recommendation_input,
    recommend_from_frozen_policy,
    recommend_methods,
    write_recommendation_report,
)
from spatialgabench.registry import list_methods, list_tasks
from spatialgabench.representations import (
    build_matched_representations,
    export_fragment_h5ad,
    filter_fragments_for_representation,
    freeze_macs3_summits,
    write_development_authorization,
)
from spatialgabench.simulation.api import (
    generate_empirical_semisynthetic_simulation,
    generate_fragments_from_peak_matrix,
    generate_mechanistic_simulation,
    load_simulation_config,
    run_g3_simulation_smoke,
    summarize_simulation,
    validate_fragments_for_simulation_dir,
    validate_simulation_outputs,
    write_factorial_design,
)
from spatialgabench.software_wrappers import (
    build_real_data_contract_fixture,
    write_g4_contract_preflight_evidence,
    write_wrapper_input_validation,
)
from spatialgabench.software_wrappers.summary import write_software_smoke_summary
from spatialgabench.software_wrappers.toy import generate_wrapper_toy_fixture
from spatialgabench.toy import run_toy_benchmark


def _print_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No rows.")
        return
    columns = sorted({key for row in rows for key in row})
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(col, "")) for col in columns))


def cmd_validate_config(args: argparse.Namespace) -> int:
    result = validate_config(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_validate_project(args: argparse.Namespace) -> int:
    payload = write_project_validation(args.config, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["validation_status"] == "pass" else 1


def cmd_validate_g4_preregistration(args: argparse.Namespace) -> int:
    payload = write_g4_preregistration_validation(args.preregistration, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


def cmd_export_fragment_h5ad(args: argparse.Namespace) -> int:
    payload = export_fragment_h5ad(
        args.h5ad,
        args.fragments,
        args.positions,
        args.audit,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


def cmd_validate_g4_m1_inputs(args: argparse.Namespace) -> int:
    outputs = write_g4_m1_input_manifest(args.config, args.outdir)
    payload = json.loads(outputs["evidence"].read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


def cmd_summarize_g4_m1(args: argparse.Namespace) -> int:
    outputs = write_g4_m1_completion_evidence(args.config, args.outdir)
    payload = json.loads(outputs["evidence"].read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


def cmd_audit_paired_spatial(args: argparse.Namespace) -> int:
    outputs = write_paired_spatial_audit(
        args.fragments,
        args.rna_matrix,
        args.positions,
        args.outdir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_audit_fragment_spatial(args: argparse.Namespace) -> int:
    outputs = write_fragment_spatial_audit(
        args.fragments,
        args.positions,
        args.outdir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_audit_paired_spatial_10x(args: argparse.Namespace) -> int:
    outputs = write_paired_spatial_10x_audit(
        args.fragments,
        args.rna_barcodes,
        args.rna_features,
        args.rna_matrix,
        args.atac_positions,
        args.rna_positions,
        args.outdir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_verify_download_manifest(args: argparse.Namespace) -> int:
    output = write_download_receipt(
        args.manifest,
        args.raw_root,
        args.out,
        args.verified_on,
    )
    receipt = pd.read_csv(output, sep="\t")
    payload = {
        "receipt": str(output),
        "files": int(len(receipt)),
        "passed": int((receipt["verification_status"] == "pass").sum()),
        "failed": int((receipt["verification_status"] == "fail").sum()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["failed"] == 0 else 1


def cmd_audit_peak_h5_spatial(args: argparse.Namespace) -> int:
    outputs = write_peak_h5_spatial_audit(
        args.h5,
        args.metadata,
        args.outdir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_audit_fragment_h5ad(args: argparse.Namespace) -> int:
    outputs = write_fragment_h5ad_audit(args.h5ad, args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_audit_paired_spatial_h5(args: argparse.Namespace) -> int:
    outputs = write_paired_spatial_h5_audit(
        args.fragments,
        args.rna_h5,
        args.positions,
        args.spot_metadata,
        args.fragment_semantics,
        args.outdir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_authorize_development_study(args: argparse.Namespace) -> int:
    output = write_development_authorization(args.config, args.study_id, args.out)
    print(json.dumps({"authorization": str(output)}, indent=2, sort_keys=True))
    return 0


def cmd_freeze_macs3_peaks(args: argparse.Namespace) -> int:
    result = freeze_macs3_summits(
        args.narrowpeak,
        args.chrom_sizes,
        args.blacklist,
        args.out,
        args.audit,
        peak_width=args.peak_width,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def cmd_filter_fragments_for_representation(args: argparse.Namespace) -> int:
    result = filter_fragments_for_representation(
        args.config,
        args.study_id,
        args.fragments,
        args.positions,
        args.chrom_sizes,
        args.out,
        args.audit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def cmd_build_matched_representations(args: argparse.Namespace) -> int:
    outputs = build_matched_representations(
        args.config,
        args.study_id,
        args.dataset_id,
        args.sample_id,
        args.fragments,
        args.positions,
        args.chrom_sizes,
        args.peaks,
        args.genome_build,
        [int(value) for value in args.bin_sizes.split(",")],
        args.outdir,
        chunk_size=args.chunk_size,
        fragment_filter_audit_path=args.fragment_filter_audit,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    validation = json.loads(Path(outputs["validation"]).read_text(encoding="utf-8"))
    return 0 if validation["status"] == "pass" else 1


def cmd_validate_wrapper_input(args: argparse.Namespace) -> int:
    payload = write_wrapper_input_validation(args.config, args.root, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


def cmd_generate_wrapper_toy(args: argparse.Namespace) -> int:
    outputs = generate_wrapper_toy_fixture(args.outdir, seed=args.seed)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_build_g4_preflight_fixture(args: argparse.Namespace) -> int:
    outputs = build_real_data_contract_fixture(
        project_config=args.config,
        study_id=args.study_id,
        dataset_id=args.dataset_id,
        sample_id=args.sample_id,
        genome_build=args.genome_build,
        fragments_path=args.fragments,
        positions_path=args.positions,
        gene_annotation_path=args.gene_annotation,
        chrom_sizes_path=args.chrom_sizes,
        output_dir=args.outdir,
        n_observations=args.n_observations,
        seed=args.seed,
        bin_size=args.bin_size,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_summarize_g4_preflight(args: argparse.Namespace) -> int:
    outputs = write_g4_contract_preflight_evidence(args.root, args.outdir)
    payload = json.loads(outputs["evidence"].read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


def cmd_summarize_wrapper_smoke(args: argparse.Namespace) -> int:
    outputs = write_software_smoke_summary(args.config, args.root, args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_list_methods(args: argparse.Namespace) -> int:
    _print_rows(list_methods())
    return 0


def cmd_list_tasks(args: argparse.Namespace) -> int:
    _print_rows(list_tasks())
    return 0


def cmd_run_toy_example(args: argparse.Namespace) -> int:
    outputs = run_toy_benchmark(args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_summarize_results(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(results_dir.rglob("*")) if results_dir.exists() else []:
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(results_dir)),
                    "suffix": path.suffix,
                    "size_bytes": path.stat().st_size,
                }
            )
    inventory = pd.DataFrame(rows, columns=["relative_path", "suffix", "size_bytes"])
    inventory_path = outdir / "result_file_inventory.tsv"
    inventory.to_csv(inventory_path, sep="\t", index=False)
    report_path = outdir / "summary_report.md"
    report_path.write_text(
        f"# Result directory summary\n\nInput: `{results_dir}`\n\nFiles found: {len(inventory)}\n",
        encoding="utf-8",
    )
    print(json.dumps({"inventory": str(inventory_path), "report": str(report_path), "n_files": len(inventory)}, indent=2))
    return 0


def cmd_generate_runbook(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "spatialgabench_runbook.md"
    path.write_text(
        "\n".join(
            [
                "# SpatialGABench runbook",
                "",
                "## Toy example",
                "",
                "Run `python -m spatialgabench.cli.main run-toy-example --outdir results/examples/toy_run`.",
                "",
                "## Full benchmark",
                "",
                "Use the project runbook and config templates. Full real-data benchmarks require source data and are not run by this command.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"runbook": str(path)}, indent=2))
    return 0


def cmd_simulate_mechanistic(args: argparse.Namespace) -> int:
    config = load_simulation_config(args.config)
    outputs = generate_mechanistic_simulation(config, args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_simulate_semisynthetic(args: argparse.Namespace) -> int:
    config = load_simulation_config(args.config)
    outputs = generate_empirical_semisynthetic_simulation(config, args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_expand_simulation_design(args: argparse.Namespace) -> int:
    outputs = write_factorial_design(args.config, args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_run_g3_simulation_smoke(args: argparse.Namespace) -> int:
    outputs = run_g3_simulation_smoke(args.config, args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_generate_fragments(args: argparse.Namespace) -> int:
    simulation_dir = Path(args.simulation_dir)
    adata = ad.read_h5ad(simulation_dir / "simulated_atac_peak_matrix.h5ad")
    config = {"simulation_id": adata.uns.get("simulation_id", simulation_dir.name), "random_seed": int(args.random_seed)}
    outputs = generate_fragments_from_peak_matrix(adata, config, args.outdir)
    validation = validate_fragments_for_simulation_dir(simulation_dir, args.outdir)
    payload = {key: str(value) for key, value in outputs.items()}
    payload["validation_status"] = str(validation["validation_status"].iloc[0])
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_validate_simulation(args: argparse.Namespace) -> int:
    validation = validate_simulation_outputs(args.simulation_dir)
    out = Path(args.simulation_dir) / "validation_summary.tsv"
    validation.to_csv(out, sep="\t", index=False)
    print(json.dumps({"validation": str(out), "status_counts": validation["validation_status"].value_counts().to_dict()}, indent=2, sort_keys=True))
    return 0


def cmd_summarize_simulation(args: argparse.Namespace) -> int:
    outputs = summarize_simulation(args.simulation_dir, args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_recommend_methods(args: argparse.Namespace) -> int:
    config = load_recommendation_input(args.config)
    recommendation = recommend_methods(config)
    outputs = write_recommendation_report(recommendation, args.outdir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


def cmd_recommend_frozen_policy(args: argparse.Namespace) -> int:
    recommendation = recommend_from_frozen_policy(
        primary_task=args.primary_task,
        input_track=args.input_track,
        representation_id=args.representation_id,
        species=args.species,
        genome_build=args.genome_build,
        fragments_available=args.fragments_available,
        policy_path=args.policy,
    )
    print(json.dumps(recommendation, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spatialgabench", description="SpatialGABench benchmark resource CLI")
    parser.add_argument("--version", action="version", version=f"spatialgabench {__version__}")
    sub = parser.add_subparsers(dest="command")

    validate = sub.add_parser("validate-config", help="Validate a YAML config template")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=cmd_validate_config)

    project = sub.add_parser("validate-project", help="Validate the active v1 project contract")
    project.add_argument("--config", required=True)
    project.add_argument("--out", required=True)
    project.set_defaults(func=cmd_validate_project)

    g4_prereg = sub.add_parser("validate-g4-preregistration", help="Validate the frozen G4 benchmark plan")
    g4_prereg.add_argument("--preregistration", required=True)
    g4_prereg.add_argument("--out", required=True)
    g4_prereg.set_defaults(func=cmd_validate_g4_preregistration)

    fragment_h5ad_export = sub.add_parser(
        "export-fragment-h5ad",
        help="Losslessly export SnapATAC2 fragment-backed AnnData to fragments and positions",
    )
    fragment_h5ad_export.add_argument("--h5ad", required=True)
    fragment_h5ad_export.add_argument("--fragments", required=True)
    fragment_h5ad_export.add_argument("--positions", required=True)
    fragment_h5ad_export.add_argument("--audit", required=True)
    fragment_h5ad_export.set_defaults(func=cmd_export_fragment_h5ad)

    g4_m1_inputs = sub.add_parser(
        "validate-g4-m1-inputs",
        help="Validate the authorized five-study matched-representation input panel",
    )
    g4_m1_inputs.add_argument("--config", required=True)
    g4_m1_inputs.add_argument("--outdir", required=True)
    g4_m1_inputs.set_defaults(func=cmd_validate_g4_m1_inputs)

    g4_m1_summary = sub.add_parser(
        "summarize-g4-m1",
        help="Validate and summarize completed G4-M1 matched representations",
    )
    g4_m1_summary.add_argument("--config", required=True)
    g4_m1_summary.add_argument("--outdir", required=True)
    g4_m1_summary.set_defaults(func=cmd_summarize_g4_m1)

    paired = sub.add_parser("audit-paired-spatial", help="Audit fragment/RNA/spatial barcode contracts")
    paired.add_argument("--fragments", required=True)
    paired.add_argument("--rna-matrix", required=True)
    paired.add_argument("--positions", required=True)
    paired.add_argument("--outdir", required=True)
    paired.set_defaults(func=cmd_audit_paired_spatial)

    fragment_spatial = sub.add_parser(
        "audit-fragment-spatial",
        help="Audit fragment/spatial barcode contracts when paired RNA is unavailable",
    )
    fragment_spatial.add_argument("--fragments", required=True)
    fragment_spatial.add_argument("--positions", required=True)
    fragment_spatial.add_argument("--outdir", required=True)
    fragment_spatial.set_defaults(func=cmd_audit_fragment_spatial)

    paired_10x = sub.add_parser(
        "audit-paired-spatial-10x",
        help="Audit fragment and 10x RNA matrices with assay-specific coordinates",
    )
    paired_10x.add_argument("--fragments", required=True)
    paired_10x.add_argument("--rna-barcodes", required=True)
    paired_10x.add_argument("--rna-features", required=True)
    paired_10x.add_argument("--rna-matrix", required=True)
    paired_10x.add_argument("--atac-positions", required=True)
    paired_10x.add_argument("--rna-positions", required=True)
    paired_10x.add_argument("--outdir", required=True)
    paired_10x.set_defaults(func=cmd_audit_paired_spatial_10x)

    downloads = sub.add_parser(
        "verify-download-manifest",
        help="Verify expected sizes, gzip integrity, and SHA-256 for downloaded files",
    )
    downloads.add_argument("--manifest", required=True)
    downloads.add_argument("--raw-root", required=True)
    downloads.add_argument("--out", required=True)
    downloads.add_argument("--verified-on", required=True)
    downloads.set_defaults(func=cmd_verify_download_manifest)

    peak_h5 = sub.add_parser(
        "audit-peak-h5-spatial",
        help="Audit a sparse peak H5 matrix and spatial metadata contract",
    )
    peak_h5.add_argument("--h5", required=True)
    peak_h5.add_argument("--metadata", required=True)
    peak_h5.add_argument("--outdir", required=True)
    peak_h5.set_defaults(func=cmd_audit_peak_h5_spatial)

    fragment_h5ad = sub.add_parser(
        "audit-fragment-h5ad",
        help="Audit a SnapATAC2-style fragment-backed AnnData file",
    )
    fragment_h5ad.add_argument("--h5ad", required=True)
    fragment_h5ad.add_argument("--outdir", required=True)
    fragment_h5ad.set_defaults(func=cmd_audit_fragment_h5ad)

    paired_h5 = sub.add_parser(
        "audit-paired-spatial-h5",
        help="Audit fragments, 10x RNA H5, coordinates, and spot metadata",
    )
    paired_h5.add_argument("--fragments", required=True)
    paired_h5.add_argument("--rna-h5", required=True)
    paired_h5.add_argument("--positions", required=True)
    paired_h5.add_argument("--spot-metadata", required=True)
    paired_h5.add_argument("--fragment-semantics", required=True)
    paired_h5.add_argument("--outdir", required=True)
    paired_h5.set_defaults(func=cmd_audit_paired_spatial_h5)

    authorize = sub.add_parser(
        "authorize-development-study",
        help="Reject holdouts and authorize a registered development study",
    )
    authorize.add_argument("--config", required=True)
    authorize.add_argument("--study-id", required=True)
    authorize.add_argument("--out", required=True)
    authorize.set_defaults(func=cmd_authorize_development_study)

    freeze_peaks = sub.add_parser(
        "freeze-macs3-peaks",
        help="Freeze fixed-width non-overlapping MACS3 summit peaks",
    )
    freeze_peaks.add_argument("--narrowpeak", required=True)
    freeze_peaks.add_argument("--chrom-sizes", required=True)
    freeze_peaks.add_argument("--blacklist", required=True)
    freeze_peaks.add_argument("--out", required=True)
    freeze_peaks.add_argument("--audit", required=True)
    freeze_peaks.add_argument("--peak-width", type=int, default=501)
    freeze_peaks.set_defaults(func=cmd_freeze_macs3_peaks)

    filter_fragments = sub.add_parser(
        "filter-fragments-for-representation",
        help="Create one audited fragment input for matched peak and bin representations",
    )
    filter_fragments.add_argument("--config", required=True)
    filter_fragments.add_argument("--study-id", required=True)
    filter_fragments.add_argument("--fragments", required=True)
    filter_fragments.add_argument("--positions", required=True)
    filter_fragments.add_argument("--chrom-sizes", required=True)
    filter_fragments.add_argument("--out", required=True)
    filter_fragments.add_argument("--audit", required=True)
    filter_fragments.set_defaults(func=cmd_filter_fragments_for_representation)

    matched = sub.add_parser(
        "build-matched-representations",
        help="Build peak and fixed-bin matrices from the same development fragments",
    )
    matched.add_argument("--config", required=True)
    matched.add_argument("--study-id", required=True)
    matched.add_argument("--dataset-id", required=True)
    matched.add_argument("--sample-id", required=True)
    matched.add_argument("--fragments", required=True)
    matched.add_argument("--positions", required=True)
    matched.add_argument("--chrom-sizes", required=True)
    matched.add_argument("--peaks", required=True)
    matched.add_argument("--genome-build", required=True)
    matched.add_argument("--bin-sizes", default="10000,25000,50000")
    matched.add_argument("--chunk-size", type=int, default=500_000)
    matched.add_argument("--fragment-filter-audit")
    matched.add_argument("--outdir", required=True)
    matched.set_defaults(func=cmd_build_matched_representations)

    wrapper_input = sub.add_parser(
        "validate-wrapper-input",
        help="Validate a leakage-safe external software wrapper input contract",
    )
    wrapper_input.add_argument("--config", required=True)
    wrapper_input.add_argument("--root", default=".")
    wrapper_input.add_argument("--out", required=True)
    wrapper_input.set_defaults(func=cmd_validate_wrapper_input)

    wrapper_toy = sub.add_parser(
        "generate-wrapper-toy",
        help="Generate an accessibility-only fixture for external wrapper smoke tests",
    )
    wrapper_toy.add_argument("--outdir", required=True)
    wrapper_toy.add_argument("--seed", type=int, default=1601)
    wrapper_toy.set_defaults(func=cmd_generate_wrapper_toy)

    preflight_fixture = sub.add_parser(
        "build-g4-preflight-fixture",
        help="Build a deterministic accessibility-only fixture from a development sample",
    )
    preflight_fixture.add_argument("--config", required=True)
    preflight_fixture.add_argument("--study-id", required=True)
    preflight_fixture.add_argument("--dataset-id", required=True)
    preflight_fixture.add_argument("--sample-id", required=True)
    preflight_fixture.add_argument("--genome-build", required=True)
    preflight_fixture.add_argument("--fragments", required=True)
    preflight_fixture.add_argument("--positions", required=True)
    preflight_fixture.add_argument("--gene-annotation", required=True)
    preflight_fixture.add_argument("--chrom-sizes", required=True)
    preflight_fixture.add_argument("--outdir", required=True)
    preflight_fixture.add_argument("--n-observations", type=int, default=64)
    preflight_fixture.add_argument("--seed", type=int, default=4160)
    preflight_fixture.add_argument("--bin-size", type=int, default=10_000)
    preflight_fixture.set_defaults(func=cmd_build_g4_preflight_fixture)

    preflight_summary = sub.add_parser(
        "summarize-g4-preflight",
        help="Validate and summarize the two-species G4 software contract preflight",
    )
    preflight_summary.add_argument("--root", default=".")
    preflight_summary.add_argument("--outdir", required=True)
    preflight_summary.set_defaults(func=cmd_summarize_g4_preflight)

    wrapper_summary = sub.add_parser(
        "summarize-wrapper-smoke",
        help="Verify and aggregate external software wrapper smoke evidence",
    )
    wrapper_summary.add_argument("--config", required=True)
    wrapper_summary.add_argument("--root", default=".")
    wrapper_summary.add_argument("--outdir", required=True)
    wrapper_summary.set_defaults(func=cmd_summarize_wrapper_smoke)

    methods = sub.add_parser("list-methods", help="List methods from the registry or public fallback")
    methods.set_defaults(func=cmd_list_methods)

    tasks = sub.add_parser("list-tasks", help="List tasks from the registry or public fallback")
    tasks.set_defaults(func=cmd_list_tasks)

    toy = sub.add_parser("run-toy-example", help="Run the self-contained toy benchmark")
    toy.add_argument("--outdir", required=True)
    toy.set_defaults(func=cmd_run_toy_example)

    summarize = sub.add_parser("summarize-results", help="Create a file inventory for a result directory")
    summarize.add_argument("--results-dir", required=True)
    summarize.add_argument("--outdir", required=True)
    summarize.set_defaults(func=cmd_summarize_results)

    runbook = sub.add_parser("generate-runbook", help="Write a minimal public runbook")
    runbook.add_argument("--outdir", required=True)
    runbook.set_defaults(func=cmd_generate_runbook)

    sim_mechanistic = sub.add_parser("simulate-mechanistic", help="Generate an identifiable mechanistic simulation")
    sim_mechanistic.add_argument("--config", required=True)
    sim_mechanistic.add_argument("--outdir", required=True)
    sim_mechanistic.set_defaults(func=cmd_simulate_mechanistic)

    sim_semisynthetic = sub.add_parser(
        "simulate-semisynthetic", help="Generate a development-authorized empirical semi-synthetic simulation"
    )
    sim_semisynthetic.add_argument("--config", required=True)
    sim_semisynthetic.add_argument("--outdir", required=True)
    sim_semisynthetic.set_defaults(func=cmd_simulate_semisynthetic)

    sim_design = sub.add_parser("expand-simulation-design", help="Expand and audit the frozen replicated factorial design")
    sim_design.add_argument("--config", required=True)
    sim_design.add_argument("--outdir", required=True)
    sim_design.set_defaults(func=cmd_expand_simulation_design)

    sim_g3 = sub.add_parser("run-g3-simulation-smoke", help="Run the frozen first-seed OAT simulation readiness smoke")
    sim_g3.add_argument("--config", required=True)
    sim_g3.add_argument("--outdir", required=True)
    sim_g3.set_defaults(func=cmd_run_g3_simulation_smoke)

    fragments = sub.add_parser("generate-fragments", help="Generate reusable simulated fragments for a simulation directory")
    fragments.add_argument("--simulation-dir", required=True)
    fragments.add_argument("--outdir", required=True)
    fragments.add_argument("--random-seed", type=int, default=1)
    fragments.set_defaults(func=cmd_generate_fragments)

    validate_sim = sub.add_parser("validate-simulation", help="Validate reusable simulation output contract")
    validate_sim.add_argument("--simulation-dir", required=True)
    validate_sim.set_defaults(func=cmd_validate_simulation)

    summarize_sim = sub.add_parser("summarize-simulation", help="Summarize a reusable simulation directory")
    summarize_sim.add_argument("--simulation-dir", required=True)
    summarize_sim.add_argument("--outdir", required=True)
    summarize_sim.set_defaults(func=cmd_summarize_simulation)

    recommend = sub.add_parser("recommend-methods", help="Run QC-driven gene activity method recommendations")
    recommend.add_argument("--config", required=True)
    recommend.add_argument("--outdir", required=True)
    recommend.set_defaults(func=cmd_recommend_methods)

    frozen = sub.add_parser("recommend-frozen-policy", help="Query one exact-contract frozen G4 policy leaf")
    frozen.add_argument("--primary-task", required=True)
    frozen.add_argument("--input-track", required=True, choices=["F", "P", "B"])
    frozen.add_argument("--representation-id", required=True)
    frozen.add_argument("--species", required=True, choices=["mouse", "human"])
    frozen.add_argument("--genome-build", required=True, choices=["mm10", "hg38"])
    frozen.add_argument("--fragments-available", action=argparse.BooleanOptionalAction, required=True)
    frozen.add_argument(
        "--policy",
        default="results/v1/g4/m4_recommendation_policy/frozen_task_specific_recommendation_policy.json",
    )
    frozen.set_defaults(func=cmd_recommend_frozen_policy)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
