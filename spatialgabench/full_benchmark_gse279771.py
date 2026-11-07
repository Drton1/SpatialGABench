"""FB4 GSE279771 insertion-derived construction and paired-RNA evaluation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import sparse


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    loader = json.loads if path.suffix == ".json" else yaml.safe_load
    value = loader(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def _validate_authorization(
    config_path: Path, authorization_path: Path, root: Path, milestone: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, authorization = _mapping(config_path), _mapping(authorization_path)
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != milestone
        or authorization.get("policy_refit_authorized") is not False
        or authorization.get("input_track_aliasing_authorized") is not False
        or authorization.get("universal_ranking_authorized") is not False
        or authorization.get("final_ranking_authorized") is not False
    ):
        raise PermissionError(f"Invalid {milestone} authorization boundary")
    bindings = {
        "config": config_path,
        "implementation": Path(__file__).resolve(),
        "protocol": root / config["protocol"],
        "completion_protocol": root / config["completion_protocol"],
    }
    if milestone == "FB4A_GSE279771_insertion_construction":
        if (
            authorization.get("construction_authorized") is not True
            or authorization.get("rna_value_access_authorized") is not False
            or authorization.get("metadata_value_access_authorized") is not False
        ):
            raise PermissionError("FB4A construction boundary is invalid")
    else:
        if (
            authorization.get("rna_evaluation_authorized") is not True
            or authorization.get("construction_rerun_authorized") is not False
            or authorization.get("metadata_value_access_authorized") is not False
        ):
            raise PermissionError("FB4B RNA boundary is invalid")
        bindings["construction_evidence"] = root / authorization["construction_evidence"]["path"]
    for key, path in bindings.items():
        record = authorization.get(key, {})
        if not path.is_file() or record.get("path") != str(path.relative_to(root)) or record.get("sha256") != sha256(path):
            raise ValueError(f"FB4 authorization binding mismatch: {key}")
    return config, authorization


def _check_artifact(root: Path, record: dict[str, Any], name: str) -> Path:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"FB4 frozen artifact mismatch: {name}")
    return path


def _read_10x_rna_h5(path: Path) -> tuple[sparse.csr_matrix, list[str], pd.DataFrame]:
    """Read 10x RNA while ignoring feature metadata arrays with non-feature lengths."""

    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(value) for value in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape
        )
        barcodes = [str(value) for value in group["barcodes"].asstr()[:]]
        feature_group = group["features"]
        n_features = shape[0]
        columns = {
            key: [str(value) for value in dataset.asstr()[:]]
            for key, dataset in feature_group.items()
            if isinstance(dataset, h5py.Dataset)
            and dataset.ndim == 1
            and dataset.shape[0] == n_features
        }
    if "id" not in columns or len(barcodes) != shape[1]:
        raise ValueError("FB4 RNA H5 lacks aligned feature IDs or barcodes")
    var = pd.DataFrame(columns)
    var.index = var["id"].astype(str)
    if var.index.duplicated().any() or len(set(barcodes)) != len(barcodes):
        raise ValueError("FB4 RNA H5 contains duplicate feature or barcode IDs")
    return matrix.transpose().tocsr(), barcodes, var


def _strip_ensembl_versions(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.\d+$", "", regex=True)


def _chromosome_bins(chrom_sizes_path: Path, bin_size: int) -> tuple[pd.DataFrame, dict[str, tuple[int, int, int]]]:
    allowed = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    records, lookup = [], {}
    offset = 0
    with chrom_sizes_path.open(encoding="utf-8") as handle:
        for line in handle:
            chromosome, size, *_ = line.rstrip("\n").split("\t")
            if chromosome not in allowed:
                continue
            size = int(size)
            count = math.ceil(size / bin_size)
            lookup[chromosome] = (offset, count, size)
            for index in range(count):
                start, end = index * bin_size, min((index + 1) * bin_size, size)
                records.append({
                    "peak_id": f"{chromosome}:{start}-{end}", "chr": chromosome,
                    "start": start, "end": end, "width": end - start,
                    "representation_id": "insertion_derived_fixed_bin_25kb",
                })
            offset += count
    frame = pd.DataFrame(records).set_index("peak_id", drop=False)
    return frame, lookup


def build_insertion_matrix(config: dict[str, Any], root: Path) -> tuple[Path, Path]:
    output = root / config["output_root"] / "construction"
    output.mkdir(parents=True, exist_ok=True)
    matrix_path, audit_path = output / "insertion_derived_fixed_bin_25kb.h5ad", output / "insertion_matrix_audit.json"
    if audit_path.is_file() and matrix_path.is_file():
        audit = _mapping(audit_path)
        if audit.get("status") == "pass" and audit.get("output_sha256") == sha256(matrix_path):
            return matrix_path, audit_path
        raise ValueError("Existing FB4 insertion matrix failed its audit")
    source = _check_artifact(root, config["source_atac"], "source_atac")
    positions_path = _check_artifact(root, config["positions"], "positions")
    chrom_sizes = _check_artifact(root, config["references"]["chrom_sizes"], "chrom_sizes")
    positions = pd.read_csv(positions_path)
    positions = positions.loc[pd.to_numeric(positions.in_tissue, errors="raise").eq(1)].copy()
    if len(positions) != int(config["expected_observations"]) or positions.barcode.duplicated().any():
        raise ValueError("FB4 in-tissue position contract failed")
    positions = positions.set_index(positions.barcode.astype(str), drop=True)
    barcode_to_row = {barcode: index for index, barcode in enumerate(positions.index)}
    features, chromosomes = _chromosome_bins(chrom_sizes, int(config["bin_size_bp"]))
    matrix = sparse.csr_matrix((len(positions), len(features)), dtype=np.int32)
    row_buffer: list[int] = []
    column_buffer: list[int] = []
    source_rows = retained_events = excluded_chromosome_events = excluded_barcode_events = invalid_events = 0

    def flush() -> None:
        nonlocal matrix
        if not row_buffer:
            return
        chunk = sparse.coo_matrix(
            (np.ones(len(row_buffer), dtype=np.int32), (row_buffer, column_buffer)),
            shape=matrix.shape,
        ).tocsr()
        matrix = matrix + chunk
        row_buffer.clear(); column_buffer.clear()

    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            source_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                invalid_events += 2; continue
            chromosome, barcode = fields[0], fields[3]
            try:
                events = (int(fields[1]), int(fields[2]))
            except ValueError:
                invalid_events += 2; continue
            if chromosome not in chromosomes:
                excluded_chromosome_events += 2; continue
            if barcode not in barcode_to_row:
                excluded_barcode_events += 2; continue
            offset, count, chromosome_size = chromosomes[chromosome]
            for coordinate in events:
                if coordinate < 0 or coordinate >= chromosome_size:
                    invalid_events += 1; continue
                column = offset + coordinate // int(config["bin_size_bp"])
                if column >= offset + count:
                    invalid_events += 1; continue
                row_buffer.append(barcode_to_row[barcode]); column_buffer.append(column); retained_events += 1
            if len(row_buffer) >= 2_000_000:
                flush()
    flush()
    matrix.sum_duplicates(); matrix.sort_indices(); matrix.eliminate_zeros()
    if source_rows != int(config["expected_source_rows"]) or invalid_events or excluded_barcode_events:
        raise ValueError("FB4 insertion event parsing contract failed")
    if int(matrix.sum()) != retained_events:
        raise ValueError("FB4 insertion count conservation failed")
    obs = positions.copy()
    obs.index.name = "barcode"
    adata = ad.AnnData(X=matrix, obs=obs, var=features)
    adata.obsm["spatial"] = obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    adata.uns.update({
        "study_id": config["study_id"], "dataset_id": config["dataset_id"], "sample_id": config["sample_id"],
        "genome_build": config["genome_build"], "modality": "spatial_atac_insertion_counts",
        "representation_id": config["representation_id"], "input_track": "I",
        "source_semantics": config["source_semantics"],
        "counting_unit": "one_source_endpoint_insertion_event",
        "zero_span_source_row_contribution": "two_events_at_same_coordinate",
        "construction_leakage_inputs": "none",
        "rna_accessed": False, "metadata_accessed": False,
    })
    adata.write_h5ad(matrix_path, compression="gzip")
    audit = {
        "schema_version": 1, "status": "pass", "source_rows": source_rows,
        "source_endpoint_events": source_rows * 2, "retained_insertion_events": retained_events,
        "excluded_noncanonical_chromosome_events": excluded_chromosome_events,
        "excluded_barcode_events": excluded_barcode_events, "invalid_events": invalid_events,
        "matrix_total_counts": int(matrix.sum()), "matrix_nnz": int(matrix.nnz),
        "observations": matrix.shape[0], "features": matrix.shape[1],
        "representation_id": config["representation_id"], "input_track": "I",
        "standard_fragment_alias": False, "rna_accessed": False, "metadata_accessed": False,
        "output_path": str(matrix_path.relative_to(root)), "output_sha256": sha256(matrix_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return matrix_path, audit_path


def _method_mapping(method: str, adata: ad.AnnData, config: dict[str, Any], root: Path):
    from spatialgabench.gene_activity.accessibility_matched_random import compute_peak_accessibility_metrics, sample_accessibility_matched_peaks
    from spatialgabench.gene_activity.distal import build_distal_peak_gene_mapping, build_promoter_plus_distal_mapping, load_tss_table
    from spatialgabench.gene_activity.gene_body import build_gene_body_mapping
    from spatialgabench.gene_activity.interval_aggregation import build_mapping_from_intervals, load_bed_intervals, load_gene_metadata, parse_peak_coordinates
    from spatialgabench.gene_activity.promoter_gene_body import build_promoter_gene_body_mapping

    refs, params = config["references"], config["parameters"]
    peaks = parse_peak_coordinates(adata)
    genes = load_gene_metadata(root / refs["genes"]["path"])
    promoters = load_bed_intervals(root / refs["promoters"]["path"], "promoter")
    gene_body = load_bed_intervals(root / refs["gene_body"]["path"], "gene_body")
    tss = load_tss_table(root / refs["tss"]["path"], genes)
    if method == "promoter_only_2kb_pc":
        return build_mapping_from_intervals(peaks, promoters, genes, deduplicate=True)
    if method == "gene_body_pc":
        return build_gene_body_mapping(peaks, gene_body, genes)
    if method == "promoter_plus_gene_body_2kb_pc":
        return build_promoter_gene_body_mapping(peaks, promoters, gene_body, genes)
    if method == "distance_weighted_distal_250kb_decay50kb_pc":
        return build_distal_peak_gene_mapping(peaks, tss, genes, distal_window_bp=params["distal_window_bp"], promoter_exclusion_window_bp=params["promoter_window_bp"], distance_decay_bp=params["distance_decay_bp"])
    if method == "promoter_plus_distal_2kb_250kb_decay50kb_pc":
        return build_promoter_plus_distal_mapping(peaks, promoters, tss, genes, distal_window_bp=params["distal_window_bp"], promoter_exclusion_window_bp=params["promoter_window_bp"], distance_decay_bp=params["distance_decay_bp"])
    if method == "accessibility_matched_random_peak_to_gene_control_pc":
        promoter = build_mapping_from_intervals(peaks, promoters, genes, deduplicate=True)
        return sample_accessibility_matched_peaks(
            peak_metrics=compute_peak_accessibility_metrics(adata.X, peaks),
            reference_promoter_links=promoter.links, gene_metadata=genes,
            seed=params["control_seed"], n_bins=params["accessibility_bins"], within_gene_replacement=False,
        ).mapping
    raise ValueError(f"Unsupported FB4 method: {method}")


def construct(config_path: Path, authorization_path: Path, root: Path) -> Path:
    config, _ = _validate_authorization(config_path, authorization_path, root, "FB4A_GSE279771_insertion_construction")
    _check_artifact(root, config["schema_audit"], "schema_audit")
    for name, record in config["references"].items():
        _check_artifact(root, record, f"reference/{name}")
    matrix_path, audit_path = build_insertion_matrix(config, root)
    input_adata = ad.read_h5ad(matrix_path)
    output = root / config["output_root"] / "construction"
    input_validation = output / "construction_input_validation.json"
    input_validation.write_text(json.dumps({
        "schema_version": 1, "status": "pass", "input_track": "I",
        "representation_id": config["representation_id"], "source_atac_sha256": config["source_atac"]["sha256"],
        "rna_accessed": False, "metadata_accessed": False, "forbidden_inputs_used": [],
    }, indent=2, sort_keys=True) + "\n")
    from spatialgabench.gene_activity.interval_aggregation import compute_gene_activity
    from spatialgabench.software_wrappers.common import ensure_standard_gene_activity_anndata, validate_standard_output
    rows = []
    for method in config["methods"]:
        started = time.time()
        run_dir = output / "methods" / method
        run_dir.mkdir(parents=True, exist_ok=True)
        method_output = run_dir / "gene_activity.h5ad"
        if method_output.is_file() and (run_dir / "output_manifest.json").is_file():
            manifest = _mapping(run_dir / "output_manifest.json")
            if manifest.get("status") == "pass" and manifest.get("output_sha256") == sha256(method_output):
                rows.append(manifest); continue
            raise ValueError(f"Existing FB4 method output failed audit: {method}")
        mapping = _method_mapping(method, input_adata, config, root)
        activity = compute_gene_activity(input_adata.X, mapping.peak_gene_matrix)
        var = mapping.genes.copy(); var.index = var.gene_id.astype(str)
        result = ensure_standard_gene_activity_anndata(
            input_adata=input_adata, X=activity, var=var, method_id=method,
            software_name="SpatialGABench", software_version="1.0.0.dev0", wrapper_version="fb4_insertion_v1",
            genome_build="hg38", normalization_status="raw_insertion_accessibility_aggregation",
            output_path=method_output, extra_uns={
                "run_id": f"fb4__{method}", "study_id": config["study_id"], "sample_id": config["sample_id"],
                "representation_id": config["representation_id"], "input_track": "I",
                "comparison_scope": "GSE279771_insertion_track_only", "subset_policy": "full",
                "construction_input_classes": ["insertion_derived_accessibility_matrix", "gene_annotation"],
                "construction_leakage_inputs": "none", "input_validation_sha256": sha256(input_validation),
                "environment_lock_sha256": sha256(root / "envs/core-resolved.yaml"),
                "authorization_sha256": sha256(authorization_path),
            },
        )
        contract = validate_standard_output(result, input_adata, method)
        contract.to_csv(run_dir / "contract_validation.tsv", sep="\t", index=False)
        if not contract.status.eq("pass").all():
            raise ValueError(f"FB4 output contract failed: {method}")
        manifest = {
            "schema_version": 1, "status": "pass", "method_id": method,
            "representation_id": config["representation_id"], "input_track": "I",
            "runtime_seconds": time.time() - started, "output_path": str(method_output.relative_to(root)),
            "output_sha256": sha256(method_output), "observations": result.n_obs, "genes": result.n_vars,
            "output_nnz": int(result.X.nnz), "rna_accessed": False, "metadata_accessed": False,
        }
        (run_dir / "output_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        rows.append(manifest)
    method_manifest = output / "method_manifest.tsv"
    pd.DataFrame(rows).to_csv(method_manifest, sep="\t", index=False)
    not_applicable = output / "software_not_applicable.tsv"
    pd.DataFrame([{"method_id": method, "status": "not_applicable", "reason": reason, "included_in_denominator": False}
                  for method, reason in config["not_applicable_methods"].items()]).to_csv(not_applicable, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "gate": "FULL_BENCHMARK", "milestone": "FB4A_GSE279771_insertion_construction",
        "status": "pass", "representation_id": config["representation_id"], "input_track": "I",
        "methods_expected": len(config["methods"]), "methods_passed": len(rows),
        "standard_fragment_alias": False, "rna_accessed": False, "metadata_accessed": False,
        "policy_refit_run": False, "universal_ranking_run": False, "final_ranking_run": False,
        "insertion_matrix_audit": {"path": str(audit_path.relative_to(root)), "sha256": sha256(audit_path)},
        "method_manifest": {"path": str(method_manifest.relative_to(root)), "sha256": sha256(method_manifest)},
        "software_not_applicable": {"path": str(not_applicable.relative_to(root)), "sha256": sha256(not_applicable)},
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256(authorization_path)},
        "implementation": {"path": "spatialgabench/full_benchmark_gse279771.py", "sha256": sha256(Path(__file__))},
    }
    evidence_path = output / "construction_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence_path


def evaluate(config_path: Path, authorization_path: Path, root: Path) -> Path:
    config, authorization = _validate_authorization(config_path, authorization_path, root, "FB4B_GSE279771_paired_RNA_evaluation")
    construction = _mapping(root / authorization["construction_evidence"]["path"])
    if construction.get("status") != "pass" or construction.get("rna_accessed") is not False:
        raise ValueError("FB4A construction evidence is not eligible for RNA evaluation")
    rna_path = _check_artifact(root, config["rna_h5"], "rna_h5")
    method_manifest = pd.read_csv(root / construction["method_manifest"]["path"], sep="\t")
    from spatialgabench.g4_m3b_rna import _canonicalize_ga_columns, _evaluate_method, _rna_ranks, normalize_total_log1p

    rna, barcodes, var = _read_10x_rna_h5(rna_path)
    first = ad.read_h5ad(root / method_manifest.iloc[0].output_path, backed="r")
    obs_names = first.obs_names.copy(); first.file.close()
    barcode_index = {barcode: index for index, barcode in enumerate(barcodes)}
    if any(barcode not in barcode_index for barcode in obs_names):
        raise ValueError("FB4 RNA does not cover every insertion observation")
    rna = rna[np.asarray([barcode_index[x] for x in obs_names]), :].tocsr()
    gene_ids = _strip_ensembl_versions(var["id"])
    gene_names = var["name"].astype(str) if "name" in var else gene_ids
    genes = pd.read_csv(root / config["references"]["genes"]["path"], sep="\t", header=None,
                        names=["chr", "start", "end", "gene_id", "score", "strand", "gene_name", "gene_type"])
    genes["canonical_gene_id"] = _strip_ensembl_versions(genes["gene_id"])
    protein_ids = set(genes.canonical_gene_id)
    detection = np.asarray((rna > 0).sum(axis=0)).ravel()
    threshold = math.ceil(config["rna_gene_eligibility"]["minimum_detected_spot_fraction"] * len(obs_names))
    eligible_mask = gene_ids.isin(protein_ids).to_numpy() & (detection >= threshold)
    eligible_indices = np.flatnonzero(eligible_mask)
    eligible = pd.DataFrame({
        "gene_id": gene_ids.iloc[eligible_indices].to_numpy(),
        "gene_name": gene_names.iloc[eligible_indices].to_numpy(),
        "detected_spots": detection[eligible_indices],
        "detected_spot_fraction": detection[eligible_indices] / len(obs_names),
    })
    if eligible.empty:
        raise ValueError("FB4 RNA eligible-gene panel is empty after canonical ID matching")
    normalized = normalize_total_log1p(rna, config["normalization"]["target_sum"])
    ranks = _rna_ranks(normalized, eligible_indices, config["correlation_chunk_genes"])
    name_counts = genes.groupby("gene_name").canonical_gene_id.agg(list)
    unique_names = {name: ids[0] for name, ids in name_counts.items() if len(set(ids)) == 1}
    metric_rows, gene_frames = [], []
    for row in method_manifest.itertuples(index=False):
        record = pd.Series({
            "study_id": config["study_id"], "dataset_id": config["dataset_id"], "sample_id": config["sample_id"],
            "genome_build": "hg38", "representation_id": config["representation_id"], "input_track": "I",
            "method_id": row.method_id,
            "method_category": "negative_control" if row.method_id == "accessibility_matched_random_peak_to_gene_control_pc" else "hand_written_annotation",
            "gene_activity_h5ad": row.output_path, "gene_activity_sha256": row.output_sha256,
        })
        metric, gene_frame = _evaluate_method(
            record, root, obs_names, eligible, normalized, eligible_indices, ranks, unique_names,
            config["normalization"]["target_sum"], config["correlation_chunk_genes"],
        )
        metric_rows.append(metric); gene_frames.append(gene_frame)
    metrics, gene_metrics = pd.DataFrame(metric_rows), pd.concat(gene_frames, ignore_index=True)
    primary = config["primary_endpoint"]
    if not np.isfinite(metrics[primary].to_numpy(float)).all():
        raise ValueError("FB4 primary RNA endpoint contains non-finite values")
    expected_primary = metrics.median_gene_wise_spearman * metrics.evaluable_gene_fraction
    if not np.allclose(metrics[primary], expected_primary, rtol=1e-12, atol=1e-12):
        raise ValueError("FB4 primary RNA endpoint formula mismatch")
    control = float(metrics.loc[metrics.method_id.eq("accessibility_matched_random_peak_to_gene_control_pc"), primary].iloc[0])
    metrics["accessibility_matched_control"] = control
    metrics["delta_vs_accessibility_matched_control"] = metrics[primary] - control
    output = root / config["output_root"] / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path, genes_path, eligible_path = output / "rna_metrics.tsv", output / "gene_level_correlations.tsv.gz", output / "eligible_genes.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    gene_metrics.to_csv(genes_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    eligible.to_csv(eligible_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "gate": "FULL_BENCHMARK", "milestone": "FB4B_GSE279771_paired_RNA_evaluation",
        "status": "pass", "methods_evaluated": len(metrics), "rna_eligible_genes": len(eligible),
        "observations": len(obs_names), "annotation_methods_positive_vs_matched_control": int(
            (metrics.loc[metrics.method_category.eq("hand_written_annotation"), "delta_vs_accessibility_matched_control"] > 0).sum()),
        "representation_id": config["representation_id"], "input_track": "I",
        "rna_used_for_construction": False, "metadata_accessed": False, "pathology_truth_available": False,
        "independent_pathology_claim_allowed": False, "policy_refit_run": False,
        "universal_ranking_run": False, "final_ranking_run": False,
        "prior_rna_schema_value_access_disclosed": True,
        "metrics": {"path": str(metrics_path.relative_to(root)), "sha256": sha256(metrics_path)},
        "gene_metrics": {"path": str(genes_path.relative_to(root)), "sha256": sha256(genes_path)},
        "eligible_genes": {"path": str(eligible_path.relative_to(root)), "sha256": sha256(eligible_path)},
        "construction_evidence": authorization["construction_evidence"],
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256(authorization_path)},
        "implementation": {"path": "spatialgabench/full_benchmark_gse279771.py", "sha256": sha256(Path(__file__))},
    }
    evidence_path = output / "evaluation_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["construct", "evaluate"])
    parser.add_argument("--config", default="configs/full_benchmark_fb4_gse279771_insertion_rna.yaml")
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root, config_path, authorization_path = Path(args.root).resolve(), Path(args.config).resolve(), Path(args.authorization).resolve()
    result = construct(config_path, authorization_path, root) if args.command == "construct" else evaluate(config_path, authorization_path, root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
