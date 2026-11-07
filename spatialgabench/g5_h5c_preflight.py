"""Freeze H5C evaluation inputs and implementation without computing outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import yaml

from spatialgabench.g4_m3b_rna import _gtf_annotations


FALSE_FIELDS = (
    "evaluation_run",
    "rna_metric_computation_run",
    "regret_computation_run",
    "policy_refit_run",
    "method_rerun",
    "GSE279771_accessed",
    "full_benchmark_run",
    "results_writing_run",
    "universal_ranking_run",
    "final_ranking_run",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wide_csv_identity(path: Path) -> tuple[list[str], list[str]]:
    """Read barcode and gene identities while never parsing expression entries."""
    genes: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n")
        if not header:
            raise ValueError("RNA CSV header is empty")
        barcodes = next(csv.reader([header]))
        if barcodes and barcodes[0] == "":
            barcodes = barcodes[1:]
        for line in handle:
            if not line.strip():
                continue
            gene, separator, _unparsed_counts = line.partition(",")
            if not gene or not separator:
                raise ValueError("RNA CSV row has no gene/count separator")
            genes.append(gene[1:-1] if gene.startswith('"') and gene.endswith('"') else gene)
    return barcodes, genes


def _position_barcodes(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [row[0] for row in csv.reader(handle) if row]


def _canonical_rna_gene_ids(
    identifiers: list[str], biotypes: dict[str, str], unique_names: dict[str, str]
) -> list[str | None]:
    mapped: list[str | None] = []
    for identifier in identifiers:
        direct = re.sub(r"\.\d+$", "", identifier)
        mapped.append(direct if direct in biotypes else unique_names.get(identifier))
    return mapped


def run(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _yaml(config_path)
    for key in (
        "previous_gate",
        "h4_gate",
        "h5b_completion_evidence",
        "leaf_manifest",
        "evaluation_manifest",
        "comparator_manifest",
        "h5b_run_summary",
        "rna_matrix",
        "spatial_positions",
        "gene_annotation",
        "deviation_amendment",
    ):
        path = root / config[key]
        if not path.is_file() or sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"H5C preflight frozen artifact mismatch: {key}")
    invalid = [field for field in FALSE_FIELDS if config.get(field) is not False]
    if invalid:
        raise PermissionError(f"H5C preflight forbidden flags are not false: {invalid}")
    previous = _json(root / config["previous_gate"])
    h4 = _json(root / config["h4_gate"])
    h5b = _json(root / config["h5b_completion_evidence"])
    if (
        previous.get("status") != "pass"
        or previous.get("runs_passed") != 50
        or previous.get("rna_value_access_run") is not False
        or previous.get("next_milestone_authorized") is not False
    ):
        raise ValueError("H5B completion boundary is invalid")
    if h4.get("status") != "pass" or h4.get("admissible_evaluation_rows") != 69:
        raise ValueError("H4 evaluation protocol is invalid")
    if h5b.get("status") != "pass" or h5b.get("runs_passed") != 50:
        raise ValueError("H5B output evidence is invalid")

    leaves = pd.read_csv(root / config["leaf_manifest"], sep="\t")
    evaluations = pd.read_csv(root / config["evaluation_manifest"], sep="\t")
    comparators = pd.read_csv(root / config["comparator_manifest"], sep="\t")
    construction = pd.read_csv(root / config["h5b_run_summary"], sep="\t")
    if len(leaves) != 15 or len(evaluations) != 69 or len(comparators) != 15 or len(construction) != 50:
        raise ValueError("H5C frozen manifest dimensions changed")
    construction_keys = set(
        zip(construction.study_id, construction.representation_id, construction.method_id)
    )
    evaluation_keys = set(zip(evaluations.study_id, evaluations.representation_id, evaluations.method_id))
    if not evaluation_keys <= construction_keys:
        raise ValueError("H5C evaluation plan references a missing H5B output")
    if not all(construction["status"] == "pass"):
        raise ValueError("H5C construction status is not fully frozen as pass")

    method_rows: list[dict[str, Any]] = []
    for row in construction.itertuples(index=False):
        output = root / str(row.output_dir) / "gene_activity.h5ad"
        if not output.is_file() or sha256(output) != row.output_h5ad_sha256:
            raise ValueError(f"H5C live output hash mismatch: {output}")
        method_rows.append(
            {
                "study_id": row.study_id,
                "accession": row.accession,
                "dataset_id": row.dataset_id,
                "sample_id": row.sample_id,
                "representation_id": row.representation_id,
                "input_track": row.input_track,
                "method_id": row.method_id,
                "method_category": (
                    "negative_control"
                    if "control" in str(row.method_id)
                    else "fragment_software"
                    if row.run_type == "software"
                    else "hand_written_annotation"
                ),
                "gene_activity_h5ad": str(output.relative_to(root)),
                "gene_activity_sha256": row.output_h5ad_sha256,
                "construction_status": row.status,
            }
        )
    methods = pd.DataFrame(method_rows).sort_values(
        ["study_id", "representation_id", "method_id"], kind="mergesort"
    )

    rna_barcodes, genes = wide_csv_identity(root / config["rna_matrix"])
    position_barcodes = _position_barcodes(root / config["spatial_positions"])
    first_path = root / methods[methods.dataset_id == "gse308623_p2s3"].iloc[0].gene_activity_h5ad
    first = ad.read_h5ad(first_path, backed="r")
    output_barcodes = first.obs_names.astype(str).tolist()
    output_shape = first.shape
    first.file.close()
    if (
        len(rna_barcodes) != int(config["expected_rna_observations"])
        or len(genes) != int(config["expected_rna_genes"])
        or len(output_barcodes) != int(config["expected_rna_observations"])
        or set(rna_barcodes) != set(output_barcodes)
        or not set(output_barcodes) <= set(position_barcodes)
    ):
        raise ValueError("H5C RNA/ATAC/spatial identity contract failed")
    rna_index = {barcode: index for index, barcode in enumerate(rna_barcodes)}
    biotypes, unique_names = _gtf_annotations(root / config["gene_annotation"])
    mapped = _canonical_rna_gene_ids(genes, biotypes, unique_names)
    mapped_biotypes = [biotypes.get(gene_id, "") if gene_id else "" for gene_id in mapped]
    if not any(value == "protein_coding" for value in mapped_biotypes):
        raise ValueError("H5C RNA genes do not map to canonical protein-coding annotation")

    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    method_path = output / "method_input_manifest.tsv"
    observation_path = output / "rna_observation_contract.tsv"
    gene_path = output / "rna_gene_identity_preflight.tsv"
    implementation_path = output / "implementation_manifest.tsv"
    methods.to_csv(method_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "barcode": output_barcodes,
            "gene_activity_order": range(len(output_barcodes)),
            "rna_column": [rna_index[barcode] for barcode in output_barcodes],
            "in_spatial_positions": [barcode in set(position_barcodes) for barcode in output_barcodes],
            "evaluation_included": True,
        }
    ).to_csv(observation_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "rna_gene_name": genes,
            "canonical_gene_id": mapped,
            "canonical_gene_biotype": mapped_biotypes,
            "unique_gtf_name_mapping": [value is not None for value in mapped],
        }
    ).to_csv(gene_path, sep="\t", index=False)
    implementation_rows = []
    for relative in config["implementation_files"]:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"H5C implementation file is missing: {relative}")
        implementation_rows.append(
            {"path": relative, "sha256": sha256(path), "size_bytes": path.stat().st_size}
        )
    pd.DataFrame(implementation_rows).to_csv(implementation_path, sep="\t", index=False)

    evidence = {
        "schema_version": 1,
        "gate": "G5",
        "milestone": "H5C_external_evaluation_preflight",
        "status": "pass",
        "study_leaf_units": len(leaves),
        "admissible_evaluation_rows": len(evaluations),
        "comparator_rows": len(comparators),
        "frozen_method_outputs": len(methods),
        "rna_method_and_control_outputs": int((methods.dataset_id == "gse308623_p2s3").sum()),
        "rna_observations": len(output_barcodes),
        "rna_gene_rows": len(genes),
        "mapped_gtf_gene_names": sum(value is not None for value in mapped),
        "mapped_protein_coding_gene_names": sum(value == "protein_coding" for value in mapped_biotypes),
        "rna_matrix_format": "gene_by_spot_wide_csv_gzip",
        "rna_values_parsed_by_preflight": False,
        "limited_preauthorization_access_deviation_disclosed": True,
        "minimum_detected_spot_fraction": config["minimum_detected_spot_fraction"],
        "normalization_target_sum": config["normalization_target_sum"],
        "correlation_chunk_genes": config["correlation_chunk_genes"],
        "primary_endpoint": config["primary_endpoint"],
        "utility_transform": config["utility_transform"],
        "random_draws": config["random_draws"],
        "random_seed": config["random_seed"],
        "regret_margin": config["regret_margin"],
        "failure_margin": config["failure_margin"],
        "output_root": config["evaluation_output_root"],
        "exact_command": config["exact_command"],
        "method_input_manifest": {
            "path": str(method_path.relative_to(root)),
            "sha256": sha256(method_path),
        },
        "rna_observation_contract": {
            "path": str(observation_path.relative_to(root)),
            "sha256": sha256(observation_path),
        },
        "rna_gene_identity_preflight": {
            "path": str(gene_path.relative_to(root)),
            "sha256": sha256(gene_path),
        },
        "implementation_manifest": {
            "path": str(implementation_path.relative_to(root)),
            "sha256": sha256(implementation_path),
        },
        "evaluation_manifest": {"path": config["evaluation_manifest"], "sha256": config["evaluation_manifest_sha256"]},
        "comparator_manifest": {"path": config["comparator_manifest"], "sha256": config["comparator_manifest_sha256"]},
        "rna_matrix": {"path": config["rna_matrix"], "sha256": config["rna_matrix_sha256"]},
        "gene_annotation": {"path": config["gene_annotation"], "sha256": config["gene_annotation_sha256"]},
        "deviation_amendment": {"path": config["deviation_amendment"], "sha256": config["deviation_amendment_sha256"]},
        "implementation": {
            "path": "spatialgabench/g5_h5c_evaluation.py",
            "sha256": sha256(root / "spatialgabench/g5_h5c_evaluation.py"),
        },
        "source_gene_activity_shape_reference_only": list(output_shape),
        "evaluation_run": False,
        "rna_metric_computation_run": False,
        "regret_computation_run": False,
        "policy_refit_run": False,
        "method_rerun": False,
        "GSE279771_accessed": False,
        "full_benchmark_run": False,
        "results_writing_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "next_milestone_authorized": False,
    }
    evidence_path = output / "completion_evidence.json"
    _write_json(evidence_path, evidence)
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g5_h5c_external_evaluation_preflight.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    path = run(args.config, args.root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
