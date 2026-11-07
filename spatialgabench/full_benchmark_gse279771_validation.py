"""Independent completion validation for FB4 GSE279771 extension."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def validate(root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    base = root / "results/v1/full_benchmark/fb4_gse279771_insertion_rna"
    construction_path = base / "construction/construction_evidence.json"
    evaluation_path = base / "evaluation/evaluation_evidence.json"
    construction, evaluation = _json(construction_path), _json(evaluation_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({"check_id": check_id, "observed": observed, "expected": expected, "pass": bool(observed == expected)})

    check("construction_status", construction.get("status"), "pass")
    check("construction_methods", construction.get("methods_passed"), 6)
    check("construction_input_track", construction.get("input_track"), "I")
    check("standard_fragment_alias", construction.get("standard_fragment_alias"), False)
    check("construction_rna_access", construction.get("rna_accessed"), False)
    check("construction_metadata_access", construction.get("metadata_accessed"), False)
    audit_path = root / construction["insertion_matrix_audit"]["path"]
    check("insertion_audit_checksum", _sha256(audit_path), construction["insertion_matrix_audit"]["sha256"])
    audit = _json(audit_path)
    check("source_rows", audit.get("source_rows"), 35798529)
    check("source_endpoint_events", audit.get("source_endpoint_events"), 71597058)
    check("retained_events", audit.get("retained_insertion_events"), 58892856)
    check("count_conservation", audit.get("matrix_total_counts"), 58892856)
    check("observations", audit.get("observations"), 2768)
    check("invalid_events", audit.get("invalid_events"), 0)
    check("excluded_barcodes", audit.get("excluded_barcode_events"), 0)

    manifest_path = root / construction["method_manifest"]["path"]
    check("method_manifest_checksum", _sha256(manifest_path), construction["method_manifest"]["sha256"])
    manifest = pd.read_csv(manifest_path, sep="\t")
    check("method_manifest_rows", len(manifest), 6)
    for row in manifest.itertuples(index=False):
        output = root / row.output_path
        check(f"method_output_checksum__{row.method_id}", _sha256(output), row.output_sha256)
        contract = pd.read_csv(output.parent / "contract_validation.tsv", sep="\t")
        check(f"method_contract_18_of_18__{row.method_id}", int(contract.status.eq("pass").sum()), 18)

    not_applicable_path = root / construction["software_not_applicable"]["path"]
    not_applicable = pd.read_csv(not_applicable_path, sep="\t")
    check("software_not_applicable_rows", len(not_applicable), 4)
    check("software_excluded_from_denominator", bool((~not_applicable.included_in_denominator.astype(bool)).all()), True)

    check("evaluation_status", evaluation.get("status"), "pass")
    check("evaluation_methods", evaluation.get("methods_evaluated"), 6)
    check("eligible_genes", evaluation.get("rna_eligible_genes"), 8100)
    check("evaluation_observations", evaluation.get("observations"), 2768)
    check("annotation_positive_vs_control", evaluation.get("annotation_methods_positive_vs_matched_control"), 5)
    check("metadata_not_accessed", evaluation.get("metadata_accessed"), False)
    check("pathology_truth_unavailable", evaluation.get("pathology_truth_available"), False)
    check("pathology_claim_forbidden", evaluation.get("independent_pathology_claim_allowed"), False)
    metrics_path = root / evaluation["metrics"]["path"]
    check("metrics_checksum", _sha256(metrics_path), evaluation["metrics"]["sha256"])
    metrics = pd.read_csv(metrics_path, sep="\t")
    primary = "coverage_adjusted_median_gene_wise_spearman"
    check("metrics_rows", len(metrics), 6)
    check("primary_finite", bool(np.isfinite(metrics[primary]).all()), True)
    check("primary_formula", bool(np.allclose(
        metrics[primary], metrics.median_gene_wise_spearman * metrics.evaluable_gene_fraction,
        rtol=1e-12, atol=1e-12,
    )), True)
    annotation = metrics[metrics.method_category.eq("hand_written_annotation")]
    check("all_five_annotation_deltas_positive", int((annotation.delta_vs_accessibility_matched_control > 0).sum()), 5)
    gene_metrics_path = root / evaluation["gene_metrics"]["path"]
    with gzip.open(gene_metrics_path, "rt", encoding="utf-8") as handle:
        gene_rows = sum(1 for _ in handle) - 1
    check("gene_metric_rows", gene_rows, 48600)

    for field in ("policy_refit_run", "universal_ranking_run", "final_ranking_run"):
        check(f"boundary_false__{field}", evaluation.get(field), False)
    check("reader_attempt_retained", (base / "attempts/attempt_1_feature_tag_length_mismatch").is_dir(), True)
    check("gene_id_attempt_retained", (base / "attempts/attempt_2_zero_eligible_gene_id_version_mismatch").is_dir(), True)
    for stage in ("construction", "evaluation"):
        memory = _json(base / stage / "process_tree_memory.json")
        check(f"{stage}_memory_status", memory.get("status"), "pass")
        check(f"{stage}_memory_exit", memory.get("exit_code"), 0)

    table = pd.DataFrame(checks)
    evidence_dir = base / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    table_path = evidence_dir / "validation_checks.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    passed = int(table["pass"].sum())
    result = {
        "schema_version": 1, "milestone": "FB4_GSE279771_insertion_track_paired_RNA",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed, "checks_total": len(table),
        "representation_id": "insertion_derived_fixed_bin_25kb", "input_track": "I",
        "independent_pathology_claim_allowed": False, "universal_ranking_allowed": False,
        "construction_evidence": {"path": str(construction_path.relative_to(root)), "sha256": _sha256(construction_path)},
        "evaluation_evidence": {"path": str(evaluation_path.relative_to(root)), "sha256": _sha256(evaluation_path)},
        "validation_checks": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
    }
    result_path = evidence_dir / "completion_audit.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "pass":
        raise ValueError(f"FB4 validation failed: {table.loc[~table['pass'], 'check_id'].tolist()}")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
