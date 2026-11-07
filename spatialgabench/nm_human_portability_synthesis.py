"""Synthesize the common NU1/NU2 human portability contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from spatialgabench.nm_human_method_setup import ANNOTATION_METHODS, NU1, NU2, SOFTWARE_METHODS


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


def synthesize(completion_path: str | Path, root: str | Path = ".") -> Path:
    root, completion_path = Path(root).resolve(), Path(completion_path).resolve()
    completion = _json(completion_path)
    if completion.get("status") != "pass" or completion.get("contracts_accounted") != 50:
        raise ValueError("NU human method construction is not fully accounted")
    summary_path = root / completion["run_summary"]["path"]
    if _sha256(summary_path) != completion["run_summary"]["sha256"]:
        raise ValueError("NU human run summary checksum mismatch")
    summary = pd.read_csv(summary_path, sep="\t")
    common_annotation = ANNOTATION_METHODS[:-1]
    common = summary.loc[
        summary.method_id.isin(common_annotation + SOFTWARE_METHODS)
        & summary.dataset_id.isin([NU1, NU2])
    ].copy()
    if len(common) != 46 or common.groupby("dataset_id").size().to_dict() != {NU1: 23, NU2: 23}:
        raise ValueError("NU human exact common portability panel changed")
    common["portable"] = common.outcome.eq("success")
    common["failure_retained"] = common.outcome.eq("structured_failure")
    dataset = (
        common.groupby(["block", "study_id", "dataset_id", "species", "genome_build"], dropna=False)
        .agg(
            planned_contracts=("portable", "size"),
            successful_contracts=("portable", "sum"),
            structured_failures=("failure_retained", "sum"),
        )
        .reset_index()
    )
    dataset["portability_success_rate"] = dataset.successful_contracts / dataset.planned_contracts
    dataset["independent_study_increment"] = 0
    dataset["status"] = "accounted"

    output = root / "results/v1/nm_upgrade/nu2_human_portability"
    output.mkdir(parents=True, exist_ok=True)
    common_path = output / "common_contract_outcomes.tsv"
    dataset_path = output / "dataset_summary.tsv"
    common.to_csv(common_path, sep="\t", index=False)
    dataset.to_csv(dataset_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "milestone": "NU2_human_standard_fragment_portability",
        "status": "pass",
        "datasets": 2,
        "study_families_observed": 2,
        "independent_study_increment": 0,
        "common_contracts_planned": 46,
        "common_contracts_accounted": len(common),
        "successful_contracts": int(common.portable.sum()),
        "structured_failures": int(common.failure_retained.sum()),
        "rna_accessed": False,
        "labels_pathology_images_accessed": False,
        "performance_ranking_run": False,
        "policy_refit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "interpretation_boundary": "within_existing_family_human_portability_only",
        "common_contract_outcomes": {"path": str(common_path.relative_to(root)), "sha256": _sha256(common_path)},
        "dataset_summary": {"path": str(dataset_path.relative_to(root)), "sha256": _sha256(dataset_path)},
        "method_completion": {"path": str(completion_path.relative_to(root)), "sha256": _sha256(completion_path)},
        "implementation": {"path": "spatialgabench/nm_human_portability_synthesis.py", "sha256": _sha256(Path(__file__))},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def validate(evidence_path: str | Path, root: str | Path = ".") -> Path:
    root, evidence_path = Path(root).resolve(), Path(evidence_path).resolve()
    evidence = _json(evidence_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({"check_id": check_id, "observed": observed, "expected": expected, "pass": bool(observed == expected)})

    check("status", evidence.get("status"), "pass")
    check("contracts", evidence.get("common_contracts_accounted"), 46)
    check("independent_increment", evidence.get("independent_study_increment"), 0)
    for name in ("common_contract_outcomes", "dataset_summary", "method_completion", "implementation"):
        record = evidence[name]
        check(f"checksum__{name}", _sha256(root / record["path"]), record["sha256"])
    common = pd.read_csv(root / evidence["common_contract_outcomes"]["path"], sep="\t")
    dataset = pd.read_csv(root / evidence["dataset_summary"]["path"], sep="\t")
    check("common_rows", len(common), 46)
    check("dataset_rows", len(dataset), 2)
    check("each_dataset_23", bool(common.groupby("dataset_id").size().eq(23).all()), True)
    check("success_accounting", int(common.portable.astype(bool).sum()), int(evidence["successful_contracts"]))
    check("failure_accounting", int(common.failure_retained.astype(bool).sum()), int(evidence["structured_failures"]))
    check("all_accounted", bool(common.portable.astype(bool).ne(common.failure_retained.astype(bool)).all()), True)
    check("dataset_accounting", bool((dataset.successful_contracts + dataset.structured_failures).eq(dataset.planned_contracts).all()), True)
    for field in ("rna_accessed", "labels_pathology_images_accessed", "performance_ranking_run", "policy_refit_run", "universal_ranking_run", "final_ranking_run"):
        check(f"boundary_false__{field}", evidence.get(field), False)
    table = pd.DataFrame(checks)
    table_path = evidence_path.parent / "validation_checks.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    passed = int(table["pass"].sum())
    audit = {
        "schema_version": 1,
        "milestone": evidence["milestone"],
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "validation_checks": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path)},
    }
    audit_path = evidence_path.parent / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        raise ValueError(f"NU human portability validation failed: {table.loc[~table['pass'], 'check_id'].tolist()}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["synthesize", "validate"])
    parser.add_argument("--completion", default="results/v1/nm_upgrade/nu1_nu2_human_methods/evidence/completion_evidence.json")
    parser.add_argument("--evidence", default="results/v1/nm_upgrade/nu2_human_portability/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(synthesize(args.completion, args.root) if args.command == "synthesize" else validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
