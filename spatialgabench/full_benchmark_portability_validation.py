"""Independent completion validator for FB2 portability synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def validate(evidence_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    evidence_path = Path(evidence_path).resolve()
    evidence = _mapping(evidence_path)
    config_record = evidence["config"]
    config_path = root / config_record["path"]
    config = _mapping(config_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({
            "check_id": check_id,
            "observed": observed,
            "expected": expected,
            "pass": bool(observed == expected),
        })

    check("evidence_status", evidence.get("status"), "pass")
    check("config_checksum", _sha256(config_path), config_record["sha256"])
    artifacts = evidence["artifacts"]
    for name, record in artifacts.items():
        path = root / record["path"]
        check(f"artifact_checksum__{name}", _sha256(path), record["sha256"])

    manifest = pd.read_csv(root / artifacts["common_contract_manifest.tsv"]["path"], sep="\t")
    check("common_rows", len(manifest), 161)
    check("study_families", manifest.study_id.nunique(), 7)
    check("contracts_per_study", bool(manifest.groupby("study_id").size().eq(23).all()), True)
    check("unique_contract_keys", int(manifest.duplicated(["study_id", "representation_id", "method_id"]).sum()), 0)
    check("development_studies", manifest.loc[manifest.study_role.eq("development"), "study_id"].nunique(), 5)
    check("external_studies", manifest.loc[manifest.study_role.eq("external"), "study_id"].nunique(), 2)
    check("human_hg38_studies", manifest.loc[manifest.genome_build.eq("hg38"), "study_id"].nunique(), 1)
    check("all_contracts_success", int(manifest.success.astype(bool).sum()), 161)
    check("resource_comparison_forbidden", bool((~manifest.resource_comparison_allowed.astype(bool)).all()), True)

    expected_pairs = {
        (representation, method)
        for representation in config["matrix_representations"]
        for method in config["annotation_methods"]
    } | {
        (config["fragment_representation"], method) for method in config["software_methods"]
    }
    observed_pairs = set(zip(manifest.representation_id, manifest.method_id))
    check("exact_common_contract_panel", observed_pairs, expected_pairs)

    summary = pd.read_csv(root / artifacts["success_rate_wilson_summary.tsv"]["path"], sep="\t")
    overall = summary[(summary.grouping_scope == "overall") & (summary.grouping_value == "all")].iloc[0]
    check("overall_successes", int(overall.successful_contracts), 161)
    check("overall_failures", int(overall.failed_contracts), 0)
    check("overall_wilson_lower_rounded", round(float(overall.wilson_95_ci_low), 6), 0.976696)

    coverage = pd.read_csv(root / artifacts["seven_study_coverage_matrix.tsv"]["path"], sep="\t")
    check("coverage_rows", len(coverage), 23)
    check("all_pairs_cover_seven", bool(coverage.studies_covered.eq(7).all()), True)

    cicero = pd.read_csv(root / artifacts["subset_cicero_accounting.tsv"]["path"], sep="\t")
    check("cicero_rows", len(cicero), 20)
    check("cicero_eligible_pass", int((cicero.eligible.astype(bool) & cicero.run_status.eq("pass")).sum()), 14)
    check("cicero_not_applicable", int(cicero.run_status.eq("not_applicable").sum()), 6)

    for field in (
        "method_superiority_inference_allowed", "resource_measurements_comparative",
        "subset_cicero_in_common_denominator", "construction_rerun", "policy_refit_run",
        "resource_ranking_run", "universal_ranking_run", "final_ranking_run",
    ):
        check(f"boundary_false__{field}", evidence.get(field), False)

    table = pd.DataFrame(checks)
    output = evidence_path.parent / "evidence"
    output.mkdir(exist_ok=True)
    table_path = output / "validation_checks.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    passed = int(table["pass"].sum())
    audit = {
        "schema_version": 1,
        "milestone": "FB2_seven_study_portability_synthesis",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "all_pass_ceiling_confirmed": bool(manifest.success.astype(bool).all()),
        "method_superiority_inference_allowed": False,
        "resource_ranking_allowed": False,
        "validation_checks": {
            "path": str(table_path.relative_to(root)),
            "sha256": _sha256(table_path),
        },
        "completion_evidence": {
            "path": str(evidence_path.relative_to(root)),
            "sha256": _sha256(evidence_path),
        },
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n")
    if audit["status"] != "pass":
        failed = table.loc[~table["pass"], "check_id"].tolist()
        raise ValueError(f"FB2 independent validation failed: {failed}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="results/v1/full_benchmark/fb2_seven_study_portability/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
