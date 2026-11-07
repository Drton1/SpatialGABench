"""Independent completion validator for corrected FB1 RNA synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def validate(evidence_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    evidence_path = Path(evidence_path).resolve()
    evidence = _json(evidence_path)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({"check_id": check_id, "observed": observed, "expected": expected, "pass": bool(observed == expected)})

    check("evidence_status", evidence.get("status"), "pass")
    for name, record in evidence["artifacts"].items():
        path = root / record["path"]
        check(f"artifact_checksum__{name}", _sha256(path), record["sha256"])
        check(f"artifact_rows__{name}", len(pd.read_csv(path, sep="\t")), int(record["rows"]))

    common = pd.read_csv(root / evidence["artifacts"]["common_study_metrics.tsv"]["path"], sep="\t")
    check("common_rows", len(common), 81)
    check("study_families", common.study_family.nunique(), 3)
    check("development_studies", common.loc[common.study_role.eq("development"), "study_family"].nunique(), 2)
    check("external_studies", common.loc[common.study_role.eq("external"), "study_family"].nunique(), 1)
    check("all_common_status_pass", bool(common.status.eq("pass").all()), True)

    direction = pd.read_csv(root / evidence["artifacts"]["direction_replication.tsv"]["path"], sep="\t")
    check("direction_rows", len(direction), 20)
    check("positive_all_three", int(direction.positive_direction_all_three_studies.sum()), 20)

    ranges = pd.read_csv(root / evidence["artifacts"]["study_method_representation_ranges.tsv"]["path"], sep="\t")
    pairs = pd.read_csv(root / evidence["artifacts"]["paired_representation_effects.tsv"]["path"], sep="\t")
    interactions = pd.read_csv(root / evidence["artifacts"]["method_representation_interactions.tsv"]["path"], sep="\t")
    reversals = pd.read_csv(root / evidence["artifacts"]["ranking_reversals.tsv"]["path"], sep="\t")
    check("material_ranges", int(ranges.material_at_0_05.sum()), 0)
    check("material_pairs", int(pairs.material_at_0_05.sum()), 0)
    check("material_interactions", int(interactions.material_at_0_05.sum()), 0)
    check("interaction_estimates_inside_corresponding_ci", bool(
        ((interactions.interaction_estimate >= interactions.bootstrap_ci_low - 1e-15)
         & (interactions.interaction_estimate <= interactions.bootstrap_ci_high + 1e-15)).all()
    ), True)
    check("ranking_reversals", int(reversals.ranking_reversal_across_representations.sum()), 9)
    check("max_range_rounded", round(float(ranges.representation_range.max()), 7), 0.0422827)
    check("max_abs_pair_rounded", round(float(pairs.mean_paired_delta.abs().max()), 7), 0.0238807)
    check("max_abs_interaction_rounded", round(float(interactions.interaction_estimate.abs().max()), 7), 0.0029459)
    check("c1_status", evidence.get("c1_status"), "below_prespecified_materiality_despite_ranking_reversals")
    check("external_access_deviation_carried", evidence.get("external_rna_access_order_deviation_carried_forward"), True)
    for field in ("input_tracks_pooled", "policy_refit_run", "construction_rerun", "universal_ranking_run", "final_ranking_run"):
        check(f"boundary_false__{field}", evidence.get(field), False)

    table = pd.DataFrame(checks)
    output = evidence_path.parent / "evidence"
    output.mkdir(exist_ok=True)
    table_path = output / "validation_checks.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    passed = int(table["pass"].sum())
    audit = {
        "schema_version": 1,
        "milestone": "FB1_three_study_paired_RNA_representation_synthesis",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "c1_status": evidence["c1_status"],
        "external_rna_access_order_deviation_carried_forward": True,
        "validation_checks": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path)},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        raise ValueError(f"FB1 independent validation failed: {table.loc[~table['pass'], 'check_id'].tolist()}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="results/v1/full_benchmark/fb1_three_study_rna_synthesis/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
