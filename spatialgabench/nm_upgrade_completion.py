"""Final experiment-only completion gate for the NM evidence upgrade."""

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


MILESTONES = {
    "FB1_three_study_RNA": "results/v1/full_benchmark/fb1_three_study_rna_synthesis/evidence/completion_audit.json",
    "FB2_seven_study_portability": "results/v1/full_benchmark/fb2_seven_study_portability/evidence/completion_audit.json",
    "FB3_two_study_depth_stress": "results/v1/full_benchmark/fb3_depth_resource_stress/evidence/completion_audit.json",
    "FB4_GSE279771_insertion_RNA": "results/v1/full_benchmark/fb4_gse279771_insertion_rna/evidence/completion_audit.json",
    "NU1_NU2_human_representations": "results/v1/nm_upgrade/nu1_nu2_human_representations/evidence/completion_audit.json",
    "NU1_NU2_human_methods": "results/v1/nm_upgrade/nu1_nu2_human_methods/evidence/completion_audit.json",
    "NU2_human_portability": "results/v1/nm_upgrade/nu2_human_portability/completion_audit.json",
    "NU1_human_RNA": "results/v1/nm_upgrade/nu1_human_rna/evidence/completion_audit.json",
    "NU3_external_depth_stress": "results/v1/nm_upgrade/nu3_external_depth_stress/evidence/completion_audit.json",
    "NU4_spatial_programs": "results/v1/nm_upgrade/nu4_spatial_programs/evidence/completion_audit.json",
}


def run(root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for milestone, relative in MILESTONES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Experiment completion waits for: {milestone}")
        evidence = _json(path)
        rows.append({
            "milestone": milestone,
            "status": evidence.get("status"),
            "evidence_path": relative,
            "evidence_sha256": _sha256(path),
        })
    matrix = pd.DataFrame(rows)
    if not matrix.status.eq("pass").all():
        raise ValueError(f"Experiment completion has non-pass milestones: {matrix.loc[~matrix.status.eq('pass'), 'milestone'].tolist()}")

    fb1 = pd.read_csv(root / "results/v1/full_benchmark/fb1_three_study_rna_synthesis/matched_control_deltas.tsv", sep="\t")
    nu1 = pd.read_csv(root / "results/v1/nm_upgrade/nu1_human_rna/matched_control_deltas.tsv", sep="\t")
    fb4 = pd.read_csv(root / "results/v1/full_benchmark/fb4_gse279771_insertion_rna/evaluation/rna_metrics.tsv", sep="\t")
    nu2 = _json(root / "results/v1/nm_upgrade/nu2_human_portability/completion_evidence.json")
    fb3 = _json(root / "results/v1/full_benchmark/fb3_depth_resource_stress/evidence/completion_audit.json")
    nu3 = _json(root / "results/v1/nm_upgrade/nu3_external_depth_stress/evidence/completion_audit.json")
    nu4 = _json(root / "results/v1/nm_upgrade/nu4_spatial_programs/completion_evidence.json")
    nu4_delta = pd.read_csv(root / nu4["artifacts"]["matched_control_deltas.tsv"]["path"], sep="\t")

    fb1_positive = int(fb1.delta_vs_accessibility_matched_control.gt(0).sum())
    nu1_positive = int(nu1.delta_vs_matched_control.gt(0).sum())
    fb4_annotation = fb4.loc[~fb4.method_id.eq("accessibility_matched_random_peak_to_gene_control_pc")]
    fb4_positive = int(fb4_annotation.delta_vs_accessibility_matched_control.gt(0).sum())
    nu4_positive = int(nu4_delta.delta_heldout_component_spearman_vs_control.gt(0).sum())

    output = root / "results/v1/nm_upgrade"
    matrix_path = output / "experiment_completion_matrix.tsv"
    matrix.to_csv(matrix_path, sep="\t", index=False)
    report_path = root / "reports/NM_UPGRADE_EXPERIMENT_COMPLETION_AUDIT_2026-07-30.md"
    report = f"""# NM evidence-upgrade experiment completion audit

Status: **PASS**

This gate covers experiments and evidence validation only. Figure production,
manuscript drafting, license review, offline reconstruction, and release packaging
remain explicitly deferred.

## Completed evidence

- All {len(matrix)} required FB/NU milestone audits pass.
- NU2 human common-contract portability: {nu2['successful_contracts']}/46 successful,
  with {nu2['structured_failures']} structured failures retained.
- Controlled depth stress: FB3 {fb3['successful_contracts']}/{fb3['planned_contracts']}
  successful; NU3 {nu3['successful_contracts']}/{nu3['planned_contracts']} successful.
- Standard paired-RNA matched-control directions: FB1 {fb1_positive}/{len(fb1)}
  positive and NU1 human {nu1_positive}/{len(nu1)} positive.
- GSE279771 insertion-track annotation directions: {fb4_positive}/{len(fb4_annotation)}
  above its matched control; this remains directional paired-RNA evidence, not
  independent pathology truth.
- Held-out spatial programs: {nu4_positive}/{len(nu4_delta)} annotation-contract
  deltas exceed their representation-matched controls across five samples; fold ×
  component units are descriptive repeated measures, not independent studies.

## Claim boundaries

- Study family, not sample, is the independent generalization unit.
- P, B, F, and I tracks remain separate; fixed-bin and author/peak-like outputs are
  not pooled into a universal score.
- RNA was evaluation-only. Labels, clusters, pathology, and RNA did not alter gene
  activity construction or method parameters.
- Portability success is not biological accuracy and ceiling results cannot support
  method superiority.
- Runtime and memory remain descriptive within contract; subset-only methods remain
  separate.
- No universal or final ranking was run.
- No independent pathology truth is available for GSE279771; pathology claims remain
  excluded.

## Next gate

The project may now enter statistical consolidation and figure specification. It
must not enter release packaging until manuscript claims, licenses, and offline
reconstruction receive separate review.
"""
    report_path.write_text(report, encoding="utf-8")
    gate = {
        "schema_version": 1,
        "gate": "NM_UPGRADE_EXPERIMENT_COMPLETION",
        "date": "2026-07-30",
        "status": "pass",
        "milestones_required": len(matrix),
        "milestones_passed": int(matrix.status.eq("pass").sum()),
        "figures_generated": False,
        "manuscript_drafted": False,
        "release_packaging_run": False,
        "rna_used_for_construction": False,
        "labels_clusters_pathology_used_for_construction": False,
        "policy_refit_run": False,
        "input_tracks_pooled": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "experiment_completion_matrix": {"path": str(matrix_path.relative_to(root)), "sha256": _sha256(matrix_path)},
        "report": {"path": str(report_path.relative_to(root)), "sha256": _sha256(report_path)},
        "implementation": {"path": "spatialgabench/nm_upgrade_completion.py", "sha256": _sha256(Path(__file__))},
    }
    gate_path = root / "governance/gates/NM_UPGRADE_EXPERIMENT_COMPLETION_2026-07-30.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(run(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
