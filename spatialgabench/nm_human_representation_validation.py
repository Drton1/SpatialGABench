"""Independent live-artifact audit for NU1/NU2 human representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import yaml


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
    root, evidence_path = Path(root).resolve(), Path(evidence_path).resolve()
    evidence = _json(evidence_path)
    config_path = root / evidence["config"]["path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(check_id: str, observed: Any, expected: Any) -> None:
        checks.append({"check_id": check_id, "observed": observed, "expected": expected, "pass": bool(observed == expected)})

    check("completion_status", evidence.get("status"), "pass")
    check("sample_count", evidence.get("samples"), 2)
    check("representation_count", evidence.get("representations"), 8)
    check("config_checksum", _sha256(config_path), evidence["config"]["sha256"])
    check("authorization_checksum", _sha256(root / evidence["authorization"]["path"]), evidence["authorization"]["sha256"])
    check("sample_summary_checksum", _sha256(root / evidence["sample_summary"]["path"]), evidence["sample_summary"]["sha256"])
    representations = 0
    for dataset_id, sample in config["samples"].items():
        base = root / config["output_root"] / dataset_id
        validation_path = base / "matched/matched_representation_validation.json"
        audit_path = base / "filtered_fragments/fragment_filter_audit.json"
        manifest_path = base / "matched/representation_manifest.tsv"
        validation, audit = _json(validation_path), _json(audit_path)
        manifest = pd.read_csv(manifest_path, sep="\t")
        check(f"validation_status__{dataset_id}", validation.get("status"), "pass")
        check(f"filter_status__{dataset_id}", audit.get("status"), "pass")
        check(f"fragment_contract__{dataset_id}", validation.get("fragment_filter_contract_verified"), True)
        positions = pd.read_csv(root / sample["positions"], header=None)
        expected_in_tissue = int(pd.to_numeric(positions.iloc[:, 1], errors="coerce").fillna(0).eq(1).sum())
        check(f"observation_count__{dataset_id}", validation.get("barcode_rows"), expected_in_tissue)
        check(f"manifest_rows__{dataset_id}", len(manifest), 4)
        check(f"manifest_status__{dataset_id}", bool(manifest.status.eq("pass").all()), True)
        check(f"manifest_unique__{dataset_id}", manifest.representation_id.nunique(), 4)
        check(f"fixed_bin_count_contract__{dataset_id}", bool(manifest.loc[manifest.input_track.eq("B"), "assigned_fragment_rows"].eq(validation["eligible_fragment_rows"]).all()), True)
        for row in manifest.itertuples(index=False):
            path = root / str(row.output_path)
            check(f"output_checksum__{dataset_id}__{row.representation_id}", _sha256(path), row.output_sha256)
            matrix = ad.read_h5ad(path, backed="r")
            shape = matrix.shape
            matrix.file.close()
            check(f"output_shape__{dataset_id}__{row.representation_id}", shape, (int(validation["barcode_rows"]), int(row.n_features)))
        representations += len(manifest)
    check("live_representation_total", representations, 8)
    for field in ("rna_accessed", "labels_pathology_images_accessed", "method_execution_run", "policy_refit_run", "universal_ranking_run", "final_ranking_run"):
        check(f"boundary_false__{field}", evidence.get(field), False)
    table = pd.DataFrame(checks)
    output = evidence_path.parent
    table_path = output / "validation_checks.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    passed = int(table["pass"].sum())
    audit = {
        "schema_version": 1,
        "milestone": "NM-U1-U2_human_matched_representation_construction",
        "status": "pass" if passed == len(table) else "fail",
        "checks_passed": passed,
        "checks_total": len(table),
        "samples": 2,
        "representations": 8,
        "rna_accessed": False,
        "validation_checks": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
        "completion_evidence": {"path": str(evidence_path.relative_to(root)), "sha256": _sha256(evidence_path)},
    }
    audit_path = output / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "pass":
        raise ValueError(f"NU human representation validation failed: {table.loc[~table['pass'], 'check_id'].tolist()}")
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="results/v1/nm_upgrade/nu1_nu2_human_representations/evidence/completion_evidence.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(validate(args.evidence, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
