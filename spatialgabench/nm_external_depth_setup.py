"""Materialize immutable GSE308623 full-depth anchors for NM-U3 stress."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardlink_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_path(target) != sha256_path(source):
            raise ValueError(f"Existing anchor differs from source: {target}")
        return
    os.link(source, target)


def setup(config_path: Path, root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    anchor = config["full_anchor"]
    for key in ("h5a_evidence", "h5b_evidence"):
        record = anchor[key]
        path = root / record["path"]
        if sha256_path(path) != record["sha256"]:
            raise ValueError(f"NU3 frozen anchor evidence mismatch: {key}")

    condition_id = anchor["condition_id"]
    base = root / config["output_root"] / "conditions" / condition_id
    source_root = root / anchor["h5b_root"]
    records: list[dict[str, Any]] = []
    for representation in config["matrix_representations"]:
        for method in config["annotation_methods"]:
            source_dir = source_root / "annotation/gse308623_p2s3" / representation / method
            target_dir = base / "methods/annotation" / condition_id / representation / method
            for name in ("gene_activity.h5ad", "process_tree_memory.json"):
                _hardlink_verified(source_dir / name, target_dir / name)
            manifest = {
                "schema_version": 1,
                "status": "pass",
                "anchor_reused": True,
                "source": str((source_dir / "gene_activity.h5ad").relative_to(root)),
                "source_sha256": sha256_path(source_dir / "gene_activity.h5ad"),
                "method_id": method,
                "representation_id": representation,
            }
            (target_dir / "output_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            records.append(manifest)

    for method in config["software_methods"]:
        source_dir = source_root / "software/gse308623_p2s3" / method
        target_dir = base / "methods/software" / method
        for name in ("gene_activity.h5ad", "process_tree_memory.json"):
            _hardlink_verified(source_dir / name, target_dir / name)
        manifest = {
            "schema_version": 1,
            "status": "pass",
            "anchor_reused": True,
            "source": str((source_dir / "gene_activity.h5ad").relative_to(root)),
            "source_sha256": sha256_path(source_dir / "gene_activity.h5ad"),
            "method_id": method,
            "representation_id": "filtered_fragments_depth_stress_contract",
        }
        (target_dir / "output_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append(manifest)

    stage_payload = {
        "schema_version": 1,
        "status": "pass",
        "condition_id": condition_id,
        "anchor_reused": True,
        "new_scientific_execution": False,
    }
    for relative in (
        "fragments/thinning_audit.json",
        "peaks/peak_stage_audit.json",
        "representations/matched_representation_validation.json",
    ):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stage_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([
        {"representation_id": representation, "available": True, "reason": "reused_frozen_full_anchor"}
        for representation in config["matrix_representations"]
    ]).to_csv(base / "representations/representation_availability.tsv", sep="\t", index=False)
    source_representation_manifest = pd.read_csv(
        root / "results/v1/g5/h5a_external_representations/gse308623_p2s3/matched/representation_manifest.tsv",
        sep="\t",
    )
    source_representation_manifest = source_representation_manifest.loc[
        source_representation_manifest.representation_id.isin(config["matrix_representations"])
    ].copy()
    source_representation_manifest["anchor_reused"] = True
    source_representation_manifest["new_scientific_execution"] = False
    source_representation_manifest.to_csv(
        base / "representations/representation_manifest.tsv", sep="\t", index=False
    )

    output = root / config["output_root"] / "full_anchor_manifest.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, sep="\t", index=False)
    evidence = {
        "schema_version": 1,
        "milestone": "NM-U3_external_full_depth_anchor_setup",
        "status": "pass" if len(records) == 13 else "blocked",
        "anchors": len(records),
        "new_scientific_executions": 0,
        "manifest": {"path": str(output.relative_to(root)), "sha256": sha256_path(output)},
        "rna_accessed": False,
        "policy_refit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    evidence_path = root / config["output_root"] / "full_anchor_setup_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nm_upgrade_external_depth_stress.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(setup(root / args.config, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
