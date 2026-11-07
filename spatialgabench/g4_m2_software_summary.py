"""Independent evidence aggregation for G4-M2B software construction."""

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


def _id_hash(values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def summarize(config_path: str | Path, root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    panel = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset_id, sample in panel["samples"].items():
        for method_id in panel["methods"]:
            run_dir = root / panel["output_root"] / dataset_id / method_id
            paths = {name: run_dir / name for name in (
                "wrapper_config.yaml", "input_validation.json", "gene_activity.h5ad",
                "contract_validation.tsv", "runtime.tsv", "output_manifest.json",
                "process_tree_memory.json",
            )}
            if not all(path.is_file() for path in paths.values()):
                rows.append({"dataset_id": dataset_id, "study_id": sample["study_id"], "method_id": method_id, "status": "missing"})
                continue
            wrapper = yaml.safe_load(paths["wrapper_config.yaml"].read_text(encoding="utf-8"))
            validation = json.loads(paths["input_validation.json"].read_text(encoding="utf-8"))
            contract = pd.read_csv(paths["contract_validation.tsv"], sep="\t")
            runtime = pd.read_csv(paths["runtime.tsv"], sep="\t").iloc[0]
            manifest = json.loads(paths["output_manifest.json"].read_text(encoding="utf-8"))
            memory = json.loads(paths["process_tree_memory.json"].read_text(encoding="utf-8"))
            input_adata = ad.read_h5ad(root / wrapper["input_h5ad"], backed="r")
            output_adata = ad.read_h5ad(paths["gene_activity.h5ad"], backed="r")
            observation_match = _id_hash(input_adata.obs_names) == _id_hash(output_adata.obs_names)
            n_observations, n_genes = output_adata.shape
            input_adata.file.close()
            output_adata.file.close()
            artifact_hashes = all(
                (root / manifest[key]).is_file()
                and _sha256(root / manifest[key]) == manifest[f"{key}_sha256"]
                for key in ("output_h5ad", "contract_tsv", "runtime_tsv")
            )
            current_inputs_match = all(
                Path(validation["resolved_paths"][key]).is_file()
                and _sha256(Path(validation["resolved_paths"][key])) == digest
                for key, digest in validation["input_sha256"].items()
            )
            checks_passed = int((contract["status"] == "pass").sum())
            run_pass = all([
                validation.get("status") == "pass", manifest.get("status") == "pass",
                memory.get("status") == "pass", memory.get("exit_code") == 0,
                checks_passed == len(contract) == 18, observation_match, artifact_hashes,
                current_inputs_match, n_observations > 0, n_genes > 0,
                str(runtime["status"]) == "pass",
            ])
            rows.append({
                "dataset_id": dataset_id, "study_id": sample["study_id"],
                "sample_id": sample["sample_id"], "genome_build": sample["genome_build"],
                "method_id": method_id, "n_observations": n_observations,
                "n_output_genes": n_genes, "contract_checks_passed": checks_passed,
                "contract_checks_total": len(contract), "observation_ids_and_order_match": observation_match,
                "manifest_hashes_valid": artifact_hashes, "current_input_hashes_valid": current_inputs_match,
                "runtime_seconds_noncomparative": float(memory["runtime_seconds"]),
                "peak_aggregate_pss_mb_noncomparative": float(memory["peak_aggregate_pss_mb"]),
                "peak_aggregate_rss_mb_noncomparative": float(memory["peak_aggregate_rss_mb"]),
                "memory_samples": int(memory["sample_count"]),
                "fragment_sha256": validation["input_sha256"]["fragments"],
                "annotation_sha256": validation["input_sha256"]["gene_annotation"],
                "chrom_sizes_sha256": validation["input_sha256"]["chrom_sizes"],
                "output_manifest_sha256": _sha256(paths["output_manifest.json"]),
                "memory_evidence_sha256": _sha256(paths["process_tree_memory.json"]),
                "status": "pass" if run_pass else "fail",
            })
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "g4_m2_software_run_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    expected = len(panel["samples"]) * len(panel["methods"])
    passed = int((summary["status"] == "pass").sum())
    evidence = {
        "schema_version": 1, "gate": "G4", "milestone": "M2B_fragment_software_construction",
        "status": "pass" if passed == expected else "partial",
        "runs_expected": expected, "runs_passed": passed,
        "runs_failed": int((summary["status"] == "fail").sum()),
        "runs_missing": int((summary["status"] == "missing").sum()),
        "datasets_expected": len(panel["samples"]), "methods_expected": len(panel["methods"]),
        "run_summary": {"path": str(summary_path.relative_to(root)), "sha256": _sha256(summary_path)},
        "config": {"path": str(config_path.relative_to(root)), "sha256": _sha256(config_path)},
        "authorization": {"path": panel["authorization"], "sha256": panel["authorization_sha256"]},
        "memory_gate": {"path": panel["memory_gate"], "sha256": panel["memory_gate_sha256"]},
        "implementation": {"path": "spatialgabench/g4_m2_software_summary.py", "sha256": _sha256(Path(__file__))},
        "retained_attempts": sorted(str(path.relative_to(root)) for path in (root / panel["output_root"]).glob("*/*/attempt_*")),
        "resource_measurements_comparative": False,
        "resource_noncomparative_reason": "construction evidence only; final resource analysis requires a separately authorized interpretation stage",
        "holdout_accessed": False, "construction_evaluation_inputs_used": False,
        "rna_evaluation_run": False, "label_evaluation_run": False,
        "full_benchmark_run": False, "final_ranking_run": False,
        "performance_outcomes_interpreted": False, "next_milestone_authorized": False,
    }
    evidence_path = output_dir / "g4_m2_software_completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary_path, "evidence": evidence_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g4_m2_software_methods.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="results/v1/g4/m2/evidence")
    args = parser.parse_args()
    outputs = summarize(args.config, args.root, args.outdir)
    evidence = json.loads(outputs["evidence"].read_text(encoding="utf-8"))
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
