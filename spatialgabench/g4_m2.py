"""Independent evidence aggregation for G4-M2 annotation construction."""

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


def _id_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def summarize_g4_m2_annotation(config_path: str | Path, root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    m1 = pd.read_csv(root / config["representation_summary"], sep="\t")
    methods = list(config["methods"])
    run_rows: list[dict[str, Any]] = []
    input_obs_hashes: dict[tuple[str, str], str] = {}

    for m1_row in m1.itertuples(index=False):
        key = (str(m1_row.dataset_id), str(m1_row.representation_id))
        input_adata = ad.read_h5ad(root / str(m1_row.output_path), backed="r")
        input_obs_hashes[key] = _id_hash(input_adata.obs_names)
        input_adata.file.close()
        for method in methods:
            run_dir = root / config["output_root"] / key[0] / key[1] / method
            manifest_path = run_dir / "output_manifest.json"
            memory_path = run_dir / "process_tree_memory.json"
            contract_path = run_dir / "contract_validation.tsv"
            runtime_path = run_dir / "runtime.tsv"
            qc_path = run_dir / "construction_qc.tsv"
            validation_path = run_dir / "input_validation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            contract = pd.read_csv(contract_path, sep="\t")
            runtime = pd.read_csv(runtime_path, sep="\t").iloc[0]
            qc = pd.read_csv(qc_path, sep="\t").iloc[0]
            artifacts_valid = all(
                (run_dir / relative).is_file()
                and _sha256(run_dir / relative) == digest
                for relative, digest in manifest.get("artifacts", {}).items()
            )
            output_h5ad = run_dir / "gene_activity.h5ad"
            output = ad.read_h5ad(output_h5ad, backed="r")
            output_obs_hash = _id_hash(output.obs_names)
            output_gene_hash = _id_hash(output.var_names)
            n_obs = output.n_obs
            n_genes = output.n_vars
            output.file.close()
            contract_passed = int((contract["status"] == "pass").sum())
            boundaries_pass = (
                validation.get("forbidden_inputs_used") == []
                and validation.get("holdout_accessed") is False
                and validation.get("rna_accessed") is False
                and validation.get("labels_accessed") is False
                and manifest.get("evaluation_run") is False
                and manifest.get("full_benchmark_run") is False
                and manifest.get("final_ranking_run") is False
            )
            status = "pass" if all(
                [
                    manifest.get("status") == "pass",
                    memory.get("status") == "pass",
                    memory.get("exit_code") == 0,
                    memory.get("peak_aggregate_pss_mb") is not None,
                    validation.get("status") == "pass",
                    artifacts_valid,
                    contract.shape[0] == 18,
                    contract_passed == 18,
                    output_obs_hash == input_obs_hashes[key],
                    n_obs == int(m1_row.n_observations),
                    n_genes > 0,
                    boundaries_pass,
                    str(runtime["status"]) == "pass",
                    str(qc["status"]) == "pass",
                ]
            ) else "fail"
            run_rows.append(
                {
                    "study_id": m1_row.study_id,
                    "dataset_id": key[0],
                    "sample_id": m1_row.sample_id,
                    "species": m1_row.species,
                    "genome_build": m1_row.genome_build,
                    "representation_id": key[1],
                    "method_id": method,
                    "n_observations": n_obs,
                    "n_input_features": int(qc["n_input_features"]),
                    "n_output_genes": n_genes,
                    "output_nnz": int(qc["output_nnz"]),
                    "mapping_matrix_nnz": int(qc["mapping_matrix_nnz"]),
                    "runtime_seconds_noncomparative": float(runtime["runtime_seconds"]),
                    "peak_aggregate_pss_mb_noncomparative": float(memory["peak_aggregate_pss_mb"]),
                    "peak_aggregate_rss_mb_noncomparative": float(memory["peak_aggregate_rss_mb"]),
                    "memory_sample_count": int(memory["sample_count"]),
                    "contract_checks_passed": contract_passed,
                    "contract_checks_total": contract.shape[0],
                    "manifest_hashes_valid": artifacts_valid,
                    "observation_ids_and_order_match_input": output_obs_hash == input_obs_hashes[key],
                    "output_gene_id_hash": output_gene_hash,
                    "forbidden_inputs_used": "",
                    "holdout_accessed": False,
                    "output_manifest": str(manifest_path.relative_to(root)),
                    "output_manifest_sha256": _sha256(manifest_path),
                    "memory_evidence_sha256": _sha256(memory_path),
                    "status": status,
                }
            )

    runs = pd.DataFrame(run_rows)
    representation_rows: list[dict[str, Any]] = []
    for (dataset_id, representation_id), group in runs.groupby(["dataset_id", "representation_id"], sort=True):
        counts = group.set_index("method_id")["mapping_matrix_nnz"].to_dict()
        promoter_count = counts.get("promoter_only_2kb_pc")
        random_counts_preserved = (
            promoter_count == counts.get("random_peak_to_gene_negative_control_pc")
            == counts.get("accessibility_matched_random_peak_to_gene_control_pc")
        )
        representation_rows.append(
            {
                "dataset_id": dataset_id,
                "representation_id": representation_id,
                "methods_expected": len(methods),
                "methods_observed": group["method_id"].nunique(),
                "all_methods_pass": bool((group["status"] == "pass").all()),
                "all_observation_ids_and_order_match": bool(group["observation_ids_and_order_match_input"].all()),
                "one_gene_universe_across_methods": group["output_gene_id_hash"].nunique() == 1,
                "negative_control_link_counts_preserved": random_counts_preserved,
                "contract_checks_passed": int(group["contract_checks_passed"].sum()),
                "contract_checks_total": int(group["contract_checks_total"].sum()),
                "status": "pass" if (
                    group["method_id"].nunique() == len(methods)
                    and bool((group["status"] == "pass").all())
                    and bool(group["observation_ids_and_order_match_input"].all())
                    and group["output_gene_id_hash"].nunique() == 1
                    and random_counts_preserved
                ) else "fail",
            }
        )
    representations = pd.DataFrame(representation_rows)

    run_path = output_dir / "g4_m2_annotation_run_summary.tsv"
    representation_path = output_dir / "g4_m2_annotation_representation_summary.tsv"
    runs.to_csv(run_path, sep="\t", index=False)
    representations.to_csv(representation_path, sep="\t", index=False)
    failure_paths = sorted(
        str(path.relative_to(root)) for path in (root / config["output_root"]).rglob("failure.tsv")
    )
    unresolved_failures = [path for path in failure_paths if "attempt_1_schema_failure" not in path]
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M2A_annotation_construction",
        "status": "pass" if (
            runs.shape[0] == 140
            and representations.shape[0] == 20
            and bool((runs["status"] == "pass").all())
            and bool((representations["status"] == "pass").all())
            and not unresolved_failures
        ) else "fail",
        "runs": int(runs.shape[0]),
        "datasets": int(runs["dataset_id"].nunique()),
        "representations": int(representations.shape[0]),
        "methods": int(runs["method_id"].nunique()),
        "contract_checks_passed": int(runs["contract_checks_passed"].sum()),
        "contract_checks_total": int(runs["contract_checks_total"].sum()),
        "run_summary": {"path": str(run_path.relative_to(root)), "sha256": _sha256(run_path)},
        "representation_summary": {
            "path": str(representation_path.relative_to(root)),
            "sha256": _sha256(representation_path),
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": _sha256(config_path)},
        "authorization": {
            "path": config["authorization"],
            "sha256": config["authorization_sha256"],
        },
        "implementation": {
            relative: _sha256(root / relative)
            for relative in (
                "adapters/g4_m2_annotation_driver.py",
                "spatialgabench/g4_m2.py",
                "spatialgabench/gene_activity/interval_aggregation.py",
                "spatialgabench/gene_activity/gene_body.py",
                "spatialgabench/gene_activity/promoter_gene_body.py",
                "spatialgabench/gene_activity/distal.py",
                "spatialgabench/gene_activity/random_control.py",
                "spatialgabench/gene_activity/accessibility_matched_random.py",
                "spatialgabench/software_wrappers/common.py",
                "spatialgabench/runtime/process_tree.py",
                "containers/tools/run_with_memory_tracking.py",
                "workflows/g4_m2_annotation_methods.smk",
            )
        },
        "retained_resolved_failures": failure_paths,
        "unresolved_failures": unresolved_failures,
        "resource_measurements_comparative": False,
        "resource_noncomparative_reason": "construction ran with controlled concurrency; final resource comparisons require sequential hardware-stratified runs",
        "holdout_accessed": False,
        "construction_evaluation_inputs_used": False,
        "rna_evaluation_run": False,
        "label_evaluation_run": False,
        "software_methods_run": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "performance_outcomes_interpreted": False,
        "next_milestone_authorized": False,
    }
    evidence_path = output_dir / "g4_m2_annotation_completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"runs": run_path, "representations": representation_path, "evidence": evidence_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outputs = summarize_g4_m2_annotation(args.config, args.root, args.outdir)
    evidence = json.loads(outputs["evidence"].read_text(encoding="utf-8"))
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
