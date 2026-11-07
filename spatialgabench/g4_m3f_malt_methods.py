"""Independent construction-QC aggregation for the GSE288178 MALT method panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from spatialgabench.software_wrappers.common import sha256_file


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_method_construction(
    annotation_config_path: str | Path,
    raw_lsi_config_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root = Path(root).resolve()
    annotation_config_path = Path(annotation_config_path).resolve()
    raw_lsi_config_path = Path(raw_lsi_config_path).resolve()
    annotation = yaml.safe_load(annotation_config_path.read_text(encoding="utf-8"))
    raw = yaml.safe_load(raw_lsi_config_path.read_text(encoding="utf-8"))
    if annotation["authorization"] != raw["authorization"]:
        raise ValueError("MALT annotation and Raw LSI configurations use different authorizations")
    for config in (annotation, raw):
        for field in config["frozen_artifacts"]:
            path = root / config[field]
            if sha256_file(path) != config[f"{field}_sha256"]:
                raise ValueError(f"Frozen MALT method-construction artifact changed: {field}")
    authorization = _json(root / annotation["authorization"])
    expected_methods = [*annotation["methods"], raw["method_id"]]
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "M3F_MALT_METHOD_CONSTRUCTION"
        or authorization.get("authorized_method_panel") != expected_methods
        or authorization.get("evaluation_tasks_authorized") is not False
        or authorization.get("morphology_evaluation_authorized") is not False
    ):
        raise PermissionError("MALT method-construction authorization boundary is invalid")

    representations = pd.read_csv(root / annotation["representation_summary"], sep="\t")
    expected_representations = set(authorization["authorized_representations"])
    if set(representations["representation_id"]) != expected_representations or len(representations) != 4:
        raise ValueError("MALT method-construction representation panel changed")
    source_obs: dict[str, pd.Index] = {}
    source_hash: dict[str, str] = {}
    for source in representations.itertuples(index=False):
        source_path = root / str(source.output_path)
        if sha256_file(source_path) != str(source.output_sha256):
            raise ValueError(f"MALT source representation changed: {source.representation_id}")
        source_data = ad.read_h5ad(source_path, backed="r")
        source_obs[str(source.representation_id)] = source_data.obs_names.copy()
        source_hash[str(source.representation_id)] = str(source.output_sha256)
        source_data.file.close()

    rows: list[dict[str, Any]] = []
    bound_sidecars: dict[str, str] = {}
    annotation_config_hash = sha256_file(annotation_config_path)
    raw_config_hash = sha256_file(raw_lsi_config_path)
    dataset_id = annotation["dataset_id"]
    for representation_id in sorted(expected_representations):
        for method_id in annotation["methods"]:
            run_dir = (
                root
                / annotation["output_root"]
                / dataset_id
                / representation_id
                / method_id
            )
            manifest_path = run_dir / "output_manifest.json"
            memory_path = run_dir / "process_tree_memory.json"
            contract_path = run_dir / "contract_validation.tsv"
            validation_path = run_dir / "input_validation.json"
            manifest = _json(manifest_path)
            memory = _json(memory_path)
            validation = _json(validation_path)
            contract = pd.read_csv(contract_path, sep="\t")
            output_path = run_dir / "gene_activity.h5ad"
            output = ad.read_h5ad(output_path, backed="r")
            artifact_hashes_valid = all(
                (run_dir / name).is_file()
                and sha256_file(run_dir / name) == digest
                for name, digest in manifest.get("artifacts", {}).items()
            )
            checks = {
                "manifest_pass": manifest.get("status") == "pass",
                "manifest_identity": manifest.get("method_id") == method_id
                and manifest.get("representation_id") == representation_id,
                "manifest_config_current": manifest.get("config_sha256")
                == annotation_config_hash,
                "manifest_authorization_current": manifest.get("authorization_sha256")
                == annotation["authorization_sha256"],
                "artifacts_current": artifact_hashes_valid,
                "memory_pass": memory.get("status") == "pass" and memory.get("exit_code") == 0,
                "contract_18_of_18": len(contract) == 18
                and bool((contract["status"] == "pass").all()),
                "input_current": validation.get("input_sha256")
                == source_hash[representation_id],
                "input_boundaries": validation.get("forbidden_inputs_used") == []
                and validation.get("holdout_accessed") is False
                and validation.get("rna_accessed") is False
                and validation.get("labels_accessed") is False
                and validation.get("clusters_accessed") is False
                and validation.get("pathology_accessed") is False,
                "output_shape": output.shape == (7825, 20065),
                "output_observation_identity": output.obs_names.equals(
                    source_obs[representation_id]
                ),
                "output_identity": output.uns.get("method_id") == method_id
                and output.uns.get("representation_id") == representation_id,
                "output_leakage_absent": output.uns.get("construction_leakage_inputs")
                == "none",
            }
            output.file.close()
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "representation_id": representation_id,
                    "input_track": "B" if representation_id.startswith("fixed_bin") else "P",
                    "method_id": method_id,
                    "method_category": "mapping_control"
                    if "random_peak_to_gene" in method_id
                    else "annotation_method",
                    "output_type": "gene_activity_matrix",
                    "n_observations": 7825,
                    "n_output_features": 20065,
                    "contract_checks_passed": int((contract["status"] == "pass").sum()),
                    "contract_checks_total": len(contract),
                    "runtime_seconds_noncomparative": float(memory["runtime_seconds"]),
                    "peak_aggregate_pss_mb_noncomparative": float(
                        memory["peak_aggregate_pss_mb"]
                    ),
                    "output_path": str(output_path.relative_to(root)),
                    "output_sha256": manifest["artifacts"]["gene_activity.h5ad"],
                    "validation_checks_passed": sum(checks.values()),
                    "validation_checks_total": len(checks),
                    "failed_checks": ";".join(key for key, passed in checks.items() if not passed),
                    "status": "pass" if all(checks.values()) else "fail",
                }
            )
            bound_sidecars[str(manifest_path.relative_to(root))] = sha256_file(manifest_path)
            bound_sidecars[str(memory_path.relative_to(root))] = sha256_file(memory_path)

        run_dir = root / raw["output_root"] / dataset_id / representation_id
        manifest_path = run_dir / "output_manifest.json"
        memory_path = run_dir / "process_tree_memory.json"
        contract_path = run_dir / "contract_validation.tsv"
        validation_path = run_dir / "input_validation.json"
        manifest = _json(manifest_path)
        memory = _json(memory_path)
        validation = _json(validation_path)
        contract = pd.read_csv(contract_path, sep="\t")
        output_path = run_dir / "raw_peak_lsi.h5ad"
        output = ad.read_h5ad(output_path, backed="r")
        embedding = np.asarray(output.obsm["X_lsi"])
        artifact_hashes_valid = all(
            (run_dir / name).is_file() and sha256_file(run_dir / name) == digest
            for name, digest in manifest.get("artifacts", {}).items()
        )
        checks = {
            "manifest_pass": manifest.get("status") == "pass",
            "manifest_config_current": manifest.get("config_sha256") == raw_config_hash,
            "manifest_authorization_current": manifest.get("authorization_sha256")
            == raw["authorization_sha256"],
            "artifacts_current": artifact_hashes_valid,
            "memory_pass": memory.get("status") == "pass" and memory.get("exit_code") == 0,
            "contract_18_of_18": len(contract) == 18
            and bool((contract["status"] == "pass").all()),
            "input_current": validation.get("input_sha256") == source_hash[representation_id],
            "input_boundaries": validation.get("forbidden_inputs_used") == []
            and validation.get("holdout_accessed") is False
            and validation.get("rna_accessed") is False
            and validation.get("labels_accessed") is False
            and validation.get("clusters_accessed") is False
            and validation.get("pathology_accessed") is False
            and validation.get("image_target_accessed") is False,
            "output_observations": output.n_obs == 7825
            and output.obs_names.equals(source_obs[representation_id]),
            "embedding_shape": embedding.shape == (7825, 50),
            "embedding_finite": bool(np.isfinite(embedding).all()),
            "output_identity": output.uns.get("method_id") == raw["method_id"]
            and output.uns.get("representation_id") == representation_id,
            "output_leakage_absent": output.uns.get("construction_leakage_inputs") == "none",
        }
        output.file.close()
        rows.append(
            {
                "dataset_id": dataset_id,
                "representation_id": representation_id,
                "input_track": "B" if representation_id.startswith("fixed_bin") else "P",
                "method_id": raw["method_id"],
                "method_category": "representation_control",
                "output_type": "cell_embedding",
                "n_observations": 7825,
                "n_output_features": 50,
                "contract_checks_passed": int((contract["status"] == "pass").sum()),
                "contract_checks_total": len(contract),
                "runtime_seconds_noncomparative": float(memory["runtime_seconds"]),
                "peak_aggregate_pss_mb_noncomparative": float(memory["peak_aggregate_pss_mb"]),
                "output_path": str(output_path.relative_to(root)),
                "output_sha256": manifest["artifacts"]["raw_peak_lsi.h5ad"],
                "validation_checks_passed": sum(checks.values()),
                "validation_checks_total": len(checks),
                "failed_checks": ";".join(key for key, passed in checks.items() if not passed),
                "status": "pass" if all(checks.values()) else "fail",
            }
        )
        bound_sidecars[str(manifest_path.relative_to(root))] = sha256_file(manifest_path)
        bound_sidecars[str(memory_path.relative_to(root))] = sha256_file(memory_path)

    frame = pd.DataFrame(rows).sort_values(
        ["representation_id", "method_category", "method_id"], kind="mergesort"
    )
    output_dir = root / "results/v1/g4/m3f_gse288178_malt_method_construction/evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "method_construction_summary.tsv"
    frame.to_csv(summary_path, sep="\t", index=False)
    passed = int((frame["status"] == "pass").sum())
    evidence_path = output_dir / "gse288178_malt_method_construction_evidence.json"
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M3F_GSE288178_MALT_method_construction",
        "status": "pass" if len(frame) == passed == 32 else "fail",
        "dataset_id": dataset_id,
        "representations": 4,
        "methods": 8,
        "runs_expected": 32,
        "runs_passed": passed,
        "annotation_control_runs_passed": int(
            ((frame["output_type"] == "gene_activity_matrix") & (frame["status"] == "pass")).sum()
        ),
        "raw_lsi_runs_passed": int(
            ((frame["output_type"] == "cell_embedding") & (frame["status"] == "pass")).sum()
        ),
        "contract_checks_passed": int(frame["contract_checks_passed"].sum()),
        "contract_checks_total": int(frame["contract_checks_total"].sum()),
        "validation_checks_passed": int(frame["validation_checks_passed"].sum()),
        "validation_checks_total": int(frame["validation_checks_total"].sum()),
        "summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256_file(summary_path)},
        "authorization": {
            "path": annotation["authorization"],
            "sha256": annotation["authorization_sha256"],
        },
        "previous_gate": {
            "path": annotation["previous_gate"],
            "sha256": annotation["previous_gate_sha256"],
        },
        "representation_summary": {
            "path": annotation["representation_summary"],
            "sha256": annotation["representation_summary_sha256"],
        },
        "configs": {
            str(annotation_config_path.relative_to(root)): annotation_config_hash,
            str(raw_lsi_config_path.relative_to(root)): raw_config_hash,
        },
        "bound_run_sidecars": bound_sidecars,
        "construction_qc_only": True,
        "performance_outcomes_interpreted": False,
        "image_target_accessed": False,
        "he_pixels_accessed": False,
        "mask_values_accessed": False,
        "rna_accessed": False,
        "labels_or_clusters_accessed": False,
        "morphology_evaluation_run": False,
        "holdout_accessed": False,
        "full_real_data_benchmark_run": False,
        "recommendation_policy_fit_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "next_milestone_authorized": False,
    }
    _write_json(evidence_path, evidence)
    if evidence["status"] != "pass":
        raise RuntimeError("GSE288178 MALT method construction failed independent aggregation")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-config",
        default="configs/g4_m3f_gse288178_malt_method_construction.yaml",
    )
    parser.add_argument("--raw-lsi-config", default="configs/g4_m3f_gse288178_malt_raw_lsi.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    path = summarize_method_construction(args.annotation_config, args.raw_lsi_config, args.root)
    print(json.dumps({"output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
