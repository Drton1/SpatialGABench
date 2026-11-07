"""FB3 deterministic depth thinning and controlled resource stress runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pysam
import yaml
from scipy import sparse
from scipy.stats import spearmanr


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    loader = json.loads if path.suffix == ".json" else yaml.safe_load
    value = loader(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def condition_id(dataset_id: str, fraction: float, seed: int) -> str:
    depth = f"d{int(round(fraction * 100)):03d}"
    return f"{dataset_id}__{depth}__s{seed}"


def planned_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id, study in config["studies"].items():
        for fraction in config["depth_fractions"]:
            seeds = [config["full_depth_seed"]] if float(fraction) == 1.0 else config["thinning_seeds"]
            for seed in seeds:
                rows.append({
                    "condition_id": condition_id(dataset_id, float(fraction), int(seed)),
                    "dataset_id": dataset_id,
                    "study_id": study["study_id"],
                    "species": study["species"],
                    "genome_build": study["genome_build"],
                    "depth_fraction": float(fraction),
                    "seed": int(seed),
                })
    return rows


def retain_fragment(raw_line: bytes, row_index: int, fraction: float, seed: int) -> bool:
    if fraction >= 1.0:
        return True
    prefix = f"{seed}\0{row_index}\0".encode("ascii")
    value = int.from_bytes(hashlib.sha256(prefix + raw_line).digest(), "big")
    return value < int(fraction * (1 << 256))


def serialized_fragment_row(raw_line: bytes) -> bytes:
    """Restore the line terminator stripped by pysam.BGZFile iteration."""

    return raw_line.rstrip(b"\r\n") + b"\n"


def _validate_authorization(config_path: Path, authorization_path: Path, root: Path) -> dict[str, Any]:
    config = _mapping(config_path)
    authorization = _mapping(authorization_path)
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != "FB3_controlled_depth_resource_stress"
        or authorization.get("method_execution_authorized") is not True
        or authorization.get("rna_value_access_authorized") is not False
        or authorization.get("label_or_pathology_value_access_authorized") is not False
        or authorization.get("policy_refit_authorized") is not False
        or authorization.get("universal_ranking_authorized") is not False
        or authorization.get("final_ranking_authorized") is not False
    ):
        raise PermissionError("FB3 authorization boundary is invalid")
    bindings = {
        "config": config_path,
        "implementation": Path(__file__).resolve(),
        "protocol": root / config["protocol"],
        "completion_protocol": root / config["completion_protocol"],
        "project_config": root / config["project_config"],
        "software_panel": root / config["software_panel"],
    }
    for key, path in bindings.items():
        record = authorization.get(key, {})
        if (
            not path.is_file()
            or record.get("path") != str(path.relative_to(root))
            or record.get("sha256") != sha256(path)
        ):
            raise ValueError(f"FB3 authorization binding mismatch: {key}")
    if len(planned_conditions(config)) != int(config["planned_conditions"]):
        raise ValueError("FB3 planned condition count changed")
    return config


def _condition_dir(root: Path, config: dict[str, Any], condition: dict[str, Any]) -> Path:
    return root / config["output_root"] / "conditions" / condition["condition_id"]


def _find_condition(config: dict[str, Any], value: str) -> dict[str, Any]:
    matches = [row for row in planned_conditions(config) if row["condition_id"] == value]
    if len(matches) != 1:
        raise ValueError(f"Unknown FB3 condition: {value}")
    return matches[0]


def prepare_condition(config: dict[str, Any], condition: dict[str, Any], root: Path) -> Path:
    study = config["studies"][condition["dataset_id"]]
    source = root / study["fragments"]
    source_index = root / study["fragments_index"]
    if sha256(source) != study["fragments_sha256"] or sha256(source_index) != study["fragments_index_sha256"]:
        raise ValueError("FB3 frozen fragment parent checksum mismatch")
    output_dir = _condition_dir(root, config, condition) / "fragments"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "eligible.fragments.tsv.gz"
    index = Path(f"{target}.tbi")
    audit_path = output_dir / "thinning_audit.json"
    if audit_path.is_file():
        audit = _mapping(audit_path)
        if (
            audit.get("status") == "pass"
            and target.is_file() and index.is_file()
            and audit.get("output_fragments_sha256") == sha256(target)
            and audit.get("output_index_sha256") == sha256(index)
        ):
            return audit_path
        raise ValueError(f"Existing FB3 thinning output failed audit: {output_dir}")

    fraction, seed = float(condition["depth_fraction"]), int(condition["seed"])
    retained = 0
    source_rows = 0
    if fraction == 1.0:
        os.link(source, target)
        os.link(source_index, index)
        source_rows = retained = int(study["fragment_rows"])
        transform = "hardlink_to_checksum_bound_sorted_bgzf_parent"
    else:
        temporary = target.with_name(f".{target.name}.tmp")
        with pysam.BGZFile(source, "rb") as handle, pysam.BGZFile(temporary, "wb") as output:
            for raw_line in handle:
                if raw_line.startswith(b"#"):
                    continue
                source_rows += 1
                if retain_fragment(raw_line, source_rows, fraction, seed):
                    output.write(serialized_fragment_row(raw_line))
                    retained += 1
        temporary.replace(target)
        with pysam.BGZFile(target, "rb") as verification_handle:
            verified_rows = sum(1 for _ in verification_handle)
        if verified_rows != retained:
            raise ValueError(
                f"FB3 BGZF row verification failed: retained={retained}, observed={verified_rows}"
            )
        pysam.tabix_index(str(target), preset="bed", force=True)
        transform = "sha256_row_index_seed_bernoulli_thinning_preserving_coordinate_order"
    if source_rows != int(study["fragment_rows"]) or retained <= 0:
        raise ValueError("FB3 thinning row accounting failed")
    expected = source_rows * fraction
    sd = math.sqrt(source_rows * fraction * (1 - fraction)) if fraction < 1 else 0.0
    z = (retained - expected) / sd if sd else 0.0
    audit = {
        "schema_version": 1,
        "status": "pass" if abs(z) <= 6 else "fail",
        "condition_id": condition["condition_id"],
        "study_id": condition["study_id"],
        "dataset_id": condition["dataset_id"],
        "depth_fraction": fraction,
        "seed": seed,
        "source_fragments": str(source.relative_to(root)),
        "source_fragments_sha256": sha256(source),
        "source_fragment_rows": source_rows,
        "output_fragments": str(target.relative_to(root)),
        "output_fragments_sha256": sha256(target),
        "output_index": str(index.relative_to(root)),
        "output_index_sha256": sha256(index),
        "included_fragment_rows": retained,
        "output_fragment_rows_verified": retained,
        "observed_retained_fraction": retained / source_rows,
        "expected_retained_rows": expected,
        "binomial_z_residual": z,
        "read_support_policy": "fifth_column_preserved_as_one_per_retained_source_row",
        "filter_contract": "deterministic_depth_thinning_of_frozen_eligible_fragments",
        "row_transform": transform,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise ValueError(f"FB3 thinning residual outside frozen tolerance: {z}")
    return audit_path


def call_peaks(config: dict[str, Any], condition: dict[str, Any], root: Path) -> Path:
    base = _condition_dir(root, config, condition)
    audit = _mapping(base / "fragments/thinning_audit.json")
    fragments = root / audit["output_fragments"]
    output = base / "peaks"
    output.mkdir(parents=True, exist_ok=True)
    narrow = output / f"{condition['condition_id']}_peaks.narrowPeak"
    frozen = output / "macs3_summit_501bp.bed"
    peak_audit = output / "peak_freeze_audit.json"
    stage_audit = output / "peak_stage_audit.json"
    if _valid_json(stage_audit):
        return stage_audit
    if peak_audit.is_file() and narrow.is_file():
        result = _mapping(peak_audit)
    else:
        macs3 = str(Path(sys.executable).with_name("macs3"))
        study = config["studies"][condition["dataset_id"]]
        command = [
            macs3, "callpeak", "-f", config["macs3"]["format"], "--max-count",
            str(config["macs3"]["max_count"]), "-t", str(fragments), "-g",
            study["macs3_genome_size"], "-n", condition["condition_id"], "-q",
            str(config["macs3"]["qvalue"]), "--call-summits", "--outdir", str(output),
        ]
        log = output / "macs3.log"
        with log.open("w") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0 or not narrow.is_file():
            raise RuntimeError(f"MACS3 failed for {condition['condition_id']}")
        from spatialgabench.representations.peaks import freeze_macs3_summits

        reference = config["references"][condition["genome_build"]]
        result = freeze_macs3_summits(
            narrow, root / reference["chrom_sizes"]["path"], root / reference["blacklist"]["path"],
            frozen, peak_audit, peak_width=int(config["macs3"]["peak_width_bp"]),
        )
    stage = {
        "schema_version": 1,
        "status": "pass",
        "condition_id": condition["condition_id"],
        "macs3_completed": True,
        "input_peak_rows": int(result.get("input_peak_rows", 0)),
        "frozen_peak_rows": int(result.get("frozen_peak_rows", 0)),
        "peak_representation_available": result.get("status") == "pass" and int(result.get("frozen_peak_rows", 0)) > 0,
        "peak_representation_failure_reason": "" if result.get("status") == "pass" else "no_nonblacklisted_frozen_peaks",
        "fixed_bin_and_fragment_branches_may_continue": True,
        "peak_freeze_audit": str(peak_audit.relative_to(root)),
        "peak_freeze_audit_sha256": sha256(peak_audit),
    }
    stage_audit.write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n")
    return stage_audit


def build_representations(config: dict[str, Any], condition: dict[str, Any], root: Path) -> Path:
    base = _condition_dir(root, config, condition)
    audit = _mapping(base / "fragments/thinning_audit.json")
    outdir = base / "representations"
    validation = outdir / "matched_representation_validation.json"
    if validation.is_file() and _mapping(validation).get("status") == "pass":
        return validation
    from spatialgabench.representations.matched import build_matched_representations

    study = config["studies"][condition["dataset_id"]]
    reference = config["references"][condition["genome_build"]]
    peak_stage = _mapping(base / "peaks/peak_stage_audit.json")
    peaks_path = base / "peaks/macs3_summit_501bp.bed"
    if not peak_stage["peak_representation_available"]:
        peaks_path = base / "peaks/sentinel_for_fixed_bin_construction_only.bed"
        peaks_path.write_text("chr1\t0\t501\tsentinel_no_peak\t0\t0\t250\n", encoding="utf-8")
    outputs = build_matched_representations(
        project_config=root / config["project_config"], study_id=condition["study_id"],
        dataset_id=condition["condition_id"], sample_id=study["sample_id"],
        fragments_path=root / audit["output_fragments"], positions_path=root / study["positions"],
        chrom_sizes_path=root / reference["chrom_sizes"]["path"],
        peaks_path=peaks_path, genome_build=condition["genome_build"],
        bin_sizes=list(config["bin_sizes_bp"]), output_dir=outdir,
        fragment_filter_audit_path=base / "fragments/thinning_audit.json",
    )
    availability = pd.DataFrame([
        {"representation_id": "fixed_bin_25kb", "available": True, "reason": "constructed"},
        {"representation_id": "macs3_summit_501bp", "available": bool(peak_stage["peak_representation_available"]),
         "reason": "constructed" if peak_stage["peak_representation_available"] else peak_stage["peak_representation_failure_reason"]},
        {"representation_id": "filtered_fragments_depth_stress_contract", "available": True, "reason": "constructed"},
    ])
    availability.to_csv(outdir / "representation_availability.tsv", sep="\t", index=False)
    return outputs["validation"]


def _representation_summary(config: dict[str, Any], condition: dict[str, Any], root: Path) -> Path:
    base = _condition_dir(root, config, condition)
    manifest = pd.read_csv(base / "representations/representation_manifest.tsv", sep="\t")
    manifest = manifest[manifest.representation_id.isin(config["matrix_representations"])].copy()
    availability = pd.read_csv(base / "representations/representation_availability.tsv", sep="\t")
    available = set(availability.loc[availability.available.astype(bool), "representation_id"])
    manifest = manifest[manifest.representation_id.isin(available)].copy()
    study = config["studies"][condition["dataset_id"]]
    manifest["study_id"] = condition["study_id"]
    manifest["dataset_id"] = condition["condition_id"]
    manifest["sample_id"] = study["sample_id"]
    manifest["species"] = condition["species"]
    manifest["genome_build"] = condition["genome_build"]
    path = base / "representation_summary.tsv"
    manifest.to_csv(path, sep="\t", index=False)
    return path


def _annotation_config(
    config: dict[str, Any], condition: dict[str, Any], authorization_path: Path, root: Path
) -> Path:
    base = _condition_dir(root, config, condition)
    references = {}
    for build, values in config["references"].items():
        references[build] = {
            name: values[name] for name in ("genes", "promoters", "gene_body", "tss")
        }
    payload = {
        "schema_version": 1, "gate": "FULL_BENCHMARK", "milestone": "FB3",
        "authorization": str(authorization_path.relative_to(root)),
        "authorization_sha256": sha256(authorization_path),
        "authorization_milestone": "FB3_controlled_depth_resource_stress",
        "stress_protocol": config["protocol"],
        "stress_protocol_sha256": sha256(root / config["protocol"]),
        "frozen_artifacts": ["authorization", "stress_protocol"],
        "representation_summary": str(_representation_summary(config, condition, root).relative_to(root)),
        "output_root": str((base / "methods/annotation").relative_to(root)),
        "methods": config["annotation_methods"], "parameters": config["parameters"],
        "references": references, "forbidden_holdout_accessions": [],
        "dataset_role": "development", "comparison_scope": "FB3_depth_resource_stress_construction_only",
        "subset_policy": "full_depth_stress_observation_contract",
        "run_id_prefix": "fb3", "driver_stage": "FB3_annotation_stress",
    }
    path = base / "execution/annotation_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _software_files(panel: dict[str, Any], method_id: str) -> tuple[str, str | None, list[str]]:
    method = panel["methods"][method_id]
    return method["adapter"], method["method_script"], method["construction_input_classes"]


def prepare_software_run(
    config: dict[str, Any], condition: dict[str, Any], method_id: str,
    authorization_path: Path, root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    panel = _mapping(root / config["software_panel"])
    method = panel["methods"][method_id]
    base = _condition_dir(root, config, condition)
    output = base / "methods/software" / method_id
    output.mkdir(parents=True, exist_ok=True)
    availability = pd.read_csv(base / "representations/representation_availability.tsv", sep="\t")
    peak_available = bool(availability.loc[availability.representation_id.eq("macs3_summit_501bp"), "available"].iloc[0])
    shell_representation = "macs3_summit_501bp" if peak_available else "fixed_bin_25kb"
    representation = base / "representations" / f"{condition['condition_id']}__{shell_representation}.h5ad"
    fragments = base / "fragments/eligible.fragments.tsv.gz"
    reference = config["references"][condition["genome_build"]]
    adapter, script, classes = _software_files(panel, method_id)
    wrapper = {
        "schema_version": 1, "run_id": f"fb3__{condition['condition_id']}__{method_id}",
        "project_config": config["project_config"], "study_id": condition["study_id"],
        "dataset_id": condition["condition_id"], "sample_id": config["studies"][condition["dataset_id"]]["sample_id"],
        "dataset_role": "development", "method_id": method_id, "software_name": method["software_name"],
        "genome_build": condition["genome_build"], "input_track": "F",
        "representation_id": "filtered_fragments_depth_stress_contract",
        "comparison_scope": "FB3_depth_resource_stress_construction_only",
        "input_h5ad": str(representation.relative_to(root)), "fragments": str(fragments.relative_to(root)),
        "gene_annotation": reference["software_annotation"]["path"],
        "chrom_sizes": reference["software_chrom_sizes"]["path"],
        "environment_lock": method["environment_lock"], "adapter": adapter,
        "method_script": script, "implementation_files": method["implementation_files"],
        "construction_input_classes": classes, "subset_policy": "full", "threads": int(config["threads"]),
        "seed": int(condition["seed"]), "output_dir": str(output.relative_to(root)),
        "observation_shell_representation": shell_representation,
        "execution_authorization": str(authorization_path.relative_to(root)),
        "execution_authorization_sha256": sha256(authorization_path),
    }
    wrapper_path = output / "wrapper_config.yaml"
    wrapper_path.write_text(yaml.safe_dump(wrapper, sort_keys=False), encoding="utf-8")
    fields = ["input_h5ad", "fragments", "gene_annotation", "chrom_sizes", "environment_lock", "adapter"]
    if script:
        fields.append("method_script")
    resolved = {field: str((root / wrapper[field]).resolve()) for field in fields}
    validation = {
        "schema_version": 1, "status": "pass", "run_id": wrapper["run_id"],
        "method_id": method_id, "dataset_id": wrapper["dataset_id"], "study_id": wrapper["study_id"],
        "dataset_role": "development", "config_path": str(wrapper_path),
        "config_sha256": sha256(wrapper_path), "resolved_paths": resolved,
        "input_sha256": {field: sha256(path) for field, path in resolved.items()},
        "authorization": {"milestone": "FB3_controlled_depth_resource_stress", "status": "authorized"},
        "checks": [{"check_id": "fb3_checksum_bound_construction_only", "status": "pass", "detail": condition["condition_id"]}],
    }
    validation_path = output / "input_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return panel, wrapper_path, validation_path


def run_software(
    config: dict[str, Any], condition: dict[str, Any], method_id: str,
    authorization_path: Path, root: Path,
) -> int:
    panel, wrapper, validation = prepare_software_run(config, condition, method_id, authorization_path, root)
    from spatialgabench.g4_m2_software import execute_run

    return execute_run(panel, wrapper, validation, root)


def _run_tracked(command: list[str], memory_path: Path, root: Path, interval: float) -> int:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = root / "containers/tools/run_with_memory_tracking.py"
    full = [sys.executable, str(tracker), "--output", str(memory_path), "--sample-interval", str(interval), "--", *command]
    return int(subprocess.run(full, cwd=root, check=False).returncode)


def _valid_json(path: Path, field: str = "status", expected: Any = "pass") -> bool:
    try:
        return path.is_file() and _mapping(path).get(field) == expected
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def worker_command(stage: str) -> str:
    commands = {
        "prepare_fragments": "prepare-condition",
        "call_peaks": "call-peaks",
        "build_representations": "build-representations",
    }
    if stage not in commands:
        raise ValueError(f"Unknown FB3 worker stage: {stage}")
    return commands[stage]


def orchestrate(config_path: Path, authorization_path: Path, root: Path, limit: int | None = None) -> Path:
    config = _validate_authorization(config_path, authorization_path, root)
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    conditions = planned_conditions(config)
    pd.DataFrame(conditions).to_csv(output / "planned_conditions.tsv", sep="\t", index=False)
    if limit is not None:
        conditions = conditions[:limit]
    ledger: list[dict[str, Any]] = []
    for condition in conditions:
        base = _condition_dir(root, config, condition)
        workers = [
            ("prepare_fragments", base / "fragments/thinning_audit.json"),
            ("call_peaks", base / "peaks/peak_stage_audit.json"),
            ("build_representations", base / "representations/matched_representation_validation.json"),
        ]
        for stage, target in workers:
            if _valid_json(target):
                code = 0
            else:
                memory = base / "resources" / f"{stage}.json"
                code = _run_tracked(
                    [sys.executable, "-m", "spatialgabench.full_benchmark_depth_stress", worker_command(stage),
                     "--config", str(config_path), "--authorization", str(authorization_path),
                     "--condition", condition["condition_id"], "--root", str(root)],
                    memory, root, float(config["sample_interval_seconds"]),
                )
            ledger.append({**condition, "stage": stage, "method_id": "", "representation_id": "", "exit_code": code})
            if code != 0:
                break
        if any(row["exit_code"] != 0 and row["condition_id"] == condition["condition_id"] for row in ledger):
            continue
        annotation_config = _annotation_config(config, condition, authorization_path, root)
        availability = pd.read_csv(base / "representations/representation_availability.tsv", sep="\t")
        available_representations = set(
            availability.loc[availability.available.astype(bool), "representation_id"]
        )
        for representation in config["matrix_representations"]:
            for method in config["annotation_methods"]:
                run_dir = base / "methods/annotation" / condition["condition_id"] / representation / method
                manifest = run_dir / "output_manifest.json"
                if representation not in available_representations:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    failure_path = run_dir / "failure.tsv"
                    if not failure_path.is_file():
                        pd.DataFrame([{
                            "dataset_id": condition["condition_id"], "representation_id": representation,
                            "method_id": method, "stage": "representation_unavailable",
                            "failure_type": "NoFrozenPeaks", "message": "no_nonblacklisted_frozen_peaks",
                            "status": "fail",
                        }]).to_csv(failure_path, sep="\t", index=False)
                    code = 2
                elif _valid_json(manifest):
                    code = 0
                elif (run_dir / "failure.tsv").is_file():
                    code = 1
                else:
                    code = _run_tracked(
                        [sys.executable, "adapters/g4_m2_annotation_driver.py", "--config", str(annotation_config),
                         "--dataset", condition["condition_id"], "--representation", representation,
                         "--method", method, "--root", str(root)],
                        run_dir / "process_tree_memory.json", root, float(config["sample_interval_seconds"]),
                    )
                ledger.append({**condition, "stage": "annotation", "method_id": method, "representation_id": representation, "exit_code": code})
        for method in config["software_methods"]:
            run_dir = base / "methods/software" / method
            if _valid_json(run_dir / "output_manifest.json"):
                code = 0
            elif (run_dir / "failure.tsv").is_file():
                code = 1
            else:
                code = run_software(config, condition, method, authorization_path, root)
            ledger.append({**condition, "stage": "software", "method_id": method, "representation_id": "filtered_fragments_depth_stress_contract", "exit_code": code})
        pd.DataFrame(ledger).to_csv(output / "execution_ledger.tsv", sep="\t", index=False)
    pd.DataFrame(ledger).to_csv(output / "execution_ledger.tsv", sep="\t", index=False)
    return output / "execution_ledger.tsv"


def _output_path(config: dict[str, Any], condition: dict[str, Any], method: str, representation: str, root: Path) -> Path:
    base = _condition_dir(root, config, condition)
    if representation == "filtered_fragments_depth_stress_contract":
        return base / "methods/software" / method / "gene_activity.h5ad"
    return base / "methods/annotation" / condition["condition_id"] / representation / method / "gene_activity.h5ad"


def _memory_path(config: dict[str, Any], condition: dict[str, Any], method: str, representation: str, root: Path) -> Path:
    return _output_path(config, condition, method, representation, root).parent / "process_tree_memory.json"


def _matrix_metrics(current_path: Path, full_path: Path) -> dict[str, float]:
    current, full = ad.read_h5ad(current_path), ad.read_h5ad(full_path)
    common_genes = full.var_names.intersection(current.var_names)
    if not current.obs_names.equals(full.obs_names) or len(common_genes) == 0:
        raise ValueError("FB3 output identity differs from full-depth anchor")
    x = current[:, common_genes].X.tocsr() if sparse.issparse(current.X) else sparse.csr_matrix(current[:, common_genes].X)
    y = full[:, common_genes].X.tocsr() if sparse.issparse(full.X) else sparse.csr_matrix(full[:, common_genes].X)
    pseudo_x, pseudo_y = np.asarray(x.sum(axis=0)).ravel(), np.asarray(y.sum(axis=0)).ravel()
    correlation = spearmanr(pseudo_x, pseudo_y).statistic
    dots = np.asarray(x.multiply(y).sum(axis=1)).ravel()
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel() * np.asarray(y.multiply(y).sum(axis=1)).ravel())
    cosine = np.divide(dots, norms, out=np.zeros_like(dots, dtype=float), where=norms > 0)
    return {
        "common_genes": len(common_genes),
        "pseudobulk_gene_spearman_vs_full": float(correlation),
        "median_spot_cosine_vs_full": float(np.median(cosine)),
        "zero_output_spots": int((np.asarray(x.sum(axis=1)).ravel() == 0).sum()),
        "output_nnz": int(x.nnz), "output_total_counts": float(x.sum()),
        "full_output_nnz": int(y.nnz), "full_output_total_counts": float(y.sum()),
        "output_nnz_fraction_vs_full": x.nnz / y.nnz if y.nnz else math.nan,
        "output_count_fraction_vs_full": float(x.sum() / y.sum()) if y.sum() else math.nan,
    }


def summarize(config_path: Path, authorization_path: Path, root: Path) -> Path:
    config = _validate_authorization(config_path, authorization_path, root)
    conditions = planned_conditions(config)
    full_by_dataset = {row["dataset_id"]: row for row in conditions if row["depth_fraction"] == 1.0}
    rows = []
    panel = [
        (method, representation)
        for representation in config["matrix_representations"] for method in config["annotation_methods"]
    ] + [(method, "filtered_fragments_depth_stress_contract") for method in config["software_methods"]]
    for condition in conditions:
        for method, representation in panel:
            path = _output_path(config, condition, method, representation, root)
            full_path = _output_path(config, full_by_dataset[condition["dataset_id"]], method, representation, root)
            memory_path = _memory_path(config, condition, method, representation, root)
            success = path.is_file() and memory_path.is_file() and _mapping(memory_path).get("exit_code") == 0
            row = {**condition, "method_id": method, "representation_id": representation, "success": success,
                   "output_path": str(path.relative_to(root)) if path.is_file() else "",
                   "memory_path": str(memory_path.relative_to(root)) if memory_path.is_file() else ""}
            if success and full_path.is_file():
                row.update(_matrix_metrics(path, full_path))
                memory = _mapping(memory_path)
                full_memory = _mapping(_memory_path(config, full_by_dataset[condition["dataset_id"]], method, representation, root))
                row.update({
                    "runtime_seconds": memory.get("runtime_seconds"),
                    "peak_aggregate_pss_mb": memory.get("peak_aggregate_pss_mb"),
                    "runtime_ratio_vs_full": memory.get("runtime_seconds") / full_memory.get("runtime_seconds"),
                    "pss_ratio_vs_full": memory.get("peak_aggregate_pss_mb") / full_memory.get("peak_aggregate_pss_mb"),
                })
            rows.append(row)
    output = root / config["output_root"]
    metrics = pd.DataFrame(rows)
    metrics_path = output / "method_depth_robustness_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    summary = (
        metrics.groupby(["study_id", "species", "genome_build", "method_id", "representation_id", "depth_fraction"], dropna=False)
        .agg(planned_runs=("success", "size"), successful_runs=("success", "sum"),
             median_pseudobulk_spearman=("pseudobulk_gene_spearman_vs_full", "median"),
             median_spot_cosine=("median_spot_cosine_vs_full", "median"),
             median_runtime_ratio=("runtime_ratio_vs_full", "median"), median_pss_ratio=("pss_ratio_vs_full", "median"))
        .reset_index()
    )
    summary["success_rate"] = summary.successful_runs / summary.planned_runs
    summary_path = output / "depth_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "gate": "FULL_BENCHMARK", "milestone": "FB3_controlled_depth_resource_stress",
        "status": "pass" if len(metrics) == int(config["planned_contracts"]) else "incomplete",
        "planned_conditions": int(config["planned_conditions"]), "planned_contracts": int(config["planned_contracts"]),
        "observed_contracts": len(metrics), "successful_contracts": int(metrics.success.sum()),
        "failed_contracts": int((~metrics.success).sum()), "rna_accessed": False, "labels_accessed": False,
        "pathology_accessed": False, "policy_refit_run": False, "universal_ranking_run": False,
        "final_ranking_run": False,
        "metrics": {"path": str(metrics_path.relative_to(root)), "sha256": sha256(metrics_path)},
        "summary": {"path": str(summary_path.relative_to(root)), "sha256": sha256(summary_path)},
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256(authorization_path)},
        "implementation": {"path": "spatialgabench/full_benchmark_depth_stress.py", "sha256": sha256(Path(__file__))},
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare-condition", "call-peaks", "build-representations", "run", "summarize"])
    parser.add_argument("--config", default="configs/full_benchmark_fb3_depth_resource_stress.yaml")
    parser.add_argument("--authorization", default="governance/authorizations/FULL_BENCHMARK_FB3_DEPTH_RESOURCE_STRESS_AUTHORIZATION_2026-07-18.json")
    parser.add_argument("--condition")
    parser.add_argument("--root", default=".")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root, config_path, authorization_path = Path(args.root).resolve(), Path(args.config).resolve(), Path(args.authorization).resolve()
    config = _validate_authorization(config_path, authorization_path, root)
    if args.command == "run":
        print(orchestrate(config_path, authorization_path, root, args.limit)); return 0
    if args.command == "summarize":
        print(summarize(config_path, authorization_path, root)); return 0
    if not args.condition:
        parser.error("--condition is required for worker commands")
    condition = _find_condition(config, args.condition)
    if args.command == "prepare-condition":
        print(prepare_condition(config, condition, root))
    elif args.command == "call-peaks":
        print(call_peaks(config, condition, root))
    else:
        print(build_representations(config, condition, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
