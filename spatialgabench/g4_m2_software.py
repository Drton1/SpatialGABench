"""Governed configuration and execution helpers for G4-M2B software construction."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
import pysam
import pandas as pd

from spatialgabench.software_wrappers.contracts import write_wrapper_input_validation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def prepare_reference(panel: dict[str, Any], genome_build: str, root: str | Path) -> Path:
    root = Path(root).resolve()
    reference = panel["references"][genome_build]
    source = root / reference["source_gene_annotation"]
    source_chrom_sizes = root / reference["source_chrom_sizes"]
    chrom_sizes = root / reference["chrom_sizes"]
    target = root / reference["gene_annotation"]
    audit_path = root / reference["audit"]
    if _sha256(source) != reference["source_gene_annotation_sha256"]:
        raise ValueError(f"Source annotation checksum mismatch: {source}")
    if _sha256(source_chrom_sizes) != reference["source_chrom_sizes_sha256"]:
        raise ValueError(f"Source chromosome-size checksum mismatch: {source_chrom_sizes}")
    from spatialgabench.software_wrappers.preflight import (
        _normalize_gtf,
        _read_canonical_chrom_sizes,
    )

    canonical_sizes = _read_canonical_chrom_sizes(source_chrom_sizes)
    expected_sizes = "".join(f"{chromosome}\t{size}\n" for chromosome, size in canonical_sizes.items())
    if not chrom_sizes.exists():
        chrom_sizes.parent.mkdir(parents=True, exist_ok=True)
        chrom_sizes.write_text(expected_sizes, encoding="utf-8")
    if chrom_sizes.read_text(encoding="utf-8") != expected_sizes:
        raise ValueError(f"Canonical chromosome-size content mismatch: {chrom_sizes}")
    if _sha256(chrom_sizes) != reference["chrom_sizes_sha256"]:
        raise ValueError(f"Canonical chromosome-size checksum mismatch: {chrom_sizes}")
    if target.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("output_sha256") == _sha256(target) == reference["gene_annotation_sha256"]
            and _sha256(audit_path) == reference["audit_sha256"]
            and audit.get("status") == "pass"
        ):
            return target
        raise ValueError(f"Existing normalized annotation failed checksum audit: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    normalization = _normalize_gtf(source, target, canonical_sizes)
    audit = {
        "schema_version": 1,
        "genome_build": genome_build,
        "source_path": str(source.relative_to(root)),
        "source_sha256": _sha256(source),
        "chrom_sizes_path": str(chrom_sizes.relative_to(root)),
        "source_chrom_sizes_path": str(source_chrom_sizes.relative_to(root)),
        "source_chrom_sizes_sha256": _sha256(source_chrom_sizes),
        "chrom_sizes_sha256": _sha256(chrom_sizes),
        "output_path": str(target.relative_to(root)),
        "output_sha256": _sha256(target),
        "canonical_chromosomes_only": True,
        **normalization,
        "status": "pass",
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def prepare_indexed_fragments(panel: dict[str, Any], dataset_id: str, root: str | Path) -> Path:
    root = Path(root).resolve()
    sample = panel["samples"][dataset_id]
    source = root / sample["source_fragments"]
    if _sha256(source) != sample["source_fragments_sha256"]:
        raise ValueError(f"Source fragment checksum mismatch: {source}")
    output_dir = root / panel["output_root"] / "_inputs" / dataset_id
    target = output_dir / "eligible.fragments.tsv.gz"
    index = Path(f"{target}.tbi")
    audit_path = output_dir / "eligible.fragments.audit.json"
    if target.is_file() and index.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status") == "pass"
            and audit.get("source_sha256") == _sha256(source)
            and audit.get("output_sha256") == _sha256(target)
            and audit.get("index_sha256") == _sha256(index)
        ):
            return target
        raise ValueError(f"Existing indexed fragments failed checksum audit: {target}")

    chrom_sizes_path = root / panel["references"][sample["genome_build"]]["chrom_sizes"]
    chromosome_order: dict[str, int] = {}
    with chrom_sizes_path.open(encoding="utf-8") as handle:
        for line in handle:
            chromosome = line.split("\t", 1)[0]
            chromosome_order[chromosome] = len(chromosome_order)
    output_dir.mkdir(parents=True, exist_ok=True)
    modulus = 1 << 256
    input_fingerprint = 0
    output_fingerprint = 0
    rows = 0
    output_rows = 0
    source_was_sorted = True
    previous_chromosome: str | None = None
    previous_position: tuple[int, int] | None = None
    source_seen_chromosomes: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=f"spgab-m2b-{dataset_id}-", dir=output_dir) as temporary:
        unsorted_path = Path(temporary) / "fragments.unsorted.tsv"
        sorted_path = Path(temporary) / "fragments.sorted.tsv"
        presorted_bgzf = Path(temporary) / "fragments.presorted.tsv.gz"
        with (
            gzip.open(source, "rb") as source_handle,
            pysam.BGZFile(presorted_bgzf, "wb") as presorted_handle,
        ):
            for raw_line in source_handle:
                if raw_line.startswith(b"#"):
                    continue
                fields = raw_line.rstrip(b"\n").split(b"\t")
                if len(fields) < 4:
                    raise ValueError(f"Malformed fragment row in {source}")
                chromosome = fields[0].decode()
                if chromosome not in chromosome_order:
                    raise ValueError(f"Noncanonical chromosome in frozen fragments: {chromosome}")
                position = (int(fields[1]), int(fields[2]))
                if chromosome != previous_chromosome:
                    source_was_sorted = source_was_sorted and chromosome not in source_seen_chromosomes
                    source_seen_chromosomes.add(chromosome)
                    previous_chromosome = chromosome
                    previous_position = None
                source_was_sorted = source_was_sorted and (
                    previous_position is None or position >= previous_position
                )
                previous_position = position
                input_fingerprint = (input_fingerprint + int.from_bytes(hashlib.sha256(raw_line).digest())) % modulus
                presorted_handle.write(raw_line)
                rows += 1
        if source_was_sorted:
            shutil.move(presorted_bgzf, target)
            output_fingerprint = input_fingerprint
            output_rows = rows
        else:
            with pysam.BGZFile(presorted_bgzf, "rb") as compressed, unsorted_path.open("wb") as plain:
                shutil.copyfileobj(compressed, plain)
            sort_environment = dict(os.environ)
            sort_environment["LC_ALL"] = "C"
            with sorted_path.open("wb") as sorted_handle:
                subprocess.run(
                    ["sort", "--version-sort", "-k1,1", "-k2,2n", "-k3,3n", "-k4,4", str(unsorted_path)],
                    stdout=sorted_handle,
                    env=sort_environment,
                    check=True,
                )
            previous_chromosome = None
            previous_position = None
            seen_chromosomes: set[str] = set()
            with sorted_path.open("rb") as sorted_handle, pysam.BGZFile(target, "wb") as target_handle:
                for raw_line in sorted_handle:
                    fields = raw_line.rstrip(b"\n").split(b"\t")
                    chromosome = fields[0].decode()
                    position = (int(fields[1]), int(fields[2]))
                    if chromosome != previous_chromosome:
                        if chromosome in seen_chromosomes:
                            raise ValueError(f"External fragment sort produced split chromosome blocks: {source}")
                        seen_chromosomes.add(chromosome)
                        previous_chromosome = chromosome
                        previous_position = None
                    if previous_position is not None and position < previous_position:
                        raise ValueError(f"External fragment sort failed: {source}")
                    previous_position = position
                    output_fingerprint = (
                        output_fingerprint + int.from_bytes(hashlib.sha256(raw_line).digest())
                    ) % modulus
                    target_handle.write(raw_line)
                    output_rows += 1
    if rows != output_rows or input_fingerprint != output_fingerprint:
        raise ValueError(f"Fragment rows changed during coordinate sort: {source}")
    pysam.tabix_index(str(target), preset="bed", force=True)
    audit = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "source_path": str(source.relative_to(root)),
        "source_sha256": _sha256(source),
        "output_path": str(target.relative_to(root)),
        "output_sha256": _sha256(target),
        "index_path": str(index.relative_to(root)),
        "index_sha256": _sha256(index),
        "order_independent_row_fingerprint_sha256_sum": f"{input_fingerprint:064x}",
        "fragment_rows": rows,
        "coordinate_sorted": True,
        "source_was_coordinate_sorted": source_was_sorted,
        "row_transform": (
            "validated_presorted_bgzf_recompression"
            if source_was_sorted
            else "deterministic_coordinate_sort_then_bgzf_recompression"
        ),
        "status": "pass",
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def prepare_run(config_path: str | Path, dataset_id: str, method_id: str, root: str | Path) -> tuple[dict[str, Any], Path, Path]:
    root = Path(root).resolve()
    panel = _load_yaml(Path(config_path).resolve())
    if dataset_id not in panel["samples"]:
        raise ValueError(f"Dataset is outside the frozen M2B panel: {dataset_id}")
    if method_id not in panel["methods"]:
        raise ValueError(f"Method is outside the frozen M2B panel: {method_id}")
    for field in panel.get("frozen_artifacts", ["authorization", "memory_gate"]):
        path = root / str(panel[field])
        if _sha256(path) != str(panel[f"{field}_sha256"]):
            raise ValueError(f"Frozen {field} checksum mismatch: {path}")

    sample = panel["samples"][dataset_id]
    method = panel["methods"][method_id]
    if panel.get("dataset_role") == "external_holdout":
        authorization = json.loads((root / panel["authorization"]).read_text(encoding="utf-8"))
        plan_path = root / panel["planned_execution_manifest"]
        if _sha256(plan_path) != panel["planned_execution_manifest_sha256"]:
            raise ValueError("External planned execution manifest checksum mismatch")
        if (
            authorization.get("planned_execution_manifest", {}).get("sha256")
            != panel["planned_execution_manifest_sha256"]
        ):
            raise PermissionError("External authorization does not bind the execution manifest")
        plan = pd.read_csv(plan_path, sep="\t")
        planned = plan[
            (plan["dataset_id"] == dataset_id)
            & (plan["method_id"] == method_id)
            & (plan["run_type"] == "software")
        ]
        if planned.shape[0] != 1:
            raise PermissionError("Software run is outside the exact H5B output manifest")
    build = str(sample["genome_build"])
    reference = panel["references"][build]
    prepare_reference(panel, build, root)
    fragments = prepare_indexed_fragments(panel, dataset_id, root)
    output_dir = root / str(panel["output_root"]) / dataset_id / method_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "output_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed M2B run: {output_dir}")
    wrapper = {
        "schema_version": 1,
        "run_id": f"{panel.get('run_id_prefix', 'g4_m2b')}__{dataset_id}__{method_id}",
        "project_config": panel["project_config"], "study_id": sample["study_id"],
        "dataset_id": dataset_id,
        "sample_id": sample["sample_id"],
        "dataset_role": panel.get("dataset_role", "development"),
        "method_id": method_id, "software_name": method["software_name"], "genome_build": build,
        "input_track": "F", "representation_id": panel["representation_id"],
        "comparison_scope": panel["comparison_scope"], "input_h5ad": sample["input_h5ad"],
        "fragments": str(fragments.relative_to(root)), "gene_annotation": reference["gene_annotation"],
        "chrom_sizes": reference["chrom_sizes"], "environment_lock": method["environment_lock"],
        "adapter": method["adapter"], "method_script": method["method_script"],
        "implementation_files": method["implementation_files"],
        "construction_input_classes": method["construction_input_classes"],
        "subset_policy": panel["subset_policy"], "threads": int(panel["threads"]),
        "seed": int(panel["seed"]), "output_dir": str(output_dir.relative_to(root)),
    }
    if wrapper["dataset_role"] == "external_holdout":
        wrapper["execution_authorization"] = panel["authorization"]
        wrapper["execution_authorization_sha256"] = panel["authorization_sha256"]
    wrapper_path = output_dir / "wrapper_config.yaml"
    wrapper_path.write_text(yaml.safe_dump(wrapper, sort_keys=False), encoding="utf-8")
    validation_path = output_dir / "input_validation.json"
    validation = write_wrapper_input_validation(wrapper_path, root, validation_path)
    if validation["status"] != "pass":
        raise ValueError(f"M2B wrapper input contract failed: {validation_path}")
    return panel, wrapper_path, validation_path


def execute_run(panel: dict[str, Any], wrapper_path: Path, validation_path: Path, root: str | Path) -> int:
    root = Path(root).resolve()
    wrapper = _load_yaml(wrapper_path)
    method = panel["methods"][wrapper["method_id"]]
    output_dir = root / wrapper["output_dir"]
    memory_path, conmon_pidfile = output_dir / "process_tree_memory.json", output_dir / "conmon.pid"
    oci = panel["oci"]
    podman = [str(oci["podman"]), "--root", str(oci["root"]), "--runroot", str(oci["runroot"]), "--storage-driver", str(oci["storage_driver"]), "--runtime", str((root / str(oci["runtime"])).resolve()), "--conmon", str(oci["conmon"]), "--cgroup-manager", str(oci["cgroup_manager"])]
    environment = dict(os.environ)
    environment["XDG_RUNTIME_DIR"] = str(oci["runtime_dir"])
    if wrapper["method_id"] == "snapatac2_geneactivity":
        method_command = podman + ["run", "--rm", "--network", "none", "--pid=host", "--conmon-pidfile", str(conmon_pidfile), "--volume", f"{root}:{root}", "--workdir", str(root), str(method["image"]), "python", str(wrapper["adapter"]), "--config", str(wrapper_path), "--validation", str(validation_path), "--root", str(root)]
    else:
        environment.update({"SPGAB_R_IMAGE": str(method["image"]), "SPGAB_REPO_ROOT": str(root), "SPGAB_PODMAN_BIN": str(oci["podman"]), "SPGAB_PODMAN_ROOT": str(oci["root"]), "SPGAB_PODMAN_RUNROOT": str(oci["runroot"]), "SPGAB_PODMAN_STORAGE_DRIVER": str(oci["storage_driver"]), "SPGAB_OCI_RUNTIME": str((root / str(oci["runtime"])).resolve()), "SPGAB_CONMON": str(oci["conmon"]), "SPGAB_CGROUP_MANAGER": str(oci["cgroup_manager"]), "SPGAB_CONMON_PIDFILE": str(conmon_pidfile)})
        method_command = [sys.executable, str(root / wrapper["adapter"]), "--config", str(wrapper_path), "--validation", str(validation_path), "--root", str(root), "--rscript", str(root / "containers/tools/run_rscript_in_oci.sh"), "--r-library", "/usr/local/lib/R/site-library"]
    command = [sys.executable, str(root / "containers/tools/run_with_memory_tracking.py"), "--output", str(memory_path), "--sample-interval", "0.05", "--anchor-pid-file", str(conmon_pidfile), "--cleanup-anchor-pid-files", "--", *method_command]
    return int(subprocess.run(command, cwd=root, env=environment, check=False).returncode)
