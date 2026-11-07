"""Freeze the exact H5B method-construction plan without executing a method."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


FALSE_FIELDS = (
    "method_execution_run",
    "rna_value_access_run",
    "label_or_pathology_access_run",
    "holdout_utility_access_run",
    "regret_computation_run",
    "policy_refit_run",
    "GSE279771_accessed",
    "full_benchmark_run",
    "results_writing_run",
    "universal_ranking_run",
    "final_ranking_run",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, path: str, expected: str) -> Path:
    resolved = root / path
    if not resolved.is_file() or sha256(resolved) != expected:
        raise ValueError(f"Frozen H5B artifact mismatch: {path}")
    return resolved


def _available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(page_size * os.sysconf("SC_AVPHYS_PAGES"))


def _inspect_images(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    oci = config["oci"]
    podman = [
        str(oci["podman"]),
        "--root",
        str(oci["root"]),
        "--runroot",
        str(oci["runroot"]),
        "--storage-driver",
        str(oci["storage_driver"]),
        "--runtime",
        str((root / oci["runtime"]).resolve()),
        "--conmon",
        str(oci["conmon"]),
        "--cgroup-manager",
        str(oci["cgroup_manager"]),
    ]
    environment = dict(os.environ)
    environment["XDG_RUNTIME_DIR"] = str(oci["runtime_dir"])
    rows: list[dict[str, Any]] = []
    for method_id, image in oci["images"].items():
        completed = subprocess.run(
            [*podman, "image", "inspect", "--format", "{{.Id}}", image],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        observed_id = completed.stdout.strip()
        digest = str(image).rsplit("@sha256:", 1)[-1]
        rows.append(
            {
                "method_id": method_id,
                "image": image,
                "expected_digest": digest,
                "observed_image_id": observed_id,
                "inspect_exit_code": completed.returncode,
                "status": "pass" if completed.returncode == 0 and len(observed_id) == 64 else "fail",
            }
        )
    return rows


def run_preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("H5B preflight config must be a mapping")
    invalid_flags = [field for field in FALSE_FIELDS if config.get(field) is not False]
    if invalid_flags:
        raise PermissionError(f"H5B preflight forbidden flags are not false: {invalid_flags}")

    previous_gate_path = _artifact(root, config["previous_gate"], config["previous_gate_sha256"])
    previous_gate = json.loads(previous_gate_path.read_text(encoding="utf-8"))
    if previous_gate.get("status") != "pass" or previous_gate.get("method_execution_run") is not False:
        raise ValueError("H5A previous gate does not preserve the method boundary")
    execution_path = _artifact(
        root, config["frozen_execution_manifest"], config["frozen_execution_manifest_sha256"]
    )
    representation_path = _artifact(
        root, config["representation_summary"], config["representation_summary_sha256"]
    )
    for group in config["references"].values():
        for artifact in group.values():
            _artifact(root, artifact["path"], artifact["sha256"])

    implementations: list[dict[str, str]] = []
    for relative in config["implementation_files"]:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"H5B implementation missing: {relative}")
        implementations.append({"path": relative, "sha256": sha256(path), "status": "pass"})

    representations = pd.read_csv(representation_path, sep="\t")
    if len(representations) != 8 or set(representations["status"]) != {"pass"}:
        raise ValueError("H5A representation summary is not the exact 8-row pass panel")
    representation_lookup = {
        (str(row.dataset_id), str(row.representation_id)): row
        for row in representations.itertuples(index=False)
    }
    accession_to_dataset = {
        str(sample["accession"]): dataset_id for dataset_id, sample in config["samples"].items()
    }
    execution = pd.read_csv(execution_path, sep="\t")
    if (
        len(execution) != config["expected_outputs"]
        or set(execution["attempts_authorized"].astype(int)) != {0}
        or set(execution["execution_status"]) != {"not_authorized"}
    ):
        raise ValueError("H4 execution manifest is not the disabled 50-output plan")

    annotation_methods = set(config["annotation_methods"])
    software_methods = set(config["software_methods"])
    planned: list[dict[str, Any]] = []
    for frozen in execution.itertuples(index=False):
        dataset_id = accession_to_dataset.get(str(frozen.accession))
        if dataset_id is None:
            raise ValueError(f"No H5A dataset maps accession {frozen.accession}")
        sample = config["samples"][dataset_id]
        if str(frozen.study_id) != sample["study_id"]:
            raise ValueError(f"Study/accession mismatch in frozen output: {frozen.study_id}")
        method_id = str(frozen.method_id)
        representation_id = str(frozen.representation_id)
        if method_id in software_methods:
            if str(frozen.input_track) != "F":
                raise ValueError(f"Software method is outside F track: {method_id}")
            run_type = "software"
            input_path = sample["peak_h5ad"]
            fragments = sample["fragments"]
            output_dir = f"{config['output_root']}/software/{dataset_id}/{method_id}"
            command = (
                "environment/v1-core/bin/python adapters/g4_m2_software_driver.py "
                f"--config configs/g5_h5b_software_methods.yaml --dataset {dataset_id} "
                f"--method {method_id} --root ."
            )
        elif method_id in annotation_methods:
            if str(frozen.input_track) not in {"P", "B"}:
                raise ValueError(f"Annotation method is outside P/B tracks: {method_id}")
            run_type = "annotation"
            row = representation_lookup.get((dataset_id, representation_id))
            if row is None or str(row.input_track) != str(frozen.input_track):
                raise ValueError(f"H5A representation does not match H4 row: {dataset_id}/{representation_id}")
            input_path = str(row.output_path)
            fragments = ""
            output_dir = (
                f"{config['output_root']}/annotation/{dataset_id}/{representation_id}/{method_id}"
            )
            command = (
                "environment/v1-core/bin/python containers/tools/run_with_memory_tracking.py "
                f"--output {output_dir}/process_tree_memory.json --sample-interval 0.05 -- "
                "environment/v1-core/bin/python adapters/g4_m2_annotation_driver.py "
                f"--config configs/g5_h5b_annotation_methods.yaml --dataset {dataset_id} "
                f"--representation {representation_id} --method {method_id} --root ."
            )
        else:
            raise ValueError(f"Frozen method is outside H5B implementation: {method_id}")
        planned.append(
            {
                "study_id": frozen.study_id,
                "accession": frozen.accession,
                "dataset_id": dataset_id,
                "sample_id": sample["sample_id"],
                "representation_id": representation_id,
                "input_track": frozen.input_track,
                "method_id": method_id,
                "role": frozen.role,
                "tasks_using_output": frozen.tasks_using_output,
                "run_type": run_type,
                "input_path": input_path,
                "fragments": fragments,
                "output_dir": output_dir,
                "command": command,
                "attempts_authorized": 0,
                "execution_status": "not_authorized",
            }
        )

    planned_frame = pd.DataFrame(planned)
    duplicate_outputs = planned_frame["output_dir"].duplicated().any()
    annotation_count = int((planned_frame["run_type"] == "annotation").sum())
    software_count = int((planned_frame["run_type"] == "software").sum())
    if (
        duplicate_outputs
        or annotation_count != config["expected_annotation_outputs"]
        or software_count != config["expected_software_outputs"]
    ):
        raise ValueError("H5B planned output dimensions are invalid")

    for sample in config["samples"].values():
        _artifact(root, sample["fragments"], sample["fragments_sha256"])
    for row in representations.itertuples(index=False):
        _artifact(root, str(row.output_path), str(row.output_sha256))
    images = _inspect_images(config, root)
    disk = os.statvfs(root)
    available_disk = int(disk.f_bavail * disk.f_frsize)
    available_memory = _available_memory_bytes()
    resources_ready = (
        available_disk >= config["minimum_available_disk_bytes"]
        and available_memory >= config["minimum_available_memory_bytes"]
    )

    evidence_dir = root / config["evidence_dir"]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    plan_path = evidence_dir / "planned_method_execution_manifest.tsv"
    implementation_path = evidence_dir / "implementation_manifest.tsv"
    image_path = evidence_dir / "runtime_image_manifest.tsv"
    planned_frame.to_csv(plan_path, sep="\t", index=False)
    pd.DataFrame(implementations).to_csv(implementation_path, sep="\t", index=False)
    pd.DataFrame(images).to_csv(image_path, sep="\t", index=False)
    passed = resources_ready and all(row["status"] == "pass" for row in images)
    evidence = {
        "schema_version": 1,
        "gate": "G5",
        "milestone": "H5B_method_construction_preflight",
        "status": "pass" if passed else "blocked",
        "planned_outputs": len(planned),
        "planned_annotation_outputs": annotation_count,
        "planned_software_outputs": software_count,
        "planned_outputs_authorized": 0,
        "studies": planned_frame["study_id"].nunique(),
        "available_disk_bytes": available_disk,
        "available_memory_bytes": available_memory,
        "resources_ready": resources_ready,
        "runtime_images_ready": all(row["status"] == "pass" for row in images),
        "method_execution_run": False,
        "rna_value_access_run": False,
        "label_or_pathology_access_run": False,
        "holdout_utility_access_run": False,
        "regret_computation_run": False,
        "policy_refit_run": False,
        "GSE279771_accessed": False,
        "full_benchmark_run": False,
        "results_writing_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
        "planned_execution_manifest": {
            "path": str(plan_path.relative_to(root)),
            "sha256": sha256(plan_path),
        },
        "implementation_manifest": {
            "path": str(implementation_path.relative_to(root)),
            "sha256": sha256(implementation_path),
        },
        "runtime_image_manifest": {
            "path": str(image_path.relative_to(root)),
            "sha256": sha256(image_path),
        },
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "previous_gate": {"path": config["previous_gate"], "sha256": config["previous_gate_sha256"]},
        "frozen_execution_manifest": {
            "path": config["frozen_execution_manifest"],
            "sha256": config["frozen_execution_manifest_sha256"],
        },
        "representation_summary": {
            "path": config["representation_summary"],
            "sha256": config["representation_summary_sha256"],
        },
    }
    evidence_path = evidence_dir / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("H5B method-construction preflight is blocked")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g5_h5b_method_construction_preflight.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    output = run_preflight(args.config, args.root)
    print(json.dumps({"output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
