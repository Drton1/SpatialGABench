"""Generate and validate G4-M2 process-tree memory preflight evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from spatialgabench.runtime.process_tree import run_with_process_tree_memory


PROFILES = {
    "synthetic": {
        "file": "synthetic_process_tree_memory.json",
        "minimum_processes": 3,
        "minimum_rss_ratio": 1.8,
        "anchor_required": False,
    },
    "signac_oci": {
        "file": "signac_r_process_tree_memory.json",
        "minimum_processes": 6,
        "minimum_rss_ratio": 2.5,
        "anchor_required": True,
    },
    "archr_oci": {
        "file": "archr_r_process_tree_memory.json",
        "minimum_processes": 6,
        "minimum_rss_ratio": 2.5,
        "anchor_required": True,
    },
    "mouse_signac_m0": {
        "file": "mouse_signac_m0_process_tree_memory.json",
        "minimum_processes": 5,
        "minimum_rss_ratio": 1.1,
        "anchor_required": True,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_synthetic_preflight(output_path: str | Path) -> dict[str, object]:
    child = "import time; payload=bytearray(32*1024*1024); time.sleep(0.8)"
    parent = (
        "import subprocess,sys,time; "
        "payload=bytearray(24*1024*1024); "
        "children=[subprocess.Popen([sys.executable,'-c',sys.argv[1]]) for _ in range(2)]; "
        "time.sleep(0.65); "
        "[child.wait() for child in children]"
    )
    return run_with_process_tree_memory(
        [sys.executable, "-c", parent, child],
        output_path,
        sample_interval_seconds=0.05,
    )


def _cgroup_capability() -> dict[str, object]:
    cgroup_type = "unknown"
    try:
        cgroup_type = os.statvfs("/sys/fs/cgroup").f_fsid
    except OSError:
        pass
    current = ""
    try:
        current = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    relative = current.split("::", 1)[1] if "::" in current else ""
    current_dir = Path("/sys/fs/cgroup") / relative.lstrip("/")
    return {
        "kernel_interface": "cgroup_v2" if Path("/sys/fs/cgroup/cgroup.controllers").is_file() else "not_cgroup_v2",
        "filesystem_id": cgroup_type,
        "current_membership": current,
        "current_cgroup_path": str(current_dir),
        "current_cgroup_writable": os.access(current_dir, os.W_OK),
        "memory_peak_readable": os.access(current_dir / "memory.peak", os.R_OK),
        "per_run_container_cgroup_verified": False,
        "decision": "procfs_process_tree_fallback",
    }


def _validate_profile(profile: str, path: Path) -> dict[str, object]:
    requirements = PROFILES[profile]
    payload = json.loads(path.read_text(encoding="utf-8"))
    peak_rss = int(payload.get("peak_aggregate_rss_bytes") or 0)
    max_rss = int(payload.get("max_single_process_rss_bytes") or 0)
    ratio = peak_rss / max_rss if max_rss else 0.0
    checks = {
        "run_pass": payload.get("status") == "pass" and payload.get("exit_code") == 0,
        "method_exact": payload.get("measurement_method") == "linux_procfs_process_tree_sampling_v1",
        "pss_available": int(payload.get("samples_with_pss") or 0) > 0
        and payload.get("peak_aggregate_pss_bytes") is not None,
        "process_scope": int(payload.get("max_concurrent_processes") or 0)
        >= int(requirements["minimum_processes"]),
        "aggregate_exceeds_single": ratio >= float(requirements["minimum_rss_ratio"]),
        "anchor_scope": not requirements["anchor_required"]
        or int(payload.get("anchor_processes_discovered") or 0) >= 1,
    }
    return {
        "profile": profile,
        "path": str(path),
        "sha256": _sha256(path),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "runtime_seconds": payload.get("runtime_seconds"),
        "sample_count": payload.get("sample_count"),
        "sample_interval_seconds": payload.get("sample_interval_seconds"),
        "anchor_processes_discovered": payload.get("anchor_processes_discovered", 0),
        "processes_seen": payload.get("processes_seen"),
        "max_concurrent_processes": payload.get("max_concurrent_processes"),
        "peak_aggregate_rss_mb": payload.get("peak_aggregate_rss_mb"),
        "peak_aggregate_pss_mb": payload.get("peak_aggregate_pss_mb"),
        "max_single_process_rss_mb": payload.get("max_single_process_rss_mb"),
        "aggregate_to_single_rss_ratio": ratio,
    }


def write_memory_preflight_summary(root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    rows = [_validate_profile(profile, output_dir / spec["file"]) for profile, spec in PROFILES.items()]
    method_run_dir = root / "results/v1/g4/preflight/memory_runs/mouse/signac"
    method_row = next(row for row in rows if row["profile"] == "mouse_signac_m0")
    input_validation = json.loads((method_run_dir / "input_validation.json").read_text(encoding="utf-8"))
    output_manifest = json.loads((method_run_dir / "output_manifest.json").read_text(encoding="utf-8"))
    with (method_run_dir / "contract_validation.tsv").open(encoding="utf-8", newline="") as handle:
        contract_rows = list(csv.DictReader(handle, delimiter="\t"))
    with (method_run_dir / "runtime.tsv").open(encoding="utf-8", newline="") as handle:
        driver_runtime = next(csv.DictReader(handle, delimiter="\t"))
    manifest_hashes_valid = all(
        _sha256(root / str(output_manifest[path_key])) == output_manifest[hash_key]
        for path_key, hash_key in (
            ("output_h5ad", "output_h5ad_sha256"),
            ("contract_tsv", "contract_tsv_sha256"),
            ("runtime_tsv", "runtime_tsv_sha256"),
        )
    )
    method_checks = {
        "input_contract_pass": input_validation.get("status") == "pass",
        "output_contract_18_of_18": len(contract_rows) == 18
        and all(row.get("status") == "pass" for row in contract_rows),
        "output_manifest_pass": output_manifest.get("status") == "pass",
        "output_manifest_hashes_valid": manifest_hashes_valid,
        "failure_artifact_absent": not (method_run_dir / "failure.tsv").exists(),
    }
    method_row["checks"].update(method_checks)
    method_row["status"] = "pass" if all(method_row["checks"].values()) else "fail"
    method_row["driver_observed_peak_memory_mb"] = float(driver_runtime["peak_memory_mb"])
    method_row["output_contract_checks_passed"] = sum(
        row.get("status") == "pass" for row in contract_rows
    )
    for row in rows:
        row["path"] = str(Path(str(row["path"])).relative_to(root))
    table_path = output_dir / "memory_preflight_summary.tsv"
    table_fields = [
        "profile",
        "status",
        "runtime_seconds",
        "sample_count",
        "sample_interval_seconds",
        "anchor_processes_discovered",
        "processes_seen",
        "max_concurrent_processes",
        "peak_aggregate_rss_mb",
        "peak_aggregate_pss_mb",
        "max_single_process_rss_mb",
        "aggregate_to_single_rss_ratio",
        "driver_observed_peak_memory_mb",
        "output_contract_checks_passed",
        "path",
        "sha256",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    implementation_paths = [
        "spatialgabench/runtime/process_tree.py",
        "spatialgabench/runtime/preflight.py",
        "containers/tools/run_with_memory_tracking.py",
        "containers/tools/run_rscript_in_oci.sh",
        "tests/test_process_tree_memory.py",
    ]
    evidence = {
        "schema_version": 1,
        "gate": "G4",
        "milestone": "M2_memory_readiness",
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "measurement_policy": {
            "primary_memory_metric": "peak_aggregate_pss_mb",
            "supplemental_memory_metric": "peak_aggregate_rss_mb",
            "runtime_scope": "end_to_end_driver_process_tree_with_dynamic_conmon_anchor",
            "sample_interval_seconds": 0.05,
            "driver_ru_maxrss_admissible_for_comparison": False,
        },
        "cgroup_capability": _cgroup_capability(),
        "profiles": rows,
        "summary_table": {
            "path": str(table_path.relative_to(root)),
            "sha256": _sha256(table_path),
            "rows": len(rows),
        },
        "implementation": {
            path: _sha256(root / path) for path in implementation_paths
        },
        "method_contract_preflight": {
            "config": {
                "path": "configs/wrappers/g4_memory_preflight_mouse_signac.yaml",
                "sha256": _sha256(root / "configs/wrappers/g4_memory_preflight_mouse_signac.yaml"),
            },
            "input_validation": {
                "path": "results/v1/g4/preflight/memory_runs/mouse/signac/input_validation.json",
                "sha256": _sha256(method_run_dir / "input_validation.json"),
            },
            "output_manifest": {
                "path": "results/v1/g4/preflight/memory_runs/mouse/signac/output_manifest.json",
                "sha256": _sha256(method_run_dir / "output_manifest.json"),
            },
            "scope": "64_observation_mm10_contract_only_no_performance_inference",
        },
        "holdout_accessed": False,
        "construction_evaluation_inputs_used": False,
        "contract_preflight_method_run": True,
        "development_benchmark_run": False,
        "rna_evaluation_run": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "performance_outcomes_interpreted": False,
        "m2_authorized_by_this_evidence": False,
    }
    evidence_path = output_dir / "g4_m2_memory_readiness_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": table_path, "evidence": evidence_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--out", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--root", default=".")
    summarize.add_argument("--outdir", required=True)
    args = parser.parse_args()
    if args.command == "synthetic":
        run_synthetic_preflight(args.out)
    else:
        write_memory_preflight_summary(args.root, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
