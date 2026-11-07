"""Sample aggregate memory for a command and its Linux process tree."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    """A PID paired with its kernel start time to prevent PID-reuse errors."""

    pid: int
    start_time_ticks: int


@dataclass(frozen=True)
class ProcessStat:
    identity: ProcessIdentity
    parent_pid: int
    process_group: int


def _status_namespace_pids(pid: int) -> tuple[int, ...]:
    try:
        with Path(f"/proc/{pid}/status").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("NSpid:"):
                    return tuple(int(value) for value in line.split()[1:])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return ()
    return ()


def _pid_namespace_inode(pid: int | str) -> int | None:
    try:
        return Path(f"/proc/{pid}/ns/pid").stat().st_ino
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def _resolve_current_namespace_pid(pid: int) -> int:
    """Map a local PID to the PID visible in the mounted procfs."""
    current_namespace = _pid_namespace_inode("self")
    try:
        entries = os.scandir("/proc")
    except OSError:
        return pid
    candidates: list[int] = []
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            proc_pid = int(entry.name)
            namespace_pids = _status_namespace_pids(proc_pid)
            if not namespace_pids or namespace_pids[-1] != pid:
                continue
            if current_namespace is not None and _pid_namespace_inode(proc_pid) != current_namespace:
                continue
            candidates.append(proc_pid)
    if len(candidates) == 1:
        return candidates[0]
    if pid in candidates:
        return pid
    return pid


def _read_process_stat(pid: int) -> ProcessStat | None:
    try:
        line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = line[line.rfind(")") + 2 :].split()
        return ProcessStat(
            identity=ProcessIdentity(pid=pid, start_time_ticks=int(fields[19])),
            parent_pid=int(fields[1]),
            process_group=int(fields[2]),
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError, ValueError):
        return None


def _scan_process_table() -> dict[int, ProcessStat]:
    processes: dict[int, ProcessStat] = {}
    try:
        entries = os.scandir("/proc")
    except OSError:
        return processes
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            stat = _read_process_stat(int(entry.name))
            if stat is not None:
                processes[stat.identity.pid] = stat
    return processes


def _read_status_value_bytes(pid: int, key: str) -> int | None:
    try:
        with Path(f"/proc/{pid}/status").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{key}:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None
    return None


def _read_pss_bytes(pid: int) -> int | None:
    try:
        with Path(f"/proc/{pid}/smaps_rollup").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None
    return None


def _discover_scope(
    process_table: dict[int, ProcessStat],
    known: set[ProcessIdentity],
    root_process_group: int,
) -> set[ProcessIdentity]:
    current_known_pids = {
        identity.pid
        for identity in known
        if process_table.get(identity.pid) is not None
        and process_table[identity.pid].identity == identity
    }
    selected_pids = {
        pid for pid, stat in process_table.items() if stat.process_group == root_process_group
    }
    selected_pids.update(current_known_pids)
    changed = True
    while changed:
        changed = False
        for pid, stat in process_table.items():
            if pid not in selected_pids and stat.parent_pid in selected_pids:
                selected_pids.add(pid)
                changed = True
    return {process_table[pid].identity for pid in selected_pids}


def _sample_memory(identities: set[ProcessIdentity]) -> dict[str, int | None]:
    aggregate_rss = 0
    aggregate_pss = 0
    max_single_rss = 0
    max_single_pss = 0
    rss_processes = 0
    pss_processes = 0
    for identity in identities:
        current = _read_process_stat(identity.pid)
        if current is None or current.identity != identity:
            continue
        rss = _read_status_value_bytes(identity.pid, "VmRSS")
        if rss is not None:
            aggregate_rss += rss
            max_single_rss = max(max_single_rss, rss)
            rss_processes += 1
        pss = _read_pss_bytes(identity.pid)
        if pss is not None:
            aggregate_pss += pss
            max_single_pss = max(max_single_pss, pss)
            pss_processes += 1
    return {
        "aggregate_rss_bytes": aggregate_rss,
        "aggregate_pss_bytes": aggregate_pss if pss_processes else None,
        "max_single_process_rss_bytes": max_single_rss,
        "max_single_process_pss_bytes": max_single_pss if pss_processes else None,
        "rss_process_count": rss_processes,
        "pss_process_count": pss_processes,
    }


def _read_anchor_identities(paths: Sequence[Path]) -> set[ProcessIdentity]:
    identities: set[ProcessIdentity] = set()
    for path in paths:
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        stat = _read_process_stat(_resolve_current_namespace_pid(pid))
        if stat is not None:
            identities.add(stat.identity)
    return identities


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _terminate_tracked_processes(
    process: subprocess.Popen[bytes],
    root_process_group: int,
    identities: set[ProcessIdentity],
) -> None:
    process_groups = {root_process_group}
    for identity in identities:
        current = _read_process_stat(identity.pid)
        if current is not None and current.identity == identity:
            process_groups.add(current.process_group)
    process_groups.discard(os.getpgrp())
    for process_group in process_groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        for process_group in process_groups:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        process.wait()


def run_with_process_tree_memory(
    command: Sequence[str],
    output_path: str | Path,
    sample_interval_seconds: float = 0.1,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    anchor_pid_files: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Run a command and write process-tree RSS/PSS measurement evidence."""

    if not command:
        raise ValueError("command must not be empty")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")

    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    root_procfs_pid = _resolve_current_namespace_pid(process.pid)
    root_stat = _read_process_stat(root_procfs_pid)
    if root_stat is None:
        process.wait()
        raise RuntimeError("could not read root process identity from /proc")

    known: set[ProcessIdentity] = {root_stat.identity}
    anchor_paths = [Path(path) for path in anchor_pid_files]
    anchored: set[ProcessIdentity] = set()
    sample_count = 0
    samples_with_pss = 0
    peak_rss = 0
    peak_pss = 0
    peak_rss_process_count = 0
    peak_pss_process_count = 0
    max_single_rss = 0
    max_single_pss = 0
    max_concurrent_processes = 0

    try:
        while True:
            process_table = _scan_process_table()
            current_anchors = _read_anchor_identities(anchor_paths)
            anchored.update(current_anchors)
            known.update(current_anchors)
            known.update(_discover_scope(process_table, known, root_stat.process_group))
            sample = _sample_memory(known)
            sample_count += 1
            rss = int(sample["aggregate_rss_bytes"] or 0)
            pss_value = sample["aggregate_pss_bytes"]
            process_count = int(sample["rss_process_count"] or 0)
            if rss > peak_rss:
                peak_rss = rss
                peak_rss_process_count = process_count
            if pss_value is not None:
                samples_with_pss += 1
                pss = int(pss_value)
                if pss > peak_pss:
                    peak_pss = pss
                    peak_pss_process_count = int(sample["pss_process_count"] or 0)
            max_single_rss = max(max_single_rss, int(sample["max_single_process_rss_bytes"] or 0))
            max_single_pss = max(max_single_pss, int(sample["max_single_process_pss_bytes"] or 0))
            max_concurrent_processes = max(max_concurrent_processes, process_count)
            if process.poll() is not None:
                break
            time.sleep(sample_interval_seconds)
    except BaseException:
        _terminate_tracked_processes(process, root_stat.process_group, known)
        raise

    finished_at = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if process.returncode == 0 and sample_count > 0 else "fail",
        "measurement_method": "linux_procfs_process_tree_sampling_v1",
        "measurement_scope": "root_process_group_dynamic_pid_anchors_and_discovered_descendants",
        "command": list(command),
        "anchor_pid_files": [str(path) for path in anchor_paths],
        "anchor_processes_discovered": len(anchored),
        "launcher_pid": process.pid,
        "root_pid": root_stat.identity.pid,
        "root_start_time_ticks": root_stat.identity.start_time_ticks,
        "root_process_group": root_stat.process_group,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "sample_interval_seconds": sample_interval_seconds,
        "sample_count": sample_count,
        "samples_with_pss": samples_with_pss,
        "processes_seen": len(known),
        "max_concurrent_processes": max_concurrent_processes,
        "peak_aggregate_rss_bytes": peak_rss,
        "peak_aggregate_rss_mb": peak_rss / (1024**2),
        "peak_aggregate_rss_process_count": peak_rss_process_count,
        "peak_aggregate_pss_bytes": peak_pss if samples_with_pss else None,
        "peak_aggregate_pss_mb": peak_pss / (1024**2) if samples_with_pss else None,
        "peak_aggregate_pss_process_count": peak_pss_process_count if samples_with_pss else None,
        "max_single_process_rss_bytes": max_single_rss,
        "max_single_process_rss_mb": max_single_rss / (1024**2),
        "max_single_process_pss_bytes": max_single_pss if samples_with_pss else None,
        "max_single_process_pss_mb": max_single_pss / (1024**2) if samples_with_pss else None,
        "exit_code": process.returncode,
        "host_pid_namespace_required_for_container_scope": True,
        "sampling_limitations": [
            "peaks shorter than the sampling interval may be underestimated",
            "aggregate RSS can double-count shared pages; aggregate PSS is the preferred comparison measure",
            "processes that fork and exit entirely between samples may be missed",
        ],
    }
    _write_json_atomic(Path(output_path), payload)
    return payload
