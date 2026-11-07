"""Deterministic post-processing of MACS3 summits into a frozen peak set."""

from __future__ import annotations

import bisect
import gzip
import json
from pathlib import Path

import pandas as pd


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            chromosome, size, *_ = line.rstrip("\n").split("\t")
            sizes[chromosome] = int(size)
    return sizes


def _read_blacklist(path: Path) -> dict[str, tuple[list[int], list[int]]]:
    records: dict[str, list[tuple[int, int]]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chromosome, start, end, *_ = line.rstrip("\n").split("\t")
            records.setdefault(chromosome, []).append((int(start), int(end)))
    result: dict[str, tuple[list[int], list[int]]] = {}
    for chromosome, intervals in records.items():
        intervals.sort()
        result[chromosome] = ([start for start, _ in intervals], [end for _, end in intervals])
    return result


def _overlaps_blacklist(
    chromosome: str,
    start: int,
    end: int,
    blacklist: dict[str, tuple[list[int], list[int]]],
) -> bool:
    if chromosome not in blacklist:
        return False
    starts, ends = blacklist[chromosome]
    index = bisect.bisect_left(starts, end)
    return index > 0 and ends[index - 1] > start


def freeze_macs3_summits(
    narrowpeak_path: str | Path,
    chrom_sizes_path: str | Path,
    blacklist_path: str | Path,
    output_bed: str | Path,
    audit_path: str | Path,
    peak_width: int = 501,
) -> dict[str, object]:
    """Create fixed-width, non-overlapping, blacklist-filtered summit peaks."""

    if peak_width <= 0 or peak_width % 2 == 0:
        raise ValueError("peak_width must be a positive odd integer")
    narrowpeak_path = Path(narrowpeak_path)
    chrom_sizes = _read_chrom_sizes(Path(chrom_sizes_path))
    blacklist = _read_blacklist(Path(blacklist_path))
    standard = {
        chromosome
        for chromosome in chrom_sizes
        if chromosome in {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY"}
    }
    half_width = peak_width // 2
    candidates: list[dict[str, object]] = []
    input_rows = 0
    invalid_rows = 0
    nonstandard_rows = 0
    blacklist_rows = 0
    with narrowpeak_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            input_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                invalid_rows += 1
                continue
            chromosome = fields[0]
            if chromosome not in standard:
                nonstandard_rows += 1
                continue
            try:
                source_start = int(fields[1])
                summit_offset = int(fields[9])
                signal = float(fields[6])
                qscore = float(fields[8])
            except ValueError:
                invalid_rows += 1
                continue
            summit = source_start + summit_offset
            start = max(0, summit - half_width)
            end = start + peak_width
            if end > chrom_sizes[chromosome]:
                end = chrom_sizes[chromosome]
                start = max(0, end - peak_width)
            if end - start != peak_width:
                invalid_rows += 1
                continue
            if _overlaps_blacklist(chromosome, start, end, blacklist):
                blacklist_rows += 1
                continue
            candidates.append(
                {
                    "chromosome": chromosome,
                    "start": start,
                    "end": end,
                    "summit": summit,
                    "signal": signal,
                    "qscore": qscore,
                }
            )

    candidates.sort(key=lambda row: (-row["qscore"], -row["signal"], row["chromosome"], row["start"]))
    buckets: dict[tuple[str, int], list[int]] = {}
    selected: list[dict[str, object]] = []
    overlap_rows = 0
    for candidate in candidates:
        chromosome = str(candidate["chromosome"])
        start = int(candidate["start"])
        end = int(candidate["end"])
        bucket_ids = range(start // peak_width, (end - 1) // peak_width + 1)
        neighbor_ids = {
            peak_id
            for bucket_id in bucket_ids
            for peak_id in buckets.get((chromosome, bucket_id), [])
        }
        if any(
            int(selected[peak_id]["start"]) < end and int(selected[peak_id]["end"]) > start
            for peak_id in neighbor_ids
        ):
            overlap_rows += 1
            continue
        peak_id = len(selected)
        selected.append(candidate)
        for bucket_id in bucket_ids:
            buckets.setdefault((chromosome, bucket_id), []).append(peak_id)

    selected.sort(key=lambda row: (row["chromosome"], row["start"], row["end"]))
    output_bed = Path(output_bed)
    output_bed.parent.mkdir(parents=True, exist_ok=True)
    with output_bed.open("w", encoding="utf-8") as handle:
        for index, peak in enumerate(selected, start=1):
            handle.write(
                f"{peak['chromosome']}\t{peak['start']}\t{peak['end']}\tpeak_{index:06d}\t"
                f"{peak['qscore']}\t{peak['signal']}\t{peak['summit']}\n"
            )

    result = {
        "narrowpeak_path": str(narrowpeak_path),
        "output_bed": str(output_bed),
        "peak_width": peak_width,
        "input_peak_rows": input_rows,
        "invalid_peak_rows": invalid_rows,
        "nonstandard_chromosome_rows": nonstandard_rows,
        "blacklist_overlap_rows": blacklist_rows,
        "overlapping_candidate_rows": overlap_rows,
        "frozen_peak_rows": len(selected),
        "standard_chromosomes": sorted(standard),
        "status": "pass" if selected and invalid_rows == 0 else "fail",
    }
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([result]).to_csv(audit_path.with_suffix(".tsv"), sep="\t", index=False)
    return result
