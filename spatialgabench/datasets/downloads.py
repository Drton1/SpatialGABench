"""Reproducible verification receipts for accession-level downloads."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "study_id",
    "accession",
    "file_name",
    "size_bytes",
    "url",
    "benchmark_role",
    "license_note",
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_gzip(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if path.suffix != ".gz":
        return "not_applicable"
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(chunk_size):
                pass
    except (EOFError, OSError):
        return "fail"
    return "pass"


def verify_download_manifest(
    manifest_path: str | Path,
    raw_root: str | Path,
    verified_on: str,
) -> pd.DataFrame:
    """Verify expected size, compressed-stream integrity, and SHA-256."""

    manifest_path = Path(manifest_path)
    raw_root = Path(raw_root)
    manifest = pd.read_csv(manifest_path, sep="\t", dtype={"accession": str, "file_name": str})
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Download manifest is missing columns: {sorted(missing_columns)}")

    rows: list[dict[str, object]] = []
    resolved_raw_root = raw_root.resolve()
    for record in manifest.to_dict(orient="records"):
        explicit_path = record.get("local_path")
        if explicit_path is not None and not pd.isna(explicit_path) and str(explicit_path).strip():
            path = raw_root / str(explicit_path)
        else:
            path = raw_root / str(record["accession"]) / str(record["file_name"])
        try:
            path.resolve().relative_to(resolved_raw_root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes raw root: {path}") from exc
        expected_size = int(record["size_bytes"])
        exists = path.is_file()
        observed_size = path.stat().st_size if exists else 0
        size_status = "pass" if exists and observed_size == expected_size else "fail"
        gzip_status = _verify_gzip(path) if exists else "not_checked_missing"
        sha256 = _sha256(path) if exists else ""
        verification_status = (
            "pass"
            if exists and size_status == "pass" and gzip_status in {"pass", "not_applicable"}
            else "fail"
        )
        rows.append(
            {
                "study_id": record["study_id"],
                "accession": record["accession"],
                "file_name": record["file_name"],
                "benchmark_role": record["benchmark_role"],
                "source_url": record["url"],
                "local_path": str(path),
                "expected_size_bytes": expected_size,
                "observed_size_bytes": observed_size,
                "size_status": size_status,
                "gzip_status": gzip_status,
                "sha256": sha256,
                "license_note": record["license_note"],
                "verified_on": verified_on,
                "verification_status": verification_status,
            }
        )
    return pd.DataFrame(rows)


def write_download_receipt(
    manifest_path: str | Path,
    raw_root: str | Path,
    output_path: str | Path,
    verified_on: str,
) -> Path:
    """Write a deterministic TSV receipt and return its path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = verify_download_manifest(manifest_path, raw_root, verified_on)
    receipt.to_csv(output_path, sep="\t", index=False)
    return output_path
