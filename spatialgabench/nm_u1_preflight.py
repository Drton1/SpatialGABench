"""Expression-blind input preflight for the NM-U1 human hippocampus extension."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_NUCLEAR = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rna_header(path: Path) -> list[str]:
    # Deliberately read exactly one line; expression rows remain inaccessible.
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.readline().rstrip("\r\n").split("\t")


def _positions(path: Path) -> tuple[list[str], int]:
    barcodes: list[str] = []
    rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            rows += 1
            if len(row) != 6:
                raise ValueError(f"Expected six spatial columns, observed {len(row)} at row {rows}")
            if row[1].strip() == "1":
                barcodes.append(row[0].strip())
    return barcodes, rows


def _fragment_scan(path: Path, allowed_barcodes: set[str]) -> dict[str, Any]:
    header_rows = 0
    data_rows = 0
    weighted_fragments = 0
    invalid_rows = 0
    unmatched_barcode_rows = 0
    coordinate_order_violations = 0
    repeated_contig_blocks = 0
    canonical_rows = 0
    noncanonical_rows = 0
    observed_barcodes: set[str] = set()
    contig_counts: Counter[str] = Counter()
    closed_contigs: set[str] = set()
    current_contig: str | None = None
    previous_start = -1
    reference_path = ""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                header_rows += 1
                if line.startswith("# reference_path="):
                    reference_path = line.rstrip("\r\n").split("=", 1)[1]
                continue
            fields = line.rstrip("\r\n").split("\t")
            data_rows += 1
            if len(fields) != 5:
                invalid_rows += 1
                continue
            chrom, start_text, end_text, barcode_raw, count_text = fields
            try:
                start = int(start_text)
                end = int(end_text)
                count = int(count_text)
            except ValueError:
                invalid_rows += 1
                continue
            if start < 0 or end <= start or count <= 0:
                invalid_rows += 1
                continue
            barcode = barcode_raw[:-2] if barcode_raw.endswith("-1") else barcode_raw
            observed_barcodes.add(barcode)
            if barcode not in allowed_barcodes:
                unmatched_barcode_rows += 1
            weighted_fragments += count
            contig_counts[chrom] += 1
            if chrom in CANONICAL_NUCLEAR:
                canonical_rows += 1
            else:
                noncanonical_rows += 1
            if chrom != current_contig:
                if current_contig is not None:
                    closed_contigs.add(current_contig)
                if chrom in closed_contigs:
                    repeated_contig_blocks += 1
                current_contig = chrom
                previous_start = -1
            if start < previous_start:
                coordinate_order_violations += 1
            previous_start = start

    return {
        "header_rows": header_rows,
        "data_rows": data_rows,
        "weighted_fragments": weighted_fragments,
        "invalid_rows": invalid_rows,
        "unmatched_barcode_rows": unmatched_barcode_rows,
        "coordinate_order_violations": coordinate_order_violations,
        "repeated_contig_blocks": repeated_contig_blocks,
        "canonical_nuclear_rows": canonical_rows,
        "noncanonical_rows": noncanonical_rows,
        "unique_fragment_barcodes": len(observed_barcodes),
        "fragment_position_barcode_overlap": len(observed_barcodes & allowed_barcodes),
        "reference_path": reference_path,
        "contig_counts": dict(sorted(contig_counts.items())),
    }


def run_preflight(root: Path, authorization_path: Path, outdir: Path) -> Path:
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "authorized":
        raise ValueError("NM-U1 download authorization is not active")
    if authorization.get("rna_expression_access_allowed") is not False:
        raise ValueError("NM-U1 preflight requires expression access to remain forbidden")

    manifest_path = root / authorization["download_manifest"]["path"]
    if sha256_path(manifest_path) != authorization["download_manifest"]["sha256"]:
        raise ValueError("NM-U1 download manifest checksum mismatch")
    protocol_path = root / authorization["upgrade_protocol"]["path"]
    if sha256_path(protocol_path) != authorization["upgrade_protocol"]["sha256"]:
        raise ValueError("NM upgrade protocol checksum mismatch")

    manifest = pd.read_csv(manifest_path, sep="\t")
    raw = root / "data/raw/v1/GSE205055"
    receipt_path = root / "data/manifests/nm_u1_human_hippocampus_download_receipt.tsv"
    receipt = pd.read_csv(receipt_path, sep="\t")
    receipt_by_name = receipt.set_index("file_name")
    file_rows: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        path = raw / row["file_name"]
        observed_size = path.stat().st_size if path.is_file() else -1
        receipt_row = receipt_by_name.loc[row["file_name"]]
        digest = sha256_path(path) if path.is_file() else ""
        file_rows.append(
            {
                "file_name": row["file_name"],
                "expected_size_bytes": int(row["expected_size_bytes"]),
                "observed_size_bytes": observed_size,
                "sha256": digest,
                "receipt_sha256": receipt_row["sha256"],
                "size_pass": observed_size == int(row["expected_size_bytes"]),
                "receipt_hash_pass": digest == receipt_row["sha256"],
            }
        )

    positions_path = raw / "spatial/HumanBrain_50um_spatial/tissue_positions_list.csv"
    position_barcodes, position_rows = _positions(positions_path)
    rna_barcodes = _rna_header(raw / "GSM6206885_HumanBrain_50um_matrix.tsv.gz")
    position_set, rna_set = set(position_barcodes), set(rna_barcodes)
    fragment = _fragment_scan(
        raw / "GSM6206884_HumanBrain_50um_fragments.tsv.gz", position_set
    )

    checks = {
        "three_files_verified": len(file_rows) == 3
        and all(row["size_pass"] and row["receipt_hash_pass"] for row in file_rows),
        "positions_2500_unique_in_tissue": position_rows == len(position_barcodes) == len(position_set) == 2500,
        "rna_header_2500_unique": len(rna_barcodes) == len(rna_set) == 2500,
        "rna_position_barcode_sets_equal": rna_set == position_set,
        "fragment_rows_valid": fragment["data_rows"] > 0 and fragment["invalid_rows"] == 0,
        "fragment_barcodes_all_match_positions": fragment["unmatched_barcode_rows"] == 0
        and fragment["fragment_position_barcode_overlap"] == len(position_set),
        "fragment_coordinate_order_valid": fragment["coordinate_order_violations"] == 0
        and fragment["repeated_contig_blocks"] == 0,
        "genome_build_header_is_GRCh38_2020_A": "GRCh38-2020-A" in fragment["reference_path"],
        "rna_expression_rows_read": False,
        "rna_used_for_construction": False,
        "labels_or_pathology_accessed": False,
    }
    negative_sentinels = {
        "rna_expression_rows_read",
        "rna_used_for_construction",
        "labels_or_pathology_accessed",
    }
    passed = all(value is True for key, value in checks.items() if key not in negative_sentinels)
    passed = passed and all(checks[key] is False for key in negative_sentinels)

    outdir.mkdir(parents=True, exist_ok=True)
    files_path = outdir / "download_identity.tsv"
    pd.DataFrame(file_rows).to_csv(files_path, sep="\t", index=False)
    contigs_path = outdir / "fragment_contig_inventory.tsv"
    pd.DataFrame(
        [{"contig": key, "rows": value, "canonical_nuclear": key in CANONICAL_NUCLEAR}
         for key, value in fragment.pop("contig_counts").items()]
    ).to_csv(contigs_path, sep="\t", index=False)
    barcode_path = outdir / "barcode_contract.tsv"
    pd.DataFrame(
        [{
            "position_rows": position_rows,
            "unique_position_barcodes": len(position_set),
            "rna_header_barcodes": len(rna_barcodes),
            "unique_rna_header_barcodes": len(rna_set),
            "rna_position_overlap": len(rna_set & position_set),
            "rna_position_sets_equal": rna_set == position_set,
            "rna_position_order_equal": rna_barcodes == position_barcodes,
            "unique_fragment_barcodes": fragment["unique_fragment_barcodes"],
            "fragment_position_overlap": fragment["fragment_position_barcode_overlap"],
        }]
    ).to_csv(barcode_path, sep="\t", index=False)

    evidence = {
        "schema_version": 1,
        "milestone": "NM-U1_human_hippocampus_expression_blind_preflight",
        "status": "pass" if passed else "blocked",
        "species": "human",
        "genome_build": "GRCh38",
        "study_family": "zhang_2023",
        "independent_study_increment": 0,
        "within_family_cross_species_extension": True,
        "fragment_scan": fragment,
        "checks": checks,
        "download_identity": {"path": str(files_path.relative_to(root)), "sha256": sha256_path(files_path)},
        "barcode_contract": {"path": str(barcode_path.relative_to(root)), "sha256": sha256_path(barcode_path)},
        "contig_inventory": {"path": str(contigs_path.relative_to(root)), "sha256": sha256_path(contigs_path)},
        "positions": {"path": str(positions_path.relative_to(root)), "sha256": sha256_path(positions_path)},
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
        "rna_expression_accessed": False,
        "method_execution_run": False,
        "universal_ranking_run": False,
        "final_ranking_run": False,
    }
    evidence_path = outdir / "preflight_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--authorization",
        default="governance/authorizations/NM_U1_HUMAN_HIPPOCAMPUS_MINIMAL_DOWNLOAD_AUTHORIZATION_2026-07-30.json",
    )
    parser.add_argument("--outdir", default="results/v1/nm_upgrade/nu1_input_preflight")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(run_preflight(root, root / args.authorization, root / args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
