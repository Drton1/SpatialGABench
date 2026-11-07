"""Deterministic accessibility-only fixture for software wrapper smoke tests."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pysam
from scipy import sparse


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_wrapper_toy_fixture(output_dir: str | Path, seed: int = 1601) -> dict[str, Path]:
    """Write a small peak matrix, indexed fragments, and matching annotation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    barcodes = [f"TOYBC{index:03d}" for index in range(64)]
    gene_records = [
        ("TOYGENE_A", "ToyGeneA", "TOYTX_A", 200_001, 208_000, "+"),
        ("TOYGENE_B", "ToyGeneB", "TOYTX_B", 590_001, 602_000, "-"),
        ("TOYGENE_C", "ToyGeneC", "TOYTX_C", 1_000_001, 1_018_000, "+"),
        ("TOYGENE_D", "ToyGeneD", "TOYTX_D", 1_400_001, 1_425_000, "-"),
    ]
    peak_rows: list[tuple[str, int, int, str, int]] = []
    offsets = [-120_000, -80_000, -40_000, 0, 25_000, 60_000, 100_000, 140_000]
    for module_id, (_, _, _, gene_start, gene_end, strand) in enumerate(gene_records):
        tss = gene_start if strand == "+" else gene_end
        for offset in offsets:
            center = tss + offset
            peak_rows.append(("chr1", center - 500, center + 500, "", module_id))
    for center in np.linspace(1_750_000, 2_450_000, 8, dtype=int):
        peak_rows.append(("chr1", int(center - 500), int(center + 500), "", -1))
    peaks = pd.DataFrame(
        peak_rows,
        columns=["chromosome", "start", "end", "feature_name", "_module_id"],
    )
    peaks["feature_name"] = [f"toy_peak_{index:03d}" for index in range(len(peaks))]
    peaks["width"] = peaks["end"] - peaks["start"]
    peaks["representation_id"] = "toy_frozen_peaks"
    peaks.index = [
        f"{row.chromosome}:{row.start}-{row.end}"
        for row in peaks.itertuples(index=False)
    ]

    latent_accessibility = rng.normal(size=(len(barcodes), len(gene_records)))
    depth_effect = rng.normal(scale=0.35, size=len(barcodes))
    peak_effect = rng.normal(scale=0.25, size=len(peaks))
    probabilities = np.empty((len(barcodes), len(peaks)), dtype=float)
    for peak_index, module_id in enumerate(peaks["_module_id"].to_numpy()):
        if module_id >= 0:
            logits = -0.55 + 1.45 * latent_accessibility[:, module_id] + 0.25 * depth_effect
        else:
            logits = -0.9 + 0.25 * depth_effect
        probabilities[:, peak_index] = 1.0 / (1.0 + np.exp(-(logits + peak_effect[peak_index])))
    accessible = rng.random(probabilities.shape) < probabilities
    for peak_index in range(accessible.shape[1]):
        order = np.argsort(probabilities[:, peak_index])
        if accessible[:, peak_index].sum() < 4:
            accessible[order[-4:], peak_index] = True
        if accessible[:, peak_index].sum() > len(barcodes) - 4:
            accessible[order[:4], peak_index] = False
    for cell_index in range(accessible.shape[0]):
        if accessible[cell_index].sum() < 4:
            top = np.argsort(probabilities[cell_index])[-4:]
            accessible[cell_index, top] = True

    fragment_counter: Counter[tuple[str, int, int, str]] = Counter()
    for cell_index, barcode in enumerate(barcodes):
        for peak_index in np.flatnonzero(accessible[cell_index]):
            peak = peaks.iloc[int(peak_index)]
            expected = 7.0 * np.exp(0.25 * depth_effect[cell_index] + 0.15 * peak_effect[peak_index])
            n_fragments = max(1, int(rng.poisson(expected)))
            peak_start = int(peak["start"])
            peak_end = int(peak["end"])
            center = (peak_start + peak_end) // 2
            for _ in range(n_fragments):
                midpoint = int(np.clip(rng.normal(center, 130), peak_start + 60, peak_end - 60))
                length = int(rng.integers(35, 120))
                start = max(peak_start, midpoint - length // 2)
                end = min(peak_end, start + length)
                fragment_counter[("chr1", start, end, barcode)] += 1

    fragments_tsv = output_dir / "toy.fragments.tsv"
    with fragments_tsv.open("w", encoding="utf-8") as handle:
        for (chromosome, start, end, barcode), count in sorted(fragment_counter.items()):
            handle.write(f"{chromosome}\t{start}\t{end}\t{barcode}\t{count}\n")
    fragments_gz = output_dir / "toy.fragments.tsv.gz"
    pysam.tabix_compress(str(fragments_tsv), str(fragments_gz), force=True)
    pysam.tabix_index(str(fragments_gz), preset="bed", force=True)
    fragments_tsv.unlink()

    barcode_to_row = {barcode: index for index, barcode in enumerate(barcodes)}
    matrix = sparse.dok_matrix((len(barcodes), len(peaks)), dtype=np.int32)
    for (chromosome, start, end, barcode), count in fragment_counter.items():
        midpoint = (start + end) // 2
        matches = np.flatnonzero(
            (peaks["chromosome"].to_numpy() == chromosome)
            & (peaks["start"].to_numpy() <= midpoint)
            & (midpoint < peaks["end"].to_numpy())
        )
        for column in matches:
            matrix[barcode_to_row[barcode], int(column)] += count

    side = 8
    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    obs["array_row"] = np.repeat(np.arange(side), side)
    obs["array_col"] = np.tile(np.arange(side), side)
    obs["in_tissue"] = 1
    adata = ad.AnnData(X=matrix.tocsr(), obs=obs, var=peaks.drop(columns="_module_id").copy())
    adata.obsm["spatial"] = obs[["array_col", "array_row"]].to_numpy(dtype=float)
    adata.uns.update(
        {
            "study_id": "toy_wrapper_contract",
            "dataset_id": "toy_wrapper_dataset",
            "sample_id": "toy_wrapper_sample",
            "genome_build": "mm10",
            "representation_id": "toy_frozen_peaks",
            "representation_source": "deterministic_accessibility_only_fixture",
            "input_track": "P",
            "comparison_scope": "toy_contract_only",
            "construction_leakage_inputs": "none",
            "seed": seed,
        }
    )
    h5ad_path = output_dir / "toy_peak_matrix.h5ad"
    adata.write_h5ad(h5ad_path, compression="gzip")

    chrom_sizes_path = output_dir / "toy.chrom.sizes.tsv"
    chrom_sizes_path.write_text("chr1\t195471971\n", encoding="utf-8")
    gtf_path = output_dir / "toy.annotation.gtf"
    gtf_lines: list[str] = []
    for gene_id, gene_name, transcript_id, start, end, strand in gene_records:
        gene_attributes = (
            f'gene_id "{gene_id}"; gene_name "{gene_name}"; gene_biotype "protein_coding";'
        )
        transcript_attributes = (
            f'gene_id "{gene_id}"; gene_name "{gene_name}"; transcript_id "{transcript_id}"; '
            f'transcript_name "{transcript_id}";'
        )
        gtf_lines.extend(
            [
                f"chr1\tSpatialGABench\tgene\t{start}\t{end}\t.\t{strand}\t.\t{gene_attributes}",
                f"chr1\tSpatialGABench\ttranscript\t{start}\t{end}\t.\t{strand}\t.\t{transcript_attributes}",
                f"chr1\tSpatialGABench\texon\t{start}\t{end}\t.\t{strand}\t.\t{transcript_attributes} exon_number \"1\";",
            ]
        )
    gtf_path.write_text("\n".join(gtf_lines) + "\n", encoding="utf-8")

    artifacts = {
        "input_h5ad": h5ad_path,
        "fragments": fragments_gz,
        "fragments_index": Path(str(fragments_gz) + ".tbi"),
        "gene_annotation": gtf_path,
        "chrom_sizes": chrom_sizes_path,
    }
    manifest = {
        "schema_version": 1,
        "fixture_id": "toy_wrapper_dataset",
        "seed": seed,
        "n_observations": adata.n_obs,
        "n_peaks": adata.n_vars,
        "fragment_rows": len(fragment_counter),
        "artifacts": {
            key: {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for key, path in artifacts.items()
        },
        "forbidden_inputs": [],
        "status": "pass",
    }
    manifest_path = output_dir / "toy_fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["manifest"] = manifest_path
    return artifacts
