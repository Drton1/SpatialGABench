"""Reusable simulated-fragments generator and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pysam
from scipy import sparse, stats


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(stats.pearsonr(x, y).statistic)


def generate_fragments_from_peak_matrix(adata: ad.AnnData, config: dict[str, Any], outdir: str | Path) -> dict[str, Path]:
    """Generate one fragment per nonzero peak/cell count unit."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    simulation_id = str(config.get("simulation_id", adata.uns.get("simulation_id", "simulation")))
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    var = adata.var
    required = {"chr", "start", "end"}
    missing = sorted(required.difference(var.columns))
    if missing:
        raise ValueError(f"Peak matrix var missing required coordinate columns: {missing}")
    records = []
    coo = X.tocoo()
    for cell_idx, peak_idx, count in zip(coo.row, coo.col, coo.data):
        n = int(max(0, round(float(count))))
        if n == 0:
            continue
        chrom = str(var.iloc[int(peak_idx)]["chr"])
        start = int(var.iloc[int(peak_idx)]["start"])
        end = int(var.iloc[int(peak_idx)]["end"])
        barcode = str(adata.obs_names[int(cell_idx)])
        for _ in range(n):
            records.append((chrom, start, end, barcode))
    records.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    text_path = outdir / f"{simulation_id}_fragments.tsv"
    gz_path = outdir / f"{simulation_id}_fragments.tsv.gz"
    with text_path.open("w", encoding="utf-8") as handle:
        for chrom, start, end, barcode in records:
            handle.write(f"{chrom}\t{start}\t{end}\t{barcode}\n")
    pysam.tabix_compress(str(text_path), str(gz_path), force=True)
    pysam.tabix_index(str(gz_path), seq_col=0, start_col=1, end_col=2, force=True)
    text_path.unlink()
    tbi_path = Path(f"{gz_path}.tbi")
    qc = pd.DataFrame(
        [
            {
                "simulation_id": simulation_id,
                "fragments_written": len(records),
                "barcodes_with_fragments": len({r[3] for r in records}),
                "fragments_path": str(gz_path),
                "index_path": str(tbi_path),
                "index_policy": "pysam_bgzf_and_real_tabix_index",
                "status": "pass" if records else "empty",
            }
        ]
    )
    qc_path = outdir / "fragment_generation_qc.tsv"
    qc.to_csv(qc_path, sep="\t", index=False)
    return {"fragments": gz_path, "index": tbi_path, "qc": qc_path}


def reconstruct_peak_matrix_from_fragments(simulation_dir: str | Path, fragments_path: str | Path) -> sparse.csr_matrix:
    simulation_dir = Path(simulation_dir)
    adata = ad.read_h5ad(simulation_dir / "simulated_atac_peak_matrix.h5ad")
    peaks = adata.var[["chr", "start", "end"]].copy().reset_index(drop=True)
    cell_index = {str(name): i for i, name in enumerate(adata.obs_names)}
    rows: list[int] = []
    cols: list[int] = []
    with pysam.BGZFile(str(fragments_path), "r") as handle:
        for line in handle:
            chrom, start_s, end_s, barcode = line.decode("utf-8").rstrip("\n").split("\t")[:4]
            start, end = int(start_s), int(end_s)
            if barcode not in cell_index:
                continue
            hit = peaks[(peaks["chr"].astype(str) == chrom) & (peaks["start"].astype(int) <= start) & (peaks["end"].astype(int) >= end)]
            if hit.empty:
                continue
            rows.append(cell_index[barcode])
            cols.append(int(hit.index[0]))
    data = np.ones(len(rows), dtype=np.float32)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=adata.shape, dtype=np.float32)
    mat.sum_duplicates()
    return mat


def validate_fragments_for_simulation_dir(simulation_dir: str | Path, fragments_dir: str | Path) -> pd.DataFrame:
    simulation_dir = Path(simulation_dir)
    fragments_dir = Path(fragments_dir)
    adata = ad.read_h5ad(simulation_dir / "simulated_atac_peak_matrix.h5ad")
    simulation_id = str(adata.uns.get("simulation_id", simulation_dir.name))
    frag = fragments_dir / f"{simulation_id}_fragments.tsv.gz"
    original = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    reconstructed = reconstruct_peak_matrix_from_fragments(simulation_dir, frag)
    global_pearson = _corr(original.toarray(), reconstructed.toarray())
    per_cell_total_correlation = _corr(np.asarray(original.sum(axis=1)).ravel(), np.asarray(reconstructed.sum(axis=1)).ravel())
    per_peak_total_correlation = _corr(np.asarray(original.sum(axis=0)).ravel(), np.asarray(reconstructed.sum(axis=0)).ravel())
    nnz_overlap = int(((original > 0).multiply(reconstructed > 0)).nnz)
    barcode_overlap = int((np.asarray(reconstructed.sum(axis=1)).ravel() > 0).sum())
    peak_overlap = int((np.asarray(reconstructed.sum(axis=0)).ravel() > 0).sum())
    out = pd.DataFrame(
        [
            {
                "simulation_id": simulation_id,
                "global_pearson": global_pearson,
                "per_cell_total_correlation": per_cell_total_correlation,
                "per_peak_total_correlation": per_peak_total_correlation,
                "nnz_overlap": nnz_overlap,
                "barcode_overlap": barcode_overlap,
                "peak_overlap": peak_overlap,
                "validation_status": "pass" if global_pearson > 0.99 else "partial",
            }
        ]
    )
    path = fragments_dir / "fragment_reconstruction_validation.tsv"
    out.to_csv(path, sep="\t", index=False)
    return out
