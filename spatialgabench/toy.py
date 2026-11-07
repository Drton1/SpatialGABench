"""Toy spatial ATAC example generation and benchmark runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats

from .gene_activity.gene_body import build_gene_body_mapping
from .gene_activity.interval_aggregation import (
    compute_gene_activity,
    load_bed_intervals,
    load_gene_metadata,
    parse_peak_coordinates,
    write_gene_activity_h5ad,
)
from .gene_activity.promoter import build_peak_gene_overlap, load_promoters_bed
from .gene_activity.promoter_gene_body import build_promoter_gene_body_mapping
from .gene_activity.random_control import build_random_control_mapping


TOY_METHODS = [
    "promoter_only_2kb_pc",
    "gene_body_pc",
    "promoter_plus_gene_body_2kb_pc",
    "random_peak_to_gene_negative_control_pc",
]


def _toy_peak_matrix() -> ad.AnnData:
    X = sparse.csr_matrix(
        np.asarray(
            [
                [5, 1, 0, 0, 3, 1],
                [4, 2, 0, 1, 2, 2],
                [0, 1, 6, 2, 1, 0],
                [1, 0, 5, 3, 0, 1],
                [2, 4, 1, 0, 5, 2],
                [0, 2, 2, 5, 1, 4],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "spot_id": [f"toy_spot_{i}" for i in range(6)],
            "domain": ["A", "A", "B", "B", "C", "C"],
            "x": [0, 1, 0, 1, 0, 1],
            "y": [0, 0, 1, 1, 2, 2],
        }
    ).set_index("spot_id")
    var = pd.DataFrame(
        {
            "peak_id": ["pA_prom", "pA_body", "pB_prom", "pB_body", "pC_prom", "pC_body"],
            "chr": ["chr1", "chr1", "chr1", "chr1", "chr2", "chr2"],
            "start": [100, 180, 310, 360, 130, 210],
            "end": [150, 220, 335, 410, 150, 250],
        }
    ).set_index("peak_id", drop=False)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["dataset_id"] = "toy_spatial_atac"
    adata.uns["genome_build"] = "toy_genome"
    return adata


def _toy_gene_annotation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_id": ["geneA", "geneB", "geneC"],
            "gene_name": ["GeneA", "GeneB", "GeneC"],
            "gene_type": ["protein_coding", "protein_coding", "protein_coding"],
            "chr": ["chr1", "chr1", "chr2"],
            "start": [90, 290, 100],
            "end": [230, 420, 260],
            "strand": ["+", "+", "-"],
            "promoter_start": [95, 300, 120],
            "promoter_end": [170, 340, 155],
        }
    )


def _write_bed(df: pd.DataFrame, path: Path, *, start_col: str, end_col: str) -> None:
    bed = df[["chr", start_col, end_col, "gene_id", "strand", "gene_name", "gene_type"]].copy()
    bed.insert(4, "score", 0)
    bed.to_csv(path, sep="\t", header=False, index=False)


def generate_toy_inputs(outdir: str | Path) -> dict[str, Path]:
    """Generate a tiny self-contained spatial ATAC example."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    adata = _toy_peak_matrix()
    genes = _toy_gene_annotation()

    peak_h5ad = outdir / "toy_peak_matrix.h5ad"
    gene_tsv = outdir / "toy_gene_annotation.tsv"
    promoter_bed = outdir / "toy_promoters.bed"
    gene_body_bed = outdir / "toy_gene_body.bed"
    config_yaml = outdir / "toy_config.yaml"
    readme = outdir / "README.md"

    adata.write_h5ad(peak_h5ad, compression="gzip")
    genes.to_csv(gene_tsv, sep="\t", index=False)
    _write_bed(genes, promoter_bed, start_col="promoter_start", end_col="promoter_end")
    _write_bed(genes, gene_body_bed, start_col="start", end_col="end")

    config = {
        "name": "toy_spatial_atac",
        "description": "Small public toy example for SpatialGABench smoke tests.",
        "inputs": {
            "peak_matrix_h5ad": "toy_peak_matrix.h5ad",
            "gene_annotation_tsv": "toy_gene_annotation.tsv",
            "promoters_bed": "toy_promoters.bed",
            "gene_body_bed": "toy_gene_body.bed",
        },
        "methods": TOY_METHODS,
        "tasks": ["construction_qc", "basic_true_ga_recovery"],
        "outputs": {"outdir": "results/examples/toy_run"},
        "comparison_scope": "toy_example_only",
        "caveats": [
            "Toy data are for smoke testing only.",
            "No real-data ranking or biological claim should be made from this example.",
        ],
    }
    with config_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    readme.write_text(
        "\n".join(
            [
                "# Toy spatial ATAC example",
                "",
                "This directory contains a tiny synthetic spatial ATAC dataset for CLI and test smoke checks.",
                "It is not a biological benchmark and does not require any external downloads.",
                "",
                "Run:",
                "",
                "```bash",
                "python -m spatialgabench.cli.main run-toy-example --outdir results/examples/toy_run",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "peak_h5ad": peak_h5ad,
        "gene_annotation_tsv": gene_tsv,
        "promoters_bed": promoter_bed,
        "gene_body_bed": gene_body_bed,
        "config_yaml": config_yaml,
        "readme": readme,
    }


def _truth_matrix(adata: ad.AnnData) -> pd.DataFrame:
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    truth = np.column_stack(
        [
            X[:, 0] + 0.5 * X[:, 1],
            X[:, 2] + 0.5 * X[:, 3],
            X[:, 4] + 0.5 * X[:, 5],
        ]
    )
    return pd.DataFrame(truth, index=adata.obs_names, columns=["geneA", "geneB", "geneC"])


def _spearman_flat(X: sparse.spmatrix, truth: pd.DataFrame) -> float:
    dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
    if dense.shape != truth.shape:
        return float("nan")
    result = stats.spearmanr(dense.ravel(), truth.to_numpy(dtype=float).ravel())
    return float(result.statistic) if np.isfinite(result.statistic) else float("nan")


def run_toy_benchmark(outdir: str | Path) -> dict[str, Path]:
    """Run the minimal toy gene activity benchmark."""

    outdir = Path(outdir)
    inputs_dir = outdir / "inputs"
    methods_dir = outdir / "gene_activity"
    tables_dir = outdir / "tables"
    reports_dir = outdir / "reports"
    for path in [inputs_dir, methods_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    paths = generate_toy_inputs(inputs_dir)
    adata = ad.read_h5ad(paths["peak_h5ad"])
    peaks = parse_peak_coordinates(adata)
    promoters = load_promoters_bed(paths["promoters_bed"])
    gene_body = load_bed_intervals(paths["gene_body_bed"], "gene_body")
    gene_metadata = load_gene_metadata(paths["gene_body_bed"])
    truth = _truth_matrix(adata)
    truth_path = tables_dir / "toy_true_gene_activity.tsv"
    truth.to_csv(truth_path, sep="\t", index=True)

    promoter_mapping = build_peak_gene_overlap(peaks, promoters)
    gene_body_mapping = build_gene_body_mapping(peaks, gene_body, gene_metadata)
    promoter_gene_body_mapping = build_promoter_gene_body_mapping(peaks, promoters, gene_body, gene_metadata)
    random_mapping = build_random_control_mapping(
        n_peaks=peaks.shape[0],
        gene_metadata=gene_metadata,
        promoter_peak_counts=promoter_mapping.gene_assigned_peak_counts,
        seed=25,
        peaks=peaks,
    )

    mappings = {
        "promoter_only_2kb_pc": promoter_mapping,
        "gene_body_pc": gene_body_mapping,
        "promoter_plus_gene_body_2kb_pc": promoter_gene_body_mapping,
        "random_peak_to_gene_negative_control_pc": random_mapping,
    }
    rows: list[dict[str, Any]] = []
    for method_id, mapping in mappings.items():
        ga = compute_gene_activity(adata.X, mapping.peak_gene_matrix)
        output_path = methods_dir / f"{method_id}.h5ad"
        write_gene_activity_h5ad(
            input_adata=adata,
            gene_activity_X=ga,
            genes=mapping.genes,
            output_path=output_path,
            dataset_id="toy_spatial_atac",
            method_id=method_id,
            genome_build="toy_genome",
            annotation_source="toy_gene_annotation",
            input_peak_h5ad=str(paths["peak_h5ad"].relative_to(outdir)),
            method_notes="Toy smoke-test output; not a biological benchmark.",
            extra_uns={"stage25a_role": "toy_example"},
        )
        rows.append(
            {
                "method_id": method_id,
                "status": "pass",
                "n_spots": int(ga.shape[0]),
                "n_genes": int(ga.shape[1]),
                "nnz": int(ga.nnz),
                "true_ga_spearman_flat": _spearman_flat(ga, truth),
                "output_file": str(output_path.relative_to(outdir)),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = tables_dir / "toy_gene_activity_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    report_path = reports_dir / "toy_summary_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Toy SpatialGABench run",
                "",
                "This is a tiny smoke-test benchmark. It does not use external data and does not support biological ranking claims.",
                "",
                "Methods run:",
                *[f"- {row['method_id']}: Spearman={row['true_ga_spearman_flat']:.3f}" for row in rows],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "outdir": outdir,
        "summary": summary_path,
        "truth": truth_path,
        "report": report_path,
    }
