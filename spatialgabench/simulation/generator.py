"""Identifiable mechanistic and empirical semi-synthetic generators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats

from .calibration import load_empirical_background, sha256_file
from .config import SIMULATION_SCHEMA_VERSION, validate_simulation_config
from .truth import write_h5ad


GENERATOR_VERSION = "2.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RNG_STREAMS = (
    "coordinates",
    "genome",
    "latent",
    "background",
    "library",
    "atac_counts",
    "zero_inflation",
    "rna",
    "empirical_sampling",
)


def _spawn_rngs(seed: int) -> dict[str, np.random.Generator]:
    streams = np.random.SeedSequence(int(seed)).spawn(len(RNG_STREAMS))
    return {name: np.random.default_rng(stream) for name, stream in zip(RNG_STREAMS, streams)}


def _grid_coordinates(n_cells: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.ceil(np.sqrt(n_cells)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    coordinates = np.column_stack([xs.ravel(), ys.ravel()])[:n_cells].astype(np.float64)
    coordinates /= max(1.0, float(side - 1))
    coordinates += rng.normal(0.0, 0.006, size=coordinates.shape)
    return np.clip(coordinates, 0.0, 1.0).astype(np.float32)


def _balanced_labels(score: np.ndarray, n_domains: int) -> np.ndarray:
    order = np.argsort(np.asarray(score), kind="mergesort")
    labels = np.empty(order.size, dtype=np.int32)
    labels[order] = np.minimum(n_domains - 1, np.arange(order.size) * n_domains // order.size)
    return labels


def _domain_labels(coordinates: np.ndarray, geometry: str, n_domains: int) -> np.ndarray:
    x, y = coordinates[:, 0], coordinates[:, 1]
    if geometry == "gradient":
        labels = _balanced_labels(x + y, n_domains)
    elif geometry == "nested":
        labels = _balanced_labels(np.sqrt(np.square(x - 0.5) + np.square(y - 0.5)), n_domains)
    elif n_domains == 4:
        labels = (x > np.median(x)).astype(np.int32) + 2 * (y > np.median(y)).astype(np.int32)
        if np.unique(labels).size != n_domains:
            labels = _balanced_labels(np.arctan2(y - 0.5, x - 0.5), n_domains)
    else:
        labels = _balanced_labels(np.arctan2(y - 0.5, x - 0.5), n_domains)
    return labels.astype(np.int32)


def _make_observations(config: dict[str, Any], coordinates: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = _domain_labels(coordinates, str(config["domain_geometry"]), int(config["n_domains"]))
    cell_ids = [f"cell_{index:05d}" for index in range(int(config["n_cells"]))]
    input_obs = pd.DataFrame(
        {"x": coordinates[:, 0], "y": coordinates[:, 1], "input_role": "accessibility_only"}, index=cell_ids
    )
    input_obs.index.name = "cell_id"
    truth_labels = pd.DataFrame(
        {
            "true_domain": [f"domain_{value}" for value in labels],
            "true_domain_index": labels,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        },
        index=cell_ids,
    )
    truth_labels.index.name = "cell_id"
    return input_obs, truth_labels


def _distal_offsets(n_distal: int) -> list[int]:
    offsets = []
    for index in range(n_distal):
        magnitude = 50000 * (1 + index // 2)
        offsets.append(-magnitude if index % 2 == 0 else magnitude)
    return offsets


def _make_genome(
    config: dict[str, Any], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    n_genes = int(config["n_genes"])
    n_peaks = int(config["n_peaks"])
    n_distal = int(config["distal_peaks_per_gene"])
    distal_fraction = float(config["factors"]["distal_fraction"])
    decoy_fraction = float(config["factors"]["decoy_fraction"])
    offsets = _distal_offsets(n_distal)
    max_offset = max(map(abs, offsets))
    gene_spacing = 2 * max_offset + 250000
    tss_positions = 2 * max_offset + 200000 + np.arange(n_genes) * gene_spacing

    gene_rows = []
    for gene_index, tss in enumerate(tss_positions):
        strand = "+" if gene_index % 2 == 0 else "-"
        length = 3000 + (gene_index % 5) * 750
        start = int(tss if strand == "+" else tss - length)
        end = int(tss + length if strand == "+" else tss)
        gene_rows.append(
            {
                "gene_id": f"sim_gene_{gene_index:04d}",
                "gene_name": f"SimGene{gene_index:04d}",
                "gene_type": "protein_coding",
                "chr": "chrSim",
                "start": start,
                "end": end,
                "strand": strand,
                "tss": int(tss),
                "is_tf": bool(config["tf_program_enabled"] and gene_index < min(6, n_genes)),
            }
        )
    genes = pd.DataFrame(gene_rows)

    peak_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    def add_peak(
        peak_id: str,
        start: int,
        end: int,
        peak_type: str,
        *,
        linked_gene_index: int = -1,
        signal_source_gene_index: int = -1,
        auxiliary_index: int = -1,
        is_decoy: bool = False,
    ) -> None:
        linked_gene_id = "" if linked_gene_index < 0 else str(genes.loc[linked_gene_index, "gene_id"])
        source_gene_id = "" if signal_source_gene_index < 0 else str(genes.loc[signal_source_gene_index, "gene_id"])
        peak_rows.append(
            {
                "peak_id": peak_id,
                "chr": "chrSim",
                "start": int(start),
                "end": int(end),
                "peak_type": peak_type,
                "linked_gene_index": int(linked_gene_index),
                "linked_gene_id": linked_gene_id,
                "signal_source_gene_index": int(signal_source_gene_index),
                "signal_source_gene_id": source_gene_id,
                "auxiliary_index": int(auxiliary_index),
                "is_decoy": bool(is_decoy),
                "is_causal": peak_type in {"promoter", "distal_enhancer"},
            }
        )

    for gene_index, gene in genes.iterrows():
        tss = int(gene["tss"])
        promoter_id = f"peak_g{gene_index:04d}_promoter"
        add_peak(promoter_id, tss - 250, tss + 250, "promoter", linked_gene_index=gene_index)
        link_rows.append(
            {
                "peak_id": promoter_id,
                "gene_index": gene_index,
                "gene_id": gene["gene_id"],
                "gene_name": gene["gene_name"],
                "link_type": "promoter",
                "true_weight": 1.0 - distal_fraction,
                "distance_to_tss": 0.0,
                "is_active_link": (1.0 - distal_fraction) > 0,
                "tf_id": "",
            }
        )
        body_start = int(gene["start"] + max(1, (int(gene["end"]) - int(gene["start"])) // 3))
        add_peak(
            f"peak_g{gene_index:04d}_body",
            body_start,
            body_start + 500,
            "gene_body_proxy",
            linked_gene_index=gene_index,
        )
        for distal_index, offset in enumerate(offsets):
            center = tss + offset
            peak_id = f"peak_g{gene_index:04d}_distal{distal_index:02d}"
            add_peak(peak_id, center - 250, center + 250, "distal_enhancer", linked_gene_index=gene_index)
            tf_id = ""
            if bool(config["tf_program_enabled"]) and rng.random() < 0.35:
                tf_id = str(genes.loc[int(rng.integers(0, min(6, n_genes))), "gene_id"])
            link_rows.append(
                {
                    "peak_id": peak_id,
                    "gene_index": gene_index,
                    "gene_id": gene["gene_id"],
                    "gene_name": gene["gene_name"],
                    "link_type": "distal",
                    "true_weight": distal_fraction / n_distal,
                    "distance_to_tss": float(abs(offset)),
                    "is_active_link": distal_fraction > 0,
                    "tf_id": tf_id,
                }
            )

    n_auxiliary = n_peaks - len(peak_rows)
    decoy_order = rng.permutation(n_auxiliary)
    n_decoys = int(round(decoy_fraction * n_auxiliary))
    decoy_indices = set(map(int, decoy_order[:n_decoys]))
    per_target_slots = np.zeros(n_genes, dtype=int)
    background_start = int(tss_positions[-1] + max_offset + 200000)
    source_gene_indices = rng.integers(0, n_genes, size=n_auxiliary)
    target_gene_indices = rng.integers(0, n_genes, size=n_auxiliary)
    target_gene_indices = np.where(target_gene_indices == source_gene_indices, (target_gene_indices + 1) % n_genes, target_gene_indices)
    for auxiliary_index in range(n_auxiliary):
        source_gene = int(source_gene_indices[auxiliary_index])
        if auxiliary_index in decoy_indices:
            target_gene = int(target_gene_indices[auxiliary_index])
            slot = int(per_target_slots[target_gene])
            per_target_slots[target_gene] += 1
            center = int(genes.loc[target_gene, "tss"]) + 15000 + slot * 1000
            add_peak(
                f"peak_aux_{auxiliary_index:05d}",
                center - 200,
                center + 200,
                "proximity_decoy",
                signal_source_gene_index=source_gene,
                auxiliary_index=auxiliary_index,
                is_decoy=True,
            )
        else:
            center = background_start + auxiliary_index * 1000
            add_peak(
                f"peak_aux_{auxiliary_index:05d}",
                center - 200,
                center + 200,
                "background",
                auxiliary_index=auxiliary_index,
            )

    peaks = pd.DataFrame(peak_rows)
    peaks = peaks.iloc[rng.permutation(peaks.shape[0])].reset_index(drop=True)
    peaks["peak_index"] = np.arange(peaks.shape[0], dtype=np.int32)
    peak_index = peaks.set_index("peak_id")["peak_index"].to_dict()
    links = pd.DataFrame(link_rows)
    links.insert(0, "peak_index", links["peak_id"].map(peak_index).astype(np.int32))
    links = links.sort_values(["gene_index", "link_type", "distance_to_tss", "peak_id"], kind="mergesort").reset_index(drop=True)
    chromosome_size = int(max(peaks["end"].max(), genes["end"].max()) + 10000)
    return genes, peaks, links, chromosome_size


def _make_latent_truth(
    config: dict[str, Any], truth_labels: pd.DataFrame, genes: pd.DataFrame, rng: np.random.Generator
) -> dict[str, Any]:
    n_cells = truth_labels.shape[0]
    n_genes = genes.shape[0]
    n_domains = int(config["n_domains"])
    labels = truth_labels["true_domain_index"].to_numpy(dtype=int)
    domain_effect = float(config["factors"]["domain_log_effect"])
    distal_fraction = float(config["factors"]["distal_fraction"])
    marker_domain = np.arange(n_genes, dtype=int) % n_domains
    pattern = np.where(
        labels[:, None] == marker_domain[None, :],
        1.0,
        -1.0 / max(1, n_domains - 1),
    )
    baseline = rng.normal(0.0, 0.35, size=n_genes)
    cell_gene_noise = rng.normal(0.0, 0.12, size=(n_cells, n_genes))
    log_activity = baseline[None, :] + cell_gene_noise + domain_effect * pattern
    true_ga = np.exp(log_activity)
    true_ga /= np.maximum(true_ga.mean(axis=1, keepdims=True), 1e-12)
    allocation_signal = rng.normal(size=(n_cells, n_genes))
    allocation_signal -= allocation_signal.mean(axis=0, keepdims=True)
    allocation_scale = np.max(np.abs(allocation_signal), axis=0, keepdims=True)
    allocation_scale[allocation_scale == 0] = 1.0
    allocation_signal /= allocation_scale
    allocation_amplitude = float(config.get("component_allocation_amplitude_fraction", 0.5)) * min(
        distal_fraction, 1.0 - distal_fraction
    )
    distal_allocation = distal_fraction + allocation_amplitude * allocation_signal
    distal_allocation = np.clip(distal_allocation, 0.0, 1.0)
    distal = distal_allocation * true_ga
    promoter = true_ga - distal
    domain_program = np.maximum(np.expm1(domain_effect * np.maximum(pattern, 0.0)), 0.0)
    tf_n = int(genes["is_tf"].sum())
    tf_activity = np.log1p(true_ga[:, :tf_n]) if tf_n else np.zeros((n_cells, 0), dtype=np.float64)
    return {
        "true_ga": true_ga.astype(np.float32),
        "promoter": promoter.astype(np.float32),
        "distal": distal.astype(np.float32),
        "domain_program": domain_program.astype(np.float32),
        "tf_activity": tf_activity.astype(np.float32),
        "marker_domain": marker_domain,
        "distal_allocation": distal_allocation.astype(np.float32),
    }


def _mechanistic_background(
    n_cells: int,
    n_auxiliary: int,
    coordinates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    base = rng.gamma(shape=1.5, scale=1.0, size=n_auxiliary)
    x_weight = rng.normal(0.0, 0.22, size=n_auxiliary)
    y_weight = rng.normal(0.0, 0.22, size=n_auxiliary)
    cell_scale = rng.lognormal(mean=0.0, sigma=0.18, size=n_cells)
    log_pattern = coordinates[:, [0]] * x_weight[None, :] + coordinates[:, [1]] * y_weight[None, :]
    return np.maximum(cell_scale[:, None] * base[None, :] * np.exp(log_pattern), 1e-10)


def _peak_probabilities(
    config: dict[str, Any],
    peaks: pd.DataFrame,
    truth: dict[str, Any],
    coordinates: np.ndarray,
    rng: np.random.Generator,
    empirical_background: sparse.csr_matrix | None,
) -> np.ndarray:
    n_cells, n_peaks = int(config["n_cells"]), int(config["n_peaks"])
    n_auxiliary = int((peaks["auxiliary_index"] >= 0).sum())
    causal = np.zeros((n_cells, n_peaks), dtype=np.float64)
    for peak_index, peak in peaks.iterrows():
        gene_index = int(peak["linked_gene_index"])
        if peak["peak_type"] == "promoter":
            causal[:, peak_index] = truth["promoter"][:, gene_index]
        elif peak["peak_type"] == "distal_enhancer":
            causal[:, peak_index] = truth["distal"][:, gene_index] / int(config["distal_peaks_per_gene"])
        elif peak["peak_type"] == "gene_body_proxy":
            causal[:, peak_index] = 0.2 * truth["true_ga"][:, gene_index]
    causal /= np.maximum(causal.sum(axis=1, keepdims=True), 1e-12)

    if empirical_background is None:
        auxiliary = _mechanistic_background(n_cells, n_auxiliary, coordinates, rng)
    else:
        if empirical_background.shape != (n_cells, n_auxiliary):
            raise ValueError("Empirical background shape does not match auxiliary peak budget")
        auxiliary = empirical_background.toarray().astype(np.float64) + 0.05

    for _, peak in peaks.loc[peaks["is_decoy"]].iterrows():
        auxiliary_index = int(peak["auxiliary_index"])
        source_gene = int(peak["signal_source_gene_index"])
        replacement = np.asarray(truth["true_ga"][:, source_gene], dtype=np.float64)
        replacement *= np.mean(auxiliary[:, auxiliary_index]) / max(float(np.mean(replacement)), 1e-12)
        auxiliary[:, auxiliary_index] = replacement
    auxiliary /= np.maximum(auxiliary.sum(axis=1, keepdims=True), 1e-12)

    auxiliary_full = np.zeros((n_cells, n_peaks), dtype=np.float64)
    for peak_index, peak in peaks.loc[peaks["auxiliary_index"] >= 0].iterrows():
        auxiliary_full[:, peak_index] = auxiliary[:, int(peak["auxiliary_index"])]
    causal_fraction = float(config["causal_signal_fraction"])
    probabilities = causal_fraction * causal + (1.0 - causal_fraction) * auxiliary_full
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return probabilities


def _sample_atac(
    config: dict[str, Any],
    probabilities: np.ndarray,
    rngs: dict[str, np.random.Generator],
    empirical_library_sizes: np.ndarray | None,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, float]]:
    n_cells = probabilities.shape[0]
    if empirical_library_sizes is None:
        multipliers = rngs["library"].lognormal(mean=0.0, sigma=0.25, size=n_cells)
        multipliers /= np.median(multipliers)
        baseline_library = float(config["base_library_size"]) * multipliers
        library_source = "mechanistic_lognormal"
    else:
        baseline_library = np.asarray(empirical_library_sizes, dtype=np.float64)
        positive = baseline_library[baseline_library > 0]
        if positive.size == 0:
            raise ValueError("Empirical calibration produced no positive library sizes")
        lower, upper = np.quantile(positive, [0.05, 0.95])
        baseline_library = np.clip(baseline_library, max(1.0, lower), max(lower, upper))
        library_source = "development_accessibility"
    target_library = baseline_library * float(config["factors"]["depth_multiplier"])
    expected = target_library[:, None] * probabilities
    counts = rngs["atac_counts"].poisson(expected).astype(np.float32)
    zero_rate = float(config["factors"]["zero_inflation_rate"])
    zero_mask = rngs["zero_inflation"].random(size=counts.shape) < zero_rate
    counts[zero_mask] = 0.0
    matrix = sparse.csr_matrix(counts)
    matrix.eliminate_zeros()
    qc = {
        "library_source": library_source,
        "target_library_median": float(np.median(target_library)),
        "observed_library_median": float(np.median(np.asarray(matrix.sum(axis=1)).ravel())),
        "observed_sparsity": float(1.0 - matrix.nnz / np.prod(matrix.shape)),
        "observed_nnz": int(matrix.nnz),
    }
    return matrix, expected.astype(np.float32), qc


def _make_rna(
    config: dict[str, Any], true_ga: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, float]]:
    log_ga = np.log1p(np.asarray(true_ga, dtype=np.float64))
    centered_ga = log_ga - log_ga.mean(axis=0, keepdims=True)
    ga_scale = centered_ga.std(axis=0, keepdims=True)
    ga_scale[ga_scale == 0] = 1.0
    standardized_ga = centered_ga / ga_scale
    noise = rng.normal(size=standardized_ga.shape)
    for gene_index in range(noise.shape[1]):
        z = standardized_ga[:, gene_index]
        n = noise[:, gene_index] - noise[:, gene_index].mean()
        denominator = float(np.dot(z, z))
        if denominator > 0:
            n -= float(np.dot(n, z)) / denominator * z
        scale = float(np.std(n))
        noise[:, gene_index] = n / (scale if scale > 0 else 1.0)
    rho = float(config["factors"]["rna_latent_correlation"])
    rna_latent = rho * standardized_ga + np.sqrt(max(0.0, 1.0 - rho**2)) * noise
    expected = np.exp(0.65 * rna_latent)
    expected *= float(config["rna_mean_count"]) / np.maximum(expected.mean(axis=0, keepdims=True), 1e-12)
    counts = rng.poisson(expected).astype(np.float32)
    latent_correlation = float(stats.pearsonr(standardized_ga.ravel(), rna_latent.ravel()).statistic)
    centered_observed = np.log1p(counts) - np.log1p(counts).mean(axis=0, keepdims=True)
    observed_correlation = float(stats.pearsonr(centered_ga.ravel(), centered_observed.ravel()).statistic)
    return counts, {
        "target_latent_correlation": rho,
        "realized_latent_correlation": latent_correlation,
        "realized_observed_log_correlation": observed_correlation,
        "rna_mean_count": float(counts.mean()),
    }


def _write_reference_files(outdir: Path, genes: pd.DataFrame, chromosome_size: int) -> None:
    reference_dir = outdir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    body = genes[["chr", "start", "end", "gene_id", "strand", "gene_name", "gene_type"]].copy()
    body.insert(4, "score", 0)
    body.to_csv(reference_dir / "gene_body.bed", sep="\t", header=False, index=False)
    promoter_rows = []
    for gene in genes.itertuples(index=False):
        if gene.strand == "+":
            start, end = max(0, int(gene.tss) - 2000), int(gene.tss) + 500
        else:
            start, end = max(0, int(gene.tss) - 500), int(gene.tss) + 2000
        promoter_rows.append([gene.chr, start, end, gene.gene_id, 0, gene.strand, gene.gene_name, gene.gene_type])
    pd.DataFrame(promoter_rows).to_csv(reference_dir / "promoters_2kb.bed", sep="\t", header=False, index=False)
    pd.DataFrame([["chrSim", chromosome_size]]).to_csv(
        reference_dir / "chrom_sizes.tsv", sep="\t", header=False, index=False
    )
    with (reference_dir / "synthetic_annotation.gtf").open("w", encoding="utf-8") as handle:
        for gene in genes.itertuples(index=False):
            start_1 = int(gene.start) + 1
            attributes = (
                f'gene_id "{gene.gene_id}"; gene_name "{gene.gene_name}"; '
                'gene_type "protein_coding"; gene_biotype "protein_coding";'
            )
            transcript = attributes + f' transcript_id "{gene.gene_id}.t1";'
            handle.write(
                "\t".join(
                    [gene.chr, "SpatialGABench", "gene", str(start_1), str(gene.end), ".", gene.strand, ".", attributes]
                )
                + "\n"
            )
            handle.write(
                "\t".join(
                    [gene.chr, "SpatialGABench", "transcript", str(start_1), str(gene.end), ".", gene.strand, ".", transcript]
                )
                + "\n"
            )
            handle.write(
                "\t".join(
                    [gene.chr, "SpatialGABench", "exon", str(start_1), str(gene.end), ".", gene.strand, ".", transcript]
                )
                + "\n"
            )


def _tf_links(genes: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf_index, tf in genes.loc[genes["is_tf"]].reset_index(drop=True).iterrows():
        hits = links.loc[links["tf_id"] == tf["gene_id"]].head(8)
        for link in hits.itertuples(index=False):
            rows.append(
                {
                    "tf_gene_id": tf["gene_id"],
                    "tf_gene_name": tf["gene_name"],
                    "motif_id": f"motif_{tf_index:02d}",
                    "target_gene_id": link.gene_id,
                    "true_association": True,
                }
            )
    return pd.DataFrame(rows, columns=["tf_gene_id", "tf_gene_name", "motif_id", "target_gene_id", "true_association"])


def _write_outputs(
    config: dict[str, Any],
    outdir: str | Path,
    *,
    input_obs: pd.DataFrame,
    truth_labels: pd.DataFrame,
    coordinates: np.ndarray,
    genes: pd.DataFrame,
    peaks: pd.DataFrame,
    links: pd.DataFrame,
    chromosome_size: int,
    truth: dict[str, Any],
    atac: sparse.csr_matrix,
    expected_atac: np.ndarray,
    rna_counts: np.ndarray,
    sampling_qc: dict[str, float],
    rna_qc: dict[str, float],
    calibration_profile: dict[str, Any] | None,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    var_peak = peaks.set_index("peak_id", drop=False)
    var_gene = genes.set_index("gene_id", drop=False)
    input_uns = {
        "simulation_id": str(config["simulation_id"]),
        "simulation_type": str(config["simulation_type"]),
        "generator_version": GENERATOR_VERSION,
        "genome_build": "synthetic_chrSim_v2",
        "input_role": "benchmark_accessibility_input",
        "evaluation_fields_present_in_obs": False,
        "construction_evaluation_inputs_used": False,
    }
    truth_uns = {
        **input_uns,
        "object_role": "evaluation_truth_not_method_input",
        "factor_values_json": json.dumps(config["factors"], sort_keys=True),
    }
    outputs = {
        "simulated_atac_peak_matrix": outdir / "simulated_atac_peak_matrix.h5ad",
        "expected_peak_accessibility": outdir / "expected_peak_accessibility.h5ad",
        "true_gene_activity": outdir / "true_gene_activity.h5ad",
        "true_promoter_component": outdir / "true_promoter_component.h5ad",
        "true_distal_component": outdir / "true_distal_component.h5ad",
        "true_domain_program_component": outdir / "true_domain_program_component.h5ad",
        "simulated_rna": outdir / "simulated_rna.h5ad",
    }
    write_h5ad(atac, input_obs, var_peak, outputs["simulated_atac_peak_matrix"], uns=input_uns, spatial=coordinates)
    write_h5ad(
        expected_atac,
        input_obs,
        var_peak,
        outputs["expected_peak_accessibility"],
        uns={**truth_uns, "expectation_stage": "before_zero_inflation"},
        spatial=coordinates,
    )
    write_h5ad(truth["true_ga"], input_obs, var_gene, outputs["true_gene_activity"], uns=truth_uns, spatial=coordinates)
    write_h5ad(truth["promoter"], input_obs, var_gene, outputs["true_promoter_component"], uns=truth_uns, spatial=coordinates)
    write_h5ad(truth["distal"], input_obs, var_gene, outputs["true_distal_component"], uns=truth_uns, spatial=coordinates)
    write_h5ad(
        truth["domain_program"],
        input_obs,
        var_gene,
        outputs["true_domain_program_component"],
        uns=truth_uns,
        spatial=coordinates,
    )
    write_h5ad(
        rna_counts,
        input_obs,
        var_gene,
        outputs["simulated_rna"],
        uns={**truth_uns, "rna_role": "evaluation_only_never_construction_tuning"},
        spatial=coordinates,
    )

    if truth["tf_activity"].shape[1]:
        tf_var = genes.loc[genes["is_tf"], ["gene_id", "gene_name"]].set_index("gene_id", drop=False)
        write_h5ad(truth["tf_activity"], input_obs, tf_var, outdir / "true_tf_activity.h5ad", uns=truth_uns, spatial=coordinates)
        motif_ids = [f"motif_{index:02d}" for index in range(truth["tf_activity"].shape[1])]
        motif_var = pd.DataFrame({"motif_id": motif_ids}).set_index("motif_id", drop=False)
        write_h5ad(
            truth["tf_activity"],
            input_obs,
            motif_var,
            outdir / "true_motif_accessibility.h5ad",
            uns=truth_uns,
            spatial=coordinates,
        )
        _tf_links(genes, links).to_csv(outdir / "true_tf_motif_links.tsv", sep="\t", index=False)

    links.to_csv(outdir / "true_peak_gene_links.tsv", sep="\t", index=False)
    truth_labels.to_csv(outdir / "true_domain_labels.tsv", sep="\t")
    pd.DataFrame([config["factors"]]).assign(
        simulation_id=config["simulation_id"], simulation_type=config["simulation_type"], random_seed=config["random_seed"]
    ).to_csv(outdir / "simulation_factor_values.tsv", sep="\t", index=False)
    pd.DataFrame([{**sampling_qc, "simulation_id": config["simulation_id"]}]).to_csv(
        outdir / "atac_sampling_qc.tsv", sep="\t", index=False
    )
    pd.DataFrame([{**rna_qc, "simulation_id": config["simulation_id"]}]).to_csv(
        outdir / "rna_calibration_qc.tsv", sep="\t", index=False
    )
    _write_reference_files(outdir, genes, chromosome_size)

    config_out = dict(config)
    config_out["output_dir"] = str(outdir)
    (outdir / "simulation_config.yaml").write_text(yaml.safe_dump(config_out, sort_keys=False), encoding="utf-8")
    if calibration_profile is not None:
        (outdir / "empirical_calibration_profile.json").write_text(
            json.dumps(calibration_profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    artifact_names = [path.name for path in outputs.values()] + [
        "true_peak_gene_links.tsv",
        "true_domain_labels.tsv",
        "simulation_factor_values.tsv",
        "atac_sampling_qc.tsv",
        "rna_calibration_qc.tsv",
        "simulation_config.yaml",
    ]
    if calibration_profile is not None:
        artifact_names.append("empirical_calibration_profile.json")
    provenance = {
        "schema_version": 1,
        "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "simulation_id": str(config["simulation_id"]),
        "simulation_type": str(config["simulation_type"]),
        "replicate_id": str(config["replicate_id"]),
        "random_seed": int(config["random_seed"]),
        "rng_stream_policy": "numpy_SeedSequence_spawn_by_fixed_named_stream_order",
        "rng_streams": list(RNG_STREAMS),
        "factor_values": {key: float(value) for key, value in config["factors"].items()},
        "construction_input_classes_used": ["accessibility", "gene_annotation", "spatial_coordinates"]
        if calibration_profile is not None
        else ["synthetic_randomness", "gene_annotation", "spatial_coordinates"],
        "construction_evaluation_inputs_used": False,
        "forbidden_inputs_used": [],
        "holdout_accessed": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "truth_exposed_in_accessibility_obs": False,
        "artifacts": {name: sha256_file(outdir / name) for name in artifact_names},
    }
    if calibration_profile is not None:
        provenance["calibration"] = calibration_profile
    (outdir / "simulation_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from .validation import validate_simulation_outputs

    validation = validate_simulation_outputs(outdir)
    if not (validation["validation_status"] == "pass").all():
        failures = validation.loc[validation["validation_status"] != "pass", "object_name"].tolist()
        raise ValueError(f"Generated simulation failed output validation: {failures}")
    summary = pd.DataFrame(
        [
            {
                "simulation_id": config["simulation_id"],
                "simulation_type": config["simulation_type"],
                "n_cells": atac.shape[0],
                "n_peaks": atac.shape[1],
                "n_genes": genes.shape[0],
                "n_true_links": links.shape[0],
                "n_decoy_peaks": int(peaks["is_decoy"].sum()),
                "atac_nnz": int(atac.nnz),
                "atac_total": float(atac.sum()),
                "random_seed": int(config["random_seed"]),
                "holdout_accessed": False,
                "full_benchmark_run": False,
                "status": "pass",
            }
        ]
    )
    summary_path = outdir / "simulation_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    return outputs | {
        "outdir": outdir,
        "validation": outdir / "validation_summary.tsv",
        "summary": summary_path,
        "provenance": outdir / "simulation_provenance.json",
    }


def _generate(config: dict[str, Any], outdir: str | Path, project_root: str | Path) -> dict[str, Path]:
    validation = validate_simulation_config(config)
    rngs = _spawn_rngs(int(config["random_seed"]))
    empirical = None
    if config["simulation_type"] == "empirical_semisynthetic":
        empirical = load_empirical_background(
            config["calibration"],
            project_root=project_root,
            n_cells=int(config["n_cells"]),
            n_features=int(validation["n_auxiliary_peaks"]),
            rng=rngs["empirical_sampling"],
        )
        coordinates = empirical["coordinates"]
    else:
        coordinates = _grid_coordinates(int(config["n_cells"]), rngs["coordinates"])
    input_obs, truth_labels = _make_observations(config, coordinates)
    genes, peaks, links, chromosome_size = _make_genome(config, rngs["genome"])
    truth = _make_latent_truth(config, truth_labels, genes, rngs["latent"])
    probabilities = _peak_probabilities(
        config,
        peaks,
        truth,
        coordinates,
        rngs["background"],
        None if empirical is None else empirical["background"],
    )
    atac, expected_atac, sampling_qc = _sample_atac(
        config,
        probabilities,
        rngs,
        None if empirical is None else empirical["library_sizes"],
    )
    rna_counts, rna_qc = _make_rna(config, truth["true_ga"], rngs["rna"])
    return _write_outputs(
        config,
        outdir,
        input_obs=input_obs,
        truth_labels=truth_labels,
        coordinates=coordinates,
        genes=genes,
        peaks=peaks,
        links=links,
        chromosome_size=chromosome_size,
        truth=truth,
        atac=atac,
        expected_atac=expected_atac,
        rna_counts=rna_counts,
        sampling_qc=sampling_qc,
        rna_qc=rna_qc,
        calibration_profile=None if empirical is None else empirical["profile"],
    )


def generate_mechanistic_simulation(config: dict[str, Any], outdir: str | Path) -> dict[str, Path]:
    if config.get("simulation_type") != "mechanistic":
        raise ValueError("generate_mechanistic_simulation requires simulation_type=mechanistic")
    return _generate(dict(config), outdir, PROJECT_ROOT)


def generate_empirical_semisynthetic_simulation(
    config: dict[str, Any], outdir: str | Path, *, project_root: str | Path = PROJECT_ROOT
) -> dict[str, Path]:
    if config.get("simulation_type") != "empirical_semisynthetic":
        raise ValueError("generate_empirical_semisynthetic_simulation requires simulation_type=empirical_semisynthetic")
    return _generate(dict(config), outdir, project_root)
