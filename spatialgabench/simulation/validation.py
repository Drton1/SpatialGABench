"""Strict validation and compact summaries for active simulations."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from .calibration import sha256_file
from .config import FACTOR_RANGES, SIMULATION_SCHEMA_VERSION
from .truth import truth_contract_table


FORBIDDEN_INPUT_OBS_TOKENS = ("truth", "true_", "label", "cluster", "pathology", "disease", "rna", "marker")


def _matrix(obj: ad.AnnData) -> sparse.csr_matrix:
    return obj.X.tocsr() if sparse.issparse(obj.X) else sparse.csr_matrix(obj.X)


def _matrix_is_valid(obj: ad.AnnData, *, require_nonzero: bool) -> bool:
    matrix = _matrix(obj)
    return bool(
        matrix.shape[0] > 0
        and matrix.shape[1] > 0
        and (matrix.nnz > 0 or not require_nonzero)
        and np.all(np.isfinite(matrix.data))
        and np.all(matrix.data >= 0)
        and not np.any(matrix.data == 0)
    )


def _row(name: str, passed: bool, detail: str, *, required: bool = True, size_bytes: int = 0, shape: str = "") -> dict:
    return {
        "object_name": name,
        "required": required,
        "exists": passed if name.startswith("invariant:") else size_bytes > 0,
        "size_bytes": int(size_bytes),
        "shape": shape,
        "validation_status": "pass" if passed else "fail",
        "detail": detail,
    }


def validate_simulation_outputs(simulation_dir: str | Path) -> pd.DataFrame:
    """Validate files, matrix contracts, leakage boundaries, and truth invariants."""

    simulation_dir = Path(simulation_dir)
    rows: list[dict] = []
    objects: dict[str, ad.AnnData] = {}
    contract = truth_contract_table()
    for record in contract.to_dict("records"):
        path = simulation_dir / record["object_name"]
        required = bool(record["required"])
        exists = path.exists() and path.stat().st_size > 0
        passed = exists or not required
        shape = ""
        detail = "present" if exists else ("optional_absent" if not required else "missing_required")
        if exists and path.suffix == ".h5ad":
            obj = ad.read_h5ad(path)
            objects[path.name] = obj
            shape = f"{obj.n_obs}x{obj.n_vars}"
            require_nonzero = path.name in {
                "simulated_atac_peak_matrix.h5ad",
                "expected_peak_accessibility.h5ad",
                "true_gene_activity.h5ad",
                "simulated_rna.h5ad",
            }
            passed = _matrix_is_valid(obj, require_nonzero=require_nonzero)
            detail = "finite_nonnegative_sparse_no_stored_zeros" if passed else "invalid_matrix_values_or_shape"
        rows.append(
            _row(
                record["object_name"],
                passed,
                detail,
                required=required,
                size_bytes=path.stat().st_size if exists else 0,
                shape=shape,
            )
        )

    required_h5ad = [
        "simulated_atac_peak_matrix.h5ad",
        "expected_peak_accessibility.h5ad",
        "true_gene_activity.h5ad",
        "true_promoter_component.h5ad",
        "true_distal_component.h5ad",
        "true_domain_program_component.h5ad",
        "simulated_rna.h5ad",
    ]
    if all(name in objects for name in required_h5ad):
        atac = objects["simulated_atac_peak_matrix.h5ad"]
        aligned = all(objects[name].obs_names.equals(atac.obs_names) for name in required_h5ad[1:])
        rows.append(_row("invariant:obs_alignment", aligned, "all required h5ad obs_names match ATAC"))
        peak_aligned = objects["expected_peak_accessibility.h5ad"].var_names.equals(atac.var_names)
        rows.append(_row("invariant:peak_alignment", peak_aligned, "expected accessibility var_names match ATAC"))
        gene_names = objects["true_gene_activity.h5ad"].var_names
        gene_aligned = all(objects[name].var_names.equals(gene_names) for name in required_h5ad[3:7])
        rows.append(_row("invariant:gene_alignment", gene_aligned, "all gene-level var_names match"))

        forbidden_columns = [
            column
            for column in atac.obs.columns
            if any(token in str(column).lower() for token in FORBIDDEN_INPUT_OBS_TOKENS)
        ]
        no_truth_obs = not forbidden_columns and not any(
            key.lower().startswith("truth") or "label" in key.lower() for key in atac.uns.keys()
        )
        rows.append(
            _row(
                "invariant:truth_absent_from_atac_input",
                no_truth_obs,
                f"forbidden_obs_columns={forbidden_columns}",
            )
        )
        spatial_valid = "spatial" in atac.obsm and atac.obsm["spatial"].shape == (atac.n_obs, 2)
        rows.append(_row("invariant:spatial_coordinates", spatial_valid, "ATAC obsm/spatial is n_obs x 2"))
        integer_counts = np.allclose(_matrix(atac).data, np.round(_matrix(atac).data), atol=1e-6)
        rows.append(_row("invariant:atac_integer_counts", bool(integer_counts), "ATAC nonzero values are integer-like"))

        true_ga = _matrix(objects["true_gene_activity.h5ad"]).astype(np.float64)
        promoter = _matrix(objects["true_promoter_component.h5ad"]).astype(np.float64)
        distal = _matrix(objects["true_distal_component.h5ad"]).astype(np.float64)
        difference = true_ga - promoter - distal
        max_difference = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
        rows.append(
            _row(
                "invariant:true_ga_decomposition",
                max_difference < 1e-5,
                f"max_abs_true_ga_minus_promoter_minus_distal={max_difference:.3g}",
            )
        )
        component_config_path = simulation_dir / "simulation_config.yaml"
        if component_config_path.exists():
            component_config = yaml.safe_load(component_config_path.read_text(encoding="utf-8"))
            distal_fraction = float(component_config.get("factors", {}).get("distal_fraction", -1.0))
            model = str(component_config.get("component_allocation_model", "centered_bounded_cell_gene_v1"))
            if distal_fraction == 0.0:
                component_identifiable = bool(distal.nnz == 0 and max_difference < 1e-5)
                component_detail = "zero_distal_control_has_zero_distal_component"
            else:
                ratio = distal.toarray() / np.maximum(true_ga.toarray(), 1e-12)
                ratio_range = float(np.ptp(ratio))
                component_identifiable = bool(model == "centered_bounded_cell_gene_v1" and ratio_range > 0.05)
                component_detail = f"model={model};distal_allocation_ratio_range={ratio_range:.6g}"
            rows.append(
                _row(
                    "invariant:component_truth_identifiability",
                    component_identifiable,
                    component_detail,
                )
            )
        rna_role = str(objects["simulated_rna.h5ad"].uns.get("rna_role", ""))
        rows.append(
            _row(
                "invariant:rna_evaluation_only",
                rna_role == "evaluation_only_never_construction_tuning",
                f"rna_role={rna_role}",
            )
        )

        labels_path = simulation_dir / "true_domain_labels.tsv"
        links_path = simulation_dir / "true_peak_gene_links.tsv"
        if labels_path.exists():
            labels = pd.read_csv(labels_path, sep="\t", index_col=0)
            labels_valid = labels.index.astype(str).tolist() == atac.obs_names.astype(str).tolist() and {
                "true_domain",
                "true_domain_index",
                "x",
                "y",
            }.issubset(labels.columns)
            rows.append(_row("invariant:domain_label_alignment", labels_valid, "domain truth matches ATAC observations"))
        if links_path.exists():
            links = pd.read_csv(links_path, sep="\t")
            required_columns = {"peak_id", "gene_id", "link_type", "true_weight", "is_active_link"}
            links_valid = (
                required_columns.issubset(links.columns)
                and set(links["peak_id"]).issubset(set(atac.var_names.astype(str)))
                and set(links["gene_id"]).issubset(set(gene_names.astype(str)))
                and np.all(np.isfinite(links["true_weight"].to_numpy(float)))
                and np.all(links["true_weight"].to_numpy(float) >= 0)
            )
            rows.append(_row("invariant:peak_gene_link_integrity", bool(links_valid), "all links resolve to peak and gene truth"))

    config_path = simulation_dir / "simulation_config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        schema_valid = int(config.get("schema_version", -1)) == SIMULATION_SCHEMA_VERSION
        factors_valid = set(config.get("factors", {})) == set(FACTOR_RANGES)
        rows.append(_row("invariant:simulation_schema_version", schema_valid, f"schema_version={config.get('schema_version')}"))
        rows.append(_row("invariant:factor_contract", factors_valid, "factor names match the frozen schema"))

    provenance_path = simulation_dir / "simulation_provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        boundary_valid = (
            provenance.get("holdout_accessed") is False
            and provenance.get("full_benchmark_run") is False
            and provenance.get("final_ranking_run") is False
            and provenance.get("construction_evaluation_inputs_used") is False
            and provenance.get("forbidden_inputs_used") == []
            and provenance.get("truth_exposed_in_accessibility_obs") is False
        )
        rows.append(_row("invariant:governance_boundary", boundary_valid, "no holdout, ranking, or evaluation-input construction"))
        hash_checks = []
        for name, expected_hash in provenance.get("artifacts", {}).items():
            path = simulation_dir / name
            hash_checks.append(path.exists() and sha256_file(path) == expected_hash)
        rows.append(_row("invariant:artifact_hashes", bool(hash_checks) and all(hash_checks), f"verified={sum(hash_checks)}/{len(hash_checks)}"))
        if provenance.get("simulation_type") == "empirical_semisynthetic":
            calibration = provenance.get("calibration", {})
            calibration_valid = (
                calibration.get("source_role") == "development_only"
                and calibration.get("holdout_accessed") is False
                and calibration.get("construction_evaluation_inputs_used") is False
                and calibration.get("forbidden_inputs_used") == []
                and calibration.get("redistribution_policy") == "local_only_source_derived"
            )
            rows.append(_row("invariant:empirical_calibration_boundary", calibration_valid, "development accessibility and coordinates only"))

    reference_files = [
        simulation_dir / "reference/gene_body.bed",
        simulation_dir / "reference/promoters_2kb.bed",
        simulation_dir / "reference/chrom_sizes.tsv",
        simulation_dir / "reference/synthetic_annotation.gtf",
    ]
    reference_valid = all(path.exists() and path.stat().st_size > 0 for path in reference_files)
    rows.append(_row("invariant:synthetic_reference", reference_valid, "BED, chrom sizes, and GTF are present"))

    output = pd.DataFrame(rows)
    output.to_csv(simulation_dir / "validation_summary.tsv", sep="\t", index=False)
    return output


def summarize_simulation(simulation_dir: str | Path, outdir: str | Path) -> dict[str, Path]:
    simulation_dir = Path(simulation_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    validation = validate_simulation_outputs(simulation_dir)
    atac = ad.read_h5ad(simulation_dir / "simulated_atac_peak_matrix.h5ad")
    summary = pd.DataFrame(
        [
            {
                "simulation_dir": str(simulation_dir),
                "n_cells": atac.n_obs,
                "n_peaks": atac.n_vars,
                "checks_passed": int((validation["validation_status"] == "pass").sum()),
                "checks_total": int(validation.shape[0]),
                "status": "pass" if (validation["validation_status"] == "pass").all() else "fail",
            }
        ]
    )
    summary_path = outdir / "simulation_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    report_path = outdir / "simulation_summary.md"
    report_path.write_text(
        "# Simulation summary\n\n"
        f"Simulation: `{simulation_dir}`\n\n"
        f"Contract checks: {summary.loc[0, 'checks_passed']}/{summary.loc[0, 'checks_total']} pass.\n",
        encoding="utf-8",
    )
    return {"summary": summary_path, "report": report_path, "validation": simulation_dir / "validation_summary.tsv"}
