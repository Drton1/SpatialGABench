"""G3-only simulation smoke execution and paired factor audits."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .calibration import sha256_file
from .design import expand_factorial_design, load_design_config
from .generator import generate_empirical_semisynthetic_simulation, generate_mechanistic_simulation


TRUTH_OUTPUTS = [
    "true_gene_activity.h5ad",
    "true_promoter_component.h5ad",
    "true_distal_component.h5ad",
    "true_domain_program_component.h5ad",
    "simulated_rna.h5ad",
    "true_peak_gene_links.tsv",
    "true_domain_labels.tsv",
]
VALIDATION_OUTPUTS = ["validation_summary.tsv", "simulation_provenance.json"]


def _matrix(path: Path) -> sparse.csr_matrix:
    obj = ad.read_h5ad(path)
    return obj.X.tocsr() if sparse.issparse(obj.X) else sparse.csr_matrix(obj.X)


def _matrix_equal(left: Path, right: Path, *, atol: float = 0.0) -> bool:
    difference = _matrix(left).astype(np.float64) - _matrix(right).astype(np.float64)
    if difference.nnz == 0:
        return True
    return bool(np.max(np.abs(difference.data)) <= atol)


def _domain_contrast(run_dir: Path, n_domains: int) -> float:
    activity = _matrix(run_dir / "true_gene_activity.h5ad").toarray()
    labels = pd.read_csv(run_dir / "true_domain_labels.tsv", sep="\t", index_col=0)["true_domain_index"].to_numpy(int)
    contrasts = []
    for gene_index in range(activity.shape[1]):
        marker = gene_index % n_domains
        inside = activity[labels == marker, gene_index]
        outside = activity[labels != marker, gene_index]
        contrasts.append(float(inside.mean() - outside.mean()))
    return float(np.mean(contrasts))


def _causal_geometry_equal(left: Path, right: Path) -> bool:
    left_var = ad.read_h5ad(left / "simulated_atac_peak_matrix.h5ad").var
    right_var = ad.read_h5ad(right / "simulated_atac_peak_matrix.h5ad").var
    left_core = left_var.loc[left_var["is_causal"] | (left_var["peak_type"] == "gene_body_proxy")]
    right_core = right_var.loc[right_var["is_causal"] | (right_var["peak_type"] == "gene_body_proxy")]
    columns = ["chr", "start", "end", "peak_type", "linked_gene_id"]
    if left_core.index.astype(str).tolist() != right_core.index.astype(str).tolist():
        return False
    left_values = left_core[columns].astype({"chr": str, "start": int, "end": int, "peak_type": str, "linked_gene_id": str})
    right_values = right_core[columns].astype({"chr": str, "start": int, "end": int, "peak_type": str, "linked_gene_id": str})
    return bool(np.array_equal(left_values.to_numpy(), right_values.to_numpy()))


def _factor_effect(factor: str, baseline_dir: Path, changed_dir: Path, n_domains: int) -> tuple[float, str, bool]:
    if factor == "distal_fraction":
        baseline_value = float(_matrix(baseline_dir / "true_distal_component.h5ad").sum())
        changed_value = float(_matrix(changed_dir / "true_distal_component.h5ad").sum())
        return changed_value - baseline_value, "true_distal_sum_increases", changed_value > baseline_value
    if factor == "depth_multiplier":
        baseline_value = float(_matrix(baseline_dir / "expected_peak_accessibility.h5ad").sum())
        changed_value = float(_matrix(changed_dir / "expected_peak_accessibility.h5ad").sum())
        return changed_value / baseline_value, "expected_total_ratio_greater_than_one", changed_value > baseline_value
    if factor == "zero_inflation_rate":
        baseline_value = int(_matrix(baseline_dir / "simulated_atac_peak_matrix.h5ad").nnz)
        changed_value = int(_matrix(changed_dir / "simulated_atac_peak_matrix.h5ad").nnz)
        return float(changed_value - baseline_value), "observed_nnz_decreases", changed_value < baseline_value
    if factor == "domain_log_effect":
        baseline_value = _domain_contrast(baseline_dir, n_domains)
        changed_value = _domain_contrast(changed_dir, n_domains)
        return changed_value - baseline_value, "marker_domain_contrast_increases", changed_value > baseline_value
    if factor == "decoy_fraction":
        baseline_value = int(ad.read_h5ad(baseline_dir / "simulated_atac_peak_matrix.h5ad").var["is_decoy"].sum())
        changed_value = int(ad.read_h5ad(changed_dir / "simulated_atac_peak_matrix.h5ad").var["is_decoy"].sum())
        return float(changed_value - baseline_value), "decoy_peak_count_increases", changed_value > baseline_value
    if factor == "rna_latent_correlation":
        baseline_value = float(pd.read_csv(baseline_dir / "rna_calibration_qc.tsv", sep="\t").loc[0, "realized_latent_correlation"])
        changed_value = float(pd.read_csv(changed_dir / "rna_calibration_qc.tsv", sep="\t").loc[0, "realized_latent_correlation"])
        return changed_value - baseline_value, "realized_latent_correlation_increases", changed_value > baseline_value
    raise ValueError(f"Unsupported factor: {factor}")


def _audit_factor_pair(
    track: str,
    factor: str,
    baseline_id: str,
    changed_id: str,
    runs_dir: Path,
    n_domains: int,
) -> dict[str, Any]:
    baseline_dir = runs_dir / baseline_id
    changed_dir = runs_dir / changed_id
    labels_equal = (baseline_dir / "true_domain_labels.tsv").read_bytes() == (changed_dir / "true_domain_labels.tsv").read_bytes()
    invariants = {
        "spatial_and_domain_labels_equal": labels_equal,
        "causal_peak_geometry_equal": _causal_geometry_equal(baseline_dir, changed_dir),
        "true_ga_equal_when_not_domain_factor": True
        if factor == "domain_log_effect"
        else _matrix_equal(baseline_dir / "true_gene_activity.h5ad", changed_dir / "true_gene_activity.h5ad", atol=1e-6),
        "rna_equal_when_not_domain_or_rna_factor": True
        if factor in {"domain_log_effect", "rna_latent_correlation"}
        else _matrix_equal(baseline_dir / "simulated_rna.h5ad", changed_dir / "simulated_rna.h5ad"),
        "atac_equal_for_rna_factor": True
        if factor != "rna_latent_correlation"
        else _matrix_equal(
            baseline_dir / "simulated_atac_peak_matrix.h5ad", changed_dir / "simulated_atac_peak_matrix.h5ad"
        ),
        "expectation_equal_for_zero_inflation_or_rna_factor": True
        if factor not in {"zero_inflation_rate", "rna_latent_correlation"}
        else _matrix_equal(
            baseline_dir / "expected_peak_accessibility.h5ad", changed_dir / "expected_peak_accessibility.h5ad", atol=1e-6
        ),
    }
    effect_value, expectation, direction_pass = _factor_effect(factor, baseline_dir, changed_dir, n_domains)
    return {
        "simulation_type": track,
        "factor": factor,
        "baseline_simulation_id": baseline_id,
        "changed_simulation_id": changed_id,
        "paired_seed_equal": True,
        "invariants": json.dumps(invariants, sort_keys=True),
        "all_non_target_invariants_pass": all(invariants.values()),
        "effect_value": effect_value,
        "expected_direction": expectation,
        "expected_direction_pass": direction_pass,
        "status": "pass" if all(invariants.values()) and direction_pass else "fail",
    }


def _runtime_config(design_config: dict[str, Any], row: pd.Series, outdir: Path) -> dict[str, Any]:
    fixed = dict(design_config["fixed_parameters"])
    fixed.pop("generator_schema_version")
    factors = {factor: float(row[factor]) for factor in design_config["factors"]}
    config = {
        "schema_version": int(design_config["fixed_parameters"]["generator_schema_version"]),
        "simulation_id": str(row["simulation_id"]),
        "simulation_type": str(row["simulation_type"]),
        "random_seed": int(row["random_seed"]),
        "replicate_id": str(row["replicate_id"]),
        **fixed,
        "factors": factors,
        "output_dir": str(outdir),
        "truth_outputs": TRUTH_OUTPUTS,
        "validation_outputs": VALIDATION_OUTPUTS,
    }
    if row["simulation_type"] == "empirical_semisynthetic":
        config["calibration"] = dict(design_config["calibration"])
    return config


def run_g3_simulation_smoke(
    design_config_path: str | Path,
    outdir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Run only the pre-declared G3 OAT subset; never run the full factorial."""

    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    outdir = Path(outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    design_config = load_design_config(design_config_path)
    design, design_audit = expand_factorial_design(design_config)
    design_path = outdir / "simulation_factorial_design.tsv"
    design_audit_path = outdir / "simulation_factorial_design_audit.json"
    design.to_csv(design_path, sep="\t", index=False)
    design_audit_path.write_text(json.dumps(design_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    smoke = design.loc[design["smoke_member"]].copy()
    run_rows = []
    failure_rows = []
    for _, row in smoke.iterrows():
        run_dir = runs_dir / str(row["simulation_id"])
        config = _runtime_config(design_config, row, run_dir)
        started = time.perf_counter()
        try:
            if row["simulation_type"] == "mechanistic":
                outputs = generate_mechanistic_simulation(config, run_dir)
            else:
                outputs = generate_empirical_semisynthetic_simulation(config, run_dir, project_root=root)
            validation = pd.read_csv(outputs["validation"], sep="\t")
            run_rows.append(
                {
                    "simulation_id": row["simulation_id"],
                    "simulation_type": row["simulation_type"],
                    "random_seed": int(row["random_seed"]),
                    "runtime_seconds": time.perf_counter() - started,
                    "validation_checks_passed": int((validation["validation_status"] == "pass").sum()),
                    "validation_checks_total": int(validation.shape[0]),
                    "provenance_sha256": sha256_file(outputs["provenance"]),
                    "status": "pass",
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "simulation_id": row["simulation_id"],
                    "simulation_type": row["simulation_type"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "runtime_seconds": time.perf_counter() - started,
                    "status": "fail",
                }
            )

    runs = pd.DataFrame(run_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["simulation_id", "simulation_type", "error_type", "error_message", "runtime_seconds", "status"],
    )
    runs_path = outdir / "simulation_smoke_runs.tsv"
    failures_path = outdir / "simulation_smoke_failures.tsv"
    runs.to_csv(runs_path, sep="\t", index=False)
    failures.to_csv(failures_path, sep="\t", index=False)

    factor_rows = []
    if failures.empty:
        factor_names = list(design_config["factors"])
        first_levels = {factor: float(design_config["factors"][factor][0]) for factor in factor_names}
        for track in design_config["tracks"]:
            track_smoke = smoke.loc[smoke["simulation_type"] == track]
            baseline_mask = np.logical_and.reduce(
                [np.isclose(track_smoke[factor].to_numpy(float), value) for factor, value in first_levels.items()]
            )
            baseline_id = str(track_smoke.loc[baseline_mask, "simulation_id"].iloc[0])
            for factor in factor_names:
                mask = np.ones(track_smoke.shape[0], dtype=bool)
                for candidate in factor_names:
                    target = float(design_config["factors"][candidate][1 if candidate == factor else 0])
                    mask &= np.isclose(track_smoke[candidate].to_numpy(float), target)
                changed_id = str(track_smoke.loc[mask, "simulation_id"].iloc[0])
                factor_rows.append(
                    _audit_factor_pair(
                        track,
                        factor,
                        baseline_id,
                        changed_id,
                        runs_dir,
                        int(design_config["fixed_parameters"]["n_domains"]),
                    )
                )
    factor_audit = pd.DataFrame(factor_rows)
    factor_audit_path = outdir / "simulation_factor_isolation_audit.tsv"
    factor_audit.to_csv(factor_audit_path, sep="\t", index=False)

    status = "pass" if (
        design_audit["status"] == "pass"
        and failures.empty
        and runs.shape[0] == smoke.shape[0]
        and not factor_audit.empty
        and (factor_audit["status"] == "pass").all()
    ) else "fail"
    manifest = {
        "schema_version": 1,
        "gate": "G3",
        "scope": "design_expansion_and_first_seed_OAT_smoke_only",
        "status": status,
        "design_id": design_config["design_id"],
        "planned_full_factorial_rows": int(design.shape[0]),
        "smoke_rows_attempted": int(smoke.shape[0]),
        "smoke_rows_passed": int((runs["status"] == "pass").sum()) if not runs.empty else 0,
        "failure_rows": int(failures.shape[0]),
        "factor_pairs_passed": int((factor_audit["status"] == "pass").sum()) if not factor_audit.empty else 0,
        "factor_pairs_total": int(factor_audit.shape[0]),
        "software_methods_run": False,
        "full_benchmark_run": False,
        "final_ranking_run": False,
        "holdout_accessed": False,
        "construction_evaluation_inputs_used": False,
        "artifacts": {
            "design": {"path": str(design_path.relative_to(root)), "sha256": sha256_file(design_path)},
            "design_audit": {"path": str(design_audit_path.relative_to(root)), "sha256": sha256_file(design_audit_path)},
            "runs": {"path": str(runs_path.relative_to(root)), "sha256": sha256_file(runs_path)},
            "failures": {"path": str(failures_path.relative_to(root)), "sha256": sha256_file(failures_path)},
            "factor_audit": {"path": str(factor_audit_path.relative_to(root)), "sha256": sha256_file(factor_audit_path)},
        },
    }
    manifest_path = outdir / "simulation_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "pass":
        raise RuntimeError(f"G3 simulation smoke failed; inspect {manifest_path}")
    return {
        "design": design_path,
        "design_audit": design_audit_path,
        "runs": runs_path,
        "failures": failures_path,
        "factor_audit": factor_audit_path,
        "manifest": manifest_path,
    }
