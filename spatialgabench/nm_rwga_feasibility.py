"""Low-depth real-data feasibility runner for reliability-weighted gene activity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pysam
import yaml
from scipy import sparse

from spatialgabench.gene_activity.reliability_multiscale import (
    fit_reliability_multiscale,
    weighted_activity_moments,
)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    loader = json.loads if path.suffix == ".json" else yaml.safe_load
    value = loader(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _fold_index(raw_line: bytes, row_index: int, seed: int, groups: int) -> int:
    prefix = f"{seed}\0{row_index}\0".encode("ascii")
    return int.from_bytes(hashlib.sha256(prefix + raw_line).digest()[:8], "big") % groups


def preflight(config_path: str | Path, root: str | Path = ".") -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _load(config_path)
    bindings = [
        config["protocol"], config["proposal"], config["source_fragments"],
        config["source_fragment_audit"], config["positions"], config["project_config"],
        config["full_peak_representation"], config["full_bin_representation"],
        config["full_peak_mapping"], config["full_bin_mapping"],
    ]
    for value in bindings:
        if not (root / value).is_file():
            raise ValueError(f"RWGA feasibility input missing: {value}")
    if sha256_path(root / config["source_fragments"]) != config["source_fragments_sha256"]:
        raise ValueError("RWGA source fragment checksum mismatch")
    source_audit = _load(root / config["source_fragment_audit"])
    if (
        source_audit.get("status") != "pass"
        or int(source_audit["included_fragment_rows"]) != int(config["expected_source_fragment_rows"])
        or source_audit["output_fragments_sha256"] != config["source_fragments_sha256"]
    ):
        raise ValueError("RWGA source fragment audit mismatch")
    for artifact in config["references"].values():
        path = root / artifact["path"]
        if not path.is_file() or sha256_path(path) != artifact["sha256"]:
            raise ValueError(f"RWGA reference mismatch: {artifact['path']}")
    output = root / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    bindings_frame = pd.DataFrame([
        {"path": value, "sha256": sha256_path(root / value)} for value in bindings
    ])
    bindings_path = output / "input_bindings.tsv"
    bindings_frame.to_csv(bindings_path, sep="\t", index=False)
    authorization = {
        "schema_version": 1,
        "authorization": "NM_RWGA_LOW_DEPTH_FEASIBILITY",
        "milestone": config["milestone"],
        "status": "authorized",
        "construction_authorized": True,
        "rna_access_authorized": False,
        "labels_clusters_pathology_authorized": False,
        "parameter_tuning_authorized": False,
        "universal_ranking_authorized": False,
        "final_ranking_authorized": False,
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_path(config_path)},
        "protocol": {"path": config["protocol"], "sha256": sha256_path(root / config["protocol"])},
        "proposal": {"path": config["proposal"], "sha256": sha256_path(root / config["proposal"])},
        "input_bindings": {"path": str(bindings_path.relative_to(root)), "sha256": sha256_path(bindings_path)},
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": sha256_path(Path(__file__).resolve()),
        },
        "core": {
            "path": "spatialgabench/gene_activity/reliability_multiscale.py",
            "sha256": sha256_path(root / "spatialgabench/gene_activity/reliability_multiscale.py"),
        },
    }
    authorization_path = root / "governance/authorizations/NM_RWGA_LOW_DEPTH_FEASIBILITY_2026-08-01.json"
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    evidence = {
        "schema_version": 1, "milestone": f"{config['milestone']}_PREFLIGHT", "status": "pass",
        "source_fragment_rows": int(config["expected_source_fragment_rows"]),
        "delete_groups": int(config["delete_groups"]), "rna_expression_values_read": False,
        "labels_clusters_pathology_read": False, "parameter_tuning_run": False,
        "universal_ranking_run": False, "final_ranking_run": False,
        "authorization": {"path": str(authorization_path.relative_to(root)), "sha256": sha256_path(authorization_path)},
    }
    path = output / "preflight_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


def _validate_authorization(config_path: Path, authorization_path: Path, root: Path) -> dict[str, Any]:
    config, authorization = _load(config_path), _load(authorization_path)
    required_false = [
        "rna_access_authorized", "labels_clusters_pathology_authorized",
        "parameter_tuning_authorized", "universal_ranking_authorized", "final_ranking_authorized",
    ]
    if (
        authorization.get("status") != "authorized"
        or authorization.get("milestone") != config["milestone"]
        or authorization.get("construction_authorized") is not True
        or any(authorization.get(field) is not False for field in required_false)
        or authorization["config"]["sha256"] != sha256_path(config_path)
        or authorization["implementation"]["sha256"] != sha256_path(Path(__file__).resolve())
        or authorization["core"]["sha256"]
        != sha256_path(root / "spatialgabench/gene_activity/reliability_multiscale.py")
    ):
        raise PermissionError("RWGA feasibility authorization is invalid")
    return config


def prepare_folds(
    config_path: str | Path,
    authorization_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _validate_authorization(config_path, Path(authorization_path).resolve(), root)
    source = root / config["source_fragments"]
    output = root / config["output_root"] / "folds"
    output.mkdir(parents=True, exist_ok=True)
    groups = int(config["delete_groups"])
    targets = [output / f"delete_group_{index}.fragments.tsv.gz" for index in range(groups)]
    audits = [output / f"delete_group_{index}.audit.json" for index in range(groups)]
    if all(path.is_file() for path in audits):
        for target, audit_path in zip(targets, audits):
            audit = _load(audit_path)
            if audit.get("status") != "pass" or audit["output_fragments_sha256"] != sha256_path(target):
                raise ValueError("existing RWGA fold failed checksum validation")
        return output / "fold_manifest.tsv"
    temporary = [path.with_name(f".{path.name}.tmp") for path in targets]
    handles = [pysam.BGZFile(path, "wb") for path in temporary]
    source_rows = 0
    omitted = np.zeros(groups, dtype=np.int64)
    retained = np.zeros(groups, dtype=np.int64)
    try:
        with pysam.BGZFile(source, "rb") as source_handle:
            for raw_line in source_handle:
                if raw_line.startswith(b"#"):
                    continue
                source_rows += 1
                excluded = _fold_index(raw_line, source_rows, int(config["fold_hash_seed"]), groups)
                omitted[excluded] += 1
                line = raw_line.rstrip(b"\r\n") + b"\n"
                for index, handle in enumerate(handles):
                    if index != excluded:
                        handle.write(line)
                        retained[index] += 1
    finally:
        for handle in handles:
            handle.close()
    if source_rows != int(config["expected_source_fragment_rows"]):
        raise ValueError("RWGA fold source row count changed")
    rows: list[dict[str, Any]] = []
    for index, (tmp, target, audit_path) in enumerate(zip(temporary, targets, audits)):
        tmp.replace(target)
        pysam.tabix_index(str(target), preset="bed", force=True)
        audit = {
            "schema_version": 1, "status": "pass", "fold": index,
            "source_fragments": str(source.relative_to(root)),
            "source_fragments_sha256": sha256_path(source), "source_fragment_rows": source_rows,
            "omitted_fragment_rows": int(omitted[index]), "included_fragment_rows": int(retained[index]),
            "retained_fraction": float(retained[index] / source_rows),
            "output_fragments": str(target.resolve()), "output_fragments_sha256": sha256_path(target),
            "fold_rule": "sha256_seed_row_index_raw_line_modulo_K_delete_group_complement",
        }
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        rows.append({
            "fold": index, "fragments": str(target.relative_to(root)),
            "fragments_sha256": sha256_path(target), "included_fragment_rows": int(retained[index]),
            "retained_fraction": float(retained[index] / source_rows),
            "audit": str(audit_path.relative_to(root)), "audit_sha256": sha256_path(audit_path),
        })
    manifest = output / "fold_manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest, sep="\t", index=False)
    return manifest


def _construct_activity(
    representation_path: Path,
    config: dict[str, Any],
    root: Path,
    output: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from spatialgabench.gene_activity.distal import (
        build_promoter_plus_distal_mapping,
        load_tss_table,
    )
    from spatialgabench.gene_activity.interval_aggregation import (
        load_bed_intervals, load_gene_metadata, parse_peak_coordinates,
    )
    data = ad.read_h5ad(representation_path)
    genes = load_gene_metadata(root / config["references"]["genes"]["path"])
    promoters = load_bed_intervals(root / config["references"]["promoters"]["path"], "promoter")
    tss = load_tss_table(root / config["references"]["tss"]["path"], genes)
    mapping = build_promoter_plus_distal_mapping(
        parse_peak_coordinates(data), promoters, tss, genes,
        distal_window_bp=int(config["parameters"]["distal_window_bp"]),
        promoter_exclusion_window_bp=int(config["parameters"]["promoter_window_bp"]),
        distance_decay_bp=int(config["parameters"]["distance_decay_bp"]),
    )
    activity, event_variance, floor = weighted_activity_moments(data.X, mapping.peak_gene_matrix)
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "activity.npz", activity.astype(np.float32), compressed=True)
    sparse.save_npz(output / "event_variance.npz", event_variance.astype(np.float32), compressed=True)
    np.save(output / "variance_floor.npy", floor.astype(np.float64))
    valid = np.asarray(mapping.peak_gene_matrix.getnnz(axis=0)).ravel() > 0
    np.save(output / "gene_valid.npy", valid)
    pd.Series(data.obs_names.astype(str), name="observation_id").to_csv(output / "observations.tsv", sep="\t", index=False)
    pd.Series(mapping.genes.gene_id.astype(str), name="gene_id").to_csv(output / "genes.tsv", sep="\t", index=False)
    exposure = np.asarray(data.X.sum(axis=1)).ravel().astype(np.float64)
    return exposure, valid, data.obs_names.astype(str).to_numpy(), mapping.genes.gene_id.astype(str).to_numpy(), floor


def construct_fold(
    config_path: str | Path,
    authorization_path: str | Path,
    fold: int,
    root: str | Path = ".",
) -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _validate_authorization(config_path, Path(authorization_path).resolve(), root)
    manifest = pd.read_csv(root / config["output_root"] / "folds/fold_manifest.tsv", sep="\t")
    selected = manifest.loc[manifest.fold.eq(fold)]
    if selected.shape[0] != 1:
        raise ValueError("unknown RWGA fold")
    row = selected.iloc[0]
    fragments = root / row.fragments
    fold_root = root / config["output_root"] / "folds" / f"delete_group_{fold}"
    evidence_path = fold_root / "construction_evidence.json"
    if evidence_path.is_file() and _load(evidence_path).get("status") == "pass":
        return evidence_path
    peaks_dir = fold_root / "peaks"
    peaks_dir.mkdir(parents=True, exist_ok=True)
    name = f"{config['condition_id']}__delete_group_{fold}"
    narrow = peaks_dir / f"{name}_peaks.narrowPeak"
    macs3 = str(Path(sys.executable).with_name("macs3"))
    command = [
        macs3, "callpeak", "-f", str(config["macs3"]["format"]), "--max-count",
        str(config["macs3"]["max_count"]), "-t", str(fragments), "-g",
        str(config["macs3"]["genome_size"]), "-n", name, "-q", str(config["macs3"]["qvalue"]),
        "--call-summits", "--outdir", str(peaks_dir),
    ]
    with (peaks_dir / "macs3.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not narrow.is_file():
        raise RuntimeError(f"RWGA MACS3 failed for fold {fold}")
    from spatialgabench.representations.peaks import freeze_macs3_summits
    frozen = peaks_dir / "macs3_summit_501bp.bed"
    peak_audit = freeze_macs3_summits(
        narrow, root / config["references"]["chrom_sizes"]["path"],
        root / config["references"]["blacklist"]["path"], frozen,
        peaks_dir / "peak_freeze_audit.json", peak_width=int(config["macs3"]["peak_width_bp"]),
    )
    if peak_audit.get("status") != "pass" or int(peak_audit.get("frozen_peak_rows", 0)) <= 0:
        raise ValueError(f"RWGA fold {fold} produced no valid peaks")
    from spatialgabench.representations.matched import build_matched_representations
    representations = fold_root / "representations"
    outputs = build_matched_representations(
        project_config=root / config["project_config"], study_id=config["study_id"],
        dataset_id=name, sample_id=config["sample_id"], fragments_path=fragments,
        positions_path=root / config["positions"],
        chrom_sizes_path=root / config["references"]["chrom_sizes"]["path"], peaks_path=frozen,
        genome_build=config["genome_build"], bin_sizes=list(config["bin_sizes_bp"]),
        output_dir=representations, fragment_filter_audit_path=root / row.audit,
    )
    _, peak_valid, obs, genes, _ = _construct_activity(
        outputs["macs3_summit_501bp"], config, root, fold_root / "activity/peak",
    )
    bin_exp, _, bin_obs, bin_genes, _ = _construct_activity(
        outputs["fixed_bin_25kb"], config, root, fold_root / "activity/bin",
    )
    if not np.array_equal(obs, bin_obs) or not np.array_equal(genes, bin_genes):
        raise ValueError("RWGA fold branch identity mismatch")
    np.save(fold_root / "activity/exposure.npy", bin_exp)
    np.save(fold_root / "activity/peak_gene_valid.npy", peak_valid)
    evidence = {
        "schema_version": 1, "milestone": config["milestone"], "status": "pass", "fold": fold,
        "fragment_rows": int(row.included_fragment_rows), "observations": len(obs), "genes": len(genes),
        "frozen_peaks": int(peak_audit["frozen_peak_rows"]),
        "structurally_valid_peak_genes": int(peak_valid.sum()),
        "rna_expression_values_read": False, "labels_clusters_pathology_read": False,
        "parameter_tuning_run": False, "universal_ranking_run": False, "final_ranking_run": False,
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence_path


def _load_branch(path: Path) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    activity = sparse.load_npz(path / "activity.npz").tocsr()
    event = sparse.load_npz(path / "event_variance.npz").tocsr()
    floor = np.load(path / "variance_floor.npy")
    return activity, event, floor


def summarize(
    config_path: str | Path,
    authorization_path: str | Path,
    root: str | Path = ".",
) -> Path:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = _validate_authorization(config_path, Path(authorization_path).resolve(), root)
    output = root / config["output_root"]
    for fold in range(int(config["delete_groups"])):
        evidence = _load(output / "folds" / f"delete_group_{fold}" / "construction_evidence.json")
        if evidence.get("status") != "pass":
            raise ValueError("RWGA fold construction incomplete")

    peak_data = ad.read_h5ad(root / config["full_peak_representation"])
    bin_data = ad.read_h5ad(root / config["full_bin_representation"])
    if not peak_data.obs_names.equals(bin_data.obs_names):
        raise ValueError("RWGA full branch observations differ")
    peak_mapping = sparse.load_npz(root / config["full_peak_mapping"]).tocsr()
    bin_mapping = sparse.load_npz(root / config["full_bin_mapping"]).tocsr()
    peak_activity, peak_event, peak_floor = weighted_activity_moments(peak_data.X, peak_mapping)
    bin_activity, bin_event, bin_floor = weighted_activity_moments(bin_data.X, bin_mapping)
    exposure = np.asarray(bin_data.X.sum(axis=1)).ravel().astype(np.float64)
    full_peak_valid = np.asarray(peak_mapping.getnnz(axis=0)).ravel() > 0
    fold_peak, fold_bin, fold_exposure, fold_valid = [], [], [], []
    for fold in range(int(config["delete_groups"])):
        base = output / "folds" / f"delete_group_{fold}" / "activity"
        fold_peak.append(_load_branch(base / "peak")[0])
        fold_bin.append(_load_branch(base / "bin")[0])
        fold_exposure.append(np.load(base / "exposure.npy"))
        fold_valid.append(np.load(base / "peak_gene_valid.npy"))
    valid = np.stack(fold_valid, axis=0)
    if peak_activity.shape != bin_activity.shape or peak_activity.shape[0] != int(config["expected_observations"]):
        raise ValueError("RWGA full activity dimensions changed")
    genes = peak_activity.shape[1]
    rows: list[dict[str, Any]] = []
    total_values = 0
    zero_weights = 0
    finite = True
    bounded = True
    for start in range(0, genes, int(config["gene_chunk_size"])):
        stop = min(start + int(config["gene_chunk_size"]), genes)
        p = peak_activity[:, start:stop].toarray()
        b = bin_activity[:, start:stop].toarray()
        pv = peak_event[:, start:stop].toarray() + peak_floor[start:stop][None, :]
        bv = bin_event[:, start:stop].toarray() + bin_floor[start:stop][None, :]
        pp = [value[:, start:stop].toarray() for value in fold_peak]
        bp = [value[:, start:stop].toarray() for value in fold_bin]
        result = fit_reliability_multiscale(
            p, b, pv, bv, exposure, pp, bp, valid[:, start:stop], full_peak_valid[start:stop],
            perturbation_exposures=fold_exposure, scale=float(config["scale"]),
            minimum_spots=int(config["minimum_spots"]),
        )
        total_values += result.peak_weight.size
        zero_weights += int(np.sum(result.peak_weight == 0))
        finite &= bool(
            np.all(np.isfinite(result.activity_rate))
            and np.all(np.isfinite(result.peak_weight))
            and np.all(np.isfinite(result.continuous_risk))
        )
        bounded &= bool(np.all(result.activity_rate >= 0) and np.all((result.peak_weight >= 0) & (result.peak_weight <= 1)))
        for local, gene in enumerate(range(start, stop)):
            nonzero = result.peak_weight[:, local][result.peak_weight[:, local] > 0]
            rows.append({
                "gene_index": gene,
                "peak_signal_variance": float(result.peak_signal_variance[local]),
                "structural_risk": float(result.peak_structural_risk[local]),
                "valid_peak_perturbations": int(result.valid_peak_perturbations[local]),
                "median_peak_weight": float(np.median(result.peak_weight[:, local])),
                "median_nonzero_peak_weight": float(np.median(nonzero)) if nonzero.size else np.nan,
                "mean_peak_weight": float(np.mean(result.peak_weight[:, local])),
                "median_continuous_risk": float(np.median(result.continuous_risk[:, local])),
            })
    gene_summary = pd.DataFrame(rows)
    gene_path = output / "gene_feasibility_summary.tsv"
    gene_summary.to_csv(gene_path, sep="\t", index=False)
    nonzero_genes = gene_summary.loc[gene_summary.peak_signal_variance.gt(0)]
    valid_fraction = float(gene_summary.structural_risk.eq(0).mean())
    q2_fraction = float(gene_summary.peak_signal_variance.gt(0).mean())
    median_nonzero = float(nonzero_genes.median_nonzero_peak_weight.median()) if len(nonzero_genes) else np.nan
    gates = {
        "finite_bounded_outputs": finite and bounded,
        "nonzero_q2_gene_fraction": q2_fraction >= float(config["gates"]["minimum_nonzero_q2_gene_fraction"]),
        "structurally_valid_gene_fraction": valid_fraction >= float(config["gates"]["minimum_structurally_valid_gene_fraction"]),
        "nonzero_weight_lower": bool(
            np.isfinite(median_nonzero)
            and median_nonzero >= float(config["gates"]["minimum_nonzero_gene_median_peak_weight"])
        ),
        "nonzero_weight_upper": bool(
            np.isfinite(median_nonzero)
            and median_nonzero <= float(config["gates"]["maximum_nonzero_gene_median_peak_weight"])
        ),
    }
    summary = pd.DataFrame([{
        "condition_id": config["condition_id"], "genes": genes, "observations": peak_activity.shape[0],
        "nonzero_q2_gene_fraction": q2_fraction, "structurally_valid_gene_fraction": valid_fraction,
        "median_nonzero_gene_peak_weight": median_nonzero,
        "zero_peak_weight_fraction": zero_weights / total_values,
        "finite_bounded_outputs": finite and bounded, "status": "pass" if all(gates.values()) else "fail",
    }])
    summary_path = output / "feasibility_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    evidence = {
        "schema_version": 1, "milestone": config["milestone"],
        "status": "pass" if all(gates.values()) else "fail", "gates": gates,
        "rna_expression_values_read": False, "labels_clusters_pathology_read": False,
        "parameter_tuning_run": False, "universal_ranking_run": False, "final_ranking_run": False,
        "artifacts": {
            "feasibility_summary.tsv": {"path": str(summary_path.relative_to(root)), "sha256": sha256_path(summary_path)},
            "gene_feasibility_summary.tsv": {"path": str(gene_path.relative_to(root)), "sha256": sha256_path(gene_path)},
        },
    }
    evidence_path = output / "completion_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if evidence["status"] != "pass":
        raise ValueError(f"RWGA low-depth feasibility failed: {[k for k,v in gates.items() if not v]}")
    return evidence_path


def run(config_path: str | Path, authorization_path: str | Path, root: str | Path = ".") -> Path:
    prepare_folds(config_path, authorization_path, root)
    config = _load(Path(config_path).resolve())
    for fold in range(int(config["delete_groups"])):
        construct_fold(config_path, authorization_path, fold, root)
    return summarize(config_path, authorization_path, root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "prepare-folds", "construct-fold", "summarize", "run"])
    parser.add_argument("--config", default="configs/nm_rwga_low_depth_feasibility.yaml")
    parser.add_argument(
        "--authorization",
        default="governance/authorizations/NM_RWGA_LOW_DEPTH_FEASIBILITY_2026-08-01.json",
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.config, args.root)
    elif args.command == "prepare-folds":
        result = prepare_folds(args.config, args.authorization, args.root)
    elif args.command == "construct-fold":
        if args.fold is None:
            raise ValueError("--fold is required")
        result = construct_fold(args.config, args.authorization, args.fold, args.root)
    elif args.command == "summarize":
        result = summarize(args.config, args.authorization, args.root)
    else:
        result = run(args.config, args.authorization, args.root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
