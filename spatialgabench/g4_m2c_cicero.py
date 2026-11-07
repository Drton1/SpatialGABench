"""Prepare and execute deterministic Cicero scalability subsets."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from spatialgabench.software_wrappers.contracts import write_wrapper_input_validation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(config_path: str | Path, dataset_id: str, cells: int, root: str | Path) -> tuple[dict, Path, Path]:
    root = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if cells not in config["cell_ladder"]:
        raise ValueError("Cell count is outside the frozen Cicero ladder")
    for key in ("authorization", "representation_summary"):
        if sha256(root / config[key]) != config[f"{key}_sha256"]:
            raise ValueError(f"Frozen artifact checksum mismatch: {key}")
    summary = pd.read_csv(root / config["representation_summary"], sep="\t")
    selected = summary[(summary.dataset_id == dataset_id) & (summary.representation_id == "macs3_summit_501bp")]
    if selected.shape[0] != 1:
        raise ValueError("Dataset is outside the frozen peak panel")
    row = selected.iloc[0]
    if cells > int(row.n_observations):
        raise ValueError("Requested Cicero cell count exceeds available observations")
    source_path = root / str(row.output_path)
    if sha256(source_path) != row.output_sha256:
        raise ValueError("Frozen peak matrix checksum mismatch")
    source = ad.read_h5ad(source_path)
    ordered_cells = sorted(source.obs_names.astype(str), key=lambda value: (hashlib.sha256(f"{config['seed']}\0{value}".encode()).hexdigest(), value))[:cells]
    matrix = source[ordered_cells, :].X
    matrix = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    var = source.var.copy()
    chromosome = var["chromosome"] if "chromosome" in var else var["chr"]
    ranks = pd.DataFrame({"index": np.arange(source.n_vars), "total": totals,
        "chromosome": chromosome.astype(str).to_numpy(), "start": var["start"].astype(int).to_numpy(),
        "end": var["end"].astype(int).to_numpy(), "feature": source.var_names.astype(str)})
    ranks = ranks[ranks.total > 0].sort_values(["total", "chromosome", "start", "end", "feature"], ascending=[False, True, True, True, True], kind="mergesort")
    feature_indices = ranks["index"].to_numpy(dtype=int)[: int(config["max_peaks"])]
    subset = source[ordered_cells, feature_indices].copy()
    subset.uns.update({"dataset_id": dataset_id, "study_id": str(row.study_id), "sample_id": str(row.sample_id),
        "genome_build": str(row.genome_build), "representation_id": "macs3_summit_501bp_cicero_subset",
        "selection_policy": "deterministic_sha256_cells_top_accessibility_peaks_coordinate_tiebreak",
        "source_h5ad": str(source_path.relative_to(root)), "source_sha256": sha256(source_path),
        "requested_cells": cells, "max_peaks": int(config["max_peaks"]), "construction_leakage_inputs": "none"})
    output_dir = root / config["output_root"] / dataset_id / f"cells_{cells}"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "cicero_subset.h5ad"
    subset.write_h5ad(subset_path, compression="gzip")
    wrapper = {"schema_version": 1, "run_id": f"g4_m2c__{dataset_id}__cicero__cells_{cells}",
        "project_config": config["project_config"], "study_id": str(row.study_id), "dataset_id": dataset_id,
        "sample_id": str(row.sample_id), "dataset_role": "development", "method_id": "cicero_geneactivity",
        "software_name": "Cicero", "genome_build": str(row.genome_build), "input_track": "P",
        "representation_id": "macs3_summit_501bp_cicero_subset", "comparison_scope": "subset_scalability_construction_only_no_pooling",
        "input_h5ad": str(subset_path.relative_to(root)), "fragments": None,
        "gene_annotation": config["gene_annotations"][str(row.genome_build)], "chrom_sizes": config["chrom_sizes"][str(row.genome_build)],
        "environment_lock": config["environment_lock"], "adapter": "adapters/r_geneactivity_driver.py",
        "method_script": "adapters/cicero_geneactivity.R",
        "implementation_files": ["configs/g4_m2c_cicero.yaml", "spatialgabench/g4_m2c_cicero.py", "adapters/r_geneactivity_driver.py", "adapters/cicero_geneactivity.R", "spatialgabench/software_wrappers/contracts.py", "spatialgabench/software_wrappers/common.py", "spatialgabench/runtime/process_tree.py", "containers/tools/run_rscript_in_oci.sh", "containers/tools/run_with_memory_tracking.py"],
        "construction_input_classes": ["peak_matrix", "gene_annotation"], "subset_policy": "scalability_subset",
        "threads": int(config["threads"]), "seed": int(config["seed"]), "output_dir": str(output_dir.relative_to(root))}
    wrapper_path = output_dir / "wrapper_config.yaml"
    wrapper_path.write_text(yaml.safe_dump(wrapper, sort_keys=False), encoding="utf-8")
    validation_path = output_dir / "input_validation.json"
    validation = write_wrapper_input_validation(wrapper_path, root, validation_path)
    if validation["status"] != "pass": raise ValueError(f"Cicero input contract failed: {validation_path}")
    return config, wrapper_path, validation_path


def execute(config: dict, wrapper_path: Path, validation_path: Path, root: Path) -> int:
    wrapper = yaml.safe_load(wrapper_path.read_text())
    out = root / wrapper["output_dir"]
    oci = config["oci"]
    env = dict(os.environ)
    env.update({"XDG_RUNTIME_DIR": str(oci["runtime_dir"]), "SPGAB_R_IMAGE": config["image"], "SPGAB_REPO_ROOT": str(root),
        "SPGAB_PODMAN_BIN": str(oci["podman"]), "SPGAB_PODMAN_ROOT": str(oci["root"]), "SPGAB_PODMAN_RUNROOT": str(oci["runroot"]),
        "SPGAB_PODMAN_STORAGE_DRIVER": str(oci["storage_driver"]), "SPGAB_OCI_RUNTIME": str((root / oci["runtime"]).resolve()),
        "SPGAB_CONMON": str(oci["conmon"]), "SPGAB_CGROUP_MANAGER": str(oci["cgroup_manager"]), "SPGAB_CONMON_PIDFILE": str(out / "conmon.pid")})
    method = [sys.executable, str(root / wrapper["adapter"]), "--config", str(wrapper_path), "--validation", str(validation_path), "--root", str(root), "--rscript", str(root / "containers/tools/run_rscript_in_oci.sh"), "--r-library", "/usr/local/lib/R/site-library"]
    command = [sys.executable, str(root / "containers/tools/run_with_memory_tracking.py"), "--output", str(out / "process_tree_memory.json"), "--sample-interval", "0.05", "--anchor-pid-file", str(out / "conmon.pid"), "--cleanup-anchor-pid-files", "--", *method]
    return subprocess.run(command, cwd=root, env=env, check=False).returncode
