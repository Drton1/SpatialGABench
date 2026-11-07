"""Truth-object contracts and serialization helpers for simulations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


TRUTH_CONTRACT_ROWS = [
    ("simulated_atac_peak_matrix.h5ad", True, "h5ad", "cells x peaks", "method input without truth fields", "peak coordinates", "X, obs, var, obsm/spatial", "all benchmark methods", "Only construction input exposed to methods."),
    ("expected_peak_accessibility.h5ad", True, "h5ad", "cells x peaks", "obs_names match ATAC", "peak coordinates", "X", "observation-model audit", "Expectation before zero inflation; evaluation only."),
    ("true_gene_activity.h5ad", True, "h5ad", "cells x genes", "obs_names match ATAC", "gene_id/gene_name", "X", "true GA recovery", "Primary latent truth."),
    ("true_promoter_component.h5ad", True, "h5ad", "cells x genes", "obs_names match ATAC", "gene_id/gene_name", "X", "promoter recovery", "Must sum with distal component to true GA."),
    ("true_distal_component.h5ad", True, "h5ad", "cells x genes", "obs_names match ATAC", "gene_id/gene_name", "X", "distal recovery", "Must sum with promoter component to true GA."),
    ("true_domain_program_component.h5ad", True, "h5ad", "cells x genes", "obs_names match ATAC", "gene_id/gene_name", "X", "domain/marker recovery", "Evaluation only."),
    ("simulated_rna.h5ad", True, "h5ad", "cells x genes", "obs_names match ATAC", "gene_id/gene_name", "X", "RNA proxy evaluation", "Evaluation only; never construction tuning."),
    ("true_peak_gene_links.tsv", True, "tsv", "links x fields", "not applicable", "not applicable", "peak_id,gene_id,link_type,true_weight", "link recovery", "Ground-truth causal links."),
    ("true_domain_labels.tsv", True, "tsv", "cells x labels", "cell_id matches ATAC", "not applicable", "true_domain,true_domain_index,x,y", "domain recovery", "Labels are absent from ATAC obs."),
    ("simulation_factor_values.tsv", True, "tsv", "one row", "not applicable", "not applicable", "six factor values", "factor-effect audit", "Pre-registered numeric factors."),
    ("simulation_provenance.json", True, "json", "one object", "not applicable", "not applicable", "hashes,input classes,holdout flags", "provenance audit", "Must record no holdout and no forbidden construction input."),
    ("true_tf_activity.h5ad", False, "h5ad", "cells x TFs", "obs_names match ATAC", "TF gene ids", "X", "TF recovery", "Present only when TF program is enabled."),
    ("true_motif_accessibility.h5ad", False, "h5ad", "cells x motifs", "obs_names match ATAC", "motif ids", "X", "motif recovery", "Present only when TF program is enabled."),
    ("true_tf_motif_links.tsv", False, "tsv", "links x fields", "not applicable", "not applicable", "tf_gene_id,motif_id,target_gene_id", "TF/motif consistency", "Present only when TF program is enabled."),
]


def truth_contract_table() -> pd.DataFrame:
    return pd.DataFrame(
        TRUTH_CONTRACT_ROWS,
        columns=[
            "object_name",
            "required",
            "format",
            "shape_expectation",
            "obs_contract",
            "var_contract",
            "required_fields",
            "evaluation_tasks_using_it",
            "notes",
        ],
    )


def matrix_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_h5ad(
    X: np.ndarray | sparse.spmatrix,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    path: str | Path,
    *,
    uns: dict[str, object],
    spatial: np.ndarray | None = None,
) -> ad.AnnData:
    X_out = X.tocsr().astype(np.float32) if sparse.issparse(X) else sparse.csr_matrix(np.asarray(X, dtype=np.float32))
    X_out.eliminate_zeros()
    obj = ad.AnnData(X=X_out, obs=obs.copy(), var=var.copy())
    if spatial is not None:
        obj.obsm["spatial"] = np.asarray(spatial, dtype=np.float32)
    for key, value in uns.items():
        obj.uns[key] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.write_h5ad(path, compression="gzip")
    return obj
