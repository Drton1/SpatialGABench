"""Raw peak binary TF-IDF plus truncated SVD LSI baseline."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def binarize_peak_matrix(X) -> sparse.csr_matrix:
    """Return a sparse binary accessibility matrix."""

    if not sparse.issparse(X):
        raise ValueError("Input peak matrix must be sparse; refusing to densify.")
    X_binary = X.tocsr(copy=True)
    X_binary.eliminate_zeros()
    if X_binary.data.size and (X_binary.data < 0).any():
        raise ValueError("Input peak matrix contains negative values.")
    X_binary.data[:] = 1.0
    return X_binary.astype(np.float32, copy=False)


def compute_tfidf_binary(X_binary) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Compute TF-IDF from a sparse binary peak matrix."""

    if not sparse.issparse(X_binary):
        raise ValueError("Binary peak matrix must be sparse; refusing to densify.")
    X_binary = X_binary.tocsr(copy=True)
    X_binary.eliminate_zeros()
    if X_binary.data.size and not np.allclose(X_binary.data, 1.0):
        raise ValueError("X_binary must contain only binary non-zero values.")
    n_cells, _ = X_binary.shape
    accessible_per_cell = np.asarray(X_binary.sum(axis=1)).ravel().astype(np.float64)
    row_scale = np.divide(
        1.0,
        accessible_per_cell,
        out=np.zeros_like(accessible_per_cell, dtype=np.float64),
        where=accessible_per_cell > 0,
    )
    tf = sparse.diags(row_scale.astype(np.float32), format="csr") @ X_binary
    df = np.asarray((X_binary > 0).sum(axis=0)).ravel().astype(np.float64)
    idf = np.log1p(float(n_cells) / (1.0 + df)).astype(np.float32)
    tfidf = tf @ sparse.diags(idf, format="csr")
    tfidf = tfidf.tocsr()
    tfidf.eliminate_zeros()
    if np.isnan(tfidf.data).any():
        raise ValueError("TF-IDF matrix contains NaN values.")
    return tfidf.astype(np.float32, copy=False), idf, accessible_per_cell


def run_truncated_svd(X_tfidf, n_components: int = 50, random_seed: int = 202603) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run sparse TruncatedSVD and return embedding plus variance summaries."""

    if not sparse.issparse(X_tfidf):
        raise ValueError("TF-IDF matrix must be sparse; refusing to densify.")
    max_components = min(X_tfidf.shape) - 1
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if n_components > max_components:
        raise ValueError(f"n_components={n_components} exceeds sparse SVD limit {max_components}")
    svd = TruncatedSVD(n_components=n_components, random_state=random_seed, algorithm="randomized")
    embedding = svd.fit_transform(X_tfidf)
    embedding = np.asarray(embedding, dtype=np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError("LSI embedding contains non-finite values.")
    return embedding, svd.explained_variance_ratio_.astype(np.float64), svd.singular_values_.astype(np.float64)


def write_lsi_h5ad(
    *,
    input_adata: ad.AnnData,
    embedding: np.ndarray,
    output_path: str | Path,
    dataset_id: str,
    method_id: str,
    genome_build: str,
    input_peak_h5ad: str,
    n_components: int,
    random_seed: int,
    explained_variance_ratio: np.ndarray,
    singular_values: np.ndarray,
    lightweight_X: bool = True,
) -> ad.AnnData:
    """Write an embedding-focused AnnData output preserving obs and spatial."""

    if embedding.shape != (input_adata.n_obs, n_components):
        raise ValueError(f"Embedding shape {embedding.shape} does not match expected {(input_adata.n_obs, n_components)}")
    if lightweight_X:
        X_out = sparse.csr_matrix((input_adata.n_obs, 0), dtype=np.float32)
        var = pd.DataFrame(index=pd.Index([], name="feature_id"))
    else:
        if not sparse.issparse(input_adata.X):
            raise ValueError("Input X must be sparse when preserving raw peak X.")
        X_out = input_adata.X.tocsr().copy()
        var = input_adata.var.copy()
    output = ad.AnnData(X=X_out, obs=input_adata.obs.copy(), var=var)
    output.obs_names = input_adata.obs_names.copy()
    if "spatial" in input_adata.obsm:
        output.obsm["spatial"] = input_adata.obsm["spatial"].copy()
    output.obsm["X_lsi"] = embedding
    output.uns["dataset_id"] = dataset_id
    output.uns["source_modality"] = "spatial_atac"
    output.uns["method_id"] = method_id
    output.uns["gene_activity_method"] = method_id
    output.uns["is_gene_activity"] = False
    output.uns["output_type"] = "cell_embedding"
    output.uns["genome_build"] = genome_build
    output.uns["input_peak_h5ad"] = input_peak_h5ad
    output.uns["normalization"] = "TF-IDF on binary peak matrix"
    output.uns["n_components"] = int(n_components)
    output.uns["random_seed"] = int(random_seed)
    output.uns["explained_variance_ratio"] = explained_variance_ratio.astype(float).tolist()
    output.uns["cumulative_explained_variance"] = np.cumsum(explained_variance_ratio).astype(float).tolist()
    output.uns["singular_values"] = singular_values.astype(float).tolist()
    output.uns["X_storage"] = "empty_sparse_matrix_embedding_focused" if lightweight_X else "raw_peak_counts_sparse"
    output.uns["method_notes"] = "Raw peak binary TF-IDF plus TruncatedSVD LSI baseline; no gene activity conversion or downstream clustering."
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(output_path, compression="gzip")
    return output
