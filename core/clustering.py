"""
core/clustering.py
------------------
Lloyd KMeans puro NumPy + selezione marker genes.
Nessuna dipendenza sklearn — ottimizzato per Raspberry Pi.
"""

from __future__ import annotations
import numpy as np
import anndata as ad
from dataclasses import dataclass, field
from typing import Optional
from scipy.sparse import issparse


@dataclass
class ClusterResult:
    labels: np.ndarray        # (n_cells,)
    centroids: np.ndarray     # (k, n_features)
    inertia: float
    n_iter: int
    marker_genes: list[str] = field(default_factory=list)
    marker_scores: np.ndarray = field(default_factory=lambda: np.array([]))


def lloyd_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    progress_callback=None,
) -> ClusterResult:
    """
    Algoritmo di Lloyd con inizializzazione KMeans++.
    Operazioni vettorizzate NumPy, nessun loop Python sulle celle.
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    # KMeans++ initialization
    idx = rng.integers(0, n)
    centroids = [X[idx].copy()]
    for _ in range(1, k):
        diffs = X[:, np.newaxis, :] - np.array(centroids)[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2).min(axis=1)
        probs = dists / dists.sum()
        centroids.append(X[rng.choice(n, p=probs)].copy())
    centroids = np.array(centroids, dtype=np.float32)

    labels = np.zeros(n, dtype=np.int32)
    dist_sq = np.zeros((n, k), dtype=np.float32)

    for iteration in range(max_iter):
        if progress_callback:
            progress_callback("Clustering", int(5 + (iteration / max_iter) * 20))

        # Assegnazione: broadcast (n,1,d) - (1,k,d)
        np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
               axis=2, out=dist_sq)
        new_labels = np.argmin(dist_sq, axis=1).astype(np.int32)

        # Aggiornamento centroidi
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = new_labels == j
            new_centroids[j] = X[mask].mean(axis=0) if mask.sum() > 0 else X[rng.integers(0, n)]

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels
        if shift < tol:
            break

    inertia = float(np.min(dist_sq, axis=1).sum())
    return ClusterResult(labels=labels, centroids=centroids, inertia=inertia, n_iter=iteration + 1)


def cluster_anndata(
    adata: ad.AnnData,
    k: int,
    use_rep: str = "X_pca",
    layer: Optional[str] = None,
    max_iter: int = 200,
    random_state: int = 42,
    progress_callback=None,
) -> ClusterResult:
    """Wrapper AnnData: usa PCA se disponibile, altrimenti X grezzo."""
    if use_rep in adata.obsm:
        X = adata.obsm[use_rep].astype(np.float32)
    elif layer and layer in adata.layers:
        mat = adata.layers[layer]
        X = (mat.toarray() if issparse(mat) else mat).astype(np.float32)
    else:
        mat = adata.X
        X = (mat.toarray() if issparse(mat) else mat).astype(np.float32)

    result = lloyd_kmeans(X, k=k, max_iter=max_iter, random_state=random_state,
                          progress_callback=progress_callback)
    adata.obs["lloyd_cluster"] = result.labels.astype(str)
    return result


def select_marker_genes(
    adata: ad.AnnData,
    cluster_labels: np.ndarray,
    n_top: int = 200,
    method: str = "variance_ratio",
) -> tuple[list[str], np.ndarray]:
    """
    Identifica i top marker genes per distinguere i cluster.

    variance_ratio (default): score = varianza inter-cluster / varianza totale
    mean_diff: score = max differenza di media tra coppie di cluster

    Riduce da ~20k geni a n_top → training RF 10-50x più veloce su Pi.
    """
    mat = adata.X
    X = (mat.toarray() if issparse(mat) else mat).astype(np.float32)
    gene_names = np.array(adata.var_names)
    k = int(cluster_labels.max()) + 1

    if method == "variance_ratio":
        total_var = X.var(axis=0)
        total_var = np.where(total_var == 0, 1e-9, total_var)
        cluster_means = np.zeros((k, X.shape[1]), dtype=np.float32)
        cluster_sizes = np.zeros(k, dtype=np.int32)
        for j in range(k):
            mask = cluster_labels == j
            cluster_sizes[j] = mask.sum()
            if mask.sum() > 0:
                cluster_means[j] = X[mask].mean(axis=0)
        global_mean = X.mean(axis=0)
        inter_var = np.average((cluster_means - global_mean) ** 2,
                               axis=0, weights=cluster_sizes)
        scores = inter_var / total_var

    elif method == "mean_diff":
        cluster_means = np.zeros((k, X.shape[1]), dtype=np.float32)
        for j in range(k):
            mask = cluster_labels == j
            if mask.sum() > 0:
                cluster_means[j] = X[mask].mean(axis=0)
        scores = np.zeros(X.shape[1], dtype=np.float32)
        for j in range(k):
            for l in range(j + 1, k):
                scores = np.maximum(scores, np.abs(cluster_means[j] - cluster_means[l]))
    else:
        raise ValueError(f"Metodo sconosciuto: {method}")

    top_idx = np.argsort(scores)[::-1][:n_top]
    return list(gene_names[top_idx]), scores[top_idx]
