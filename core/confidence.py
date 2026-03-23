"""
core/confidence.py
------------------
Confidence score basato su distanza euclidea cellula → centroide classe.
Soglia adattiva per classe (95° percentile intra-classe sul training set).
"""

from __future__ import annotations
import numpy as np


def compute_class_centroids(X_train: np.ndarray, y_train: np.ndarray) -> dict[int, np.ndarray]:
    """Centroide per ogni classe nel training set."""
    return {
        int(label): X_train[y_train == label].mean(axis=0)
        for label in np.unique(y_train)
    }


def euclidean_confidence(
    X_test: np.ndarray,
    y_pred: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    percentile_threshold: float = 95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola confidence score per ogni cellula del test set.

    Score = dist(cellula, centroide_classe_predetta) / soglia_classe
    Score > 1.0  →  Unknown / Potential New Cell Type

    La soglia è il percentile_threshold-esimo delle distanze intra-classe
    nel training set (calibrazione adattiva per classe).

    Returns
    -------
    scores       : float array, score normalizzato (0=vicino, >1=Unknown)
    unknown_mask : bool array, True dove la cellula è Unknown
    """
    centroids = compute_class_centroids(X_train, y_train)

    # Soglia per classe: percentile delle distanze intra-classe nel training
    thresholds: dict[int, float] = {}
    for label, centroid in centroids.items():
        mask = y_train == label
        if mask.sum() == 0:
            thresholds[label] = np.inf
            continue
        intra_dists = np.linalg.norm(X_train[mask] - centroid, axis=1)
        thresholds[label] = float(np.percentile(intra_dists, percentile_threshold))

    # Distanza test → centroide predetto, normalizzata per soglia
    raw = np.array([
        np.linalg.norm(X_test[i] - centroids[y_pred[i]])
        if y_pred[i] in centroids else np.inf
        for i in range(len(y_pred))
    ])
    thresh_arr = np.array([thresholds.get(int(p), 1.0) for p in y_pred])
    thresh_arr = np.where(thresh_arr == 0, 1e-9, thresh_arr)

    scores = raw / thresh_arr
    unknown_mask = scores > 1.0
    return scores, unknown_mask


def label_with_confidence(
    y_pred: np.ndarray,
    unknown_mask: np.ndarray,
    label_encoder,
    unknown_label: str = "Unknown / New Cell Type",
) -> np.ndarray:
    """Decodifica label RF e sovrascrive Unknown dove indicato dalla maschera."""
    decoded = label_encoder.inverse_transform(y_pred).astype(object)
    decoded[unknown_mask] = unknown_label
    return decoded
