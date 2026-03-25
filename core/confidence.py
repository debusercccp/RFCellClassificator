"""
core/confidence.py
------------------
Confidence score basato su distanza euclidea cellula → centroide classe.
Soglia adattiva per classe (95° percentile intra-classe sul training set).

v2.1: euclidean_confidence completamente vettorizzata — nessun loop Python.
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

    Implementazione completamente vettorizzata:
    - Centroid matrix (n_classes, n_features)
    - Gather dei centroidi per y_pred con advanced indexing
    - Distanza euclidea con np.linalg.norm(axis=1)
    - Nessun loop Python → scala a 50k+ cellule senza overhead

    Returns
    -------
    scores       : float array, score normalizzato (0=vicino, >1=Unknown)
    unknown_mask : bool array, True dove la cellula è Unknown
    """
    if not (0.0 <= percentile_threshold <= 100.0):
        raise ValueError(
            f"percentile_threshold deve essere tra 0 e 100, ricevuto: {percentile_threshold}"
        )

    y_pred = np.asarray(y_pred, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)

    unique_labels = np.unique(y_train)
    n_classes = int(unique_labels.max()) + 1
    n_features = X_train.shape[1]

    if len(unique_labels) == 0:
        scores = np.full(len(y_pred), np.inf, dtype=np.float32)
        return scores, np.ones(len(y_pred), dtype=bool)

    # ── Centroid matrix (n_classes, n_features) ────────────────────────
    # Righe non usate (classi assenti) restano a zero — non vengono mai
    # accedute perché y_pred contiene solo label presenti in y_train.
    centroid_matrix = np.zeros((n_classes, n_features), dtype=np.float32)
    for label in unique_labels:
        centroid_matrix[int(label)] = X_train[y_train == label].mean(axis=0)

    # ── Threshold per classe (percentile intra-classe) ─────────────────
    # Vettorizzato per ogni classe: distanze intra-classe calcolate in batch
    threshold_arr = np.full(n_classes, np.inf, dtype=np.float32)
    for label in unique_labels:
        lbl = int(label)
        mask = y_train == lbl
        if mask.sum() == 0:
            continue
        # (n_samples_class, n_features) - (1, n_features) → norma per riga
        intra_dists = np.linalg.norm(X_train[mask] - centroid_matrix[lbl], axis=1)
        threshold_arr[lbl] = float(np.percentile(intra_dists, percentile_threshold))

    # ── Gather centroidi predetti: (n_test, n_features) ───────────────
    # y_pred come indice di riga → nessun loop
    pred_centroids = centroid_matrix[y_pred]          # (n_test, n_features)

    # ── Distanza euclidea vettorizzata ─────────────────────────────────
    raw_distances = np.linalg.norm(X_test - pred_centroids, axis=1)  # (n_test,)

    # ── Gather soglie predette ─────────────────────────────────────────
    pred_thresholds = threshold_arr[y_pred]           # (n_test,)
    pred_thresholds = np.where(pred_thresholds == 0, 1e-9, pred_thresholds)

    scores = raw_distances / pred_thresholds
    unknown_mask = scores > 1.0
    return scores.astype(np.float32), unknown_mask


def label_with_confidence(
    y_pred: np.ndarray,
    unknown_mask: np.ndarray,
    label_encoder,
    unknown_label: str = "Unknown / New Cell Type",
) -> np.ndarray:
    """Decodifica label RF e sovrascrive Unknown dove indicato dalla maschera."""
    y_pred = np.asarray(y_pred)
    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    decoded = np.empty(shape=y_pred.shape, dtype=object)
    decoded[:] = unknown_label
    valid = y_pred >= 0
    if np.any(valid):
        decoded[valid] = label_encoder.inverse_transform(y_pred[valid]).astype(object)
    decoded[unknown_mask] = unknown_label
    return decoded
