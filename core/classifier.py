"""
core/classifier.py
------------------
Pipeline semi-supervisionata completa:
  1. Lloyd KMeans clustering
  2. Marker gene selection (riduzione feature space)
  3. Random Forest training
  4. Consensus check RF ↔ Cluster
  5. Euclidean confidence → Unknown labeling
"""

from __future__ import annotations

import numpy as np
import anndata as ad
from dataclasses import dataclass, field
from typing import Optional, Callable
from scipy.sparse import issparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from .clustering import cluster_anndata, select_marker_genes, ClusterResult
from .confidence import euclidean_confidence


@dataclass
class PipelineConfig:
    # Clustering
    n_clusters: int = 10
    cluster_use_rep: str = "X_pca"
    marker_n_top: int = 200
    marker_method: str = "variance_ratio"   # "variance_ratio" | "mean_diff"
    # Random Forest
    n_estimators: int = 100
    max_depth: Optional[int] = None
    criterion: str = "gini"
    n_jobs: int = -1
    random_state: int = 42
    test_size: float = 0.2
    # Confidence
    confidence_percentile: float = 95
    enable_consensus: bool = True
    enable_confidence: bool = True


@dataclass
class PipelineResult:
    # Predizioni
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    y_test: np.ndarray
    test_indices: np.ndarray
    # Metriche
    confusion_mat: np.ndarray
    class_report: str
    class_report_dict: dict
    feature_importances: np.ndarray
    selected_genes: list[str]
    marker_scores: np.ndarray
    # Clustering
    cluster_result: ClusterResult
    # Consensus
    consensus_flags: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    # Confidence
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    unknown_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    # Modello
    rf_model: Optional[RandomForestClassifier] = None
    label_encoder: Optional[LabelEncoder] = None
    # Statistiche pipeline
    n_genes_original: int = 0
    n_genes_selected: int = 0
    n_conflicts: int = 0
    n_unknown: int = 0


ProgressCB = Callable[[str, int], None]


class SemiSupervisedPipeline:
    """
    Uso:
        cfg = PipelineConfig(n_clusters=10, marker_n_top=200)
        pipeline = SemiSupervisedPipeline(cfg)
        result = pipeline.fit_predict(adata, label_key="cell_type",
                                      progress_callback=cb)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._result: Optional[PipelineResult] = None

    def fit_predict(
        self,
        adata: ad.AnnData,
        label_key: str = "cell_type",
        progress_callback: Optional[ProgressCB] = None,
    ) -> PipelineResult:
        cfg = self.config
        cb = progress_callback or (lambda s, p: None)

        # ── 1. Clustering ──────────────────────────────────────────────
        cb("Clustering (Lloyd KMeans)", 5)
        cluster_result = cluster_anndata(
            adata, k=cfg.n_clusters, use_rep=cfg.cluster_use_rep,
            max_iter=200, random_state=cfg.random_state,
            progress_callback=progress_callback,
        )

        # ── 2. Marker gene selection ───────────────────────────────────
        cb("Selezione marker genes", 28)
        n_genes_original = adata.n_vars
        selected_genes, marker_scores = select_marker_genes(
            adata, cluster_labels=cluster_result.labels,
            n_top=cfg.marker_n_top, method=cfg.marker_method,
        )
        cluster_result.marker_genes = selected_genes
        cluster_result.marker_scores = marker_scores

        # Subset AnnData sui soli marker genes
        adata_sub = adata[:, selected_genes]
        mat = adata_sub.X
        X_full = (mat.toarray() if issparse(mat) else mat).astype(np.float32)

        # ── 3. Label encoding + split ──────────────────────────────────
        cb("Preparazione dati RF", 38)
        le = LabelEncoder()
        y_full = le.fit_transform(adata.obs[label_key].values)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_full, y_full, np.arange(len(y_full)),
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y_full,
        )

        # ── 4. Training RF ─────────────────────────────────────────────
        cb(f"Training RF ({cfg.n_estimators} alberi, {cfg.criterion})", 45)
        rf = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            criterion=cfg.criterion,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
        )
        rf.fit(X_train, y_train)

        # ── 5. Predizione + metriche ───────────────────────────────────
        cb("Predizione e metriche", 72)
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(
            y_test, y_pred, target_names=le.classes_,
            zero_division=0, output_dict=True,
        )
        report_str = classification_report(
            y_test, y_pred, target_names=le.classes_, zero_division=0,
        )

        # ── 6. Consensus check ─────────────────────────────────────────
        consensus_flags = np.zeros(len(y_test), dtype=bool)
        if cfg.enable_consensus:
            cb("Consensus RF ↔ Cluster", 82)
            consensus_flags = self._consensus_check(
                rf_labels=y_pred,
                cluster_labels=cluster_result.labels[idx_test],
                true_labels=y_test,
                k=cfg.n_clusters,
            )
            # Salva nell'AnnData per ispezione esterna
            flag_full = np.zeros(len(adata), dtype=bool)
            flag_full[idx_test] = consensus_flags
            adata.obs["consensus_conflict"] = flag_full

        # ── 7. Confidence score ────────────────────────────────────────
        confidence_scores = np.zeros(len(y_test))
        unknown_mask = np.zeros(len(y_test), dtype=bool)
        if cfg.enable_confidence:
            cb("Confidence score euclideo", 90)
            confidence_scores, unknown_mask = euclidean_confidence(
                X_test=X_test,
                y_pred=y_pred,
                X_train=X_train,
                y_train=y_train,
                percentile_threshold=cfg.confidence_percentile,
            )
            adata.obs["is_unknown"] = False
            adata.obs.loc[adata.obs.index[idx_test[unknown_mask]], "is_unknown"] = True

        cb("Completato", 100)

        self._result = PipelineResult(
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            y_test=y_test,
            test_indices=idx_test,
            confusion_mat=cm,
            class_report=report_str,
            class_report_dict=report_dict,
            feature_importances=rf.feature_importances_,
            selected_genes=selected_genes,
            marker_scores=marker_scores,
            cluster_result=cluster_result,
            consensus_flags=consensus_flags,
            confidence_scores=confidence_scores,
            unknown_mask=unknown_mask,
            rf_model=rf,
            label_encoder=le,
            n_genes_original=n_genes_original,
            n_genes_selected=len(selected_genes),
            n_conflicts=int(consensus_flags.sum()),
            n_unknown=int(unknown_mask.sum()),
        )
        return self._result

    def _consensus_check(
        self,
        rf_labels: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Per ogni cluster, trova il cell type dominante (ground truth).
        Conflitto = RF predice X, ma il cluster è dominato da Y ≠ X.
        Indica errori di annotazione o cellule in stato di transizione.
        """
        cluster_dominant: dict[int, int] = {}
        for c in range(k):
            mask = cluster_labels == c
            if mask.sum() == 0:
                continue
            counts = np.bincount(true_labels[mask])
            cluster_dominant[c] = int(counts.argmax())

        return np.array([
            rf_labels[i] != cluster_dominant.get(int(cluster_labels[i]), int(rf_labels[i]))
            for i in range(len(rf_labels))
        ], dtype=bool)

    @property
    def result(self) -> Optional[PipelineResult]:
        return self._result
