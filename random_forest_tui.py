#!/usr/bin/env python3
"""
TREE / RANDOM FOREST TUI  —  v2.0 (Semi-Supervised Edition)
Classificazione di tipi cellulari da espressione genica
Formati: CSV · TSV · Excel · H5AD (AnnData / CELLxGENE)
Novità v2.0:
  [S]  Pipeline semi-supervisionata (Clustering → Marker Genes → RF)
  [K]  Clustering Lloyd KMeans standalone
  [C]  Confidence score euclideo + Unknown labeling
"""
import os
import sys
import pickle
import random
import argparse
import threading
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, track,
)
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.rule import Rule
from rich.columns import Columns
from rich import box
# ── Moduli semi-supervisionati ─────────────────────────────────────────────
try:
    from core import (
        SemiSupervisedPipeline, PipelineConfig, PipelineResult,
        cluster_anndata, select_marker_genes,
        euclidean_confidence, label_with_confidence,
    )
    SEMI_AVAILABLE = True
except ImportError:
    SEMI_AVAILABLE = False
console = Console()
# ══════════════════════════════════════════════════════════════════════════════
# Utilità path
# ══════════════════════════════════════════════════════════════════════════════
def normalize_path(path: str) -> str:
    if path is None:
        return path
    p = os.path.expandvars(str(path).strip())
    p = os.path.expanduser(p)
    return os.path.abspath(p)
def resolve_dataset_path_input(path_in: str) -> str:
    p = normalize_path(path_in)
    if not p:
        return p
    if os.path.isdir(p):
        exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".h5ad", ".h5", ".hdf5"}
        files: list[str] = []
        for name in os.listdir(p):
            fp = os.path.join(p, name)
            if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in exts:
                files.append(fp)
        files.sort()
        if not files:
            raise FileNotFoundError(
                f"Nessun dataset supportato trovato in directory: {p}. "
                f"Estensioni attese: {sorted(exts)}"
            )
        console.print(f"[dim]Dataset trovati in directory:[/dim] {p}")
        for i, fp in enumerate(files):
            console.print(f"  [{i}] {os.path.basename(fp)}")
        idx = IntPrompt.ask("Indice dataset da usare", default=0)
        idx = max(0, min(idx, len(files) - 1))
        return files[idx]
    return p
def resolve_h5ad_path_input(path_in: str) -> str:
    p = normalize_path(path_in)
    if not p:
        return p
    if os.path.isdir(p):
        files: list[str] = []
        for name in os.listdir(p):
            fp = os.path.join(p, name)
            if os.path.isfile(fp) and name.lower().endswith(".h5ad"):
                files.append(fp)
        files.sort()
        if not files:
            raise FileNotFoundError(f"Nessun file .h5ad trovato in directory: {p}")
        console.print(f"[dim]File .h5ad trovati in directory:[/dim] {p}")
        for i, fp in enumerate(files):
            console.print(f"  [{i}] {os.path.basename(fp)}")
        idx = IntPrompt.ask("Indice file .h5ad da usare", default=0)
        idx = max(0, min(idx, len(files) - 1))
        return files[idx]
    return p
# ══════════════════════════════════════════════════════════════════════════════
# Colori cell type
# ══════════════════════════════════════════════════════════════════════════════
CELL_COLORS = [
    "bold red", "bold green", "bold blue", "bold yellow",
    "bold magenta", "bold cyan", "bold white", "bright_red",
    "bright_green", "bright_blue", "bright_magenta", "bright_cyan",
    "orange3", "deep_pink4", "dark_cyan", "gold1", "purple", "chartreuse3",
]
class CellColorMapper:
    def __init__(self):
        self.mapping: dict[str, str] = {}
        self._idx = 0
    def get(self, label: str) -> str:
        label = str(label)
        if label not in self.mapping:
            self.mapping[label] = CELL_COLORS[self._idx % len(CELL_COLORS)]
            self._idx += 1
        return self.mapping[label]
    def legend(self):
        table = Table(title="Legenda Tipi Cellulari", box=box.ROUNDED, border_style="dim")
        table.add_column("Tipo Cellulare", style="bold")
        table.add_column("Colore")
        for label, color in self.mapping.items():
            table.add_row(Text(label, style=color), color)
        console.print(table)
color_map = CellColorMapper()
# ══════════════════════════════════════════════════════════════════════════════
# Banner
# ══════════════════════════════════════════════════════════════════════════════
def print_banner():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]TREE / RANDOM FOREST TUI[/bold cyan]  [dim]v2.0 Semi-Supervised[/dim]\n"
        "[dim]Classificazione di tipi cellulari da espressione genica[/dim]\n"
        "[dim]Formati: CSV · TSV · Excel · .h5ad (AnnData / CELLxGENE)[/dim]",
        border_style="cyan", padding=(1, 4),
    ))
    console.print()
# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dataset
# ══════════════════════════════════════════════════════════════════════════════
def load_h5ad(path, label_col=None):
    path = normalize_path(path)
    if not path:
        raise FileNotFoundError("Path .h5ad vuoto")
    if os.path.isdir(path):
        raise IsADirectoryError(f"Path è una directory (atteso file .h5ad): {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File .h5ad non trovato: {path}")
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata non installata. Installa con: pip install anndata")
    console.print(f"[dim]Lettura AnnData:[/dim] [yellow]{path}[/yellow]")
    adata = ad.read_h5ad(path)
    console.print(f"[green]Letto:[/green] {adata.n_obs} cellule x {adata.n_vars} geni")
    obs_cols = list(adata.obs.columns)
    if not obs_cols:
        raise ValueError("Nessuna colonna obs trovata in AnnData")
    if label_col is None or label_col not in obs_cols:
        console.print("\n[bold]Colonne metadati (obs):[/bold]")
        for i, col in enumerate(obs_cols):
            console.print(f"  [{i}] {col} [dim]({adata.obs[col].nunique()} unici)[/dim]")
        idx = IntPrompt.ask("Indice colonna target", default=0)
        label_col = obs_cols[idx]
    console.print(f"[green]Target:[/green] [bold]{label_col}[/bold]")
    console.print("[dim]Conversione matrice...[/dim]")
    import scipy.sparse as sp
    X_mat = adata.X
    if sp.issparse(X_mat):
        X_mat = X_mat.toarray()
    X = pd.DataFrame(X_mat, columns=adata.var_names)
    y = adata.obs[label_col].astype(str).values
    _print_distribution(y)
    return X, pd.Series(y), list(adata.var_names)
def load_dataset(path, label_col=None):
    path = normalize_path(path)
    if not path:
        raise FileNotFoundError("Path dataset vuoto")
    if os.path.isdir(path):
        raise IsADirectoryError(f"Path è una directory (atteso file): {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset non trovato: {path}")
    console.print(f"[dim]Caricamento:[/dim] [yellow]{path}[/yellow]")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5ad", ".h5", ".hdf5"):
        return load_h5ad(path, label_col)
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Formato dataset non supportato: {ext}")
    console.print(f"[green]Caricato:[/green] {df.shape[0]} campioni, {df.shape[1]} colonne")
    if label_col is None:
        console.print("\n[bold]Colonne:[/bold]")
        for i, col in enumerate(df.columns):
            console.print(f"  [{i}] {col}")
        idx = IntPrompt.ask("Indice colonna target", default=df.shape[1] - 1)
        label_col = df.columns[idx]
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(str)
    _print_distribution(y.values)
    for cls in y.unique():
        color_map.get(str(cls))
    if isinstance(X, pd.DataFrame):
        bad_cols = []
        X_num = X.copy()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                continue
            converted = pd.to_numeric(X[col], errors="coerce")
            non_null_mask = X[col].notna()
            if (non_null_mask & converted.isna()).any():
                bad_cols.append(col)
            X_num[col] = converted
        if bad_cols:
            raise ValueError(
                "Le feature contengono valori non numerici. Probabilmente hai selezionato "
                "come target una colonna sbagliata (es. i label tipo 'CD4 T cell' sono rimasti in X). "
                f"Colonne problematiche: {bad_cols[:10]}"
            )
        X = X_num
    return X, y, list(X.columns)
def _print_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    table = Table(title="Distribuzione classi cellulari", box=box.ROUNDED)
    table.add_column("Tipo Cellulare")
    table.add_column("N", justify="right")
    table.add_column("%", justify="right")
    for cls, cnt in zip(unique, counts):
        table.add_row(Text(str(cls), style=color_map.get(str(cls))), str(cnt), f"{cnt/len(y)*100:.1f}%")
    console.print(table)
# ══════════════════════════════════════════════════════════════════════════════
# Training RF classico
# ══════════════════════════════════════════════════════════════════════════════
def ask_criterion():
    console.print("\n[bold]Criterio di impurità:[/bold]")
    console.print("  [cyan]1[/cyan] - [bold]Gini[/bold]     impurità di Gini (default, veloce)")
    console.print("  [cyan]2[/cyan] - [bold]Entropy[/bold]  information gain, entropia di Shannon")
    console.print("  [cyan]3[/cyan] - [bold]Log-loss[/bold] cross-entropy (sklearn >= 1.1)")
    choice = Prompt.ask("Scelta", choices=["1", "2", "3"], default="1")
    return {"1": "gini", "2": "entropy", "3": "log_loss"}[choice]
def train_model(X, y, n_trees=100, max_depth=None, criterion="gini"):
    console.print(f"\n[bold cyan]Training: {n_trees} alberi | criterio: [yellow]{criterion}[/yellow][/bold cyan]")
    if isinstance(X, pd.DataFrame):
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            raise ValueError(
                "Dataset non valido: alcune feature non sono numeriche. "
                "Riprova scegliendo la colonna target corretta (label) e assicurati "
                "che le feature siano solo numeri."
            )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(
        n_estimators=n_trees, max_depth=max_depth,
        criterion=criterion, random_state=42, n_jobs=-1,
    )
    for _ in track(range(1), description="Allenamento..."):
        try:
            model.fit(X_train, y_train)
        except ValueError as e:
            raise ValueError(
                f"Errore in training (features non convertibili): {e}. "
                "Probabilmente hai lasciato i label (stringhe) in X. Ricarica dataset e seleziona correttamente la colonna target."
            ) from e
    console.print("[green]Training completato![/green]")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    table = Table(title=f"Risultati test set (20%) - {criterion}", box=box.ROUNDED)
    table.add_column("Classe", style="bold")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")
    for cls, m in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        table.add_row(
            Text(cls, style=color_map.get(cls)),
            f"{m['precision']:.2f}", f"{m['recall']:.2f}",
            f"{m['f1-score']:.2f}", str(int(m['support'])),
        )
    console.print(table)
    console.print(f"[bold]Accuracy:[/bold] [green]{report['accuracy']:.2%}[/green]")
    return model
# ══════════════════════════════════════════════════════════════════════════════
# Visualizzazione alberi
# ══════════════════════════════════════════════════════════════════════════════
def _align_feature_names(model_or_tree, feature_names):
    if hasattr(model_or_tree, "estimators_"):
        n = model_or_tree.estimators_[0].tree_.n_features
    else:
        n = model_or_tree.tree_.n_features
    fn = list(feature_names) if feature_names else []
    return fn if len(fn) == n else [f"feature_{i}" for i in range(n)]
def show_tree(model, feature_names, mode="random", tree_idx=None):
    n = len(model.estimators_)
    if mode == "random":
        tree_idx = random.randint(0, n - 1)
        console.print(f"\n[dim]Albero random:[/dim] #{tree_idx}")
    else:
        if tree_idx is None:
            tree_idx = IntPrompt.ask(f"Indice albero (0-{n-1})", default=0)
        tree_idx = max(0, min(tree_idx, n - 1))
    fn = _align_feature_names(model, feature_names)
    text = export_text(model.estimators_[tree_idx], feature_names=fn, max_depth=4)
    console.print(Panel(
        f"[green]{text}[/green]",
        title=f"Albero #{tree_idx} | {model.criterion}",
        border_style="green", expand=False,
    ))
def show_node_impurity(model, feature_names):
    n = len(model.estimators_)
    tree_idx = IntPrompt.ask(f"Indice albero (0-{n-1})", default=0)
    tree_idx = max(0, min(tree_idx, n - 1))
    t = model.estimators_[tree_idx].tree_
    fn = _align_feature_names(model, feature_names)
    criterion = model.criterion
    label = {"gini": "Gini", "entropy": "Entropia", "log_loss": "Log-loss"}.get(criterion, criterion)
    table = Table(title=f"Impurità nodi - Albero #{tree_idx} | {label}", box=box.ROUNDED)
    table.add_column("Nodo", justify="right", style="dim")
    table.add_column("Gene / Feature")
    table.add_column("Soglia expr.", justify="right")
    table.add_column(label, justify="right")
    table.add_column("Campioni", justify="right")
    table.add_column("Tipo")
    for i in range(t.node_count):
        is_leaf = t.children_left[i] == t.children_right[i]
        imp = t.impurity[i]
        imp_style = "green" if imp < 0.2 else "yellow" if imp < 0.4 else "red"
        if is_leaf:
            table.add_row(str(i), "-", "-", Text(f"{imp:.4f}", style=imp_style),
                          str(t.n_node_samples[i]), Text("foglia", style="dim"))
        else:
            fname = fn[t.feature[i]] if t.feature[i] < len(fn) else f"f{t.feature[i]}"
            table.add_row(str(i), Text(fname, style="cyan"), f"{t.threshold[i]:.4f}",
                          Text(f"{imp:.4f}", style=imp_style), str(t.n_node_samples[i]),
                          Text("split", style="bold white"))
    console.print(table)
    console.print("[dim]Colori:[/dim] [green]< 0.2 bassa[/green]  [yellow]0.2-0.4 media[/yellow]  [red]> 0.4 alta[/red]")
def show_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    fn = _align_feature_names(model, feature_names)
    if len(fn) != len(importances):
        fn = [f"feature_{i}" for i in range(len(importances))]
    std = np.std([t.feature_importances_ for t in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:top_n]
    criterion = model.criterion
    label = {"gini": "Gini MDI", "entropy": "Entropy MDI", "log_loss": "Log-loss MDI"}.get(criterion, "MDI")
    table = Table(title=f"Top {top_n} geni - {label}", box=box.ROUNDED)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Gene")
    table.add_column("Importanza MDI", justify="right")
    table.add_column("Marker Score", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Barra")
    max_imp = importances[indices[0]] if len(indices) else 1.0
    marker_score_map: dict[str, float] = {}
    if hasattr(model, "_marker_scores") and model._marker_scores is not None:
        for g, s in zip(model._marker_genes, model._marker_scores):
            marker_score_map[g] = float(s)
    for rank, idx in enumerate(indices, 1):
        idx = int(idx)
        imp = importances[idx]
        bar = "#" * int((imp / max_imp) * 30)
        bar_style = "gold1" if rank <= 3 else "cyan" if rank <= 10 else "dim cyan"
        gene = fn[idx]
        ms = f"{marker_score_map[gene]:.4f}" if gene in marker_score_map else "[dim]-[/dim]"
        table.add_row(
            str(rank), Text(gene, style="bold"),
            f"{imp:.5f}", ms, f"±{std[idx]:.5f}",
            Text(bar, style=bar_style),
        )
    console.print(table)
    console.print(f"[dim]MDI = riduzione media impurità ({criterion}) pesata per campioni.[/dim]")
    console.print(f"[dim]Marker Score = variance_ratio dal clustering (colonna extra v2.0).[/dim]")
def compare_criteria(X, y, n_trees=100, max_depth=None):
    console.print("\n[bold cyan]Confronto Gini vs Entropy...[/bold cyan]")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    for crit in ["gini", "entropy"]:
        m = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                   criterion=crit, random_state=42, n_jobs=-1)
        for _ in track(range(1), description=f"Training {crit}..."):
            m.fit(X_train, y_train)
        rep = classification_report(y_test, m.predict(X_test), output_dict=True)
        results[crit] = (m, rep)
    table = Table(title="Gini vs Entropy - F1 per classe", box=box.ROUNDED)
    table.add_column("Classe")
    table.add_column("Gini F1", justify="right")
    table.add_column("Entropy F1", justify="right")
    table.add_column("Migliore", justify="center")
    all_classes = {k for c in ["gini", "entropy"] for k in results[c][1]
                   if k not in ("accuracy", "macro avg", "weighted avg")}
    for cls in sorted(all_classes):
        f1g = results["gini"][1].get(cls, {}).get("f1-score", 0)
        f1e = results["entropy"][1].get(cls, {}).get("f1-score", 0)
        best = "gini" if f1g >= f1e else "entropy"
        table.add_row(
            Text(cls, style=color_map.get(cls)),
            f"{f1g:.3f}", f"{f1e:.3f}",
            Text(best, style="bold green"),
        )
    acc_g = results["gini"][1]["accuracy"]
    acc_e = results["entropy"][1]["accuracy"]
    best_acc = "gini" if acc_g >= acc_e else "entropy"
    table.add_row(
        Text("ACCURACY TOTALE", style="bold"),
        Text(f"{acc_g:.2%}", style="bold"),
        Text(f"{acc_e:.2%}", style="bold"),
        Text(best_acc, style="bold yellow"),
    )
    console.print(table)
    return results[best_acc][0]
def predict_test(model, feature_names, path=None, label_col=None):
    if path is None:
        path = Prompt.ask("Path del dataset di test")
    X_test, y_true, _ = load_dataset(path, label_col)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    table = Table(title="Predizioni dataset di test", box=box.ROUNDED)
    table.add_column("Campione", justify="right", style="dim")
    table.add_column("Predizione")
    table.add_column("Confidenza", justify="right")
    if y_true is not None:
        table.add_column("Reale")
    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        pred_str = str(pred)
        row = [str(i), Text(pred_str, style=color_map.get(pred_str)), f"{max(prob):.1%}"]
        if y_true is not None:
            real = str(y_true.iloc[i])
            match = "OK" if real == pred_str else "NO"
            row.append(Text(f"{match} {real}", style="green" if real == pred_str else "red"))
        table.add_row(*row)
    console.print(table)
# ══════════════════════════════════════════════════════════════════════════════
# NOVITÀ v2.0 — Pipeline Semi-Supervisionata
# ══════════════════════════════════════════════════════════════════════════════
def run_semi_supervised(current_adata=None):
    if not SEMI_AVAILABLE:
        console.print("[red]Moduli core/ non trovati.[/red] Assicurati che core/ sia nella stessa directory.")
        return None, None
    try:
        import anndata as ad
    except ImportError:
        console.print("[red]anndata non installata.[/red]")
        return None, None
    if current_adata is None:
        path_in = Prompt.ask("Path dataset H5AD (file o directory)")
        try:
            path = resolve_h5ad_path_input(path_in)
        except Exception as e:
            console.print(f"[red]Impossibile scegliere .h5ad:[/red] {type(e).__name__}: {e}")
            return None, None
        if not path:
            console.print("[red]Path .h5ad vuoto: annullo la pipeline.[/red]")
            return None, None
        try:
            console.print(f"[dim]Lettura:[/dim] [yellow]{path}[/yellow]")
            adata = ad.read_h5ad(path)
        except Exception as e:
            console.print(f"[red]Impossibile leggere .h5ad:[/red] {type(e).__name__}: {e}")
            return None, None
    else:
        adata = current_adata
        console.print(f"[green]Uso AnnData già caricato:[/green] {adata.n_obs} cellule × {adata.n_vars} geni")
    obs_cols = list(adata.obs.columns)
    console.print("\n[bold]Colonne disponibili:[/bold]")
    for i, col in enumerate(obs_cols):
        console.print(f"  [{i}] {col} [dim]({adata.obs[col].nunique()} unici)[/dim]")
    idx = IntPrompt.ask("Colonna target (cell type)", default=0)
    label_key = obs_cols[idx]
    console.print("\n[bold cyan]Configurazione pipeline semi-supervisionata:[/bold cyan]")
    n_clusters = IntPrompt.ask("Numero di cluster KMeans", default=10)
    n_markers   = IntPrompt.ask("Marker genes da selezionare", default=200)
    n_trees     = IntPrompt.ask("Alberi Random Forest", default=100)
    criterion   = ask_criterion()
    md = Prompt.ask("Profondità massima RF (invio = illimitata)", default="")
    max_depth = int(md) if md.strip() else None
    conf_pct = IntPrompt.ask("Percentile soglia confidence (0-100)", default=95)
    cfg = PipelineConfig(
        n_clusters=n_clusters,
        marker_n_top=n_markers,
        marker_method="variance_ratio",
        n_estimators=n_trees,
        max_depth=max_depth,
        criterion=criterion,
        confidence_percentile=float(conf_pct),
        enable_consensus=True,
        enable_confidence=True,
    )
    pipeline = SemiSupervisedPipeline(cfg)
    result_holder: list = []
    error_holder:  list = []
    progress_status = {"step": "Avvio...", "pct": 0}
    def cb(step: str, pct: int):
        progress_status["step"] = step
        progress_status["pct"] = pct
    def worker():
        try:
            r = pipeline.fit_predict(adata, label_key=label_key, progress_callback=cb)
            result_holder.append(r)
        except Exception as e:
            error_holder.append(e)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Pipeline semi-supervisionata", total=100)
        while t.is_alive():
            progress.update(task, completed=progress_status["pct"],
                            description=progress_status["step"])
            time.sleep(0.15)
        progress.update(task, completed=100, description="Completato")
    if error_holder:
        console.print(f"[red]Errore durante la pipeline:[/red] {error_holder[0]}")
        import traceback; traceback.print_exc()
        return None, None
    result: PipelineResult = result_holder[0]
    _print_pipeline_summary(result)
    rf = result.rf_model
    rf._marker_genes = result.selected_genes
    rf._marker_scores = result.marker_scores
    return rf, result.selected_genes, result

def _print_pipeline_summary(result: "PipelineResult"):
    le = result.label_encoder
    report = result.class_report_dict
    ratio = result.n_genes_original / max(result.n_genes_selected, 1)
    summary = Panel(
        f"[bold]Geni originali:[/bold]  {result.n_genes_original:>6}\n"
        f"[bold]Marker genes:[/bold]    {result.n_genes_selected:>6}  [dim](riduzione {ratio:.0f}x)[/dim]\n"
        f"[bold]Cluster trovati:[/bold] {result.cluster_result.n_iter:>6} iter → "
        f"{len(np.unique(result.cluster_result.labels))} cluster\n"
        f"[bold]Conflitti RF↔K:[/bold] {result.n_conflicts:>6}  "
        f"[dim]({100*result.n_conflicts/max(len(result.y_pred),1):.1f}% test set)[/dim]\n"
        f"[bold]Unknown cells:[/bold]  {result.n_unknown:>6}  "
        f"[dim]({100*result.n_unknown/max(len(result.y_pred),1):.1f}% test set)[/dim]",
        title="[bold cyan]Pipeline Semi-Supervisionata — Riepilogo[/bold cyan]",
        border_style="cyan",
        padding=(0, 2),
    )
    console.print(summary)
    table = Table(title="Risultati RF su Marker Genes", box=box.ROUNDED)
    table.add_column("Classe", style="bold")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")
    for cls, m in report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        table.add_row(
            Text(cls, style=color_map.get(cls)),
            f"{m['precision']:.2f}", f"{m['recall']:.2f}",
            f"{m['f1-score']:.2f}", str(int(m['support'])),
        )
    console.print(table)
    console.print(f"[bold]Accuracy:[/bold] [green]{report['accuracy']:.2%}[/green]")
    _print_marker_genes(result.selected_genes[:15], result.marker_scores[:15])
    if result.n_conflicts > 0:
        _print_consensus_report(result, le)
    _print_confusion_matrix(result.confusion_mat, le.classes_)

def _print_marker_genes(genes: list[str], scores: np.ndarray):
    table = Table(title="Top Marker Genes (Clustering → RF)", box=box.SIMPLE)
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Gene", style="bold cyan")
    table.add_column("Variance Ratio Score", justify="right")
    table.add_column("Barra")
    max_s = scores[0] if len(scores) > 0 else 1.0
    for i, (g, s) in enumerate(zip(genes, scores), 1):
        bar = "█" * int((s / max_s) * 20)
        table.add_row(str(i), g, f"{s:.5f}", Text(bar, style="gold1"))
    console.print(table)

def _print_consensus_report(result: "PipelineResult", le: LabelEncoder):
    n_total = len(result.y_pred)
    conflict_idx = np.where(result.consensus_flags)[0]
    table = Table(
        title=f"[yellow]⚠  Consensus Conflicts RF ↔ Cluster[/yellow]  "
              f"[dim]({result.n_conflicts}/{n_total} celle)[/dim]",
        box=box.SIMPLE,
    )
    table.add_column("Cella (idx)", style="dim", justify="right")
    table.add_column("RF Predizione")
    table.add_column("Conf. Score", justify="right")
    table.add_column("Nota")
    for i in conflict_idx[:20]:
        rf_label = le.inverse_transform([result.y_pred[i]])[0]
        conf = result.confidence_scores[i] if len(result.confidence_scores) > 0 else 0.0
        conf_style = "red" if conf > 1.0 else "yellow"
        nota = "[red]Unknown[/red]" if conf > 1.0 else "[yellow]Borderline[/yellow]"
        table.add_row(
            str(result.test_indices[i]),
            Text(rf_label, style=color_map.get(rf_label)),
            Text(f"{conf:.2f}", style=conf_style),
            nota,
        )
    if len(conflict_idx) > 20:
        console.print(f"[dim]... e altri {len(conflict_idx)-20} conflitti[/dim]")
    console.print(table)
    console.print(
        "[dim]Un conflitto indica una cellula dove RF e cluster "
        "dominante disaccordano — potenziale errore di annotazione "
        "o cellula in stato di transizione.[/dim]"
    )
def _print_confusion_matrix(cm: np.ndarray, class_names: np.ndarray):
    n = len(class_names)
    if n > 10:
        console.print(f"[dim]Confusion matrix {n}×{n} — troppo grande per la TUI, "
                      f"salva con [S] e visualizza con matplotlib.[/dim]")
        return
    table = Table(title="Confusion Matrix", box=box.ROUNDED)
    table.add_column("Reale \\ Pred", style="bold")
    for cls in class_names:
        table.add_column(cls[:12], justify="right")
    for i, cls in enumerate(class_names):
        row_vals = []
        for j in range(n):
            val = cm[i][j]
            style = "bold green" if i == j and val > 0 else ("red" if val > 0 else "dim")
            row_vals.append(Text(str(val), style=style))
        table.add_row(Text(cls[:12], style=color_map.get(cls)), *row_vals)
    console.print(table)
# ══════════════════════════════════════════════════════════════════════════════
# NOVITÀ v2.0 — Clustering Standalone
# ══════════════════════════════════════════════════════════════════════════════
def run_clustering_standalone():
    if not SEMI_AVAILABLE:
        console.print("[red]Moduli core/ non trovati.[/red]")
        return
    try:
        import anndata as ad
    except ImportError:
        console.print("[red]anndata non installata.[/red]")
        return
    path_in = Prompt.ask("Path dataset H5AD (file o directory)")
    try:
        path = resolve_h5ad_path_input(path_in)
    except Exception as e:
        console.print(f"[red]Impossibile scegliere .h5ad:[/red] {type(e).__name__}: {e}")
        return
    if not path:
        console.print("[red]Path .h5ad vuoto: annullo clustering.[/red]")
        return
    try:
        adata = ad.read_h5ad(path)
        console.print(f"[green]Letto:[/green] {adata.n_obs} cellule × {adata.n_vars} geni")
    except Exception as e:
        console.print(f"[red]Impossibile leggere .h5ad:[/red] {type(e).__name__}: {e}")
        return
    k = IntPrompt.ask("Numero di cluster", default=10)
    n_markers = IntPrompt.ask("Marker genes da mostrare", default=20)
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  BarColumn(), TimeElapsedColumn(), console=console, transient=True) as prog:
        task = prog.add_task("Lloyd KMeans...", total=None)
        result = cluster_anndata(adata, k=k, max_iter=200)
        prog.update(task, completed=True)
    console.print(f"\n[green]Clustering completato:[/green] {result.n_iter} iterazioni | "
                  f"inertia: {result.inertia:.2f}")
    unique, counts = np.unique(result.labels, return_counts=True)
    table = Table(title=f"Distribuzione {k} Cluster (Lloyd KMeans)", box=box.ROUNDED)
    table.add_column("Cluster", justify="right")
    table.add_column("Cellule", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Barra")
    max_c = counts.max()
    for cl, cnt in zip(unique, counts):
        bar = "█" * int((cnt / max_c) * 25)
        table.add_row(str(cl), str(cnt), f"{cnt/len(result.labels)*100:.1f}%",
                      Text(bar, style="cyan"))
    console.print(table)
    console.print("\n[dim]Selezione marker genes...[/dim]")
    genes, scores = select_marker_genes(adata, result.labels, n_top=n_markers)
    _print_marker_genes(genes[:n_markers], scores[:n_markers])
    console.print(f"\n[dim]Label salvate in adata.obs['lloyd_cluster'][/dim]")
# ══════════════════════════════════════════════════════════════════════════════
# NOVITÀ v2.0 — Confidence Score su modello esistente
# ══════════════════════════════════════════════════════════════════════════════
def run_confidence_check(model, feature_names, X=None, y=None):
    if not SEMI_AVAILABLE:
        console.print("[red]Moduli core/ non trovati.[/red]")
        return
    if model is None:
        console.print("[red]Nessun modello caricato. Usa opzione 1 o 9.[/red]")
        return
    if X is None:
        console.print("[red]Nessun dataset caricato. Usa opzione 1 prima.[/red]")
        return
    console.print("\n[bold cyan]Confidence Score Euclideo[/bold cyan]")
    conf_pct = IntPrompt.ask("Percentile soglia (0-100)", default=95)
    feature_order = list(feature_names) if feature_names is not None else None
    if hasattr(X, "columns"):
        X_df = X.copy()
        if feature_order:
            X_df = X_df.loc[:, feature_order]
    else:
        if not feature_order:
            console.print("[red]Impossibile ricostruire l'ordine delle feature.[/red]")
            return
        X_df = pd.DataFrame(np.asarray(X), columns=feature_order)
    X_arr = X_df.values.astype(np.float32)
    y_arr = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_arr)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)
    y_pred = model.predict(X_test_df)
    try:
        y_pred_enc = le.transform(y_pred)
    except Exception:
        known = set(le.classes_)
        y_pred_enc = np.array([le.transform([lbl])[0] if lbl in known else -1 for lbl in y_pred], dtype=int)
    try:
        scores, unknown_mask = euclidean_confidence(
            X_test=X_test,
            y_pred=y_pred_enc,
            X_train=X_train,
            y_train=y_train,
            percentile_threshold=float(conf_pct),
        )
        final_labels = label_with_confidence(y_pred_enc, unknown_mask, le)
    except Exception as e:
        console.print(f"[red]Errore durante confidence: {type(e).__name__}: {e}[/red]")
        console.print("[dim]Suggerimento: prova di nuovo con un altro dataset o riduci il percentile.[/dim]")
        return
    n_unknown = unknown_mask.sum()
    console.print(Panel(
        f"[bold]Cellule analizzate:[/bold]  {len(y_pred)}\n"
        f"[bold]Unknown:[/bold]             {n_unknown}  "
        f"[dim]({100*n_unknown/len(y_pred):.1f}%)[/dim]\n"
        f"[bold]Soglia:[/bold]              {conf_pct}° percentile intra-classe",
        title="[bold]Risultati Confidence[/bold]", border_style="yellow",
    ))
    table = Table(title="Campione cellule Unknown (prime 20)", box=box.SIMPLE)
    table.add_column("Idx", style="dim", justify="right")
    table.add_column("RF Label")
    table.add_column("Final Label")
    table.add_column("Score", justify="right")

    unknown_idx = np.where(unknown_mask)[0]

    for i in unknown_idx[:20]:
        rf_label = str(y_pred[i]) 
        table.add_row(
            str(i),
            Text(rf_label, style=color_map.get(rf_label)),
            Text(str(final_labels[i]), style="red bold"), 
            Text(f"{scores[i]:.2f}", style="red"),
        )   

    if len(unknown_idx) > 20:
        console.print(f"[dim]... e altri {len(unknown_idx)-20} Unknown[/dim]")
    console.print(table)
    table2 = Table(title="Distribuzione score per classe", box=box.SIMPLE)
    table2.add_column("Classe")
    table2.add_column("Score medio", justify="right")
    table2.add_column("Unknown", justify="right")
    for label in np.unique(y_pred_enc):
        mask = y_pred_enc == label
        cls_name = le.inverse_transform([label])[0] if label >= 0 else "Unknown / New Cell Type"
        avg_score = scores[mask].mean()
        n_unk = unknown_mask[mask].sum()
        style = "red" if avg_score > 0.8 else "yellow" if avg_score > 0.5 else "green"
        table2.add_row(
            Text(cls_name, style=color_map.get(cls_name)),
            Text(f"{avg_score:.3f}", style=style),
            str(n_unk),
        )
    console.print(table2)
# ══════════════════════════════════════════════════════════════════════════════
# Save / Load modello
# ══════════════════════════════════════════════════════════════════════════════
def save_model(model, path="forest_model.pkl"):
    path = normalize_path(path)
    if not path:
        console.print("[red]Path salvataggio vuoto.[/red]")
        return False
    if os.path.isdir(path):
        console.print(f"[red]Il percorso di salvataggio è una directory:[/red] [yellow]{path}[/yellow]")
        return False
    if not path.lower().endswith(".pkl"):
        console.print("[yellow]Attenzione: il nome file non termina con .pkl.[/yellow] Proseguo comunque.")
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        console.print(f"[red]Directory di output non esistente:[/red] [yellow]{parent}[/yellow]")
        return False
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        console.print(f"[red]Impossibile salvare modello:[/red] {type(e).__name__}: {e}")
        return False
    console.print(f"[green]Modello salvato:[/green] [yellow]{path}[/yellow]")
    return True
def load_model(path):
    path = normalize_path(path)
    if not path:
        console.print("[red]Path modello vuoto.[/red]")
        return None
    if os.path.isdir(path):
        console.print(f"[red]Il percorso non è un file (è una directory):[/red] [yellow]{path}[/yellow]")
        return None
    if not os.path.exists(path):
        console.print(f"[red]File modello non trovato:[/red] [yellow]{path}[/yellow]")
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except (IsADirectoryError, FileNotFoundError):
        console.print(f"[red]Impossibile aprire il file modello:[/red] [yellow]{path}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Impossibile caricare il modello:[/red] {type(e).__name__}: {e}")
        return None
    console.print(f"[green]Modello caricato:[/green] [yellow]{path}[/yellow]")
    return model
# ══════════════════════════════════════════════════════════════════════════════
# Menu principale
# ══════════════════════════════════════════════════════════════════════════════
def main_menu(model=None, feature_names=None, X=None, y=None, adata=None):
    semi_result = None
    while True:
        console.print(Rule(style="dim"))
        console.print("[bold]Menu:[/bold]")
        console.print("  [cyan]1[/cyan]  Carica dataset e allena        [dim](CSV/TSV/Excel/H5AD)[/dim]")
        console.print("  [cyan]2[/cyan]  Albero random")
        console.print("  [cyan]3[/cyan]  Albero a scelta")
        console.print("  [cyan]4[/cyan]  Impurità nodi di un albero     [dim](Gini/Entropia/Log-loss)[/dim]")
        console.print("  [cyan]5[/cyan]  Feature importance             [dim](top geni MDI + Marker Score)[/dim]")
        console.print("  [cyan]6[/cyan]  Confronto Gini vs Entropy")
        console.print("  [cyan]7[/cyan]  Predici su dataset di test")
        console.print("  [cyan]8[/cyan]  Salva modello")
        console.print("  [cyan]9[/cyan]  Carica modello")
        console.print("  [cyan]L[/cyan]  Legenda colori")
        console.print("  [cyan]E[/cyan]  Converti .h5ad → CSV")
        console.print(Rule(characters="─", style="dim cyan"))
        semi_ok = "[green]✓[/green]" if SEMI_AVAILABLE else "[red]✗[/red]"
        console.print(f"  [bold cyan]S[/bold cyan]  Pipeline Semi-Supervisionata   "
                      f"[dim](Clustering → Marker Genes → RF → Consensus → Confidence)[/dim] {semi_ok}")
        console.print(f"  [bold cyan]K[/bold cyan]  Clustering Lloyd KMeans        "
                      f"[dim](standalone, visualizza cluster + marker genes)[/dim] {semi_ok}")
        console.print(f"  [bold cyan]C[/bold cyan]  Confidence Score euclideo      "
                      f"[dim](Unknown / New Cell Type labeling)[/dim] {semi_ok}")
        console.print("  [cyan]0[/cyan]  Esci")
        console.print()
        valid = ["0","1","2","3","4","5","6","7","8","9","L","l","E","e",
                 "S","s","K","k","C","c"]
        choice = Prompt.ask("Scelta", choices=valid).upper()
        if choice == "1":
            loaded = False
            for _attempt in range(3):
                path_in = Prompt.ask("Path dataset di training (file o directory)")
                try:
                    path = resolve_dataset_path_input(path_in)
                except Exception as e:
                    console.print(f"[red]Impossibile scegliere dataset:[/red] {type(e).__name__}: {e}")
                    path = normalize_path(path_in)
                try:
                    X_tmp, y_tmp, feature_names_tmp = load_dataset(path)
                    loaded = True
                    break
                except Exception as e:
                    console.print(f"[red]Impossibile caricare dataset:[/red] {type(e).__name__}: {e}")
            if not loaded:
                console.print("[red]Caricamento dataset fallito. Torno al menu.[/red]")
                continue
            n_trees = IntPrompt.ask("Numero di alberi", default=100)
            md = Prompt.ask("Profondità massima (invio = illimitata)", default="")
            max_depth = int(md) if md.strip() else None
            criterion = ask_criterion()
            X, y, feature_names = X_tmp, y_tmp, feature_names_tmp
            try:
                model = train_model(X, y, n_trees, max_depth, criterion)
            except ValueError as e:
                console.print(f"[red]Training fallito:[/red] {e}")
                model = None; feature_names = X = y = None
                continue
            if path.lower().endswith(".h5ad"):
                try:
                    import anndata as ad
                    adata = ad.read_h5ad(path)
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] lettura adata fallita: {type(e).__name__}: {e}")
        elif choice == "2":
            if model is None: console.print("[red]Nessun modello caricato.[/red]")
            else: show_tree(model, feature_names, mode="random")
        elif choice == "3":
            if model is None: console.print("[red]Nessun modello caricato.[/red]")
            else: show_tree(model, feature_names, mode="choose")
        elif choice == "4":
            if model is None: console.print("[red]Nessun modello caricato.[/red]")
            else: show_node_impurity(model, feature_names)
        elif choice == "5":
            if model is None: console.print("[red]Nessun modello caricato.[/red]")
            else:
                top_n = IntPrompt.ask("Quanti geni mostrare", default=20)
                show_feature_importance(model, feature_names, top_n)
        elif choice == "6":
            if X is None: console.print("[red]Carica prima un dataset (opzione 1).[/red]")
            else:
                n_trees = IntPrompt.ask("Numero di alberi", default=100)
                md = Prompt.ask("Profondità massima (invio = illimitata)", default="")
                max_depth = int(md) if md.strip() else None
                best = compare_criteria(X, y, n_trees, max_depth)
                if Prompt.ask("Usare il modello migliore?", choices=["s","n"], default="s") == "s":
                    model = best
        elif choice == "7":
            if model is None: console.print("[red]Nessun modello caricato.[/red]")
            else: predict_test(model, feature_names)
        elif choice == "8":
            if model is None: console.print("[red]Nessun modello caricato.[/red]")
            else:
                ok = False
                for _attempt in range(3):
                    path = Prompt.ask("Path salvataggio", default="forest_model.pkl")
                    ok = save_model(model, path)
                    if ok: break
                if not ok:
                    console.print("[red]Salvataggio fallito. Torno al menu.[/red]")
        elif choice == "9":
            model = None
            for _attempt in range(3):
                path = Prompt.ask("Path modello .pkl", default="forest_model.pkl")
                path_n = normalize_path(path)
                if os.path.isdir(path_n):
                    console.print(f"[red]Hai inserito una directory:[/red] [yellow]{path_n}[/yellow]")
                    continue
                model = load_model(path_n)
                if model is not None: break
                console.print("[yellow]Caricamento fallito. Riprova.[/yellow]")
            if model is None:
                console.print("[red]Impossibile caricare un modello. Torno al menu.[/red]")
                continue
            fn = Prompt.ask("Feature names (virgola, o invio)", default="")
            feature_names = [f.strip() for f in fn.split(",")] if fn.strip() else []
        elif choice == "L":
            color_map.legend()
        elif choice == "E":
            h5ad_path_in = Prompt.ask("Path .h5ad (file o directory)")
            try:
                h5ad_path = resolve_h5ad_path_input(h5ad_path_in)
            except Exception as e:
                console.print(f"[red]Impossibile scegliere .h5ad:[/red] {type(e).__name__}: {e}")
                continue
            out_path = normalize_path(Prompt.ask("Output CSV", default="expression_matrix.csv"))
            Xe, ye, genes = load_h5ad(h5ad_path)
            df_out = Xe.copy()
            df_out["cell_type"] = ye.values
            df_out.to_csv(out_path, index=False)
            console.print(f"[green]Esportato:[/green] [yellow]{out_path}[/yellow] "
                          f"({df_out.shape[0]} cellule × {len(genes)} geni)")
        elif choice == "S":
            ret = run_semi_supervised(current_adata=adata)
            if ret is not None and len(ret) == 3:
                model, feature_names, semi_result = ret
        elif choice == "K":
            run_clustering_standalone()
        elif choice == "C":
            run_confidence_check(model, feature_names, X, y)
        elif choice == "0":
            console.print("[dim]Arrivederci![/dim]")
            break
# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RF Cell Classificator TUI v2.0")
    parser.add_argument("--dataset", help="Path dataset di training")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--criterion", choices=["gini","entropy","log_loss"], default="gini")
    parser.add_argument("--model", help="Path modello .pkl")
    parser.add_argument("--semi", action="store_true",
                        help="Avvia direttamente la pipeline semi-supervisionata")
    args = parser.parse_args()
    print_banner()
    model = feature_names = X = y = adata = None
    if args.model:
        model = load_model(args.model)
        if model is None:
            console.print("[yellow]Opzione --model fallita: continuo senza modello.[/yellow]")
    if args.dataset:
        try:
            X, y, feature_names = load_dataset(args.dataset)
            model = train_model(X, y, args.trees, criterion=args.criterion)
            if args.dataset.endswith(".h5ad"):
                try:
                    import anndata as ad
                    adata = ad.read_h5ad(normalize_path(args.dataset))
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] lettura adata fallita: {type(e).__name__}: {e}")
        except Exception as e:
            console.print(f"[red]Errore caricamento/training:[/red] {type(e).__name__}: {e}")
            model = feature_names = X = y = adata = None
    if args.semi:
        ret = run_semi_supervised(current_adata=adata)
        if ret:
            return
    main_menu(model, feature_names, X, y, adata)
if __name__ == "__main__":
    main()
