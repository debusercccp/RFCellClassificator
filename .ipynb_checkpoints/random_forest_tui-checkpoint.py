#!/usr/bin/env python3
"""
TREE / RANDOM FOREST TUI
Classificazione di tipi cellulari da espressione genica
Supporta: CSV, TSV, Excel, H5AD (AnnData / single-cell)
"""

import os
import sys
import pickle
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import track
from rich.text import Text
from rich import box
from rich.rule import Rule

console = Console()

CELL_COLORS = [
    "bold red", "bold green", "bold blue", "bold yellow",
    "bold magenta", "bold cyan", "bold white", "bright_red",
    "bright_green", "bright_blue", "bright_magenta", "bright_cyan",
    "orange3", "deep_pink4", "dark_cyan", "gold1", "purple", "chartreuse3",
]

class CellColorMapper:
    def __init__(self):
        self.mapping = {}
        self._idx = 0

    def get(self, label):
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


def print_banner():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]TREE / RANDOM FOREST TUI[/bold cyan]\n"
        "[dim]Classificazione di tipi cellulari da espressione genica[/dim]\n"
        "[dim]Formati: CSV - TSV - Excel - .h5ad (AnnData / CELLxGENE)[/dim]",
        border_style="cyan",
        padding=(1, 4)
    ))
    console.print()


def load_h5ad(path, label_col=None):
    try:
        import anndata as ad
    except ImportError:
        console.print("[red]anndata non installata.[/red] pip install anndata")
        sys.exit(1)

    console.print(f"[dim]Lettura AnnData:[/dim] [yellow]{path}[/yellow]")
    adata = ad.read_h5ad(path)
    console.print(f"[green]Letto:[/green] {adata.n_obs} cellule x {adata.n_vars} geni")

    obs_cols = list(adata.obs.columns)
    if not obs_cols:
        console.print("[red]Nessuna colonna obs trovata.[/red]")
        sys.exit(1)

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
        console.print("[red]Formato non supportato.[/red] Usa: csv tsv txt xlsx h5ad")
        sys.exit(1)

    console.print(f"[green]Caricato:[/green] {df.shape[0]} campioni, {df.shape[1]} colonne")

    if label_col is None:
        console.print("\n[bold]Colonne:[/bold]")
        for i, col in enumerate(df.columns):
            console.print(f"  [{i}] {col}")
        idx = IntPrompt.ask("Indice colonna target", default=df.shape[1]-1)
        label_col = df.columns[idx]

    X = df.drop(columns=[label_col])
    y = df[label_col]
    _print_distribution(y.values)
    for cls in y.unique():
        color_map.get(str(cls))
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


def ask_criterion():
    console.print("\n[bold]Criterio di impurita:[/bold]")
    console.print("  [cyan]1[/cyan] - [bold]Gini[/bold]     impurita di Gini (default, veloce)")
    console.print("  [cyan]2[/cyan] - [bold]Entropy[/bold]  information gain, entropia di Shannon")
    console.print("  [cyan]3[/cyan] - [bold]Log-loss[/bold] cross-entropy (sklearn >= 1.1)")
    choice = Prompt.ask("Scelta", choices=["1","2","3"], default="1")
    return {"1": "gini", "2": "entropy", "3": "log_loss"}[choice]


def train_model(X, y, n_trees=100, max_depth=None, criterion="gini"):
    console.print(f"\n[bold cyan]Training: {n_trees} alberi | criterio: [yellow]{criterion}[/yellow][/bold cyan]")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=n_trees, max_depth=max_depth,
        criterion=criterion, random_state=42, n_jobs=-1
    )

    for _ in track(range(1), description="Allenamento..."):
        model.fit(X_train, y_train)

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
            f"{m['f1-score']:.2f}", str(int(m['support']))
        )

    console.print(table)
    console.print(f"[bold]Accuracy:[/bold] [green]{report['accuracy']:.2%}[/green]")
    return model


def show_tree(model, feature_names, mode="random", tree_idx=None):
    n = len(model.estimators_)
    if mode == "random":
        tree_idx = random.randint(0, n-1)
        console.print(f"\n[dim]Albero random:[/dim] #{tree_idx}")
    else:
        if tree_idx is None:
            tree_idx = IntPrompt.ask(f"Indice albero (0-{n-1})", default=0)
        tree_idx = max(0, min(tree_idx, n-1))

    # Allinea feature_names
    feature_names = list(feature_names) if feature_names else []
    n_features = model.estimators_[tree_idx].tree_.n_features
    if len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    text = export_text(model.estimators_[tree_idx], feature_names=list(feature_names), max_depth=4)
    console.print(Panel(
        f"[green]{text}[/green]",
        title=f"Albero #{tree_idx} | {model.criterion}",
        border_style="green", expand=False
    ))


def show_node_impurity(model, feature_names):
    """Tabella dei nodi con impurita (Gini/Entropia), soglia di espressione e campioni."""
    n = len(model.estimators_)
    tree_idx = IntPrompt.ask(f"Indice albero (0-{n-1})", default=0)
    tree_idx = max(0, min(tree_idx, n-1))

    t = model.estimators_[tree_idx].tree_
    criterion = model.criterion
    label = {"gini": "Gini", "entropy": "Entropia", "log_loss": "Log-loss"}.get(criterion, criterion)

    # Allinea feature_names
    feature_names = list(feature_names) if feature_names else []
    n_features = t.n_features
    if len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    table = Table(
        title=f"Impurita nodi - Albero #{tree_idx} | {label}",
        box=box.ROUNDED
    )
    table.add_column("Nodo", justify="right", style="dim")
    table.add_column("Gene / Feature")
    table.add_column("Soglia expr.", justify="right")
    table.add_column(label, justify="right")
    table.add_column("Campioni", justify="right")
    table.add_column("Tipo")

    for i in range(t.node_count):
        is_leaf = t.children_left[i] == t.children_right[i]
        imp = t.impurity[i]

        if imp < 0.2:   imp_style = "green"
        elif imp < 0.4: imp_style = "yellow"
        else:           imp_style = "red"

        if is_leaf:
            table.add_row(
                str(i), "-", "-",
                Text(f"{imp:.4f}", style=imp_style),
                str(t.n_node_samples[i]),
                Text("foglia", style="dim")
            )
        else:
            fname = feature_names[t.feature[i]] if t.feature[i] < len(feature_names) else f"f{t.feature[i]}"
            table.add_row(
                str(i),
                Text(fname, style="cyan"),
                f"{t.threshold[i]:.4f}",
                Text(f"{imp:.4f}", style=imp_style),
                str(t.n_node_samples[i]),
                Text("split", style="bold white")
            )

    console.print(table)
    console.print("[dim]Colori impurita:[/dim] [green]< 0.2 bassa[/green]  [yellow]0.2-0.4 media[/yellow]  [red]> 0.4 alta[/red]")


def show_feature_importance(model, feature_names, top_n=20):
    """Top N geni per Mean Decrease Impurity (MDI)."""
    importances = model.feature_importances_
    n_features = len(importances)

    # Allinea feature_names alla dimensione effettiva del modello
    feature_names = list(feature_names) if feature_names else []
    if len(feature_names) != n_features:
        console.print(f"[yellow]Attenzione: feature_names ha {len(feature_names)} elementi, il modello ne ha {n_features}. Uso indici numerici.[/yellow]")
        feature_names = [f"feature_{i}" for i in range(n_features)]

    std = np.std([t.feature_importances_ for t in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:top_n]
    criterion = model.criterion
    label = {"gini": "Gini MDI", "entropy": "Entropy MDI", "log_loss": "Log-loss MDI"}.get(criterion, "MDI")

    table = Table(title=f"Top {top_n} geni - {label}", box=box.ROUNDED)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Gene")
    table.add_column("Importanza", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Barra")

    max_imp = importances[indices[0]] if len(indices) else 1.0

    for rank, idx in enumerate(indices, 1):
        idx = int(idx)  # converti da numpy.int64 a int Python
        imp = importances[idx]
        bar = "#" * int((imp / max_imp) * 30)
        bar_style = "gold1" if rank <= 3 else "cyan" if rank <= 10 else "dim cyan"
        table.add_row(
            str(rank),
            Text(feature_names[idx], style="bold"),
            f"{imp:.5f}",
            f"+-{std[idx]:.5f}",
            Text(bar, style=bar_style)
        )

    console.print(table)
    console.print(f"[dim]MDI = riduzione media dell'impurita ({criterion}) pesata per campioni. Piu alto = gene piu discriminante.[/dim]")


def compare_criteria(X, y, n_trees=100, max_depth=None):
    """Allena gini ed entropy sullo stesso split e confronta F1 per classe."""
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

    all_classes = {k for c in ["gini","entropy"] for k in results[c][1]
                   if k not in ("accuracy","macro avg","weighted avg")}

    for cls in sorted(all_classes):
        f1g = results["gini"][1].get(cls, {}).get("f1-score", 0)
        f1e = results["entropy"][1].get(cls, {}).get("f1-score", 0)
        best = "gini" if f1g >= f1e else "entropy"
        table.add_row(
            Text(cls, style=color_map.get(cls)),
            f"{f1g:.3f}", f"{f1e:.3f}",
            Text(best, style="bold green")
        )

    acc_g = results["gini"][1]["accuracy"]
    acc_e = results["entropy"][1]["accuracy"]
    best_acc = "gini" if acc_g >= acc_e else "entropy"
    table.add_row(
        Text("ACCURACY TOTALE", style="bold"),
        Text(f"{acc_g:.2%}", style="bold"),
        Text(f"{acc_e:.2%}", style="bold"),
        Text(best_acc, style="bold yellow")
    )

    console.print(table)
    best_model = results[best_acc][0]
    console.print(f"[green]Modello migliore:[/green] [bold]{best_acc}[/bold] (accuracy {max(acc_g, acc_e):.2%})")
    return best_model


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


def save_model(model, path="forest_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    console.print(f"[green]Modello salvato:[/green] [yellow]{path}[/yellow]")


def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    console.print(f"[green]Modello caricato:[/green] [yellow]{path}[/yellow]")
    return model


def main_menu(model=None, feature_names=None, X=None, y=None):
    while True:
        console.print(Rule(style="dim"))
        console.print("[bold]Menu:[/bold]")
        console.print("  [cyan]1[/cyan]  Carica dataset e allena        [dim](CSV/TSV/Excel/H5AD - scelta criterio)[/dim]")
        console.print("  [cyan]2[/cyan]  Albero random")
        console.print("  [cyan]3[/cyan]  Albero a scelta")
        console.print("  [cyan]4[/cyan]  Impurita nodi di un albero     [dim](Gini/Entropia per ogni nodo)[/dim]")
        console.print("  [cyan]5[/cyan]  Feature importance             [dim](top geni per MDI)[/dim]")
        console.print("  [cyan]6[/cyan]  Confronto Gini vs Entropy      [dim](allena entrambi e confronta F1)[/dim]")
        console.print("  [cyan]7[/cyan]  Predici su dataset di test")
        console.print("  [cyan]8[/cyan]  Salva modello")
        console.print("  [cyan]9[/cyan]  Carica modello")
        console.print("  [cyan]L[/cyan]  Legenda colori")
        console.print("  [cyan]E[/cyan]  Converti .h5ad in CSV")
        console.print("  [cyan]0[/cyan]  Esci")
        console.print()

        choice = Prompt.ask("Scelta", choices=["0","1","2","3","4","5","6","7","8","9","L","l","E","e"]).upper()

        if choice == "1":
            path = Prompt.ask("Path dataset di training")
            n_trees = IntPrompt.ask("Numero di alberi", default=100)
            md = Prompt.ask("Profondita massima (invio = illimitata)", default="")
            max_depth = int(md) if md.strip() else None
            criterion = ask_criterion()
            X, y, feature_names = load_dataset(path)
            model = train_model(X, y, n_trees, max_depth, criterion)

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
                md = Prompt.ask("Profondita massima (invio = illimitata)", default="")
                max_depth = int(md) if md.strip() else None
                best = compare_criteria(X, y, n_trees, max_depth)
                if Prompt.ask("Usare il modello migliore come attivo?", choices=["s","n"], default="s") == "s":
                    model = best

        elif choice == "7":
            if model is None: console.print("[red]Nessun modello caricato.[/red]")
            else: predict_test(model, feature_names)

        elif choice == "8":
            if model is None: console.print("[red]Nessun modello caricato.[/red]")
            else:
                path = Prompt.ask("Path salvataggio", default="forest_model.pkl")
                save_model(model, path)

        elif choice == "9":
            path = Prompt.ask("Path modello .pkl")
            model = load_model(path)
            fn = Prompt.ask("Feature names (virgola, o invio)", default="")
            feature_names = [f.strip() for f in fn.split(",")] if fn.strip() else []

        elif choice == "L":
            color_map.legend()

        elif choice == "E":
            h5ad_path = Prompt.ask("Path .h5ad")
            out_path = Prompt.ask("Output CSV", default="expression_matrix.csv")
            Xe, ye, genes = load_h5ad(h5ad_path)
            df_out = Xe.copy()
            df_out["cell_type"] = ye.values
            df_out.to_csv(out_path, index=False)
            console.print(f"[green]Esportato:[/green] [yellow]{out_path}[/yellow] ({df_out.shape[0]} cellule x {len(genes)} geni)")

        elif choice == "0":
            console.print("[dim]Arrivederci![/dim]")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path dataset di training")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--criterion", choices=["gini","entropy","log_loss"], default="gini")
    parser.add_argument("--model", help="Path modello .pkl")
    args = parser.parse_args()

    print_banner()

    model = feature_names = X = y = None

    if args.model:
        model = load_model(args.model)
    if args.dataset:
        X, y, feature_names = load_dataset(args.dataset)
        model = train_model(X, y, args.trees, criterion=args.criterion)

    main_menu(model, feature_names, X, y)


if __name__ == "__main__":
    main()
