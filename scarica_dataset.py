#!/usr/bin/env python3
"""
Script di setup: scarica e prepara il dataset PBMC3k
(2700 cellule del sangue umano, livelli di espressione genica, tipi cellulari annotati)
Pronto per essere usato con random_forest_tui.py
"""

import os
import sys

def check_and_install(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f"  Installazione {package}...")
        os.system(f"{sys.executable} -m pip install {package} -q")

print("=" * 60)
print("  SETUP DATASET PBMC3k - Espressione Genica Cellule Sangue")
print("=" * 60)

print("\n[1/4] Controllo dipendenze...")
check_and_install("scanpy")
check_and_install("anndata")
check_and_install("scipy")
check_and_install("pandas")
check_and_install("leidenalg")   # per clustering
check_and_install("igraph", "igraph")
print("   Dipendenze OK")

print("\n[2/4] Download dataset PBMC3k (prima volta ~6MB)...")
import scanpy as sc
import pandas as pd
import numpy as np

sc.settings.verbosity = 1

# Scarica il dataset raw (~6MB, viene cachato automaticamente)
adata = sc.datasets.pbmc3k()
print(f"   Scaricato: {adata.n_obs} cellule × {adata.n_vars} geni")

print("\n[3/4] Preprocessing e annotazione tipi cellulari...")

# Filtro qualità base
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalizzazione (10.000 counts per cellula = livelli di espressione comparabili)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)  # log(x+1) — standard per scRNA-seq

# Geni altamente variabili (i più informativi per distinguere i tipi cellulari)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]

# PCA + clustering
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.5)

# Annotazione manuale dei cluster (standard per PBMC3k, dalla letteratura)
# Basata sui marker genes tipici di ogni tipo cellulare:
# CD4 T: IL7R, CCR7 | CD8 T: CD8A | NK: GNLY, NKG7
# B cell: MS4A1 | Monocyte CD14+: LYZ, CD14 | Monocyte FCGR3A+: FCGR3A
# Dendritic: FCER1A, CST3 | Megakaryocyte: PPBP
cluster_to_celltype = {
    '0': 'CD4 T cell',
    '1': 'CD14+ Monocyte',
    '2': 'CD4 T cell',
    '3': 'B cell',
    '4': 'CD8 T cell',
    '5': 'NK cell',
    '6': 'CD14+ Monocyte',
    '7': 'Dendritic cell',
    '8': 'FCGR3A+ Monocyte',
    '9': 'Megakaryocyte',
    '10': 'CD4 T cell',
    '11': 'B cell',
    '12': 'CD8 T cell',
    '13': 'Dendritic cell',
}

adata.obs['cell_type'] = adata.obs['leiden'].map(
    lambda x: cluster_to_celltype.get(str(x), f'Cluster_{x}')
)

# Statistiche
print("\n  Distribuzione tipi cellulari:")
counts = adata.obs['cell_type'].value_counts()
for ct, n in counts.items():
    print(f"    {ct:<25} {n:>4} cellule  ({n/len(adata)*100:.1f}%)")

print(f"\n  Totale: {adata.n_obs} cellule, {adata.n_vars} geni (altamente variabili)")

print("\n[4/4] Salvataggio file...")

# Salva come h5ad (formato completo)
adata.write("pbmc3k_pronto.h5ad", compression="gzip")
print("   Salvato: pbmc3k_pronto.h5ad")

# Salva anche come CSV (più semplice, subset di 500 geni per dimensioni ragionevoli)
# Per il CSV usiamo i valori di espressione normalizzati
import scipy.sparse as sp
X = adata.X
if sp.issparse(X):
    X = X.toarray()

df_csv = pd.DataFrame(X, columns=adata.var_names)
df_csv['cell_type'] = adata.obs['cell_type'].values
df_csv.to_csv("pbmc3k_pronto.csv", index=False)
print("   Salvato: pbmc3k_pronto.csv")

print("\n" + "=" * 60)
print("  SETUP COMPLETATO!")
print("=" * 60)
print("""
  File pronti:
     pbmc3k_pronto.h5ad  → usa con il TUI (opzione 1)
     pbmc3k_pronto.csv   → alternativa CSV

  Come usarlo con il TUI:
    python random_forest_tui.py

    → Opzione 1 (Carica dataset)
    → Path: pbmc3k_pronto.h5ad  (o .csv)
    → Colonna target: cell_type
""")
