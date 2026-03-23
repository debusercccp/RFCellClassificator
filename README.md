# RFCellClassificator v2.0 — Semi-Supervised Edition

Classificazione di tipi cellulari da espressione genica (scRNA-seq).
Interfaccia TUI Rich, pipeline semi-supervisionata, ottimizzata per Raspberry Pi.

## Struttura

```
RFCellClassificator/
├── random_forest_tui.py    ← entry point (tutto in uno, come prima)
├── core/
│   ├── clustering.py       ← Lloyd KMeans + marker gene selection
│   ├── classifier.py       ← SemiSupervisedPipeline
│   └── confidence.py       ← Euclidean confidence + Unknown labeling
├── scarica_dataset.py      ← setup PBMC3k (invariato)
├── requirements.txt
└── flake.nix               ← Nix dev shell con openblas
```

## Avvio rapido

```bash
# Classico (identico a v1.0)
python random_forest_tui.py

# Pipeline semi-supervisionata diretta
python random_forest_tui.py --semi

# Nix (con openblas ottimizzato per Pi)
nix develop
python random_forest_tui.py
```

## Opzioni menu v2.0

| Tasto | Funzione |
|-------|---------|
| `1-9` | Tutte le opzioni originali (invariate) |
| `S`   | **Pipeline semi-supervisionata**: Clustering → Marker Genes → RF → Consensus → Confidence |
| `K`   | **Clustering standalone**: Lloyd KMeans + visualizzazione cluster + marker genes |
| `C`   | **Confidence score**: distanza euclidea cellula → centroide, Unknown labeling |

## Pipeline semi-supervisionata [S]

```
AnnData (X_pca)
    │
    ▼
Lloyd KMeans (k cluster, puro NumPy)
    │
    ▼
Marker Gene Selection (variance_ratio)
20k geni → 200 marker  →  training RF 10-50x più veloce su Pi
    │
    ▼
Random Forest (su soli marker genes)
    │
    ├── Consensus Check: RF ↔ Cluster dominant type
    │   flagga errori di annotazione / cellule in transizione
    │
    └── Euclidean Confidence: dist(cellula, centroide classe)
        score > soglia(95° percentile intra-classe) → "Unknown"
```

## Performance stimata su Raspberry Pi 5 (50k cellule, 20k geni)

| Step | v1.0 | v2.0 |
|------|------|------|
| Training RF | ~90s | ~8s |
| Feature space | 20k geni | 200 marker genes |
| Riduzione | — | 100x |
