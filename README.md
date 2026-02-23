# Random Forest TUI — Classificazione Tipi Cellulari

Applicazione da terminale per classificare tipi cellulari a partire dai **livelli di espressione genica**, usando un modello Random Forest. Interfaccia TUI colorata basata su `rich`, con supporto nativo a file AnnData (`.h5ad`).

---

## Funzionalita

- Caricamento dataset in formato **CSV, TSV, Excel** e **.h5ad** (AnnData / CELLxGENE / Scanpy)
- Scelta del **criterio di impurita**: Gini, Entropy (Shannon), Log-loss
- Visualizzazione dell'**impurita per ogni nodo** di un albero (Gini o Entropia), con colori per livello
- **Feature importance** (Mean Decrease Impurity) con barra ASCII e deviazione standard
- **Confronto automatico Gini vs Entropy** sullo stesso dataset con tabella F1 per classe
- Visualizzazione albero random o scelto dalla foresta
- Predizione su dataset di test con colonna di confronto reale vs predetto
- Ogni tipo cellulare ha un **colore univoco** in tutti i report
- Salvataggio e caricamento del modello (`.pkl`)
- Conversione `.h5ad` → CSV
- Download e preparazione automatica del dataset **PBMC3k** (via `scarica_dataset.py`)

---

## Struttura del progetto

```
.
├── random_forest_tui.py   # Applicazione principale
├── scarica_dataset.py     # Scarica e prepara PBMC3k automaticamente
├── README.md
└── .gitignore
```

Dataset, modelli e cache non sono versionati (vedi `.gitignore`).

---

## Installazione

```bash
git clone https://github.com/tuonome/random-forest-tui.git
cd random-forest-tui

python -m venv venv
source venv/bin/activate       # Mac/Linux

pip install scikit-learn pandas numpy rich openpyxl anndata scipy scanpy leidenalg igraph
```

---

## Utilizzo

### 1. Scarica il dataset di esempio (PBMC3k)

```bash
python scarica_dataset.py
```

Genera `pbmc3k_pronto.h5ad` e `pbmc3k_pronto.csv` con 2700 cellule del sangue umano, livelli di espressione normalizzati e tipi cellulari gia annotati.

### 2. Avvia il TUI

```bash
# Interattivo
python random_forest_tui.py

# Con dataset precaricato
python random_forest_tui.py --dataset pbmc3k_pronto.h5ad --trees 200 --criterion entropy

# Con modello gia allenato
python random_forest_tui.py --model forest_model.pkl
```

### 3. Menu

| Tasto | Funzione |
|-------|----------|
| `1` | Carica dataset e allena (con scelta criterio) |
| `2` | Visualizza albero random |
| `3` | Visualizza albero a scelta |
| `4` | **Impurita nodi** — tabella Gini/Entropia per ogni nodo dell'albero |
| `5` | **Feature importance** — top N geni per Mean Decrease Impurity |
| `6` | **Confronto Gini vs Entropy** — allena entrambi, tabella F1 per classe |
| `7` | Predici su dataset di test |
| `8` | Salva modello |
| `9` | Carica modello |
| `L` | Legenda colori classi cellulari |
| `E` | Converti `.h5ad` in CSV |
| `0` | Esci |

---

## Criteri di impurita

### Gini
Misura la probabilita che un campione scelto casualmente venga classificato in modo errato. Valore 0 = nodo puro (tutti della stessa classe), valore massimo = massima mistura.

```
Gini(t) = 1 - sum(p_i^2)
```

### Entropy (Shannon)
Misura il disordine del nodo in termini di information gain. Piu sensibile di Gini a distribuzioni sbilanciate, ma computazionalmente piu costosa.

```
Entropy(t) = - sum(p_i * log2(p_i))
```

### Log-loss (cross-entropy)
Equivalente all'entropia ma con logaritmo naturale. Disponibile da sklearn >= 1.1.

---

## Visualizzazione impurita nodi (opzione 4)

Mostra ogni nodo dell'albero scelto con:

| Colonna | Significato |
|---------|-------------|
| Gene / Feature | Gene su cui viene fatto lo split |
| Soglia expr. | Valore di espressione che separa le due rami |
| Gini / Entropia | Impurita del nodo (verde bassa, rosso alta) |
| Campioni | Quante cellule passano per quel nodo |
| Tipo | `split` (nodo interno) o `foglia` |

---

## Feature importance (opzione 5)

Calcola la **Mean Decrease Impurity (MDI)**: quanto ogni gene riduce l'impurita media della foresta, pesato per il numero di campioni che attraversano i nodi in cui quel gene viene usato come split.

I top 3 geni sono evidenziati in oro, i successivi in ciano. La deviazione standard tra gli alberi indica la stabilita dell'importanza.

---

## Confronto Gini vs Entropy (opzione 6)

Allena due modelli identici (stesso split train/test, stesso numero di alberi) cambiando solo il criterio, poi mostra:
- F1-score per ogni classe cellulare
- Accuracy globale
- Quale criterio vince su ogni classe
- Offre di impostare il modello migliore come attivo

---

## Formato dataset

Righe = cellule, colonne = geni (valori di espressione normalizzati), piu una colonna per il tipo cellulare:

| cell_id | GENE1 | GENE2 | ... | cell_type  |
|---------|-------|-------|-----|------------|
| cell_1  | 0.0   | 3.2   | ... | CD4 T cell |
| cell_2  | 5.1   | 0.0   | ... | B cell     |

Per `.h5ad` la colonna target si sceglie interattivamente tra i metadati `obs`.

---

## Dataset consigliati

| Dataset | Formato | Cellule | Come ottenerlo |
|---------|---------|---------|----------------|
| PBMC3k | H5AD / CSV | 2.700 | `python scarica_dataset.py` |
| CELLxGENE | H5AD | variabile | [cellxgene.cziscience.com](https://cellxgene.cziscience.com) |
| Human Cell Atlas | H5AD / CSV | variabile | [humancellatlas.org](https://www.humancellatlas.org) |
| GEO NCBI | CSV / TSV | variabile | [ncbi.nlm.nih.gov/geo](https://www.ncbi.nlm.nih.gov/geo) |

---

## Tipi cellulari nel dataset PBMC3k

| Tipo cellulare | Descrizione |
|----------------|-------------|
| CD4 T cell | Linfociti T helper |
| CD8 T cell | Linfociti T citotossici |
| NK cell | Natural Killer |
| B cell | Linfociti B |
| CD14+ Monocyte | Monociti classici |
| FCGR3A+ Monocyte | Monociti non classici |
| Dendritic cell | Cellule dendritiche |
| Megakaryocyte | Precursori delle piastrine |

---

## Dipendenze

| Libreria | Utilizzo |
|----------|----------|
| `scikit-learn` | Random Forest, metriche, export alberi |
| `pandas` / `numpy` | Manipolazione dati |
| `rich` | Interfaccia TUI colorata |
| `anndata` | Lettura file `.h5ad` |
| `scanpy` | Download e preprocessing PBMC3k |
| `scipy` | Matrici sparse |
| `openpyxl` | Lettura file Excel |
# RFCellClassificator
