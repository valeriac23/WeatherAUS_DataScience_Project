# WeatherAUS — Data Science Project

Progetto di Data Science sul dataset **Rain in Australia (WeatherAUS)**: un unico dataset meteorologico affrontato con tre famiglie di tecniche — **classificazione**, **clustering** e **analisi di serie temporali**.

Progetto per il corso di Data Science, Laurea Magistrale in Ingegneria Informatica e dell'Automazione, Università Politecnica delle Marche (UNIVPM), A.A. 2025/2026.

---

## Dataset

I dati provengono dal dataset pubblico **[Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)** (Kaggle), basato su osservazioni del Bureau of Meteorology australiano.

| | |
|---|---|
| Osservazioni | 145 460 giornaliere |
| Stazioni | 49 località |
| Periodo | novembre 2007 – giugno 2017 |
| Variabili | 23 (temperature, umidità, pressione, vento, pioggia…) |
| Target (classificazione) | `RainTomorrow` — pioggia il giorno successivo (~22% di casi positivi) |
---

## Struttura del repository

```
WeatherAUS_DataScience_Project/
├── classification_and_clustering/
│   ├── eda_data_analysis.ipynb      # Analisi esplorativa (EDA)
│   ├── classification.ipynb         # Classificazione binaria di RainTomorrow
│   ├── clustering.ipynb             # Clustering dei regimi meteorologici
│   └── MetaCost.py                  # Implementazione dell'algoritmo MetaCost
├── temporal_series/
│   └── time_series.ipynb            # Serie temporali (SARIMAX) su Temp3pm a Sydney
├── dataset/                     # weatherAUS.csv (da scaricare, non versionato)
├── README.md
└── .gitignore
```

---

## Le tre analisi

### 1. Analisi esplorativa (EDA)
`eda_data_analysis.ipynb` — Esame della struttura del dataset: valori mancanti (con scarto delle colonne a mancanza strutturale come `Sunshine`, `Evaporation`, `Cloud9am`, `Cloud3pm`), distribuzioni delle variabili, matrice di correlazione, sbilanciamento del target e relazioni fra `RainToday` e `RainTomorrow`.

### 2. Classificazione binaria — *Pioverà domani?*
`classification.ipynb` — Previsione di `RainTomorrow` confrontando più algoritmi in cross-validation e gestendo esplicitamente lo sbilanciamento delle classi.

- **Modello migliore:** XGBoost (selezionato per F1 in CV), con ottimizzazione della soglia decisionale.
- **Prestazioni sul test set** (classe *pioggia*): **F1 ≈ 0,66**, precision 0,63, recall 0,70; **ROC-AUC 0,887**, **PR-AUC 0,742**, accuracy 0,84.
- **Alternativa cost-sensitive:** `MetaCost` (matrice dei costi che penalizza 5× i falsi negativi) spinge la **recall fino a 0,815**, utile quando mancare una giornata di pioggia è più costoso di un falso allarme.
- Strategie di bilanciamento confrontate: `class_weight`, SMOTE, NearMiss+SMOTE, LDA.

### 3. Clustering — regimi meteorologici
`clustering.ipynb` — Individuazione non supervisionata dei principali regimi meteorologici a partire da 12 variabili continue (campione di 10 000 giornate su 120 381 osservazioni complete, 44 località).

- **Preprocessing:** `log1p` sulla pioggia, standardizzazione, PCA (le prime 2 componenti spiegano il **61,5%** della varianza; 4 componenti superano l'80%).
- **K-Means con k = 4.** Nessun indice interno seleziona da solo k=4 (Silhouette e Calinski-Harabasz preferiscono k=2, Davies-Bouldin k=3): la scelta è motivata da **stabilità bootstrap** (ARI 0,980 ± 0,006, ultimo k stabile prima del crollo a k=5) e **interpretabilità**.
- **Quattro regimi:** freddo/invernale stabile · mite/secco · caldo/arido estivo · perturbato/ventoso.
- **DBSCAN** conferma che i dati formano un unico blocco denso e continuo (density-based inadatto alla segmentazione), con proiezione **t-SNE** a supporto.
- **Validazione esterna** contro stagione, `RainTomorrow` e località, con mappatura geografica dei regimi sull'Australia.

### 4. Serie temporali — temperatura mensile a Sydney
`temporal_series/time_series.ipynb` — Modellazione della temperatura pomeridiana media mensile (`Temp3pm`) a **Sydney**, 113 mesi (2008–2017).

- **Modello:** **SARIMAX(1,0,0)(0,1,1)₁₂**, identificato da ACF/PACF e confermato per AIC (238,6). Poiché `pmdarima`/`auto_arima` non è compatibile con Python 3.13, la selezione automatica è realizzata con una **ricerca a griglia** su statsmodels, che seleziona lo stesso modello.
- **Validazione** con split temporale (ultimi 24 mesi come test); residui **rumore bianco** (Ljung-Box non significativo) ed errore inferiore al grado.
- **Risultati onesti e istruttivi:**
  - un **baseline naïve stagionale** (MAE 0,893 °C) **batte** i modelli SARIMAX (MAE 0,995) — su serie corte e fortemente stagionali i modelli semplici sono molto competitivi;
  - le **variabili esogene** (umidità e pressione) migliorano il MAE del 27% (0,995 → 0,727) **solo nello scenario "oracolo"**, in cui se ne conoscono i valori futuri; nella previsione onesta, dovendole prevedere, il loro contributo si annulla (MAE 1,023). Il guadagno apparente è un **artefatto di lookahead**, quantificato e dichiarato.
- Previsione finale a 12 mesi con intervalli di confidenza al 95%.

---

## Esecuzione

**Requisiti:** Python 3.10+ (i notebook sono stati eseguiti con Python 3.13).

Dipendenze principali:

```bash
pip install pandas numpy scikit-learn statsmodels xgboost imbalanced-learn matplotlib seaborn jupyter
```

Passi:

1. Clona il repository:
   ```bash
   git clone https://github.com/valeriac23/WeatherAUS_DataScience_Project.git
   cd WeatherAUS_DataScience_Project
   ```
2. Scarica `weatherAUS.csv` da Kaggle e mettilo in `dataset/` nella root del progetto.
3. Avvia Jupyter ed esegui i notebook nell'ordine consigliato: EDA → classificazione → clustering → serie temporali.
   ```bash
   jupyter notebook
   ```

Tutti i risultati sono riproducibili: seed fissato (RANDOM_STATE = 42) in ogni notebook, incluso il bootstrap di MetaCost.
---

## Autori

- **Valeria Cannone** 
- **Giada Remedia**
- **Alessandro Pettinaro**

Università Politecnica delle Marche — A.A. 2025/2026