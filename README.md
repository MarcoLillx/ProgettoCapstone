# Cybersecurity Capstone Project: Adversarial Attacks on Critical Infrastructure (Water Treatment)

Questo progetto è un caso di studio sulla robustezza dei sistemi di Intrusion Detection (IDS) basati su Machine Learning in scenari **Multidominio** (Cyber-Physical Systems).

Il lavoro replica la metodologia della tesi *"Adversarial Attacks on IDS and Multidomain Impact Analysis for Threat Intelligence in Military Automotive Scenarios"* (del Vescovo, Barletta), applicandola però al dominio delle infrastrutture critiche idriche (**Water Treatment**).

## Obiettivo
Valutare l'efficacia degli attacchi **Adversarial Machine Learning (Black-Box)** contro modelli IDS addestrati sul dataset **CISS 2020 / SWaT** (Secure Water Treatment).

## Struttura del Progetto
*   `src/`: Codice sorgente Python.
    *   `pre_elaboration_ciss.py`: Pulizia, unione e preprocessing del dataset CISS.
    *   `eda.py`: Analisi esplorativa dei dati (grafici e correlazioni).
    *   `feature_engineering.py`: Selezione delle feature e rimozione colonne costanti.
    *   `main.py`: Training dei modelli, esecuzione attacchi e valutazione.
    *   `utils.py`: Funzioni di supporto e wrapper per la libreria ART.
*   `dataset/`: Contiene i dati grezzi e processati (ignorati da git).
*   `models/`: Modelli addestrati (.joblib).
*   `adv_examples/`: Esempi avversari generati dagli attacchi.
*   `figure/`: Matrici di confusione e grafici EDA.
*   `logs/`: Report metriche in formato CSV.

## Tecnologie Utilizzate
*   **Linguaggio:** Python 3.x
*   **Librerie ML:** Scikit-Learn, XGBoost
*   **Adversarial ML:** Adversarial Robustness Toolbox (ART)
*   **Data Processing:** Pandas, NumPy

## Istruzioni per l'uso

### 1. Setup Ambiente
```bash
# Crea ambiente virtuale
python -m venv .venv

# Attiva ambiente (Windows)
.\.venv\Scripts\Activate.ps1
# oppure
.venv/Scripts/activate

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Preparazione dati
Scaricare il dataset **CISS 2020** e posizionare i file `.xlsx` in `dataset/CISS2020/`.
Eseguire lo script di preprocessing:

```bash
python src/pre_elaboration_ciss.py
```

### 3. Esecuzione Attacchi

Eseguire il main per addestrare i modelli (DT, RF, XGB) e lanciare gli attacchi (ZOO, Boundary, HopSkipJump, Sign-OPT):

```bash
python src/main.py
```

## Risultati Preliminari
I modelli Random Forest e XGBoost mostrano alta accuratezza in condizioni normali, ma sono vulnerabili ad attacchi avversari mirati, in particolare Sign-OPT e HopSkipJump.
