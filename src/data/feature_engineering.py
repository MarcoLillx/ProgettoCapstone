import pandas as pd
import numpy as np
import os

# Configurazione
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_processed.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_refined.csv')

def refine_dataset():
    print("--- STARTING ADVANCED PREPROCESSING ---")
    
    df = pd.read_csv(INPUT_PATH)
    
    # 1. Rimozione Leakage (Definitiva)
    cols_leakage = [
        'Attack Id', 'Attack Name', 'Attack Stat', 'Intent', 'ASD', 
        'PLC1', 'PLC2', 'PLC3', 'PLC4', 'PLC5', 'PLC6', 
        'P1SA1', 'Plant', 'Attack Hash'
    ]
    existing_leakage = [c for c in cols_leakage if c in df.columns]
    df = df.drop(columns=existing_leakage)
    print(f"[STEP 1] Rimosse {len(existing_leakage)} colonne di Leakage.")

    # 2. Rimozione Colonne Costanti (Varianza Zero)
    # In un sistema fisico, se un sensore vale sempre 0.0 o 1.0 spaccato per 40k righe,
    # non porta informazione utile per la classificazione.
    description = df.describe()
    # Colonne dove min == max (quindi std == 0)
    constant_cols = [c for c in df.columns if description[c]['min'] == description[c]['max']]
    
    # Non rimuovere mai la Label!
    if 'Label' in constant_cols: constant_cols.remove('Label')
    
    df = df.drop(columns=constant_cols)
    print(f"[STEP 2] Rimosse {len(constant_cols)} colonne Costanti.")
    print(f"         -> {constant_cols}")

    # 3. Rimozione Feature Altamente Correlate (Collinearità)
    # Se due sensori dicono la stessa cosa al 99%, ne basta uno.
    # Questo aiuta molto modelli come Logistic Regression o SVM, meno RF, ma riduce la dimensione per ZOO.
    
    # Calcoliamo la matrice di correlazione (valore assoluto)
    corr_matrix = df.drop(columns=['Label']).corr().abs()
    
    # Selezioniamo il triangolo superiore della matrice
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Troviamo colonne con correlazione > 0.98
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    
    df = df.drop(columns=to_drop)
    print(f"[STEP 3] Rimosse {len(to_drop)} colonne Altamente Correlate (>0.98).")
    print(f"         -> {to_drop}")

    # Salvataggio
    print(f"[INFO] Dataset finale: {df.shape}")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Dataset raffinato salvato in: {OUTPUT_PATH}")

if __name__ == "__main__":
    refine_dataset()