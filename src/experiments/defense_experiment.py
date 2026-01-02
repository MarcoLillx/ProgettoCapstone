import pandas as pd
import numpy as np
import sys
import os
# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from sklearn.model_selection import train_test_split
from utils import train_model, test_model, generate_adv_examples

# Configurazione
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_refined.csv')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
SEED = 42
np.random.seed(SEED)

def run_trap_experiment():
    print("--- ESPERIMENTO DIFENSIVO: TRAP FEATURES (OBFUSCATION) ---")
    
    # 1. Caricamento Dati
    if not os.path.exists(DATASET_PATH):
        print("Errore: Dataset non trovato.")
        return

    df = pd.read_csv(DATASET_PATH)
    X_clean = df.drop(columns=['Label']).values
    y = df['Label'].values
    
    # 2. Iniezione "Trap Features" (Rumore)
    # Aggiungiamo 20 colonne di rumore uniforme tra 0 e 1
    n_samples = X_clean.shape[0]
    n_traps = 20
    print(f"[INFO] Generazione di {n_traps} Trap Features...")
    
    # Rumore casuale (simula sensori non correlati)
    X_noise = np.random.rand(n_samples, n_traps)
    
    # Uniamo le feature reali con quelle trappola
    X_trapped = np.hstack((X_clean, X_noise))
    
    datasets = {
        "Clean (46 feat)": X_clean,
        "Trapped (66 feat)": X_trapped
    }
    
    # Usiamo Decision Tree (DT) e Random Forest (RF)
    models_to_test = ['dt', 'rf']
    
    # Campione piccolo per ZOO (lento)
    SAMPLE_SIZE = 30 
    
    results = []

    for dataset_name, X_data in datasets.items():
        print(f"\n=== Scenario: {dataset_name} ===")
        nb_features = X_data.shape[1]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, train_size=0.6, random_state=SEED, stratify=y)
        
        # Sottoinsieme per attacco
        indices = np.random.choice(len(X_test), SAMPLE_SIZE, replace=False)
        X_att = X_test[indices]
        y_att = y_test[indices]
        
        for m_name in models_to_test:
            # 1. Training
            # print(f"[{m_name}] Training...") # Rimosso print ridondante
            model = train_model(m_name, X_train, y_train, SEED)
            
            # 2. Valutazione Performance (F1)
            metrics = test_model(m_name, model, X_test, y_test, None, None)
            f1 = metrics['F1_W']
            
            # 3. Attacco ZOO
            print(f"[{m_name}] Esecuzione Attacco ZOO...")
            # Passiamo y_att anche se ZOO è untargeted nel wrapper, per compatibilità con la firma
            _, duration = generate_adv_examples('zoo', model, X_att, y_att, m_name, nb_features)
            
            results.append({
                'Dataset': dataset_name,
                'Model': m_name,
                'F1_Score': f1,
                'Attack_Time': duration
            })

    # Salvataggio Risultati
    df_res = pd.DataFrame(results)
    print("\n--- RISULTATI FINALI ---")
    print(df_res)
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    df_res.to_csv(os.path.join(LOGS_DIR, 'defense_results.csv'), index=False)
    print(f"[SUCCESS] Risultati salvati in {os.path.join(LOGS_DIR, 'defense_results.csv')}")

if __name__ == "__main__":
    run_trap_experiment()