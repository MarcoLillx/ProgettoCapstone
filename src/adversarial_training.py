import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import train_model, test_model

# Configurazione
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_refined.csv')
ADV_DIR = os.path.join(BASE_DIR, 'adv_examples')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SEED = 42
np.random.seed(SEED)

def run_adversarial_training():
    print("--- AVVIO ADVERSARIAL TRAINING (DIFESA ATTIVA) ---")
    
    if not os.path.exists(DATASET_PATH):
        print("Errore: Dataset non trovato.")
        return

    # 1. Caricamento e Split (Identico a main.py per coerenza)
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    
    # Split A (Train), B (Adv Gen), C (Test)
    X_A, X_BC, y_A, y_BC = train_test_split(X, y, train_size=0.60, random_state=SEED, stratify=y)
    X_B, X_C, y_B, y_C = train_test_split(X_BC, y_BC, train_size=0.50, random_state=SEED, stratify=y_BC)
    
    # Recuperiamo gli indici del sottoinsieme usato per gli attacchi (50 sample)
    SAMPLE_SIZE = 50
    if len(X_B) > SAMPLE_SIZE:
        indices = np.random.choice(len(X_B), SAMPLE_SIZE, replace=False)
        X_B_sub = X_B[indices]
        y_B_sub = y_B[indices] # Queste sono le etichette VERE degli esempi avversari
    else:
        y_B_sub = y_B

    models = ['dt', 'rf', 'xgb']
    attacks = ['zoo', 'hop', 'bound', 'sign']
    
    results = []

    for m_name in models:
        for atk in attacks:
            filename = f"adv_{m_name}_{atk}.txt"
            filepath = os.path.join(ADV_DIR, filename)
            
            if not os.path.exists(filepath):
                print(f"[SKIP] File non trovato: {filename}")
                continue
                
            print(f"\n>>> Hardening {m_name} contro {atk} <<<")
            
            # 2. Caricamento Esempi Avversari
            try:
                X_adv = np.loadtxt(filepath)
            except Exception as e:
                print(f"Errore caricamento {filename}: {e}")
                continue

            # 3. Data Augmentation (Train Set + Adv Examples)
            # Uniamo il training set originale (A) con gli esempi avversari generati
            # Assegniamo agli esempi avversari la loro etichetta VERA (y_B_sub)
            # Così il modello impara che "quell'esempio strano" è in realtà un attacco (o un normale)
            X_train_aug = np.vstack((X_A, X_adv))
            y_train_aug = np.concatenate((y_A, y_B_sub))
            
            # 4. Retraining (Modello Robusto)
            print(f"   Retraining su {len(X_train_aug)} campioni...")
            model_robust = train_model(m_name, X_train_aug, y_train_aug, SEED)
            
            # 5. Valutazione Robustezza (Sul set avversario)
            # Il modello riesce ora a classificare correttamente gli attacchi che prima lo ingannavano?
            print("   Test Robustezza (su esempi avversari)...")
            metrics_rob = test_model(m_name, model_robust, X_adv, y_B_sub, None, None)
            
            # 6. Valutazione Clean (Sul set di test C)
            # Abbiamo rovinato le performance sui dati normali?
            print("   Test Clean (su dati originali)...")
            metrics_clean = test_model(m_name, model_robust, X_C, y_C, None, None)
            
            results.append({
                'Model': m_name,
                'Defended_Against': atk,
                'Robust_Accuracy': metrics_rob['Overall_Acc'], # Quanto è bravo ora sugli attacchi
                'Clean_F1': metrics_clean['F1_W'] # Quanto è bravo sui dati normali
            })

    # Salvataggio
    df_res = pd.DataFrame(results)
    os.makedirs(LOGS_DIR, exist_ok=True)
    df_res.to_csv(os.path.join(LOGS_DIR, 'adv_training_results.csv'), index=False)
    print(f"\n[SUCCESS] Risultati salvati in {os.path.join(LOGS_DIR, 'adv_training_results.csv')}")

if __name__ == "__main__":
    run_adversarial_training()