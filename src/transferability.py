import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import train_model, generate_adv_examples

# Configurazione
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_refined.csv')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
SEED = 42
np.random.seed(SEED)

def run_transferability():
    print("--- ESPERIMENTO: TRANSFERABILITY (GRAY-BOX) ---")
    
    if not os.path.exists(DATASET_PATH):
        print("Errore: Dataset non trovato.")
        return

    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    nb_features = X.shape[1]
    
    # Split A/B/C
    X_A, X_BC, y_A, y_BC = train_test_split(X, y, train_size=0.60, random_state=SEED, stratify=y)
    X_B, X_C, y_B, y_C = train_test_split(X_BC, y_BC, train_size=0.50, random_state=SEED, stratify=y_BC)
    
    # Sottoinsieme per attacco (50 sample)
    indices = np.random.choice(len(X_B), 50, replace=False)
    X_att = X_B[indices]
    y_att = y_B[indices]

    # 1. Training Modelli
    print("[INFO] Training models...")
    models = {}
    for m_name in ['dt', 'rf', 'xgb']:
        models[m_name] = train_model(m_name, X_A, y_A, SEED)

    # 2. Generazione Attacchi sul SOURCE (Decision Tree)
    source_name = 'dt'
    attack_type = 'hop' # HopSkipJump trasferisce meglio di ZOO
    print(f"\n[ATTACK] Generazione esempi avversari su SOURCE: {source_name} ({attack_type})...")
    
    # Passiamo y_att per compatibilità wrapper
    X_adv_source, _ = generate_adv_examples(attack_type, models[source_name], X_att, y_att, source_name, nb_features)
    
    # 3. Test di Trasferibilità sui TARGET
    results = []
    
    print("\n[TEST] Verifica Trasferibilità:")
    for target_name, target_model in models.items():
        # Prediciamo sugli esempi generati dal DT
        y_pred_adv = target_model.predict(X_adv_source)
        
        # Calcoliamo l'accuratezza (più è bassa, più l'attacco ha avuto successo)
        acc_adv = accuracy_score(y_att, y_pred_adv)
        
        # Attack Success Rate (ASR) = 1 - Accuracy (su esempi che erano corretti)
        # Semplificazione: Quanti sono stati classificati male?
        # Nota: Idealmente dovremmo filtrare solo quelli predetti correttamente all'inizio, 
        # ma per semplicità guardiamo l'accuratezza grezza sugli avversari.
        asr = 1.0 - acc_adv
        
        role = "SOURCE" if target_name == source_name else "TARGET"
        print(f"   -> {target_name.upper()} ({role}): Accuracy sotto attacco = {acc_adv:.2%} (ASR: {asr:.2%})")
        
        results.append({
            'Source_Model': source_name,
            'Target_Model': target_name,
            'Role': role,
            'Adv_Accuracy': acc_adv,
            'Transfer_ASR': asr
        })

    # Salvataggio
    df_res = pd.DataFrame(results)
    os.makedirs(LOGS_DIR, exist_ok=True)
    df_res.to_csv(os.path.join(LOGS_DIR, 'transferability_results.csv'), index=False)
    print(f"\n[SUCCESS] Dati salvati in {os.path.join(LOGS_DIR, 'transferability_results.csv')}")

if __name__ == "__main__":
    run_transferability()