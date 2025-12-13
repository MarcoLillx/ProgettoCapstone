import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from joblib import dump, load
from utils import train_model, test_model, generate_adv_examples, cosine_similarity

# Configurazione
SEED = 42
np.random.seed(SEED)

# Percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_processed.csv')
# Uncomment the following line to use the refined dataset after feature engineering
# DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_refined.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ADV_DIR = os.path.join(BASE_DIR, 'adv_examples')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
FIGURE_DIR = os.path.join(BASE_DIR, 'figure')

# Creazione cartelle
for d in [MODELS_DIR, ADV_DIR, LOGS_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    print("--- STARTING CAPSTONE PROJECT: CISS 2020 (Water Treatment) ---")
    
    # 1. Caricamento Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"ERRORE: Dataset non trovato in {DATASET_PATH}.")
        exit()
        
    print(f"[INFO] Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    # --- PULIZIA CRITICA (DATA LEAKAGE REMOVAL) ---
    cols_to_drop = [
        'Label', 
        'Attack Id', 'Attack Name', 'Attack Stat', 'Intent', 'ASD', 
        'PLC1', 'PLC2', 'PLC3', 'PLC4', 'PLC5', 'PLC6', 
        'P1SA1', 'Plant', 'Attack Hash'
    ]
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=existing_cols_to_drop).values
    y = df['Label'].values
    
    nb_features = X.shape[1]
    print(f"[INFO] Features finali (Sensori): {nb_features}, Samples: {len(y)}")

    # 2. Split Dataset (A=60%, B=20%, C=20%)
    X_A, X_BC, y_A, y_BC = train_test_split(X, y, train_size=0.60, random_state=SEED, stratify=y)
    X_B, X_C, y_B, y_C = train_test_split(X_BC, y_BC, train_size=0.50, random_state=SEED, stratify=y_BC)
    
    # Riduzione B per demo (Gli attacchi Black-Box sono lenti)
    SAMPLE_SIZE = 50 
    if len(X_B) > SAMPLE_SIZE:
        print(f"[INFO] Reducing B set to {SAMPLE_SIZE} samples for attack demonstration...")
        indices = np.random.choice(len(X_B), SAMPLE_SIZE, replace=False)
        X_B_sub = X_B[indices]
        y_B_sub = y_B[indices]
    else:
        X_B_sub = X_B
        y_B_sub = y_B

    # 3. Training Modelli Vittima
    models_to_train = ['dt', 'rf', 'xgb'] 
    trained_models = {}
    final_results = []

    for m_name in models_to_train:
        model_path = os.path.join(MODELS_DIR, f"{m_name}_ciss.joblib")
        
        # Addestramento
        model = train_model(m_name, X_A, y_A, SEED)
        dump(model, model_path)
        trained_models[m_name] = model
        
        # --- TEST NORMALE (Su C) ---
        print(f"--- Normal Test for {m_name} ---")
        metrics_norm = test_model(
            m_name, 
            model, 
            X_C, 
            y_C, 
            save_path_cm=os.path.join(FIGURE_DIR, "normal_test"), 
            prefix_cm="Normal"
        )
        
        res_entry = {'Model': m_name, 'Scenario': 'Normal', 'Attack': 'None'}
        res_entry.update(metrics_norm)
        final_results.append(res_entry)

    # 4. Esecuzione Attacchi (Evasione)
    attacks_to_run = ['zoo', 'bound', 'hop', 'sign']
    
    for m_name, model in trained_models.items():
        for attack_type in attacks_to_run:
            print(f"\n=== ATTACKING {m_name} with {attack_type.upper()} ===")
            
            # Generazione Esempi Avversari su B (sottoinsieme)
            # Passiamo y_B_sub per supportare Sign-OPT Targeted
            X_adv, time_taken = generate_adv_examples(attack_type, model, X_B_sub, y_B_sub, m_name, nb_features)
            
            # Calcolo Similarità
            similarity = cosine_similarity(X_adv, X_B_sub)
            print(f"[METRICS] Avg Cosine Similarity: {similarity:.4f}")
            
            # --- SALVATAGGIO ESEMPI (ORA ATTIVO) ---
            filename = f"adv_{m_name}_{attack_type}.txt"
            save_path = os.path.join(ADV_DIR, filename)
            np.savetxt(save_path, X_adv)
            print(f"[INFO] Saved adversarial examples to {filename}")
            
            # --- TEST AVVERSARIO (Su B_adv) ---
            print(f"--- Adversarial Test for {m_name} ({attack_type}) ---")
            metrics_adv = test_model(
                m_name, 
                model, 
                X_adv, 
                y_B_sub, 
                save_path_cm=os.path.join(FIGURE_DIR, "adv_test"), 
                prefix_cm=f"Adv_{attack_type}"
            )
            
            # Aggiungiamo al report
            res_entry = {
                'Model': m_name, 
                'Scenario': 'Adversarial', 
                'Attack': attack_type,
                'Time_Gen_Attack': time_taken,
                'Cosine_Similarity': similarity
            }
            res_entry.update(metrics_adv)
            final_results.append(res_entry)

    # 5. Riepilogo Finale e Salvataggio CSV
    print("\n=== FINAL REPORT ===")
    res_df = pd.DataFrame(final_results)
    
    cols = ['Model', 'Scenario', 'Attack', 'Precision_W', 'Recall_W', 'F1_W', 'Overall_Acc', 'Average_Acc', 'Time_Test', 'Time_Gen_Attack', 'Cosine_Similarity']
    cols = [c for c in cols if c in res_df.columns]
    res_df = res_df[cols]
    
    print(res_df)
    res_df.to_csv(os.path.join(LOGS_DIR, 'final_metrics_report_full.csv'), index=False)
    print(f"[SUCCESS] Report saved to {os.path.join(LOGS_DIR, 'final_metrics_report_full.csv')}")