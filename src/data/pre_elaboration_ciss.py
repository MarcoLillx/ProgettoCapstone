import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Inserisci qui il percorso dove hai salvato i file Excel scaricati
RAW_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'CISS 2020')
OUTPUT_FILE = r"dataset\ciss_processed.csv"
SCALER_FILE = r"dataset\scaler.pkl"

# Colonne da escludere (Metadati dell'attacco presenti solo nei file CISS)
# Basato sulla tua descrizione e immagine
DROP_COLS_INFO = [
    'Timestamp', 'Annotation', 'Other', # Spesso non utili per ML statico
    'Attack Stat', 'Attack Tar', 'Attack Typ', 'Attack Int', 
    'Attack Mod', 'Attack Out', 'Target Sel', 'Entry Poin', 
    'Attacker', 'Attack ID', 'Attack Sub', 'Plant', 'Attack Name',
    'Attack Hash', 'Attack Start', 'Attack End', 'Point ASD'
]

def load_and_process_ciss():
    all_data = []
    
    # 1. Caricamento File TARGET (Solo dati Normali)
    target_files = glob.glob(os.path.join(RAW_DATA_DIR, "Target*.xlsx"))
    print(f"[INFO] Trovati {len(target_files)} file Target (Normali).")
    
    for f in target_files:
        print(f"  -> Caricamento {os.path.basename(f)}...")
        df = pd.read_excel(f, engine='openpyxl')
        
        # Aggiungiamo la Label 0 (Normale)
        df['Label'] = 0
        
        # Rimuoviamo Timestamp se presente
        if 'Timestamp' in df.columns:
            df = df.drop(columns=['Timestamp'])
            
        all_data.append(df)

    # 2. Caricamento File CISS_OL (Misti: Normali + Attacchi)
    ciss_files = glob.glob(os.path.join(RAW_DATA_DIR, "CISS2020_OL*.xlsx"))
    print(f"[INFO] Trovati {len(ciss_files)} file CISS_OL (Misti).")
    
    for f in ciss_files:
        print(f"  -> Caricamento {os.path.basename(f)}...")
        df = pd.read_excel(f, engine='openpyxl')
        
        # --- LOGICA DI ETICHETTATURA (LABELING) ---
        # Guardando l'immagine, usiamo 'Attack ID' o 'Attack Name'.
        # Se è 'None', NaN o vuoto -> Normale (0). Altrimenti -> Attacco (1).
        
        # Cerchiamo la colonna corretta (a volte i nomi variano leggermente)
        attack_col = None
        possible_cols = ['Attack ID', 'Attack Name', 'Attack Stat']
        for col in possible_cols:
            if col in df.columns:
                attack_col = col
                break
        
        if attack_col:
            # Riempiamo i NaN con 'None'
            df[attack_col] = df[attack_col].fillna('None').astype(str)
            
            # Logica: Se contiene "None" o è vuoto è Normale, altrimenti Attacco
            # Nota: Nell'immagine vedo "None" sotto Attack Name per le prime righe
            df['Label'] = df[attack_col].apply(lambda x: 0 if x.strip() == 'None' or x.strip() == '' else 1)
        else:
            print(f"[WARNING] Colonna attacco non trovata in {f}. Imposto tutto a 0.")
            df['Label'] = 0

        # --- PULIZIA COLONNE EXTRA ---
        # Dobbiamo tenere solo le colonne dei sensori (comuni ai file Target)
        # Identifichiamo le colonne da rimuovere (quelle che non sono sensori)
        cols_to_drop = [c for c in df.columns if any(x in c for x in DROP_COLS_INFO) and c != 'Label']
        
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        all_data.append(df)

    # 3. Unione
    print("[INFO] Unione dei DataFrame...")
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Pulizia finale nomi colonne (rimuove spazi extra tipo " FIT101")
    full_df.columns = full_df.columns.str.strip()
    
    # Verifica Label
    print(f"[INFO] Distribuzione Classi Totale:\n{full_df['Label'].value_counts()}")

    return full_df

def preprocess_and_save(df):
    # 4. Downsampling per SVM
    # SVM non gestisce milioni di righe. Riduciamo il dataset.
    
    df_attack = df[df['Label'] == 1]
    df_normal = df[df['Label'] == 0]
    
    print(f"[INFO] Attacchi totali: {len(df_attack)}")
    
    # Prendiamo tutti gli attacchi e un numero simile di normali (es. 20.000 normali)
    # Se gli attacchi sono pochi (<10k), prendiamo più normali per bilanciare meglio
    n_samples_normal = 30000 
    if len(df_normal) > n_samples_normal:
        df_normal = df_normal.sample(n=n_samples_normal, random_state=42)
    
    df_reduced = pd.concat([df_normal, df_attack])
    df_reduced = df_reduced.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    
    print(f"[INFO] Dataset ridotto finale: {len(df_reduced)} righe.")
    
    # 5. Scaling (MinMax)
    y = df_reduced['Label']
    X = df_reduced.drop(columns=['Label'])
    
    # Gestione valori non numerici (se presenti per errore)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final['Label'] = y.values
    
    # Salvataggio
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, SCALER_FILE)
    
    print(f"[SUCCESS] Dataset salvato in {OUTPUT_FILE}")

if __name__ == "__main__":
    # Assicurati che la cartella raw esista e contenga i file xlsx
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[ERRORE] Cartella {RAW_DATA_DIR} non trovata.")
    else:
        df = load_and_process_ciss()
        preprocess_and_save(df)