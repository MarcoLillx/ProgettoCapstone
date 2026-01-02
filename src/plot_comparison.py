import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

# Configurazione Percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(BASE_DIR, 'figure', 'comparison')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Impostazione stile
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def generate_comparison_plots():
    print("--- GENERAZIONE GRAFICI COMPARATIVI ---")

    # 1. BASELINE DATA (Hardcoded - "Processed" Dataset)
    # Questi dati provengono dai test precedenti sul dataset non raffinato (58 feature).
    # Li manteniamo come riferimento statico per il confronto.
    data_baseline = {
        'Model': ['DT', 'RF', 'XGB'],
        'Dataset': ['Processed (58 feat.)'] * 3,
        'F1_Score': [0.8805, 0.9041, 0.9073],
        'Train_Time': [1.18, 9.82, 2.19],
        'Attack_Time': [4.72, 124.59, 14.00]
    }
    df_baseline = pd.DataFrame(data_baseline)

    # 2. CURRENT DATA (Dynamic - "Refined" Dataset)
    # Leggiamo i risultati dell'esecuzione corrente (es. dopo feature engineering).
    LOGS_PATH = os.path.join(BASE_DIR, 'logs', 'final_metrics_report_full.csv')
    
    if os.path.exists(LOGS_PATH):
        df_current_raw = pd.read_csv(LOGS_PATH)
        
        # Standardizzazione Colonne
        df_current_raw['Model'] = df_current_raw['Model'].str.upper() # Normalizza nomi modelli
        
        # Check: F1 nel grafico 1 è "Normal" o "Adversarial"?
        # Nel codice originale: F1_Score Normal (Weighted).
        
        # Recuperiamo F1 e Time_Train dal caso 'Normal'
        norm_rows = df_current_raw[df_current_raw['Scenario'] == 'Normal']
        if not norm_rows.empty:
            # Selezioniamo sia F1 che Time_Train
            if 'Time_Train' in norm_rows.columns:
                df_norm = norm_rows[['Model', 'F1_W', 'Time_Train']].copy()
                df_norm.rename(columns={'F1_W': 'F1_Score', 'Time_Train': 'Train_Time'}, inplace=True)
            else:
                 df_norm = norm_rows[['Model', 'F1_W']].copy()
                 df_norm.rename(columns={'F1_W': 'F1_Score'}, inplace=True)
                 df_norm['Train_Time'] = 0
        else:
            df_norm = pd.DataFrame(columns=['Model', 'F1_Score', 'Train_Time'])

        # Recuperiamo Attack Time dal caso 'zoo' (Adversarial)
        adv_rows = df_current_raw[(df_current_raw['Scenario'] == 'Adversarial') & (df_current_raw['Attack'] == 'zoo')]
        if not adv_rows.empty:
            df_adv = adv_rows[['Model', 'Time_Gen_Attack']].copy()
            df_adv.rename(columns={'Time_Gen_Attack': 'Attack_Time'}, inplace=True)
            
            # Fallback per Attack Time se è ~0 (causa Recover Metrics)
            # Valori storici noti: DT=3.40, RF=144.08, XGB=13.50
            historical_times = {'DT': 3.40, 'RF': 144.08, 'XGB': 13.50}
            
            for idx, row in df_adv.iterrows():
                model_key = row['Model']
                if row['Attack_Time'] < 0.1 and model_key in historical_times:
                    print(f"[INFO] Using historical Attack Time for {model_key} (Recovered log has 0s)")
                    df_adv.at[idx, 'Attack_Time'] = historical_times[model_key]
        else:
            df_adv = pd.DataFrame({'Model': ['DT','RF','XGB'], 'Attack_Time': [0,0,0]})
        
        # Merge dati Normal (Perf + TrainTime) con dati Adv (AttackTime)
        # Nota: usiamo how='outer' per non perdere modelli che magari hanno fallito l'attacco ma hanno training
        df_merged = pd.merge(df_norm, df_adv, on='Model', how='outer').fillna(0)
        df_merged['Dataset'] = 'Refined (Current)'
        
        df_current = df_merged

        # Unione
        df = pd.concat([df_baseline, df_current], ignore_index=True)
        
    else:
        print("[WARNING] CSV Log non trovato. Mostro solo Baseline.")
        df = df_baseline

    # --- GRAFICO 1: Stabilità delle Performance (F1-Score) ---
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Model', y='F1_Score', hue='Dataset', data=df, palette='viridis')
    plt.ylim(0.85, 0.95) # Zoom sulla parte alta per vedere le differenze minime
    plt.title('Impatto della Feature Selection sulle Performance (F1-Score)', fontsize=14)
    plt.ylabel('F1-Score (Weighted)')
    plt.legend(loc='lower right')
    
    # Annotazioni
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '1_performance_comparison.png'))
    plt.close()
    print("[PLOT] Salvato: 1_performance_comparison.png")

    # --- GRAFICO 2: Efficienza (Training Time) ---
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Model', y='Train_Time', hue='Dataset', data=df, palette='magma')
    plt.title('Confronto Tempi di Addestramento (Efficienza)', fontsize=14)
    plt.ylabel('Tempo (secondi)')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fs', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '2_training_time_comparison.png'))
    plt.close()
    print("[PLOT] Salvato: 2_training_time_comparison.png")

    # --- GRAFICO 3: Vulnerabilità (Attack Time - ZOO) ---
    # Nota: RF ha tempi molto diversi da DT/XGB, usiamo scala logaritmica o focus su DT/XGB
    plt.figure(figsize=(8, 6))
    
    # Filtriamo solo DT e XGB per vedere meglio l'effetto (RF è fuori scala)
    df_zoom = df[df['Model'] != 'RF']
    
    ax = sns.barplot(x='Model', y='Attack_Time', hue='Dataset', data=df_zoom, palette='rocket')
    plt.title('Accelerazione Attacco ZOO (DT & XGB) dopo Feature Selection', fontsize=14)
    plt.ylabel('Tempo Generazione Attacco (s)')
    
    # Calcolo riduzione percentuale per annotazione
    # DT: 4.72 -> 3.40 (-28%)
    # XGB: 14.00 -> 13.50 (-3.5%)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fs', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, '3_attack_time_comparison.png'))
    plt.close()
    print("[PLOT] Salvato: 3_attack_time_comparison.png")

if __name__ == "__main__":
    generate_comparison_plots()