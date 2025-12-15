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

    # DATI ESTRATTI DAI LOG (Processed vs Refined)
    # Li inseriamo manualmente per creare il dataframe di confronto
    data = {
        'Model': ['DT', 'RF', 'XGB'] * 2,
        'Dataset': ['Processed (58 feat.)'] * 3 + ['Refined (46 feat.)'] * 3,
        
        # F1-Score Normal (Weighted)
        'F1_Score': [0.8805, 0.9041, 0.9073,  # Processed
                     0.8842, 0.9027, 0.9067], # Refined
        
        # Training Time (secondi)
        'Train_Time': [1.18, 9.82, 2.19,    # Processed
                       0.95, 8.56, 2.50],   # Refined
        
        # ZOO Attack Generation Time (secondi per 50 samples)
        'Attack_Time': [4.72, 124.59, 14.00,  # Processed
                        3.40, 144.08, 13.50]  # Refined
    }

    df = pd.read_json(pd.DataFrame(data).to_json()) # Trick per evitare errori di tipo

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