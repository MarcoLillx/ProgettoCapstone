import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configurazione
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_processed.csv')
FIGURE_DIR = os.path.join(BASE_DIR, 'figure', 'eda')

os.makedirs(FIGURE_DIR, exist_ok=True)

def perform_eda():
    print("--- STARTING EXPLORATORY DATA ANALYSIS (EDA) ---")
    
    if not os.path.exists(DATASET_PATH):
        print("Dataset non trovato.")
        return

    df = pd.read_csv(DATASET_PATH)
    
    # 1. Rimuoviamo le colonne di Leakage per l'analisi (come fatto nel main)
    cols_to_drop = [
        'Attack Id', 'Attack Name', 'Attack Stat', 'Intent', 'ASD', 
        'PLC1', 'PLC2', 'PLC3', 'PLC4', 'PLC5', 'PLC6', 
        'P1SA1', 'Plant', 'Attack Hash'
    ]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df_clean = df.drop(columns=existing_cols)
    
    print(f"[INFO] Dataset Shape: {df_clean.shape}")

    # --- PLOT 1: Distribuzione Classi ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Label', data=df_clean, palette='viridis')
    plt.title('Distribuzione Classi (0=Normal, 1=Attack)')
    plt.savefig(os.path.join(FIGURE_DIR, 'class_distribution.png'))
    plt.close()
    print("[PLOT] Class distribution saved.")

    # --- PLOT 2: Matrice di Correlazione (Top 20 Features) ---
    # Calcoliamo la correlazione solo sulle feature numeriche
    corr = df_clean.drop(columns=['Label']).corr()
    
    # Prendiamo solo le feature con varianza maggiore (più interessanti)
    # Altrimenti la matrice 50x50 è illeggibile
    variances = df_clean.drop(columns=['Label']).var()
    top_vars = variances.sort_values(ascending=False).head(20).index
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_clean[top_vars].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matrice di Correlazione (Top 20 Features per Varianza)')
    plt.savefig(os.path.join(FIGURE_DIR, 'correlation_matrix.png'))
    plt.close()
    print("[PLOT] Correlation matrix saved.")

    # --- PLOT 3: Boxplot Feature Importanti ---
    # Vediamo come cambiano le distribuzioni tra Normal e Attack per le top 5 feature
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(top_vars[:6]): # Primi 6
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='Label', y=col, data=df_clean, palette='Set2')
        plt.title(f'Distribuzione: {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'boxplots_top_features.png'))
    plt.close()
    print("[PLOT] Boxplots saved.")

    # --- ANALISI COLONNE COSTANTI ---
    # Colonne con deviazione standard = 0 (non cambiano mai)
    std_dev = df_clean.std()
    constant_cols = std_dev[std_dev == 0].index.tolist()
    print(f"\n[ANALYSIS] Colonne Costanti (Inutili): {len(constant_cols)}")
    print(constant_cols)
    
    # Salviamo la lista per il preprocessing successivo
    with open(os.path.join(BASE_DIR, 'logs', 'constant_columns.txt'), 'w') as f:
        f.write('\n'.join(constant_cols))

if __name__ == "__main__":
    perform_eda()