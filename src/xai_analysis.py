import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from joblib import load

# Configurazione
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'ciss_refined.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ADV_DIR = os.path.join(BASE_DIR, 'adv_examples')
FIGURE_DIR = os.path.join(BASE_DIR, 'figure', 'xai')

os.makedirs(FIGURE_DIR, exist_ok=True)
SEED = 42

def run_xai_analysis():
    print("--- EXPLAINABLE AI (XAI) ANALYSIS WITH SHAP ---")

    # 1. Caricamento Dati e Nomi Feature
    df = pd.read_csv(DATASET_PATH)
    feature_names = df.drop(columns=['Label']).columns.tolist()
    X = df.drop(columns=['Label']).values
    y = df['Label'].values

    # 2. Ricostruzione dello Split (per avere gli stessi sample usati in main.py)
    # A=Train, B=AdvGen
    X_A, X_BC, y_A, y_BC = train_test_split(X, y, train_size=0.60, random_state=SEED, stratify=y)
    X_B, X_C, y_B, y_C = train_test_split(X_BC, y_BC, train_size=0.50, random_state=SEED, stratify=y_BC)
    
    # Prendiamo gli stessi 50 indici usati per generare gli attacchi
    SAMPLE_SIZE = 50
    np.random.seed(SEED) # Importante: resettare il seed numpy prima di choice
    indices = np.random.choice(len(X_B), SAMPLE_SIZE, replace=False)
    
    X_clean = X_B[indices] # Questi sono i dati originali (Clean)
    
    # 3. Caricamento Modello e Attacco
    # Analizziamo XGBoost contro HopSkipJump (o Sign-OPT se preferisci)
    model_name = 'xgb'
    attack_name = 'hop' 
    
    print(f"[INFO] Analisi Modello: {model_name}, Attacco: {attack_name}")
    
    model = load(os.path.join(MODELS_DIR, f'{model_name}_ciss.joblib'))
    
    # Carichiamo gli esempi avversari generati
    adv_file = os.path.join(ADV_DIR, f'adv_{model_name}_{attack_name}.txt')
    if not os.path.exists(adv_file):
        print(f"Errore: File {adv_file} non trovato. Esegui prima main.py")
        return
        
    X_adv = np.loadtxt(adv_file)

    # 4. Calcolo SHAP
    print("[INFO] Calcolo valori SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_adv)

    # --- GRAFICO 1: Feature Impact (Summary Plot) ---
    # Mostra quali feature hanno influenzato di più la decisione sugli esempi avversari
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_adv, feature_names=feature_names, show=False, plot_type="bar")
    plt.title(f'SHAP Feature Importance on Adversarial Examples ({attack_name.upper()})')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'shap_summary_bar.png'))
    plt.close()
    print("[PLOT] Salvato shap_summary_bar.png")

    # --- GRAFICO 2: Analisi delle Perturbazioni (Fisica) ---
    # Calcoliamo la differenza media assoluta tra Clean e Adv per ogni feature
    perturbations = np.mean(np.abs(X_adv - X_clean), axis=0)
    
    # Creiamo un DataFrame per ordinare
    pert_df = pd.DataFrame({'Feature': feature_names, 'Perturbation': perturbations})
    pert_df = pert_df.sort_values(by='Perturbation', ascending=False).head(10) # Top 10
    
    plt.figure(figsize=(10, 6))
    plt.barh(pert_df['Feature'], pert_df['Perturbation'], color='crimson')
    plt.xlabel('Mean Absolute Perturbation (Normalized Scale)')
    plt.title(f'Top 10 Features Modified by {attack_name.upper()} Attack')
    plt.gca().invert_yaxis() # Feature più modificata in alto
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'perturbation_analysis.png'))
    plt.close()
    print("[PLOT] Salvato perturbation_analysis.png")

    # --- GRAFICO 3: Waterfall Plot (Singolo Esempio) ---
    # Prendiamo il primo esempio avversario per vedere nel dettaglio cosa è successo
    plt.figure(figsize=(8, 6))
    # Nota: shap.plots.waterfall richiede un oggetto Explanation
    exp = shap.Explanation(values=shap_values[0], 
                           base_values=explainer.expected_value, 
                           data=X_adv[0], 
                           feature_names=feature_names)
    shap.plots.waterfall(exp, show=False, max_display=10)
    plt.title(f'Why did the model fail? (Single Instance Analysis)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'shap_waterfall_single.png'), bbox_inches='tight')
    plt.close()
    print("[PLOT] Salvato shap_waterfall_single.png")

if __name__ == "__main__":
    run_xai_analysis()