import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_PATH = os.path.join(BASE_DIR, 'logs', 'adv_training_results.csv')
FIGURE_DIR = os.path.join(BASE_DIR, 'figure', 'defense')

os.makedirs(FIGURE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def plot_adv_training():
    if not os.path.exists(LOGS_PATH):
        print("Esegui prima adversarial_training.py!")
        return

    df = pd.read_csv(LOGS_PATH)

    # --- GRAFICO 1: Robustezza Acquisita ---
    plt.figure(figsize=(10, 6))
    # Filtriamo per mostrare quanto il modello è diventato bravo a riconoscere gli attacchi
    # (Prima dell'addestramento, l'accuratezza su questi esempi era 0% o molto bassa)
    ax = sns.barplot(x='Model', y='Robust_Accuracy', hue='Defended_Against', data=df, palette='viridis')
    
    plt.title('Efficacia dell\'Adversarial Training (Robustezza)', fontsize=14)
    plt.ylabel('Accuratezza su Esempi Avversari (Post-Training)')
    plt.ylim(0, 1.1)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
    plt.legend(title='Attacco Mitigato')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'adv_training_robustness.png'))
    print("[PLOT] Salvato adv_training_robustness.png")

    # --- GRAFICO 2: Impatto sulla Clean Accuracy ---
    plt.figure(figsize=(10, 6))
    # Vogliamo vedere se Clean F1 è rimasto alto (>0.88)
    ax = sns.barplot(x='Model', y='Clean_F1', hue='Defended_Against', data=df, palette='Blues')
    
    plt.title('Impatto della Difesa sulle Prestazioni Normali', fontsize=14)
    plt.ylabel('F1-Score (Clean Test Set)')
    plt.ylim(0.80, 0.95) # Zoom sulla parte alta
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'adv_training_clean_impact.png'))
    print("[PLOT] Salvato adv_training_clean_impact.png")

if __name__ == "__main__":
    plot_adv_training()