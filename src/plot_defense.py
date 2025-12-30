import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(BASE_DIR, 'logs', 'defense_results.csv')
FIGURE_DIR = os.path.join(BASE_DIR, 'figure', 'defense')

os.makedirs(FIGURE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def plot_defense():
    if not os.path.exists(LOGS_PATH):
        print("Esegui prima defense_experiment.py!")
        return

    df = pd.read_csv(LOGS_PATH)

    # --- GRAFICO 1: Rallentamento Attacco (Il Successo) ---
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Model', y='Attack_Time', hue='Dataset', data=df, palette='Reds')
    plt.title('Efficacia delle Trap Features nel Rallentare ZOO', fontsize=14)
    plt.ylabel('Tempo di Generazione Attacco (s)')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fs', padding=3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'defense_time_impact.png'))
    print("[PLOT] Salvato defense_time_impact.png")

    # --- GRAFICO 2: Impatto sulle Performance (Il Costo) ---
    plt.figure(figsize=(8, 6))
    # Zoomiamo l'asse Y per vedere le piccole differenze
    ax = sns.barplot(x='Model', y='F1_Score', hue='Dataset', data=df, palette='Blues')
    plt.ylim(0.80, 0.95) 
    plt.title('Impatto delle Trap Features sulla Precisione (F1)', fontsize=14)
    plt.ylabel('F1-Score (Weighted)')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'defense_performance_impact.png'))
    print("[PLOT] Salvato defense_performance_impact.png")

if __name__ == "__main__":
    plot_defense()