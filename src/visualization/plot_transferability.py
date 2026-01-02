import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_PATH = os.path.join(BASE_DIR, 'logs', 'transferability_results.csv')
FIGURE_DIR = os.path.join(BASE_DIR, 'figure', 'analysis')

os.makedirs(FIGURE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def plot_transf():
    if not os.path.exists(LOGS_PATH):
        print("Esegui prima transferability.py!")
        return

    df = pd.read_csv(LOGS_PATH)

    plt.figure(figsize=(8, 6))
    
    # Barplot dell'ASR (Attack Success Rate)
    # Più è alto, più l'attacco si è trasferito
    ax = sns.barplot(x='Target_Model', y='Transfer_ASR', hue='Role', data=df, palette={'SOURCE': 'firebrick', 'TARGET': 'steelblue'})
    
    plt.title('Transferability of Adversarial Examples (Source: DT)', fontsize=14)
    plt.ylabel('Attack Success Rate (ASR)')
    plt.ylim(0, 1.1)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'transferability_plot.png'))
    print("[PLOT] Salvato transferability_plot.png")

if __name__ == "__main__":
    plot_transf()