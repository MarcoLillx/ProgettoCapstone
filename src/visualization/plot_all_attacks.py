import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_PATH = os.path.join(BASE_DIR, 'logs', 'vulnerability_all_attacks.csv')
FIGURE_DIR = os.path.join(BASE_DIR, 'figure', 'analysis')

os.makedirs(FIGURE_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def plot_all():
    if not os.path.exists(LOGS_PATH):
        print("Esegui prima vulnerability_all.py!")
        return

    df = pd.read_csv(LOGS_PATH)

    # --- GRAFICO 1: TEMPO DI ATTACCO ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Num_Features', y='Attack_Time', hue='Attack', data=df, marker='o', linewidth=2.5)
    plt.title('Impact of Dimensionality on Attack Time (All Attacks)', fontsize=14)
    plt.ylabel('Time (seconds)')
    plt.xlabel('Number of Features')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(FIGURE_DIR, 'all_attacks_time.png'))
    print("[PLOT] Salvato all_attacks_time.png")

    # --- GRAFICO 2: ATTACK SUCCESS RATE ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Num_Features', y='ASR', hue='Attack', data=df, marker='s', linewidth=2.5)
    plt.title('Impact of Dimensionality on Success Rate (All Attacks)', fontsize=14)
    plt.ylabel('Attack Success Rate (ASR)')
    plt.xlabel('Number of Features')
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(FIGURE_DIR, 'all_attacks_asr.png'))
    print("[PLOT] Salvato all_attacks_asr.png")

if __name__ == "__main__":
    plot_all()