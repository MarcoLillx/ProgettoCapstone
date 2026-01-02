import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_PATH = os.path.join(BASE_DIR, 'logs', 'vulnerability_curve_advanced.csv')
FIGURE_DIR = os.path.join(BASE_DIR, 'figure', 'analysis')

os.makedirs(FIGURE_DIR, exist_ok=True)
sns.set_theme(style="white") # Stile pulito per doppio asse

def plot_advanced_curve():
    if not os.path.exists(LOGS_PATH):
        print("Esegui prima vulnerability_advanced.py!")
        return

    df = pd.read_csv(LOGS_PATH)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- ASSE SINISTRO: TEMPO (Rosso) ---
    color = 'tab:red'
    ax1.set_xlabel('Number of Features (Dimensionality)', fontsize=12)
    ax1.set_ylabel('Attack Generation Time (s)', color=color, fontsize=12)
    sns.lineplot(x='Num_Features', y='Attack_Time', data=df, marker='o', color=color, ax=ax1, linewidth=2.5, label='Time')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- ASSE DESTRO: SUCCESS RATE (Blu) ---
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Attack Success Rate (ASR)', color=color, fontsize=12)
    sns.lineplot(x='Num_Features', y='ASR', data=df, marker='s', color=color, ax=ax2, linewidth=2.5, linestyle='--', label='ASR')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.1) # ASR va da 0 a 1

    # Titolo
    plt.title('Advanced Vulnerability Analysis: Time vs Success Rate', fontsize=14)
    
    # Uniamo le legende (opzionale, ma elegante)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    output_path = os.path.join(FIGURE_DIR, 'vulnerability_advanced_zoo.png')
    plt.savefig(output_path)
    print(f"[PLOT] Salvato: {output_path}")

if __name__ == "__main__":
    plot_advanced_curve()