# Cybersecurity Capstone Project: Adversarial Attacks on Critical Infrastructure (Water Treatment)

This project is a case study on the robustness of Machine Learning-based Intrusion Detection Systems (IDS) in Cyber-Physical Systems (CPS).

The work replicates the methodology of the thesis *"Adversarial Attacks on IDS and Multidomain Impact Analysis for Threat Intelligence in Military Automotive Scenarios"* (Barletta & del Vescovo), applying it to the domain of critical water infrastructures (**Water Treatment**) using the **CISS 2020 (SWaT)** dataset.

## Objectives
*   **Domain Transfer:** Verify if Black-Box attacks effective in the Automotive domain are also effective in the Water Treatment domain.
*   **Robustness Evaluation:** Evaluate Decision Tree, Random Forest, and XGBoost models against **ZOO**, **Boundary**, **HopSkipJump**, and **Sign-OPT** attacks.
*   **Efficiency vs. Security:** Analyze the trade-off between Feature Engineering (efficiency) and the computational cost of attacks (vulnerability).
*   **Defense Validation:** Implement and test mitigation strategies like Adversarial Obfuscation and Adversarial Training.

## Key Findings
*   **Random Forest Vulnerability:** Unlike in the Automotive domain, the Random Forest model was completely bypassed by the **Sign-OPT** attack (F1-Score dropped to ~0.01), although generating the attack required significant time (~13 minutes for 50 samples).
*   **XGBoost Resilience:** XGBoost proved to be the most balanced model, showing high resilience against geometric attacks (HopSkipJump) where other models failed.
*   **The Security Paradox:** Removing redundant features via Feature Engineering improved training efficiency by 13% but inadvertently accelerated the generation of ZOO attacks by **28%**, reducing the defender's reaction window.
*   **Attacker's Sweet Spot:** Advanced analysis revealed a non-linear relationship between dimensionality and vulnerability, identifying a "sweet spot" (10-25 features) where attacks are most efficient.

## Project Structure
*   `src/`: Python source code.
    *   **Preprocessing & EDA:**
        *   `pre_elaboration_ciss.py`: Cleaning, merging, and initial labeling of the CISS dataset.
        *   `eda.py`: Exploratory Data Analysis (class distribution, correlation matrices, boxplots).
        *   `feature_engineering.py`: Advanced preprocessing, removal of constant and highly correlated features.
    *   **Core Pipeline:**
        *   `main.py`: Model training, execution of adversarial attacks, and metric evaluation.
        *   `utils.py`: Helper functions and wrappers for the Adversarial Robustness Toolbox (ART).
    *   **Advanced Analysis:**
        *   `xai_analysis.py`: Runs SHAP (Explainable AI) analysis.
        *   `plot_comparison.py`: Generates charts comparing performance/timing (Processed vs Refined).
        *   `vulnerability_advanced.py` & `plot_advanced.py`: Stress test for the Vulnerability Curve (Time vs ASR).
        *   `vulnerability_all.py` & `plot_all_attacks.py`: Comparative analysis of all attacks vs dimensionality.
        *   `transferability.py` & `plot_transferability.py`: Gray-Box attack analysis (Source: DT -> Target: RF/XGB).
    *   **Defenses:**
        *   `defense_experiment.py` & `plot_defense.py`: Adversarial Obfuscation (Trap Features).
        *   `adversarial_training.py` & `plot_adv_training.py`: Active defense via retraining.
    *   `clean.py`: Utility to reset the environment.
*   `dataset/`: Contains raw and processed data (ignored by git).
*   `models/`: Trained models (.joblib).
*   `adv_examples/`: Generated adversarial examples (.txt).
*   `figure/`: Confusion matrices and EDA plots.
*   `logs/`: Metric reports in CSV format.

## Technologies Used
*   **Language:** Python 3.x
*   **ML Libraries:** Scikit-Learn, XGBoost, SHAP
*   **Adversarial ML:** Adversarial Robustness Toolbox (ART)
*   **Data Processing:** Pandas, NumPy

## Usage Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.\.venv\Scripts\Activate.ps1
# Or Linux/Mac:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Download the CISS 2020 dataset (Target & CISS_OL Excel files) and place them in dataset/raw/.
Run the initial preprocessing script:

```bash
python src/pre_elaboration_ciss.py
```

### 3. Exploratory Data Analysis (EDA)
Generate analysis plots to understand class imbalance and correlations:
```bash
python src/eda.py
```

### 4. Feature Engineering
Refine the dataset by removing constant and correlated features (>0.99) and creates a new dataset " `class_refined.csv` ":
```bash
python src/feature_engineering.py
```

### 5. Training and Attacks (Main Execution)
Train the models (DT, RF, XGB) and execute the attacks (ZOO, Boundary, HopSkipJump, Sign-OPT):
```bash
python src/main.py
```

### 6. Comparison Analysis
Generate charts to visualize the trade-off between efficiency and security:
```bash
python src/plot_comparison.py
```

### 7. Explainable AI (XAI) Analysis
Run SHAP analysis to visualize feature importance and perturbation effects on adversarial examples:
```bash
python src/xai_analysis.py
```

### 8. Mitigation Strategies
Test passive and active defenses:
```bash
# 1. Adversarial Obfuscation (Trap Features)
python src/defense_experiment.py
python src/plot_defense.py

# 2. Adversarial Training (Active Defense)
python src/adversarial_training.py
python src/plot_adv_training.py
```

### 9. Advanced Vulnerability & Transferability
```bash
# Vulnerability Curve (Stress Test)
python src/vulnerability_advanced_zoo.py
python src/plot_advanced_zoo.py

# Comparative Analysis (All Attacks)
python src/vulnerability_all.py
python src/plot_all_attacks.py

# Transferability (Gray-Box)
python src/transferability.py
python src/plot_transferability.py
```


## Results Summary (Sample)

| Model | Scenario | Attack | F1-Score (Weighted) | Attack Time (50 samples) |
| :--- | :--- | :--- | :--- | :--- |
| **DT** | Normal | - | 0.88 | - |
| **DT** | Adversarial | ZOO | 0.67 | **3.40s** (Fastest) |
| **RF** | Normal | - | 0.90 | - |
| **RF** | Adversarial | Sign-OPT | **0.011** (Failed) | 793.67s |
| **XGB** | Normal | - | **0.91** | - |
| **XGB** | Adversarial | HopSkipJump | **0.87** (Robust) | 8.07s |

## References
*   **Reference Thesis:** Barletta, V. S., & del Vescovo, S. (2024). *Adversarial Attacks on IDS and Multidomain Impact Analysis for Threat Intelligence in Military Automotive Scenarios*.
*   **Dataset:** iTrust. (2020). *CISS 2020 Dataset: Critical Infrastructure Security Showdown*. Secure Water Treatment (SWaT) Testbed.
*   **Library:** Nicolae, M. I., et al. (2018). *Adversarial Robustness Toolbox v1.0.0*.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
