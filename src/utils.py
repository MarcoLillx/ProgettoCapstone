import numpy as np
import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import xgboost as xgb
from joblib import dump, load

# ART Imports
from art.utils import to_categorical
from art.estimators.classification import BlackBoxClassifier, XGBoostClassifier, SklearnClassifier
from art.attacks.evasion import ZooAttack, HopSkipJump, BoundaryAttack, SignOPTAttack
from numpy.linalg import norm

warnings.simplefilter(action='ignore', category=FutureWarning)

# Configurazione Classi
NB_CLASSES_REAL = 2 
CLIP_VALUES = (0.0, 1.0) 

def train_model(type_model, X, y, seed, estimator=None):
    print(f"[TRAIN] Training {type_model}...")
    model = None
    start_train = time.time()

    if type_model == 'dt':
        model = DecisionTreeClassifier(criterion='entropy', random_state=seed)
        model.fit(X, y)

    elif type_model == 'rf':
        n_est = estimator if estimator else 100
        model = RandomForestClassifier(n_estimators=n_est, criterion='gini', random_state=seed)
        model.fit(X, y)

    elif type_model == 'xgb':
        n_est = estimator if estimator else 100
        model = xgb.XGBClassifier(
            n_estimators=n_est, 
            objective='binary:logistic',
            booster='gbtree',
            tree_method='hist', 
            seed=seed
        )
        model.fit(X, y)

    end_train = time.time()
    duration = end_train - start_train
    print(f"[TRAIN] {type_model} trained in {duration:.2f} seconds")
    
    # Store training duration on the model for optional downstream use
    setattr(model, "train_duration", duration)

    return model, duration

def plot_save_confMatrix(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(title)
    plt.savefig(path)
    plt.close()

def test_model(type_model, model, X, y, save_path_cm, prefix_cm, label_names=['Normal', 'Attack']):
    print(f"[TEST] Testing {type_model}...")
    start_test = time.time()
    y_pred = model.predict(X)
    end_test = time.time()
    
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    overall_acc = accuracy_score(y, y_pred)
    avg_acc = balanced_accuracy_score(y, y_pred)
    
    time_elapsed = end_test - start_test

    print(f"--- Results for {type_model} ---")
    print(f"Time: {time_elapsed:.4f}s | F1(W): {f1:.4f} | Avg Acc: {avg_acc:.4f}")
    
    if save_path_cm:
        os.makedirs(save_path_cm, exist_ok=True)
        full_path = os.path.join(save_path_cm, f"cm_{prefix_cm}_{type_model}.png")
        plot_save_confMatrix(y, y_pred, label_names, full_path, f"CM {type_model} - {prefix_cm}")

    return {
        'Precision_W': precision,
        'Recall_W': recall,
        'F1_W': f1,
        'Overall_Acc': overall_acc,
        'Average_Acc': avg_acc,
        'Time_Test': time_elapsed
    }

def generate_adv_examples(type_attack, target_model, X, y, type_attacked_model, nb_features, max_iter=50):
    x_test_adv = None
    start_gen = time.time()
    
    # --- GESTIONE SPECIALE PER SIGN-OPT (Fake Multiclass) ---
    if type_attack == 'sign':
        def predict_wrapper_sign(x):
            x = np.array(x, dtype=np.float32)
            if hasattr(target_model, "predict_proba"):
                res = target_model.predict_proba(x)
            else:
                res = to_categorical(target_model.predict(x), NB_CLASSES_REAL)
            
            if res.ndim == 1: res = res.reshape(-1, 1)
            if res.shape[1] == 1:
                res = np.column_stack((1.0 - res, res))
            
            dummy = np.zeros((res.shape[0], 1), dtype=np.float32)
            return np.hstack((res, dummy)).astype(np.float32)

        classifier = BlackBoxClassifier(
            predict_wrapper_sign,
            input_shape=(nb_features,),
            nb_classes=3, 
            clip_values=CLIP_VALUES
        )

        attack = SignOPTAttack(
            estimator=classifier,
            targeted=False, 
            epsilon=0.001,
            num_trial=100,
            max_iter=max_iter,
            query_limit=1000, 
            k=200,
            alpha=0.2,
            beta=0.001,
            eval_perform=False,
            batch_size=64,
            verbose=False
        )
        x_test_adv = attack.generate(X)

    # --- ALTRI ATTACCHI ---
    else:
        if type_attacked_model == 'xgb':
            classifier = XGBoostClassifier(
                model=target_model, 
                nb_features=nb_features, 
                nb_classes=NB_CLASSES_REAL, 
                clip_values=CLIP_VALUES
            )
        else:
            classifier = SklearnClassifier(
                model=target_model, 
                clip_values=CLIP_VALUES
            )

        if type_attack == 'zoo':
            attack = ZooAttack(
                classifier=classifier, 
                confidence=0.0, 
                targeted=False, 
                learning_rate=1e-1, 
                max_iter=max_iter, 
                binary_search_steps=10, 
                initial_const=1e-3, 
                abort_early=True, 
                use_resize=False, 
                use_importance=False, 
                nb_parallel=1, 
                batch_size=1, 
                variable_h=0.2
            )
            x_test_adv = attack.generate(X)

        elif type_attack == 'hop':
            attack = HopSkipJump(
                classifier=classifier, 
                batch_size=64, 
                targeted=False, 
                norm='inf', 
                max_iter=max_iter, 
                max_eval=10000,
                init_eval=100, 
                init_size=100, 
                verbose=False
            )
            x_test_adv = attack.generate(X, x_adv_init=X)

        elif type_attack == 'bound':
            attack = BoundaryAttack(
                estimator=classifier, 
                batch_size=64, 
                delta=0.01, 
                epsilon=0.01,
                step_adapt=0.667, 
                max_iter=max_iter, 
                num_trial=25, 
                sample_size=20, 
                init_size=100,
                min_epsilon=0.0, 
                verbose=False, 
                targeted=False
            )
            x_test_adv = attack.generate(X, x_adv_init=X)

    end_gen = time.time()
    duration = end_gen - start_gen
    
    return x_test_adv, duration

def cosine_similarity(X_adv, X):
    sum_sim = 0
    for i in range(len(X_adv)):
        norm_adv = norm(X_adv[i]) + 1e-10
        norm_x = norm(X[i]) + 1e-10
        cosine = np.dot(X_adv[i], X[i]) / (norm_adv * norm_x)
        sum_sim += cosine
    return sum_sim / len(X_adv)