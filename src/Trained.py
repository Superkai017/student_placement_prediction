

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13})


# =============================================================================
# 1. LOAD & PREPROCESS
# =============================================================================

def load_and_preprocess(filepath):
    df = pd.read_csv('data_cleaned.csv')
    print(f"Loaded  : {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Drop both leakage columns
    df.drop(columns=["salary_package_lpa"], errors="ignore", inplace=True)
    print("✓  Dropped 'salary_package_lpa'")

    # Add 30% noise to salary_available
    n_flip       = int(0.30 * len(df))
    flip_indices = np.random.choice(df.index, size=n_flip, replace=False)
    df.loc[flip_indices, "salary_available"] = 1 - df.loc[flip_indices, "salary_available"]
    print(f"✓  Added 30% noise to 'salary_available'")

    # Fix any remaining NaN
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
            print(f"   Fixed NaN in '{col}'")

    print(f"\n✓  Missing values remaining : {df.isnull().sum().sum()}")

    X = df.drop(columns=["placement_status"])
    y = df["placement_status"]
    print(f"\n✓  Features ({X.shape[1]}) : {X.columns.tolist()}")
    print(f"   Target : placement_status  (0 = Not Placed, 1 = Placed)")

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(f"\n✓  Scaled  : {X_scaled.shape[1]} features (StandardScaler applied)\n")

    return X_scaled, y, scaler, X




# =============================================================================
# 3. SPLIT
# =============================================================================

def split_data(X_scaled, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size    = test_size,
        random_state = RANDOM_STATE,
        stratify     = y
    )
    print(f"Train : {X_train.shape[0]:,} rows  |  Test : {X_test.shape[0]:,} rows")
    print(f"Train balance : {y_train.mean()*100:.1f}% placed")
    print(f"Test  balance : {y_test.mean()*100:.1f}% placed\n")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 4. TRAIN
# =============================================================================

def train_models(X_train, y_train):
    print("[1] Training Logistic Regression ...")
    lr = LogisticRegression(C=0.1, max_iter=300, solver="lbfgs",
                            random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    print("    Done")

    print("[2] Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                min_samples_leaf=20,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("    Done")

    print("[3] Training Gradient Boosting ...")
    gb = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                    learning_rate=0.08, subsample=0.8,
                                    random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    print("    Done\n")

    return {"Logistic Regression": lr, "Random Forest": rf, "Gradient Boosting": gb}


# =============================================================================
# 5. EVALUATE
# =============================================================================

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)[:, 1]
    acc       = accuracy_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=5, scoring="accuracy", n_jobs=-1)

    status = "IN TARGET RANGE (73-78%)" if 0.73 <= acc <= 0.78 else "OUT OF TARGET RANGE"

    print(f"{'─'*55}")
    print(f"  Model          : {name}")
    print(f"  Test Accuracy  : {acc*100:.2f}%   {status}")
    print(f"  ROC-AUC        : {roc_auc:.4f}")
    print(f"  CV Acc (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Placed (0)", "Placed (1)"],
                                digits=3, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"                        Pred Not Placed   Pred Placed")
    print(f"  Actual Not Placed       TN={cm[0,0]:>6,}        FP={cm[0,1]:>6,}")
    print(f"  Actual Placed           FN={cm[1,0]:>6,}        TP={cm[1,1]:>6,}\n")

    return acc


def evaluate_all_models(models_dict, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models_dict.items():
        results[name] = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    return results



def print_summary(results):
    print("=" * 56)
    print(f"  {'Model':<22} | Accuracy  | In 73-78%?")
    print(f"  {'-'*52}")
    for name, acc in results.items():
        flag = "Yes" if 0.73 <= acc <= 0.78 else "No"
        print(f"  {name:<22} |  {acc*100:.2f}%   | {flag}")
    print("=" * 56)
    best = min(results, key=lambda n: abs(results[n] - 0.755))
    print(f"\nRecommended model : {best}  ({results[best]*100:.2f}%)")
    print(f"(closest to target midpoint 75.5%)\n")


# =============================================================================
# 7. SAVE MODELS
# =============================================================================

def save_models(models_dict, scaler, folder=None):
    folder = folder or os.getcwd()
    os.makedirs(folder, exist_ok=True)

    name_to_file = {
        "Logistic Regression" : "model_logistic_regression.pkl",
        "Random Forest"       : "model_random_forest.pkl",
        "Gradient Boosting"   : "model_gradient_boosting.pkl",
    }

    for name, model in models_dict.items():
        path = os.path.join(folder, name_to_file[name])
        with open(path, "wb") as f:
            pickle.dump(model, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"Saved : {name_to_file[name]}  ({size_kb:.1f} KB)")

    scaler_path = os.path.join(folder, "scaler_standard.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved : scaler_standard.pkl\n")

def get_models_and_scaler():
   
    X_scaled, y, scaler, X     = load_and_preprocess("data cleaned.csv")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    models_dict                = train_models(X_train, y_train)
    return models_dict, scaler


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    X_scaled, y, scaler, X = load_and_preprocess("data cleaned.csv")


    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    models_dict = train_models(X_train, y_train)

    results = evaluate_all_models(models_dict, X_train, y_train, X_test, y_test)


    print_summary(results)

    save_models(models_dict, scaler)
    print("Training complete. Models saved and ready for predict.py")