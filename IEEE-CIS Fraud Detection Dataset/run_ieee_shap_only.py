"""
=============================================================================
  SHAP FEATURE IMPORTANCE — IEEE-CIS Fraud Detection (Standalone)
  Author   : Faheem Abbas
  Dataset  : IEEE-CIS Fraud Detection Dataset
  Purpose  : Run SHAP analysis only (RF + XGBoost already trained)

  REQUIREMENTS:
    - Saved models: ieee_fast_rf_model.pkl  +  ieee_fast_xgb_model.pkl
    - OR: set RETRAIN_MODELS = True to retrain quickly from scratch
  Output   : ieee_fast_shap_feature_importance.png
=============================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =============================================================================
# CONFIGURATION — edit these paths as needed
# =============================================================================
TRAIN_PATH   = r"D:\Educational\Research\DataSets\IEEE-CIS Fraud Detection Dataset\fraud_train_preprocessed.csv"
TEST_PATH    = r"D:\Educational\Research\DataSets\IEEE-CIS Fraud Detection Dataset\fraud_test_preprocessed.csv"

RF_MODEL_PATH  = "ieee_fast_rf_model.pkl"   # saved RF model
XGB_MODEL_PATH = "ieee_fast_xgb_model.pkl"  # saved XGB model

# Set True to retrain RF + XGBoost from scratch (if .pkl files don't exist)
RETRAIN_MODELS = not (os.path.exists(RF_MODEL_PATH) and os.path.exists(XGB_MODEL_PATH))

PREFIX       = "ieee_fast_"
SHAP_SAMPLES = 500   # number of test samples for SHAP (increase for more accuracy)

# =============================================================================
# 1. LOAD & PREPROCESS
# =============================================================================
def load_and_preprocess():
    print("[1/3] Loading and preprocessing data ...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"      Combined: {len(df):,} rows")

    # Normalise label column name
    if "isFraud" in df.columns:
        df.rename(columns={"isFraud": "Class"}, inplace=True)
    elif "target" in df.columns:
        df.rename(columns={"target": "Class"}, inplace=True)

    # Drop non-numeric columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df.drop(columns=non_numeric, inplace=True)

    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].values
    feature_names = list(X.columns)

    # Same preprocessing as main pipeline
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X.values)

    selector   = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X_scaled)
    selected_mask = selector.get_support()
    feature_names = [feature_names[i] for i, s in enumerate(selected_mask) if s]
    print(f"      Features after variance selection: {X_selected.shape[1]}")

    _, X_test, _, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Test set: {len(X_test):,} rows ({y_test.sum():,} fraud)")
    return X_selected, X_test, y, y_test, feature_names

# =============================================================================
# 2. LOAD OR RETRAIN MODELS
# =============================================================================
def get_models(X_train_full, y_train_full):
    if not RETRAIN_MODELS:
        print("[2/3] Loading saved models ...")
        rf  = joblib.load(RF_MODEL_PATH)
        xgb = joblib.load(XGB_MODEL_PATH)
        print(f"      Loaded  → {RF_MODEL_PATH}")
        print(f"      Loaded  → {XGB_MODEL_PATH}")
    else:
        print("[2/3] Saved models not found — retraining RF + XGBoost (fast) ...")

        # Split for training (80%)
        X_tr, _, y_tr, _ = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        print("      → Random Forest ...")
        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        joblib.dump(rf, RF_MODEL_PATH)
        print(f"         Saved → {RF_MODEL_PATH}")

        print("      → XGBoost ...")
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
            n_jobs=-1, use_label_encoder=False
        )
        xgb.fit(X_tr, y_tr)
        joblib.dump(xgb, XGB_MODEL_PATH)
        print(f"         Saved → {XGB_MODEL_PATH}")

    return rf, xgb

# =============================================================================
# 3. SHAP ANALYSIS
# =============================================================================
def shap_analysis(rf_model, xgb_model, X_test, feature_names):
    print("[3/3] Running SHAP analysis ...")
    X_sample = X_test[:SHAP_SAMPLES]
    feature_names_list = list(feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "SHAP Mean Absolute Feature Importance — IEEE-CIS Fast Optimised",
        fontsize=14, fontweight="bold"
    )

    for ax, (name, model) in zip(axes, [("Random Forest", rf_model),
                                         ("XGBoost",       xgb_model)]):
        print(f"      Computing SHAP values for {name} (n={SHAP_SAMPLES}) ...")
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)

            # Handle both list-style (RF) and array-style (XGB) outputs
            if isinstance(sv, list):
                sv = sv[1]
            if hasattr(sv, 'ndim') and sv.ndim == 3:
                sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]

            mean_abs    = np.abs(sv).mean(axis=0).flatten()
            n_top       = min(15, len(mean_abs))
            top_idx     = [int(i)   for i in np.argsort(mean_abs)[::-1][:n_top]]
            top_features = [feature_names_list[i] for i in top_idx]
            top_values  = [float(mean_abs[i])    for i in top_idx]

            ax.barh(top_features[::-1], top_values[::-1], color="#4C72B0")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"{name} — IEEE-CIS")
            ax.tick_params(axis="y", labelsize=9)

            print(f"      {name} — Top 5 features:")
            for feat, val in zip(top_features[:5], top_values[:5]):
                print(f"         {feat:40s}  {val:.4f}")

        except Exception as e:
            print(f"      WARNING: SHAP failed for {name}: {e}")
            ax.set_title(f"{name} — SHAP unavailable")

    plt.tight_layout()
    fname = f"{PREFIX}shap_feature_importance.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n      Saved → {fname}")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================
def run():
    print("=" * 60)
    print("  SHAP ANALYSIS — IEEE-CIS Fraud Detection (Standalone)")
    print("=" * 60)

    X_full, X_test, y_full, y_test, feature_names = load_and_preprocess()
    rf, xgb = get_models(X_full, y_full)
    shap_analysis(rf, xgb, X_test, feature_names)

    print("\n✅  SHAP analysis complete!")
    print(f"   • {PREFIX}shap_feature_importance.png\n")

if __name__ == "__main__":
    run()
