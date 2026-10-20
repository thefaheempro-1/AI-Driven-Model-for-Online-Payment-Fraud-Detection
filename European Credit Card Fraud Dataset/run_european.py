"""
=============================================================================
  AI-DRIVEN FRAUD DETECTION — THESIS EXPERIMENT PIPELINE
  Author  : Faheem Abbas
  Dataset : European Credit Card Fraud Dataset (284,807 transactions)
  Source  : D:\Educational\Research\DataSets\European Credit Card Fraud Dataset
  Models  : Random Forest | XGBoost | SVM | Isolation Forest | Autoencoder
  Metrics : Precision, Recall, F1, AUC-ROC, FPR, Inference Time
  Extras  : Stratified 5-Fold CV, Friedman Test, SHAP Analysis
  Output  : european_model_results.csv + european_*.png
=============================================================================

HOW TO RUN
----------
1. Install dependencies (one-time):
       pip install kagglehub pandas numpy scikit-learn xgboost tensorflow shap scipy matplotlib seaborn

2. Place your kaggle.json API key in ~/.kaggle/kaggle.json
   (Download from https://www.kaggle.com/settings → API → Create Token)

3. Run the script:
       python fraud_detection_thesis.py

   The script will:
   - Download the dataset automatically via KaggleHub (if not cached)
   - Preprocess, train all 5 models, evaluate, cross-validate
   - Print a full results table and save plots + SHAP figures

4. To switch to the IEEE-CIS secondary dataset later, set at the bottom:
       run_pipeline(dataset="ieee")
=============================================================================
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ─── Data / numerics ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─── Sklearn ─────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ─── XGBoost ─────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier

# ─── TensorFlow / Keras ──────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ─── SHAP ────────────────────────────────────────────────────────────────────
import shap

# ─── Statistical test ────────────────────────────────────────────────────────
from scipy.stats import friedmanchisquare

# ─── Plotting ────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ─── KaggleHub (dataset downloader) ──────────────────────────────────────────
import kagglehub


# =============================================================================
# 1. DATASET LOADING
# =============================================================================

def load_european_dataset():
    """Load the European Credit Card Fraud dataset from local CSV file."""
    print("\n[1/6] Loading European Credit Card Fraud Dataset from local file …")

    # ── Local file path ───────────────────────────────────────────────────────
    csv_path = r"D:\Educational\Research\DataSets\European Credit Card Fraud Dataset\European Credit Card Fraud Dataset.csv"

    print("      Reading European Credit Card Fraud Dataset.csv …")
    df = pd.read_csv(csv_path)
    print(f"      Loaded {len(df):,} rows x {df.shape[1]} columns")
    print(f"      Fraud rate: {df['Class'].mean()*100:.4f}% "
          f"({df['Class'].sum()} fraudulent / {(df['Class']==0).sum()} legitimate)")
    return df


def load_ieee_dataset():
    """Load the IEEE-CIS Fraud Detection dataset from local CSV files."""
    print("\n[1/6] Loading IEEE-CIS Fraud Detection Dataset from local files …")

    # ── Local file paths ──────────────────────────────────────────────────────
    train_path = r"D:\Educational\Research\DataSets\IEEE-CIS Fraud Detection Dataset\fraud_train_preprocessed.csv"
    test_path  = r"D:\Educational\Research\DataSets\IEEE-CIS Fraud Detection Dataset\fraud_test_preprocessed.csv"

    # Load both files
    print("      Reading fraud_train_preprocessed.csv …")
    df_train = pd.read_csv(train_path)
    print("      Reading fraud_test_preprocessed.csv …")
    df_test  = pd.read_csv(test_path)

    # Combine into one dataframe (pipeline will re-split with stratification)
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"      Combined: {len(df_train):,} train + {len(df_test):,} test = {len(df):,} total rows")

    # Normalise target column name
    if "isFraud" in df.columns:
        df.rename(columns={"isFraud": "Class"}, inplace=True)
    elif "target" in df.columns:
        df.rename(columns={"target": "Class"}, inplace=True)

    # Drop non-numeric columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"      Dropping non-numeric columns: {non_numeric}")
        df.drop(columns=non_numeric, inplace=True)

    # Drop rows where Class is missing
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)

    print(f"      Final dataset: {len(df):,} rows x {df.shape[1]} columns")
    print(f"      Fraud rate: {df['Class'].mean()*100:.4f}% ({df['Class'].sum():,} fraud / {(df['Class']==0).sum():,} legitimate)")
    return df


# =============================================================================
# 2. PREPROCESSING  (Section 3.4 of thesis)
# =============================================================================

def preprocess(df):
    """
    Steps:
    2.1  Missing value check
    2.2  Feature scaling (Amount, Time)
    2.3  Stratified 80/20 train-test split
    Returns X_train, X_test, y_train, y_test, feature_names
    """
    print("\n[2/6] Preprocessing …")

    # 2.1 Missing values
    missing = df.isnull().sum().sum()
    print(f"      Missing values: {missing}")

    # Features / target
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].values
    feature_names = X.columns.tolist()

    # 2.2 Scale Amount and Time (V1-V28 are already PCA-scaled)
    scaler = StandardScaler()
    cols_to_scale = [c for c in ["Amount", "Time"] if c in X.columns]
    if cols_to_scale:
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    # 2.3 Stratified 80/20 split (random_state=42 for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"      Fraud in train: {y_train.sum()}  |  Fraud in test: {y_test.sum()}")
    return X_train, X_test, y_train, y_test, feature_names


# =============================================================================
# 3. MODEL DEFINITIONS  (Section 3.5 of thesis)
# =============================================================================

def build_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )


def build_xgboost(y_train):
    # scale_pos_weight = count(negative) / count(positive)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos)
    print(f"      XGBoost scale_pos_weight = {spw}")
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        gamma=0.1,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )


def build_svm():
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    )


def build_isolation_forest():
    return IsolationForest(
        n_estimators=300,
        contamination=0.002,
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )


def build_autoencoder(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(24, activation="relu")(inp)
    x = Dense(16, activation="relu")(x)
    bottleneck = Dense(8, activation="relu", name="bottleneck")(x)
    x = Dense(16, activation="relu")(bottleneck)
    x = Dense(24, activation="relu")(x)
    out = Dense(input_dim, activation="sigmoid")(x)
    ae = Model(inputs=inp, outputs=out)
    ae.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return ae


# =============================================================================
# 4. TRAINING & EVALUATION HELPERS
# =============================================================================

def evaluate_supervised(name, model, X_test, y_test, y_prob):
    """Compute all metrics for a supervised classifier."""
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "Model": name,
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC-ROC":   round(roc_auc_score(y_test, y_prob), 4),
        "FPR":       round(fpr, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }


def evaluate_unsupervised(name, scores_normal, scores_test, y_test, threshold_pct):
    """
    Isolation Forest  → anomaly_score (lower = more anomalous)
    Autoencoder       → reconstruction error (higher = more anomalous)
    threshold_pct: percentile of *normal* scores used as decision boundary
    """
    threshold = np.percentile(scores_normal, threshold_pct)
    y_pred = (scores_test > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        auc = roc_auc_score(y_test, scores_test)
    except Exception:
        auc = float("nan")
    return {
        "Model": name,
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC-ROC":   round(auc, 4),
        "FPR":       round(fpr_val, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }


# =============================================================================
# 5. CROSS-VALIDATION  (Section 3.7.1)
# =============================================================================

def cross_validate_supervised(models_dict, X_train, y_train):
    """
    Stratified 5-fold CV for supervised models.
    Returns a dict: model_name → list of 5 AUC-ROC scores.
    """
    print("\n[5/6] Stratified 5-Fold Cross-Validation …")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models_dict.items():
        fold_aucs = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]

            # SVM subsample (O(n²) complexity)
            if "SVM" in name:
                sample_idx = np.random.choice(
                    np.where(ytr == 0)[0], size=30000, replace=False
                )
                fraud_idx = np.where(ytr == 1)[0]
                idx = np.concatenate([sample_idx, fraud_idx])
                np.random.shuffle(idx)
                Xtr, ytr = Xtr[idx], ytr[idx]

            model.fit(Xtr, ytr)
            prob = model.predict_proba(Xval)[:, 1]
            auc = roc_auc_score(yval, prob)
            fold_aucs.append(round(auc, 4))
            print(f"      {name:20s} | fold {fold} AUC-ROC = {auc:.4f}")

        cv_results[name] = fold_aucs
        print(f"      {name:20s} | mean = {np.mean(fold_aucs):.4f}  "
              f"std = {np.std(fold_aucs):.4f}\n")

    return cv_results


def friedman_test(cv_results):
    """Friedman non-parametric test across models (Section 3.7.2)."""
    print("[5b] Friedman Statistical Test …")
    groups = [np.array(v) for v in cv_results.values()]
    stat, p = friedmanchisquare(*groups)
    print(f"     Friedman χ² = {stat:.4f},  p-value = {p:.6f}")
    if p < 0.05:
        print("     → Statistically significant differences exist between models (p < 0.05)")
    else:
        print("     → No statistically significant differences (p ≥ 0.05)")
    return stat, p


# =============================================================================
# 6. SHAP ANALYSIS  (Section 3.7.3)
# =============================================================================

def shap_analysis(rf_model, xgb_model, X_test, feature_names, n_samples=500, prefix=""):
    """SHAP mean |shap| feature importance for RF and XGBoost.
    Fixed to work with all SHAP versions (list output, ndarray, 3D arrays).
    """
    print("\n[6/6] SHAP Feature Importance Analysis …")
    X_sample = X_test[:n_samples]
    feature_names_list = list(feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SHAP Mean Absolute Feature Importance", fontsize=14, fontweight="bold")

    for ax, (name, model) in zip(axes, [("Random Forest", rf_model),
                                         ("XGBoost", xgb_model)]):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Handle all SHAP output formats across library versions
            if isinstance(shap_values, list):
                # Old SHAP: list of arrays [class0, class1]
                sv = shap_values[1]
            else:
                sv = shap_values

            # Handle 3D arrays (new SHAP versions: samples x features x classes)
            if hasattr(sv, 'ndim') and sv.ndim == 3:
                sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]

            # Flatten to 1D and convert to plain Python types (fixes index errors)
            mean_abs = np.abs(sv).mean(axis=0).flatten()
            n_top = min(15, len(mean_abs))
            top_idx      = [int(i)   for i in np.argsort(mean_abs)[::-1][:n_top]]
            top_features = [feature_names_list[i] for i in top_idx]
            top_values   = [float(mean_abs[i])    for i in top_idx]

            ax.barh(top_features[::-1], top_values[::-1], color="#E07B54")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"{name}")
            ax.tick_params(axis="y", labelsize=9)
            print(f"      {name} SHAP done — top feature: {top_features[0]} ({top_values[0]:.4f})")

        except Exception as e:
            print(f"      WARNING: SHAP failed for {name}: {e}")
            ax.set_title(f"{name} — SHAP unavailable")

    plt.tight_layout()
    fname = f"{prefix}shap_feature_importance.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"      Saved → {fname}")
    plt.close()

def plot_results(results_df, prefix=""):
    metrics = ["Precision", "Recall", "F1", "AUC-ROC"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Performance Comparison — Credit Card Fraud Detection",
                 fontsize=14, fontweight="bold")

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for ax, metric in zip(axes.flat, metrics):
        bars = ax.bar(results_df["Model"], results_df[metric],
                      color=palette[:len(results_df)])
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fname = f"{prefix}model_performance_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"      Saved → {fname}")
    plt.close()


def plot_cv_results(cv_results, prefix=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = range(len(cv_results))
    ax.boxplot(cv_results.values(), positions=positions, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(cv_results.keys()), rotation=15)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("5-Fold Cross-Validation AUC-ROC Distribution")
    plt.tight_layout()
    fname = f"{prefix}cross_validation_results.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"      Saved → {fname}")
    plt.close()


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def run_pipeline(dataset="european"):
    print("=" * 65)
    print("  FRAUD DETECTION THESIS — FULL EXPERIMENT PIPELINE")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    if dataset == "european":
        df = load_european_dataset()
    else:
        df = load_ieee_dataset()

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    results = []

    # ── 3A: Random Forest ─────────────────────────────────────────────────────
    print("\n[3/6] Training models …")
    print("      → Random Forest")
    rf = build_random_forest()
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    rf_prob = rf.predict_proba(X_test)[:, 1]
    inf_time = (time.perf_counter() - t0) / len(X_test) * 1000  # ms per tx
    res = evaluate_supervised("Random Forest", rf, X_test, y_test, rf_prob)
    res["TrainTime(s)"] = round(train_time, 2)
    res["InfTime(ms/tx)"] = round(inf_time, 4)
    results.append(res)
    print(f"         Done — F1={res['F1']}  AUC={res['AUC-ROC']}  "
          f"train={train_time:.1f}s")

    # ── 3B: XGBoost ───────────────────────────────────────────────────────────
    print("      → XGBoost")
    xgb = build_xgboost(y_train)
    t0 = time.perf_counter()
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    inf_time = (time.perf_counter() - t0) / len(X_test) * 1000
    res = evaluate_supervised("XGBoost", xgb, X_test, y_test, xgb_prob)
    res["TrainTime(s)"] = round(train_time, 2)
    res["InfTime(ms/tx)"] = round(inf_time, 4)
    results.append(res)
    print(f"         Done — F1={res['F1']}  AUC={res['AUC-ROC']}  "
          f"train={train_time:.1f}s")

    # ── 3C: SVM (subsample) ───────────────────────────────────────────────────
    print("      → SVM (60,000-sample subsample for tractability)")
    svm_idx = np.concatenate([
        np.random.choice(np.where(y_train == 0)[0], size=59000, replace=False),
        np.where(y_train == 1)[0]
    ])
    np.random.shuffle(svm_idx)
    svm = build_svm()
    t0 = time.perf_counter()
    svm.fit(X_train[svm_idx], y_train[svm_idx])
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    svm_prob = svm.predict_proba(X_test)[:, 1]
    inf_time = (time.perf_counter() - t0) / len(X_test) * 1000
    res = evaluate_supervised("SVM", svm, X_test, y_test, svm_prob)
    res["TrainTime(s)"] = round(train_time, 2)
    res["InfTime(ms/tx)"] = round(inf_time, 4)
    results.append(res)
    print(f"         Done — F1={res['F1']}  AUC={res['AUC-ROC']}  "
          f"train={train_time:.1f}s")

    # ── 3D: Isolation Forest ──────────────────────────────────────────────────
    print("      → Isolation Forest")
    iso = build_isolation_forest()
    t0 = time.perf_counter()
    iso.fit(X_train)
    train_time = time.perf_counter() - t0

    # Anomaly score: score_samples returns negative values; higher = more normal
    # Flip sign so higher = more anomalous (consistent with thesis framing)
    t0 = time.perf_counter()
    iso_scores_test = -iso.score_samples(X_test)
    inf_time = (time.perf_counter() - t0) / len(X_test) * 1000
    iso_scores_train_legit = -iso.score_samples(X_train[y_train == 0])

    # Threshold = 99th percentile of legitimate training scores
    res = evaluate_unsupervised(
        "Isolation Forest",
        iso_scores_train_legit,
        iso_scores_test,
        y_test,
        threshold_pct=99
    )
    res["TrainTime(s)"] = round(train_time, 2)
    res["InfTime(ms/tx)"] = round(inf_time, 4)
    results.append(res)
    print(f"         Done — F1={res['F1']}  AUC={res['AUC-ROC']}  "
          f"train={train_time:.1f}s")

    # ── 3E: Autoencoder ───────────────────────────────────────────────────────
    print("      → Autoencoder (training on legitimate transactions only)")
    X_train_legit = X_train[y_train == 0]
    ae = build_autoencoder(X_train.shape[1])
    es = EarlyStopping(patience=10, restore_best_weights=True)

    t0 = time.perf_counter()
    ae.fit(
        X_train_legit, X_train_legit,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )
    train_time = time.perf_counter() - t0

    # Reconstruction error
    t0 = time.perf_counter()
    recon_test = ae.predict(X_test, verbose=0)
    inf_time = (time.perf_counter() - t0) / len(X_test) * 1000
    ae_scores_test = np.mean(np.power(X_test - recon_test, 2), axis=1)

    recon_legit_train = ae.predict(X_train_legit, verbose=0)
    ae_scores_train_legit = np.mean(
        np.power(X_train_legit - recon_legit_train, 2), axis=1
    )

    res = evaluate_unsupervised(
        "Autoencoder",
        ae_scores_train_legit,
        ae_scores_test,
        y_test,
        threshold_pct=99
    )
    res["TrainTime(s)"] = round(train_time, 2)
    res["InfTime(ms/tx)"] = round(inf_time, 4)
    results.append(res)
    print(f"         Done — F1={res['F1']}  AUC={res['AUC-ROC']}  "
          f"train={train_time:.1f}s")

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n[4/6] CONSOLIDATED RESULTS")
    print("=" * 65)
    results_df = pd.DataFrame(results)
    display_cols = ["Model", "Precision", "Recall", "F1",
                    "AUC-ROC", "FPR", "TrainTime(s)", "InfTime(ms/tx)"]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 65)
    csv_fname = f"{prefix}model_results.csv"
    results_df.to_csv(csv_fname, index=False)
    print(f"      Saved → {csv_fname}")

    # ── Cross-Validation ──────────────────────────────────────────────────────
    supervised_models = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "SVM": svm
    }
    cv_results = cross_validate_supervised(supervised_models, X_train, y_train)
    friedman_test(cv_results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_results(results_df[display_cols[:6]])
    plot_cv_results(cv_results)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_analysis(rf, xgb, X_test, feature_names)

    print(f"\n✅  Pipeline complete for [{dataset.upper()}]. Output files:")
    print(f"   • {prefix}model_results.csv")
    print(f"   • {prefix}model_performance_comparison.png")
    print(f"   • {prefix}cross_validation_results.png")
    print(f"   • {prefix}shap_feature_importance.png\n")


# =============================================================================
if __name__ == "__main__":
    # ── PRIMARY: European Credit Card Fraud Dataset ──────────────────────────
    run_pipeline(dataset="european")

    # ── SECONDARY: IEEE-CIS Fraud Detection Dataset ───────────────────────────
    run_pipeline(dataset="ieee")
