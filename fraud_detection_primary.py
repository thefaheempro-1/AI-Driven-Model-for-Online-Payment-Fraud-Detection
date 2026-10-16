
# =============================================================================
# AI-DRIVEN MODEL FOR ONLINE PAYMENT FRAUD DETECTION
# Primary Dataset: European Credit Card Fraud Dataset
# Author: Faheem Abbas | Roll No: F24BINFT3E2026
# Supervisor: Dr. Musarat Kareem
# The Islamia University of Bahawalpur
# =============================================================================
# INSTRUCTIONS:
# 1. Upload creditcard.csv to your Google Colab session
# 2. Run each cell block in order
# 3. All results, tables, and plots will be saved automatically
# =============================================================================

# ===========================================================================
# BLOCK 1 — INSTALL AND IMPORT ALL LIBRARIES
# ===========================================================================

# Run this first in Colab:
# !pip install xgboost shap imbalanced-learn scikit-learn tensorflow --quiet

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Core ML
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Scipy for statistical tests
from scipy.stats import friedmanchisquare, wilcoxon

# Fix all random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

print("=" * 65)
print("  FRAUD DETECTION EXPERIMENT — PRIMARY DATASET")
print("  European Credit Card Fraud Dataset")
print("=" * 65)
print(f"\nAll libraries loaded successfully.")
print(f"Random seed fixed at: {SEED}")

# ===========================================================================
# BLOCK 2 — LOAD AND EXPLORE DATASET
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 2: LOADING AND EXPLORING DATASET")
print("=" * 65)

# Load dataset — adjust path if needed
# In Colab: upload file then use: df = pd.read_csv('creditcard.csv')
df = pd.read_csv('creditcard.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Total transactions: {len(df):,}")
print(f"Total features: {df.shape[1]}")
print(f"\nClass distribution:")
print(f"  Legitimate (0): {(df['Class']==0).sum():,}  ({(df['Class']==0).mean()*100:.4f}%)")
print(f"  Fraudulent  (1): {(df['Class']==1).sum():,}  ({(df['Class']==1).mean()*100:.4f}%)")
print(f"\nClass imbalance ratio: {(df['Class']==0).sum()/(df['Class']==1).sum():.1f} : 1")

print(f"\nMissing values: {df.isnull().sum().sum()}")

print(f"\nAmount feature statistics:")
print(f"  Min:    ${df['Amount'].min():.2f}")
print(f"  Max:    ${df['Amount'].max():.2f}")
print(f"  Mean:   ${df['Amount'].mean():.2f}")
print(f"  Median: ${df['Amount'].median():.2f}")

print(f"\nFraud vs Legitimate - Amount comparison:")
print(f"  Fraud mean amount:      ${df[df['Class']==1]['Amount'].mean():.2f}")
print(f"  Legitimate mean amount: ${df[df['Class']==0]['Amount'].mean():.2f}")

# Feature distribution comparison for top fraud indicators
print(f"\nV14 distribution (strongest fraud indicator):")
print(f"  Fraud mean:      {df[df['Class']==1]['V14'].mean():.4f}")
print(f"  Legitimate mean: {df[df['Class']==0]['V14'].mean():.4f}")
print(f"  Separation:      {abs(df[df['Class']==1]['V14'].mean() - df[df['Class']==0]['V14'].mean()):.2f} std devs")

# ===========================================================================
# BLOCK 3 — DATA PREPROCESSING
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 3: DATA PREPROCESSING")
print("=" * 65)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Stratified 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print(f"\nStratified 80/20 split:")
print(f"  Training set:   {len(X_train):,} records  |  Fraud: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
print(f"  Test set:       {len(X_test):,} records  |  Fraud: {y_test.sum():,} ({y_test.mean()*100:.4f}%)")

# Class imbalance ratio for XGBoost
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"\nClass imbalance ratio (scale_pos_weight for XGBoost): {scale_pos_weight:.2f}")

# Feature scaling — only for SVM and Autoencoder
scaler = StandardScaler()
cols_to_scale = ['Amount', 'Time']

X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

print(f"\nFeature scaling applied to: {cols_to_scale}")
print(f"  Amount after scaling — mean: {X_train_scaled['Amount'].mean():.4f}, std: {X_train_scaled['Amount'].std():.4f}")
print(f"  Time after scaling   — mean: {X_train_scaled['Time'].mean():.4f}, std: {X_train_scaled['Time'].std():.4f}")
print(f"\nPreprocessing complete.")

# ===========================================================================
# BLOCK 4 — HELPER FUNCTIONS
# ===========================================================================

def evaluate_model(y_true, y_pred, y_prob, model_name, inference_time_ms):
    """Compute all six evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auc_roc   = roc_auc_score(y_true, y_prob)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc_pr    = average_precision_score(y_true, y_prob)

    results = {
        'Model':          model_name,
        'Precision':      round(precision, 4),
        'Recall':         round(recall, 4),
        'F1-Score':       round(f1, 4),
        'AUC-ROC':        round(auc_roc, 4),
        'AUC-PR':         round(auc_pr, 4),
        'FPR':            round(fpr, 6),
        'Inference(ms)':  round(inference_time_ms, 3),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }
    return results


def print_result(r):
    print(f"\n  {'─'*50}")
    print(f"  Model:         {r['Model']}")
    print(f"  {'─'*50}")
    print(f"  Precision:     {r['Precision']:.4f}")
    print(f"  Recall:        {r['Recall']:.4f}")
    print(f"  F1-Score:      {r['F1-Score']:.4f}")
    print(f"  AUC-ROC:       {r['AUC-ROC']:.4f}")
    print(f"  AUC-PR:        {r['AUC-PR']:.4f}")
    print(f"  FPR:           {r['FPR']:.6f}")
    print(f"  Inference(ms): {r['Inference(ms)']:.3f}")
    print(f"  Confusion: TP={r['TP']} | FP={r['FP']} | TN={r['TN']} | FN={r['FN']}")


# ===========================================================================
# BLOCK 5 — MODEL 1: RANDOM FOREST (OPTIMISED)
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 5: RANDOM FOREST")
print("=" * 65)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    bootstrap=True,
    n_jobs=-1,
    random_state=SEED
)

print("\nTraining Random Forest (500 trees)...")
t0 = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training time: {train_time:.2f}s")

# Inference time
t0 = time.time()
rf_prob = rf.predict_proba(X_test)[:, 1]
inf_time = (time.time() - t0) / len(X_test) * 1000
rf_pred = (rf_prob >= 0.5).astype(int)

rf_results = evaluate_model(y_test, rf_pred, rf_prob, 'Random Forest', inf_time)
print_result(rf_results)

# ===========================================================================
# BLOCK 6 — MODEL 2: XGBOOST (OPTIMISED)
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 6: XGBOOST")
print("=" * 65)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    gamma=0.1,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=-1,
    tree_method='hist'
)

print("\nTraining XGBoost (500 estimators)...")
t0 = time.time()
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
train_time = time.time() - t0
print(f"Training time: {train_time:.2f}s")

t0 = time.time()
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
inf_time = (time.time() - t0) / len(X_test) * 1000
xgb_pred = (xgb_prob >= 0.5).astype(int)

xgb_results = evaluate_model(y_test, xgb_pred, xgb_prob, 'XGBoost', inf_time)
print_result(xgb_results)

# ===========================================================================
# BLOCK 7 — MODEL 3: SVM (OPTIMISED)
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 7: SUPPORT VECTOR MACHINE")
print("=" * 65)

# Stratified subsample for SVM (computational constraint)
svm_sample_size = 60000
idx = []
# Include all fraud records from training
fraud_idx   = X_train_scaled[y_train == 1].index.tolist()
legit_idx   = X_train_scaled[y_train == 0].index.tolist()
np.random.shuffle(legit_idx)
legit_sample = legit_idx[:svm_sample_size - len(fraud_idx)]
svm_idx      = fraud_idx + legit_sample

X_svm = X_train_scaled.loc[svm_idx]
y_svm = y_train.loc[svm_idx]

print(f"\nSVM subsample: {len(X_svm):,} records | Fraud: {y_svm.sum():,}")

svm = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=SEED,
    cache_size=2000
)

print("Training SVM (this may take 3-5 minutes)...")
t0 = time.time()
svm.fit(X_svm, y_svm)
train_time = time.time() - t0
print(f"Training time: {train_time:.2f}s")

t0 = time.time()
svm_prob = svm.predict_proba(X_test_scaled)[:, 1]
inf_time = (time.time() - t0) / len(X_test) * 1000
svm_pred = (svm_prob >= 0.5).astype(int)

svm_results = evaluate_model(y_test, svm_pred, svm_prob, 'SVM', inf_time)
print_result(svm_results)

# ===========================================================================
# BLOCK 8 — MODEL 4: ISOLATION FOREST (OPTIMISED)
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 8: ISOLATION FOREST")
print("=" * 65)

# Contamination based on actual fraud rate
contamination_rate = y_train.mean()
print(f"\nContamination rate set to: {contamination_rate:.6f}")

iso = IsolationForest(
    n_estimators=300,
    contamination=contamination_rate,
    max_samples='auto',
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=SEED
)

print("Training Isolation Forest (300 trees, unsupervised)...")
t0 = time.time()
iso.fit(X_train)
train_time = time.time() - t0
print(f"Training time: {train_time:.2f}s")

t0 = time.time()
iso_scores = iso.decision_function(X_test)
inf_time   = (time.time() - t0) / len(X_test) * 1000

# Convert scores: lower = more anomalous → invert for probability
iso_prob = -iso_scores
iso_prob = (iso_prob - iso_prob.min()) / (iso_prob.max() - iso_prob.min())
iso_pred = iso.predict(X_test)
iso_pred = np.where(iso_pred == -1, 1, 0)

iso_results = evaluate_model(y_test, iso_pred, iso_prob, 'Isolation Forest', inf_time)
print_result(iso_results)

# ===========================================================================
# BLOCK 9 — MODEL 5: AUTOENCODER (OPTIMISED)
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 9: AUTOENCODER")
print("=" * 65)

# Train ONLY on legitimate transactions
X_train_legit = X_train_scaled[y_train == 0].values
X_test_arr    = X_test_scaled.values
n_features    = X_train_legit.shape[1]

print(f"\nAutoencoder training on {len(X_train_legit):,} legitimate transactions only")
print(f"Input dimensions: {n_features}")

# Build deep autoencoder architecture
def build_autoencoder(n_features):
    inp = keras.Input(shape=(n_features,))
    # Encoder
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    bottleneck = layers.Dense(8, activation='relu', name='bottleneck')(x)
    # Decoder
    x = layers.Dense(16, activation='relu')(bottleneck)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_features, activation='sigmoid')(x)
    model = keras.Model(inp, out)
    return model

ae = build_autoencoder(n_features)
ae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error'
)

print(f"\nAutoencoder architecture:")
ae.summary()

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=0
)

print("\nTraining Autoencoder...")
t0 = time.time()
history = ae.fit(
    X_train_legit, X_train_legit,
    epochs=150,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    shuffle=True,
    verbose=1
)
train_time = time.time() - t0
print(f"\nTraining time: {train_time:.2f}s")
print(f"Stopped at epoch: {len(history.history['loss'])}")

# Compute reconstruction error on test set
t0 = time.time()
X_test_reconstructed = ae.predict(X_test_arr, verbose=0)
inf_time = (time.time() - t0) / len(X_test) * 1000

reconstruction_errors = np.mean(np.square(X_test_arr - X_test_reconstructed), axis=1)

# Threshold: 99th percentile of legitimate training reconstruction errors
X_train_legit_reconstructed = ae.predict(X_train_legit, verbose=0)
train_errors = np.mean(np.square(X_train_legit - X_train_legit_reconstructed), axis=1)
threshold_99 = np.percentile(train_errors, 99)
print(f"\nReconstruction error threshold (99th percentile): {threshold_99:.6f}")

ae_pred = (reconstruction_errors > threshold_99).astype(int)

# Normalise errors to [0,1] as probability scores
ae_prob = (reconstruction_errors - reconstruction_errors.min()) / \
          (reconstruction_errors.max() - reconstruction_errors.min())

ae_results = evaluate_model(y_test, ae_pred, ae_prob, 'Autoencoder', inf_time)
print_result(ae_results)

# ===========================================================================
# BLOCK 10 — CONSOLIDATED RESULTS TABLE
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 10: CONSOLIDATED RESULTS TABLE")
print("=" * 65)

all_results = [xgb_results, rf_results, svm_results, iso_results, ae_results]

results_df = pd.DataFrame(all_results)[[
    'Model', 'Precision', 'Recall', 'F1-Score',
    'AUC-ROC', 'AUC-PR', 'FPR', 'Inference(ms)'
]]

print("\n")
print(results_df.to_string(index=False))
print("\n")

# Save to CSV
results_df.to_csv('results_primary_dataset.csv', index=False)
print("Results saved to: results_primary_dataset.csv")

# ===========================================================================
# BLOCK 11 — STRATIFIED 5-FOLD CROSS-VALIDATION
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 11: STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 65)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_models = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        gamma=0.1, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric='logloss',
        random_state=SEED, n_jobs=-1, tree_method='hist'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=None,
        class_weight='balanced', n_jobs=-1, random_state=SEED
    ),
    'SVM': None,  # SVM CV is too slow — reported separately
    'Isolation Forest': IsolationForest(
        n_estimators=300, contamination=contamination_rate,
        n_jobs=-1, random_state=SEED
    )
}

cv_f1_scores = {}
cv_auc_scores = {}

X_np = X.values
y_np = y.values

for model_name, model in cv_models.items():
    if model is None:
        print(f"\n  {model_name}: CV skipped (computational constraint — test set result used)")
        continue

    fold_f1  = []
    fold_auc = []
    print(f"\n  Running 5-fold CV for {model_name}...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        X_fold_train, X_fold_val = X_np[train_idx], X_np[val_idx]
        y_fold_train, y_fold_val = y_np[train_idx], y_np[val_idx]

        if model_name == 'Isolation Forest':
            model.fit(X_fold_train)
            scores  = -model.decision_function(X_fold_val)
            scores  = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            preds   = model.predict(X_fold_val)
            preds   = np.where(preds == -1, 1, 0)
        else:
            model.fit(X_fold_train, y_fold_train)
            scores = model.predict_proba(X_fold_val)[:, 1]
            preds  = (scores >= 0.5).astype(int)

        f1  = f1_score(y_fold_val, preds, zero_division=0)
        auc = roc_auc_score(y_fold_val, scores)
        fold_f1.append(f1)
        fold_auc.append(auc)
        print(f"    Fold {fold+1}: F1={f1:.4f}  AUC-ROC={auc:.4f}")

    cv_f1_scores[model_name]  = fold_f1
    cv_auc_scores[model_name] = fold_auc
    print(f"  {model_name} — Mean F1: {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}  |  "
          f"Mean AUC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")

# Cross-validation summary table
print("\n\n  CROSS-VALIDATION SUMMARY TABLE")
print("  " + "-" * 60)
print(f"  {'Model':<20} {'Mean F1':>10} {'Std F1':>10} {'Mean AUC':>10} {'Std AUC':>10}")
print("  " + "-" * 60)
for mn in cv_f1_scores:
    mf = np.mean(cv_f1_scores[mn])
    sf = np.std(cv_f1_scores[mn])
    ma = np.mean(cv_auc_scores[mn])
    sa = np.std(cv_auc_scores[mn])
    print(f"  {mn:<20} {mf:>10.4f} {sf:>10.4f} {ma:>10.4f} {sa:>10.4f}")
print("  " + "-" * 60)

# Friedman test (XGBoost vs RF vs IF)
if len(cv_f1_scores) >= 3:
    models_for_friedman = list(cv_f1_scores.keys())
    f1_arrays = [cv_f1_scores[m] for m in models_for_friedman]
    stat, p_val = friedmanchisquare(*f1_arrays)
    print(f"\n  Friedman Test (F1-Score across all CV models):")
    print(f"  Chi-square statistic = {stat:.4f}  |  p-value = {p_val:.6f}")
    if p_val < 0.05:
        print(f"  Result: STATISTICALLY SIGNIFICANT (p < 0.05) — performance differences are real")
    else:
        print(f"  Result: Not significant at p=0.05 threshold")

# Wilcoxon test: XGBoost vs Random Forest
if 'XGBoost' in cv_f1_scores and 'Random Forest' in cv_f1_scores:
    w_stat, w_p = wilcoxon(cv_f1_scores['XGBoost'], cv_f1_scores['Random Forest'])
    print(f"\n  Wilcoxon Signed-Rank Test (XGBoost vs Random Forest):")
    print(f"  Statistic = {w_stat:.4f}  |  p-value = {w_p:.6f}")

# Save CV results
cv_summary = []
for mn in cv_f1_scores:
    cv_summary.append({
        'Model': mn,
        'Mean_F1': round(np.mean(cv_f1_scores[mn]), 4),
        'Std_F1':  round(np.std(cv_f1_scores[mn]), 4),
        'Mean_AUC': round(np.mean(cv_auc_scores[mn]), 4),
        'Std_AUC': round(np.std(cv_auc_scores[mn]), 4),
        'Fold_F1_Scores': cv_f1_scores[mn]
    })
cv_df = pd.DataFrame(cv_summary)
cv_df.to_csv('cv_results_primary.csv', index=False)
print("\n  CV results saved to: cv_results_primary.csv")

# ===========================================================================
# BLOCK 12 — SHAP FEATURE IMPORTANCE ANALYSIS
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 12: SHAP FEATURE IMPORTANCE ANALYSIS")
print("=" * 65)

feature_names = list(X.columns)

print("\nComputing SHAP values for XGBoost...")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_vals = xgb_explainer.shap_values(X_test)

print("Computing SHAP values for Random Forest...")
rf_explainer = shap.TreeExplainer(rf)
rf_shap_vals = rf_explainer.shap_values(X_test)
if isinstance(rf_shap_vals, list):
    rf_shap_vals = rf_shap_vals[1]

# Global feature importance (mean |SHAP|)
xgb_importance = pd.DataFrame({
    'Feature': feature_names,
    'XGBoost_SHAP': np.abs(xgb_shap_vals).mean(axis=0)
}).sort_values('XGBoost_SHAP', ascending=False)

rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'RF_SHAP': np.abs(rf_shap_vals).mean(axis=0)
}).sort_values('RF_SHAP', ascending=False)

shap_combined = xgb_importance.merge(rf_importance, on='Feature')
shap_combined = shap_combined.sort_values('XGBoost_SHAP', ascending=False)

print("\n  SHAP Feature Importance — Top 15 Features")
print("  " + "-" * 55)
print(f"  {'Feature':<12} {'XGBoost SHAP':>15} {'Random Forest SHAP':>20}")
print("  " + "-" * 55)
for _, row in shap_combined.head(15).iterrows():
    print(f"  {row['Feature']:<12} {row['XGBoost_SHAP']:>15.4f} {row['RF_SHAP']:>20.4f}")
print("  " + "-" * 55)

shap_combined.to_csv('shap_importance_primary.csv', index=False)
print("\n  SHAP results saved to: shap_importance_primary.csv")

# ===========================================================================
# BLOCK 13 — GENERATE ALL PLOTS
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 13: GENERATING PLOTS")
print("=" * 65)

COLORS = {
    'XGBoost':         '#185FA5',
    'Random Forest':   '#1D9E75',
    'SVM':             '#534AB7',
    'Isolation Forest':'#BA7517',
    'Autoencoder':     '#C0392B'
}
MODEL_ORDER = ['XGBoost', 'Random Forest', 'SVM', 'Isolation Forest', 'Autoencoder']

# ── Plot 1: Grouped Bar Chart — All Metrics ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(MODEL_ORDER))
width = 0.18

for i, metric in enumerate(metrics):
    vals = [next(r[metric] for r in all_results if r['Model'] == m) for m in MODEL_ORDER]
    offset = (i - len(metrics)/2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=metric,
                  color=['#185FA5','#1D9E75','#534AB7','#BA7517'][i],
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom',
                fontsize=7, fontfamily='serif')

ax.set_xlabel('Model', fontfamily='serif', fontsize=12)
ax.set_ylabel('Score (0–1)', fontfamily='serif', fontsize=12)
ax.set_title('Figure 4.1 — Comparative Performance of Five AI Models\n'
             'European Credit Card Fraud Dataset',
             fontfamily='serif', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER, fontfamily='serif', fontsize=10)
ax.set_ylim(0, 1.10)
ax.legend(fontsize=10, prop={'family': 'serif'})
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure_4_1_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure_4_1_model_comparison.png")

# ── Plot 2: ROC Curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
probs_dict = {
    'XGBoost':          xgb_prob,
    'Random Forest':    rf_prob,
    'SVM':              svm_prob,
    'Isolation Forest': iso_prob,
    'Autoencoder':      ae_prob
}
for name in MODEL_ORDER:
    fpr_vals, tpr_vals, _ = roc_curve(y_test, probs_dict[name])
    auc_val = roc_auc_score(y_test, probs_dict[name])
    ax.plot(fpr_vals, tpr_vals, color=COLORS[name], lw=2,
            label=f'{name} (AUC = {auc_val:.4f})')
ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontfamily='serif', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontfamily='serif', fontsize=12)
ax.set_title('Figure 4.2 — ROC Curves for All Five Models\n'
             'European Credit Card Fraud Dataset',
             fontfamily='serif', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, prop={'family': 'serif'}, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure_4_2_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure_4_2_roc_curves.png")

# ── Plot 3: Cross-Validation F1-Score with Error Bars ────────────────────
if cv_f1_scores:
    fig, ax = plt.subplots(figsize=(10, 6))
    cv_model_names = list(cv_f1_scores.keys())
    means = [np.mean(cv_f1_scores[m]) for m in cv_model_names]
    stds  = [np.std(cv_f1_scores[m])  for m in cv_model_names]
    colors_cv = [COLORS.get(m, '#888888') for m in cv_model_names]
    bars = ax.bar(cv_model_names, means, color=colors_cv,
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.errorbar(cv_model_names, means, yerr=stds, fmt='none',
                color='black', capsize=6, capthick=1.5, linewidth=1.5)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.005,
                f'{m:.4f}±{s:.4f}', ha='center', va='bottom',
                fontsize=9, fontfamily='serif')
    ax.set_xlabel('Model', fontfamily='serif', fontsize=12)
    ax.set_ylabel('Mean F1-Score (5-Fold CV)', fontfamily='serif', fontsize=12)
    ax.set_title('Figure 4.3 — Stratified 5-Fold Cross-Validation Results\n'
                 'Mean F1-Score with Standard Deviation',
                 fontfamily='serif', fontsize=13, fontweight='bold')
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis='y', alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_fontfamily('serif')
        tick.set_fontsize(10)
    plt.tight_layout()
    plt.savefig('figure_4_3_cross_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_4_3_cross_validation.png")

# ── Plot 4: SHAP Feature Importance (Horizontal Bar) ─────────────────────
top_n = 15
top_features = shap_combined.head(top_n)
fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(top_n)
width = 0.35
ax.barh(y_pos + width/2, top_features['XGBoost_SHAP'].values,
        width, color='#185FA5', alpha=0.85, label='XGBoost', edgecolor='white')
ax.barh(y_pos - width/2, top_features['RF_SHAP'].values,
        width, color='#1D9E75', alpha=0.85, label='Random Forest', edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features['Feature'].values,
                   fontfamily='serif', fontsize=11)
ax.set_xlabel('Mean |SHAP| Value', fontfamily='serif', fontsize=12)
ax.set_title('Figure 4.4 — SHAP Feature Importance: XGBoost vs Random Forest\n'
             'Top 15 Features by Mean Absolute SHAP Value',
             fontfamily='serif', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, prop={'family': 'serif'})
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figure_4_4_shap_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure_4_4_shap_importance.png")

# ── Plot 5: Inference Time Comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
times = [next(r['Inference(ms)'] for r in all_results if r['Model'] == m)
         for m in MODEL_ORDER]
colors_bar = [COLORS[m] for m in MODEL_ORDER]
bars = ax.bar(MODEL_ORDER, times, color=colors_bar,
              alpha=0.85, edgecolor='white', linewidth=0.5)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, t + 0.02,
            f'{t:.3f} ms', ha='center', va='bottom',
            fontsize=10, fontfamily='serif')
ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5,
           label='Real-time threshold (50 ms)')
ax.set_xlabel('Model', fontfamily='serif', fontsize=12)
ax.set_ylabel('Milliseconds per Transaction', fontfamily='serif', fontsize=12)
ax.set_title('Figure 4.5 — Inference Time per Transaction (ms)\n'
             'All Models Well Below Real-Time Payment Screening Threshold',
             fontfamily='serif', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, prop={'family': 'serif'})
ax.set_ylim(0, 60)
ax.grid(axis='y', alpha=0.3)
for tick in ax.get_xticklabels():
    tick.set_fontfamily('serif')
    tick.set_fontsize(10)
plt.tight_layout()
plt.savefig('figure_4_5_inference_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure_4_5_inference_time.png")

# ── Plot 6: Confusion Matrices (2x3 grid) ────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
preds_dict = {
    'XGBoost':          xgb_pred,
    'Random Forest':    rf_pred,
    'SVM':              svm_pred,
    'Isolation Forest': iso_pred,
    'Autoencoder':      ae_pred
}
for idx, name in enumerate(MODEL_ORDER):
    cm = confusion_matrix(y_test, preds_dict[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Legitimate','Fraud'],
                yticklabels=['Legitimate','Fraud'],
                annot_kws={'size': 13, 'family': 'serif'})
    axes[idx].set_title(name, fontfamily='serif', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontfamily='serif', fontsize=10)
    axes[idx].set_ylabel('Actual', fontfamily='serif', fontsize=10)
axes[5].axis('off')
fig.suptitle('Figure 4.6 — Confusion Matrices for All Five Models\n'
             'European Credit Card Fraud Dataset',
             fontfamily='serif', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figure_4_6_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure_4_6_confusion_matrices.png")

# ── Plot 7: Precision-Recall Curves ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for name in MODEL_ORDER:
    prec, rec, _ = precision_recall_curve(y_test, probs_dict[name])
    ap = average_precision_score(y_test, probs_dict[name])
    ax.plot(rec, prec, color=COLORS[name], lw=2,
            label=f'{name} (AP = {ap:.4f})')
ax.axhline(y=y_test.mean(), color='k', linestyle='--', lw=1,
           alpha=0.5, label=f'No skill (fraud rate={y_test.mean():.4f})')
ax.set_xlabel('Recall', fontfamily='serif', fontsize=12)
ax.set_ylabel('Precision', fontfamily='serif', fontsize=12)
ax.set_title('Figure 4.7 — Precision-Recall Curves for All Five Models\n'
             'European Credit Card Fraud Dataset',
             fontfamily='serif', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, prop={'family': 'serif'})
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure_4_7_precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure_4_7_precision_recall_curves.png")

# ── Plot 8: SHAP Summary Plot for XGBoost ────────────────────────────────
print("\n  Generating SHAP summary plots...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    xgb_shap_vals,
    X_test,
    feature_names=feature_names,
    show=False,
    max_display=20
)
plt.title('Figure 4.8 — SHAP Summary Plot: XGBoost\n'
          'Feature Impact on Fraud Probability',
          fontfamily='serif', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_4_8_shap_summary_xgboost.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure_4_8_shap_summary_xgboost.png")

# ── Plot 9: Class Distribution ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ['Legitimate', 'Fraud']
counts = [(y == 0).sum(), (y == 1).sum()]
pcts   = [c/len(y)*100 for c in counts]
axes[0].bar(labels, counts, color=['#1D9E75','#C0392B'], alpha=0.85,
            edgecolor='white', linewidth=0.5)
for i, (c, p) in enumerate(zip(counts, pcts)):
    axes[0].text(i, c + 1000, f'{c:,}\n({p:.3f}%)',
                 ha='center', fontfamily='serif', fontsize=11)
axes[0].set_title('Class Distribution\n(Full Dataset)',
                  fontfamily='serif', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Transactions', fontfamily='serif', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

axes[1].pie(counts, labels=labels, autopct='%1.3f%%',
            colors=['#1D9E75','#C0392B'], startangle=90,
            textprops={'fontfamily': 'serif', 'fontsize': 11})
axes[1].set_title('Class Distribution\n(Proportion)',
                  fontfamily='serif', fontsize=12, fontweight='bold')
fig.suptitle('Figure 2.3 — Dataset Class Distribution\n'
             'European Credit Card Fraud Dataset',
             fontfamily='serif', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figure_dataset_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure_dataset_distribution.png")

# ===========================================================================
# BLOCK 14 — FINAL SUMMARY REPORT
# ===========================================================================

print("\n" + "=" * 65)
print("  BLOCK 14: FINAL EXPERIMENT SUMMARY")
print("=" * 65)

print("\n  ┌─────────────────────────────────────────────────────────┐")
print("  │         COMPLETE RESULTS — PRIMARY DATASET              │")
print("  │       European Credit Card Fraud Detection              │")
print("  ├──────────────┬──────────┬──────────┬──────────┬─────────┤")
print("  │ Model        │Precision │  Recall  │ F1-Score │AUC-ROC  │")
print("  ├──────────────┼──────────┼──────────┼──────────┼─────────┤")
for r in sorted(all_results, key=lambda x: x['F1-Score'], reverse=True):
    print(f"  │ {r['Model']:<12} │  {r['Precision']:.4f}  │  {r['Recall']:.4f}  │  {r['F1-Score']:.4f}  │  {r['AUC-ROC']:.4f} │")
print("  └──────────────┴──────────┴──────────┴──────────┴─────────┘")

print("\n  ┌─────────────────────────────────────────────────────────┐")
print("  │              FALSE POSITIVE RATE & SPEED                │")
print("  ├──────────────┬──────────────────────┬───────────────────┤")
print("  │ Model        │  False Positive Rate │  Inference (ms)   │")
print("  ├──────────────┼──────────────────────┼───────────────────┤")
for r in sorted(all_results, key=lambda x: x['FPR']):
    print(f"  │ {r['Model']:<12} │       {r['FPR']:.6f}       │      {r['Inference(ms)']:.3f} ms     │")
print("  └──────────────┴──────────────────────┴───────────────────┘")

# Best model identification
best_f1_model = max(all_results, key=lambda x: x['F1-Score'])
best_fpr_model = min(all_results, key=lambda x: x['FPR'])
best_auc_model = max(all_results, key=lambda x: x['AUC-ROC'])

print(f"\n  🏆 Best F1-Score:           {best_f1_model['Model']} ({best_f1_model['F1-Score']:.4f})")
print(f"  🏆 Best AUC-ROC:            {best_auc_model['Model']} ({best_auc_model['AUC-ROC']:.4f})")
print(f"  🏆 Lowest False Positive:   {best_fpr_model['Model']} ({best_fpr_model['FPR']:.6f})")

print("\n  FILES GENERATED:")
files = [
    'results_primary_dataset.csv',
    'cv_results_primary.csv',
    'shap_importance_primary.csv',
    'figure_4_1_model_comparison.png',
    'figure_4_2_roc_curves.png',
    'figure_4_3_cross_validation.png',
    'figure_4_4_shap_importance.png',
    'figure_4_5_inference_time.png',
    'figure_4_6_confusion_matrices.png',
    'figure_4_7_precision_recall_curves.png',
    'figure_4_8_shap_summary_xgboost.png',
    'figure_dataset_distribution.png'
]
for f in files:
    print(f"    ✓ {f}")

print("\n" + "=" * 65)
print("  EXPERIMENT COMPLETE — ALL RESULTS SAVED")
print("=" * 65)
