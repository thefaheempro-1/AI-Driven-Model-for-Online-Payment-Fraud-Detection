"""
Fraud detection benchmark v2  (European Credit Card  +  IEEE-CIS)
=================================================================
One script for both datasets.  Differences from run_*_fast.py:

  1. Three-way split (train / validation / test).  Thresholds, early stopping,
     scaling statistics and the stacking meta-learner use TRAIN/VALIDATION only.
     The test set is used once, for the reported numbers.
  2. Scaler, imputer and variance filter are fitted on the training part only.
  3. SMOTE is applied INSIDE every training fold (imblearn Pipeline), both in
     the hyper-parameter search and in the cross-validation used for Friedman.
  4. All three supervised models get the same search protocol (average precision).
  5. LightGBM early stopping uses the validation set (old script used the TEST set);
     subsample_freq=1 so `subsample` actually takes effect.
  6. Isolation Forest uses contamination='auto' (contamination does not change
     score_samples, so the label-based setting is unnecessary).
  7. Unsupervised scores are standardised with validation statistics (not with
     test min/max).
  8. Stacking = hold-out stacking: base models are fitted ONCE on the training set
     (exactly the models evaluated in the tables); the logistic-regression
     meta-learner is fitted on their VALIDATION scores (logit-transformed for the
     supervised models).  The threshold comes from out-of-fold meta predictions
     on the validation set.  A trees-only stack is reported as well.
  9. Reported: AUC-ROC, PR-AUC, precision/recall/F1/FPR at the validation
     threshold, stratified-bootstrap 95% CIs (shared resamples) and paired
     bootstrap differences against a reference model.
 10. Latency: batch time per transaction AND single-transaction latency
     (mean, p50, p95, p99) for the full pipeline, n_jobs=1.
 11. CV for the Friedman test uses the tuned configurations, un-rounded AUC and
     PR-AUC (the old script rounded to 4 decimals, which creates artificial ties).
 12. SHAP sample contains test-set fraud cases (old script used the first 500 rows).
 13. Seeds for numpy / python / TensorFlow; library versions saved to run_info.json.

Default data locations (built in, so --data is optional)
--------------------------------------------------------
  euro : D:\\Educational\\Research\\Tests\\European Credit Card Fraud Dataset 284,807 transaction records\\creditcard.csv
  ieee : D:\\Educational\\Research\\Tests\\IEEE-CIS Fraud Detection Dataset 590,000 records\\fraud_train_preprocessed.csv
         (fraud_test_preprocessed.csv is also read if it contains a labelled target column;
          the original Kaggle IEEE-CIS test file has no isFraud labels, in which case it is skipped)

Usage
-----
  python fraud_benchmark_v2.py --dataset euro --split random
  python fraud_benchmark_v2.py --dataset euro --split temporal
  python fraud_benchmark_v2.py --dataset ieee --split random
  python fraud_benchmark_v2.py --dataset ieee --split temporal --time-col TransactionDT

You can still override the paths with --data <file> [<file> ...].

IEEE-CIS: for a temporal split the file must contain a time column
(TransactionDT).  If the pre-processed files do not have it, build the table
from train_transaction.csv (+ train_identity.csv) so that TransactionDT is kept.
"""
import argparse, json, os, platform, random, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import lightgbm as lgb

SEED = 42

# ----------------------------------------------------------------------------
# default locations
# ----------------------------------------------------------------------------
BASE_DIR = r"D:\Educational\Research\Tests"
EURO_DIR = os.path.join(BASE_DIR, "European Credit Card Fraud Dataset 284,807 transaction records")
IEEE_DIR = os.path.join(BASE_DIR, "IEEE-CIS Fraud Detection Dataset 590,000 records")
DEFAULT_DATA = {
    "euro": [os.path.join(EURO_DIR, "creditcard.csv")],
    "ieee": [os.path.join(IEEE_DIR, "fraud_train_preprocessed.csv"),
             os.path.join(IEEE_DIR, "fraud_test_preprocessed.csv")],
}
DEFAULT_OUT = os.path.join(BASE_DIR, "results_v2")


# ----------------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------------
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def best_threshold(y, s):
    """Threshold maximising F1 on (y, s). Called ONLY with validation data."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i])


def point_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(Precision=prec, Recall=rec, F1=f1, FPR=fpr, TP=tp, FP=fp, FN=fn, TN=tn)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def _read_labelled_ieee(paths):
    """Read the IEEE files; keep only those that contain a usable label column."""
    frames = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[data] file not found, skipping: {p}")
            continue
        d = pd.read_csv(p)
        for cand in ("isFraud", "target"):
            if cand in d.columns:
                d = d.rename(columns={cand: "Class"})
        if "Class" not in d.columns or d["Class"].notna().sum() == 0:
            print(f"[data] no labels in {os.path.basename(p)} -> skipped (cannot be scored)")
            continue
        print(f"[data] loaded {os.path.basename(p)}: {len(d):,} rows")
        frames.append(d)
    if not frames:
        sys.exit("No labelled IEEE-CIS file could be read. Check the --data paths.")
    return pd.concat(frames, ignore_index=True)


def load_data(args):
    if args.dataset == "euro":
        df = pd.read_csv(args.data[0])
        target = "Class"
        time_col = args.time_col or "Time"
        scale_cols = ["Time", "Amount"]
        variance_filter = False
    else:
        df = _read_labelled_ieee(args.data)
        target = "Class"
        time_col = args.time_col
        scale_cols = None            # all numeric columns
        variance_filter = True
        obj = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if obj:
            print(f"[data] dropping non-numeric columns: {obj}")
            df = df.drop(columns=obj)
    df = df.dropna(subset=[target])
    y = df[target].astype(int).values
    t = None
    if args.split == "temporal":
        if time_col is None or time_col not in df.columns:
            sys.exit("Temporal split needs a time column present in the data "
                     f"(--time-col). Columns available: {list(df.columns)[:15]} ...")
        t = df[time_col].values
    X = df.drop(columns=[target])
    if args.split == "temporal" and time_col in X.columns:
        X = X.drop(columns=[time_col])          # do not let the model extrapolate on time
        if scale_cols:
            scale_cols = [c for c in scale_cols if c != time_col]
    print(f"[data] {len(df):,} rows, {X.shape[1]} features, "
          f"fraud={y.sum():,} ({y.mean()*100:.3f}%)")
    return X, y, t, scale_cols, variance_filter


def make_split(y, t, mode, seed=SEED):
    idx = np.arange(len(y))
    if mode == "random":
        trva, te = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
        tr, va = train_test_split(trva, test_size=0.25, stratify=y[trva], random_state=seed)
    else:
        order = np.argsort(t, kind="stable")
        n = len(y)
        tr, va, te = order[: int(.6 * n)], order[int(.6 * n): int(.8 * n)], order[int(.8 * n):]
    for nm, ix in (("train", tr), ("val", va), ("test", te)):
        print(f"[split:{mode}] {nm:5s} n={len(ix):>8,}  fraud={y[ix].sum():>6,} ({y[ix].mean()*100:.3f}%)")
        if y[ix].sum() < 10:
            sys.exit(f"Too few fraud cases in {nm} split.")
    return tr, va, te


class Prep:
    """median imputer + RobustScaler (+ variance filter); fitted on TRAIN only."""
    def __init__(self, scale_cols, variance_filter):
        self.scale_cols, self.vf = scale_cols, variance_filter

    def fit(self, X):
        self.cols = list(X.columns)
        self.med = X.median(numeric_only=True)
        Xf = X.fillna(self.med)
        self.sc_idx = ([self.cols.index(c) for c in self.scale_cols if c in self.cols]
                       if self.scale_cols else list(range(len(self.cols))))
        self.scaler = RobustScaler().fit(Xf.values[:, self.sc_idx])
        A = self._scale(Xf.values)
        self.sel = VarianceThreshold(0.01).fit(A) if self.vf else None
        keep = self.sel.get_support() if self.sel is not None else np.ones(A.shape[1], bool)
        self.feature_names = [c for c, k in zip(self.cols, keep) if k]
        return self

    def _scale(self, A):
        A = A.astype(np.float64).copy()
        A[:, self.sc_idx] = self.scaler.transform(A[:, self.sc_idx])
        return A

    def transform(self, X):
        A = self._scale(X.fillna(self.med).values)
        return self.sel.transform(A) if self.sel is not None else A


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def tune(name, base, grid, n_iter, Xtr, ytr):
    pipe = ImbPipeline([("smote", SMOTE(k_neighbors=5, random_state=SEED)), ("clf", base)])
    dist = {f"clf__{k}": v for k, v in grid.items()}
    search = RandomizedSearchCV(
        pipe, dist, n_iter=n_iter, scoring="average_precision",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=1, refit=True)
    t0 = time.perf_counter()
    search.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0       # includes the search
    print(f"   {name}: best={ {k[5:]: v for k, v in search.best_params_.items()} }  "
          f"(cv AP={search.best_score_:.4f}, {train_time:.0f}s)")
    return search.best_estimator_, search.best_params_, train_time


def build_supervised(Xtr, ytr, n_iter_rf, n_iter_xgb, n_iter_lgb):
    out = {}
    out["Random Forest"] = tune(
        "Random Forest", RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 300], "max_depth": [None, 20],
         "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]},
        n_iter_rf, Xtr, ytr)
    out["XGBoost"] = tune(
        "XGBoost", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
        {"n_estimators": [300, 500], "max_depth": [4, 6], "learning_rate": [0.05, 0.1],
         "subsample": [0.8, 0.9], "colsample_bytree": [0.8, 0.9], "gamma": [0, 0.1],
         "min_child_weight": [1, 3]},
        n_iter_xgb, Xtr, ytr)
    out["LightGBM"] = tune(
        "LightGBM", lgb.LGBMClassifier(subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1),
        {"n_estimators": [300, 500], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1],
         "subsample": [0.8, 0.9], "colsample_bytree": [0.8, 0.9], "reg_lambda": [0, 1.0]},
        n_iter_lgb, Xtr, ytr)
    return out


class Unsup:
    """Unsupervised scorer; output standardised with VALIDATION statistics."""
    def __init__(self, kind, dataset):
        self.kind, self.dataset = kind, dataset

    def fit(self, Xtr, ytr, Xva):
        t0 = time.perf_counter()
        if self.kind == "iso":
            self.m = IsolationForest(n_estimators=300, contamination="auto",
                                     random_state=SEED, n_jobs=-1).fit(Xtr)
        else:
            self._fit_ae(Xtr[ytr == 0])
        self.train_time = time.perf_counter() - t0
        raw = self._raw(Xva)
        self.mu, self.sd = raw.mean(), raw.std() + 1e-12
        return self

    def _fit_ae(self, X):
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam
        d = X.shape[1]
        hid, ep, bs, dr = (([32, 16, 8], 100, 512, 0.1) if self.dataset == "euro"
                           else ([128, 64, 32, 16], 80, 1024, 0.2))
        inp = Input(shape=(d,))
        x = inp
        for i, h in enumerate(hid[:-1]):
            x = Dense(h, activation="relu")(x)
            if i == 0:
                x = BatchNormalization()(x); x = Dropout(dr)(x)
        x = Dense(hid[-1], activation="relu")(x)
        for h in hid[-2::-1]:
            x = Dense(h, activation="relu")(x)
        out = Dense(d, activation="linear")(x)
        self.ae = Model(inp, out)
        self.ae.compile(optimizer=Adam(1e-3), loss="mse")
        self.ae.fit(X, X, epochs=ep, batch_size=bs, validation_split=0.1, verbose=0,
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                               ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)])

    def _raw(self, X):
        if self.kind == "iso":
            return -self.m.score_samples(X)
        rec = self.ae.predict(X, batch_size=4096, verbose=0)
        return np.log(np.mean((X - rec) ** 2, axis=1) + 1e-12)   # log: heavy-tailed errors

    def score(self, X):
        return (self._raw(X) - self.mu) / self.sd

    def score_one(self, x):                 # single-row path for latency
        if self.kind == "iso":
            return self._raw(x)
        rec = self.ae(x.astype(np.float32), training=False).numpy()
        return np.log(np.mean((x - rec) ** 2, axis=1) + 1e-12)


# ----------------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------------
def bootstrap(y, scores, thr, ref, B, seed=SEED):
    """Stratified bootstrap with SHARED resamples; paired differences vs `ref`."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    names = list(scores)
    keys = ["AUC", "PR-AUC", "Precision", "Recall", "F1"]
    store = {n: {k: [] for k in keys} for n in names}
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yb = y[idx]
        for n in names:
            sb = scores[n][idx]
            pm = point_metrics(yb, sb, thr[n])
            store[n]["AUC"].append(roc_auc_score(yb, sb))
            store[n]["PR-AUC"].append(average_precision_score(yb, sb))
            for k in ("Precision", "Recall", "F1"):
                store[n][k].append(pm[k])
    ci = {n: {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in d.items()}
          for n, d in store.items()}
    diffs = []
    for n in names:
        if n == ref:
            continue
        row = {"Model": n, "vs": ref}
        for k in ("AUC", "PR-AUC", "F1"):
            d = np.array(store[n][k]) - np.array(store[ref][k])
            lo, hi = np.percentile(d, [2.5, 97.5])
            row[f"d{k}"] = d.mean(); row[f"d{k}_lo"] = lo; row[f"d{k}_hi"] = hi
        diffs.append(row)
    return ci, pd.DataFrame(diffs)


def latency_single(fn, X, n, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = X[rng.integers(0, len(X), n)]
    for r in rows[:10]:
        fn(r[None, :])
    ts = []
    for r in rows:
        t0 = time.perf_counter(); fn(r[None, :]); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return dict(mean=ts.mean(), p50=np.percentile(ts, 50), p95=np.percentile(ts, 95), p99=np.percentile(ts, 99))


def cv_scores(pipes, Xtr, ytr):
    """5-fold CV of the TUNED pipelines (SMOTE inside each training fold). Un-rounded."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    auc = {n: [] for n in pipes}; ap = {n: [] for n in pipes}
    for tr, va in skf.split(Xtr, ytr):
        for n, p in pipes.items():
            m = clone(p).fit(Xtr[tr], ytr[tr])
            s = m.predict_proba(Xtr[va])[:, 1]
            auc[n].append(roc_auc_score(ytr[va], s)); ap[n].append(average_precision_score(ytr[va], s))
    return auc, ap


def shap_top(model, Xte, yte, names, n_fraud=250, n_legit=250, top=15):
    import shap
    rng = np.random.default_rng(SEED)
    fi = np.where(yte == 1)[0]; li = np.where(yte == 0)[0]
    fi = rng.choice(fi, min(n_fraud, len(fi)), replace=False)
    li = rng.choice(li, n_legit, replace=False)
    idx = np.concatenate([fi, li]); is_f = np.r_[np.ones(len(fi), bool), np.zeros(len(li), bool)]
    sv = shap.TreeExplainer(model).shap_values(Xte[idx])
    if isinstance(sv, list):
        sv = sv[1]
    if getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    all_m, fr_m = np.abs(sv).mean(0), np.abs(sv[is_f]).mean(0)
    order = np.argsort(all_m)[::-1][:top]
    return pd.DataFrame({"feature": [names[i] for i in order], "mean|SHAP| (all sampled)": all_m[order],
                         "mean|SHAP| (fraud only)": fr_m[order]}), len(fi), len(li)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["euro", "ieee"], required=True)
    ap.add_argument("--data", nargs="+", default=None,
                    help="CSV path(s); defaults to the built-in D:\\ locations")
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-iter-rf", type=int, default=10)
    ap.add_argument("--n-iter-xgb", type=int, default=15)
    ap.add_argument("--n-iter-lgb", type=int, default=10)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--lat-n", type=int, default=300)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    args = ap.parse_args()
    if not args.data:
        args.data = DEFAULT_DATA[args.dataset]
    print(f"[data] source file(s): {args.data}")
    set_seeds()
    tag = f"{args.out}_{args.dataset}_{args.split}"
    os.makedirs(tag, exist_ok=True)

    X, y, t, scale_cols, vf = load_data(args)
    tr, va, te = make_split(y, t, args.split)
    prep = Prep(scale_cols, vf).fit(X.iloc[tr])           # train only
    Xtr, Xva, Xte = (prep.transform(X.iloc[i]) for i in (tr, va, te))
    ytr, yva, yte = y[tr], y[va], y[te]
    names = prep.feature_names
    print(f"[prep] {len(names)} features after train-only preprocessing")

    # ---- supervised
    print("[train] supervised models (SMOTE inside CV; AP-scored search)")
    sup = build_supervised(Xtr, ytr, args.n_iter_rf, args.n_iter_xgb, args.n_iter_lgb)
    pipes = {n: v[0] for n, v in sup.items()}
    train_time = {n: v[2] for n, v in sup.items()}
    score_fn = {n: (lambda X, p=p: p.predict_proba(X)[:, 1]) for n, p in pipes.items()}

    # ---- unsupervised
    print("[train] unsupervised models")
    iso = Unsup("iso", args.dataset).fit(Xtr, ytr, Xva)
    ae = Unsup("ae", args.dataset).fit(Xtr, ytr, Xva)
    score_fn["Isolation Forest"] = iso.score
    score_fn["Autoencoder"] = ae.score
    train_time["Isolation Forest"], train_time["Autoencoder"] = iso.train_time, ae.train_time

    base = ["Random Forest", "XGBoost", "LightGBM", "Isolation Forest", "Autoencoder"]
    sv_, st_, bt = {}, {}, {}
    for n in base:
        sv_[n] = score_fn[n](Xva)
        t0 = time.perf_counter(); st_[n] = score_fn[n](Xte)
        bt[n] = (time.perf_counter() - t0) / len(Xte) * 1000

    # ---- hold-out stacking (meta-learner on VALIDATION scores)
    def meta_in(d, cols):
        return np.column_stack([logit(d[c]) if c in pipes else d[c] for c in cols])

    stacks = {"Stacked Ensemble": base,
              "Stacked (trees only)": ["Random Forest", "XGBoost", "LightGBM"]}
    meta_models = {}
    for sn, cols in stacks.items():
        Mva, Mte = meta_in(sv_, cols), meta_in(st_, cols)
        meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        oof = cross_val_predict(meta, Mva, yva, method="predict_proba",
                                cv=StratifiedKFold(5, shuffle=True, random_state=SEED))[:, 1]
        meta.fit(Mva, yva)
        sv_[sn] = oof                                   # validation-side scores for the threshold
        t0 = time.perf_counter(); st_[sn] = meta.predict_proba(Mte)[:, 1]
        bt[sn] = (time.perf_counter() - t0) / len(Xte) * 1000 + sum(bt[c] for c in cols)
        meta_models[sn] = (meta, cols)
        print(f"[stack] {sn}: coefficients={dict(zip(cols, np.round(meta.coef_[0], 3)))}")

    models = base + list(stacks)
    thr = {n: best_threshold(yva, sv_[n]) for n in models}      # validation-derived thresholds

    # ---- metrics
    ref = "XGBoost"
    ci, diffs = bootstrap(yte, {n: st_[n] for n in models}, thr, ref, args.boot)
    rows = []
    for n in models:
        pm = point_metrics(yte, st_[n], thr[n])
        r = {"Model": n, **{k: pm[k] for k in ("Precision", "Recall", "F1", "FPR")},
             "AUC-ROC": roc_auc_score(yte, st_[n]), "PR-AUC": average_precision_score(yte, st_[n]),
             "Threshold(val)": thr[n], "TP": pm["TP"], "FP": pm["FP"], "FN": pm["FN"], "TN": pm["TN"],
             "BatchInf(ms/tx)": bt[n], "TrainTime(s)": train_time.get(n, np.nan)}
        for k in ("AUC", "PR-AUC", "Recall", "F1", "Precision"):
            lo, hi = ci[n][k]; r[f"{k} 95%CI"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f"{tag}/results.csv", index=False)
    diffs.to_csv(f"{tag}/paired_bootstrap_diffs_vs_{ref.replace(' ', '')}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n[RESULTS]\n", res[["Model", "Precision", "Recall", "F1", "AUC-ROC", "PR-AUC", "FPR",
                               "BatchInf(ms/tx)"]].round(4).to_string(index=False))
    print("\n[PAIRED BOOTSTRAP vs", ref, "]\n", diffs.round(4).to_string(index=False))

    # ---- single-transaction latency (full pipeline, n_jobs=1)
    for n, p in pipes.items():
        try:
            p.set_params(clf__n_jobs=1)
        except Exception:
            pass
    one = {n: (lambda x, p=p: p.predict_proba(x)[:, 1]) for n, p in pipes.items()}
    one["Isolation Forest"], one["Autoencoder"] = iso.score_one, ae.score_one

    def stack_one(sn):
        meta, cols = meta_models[sn]
        def f(x):
            v = [one[c](x) for c in cols]
            M = np.column_stack([logit(a) if c in pipes else a for a, c in zip(v, cols)])
            return meta.predict_proba(M)[:, 1]
        return f
    for sn in stacks:
        one[sn] = stack_one(sn)
    lat = pd.DataFrame({n: latency_single(one[n], Xte, args.lat_n) for n in models}).T
    lat.index.name = "Model"; lat.round(4).to_csv(f"{tag}/latency_single_ms.csv")
    print("\n[LATENCY single transaction, ms]\n", lat.round(4).to_string())

    # ---- CV + Friedman (tuned configs, SMOTE inside folds, un-rounded)
    if not args.skip_cv:
        print("\n[cv] 5-fold CV of tuned pipelines ...")
        auc, apr = cv_scores({n: pipes[n] for n in ["Random Forest", "XGBoost", "LightGBM"]}, Xtr, ytr)
        out = []
        for lab, d in (("AUC-ROC", auc), ("PR-AUC", apr)):
            stat, p = friedmanchisquare(*[d[n] for n in d])
            print(f"   Friedman on {lab}: chi2={stat:.4f}, p={p:.4f}")
            for n in d:
                out.append({"metric": lab, "model": n, **{f"fold{i+1}": v for i, v in enumerate(d[n])},
                            "mean": np.mean(d[n]), "sd(ddof=1)": np.std(d[n], ddof=1),
                            "friedman_chi2": stat, "friedman_p": p})
        pd.DataFrame(out).to_csv(f"{tag}/cv_friedman.csv", index=False)

    # ---- SHAP (sample contains fraud cases)
    if not args.skip_shap:
        for n in ("Random Forest", "XGBoost"):
            df_s, nf, nl = shap_top(pipes[n].named_steps["clf"], Xte, yte, names)
            df_s.to_csv(f"{tag}/shap_{n.replace(' ', '')}.csv", index=False)
            print(f"\n[SHAP {n}] sample: {nf} fraud + {nl} legitimate test rows\n", df_s.round(4).to_string(index=False))

    # ---- run info
    import sklearn, scipy
    info = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__, "xgboost": __import__("xgboost").__version__,
            "lightgbm": lgb.__version__, "imblearn": __import__("imblearn").__version__,
            "shap": getattr(__import__("shap"), "__version__", "n/a") if not args.skip_shap else "skipped",
            "seed": SEED, "args": vars(args),
            "best_params": {n: {k[5:]: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                for k, v in sup[n][1].items()} for n in sup}}
    try:
        import tensorflow as tf; info["tensorflow"] = tf.__version__
    except Exception:
        pass
    json.dump(info, open(f"{tag}/run_info.json", "w"), indent=2, default=str)
    print(f"\nDone. Outputs in {tag}")


if __name__ == "__main__":
    main()
