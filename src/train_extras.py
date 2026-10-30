"""
Train two additional models and save them:
  1. Revenue estimator  — XGBRegressor predicting log1p(ProductRelated_Duration)
     as a proxy for "session engagement value" (no revenue amount in UCI dataset,
     so we define value = ProductRelated_Duration * (1 + BounceRates_inverse) * (1 + ExitRates_inverse))
  2. Anomaly detector   — IsolationForest flagging bot / suspicious sessions

Run with:
  .venv\Scripts\python.exe src/train_extras.py
"""
from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / "data" / "raw"
DATA_PATH= ROOT / "data" / "processed" / "shoppers_clean.csv"
OUT_DIR  = ROOT / "models"
SEED     = 42

SKEWED   = ["Administrative","Administrative_Duration","Informational",
            "Informational_Duration","ProductRelated","ProductRelated_Duration",
            "BounceRates","ExitRates"]
PASSTHRU = ["SpecialDay","Weekend"]
CATS     = ["Month","VisitorType","OperatingSystems","Browser","Region","TrafficType"]

# ─── Load data ───────────────────────────────────────────────
def load():
    if DATA_PATH.is_file():
        df = pd.read_csv(DATA_PATH)
    else:
        csvs = list(RAW_DIR.rglob("online_shoppers_intention.csv"))
        if not csvs:
            zips = list(RAW_DIR.glob("*.zip"))
            if zips:
                with zipfile.ZipFile(zips[0]) as z:
                    z.extractall(RAW_DIR / "uci_extracted")
                csvs = list(RAW_DIR.rglob("online_shoppers_intention.csv"))
        df = pd.read_csv(csvs[0])
        df = df.drop_duplicates().reset_index(drop=True)
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    return df


def make_prep():
    return ColumnTransformer([
        ("sk", Pipeline([("log", FunctionTransformer(np.log1p)),
                         ("sc",  StandardScaler())]), SKEWED),
        ("pt", "passthrough", PASSTHRU),
        ("ca", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATS),
    ])


# ══════════════════════════════════════════════════════════════
#  1.  SESSION VALUE ESTIMATOR
#  No actual order amount in UCI, so we engineer a proxy:
#     session_value = log1p(ProductRelated_Duration) * Revenue (buying sessions only)
#  → Model predicts how "valuable" a session is BEFORE we know Revenue.
#  We train on ALL sessions and predict estimated engagement value.
# ══════════════════════════════════════════════════════════════
def train_value_estimator(df):
    print("Training session value estimator…")
    # Engineering proxy value:
    #   For buyers: value = ProductRelated_Duration * (0.5 + PageValues_proxy)
    #   Normalised to 0-100 scale so it reads as a "value score"
    val = (
        np.log1p(df["ProductRelated_Duration"]) * (1 - df["ExitRates"]) * (1 - df["BounceRates"])
    )
    # Scale to 0-100
    val_scaled = (val - val.min()) / (val.max() - val.min()) * 100
    y = val_scaled.values

    FEAT_COLS = SKEWED + PASSTHRU + CATS
    X = df[FEAT_COLS]
    prep = make_prep()
    reg  = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=4, eval_metric="rmse",
    )
    pipe = Pipeline([("prep", prep), ("reg", reg)])
    pipe.fit(X, y)

    # Evaluate on training data (no held-out split needed — this is a proxy target)
    preds = pipe.predict(X)
    rmse  = float(np.sqrt(np.mean((preds - y)**2)))
    r2    = float(1 - np.sum((preds - y)**2) / np.sum((y - y.mean())**2))

    # Compute band thresholds from distribution
    q33, q66 = float(np.percentile(preds, 33)), float(np.percentile(preds, 66))
    high_conv = float(df[preds >= q66]["Revenue"].mean())
    med_conv  = float(df[(preds >= q33) & (preds < q66)]["Revenue"].mean())
    low_conv  = float(df[preds < q33]["Revenue"].mean())

    joblib.dump(pipe, OUT_DIR / "value_estimator.joblib", compress=3)

    meta = {
        "model_file":      "value_estimator.joblib",
        "target":          "session_engagement_value_score (0-100 proxy)",
        "train_rmse":      round(rmse, 3),
        "train_r2":        round(r2, 3),
        "value_thresholds":{"high": round(q66, 2), "low": round(q33, 2)},
        "conversion_by_value_band":{
            "High Value (top 33%)":   round(high_conv, 4),
            "Medium Value (mid 33%)": round(med_conv,  4),
            "Low Value (bottom 33%)": round(low_conv,  4),
        },
        "feature_columns": FEAT_COLS,
        "interpretation":  (
            "Score 0-100. Higher = more engaged session. "
            "Based on product page dwell time adjusted for bounce/exit resistance. "
            "Correlates strongly with actual purchase (high-value sessions buy at "
            f"{high_conv:.1%} vs low at {low_conv:.1%})."
        ),
    }
    (OUT_DIR / "value_estimator_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  Value estimator saved | RMSE={rmse:.2f} | R²={r2:.3f}")
    print(f"  High-value conv: {high_conv:.1%} | Low-value conv: {low_conv:.1%}")
    return pipe, meta, preds


# ══════════════════════════════════════════════════════════════
#  3.  ANOMALY DETECTOR — Suspicious / Bot Sessions
#  IsolationForest on numeric engagement features.
#  Flags sessions that look like bots, scrapers, or impossible behaviour.
# ══════════════════════════════════════════════════════════════
def train_anomaly_detector(df):
    print("Training anomaly detector…")
    ANOM_FEATS = [
        "Administrative", "Administrative_Duration",
        "Informational",  "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates",    "ExitRates", "SpecialDay",
    ]
    scaler = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("sc",  StandardScaler()),
    ])
    X_scaled = scaler.fit_transform(df[ANOM_FEATS])

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # expect ~5% suspicious sessions
        max_samples="auto",
        random_state=SEED,
        n_jobs=4,
    )
    iso.fit(X_scaled)
    scores = iso.decision_function(X_scaled)   # more negative = more anomalous
    labels = iso.predict(X_scaled)             # -1 = anomaly, 1 = normal
    n_anom = int((labels == -1).sum())
    pct    = n_anom / len(df)

    # Anomaly score → 0-100 (100 = most suspicious)
    anom_score_0_100 = (
        (scores.max() - scores) / (scores.max() - scores.min()) * 100
    ).clip(0, 100)

    # What do anomalies look like?
    anom_mask = labels == -1
    anom_stats = {
        "median_product_pages":      round(float(df.loc[anom_mask, "ProductRelated"].median()), 1),
        "median_product_duration":   round(float(df.loc[anom_mask, "ProductRelated_Duration"].median()), 1),
        "median_bounce_rate":        round(float(df.loc[anom_mask, "BounceRates"].median()), 4),
        "median_exit_rate":          round(float(df.loc[anom_mask, "ExitRates"].median()), 4),
        "conversion_rate":           round(float(df.loc[anom_mask, "Revenue"].mean()), 4),
    }
    normal_stats = {
        "median_product_pages":      round(float(df.loc[~anom_mask, "ProductRelated"].median()), 1),
        "median_product_duration":   round(float(df.loc[~anom_mask, "ProductRelated_Duration"].median()), 1),
        "median_bounce_rate":        round(float(df.loc[~anom_mask, "BounceRates"].median()), 4),
        "median_exit_rate":          round(float(df.loc[~anom_mask, "ExitRates"].median()), 4),
        "conversion_rate":           round(float(df.loc[~anom_mask, "Revenue"].mean()), 4),
    }

    # Top anomaly reasons (which features deviate most?)
    z_scores = np.abs(X_scaled)
    top_feat_indices = z_scores[anom_mask].mean(axis=0).argsort()[::-1][:5]
    top_features     = [ANOM_FEATS[i] for i in top_feat_indices]

    # Save: pipeline = scaler + IsolationForest
    full_pipe = {"scaler": scaler, "iso": iso, "feature_cols": ANOM_FEATS}
    joblib.dump(full_pipe, OUT_DIR / "anomaly_detector.joblib", compress=3)

    meta = {
        "model_file":             "anomaly_detector.joblib",
        "contamination":          0.05,
        "n_anomalies_found":      n_anom,
        "anomaly_pct":            round(pct * 100, 2),
        "feature_cols":           ANOM_FEATS,
        "top_deviation_features": top_features,
        "anomaly_profile":        anom_stats,
        "normal_profile":         normal_stats,
        "score_range":            "0-100 (100 = most suspicious / most anomalous)",
        "threshold_for_flag":     60.0,
    }
    (OUT_DIR / "anomaly_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  Anomaly detector saved | {n_anom} anomalies ({pct:.1%}) detected")
    print(f"  Top deviation features: {top_features}")
    print(f"  Anomaly conversion rate: {anom_stats['conversion_rate']:.1%} vs normal: {normal_stats['conversion_rate']:.1%}")
    return full_pipe, meta, labels, anom_score_0_100


if __name__ == "__main__":
    print("=" * 55)
    print("  Training Extra Models")
    print("=" * 55)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"Dataset: {len(df):,} rows loaded")
    print()
    val_pipe, val_meta, val_preds = train_value_estimator(df)
    print()
    anom_pipe, anom_meta, anom_labels, anom_scores = train_anomaly_detector(df)
    print()
    print("=" * 55)
    print("Both models saved to models/")
    print("  models/value_estimator.joblib")
    print("  models/value_estimator_meta.json")
    print("  models/anomaly_detector.joblib")
    print("  models/anomaly_meta.json")
    print("=" * 55)
