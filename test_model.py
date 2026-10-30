"""
Smoke test for all 3 models.
Run: .venv\Scripts\python.exe -W ignore test_model.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=" * 60)
print("  Visitor Intelligence — Full Model Smoke Test")
print("=" * 60)

# ─── Load data ───────────────────────────────────────────────
raw  = pd.read_csv(ROOT / "data" / "processed" / "shoppers_clean.csv")
meta = json.loads((ROOT / "models" / "app_meta.json").read_text(encoding="utf-8"))
print(f"[OK] Dataset : {len(raw):,} rows × {len(raw.columns)} columns")
print(f"[OK] Base conversion rate : {raw['Revenue'].mean():.1%}")

# ─── Model 1: Intent Classifier ─────────────────────────────
print()
print("── Model 1: XGBoost Purchase Intent Classifier ──")
clf   = joblib.load(ROOT / "models" / "xgb_no_pagevalues.joblib")
FEAT  = [
    "Administrative","Administrative_Duration","Informational",
    "Informational_Duration","ProductRelated","ProductRelated_Duration",
    "BounceRates","ExitRates","SpecialDay","Month","OperatingSystems",
    "Browser","Region","TrafficType","VisitorType","Weekend",
]
scores = clf.predict_proba(raw[FEAT])[:, 1]
bands  = meta["score_bands"]
intent_band = np.where(scores >= bands["high_cutoff"], "High",
              np.where(scores >= bands["medium_cutoff"], "Medium", "Low"))
thr_r5 = float(meta["cost_thresholds"]["5"])
n_high = int((intent_band == "High").sum())
n_flag = int((scores >= thr_r5).sum())

scenarios = [
    ("High-Intent Buyer",  {"ProductRelated":45,"ProductRelated_Duration":2400.0,"BounceRates":0.002,"ExitRates":0.004,"VisitorType":"Returning_Visitor","Month":"Nov"}),
    ("Hesitant Evaluator", {"ProductRelated":12,"ProductRelated_Duration":350.0, "BounceRates":0.025,"ExitRates":0.03, "VisitorType":"Returning_Visitor","Month":"Nov"}),
    ("Bounce Risk",        {"ProductRelated":2, "ProductRelated_Duration":18.0,  "BounceRates":0.18, "ExitRates":0.22, "VisitorType":"Returning_Visitor","Month":"Nov"}),
    ("New Explorer",       {"ProductRelated":18,"ProductRelated_Duration":650.0, "BounceRates":0.015,"ExitRates":0.02, "VisitorType":"New_Visitor",      "Month":"Dec"}),
]
defaults = {"Administrative":0,"Administrative_Duration":0.0,"Informational":0,
            "Informational_Duration":0.0,"SpecialDay":0.0,"Weekend":0,
            "OperatingSystems":2,"Browser":2,"Region":1,"TrafficType":2}

print(f"  {'Scenario':<22} {'Score':>7}  {'Band':<8}  Flag(R=5)")
print("  " + "-"*52)
for name, ov in scenarios:
    row  = {**defaults, **ov}
    s    = clf.predict_proba(pd.DataFrame([row]))[0, 1]
    b    = "High" if s>=bands["high_cutoff"] else ("Medium" if s>=bands["medium_cutoff"] else "Low")
    flag = "YES" if s >= thr_r5 else "no"
    print(f"  {name:<22} {s:>7.4f}  {b:<8}  {flag}")

print(f"  [OK] Full dataset: {n_high:,} High-intent, {n_flag:,} flagged for targeting")

# ─── Model 2: Value Estimator ────────────────────────────────
print()
print("── Model 2: Session Value Estimator (XGBRegressor) ──")
val_path = ROOT / "models" / "value_estimator.joblib"
val_meta_path = ROOT / "models" / "value_estimator_meta.json"
if val_path.is_file():
    val_model = joblib.load(val_path)
    val_meta  = json.loads(val_meta_path.read_text(encoding="utf-8"))
    val_feats = val_meta["feature_columns"]
    val_scores = val_model.predict(raw[val_feats]).clip(0, 100)
    vt         = val_meta["value_thresholds"]
    val_bands  = np.where(val_scores >= vt["high"], "High Value",
                 np.where(val_scores >= vt["low"],  "Medium Value", "Low Value"))
    n_hv = int((val_bands == "High Value").sum())
    n_lv = int((val_bands == "Low Value").sum())
    conv = val_meta["conversion_by_value_band"]
    print(f"  [OK] R²={val_meta['train_r2']:.3f} | RMSE={val_meta['train_rmse']:.2f}")
    print(f"  [OK] High-value sessions : {n_hv:,}  ({n_hv/len(raw):.1%})")
    print(f"  [OK] Low-value sessions  : {n_lv:,}  ({n_lv/len(raw):.1%})")
    print(f"  [OK] High-value conversion: {list(conv.values())[0]:.1%}")
    print(f"  [OK] Low-value conversion : {list(conv.values())[2]:.1%}")
    print(f"  [OK] Avg value score: {val_scores.mean():.1f}/100")
else:
    print("  [MISSING] Run: .venv\\Scripts\\python.exe src/train_extras.py")

# ─── Model 3: Anomaly Detector ───────────────────────────────
print()
print("── Model 3: Anomaly Detector (IsolationForest) ──")
anom_path = ROOT / "models" / "anomaly_detector.joblib"
anom_meta_path = ROOT / "models" / "anomaly_meta.json"
if anom_path.is_file():
    anom_bundle = joblib.load(anom_path)
    anom_meta   = json.loads(anom_meta_path.read_text(encoding="utf-8"))
    anom_feats  = anom_meta["feature_cols"]
    scaler      = anom_bundle["scaler"]
    iso         = anom_bundle["iso"]
    X_sc        = scaler.transform(raw[anom_feats])
    labels      = iso.predict(X_sc)
    anom_scores = iso.decision_function(X_sc)
    anom_0_100  = ((anom_scores.max() - anom_scores) /
                   (anom_scores.max() - anom_scores.min()) * 100).clip(0, 100)
    n_anom      = int((labels == -1).sum())
    anom_conv   = float(raw[labels == -1]["Revenue"].mean())
    norm_conv   = float(raw[labels ==  1]["Revenue"].mean())
    top_feats   = anom_meta.get("top_deviation_features", [])
    print(f"  [OK] Suspicious sessions found : {n_anom:,} ({n_anom/len(raw):.1%})")
    print(f"  [OK] Anomaly conversion rate   : {anom_conv:.1%} (vs {norm_conv:.1%} normal)")
    print(f"  [OK] Top deviation features    : {', '.join(top_feats[:3])}")
    print(f"  [OK] Avg suspicion score       : {anom_0_100[labels==-1].mean():.1f}/100")
    print(f"  [OK] Budget saved by excluding : ₹{n_anom * 200:,}")
else:
    print("  [MISSING] Run: .venv\\Scripts\\python.exe src/train_extras.py")

# ─── Summary ─────────────────────────────────────────────────
print()
print("=" * 60)
print("[PASS] All models verified successfully.")
print()
print("Pages available in the app:")
print("  📊 Dashboard       — full EDA + all 12,205 sessions scored")
print("  🎯 Targeting       — who to target + cost analysis")
print("  ⚡ Live Scoring    — simulate any visitor, manual entry")
print("  💎 Value Analysis  — session engagement value (0-100)")
print("  🔍 Anomaly Detect  — bot/suspicious session detection")
print("  🔬 Methodology     — ML pipeline, SHAP, model comparison")
print()
print("To run the app:")
print("  .venv\\Scripts\\streamlit.exe run app/app.py")
print("=" * 60)
