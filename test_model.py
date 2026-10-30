"""Quick smoke test — run with: .venv\Scripts\python.exe test_model.py"""
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("=" * 55)
print("  Visitor Intelligence — Model Smoke Test")
print("=" * 55)

# 1. Load artifacts
model = joblib.load(ROOT / "models" / "xgb_no_pagevalues.joblib")
meta  = json.loads((ROOT / "models" / "app_meta.json").read_text(encoding="utf-8"))
raw   = pd.read_csv(ROOT / "data" / "processed" / "shoppers_clean.csv")
print(f"[OK] Dataset loaded       : {len(raw):,} rows x {len(raw.columns)} columns")
print(f"[OK] Base conversion rate : {raw['Revenue'].mean():.1%}")
print(f"[OK] Model pipeline steps : {list(model.named_steps.keys())}")

# 2. Test 4 scenarios
scenarios = [
    ("High-Intent Buyer",    {"ProductRelated": 45, "ProductRelated_Duration": 2400.0, "BounceRates": 0.002, "ExitRates": 0.004, "VisitorType": "Returning_Visitor", "Month": "Nov"}),
    ("Hesitant Evaluator",   {"ProductRelated": 12, "ProductRelated_Duration": 350.0,  "BounceRates": 0.025, "ExitRates": 0.03,  "VisitorType": "Returning_Visitor", "Month": "Nov"}),
    ("Bounce Risk",          {"ProductRelated":  2, "ProductRelated_Duration":  18.0,  "BounceRates": 0.18,  "ExitRates": 0.22,  "VisitorType": "Returning_Visitor", "Month": "Nov"}),
    ("New Explorer",         {"ProductRelated": 18, "ProductRelated_Duration": 650.0,  "BounceRates": 0.015, "ExitRates": 0.02,  "VisitorType": "New_Visitor",       "Month": "Dec"}),
]

defaults = {
    "Administrative": 0, "Administrative_Duration": 0.0,
    "Informational":  0, "Informational_Duration":  0.0,
    "SpecialDay": 0.0, "Weekend": 0,
    "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2,
}

bands_cfg = meta["score_bands"]
thr_r5    = float(meta["cost_thresholds"]["5"])

print()
print(f"{'Scenario':<22}  {'Score':>7}  {'Band':<8}  {'Flag (R=5)'}")
print("-" * 55)

for name, overrides in scenarios:
    row = {**defaults, **overrides}
    score = model.predict_proba(pd.DataFrame([row]))[0, 1]
    if score >= bands_cfg["high_cutoff"]:
        band = "High"
    elif score >= bands_cfg["medium_cutoff"]:
        band = "Medium"
    else:
        band = "Low"
    flag = "YES — intervene" if score >= thr_r5 else "no"
    print(f"  {name:<20}  {score:>7.4f}  {band:<8}  {flag}")

# 3. Batch test with sample file
sample_path = ROOT / "data" / "sample_batch_scoring.csv"
if sample_path.is_file():
    batch = pd.read_csv(sample_path)
    scores = model.predict_proba(batch)[: , 1]
    print()
    print(f"[OK] Batch scoring ({len(batch)} rows):")
    for b, cnt in zip(["High", "Medium", "Low"],
                      [(scores >= bands_cfg["high_cutoff"]).sum(),
                       ((scores >= bands_cfg["medium_cutoff"]) & (scores < bands_cfg["high_cutoff"])).sum(),
                       (scores < bands_cfg["medium_cutoff"]).sum()]):
        print(f"     {b:<8}: {cnt} sessions")

print()
print("[PASS] All checks completed successfully.")
print()
print("To run the app:")
print('  .venv\\Scripts\\streamlit.exe run app/app.py')
