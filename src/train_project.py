from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, learning_curve, train_test_split)
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
sns.set_theme(style="whitegrid", palette=["#18324B", "#147D7A", "#E9A23B"])

sys.path.insert(0, str(ROOT))
from src.pipeline import (CATEGORICAL, MONTHS, PASSTHROUGH, SKEWED,
                          make_model, make_preprocessor, make_xgb)


def load_data() -> pd.DataFrame:
    csvs = list(RAW_DIR.rglob("online_shoppers_intention.csv"))
    if not csvs:
        zips = list(RAW_DIR.glob("*.zip"))
        if zips:
            with zipfile.ZipFile(zips[0]) as archive:
                archive.extractall(RAW_DIR / "uci_extracted")
            csvs = list(RAW_DIR.rglob("online_shoppers_intention.csv"))
    if not csvs:
        raise FileNotFoundError("Place online_shoppers_intention.csv in data/raw or download the UCI archive.")
    df = pd.read_csv(csvs[0])
    df = df.drop_duplicates().reset_index(drop=True)
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[c] = df[c].astype("category")
    return df


def serialize(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df.to_csv(ROOT / "data" / "processed" / "shoppers_clean.csv", index=False)
    X = df.drop(columns="Revenue")
    y = df["Revenue"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    result: dict = {
        "seed": SEED,
        "rows_after_dedup": int(len(df)),
        "duplicates_removed": int(12330 - len(df)),
        "positive_rate": float(y.mean()),
        "training_rows_for_cv": int(len(X_train)),
        "heldout_test_rerun": False,
        "note": "Recorded report test results were not re-evaluated. New robustness work uses training folds only.",
    }

    # Matched model family comparison: same architecture settings in both feature sets.
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                min_samples_leaf=3, n_jobs=4, random_state=SEED)
    xgb = make_xgb(pos_weight)
    comparison = []
    for feature_set, drop in [("with PageValues", ()), ("without PageValues", ("PageValues",))]:
        for name, estimator in [("Logistic regression", lr), ("Random forest", rf), ("XGBoost", xgb)]:
            pipe = Pipeline([("prep", make_preprocessor(drop)), ("clf", estimator)])
            cv_scores = cross_validate(pipe, X_train, y_train, cv=CV,
                                       scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
                                       n_jobs=1, return_train_score=False)
            comparison.append({"feature_set": feature_set, "model": name,
                               "fold_pr_auc": cv_scores["test_pr_auc"].round(4).tolist(),
                               "mean_pr_auc": float(cv_scores["test_pr_auc"].mean()),
                               "std_pr_auc": float(cv_scores["test_pr_auc"].std()),
                               "mean_roc_auc": float(cv_scores["test_roc_auc"].mean())})
    result["matched_model_comparison"] = comparison

    # Learning curve on the training partition only.
    no_pv_train = X_train.drop(columns="PageValues")
    lc_model = make_model(pos_weight)
    sizes, train_scores, cv_scores = learning_curve(
        lc_model, no_pv_train, y_train, cv=CV,
        train_sizes=[0.25, 0.50, 0.75, 1.0], scoring="average_precision",
        n_jobs=1, shuffle=True, random_state=SEED,
    )
    result["learning_curve"] = [
        {"training_rows": int(n), "train_pr_auc_mean": float(a.mean()),
         "cv_pr_auc_mean": float(b.mean()), "cv_pr_auc_std": float(b.std())}
        for n, a, b in zip(sizes, train_scores, cv_scores)
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, cv_scores.mean(axis=1), marker="o", color="#147D7A", label="CV PR-AUC")
    ax.fill_between(sizes, cv_scores.mean(axis=1)-cv_scores.std(axis=1),
                    cv_scores.mean(axis=1)+cv_scores.std(axis=1), color="#147D7A", alpha=.16)
    ax.set(xlabel="Training rows", ylabel="Average precision", title="Learning curve: XGBoost without PageValues")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR/"learning_curve.png", dpi=180); plt.close(fig)

    # SMOTE stays inside every CV training fold; no resampling occurs before splitting.
    smote_pipe = ImbPipeline([
        ("prep", make_preprocessor(("PageValues",))),
        ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
        ("clf", make_xgb(1.0)),
    ])
    smote_scores = cross_validate(smote_pipe, no_pv_train, y_train, cv=CV,
                                  scoring="average_precision", n_jobs=1)
    weighted_pipe = make_model(pos_weight)
    weighted_scores = cross_validate(weighted_pipe, no_pv_train, y_train, cv=CV,
                                     scoring="average_precision", n_jobs=1)
    result["imbalance_comparison"] = {
        "class_weighted_xgb_fold_pr_auc": weighted_scores["test_score"].round(4).tolist(),
        "class_weighted_xgb_mean": float(weighted_scores["test_score"].mean()),
        "smote_in_cv_xgb_fold_pr_auc": smote_scores["test_score"].round(4).tolist(),
        "smote_in_cv_xgb_mean": float(smote_scores["test_score"].mean()),
        "note": "SMOTE is applied after one-hot encoding inside each CV fold; synthetic categorical coordinates are fractional, so interpret as a sensitivity check.",
    }

    # Nested cross-validation calibration estimates calibrated scores without test labels.
    raw_oof = cross_val_predict(make_model(pos_weight), X, y, cv=CV,
                                method="predict_proba", n_jobs=1)[:, 1]
    calibrated = CalibratedClassifierCV(estimator=make_model(pos_weight), method="sigmoid", cv=5)
    cal_oof = cross_val_predict(calibrated, X, y, cv=CV,
                                method="predict_proba", n_jobs=1)[:, 1]
    frac, mean_pred = calibration_curve(y, cal_oof, n_bins=10, strategy="quantile")
    result["calibration"] = {
        "raw_oof_brier": float(brier_score_loss(y, raw_oof)),
        "sigmoid_calibrated_oof_brier": float(brier_score_loss(y, cal_oof)),
        "reliability_fraction_positive": frac.round(4).tolist(),
        "reliability_mean_score": mean_pred.round(4).tolist(),
        "method": "Sigmoid calibration nested in 5-fold cross-validation",
        "app_displays_probability": False,
    }
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean_pred, frac, marker="o", color="#147D7A", label="Nested-CV calibrated scores")
    ax.plot([0,1],[0,1], "--", color="#8998A8", label="Ideal")
    ax.set(xlabel="Mean predicted score", ylabel="Observed purchase rate", title="Cross-validated reliability")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR/"calibration_curve.png", dpi=180); plt.close(fig)

    # Optional segmentation: fit on engagement behavior only, without PageValues or Revenue.
    seg_cols = ["Administrative", "Administrative_Duration", "Informational",
                "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                "BounceRates", "ExitRates"]
    seg = Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
    Z = seg.fit_transform(df[seg_cols])
    cluster_options = []
    for k in range(3, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        cluster_options.append({"k": k, "silhouette": float(silhouette_score(Z, km.labels_)), "model": km})
    chosen = max(cluster_options, key=lambda x: x["silhouette"])
    labels = chosen["model"].labels_
    profile = df.assign(segment=labels).groupby("segment").agg(
        sessions=("Revenue", "size"), conversion_rate=("Revenue", "mean"),
        median_product_pages=("ProductRelated", "median"),
        median_product_duration=("ProductRelated_Duration", "median"),
        median_exit_rate=("ExitRates", "median"),
    ).reset_index()
    profile.to_csv(ROOT/"reports"/"segment_profiles.csv", index=False)
    result["segmentation"] = {"selected_k": int(chosen["k"]),
                               "silhouette": float(chosen["silhouette"]),
                               "candidate_silhouettes": [{"k": x["k"], "silhouette": x["silhouette"]} for x in cluster_options],
                               "profiles": profile.to_dict(orient="records"),
                               "features_excluded": ["PageValues", "Revenue"]}

    # Cross-fitted score bands and cost thresholds for the working app.
    cut_hi, cut_mid = np.quantile(raw_oof, [0.70, 0.40])
    band = np.where(raw_oof >= cut_hi, "High", np.where(raw_oof >= cut_mid, "Medium", "Low"))
    band_stats = {}
    for name in ["High", "Medium", "Low"]:
        mask = band == name
        band_stats[name] = {"sessions": int(mask.sum()), "conversion_rate": float(y.to_numpy()[mask].mean())}
    threshold_map = {}
    cost_sensitivity = []
    grid = np.linspace(0.02, 0.98, 97)
    for ratio in [3, 5, 10]:
        costs = []
        for threshold in grid:
            pred = raw_oof >= threshold
            fp = int(((pred == 1) & (y.to_numpy() == 0)).sum())
            fn = int(((pred == 0) & (y.to_numpy() == 1)).sum())
            costs.append(fp + ratio * fn)
        idx = int(np.argmin(costs))
        threshold = float(grid[idx])
        pred = raw_oof >= threshold
        fp = int(((pred == 1) & (y.to_numpy() == 0)).sum())
        fn = int(((pred == 0) & (y.to_numpy() == 1)).sum())
        do_nothing = ratio * int(y.sum())
        flag_all = int((y == 0).sum())
        cheaper = min(do_nothing, flag_all)
        threshold_map[str(ratio)] = threshold
        cost_sensitivity.append({"R": ratio, "threshold": threshold,
                                 "precision": float(precision_score(y, pred, zero_division=0)),
                                 "recall": float(recall_score(y, pred, zero_division=0)),
                                 "flagged_percent": float(pred.mean()*100),
                                 "cost_vs_do_nothing_percent": float((fp+ratio*fn)/do_nothing*100),
                                 "cost_vs_flag_all_percent": float((fp+ratio*fn)/flag_all*100),
                                 "cost_vs_cheaper_baseline_percent": float((fp+ratio*fn)/cheaper*100)})
    result["app_oof_score_bands"] = band_stats
    result["app_oof_cost_sensitivity"] = cost_sensitivity

    # Fit the scoring artifact on all deduplicated rows only after the cross-fitted
    # estimates are collected. The previous report's held-out results are untouched.
    final_weight = float((y == 0).sum() / (y == 1).sum())
    final_model = make_model(final_weight).fit(X, y)
    serialize(final_model, OUT_DIR/"xgb_no_pagevalues.joblib")
    metadata = {
        "artifact_version": 1,
        "model_file": "xgb_no_pagevalues.joblib",
        "target": "Revenue",
        "primary_feature_set": "PageValues excluded",
        "score_bands": {"high_cutoff": float(cut_hi), "medium_cutoff": float(cut_mid), "rates": band_stats},
        "cost_thresholds": threshold_map,
        "cost_sensitivity": cost_sensitivity,
        "observed_months": sorted(df["Month"].astype(str).unique().tolist()),
        "visitor_types": sorted(df["VisitorType"].astype(str).unique().tolist()),
        "technical_category_options": {c: sorted(map(int, df[c].astype(int).unique())) for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]},
        "default_values": {"Administrative": 0, "Administrative_Duration": 0,
                           "Informational": 0, "Informational_Duration": 0,
                           "ProductRelated": 10, "ProductRelated_Duration": 300,
                           "BounceRates": 0.02, "ExitRates": 0.03,
                           "SpecialDay": 0.0, "Month": "Nov", "VisitorType": "Returning_Visitor",
                           "Weekend": 0, "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 2},
        "reference_rate_method": "5-fold cross-fitted predictions over the deduplicated dataset; not a separate held-out band estimate.",
        "fit_rows": int(len(df)),
        "fit_seed": SEED,
        "raw_score_is_probability": False,
        "calibration": result["calibration"],
        "limitations": ["Completed-session scoring only", "PageValues excluded", "Month has 10 observed categories", "Band rates are cross-validated historical estimates", "Associations are not causal"],
    }
    (OUT_DIR/"app_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Add current data audit and feature diagnostics for the report.
    summary = {"dataset_rows_source": 12330, "rows_after_dedup": int(len(df)),
               "duplicates_removed": int(12330-len(df)), "positive_rate": float(y.mean()),
               "pagevalues_zero_share": float((df.PageValues == 0).mean()),
               "conversion_pagevalues_zero": float(df.loc[df.PageValues == 0, "Revenue"].mean()),
               "conversion_pagevalues_positive": float(df.loc[df.PageValues > 0, "Revenue"].mean()),
               "duration_over_3h_count": int((df.ProductRelated_Duration > 10800).sum()),
               "month_counts": df.Month.value_counts().to_dict(),
               "visitor_rates": df.groupby("VisitorType", observed=True).Revenue.agg(["mean", "count"]).round(4).to_dict(orient="index"),
               "matched_model_comparison": comparison,
               "learning_curve": result["learning_curve"],
               "imbalance_comparison": result["imbalance_comparison"],
               "calibration": result["calibration"],
               "segmentation": result["segmentation"],
               "app_oof_score_bands": band_stats,
               "app_oof_cost_sensitivity": cost_sensitivity,
               "runtime_seconds": round(time.time()-t0, 2)}
    (ROOT/"reports"/"supplemental_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
