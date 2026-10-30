# Visitor Intelligence — AI-Powered Purchase Intent Engine
### Data Science Course Project | VIT | 3rd Semester

---

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Our Solution](#our-solution)
3. [Dataset Overview](#dataset-overview)
4. [Project Architecture](#project-architecture)
5. [Data Science Techniques Used](#data-science-techniques-used)
6. [Model Results & Evaluation](#model-results--evaluation)
7. [Key Findings](#key-findings)
8. [Streamlit Web Application](#streamlit-web-application)
9. [Tech Stack](#tech-stack)
10. [Limitations](#limitations)
11. [References](#references)

---

## 🔴 Problem Statement

E-commerce websites receive thousands of visitors every day, but only a small fraction of those visits actually end in a purchase. The industry-wide average conversion rate is roughly **2–3%**, and in our dataset it is **15.6%**.

The core business problem:

> **Treating every visitor the same wastes marketing budget.**
> Showing discount popups, live-chat prompts, and retargeting ads to visitors who were never going to buy anyway costs money and degrades the user experience for genuine browsers.

**Specific challenges we address:**

- How do you identify *which* sessions are likely to convert — before the visitor leaves?
- How do you handle severely **imbalanced classes** (84.4% non-buyers vs 15.6% buyers)?
- How do you avoid **data leakage** that inflates model performance and makes it useless in production?
- How do you translate a probability score into an **actionable business decision** (intervene or not)?

---

## ✅ Our Solution

We built a **session-level purchase intent classifier** using machine learning on the UCI Online Shoppers Purchasing Intention Dataset.

The system:

1. Takes **16 aggregated session features** (page counts, dwell times, bounce/exit rates, visitor type, month, etc.)
2. Passes them through a **leakage-free XGBoost pipeline** (PageValues deliberately excluded)
3. Outputs an **intent band — High / Medium / Low** — along with a ranking score
4. Applies **cost-sensitive thresholding** to decide which sessions deserve a real-time intervention

This is deployed as a fully working **Streamlit web application** with Firebase authentication, single-session scoring, CSV batch scoring, and interactive dashboards.

---

## 📊 Dataset Overview

| Property | Detail |
|---|---|
| **Name** | UCI Online Shoppers Purchasing Intention Dataset |
| **Source** | UCI Machine Learning Repository |
| **Authors** | C. Sakar & S. O. Kastro (2018) |
| **DOI** | [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q) |
| **Original rows** | 12,330 sessions |
| **After deduplication** | **12,205 rows** (125 exact duplicates removed) |
| **Total columns** | **18** (17 features + 1 target) |
| **Used for modeling** | **16 features** (PageValues excluded — see leakage section) |
| **Target column** | `Revenue` — True = purchase, False = no purchase |
| **Positive class rate** | **15.6%** (1,908 purchasing sessions) |
| **Time period** | 1 year of sessions (Jan and Apr are missing from the data) |

### Feature Breakdown

| Group | Feature Names | Type |
|---|---|---|
| Page Counts | Administrative, Informational, ProductRelated | Numeric (integer) |
| Time Spent | Administrative_Duration, Informational_Duration, ProductRelated_Duration | Numeric (float, seconds) |
| Google Analytics | BounceRates, ExitRates, PageValues | Numeric (float, ratio/score) |
| Context | SpecialDay, Month, Weekend | Mixed |
| Technical / Traffic | OperatingSystems, Browser, Region, TrafficType | Categorical (nominal codes) |
| Visitor Type | VisitorType | Categorical (New / Returning / Other) |
| **Target** | **Revenue** | **Binary (True / False)** |

### Known Data Issues We Handled

| Issue | How We Handled It |
|---|---|
| 125 duplicate rows | Removed before any analysis |
| Heavy right skew in counts & durations | Applied `log1p` transformation |
| BounceRates & ExitRates highly correlated (~0.91) | Kept both for tree models; tested dropping one for logistic regression |
| PageValues is post-purchase derived (leakage) | Mandatory ablation — excluded from deployed model |
| Categorical IDs (Browser, OS, Region) are nominal | OneHotEncoded, not treated as ordinal |
| Only 10 of 12 months present | Noted as limitation; no seasonality generalization |

---

## 🏗️ Project Architecture

```
Raw CSV Data
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Data Cleaning                                       │
│  · Remove 125 duplicates                            │
│  · Cast Revenue, Weekend to int                     │
│  · Sanity check: rates in [0,1], durations ≥ 0     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Exploratory Data Analysis (EDA)                     │
│  · 6–8 visualizations with written takeaways        │
│  · PageValues leakage preview                       │
│  · Correlation heatmap                              │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  80/20 Stratified Train/Test Split (seed=42)         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  5-Fold Stratified Cross-Validation on Training Set  │
│                                                      │
│  sklearn Pipeline (inside every fold):              │
│  · log1p → StandardScaler  (skewed numerics)        │
│  · OneHotEncoder            (categoricals)          │
│  · class weighting / SMOTE  (imbalance)             │
│  · Model (LR / RF / XGBoost)                       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Hyperparameter Tuning (RandomizedSearchCV)          │
│  · Optimizing PR-AUC (average_precision)            │
│  · ~25–30 iterations per model                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Evaluation on Held-Out Test Set (touched once)      │
│  · PR-AUC, ROC-AUC, F1, Precision, Recall          │
│  · Confusion matrix at chosen threshold             │
│  · Decile lift table                               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PageValues Ablation + SHAP Interpretation           │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  K-Means Customer Segmentation (Optional)            │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Streamlit Web App Deployment                        │
│  · Loads saved joblib model                         │
│  · Single session + CSV batch scoring               │
│  · Firebase Auth (Google / Email / Guest)           │
└─────────────────────────────────────────────────────┘
```

---

## 🔬 Data Science Techniques Used

### 1. Exploratory Data Analysis (EDA)

We created **6–8 visualizations**, each with a one-line written takeaway:

| Plot | Key Takeaway |
|---|---|
| Class balance bar chart | Only 15.6% of sessions convert — severe class imbalance |
| Conversion rate by Month | November and May drive the highest volume; conversion peaks in November |
| Conversion by VisitorType & Weekend | New visitors convert at 24.9% vs returning at 14.1%; weekday sessions slightly edge weekends |
| KDE distributions (log scale) of PageValues, ProductRelated_Duration, BounceRates, ExitRates split by Revenue | Buyers have much higher PageValues and ProductRelated_Duration; lower exit/bounce rates |
| Correlation heatmap | BounceRates and ExitRates are correlated at 0.91 — near-collinear |
| Conversion by TrafficType (n ≥ 100) | Some traffic types convert at 2× the average; paid vs. organic patterns visible |

**Leakage Preview Finding:**

```
PageValues == 0  →  3.9%  conversion rate   (77.6% of sessions)
PageValues  > 0  →  56.3% conversion rate   (22.4% of sessions)
```

This extreme split reveals that PageValues is computed from post-purchase page monetization data — it encodes the outcome and must be excluded.

---

### 2. Preprocessing Pipeline

All transformations are fit **only on training data** using sklearn `Pipeline` and `ColumnTransformer` — no leakage from validation or test data.

| Step | Applied To | Method |
|---|---|---|
| Log1p transform | Administrative, Informational, ProductRelated counts + all durations + BounceRates, ExitRates | `FunctionTransformer(np.log1p)` |
| Standard scaling | Same skewed columns (for logistic regression compatibility) | `StandardScaler()` |
| One-hot encoding | Month, VisitorType, OS, Browser, Region, TrafficType | `OneHotEncoder(handle_unknown='ignore', min_frequency=30)` |
| Passthrough | SpecialDay, Weekend | No transformation needed |
| **Excluded** | **PageValues** | **Deliberate — lookahead leakage** |

---

### 3. Handling Class Imbalance

The dataset has **84.4% non-buyers** vs **15.6% buyers**. Two strategies were tested and compared:

**Strategy A — Class Weights (built into models):**
- Logistic Regression: `class_weight='balanced'`
- Random Forest: `class_weight='balanced_subsample'`
- XGBoost: `scale_pos_weight = n_negatives / n_positives ≈ 5.4`

**Strategy B — SMOTE (Synthetic Minority Oversampling):**
- Applied using `imblearn.pipeline.Pipeline`
- SMOTE runs **inside each CV fold on the training portion only** — never before splitting
- This is the correct way; applying SMOTE before splitting inflates scores significantly

**Results (CV PR-AUC on training set, XGBoost, no PageValues):**

| Method | Mean CV PR-AUC |
|---|---|
| Class-weighted XGBoost | **0.379** |
| SMOTE inside CV + XGBoost | 0.355 |

Class weighting performed better here; SMOTE on one-hot encoded categoricals produces fractional synthetic values which degrades quality.

---

### 4. Three Models Compared

| # | Model | Imbalance Strategy | CV PR-AUC (with PV) | CV PR-AUC (without PV) |
|---|---|---|---|---|
| 1 | Logistic Regression | class_weight='balanced' | 0.664 | 0.330 |
| 2 | Random Forest | balanced_subsample | 0.737 | 0.384 |
| 3 | **XGBoost** ✅ | scale_pos_weight | **0.749** | **0.379** |

XGBoost was selected as the final model. We focused on **depth over breadth** — only 3 models, each thoroughly evaluated.

---

### 5. Hyperparameter Tuning

Used `RandomizedSearchCV` with **25 iterations**, optimizing `average_precision` (PR-AUC):

**Search Space:**

| Parameter | Range Searched |
|---|---|
| n_estimators | 150 – 500 |
| learning_rate | 0.01 – 0.11 |
| max_depth | 2 – 6 |
| min_child_weight | 1 – 10 |
| subsample | 0.6 – 1.0 |
| colsample_bytree | 0.5 – 1.0 |

**Best Found Parameters (deployed model):**

```
n_estimators    = 265
learning_rate   = 0.037
max_depth       = 4
min_child_weight= 9
subsample       = 0.741
colsample_bytree= 0.953
scale_pos_weight= 5.39
```

---

### 6. ⚠️ PageValues Ablation Study (Leakage Analysis)

This is one of the most important contributions of the project.

**What is PageValues?**
PageValues is a Google Analytics metric representing the average value of a page that a user visited before completing a purchase. It is computed *retrospectively from completed transactions* — meaning it is not available at the start of a session.

**Ablation Results (held-out test set):**

| Model Configuration | PR-AUC | ROC-AUC |
|---|---|---|
| XGBoost **with** PageValues | **0.758** | **0.938** |
| XGBoost **without** PageValues (deployed) | **0.376** | **0.795** |
| Gap | −0.382 | −0.143 |

**Conclusion:** The with-PageValues model is an upper bound / academic benchmark. The without-PageValues model is the honest, production-realistic version. **Our deployed app uses only the leakage-free model.**

---

### 7. Evaluation Metrics

We deliberately **did not use Accuracy** as the primary metric.

> A naive model that predicts "no purchase" for everyone gets **84.4% accuracy** while catching zero buyers. This is misleading.

**Metrics we used:**

| Metric | Why It Matters |
|---|---|
| **PR-AUC** (Primary) | Summarizes precision-recall trade-off across all thresholds; ideal for imbalanced data |
| ROC-AUC | Measures ranking ability; less sensitive to class imbalance than accuracy |
| Precision | Of sessions we flagged, how many were real buyers? |
| Recall | Of all real buyers, how many did we catch? |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | At chosen threshold — actual counts of TP, FP, TN, FN |

---

### 8. Threshold Optimization (Cost-Sensitive Decision)

Instead of using the default 0.5 cutoff, we swept all thresholds and minimized a business cost function:

```
Total Cost = False Positives + R × False Negatives
```

- **False Positive** = wasted intervention on a non-buyer (cost = 1)
- **False Negative** = missed buyer who gets no offer (cost = R)
- **R** = how much worse it is to miss a buyer vs. waste an offer

**Thresholds calibrated on Out-Of-Fold predictions (not test set):**

| Assumed R | Threshold | Precision | Recall | % Sessions Flagged | Cost vs. Do Nothing |
|---|---|---|---|---|---|
| R = 3 (Conservative) | 0.60 | 35.4% | 57.9% | 25.6% | 77.3% |
| R = 5 (Balanced) ✅ | 0.49 | 29.6% | 75.9% | 40.1% | 60.2% |
| R = 10 (Aggressive) | 0.33 | 24.5% | 91.4% | 58.2% | 36.7% |

At R=5 (default), we catch **75.9% of all real buyers** while only flagging **40% of sessions** — a 40% reduction in unnecessary interventions vs. flagging everyone.

---

### 9. SHAP Explainability

Used `shap.TreeExplainer` on the test set to explain individual predictions.

**Top Features by Mean |SHAP| Value:**

| Rank | Feature | SHAP Importance | Interpretation |
|---|---|---|---|
| 1 | ProductRelated_Duration | 0.444 | Longer product page dwell time → strong buy signal |
| 2 | Month | 0.428 | November sessions convert significantly more |
| 3 | ExitRates | 0.410 | Lower exit rate → more likely to stay and convert |
| 4 | Browser | 0.207 | Some browser types correlate with higher intent |
| 5 | TrafficType | 0.160 | Paid/direct traffic types show different conversion patterns |
| 6 | BounceRates | 0.111 | Low bounce rate = engaged visitor |
| 7 | Administrative_Duration | 0.109 | Time on account/admin pages suggests intent to complete a transaction |

**Key SHAP Finding:** Sessions with high `ProductRelated_Duration` AND low `ExitRates` are very strongly associated with purchase. High exit rates are a dominant negative signal.

**Permutation Importance** was also run as a robustness check — it agreed on the top 3 features, confirming the SHAP findings are stable.

> ⚠️ Important note: SHAP shows **association**, not causation. We cannot claim that increasing product page time *causes* purchases.

---

### 10. K-Means Customer Segmentation

Applied K-Means clustering on engagement features only (no PageValues, no Revenue target):

**Features used:** Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates

**Preprocessing:** log1p → StandardScaler

**Choosing k:** Tested k = 3, 4, 5, 6 using Silhouette Score:

| k | Silhouette Score |
|---|---|
| **3** | **0.434** ✅ (best) |
| 4 | 0.409 |
| 5 | 0.369 |
| 6 | 0.363 |

**Three Discovered Segments:**

| Segment | Sessions | Conversion Rate | Median Product Pages | Median Duration | Median Exit Rate | Profile Name |
|---|---|---|---|---|---|---|
| 0 | 2,307 | **24.0%** | 43 pages | 1,664 sec | 1.9% | 🟢 Focused Buyers |
| 1 | 9,013 | 15.0% | 17 pages | 553 sec | 2.5% | 🟡 Casual Browsers |
| 2 | 885 | **0.5%** | 1 page | 0 sec | **20%** | 🔴 Bouncers |

**Takeaway:** Segment 2 ("Bouncers") visits 1 product page with zero dwell time and a 20% exit rate — these sessions have essentially zero purchase probability and should not receive expensive interventions.

---

### 11. Calibration Analysis

Raw XGBoost scores are not calibrated probabilities — they are ranking scores.

We tested **sigmoid calibration** using nested 5-fold cross-validation:

| | Brier Score (lower = better) |
|---|---|
| Raw OOF scores | 0.185 |
| Sigmoid-calibrated OOF scores | **0.115** |

Calibration curve was plotted to visualize reliability. The app uses raw scores for **ranking** only, not as probability estimates.

---

## 📈 Model Results & Evaluation

### Final Held-Out Test Set Results (XGBoost, No PageValues)

| Metric | Value |
|---|---|
| **PR-AUC (Primary)** | **0.376** |
| ROC-AUC | 0.795 |
| Precision (at R=5 threshold) | 29.6% |
| Recall (at R=5 threshold) | 75.9% |
| F1-Score (at R=5 threshold) | 0.423 |
| Majority-class accuracy baseline | 84.4% (catches 0 buyers) |

### Decile Lift Analysis

| Top % of Sessions Flagged | Conversion Rate in Segment | Lift vs. Base Rate |
|---|---|---|
| Top 10% | **44.5%** | **2.85×** |
| Top 20% | 35.0% | 2.24× |
| Top 30% | 21.6% | 1.38× |
| Bottom 40% | 6.0% | 0.38× |

**Interpretation:** If we only intervene with the top 10% of sessions ranked by our model, we hit a 44.5% conversion rate — nearly 3× the baseline. This means we can capture a large share of buyers while contacting far fewer visitors.

---

## 🔑 Key Findings

1. **ProductRelated_Duration is the strongest signal** — how long a visitor spends on product pages is the most predictive behavioral feature.

2. **Exit rate is a dominant negative signal** — sessions with exit rate > 8% are strongly associated with non-purchase outcomes.

3. **PageValues creates an artificial 0.76 PR-AUC** — but it's post-purchase data and unusable in real-time. Our honest model gets 0.376 PR-AUC, which is still 2.85× better than random at the top decile.

4. **Accuracy is the wrong metric** — with 15.6% positives, an 84.4% accuracy model that predicts no purchase for everyone is useless. PR-AUC is the correct primary metric.

5. **SMOTE needs careful handling** — applying it before splitting inflates scores. Applied correctly (inside each CV fold), it actually performs slightly worse than class weighting on this dataset.

6. **"Bouncers" segment (7.2% of sessions) has a 0.5% conversion rate** — these should be excluded from any intervention strategy entirely.

7. **November is the highest-conversion month** — seasonal context matters; the model leverages month as the second most important feature.

---

## 🌐 Streamlit Web Application

A fully functional interactive web app was built and deployed locally.

### Pages & Features

| Page | Access | Description |
|---|---|---|
| 🏠 Home | Public | Landing page with live metrics, feature overview, animated charts |
| 🔑 Login | Public | Firebase Auth: Google OAuth, Email/Password, Guest Demo |
| 📊 Dashboard (Overview) | Protected | KPI cards, donut chart, monthly conversion trend, intent band distribution |
| 🔮 Predict Intent | Protected | Single session scoring + CSV batch scoring with download |
| 📈 Insights | Protected | PR-AUC results, ablation chart, SHAP/permutation importance, decile lift |
| 🔬 Methodology | Public | Full ML pipeline explanation, hyperparameters, cost formula |
| ℹ️ About | Public | Project background, limitations, dataset citation |

### Unique App Features

- **Session presets** — One-click load of "High-Intent Buyer", "Hesitant Evaluator", "Bounce Risk", "New Explorer" scenarios
- **Speedometer gauge** — Visual indicator of where the session scores relative to band cutoffs
- **Behavioral DNA Radar Chart** — Compares the current session's 5 key engagement signals against a converting-buyer benchmark
- **CSV batch scoring** — Upload a CSV, score all rows through the real model, download results
- **Session persistence** — Cryptographically signed tokens (itsdangerous) keep users logged in across browser refreshes
- **Firebase Auth** — Google OAuth + Email/Password + Guest Demo, with clean logout pipeline

---

## 🛠️ Tech Stack

| Category | Tools / Libraries | Version |
|---|---|---|
| Language | Python | 3.11+ |
| Data manipulation | pandas, numpy | 2.2.3, 2.1.3 |
| Visualization | matplotlib, seaborn, plotly | 3.10.0, 0.13.2 |
| Machine Learning | scikit-learn | 1.6.1 |
| Boosting | XGBoost | 3.4.1 |
| Imbalanced learning | imbalanced-learn | 0.14.2 |
| Explainability | shap | 0.54.1 |
| Statistical | scipy | 1.16.3 |
| Model serialization | joblib | 1.6.0 |
| Dataset access | ucimlrepo | 0.0.7 |
| Web application | Streamlit | 1.66.1 |
| Authentication | Google Firebase (compat SDK 10.14.1) | — |
| Session security | itsdangerous | — |

---

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| No timestamps | A time-based validation split was impossible; we used random stratified splitting |
| Single-site, single-year data | Results may not generalize to other retailers or time periods |
| Two months missing | January and April are absent from the dataset; seasonal conclusions are partial |
| PageValues excluded | The most predictive feature was removed for production realism — honest but lower PR-AUC |
| Session aggregates only | Features are session-level totals; the sequence and order of individual page visits is lost |
| Assumed cost ratios | The R=3/5/10 values are illustrative assumptions, not measured from real business outcomes |
| Associations ≠ Causation | All SHAP findings are correlational — we cannot claim that changing any feature will change purchase behavior |

---

## 📚 References

1. **Dataset:** Sakar, C. O., & Kastro, O. S. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)

2. **XGBoost:** Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.

3. **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.

4. **SMOTE:** Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.

5. **scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.

---

## 📁 Project Folder Structure

```
Visitor-Intelligence/
│
├── app/
│   ├── app.py                      # Streamlit web application (1745 lines)
│   ├── logic.py                    # Feature schema, validation, scoring helpers
│   └── firebase_auth/
│       └── index.html              # Firebase Auth UI component
│
├── data/
│   ├── raw/
│   │   ├── online_shoppers.zip     # Original UCI archive
│   │   └── uci_extracted/
│   │       └── online_shoppers_intention.csv
│   ├── processed/
│   │   └── shoppers_clean.csv      # 12,205 deduplicated rows
│   └── sample_batch_scoring.csv    # Demo batch file
│
├── models/
│   ├── xgb_no_pagevalues.joblib    # Trained XGBoost pipeline (deployed)
│   └── app_meta.json               # Score bands, thresholds, category maps
│
├── notebooks/
│   └── 01_visitor_engagement_analysis.ipynb
│
├── reports/
│   ├── project_report.pdf
│   ├── project_report.md
│   ├── supplemental_results.json   # All CV scores, segmentation, calibration
│   ├── segment_profiles.csv        # K-means cluster profiles
│   └── figures/
│       ├── learning_curve.png
│       └── calibration_curve.png
│
├── src/
│   ├── pipeline.py                 # Shared preprocessing + model factory
│   └── train_project.py            # Reproducible full training script
│
├── DS_CP.ipynb                     # Main analysis notebook (Colab)
├── requirements.txt                # Pinned dependency versions
├── README.md                       # Setup & usage instructions
├── PROJECT_STATUS.md               # Completed features & verification log
└── PROJECT_PRESENTATION.md        # ← This file
```

---

*Built for VIT 3rd Semester Data Science Course Project | Visitor Intelligence v2.2*
