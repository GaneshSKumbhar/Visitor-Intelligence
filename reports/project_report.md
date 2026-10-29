# Website Visitor Engagement Analytics
## Predicting purchase intent from browsing behavior

**Data Science project report | UCI Online Shoppers Purchasing Intention**

## Executive summary

This project ranks e-commerce sessions by whether they end in a recorded purchase. The central finding is the PageValues ablation: the tuned XGBoost model achieved test PR-AUC 0.758 with `PageValues` and 0.376 without it. UCI defines PageValue using pages visited before a transaction, so the higher score may reflect a late-funnel signal. The no-PageValues model is the primary candidate for earlier targeting. It is above the 0.1565 test purchase prevalence and the strongest single-feature training benchmark (ExitRates, 0.259), but its performance is moderate.

At the preselected threshold 0.52 and assumed false-negative:false-positive cost ratio R=5, the model flagged 36.7% of test sessions, with 0.311 precision and 0.728 recall. Its modeled cost was 59.5% of doing nothing under this simple formula. This is a scenario estimate, not a measured business saving or causal effect.

## 1. Problem and decision context

The target is session-level binary classification (`Revenue`). The intended decision is which sessions might merit a low-cost intervention. This work evaluates ranking and hypothetical classification costs; it does not establish whether a discount or popup increases conversion. PR-AUC is the headline metric because buyers are a minority class. Scores are ranking scores, not calibrated probabilities.

## 2. Dataset and preparation

UCI's Online Shoppers Purchasing Intention dataset contains 12,330 sessions, 17 predictors, and the binary Revenue target. The features cover page counts and durations, Google Analytics metrics, calendar context, and visitor or traffic categories. UCI reports no missing values. The dataset description defines Bounce Rate and Exit Rate as Google Analytics page metrics and Page Value as the average value of pages visited before a transaction. [UCI dataset documentation](https://archive.ics.uci.edu/dataset/468/online)

The analysis removed 125 exact duplicates, leaving 12,205 rows. Conversion after cleaning was 15.63%. A stratified 80/20 split with seed 42 produced 9,764 training sessions and 2,441 test sessions; test prevalence was 15.65%. Numeric skewed features were log1p transformed and standardized, while categorical and technical code fields were one-hot encoded inside pipelines.

The majority classifier reaches 84.37% accuracy by predicting no purchase for every row. It identifies no buyers, illustrating why accuracy is not an appropriate headline measure.

## 3. Exploratory findings

New visitors converted at 24.9% (n=1,693) and returning visitors at 14.1% (n=10,431). The `Other` category has only 81 sessions and is not interpreted. Ten months appear; January and April are absent. February has about 181 sessions, too few to support a seasonal conclusion. Product-related browsing duration is positively associated with conversion in test-set bins. High ExitRates are associated with lower conversion; among sessions with ExitRates above 0.08, 4 of 305 converted. BounceRates and ExitRates are strongly correlated. These are descriptive associations, not causes.

## 4. Modeling and validation

The models were balanced logistic regression, class-weighted random forest, and class-weighted XGBoost. Five-fold stratified cross-validation was used on training data. Preprocessing remained within pipelines. XGBoost was tuned by 25-iteration randomized search separately with and without PageValues. The best CV PR-AUC was 0.7522 with PageValues and 0.3816 without; OOF PR-AUC for the selected no-PageValues model was 0.3751.

The selected configurations differ (230 trees with PageValues and 265 without), so the recorded ablation is not an identical-parameter comparison. Best-of-search scores also have selection optimism. A separate local matched-setting robustness analysis is reported in Section 9.

## 5. Held-out results and PageValues ablation

| Model | Test PR-AUC (95% bootstrap CI) | ROC-AUC | Interpretation |
|---|---:|---:|---|
| XGBoost with PageValues | 0.7577 (0.715-0.800) | 0.9378 | Purchase-path-tied upper bound |
| XGBoost without PageValues | 0.3762 (0.331-0.426) | 0.7949 | Primary candidate |
| No-skill reference | 0.1565 | - | Test purchase prevalence |

Removing PageValues reduces test PR-AUC by 0.3815. This supports treating it as a late-funnel signal, but does not prove leakage: the feature definition alone does not establish exactly when a live system could know it. The `PageValues > 0` rule scored test precision 0.571, recall 0.812, and F1 0.670; treat it as a descriptive late-funnel reference.

Training single-feature PR-AUC was 0.259 for inverted ExitRates, 0.244 for ProductRelated_Duration, and 0.238 for ProductRelated. The no-PageValues model adds signal, though not enough to call it a strong buyer detector.

## 6. Threshold and cost scenario

Threshold selection used training out-of-fold scores, minimizing `FP + R × FN`. R is an assumed cost ratio. At R=5, the no-PageValues threshold is 0.52; test precision is 0.311, recall 0.728, and 36.7% of sessions are flagged.

| Assumed R | Threshold | Precision | Recall | Flagged | Cost / do nothing | Cost / flag all |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.58 | 0.348 | 0.631 | 28.4% | 76.4% | 42.5% |
| 5 | 0.52 | 0.311 | 0.728 | 36.7% | 59.5% | 55.2% |
| 10 | 0.33 | 0.247 | 0.932 | 59.0% | 35.2% | 65.3% |

The cheaper baseline is the lower-cost of doing nothing or flagging everyone. Thus modeled reductions versus the cheaper baseline are 23.6%, 40.5%, and 34.7% at R=3, 5, and 10. Actual margin, intervention cost, and intervention effectiveness are not measured.

## 7. Ranking and interpretation

The highest-scored 30% of test sessions contained 247 of 382 buyers (64.7% cumulative recall); the bottom 40% contained 23 buyers (6.0%). The top decile conversion rate was 44.5%, or 2.84 times the prevalence. Small buyer counts make decile rates noisy.

Corrected grouped SHAP ranked ProductRelated_Duration, Month, and ExitRates highest. Test permutation importance ranked ProductRelated_Duration (0.0794), ExitRates (0.0738), and Month (0.0588). Removing Month reduced CV PR-AUC from 0.3816 to 0.3412. Removing Browser, OperatingSystems, Region, and TrafficType reduced it to 0.3743, a 0.0073 change that is small relative to fold variation. Importances describe model reliance, not causal drivers.

## 8. Limitations and deployment boundary

- One site and period; no timestamps for forward-in-time validation.
- January and April are absent; predictions for unseen months extrapolate.
- PageValues is tied by definition to pages visited before a transaction; exact live availability needs to be established.
- BounceRates and ExitRates are page-level analytics aggregated to sessions. Live use requires past-traffic-only computation and availability before intervention.
- Session aggregates contain no event order, so this is not true early-session prediction.
- XGBoost settings were tuned separately for the PageValues ablation.
- R is assumed, scores are uncalibrated, and no causal intervention effect is estimated.
- Bootstrap intervals cover sampling variation in the held-out test set only.
- The original notebook run did not complete SMOTE, calibration, learning curves, segmentation, or app validation. A supplemental local pipeline now covers these items; results are reported separately in Section 9.
- The downloaded notebook has not been freshly run end-to-end in this workspace.

## Conclusion

PageValues accounts for a large share of apparent predictive performance, while the no-PageValues model retains useful but moderate ranking skill over prevalence and single-feature baselines. Any business value depends on a cost ratio that must be measured. A stronger next study would use event-level data, features from the first browsing actions, and a time-based test split.

## 9. Supplemental robustness and deliverable

To complete the PRD items beyond the recorded Colab run, a local pipeline was run on the deduplicated UCI data. These cross-validation results use a fixed stratified 80% training partition and do not evaluate the historical held-out test labels. The app model is subsequently fit on all 12,205 rows for demonstration.

| Fixed-settings CV model | Mean PR-AUC with PageValues | Mean PR-AUC without PageValues |
|---|---:|---:|
| Logistic regression | 0.664 | 0.330 |
| Random forest | 0.737 | 0.384 |
| XGBoost | 0.749 | 0.379 |

The same settings were used for each paired feature set, including the same tree count for XGBoost. Removing PageValues lowered PR-AUC in all three models. The fixed no-PageValues XGBoost mean (0.379) is close to the original selected model's training CV (0.382), but this is a robustness comparison rather than a replacement estimate.

The no-PageValues XGBoost learning curve rose from CV PR-AUC 0.326 at 1,952 training rows to 0.380 at 7,811 rows; the last increase was small, suggesting diminishing returns at the observed sample size. SMOTE applied inside each CV fold after one-hot encoding produced mean PR-AUC 0.355, below the class-weighted model's 0.379. Because synthetic samples have fractional one-hot coordinates, treat this as a sensitivity check rather than a preferred approach.

Nested five-fold sigmoid calibration reduced cross-fitted Brier score from 0.185 to 0.115. The deployed demo still shows intent bands rather than probability estimates. K-means on engagement-only standardized features selected three groups by silhouette score (0.434): a high-engagement group (n=2,307; 24.0% conversion), a middle group (n=9,013; 15.0%), and a low-engagement group (n=885; 0.5%). `PageValues` and `Revenue` were excluded from clustering inputs; Revenue was used only to describe the resulting segments.

The Streamlit app supports single-session and CSV scoring, displays High/Medium/Low intent bands, and offers cost-ratio scenarios. Five-fold cross-fitted reference conversion rates were 33.2%, 14.8%, and 3.1% for the three score bands. The app launch and scoring paths were smoke-checked locally. These rates are descriptive cross-fitted references, not a separate held-out estimate. The app is a completed-session demonstration, not an early-session production service.

Reproduce the supplemental analysis and app from the project root with `python -m src.train_project` and `streamlit run app/app.py`. Detailed fold values and artifacts are in `reports/supplemental_results.json` and `reports/figures/`.

## References

Sakar, C. and Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Dataset documentation: https://archive.ics.uci.edu/dataset/468/online

Supplemental calculations were run locally with the Python dependencies listed in `requirements.txt`; see the README for reproduction instructions and caveats.
