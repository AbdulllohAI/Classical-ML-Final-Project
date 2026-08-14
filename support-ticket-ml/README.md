<div align="center">

# Support Ticket Priority — Classification

### End-to-End Multiclass ML Pipeline on a 50K Support Ticket Dataset

EDA → Preprocessing → Feature Engineering → Feature Selection → Classical ML → Ensemble → Gradient Boosting → Tuning → Feature Importance → Evaluation → Business Analysis

</div>

---

## Project Description

A company's support center receives thousands of tickets every day. This project builds a **multiclass classification** model that predicts a ticket's priority — **Low / Medium / High** — from ticket metadata available at the moment the ticket is opened (no future/post-resolution information is used).

## Dataset Source

[Support Ticket Priority Dataset (50K) — Kaggle](https://www.kaggle.com/datasets/albertobircoci/support-ticket-priority-dataset-50k) (`albertobircoci/support-ticket-priority-dataset-50k`), downloaded via `kagglehub`.

- **50,000 rows**, 33 raw columns
- Target: `priority` ∈ {`low`, `medium`, `high`}
- Class balance: Low 50.0% / Medium 35.0% / High 15.0% — **imbalanced**

## Problem Statement

Given ticket metadata (company profile, incident metrics, reporter/channel info, sentiment, description length), predict the priority level a support ticket should be assigned, so that high-impact incidents can be triaged and escalated correctly.

## EDA Findings

Full analysis in [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb).

1. **Imbalanced target**: Low ≈ 50%, Medium ≈ 35%, High ≈ 15% — the smallest class is also the most business-critical one, so accuracy alone is a poor model-selection metric.
2. **Minimal missing data**: only `customer_sentiment` has missing values (906 rows, ≈1.81%); no other column has any.
3. **No duplicate rows** (with or without `ticket_id`).
4. **Severity-related features correlate most strongly with priority**: `customers_affected`, `downtime_min`, `error_rate_pct`, `security_incident_flag`, `payment_impact_flag`, `data_loss_flag`.
5. **Every raw categorical column has a pre-encoded `_cat` twin** (e.g. `industry` / `industry_cat`), and critically **`priority_cat` is a 1:1 numeric encoding of the target itself** — a label-leakage trap that must be dropped, not fed to any model.
6. Outliers in `customers_affected`, `downtime_min`, `past_90d_incidents` are **real severe incidents**, not data errors — removing them would strip the model of its strongest `High`-priority signal.

## Data Preprocessing

Full pipeline in [`notebooks/02_Preprocessing.ipynb`](notebooks/02_Preprocessing.ipynb).

| Step | Decision | Why |
|---|---|---|
| Missing values | `customer_sentiment` → filled with `"Unknown"` | Missing is rare (<2%) and dropping the column loses signal; treating absence as its own category is safer than imputing a guessed sentiment |
| Duplicates | None found — no action needed | Dataset is clean |
| Outliers | Kept as-is | Represent real severe incidents, strongly predictive of `High` priority |
| Label leakage | `priority_cat` dropped | Direct numeric encoding of the target |
| Redundant columns | 9 pre-encoded `_cat` columns dropped | Duplicate their categorical originals, kept the interpretable string version and One-Hot encoded it instead |
| ID columns | `ticket_id`, `company_id` dropped | Pure identifiers / high-cardinality IDs that generalize poorly |
| Encoding | One-Hot Encoding on 9 categorical columns | Consistent input format across all classical and boosting models |
| Scaling | `StandardScaler`, applied only for KNN and SVM | Distance-based models need it; tree-based models don't |
| Split | 80% train / 20% test, **stratified** on `priority` | Preserves class ratios across train/test for imbalanced multiclass data |

## Feature Engineering

Six new features were engineered from the raw ticket data:

| Feature | What it does | Why it helps |
|---|---|---|
| `severity_flag_score` | Sum of payment/security/data-loss flags + missing-runbook flag | Aggregates scattered severity signals into one score |
| `impact_per_user` | `customers_affected / org_users` | Normalizes impact by company size — a fairer severity signal than raw counts |
| `incident_to_ticket_ratio` | `past_90d_incidents / (past_30d_tickets + 1)` | Separates "noisy but minor" accounts from "few but serious" ones |
| `downtime_per_customer` | `downtime_min / (customers_affected + 1)` | Captures per-customer severity of an outage |
| `is_weekend` | Ticket opened on Sat/Sun | Weekend support capacity is typically lower |
| `is_long_description` | Description length above the median | Longer descriptions often signal more complex/severe issues |

## Models

Trained and evaluated (Accuracy, Precision, Recall, F1-macro, ROC-AUC OVR) in [`notebooks/03_Classical_Models.ipynb`](notebooks/03_Classical_Models.ipynb) and [`notebooks/04_Boosting_Tuning.ipynb`](notebooks/04_Boosting_Tuning.ipynb):

- Logistic Regression (baseline)
- KNN (k = 3, 5, 7, 9, 11 — scaled vs. unscaled)
- SVM (linear & RBF kernels, C = 0.1 / 1 / 10, trained on a 6K stratified subsample for tractability)
- Decision Tree (default, and max_depth = 3 / 5 / 10)
- Random Forest (4 configs varying `n_estimators`, `max_depth`, `max_features`)
- XGBoost, LightGBM, CatBoost (default and manually-tuned configs)

## Hyperparameter Tuning

`RandomizedSearchCV` (`cv=3`, `scoring='f1_macro'`, `n_iter=10`) on a 15K-row stratified subsample, then refit on the full training set:

| Model | Search space | Best params | Best CV F1-macro |
|---|---|---|---|
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` | `subsample=0.7, n_estimators=400, max_depth≈6, ...` | 0.9495 |
| LightGBM | `n_estimators`, `learning_rate`, `num_leaves`, `max_depth` | `num_leaves=15, n_estimators=300, ...` | 0.9465 |
| CatBoost | `iterations`, `depth`, `learning_rate`, `l2_leaf_reg` | `learning_rate=0.05, l2_leaf_reg=3.0, ...` | 0.9545 |

## Model Comparison

Full held-out test set results (sorted by F1-macro):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **CatBoost** | **0.9758** | **0.9762** | **0.9685** | **0.9722** |
| LightGBM | 0.9727 | 0.9709 | 0.9639 | 0.9673 |
| XGBoost (tuned) | 0.9720 | 0.9708 | 0.9626 | 0.9665 |
| LightGBM (tuned) | 0.9705 | 0.9699 | 0.9595 | 0.9644 |
| CatBoost (tuned) | 0.9688 | 0.9708 | 0.9583 | 0.9642 |
| XGBoost | 0.9691 | 0.9687 | 0.9575 | 0.9628 |
| Random Forest | 0.9492 | 0.9456 | 0.9366 | 0.9409 |
| Decision Tree (depth=10) | 0.9146 | 0.9124 | 0.8923 | 0.9015 |
| SVM (linear, C=10) | 0.9038 | 0.8945 | 0.8848 | 0.8894 |
| Logistic Regression | 0.8423 | 0.8279 | 0.7968 | 0.8097 |
| KNN (k=9, scaled) | 0.7244 | 0.7165 | 0.6667 | 0.6844 |

**"Highest accuracy" alone is not the answer** — F1-macro (which weights all three classes equally, including the rare-but-critical `High` class), Precision, and Recall are all considered together, and boosting models lead on every metric simultaneously.

## Feature Importance

Compared across Random Forest, XGBoost, LightGBM, and CatBoost ([`images/feature_importance_rf.png`](images/feature_importance_rf.png), [`_xgb`](images/feature_importance_xgb.png), [`_lgbm`](images/feature_importance_lgbm.png), [`_catboost`](images/feature_importance_catboost.png)):

- **Important across all three boosting models**: `error_rate_pct`, `customers_affected`, `downtime_min`, `downtime_per_customer`, `reported_by_role_c_level`, `customer_tier_Enterprise`
- **XGBoost-specific**: company-size dummies, `org_users`
- **LightGBM-specific**: `description_length`
- **CatBoost-specific**: `product_area_auth`

This confirms the EDA findings: incident-severity metrics dominate, and the engineered `downtime_per_customer` feature ranks among the top signals across models.

## Overfitting Analysis

| Model | Train F1 | Test F1 | Gap |
|---|---|---|---|
| KNN (k=9) | 0.7571 | 0.6844 | 0.0727 |
| Random Forest | 1.0000 | 0.9409 | 0.0591 |
| Decision Tree (depth=10) | 0.9459 | 0.9015 | 0.0444 |
| XGBoost | 0.9982 | 0.9628 | 0.0354 |
| XGBoost (tuned) | 1.0000 | 0.9665 | 0.0335 |
| LightGBM | 0.9996 | 0.9673 | 0.0323 |
| LightGBM (tuned) | 0.9867 | 0.9644 | 0.0223 |
| Logistic Regression | 0.8221 | 0.8097 | 0.0123 |
| SVM (linear, C=10) | 0.8992 | 0.8894 | 0.0098 |
| CatBoost | 0.9796 | 0.9722 | 0.0074 |
| **CatBoost (tuned)** | 0.9687 | 0.9642 | **0.0046** |

Random Forest and untuned/deep models overfit the most (train F1 near-perfect, larger train/test gap); **CatBoost is the most stable**, with the smallest train/test gap while still ranking at or near the top on raw test performance.

## Final Model

**CatBoost** is selected for production:

- Highest F1-macro, Precision, and Recall of all 11 evaluated models/configurations.
- Smallest overfitting gap among boosting models — most reliable generalization.
- Native handling of categorical data (used here in One-Hot form for a fair head-to-head comparison, but this gives CatBoost a natural upgrade path if raw categorical columns are fed directly in production).

## Final Conclusion

- **Business framing**: misclassifying a `High` ticket as `Low` (missed real incidents) is far costlier than the reverse (over-escalating a minor ticket) — so **Recall on the `High` class** matters as much as overall F1 when picking a production threshold/model.
- Before deployment: validate on more recent (out-of-time) tickets, check performance per segment (industry/region/company size) for fairness, run in shadow mode against the current process, calibrate `predict_proba` outputs, and set up data-drift monitoring with human review for low-confidence predictions.

---

## Repository Structure

```
support-ticket-ml/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Classical_Models.ipynb
│   └── 04_Boosting_Tuning.ipynb
│
├── images/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance_rf.png
│   ├── feature_importance_xgb.png
│   ├── feature_importance_lgbm.png
│   ├── feature_importance_catboost.png
│   └── ... (histograms, boxplots, confusion matrix, model comparison, etc.)
│
└── data/
    ├── Support_tickets.csv
    └── processed/            # generated by 02_Preprocessing.ipynb (train/test splits)
```

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebooks in order (each writes outputs consumed by the next)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_Preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_Classical_Models.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_Boosting_Tuning.ipynb
```

Notebooks read data via relative paths (`../data/...`) and must be run from inside `notebooks/`, in order — `02` generates the processed train/test files that `03` and `04` depend on, and `03` pickles classical-model results that `04` loads for the final comparison table.

## Tech Stack

| Category | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Classical ML | `scikit-learn` |
| Gradient Boosting | `xgboost`, `lightgbm`, `catboost` |
| Environment | `jupyter` |
| Data acquisition | `kagglehub` |
