# Dataset 3 — Customer Churn Prediction (Bank)

## Loyiha tavsifi
Ushbu report Kaggle "Bank Customer Churn / Churn Modelling" dataseti asosida to'liq Machine Learning pipeline (EDA → preprocessing → feature engineering → feature importance → model training → hyperparameter tuning → model comparison) natijalarini umumlashtiradi. Batafsil tahlil `notebook.ipynb` faylida, barcha grafiklar `images/` papkasida.

## Dataset haqida qisqacha ma'lumot
- **Manba:** Kaggle — `shrutimechlearn/churn-modelling`
- **Hajmi:** 10,000 qator × 14 ustun (xom holatda)
- **Target:** `Exited` (1 = churn, 0 = qolgan)
- **Olib tashlangan ustunlar:** `RowNumber`, `CustomerId`, `Surname` → yakuniy 11 ustun
- **Missing values:** 0 ta
- **Duplicates:** 0 ta
- **Target balance:** Imbalanced — 79.63% / 20.37% (~3.9:1)

## Ishlatilgan kutubxonalar
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (RandomForest, train_test_split, RandomizedSearchCV, metrics), `xgboost`, `lightgbm`, `catboost`

## Feature Engineering
- `Balance_to_Salary` = Balance / EstimatedSalary
- `Age_Tenure_Ratio` = Age / Tenure

Encoding'dan keyin jami 13 ta feature.

## Feature Importance (Random Forest, top-5)
1. `Age` — 0.201
2. `NumOfProducts` — 0.125
3. `CreditScore` — 0.112
4. `EstimatedSalary` — 0.110
5. `Age_Tenure_Ratio` (engineered) — 0.104

Eng foydasiz: `Geography_Spain`, `HasCrCard`, `Gender_Male`.

## Model Comparison (test set, tuned)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (s) |
|---|---|---|---|---|---|---|
| XGBoost | 0.8610 | 0.7529 | 0.4717 | 0.5801 | 0.8460 | 0.18 |
| **LightGBM** | **0.8690** | 0.7843 | **0.4914** | **0.6042** | **0.8661** | 1.15 |
| CatBoost | 0.8665 | 0.7713 | 0.4889 | 0.5985 | 0.8633 | 0.73 |

## Eng yaxshi model
**LightGBM** (production tanlovi) — barcha metrikalarda (Accuracy, F1, ROC-AUC) yetakchi. Biroq barcha modellarda Recall past (~0.47-0.49) — real churn qiluvchi mijozlarning yarmigagina to'g'ri aniqlanadi.

## Asosiy xulosalar
1. Target imbalanced (20.4% churn) — F1/ROC-AUC asosiy baholash metrikasi sifatida ishlatildi.
2. `Age` va `IsActiveMember` — churn'ning eng kuchli individual signallari; faol bo'lmagan mijozlarning churn darajasi (26.9%) faol mijozlarnikidan (14.3%) deyarli 2 barobar yuqori.
3. Germaniya mijozlari boshqa mamlakatlarga (Fransiya, Ispaniya) nisbatan sezilarli yuqori churn nisbatiga ega — geografik segmentatsiya foydali.
4. Barcha modellarda Recall nisbatan past (~0.47-0.49) — bu cheklangan feature soni (13 ta) va class imbalance bilan bog'liq; production'da class_weight yoki SMOTE, shuningdek threshold tuning ko'rib chiqilishi tavsiya etiladi.
5. Hyperparameter tuning LightGBM'ni yaxshiladi, lekin XGBoost'da recall biroz pasaydi — bu RandomizedSearchCV'ning cheklangan (n_iter=15) qidiruv hajmi bilan izohlanishi mumkin.

## Notebooklarni ishga tushirish bo'yicha ko'rsatmalar
```bash
cd Dataset-3
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
Yoki to'liq qayta ishga tushirish uchun: `jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`
