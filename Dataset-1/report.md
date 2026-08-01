# Dataset 1 — Credit Card Customers (Customer Churn)

## Loyiha tavsifi
Ushbu report Kaggle "Credit Card Customers" dataseti asosida to'liq Machine Learning pipeline (EDA → preprocessing → feature engineering → feature importance → model training → hyperparameter tuning → model comparison) natijalarini umumlashtiradi. Batafsil tahlil `notebook.ipynb` faylida, barcha grafiklar `images/` papkasida.

## Dataset haqida qisqacha ma'lumot
- **Manba:** Kaggle — `sakshigoyal7/credit-card-customers`
- **Hajmi:** 10,127 qator × 23 ustun (xom holatda)
- **Target:** `Attrition_Flag` (Attrited Customer = 1 / churn, Existing Customer = 0)
- **Olib tashlangan ustunlar:** `CLIENTNUM` (unique ID), 2 ta `Naive_Bayes_Classifier_...` yordamchi ustun → yakuniy 20 ustun
- **Missing values:** 0 ta klassik NaN, lekin `Education_Level`/`Marital_Status`/`Income_Category`da "Unknown" placeholder (7–15%)
- **Duplicates:** 0 ta
- **Target balance:** Imbalanced — 83.93% / 16.07% (~5.2:1)

## Ishlatilgan kutubxonalar
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (RandomForest, train_test_split, RandomizedSearchCV, metrics), `xgboost`, `lightgbm`, `catboost`

## Feature Engineering
- `Trans_Amt_per_Trans` = Total_Trans_Amt / Total_Trans_Ct
- `Inactivity_Ratio` = Months_Inactive_12_mon / Months_on_book

Encoding'dan keyin jami 34 ta feature (One-Hot + engineered).

## Feature Importance (Random Forest, top-5)
1. `Total_Trans_Amt` — 0.154
2. `Total_Trans_Ct` — 0.153
3. `Total_Ct_Chng_Q4_Q1` — 0.106
4. `Total_Revolving_Bal` — 0.095
5. `Trans_Amt_per_Trans` (engineered) — 0.068

Eng foydasiz: `Card_Category` (Silver/Gold/Platinum) dummy ustunlari.

## Model Comparison (test set, tuned)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (s) |
|---|---|---|---|---|---|---|
| XGBoost | 0.9719 | 0.9497 | 0.8708 | 0.9085 | 0.9924 | 1.06 |
| LightGBM | 0.9704 | 0.9316 | 0.8800 | 0.9051 | 0.9923 | 0.38 |
| CatBoost | 0.9719 | 0.9497 | 0.8708 | 0.9085 | **0.9931** | 2.24 |

*(Baseline vs tuned to'liq jadval notebook'da.)*

## Eng yaxshi model
**CatBoost** (production tanlovi) — eng yuqori ROC-AUC (0.9931), XGBoost bilan teng F1 (0.9085), va inference vaqti juda past (~0.003s/qator). LightGBM esa eng tez o'qitiladigan model (0.38s) bo'lib, resurs cheklangan sharoitlar uchun yaxshi muqobil.

## Asosiy xulosalar
1. Target kuchli imbalanced (16% churn) — faqat accuracy emas, F1/ROC-AUC asosiy metrika sifatida ishlatildi.
2. Churn'ning eng kuchli signali — **tranzaksion faollik** (Total_Trans_Ct/Amt), demografik xususiyatlar (yosh, jins) emas.
3. `Credit_Limit` va `Avg_Open_To_Buy` deyarli bir xil ma'lumot tashiydi (r=0.996) — multicollinearity mavjud, lekin tree-based modellarga ta'sir qilmaydi.
4. Barcha 3 model (tuning'dan keyin) juda yaqin natija ko'rsatdi — bu dataset "oson" classification muammosi ekanligini bildiradi (ROC-AUC > 0.99).
5. Hyperparameter tuning natijalarni sezilarli yaxshilamadi (F1 +0.01 atrofida) — baseline sozlamalar allaqachon yaqin-optimal edi.

## Notebooklarni ishga tushirish bo'yicha ko'rsatmalar
```bash
cd Dataset-1
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
Yoki to'liq qayta ishga tushirish uchun: `jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`
