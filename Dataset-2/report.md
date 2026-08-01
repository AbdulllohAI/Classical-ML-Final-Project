# Dataset 2 — Adult Census Income

## Loyiha tavsifi
Ushbu report Kaggle "Adult Census Income" dataseti asosida to'liq Machine Learning pipeline (EDA → preprocessing → feature engineering → feature importance → model training → hyperparameter tuning → model comparison) natijalarini umumlashtiradi. Batafsil tahlil `notebook.ipynb` faylida, barcha grafiklar `images/` papkasida.

## Dataset haqida qisqacha ma'lumot
- **Manba:** Kaggle — `uciml/adult-census-income`
- **Hajmi:** 32,561 qator × 15 ustun (xom holatda)
- **Target:** `income` (`>50K` = 1, `<=50K` = 0)
- **Olib tashlangan ustunlar:** `fnlwgt` (sample weight) → yakuniy 14 ustun
- **Missing values:** 4262 ta (`workclass`, `occupation`, `native.country`da `?` sifatida), mode bilan to'ldirildi
- **Duplicates:** 3591 ta (11.03%) — o'chirildi
- **Target balance:** Imbalanced — 75.22% / 24.78% (~3:1)

## Ishlatilgan kutubxonalar
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (RandomForest, LabelEncoder, train_test_split, RandomizedSearchCV, metrics), `xgboost`, `lightgbm`, `catboost`

## Feature Engineering
- `capital_net` = capital.gain − capital.loss
- `age_group` = yosh 6 ta guruhga bo'lingan (binned, <25...65+)

Encoding'dan keyin jami 102 ta feature (yuqori kardinallikdagi `native.country` (41 unique) One-Hot Encoding tufayli).

## Feature Importance (Random Forest, top-5)
1. `age` — 0.164
2. `hours.per.week` — 0.116
3. `capital_net` (engineered) — 0.094
4. `marital.status_Married-civ-spouse` — 0.081
5. `capital.gain` — 0.068

Eng foydasiz: kam uchraydigan `native.country` dummy ustunlari (importance ≈ 0).

## Model Comparison (test set, tuned)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (s) |
|---|---|---|---|---|---|---|
| XGBoost | 0.8693 | 0.7822 | 0.6553 | 0.7131 | 0.9292 | 1.91 |
| LightGBM | 0.8697 | 0.7830 | 0.6560 | **0.7139** | 0.9279 | 2.70 |
| CatBoost | 0.8688 | 0.7845 | 0.6490 | 0.7104 | 0.9285 | 5.77 |

*(Baseline vs tuned to'liq jadval notebook'da.)*

## Eng yaxshi model
**LightGBM** (production tanlovi) — eng yuqori F1 (0.7139) va Accuracy (0.8697), baseline bosqichida eng tez train vaqti (0.29s). XGBoost eng yuqori ROC-AUC (0.9292) bilan yaqin muqobil.

## Asosiy xulosalar
1. Target imbalanced (24.8% yuqori daromadli) — F1/ROC-AUC asosiy baholash metrikasi sifatida ishlatildi.
2. Gender bo'yicha daromad taqsimoti keskin nomutanosib (>50K guruhining 84%i erkak) — ijtimoiy/tarixiy bias, production'da fairness monitoring tavsiya etiladi.
3. Numerik featurelar orasida kuchli korrelyatsiya yo'q (eng kuchlisi r=0.141) — signal ko'proq non-linear interaction'lardan (masalan yosh+ta'lim+soat birgalikda) keladi, shu sababli tree-based modellar mos.
4. Feature engineering foydali bo'ldi — `capital_net` top-3 muhim feature bo'lib chiqdi.
5. Hyperparameter tuning natijalarni deyarli o'zgartirmadi (F1 ±0.001-0.003) — baseline gradient boosting sozlamalari allaqachon yaqin-optimal edi.

## Notebooklarni ishga tushirish bo'yicha ko'rsatmalar
```bash
cd Dataset-2
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
Yoki to'liq qayta ishga tushirish uchun: `jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`
