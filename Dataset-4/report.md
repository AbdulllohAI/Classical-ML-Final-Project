# Dataset 4 — Crop Yield Prediction (Regression)

## Loyiha tavsifi
Ushbu report Kaggle "Crop Yield Prediction Dataset" asosida to'liq Machine Learning regression pipeline (EDA → Data Cleaning → Feature Engineering → Categorical Encoding → Baseline → Model Training → Cross-validation → Hyperparameter Tuning → Feature Importance → Error Analysis → Agricultural Business Analysis) natijalarini umumlashtiradi. Batafsil tahlil `notebook.ipynb` faylida, barcha grafiklar `images/` papkasida.

Avvalgi 3 ta dataset (1–3) **classification** masalalari edi; bu loyiha esa **regression** — target uzluksiz raqamli qiymat (`hg/ha_yield`).

## Dataset haqida qisqacha ma'lumot
- **Manba:** Kaggle — `patelris/crop-yield-prediction-dataset` (`yield_df.csv`)
- **Xom hajmi:** 28,242 qator × 8 ustun (indeks ustuni bilan)
- **Olib tashlangan ustunlar:** `Unnamed: 0` (indeks dublikati) → 7 ustun
- **Target:** `hg/ha_yield` (hektogram/gektar hosildorlik) — uzluksiz (regression)
- **Missing values:** 0 ta
- **Duplicates:** 2,310 ta topildi va olib tashlandi → yakuniy shape **25,932 × 7**
- **Ekin turlari:** 10 ta (Potatoes, Maize, Wheat, Rice paddy, Soybeans, Sorghum, Sweet potatoes, Cassava, Yams, Plantains and others)
- **Hududlar:** 101 ta davlat

## Ishlatilgan kutubxonalar
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, StandardScaler, train_test_split, RandomizedSearchCV, KFold, TimeSeriesSplit, metrics), `xgboost`, `lightgbm`, `catboost`

## EDA — asosiy topilmalar
- Eng yuqori o'rtacha hosildorlikka ega ekin: **Potatoes** (198,666 hg/ha), eng past: **Soybeans** (16,992 hg/ha) — ~11.7x farq, chunki ildizmevalar birligi tabiiy ravishda katta.
- Eng yuqori o'rtacha hosildorlikka ega hudud: **United Kingdom** (240,956 hg/ha), undan keyin Belgium, Denmark, Netherlands, Ireland.
- Yillar davomida (1990→2013) o'rtacha yield 66,722 dan 90,362 gacha o'sgan — barqaror o'sish tendensiyasi.
- Rainfall (r=-0.004), Temperature (r=-0.110) va Pesticides (r=0.066) bilan yield orasidagi **to'g'ridan-to'g'ri chiziqli korrelyatsiya juda zaif** — bu yield asosan `Item`/`Area` bilan belgilanishini, iqlim omillari esa ekin turi konteksti ichida ma'noli bo'lishini ko'rsatadi.

## Outlier Analysis (IQR)
| Ustun | Outlier % |
|---|---|
| average_rain_fall_mm_per_year | 0.00% |
| pesticides_tonnes | 4.87% |
| avg_temp | 0.06% |
| hg/ha_yield | 7.40% |

**Qaror:** outlierlar olib tashlanmadi — ular haqiqiy yuqori/past hosildorlikka ega boshqa ekin turlari yoki katta agrar davlatlar (AQSh, Hindiston) natijasi, xato emas; tree-based modellar bunga chidamli.

## Feature Engineering (3 ta yangi feature)
1. `Years_From_Start` = `Year - Year.min()` — vaqt trendini raqamli signalga aylantiradi (yillar davomidagi texnologik o'sishni tutish uchun).
2. `Rainfall_Category` (Low/Medium/High, tertile-based `qcut`) — rainfall'ning yield bilan zaif chiziqli, lekin potentsial non-linear/threshold bog'liqligini kategoriyalash orqali ushlaydi.
3. `Temp_Category` (Cold/Moderate/Hot, tertile-based `qcut`) — har bir ekin turi uchun optimal harorat oralig'i borligi sababli, xom haroratni guruhlarga bo'lish daraxt-asosli modellarga chegaralarni topishga yordam beradi.

## Categorical Encoding
`Area`, `Item`, `Rainfall_Category`, `Temp_Category` — **One-Hot Encoding** (`drop_first=True`). Ustun nomlari LightGBM talabiga ko'ra maxsus belgilardan (masalan davlat nomlaridagi apostrof/vergul) tozalandi. Encoding'dan keyin jami **119 ustun** (118 feature + target).

## Train/Test Split
- **Variant A (random, 80/20, `random_state=42`):** 20,745 train / 5,187 test — asosiy pipeline shu variant bilan bajarildi.
- **Variant B (time-based, eski yillar → train, yangi yillar → test):** alohida bo'limda solishtirish uchun ishlatildi (pastga qarang).
- Real hayotdagi hosildorlik bashoratida **Variant B real dunyoga yaqinroq** (kelajakni bilmagan holda bashorat qilish), Variant A esa temporal leakage'ga yo'l qo'yadi, lekin generalizatsiyani Area/Item bo'yicha kengroq baholaydi.

## Baseline Model
O'rtacha (mean) prediction: **MAE 64,251.59, RMSE 85,153.01, R² -0.0004**

## Model Comparison (test set, Variant A, baseline hyperparametrlar)

| Model | MAE | RMSE | R² | Train Time (s) |
|---|---|---|---|---|
| Linear Regression | 30,785.67 | 44,405.78 | 0.7279 | 0.291 |
| Decision Tree | 4,034.43 | 12,463.93 | 0.9786 | 0.262 |
| **Random Forest** | **3,965.12** | **10,507.23** | **0.9848** | 10.848 |
| XGBoost | 8,959.76 | 16,180.14 | 0.9639 | 2.116 |
| LightGBM | 11,273.27 | 19,040.16 | 0.9500 | 0.434 |
| CatBoost | 10,115.27 | 16,530.34 | 0.9623 | 4.944 |

**Eng yaxshi model:** **Random Forest** (R²=0.9848), barcha modellar baseline'dan (R²≈0) sezilarli yaxshi. Random Forest yaxshi ishlashi sababi — yield bilan featurelar orasidagi bog'liqlik kuchli non-linear va categorical (Area/Item) o'zaro ta'sirlarga boy, bagging orqali variance kamayadi; boosting modellar (XGBoost/LightGBM/CatBoost) bu datasetda default sozlamalarda RF'dan bir oz orqada qoldi — ehtimol yuqori kardinallikdagi one-hot ustunlar (Area: ~100 ta) bilan bog'liq.

## Feature Scaling
| | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (unscaled) | 30,785.67 | 44,405.78 | 0.7279 |
| Linear Regression (scaled) | 29,922.85 | 42,683.04 | 0.7486 |

Nazariy jihatdan oddiy (regularizatsiyasiz) Linear Regression scale-invariant bo'lishi kerak; kuzatilgan kichik farq (`R² +0.02`) dizayn matritsasining sonli konditsiyalanishi (pesticides_tonnes kabi 0–367,778 diapazonli ustun bilan 0/1 one-hot ustunlar birga) bilan izohlanadi, tub effekt emas. Daraxt-asosli modellar (Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost) scalingga **mutlaqo sezgir emas** — ular threshold-split'larga asoslanadi, masofaga emas.

## Cross-validation (Random Forest)
- **K-Fold (5-fold, shuffled):** mean R² = **0.9881** (± 0.0021)
- **TimeSeriesSplit (5-fold):** mean R² = **0.9513** (± 0.0076)

TimeSeriesSplit sezilarli pastroq — bu Variant A/B muhokamasini tasdiqlaydi: kelajakdagi yillarga ekstrapolyatsiya qilish tasodifiy aralashtirilgan validatsiyadan qiyinroq.

## Hyperparameter Tuning (Random Forest, `RandomizedSearchCV`, n_iter=15, cv=3, scoring='r2')
**Eng yaxshi parametrlar:** `n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_depth=30`

| | MAE | RMSE | R² |
|---|---|---|---|
| Default (300 daraxt) | 3,965.12 | 10,507.23 | 0.9848 |
| Tuned | 4,023.29 | 10,520.24 | 0.9847 |

Tuning natijasi default sozlamadan amalda farqlanmadi (R² deyarli bir xil) — default Random Forest giperparametrlari bu dataset uchun allaqachon deyarli optimal, chunki dataset nisbatan kichik (26K qator) va daraxt-asosli model uchun standart 100-300 daraxt yetarli signalni tutib oladi.

## Feature Importance (Random Forest, top-10)
1. `Item_Potatoes` — 0.373
2. `pesticides_tonnes` — 0.072
3. `Item_Rice_paddy` — 0.068
4. `Item_Sweet_potatoes` — 0.066
5. `Area_India` — 0.053
6. `avg_temp` — 0.046
7. `average_rain_fall_mm_per_year` — 0.042
8. `Item_Maize` — 0.025
9. `Item_Wheat` — 0.022
10. `Year` — 0.018

**Xulosa:** ekin turi (`Item_*`) yield shkalasini eng ko'p belgilaydigan omil, undan keyin iqlim (`avg_temp`, `rainfall`) va pestitsid ishlatilishi keladi.

## Actual vs Predicted / Error Analysis
- Predictionlar ko'p qismda ideal (y=x) chiziqqa yaqin, lekin **yuqori yield qiymatlarida** model kam bashorat qiladi (underestimate).
- Past yield kvartili (Q1) o'rtacha mutlaq xato: **1,565.71**; yuqori yield kvartili (Q4): **8,482.04** — ~5.4x farq.
- Eng ko'p xato ekin bo'yicha: `Potatoes` (10,332), `Plantains and others` (7,845), `Sweet potatoes` (5,443), `Cassava` (5,342) — barchasi yuqori/o'zgaruvchan yield'ga ega ildizmevalar.
- Eng ko'p xato hudud bo'yicha: `Bahrain` (36,423), `Belgium` (33,979), `Ireland` (20,884) — kam observationga ega yoki g'ayrioddiy yuqori yield'ga ega davlatlar.

**Xulosa:** model kam ma'lumotli yoki ekstremal yuqori yield combinatsiyalarida (kam uchraydigan ekin/hudud) ancha xato qiladi; "tipik" (keng tarqalgan ekin + keng tarqalgan hudud, past-o'rta yield) holatlarda ishonchli.

## Variant A vs Variant B (bonus)
| | MAE | RMSE | R² |
|---|---|---|---|
| Variant A (random split) | 4,023.29 | 10,520.24 | 0.9847 |
| Variant B (time-based split) | 10,184.94 | 19,865.26 | 0.9556 |

Vaqt bo'yicha split sezilarli pastroq natija beradi — bu real production stsenariysida (kelajak yilni bashorat qilish) modelni faqat random split bilan baholash haddan tashqari optimistik bo'lishini tasdiqlaydi.

## Agriculture Business Analysis
1. **Eng yuqori hosildorlikka ega ekinlar:** Potatoes, Cassava, Sweet potatoes, Yams, Plantains and others (ildizmevalar guruhi).
2. **Eng yuqori hosildorlikka ega hududlar:** United Kingdom, Belgium, Denmark, Netherlands, Ireland — intensiv/rivojlangan agrotexnologiyaga ega davlatlar.
3. **Rainfall ta'siri:** to'g'ridan-to'g'ri chiziqli ta'sir zaif (r=-0.004); ekin turi konteksti bilan birga ma'noli.
4. **Temperature ta'siri:** zaif salbiy chiziqli korrelyatsiya (r=-0.110); har bir ekin uchun optimal harorat oralig'i borligi sababli non-linear.
5. **Pesticide va yield:** kuchsiz ijobiy korrelyatsiya (r=0.066) — bu kauzallik emas, ko'proq mamlakat/fermer hajmi bilan bog'liq confound.
6. **Tavsiyalar:** past yield hududlarida yuqori yield hududlaridagi agrotexnologiyani joriy qilish; ekin turi bilan mos iqlim rejalashtirish; pestitsidni xarajat-samaradorlik nuqtai nazaridan optimallashtirish; modelni yangi yillar ma'lumoti bilan muntazam qayta o'qitish (Variant B natijasi buni tasdiqlaydi).

## Asosiy xulosalar
1. Yield asosan **ekin turi** va **hudud** bilan belgilanadi — bu ikkalasi eng yuqori feature importance'ga ega, iqlim/pestitsid omillari ikkinchi darajali.
2. Non-linear bog'liqliklar sababli tree-based modellar (ayniqsa **Random Forest**, R²=0.9848) Linear Regression'dan (R²=0.7279) sezilarli yaxshi ishlaydi.
3. Hyperparameter tuning bu datasetda amalda foyda bermadi — default Random Forest allaqachon deyarli optimal.
4. Model xatolari yuqori yield va kam ma'lumotli ekin/hudud kombinatsiyalarida kattaroq — production'da bu subgruppalar uchun ehtiyot bo'lish tavsiya etiladi.
5. Vaqt bo'yicha baholash (TimeSeriesSplit / Variant B) random split'dan qiyinroq natija beradi — bu real dunyoda kelajakni bashorat qilish tabiiy ravishda qiyinroqligini tasdiqlaydi.

## Notebooklarni ishga tushirish bo'yicha ko'rsatmalar
```bash
cd Dataset-4
pip install -r ../requirements.txt
jupyter notebook notebook.ipynb
```
Yoki to'liq qayta ishga tushirish uchun: `jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`
