# 🛒 Walmart Weekly Sales Forecasting

> - **Goal:** forecast the weekly sales of 3,169 store/department pairs across 45 Walmart stores, up to 39 weeks ahead ([Kaggle competition](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)).
> - **Result:** **WMAE 1,818** on a validation year that includes Thanksgiving and Christmas (baseline XGBoost: 2,752 on an easier validation).
> - **Kaggle:** private WMAE **2,885**, public **2,769** (≈ rank 78 of 689 on the public leaderboard, top 11%; late submission, not officially ranked). The gap with our validation most likely comes from test conditions it never covered: Christmas 2012 falls on a Tuesday (a weekday pattern never seen in training) and a third of its rows have missing CPI/unemployment, whereas the calendar period itself explains little (our validation restricted to the same Nov → Jul span only goes from 1,818 to 1,889).
> - **Model:** 50% LightGBM with 52-week lag features + 50% "same week last year" forecast denoised across stores (SVD).
> - **Key ideas:** a validation split that contains the holiday peaks, lags matched to the forecast horizon, and an ensemble of a tree model with a simple seasonal forecast.
> - **Explore:** [`Notebook.ipynb`](Notebook.ipynb) (step by step story), [`Notebook_Experiments.ipynb`](Notebook_Experiments.ipynb) (what was tested), Streamlit app: `streamlit run app.py`.

---

## 🏆 Final Model: LightGBM + seasonal naive ensemble

**Prediction = 50% × LightGBM + lag features + 50% × seasonal naive forecast** ("same week last year", on a history denoised by SVD across stores, rank 10).

Validation: Nov 2011 → Oct 2012 (52 weeks, **with Thanksgiving & Christmas**):

| Model | WMAE | MAE | R² | MAE Thanksgiving & Christmas |
| --- | --- | --- | --- | --- |
| LightGBM + lags | 2,068 | 1,960 | 0.9637 | 2,988 |
| Seasonal naive (SVD rank 10) | 1,887 | 1,813 | 0.9703 | 2,500 |
| **Ensemble 50/50** | **1,818** | **1,764** | **0.9722** | **2,285** |

* The two models make different errors: LightGBM over-estimates the pre-Christmas weeks, the seasonal naive forecast follows last year's pattern closely. Their average beats both.
* The 50/50 weight is fixed in advance, not tuned on the validation set.
* Validated on the holiday-aware split, then **retrained on all data (2010 → 2012)** for the app and the submission.
* LightGBM features = base features + `Sales_Lag_52`, `Sales_Lag_52_Mean4`, `Store_Dept_Mean`, `Store_Dept_Median`. All the logic lives in `walmart_features.py`, shared by the notebook and the app.
* LightGBM trains on CPU in ~10 seconds, and the SVD takes ~1 second.

```python
LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=63,
              subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42)
```
---

## 🚀 Streamlit App

An interactive web app is included to explore the data and make predictions.

**Features:**

* **📊 EDA tab**, interactive Plotly charts: weekly sales trends, sales by store type, year-over-year comparison, holiday impact, and markdown vs sales
* **📈 Model Performance tab**, WMAE / MAE / R² on the holiday-aware validation year, actual vs predicted sales for the whole chain and for any store/department, feature importance
* **🔮 Predict tab**, pick a store, a department and a week (past or future): store info, external factors and last year's sales are filled in automatically, then the ensemble predicts the weekly sales (with the LightGBM and seasonal naive parts shown)

**How to run:**

```bash
# Python 3.11+
pip install -r requirements.txt

# run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 💡 Key Takeaways

* **Linear models failed completely** (R² ~0.09), the data is highly non-linear due to holiday spikes and complex store interactions
* **AdaBoost collapsed** (R² = -0.47) due to its sensitivity to extreme sales outliers during holidays
* **Tree-based models dominate**, a single Decision Tree jumped to R²=0.90, and ensembles pushed it further
* **Feature engineering beat model choice**: adding the store/department sales level and 52-week lags cut LightGBM's WMAE by 41%, far more than any hyper-parameter or algorithm change
* **A simple baseline can beat a complex model**: "same week last year" (denoised across stores) beats LightGBM on its own, and averaging the two gives the best result
* **Engineered holiday features had mixed results**: `Days_to_Thanksgiving` is useful, `Days_to_SuperBowl` and `Days_to_LaborDay` were ignored by the model
* **The metric changes the winner**: among baseline models, XGBoost is best on R², Random Forest on the official WMAE
* **The validation window matters**: a validation period without Thanksgiving and Christmas gave optimistic scores
* **Temporal train/val split** is critical for time series data, random splits would leak future data and inflate scores

---

## 🧪 Experiments (`Notebook_Experiments.ipynb`)

Three improvements tested separately before being adopted in the final model:

**1. Holiday-aware validation.** New split: train up to 2011-10-28, validate on the next 52 weeks (2011-11-04 → 2012-10-26), which **includes Thanksgiving and Christmas 2011**. Every model's WMAE gets worse (+6% to +18%): the original scores were optimistic.

**2. Lag features** (only lags ≥ 52 weeks, since the Kaggle test set goes up to 39 weeks ahead): sales of the same store/department 52 weeks earlier, their 4-week mean, and the store/department historical mean and median.

| Model (new split) | WMAE base | WMAE + lags | Gain |
| --- | --- | --- | --- |
| **LightGBM** | 3,501 | **2,070** | **-41%** |
| Random Forest | 2,779 | 2,293 | -17% |
| XGBoost | 2,919 | 2,645 | -9% |

**LightGBM + lags is the best model of the project** (WMAE 2,070, R² 0.9637), and the only one that also improves on Thanksgiving/Christmas weeks (MAE 4,605 → 2,989). Most of the gain comes from the store/department historical level.

**Out-of-fold target encoding.** `Store_Dept_Mean`/`Median` computed in-sample let each training row see its own sales. With out-of-fold encoding (5 folds of weeks), **XGBoost's WMAE drops from 2,645 to 2,061 and its R² recovers from 0.812 to 0.965**: it was over-fitting the leak. LightGBM (+1%) and Random Forest (+3%) are not affected, so the deployed LightGBM is kept as is.

**Ideas from the competition winner** (David Thaler, [code](https://github.com/davidthaler/Walmart_competition_code)), tested one by one against LightGBM + lags (WMAE 2,068):

| Idea | WMAE | Verdict |
| --- | --- | --- |
| Drop external features (CPI, MarkDowns, …) | 2,034 (-1.6%) | small gain, not kept on top of the ensemble |
| Christmas day-of-week shift | 2,098 to 2,222 | ❌ LightGBM already captures it |
| SVD-denoised lag features | 2,071 to 2,079 | ➖ neutral for LightGBM |
| **Average with an SVD seasonal naive forecast** | **1,822 (-12%)** | ✅ **adopted** |

**3. Retraining on all data (2010 → 2012)** before predicting the test set. The resulting forecast matches the level of the same season one year earlier, while the original model, which never saw 2012, sits ~5% below it.

---

# 📚 Details

## 📁 Dataset

Data sourced from the [Kaggle Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) competition.

| File                     | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| `train.csv`            | Historical weekly sales per store/department                              |
| `test.csv`             | Stores/departments to predict                                             |
| `features.csv`         | External features (temperature, fuel price, markdowns, CPI, unemployment) |
| `stores.csv`           | Store type (A, B, C) and size                                             |
| `sampleSubmission.csv` | Submission format                                                         |

**Final merged training dataset:** 421,570 rows × 25 columns

---

## 📂 Project Structure

```
Walmart_Weekly_Sales_Forecast/
│
├── Data/
│   ├── train.csv
│   ├── test.csv
│   ├── features.csv
│   ├── stores.csv
│   └── sampleSubmission.csv
│
├── Notebook.ipynb          # EDA, feature engineering, modeling, evaluation, submission
├── Notebook_Experiments.ipynb  # holiday-aware validation, lag features, retraining on all data
├── walmart_features.py     # feature engineering shared by the notebook and the app
├── app.py                  # Streamlit app (EDA, model performance, prediction)
├── walmart_lgbm_model.pkl  # LightGBM component of the final ensemble, trained on all data (2.9 MB)
├── model_validation.csv.gz # validation predictions shown in the app
├── requirements.txt
└── README.md
```

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

* **Sales peak** around weeks 7, 14, 22, 27, 40, 47 and 51, driven by holidays and promotions
* **Store Type A** dominates sales volume, followed by B and C
* **Type C stores** show stable, linear sales year-round while A and B stores show sharp seasonal spikes
* **Holiday weeks** generate ~7.13% higher average sales than non-holiday weeks despite being only 10 out of 143 weeks
* **Markdowns correlate strongly** with sales spikes, promotions visibly drive purchases
* **Store size** (0.24) is the strongest single correlator with weekly sales
* Yearly totals look like a big drop in 2012, but this is a **data coverage effect**: the data runs from 2010-02-05 to 2012-10-26, so 2012 is missing its November–December peak. On the **39 weeks present in all three years**, sales are actually **stable** (1,797M$ → 1,786M$ → 1,831M$)

---

## ⚙️ Feature Engineering

| Feature                  | Description                   |
| ------------------------ | ----------------------------- |
| `Year`                 | Extracted from `Date`       |
| `Week_Of_Year`         | Week number (1–52)           |
| `Days_to_Thanksgiving` | Days until/since Thanksgiving |
| `Days_to_Christmas`    | Days until/since Christmas    |
| `Days_to_SuperBowl`    | Days until/since Super Bowl   |
| `Days_to_LaborDay`     | Days until/since Labor Day    |

Dropped columns: `Date`, `Day`, `Month`, `Week_Of_Month` (redundant after feature extraction)

---

## 🧹 Preprocessing Pipeline

```python
# Ordinal Encoding for Store Type (C=0 < B=1 < A=2, ordered by sales volume)
OrdinalEncoder(categories=[["C", "B", "A"]])

# Standard Scaling for all numerical features
StandardScaler()

# Combined via ColumnTransformer + Pipeline
ColumnTransformer([
    ("Categorical", cat_pipeline, cat_data),
    ("Numerical",   num_pipeline, num_data)
], remainder="drop")
```

---

## 📊 Train / Validation Split

Temporal split to avoid data leakage, train on the past, validate on the future:

| Set        | Years      | Rows    |
| ---------- | ---------- | ------- |
| Train      | 2010–2011 | 294,132 |
| Validation | 2012       | 127,438 |

---

## 🤖 Models & Results

### Official Kaggle metric: WMAE

The competition is scored with the **Weighted Mean Absolute Error**, where holiday weeks weigh **5×** more than normal weeks:

$$WMAE = \frac{\sum_i w_i \, |y_i - \hat{y}_i|}{\sum_i w_i}, \qquad w_i = 5 \text{ if holiday week, else } 1$$

| Model                    | **WMAE**        | MAE       | RMSE      | Val R²     | Train R²   | Gap        |
| ------------------------ | --------------- | --------- | --------- | ---------- | ---------- | ---------- |
| **Random Forest (tuned)** | **2,367** 🥇   | 2,333     | 5,149     | 0.9458     | 0.9728     | 0.0270     |
| **XGBoost (tuned)**      | 2,752           | 2,732     | **4,829** | **0.9523** 🥇 | 0.9759  | 0.0236     |
| LightGBM                 | 3,146           | 3,119     | 5,320     | 0.9422     | 0.9553     | 0.0131     |
| Decision Tree (tuned)    | 4,664           | 4,624     | 8,132     | 0.8648     | 0.8656     | 0.0008     |
| SGD Regressor            | 14,155          | 14,145    | 21,326    | 0.0705     |,          |,          |
| Linear Regression        | 14,374          | 14,299    | 21,118    | 0.0886     |,          |,          |
| AdaBoost                 | 22,599          | 22,727    | 26,874    | -0.4761    | -0.6408    |,          |

*Default (untuned) Decision Tree and Random Forest reached R² of 0.9078 and 0.9488 but overfit (train R² of 1.000 and 0.996).*

**The ranking depends on the metric.** XGBoost wins on R² and RMSE, which square the errors and reward it for handling the few very large sales values better. **Random Forest wins on WMAE and MAE**, the metrics that reflect the error on a typical week and the one Kaggle actually uses.

> ⚠️ **Validation caveat:** the 2012 validation set ends on 2012-10-26, so it contains **no Thanksgiving and no Christmas week**. Its only holiday weeks are the Super Bowl and Labor Day. These scores are therefore optimistic compared to the Kaggle test set (Nov 2012 → Jul 2013), which includes both peaks.

---

## ⚖️ Baseline Model Choice

**Baseline stage (features above, 2012 validation):** no model wins everywhere.

| | Random Forest (tuned) | XGBoost (tuned) |
| --- | --- | --- |
| WMAE (Kaggle metric) | **2,367** | 2,752 (+16%) |
| R² / RMSE | 0.9458 / 5,149 | **0.9523 / 4,829** |
| Saved model size | 182 MB (64 MB compressed) | **3.9 MB** |

Random Forest wins on the scored metric but is too heavy to deploy comfortably (GitHub's 100 MB file limit, slow loading on Streamlit Cloud). Neither is the final model: the experiments below show that **LightGBM + lag features**, averaged with a seasonal naive forecast, beats both.

---

## 📈 Predictions vs Reality & Explainability (baseline XGBoost)

On 2012 (never seen in training), the baseline XGBoost's **total weekly sales** across all stores are off by **2.59%** on average. The notebook plots predicted vs actual sales for the whole chain and for individual store/department pairs.

**Feature importance + SHAP** show that:

* **Dept, Size, Store and Type** drive most predictions: *what* is sold and *where* matters far more than anything else.
* **CPI** has a negative effect, but CPI is regional and almost constant per store, so the model most likely uses it as a **store location proxy**, not as an economic driver.
* Seasonality comes from `Week_Of_Year` and `Days_to_Thanksgiving`. `Days_to_SuperBowl` and `Days_to_LaborDay` have **zero importance** and could be dropped.
* Markdowns, Temperature and Fuel_Price have a small impact.

---

## 📤 Kaggle Submission

The last section of `Notebook.ipynb` generates `walmart_submission.csv` with the final model in the format expected by Kaggle (`Id,Weekly_Sales`, 115,064 rows), with a sanity check against `sampleSubmission.csv`.

```bash
kaggle competitions submit -c walmart-recruiting-store-sales-forecasting -f walmart_submission.csv -m "LightGBM + seasonal naive ensemble"
```

| Submission | Public WMAE | Private WMAE |
| --- | --- | --- |
| LightGBM + lag features | 2,929 | 3,070 |
| **Final ensemble (LightGBM + SVD seasonal naive)** | **2,769** | **2,885** |

The ensemble improves the real test score by 6%, confirming the direction seen on the local validation (-12%).

---

## 🛠️ Tech Stack

| Library              | Usage                            |
| -------------------- | -------------------------------- |
| `pandas`,`numpy` | Data manipulation                |
| `scikit-learn`     | Preprocessing, pipelines, models |
| `lightgbm`         | Final model (ensemble component) |
| `xgboost`          | Baseline model                   |
| `shap`             | Model explainability             |
| `plotly`           | Interactive EDA visualizations   |
| `streamlit`        | Web app deployment               |
| `joblib`           | Model serialization              |
| CUDA (RTX 5070)      | GPU acceleration for model comparison |
