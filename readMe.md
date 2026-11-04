
# 🛒 Walmart Weekly Sales Prediction

A machine learning project to forecast weekly department-level sales across 45 Walmart stores using historical data from 2010–2012.

---

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
├── app.py                  # Streamlit app (EDA, model performance, prediction)
├── walmart_xgb_model.pkl   # trained XGBoost pipeline (preprocessing + model)
├── requirements.txt
└── README.md
```

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

* **Sales peak** around weeks 7, 14, 22, 27, 40, 47 and 51 — driven by holidays and promotions
* **Store Type A** dominates sales volume, followed by B and C
* **Type C stores** show stable, linear sales year-round while A and B stores show sharp seasonal spikes
* **Holiday weeks** generate ~7.13% higher average sales than non-holiday weeks despite being only 10 out of 143 weeks
* **Markdowns correlate strongly** with sales spikes — promotions visibly drive purchases
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

Temporal split to avoid data leakage — train on the past, validate on the future:

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
| SGD Regressor            | 14,155          | 14,145    | 21,326    | 0.0705     | —          | —          |
| Linear Regression        | 14,374          | 14,299    | 21,118    | 0.0886     | —          | —          |
| AdaBoost                 | 22,599          | 22,727    | 26,874    | -0.4761    | -0.6408    | —          |

*Default (untuned) Decision Tree and Random Forest reached R² of 0.9078 and 0.9488 but overfit (train R² of 1.000 and 0.996).*

**The ranking depends on the metric.** XGBoost wins on R² and RMSE, which square the errors and reward it for handling the few very large sales values better. **Random Forest wins on WMAE and MAE**, the metrics that reflect the error on a typical week and the one Kaggle actually uses.

> ⚠️ **Validation caveat:** the 2012 validation set ends on 2012-10-26, so it contains **no Thanksgiving and no Christmas week**. Its only holiday weeks are the Super Bowl and Labor Day. These scores are therefore optimistic compared to the Kaggle test set (Nov 2012 → Jul 2013), which includes both peaks.

---

## 🏆 Model Choice

**No single model wins everywhere, so the choice depends on the use case:**

| | Random Forest (tuned) | XGBoost (tuned) |
| --- | --- | --- |
| WMAE (Kaggle metric) | **2,367** | 2,752 (+16%) |
| R² / RMSE | 0.9458 / 5,149 | **0.9523 / 4,829** |
| Saved model size | 182 MB (64 MB compressed) | **3.9 MB** |
| **Used for** | **Kaggle leaderboard** | **Deployed Streamlit app** |

* **Random Forest** is the right pick for Kaggle: it wins on the metric that is actually scored.
* **XGBoost** is deployed in the app: its model file is ~47× smaller (182 MB would exceed GitHub's 100 MB file limit and load slowly on Streamlit Cloud), and it is the most accurate on the large sales values. The 16% WMAE gap is the accepted trade-off for a lightweight app.

Deployed XGBoost configuration:

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    device="cuda",
    tree_method="hist",
    random_state=42
)
```

Train/val R² gap of 0.0236: the model generalizes well to unseen data.

---

## 📈 Predictions vs Reality & Explainability

On 2012 (never seen in training), XGBoost's **total weekly sales** across all stores are off by **2.59%** on average. The notebook plots predicted vs actual sales for the whole chain and for individual store/department pairs.

**Feature importance + SHAP** show that:

* **Dept, Size, Store and Type** drive most predictions: *what* is sold and *where* matters far more than anything else.
* **CPI** has a negative effect, but CPI is regional and almost constant per store, so the model most likely uses it as a **store location proxy**, not as an economic driver.
* Seasonality comes from `Week_Of_Year` and `Days_to_Thanksgiving`. `Days_to_SuperBowl` and `Days_to_LaborDay` have **zero importance** and could be dropped.
* Markdowns, Temperature and Fuel_Price have a small impact.

---

## 💡 Key Takeaways

* **Linear models failed completely** (R² ~0.09) — the data is highly non-linear due to holiday spikes and complex store interactions
* **AdaBoost collapsed** (R² = -0.47) due to its sensitivity to extreme sales outliers during holidays
* **Tree-based models dominate** — a single Decision Tree jumped to R²=0.90, and ensembles pushed it further
* **XGBoost's built-in regularization** and sequential boosting gave it the edge over all other models
* **Engineered holiday features had mixed results**: `Days_to_Thanksgiving` is useful, `Days_to_SuperBowl` and `Days_to_LaborDay` were ignored by the model
* **The metric changes the winner**: XGBoost is best on R², Random Forest is best on the official WMAE
* **Temporal train/val split** is critical for time series data — random splits would leak future data and inflate scores

---

## 🚀 Streamlit App

An interactive web app is included to explore the data and make predictions.

**Features:**

* **📊 EDA tab** — interactive Plotly charts: weekly sales trends, sales by store type, year-over-year comparison, holiday impact, and markdown vs sales
* **📈 Model Performance tab** — WMAE / MAE / R² on 2012, actual vs predicted sales for the whole chain and for any store/department, feature importance
* **🔮 Predict tab** — input store details and get an instant weekly sales prediction from the XGBoost model

**How to run:**

```bash
# Python 3.11+
pip install -r requirements.txt

# run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🧪 Experiments (`Notebook_Experiments.ipynb`)

Three improvements tested separately, without touching the deployed model:

**1. Holiday-aware validation.** New split: train up to 2011-10-28, validate on the next 52 weeks (2011-11-04 → 2012-10-26), which **includes Thanksgiving and Christmas 2011**. Every model's WMAE gets worse (+6% to +18%): the original scores were optimistic.

**2. Lag features** (only lags ≥ 52 weeks, since the Kaggle test set goes up to 39 weeks ahead): sales of the same store/department 52 weeks earlier, their 4-week mean, and the store/department historical mean and median.

| Model (new split) | WMAE base | WMAE + lags | Gain |
| --- | --- | --- | --- |
| **LightGBM** | 3,501 | **2,070** | **-41%** |
| Random Forest | 2,779 | 2,293 | -17% |
| XGBoost | 2,919 | 2,645 | -9% |

**LightGBM + lags is the best model of the project** (WMAE 2,070, R² 0.9637), and the only one that also improves on Thanksgiving/Christmas weeks (MAE 4,605 → 2,989). Most of the gain comes from the store/department historical level.

**3. Retraining on all data (2010 → 2012)** before predicting the test set. The resulting forecast (`walmart_submission_v2.csv`, not submitted) matches the level of the same season one year earlier, while the original model, which never saw 2012, sits ~5% below it.

---

## 📤 Kaggle Submission

The last section of the notebook generates `walmart_submission.csv` in the format expected by Kaggle (`Id,Weekly_Sales`, 115,064 rows), with a sanity check against `sampleSubmission.csv`.

```bash
kaggle competitions submit -c walmart-recruiting-store-sales-forecasting -f walmart_submission.csv -m "XGBoost tuned"
```

---

## 🛠️ Tech Stack

| Library              | Usage                            |
| -------------------- | -------------------------------- |
| `pandas`,`numpy` | Data manipulation                |
| `scikit-learn`     | Preprocessing, pipelines, models |
| `xgboost`          | Deployed model                   |
| `shap`             | Model explainability             |
| `lightgbm`         | Gradient boosting alternative    |
| `plotly`           | Interactive EDA visualizations   |
| `streamlit`        | Web app deployment               |
| `joblib`           | Model serialization              |
| CUDA (RTX 5070)      | GPU acceleration for XGBoost     |
