
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
├── Notebook.ipynb         
├── app.py                
├── walmart_xgb_model.pkl   
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
* Sales **increased from 2010 to 2011** (~$160M) then **decreased in 2012** (~$450M)

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

| Model                        | MAE             | RMSE            | Val R²          | Train R²        | Gap              |
| ---------------------------- | --------------- | --------------- | ---------------- | ---------------- | ---------------- |
| Linear Regression            | 14,299          | 21,118          | 0.0886           | —               | —               |
| SGD Regressor                | 14,145          | 21,326          | 0.0705           | —               | —               |
| Decision Tree (default)      | 2,967           | 6,717           | 0.9078           | 1.0000           | 0.0922           |
| Decision Tree (tuned)        | 4,623           | 8,132           | 0.8648           | 0.8656           | 0.0008           |
| Random Forest (default)      | 2,272           | 5,003           | 0.9488           | 0.9962           | 0.0474           |
| Random Forest (tuned)        | 2,332           | 5,148           | 0.9458           | 0.9728           | 0.0270           |
| AdaBoost                     | 22,726          | 26,874          | -0.4761          | -0.6408          | —               |
| LightGBM                     | 3,107           | 5,346           | 0.9416           | 0.9539           | 0.0123           |
| **XGBoost (tuned) 🏆** | **2,732** | **4,829** | **0.9523** | **0.9759** | **0.0236** |

---

## 🏆 Best Model — XGBoost

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

**XGBoost** emerged as the champion with a val R² of  **0.9523** , explaining 95.23% of the variance in weekly sales, and an MAE of **$2,732** per prediction. The train/val gap of 0.0236 confirms the model generalizes well to unseen data without memorizing the training set.

---

## 💡 Key Takeaways

* **Linear models failed completely** (R² ~0.09) — the data is highly non-linear due to holiday spikes and complex store interactions
* **AdaBoost collapsed** (R² = -0.47) due to its sensitivity to extreme sales outliers during holidays
* **Tree-based models dominate** — a single Decision Tree jumped to R²=0.90, and ensembles pushed it further
* **XGBoost's built-in regularization** and sequential boosting gave it the edge over all other models
* **Holiday distance features** (`Days_to_Thanksgiving`, etc.) were among the most impactful engineered features
* **Temporal train/val split** is critical for time series data — random splits would leak future data and inflate scores

---

## 🚀 Streamlit App

An interactive web app is included to explore the data and make predictions.

**Features:**

* **📊 EDA tab** — interactive Plotly charts: weekly sales trends, sales by store type, year-over-year comparison, holiday impact, and markdown vs sales
* **🔮 Predict tab** — input store details and get an instant weekly sales prediction from the XGBoost model

**How to run:**

```bash
# activate your environment
conda activate torch-nightly

# run the app
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🛠️ Tech Stack

| Library              | Usage                            |
| -------------------- | -------------------------------- |
| `pandas`,`numpy` | Data manipulation                |
| `scikit-learn`     | Preprocessing, pipelines, models |
| `xgboost`          | Champion model                   |
| `lightgbm`         | Gradient boosting alternative    |
| `plotly`           | Interactive EDA visualizations   |
| `streamlit`        | Web app deployment               |
| `joblib`           | Model serialization              |
| CUDA (RTX 5070)      | GPU acceleration for XGBoost     |
