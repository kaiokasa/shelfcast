"""Feature engineering shared by Notebook.ipynb and app.py.

Keeping it in one place guarantees that the deployed model receives exactly
the same features as during training.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = Path(__file__).parent / "Data"

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

# Friday of each holiday week (Walmart weeks in the dataset end on a Friday)
HOLIDAYS = {
    "Thanksgiving": {2010: "2010-11-26", 2011: "2011-11-25", 2012: "2012-11-23", 2013: "2013-11-29"},
    "Christmas":    {2010: "2010-12-31", 2011: "2011-12-30", 2012: "2012-12-28", 2013: "2013-12-27"},
    "SuperBowl":    {2010: "2010-02-12", 2011: "2011-02-11", 2012: "2012-02-10", 2013: "2013-02-08"},
    "LaborDay":     {2010: "2010-09-10", 2011: "2011-09-09", 2012: "2012-09-07", 2013: "2013-09-06"},
}

BASE_FEATURES = ["Store", "Dept", "IsHoliday", "Temperature", "Fuel_Price", *MARKDOWN_COLS,
                 "CPI", "Unemployment", "Type", "Size", "Year", "Week_Of_Year",
                 "Days_to_Thanksgiving", "Days_to_Christmas", "Days_to_SuperBowl", "Days_to_LaborDay"]

# Only lags of 52+ weeks: the Kaggle test set goes up to 39 weeks ahead, so shorter lags
# would not be known at prediction time.
LAG_WEEKS = [52, 53, 54, 55]
LAG_FEATURES = ["Sales_Lag_52", "Sales_Lag_52_Mean4", "Store_Dept_Mean", "Store_Dept_Median"]

FEATURES = BASE_FEATURES + LAG_FEATURES


def load_raw():
    """Return train, test and the merged features/stores table."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    feature_store = pd.read_csv(DATA_DIR / "features.csv").merge(
        pd.read_csv(DATA_DIR / "stores.csv"), how="inner", on="Store")
    return train, test, feature_store


def build_base_features(df, feature_store):
    """Merge external features and add calendar / holiday-distance features."""
    df = df.merge(feature_store, how="inner", on=["Store", "Date", "IsHoliday"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Week_Of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
    for name, dates in HOLIDAYS.items():
        ref = df["Year"].map({y: pd.Timestamp(d) for y, d in dates.items()})
        df[f"Days_to_{name}"] = (ref - df["Date"]).dt.days
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


def add_lag_features(df, history, stats_source=None):
    """
    Add 52+ week lags and store/department sales levels.

    df:           rows to enrich (needs Store, Dept, Date)
    history:      known sales (Store, Dept, Date, Weekly_Sales) the lags are read from
    stats_source: rows used for the store/department mean/median (defaults to history).
                  Must only contain data known at training time to avoid leakage.
    """
    stats_source = history if stats_source is None else stats_source
    df = df.copy()
    hist = history[["Store", "Dept", "Date", "Weekly_Sales"]]
    for k in LAG_WEEKS:
        shifted = (hist.assign(Date=hist["Date"] + pd.Timedelta(weeks=k))
                   .rename(columns={"Weekly_Sales": f"lag_{k}"}))
        df = df.merge(shifted, on=["Store", "Dept", "Date"], how="left")
    lag_cols = [f"lag_{k}" for k in LAG_WEEKS]
    df["Sales_Lag_52"] = df["lag_52"]
    df["Sales_Lag_52_Mean4"] = df[lag_cols].mean(axis=1)
    df = df.drop(columns=lag_cols)

    stats = (stats_source.groupby(["Store", "Dept"])["Weekly_Sales"]
             .agg(Store_Dept_Mean="mean", Store_Dept_Median="median").reset_index())
    return df.merge(stats, on=["Store", "Dept"], how="left")


def make_lgbm_pipeline(device="cpu"):
    """LightGBM + lag features: the best model of the project (see Notebook_Experiments.ipynb)."""
    numeric = [c for c in FEATURES if c != "Type"]
    preprocessing = ColumnTransformer(
        [("type", OrdinalEncoder(categories=[["C", "B", "A"]]), ["Type"]),
         ("num", "passthrough", numeric)],  # tree model: no scaling needed, NaN lags handled natively
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                          random_state=42, device=device, verbose=-1)
    return Pipeline([("Preprocessing", preprocessing), ("Model", model)])


# --- Ensemble: LightGBM + SVD-denoised seasonal naive (see Notebook_Experiments.ipynb, section 5) ---

SVD_RANK = 10          # departments' weeks x stores matrices are reduced from 45 to 10 components
LGBM_WEIGHT = 0.5      # fixed in advance, not tuned on the validation set


def svd_denoise(history, k=SVD_RANK):
    """
    For each department, replace the weeks x stores sales matrix by its rank-k approximation.
    Stores share most of a department's seasonality; what is specific to one store is partly noise.
    """
    parts = []
    for dept, g in history.groupby("Dept"):
        m = g.pivot_table(index="Date", columns="Store", values="Weekly_Sales")
        observed = m.notna()
        x = m.fillna(0).values
        if min(x.shape) > k:
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            x = (u[:, :k] * s[:k]) @ vt[:k]
        denoised = pd.DataFrame(x, index=m.index, columns=m.columns).where(observed)
        parts.append(denoised.stack().dropna().rename("Weekly_Sales").reset_index().assign(Dept=dept))
    return pd.concat(parts, ignore_index=True)[["Store", "Dept", "Date", "Weekly_Sales"]]


def seasonal_naive(rows, denoised_history):
    """Same week last year (denoised), falling back on the store/department mean."""
    v = add_lag_features(rows[["Store", "Dept", "Date"]], history=denoised_history)
    return v["Sales_Lag_52"].fillna(v["Store_Dept_Mean"]).fillna(0).to_numpy()


def predict_ensemble(model, rows, history, denoised_history):
    """
    rows: output of build_base_features (no lag features yet).
    Returns (ensemble, lightgbm, seasonal_naive) predictions, aligned with rows.
    """
    lgbm_pred = model.predict(add_lag_features(rows, history=history)[FEATURES])
    naive_pred = seasonal_naive(rows, denoised_history)
    return LGBM_WEIGHT * lgbm_pred + (1 - LGBM_WEIGHT) * naive_pred, lgbm_pred, naive_pred


def wmae(y_true, y_pred, is_holiday):
    """Official Kaggle metric: holiday weeks weigh 5x."""
    w = np.where(is_holiday, 5, 1)
    return np.sum(w * np.abs(np.asarray(y_true) - np.asarray(y_pred))) / np.sum(w)
