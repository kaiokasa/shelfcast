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

HOLIDAY_NAMES = ["Thanksgiving", "Christmas", "SuperBowl", "LaborDay"]


def holiday_week(name, year):
    """
    Friday ending the holiday week (Walmart weeks in the dataset end on a Friday), for any year.
    Matches the dates given by Kaggle for 2010-2013 (checked in the tests).
    """
    if name == "Thanksgiving":  # 4th Thursday of November; the week ends the next day
        nov1 = pd.Timestamp(year, 11, 1)
        return nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7) + pd.Timedelta(weeks=3, days=1)
    if name == "Christmas":     # the Friday on or after December 25
        dec25 = pd.Timestamp(year, 12, 25)
        return dec25 + pd.Timedelta(days=(4 - dec25.weekday()) % 7)
    if name == "SuperBowl":     # 1st Sunday of February until 2021, 2nd Sunday since 2022; week ends 5 days later
        feb1 = pd.Timestamp(year, 2, 1)
        sunday = feb1 + pd.Timedelta(days=(6 - feb1.weekday()) % 7) + pd.Timedelta(weeks=1 if year >= 2022 else 0)
        return sunday + pd.Timedelta(days=5)
    if name == "LaborDay":      # 1st Monday of September; week ends 4 days later
        sep1 = pd.Timestamp(year, 9, 1)
        return sep1 + pd.Timedelta(days=(0 - sep1.weekday()) % 7) + pd.Timedelta(days=4)
    raise ValueError(f"Unknown holiday: {name}")


# Friday of each holiday week, 2010-2040 (holiday_week() works for any year)
HOLIDAYS = {name: {y: holiday_week(name, y).strftime("%Y-%m-%d") for y in range(2010, 2041)}
            for name in HOLIDAY_NAMES}


def weeks_back(years):
    """Number of weeks closest to `years` real years (52, 104, 157, 209, 261...)."""
    return round(365.25 * years / 7)


def reference_week(week, years):
    """
    The same week `years` years earlier.
    - One year back: exactly 52 weeks, the lag the model was trained with.
    - Further back, a fixed 52-week step drifts by about a day per year, and Thanksgiving itself moves by up to
      a week from one year to the next. So weeks within 4 weeks of a holiday keep the same distance to that
      holiday ("2 weeks before Thanksgiving"), and the other weeks follow real years.
    """
    week = pd.Timestamp(week)
    if years == 1:
        return week - pd.Timedelta(weeks=52)
    # the closest holiday, looking at the surrounding years too (early January is close to last Christmas)
    nearby = [(abs(days), days, name, shift)
              for name in HOLIDAY_NAMES for shift in (-1, 0, 1)
              for days in [(holiday_week(name, week.year + shift) - week).days] if abs(days) <= 28]
    if nearby:
        _, days, name, shift = min(nearby)
        return holiday_week(name, week.year + shift - years) - pd.Timedelta(days=days)
    return week - pd.Timedelta(weeks=weeks_back(years))


def week_ending(date):
    """The Friday that ends the Walmart week containing `date`."""
    d = pd.Timestamp(date).normalize()
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def is_holiday_week(week):
    week = pd.Timestamp(week)
    return any(holiday_week(name, week.year) == week for name in HOLIDAY_NAMES)

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
    for name in HOLIDAY_NAMES:
        ref = df["Year"].map({y: holiday_week(name, y) for y in df["Year"].unique()})
        df[f"Days_to_{name}"] = (ref - df["Date"]).dt.days
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


def add_lag_features(df, history, stats_source=None, max_years_back=1):
    """
    Add 52+ week lags and store/department sales levels.

    df:             rows to enrich (needs Store, Dept, Date)
    history:        known sales (Store, Dept, Date, Weekly_Sales) the lags are read from
    stats_source:   rows used for the store/department mean/median (defaults to history).
                    Must only contain data known at training time to avoid leakage.
    max_years_back: 1 (training, validation, Kaggle test) = exactly one year earlier.
                    For dates further in the future, the same week of the most recent year with recorded
                    sales is used instead (up to max_years_back years earlier).
    """
    stats_source = history if stats_source is None else stats_source
    df = df.copy()
    hist = history[["Store", "Dept", "Date", "Weekly_Sales"]].rename(columns={"Date": "RefDate"})
    keys = df[["Store", "Dept", "Date"]].reset_index(drop=True)
    lag_52 = lag_mean4 = None
    for years in range(1, max_years_back + 1):
        # the same week `years` years earlier (exactly 52 weeks for one year, see reference_week)
        ref = keys["Date"].map({d: reference_week(d, years) for d in keys["Date"].unique()})
        lags = {}
        for k in LAG_WEEKS:
            lookup = keys[["Store", "Dept"]].assign(RefDate=ref - pd.Timedelta(weeks=k - 52))
            lags[k] = lookup.merge(hist, on=["Store", "Dept", "RefDate"], how="left")["Weekly_Sales"].to_numpy()
        lags = pd.DataFrame(lags)
        this_52, this_mean4 = lags[52].to_numpy(), lags.mean(axis=1).to_numpy()
        if lag_52 is None:
            lag_52, lag_mean4 = this_52, this_mean4
        else:  # only fill the weeks that have no more recent value
            lag_52 = np.where(np.isnan(lag_52), this_52, lag_52)
            lag_mean4 = np.where(np.isnan(lag_mean4), this_mean4, lag_mean4)
    df["Sales_Lag_52"] = lag_52
    df["Sales_Lag_52_Mean4"] = lag_mean4

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


def seasonal_naive(rows, denoised_history, max_years_back=1):
    """Same week last year (denoised), falling back on the store/department mean."""
    v = add_lag_features(rows[["Store", "Dept", "Date"]], history=denoised_history, max_years_back=max_years_back)
    return v["Sales_Lag_52"].fillna(v["Store_Dept_Mean"]).fillna(0).to_numpy()


def predict_ensemble(model, rows, history, denoised_history, max_years_back=1):
    """
    rows: output of build_base_features (no lag features yet).
    max_years_back > 1 lets the app predict any future date (see add_lag_features).
    Returns (ensemble, lightgbm, seasonal_naive) predictions, aligned with rows.
    """
    lgbm_pred = model.predict(add_lag_features(rows, history=history, max_years_back=max_years_back)[FEATURES])
    naive_pred = seasonal_naive(rows, denoised_history, max_years_back=max_years_back)
    return LGBM_WEIGHT * lgbm_pred + (1 - LGBM_WEIGHT) * naive_pred, lgbm_pred, naive_pred


# --- Predicting any week, including weeks after the end of features.csv (July 2013) ---

SEASONAL_FACTORS = ["Temperature", *MARKDOWN_COLS]   # repeat the same week of the most recent known year
TREND_FACTORS = ["Fuel_Price", "CPI", "Unemployment"]  # keep the last known value


def features_for_week(store, week, feature_store):
    """
    The external features of one store for one week, in the format of feature_store.
    Known weeks come from features.csv. Other weeks are estimated, and `estimated` is True:
    seasonal factors repeat the same week of the most recent year that has them, economic
    factors keep their last known value.
    """
    fs = feature_store[feature_store["Store"] == store].copy()
    fs["Date"] = pd.to_datetime(fs["Date"])
    week = pd.Timestamp(week)
    known = fs[fs["Date"] == week]
    if len(known):
        return known.iloc[0].to_dict(), False

    fs = fs.sort_values("Date").set_index("Date")
    row = {"Store": store, "Date": week, "IsHoliday": is_holiday_week(week),
           "Type": fs["Type"].iloc[-1], "Size": fs["Size"].iloc[-1]}
    # the same week in past years, most recent first, aligned like the lags (see reference_week)
    past_weeks = [reference_week(week, y) for y in range(1, 31)]
    for col in SEASONAL_FACTORS:
        known_values = [fs.at[d, col] for d in past_weeks if d in fs.index and pd.notna(fs.at[d, col])]
        row[col] = known_values[0] if known_values else (fs[col].dropna().iloc[-1] if fs[col].notna().any() else np.nan)
    for col in TREND_FACTORS:
        values = fs[col].dropna()
        row[col] = values.iloc[-1] if len(values) else np.nan
    return row, True


def wmae(y_true, y_pred, is_holiday):
    """Official Kaggle metric: holiday weeks weigh 5x."""
    w = np.where(is_holiday, 5, 1)
    return np.sum(w * np.abs(np.asarray(y_true) - np.asarray(y_pred))) / np.sum(w)
