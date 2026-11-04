"""Unit tests for walmart_features.py.

Run from the project root with:  python -m pytest
They use tiny hand-made tables, so they run in a few seconds and do not need the Kaggle data.
"""
import numpy as np
import pandas as pd
import pytest

from walmart_features import (MARKDOWN_COLS, add_lag_features, build_base_features, predict_ensemble,
                              seasonal_naive, svd_denoise, wmae)


def sales(rows):
    """Build a sales history from (store, dept, date, weekly_sales) tuples."""
    df = pd.DataFrame(rows, columns=["Store", "Dept", "Date", "Weekly_Sales"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------------------------------------------- add_lag_features

def test_lag_52_is_the_same_week_one_year_earlier():
    # Thanksgiving week 2010 and 2011: 364 days (52 weeks) apart, both on a Friday
    history = sales([(1, 1, "2010-11-26", 100.0), (1, 1, "2011-11-25", 200.0)])
    rows = history[history["Date"] == "2011-11-25"]

    result = add_lag_features(rows, history=history)

    assert result["Sales_Lag_52"].iloc[0] == 100.0


def test_lags_only_read_the_history_they_are_given():
    # The 2011 week is in the rows to predict but NOT in the history: its own sales must never be used
    known = sales([(1, 1, "2010-11-26", 100.0)])
    future = sales([(1, 1, "2011-11-25", 999.0)])

    result = add_lag_features(future, history=known)

    assert result["Sales_Lag_52"].iloc[0] == 100.0
    assert result["Store_Dept_Mean"].iloc[0] == 100.0  # the 999 of the predicted week is not in the mean


def test_missing_history_gives_missing_lag_not_an_error():
    history = sales([(1, 1, "2011-11-25", 200.0)])

    result = add_lag_features(history, history=history)

    assert result["Sales_Lag_52"].isna().all()


def test_lag_mean4_averages_lags_52_to_55():
    target = pd.Timestamp("2012-01-06")
    history = sales([(1, 1, target - pd.Timedelta(weeks=k), float(k)) for k in [52, 53, 54, 55]])
    rows = sales([(1, 1, target, 0.0)])

    result = add_lag_features(rows, history=history)

    assert result["Sales_Lag_52_Mean4"].iloc[0] == pytest.approx(np.mean([52, 53, 54, 55]))


def test_store_dept_stats_come_from_stats_source():
    history = sales([(1, 1, "2010-02-05", 10.0), (1, 1, "2010-02-12", 30.0), (1, 1, "2010-02-19", 1000.0)])
    stats_source = history.iloc[:2]  # e.g. training period only

    result = add_lag_features(history.iloc[[2]], history=history, stats_source=stats_source)

    assert result["Store_Dept_Mean"].iloc[0] == 20.0
    assert result["Store_Dept_Median"].iloc[0] == 20.0


def test_lag_features_keep_row_order():
    history = sales([(s, 1, "2010-11-26", float(s)) for s in [3, 1, 2]])
    rows = sales([(s, 1, "2011-11-25", 0.0) for s in [3, 1, 2]])

    result = add_lag_features(rows, history=history)

    assert result["Store"].tolist() == [3, 1, 2]
    assert result["Sales_Lag_52"].tolist() == [3.0, 1.0, 2.0]


# ---------------------------------------------------------------- build_base_features

def make_feature_store(dates):
    n = len(dates)
    return pd.DataFrame({
        "Store": [1] * n, "Date": dates, "IsHoliday": [False] * n,
        "Temperature": [50.0] * n, "Fuel_Price": [3.0] * n,
        **{c: [np.nan] * n for c in MARKDOWN_COLS},
        "CPI": [210.0] * n, "Unemployment": [8.0] * n, "Type": ["A"] * n, "Size": [150000] * n,
    })


def test_days_to_holiday_is_zero_on_the_holiday_week():
    dates = ["2011-11-25", "2011-12-30", "2012-02-10", "2012-09-07"]
    df = pd.DataFrame({"Store": 1, "Dept": 1, "Date": dates, "IsHoliday": False})

    result = build_base_features(df, make_feature_store(dates)).set_index("Date")

    assert result.loc["2011-11-25", "Days_to_Thanksgiving"] == 0
    assert result.loc["2011-12-30", "Days_to_Christmas"] == 0
    assert result.loc["2012-02-10", "Days_to_SuperBowl"] == 0
    assert result.loc["2012-09-07", "Days_to_LaborDay"] == 0


def test_days_to_holiday_counts_down_before_the_holiday():
    dates = ["2011-11-18"]  # one week before Thanksgiving 2011
    df = pd.DataFrame({"Store": 1, "Dept": 1, "Date": dates, "IsHoliday": False})

    result = build_base_features(df, make_feature_store(dates))

    assert result["Days_to_Thanksgiving"].iloc[0] == 7


def test_base_features_fill_missing_markdowns_and_encode_holiday():
    dates = ["2011-11-25"]
    df = pd.DataFrame({"Store": 1, "Dept": 1, "Date": dates, "IsHoliday": False})

    result = build_base_features(df, make_feature_store(dates))

    assert (result[MARKDOWN_COLS] == 0).all().all()
    assert result["IsHoliday"].iloc[0] == 0
    assert result["Year"].iloc[0] == 2011
    assert result["Week_Of_Year"].iloc[0] == 47


# ---------------------------------------------------------------- svd_denoise

def random_history(n_weeks=20, n_stores=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-02-05", periods=n_weeks, freq="7D")
    return sales([(s, 1, d, float(rng.normal(1000, 100))) for d in dates for s in range(1, n_stores + 1)])


def test_svd_with_full_rank_returns_the_original_sales():
    history = random_history()

    result = svd_denoise(history, k=6).sort_values(["Store", "Date"]).reset_index(drop=True)
    expected = history.sort_values(["Store", "Date"]).reset_index(drop=True)

    np.testing.assert_allclose(result["Weekly_Sales"], expected["Weekly_Sales"])


def test_svd_low_rank_keeps_the_same_rows_and_changes_values():
    history = random_history()

    result = svd_denoise(history, k=2)

    assert len(result) == len(history)
    assert set(map(tuple, result[["Store", "Dept", "Date"]].values)) == set(map(tuple, history[["Store", "Dept", "Date"]].values))
    assert not np.allclose(result.sort_values(["Store", "Date"])["Weekly_Sales"].values,
                           history.sort_values(["Store", "Date"])["Weekly_Sales"].values)


def test_svd_does_not_invent_weeks_that_were_never_observed():
    history = random_history().iloc[1:]  # one store/week is missing

    result = svd_denoise(history, k=2)

    assert len(result) == len(history)


# ---------------------------------------------------------------- seasonal_naive & predict_ensemble

def test_seasonal_naive_uses_last_year_then_falls_back_on_store_dept_mean():
    history = sales([(1, 1, "2010-11-26", 100.0), (1, 1, "2010-12-03", 300.0)])
    rows = sales([(1, 1, "2011-11-25", 0.0),     # 52 weeks after a known week
                  (1, 1, "2011-12-09", 0.0)])    # no known week 52 weeks before

    result = seasonal_naive(rows, history)

    assert result[0] == 100.0
    assert result[1] == 200.0  # mean of the store/department history


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_predict_ensemble_is_the_50_50_average():
    dates = ["2011-11-25"]
    history = build_base_features(
        pd.DataFrame({"Store": 1, "Dept": 1, "Date": ["2010-11-26"], "IsHoliday": False, "Weekly_Sales": [100.0]}),
        make_feature_store(["2010-11-26"]))
    rows = build_base_features(pd.DataFrame({"Store": 1, "Dept": 1, "Date": dates, "IsHoliday": False}),
                               make_feature_store(dates))

    ensemble, lgbm, naive = predict_ensemble(ConstantModel(300.0), rows, history=history, denoised_history=history)

    assert lgbm[0] == 300.0
    assert naive[0] == 100.0
    assert ensemble[0] == 200.0


# ---------------------------------------------------------------- wmae

def test_wmae_gives_holiday_weeks_five_times_more_weight():
    y_true = [100.0, 100.0]
    y_pred = [110.0, 130.0]  # errors: 10 on a normal week, 30 on a holiday week

    assert wmae(y_true, y_pred, is_holiday=[False, True]) == pytest.approx((10 * 1 + 30 * 5) / 6)


def test_wmae_without_holidays_is_the_mae():
    assert wmae([1.0, 2.0, 3.0], [2.0, 2.0, 5.0], is_holiday=[False, False, False]) == pytest.approx(1.0)
