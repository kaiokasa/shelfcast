from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from walmart_features import (HOLIDAYS, LGBM_WEIGHT, MARKDOWN_COLS, SVD_RANK, add_lag_features,
                              build_base_features, load_raw, predict_ensemble, svd_denoise, wmae)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "walmart_lgbm_model.pkl"
VALIDATION_PATH = BASE_DIR / "model_validation.csv.gz"

EXTERNAL_FACTORS = ["Temperature", "Fuel_Price", "CPI", "Unemployment", *MARKDOWN_COLS]


@st.cache_data
def load_data():
    train, _, feature_store = load_raw()
    df = build_base_features(train, feature_store)
    feature_store["Date"] = pd.to_datetime(feature_store["Date"])
    return df, feature_store, svd_denoise(df)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_validation():
    return pd.read_csv(VALIDATION_PATH, parse_dates=["Date"])


df, feature_store, denoised_history = load_data()
model = load_model()
val = load_validation()
val_wmae = wmae(val["Weekly_Sales"], val["Predicted"], val["IsHoliday"] == 1)

st.title("🛒 Walmart Weekly Sales Predictor")
st.caption(f"Ensemble: {LGBM_WEIGHT:.0%} LightGBM + lag features, {1 - LGBM_WEIGHT:.0%} seasonal naive "
           f"(SVD rank {SVD_RANK}) · WMAE {val_wmae:,.0f} on a validation year that includes Thanksgiving & Christmas")

tab1, tab2, tab3 = st.tabs(["📊 EDA", "📈 Model Performance", "🔮 Predict"])

with tab1:
    st.subheader("Total Weekly Sales by Week of Year")
    total_weekly = df.groupby("Week_Of_Year", as_index=False)["Weekly_Sales"].sum()
    fig1 = px.line(total_weekly, x="Week_Of_Year", y="Weekly_Sales",
                   labels={"Week_Of_Year": "Week of Year", "Weekly_Sales": "Total Sales ($)"})
    st.plotly_chart(fig1, width="stretch")

    st.subheader("Sales by Store Type")
    type_sales = df.groupby(["Week_Of_Year", "Type"], as_index=False)["Weekly_Sales"].sum()
    fig2 = px.line(type_sales, x="Week_Of_Year", y="Weekly_Sales", color="Type",
                   labels={"Week_Of_Year": "Week of Year", "Weekly_Sales": "Total Sales ($)"})
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Sales by Year")
    yearly = df.groupby(["Year", "Week_Of_Year"], as_index=False)["Weekly_Sales"].sum()
    fig3 = px.line(yearly, x="Week_Of_Year", y="Weekly_Sales", color="Year",
                   labels={"Week_Of_Year": "Week of Year", "Weekly_Sales": "Total Sales ($)"})
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Holiday Impact on Sales")
    holiday_effect = df.groupby("IsHoliday", as_index=False)["Weekly_Sales"].mean()
    holiday_effect["IsHoliday"] = holiday_effect["IsHoliday"].map({1: "Holiday", 0: "Non-Holiday"})
    fig4 = px.bar(holiday_effect, x="IsHoliday", y="Weekly_Sales", text_auto=".2s",
                  labels={"IsHoliday": "Week Type", "Weekly_Sales": "Avg Sales ($)"})
    st.plotly_chart(fig4, width="stretch")

    st.subheader("Markdowns vs Sales")
    weekly_md = df.groupby("Week_Of_Year", as_index=False).agg(
        {"Weekly_Sales": "sum", **{c: "sum" for c in MARKDOWN_COLS}}
    )
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=weekly_md["Week_Of_Year"], y=weekly_md["Weekly_Sales"],
                              name="Total Sales", yaxis="y1", line=dict(width=3)))
    for md in MARKDOWN_COLS:
        fig5.add_trace(go.Scatter(x=weekly_md["Week_Of_Year"], y=weekly_md[md],
                                  name=md, yaxis="y2"))
    fig5.update_layout(
        yaxis=dict(title="Total Sales ($)"),
        yaxis2=dict(title="MarkDown ($)", overlaying="y", side="right"),
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig5, width="stretch")

with tab2:
    st.subheader("How good is the model on unseen data?")
    st.caption("Model trained until 2011-10-28 and evaluated on the next 52 weeks (Nov 2011 → Oct 2012), "
               "a full year including Thanksgiving and Christmas. The deployed model is then retrained on all data.")

    err = val["Weekly_Sales"] - val["Predicted"]
    peaks = val["Date"].isin([pd.Timestamp(d) for h in ("Thanksgiving", "Christmas")
                              for d in HOLIDAYS[h].values()])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WMAE (Kaggle metric)", f"${val_wmae:,.0f}")
    c2.metric("MAE", f"${err.abs().mean():,.0f}")
    c3.metric("R²", f"{1 - (err ** 2).sum() / ((val['Weekly_Sales'] - val['Weekly_Sales'].mean()) ** 2).sum():.4f}")
    c4.metric("MAE Thanksgiving/Xmas", f"${err[peaks].abs().mean():,.0f}")

    st.markdown("**Why an ensemble?** Each component alone vs their average:")
    components = {"LightGBM + lags": "LightGBM", f"Seasonal naive (SVD rank {SVD_RANK})": "Seasonal_Naive",
                  "Ensemble": "Predicted"}
    st.dataframe(pd.DataFrame([
        {"Model": name,
         "WMAE": wmae(val["Weekly_Sales"], val[col], val["IsHoliday"] == 1),
         "MAE": (val["Weekly_Sales"] - val[col]).abs().mean(),
         "MAE Thanksgiving/Xmas": (val["Weekly_Sales"] - val[col])[peaks].abs().mean()}
        for name, col in components.items()
    ]).style.format({"WMAE": "{:,.0f}", "MAE": "{:,.0f}", "MAE Thanksgiving/Xmas": "{:,.0f}"}),
        hide_index=True)

    st.subheader("Total weekly sales: actual vs predicted")
    weekly_val = val.groupby("Date", as_index=False)[["Weekly_Sales", "Predicted", "LightGBM", "Seasonal_Naive"]].sum()
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=weekly_val["Date"], y=weekly_val["Weekly_Sales"], name="Actual",
                              line=dict(width=3, color="black")))
    fig6.add_trace(go.Scatter(x=weekly_val["Date"], y=weekly_val["Predicted"], name="Ensemble", line=dict(width=3)))
    fig6.add_trace(go.Scatter(x=weekly_val["Date"], y=weekly_val["LightGBM"], name="LightGBM alone",
                              line=dict(width=1, dash="dot")))
    fig6.add_trace(go.Scatter(x=weekly_val["Date"], y=weekly_val["Seasonal_Naive"], name="Seasonal naive alone",
                              line=dict(width=1, dash="dot")))
    fig6.update_layout(yaxis_title="Total Sales ($)", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig6, width="stretch")

    st.subheader("Zoom on one store / department")
    z1, z2 = st.columns(2)
    zoom_store = z1.selectbox("Store", sorted(val["Store"].unique()), key="zoom_store")
    zoom_dept = z2.selectbox("Department", sorted(val.loc[val["Store"] == zoom_store, "Dept"].unique()),
                             key="zoom_dept")
    zoom = val[(val["Store"] == zoom_store) & (val["Dept"] == zoom_dept)]
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=zoom["Date"], y=zoom["Weekly_Sales"], name="Actual", line=dict(width=3)))
    fig7.add_trace(go.Scatter(x=zoom["Date"], y=zoom["Predicted"], name="Predicted", line=dict(width=3, dash="dot")))
    fig7.update_layout(yaxis_title="Weekly Sales ($)", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig7, width="stretch")

    st.subheader("Feature importance (LightGBM component)")
    lgbm = model.named_steps["Model"]
    importance = (pd.DataFrame({"Feature": lgbm.feature_name_,
                                "Gain": lgbm.booster_.feature_importance(importance_type="gain")})
                  .assign(Share=lambda d: d["Gain"] / d["Gain"].sum())
                  .sort_values("Share"))
    fig8 = px.bar(importance, x="Share", y="Feature", orientation="h")
    fig8.update_layout(height=650, xaxis_tickformat=".0%", xaxis_title="Share of total gain")
    st.plotly_chart(fig8, width="stretch")

with tab3:
    st.subheader("Predict Weekly Sales")
    st.caption("Pick a store, a department and a week: store info, external factors and last year's "
               "sales are filled in automatically from the dataset.")

    col1, col2, col3 = st.columns(3)
    store = col1.selectbox("Store", sorted(df["Store"].unique()))
    dept = col2.selectbox("Department", sorted(df.loc[df["Store"] == store, "Dept"].unique()))
    store_weeks = feature_store.loc[feature_store["Store"] == store].sort_values("Date")
    last_known = df["Date"].max()
    week = col3.selectbox(
        "Week (ending Friday)", store_weeks["Date"].tolist(),
        index=int((store_weeks["Date"] <= last_known).sum()),  # first week after the known history
        format_func=lambda d: d.strftime("%Y-%m-%d") + ("" if d <= last_known else "  (future)"),
    )

    week_info = store_weeks.loc[store_weeks["Date"] == week].iloc[0]
    st.markdown(f"**Store type** {week_info['Type']} · **Size** {week_info['Size']:,} sq ft · "
                f"**Holiday week** {'yes' if week_info['IsHoliday'] else 'no'}")

    with st.expander("External factors (pre-filled from features.csv, editable)"):
        e1, e2 = st.columns(2)
        overrides = {}
        for i, col in enumerate(EXTERNAL_FACTORS):
            default = week_info[col]
            if col in MARKDOWN_COLS and pd.isna(default):
                default = 0.0
            overrides[col] = (e1 if i % 2 == 0 else e2).number_input(
                col, value=None if pd.isna(default) else float(default),
                key=f"{col}_{store}_{week.date()}")

    row = pd.DataFrame({"Store": [store], "Dept": [dept], "Date": [week.strftime("%Y-%m-%d")],
                        "IsHoliday": [bool(week_info["IsHoliday"])]})
    raw_store = feature_store.assign(Date=feature_store["Date"].dt.strftime("%Y-%m-%d"))
    row = build_base_features(row, raw_store)
    for col, value in overrides.items():
        row[col] = np.nan if value is None else value

    if st.button("Predict Weekly Sales 🚀"):
        ensemble, lgbm_part, naive_part = predict_ensemble(model, row, history=df, denoised_history=denoised_history)
        prediction = ensemble[0]
        st.success(f"### Predicted Weekly Sales: **${prediction:,.2f}**")
        st.caption(f"= {LGBM_WEIGHT:.0%} × LightGBM (${lgbm_part[0]:,.0f}) + {1 - LGBM_WEIGHT:.0%} × "
                   f"seasonal naive (${naive_part[0]:,.0f})")

        lags = add_lag_features(row, history=df)
        m1, m2, m3 = st.columns(3)
        lag = lags["Sales_Lag_52"].iloc[0]
        m1.metric("Same week last year", "n/a" if pd.isna(lag) else f"${lag:,.0f}")
        m2.metric("Store/Dept average", f"${lags['Store_Dept_Mean'].iloc[0]:,.0f}")
        actual = df.loc[(df["Store"] == store) & (df["Dept"] == dept) & (df["Date"] == week), "Weekly_Sales"]
        if len(actual):
            m3.metric("Actual sales", f"${actual.iloc[0]:,.0f}",
                      delta=f"{prediction - actual.iloc[0]:+,.0f} predicted vs actual", delta_color="off")
            st.info("This week is in the training data: the deployed model has seen it, "
                    "see the Model Performance tab for an honest evaluation.")
        else:
            m3.metric("Actual sales", "unknown (future week)")
