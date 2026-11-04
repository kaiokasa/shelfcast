from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"
MODEL_PATH = BASE_DIR / "walmart_xgb_model.pkl"

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

# Same holiday reference dates as in Notebook.ipynb: the model was trained with these
THANKSGIVING_WEEK = {2010: pd.Timestamp("2010-11-26"), 2011: pd.Timestamp("2011-11-25"),
                     2012: pd.Timestamp("2012-11-23"), 2013: pd.Timestamp("2013-11-29")}
CHRISTMAS_WEEK    = {2010: pd.Timestamp("2010-12-31"), 2011: pd.Timestamp("2011-12-30"),
                     2012: pd.Timestamp("2012-12-28"), 2013: pd.Timestamp("2013-12-27")}
SUPER_BOWL_WEEK   = {2010: pd.Timestamp("2010-02-12"), 2011: pd.Timestamp("2011-02-11"),
                     2012: pd.Timestamp("2012-02-10"), 2013: pd.Timestamp("2013-02-08")}
LABOR_DAY_WEEK    = {2010: pd.Timestamp("2010-09-10"), 2011: pd.Timestamp("2011-09-09"),
                     2012: pd.Timestamp("2012-09-07"), 2013: pd.Timestamp("2013-09-06")}


def days_to_holidays(date: pd.Timestamp) -> dict:
    return {
        "Days_to_Thanksgiving": (THANKSGIVING_WEEK[date.year] - date).days,
        "Days_to_Christmas":    (CHRISTMAS_WEEK[date.year] - date).days,
        "Days_to_SuperBowl":    (SUPER_BOWL_WEEK[date.year] - date).days,
        "Days_to_LaborDay":     (LABOR_DAY_WEEK[date.year] - date).days,
    }


def wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(np.asarray(y_true) - np.asarray(y_pred))) / np.sum(weights)


@st.cache_data
def load_data() -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / "train.csv")
    features = pd.read_csv(DATA_DIR / "features.csv")
    stores = pd.read_csv(DATA_DIR / "stores.csv")

    feature_store_merged = features.merge(stores, how="inner", on="Store")
    df = train.merge(feature_store_merged, on=["Store", "Date", "IsHoliday"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week_Of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"] = df["Date"].dt.year
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    holiday_features = pd.DataFrame([days_to_holidays(d) for d in df["Date"]], index=df.index)
    return pd.concat([df, holiday_features], axis=1)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    model.named_steps["Model"].set_params(device="cpu")
    return model


@st.cache_data
def validation_predictions(_model, df: pd.DataFrame) -> pd.DataFrame:
    """Predict 2012, the year the model never saw during training."""
    val = df[df["Year"] == 2012].copy()
    val["Predicted"] = _model.predict(val[list(_model.feature_names_in_)])
    return val


df = load_data()
model = load_model()

st.title("🛒 Walmart Weekly Sales Predictor")

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
    st.caption("The model was trained on 2010–2011 only. Everything below is measured on 2012.")

    val = validation_predictions(model, df)
    abs_err = (val["Weekly_Sales"] - val["Predicted"]).abs()
    holiday = val["IsHoliday"] == 1

    c1, c2, c3 = st.columns(3)
    c1.metric("WMAE (Kaggle metric)", f"${wmae(val['Weekly_Sales'], val['Predicted'], holiday):,.0f}")
    c2.metric("MAE", f"${abs_err.mean():,.0f}")
    c3.metric("R²", f"{1 - ((val['Weekly_Sales'] - val['Predicted']) ** 2).sum() / ((val['Weekly_Sales'] - val['Weekly_Sales'].mean()) ** 2).sum():.4f}")

    st.subheader("Total weekly sales in 2012: actual vs predicted")
    weekly_val = val.groupby("Date", as_index=False)[["Weekly_Sales", "Predicted"]].sum()
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=weekly_val["Date"], y=weekly_val["Weekly_Sales"], name="Actual", line=dict(width=3)))
    fig6.add_trace(go.Scatter(x=weekly_val["Date"], y=weekly_val["Predicted"], name="Predicted",
                              line=dict(width=3, dash="dot")))
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

    st.subheader("Feature importance")
    feature_names = [n.split("__")[1] for n in model.named_steps["Preprocessing"].get_feature_names_out()]
    importance = (pd.DataFrame({"Feature": feature_names,
                                "Importance": model.named_steps["Model"].feature_importances_})
                  .sort_values("Importance"))
    fig8 = px.bar(importance, x="Importance", y="Feature", orientation="h")
    fig8.update_layout(height=550)
    st.plotly_chart(fig8, width="stretch")

with tab3:
    st.subheader("Predict Weekly Sales")
    col1, col2 = st.columns(2)

    with col1:
        store = st.number_input("Store", min_value=1, max_value=45, value=1)
        dept = st.number_input("Department", min_value=1, max_value=99, value=1)
        store_type = st.selectbox("Store Type", ["A", "B", "C"])
        size = st.number_input("Store Size", min_value=34000, max_value=220000, value=151315)
        is_holiday = st.selectbox("Is Holiday Week?", [0, 1])
        year = st.number_input("Year", min_value=2010, max_value=2013, value=2012)
        week_of_year = st.number_input("Week of Year", min_value=1, max_value=52, value=1)

    with col2:
        temperature = st.number_input("Temperature (°F)", value=60.0)
        fuel_price = st.number_input("Fuel Price", value=3.5)
        cpi = st.number_input("CPI", value=211.0)
        unemployment = st.number_input("Unemployment", value=8.0)
        markdowns = [st.number_input(md, value=0.0) for md in MARKDOWN_COLS]

    # Walmart weeks in the dataset end on a Friday
    date = pd.Timestamp.fromisocalendar(year, week_of_year, 5)

    if st.button("Predict Weekly Sales 🚀"):
        input_data = pd.DataFrame([{
            "Store": store, "Dept": dept, "IsHoliday": is_holiday,
            "Temperature": temperature, "Fuel_Price": fuel_price,
            **dict(zip(MARKDOWN_COLS, markdowns)),
            "CPI": cpi, "Unemployment": unemployment,
            "Type": store_type, "Size": size, "Year": year,
            "Week_Of_Year": week_of_year,
            **days_to_holidays(date),
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Weekly Sales: **${prediction:,.2f}**")
