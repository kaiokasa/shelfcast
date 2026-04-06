import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# load model
model = joblib.load("walmart_xgb_model.pkl")
model.named_steps["Model"].set_params(device="cpu")

# load data
train = pd.read_csv("./Data/train.csv")
features = pd.read_csv("./Data/features.csv")
stores = pd.read_csv("./Data/stores.csv")

feature_store_merged = features.merge(stores, how="inner", on="Store")
df = train.merge(feature_store_merged, on=["Store", "Date", "IsHoliday"])
df["Date"] = pd.to_datetime(df["Date"])
df["Week_Of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
df["Year"] = df["Date"].dt.year

st.title("🛒 Walmart Weekly Sales Predictor")

# tabs
tab1, tab2 = st.tabs(["📊 EDA", "🔮 Predict"])

with tab1:
    st.subheader("Total Weekly Sales by Week of Year")
    total_weekly = df.groupby("Week_Of_Year", as_index=False)["Weekly_Sales"].sum()
    fig1 = px.line(total_weekly, x="Week_Of_Year", y="Weekly_Sales",
                   labels={"Week_Of_Year": "Week of Year", "Weekly_Sales": "Total Sales ($)"})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Sales by Store Type")
    type_sales = df.groupby(["Week_Of_Year", "Type"], as_index=False)["Weekly_Sales"].sum()
    fig2 = px.line(type_sales, x="Week_Of_Year", y="Weekly_Sales", color="Type",
                   labels={"Week_Of_Year": "Week of Year", "Weekly_Sales": "Total Sales ($)"})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Year")
    yearly = df.groupby(["Year", "Week_Of_Year"], as_index=False)["Weekly_Sales"].sum()
    fig3 = px.line(yearly, x="Week_Of_Year", y="Weekly_Sales", color="Year",
                   labels={"Week_Of_Year": "Week of Year", "Weekly_Sales": "Total Sales ($)"})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Holiday Impact on Sales")
    holiday_effect = df.groupby("IsHoliday", as_index=False)["Weekly_Sales"].mean()
    holiday_effect["IsHoliday"] = holiday_effect["IsHoliday"].map({True: "Holiday", False: "Non-Holiday",
                                                                     1: "Holiday", 0: "Non-Holiday"})
    fig4 = px.bar(holiday_effect, x="IsHoliday", y="Weekly_Sales", text_auto=".2s",
                  labels={"IsHoliday": "Week Type", "Weekly_Sales": "Avg Sales ($)"})
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Markdowns vs Sales")
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    df[markdown_cols] = df[markdown_cols].fillna(0)
    weekly_md = df.groupby("Week_Of_Year", as_index=False).agg(
        {"Weekly_Sales": "sum", **{c: "sum" for c in markdown_cols}}
    )
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=weekly_md["Week_Of_Year"], y=weekly_md["Weekly_Sales"],
                              name="Total Sales", yaxis="y1", line=dict(width=3)))
    for md in markdown_cols:
        fig5.add_trace(go.Scatter(x=weekly_md["Week_Of_Year"], y=weekly_md[md],
                                  name=md, yaxis="y2"))
    fig5.update_layout(
        yaxis=dict(title="Total Sales ($)"),
        yaxis2=dict(title="MarkDown ($)", overlaying="y", side="right"),
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig5, use_container_width=True)

with tab2:
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
        markdown1 = st.number_input("MarkDown1", value=0.0)
        markdown2 = st.number_input("MarkDown2", value=0.0)
        markdown3 = st.number_input("MarkDown3", value=0.0)
        markdown4 = st.number_input("MarkDown4", value=0.0)
        markdown5 = st.number_input("MarkDown5", value=0.0)

    THANKSGIVING_WEEK = {2010: pd.Timestamp("2010-11-26"), 2011: pd.Timestamp("2011-11-25"),
                         2012: pd.Timestamp("2012-11-23"), 2013: pd.Timestamp("2013-11-29")}
    CHRISTMAS_WEEK    = {2010: pd.Timestamp("2010-12-31"), 2011: pd.Timestamp("2011-12-30"),
                         2012: pd.Timestamp("2012-12-28"), 2013: pd.Timestamp("2013-12-27")}
    SUPER_BOWL_WEEK   = {2010: pd.Timestamp("2010-02-07"), 2011: pd.Timestamp("2011-02-06"),
                         2012: pd.Timestamp("2012-02-05"), 2013: pd.Timestamp("2013-02-03")}
    LABOR_DAY_WEEK    = {2010: pd.Timestamp("2010-09-10"), 2011: pd.Timestamp("2011-09-09"),
                         2012: pd.Timestamp("2012-09-07"), 2013: pd.Timestamp("2013-09-06")}

    date = pd.Timestamp.fromisocalendar(year, week_of_year, 1)

    if st.button("Predict Weekly Sales 🚀"):
        input_data = pd.DataFrame([{
            "Store": store, "Dept": dept, "IsHoliday": is_holiday,
            "Temperature": temperature, "Fuel_Price": fuel_price,
            "MarkDown1": markdown1, "MarkDown2": markdown2, "MarkDown3": markdown3,
            "MarkDown4": markdown4, "MarkDown5": markdown5,
            "CPI": cpi, "Unemployment": unemployment,
            "Type": store_type, "Size": size, "Year": year,
            "Week_Of_Year": week_of_year,
            "Days_to_Thanksgiving": (THANKSGIVING_WEEK[year] - date).days,
            "Days_to_Christmas":    (CHRISTMAS_WEEK[year] - date).days,
            "Days_to_SuperBowl":    (SUPER_BOWL_WEEK[year] - date).days,
            "Days_to_LaborDay":     (LABOR_DAY_WEEK[year] - date).days,
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Weekly Sales: **${prediction:,.2f}**")
@st.cache_data
def load_data():
    train = pd.read_csv("./Data/train.csv")
    features = pd.read_csv("./Data/features.csv")
    stores = pd.read_csv("./Data/stores.csv")
    
    feature_store_merged = features.merge(stores, how="inner", on="Store")
    df = train.merge(feature_store_merged, on=["Store", "Date", "IsHoliday"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week_Of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"] = df["Date"].dt.year
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    df[markdown_cols] = df[markdown_cols].fillna(0)
    return df

@st.cache_resource
def load_model():
    model = joblib.load("walmart_xgb_model.pkl")
    model.named_steps["Model"].set_params(device="cpu")
    return model

df = load_data()
model = load_model()