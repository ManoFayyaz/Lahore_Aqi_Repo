import os
import pwd

# Fully bypass getpwuid error for UID 1000 (Hugging Face specific)
def patch_getpwuid():
    try:
        pwd.getpwuid(os.getuid())
    except KeyError:
        os.getuid = lambda: 0  # Override UID to 0 (root)
        os.environ["HOME"] = "/tmp"

patch_getpwuid()


import streamlit as st
import hopsworks
import pandas as pd
from prophet import Prophet
from datetime import timedelta

st.set_page_config(page_title="Lahore AQI Forecast", layout="centered")
st.title("🌫️ Lahore AQI Forecast Dashboard")
st.markdown("Predict AQI levels for the **next 3 days** using historical data and time series forecasting (Prophet).")

# 🔐 Use Hugging Face Secrets for this
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

try:
    # Login to Hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="Lahore_aqi")
    fs = project.get_feature_store()

    # Load feature group
    fg = fs.get_feature_group(name="lahore_aqi_group", version=1)
    df = fg.select_all().read()
    df.dropna(inplace=True)

    # Prepare data for Prophet
    df_prophet = df[["timestamp", "target_aqi"]].copy()
    df_prophet.rename(columns={"timestamp": "ds", "target_aqi": "y"}, inplace=True)

    # Fit model
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    # Forecast next 3 days
    future = model.make_future_dataframe(periods=3)
    forecast = model.predict(future)

    forecast_next3 = forecast.tail(3)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # Show forecast
    st.subheader("📅 Predicted AQI for Next 3 Days")
    st.dataframe(forecast_next3.rename(columns={
        "ds": "Date",
        "yhat": "Predicted AQI",
        "yhat_lower": "Lower Bound",
        "yhat_upper": "Upper Bound"
    }))

    st.subheader("📈 Forecast Trend")
    st.line_chart(forecast.set_index("ds")[["yhat"]])

except Exception as e:
    st.error(f"Something went wrong: {e}")
