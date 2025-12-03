import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import joblib
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AAPL SARIMA Forecast", layout="wide")

st.title("📈 Apple (AAPL) Stock Price Prediction — SARIMA")

# ----- Sidebar controls -----
st.sidebar.header("Settings")

with st.sidebar.expander("Data"):
    start_date = st.date_input("Start date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End date (use today for latest)", value=pd.to_datetime("today"))
    ticker = st.text_input("Ticker", value="AAPL")

with st.sidebar.expander("Model (SARIMA)"):
    p = st.number_input("p (AR order)", min_value=0, max_value=10, value=1, step=1)
    d = st.number_input("d (diff order)", min_value=0, max_value=2, value=1, step=1)
    q = st.number_input("q (MA order)", min_value=0, max_value=10, value=1, step=1)
    P = st.number_input("P (seasonal AR)", min_value=0, max_value=5, value=1, step=1)
    D = st.number_input("D (seasonal diff)", min_value=0, max_value=2, value=1, step=1)
    Q = st.number_input("Q (seasonal MA)", min_value=0, max_value=5, value=1, step=1)
    m = st.number_input("m (seasonal period)", min_value=1, max_value=365, value=5, step=1)
    forecast_horizon = st.number_input("Days to forecast", min_value=1, max_value=365, value=15, step=1)

with st.sidebar.expander("Model persistence"):
    model_filename = st.text_input("Model filename", value="sarima_aapl.joblib")
    use_saved = st.checkbox("Load saved model if available", value=False)

train_btn = st.sidebar.button("Train / Re-train model")
forecast_btn = st.sidebar.button("Generate forecast")

# ----- Data load -----
@st.cache_data(ttl=60*60)
def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None
    df = df[['Close']].dropna()
    df.index = pd.to_datetime(df.index)
    return df

data = load_data(ticker, start_date, end_date)
if data is None:
    st.error("No data found for the selected ticker/dates.")
    st.stop()

st.subheader("Historical Close Price")
st.line_chart(data['Close'])

# ----- Train model -----
def train_sarima(series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=200)
    return res

model = None
if use_saved:
    try:
        model = joblib.load(model_filename)
        st.sidebar.success("Loaded saved model.")
    except:
        st.sidebar.warning("No saved model found.")

if train_btn or model is None:
    st.info("Training SARIMA model...")
    order = (p, d, q)
    seasonal_order = (P, D, Q, m)

    try:
        model = train_sarima(data['Close'], order, seasonal_order)
        st.success("Model trained successfully!")

        joblib.dump(model, model_filename)
        st.sidebar.info("Model saved!")

    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

# ----- Forecast -----
if forecast_btn:
    try:
        forecast = model.get_forecast(steps=forecast_horizon)
        mean_forecast = forecast.predicted_mean
        conf = forecast.conf_int()

        forecast_df = pd.DataFrame({
            "Forecast": mean_forecast,
            "Lower CI": conf.iloc[:, 0],
            "Upper CI": conf.iloc[:, 1]
        })

        st.subheader("Forecast Results")
        st.line_chart(forecast_df["Forecast"])
        st.write(forecast_df)

    except Exception as e:
        st.error(f"Forecast failed: {e}")
