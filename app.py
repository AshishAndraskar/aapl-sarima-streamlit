# app.py
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
    m = st.number_input("m (seasonal period, e.g., 12 for monthly, 5 for weekly)", min_value=1, max_value=365, value=5, step=1)
    forecast_horizon = st.number_input("Days to forecast", min_value=1, max_value=365, value=15, step=1)

with st.sidebar.expander("Model persistence"):
    model_filename = st.text_input("Model filename (for save/load)", value="sarima_aapl.joblib")
    use_saved = st.checkbox("Load saved model if available", value=False)

train_btn = st.sidebar.button("Train / Re-train model")
forecast_btn = st.sidebar.button("Generate forecast")

# ----- Data load -----
@st.cache_data(ttl=60*60)  # cache for 1 hour
def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None
    df = df[['Close']].dropna()
    df.index = pd.to_datetime(df.index)
    return df

data_load_state = st.text("Loading data...")
data = load_data(ticker, start_date, end_date)
if data is None:
    st.error("No data found for the selected ticker/dates. Check ticker symbol or date range.")
    st.stop()
data_load_state.text("Data loaded ✅")

# Show sample and chart
st.subheader("Historical Close Price")
st.line_chart(data['Close'])

# ----- Model utilities -----
def train_sarima(series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=200)
    return res

# Optionally load saved model
model = None
if use_saved:
    try:
        model = joblib.load(model_filename)
        st.sidebar.success(f"Loaded model from {model_filename}")
    except Exception as e:
        st.sidebar.warning(f"Could not load model: {e}")

# Train or re-train
if train_btn or model is None:
    st.info("Training SARIMA model — this may take a while depending on data length and parameters...")
    order = (int(p), int(d), int(q))
    seasonal_order = (int(P), int(D), int(Q), int(m))
    try:
        model = train_sarima(data['Close'], order, seasonal_order)
        st.success("Model trained successfully!")
        # Save model
        try:
            joblib.dump(model, model_filename)
            st.sidebar.info(f"Model saved to {model_filename}")
        except Exception as e:
            st.sidebar.warning(f"Failed to save model: {e}")
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

# Forecast
if forecast_btn:
    if model is None:
        st.error("No trained model available. Train the model first.")
        st.stop()
    steps = int(forecast_horizon)
    st.info(f"Generating forecast for {steps} steps...")
    try:
        # get forecast
        pred = model.get_forecast(steps=steps)
        mean_forecast = pred.predicted_mean
        conf = pred.conf_int(alpha=0.05)
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='B')  # business days
        forecast_df = pd.DataFrame({
            "Forecast": mean_forecast.values,
            "Lower CI": conf.iloc[:, 0].values,
            "Upper CI": conf.iloc[:, 1].values
        }, index=forecast_index)
        st.subheader("Forecast (next days)")
        st.line_chart(pd.concat([data['Close'].rename('Actual'), forecast_df['Forecast']], axis=0))
        st.write(forecast_df)
    except Exception as e:
        st.error(f"Forecast failed: {e}")

# ----- Optional: Evaluate on a holdout -----
if st.checkbox("Show simple train/test evaluation (last N days as test)", value=False):
    n_test = st.number_input("Test set size (days)", min_value=5, max_value=180, value=30)
    if len(data) > n_test + 10:
        train_series = data['Close'][:-n_test]
        test_series = data['Close'][-n_test:]
        st.write("Retraining on training set for evaluation...")
        order = (int(p), int(d), int(q))
        seasonal_order = (int(P), int(D), int(Q), int(m))
        try:
            eval_model = train_sarima(train_series, order, seasonal_order)
            preds = eval_model.get_forecast(steps=n_test).predicted_mean
            rmse = np.sqrt(mean_squared_error(test_series.values, preds.values))
            st.write(f"RMSE on test set: {rmse:.4f}")
            # plot
            plt.figure(figsize=(10,4))
            plt.plot(train_series.index, train_series.values, label='Train')
            plt.plot(test_series.index, test_series.values, label='Test (actual)')
            plt.plot(test_series.index, preds.values, label='Predicted')
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
    else:
        st.warning("Not enough data for the chosen test size.")

st.write("App by you — tweak SARIMA parameters or dates in the sidebar and retrain.")
