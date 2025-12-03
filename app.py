import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm

# -----------------------------
# Page Title
# -----------------------------
st.title("📈 Apple Stock Price Forecasting (SARIMA Model)")
st.write("Predict next 30 days of Apple stock prices using SARIMA model.")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload Apple Stock CSV (2012–2019)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    
    st.subheader("📊 Raw Data Preview")
    st.write(df.head())

    # -----------------------------
    # Select date range for training
    # -----------------------------
    st.subheader("📅 Select training date range")
    min_date = df.index.min()
    max_date = df.index.max()

    start = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    if start >= end:
        st.error("Start date must be before end date")
    else:
        train_df = df.loc[start:end]
        
        st.write(f"Training Data from {start} to {end}")
        st.line_chart(train_df["Close"])

        # -----------------------------
        # Train SARIMA model
        # -----------------------------
        st.subheader("🔧 Training SARIMA Model...")

        try:
            model = sm.tsa.statespace.SARIMAX(
                train_df["Close"],
                order=(2, 1, 2),
                seasonal_order=(1, 1, 1, 12)
            )
            model_fit = model.fit(disp=False)
            st.success("Model training completed!")
        except Exception as e:
            st.error(f"Error in model training: {e}")

        # -----------------------------
        # Forecast next 30 days
        # -----------------------------
        st.subheader("📈 30-Day Forecast")

        forecast_days = 30
        forecast = model_fit.forecast(steps=forecast_days)

        forecast_df = pd.DataFrame({
            "Date": pd.date_range(start=train_df.index[-1] + pd.Timedelta(days=1), periods=forecast_days),
            "Forecasted Close Price": forecast.values
        }).set_index("Date")

        st.write(forecast_df)

        st.line_chart(forecast_df)

        # Download button
        st.download_button(
            label="Download Forecast CSV",
            data=forecast_df.to_csv(),
            file_name="apple_forecast_30_days.csv",
            mime="text/csv"
        )
else:
    st.info("👆 Upload the Apple stock CSV file to begin.")
