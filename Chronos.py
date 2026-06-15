import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import torch

from chronos import BaseChronosPipeline

# =========================
# CONFIG
# =========================
HORIZON = 20
PERIOD = "1y"
STEP = 20
MIN_HISTORY = 120

# =========================
# MODEL (cached)
# =========================
@st.cache_resource
def load_model():
    return BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu"
    )

model = load_model()

# =========================
# DATA LOADER (CLOUD SAFE)
# =========================
def chronos_forecast(series: np.ndarray, horizon=20):

    import torch
    import numpy as np

    series = np.asarray(series, dtype=np.float32)
    series = series[~np.isnan(series)]

    if len(series) < 50:
        raise ValueError("Not enough data")

    inputs = torch.tensor(series).float().unsqueeze(0)  # 🔥 wichtig: batch dim

    with torch.no_grad():
        forecast = model.predict(
            inputs=inputs,
            prediction_length=horizon,
            num_samples=10
        )

    return forecast.median(dim=1).values.squeeze(0).cpu().numpy()

# =========================
# SAFE CHRONOS FORECAST
# =========================
def chronos_forecast(series: np.ndarray, horizon=HORIZON):

    series = np.asarray(series, dtype=np.float32)
    series = series[~np.isnan(series)]

    if len(series) < 50:
        raise ValueError("Not enough data for Chronos")

    context = torch.tensor(series).float().flatten()

    with torch.no_grad():
        forecast = model.predict(
            context=context,
            prediction_length=horizon,
            num_samples=10
        )

    return forecast.median(dim=0).values.cpu().numpy()

# =========================
# BASELINES
# =========================
def naive_forecast(last_value, horizon):
    return np.full(horizon, last_value, dtype=np.float32)

def sma_forecast(series, horizon, window=20):
    series = np.asarray(series, dtype=np.float32)
    return np.full(horizon, np.mean(series[-window:]))

# =========================
# BACKTESTING (SAFE)
# =========================
def backtest(series: np.ndarray):
    series = np.asarray(series, dtype=np.float32)
    series = series[~np.isnan(series)]

    results = {"chronos": [], "naive": [], "sma": []}

    for t in range(MIN_HISTORY, len(series) - HORIZON, STEP):

        context = series[:t]
        actual = series[t:t + HORIZON]

        if len(context) < 60 or len(actual) < HORIZON:
            continue

        try:
            ch = chronos_forecast(context)
        except:
            continue

        na = naive_forecast(series[t - 1], HORIZON)
        sm = sma_forecast(context, HORIZON)

        results["chronos"].append(np.mean(np.abs(ch - actual)))
        results["naive"].append(np.mean(np.abs(na - actual)))
        results["sma"].append(np.mean(np.abs(sm - actual)))

    return {
        k: float(np.mean(v)) if len(v) > 0 else None
        for k, v in results.items()
    }

# =========================
# UI
# =========================
st.set_page_config(page_title="Chronos-2 Stock Lab", layout="wide")

st.title("📊 Chronos-2 Stock Forecast Lab (Stable Version)")

ticker = st.text_input("Ticker", "AMD")

try:
    df = load_data(ticker)
    series = df["close"].values

    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🧪 Backtest", "📊 Comparison"])

    # =========================
    # FORECAST TAB
    # =========================
    with tab1:
        st.subheader("20-Day Forecast")

        if st.button("Run Forecast"):

            forecast = chronos_forecast(series)

            future_dates = pd.date_range(
                start=df["date"].iloc[-1],
                periods=HORIZON + 1,
                freq="B"
            )[1:]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["date"],
                y=series,
                name="History"
            ))

            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast,
                name="Chronos Forecast"
            ))

            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # BACKTEST TAB
    # =========================
    with tab2:
        st.subheader("Walk-forward Backtesting")

        if st.button("Run Backtest"):
            result = backtest(series)
            st.json(result)

    # =========================
    # COMPARISON TAB
    # =========================
    with tab3:
        st.subheader("Model Comparison (Last Window)")

        context = series[:-HORIZON]
        actual = series[-HORIZON:]

        ch = chronos_forecast(context)
        na = naive_forecast(series[-HORIZON - 1], HORIZON)
        sm = sma_forecast(context, HORIZON)

        fig = go.Figure()

        fig.add_trace(go.Scatter(y=actual, name="Actual"))
        fig.add_trace(go.Scatter(y=ch, name="Chronos"))
        fig.add_trace(go.Scatter(y=na, name="Naive"))
        fig.add_trace(go.Scatter(y=sm, name="SMA20"))

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
