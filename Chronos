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
MIN_HISTORY = 200

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
# DATA
# =========================
def load_data(ticker: str):
    df = yf.download(ticker, period=PERIOD)
    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df = df[["date", "close"]].dropna()
    return df

# =========================
# FORECAST
# =========================
def chronos_forecast(series: np.ndarray, horizon=HORIZON):
    tensor = torch.tensor(series).float()

    forecast = model.predict(
        context=tensor,
        prediction_length=horizon,
        num_samples=10
    )

    return forecast.median(dim=0).values.numpy()

# =========================
# BASELINES
# =========================
def naive_forecast(last_value, horizon):
    return np.full(horizon, last_value)

def sma_forecast(series, horizon, window=20):
    return np.full(horizon, np.mean(series[-window:]))

# =========================
# BACKTESTING
# =========================
def backtest(series: np.ndarray):
    errors = {"chronos": [], "naive": [], "sma": []}

    for t in range(MIN_HISTORY, len(series) - HORIZON, STEP):
        context = series[:t]
        actual = series[t:t + HORIZON]

        try:
            ch = chronos_forecast(context)
        except Exception:
            continue

        na = naive_forecast(series[t - 1], HORIZON)
        sm = sma_forecast(context, HORIZON)

        errors["chronos"].append(np.mean(np.abs(ch - actual)))
        errors["naive"].append(np.mean(np.abs(na - actual)))
        errors["sma"].append(np.mean(np.abs(sm - actual)))

    return {k: float(np.mean(v)) if len(v) > 0 else None for k, v in errors.items()}

# =========================
# UI
# =========================
st.set_page_config(page_title="Chronos-2 Stock Lab", layout="wide")

st.title("📊 Chronos-2 Stock Forecast Lab")

ticker = st.text_input("Ticker eingeben", "AMD")

df = load_data(ticker)
series = df["close"].values

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🧪 Backtesting", "📊 Comparison"])

# =========================
# TAB 1 - FORECAST
# =========================
with tab1:
    st.subheader("20-Tage Forecast (Chronos-2)")

    if st.button("Forecast starten"):
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
# TAB 2 - BACKTESTING
# =========================
with tab2:
    st.subheader("Walk-forward Backtesting")

    if st.button("Backtest starten"):
        result = backtest(series)

        st.write("### MAE Vergleich")
        st.json(result)

# =========================
# TAB 3 - COMPARISON
# =========================
with tab3:
    st.subheader("Model Comparison (letzte 20 Tage)")

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
