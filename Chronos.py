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

# =========================
# MODEL
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
def load_data(ticker):
    df = yf.download(ticker, period=PERIOD, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={"date": "date", "close": "close"})
    df = df[["date", "close"]].dropna()
    df["close"] = df["close"].astype(np.float32)

    return df

# =========================
# CHRONOS FORECAST (SAFE)
# =========================
def chronos_forecast(series, horizon=HORIZON):

    series = np.asarray(series, dtype=np.float32)
    series = series[~np.isnan(series)]

    inputs = torch.tensor(series).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model.predict(
            inputs,
            horizon,
            10
        )

    # handle list or tensor
    if isinstance(output, list):
        output = torch.stack(output)

    output = output.cpu()

    # expected: [samples, batch, horizon]
    median = output.median(dim=0).values.squeeze(0).numpy()

    return median

# =========================
# BASELINES
# =========================
def naive(series):
    return np.full(HORIZON, series[-1])

def sma(series, window=20):
    return np.full(HORIZON, np.mean(series[-window:]))

# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.title("📊 Chronos-2 Professional Forecast Dashboard")

ticker = st.text_input("Ticker", "AMD")

df = load_data(ticker)
series = df["close"].values

tab1, tab2 = st.tabs(["📈 Forecast", "🧪 Backtest"])

# =========================
# FORECAST TAB
# =========================
with tab1:

    st.subheader("Forecast + History")

    if st.button("Run Forecast"):

        forecast = chronos_forecast(series)

        # FIX: proper timeline alignment
        future_dates = pd.date_range(
            start=df["date"].iloc[-1],
            periods=HORIZON + 1,
            freq="B"
        )[1:]

        fig = go.Figure()

        # HISTORY
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=series,
            name="History",
            line=dict(color="blue")
        ))

        # FORECAST
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            name="Chronos Forecast",
            line=dict(color="orange", dash="dash")
        ))

        # SMA baseline
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=sma(series),
            name="SMA20",
            line=dict(color="green", dash="dot")
        ))

        # Naive baseline
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=naive(series),
            name="Naive",
            line=dict(color="gray", dash="dot")
        ))

        fig.update_layout(
            title=f"{ticker} Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# BACKTEST TAB
# =========================
with tab2:

    st.subheader("Simple Backtest (MAE)")

    if st.button("Run Backtest"):

        errors = []

        for i in range(120, len(series) - HORIZON, 20):

            context = series[:i]
            actual = series[i:i+HORIZON]

            try:
                pred = chronos_forecast(context)
                errors.append(np.mean(np.abs(pred - actual)))
            except:
                continue

        st.metric("MAE (Chronos)", round(np.mean(errors), 4))
