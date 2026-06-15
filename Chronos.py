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
# DATA LOADER (SAFE)
# =========================
def load_data(ticker):
    df = yf.download(ticker, period=PERIOD, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    df = df[["date", "close"]].dropna()
    df["close"] = df["close"].astype(np.float32)

    return df

# =========================
# CHRONOS FORECAST (100% SAFE)
# =========================
def chronos_forecast(series, horizon=HORIZON):

    series = np.asarray(series, dtype=np.float32)
    series = np.nan_to_num(series)

    if len(series) < 60:
        raise ValueError("Not enough data")

    inputs = torch.tensor(series).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model.predict(inputs, horizon, 10)

    # handle list output
    if isinstance(output, list):
        output = torch.stack(output)

    output = output.cpu()

    # ensure tensor
    output = torch.tensor(output)

    # median across samples
    forecast = output.median(dim=0).values

    # remove batch dimension
    forecast = forecast.squeeze().numpy()

    return forecast

# =========================
# BASELINES
# =========================
def naive(series):
    return np.full(HORIZON, series[-1], dtype=np.float32)

def sma(series, window=20):
    return np.full(HORIZON, np.mean(series[-window:]), dtype=np.float32)

# =========================
# APP UI
# =========================
st.set_page_config(layout="wide")
st.title("📊 Chronos-2 Forecast Dashboard (Stable)")

ticker = st.text_input("Ticker", "AMD")

df = load_data(ticker)
series = df["close"].values

tab1, tab2 = st.tabs(["📈 Forecast", "🧪 Debug"])

# =========================
# FORECAST TAB
# =========================
with tab1:

    st.subheader("Forecast vs History")

    if st.button("Run Forecast"):

        forecast = chronos_forecast(series)

        # 🔥 HARD FIX: make absolutely safe
        forecast = np.asarray(forecast).reshape(-1)
        forecast = np.nan_to_num(forecast)

        # DEBUG (WICHTIG!)
        st.write("Forecast preview:", forecast)

        future_dates = pd.date_range(
            start=df["date"].iloc[-1],
            periods=HORIZON + 1,
            freq="B"
        )[1:]

        fig = go.Figure()

        # History
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=series,
            name="History",
            line=dict(color="blue")
        ))

        # Forecast (VISIBLE FIX 🔥)
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            name="Chronos Forecast",
            line=dict(color="orange", width=4),
            mode="lines+markers"
        ))

        # SMA
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=sma(series),
            name="SMA20",
            line=dict(color="green", dash="dot")
        ))

        # Naive
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=naive(series),
            name="Naive",
            line=dict(color="gray", dash="dot")
        ))

        fig.update_layout(
            template="plotly_white",
            title=f"{ticker} Forecast (Chronos-2)",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# DEBUG TAB
# =========================
with tab2:

    st.subheader("Debug Info")

    st.write("Data shape:", series.shape)
    st.write("Last values:", series[-5:])

    try:
        test = chronos_forecast(series)
        st.write("Forecast OK:", test[:5])
    except Exception as e:
        st.error(str(e))
