

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AI Stock Trend Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI-Based Stock Market Trend Analyzer")
st.write(
    "Analyze historical stock prices, technical indicators, "
    "and machine-learning trend classifications."
)
st.caption(
    "Educational project only. Results are estimates, not "
    "guaranteed predictions or investment advice."
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Stock Settings")

stock_options = {
    "TCS (India)": "TCS.NS",
    "Reliance Industries (India)": "RELIANCE.NS",
    "Infosys (India)": "INFY.NS",
    "HDFC Bank (India)": "HDFCBANK.NS",
    "Apple (USA)": "AAPL",
    "Microsoft (USA)": "MSFT",
    "Tesla (USA)": "TSLA"
}

selected_stock = st.sidebar.selectbox(
    "Select a stock",
    list(stock_options.keys())
)

ticker = stock_options[selected_stock]

period = st.sidebar.selectbox(
    "Historical data period",
    ["1y", "2y", "5y", "10y"],
    index=2
)

# ---------------- DOWNLOAD DATA ----------------
@st.cache_data(ttl=900)
def load_data(symbol, history_period):
    data = yf.download(
        symbol,
        period=history_period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    # Handle different yfinance column formats
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[:, ~data.columns.duplicated()]

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    data = data[required].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    data = data[data["Close"] > 0]

    return data


with st.spinner("Downloading stock data..."):
    try:
        df = load_data(ticker, period)
    except Exception as exc:
        st.error(f"Unable to download data: {exc}")
        st.stop()

if df.empty or len(df) < 100:
    st.error(
        "Not enough historical data. Check your internet "
        "connection or select another stock."
    )
    st.stop()

# ---------------- TECHNICAL INDICATORS ----------------
df["Daily Return (%)"] = df["Close"].pct_change() * 100
df["SMA 20"] = df["Close"].rolling(20).mean()
df["SMA 50"] = df["Close"].rolling(50).mean()

# Calculate RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)
df["RSI"] = 100 - (100 / (1 + rs))

# Additional features
df["High-Low (%)"] = (
    (df["High"] - df["Low"]) / df["Close"] * 100
)

df["Close-Open (%)"] = (
    (df["Close"] - df["Open"]) / df["Open"] * 100
)

df["Return Lag 1"] = df["Daily Return (%)"].shift(1)
df["Return Lag 2"] = df["Daily Return (%)"].shift(2)
df["Volume Change (%)"] = df["Volume"].pct_change() * 100

feature_cols = [
    "Daily Return (%)",
    "High-Low (%)",
    "Close-Open (%)",
    "SMA 20",
    "SMA 50",
    "RSI",
    "Return Lag 1",
    "Return Lag 2",
    "Volume Change (%)"
]

# Target: 1 if the next closing price rises, otherwise 0.
# The last row has no known next-day outcome, so it stays missing.
df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(float)

df.loc[df["Close"].shift(-1).isna(), "Target"] = np.nan

# ---------------- PREPARE MODEL DATA ----------------
model_data = df[feature_cols + ["Target"]].replace(
    [np.inf, -np.inf], np.nan
).dropna()

if len(model_data) < 100:
    st.error("Not enough clean observations to train the model.")
    st.stop()

# Use earlier observations for training and later observations
# for testing. Do not randomly shuffle time-series observations.
split_index = int(len(model_data) * 0.80)

train = model_data.iloc[:split_index]
test = model_data.iloc[split_index:]

X_train = train[feature_cols]
y_train = train["Target"].astype(int)

X_test = test[feature_cols]
y_test = test["Target"].astype(int)

if y_train.nunique() < 2:
    st.error("Training data contains only one target class.")
    st.stop()

# ---------------- TRAIN RANDOM FOREST ----------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

# ---------------- LATEST TREND CLASSIFICATION ----------------
# Use the latest row with available features, even though its
# next-day target is unknown.
latest_features_data = df[feature_cols].replace(
    [np.inf, -np.inf], np.nan
).dropna()

latest_features = latest_features_data.iloc[[-1]]
latest_date = latest_features.index[0]
latest_price = df.loc[latest_date, "Close"]
latest_rsi = df.loc[latest_date, "RSI"]

prediction = int(model.predict(latest_features)[0])

probabilities = model.predict_proba(latest_features)[0]
classes = list(model.classes_)

up_probability = (
    probabilities[classes.index(1)] if 1 in classes else 0.0
)
down_probability = (
    probabilities[classes.index(0)] if 0 in classes else 0.0
)

daily_change = df["Daily Return (%)"].loc[latest_date]

# ---------------- SUMMARY CARDS ----------------
st.subheader("Stock Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest Closing Price", f"{latest_price:,.2f}")
c2.metric("Daily Change", f"{daily_change:+.2f}%")
c3.metric("Test Accuracy", f"{accuracy:.1%}")
c4.metric("RSI (14)", f"{latest_rsi:.2f}")

st.caption(
    f"Ticker: {ticker} | Latest observation: "
    f"{latest_date.strftime('%Y-%m-%d')}"
)

# ---------------- PRICE CHART ----------------
st.subheader("Historical Price and Moving Averages")

price_fig = go.Figure()

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    name="Closing Price",
    line=dict(width=2)
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["SMA 20"],
    name="20-Day SMA"
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["SMA 50"],
    name="50-Day SMA"
))

price_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Adjusted Price",
    hovermode="x unified"
)

st.plotly_chart(price_fig, use_container_width=True)

# ---------------- TRADING VOLUME ----------------
st.subheader("Trading Volume")

volume_fig = go.Figure()

volume_fig.add_trace(go.Bar(
    x=df.index,
    y=df["Volume"],
    name="Volume"
))

volume_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Trading Volume"
)

st.plotly_chart(volume_fig, use_container_width=True)

# ---------------- RSI CHART ----------------
st.subheader("Relative Strength Index (RSI)")

rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["RSI"],
    name="RSI (14)"
))

rsi_fig.add_hline(y=70, line_dash="dash")
rsi_fig.add_hline(y=30, line_dash="dash")

rsi_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="RSI",
    yaxis_range=[0, 100]
)

st.plotly_chart(rsi_fig, use_container_width=True)

# ---------------- MODEL OUTPUT ----------------
st.subheader("AI Trend Classification")

if prediction == 1:
    st.success("Model classification: UPWARD")
else:
    st.warning("Model classification: DOWNWARD / NOT UPWARD")

m1, m2 = st.columns(2)

m1.metric("Model probability: Upward", f"{up_probability:.1%}")
m2.metric("Model probability: Downward", f"{down_probability:.1%}")

st.caption(
    "This classification concerns the next trading day's "
    "direction relative to the latest observed close. Model "
    "probabilities are not guaranteed real-world probabilities."
)

# ---------------- MODEL EVALUATION ----------------
st.subheader("Model Evaluation")

report = classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Down/Not Up", "Up"],
    output_dict=True,
    zero_division=0
)

st.dataframe(
    pd.DataFrame(report).transpose().round(3),
    use_container_width=True
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

matrix_fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=["Predicted Down/Not Up", "Predicted Up"],
    y=["Actual Down/Not Up", "Actual Up"],
    text=matrix,
    texttemplate="%{text}",
    colorscale="Blues"
))

matrix_fig.update_layout(title="Confusion Matrix")

st.plotly_chart(matrix_fig, use_container_width=True)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("Machine-Learning Feature Importance")

importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

importance_fig = go.Figure(go.Bar(
    x=importance["Importance"],
    y=importance["Feature"],
    orientation="h"
))

importance_fig.update_layout(
    xaxis_title="Relative Importance",
    yaxis_title="Feature"
)

st.plotly_chart(importance_fig, use_container_width=True)

# ---------------- DOWNLOAD DATA ----------------
st.subheader("Download Analyzed Data")

download_df = df.copy()
download_df.index.name = "Date"

csv_data = download_df.to_csv().encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"{ticker.replace('.', '_')}_analysis.csv",
    mime="text/csv"
)

st.info(
    "This application is an educational project. Historical "
    "model performance does not guarantee future results."
)
