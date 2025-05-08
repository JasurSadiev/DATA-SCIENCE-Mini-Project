from datetime import date
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model  # type: ignore

# Load trained model
model = load_model("stock_price_prediction.h5")

# Streamlit layout
st.title("📈 Stock Price Prediction (in USD)")
st.write("Enter the stock ticker symbol (e.g., AAPL, TSLA, MSFT) to begin.")

ticker = st.text_input("Ticker Symbol", "")
search = st.button("🔍 Search")

if ticker and search:
    currentYear = date.today().year
    today = date.today().strftime("%Y-%m-%d")

    stock = yf.Ticker(ticker)
    try:
        fast_info = stock.fast_info
    except Exception:
        st.error(
            f"❌ Failed to fetch stock info for {ticker}. Please wait a while and try again."
        )
        st.stop()

    st.subheader(f"Quick Information about {ticker}")
    st.write(
        f"*Last Price:* {fast_info.get('last_price'):.2f} USD"
        if fast_info.get("last_price")
        else "N/A"
    )
    st.write(
        f"*Market Cap:* {fast_info.get('market_cap'):,} USD"
        if fast_info.get("market_cap")
        else "N/A"
    )
    st.write(f"*Volume:* {fast_info.get('volume')}")
    st.write(f"*Exchange:* {fast_info.get('exchange')}")
    st.info("Detailed company info is not shown due to Yahoo Finance rate limits.")

    @st.cache_data
    def get_history_data(ticker, end_date):
        stock = yf.Ticker(ticker)
        return stock.history(start="2010-01-01", end=end_date, actions=False)

    try:
        df = get_history_data(ticker, today)
        df = df.drop(["Open", "High", "Volume", "Low"], axis=1)
    except Exception:
        st.error("❌ Failed to load historical stock data.")
        st.stop()

    st.subheader("📉 Price over the years")
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(df["2020-05-31":today])
    ax.set_title("Price over the years")
    ax.set_ylabel("Price in USD")
    ax.set_xlabel("Time")
    st.pyplot(fig)

    st.subheader(f"📅 Price changes in year {currentYear}")
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(df[f"{currentYear}-01-01":today], color="green")
    ax.set_title(f"Price changes in year {currentYear}")
    ax.set_ylabel("Price in USD")
    ax.set_xlabel("Months")
    st.pyplot(fig)

    # Prepare data for prediction
    data = df.values
    train_len = int(len(data) * 0.92)
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = min_max_scaler.fit_transform(data)

    test_data = scaled_data[train_len - 60 :, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = min_max_scaler.inverse_transform(predictions)

    train_data = df[:train_len]
    valid_data = df[train_len:].copy()
    valid_data["Predictions"] = predictions

    st.subheader("🔍 Model Prediction vs Actual Price")
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(valid_data["Close"])
    ax.plot(valid_data["Predictions"])
    ax.set_title("Model Prediction vs Actual Price")
    ax.set_ylabel("Price in USD")
    ax.set_xlabel("Date")
    ax.legend(["Actual Price", "Model Prediction"], loc="lower right", fontsize=15)
    st.pyplot(fig)

    # Predict tomorrow's price
    last_60_days = df[-60:].values
    last_60_days_scaled = min_max_scaler.transform(last_60_days)
    X_test = [last_60_days_scaled]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    tomorrow_prediction = model.predict(X_test)
    tomorrow_prediction = min_max_scaler.inverse_transform(tomorrow_prediction)

    st.subheader("📊 Tomorrow's Predicted Price")
    st.write(
        f"The predicted price for tomorrow is: *{tomorrow_prediction[0][0]:.2f} USD*"
    )

else:
    st.info("🔎 Please enter a ticker symbol and press *Search* to begin.")
