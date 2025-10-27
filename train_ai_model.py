#!/usr/bin/env python3
"""
train_ai_model_v13.1.py
------------------------------------
Crypto AI Model Training Script
• Fetches crypto data (BTC, XRP, GALA) via Yahoo Finance
• Cleans and trains lightweight ML models
• Saves model, scaler, and summary metrics
• Compatible with GitHub Actions workflow
"""

import os
import json
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === CONFIGURATION ===
CRYPTO_SYMBOLS = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]
LOOKBACK_DAYS = 90
INTERVAL = "1h"
DATA_DIR = "data"
MODEL_DIR = "models"
SUMMARY_DIR = "training_summary"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

print("[RUN] Starting model training...")

# === FETCH FUNCTION ===
def fetch_data(symbol):
    try:
        # Ensure symbol is a string
        if isinstance(symbol, (list, tuple)):
            symbol = symbol[0]

        yf_symbol = symbol
        # Yahoo Finance uses -USD for crypto tickers
        if yf_symbol.endswith("USDT"):
            yf_symbol = yf_symbol.replace("USDT", "-USD")
        elif not yf_symbol.endswith("-USD"):
            yf_symbol = yf_symbol + "-USD"

        print(f"[{datetime.utcnow().isoformat()}] [FETCH] Downloading {yf_symbol} from Yahoo Finance...")

        df = yf.download(
            yf_symbol,
            period=f"{LOOKBACK_DAYS}d",
            interval=INTERVAL,
            progress=False,
            auto_adjust=False  # explicitly prevent warning
        )

        if df is None or df.empty:
            raise ValueError(f"No data fetched for {yf_symbol}")

        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df.to_csv(os.path.join(DATA_DIR, f"{symbol}.csv"), index=False)
        return df

    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}] [ERROR] Could not fetch data for {symbol}: {e}")
        return pd.DataFrame()

# === TRAIN FUNCTION ===
def train_model(symbol, df):
    try:
        print(f"[{datetime.utcnow().isoformat()}] [TRAIN] Starting model training for {symbol}...")

        # Prepare features
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        X = df[["Open", "High", "Low", "Volume"]].values
        y = df["Target"].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)

        model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"[{datetime.utcnow().isoformat()}] [TRAIN] Model trained successfully for {symbol}")
        return {"symbol": symbol, "mae": mae, "rmse": rmse}

    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}] [ERROR] Training failed for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}

# === MAIN RUN ===
def main():
    start_time = datetime.utcnow().isoformat()
    print(f"[{start_time}] [START] Training run — {start_time}")

    results = []
    for symbol in CRYPTO_SYMBOLS:
        df = fetch_data(symbol)
        if df.empty:
            continue
        metrics = train_model(symbol, df)
        results.append(metrics)

    summary_file = os.path.join(SUMMARY_DIR, f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "symbols_trained": [r["symbol"] for r in results],
            "results": results
        }, f, indent=4)

    print(f"[{datetime.utcnow().isoformat()}] [DONE] Training telemetry saved → {summary_file}")


if __name__ == "__main__":
    main()
