import os
import json
import time
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from colorama import Fore, Style

# === CONFIG ===
LOOKBACK_DAYS = 60
INTERVAL = "1h"
SYMBOLS = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]

MODEL_DIR = "models"
TELEMETRY_DIR = "training_summary"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TELEMETRY_DIR, exist_ok=True)

def log(msg, level="INFO"):
    """Consistent timestamped logging"""
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] [{level}] {msg}")

# === FETCHING LAYERS ===

def fetch_binance(symbol, interval=INTERVAL, limit=1000):
    """Fetch OHLC data from Binance API"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if not isinstance(data, list):
            raise ValueError(f"Invalid response: {data}")
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_1", "_2", "_3", "_4", "_5", "_6"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        log(f"Binance fetch failed for {symbol}: {e}", "ERROR")
        return pd.DataFrame()

def safe_download_yahoo(symbol, period="60d", interval="1h", retries=3, delay=5):
    """Yahoo Finance fallback with retry"""
    for attempt in range(retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
            if not df.empty:
                return df
            else:
                log(f"Empty Yahoo data for {symbol} (attempt {attempt+1})", "WARN")
        except Exception as e:
            log(f"Yahoo fetch failed for {symbol} (attempt {attempt+1}): {e}", "ERROR")
        time.sleep(delay)
    return pd.DataFrame()

def get_market_data(symbol):
    """Unified data fetcher with fallback"""
    df = fetch_binance(symbol)
    if df.empty:
        yf_symbol = symbol.replace("USDT", "-USD")
        log(f"Falling back to Yahoo for {yf_symbol}", "WARN")
        df = safe_download_yahoo(yf_symbol)
    return df

# === MODEL TRAINING ===

def prepare_features(df):
    """Prepare features for model training"""
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["volatility"] = df["close"].rolling(10).std()
    df = df.dropna()
    X = df[["open", "high", "low", "close", "volume", "ma5", "ma20", "volatility"]]
    y = df["close"].shift(-1).dropna()
    X, y = X.iloc[:-1], y
    return X, y

def train_model(symbol, df):
    """Train and save model"""
    try:
        X, y = prepare_features(df)
        if len(X) < 50:
            log(f"Not enough data for {symbol} ({len(X)} rows)", "WARN")
            return None
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump({"model": model, "scaler": scaler}, os.path.join(MODEL_DIR, f"{symbol}_model.pkl"))
        log(f"Model trained successfully for {symbol} ({len(X)} samples)")
        return True
    except Exception as e:
        log(f"Training failed for {symbol}: {e}", "ERROR")
        return None

# === MAIN EXECUTION ===

def main():
    start_time = datetime.utcnow()
    log(f"[START] Training run — {start_time}")
    telemetry = {"start_time": str(start_time), "symbols": {}}

    for symbol in SYMBOLS:
        log(f"[FETCH] Downloading {symbol} data...")
        df = get_market_data(symbol)
        if df.empty:
            log(f"[ERROR] No data fetched for {symbol}", "ERROR")
            telemetry["symbols"][symbol] = {"status": "failed_no_data"}
            continue
        status = train_model(symbol, df)
        telemetry["symbols"][symbol] = {"status": "success" if status else "failed"}
        time.sleep(2)

    telemetry["end_time"] = str(datetime.utcnow())
    telemetry_path = os.path.join(TELEMETRY_DIR, f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(telemetry_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    log(f"[DONE] Training telemetry saved → {telemetry_path}")
    log("[DONE] Model training complete.")

if __name__ == "__main__":
    main()
