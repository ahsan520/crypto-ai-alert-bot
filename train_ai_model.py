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
CACHE_DIR = "data_cache"
TELEMETRY_DIR = "training_summary"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TELEMETRY_DIR, exist_ok=True)

def log(msg, level="INFO"):
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] [{level}] {msg}")

# === FETCHING HELPERS ===

def fetch_binance(symbol, interval=INTERVAL, limit=1000):
    """Try to fetch from Binance"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Invalid response format from Binance")
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_1", "_2", "_3", "_4", "_5", "_6"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        log(f"Fetched {len(df)} rows from Binance for {symbol}")
        return df
    except Exception as e:
        log(f"Binance fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

def fetch_yahoo(symbol, period="60d", interval="1h"):
    """Try to fetch from Yahoo Finance"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df.empty:
            log(f"Yahoo Finance returned empty for {symbol}", "WARN")
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        df = df[["open", "high", "low", "close", "volume"]]
        log(f"Fetched {len(df)} rows from Yahoo for {symbol}")
        return df
    except Exception as e:
        log(f"Yahoo fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

def fetch_coingecko(symbol):
    """Try to fetch from CoinGecko"""
    try:
        cg_map = {"BTCUSDT": "bitcoin", "XRPUSDT": "ripple", "GALAUSDT": "gala"}
        if symbol not in cg_map:
            return pd.DataFrame()
        url = f"https://api.coingecko.com/api/v3/coins/{cg_map[symbol]}/market_chart?vs_currency=usd&days=60&interval=hourly"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame()
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["close"] = df["close"].astype(float)
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"].rolling(2).max()
        df["low"] = df["close"].rolling(2).min()
        df["volume"] = np.nan
        df = df.dropna()
        log(f"Fetched {len(df)} rows from CoinGecko for {symbol}")
        return df
    except Exception as e:
        log(f"CoinGecko fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

# === CACHE HANDLING ===

def load_cached_data(symbol):
    """Load cached data if available"""
    path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        log(f"Loaded {len(df)} cached rows for {symbol}")
        return df
    except Exception as e:
        log(f"Cache read failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

def save_to_cache(symbol, df):
    """Save successful fetch to cache"""
    path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    try:
        df.to_csv(path)
        log(f"Cache updated for {symbol} → {path}")
    except Exception as e:
        log(f"Cache write failed for {symbol}: {e}", "WARN")

# === FETCH LOGIC ===

def get_market_data(symbol):
    """Try live APIs first, then fallback to cache"""
    df = fetch_binance(symbol)
    if df.empty:
        yf_symbol = symbol.replace("USDT", "-USD")
        df = fetch_yahoo(yf_symbol)
    if df.empty:
        df = fetch_coingecko(symbol)
    if df.empty:
        log(f"All live sources failed for {symbol}, using cache", "WARN")
        df = load_cached_data(symbol)
    else:
        save_to_cache(symbol, df)
    return df

# === TRAINING ===

def prepare_features(df):
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
    try:
        X, y = prepare_features(df)
        if len(X) < 50:
            log(f"Not enough data for {symbol} ({len(X)} rows)", "WARN")
            return None
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=120, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump({"model": model, "scaler": scaler}, os.path.join(MODEL_DIR, f"{symbol}_model.pkl"))
        log(f"✅ Model trained successfully for {symbol} ({len(X)} samples)")
        return True
    except Exception as e:
        log(f"Training failed for {symbol}: {e}", "ERROR")
        return None

# === MAIN ===

def main():
    start_time = datetime.utcnow()
    log(f"[START] Training run — {start_time}")
    telemetry = {"start_time": str(start_time), "symbols": {}}

    for symbol in SYMBOLS:
        log(f"[FETCH] Processing {symbol}")
        df = get_market_data(symbol)
        if df.empty:
            log(f"[ERROR] No usable data for {symbol}", "ERROR")
            telemetry["symbols"][symbol] = {"status": "failed_no_data"}
            continue
        status = train_model(symbol, df)_
