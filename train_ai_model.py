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

# Demo base URL for CoinGecko (required for free/demo keys)
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

def log(msg, level="INFO"):
    """Timestamped colored log"""
    color = Fore.GREEN if level == "INFO" else Fore.YELLOW if level == "WARN" else Fore.RED
    print(f"{color}[{datetime.utcnow().isoformat()}] [{level}] {msg}{Style.RESET_ALL}")

# === FETCHERS ===

def fetch_binance(symbol, interval=INTERVAL, limit=1000):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Invalid Binance response")
        df = pd.DataFrame(data, columns=[
            "timestamp","open","high","low","close","volume",
            "_1","_2","_3","_4","_5","_6"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        log(f"Binance fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

def fetch_coingecko(symbol):
    """Fetch market chart from CoinGecko (free/demo API compatible)"""
    try:
        coin_map = {"BTCUSDT": "bitcoin", "XRPUSDT": "ripple", "GALAUSDT": "gala"}
        if symbol not in coin_map:
            raise ValueError("Unsupported symbol for CoinGecko")
        coin_id = coin_map[symbol]
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": "60"}  # ✅ added vs_currency
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "prices" not in data:
            raise ValueError(f"Invalid response: {data}")
        prices = pd.DataFrame(data["prices"], columns=["timestamp","price"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp","volume"])
        df = prices.merge(volumes, on="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.rename(columns={"price":"close"}, inplace=True)
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"].rolling(3).max()
        df["low"] = df["close"].rolling(3).min()
        df = df.dropna().set_index("timestamp")
        return df
    except Exception as e:
        log(f"CoinGecko fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

def safe_download_yahoo(symbol, period="60d", interval="1h", retries=3, delay=5):
    for attempt in range(retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
            if not df.empty:
                return df
            log(f"Empty Yahoo data for {symbol} (attempt {attempt+1})", "WARN")
        except Exception as e:
            log(f"Yahoo fetch failed for {symbol} (attempt {attempt+1}): {e}", "WARN")
        time.sleep(delay)
    return pd.DataFrame()

# === CACHE HANDLING ===

def load_from_cache(symbol):
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
            log(f"[CACHE] Loaded {len(df)} rows for {symbol}")
            return df
        except Exception:
            pass
    return pd.DataFrame()

def save_to_cache(symbol, df):
    if df.empty:
        return
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    df.to_csv(cache_path)
    log(f"[CACHE] Saved {len(df)} rows for {symbol}")

# === DATA PIPELINE ===

def get_market_data(symbol):
    """Try Binance → CoinGecko → Yahoo → cache"""
    df = fetch_binance(symbol)
    if df.empty:
        df = fetch_coingecko(symbol)
    if df.empty:
        yf_symbol = symbol.replace("USDT","-USD")
        df = safe_download_yahoo(yf_symbol)
    if df.empty:
        df = load_from_cache(symbol)
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
    X = df[["open","high","low","close","volume","m]()]()
