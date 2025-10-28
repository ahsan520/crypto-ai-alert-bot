import os
import time
import json
import pandas as pd
import numpy as np
import joblib
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from colorama import Fore, Style

# ========== CONFIG ==========
CONFIG_FILE = "config.json"
LOOKBACK_DAYS = 30  # data window for training
CACHE_DIR = "data_cache"
MODEL_DIR = "models"

# ========== LOGGING ==========
def log(msg, level="INFO"):
    colors = {"INFO": Fore.GREEN, "WARN": Fore.YELLOW, "ERROR": Fore.RED}
    print(colors.get(level, Fore.WHITE) + f"[{level}] {msg}" + Style.RESET_ALL)

# ========== CONFIG LOADING ==========
def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to load config.json: {e}", "ERROR")
        return {}

cfg = load_config()
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== FETCH FUNCTIONS ==========

def fetch_coingecko(symbol):
    """Fetch OHLC data from CoinGecko API (Pro)"""
    try:
        api_key = os.getenv("COINGECKO_API_KEY", cfg.get("coingecko_api_key", ""))
        if not api_key:
            raise ValueError("Missing CoinGecko API key")

        mapping = {
            "BTCUSDT": "bitcoin",
            "XRPUSDT": "ripple",
            "GALAUSDT": "gala"
        }
        coin_id = mapping.get(symbol.upper())
        if not coin_id:
            raise ValueError(f"No CoinGecko mapping found for {symbol}")

        url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": LOOKBACK_DAYS}
        headers = {"accept": "application/json", "x-cg-pro-api-key": api_key}

        r = requests.get(url, headers=headers, params=params, timeout=10)
        data = r.json()

        if "prices" not in data:
            raise ValueError(f"Invalid response: {data}")

        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        df["volume"] = [v[1] for v in data.get("total_volumes", [[0, 0]])]
        df["open"] = df["high"] = df["low"] = df["close"]

        log(f"[OK] Fetched {len(df)} rows from CoinGecko for {symbol}")
        return df
    except Exception as e:
        log(f"CoinGecko fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

def fetch_binance(symbol):
    """Fetch OHLC data from Binance (public API)"""
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": LOOKBACK_DAYS * 24}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if not isinstance(data, list):
            raise ValueError("Invalid Binance response")

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        log(f"[OK] Fetched {len(df)} rows from Binance for {symbol}")
        return df
    except Exception as e:
        log(f"Binance fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

def fetch_yahoo(symbol):
    """Fallback: Fetch OHLC data from Yahoo Finance"""
    try:
        yf_symbol = symbol.replace("USDT", "-USD")
        df = yf.download(yf_symbol, period=f"{LOOKBACK_DAYS}d", interval="1h", progress=False)
        if df.empty:
            raise ValueError("Yahoo returned no data")
        log(f"[OK] Fetched {len(df)} rows from Yahoo for {symbol}")
        return df
    except Exception as e:
        log(f"Yahoo fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

# ========== CACHE ==========
def cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}.csv")

def load_cache(symbol):
    try:
        path = cache_path(symbol)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
            log(f"[CACHE] Loaded {len(df)} rows for {symbol}")
            return df
        return pd.DataFrame()
    except Exception as e:
        log(f"Cache load failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

def save_cache(symbol, df):
    try:
        df.to_csv(cache_path(symbol))
        log(f"[CACHE] Saved {len(df)} rows for {symbol}")
    except Exception as e:
        log(f"Cache save failed for {symbol}: {e}", "WARN")

# ========== UNIFIED FETCHER ==========
def get_market_data(symbol):
    """Try CoinGecko → Binance → Yahoo → Cache"""
    df = fetch_coingecko(symbol)
    if df.empty:
        df = fetch_binance(symbol)
    if df.empty:
        df = fetch_yahoo(symbol)
    if df.empty:
        df = load_cache(symbol)
    if not df.empty:
        save_cache(symbol, df)
    return df

# ========== TRAINING ==========
def train_model(symbol):
    df = get_market_data(symbol)
    if df.empty or len(df) < 50:
        log(f"Not enough data for {symbol} ({len(df)} rows). Skipping.", "WARN")
        return

    df["returns"] = df["close"].pct_change()
    df["target"] = df["close"].shift(-1)
    df.dropna(inplace=True)

    X = df[["open", "high", "low", "close", "volume", "returns"]]
    y = df["target"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, os.path.join(MODEL_DIR, f"{symbol}_model.pkl"))
    log(f"[DONE] Model trained and saved for {symbol}")

# ========== MAIN ==========
if __name__ == "__main__":
    print("[RUN] Starting model training...")
    symbols = cfg.get("symbols", ["BTCUSDT", "XRPUSDT", "GALAUSDT"])
    for sym in symbols:
        log(f"[START] Training {sym}")
        train_model(sym)
    log("[FINISH] All models processed.")
