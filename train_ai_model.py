#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto AI Model Trainer v13.3
- Hybrid data source fallback (Binance → CoinGecko → Yahoo → Cache)
- CoinGecko Demo API base_url auto-detect
- Uses cached data when online fetch fails
"""

import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from colorama import Fore, Style

# === Configuration ===
CACHE_DIR = "data_cache"
MODEL_DIR = "models"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
if "demo" in COINGECKO_API_KEY.lower():
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
else:
    COINGECKO_BASE_URL = "https://pro-api.coingecko.com/api/v3"

# === Logging helper ===
def log(msg, level="info"):
    colors = {
        "info": Fore.CYAN,
        "warn": Fore.YELLOW,
        "error": Fore.RED,
        "ok": Fore.GREEN
    }
    print(colors.get(level, Fore.WHITE) + f"[{level.upper()}] {msg}" + Style.RESET_ALL)

# === Data fetchers ===
def fetch_binance(symbol="BTCUSDT", limit=500):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit={limit}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            raise ValueError("Invalid Binance response")
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
        log(f"Fetched {len(df)} rows from Binance for {symbol}", "ok")
        return df
    except Exception as e:
        log(f"Binance fetch failed for {symbol}: {e}", "warn")
        return None

def fetch_coingecko(symbol="bitcoin", days=7):
    try:
        coin_id = symbol.lower().replace("usdt", "").replace("usd", "")
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}

        headers = {"accept": "application/json"}
        if "demo" not in COINGECKO_API_KEY.lower():
            headers["x-cg-pro-api-key"] = COINGECKO_API_KEY
        else:
            params["x_cg_demo_api_key"] = COINGECKO_API_KEY

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            raise ValueError(f"Invalid response: {resp.text}")
        prices = resp.json().get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        log(f"Fetched {len(df)} rows from CoinGecko for {symbol}", "ok")
        return df
    except Exception as e:
        log(f"CoinGecko fetch failed for {symbol}: {e}", "warn")
        return None

def fetch_yahoo(symbol="BTC-USD", period="7d", interval="1h"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError("Yahoo returned no data")
        df.reset_index(inplace=True)
        df.rename(columns={"Datetime": "timestamp", "Close": "close", "Volume": "volume"}, inplace=True)
        log(f"Fetched {len(df)} rows from Yahoo for {symbol}", "ok")
        return df[["timestamp", "close", "volume"]]
    except Exception as e:
        log(f"Yahoo fetch failed for {symbol}: {e}", "warn")
        return None

# === Cache helpers ===
def cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}.csv")

def load_cache(symbol):
    try:
        path = cache_path(symbol)
        if os.path.exists(path):
            df = pd.read_csv(path)
            log(f"[CACHE] Loaded {len(df)} rows for {symbol}", "info")
            return df
        return None
    except Exception as e:
        log(f"Cache load failed for {symbol}: {e}", "warn")
        return None

def save_cache(symbol, df):
    try:
        df.to_csv(cache_path(symbol), index=False)
        log(f"[CACHE] Saved {len(df)} rows for {symbol}", "info")
    except Exception as e:
        log(f"Cache save failed for {symbol}: {e}", "warn")

# === Data preparation ===
def prepare_data(df):
    if df is None or len(df) < 100:
        return None, None
    df = df.sort_values("timestamp")
    df["returns"] = df["close"].pct_change().fillna(0)
    df["target"] = (df["returns"].shift(-1) > 0).astype(int)
    features = df[["close", "volume", "returns"]].fillna(0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    y = df["target"].values
    return X[:-1], y[:-1]

# === Model training ===
def train_model(symbol, df):
    X, y = prepare_data(df)
    if X is None or len(X) < 100:
        log(f"Not enough data for {symbol} ({len(df)} rows). Skipping.", "warn")
        return
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    path = os.path.join(MODEL_DIR, f"{symbol}_model.joblib")
    dump(model, path)
    log(f"✅ Trained model for {symbol} and saved to {path}", "ok")

# === Main loop ===
def main():
    log("[START] AI Model Training Sequence", "info")
    symbols = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]

    for sym in symbols:
        log(f"[START] Training {sym}", "info")

        df = None
        # Try Binance first
        df = fetch_binance(sym)
        if df is None or len(df) < 100:
            df = fetch_coingecko(sym)
        if (df is None or len(df) < 100) and "USDT" in sym:
            df = fetch_yahoo(sym.replace("USDT", "-USD"))
        if df is None or len(df) < 100:
            df = load_cache(sym)

        if df is not None and len(df) >= 50:
            save_cache(sym, df)
            train_model(sym, df)
        else:
            log(f"[WARN] No sufficient data for {sym}. Skipped.", "warn")

    log("[FINISH] All models processed.", "ok")


if __name__ == "__main__":
    main()
