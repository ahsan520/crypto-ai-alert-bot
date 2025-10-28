#!/usr/bin/env python3
import os
import json
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==============================
# GOOGLE FINANCE FETCHER
# ==============================
def get_google_data(symbol):
    """
    Try to fetch historical OHLCV data from Google Finance via Yahoo API proxy.
    Note: Google Finance doesn’t have an official public API, so we use a lightweight
    CSV endpoint provided by Google (may occasionally block requests).
    """
    try:
        logging.info(f"[GOOGLE] Attempting fetch for {symbol}")
        url = f"https://www.google.com/finance/quote/{symbol}"
        df = pd.read_html(url)[0]  # fallback for latest price
        if df.empty:
            raise ValueError("Empty Google data")
        df["timestamp"] = datetime.utcnow()
        df.rename(columns={"Price": "close"}, inplace=True)
        df["open"] = df["high"] = df["low"] = df["close"]
        df["volume"] = 0
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logging.warning(f"[GOOGLE] Failed for {symbol}: {e}")
        return None


# ==============================
# YAHOO FINANCE FETCHER
# ==============================
def get_yahoo_data(symbol, days=30):
    try:
        logging.info(f"[YAHOO] Attempting fetch for {symbol}")
        data = yf.download(symbol, period=f"{days}d", interval="1h", progress=False)
        if data.empty:
            raise ValueError("Empty Yahoo data")

        data.reset_index(inplace=True)
        data.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        return data[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logging.warning(f"[YAHOO] Failed for {symbol}: {e}")
        return None


# ==============================
# COINGECKO FETCHER
# ==============================
def get_coingecko_data(symbol):
    try:
        logging.info(f"[COINGECKO] Attempting fetch for {symbol}")
        coin_id = {
            "BTCUSDT": "bitcoin",
            "XRPUSDT": "ripple",
            "GALAUSDT": "gala"
        }.get(symbol, symbol.lower())

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 30, "interval": "hourly"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()

        data = r.json().get("prices", [])
        if not data:
            raise ValueError("Empty CoinGecko data")

        df = pd.DataFrame(data, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["high"] = df["low"] = df["close"]
        df["volume"] = 0
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logging.warning(f"[COINGECKO] Failed for {symbol}: {e}")
        return None


# ==============================
# MAIN UNIFIED FETCH FUNCTION
# ==============================
def get_data(symbol):
    """Fetch data using Google → Yahoo → CoinGecko → cache fallback."""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")

    # Try Google Finance
    df = get_google_data(symbol)
    if df is not None and not df.empty:
        logging.info(f"[FETCH] {symbol} → Google Finance success ✅")
        df.to_csv(cache_file, index=False)
        return df

    # Try Yahoo Finance
    df = get_yahoo_data(symbol)
    if df is not None and not df.empty:
        logging.info(f"[FETCH] {symbol} → Yahoo Finance success ✅")
        df.to_csv(cache_file, index=False)
        return df

    # Try CoinGecko
    df = get_coingecko_data(symbol)
    if df is not None and not df.empty:
        logging.info(f"[FETCH] {symbol} → CoinGecko success ✅")
        df.to_csv(cache_file, index=False)
        return df

    # Last resort: use cache
    if os.path.exists(cache_file):
        logging.warning(f"[CACHE] Using fallback cache for {symbol}")
        return pd.read_csv(cache_file)

    logging.error(f"[FAIL] All fetch methods failed for {symbol}")
    return None
