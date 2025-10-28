# utils/data_fetcher.py
import os
import time
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

DATA_CACHE_DIR = "data_cache"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

def fetch_from_google(symbol, days=3):
    """Fetch data using Google Finance CSV API"""
    print(f"[FETCH] Trying Google Finance for {symbol} ...")
    try:
        url = f"https://www.google.com/finance/quote/{symbol}"
        df = pd.read_html(requests.get(url, timeout=10).text)[0]
        if df.empty:
            raise ValueError("Empty dataframe from Google.")
        df.columns = [c.lower() for c in df.columns]
        df['timestamp'] = pd.to_datetime(datetime.utcnow())
        print(f"[OK] Google Finance fetched {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        print(f"[WARN] Google Finance failed for {symbol}: {e}")
        return None

def fetch_from_yahoo(symbol, days=3):
    """Fetch data using yfinance"""
    print(f"[FETCH] Trying Yahoo Finance for {symbol} ...")
    try:
        data = yf.download(symbol, period=f"{days}d", interval="15m", progress=False)
        if data.empty:
            raise ValueError("Empty dataframe from Yahoo.")
        data.reset_index(inplace=True)
        print(f"[OK] Yahoo Finance fetched {len(data)} rows for {symbol}")
        return data
    except Exception as e:
        print(f"[WARN] Yahoo Finance failed for {symbol}: {e}")
        return None

def fetch_from_coingecko(symbol, days=3):
    """Fetch data from CoinGecko as last fallback"""
    print(f"[FETCH] Trying CoinGecko for {symbol} ...")
    try:
        coingecko_map = {"BTCUSDT": "bitcoin", "XRPUSDT": "ripple", "GALAUSDT": "gala"}
        coin_id = coingecko_map.get(symbol.upper())
        if not coin_id:
            raise ValueError("Symbol not in coingecko_map.")
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "hourly"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json().get("prices", [])
        if not data:
            raise ValueError("Empty data from CoinGecko.")
        df = pd.DataFrame(data, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close"] = df["price"]
        df.drop(columns=["price"], inplace=True)
        print(f"[OK] CoinGecko fetched {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        print(f"[ERROR] CoinGecko failed for {symbol}: {e}")
        return None

def fetch_and_cache(symbol, days=3):
    """Try all sources in fallback chain and cache result."""
    for fetcher in [fetch_from_google, fetch_from_yahoo, fetch_from_coingecko]:
        df = fetcher(symbol, days)
        if df is not None and not df.empty:
            path = os.path.join(DATA_CACHE_DIR, f"{symbol}.csv")
            df.to_csv(path, index=False)
            print(f"[CACHE] Saved {len(df)} rows → {path}")
            return df
    print(f"[FAIL] All sources failed for {symbol}. No data cached.")
    return None

if __name__ == "__main__":
    print("[RUN] Fetching crypto data using utils/data_fetcher.py...")
    for sym in ["BTCUSDT", "XRPUSDT", "GALAUSDT"]:
        fetch_and_cache(sym)
