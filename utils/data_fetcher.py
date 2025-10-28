import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# ---------- Config ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

COINGECKO_API = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"
COINGECKO_IDS = {"BTCUSDT": "bitcoin", "XRPUSDT": "ripple", "GALAUSDT": "gala"}
YF_MAPPING = {"BTCUSDT": "BTC-USD", "XRPUSDT": "XRP-USD", "GALAUSDT": "GALA-USD"}

CACHE_EXPIRY_HOURS = 6


# ---------- Cache Helpers ----------
def cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}.csv")

def is_cache_valid(symbol):
    path = cache_path(symbol)
    if not os.path.exists(path):
        return False
    mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
    return (datetime.utcnow() - mtime) < timedelta(hours=CACHE_EXPIRY_HOURS)


# ---------- Data Fetch Logic ----------
def get_data(symbol):
    """Fetch market data with fallback: CoinGecko → Yahoo → Cache"""
    if is_cache_valid(symbol):
        return load_from_cache(symbol)

    df = fetch_from_coingecko(symbol)
    if df is None or df.empty:
        df = fetch_from_yahoo(symbol)
    if df is None or df.empty:
        df = load_from_cache(symbol)

    if df is not None and not df.empty:
        save_to_cache(symbol, df)
    return df


def fetch_from_coingecko(symbol):
    """Pull 7d hourly data from CoinGecko"""
    try:
        coin_id = COINGECKO_IDS.get(symbol)
        if not coin_id:
            return None
        url = COINGECKO_API.format(id=coin_id)
        params = {"vs_currency": "usd", "days": "7", "interval": "hourly"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = [v[1] for v in volumes[:len(df)]]
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"].rolling(3).max()
        df["low"] = df["close"].rolling(3).min()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"[WARN] CoinGecko fetch failed for {symbol}: {e}")
        return None


def fetch_from_yahoo(symbol):
    """Fallback to Yahoo Finance"""
    try:
        yf_symbol = YF_MAPPING.get(symbol)
        df = yf.download(yf_symbol, period="7d", interval="1h", auto_adjust=True, threads=False)
        if df is None or df.empty:
            raise ValueError("Yahoo returned no data")
        df = df.reset_index()
        df.rename(columns={
            "Datetime": "timestamp", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        }, inplace=True)
        return df
    except Exception as e:
        print(f"[WARN] Yahoo fetch failed for {symbol}: {e}")
        return None


# ---------- Cache Read/Write ----------
def save_to_cache(symbol, df):
    try:
        df.to_csv(cache_path(symbol), index=False)
        print(f"[CACHE] Updated cache for {symbol}")
    except Exception as e:
        print(f"[WARN] Failed to save cache for {symbol}: {e}")

def load_from_cache(symbol):
    try:
        path = cache_path(symbol)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["timestamp"])
            print(f"[CACHE] Loaded from cache for {symbol}")
            return df
        return None
    except Exception as e:
        print(f"[WARN] Failed to load cache for {symbol}: {e}")
        return None


# ---------- Candle Extract ----------
def fetch_latest_candle(symbol):
    """Return most recent candle for AI alert"""
    df = get_data(symbol)
    if df is not None and not df.empty:
        last = df.iloc[-1]
        return {
            "timestamp": last["timestamp"],
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "close": float(last["close"]),
            "volume": float(last["volume"]),
        }
    return None
