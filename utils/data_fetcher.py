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
MIN_CACHE_ROWS = 100  # ✅ ensure minimum data size before trusting cache


# ---------- Cache Helpers ----------
def cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}.csv")


def is_cache_valid(symbol):
    """Check if cache exists, is recent, and not too small."""
    path = cache_path(symbol)
    if not os.path.exists(path):
        return False

    try:
        mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
        age_ok = (datetime.utcnow() - mtime) < timedelta(hours=CACHE_EXPIRY_HOURS)

        df = pd.read_csv(path)
        size_ok = len(df) >= MIN_CACHE_ROWS

        if not size_ok:
            print(f"[CACHE] {symbol} cache too small ({len(df)} rows) — refreshing")
        if not age_ok:
            print(f"[CACHE] {symbol} cache too old — refreshing")

        return age_ok and size_ok
    except Exception as e:
        print(f"[WARN] Cache validation failed for {symbol}: {e}")
        return False


# ---------- Data Fetch Logic ----------
def get_data(symbol):
    """Fetch market data with fallback: CoinGecko → Yahoo → Cache"""
    if is_cache_valid(symbol):
        return load_from_cache(symbol)

    print(f"[FETCH] Refreshing data for {symbol} (cache invalid or small)")
    df = fetch_from_coingecko(symbol)

    if df is None or df.empty:
        print(f"[WARN] CoinGecko returned no data for {symbol}, trying Yahoo")
        df = fetch_from_yahoo(symbol)

    if df is None or df.empty:
        print(f"[WARN] Both sources failed for {symbol}, trying cache fallback")
        df = load_from_cache(symbol)

    if df is not None and not df.empty:
        save_to_cache(symbol, df)
    else:
        print(f"[ERROR] No data could be fetched for {symbol}")

    return df


def fetch_from_coingecko(symbol):
    """Pull 7d hourly data from CoinGecko"""
    try:
        coin_id = COINGECKO_IDS.get(symbol)
        if not coin_id:
            print(f"[WARN] Unknown symbol for CoinGecko: {symbol}")
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

        print(f"[COINGECKO] {symbol}: {len(df)} rows fetched")
        return df
    except Exception as e:
        print(f"[WARN] CoinGecko fetch failed for {symbol}: {e}")
        return None


def fetch_from_yahoo(symbol):
    """Fallback to Yahoo Finance (robust handling)"""
    try:
        yf_symbol = YF_MAPPING.get(symbol)
        if not yf_symbol:
            print(f"[WARN] No Yahoo mapping for {symbol}")
            return None

        df = yf.download(
            yf_symbol,
            period="7d",
            interval="1h",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            print(f"[WARN] Yahoo returned empty data for {symbol}")
            return None

        df = df.reset_index()
        if "Datetime" not in df.columns:
            print(f"[WARN] Yahoo structure unexpected for {symbol}")
            return None

        df.rename(columns={
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        df.dropna(inplace=True)
        print(f"[YAHOO] {symbol}: {len(df)} rows fetched")
        return df
    except Exception as e:
        print(f"[WARN] Yahoo fetch failed for {symbol}: {e}")
        return None


# ---------- Cache Read/Write ----------
def save_to_cache(symbol, df):
    try:
        df.to_csv(cache_path(symbol), index=False)
        print(f"[CACHE] Updated cache for {symbol} ({len(df)} rows)")
    except Exception as e:
        print(f"[WARN] Failed to save cache for {symbol}: {e}")


def load_from_cache(symbol):
    try:
        path = cache_path(symbol)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["timestamp"])
            print(f"[CACHE] Loaded from cache for {symbol} ({len(df)} rows)")
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
