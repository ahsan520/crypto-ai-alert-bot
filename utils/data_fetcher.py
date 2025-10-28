#!/usr/bin/env python3
import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
from pycoingecko import CoinGeckoAPI

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

cg = CoinGeckoAPI()


# =========================================
# SYMBOL MAPS
# =========================================
SYMBOL_MAP = {
    "BTCUSDT": {
        "google": "CURRENCY:BTCUSD",
        "yahoo": "BTC-USD",
        "coingecko": "bitcoin"
    },
    "XRPUSDT": {
        "google": "CURRENCY:XRPUSD",
        "yahoo": "XRP-USD",
        "coingecko": "ripple"
    },
    "GALAUSDT": {
        "google": "CURRENCY:GALAUSD",
        "yahoo": "GALA-USD",
        "coingecko": "gala"
    }
}


# =========================================
# SAVE CACHE
# =========================================
def _save_cache(symbol, df):
    """Save dataframe to cache as CSV."""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    try:
        df.to_csv(cache_path, index=False)
        logging.info(f"[CACHE] Saved → {cache_path}")
    except Exception as e:
        logging.error(f"[CACHE ERROR] {symbol}: {e}")


# =========================================
# GOOGLE FINANCE FETCH
# =========================================
def fetch_google(symbol):
    """Fetch from Google Finance via pandas_datareader."""
    try:
        mapped = SYMBOL_MAP.get(symbol, {}).get("google", symbol)
        logging.info(f"[GOOGLE] Fetching {symbol} as {mapped}")
        end = datetime.utcnow()
        start = end - timedelta(days=60)

        df = pdr.DataReader(mapped, "google", start, end)
        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        logging.warning(f"[GOOGLE] Failed for {symbol}: {e}")
        return None


# =========================================
# YAHOO FINANCE FETCH
# =========================================
def fetch_yahoo(symbol):
    """Fetch from Yahoo Finance."""
    try:
        mapped = SYMBOL_MAP.get(symbol, {}).get("yahoo", symbol)
        logging.info(f"[YAHOO] Fetching {symbol} as {mapped}")
        end = datetime.utcnow()
        start = end - timedelta(days=60)

        df = yf.download(mapped, start=start, end=end, progress=False)
        if df is None or df.empty:
            return None

        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        logging.warning(f"[YAHOO] Failed for {symbol}: {e}")
        return None


# =========================================
# COINGECKO FETCH
# =========================================
def fetch_coingecko(symbol):
    """Fallback fetch from CoinGecko."""
    try:
        mapped = SYMBOL_MAP.get(symbol, {}).get("coingecko")
        if not mapped:
            logging.warning(f"[COINGECKO] No mapping for {symbol}")
            return None

        logging.info(f"[COINGECKO] Fetching {symbol} ({mapped}) ...")
        data = cg.get_coin_market_chart_by_id(id=mapped, vs_currency="usd", days=60)
        prices = data.get("prices", [])
        vols = data.get("total_volumes", [])

        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = [v[1] for v in vols[:len(df)]]
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"].rolling(3, min_periods=1).max()
        df["low"] = df["close"].rolling(3, min_periods=1).min()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.warning(f"[COINGECKO] Failed for {symbol}: {e}")
        return None


# =========================================
# MAIN FETCHER LOGIC
# =========================================
def get_data(symbol):
    """Unified fetcher with Google → Yahoo → CoinGecko fallback."""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")

    # Try cache first
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            if not df.empty:
                logging.info(f"[CACHE HIT] Loaded {symbol} from cache")
                return df
        except Exception as e:
            logging.warning(f"[CACHE READ FAIL] {symbol}: {e}")

    # Google
    df = fetch_google(symbol)
    if df is not None and not df.empty:
        _save_cache(symbol, df)
        return df

    # Yahoo
    df = fetch_yahoo(symbol)
    if df is not None and not df.empty:
        _save_cache(symbol, df)
        return df

    # CoinGecko
    df = fetch_coingecko(symbol)
    if df is not None and not df.empty:
        _save_cache(symbol, df)
        return df

    logging.error(f"[FAIL] All sources failed for {symbol}")
    return None
