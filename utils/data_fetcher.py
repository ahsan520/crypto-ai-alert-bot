#!/usr/bin/env python3
"""
utils/data_fetcher.py
--------------------------------------
Unified crypto data fetcher with automatic multi-source fallback.

Sources (in order of priority):
  1. Binance (ccxt)
  2. Kraken (ccxt)
  3. Coinbase (ccxt)
  4. Yahoo Finance (yfinance)
  5. CoinGecko (pycoingecko)

All public data — no authentication required.
Cached data is saved under data_cache/{symbol}.csv.
"""

import os
import time
import logging
import pandas as pd
import yfinance as yf
from pycoingecko import CoinGeckoAPI

# optional dependency for exchanges
try:
    import ccxt
except ImportError:
    ccxt = None
    print("[WARN] ccxt not installed; only Yahoo & CoinGecko sources will work.")

# ---------------------------------
# CONFIG
# ---------------------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------
# Helper: save to cache
# ---------------------------------
def _cache(symbol: str, df: pd.DataFrame):
    if df is not None and not df.empty:
        path = os.path.join(CACHE_DIR, f"{symbol}.csv")
        df.to_csv(path, index=False)
        logging.info(f"[CACHE] Saved {len(df)} rows for {symbol} → {path}")
    else:
        logging.warning(f"[CACHE] No data to save for {symbol}")

# ---------------------------------
# Binance fetch (public OHLCV)
# ---------------------------------
def fetch_binance(symbol: str, days=30, interval="1h"):
    if ccxt is None:
        return pd.DataFrame()
    try:
        exchange = ccxt.binance()
        pair = symbol.replace("USDT", "/USDT")
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=interval, since=since)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        logging.info(f"[BINANCE] {symbol} → {len(df)} rows")
        return df
    except Exception as e:
        logging.warning(f"[BINANCE] Failed for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------------
# Kraken fetch
# ---------------------------------
def fetch_kraken(symbol: str, days=30, interval="1h"):
    if ccxt is None:
        return pd.DataFrame()
    try:
        exchange = ccxt.kraken()
        pair = symbol.replace("USDT", "/USDT")
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=interval, since=since)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        logging.info(f"[KRAKEN] {symbol} → {len(df)} rows")
        return df
    except Exception as e:
        logging.warning(f"[KRAKEN] Failed for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------------
# Coinbase fetch
# ---------------------------------
def fetch_coinbase(symbol: str, days=30, interval="1h"):
    if ccxt is None:
        return pd.DataFrame()
    try:
        exchange = ccxt.coinbase()
        pair = symbol.replace("USDT", "/USDT")
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=interval, since=since)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        logging.info(f"[COINBASE] {symbol} → {len(df)} rows")
        return df
    except Exception as e:
        logging.warning(f"[COINBASE] Failed for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------------
# Yahoo Finance fallback
# ---------------------------------
def fetch_yahoo(symbol: str, period="30d", interval="1h"):
    try:
        ticker = yf.Ticker(symbol.replace("USDT", "-USD"))
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError("No data returned.")
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df.reset_index().rename(columns={"Datetime": "timestamp"})
        logging.info(f"[YAHOO] {symbol} → {len(df)} rows")
        return df
    except Exception as e:
        logging.warning(f"[YAHOO] Failed for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------------
# CoinGecko fallback
# ---------------------------------
def fetch_coingecko(symbol: str):
    try:
        cg = CoinGeckoAPI()
        cg_id = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "XRPUSDT": "ripple",
            "GALAUSDT": "gala",
        }.get(symbol, symbol.lower().replace("usdt", ""))

        data = cg.get_coin_market_chart_by_id(id=cg_id, vs_currency="usd", days="30")
        prices = data.get("prices", [])
        if not prices:
            raise ValueError("No CoinGecko price data.")
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = [v[1] for v in data.get("total_volumes", [])[: len(df)]]
        logging.info(f"[COINGECKO] {symbol} → {len(df)} rows")
        return df
    except Exception as e:
        logging.warning(f"[COINGECKO] Failed for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------------------
# Unified get_data()
# ---------------------------------
def get_data(symbol: str, days=30, interval="1h"):
    """
    Unified fetch chain:
    Binance → Kraken → Coinbase → Yahoo → CoinGecko
    """
    for src, func in [
        ("Binance", fetch_binance),
        ("Kraken", fetch_kraken),
        ("Coinbase", fetch_coinbase),
        ("Yahoo", fetch_yahoo),
        ("CoinGecko", fetch_coingecko),
    ]:
        logging.info(f"[TRY] {src} for {symbol}...")
        df = func(symbol, days, interval) if "days" in func.__code__.co_varnames else func(symbol)
        if df is not None and not df.empty:
            _cache(symbol, df)
            df.attrs["source"] = src
            logging.info(f"[OK] {src} succeeded for {symbol}")
            return df
        time.sleep(1)
    logging.error(f"[FAIL] All sources failed for {symbol}")
    return pd.DataFrame()

# ---------------------------------
# Compatibility wrapper for alert script
# ---------------------------------
def fetch_and_cache(symbols):
    """
    Fetch and cache multiple symbols — for backward compatibility.
    Used by crypto_ai_alert_v10.py.
    """
    results = {}
    for sym in symbols:
        df = get_data(sym)
        if df is not None and not df.empty:
            results[sym] = df
            logging.info(f"[FETCH_AND_CACHE] {sym}: {len(df)} rows ready.")
        else:
            logging.warning(f"[FETCH_AND_CACHE] {sym}: no valid data.")
    return results


# ---------------------------------
# Run standalone (manual test)
# ---------------------------------
if __name__ == "__main__":
    TEST_SYMBOLS = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]
    logging.info("[RUN] Manual test: fetching sample data...")
    results = fetch_and_cache(TEST_SYMBOLS)
    for sym, df in results.items():
        print(f"{sym}: {len(df)} rows from {df.attrs.get('source', 'unknown')}")
