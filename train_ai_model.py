import os
import json
import time
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import requests

# ==============================
# CONFIGURATION
# ==============================
SYMBOLS = {
    "BTCUSDT": "bitcoin",
    "XRPUSDT": "ripple",
    "GALAUSDT": "gala"
}

CACHE_DIR = "data_cache"
MODEL_DIR = "models"
SUMMARY_DIR = "training_summary"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

# CoinGecko setup
USE_COINGECKO_DEMO = os.getenv("USE_COINGECKO_DEMO", "true").lower() == "true"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
BASE_URL = "https://api.coingecko.com/api/v3" if USE_COINGECKO_DEMO else "https://pro-api.coingecko.com/api/v3"


# ==============================
# DATA FETCH HELPERS
# ==============================
def fetch_from_coingecko(coin_id):
    try:
        url = f"{BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 90}
        headers = {"accept": "application/json"}
        if COINGECKO_API_KEY:
            headers["x-cg-pro-api-key"] = COINGECKO_API_KEY
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        if "prices" not in data:
            raise ValueError(f"Invalid response: {data}")
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        merged = pd.merge(prices, volumes, on="timestamp", how="inner")
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], unit="ms")
        merged["open"] = merged["price"]
        merged["high"] = merged["price"]
        merged["low"] = merged["price"]
        merged["close"] = merged["price"]
        df = merged[["timestamp", "open", "high", "low", "close", "volume"]]
        logging.info(f"[OK] CoinGecko returned {len(df)} rows for {coin_id}")
        return df
    except Exception as e:
        logging.warning(f"[WARN] CoinGecko fetch failed for {coin_id}: {e}")
        return pd.DataFrame()


def fetch_from_binance(symbol):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=500"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise ValueError(f"Invalid Binance response: {r.status_code}")
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Invalid Binance response format")
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_", "__", "___", "____", "_____", "______"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
        logging.info(f"[OK] Binance returned {len(df)} rows for {symbol}")
        return df
    except Exception as e:
        logging.warning(f"[WARN] Binance fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_from_yahoo(symbol):
    try:
        df = yf.download(symbol, period="90d", interval="1h", progress=False)
        if df.empty:
            raise ValueError("Yahoo returned no data")
        df.reset_index(inplace=True)
        df.rename(columns={
            "Datetime": "timestamp", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        }, inplace=True)
        logging.info(f"[OK] Yahoo returned {len(df)} rows for {symbol}")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logging.warning(f"[WARN] Yahoo fetch failed for {symbol}: {e}")
        return pd.DataFrame()


# ==============================
# CACHE HANDLING
# ==============================
def is_cache_valid(path, max_age_hours=6):
    """Return True if cache file exists and is fresh (< max_age_hours old)."""
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < max_age_hours * 3600


def cleanup_expired_cache(max_age_hours=6):
    """Delete expired cache files."""
    for file in os.listdir(CACHE_DIR):
        full_path = os.path.join(CACHE_DIR, file)
        if not is_cache_valid(full_path, max_age_hours):
            os.remove(full_path)
            logging.info(f"[CLEANUP] Removed expired cache file → {file}")


def load_or_fetch(symbol, coin_id):
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")

    # 1️⃣ Use cache only if valid *and* has enough data
    if is_cache_valid(cache_path):
        df = pd.read_csv(cache_path)
        if len(df) > 100:
            logging.info(f"[CACHE] Using valid cache for {symbol}")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        else:
            logging.info(f"[CACHE] Cache too small ({len(df)} rows) → refetching")

    # 2️⃣ Graceful fallback: CoinGecko → Binance → Yahoo
    logging.info(f"[FETCH] Attempting fresh data for {symbol}")
    df = fetch_from_coingecko(coin_id)
    if df.empty:
        df = fetch_from_binance(symbol)
    if df.empty:
        df = fetch_from_yahoo(symbol.replace("USDT", "-USD"))

    # 3️⃣ If all fetches fail but cache exists → use old cache anyway
    if df.empty and os.path.exists(cache_path):
        logging.warning(f"[FALLBACK] All fetches failed → using old cache for {symbol}")
        df = pd.read_csv(cache_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # 4️⃣ If data fetched → save/update cache
    if not df.empty:
        try:
            df.to_csv(cache_path, index=False)
            logging.info(f"[CACHE] Updated cache with {len(df)} rows for {symbol}")
        except Exception as e:
            logging.error(f"[ERROR] Failed to save cache for {symbol}: {e}")

    return df


# ==============================
# TRAINING FUNCTION
# ==============================
def train_model(symbol, df):
    if len(df) < 100:
        logging.warning(f"[WARN] Not enough data for {symbol} ({len(df)} rows)")
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target"] = df["close"].shift(-1)
    df.dropna(inplace=True)

    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"[OK] Model trained and saved → {model_path}")

    return model


# ==============================
# MAIN ENTRY
# ==============================
def main():
    logging.info("[START] AI Model Training Sequence")
    cleanup_expired_cache(max_age_hours=6)

    summary = {"timestamp": str(datetime.utcnow()), "symbols": {}}

    for symbol, coin_id in SYMBOLS.items():
        logging.info(f"[START] Training {symbol}")
        df = load_or_fetch(symbol, coin_id)
        model = train_model(symbol, df)
        summary["symbols"][symbol] = {
            "rows": len(df),
            "model_trained": model is not None
        }

    # Save summary JSON
    summary_path = os.path.join(
        SUMMARY_DIR,
        f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"[DONE] Training summary saved → {summary_path}")
    logging.info(f"[CHECK] Cached files: {os.listdir(CACHE_DIR)}")
    logging.info("[FINISH] All models processed.")


if __name__ == "__main__":
    main()
