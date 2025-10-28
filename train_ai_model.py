import os
import json
import time
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
CACHE_EXPIRY_HOURS = 6  # Cache expires after 6 hours

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

USE_COINGECKO_DEMO = os.getenv("USE_COINGECKO_DEMO", "true").lower() == "true"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
BASE_URL = "https://api.coingecko.com/api/v3" if USE_COINGECKO_DEMO else "https://pro-api.coingecko.com/api/v3"


# ==============================
# FETCH HELPERS
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
            raise ValueError(f"Invalid CoinGecko response: {data}")
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
# CACHE MANAGEMENT
# ==============================
def is_cache_valid(cache_path):
    if not os.path.exists(cache_path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
    if datetime.now() - mtime < timedelta(hours=CACHE_EXPIRY_HOURS):
        return True
    else:
        logging.info(f"[CACHE] Expired cache, deleting {cache_path}")
        os.remove(cache_path)
        return False


def load_or_fetch(symbol, coin_id):
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")

    # 1️⃣ Try cache first if valid
    if is_cache_valid(cache_path):
        logging.info(f"[CACHE] Using valid cache for {symbol}")
        df = pd.read_csv(cache_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # 2️⃣ Graceful fallback: CoinGecko → Binance → Yahoo → Old cache
    logging.info(f"[FETCH] Fetching fresh data for {symbol}")
    df = fetch_from_coingecko(coin_id)
    if df.empty:
        df = fetch_from_binance(symbol)
    if df.empty:
        df = fetch_from_yahoo(symbol.replace("USDT", "-USD"))

    # 3️⃣ Use old cache if all sources failed
    if df.empty and os.path.exists(cache_path):
        logging.warning(f"[FALLBACK] Using old expired cache for {symbol}")
        df = pd.read_csv(cache_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 4️⃣ Save new cache if we got something
    if not df.empty:
        df.to_csv(cache_path, index=False)
        logging.info(f"[CACHE] Updated cache for {symbol} ({len(df)} rows)")

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
    summary_path = os.path.join(SUMMARY_DIR, f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"[DONE] Training summary saved → {summary_path}")
    logging.info("[FINISH] All models processed.")


if __name__ == "__main__":
    main()
