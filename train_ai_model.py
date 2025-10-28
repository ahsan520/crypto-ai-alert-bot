import os
import json
import time
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

LOOKBACK_DAYS = 60
INTERVAL = "1h"
SYMBOLS = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]

MODEL_DIR = "models"
TELEMETRY_DIR = "training_summary"
CACHE_DIR = "data_cache"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TELEMETRY_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
USE_COINGECKO_DEMO = os.getenv("USE_COINGECKO_DEMO", "true").lower() in ["true", "1", "yes"]
COINGECKO_BASE_URL = (
    "https://api.coingecko.com/api/v3"
    if USE_COINGECKO_DEMO
    else "https://pro-api.coingecko.com/api/v3"
)

def log(msg, level="INFO"):
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] [{level}] {msg}")

# --- Binance ---
def fetch_binance(symbol, interval=INTERVAL, limit=1000):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.startswith('['):
            raise ValueError("Invalid Binance response")

        data = r.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_1", "_2", "_3", "_4", "_5", "_6"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        log(f"Binance fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

# --- CoinGecko ---
def fetch_coingecko(symbol):
    """Fetch OHLC data from CoinGecko API (no interval parameter)"""
    try:
        coin_map = {"BTCUSDT": "bitcoin", "XRPUSDT": "ripple", "GALAUSDT": "gala"}
        coin_id = coin_map.get(symbol.upper())
        if not coin_id:
            log(f"Unknown symbol mapping for {symbol}", "ERROR")
            return pd.DataFrame()

        headers = {"accept": "application/json"}
        params = {"days": "14"}
        if not USE_COINGECKO_DEMO and COINGECKO_API_KEY:
            params["x_cg_pro_api_key"] = COINGECKO_API_KEY

        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            raise ValueError(f"Invalid response: {r.text}")

        data = r.json().get("prices", [])
        if not data:
            raise ValueError("No price data returned")

        df = pd.DataFrame(data, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close"] = df["price"]
        df["open"] = df["high"] = df["low"] = df["price"]
        df["volume"] = 0
        df.set_index("timestamp", inplace=True)
        log(f"[OK] CoinGecko data fetched for {symbol}: {len(df)} rows")
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        log(f"CoinGecko fetch failed for {symbol}: {e}", "WARN")
        return pd.DataFrame()

# --- Yahoo Finance fallback ---
def safe_download_yahoo(symbol, period="60d", interval="1h", retries=3, delay=5):
    for attempt in range(retries):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
            if not df.empty:
                df = df.rename(columns=str.lower)
                return df
            log(f"Empty Yahoo data for {symbol} (attempt {attempt+1})", "WARN")
        except Exception as e:
            log(f"Yahoo fetch failed for {symbol} (attempt {attempt+1}): {e}", "ERROR")
        time.sleep(delay)
    return pd.DataFrame()

# --- Cache ---
def load_from_cache(symbol):
    path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        log(f"[CACHE] Loaded {len(df)} rows for {symbol}")
        return df
    return pd.DataFrame()

def save_to_cache(symbol, df):
    if not df.empty:
        df.to_csv(os.path.join(CACHE_DIR, f"{symbol}.csv"))
        log(f"[CACHE] Saved {len(df)} rows for {symbol}")

def get_market_data(symbol):
    df = fetch_binance(symbol)
    if df.empty:
        df = fetch_coingecko(symbol)
    if df.empty:
        yf_symbol = symbol.replace("USDT", "-USD")
        df = safe_download_yahoo(yf_symbol)
    if df.empty:
        df = load_from_cache(symbol)
    else:
        save_to_cache(symbol, df)
    return df

# --- Features + Training ---
def prepare_features(df):
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["volatility"] = df["close"].rolling(10).std()
    df = df.dropna()
    X = df[["open", "high", "low", "close", "volume", "ma5", "ma20", "volatility"]]
    y = df["close"].shift(-1).dropna()
    X, y = X.iloc[:-1], y
    return X, y

def train_model(symbol, df):
    try:
        X, y = prepare_features(df)
        if len(X) < 100:
            log(f"[WARN] No sufficient data for {symbol}. Skipped.")
            return False

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump({"model": model, "scaler": scaler}, os.path.join(MODEL_DIR, f"{symbol}_model.pkl"))
        log(f"[OK] Model trained for {symbol} ({len(X)} samples)")
        return True
    except Exception as e:
        log(f"[ERROR] Training failed for {symbol}: {e}")
        return False

# --- Main ---
def main():
    log("[INFO] [START] AI Model Training Sequence")
    telemetry = {"start_time": str(datetime.utcnow()), "symbols": {}}

    for symbol in SYMBOLS:
        log(f"[INFO] [START] Training {symbol}")
        df = get_market_data(symbol)
        if df.empty:
            log(f"[ERROR] No data for {symbol}.", "ERROR")
            telemetry["symbols"][symbol] = {"status": "failed_no_data"}
            continue

        success = train_model(symbol, df)
        telemetry["symbols"][symbol] = {"status": "success" if success else "failed"}

    telemetry["end_time"] = str(datetime.utcnow())
    telemetry_path = os.path.join(
        TELEMETRY_DIR, f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(telemetry_path, "w") as f:
        json.dump(telemetry, f, indent=2)

    log(f"[OK] [FINISH] All models processed.")
    log(f"[OK] Telemetry saved → {telemetry_path}")

if __name__ == "__main__":
    main()
