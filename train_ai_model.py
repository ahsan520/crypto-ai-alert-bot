import os
import json
import time
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# === CONFIG ===
LOOKBACK_DAYS = 60
INTERVAL = "1h"
SYMBOLS = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]
COINGECKO_IDS = {"BTCUSDT": "bitcoin", "XRPUSDT": "ripple", "GALAUSDT": "gala"}

MODEL_DIR = "models"
CACHE_DIR = "data_cache"
TELEMETRY_DIR = "training_summary"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TELEMETRY_DIR, exist_ok=True)

# === API CONFIG ===
USE_COINGECKO_DEMO = os.getenv("USE_COINGECKO_DEMO", "true").lower() == "true"
CG_API_KEY = os.getenv("COINGECKO_API_KEY", "")
COINGECKO_BASE_URL = (
    "https://api.coingecko.com/api/v3" if USE_COINGECKO_DEMO else "https://pro-api.coingecko.com/api/v3"
)

def log(level, msg):
    """Timestamped log output"""
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] [{level}] {msg}")

# === DATA FETCHERS ===
def fetch_from_binance(symbol, limit=1000):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={INTERVAL}&limit={limit}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise ValueError("Invalid Binance response")
        data = r.json()
        if not isinstance(data, list):
            raise ValueError("Invalid Binance response")
        df = pd.DataFrame(data, columns=[
            "timestamp","open","high","low","close","volume",
            "_1","_2","_3","_4","_5","_6"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp","open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        log("WARN", f"Binance fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def fetch_from_coingecko(coin_id, days=60):
    """Fetch market data from CoinGecko"""
    try:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        headers = {"accept": "application/json"}
        if CG_API_KEY:
            headers["x-cg-pro-api-key"] = CG_API_KEY
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        data = r.json()
        if "prices" not in data:
            raise ValueError(f"Invalid response: {r.text}")

        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        if "total_volumes" in data:
            vol_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
            df = df.merge(vol_df, on="timestamp", how="left")
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df = df[["timestamp","open","high","low","close","volume"]]
        return df
    except Exception as e:
        log("WARN", f"CoinGecko fetch failed for {coin_id}: {e}")
        return pd.DataFrame()

def fetch_from_yahoo(symbol, period="60d", interval="1h"):
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df.empty:
            log("WARN", f"Yahoo fetch failed for {symbol}: Yahoo returned no data")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df[["timestamp","Open","High","Low","Close","Volume"]].rename(
            columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
        )
    except Exception as e:
        log("WARN", f"Yahoo fetch failed for {symbol}: {e}")
        return pd.DataFrame()

# === CACHE HANDLER ===
def load_or_fetch(symbol, coingecko_id):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")
    log("INFO", f"[FETCH] Attempting fresh data for {symbol}")

    df = fetch_from_binance(symbol)
    if df is None or df.empty:
        df = fetch_from_coingecko(coingecko_id)
    if df is None or df.empty:
        df = fetch_from_yahoo(symbol.replace("USDT", "-USD"))

    # Fallback to cache if all fail
    if df is None or df.empty:
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=["timestamp"])
            log("INFO", f"[CACHE] Loaded {len(df)} rows for {symbol}")
        else:
            log("ERROR", f"No data available for {symbol}")
            return None
    else:
        df.to_csv(cache_file, index=False)
        log("INFO", f"[CACHE] Saved {len(df)} rows for {symbol}")

    return df

# === FEATURE ENGINEERING ===
def prepare_features(df):
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["volatility"] = df["close"].rolling(10).std()
    df = df.dropna()
    X = df[["open","high","low","close","volume","ma5","ma20","volatility"]]
    y = df["close"].shift(-1).dropna()
    X, y = X.iloc[:-1], y
    return X, y

# === MODEL TRAINING ===
def train_model(symbol, df):
    try:
        X, y = prepare_features(df)
        if len(X) < 50:
            log("WARN", f"Not enough data for {symbol} ({len(X)} rows)")
            return None
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        joblib.dump({"model": model, "scaler": scaler}, os.path.join(MODEL_DIR, f"{symbol}_model.pkl"))
        log("INFO", f"Model trained successfully for {symbol} ({len(X)} samples)")
        return True
    except Exception as e:
        log("ERROR", f"Training failed for {symbol}: {e}")
        return None

# === MAIN ===
def main():
    start_time = datetime.utcnow()
    log("INFO", "[START] AI Model Training Sequence")
    telemetry = {"start_time": str(start_time), "symbols": {}}

    for symbol in SYMBOLS:
        log("INFO", f"[START] Training {symbol}")
        df = load_or_fetch(symbol, COINGECKO_IDS[symbol])
        if df is None or df.empty:
            log("WARN", f"No sufficient data for {symbol}. Skipped.")
            telemetry["symbols"][symbol] = {"status": "failed_no_data"}
            continue

        status = train_model(symbol, df)
        telemetry["symbols"][symbol] = {"status": "success" if status else "failed"}
        time.sleep(2)

    telemetry["end_time"] = str(datetime.utcnow())
    telemetry_path = os.path.join(TELEMETRY_DIR, f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(telemetry_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    log("INFO", f"[FINISH] All models processed.")

if __name__ == "__main__":
    main()
