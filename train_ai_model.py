import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

USE_COINGECKO_DEMO = os.getenv("USE_COINGECKO_DEMO", "false").lower() == "true"
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
COINGECKO_BASE_URL = (
    "https://api.coingecko.com/api/v3" if USE_COINGECKO_DEMO else "https://pro-api.coingecko.com/api/v3"
)

def log(level, msg):
    print(f"[{datetime.utcnow().isoformat()}] [{level}] {msg}")

# -------------------------------
# FETCHERS
# -------------------------------
def fetch_from_binance(symbol="BTCUSDT", limit=200):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit={limit}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            raise Exception("Invalid Binance response")
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_base_vol", "taker_quote_vol", "ignore"
        ])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        log("WARN", f"Binance fetch failed for {symbol}: {e}")
        return None


def fetch_from_coingecko(symbol_id="bitcoin", vs_currency="usd", days=30):
    try:
        url = f"{COINGECKO_BASE_URL}/coins/{symbol_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        headers = {}
        if COINGECKO_API_KEY:
            headers["x-cg-pro-api-key"] = COINGECKO_API_KEY

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if "prices" not in data:
            raise Exception(f"Invalid response: {json.dumps(data)}")

        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        if "total_volumes" in data:
            vol_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
            df["volume"] = vol_df["volume"]
        else:
            df["volume"] = np.nan
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        log("WARN", f"CoinGecko fetch failed for {symbol_id.upper()}: {e}")
        return None


def fetch_from_yahoo(symbol="BTC-USD", period="7d", interval="1h"):
    try:
        import yfinance as yf
        data = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
        if data.empty:
            log("WARN", f"Empty Yahoo data for {symbol}")
            return None
        df = data.reset_index()
        df.rename(columns={
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        log("WARN", f"Yahoo fetch failed for {symbol}: {e}")
        return None


# -------------------------------
# DATA PIPELINE
# -------------------------------
def load_or_fetch(symbol, coingecko_id):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")

    # Try live fetch first
    log("INFO", f"[FETCH] Attempting fresh data for {symbol}")
    df = (
        fetch_from_binance(symbol)
        or fetch_from_coingecko(coingecko_id)
        or fetch_from_yahoo(symbol.replace("USDT", "-USD"))
    )

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


# -------------------------------
# TRAINING
# -------------------------------
def train_model(symbol, df):
    if df is None or df.empty:
        log("WARN", f"No data to train for {symbol}")
        return

    if len(df) < 50:
        log("WARN", f"Not enough data for {symbol} ({len(df)} rows)")
        return

    # Technical indicators
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["future_close"] = df["close"].shift(-1)
    df.dropna(inplace=True)

    feature_cols = ["open", "high", "low", "close", "volume", "sma_5", "sma_20", "rsi_14"]
    X = df[feature_cols].values
    y = df["future_close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)

    log("OK", f"{symbol} model trained (MAE={mae:.4f})")


# -------------------------------
# MAIN EXECUTION
# -------------------------------
def main():
    log("INFO", "[START] AI Model Training Sequence")

    pairs = {
        "BTCUSDT": "bitcoin",
        "XRPUSDT": "ripple",
        "GALAUSDT": "gala"
    }

    for symbol, cg_id in pairs.items():
        log("INFO", f"[START] Training {symbol}")
        df = load_or_fetch(symbol, cg_id)
        train_model(symbol, df)

    log("OK", "[FINISH] All models processed.")


if __name__ == "__main__":
    main()
