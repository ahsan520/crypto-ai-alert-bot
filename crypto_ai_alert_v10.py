import os
import time
import joblib
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# ✅ Configuration
# -------------------------------------------------------------------
CONFIG = {
    "BTCUSDT": {"model": "models/BTCUSDT_model.pkl"},
    "XRPUSDT": {"model": "models/XRPUSDT_model.pkl"},
    "GALAUSDT": {"model": "models/GALAUSDT_model.pkl"},
}

MAPPING = {
    "BTCUSDT": "BTC-USD",
    "XRPUSDT": "XRP-USD",
    "GALAUSDT": "GALA-USD"
}

COIN_ID_MAP = {
    "BTCUSDT": "bitcoin",
    "XRPUSDT": "ripple",
    "GALAUSDT": "gala"
}

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# -------------------------------------------------------------------
# ✅ Helper: Send Telegram Alert
# -------------------------------------------------------------------
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")

# -------------------------------------------------------------------
# ✅ Fetch Latest Candle (CoinGecko → Yahoo → Cache)
# -------------------------------------------------------------------
def fetch_latest_candle(symbol):
    coin_id = COIN_ID_MAP.get(symbol)
    yf_sym = MAPPING.get(symbol)

    # 1️⃣ Try CoinGecko first
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": "2", "interval": "hourly"}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            prices = data.get("prices", [])
            vols = data.get("total_volumes", [])
            if prices:
                last_ts, last_price = prices[-1]
                last_vol = vols[-1][1] if vols else 0
                return {
                    "timestamp": datetime.utcfromtimestamp(last_ts / 1000),
                    "open": float(prices[-2][1]) if len(prices) > 1 else float(last_price),
                    "high": float(max(p[1] for p in prices[-5:])),
                    "low": float(min(p[1] for p in prices[-5:])),
                    "close": float(last_price),
                    "volume": float(last_vol)
                }
        else:
            print(f"[WARN] CoinGecko returned {r.status_code} for {symbol}")
    except Exception as e:
        print(f"[WARN] CoinGecko fetch failed for {symbol}: {e}")

    # 2️⃣ Fallback to Yahoo Finance
    try:
        df = yf.download(yf_sym, period="2d", interval="1h", progress=False)
        if not df.empty:
            last = df.tail(1)
            return {
                "timestamp": last.index[-1].to_pydatetime(),
                "open": float(last["Open"].iloc[0]),
                "high": float(last["High"].iloc[0]),
                "low": float(last["Low"].iloc[0]),
                "close": float(last["Close"].iloc[0]),
                "volume": float(last["Volume"].iloc[0]),
            }
    except Exception as e:
        print(f"[WARN] Yahoo fetch failed for {symbol}: {e}")

    # 3️⃣ Fallback to cached CSV
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            if not df.empty:
                last = df.iloc[-1]
                return {
                    "timestamp": last["timestamp"],
                    "open": last["open"],
                    "high": last["high"],
                    "low": last["low"],
                    "close": last["close"],
                    "volume": last["volume"],
                }
        except Exception as e:
            print(f"[WARN] Cache read failed for {symbol}: {e}")

    print(f"[WARN] No valid data for {symbol}")
    return None

# -------------------------------------------------------------------
# ✅ Update Cache
# -------------------------------------------------------------------
def update_cache(symbol, candle):
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    df_new = pd.DataFrame([candle])
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            df = pd.concat([df, df_new]).drop_duplicates(subset=["timestamp"], keep="last")
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_csv(cache_path, index=False)

# -------------------------------------------------------------------
# ✅ Feature Computation
# -------------------------------------------------------------------
def compute_features(df):
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(10).std()
    df["ma"] = df["close"].rolling(10).mean()
    df = df.dropna().tail(1)
    return df[["return", "volatility", "ma"]].values

# -------------------------------------------------------------------
# ✅ Prediction + Alert Logic
# -------------------------------------------------------------------
def run_alert_cycle():
    print(f"\n[INFO] Alert cycle started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for symbol, cfg in CONFIG.items():
        try:
            candle = fetch_latest_candle(symbol)
            if not candle:
                continue
            update_cache(symbol, candle)

            df = pd.read_csv(os.path.join(CACHE_DIR, f"{symbol}.csv"), parse_dates=["timestamp"])
            if len(df) < 20:
                print(f"[WARN] Not enough data for {symbol}")
                continue

            X = compute_features(df)
            model = joblib.load(cfg["model"])
            y_pred = model.predict(X)[0]

            decision = {1: "BUY ✅", 0: "HOLD ⚪", -1: "SELL ❌"}.get(y_pred, "HOLD ⚪")
            msg = f"{symbol}: {decision}\nPrice: {round(candle['close'], 4)}\nTime: {candle['timestamp']}"
            print(msg)
            send_telegram_message(msg)
        except Exception as e:
            print(f"[ERROR] Failed {symbol}: {e}")

# -------------------------------------------------------------------
# ✅ Scheduler
# -------------------------------------------------------------------
if __name__ == "__main__":
    while True:
        run_alert_cycle()
        print("[INFO] Sleeping 30 minutes...")
        time.sleep(1800)
