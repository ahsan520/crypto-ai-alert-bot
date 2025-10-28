#!/usr/bin/env python3
# crypto_ai_alert_v10.py
# -----------------------------------------------------
# Crypto AI Hybrid v10 — Real-time alert + telemetry
# -----------------------------------------------------
import os
import time
import json
import joblib
import requests
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
CACHE_DIR = "telemetry_logs"
os.makedirs(CACHE_DIR, exist_ok=True)

CONFIG = {
    "BTCUSDT": {"model": "models/btc_rf.pkl"},
    "XRPUSDT": {"model": "models/xrp_rf.pkl"},
    "GALAUSDT": {"model": "models/gala_rf.pkl"},
}

# Use environment variables for secure secrets
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
USE_COINGECKO_DEMO = os.getenv("USE_COINGECKO_DEMO", "true").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def fetch_latest_candle(symbol):
    """Fetch latest OHLCV candle from CoinGecko or fallback."""
    base_url = "https://api.coingecko.com/api/v3"
    if COINGECKO_API_KEY and not USE_COINGECKO_DEMO:
        headers = {"x-cg-pro-api-key": COINGECKO_API_KEY}
    else:
        headers = {}

    pair = symbol.replace("USDT", "").lower()
    url = f"{base_url}/coins/{pair}/ohlc?vs_currency=usd&days=1"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 429:
            print(f"[WARN] CoinGecko rate limit for {symbol}, sleeping...")
            time.sleep(3)
            return None
        if resp.status_code != 200:
            print(f"[WARN] CoinGecko returned {resp.status_code} for {symbol}")
            return None

        data = resp.json()
        if not data or len(data) == 0:
            return None
        # latest candle = [timestamp, open, high, low, close]
        ts, o, h, l, c = data[-1]
        return {
            "timestamp": datetime.utcfromtimestamp(ts / 1000),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 1.0,  # CoinGecko free API doesn’t return volume
        }
    except Exception as e:
        print(f"[ERROR] Failed to fetch {symbol}: {e}")
        return None


def update_cache(symbol, candle):
    """Maintain rolling JSON cache of last 3 candles per symbol."""
    file_path = os.path.join(CACHE_DIR, f"{symbol}_0.json")
    data = []
    if os.path.exists(file_path):
        try:
            data = json.load(open(file_path))
        except Exception:
            data = []
    data.append(candle)
    data = sorted(data, key=lambda x: x["timestamp"])[-3:]
    json.dump(data, open(file_path, "w"), default=str, indent=2)


def send_telegram_message(message):
    """Send alert to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[INFO] Telegram not configured. Skipping alert:\n{message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")


# -----------------------------------------------------
# Feature + AI Logic
# -----------------------------------------------------
def compute_features(df):
    """Ensure 5 feature columns same as training."""
    df = df.sort_values("timestamp").tail(1)
    X = df[["open", "high", "low", "close", "volume"]].values
    return X


def run_alert_cycle():
    """Main alert loop."""
    print(f"\n[INFO] Alert cycle started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for symbol, cfg in CONFIG.items():
        try:
            candle = fetch_latest_candle(symbol)
            if not candle:
                continue
            update_cache(symbol, candle)

            # build DataFrame for this symbol
            file_path = os.path.join(CACHE_DIR, f"{symbol}_0.json")
            data = json.load(open(file_path))
            df = pd.DataFrame(data)
            if len(df) < 2:
                print(f"[WARN] Not enough data for {symbol}")
                continue

            X = compute_features(df)
            model = joblib.load(cfg["model"])
            y_pred = model.predict(X)[0]

            last_close = df["close"].iloc[-1]
            delta = y_pred - last_close
            if delta > 0:
                decision = "BUY ✅"
            elif delta < 0:
                decision = "SELL ❌"
            else:
                decision = "HOLD ⚪"

            msg = (
                f"{symbol}: {decision}\n"
                f"Predicted next price: {round(y_pred, 4)}\n"
                f"Last close: {round(last_close, 4)}\n"
                f"Δ = {round(delta, 4)}\n"
                f"Time: {candle['timestamp']}"
            )
            print(msg)
            send_telegram_message(msg)

            # brief delay to avoid rate-limit
            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] Failed {symbol}: {e}")


# -----------------------------------------------------
# Main Entry
# -----------------------------------------------------
if __name__ == "__main__":
    print('echo "[RUN] Executing crypto alert logic..."')
    run_alert_cycle()
    print("[DONE] Alert cycle complete.")
