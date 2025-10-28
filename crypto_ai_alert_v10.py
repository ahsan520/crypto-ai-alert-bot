#!/usr/bin/env python3
# crypto_ai_alert_v11.py
# -----------------------------------------------------
# Crypto AI Hybrid v11 — Dynamic model discovery + unified cache
# -----------------------------------------------------
import os
import time
import json
import joblib
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_CACHE = BASE_DIR / "data_cache"
MODEL_DIR = BASE_DIR / "models"

DATA_CACHE.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
USE_COINGECKO_DEMO = os.getenv("USE_COINGECKO_DEMO", "true").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def fetch_latest_candle(symbol):
    """Fetch latest OHLCV candle from CoinGecko."""
    base_url = "https://api.coingecko.com/api/v3"
    headers = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY and not USE_COINGECKO_DEMO else {}

    pair = symbol.replace("USDT", "").lower()
    url = f"{base_url}/coins/{pair}/ohlc?vs_currency=usd&days=1"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 429:
            print(f"[WARN] Rate limited for {symbol}, retrying...")
            time.sleep(3)
            return None
        if resp.status_code != 200:
            print(f"[WARN] CoinGecko {symbol} → {resp.status_code}")
            return None

        data = resp.json()
        if not data:
            return None
        ts, o, h, l, c = data[-1]
        return {
            "timestamp": datetime.utcfromtimestamp(ts / 1000),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 1.0,
        }
    except Exception as e:
        print(f"[ERROR] Fetch failed for {symbol}: {e}")
        return None


def update_cache(symbol, candle):
    """Maintain rolling cache (3 recent candles)."""
    file_path = DATA_CACHE / f"{symbol}.json"
    try:
        data = json.load(open(file_path)) if file_path.exists() else []
    except Exception:
        data = []
    data.append(candle)
    data = sorted(data, key=lambda x: x["timestamp"])[-3:]
    json.dump(data, open(file_path, "w"), default=str, indent=2)


def send_telegram_message(message):
    """Send alert to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[INFO] Telegram not configured, skipping alert:\n{message}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        )
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")


# -----------------------------------------------------
# AI Feature & Decision Logic
# -----------------------------------------------------
def compute_features(df):
    """Use same 5 features as training."""
    df = df.sort_values("timestamp").tail(1)
    return df[["open", "high", "low", "close", "volume"]].values


def discover_models():
    """Dynamically discover all .pkl models in models/."""
    model_map = {}
    for model_file in MODEL_DIR.glob("*.pkl"):
        symbol = (
            model_file.stem.replace("_rf", "")
            .replace("_model", "")
            .upper()
        )
        model_map[symbol] = str(model_file)
    return model_map


def run_alert_cycle():
    """Main alert loop."""
    print(f"\n[INFO] Starting alert cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    models = discover_models()
    if not models:
        print("[WARN] No model files found in models/. Skipping alert cycle.")
        return

    for symbol, model_path in models.items():
        try:
            print(f"\n[MODEL] Processing {symbol} from {os.path.basename(model_path)}")

            if not os.path.exists(model_path):
                print(f"[WARN] Missing model: {model_path}")
                continue

            candle = fetch_latest_candle(symbol)
            if not candle:
                continue

            update_cache(symbol, candle)
            cache_file = DATA_CACHE / f"{symbol}.json"
            if not cache_file.exists():
                print(f"[WARN] Cache not found for {symbol}")
                continue

            data = json.load(open(cache_file))
            df = pd.DataFrame(data)
            if len(df) < 2:
                print(f"[WARN] Not enough candles for {symbol}")
                continue

            model = joblib.load(model_path)
            X = compute_features(df)
            y_pred = model.predict(X)[0]
            last_close = df["close"].iloc[-1]
            delta = y_pred - last_close

            decision = (
                "BUY ✅" if delta > 0 else
                "SELL ❌" if delta < 0 else
                "HOLD ⚪"
            )

            msg = (
                f"{symbol} → {decision}\n"
                f"Predicted: {round(y_pred, 4)}\n"
                f"Last close: {round(last_close, 4)}\n"
                f"Δ: {round(delta, 4)}\n"
                f"Time: {candle['timestamp']}"
            )
            print(msg)
            send_telegram_message(msg)
            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] {symbol} failed: {e}")

    print("\n[DONE] Alert cycle complete.\n")


# -----------------------------------------------------
# Main Entry
# -----------------------------------------------------
if __name__ == "__main__":
    print("[RUN] Executing crypto alert logic...")
    run_alert_cycle()
