#!/usr/bin/env python3
# -----------------------------------------------------
# Crypto AI Hybrid v12 — Source-aware Alert Engine
# -----------------------------------------------------
import os
import time
import json
import joblib
import logging
import requests
import pandas as pd
from datetime import datetime
from utils.data_fetcher import get_data

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
MODEL_DIR = "models"
SYMBOLS = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -----------------------------------------------------
# Telegram Alerts
# -----------------------------------------------------
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.info(f"[INFO] Telegram not configured, skipping:\n{message}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        )
    except Exception as e:
        logging.error(f"[TELEGRAM ERROR] {e}")


# -----------------------------------------------------
# Prediction & Alert Logic
# -----------------------------------------------------
def compute_features(df):
    df = df.sort_values("timestamp").tail(1)
    return df[["open", "high", "low", "close", "volume"]].values


def run_alert_cycle():
    logging.info(f"[INFO] Starting alert cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for symbol in SYMBOLS:
        try:
            model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
            if not os.path.exists(model_path):
                logging.warning(f"[WARN] Model not found: {model_path}")
                continue

            df = get_data(symbol)
            if df is None or len(df) < 2:
                logging.warning(f"[WARN] No recent data for {symbol}")
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

            src = df.attrs.get("source", "unknown")
            msg = (
                f"{symbol} → {decision}\n"
                f"Predicted: {round(y_pred, 4)}\n"
                f"Last Close: {round(last_close, 4)}\n"
                f"Δ: {round(delta, 4)}\n"
                f"Source: {src}\n"
                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )

            logging.info(msg)
            send_telegram(msg)
            time.sleep(2)

        except Exception as e:
            logging.error(f"[ERROR] {symbol}: {e}")

    logging.info("[DONE] Alert cycle complete.")


if __name__ == "__main__":
    logging.info("[RUN] Executing Crypto AI Alert Engine (v12)")
    run_alert_cycle()
