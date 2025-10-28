#!/usr/bin/env python3
"""
train_ai_model.py - Core trainer for Crypto AI Hybrid v13.7

- Loads cached or freshly fetched data from utils/data_fetcher
- Trains RandomForest models for each symbol
- Saves trained models and JSON training summaries
- Provides train_model_if_needed() for external modules (v10 alert logic)
"""

import os
import json
import joblib
import logging
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from utils.data_fetcher import get_data, CACHE_DIR

# ==============================
# CONFIGURATION
# ==============================
SYMBOLS = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]
MODEL_DIR = "models"
SUMMARY_DIR = "training_summary"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ==============================
# TRAINING FUNCTION
# ==============================
def train_model(symbol, df):
    """Train a RandomForest model for one symbol."""
    if df is None or len(df) < 100:
        logging.warning(f"[WARN] Not enough data for {symbol} ({0 if df is None else len(df)} rows)")
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target"] = df["close"].shift(-1)
    df.dropna(inplace=True)

    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"[OK] Model trained and saved → {model_path}")

    return model


# ==============================
# WRAPPER FOR OTHER MODULES
# ==============================
def train_model_if_needed(symbols=None):
    """
    Public function for external imports (used by crypto_ai_alert_v10.py).

    Checks if models exist or are stale (>2h old). Retrains if needed.
    """
    symbols = symbols or SYMBOLS
    updated_models = {}

    for symbol in symbols:
        model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")

        retrain = True
        if os.path.exists(model_path):
            age_seconds = (datetime.now().timestamp() - os.path.getmtime(model_path))
            retrain = age_seconds > 7200  # 2 hours
            if retrain:
                logging.info(f"[INFO] {symbol} model is stale ({int(age_seconds)}s old) — retraining.")
            else:
                logging.info(f"[INFO] {symbol} model is fresh (<2h). Skipping retrain.")
                updated_models[symbol] = model_path
                continue

        df = get_data(symbol)
        if df is None or df.empty:
            logging.warning(f"[WARN] Skipping {symbol}: no data returned from fetcher.")
            continue

        model = train_model(symbol, df)
        if model is not None:
            updated_models[symbol] = model_path

    logging.info(f"[SUMMARY] Updated models: {list(updated_models.keys())}")
    return updated_models


# ==============================
# MAIN ENTRY POINT
# ==============================
def main():
    logging.info("[START] 🧠 Crypto AI Model Training Sequence")

    summary = {
        "timestamp": str(datetime.utcnow()),
        "symbols": {},
    }

    for symbol in SYMBOLS:
        try:
            logging.info(f"[FETCH] Loading data for {symbol}")
            df = get_data(symbol)

            if df is None or df.empty:
                logging.warning(f"[WARN] No valid data retrieved for {symbol}. Skipping.")
                continue

            source_tag = df.attrs.get("source", "cache/unknown")
            logging.info(f"[SOURCE] {symbol} → {source_tag} ({len(df)} rows)")

            cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")
            if not os.path.exists(cache_file):
                logging.warning(f"[WARN] Cache file missing for {symbol}: {cache_file}")
            else:
                logging.info(f"[CACHE] {symbol} data cached → {cache_file}")

            # Train and save model
            model = train_model(symbol, df)

            summary["symbols"][symbol] = {
                "rows": len(df),
                "model_trained": model is not None,
                "source": source_tag,
                "cache_file": cache_file,
                "model_path": os.path.join(MODEL_DIR, f"{symbol}_model.pkl"),
            }

        except Exception as e:
            logging.error(f"[ERROR] Training failed for {symbol}: {e}", exc_info=True)

    # Save JSON training summary
    summary_path = os.path.join(
        SUMMARY_DIR,
        f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"[DONE] Training summary saved → {summary_path}")
    logging.info("[FINISH] ✅ All models processed successfully.")


if __name__ == "__main__":
    main()
