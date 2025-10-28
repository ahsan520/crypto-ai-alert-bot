import os
import json
import time
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from utils.data_fetcher import get_data  # ✅ unified fetcher import

# ==============================
# CONFIGURATION
# ==============================
SYMBOLS = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]

CACHE_DIR = "data_cache"
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
    if df is None or len(df) < 100:
        logging.warning(f"[WARN] Not enough data for {symbol} ({0 if df is None else len(df)} rows)")
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

    for symbol in SYMBOLS:
        logging.info(f"[FETCH] Loading data for {symbol}")
        df = get_data(symbol)  # ✅ centralized fetch (CoinGecko → Yahoo → Cache)
        logging.info(f"[DATA] {symbol} → {len(df)} rows loaded")

        model = train_model(symbol, df)
        summary["symbols"][symbol] = {
            "rows": len(df) if df is not None else 0,
            "model_trained": model is not None,
        }

    # Save training summary JSON
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
