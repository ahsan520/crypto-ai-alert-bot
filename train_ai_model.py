#!/usr/bin/env python3
import os
import json
import joblib
import logging
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from utils.data_fetcher import get_data  # ✅ unified fetcher import

# =========================================
# CONFIGURATION
# =========================================
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

# =========================================
# MODEL TRAINING FUNCTION
# =========================================
def train_model(symbol: str, df: pd.DataFrame):
    """Train and save RandomForest model for given symbol."""
    if df is None or len(df) < 100:
        logging.warning(f"[WARN] Insufficient data for {symbol} ({0 if df is None else len(df)} rows)")
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target"] = df["close"].shift(-1)
    df.dropna(inplace=True)

    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"[OK] Model trained → {model_path}")

    return model


# =========================================
# MAIN TRAINING LOOP
# =========================================
def main():
    logging.info("[START] === AI Model Training Sequence ===")

    summary = {
        "timestamp": str(datetime.utcnow()),
        "symbols": {}
    }

    for symbol in SYMBOLS:
        logging.info(f"[FETCH] Getting data for {symbol} ...")
        df = get_data(symbol)

        cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")

        if df is None or df.empty:
            logging.warning(f"[WARN] No data retrieved for {symbol}. Skipping.")
            continue

        # ✅ Save valid data to cache
        try:
            df.to_csv(cache_file, index=False)
            logging.info(f"[CACHE] {symbol} data saved → {cache_file}")
        except Exception as e:
            logging.error(f"[ERROR] Could not save cache for {symbol}: {e}")

        model = train_model(symbol, df)

        summary["symbols"][symbol] = {
            "rows": len(df),
            "model_trained": model is not None,
            "cache_file": cache_file,
            "model_path": os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
        }

    # =========================================
    # SAVE TRAINING SUMMARY
    # =========================================
    summary_file = f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path = os.path.join(SUMMARY_DIR, summary_file)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"[DONE] Training summary saved → {summary_path}")
    logging.info(f"[CACHE CONTENTS] {os.listdir(CACHE_DIR)}")
    logging.info("[FINISH] === All models processed successfully ===")


if __name__ == "__main__":
    main()
