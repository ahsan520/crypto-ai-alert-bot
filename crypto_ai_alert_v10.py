# crypto_ai_alert_v10.py
import os
import pandas as pd
from datetime import datetime
from utils.data_fetcher import fetch_and_cache
from spike_predictor import run_spike_predictor
from train_ai_model import train_model_if_needed

MODEL_DIR = "models"
DATA_DIR = "data_cache"
MAX_MODEL_AGE_HRS = 2

def is_model_stale(model_path):
    if not os.path.exists(model_path):
        print(f"[WARN] Missing model: {model_path}")
        return True
    age_seconds = (datetime.now().timestamp() - os.path.getmtime(model_path))
    age_hrs = age_seconds / 3600
    print(f"[INFO] {model_path} age: {age_hrs:.2f} hours")
    return age_hrs > MAX_MODEL_AGE_HRS

def ensure_data(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        print(f"[INFO] Missing or small cache for {symbol}. Re-fetching...")
        fetch_and_cache(symbol)
    try:
        df = pd.read_csv(path)
        if len(df) < 50:
            print(f"[WARN] Not enough data for {symbol} ({len(df)} rows). Re-fetching...")
            fetch_and_cache(symbol)
    except Exception:
        fetch_and_cache(symbol)

def main():
    print(f"[RUN] Executing crypto alert logic at {datetime.now()}")
    symbols = ["BTCUSDT", "XRPUSDT", "GALAUSDT"]

    # 1️⃣ Ensure data available
    for sym in symbols:
        ensure_data(sym)

    # 2️⃣ Train if needed
    for sym in symbols:
        model_path = os.path.join(MODEL_DIR, f"{sym}_model.pkl")
        if is_model_stale(model_path):
            print(f"[TRAIN] Training model for {sym} ...")
            train_model_if_needed(sym)
        else:
            print(f"[SKIP] Model fresh for {sym}.")

    # 3️⃣ Run predictor
    run_spike_predictor(symbols)
    print("[DONE] Alert cycle complete.")

if __name__ == "__main__":
    main()
