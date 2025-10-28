#!/usr/bin/env python3
"""
spike_predictor.py - Production spike + trend model trainer

- Loads cached datasets (data_cache/{symbol}.csv) or fetches via yfinance
- Featurizes recent history and creates labels for:
    - spike: if max future return in window > threshold
    - dip: if min future return in window < -threshold
- Trains RandomForest classifiers for spike and dip detection
- Trains a simple trend LogisticRegression classifier
- Saves spike packs under models/spike_trend/{symbol}_spike_pack.pkl
"""

import os
import json
import traceback
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ========================
# CONFIG SETUP
# ========================
HERE = os.path.dirname(__file__)
CFG_PATH = os.path.join(HERE, "config.json")
if not os.path.exists(CFG_PATH):
    raise FileNotFoundError("config.json missing")

cfg = json.load(open(CFG_PATH))
SYMBOLS = cfg.get("symbols", ["BTCUSDT"])
MODEL_DIR = os.path.join(HERE, cfg.get("model_dir", "models"))
SPIKE_DIR = os.path.join(MODEL_DIR, "spike_trend")
os.makedirs(SPIKE_DIR, exist_ok=True)
CACHE_DIR = os.path.join(HERE, cfg.get("cache_dir", "data_cache"))
LOOKBACK_DAYS = int(cfg.get("lookback_days", 30))
INTERVAL = cfg.get("interval", "1h")
SPIKE_WINDOW = int(cfg.get("spike_window", 3))
SPIKE_THRESH = float(cfg.get("alert_probability_thresholds", {}).get("spike_alert", cfg.get("spike_confidence_threshold", 0.8)))

# Feature names used for spike models
SPIKE_FEATURES = ['return', 'vol_change', 'rsi', 'macd']

# ========================
# HELPERS
# ========================

def load_cache(symbol):
    """Load cached CSV file."""
    path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception as e:
        print(f"[WARN] failed loading cache {path}: {e}")
        return pd.DataFrame()


def featurize(df):
    """Feature engineering for spike/dip model training."""
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df['return'] = df['close'].pct_change().fillna(0)
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    df['rsi'] = df['close'].diff().ewm(alpha=1/14, adjust=False).mean().fillna(50)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df = df.dropna()
    return df


def label_spike(df, window=SPIKE_WINDOW, threshold=0.02):
    """Label 1 if price spikes within next `window` periods."""
    closes = df['close'].values
    lab = []
    for i in range(len(closes) - window):
        future = closes[i+1:i+1+window]
        m = (future.max() - closes[i]) / closes[i]
        lab.append(1 if m > threshold else 0)
    return pd.Series(lab, index=df.index[:-window])


def label_dip(df, window=SPIKE_WINDOW, threshold=0.02):
    """Label 1 if price dips within next `window` periods."""
    closes = df['close'].values
    lab = []
    for i in range(len(closes) - window):
        future = closes[i+1:i+1+window]
        m = (closes[i] - future.min()) / closes[i]
        lab.append(1 if m > threshold else 0)
    return pd.Series(lab, index=df.index[:-window])

# ========================
# TRAINING
# ========================

def train_for_symbol(symbol):
    """Train spike/dip/trend models for one crypto symbol."""
    print(f"[START] spike training for {symbol}")
    df = load_cache(symbol)
    if df.empty or len(df) < 120:
        print(f"[WARN] Not enough cached data for {symbol} ({len(df)} rows). Skipping.")
        return False

    try:
        dfx = featurize(df)
        if len(dfx) < (SPIKE_WINDOW + 50):
            print(f"[WARN] Not enough rows after featurize for {symbol}")
            return False

        # Labels
        spike_y = label_spike(dfx, window=SPIKE_WINDOW, threshold=0.02)
        dip_y = label_dip(dfx, window=SPIKE_WINDOW, threshold=0.02)
        dfx_trim = dfx.iloc[:-SPIKE_WINDOW]
        X = dfx_trim[SPIKE_FEATURES].fillna(0).values

        # Spike RF
        X_train, X_test, y_train, y_test = train_test_split(X, spike_y.values, test_size=0.2, random_state=42, stratify=spike_y.values)
        rf_spike = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_spike.fit(X_train, y_train)

        # Dip RF
        X_train, X_test, y_train, y_test = train_test_split(X, dip_y.values, test_size=0.2, random_state=42, stratify=dip_y.values)
        rf_dip = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_dip.fit(X_train, y_train)

        # Trend (LogisticRegression)
        trend_y = (dfx['return'].rolling(3).mean().shift(-1) > 0).astype(int).iloc[:-SPIKE_WINDOW].values
        log = LogisticRegression(max_iter=300)
        log.fit(X, trend_y)

        # Save spike pack
        pack = {
            "spike": rf_spike,
            "dip": rf_dip,
            "trend": log,
            "features": SPIKE_FEATURES,
            "trained_at_utc": datetime.utcnow().isoformat()
        }
        out_path = os.path.join(SPIKE_DIR, f"{symbol}_spike_pack.pkl")
        joblib.dump(pack, out_path)
        print(f"[SAVED] spike pack for {symbol} -> {out_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Exception training spike predictor for {symbol}: {e}\n{traceback.format_exc()}")
        return False

# ========================
# MAIN + WRAPPER
# ========================

def main(symbols=None):
    """Run spike predictor training for all configured symbols."""
    if symbols is None:
        symbols = SYMBOLS

    results = {}
    for s in symbols:
        ok = train_for_symbol(s)
        results[s] = ok

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = {"timestamp_utc": datetime.utcnow().isoformat(), "results": results}

    try:
        telemetry_dir = os.path.join(HERE, cfg.get("telemetry_viewer", {}).get("log_dir", "telemetry_logs"))
        os.makedirs(telemetry_dir, exist_ok=True)
        with open(os.path.join(telemetry_dir, f"spike_train_summary_{ts}.json"), "w") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass

    print("[DONE] spike predictor run finished.")


# 🔧 UPDATED WRAPPER
def run_spike_predictor(symbols=None):
    """Wrapper for alert_v10 compatibility."""
    return main(symbols)


if __name__ == "__main__":
    main()
