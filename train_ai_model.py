#!/usr/bin/env python3
"""
train_ai_model.py — Crypto AI Hybrid v13.1
Fetches market data (yfinance w/ fallback cache), computes indicators,
trains hybrid AI models (RandomForest + LogisticRegression), and saves models.
"""

import os, json, time, pickle, traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------
# CONFIG
# -------------------------------
CONFIG_PATH = "config.json"
cfg = json.load(open(CONFIG_PATH))
MODEL_DIR = cfg.get("model_dir", "models")
CACHE_DIR = cfg.get("cache_dir", "data_cache")
LOOKBACK_DAYS = cfg.get("lookback_days", 30)
INTERVAL = cfg.get("interval", "1h")
SYMBOLS = cfg.get("symbols", [])
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------------
# HELPERS
# -------------------------------
def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] {msg}")

def save_cache(symbol, df):
    path = os.path.join(CACHE_DIR, f"{symbol}_cache.csv")
    df.to_csv(path, index=False)
    return path

def load_cache(symbol):
    path = os.path.join(CACHE_DIR, f"{symbol}_cache.csv")
    if os.path.exists(path):
        mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
        if datetime.utcnow() - mtime < timedelta(days=1):
            return pd.read_csv(path)
    return None

def fetch_data(symbol):
    """Fetch from Yahoo Finance with 24h cache fallback."""
    df = load_cache(symbol)
    if df is not None:
        log(f"[CACHE] Loaded cached data for {symbol}")
        return df

    try:
        log(f"[FETCH] Downloading {symbol} from Yahoo Finance...")
        yf_symbol = symbol.replace("USDT", "-USD")  # Binance → Yahoo format
        df = yf.download(yf_symbol, period=f"{LOOKBACK_DAYS}d", interval=INTERVAL, progress=False)
        if df is None or df.empty:
            raise ValueError("Empty dataframe from yfinance")

        # ✅ normalize columns
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()

        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in data for {symbol}")

        save_cache(symbol, df)
        log(f"[INFO] Cached data for {symbol} ({len(df)} rows)")
        return df
    except Exception as e:
        log(f"[ERROR] Could not fetch data for {symbol}: {e}")
        return None

# -------------------------------
# FEATURES
# -------------------------------
def compute_features(df):
    """Compute basic indicators: RSI, EMA ratios, MACD, etc."""
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['rsi'] = compute_rsi(df['close'])
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_ratio'] = df['ema_fast'] / df['ema_slow']
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    df['returns'] = df['close'].pct_change()
    df['future_return'] = df['close'].shift(-3) / df['close'] - 1
    df['target'] = np.where(df['future_return'] > 0.002, 1, np.where(df['future_return'] < -0.002, -1, 0))
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# -------------------------------
# TRAINING
# -------------------------------
def train_model(df, symbol):
    features = ['rsi', 'ema_ratio', 'macd', 'macd_hist', 'returns']
    X = df[features]
    y = df['target']

    # Train two models for hybrid
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    rf.fit(X, y)
    logreg.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_hybrid.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"rf": rf, "logreg": logreg, "features": features}, f)
    return model_path

# -------------------------------
# MAIN LOOP
# -------------------------------
def train_all():
    summary = {}
    log(f"[START] Training run — {datetime.utcnow().isoformat()}")
    for symbol in SYMBOLS:
        try:
            df = fetch_data(symbol)
            if df is None or df.empty:
                summary[symbol] = {"trained": False, "error": "no_data"}
                continue

            df_feat = compute_features(df)
            if len(df_feat) < 50:
                summary[symbol] = {"trained": False, "error": "too_few_rows"}
                continue

            model_path = train_model(df_feat, symbol)
            summary[symbol] = {"trained": True, "rows": len(df_feat), "model": model_path}
            log(f"[OK] {symbol} trained ({len(df_feat)} samples)")
        except Exception as e:
            summary[symbol] = {"trained": False, "error": str(e)}
            log(f"[ERROR] Exception training {symbol}: {traceback.format_exc()}")

    # save telemetry
    telemetry_dir = "telemetry_logs"
    os.makedirs(telemetry_dir, exist_ok=True)
    fname = f"train_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(telemetry_dir, fname), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[DONE] Training telemetry saved → {fname}")
    return summary

if __name__ == "__main__":
    train_all()
