#!/usr/bin/env python3
"""
train_ai_model.py (production)

- Primary data source: yfinance (Yahoo)
- Fallback: Google finance reachability check + local cache
- Offline cache: data_cache/{symbol}.csv (only last successful)
- Trains per-symbol LogisticRegression + RandomForest ensemble and saves model pack

Outputs:
- models/crypto_ai_model_{model_version}_{symbol}.pkl
- telemetry_logs/train_summary_{timestamp}.json

Requirements:
pip install yfinance pandas numpy scikit-learn joblib requests
"""
import os
import json
import time
from datetime import datetime, timedelta
import traceback

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -----------------------
# Utilities & Config load
# -----------------------
HERE = os.path.dirname(__file__)
CFG_PATH = os.path.join(HERE, "config.json")

def load_config():
    if not os.path.exists(CFG_PATH):
        raise FileNotFoundError("config.json not found in repo root")
    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)
    # defaults
    cfg.setdefault("symbols", ["BTCUSDT"])
    cfg.setdefault("interval", "1h")
    cfg.setdefault("lookback_days", 30)
    cfg.setdefault("model_dir", "models")
    cfg.setdefault("cache_dir", "data_cache")
    cfg.setdefault("model_version", "v1")
    cfg.setdefault("max_cache_age_hours", 6)
    cfg.setdefault("alert_probability_thresholds", {"buy":0.7,"sell":0.3})
    cfg.setdefault("telemetry", {"save_json_artifacts": True})
    return cfg

cfg = load_config()
SYMBOLS = cfg["symbols"]
INTERVAL = cfg["interval"]
LOOKBACK_DAYS = int(cfg["lookback_days"])
MODEL_DIR = os.path.join(HERE, cfg["model_dir"])
CACHE_DIR = os.path.join(HERE, cfg["cache_dir"])
MODEL_VERSION = cfg["model_version"]
MAX_CACHE_AGE_HOURS = int(cfg.get("max_cache_age_hours", 6))
TELEMETRY_DIR = os.path.join(HERE, cfg.get("telemetry", {}).get("telemetry_dir", "telemetry_logs"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TELEMETRY_DIR, exist_ok=True)

# Mapping trading pair to yfinance symbol
DEFAULT_MAPPING = {
    "BTCUSDT": "BTC-USD",
    "XRPUSDT": "XRP-USD",
    "GALAUSDT": "GALA-USD"
}

MAPPING = cfg.get("symbol_mapping", DEFAULT_MAPPING)

# -----------------------
# Data fetching + caching
# -----------------------
def cache_path_for(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}.csv")

def is_cache_stale(path, max_age_hours=MAX_CACHE_AGE_HOURS):
    if not os.path.exists(path):
        return True
    mod = datetime.utcfromtimestamp(os.path.getmtime(path))
    return (datetime.utcnow() - mod) > timedelta(hours=max_age_hours)

def fetch_from_yfinance(symbol, period_days=LOOKBACK_DAYS, interval=INTERVAL):
    yf_sym = MAPPING.get(symbol, None)
    if not yf_sym:
        raise ValueError(f"No mapping for symbol {symbol} -> yfinance symbol")
    try:
        df = yf.download(yf_sym, period=f"{period_days}d", interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={"Datetime":"timestamp","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df = df[["timestamp","open","high","low","close","volume"]]
        return df
    except Exception as e:
        print(f"[WARN] yfinance fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def google_reachable_check(symbol):
    # simple reachability check to google finance page (not parsing OHLC)
    yf_sym = MAPPING.get(symbol, None)
    if not yf_sym:
        return False
    url = f"https://www.google.com/finance/quote/{yf_sym.replace('-',':')}"
    try:
        r = requests.get(url, timeout=8)
        return r.status_code == 200
    except Exception:
        return False

def load_cached(symbol):
    path = cache_path_for(symbol)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return df
    except Exception as e:
        print(f"[WARN] Failed to load cache for {symbol}: {e}")
        return pd.DataFrame()

def save_cache(symbol, df):
    try:
        df.to_csv(cache_path_for(symbol), index=False)
    except Exception as e:
        print(f"[WARN] Failed to save cache for {symbol}: {e}")

def get_price_data(symbol, period_days=LOOKBACK_DAYS, interval=INTERVAL):
    """
    Try yfinance -> if empty or failed, check Google reachability -> if reachable and cache exists use cache,
    else if cache exists use cache even if stale (last resort).
    """
    print(f"[INFO] Fetching data for {symbol} (period={period_days}d interval={interval})")
    df = fetch_from_yfinance(symbol, period_days=period_days, interval=interval)
    if not df.empty:
        save_cache(symbol, df)
        print(f"[INFO] Fetched {len(df)} rows for {symbol} from Yahoo and cached.")
        return df

    print(f"[WARN] Yahoo returned no data for {symbol}. Trying fallback strategy.")
    reachable = google_reachable_check(symbol)
    if reachable:
        print(f"[INFO] Google Finance reachable for {symbol}. Will attempt to use cache if present.")
    else:
        print(f"[WARN] Google Finance not reachable or blocked for {symbol}.")

    # fallback to cache if available
    cached = load_cached(symbol)
    if not cached.empty:
        print(f"[INFO] Loaded {len(cached)} cached rows for {symbol}. (may be stale)")
        return cached

    print(f"[ERROR] No data available for {symbol} (live or cached).")
    return pd.DataFrame()

# -----------------------
# Feature engineering
# -----------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def atr(df, period=14):
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(0)

def linear_slope(arr):
    y = np.array(arr, dtype=float)
    x = np.arange(len(y))
    xm = x.mean(); ym = y.mean()
    num = ((x - xm)*(y - ym)).sum()
    den = ((x - xm)**2).sum()
    return float(num/den) if den != 0 else 0.0

def compute_features(df):
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open']  = pd.to_numeric(df['open'], errors='coerce')
    df['high']  = pd.to_numeric(df['high'], errors='coerce')
    df['low']   = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    df['rsi'] = compute_rsi(df['close'], 14)
    df['rsi_diff'] = df['rsi'] - 50.0
    df['slope'] = df['close'].rolling(12).apply(lambda x: linear_slope(x), raw=False).fillna(0)
    df['ema20'] = ema(df['close'], 20)
    df['ema_ratio'] = (df['close'] / df['ema20']) - 1.0
    ema_fast = ema(df['close'], 12)
    ema_slow = ema(df['close'], 26)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = ema(df['macd'], 9)
    df['atr'] = atr(df, 14)
    df['atr_ratio'] = df['atr'] / df['close'].replace(0, np.nan)
    df['body'] = (df['close'] - df['open']).abs()
    df['body_ratio'] = df['body'] / (df['high'] - df['low']).replace(0, np.nan)
    df['upper_wick'] = df['high'] - df[['close','open']].max(axis=1)
    df['lower_wick'] = df[['close','open']].min(axis=1) - df['low']
    df['upper_wick_ratio'] = df['upper_wick'] / (df['high'] - df['low']).replace(0, np.nan)
    df['lower_wick_ratio'] = df['lower_wick'] / (df['high'] - df['low']).replace(0, np.nan)
    df['vol_avg20'] = df['volume'].rolling(20).mean().fillna(0)
    df['vol_ratio'] = df['volume'] / df['vol_avg20'].replace(0, np.nan)
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    # cleanup infinities / NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# -----------------------
# Labeling
# -----------------------
def label_data(df, up_threshold=0.0, forward_shift=1):
    """
    Label as 1 if next 'forward_shift' close return > up_threshold else 0.
    """
    df = df.copy()
    df['close_shift'] = df['close'].shift(-forward_shift)
    df['return_next'] = (df['close_shift'] - df['close']) / df['close']
    df = df.dropna()
    df['label'] = (df['return_next'] > up_threshold).astype(int)
    return df

# -----------------------
# Training per symbol
# -----------------------
def train_models_for_symbol(symbol):
    summary = {"symbol": symbol, "trained": False, "error": None}
    try:
        # 1) Get data (fetch or cache)
        df = get_price_data(symbol, period_days=LOOKBACK_DAYS, interval=INTERVAL)
        if df.empty or len(df) < 120:
            msg = f"Not enough data for {symbol} (rows={len(df)})"
            print(f"[WARN] {msg}")
            summary["error"] = msg
            return summary

        # 2) Feature engineering
        df_feat = compute_features(df)
        if df_feat.empty or len(df_feat) < 120:
            msg = f"Not enough feature rows for {symbol} after featurization (rows={len(df_feat)})"
            print(f"[WARN] {msg}")
            summary["error"] = msg
            return summary

        # 3) label
        df_lab = label_data(df_feat, up_threshold=0.0, forward_shift=1)
        if df_lab.empty or len(df_lab) < 120 or len(df_lab['label'].unique()) < 2:
            msg = f"Insufficient labeled data or no variance for {symbol}"
            print(f"[WARN] {msg}")
            summary["error"] = msg
            return summary

        FEATURES = [
            'rsi','rsi_diff','slope','ema_ratio','macd','macd_signal',
            'atr_ratio','body_ratio','upper_wick_ratio','lower_wick_ratio',
            'vol_ratio','vol_change'
        ]
        # ensure features present
        for f in FEATURES:
            if f not in df_lab.columns:
                df_lab[f] = 0.0

        df_lab = df_lab.dropna(subset=FEATURES + ['label'])
        X = df_lab[FEATURES].values
        y = df_lab['label'].values

        # 4) scale & train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # logistic
        log = LogisticRegression(max_iter=400)
        log.fit(X_train, y_train)

        # random forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)

        # ensemble probabilities on test
        try:
            p_log = log.predict_proba(X_test)[:,1]
        except Exception:
            p_log = log.predict(X_test)

        try:
            p_rf = rf.predict_proba(X_test)[:,1]
        except Exception:
            p_rf = rf.predict(X_test)

        hybrid_prob = 0.5 * np.array(p_log) + 0.5 * np.array(p_rf)
        preds = (hybrid_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        print(f"[INFO] {symbol} trained. hybrid_acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} rows={len(df_lab)}")

        # 5) save models & scaler
        model_pack = {
            "version": MODEL_VERSION,
            "symbol": symbol,
            "features": FEATURES,
            "scaler": scaler,
            "logistic": log,
            "random_forest": rf,
            "trained_at_utc": datetime.utcnow().isoformat()
        }
        filename = os.path.join(MODEL_DIR, f"crypto_ai_model_{MODEL_VERSION}_{symbol}.pkl")
        joblib.dump(model_pack, filename)

        summary.update({
            "trained": True,
            "rows": len(df_lab),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "model_path": filename,
            "trained_at_utc": model_pack["trained_at_utc"]
        })

        return summary

    except Exception as e:
        err = traceback.format_exc()
        print(f"[ERROR] Exception training {symbol}: {e}\n{err}")
        summary["error"] = str(e)
        return summary

# -----------------------
# Telemetry write
# -----------------------
def write_telemetry(results):
    if not cfg.get("telemetry", {}).get("save_json_artifacts", True):
        return
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_version": MODEL_VERSION,
        "results": results
    }
    fname = os.path.join(TELEMETRY_DIR, f"train_summary_{ts}.json")
    try:
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] Training telemetry saved to {fname}")
    except Exception as e:
        print(f"[WARN] Could not write telemetry file: {e}")

# -----------------------
# Main
# -----------------------
def main():
    print(f"[START] train_ai_model.py v{MODEL_VERSION} - {datetime.utcnow().isoformat()} UTC")
    results = []
    for s in SYMBOLS:
        try:
            res = train_models_for_symbol(s)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Unexpected error for {s}: {e}")
            results.append({"symbol": s, "trained": False, "error": str(e)})
    write_telemetry(results)
    print("[DONE] Training run finished.")
    # print summary table
    for r in results:
        if r.get("trained"):
            print(f" - {r['symbol']}: trained True rows={r.get('rows')} acc={r.get('accuracy'):.3f}")
        else:
            print(f" - {r['symbol']}: trained False error={r.get('error')}")

if __name__ == "__main__":
    main()
