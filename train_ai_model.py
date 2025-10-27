#!/usr/bin/env python3
"""
train_ai_model.py (v4 hybrid)
- Trains a per-symbol hybrid model (LogisticRegression + RandomForestClassifier) daily using Binance hourly candles.
- Saves per-symbol models to the models/ directory as crypto_ai_model_v4_<SYMBOL>.pkl
Requires: numpy, pandas, scikit-learn, requests, joblib
"""
import os, json, math, warnings
from datetime import datetime
import requests, joblib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
CONF = json.load(open(os.path.join(HERE, "config.json")))
SYMBOLS = CONF.get("symbols", ["BTCUSDT"])
INTERVAL = CONF.get("interval", "1h")
LIMIT = CONF.get("sample_hours_per_symbol", 24*30+200)
MODEL_DIR = os.path.join(HERE, CONF.get("model_dir","models"))
MODEL_VERSION = CONF.get("model_version","v4_hybrid")
os.makedirs(MODEL_DIR, exist_ok=True)

BINANCE_API = "https://api.binance.com"

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BINANCE_API}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for k in data:
        rows.append({
            "open_time": int(k[0])//1000,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        })
    return pd.DataFrame(rows)

# Indicator helpers
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd_components(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    return macd_line, macd_signal

def atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_features(df):
    df = df.copy()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['rsi_diff'] = df['rsi'] - 50.0
    # slope: linear fit over last 12
    df['slope'] = df['close'].rolling(12).apply(lambda x: linear_slope(x), raw=False)
    df['ema20'] = ema(df['close'], 20)
    df['ema_ratio'] = df['close'] / df['ema20'] - 1.0
    macd_line, macd_signal = macd_components(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['atr'] = atr(df, 14)
    df['atr_ratio'] = df['atr'] / df['close']
    df['body'] = (df['close'] - df['open']).abs()
    df['body_ratio'] = df['body'] / (df['high'] - df['low']).replace(0, pd.NA)
    df['upper_wick'] = df['high'] - df[['close','open']].max(axis=1)
    df['lower_wick'] = df[['close','open']].min(axis=1) - df['low']
    df['upper_wick_ratio'] = df['upper_wick'] / (df['high'] - df['low']).replace(0, pd.NA)
    df['lower_wick_ratio'] = df['lower_wick'] / (df['high'] - df['low']).replace(0, pd.NA)
    df['vol_avg20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg20']
    df['vol_change'] = df['volume'].pct_change().fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def linear_slope(arr):
    y = np.array(arr)
    x = np.arange(len(y))
    xm = x.mean(); ym = y.mean()
    num = ((x - xm)*(y - ym)).sum()
    den = ((x - xm)**2).sum()
    return num/den if den!=0 else 0.0

def label_data(df, up_threshold=0.0):
    df = df.copy()
    df['close_shift'] = df['close'].shift(-1)
    df['return_next'] = (df['close_shift'] - df['close']) / df['close']
    df = df.dropna()
    df['label'] = (df['return_next'] > up_threshold).astype(int)
    return df

def train_for_symbol(symbol):
    print(f"[{datetime.utcnow().isoformat()}] Training for {symbol}")
    df = fetch_klines(symbol)
    if df.empty or len(df) < 200:
        print("Not enough data for", symbol)
        return None
    df = compute_features(df)
    df = label_data(df)
    features = ['rsi','rsi_diff','slope','ema_ratio','macd','macd_signal','atr_ratio','body_ratio','upper_wick_ratio','lower_wick_ratio','vol_ratio','vol_change']
    df = df.dropna(subset=features+['label'])
    X = df[features].fillna(0).values
    y = df['label'].values
    if len(np.unique(y)) < 2 or len(y) < 200:
        print("Not enough class variance or samples for", symbol)
        return None
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    # Train Logistic Regression
    log = LogisticRegression(max_iter=300)
    log.fit(X_train, y_train)
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    # Evaluate
    prob_log = log.predict_proba(X_test)[:,1]
    prob_rf = rf.predict_proba(X_test)[:,1]
    hybrid_prob = 0.5 * prob_log + 0.5 * prob_rf
    preds = (hybrid_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"Trained {symbol} - hybrid_test_acc={acc:.4f} - samples={len(y)}")
    # Save models and metadata in a dict
    model_pack = {
        "version": MODEL_VERSION,
        "features": features,
        "logistic": log,
        "random_forest": rf
    }
    path = os.path.join(MODEL_DIR, f"crypto_ai_model_{MODEL_VERSION}_{symbol}.pkl")
    joblib.dump(model_pack, path)
    return {"symbol":symbol, "accuracy":acc, "path":path}

def main():
    results = []
    for s in SYMBOLS:
        try:
            res = train_for_symbol(s)
            if res:
                results.append(res)
        except Exception as e:
            print("Error training", s, e)
    print("Done training. Models saved to", MODEL_DIR)
    print(results)

if __name__ == '__main__':
    main()
