#!/usr/bin/env python3
"""
crypto_ai_alert_v4.py
- Loads per-symbol hybrid models and produces BUY/SELL/HOLD alerts using ensemble probability.
- Requires: requests, numpy, pandas, joblib
- Config via config.json (or environment variables for telegram/email secrets).
"""
import os, json, math, sys, time
from datetime import datetime
import requests, joblib
import numpy as np, pandas as pd

HERE = os.path.dirname(__file__)
CONF = json.load(open(os.path.join(HERE, "config.json")))
SYMBOLS = CONF.get("symbols", ["BTCUSDT"])
INTERVAL = CONF.get("interval", "1h")
MODEL_DIR = os.path.join(HERE, CONF.get("model_dir","models"))
MODEL_VERSION = CONF.get("model_version","v4_hybrid")
TH_BUY = CONF.get("alert_probability_thresholds", {}).get("buy", 0.7)
TH_SELL = CONF.get("alert_probability_thresholds", {}).get("sell", 0.3)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") or CONF.get("telegram_token")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or CONF.get("telegram_chat_id")
EMAIL_FALLBACK = CONF.get("email_alert_fallback", True)
BINANCE_API = "https://api.binance.com"

# indicator functions (same as trainer)
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd_components(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    return macd_line, macd_signal

def atr(df, period=14):
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

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

def build_features(df):
    df = df.copy()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['rsi_diff'] = df['rsi'] - 50.0
    df['slope'] = df['close'].rolling(12).apply(lambda x: linear_slope(x), raw=False)
    df['ema20'] = ema(df['close'], 20)
    df['ema_ratio'] = df['close'] / df['ema20'] - 1.0
    macd_line, macd_signal = macd_components(df['close'])
    df['macd'] = macd_line; df['macd_signal'] = macd_signal
    df['atr'] = atr(df, 14); df['atr_ratio'] = df['atr'] / df['close']
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

def fetch_klines(symbol, interval=INTERVAL, limit=80):
    url = f"{BINANCE_API}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
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

def load_model(symbol):
    path = os.path.join(MODEL_DIR, f"crypto_ai_model_{MODEL_VERSION}_{symbol}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def ensemble_prob(model_pack, X):
    log = model_pack.get("logistic")
    rf = model_pack.get("random_forest")
    p_log = log.predict_proba(X)[:,1] if hasattr(log, "predict_proba") else log.predict(X)
    p_rf = rf.predict_proba(X)[:,1] if hasattr(rf, "predict_proba") else rf.predict(X)
    return 0.5 * p_log + 0.5 * p_rf

def format_alert(symbol, prob, features):
    signal = "HOLD"
    if prob >= TH_BUY:
        signal = "BUY"
    elif prob <= TH_SELL:
        signal = "SELL"
    txt = f"*{symbol}* — {signal} (p_up={prob:.2f})\n"
    txt += "features: " + ", ".join([f\"{k}={v:.4f}\" for k,v in features.items()]) + "\n"
    txt += f"Time: {datetime.utcnow().isoformat()} UTC\n"
    return signal, txt

def send_telegram(text):
    token = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    if not token or "PASTE" in str(token):
        raise RuntimeError("Telegram token not configured")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=20)
    return r.ok, r.text

def send_email(subject, body_text):
    import smtplib, ssl
    if not EMAIL_FALLBACK:
        return False, "disabled"
    server = CONF.get("email_smtp_server")
    port = CONF.get("email_smtp_port")
    user = CONF.get("email_username")
    pwd = CONF.get("email_password")
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(server, port, context=ctx) as s:
            s.login(user, pwd)
            msg = f"Subject: {subject}\n\n{body_text}"
            s.sendmail(user, CONF.get("email_to", user), msg)
        return True, "sent"
    except Exception as e:
        return False, str(e)

def process_symbol(symbol):
    df = fetch_klines(symbol, limit=120)
    if df.empty or len(df) < 60:
        print("Not enough data for", symbol); return
    feats_df = build_features(df)
    last = feats_df.iloc[-1]
    features = {
        "rsi": float(last['rsi']),
        "rsi_diff": float(last['rsi_diff']),
        "slope": float(last['slope']),
        "ema_ratio": float(last['ema_ratio']),
        "macd": float(last['macd']),
        "macd_signal": float(last['macd_signal']),
        "atr_ratio": float(last['atr_ratio']),
        "body_ratio": float(last['body_ratio']),
        "upper_wick_ratio": float(last['upper_wick_ratio']),
        "lower_wick_ratio": float(last['lower_wick_ratio']),
        "vol_ratio": float(last['vol_ratio']),
        "vol_change": float(last['vol_change'])
    }
    model_pack = load_model(symbol)
    if model_pack is None:
        print("Model missing for", symbol); return
    X = np.array([features[k] for k in model_pack.get("features")]).reshape(1,-1)
    prob = float(ensemble_prob(model_pack, X)[0])
    signal, text = format_alert(symbol, prob, features)
    try:
        ok, resp = send_telegram(text)
        if ok:
            print(f"Telegram sent for {symbol} -> {signal}")
        else:
            raise Exception("tg_failed:"+str(resp))
    except Exception as e:
        print("Telegram error", e)
        if EMAIL_FALLBACK:
            subj = f"Crypto Alert - {symbol} - {signal}"
            em_ok, em_resp = send_email(subj, text)
            if em_ok:
                print("Email fallback sent for", symbol)
            else:
                print("Email fallback failed", em_resp)

def main():
    for s in SYMBOLS:
        try:
            process_symbol(s)
        except Exception as e:
            print("Error processing", s, e)

if __name__ == '__main__':
    main()
