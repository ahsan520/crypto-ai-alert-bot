#!/usr/bin/env python3
"""
crypto_ai_alert_v10.2.py - Production alert runner (stable)
Fixes:
- Robust Yahoo Finance retry with threads=False
- Graceful fallback to cache if Yahoo fails
- Compatible with sklearn >=1.7.2 unpickle
- Silences yfinance FutureWarnings
- Ensures clean telemetry even if data is missing
"""

import os
import json
import time
import traceback
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import requests
import smtplib
import ssl
from email.message import EmailMessage
import warnings

# silence warnings from sklearn/yfinance
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except Exception:
    yf = None

# ---------- Config ----------
HERE = os.path.dirname(__file__)
CFG_PATH = os.path.join(HERE, "config.json")
if not os.path.exists(CFG_PATH):
    raise FileNotFoundError("config.json not found")

cfg = json.load(open(CFG_PATH))
SYMBOLS = cfg.get("symbols", ["BTCUSDT"])
MODEL_DIR = os.path.join(HERE, cfg.get("model_dir", "models"))
SPIKE_DIR = os.path.join(MODEL_DIR, "spike_trend")
CACHE_DIR = os.path.join(HERE, cfg.get("cache_dir", "data_cache"))
TELEMETRY_DIR = os.path.join(HERE, cfg.get("telemetry_viewer", {}).get("log_dir", "telemetry_logs"))
os.makedirs(TELEMETRY_DIR, exist_ok=True)

MODEL_VERSION = cfg.get("model_version", "v10_hybrid")
USE_SPIKE = cfg.get("use_spike_predictor_in_alerts", True)
SPIKE_THRESHOLD = float(cfg.get("spike_confidence_threshold", 0.6))
ALERT_THRESH = cfg.get("alert_probability_thresholds", {"buy": 0.7, "sell": 0.3})

FEATURES = [
    'rsi','rsi_diff','slope','ema_ratio','macd','macd_signal',
    'atr_ratio','body_ratio','upper_wick_ratio','lower_wick_ratio',
    'vol_ratio','vol_change'
]

# Secrets
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or cfg.get("telegram_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME") or cfg.get("email_username")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") or cfg.get("email_password")
EMAIL_SMTP = cfg.get("email_smtp_server", "smtp.gmail.com")
EMAIL_PORT = int(cfg.get("email_smtp_port", 465))
EMAIL_FALLBACK = cfg.get("email_alert_fallback", True)

DEFAULT_MAPPING = {"BTCUSDT":"BTC-USD","XRPUSDT":"XRP-USD","GALAUSDT":"GALA-USD"}
MAPPING = cfg.get("symbol_mapping", DEFAULT_MAPPING)

# ---------- Helpers ----------
def now_utc_iso():
    return datetime.utcnow().isoformat() + "Z"

def load_model_pack(symbol):
    path = os.path.join(MODEL_DIR, f"crypto_ai_model_{MODEL_VERSION}_{symbol}.pkl")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Failed to load model pack {path}: {e}")
        return None

def load_spike_pack(symbol):
    path = os.path.join(SPIKE_DIR, f"{symbol}_spike_pack.pkl")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Failed to load spike pack {path}: {e}")
        return None

def fetch_latest_candle(symbol):
    """Fetch latest candle with robust Yahoo retry/fallback logic."""
    yf_sym = MAPPING.get(symbol)
    if yf is not None:
        for attempt, period in enumerate(["2d", "7d"], 1):
            try:
                df = yf.download(
                    yf_sym, period=period, interval="1h",
                    progress=False, auto_adjust=True, threads=False
                )
                if df is None or df.empty:
                    raise ValueError("Yahoo returned no data")

                last = df.tail(1)
                return {
                    "timestamp": last.index[-1].to_pydatetime(),
                    "open": float(last["Open"].iloc[0]),
                    "high": float(last["High"].iloc[0]),
                    "low": float(last["Low"].iloc[0]),
                    "close": float(last["Close"].iloc[0]),
                    "volume": float(last["Volume"].iloc[0]),
                }
            except Exception as e:
                print(f"[WARN] Yahoo fetch (attempt {attempt}) failed for {symbol}: {e}")
                time.sleep(1)

    # fallback to cache if Yahoo fails
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            if not df.empty:
                last = df.iloc[-1]
                return {
                    "timestamp": last["timestamp"],
                    "open": last["open"],
                    "high": last["high"],
                    "low": last["low"],
                    "close": last["close"],
                    "volume": last["volume"],
                }
        except Exception as e:
            print(f"[WARN] Cache read failed for {symbol}: {e}")
    print(f"[WARN] No valid candle data for {symbol}")
    return None

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_features_from_df(df):
    df = df.copy()
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')
    df['rsi'] = df['close'].diff().ewm(alpha=1/14, adjust=False).mean().fillna(50)
    df['rsi_diff'] = df['rsi'] - 50
    df['slope'] = df['close'].rolling(12).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>1 else 0).fillna(0)
    df['ema20'] = ema(df['close'], 20)
    df['ema_ratio'] = (df['close'] / df['ema20']) - 1
    ema_fast, ema_slow = ema(df['close'], 12), ema(df['close'], 26)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = ema(df['macd'], 9)
    df['tr'] = pd.concat([
        df['high']-df['low'],
        (df['high']-df['close'].shift(1)).abs(),
        (df['low']-df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean().fillna(0)
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
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def ensemble_prob_from_pack(model_pack, X_row):
    try:
        scaler = model_pack.get("scaler")
        if scaler:
            Xs = scaler.transform(X_row)
        else:
            Xs = X_row
        log, rf = model_pack.get("logistic"), model_pack.get("random_forest")
        p_log = log.predict_proba(Xs)[:,1] if hasattr(log,"predict_proba") else log.predict(Xs)
        p_rf = rf.predict_proba(Xs)[:,1] if hasattr(rf,"predict_proba") else rf.predict(Xs)
        return float(0.5 * (p_log[0] + p_rf[0]))
    except Exception as e:
        print(f"[WARN] ensemble prob failed: {e}")
        return 0.0

def spike_probs_from_pack(spike_pack, X_row_spike):
    spike_prob, dip_prob = 0.0, 0.0
    try:
        if not spike_pack: return spike_prob, dip_prob
        sp, dp = spike_pack.get("spike"), spike_pack.get("dip")
        if sp is not None:
            spike_prob = float(sp.predict_proba(X_row_spike)[:,1][0])
        if dp is not None:
            dip_prob = float(dp.predict_proba(X_row_spike)[:,1][0])
    except Exception as e:
        print(f"[WARN] spike prob eval failed: {e}")
    return spike_prob, dip_prob

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram credentials missing")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.ok, r.text
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")
        return False, str(e)

def send_email(subject, body, to_addr=None):
    if not EMAIL_FALLBACK:
        return False, "email fallback disabled"
    to_addr = to_addr or EMAIL_USERNAME
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        return False, "email credentials missing"
    try:
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_USERNAME, to_addr
        msg.set_content(body)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_SMTP, EMAIL_PORT, context=context) as s:
            s.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, str(e)

def decide_final_action(ai_prob, buy_thr, sell_thr, ai_signal, spike_forecast, spike_conf):
    final = "hold"
    if ai_signal == "buy":
        if USE_SPIKE and spike_forecast == "Likely spike" and spike_conf >= SPIKE_THRESHOLD:
            return "strong_buy", "ai+spike_confirm"
        elif USE_SPIKE and spike_forecast == "Likely dip" and spike_conf >= SPIKE_THRESHOLD:
            return "hold", "ai_buy_but_spike_dip"
        return "buy", "ai_only"
    if ai_signal == "sell":
        if USE_SPIKE and spike_forecast == "Likely dip" and spike_conf >= SPIKE_THRESHOLD:
            return "strong_sell", "ai+spike_confirm"
        elif USE_SPIKE and spike_forecast == "Likely spike" and spike_conf >= SPIKE_THRESHOLD:
            return "hold", "ai_sell_but_spike_up"
        return "sell", "ai_only"
    if USE_SPIKE and spike_conf >= SPIKE_THRESHOLD:
        return "watch", "spike_only"
    return "hold", "no_signal"

def run_for_symbol(symbol):
    ts = now_utc_iso()
    model_pack = load_model_pack(symbol)
    spike_pack = load_spike_pack(symbol)
    latest = fetch_latest_candle(symbol)
    if latest is None:
        print(f"[WARN] No latest candle for {symbol}")
        return None

    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        except Exception:
            df = None
    else:
        df = None

    if df is None:
        df = pd.DataFrame([latest]*50)
        df['timestamp'] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq='H')

    df_feat = compute_features_from_df(df)
    if df_feat.empty:
        print(f"[WARN] Feature computation failed for {symbol}")
        return None

    last_row = df_feat.iloc[-1]
    X = np.array([last_row[FEATURES].values])
    ai_prob, ai_signal = 0.0, "hold"
    if model_pack:
        ai_prob = ensemble_prob_from_pack(model_pack, X)
        if ai_prob >= ALERT_THRESH.get("buy",0.7):
            ai_signal = "buy"
        elif (1-ai_prob) >= ALERT_THRESH.get("sell",0.3):
            ai_signal = "sell"
    else:
        print(f"[WARN] Model missing for {symbol}")

    X_spike = np.array([[last_row['close']-last_row['open'], last_row['vol_change'], last_row['rsi'], last_row['macd']]])
    spike_prob, dip_prob = spike_probs_from_pack(spike_pack, X_spike) if spike_pack else (0,0)
    spike_forecast, spike_conf = "Stable", max(spike_prob, dip_prob)
    if spike_prob >= SPIKE_THRESHOLD and spike_prob > dip_prob:
        spike_forecast = "Likely spike"
    elif dip_prob >= SPIKE_THRESHOLD and dip_prob > spike_prob:
        spike_forecast = "Likely dip"

    final_action, reason_short = decide_final_action(ai_prob, ALERT_THRESH["buy"], ALERT_THRESH["sell"], ai_signal, spike_forecast, spike_conf)

    telemetry = {
        "symbol": symbol,
        "timestamp_utc": ts,
        "hybrid_ai_signal": ai_signal,
        "hybrid_ai_prob": ai_prob,
        "spike_forecast": spike_forecast,
        "spike_confidence": spike_conf,
        "final_action": final_action,
        "reason_short": reason_short,
        "price": float(last_row['close'])
    }

    fn = os.path.join(TELEMETRY_DIR, f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(fn, "w") as f:
        json.dump(telemetry, f, indent=2)

    if final_action in ("buy","strong_buy","sell","strong_sell"):
        text = f"🔔 {final_action.upper()} {symbol}\nPrice: {telemetry['price']}\nAI={ai_prob:.2f}\nSpike={spike_forecast} ({spike_conf:.2f})\nTime: {ts}"
        ok, resp = send_telegram(text)
        if not ok:
            subj = f"[ALERT] {final_action.upper()} {symbol}"
            send_email(subj, text)
    else:
        print(f"[INFO] {symbol}: No actionable signal")

    return telemetry

def main():
    all_t = []
    for s in SYMBOLS:
        try:
            t = run_for_symbol(s)
            if t: all_t.append(t)
        except Exception as e:
            print(f"[ERROR] {s}: {e}\n{traceback.format_exc()}")
    print(f"[DONE] Run complete at {now_utc_iso()} - {len(all_t)} symbols processed")

if __name__ == "__main__":
    main()
