#!/usr/bin/env python3
"""
crypto_ai_alert_v10.py - Production alert runner

Responsibilities:
- Load hybrid models (Logistic + RandomForest) saved by train_ai_model.py
- Load spike predictor packs saved by spike_predictor.py
- Fetch latest market data (yfinance primary, cache fallback)
- Build features to match trainer's FEATURES
- Compute hybrid probabilities, spike probs, merge decisions (config-controlled)
- Send Telegram alerts ONLY for Buy/Sell (strong variants included)
- Email fallback if Telegram fails (config.email_alert_fallback)
- Write telemetry JSON to telemetry_logs/
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

# 3rd-party
try:
    import yfinance as yf
except Exception:
    yf = None

# ---------- Config & paths ----------
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
ALERT_THRESH = cfg.get("alert_probability_thresholds", {"buy":0.7,"sell":0.3})

# features must match trainer FEATURES
FEATURES = [
    'rsi','rsi_diff','slope','ema_ratio','macd','macd_signal',
    'atr_ratio','body_ratio','upper_wick_ratio','lower_wick_ratio',
    'vol_ratio','vol_change'
]

# secrets from env (GitHub Actions will inject them)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or cfg.get("telegram_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id")
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME") or cfg.get("email_username")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") or cfg.get("email_password")
EMAIL_SMTP = cfg.get("email_smtp_server", "smtp.gmail.com")
EMAIL_PORT = int(cfg.get("email_smtp_port", 465))
EMAIL_FALLBACK = cfg.get("email_alert_fallback", True)

# symbol mapping to yfinance
DEFAULT_MAPPING = {"BTCUSDT":"BTC-USD","XRPUSDT":"XRP-USD","GALAUSDT":"GALA-USD"}
MAPPING = cfg.get("symbol_mapping", DEFAULT_MAPPING)

# util functions
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
    """
    Fetch latest 1h candle via yfinance. Returns pandas Series with open,high,low,close,volume.
    If yfinance not available or fails, try to load last row from cached CSV.
    """
    yf_sym = MAPPING.get(symbol)
    if yf is None:
        print("[WARN] yfinance not available; fallback to cache")
    try:
        if yf is not None:
            df = yf.download(yf_sym, period="2d", interval="1h", progress=False)
            if df is not None and not df.empty:
                last = df.reset_index().iloc[-1]
                return {
                    "timestamp": pd.to_datetime(last["Datetime"]) if "Datetime" in last else last["Datetime"],
                    "open": float(last["Open"]), "high": float(last["High"]),
                    "low": float(last["Low"]), "close": float(last["Close"]), "volume": float(last["Volume"])
                }
    except Exception as e:
        print(f"[WARN] yfinance latest candle failed for {symbol}: {e}")

    # fallback to cache
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            last = df.iloc[-1]
            return {"timestamp": last["timestamp"], "open": last["open"], "high": last["high"],
                    "low": last["low"], "close": last["close"], "volume": last["volume"]}
        except Exception as e:
            print(f"[WARN] Failed to read cache for latest candle {symbol}: {e}")
    return None

# feature builders (lightweight - consistent with trainer)
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_features_from_df(df):
    """
    Accepts a DataFrame with columns timestamp, open, high, low, close, volume
    Returns df with FEATURES present (may be multiple rows). For alert, we only use last row.
    """
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    # minimal features to mirror trainer behavior
    df['rsi'] = df['close'].diff().ewm(alpha=1/14, adjust=False).mean().fillna(50)  # simplified
    df['rsi_diff'] = df['rsi'] - 50.0
    df['slope'] = df['close'].rolling(12).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>1 else 0.0).fillna(0)
    df['ema20'] = ema(df['close'], 20)
    df['ema_ratio'] = (df['close'] / df['ema20']) - 1.0
    ema_fast = ema(df['close'], 12)
    ema_slow = ema(df['close'], 26)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = ema(df['macd'], 9)
    # ATR simplified
    df['tr'] = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift(1)).abs(), (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
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

# prediction helpers
def ensemble_prob_from_pack(model_pack, X_row):
    """
    model_pack contains scaler, logistic, random_forest
    X_row: 2D array shape (1, n_features)
    returns hybrid_prob (float)
    """
    try:
        scaler = model_pack.get("scaler")
        if scaler is not None:
            Xs = scaler.transform(X_row)
        else:
            Xs = X_row
        log = model_pack.get("logistic")
        rf = model_pack.get("random_forest")
        p_log = log.predict_proba(Xs)[:,1] if hasattr(log, "predict_proba") else log.predict(Xs)
        p_rf = rf.predict_proba(Xs)[:,1] if hasattr(rf, "predict_proba") else rf.predict(Xs)
        hybrid = 0.5 * np.array(p_log) + 0.5 * np.array(p_rf)
        return float(hybrid[0])
    except Exception as e:
        print(f"[WARN] ensemble prob failed: {e}")
        return 0.0

def spike_probs_from_pack(spike_pack, X_row_spike):
    """
    spike_pack expected to have 'spike', 'dip' RandomForest models.
    X_row_spike is 2D array
    returns spike_prob, dip_prob
    """
    spike_prob = 0.0
    dip_prob = 0.0
    try:
        if spike_pack is None:
            return spike_prob, dip_prob
        sp = spike_pack.get("spike")
        dp = spike_pack.get("dip")
        if sp is not None:
            spike_prob = float(sp.predict_proba(X_row_spike)[:,1][0]) if hasattr(sp, "predict_proba") else float(sp.predict(X_row_spike)[0])
        if dp is not None:
            dip_prob = float(dp.predict_proba(X_row_spike)[:,1][0]) if hasattr(dp, "predict_proba") else float(dp.predict(X_row_spike)[0])
    except Exception as e:
        print(f"[WARN] spike prob eval failed: {e}")
    return spike_prob, dip_prob

# send telegram
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram credentials not configured")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.ok, r.text
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")
        return False, str(e)

# email fallback
def send_email(subject, body, to_addr=None):
    if not EMAIL_FALLBACK:
        return False, "email fallback disabled"
    to_addr = to_addr or EMAIL_USERNAME
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        return False, "email credentials missing"
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_USERNAME
        msg["To"] = to_addr
        msg.set_content(body)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_SMTP, EMAIL_PORT, context=context) as server:
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, str(e)

def build_reason_text(ai_prob, spike_prob, dip_prob, ai_signal, spike_forecast):
    # Medium-level reason for console (adaptive)
    parts = []
    parts.append(f"Hybrid AI prob={ai_prob:.2f}")
    if spike_forecast:
        parts.append(f"Spike forecast={spike_forecast}")
    if spike_prob is not None:
        parts.append(f"spike_conf={spike_prob:.2f}")
        parts.append(f"dip_conf={dip_prob:.2f}")
    return " | ".join(parts)

def decide_final_action(ai_prob, buy_thr, sell_thr, ai_signal, spike_forecast, spike_conf):
    """
    Merge hybrid AI and spike predictor:
    - If use_spike_predictor_in_alerts True -> amplify or suppress
    Returns final_action and reason short tag
    """
    final = "hold"
    reason_short = ""
    if ai_signal == "buy":
        if USE_SPIKE:
            if spike_forecast == "Likely spike" and spike_conf >= SPIKE_THRESHOLD:
                final = "strong_buy"; reason_short = "ai+spike_confirm"
            elif spike_forecast == "Likely dip" and spike_conf >= SPIKE_THRESHOLD:
                final = "hold"; reason_short = "ai_buy_but_spike_dip"
            else:
                final = "buy"; reason_short = "ai_only"
        else:
            final = "buy"; reason_short = "ai_only"
    elif ai_signal == "sell":
        if USE_SPIKE:
            if spike_forecast == "Likely dip" and spike_conf >= SPIKE_THRESHOLD:
                final = "strong_sell"; reason_short = "ai+spike_confirm"
            elif spike_forecast == "Likely spike" and spike_conf >= SPIKE_THRESHOLD:
                final = "hold"; reason_short = "ai_sell_but_spike_up"
            else:
                final = "sell"; reason_short = "ai_only"
        else:
            final = "sell"; reason_short = "ai_only"
    else:
        # hold/watch logic if spike signals are very strong consider 'watch' (no alert)
        if USE_SPIKE and (spike_conf >= SPIKE_THRESHOLD):
            final = "watch"; reason_short = "spike_only"
        else:
            final = "hold"; reason_short = "no_signal"
    return final, reason_short

def run_for_symbol(symbol):
    ts = now_utc_iso()
    model_pack = load_model_pack(symbol)
    spike_pack = load_spike_pack(symbol)
    latest = fetch_latest_candle(symbol)
    if latest is None:
        print(f"[WARN] No latest candle for {symbol}; skipping.")
        return None

    # Build a minimal DataFrame using last 48h from cache if available, else replicate using latest only
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    df = None
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        except Exception:
            df = None
    if df is None:
        # create a tiny rolling df with repeated latest candle to allow feature calc (not ideal but safe)
        df = pd.DataFrame([latest]*50)
        df['timestamp'] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq='H')

    # append latest if timestamp newer than last
    try:
        if pd.to_datetime(df['timestamp'].iloc[-1]) < pd.to_datetime(latest['timestamp']):
            df = df.append(latest, ignore_index=True)
    except Exception:
        pass

    df_feat = compute_features_from_df(df)
    if df_feat.empty:
        print(f"[WARN] Feature computation failed for {symbol}")
        return None

    last_row = df_feat.iloc[-1]
    X = np.array([last_row[FEATURES].values])
    # hybrid AI probability
    ai_prob = 0.0
    ai_signal = "hold"
    if model_pack:
        ai_prob = ensemble_prob_from_pack(model_pack, X)
        if ai_prob >= ALERT_THRESH.get("buy", 0.7):
            ai_signal = "buy"
        elif (1.0 - ai_prob) >= ALERT_THRESH.get("sell", 0.7):
            ai_signal = "sell"
        else:
            ai_signal = "hold"
    else:
        print(f"[WARN] Model pack missing for {symbol}")

    # spike features: simple X_spike (can reuse subset)
    X_spike = np.array([[ last_row['close'] - last_row['open'],
                          last_row['vol_change'],
                          last_row['rsi'],
                          last_row['macd'] ]])
    spike_prob, dip_prob = 0.0, 0.0
    spike_forecast = None
    spike_conf = 0.0
    if spike_pack:
        spike_prob, dip_prob = spike_probs_from_pack(spike_pack, X_spike)
        if spike_prob >= SPIKE_THRESHOLD and spike_prob > dip_prob:
            spike_forecast = "Likely spike"
            spike_conf = spike_prob
        elif dip_prob >= SPIKE_THRESHOLD and dip_prob > spike_prob:
            spike_forecast = "Likely dip"
            spike_conf = dip_prob
        else:
            spike_forecast = "Stable"
            spike_conf = max(spike_prob, dip_prob)
    # Merge into final action
    final_action, reason_short = decide_final_action(ai_prob, ALERT_THRESH.get("buy",0.7), ALERT_THRESH.get("sell",0.3), ai_signal, spike_forecast, spike_conf)

    # Build human-friendly messages
    console_reason = build_reason_text(ai_prob, spike_prob, dip_prob, ai_signal, spike_forecast)
    telemetry = {
        "symbol": symbol,
        "timestamp_utc": ts,
        "hybrid_ai_signal": ai_signal,
        "hybrid_ai_prob": ai_prob,
        "spike_forecast": spike_forecast,
        "spike_confidence": spike_conf,
        "final_action": final_action,
        "reason_short": reason_short,
        "reason": {
            "summary": f"Hybrid AI={ai_signal} ({ai_prob:.2f}), spike={spike_forecast} ({spike_conf:.2f})",
            "details": {
                "ai_prob": ai_prob,
                "spike_prob": spike_prob,
                "dip_prob": dip_prob,
                "features": {f: float(last_row[f]) for f in FEATURES if f in last_row.index}
            }
        },
        "price": float(last_row['close'])
    }

    # Write telemetry JSON
    fn = os.path.join(TELEMETRY_DIR, f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(fn, "w") as f:
            json.dump(telemetry, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write telemetry: {e}")

    # Send Telegram only for buy/sell strong variants
    send_alert = final_action in ("buy", "strong_buy", "sell", "strong_sell")
    if send_alert and cfg.get("telemetry", {}).get("enable_telegram_alerts", True):
        short = f"🔔 {final_action.upper()} signal for {symbol}\nPrice: {telemetry['price']}\nProb: {ai_prob:.2f}\nSpike: {spike_forecast} ({spike_conf:.2f})\nReason: {telemetry['reason']['summary']}\nTime: {telemetry['timestamp_utc']}"
        ok, resp = False, None
        try:
            ok, resp = send_telegram(short)
        except Exception as e:
            ok, resp = False, str(e)
        # email fallback
        if not ok and EMAIL_FALLBACK:
            subj = f"[ALERT] {final_action.upper()} {symbol}"
            body = short
            ok2, resp2 = send_email(subj, body)
            print(f"[INFO] Email fallback sent? {ok2} resp={resp2}")
        print(f"[INFO] Telegram sent? {ok} resp={str(resp)[:200]}")
    else:
        print(f"[INFO] No alert to send for {symbol} (final_action={final_action})")

    # Always return telemetry for potential aggregation
    return telemetry

def main():
    all_telemetry = []
    for s in SYMBOLS:
        try:
            t = run_for_symbol(s)
            if t:
                all_telemetry.append(t)
        except Exception as e:
            print(f"[ERROR] Exception in run_for_symbol {s}: {e}\n{traceback.format_exc()}")
    # Optionally: upload or aggregate - GitHub Actions will collect telemetry_logs artifact
    print(f"[DONE] Run complete at {now_utc_iso()} - {len(all_telemetry)} symbols processed")

if __name__ == "__main__":
    main()
