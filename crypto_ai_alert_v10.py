#!/usr/bin/env python3
"""
crypto_ai_alert_v11.py
Unified alert runner — AI regression + spike predictor fusion.
Sends unified Telegram alerts (BUY/SELL) and falls back to email via SMTP.
Writes enriched telemetry (keeps last 3 files per symbol: _0,_1,_2).
"""

import os
import json
import joblib
import math
import time
import smtplib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from email.message import EmailMessage

# Local utilities
try:
    from utils.data_fetcher import get_data, CACHE_DIR
except Exception:
    # if utils package not present fall back to local dir expectation
    from data_fetcher import get_data  # noqa
    CACHE_DIR = "data_cache"

# -------------------------------
# Config & env
# -------------------------------
HERE = os.path.dirname(__file__)
CONF_PATH = os.path.join(HERE, "config.json")
if os.path.exists(CONF_PATH):
    CONF = json.load(open(CONF_PATH))
else:
    CONF = {}

SYMBOLS = CONF.get("symbols", ["BTCUSDT", "XRPUSDT", "GALAUSDT"])
MODEL_DIR = CONF.get("model_dir", "models")
SPIKE_DIR = os.path.join(MODEL_DIR, "spike_trend")
TELEMETRY_DIR = CONF.get("telemetry", {}).get("log_dir", "telemetry_logs")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SPIKE_DIR, exist_ok=True)

# thresholds
TH_BUY = CONF.get("alert_probability_thresholds", {}).get("buy", 0.7)
TH_SELL = CONF.get("alert_probability_thresholds", {}).get("sell", 0.3)
SPIKE_CONF_THRESH = float(CONF.get("spike_confidence_threshold", 0.6))
ASSUMED_SPIKE_PCT = float(CONF.get("assumed_spike_pct", 2.5))  # percent used to estimate magnitude from prob

# env for notifications (prefer GitHub secrets, else config.json)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or CONF.get("telegram_token") or os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or CONF.get("telegram_chat_id")

# email fallback configuration via secrets (recommended)
EMAIL_HOST = os.getenv("EMAIL_HOST") or CONF.get("email", {}).get("host")
EMAIL_PORT = int(os.getenv("EMAIL_PORT") or CONF.get("email", {}).get("port", 587) or 587)
EMAIL_USER = os.getenv("EMAIL_USERNAME") or CONF.get("email", {}).get("user")
EMAIL_PASS = os.getenv("EMAIL_PASSWORD") or CONF.get("email", {}).get("pass")
EMAIL_TO = os.getenv("EMAIL_USERNAME") or CONF.get("email", {}).get("to")

MAX_MODEL_AGE_HRS = int(CONF.get("max_model_age_hours", 2))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# -------------------------------
# Helpers
# -------------------------------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def read_cache_csv(symbol):
    path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            return df
        except Exception as e:
            logging.warning(f"Failed read cache CSV {path}: {e}")
    return pd.DataFrame()

def ensure_cache(symbol):
    """Ensure data_cache has a CSV for symbol. If missing or too small, call get_data to fetch."""
    path = os.path.join(CACHE_DIR, f"{symbol}.csv")
    if not os.path.exists(path) or os.path.getsize(path) < 2000:
        logging.info(f"Cache missing/small for {symbol}, fetching via get_data()")
        try:
            df = get_data(symbol)
            # get_data already caches via data_fetcher._cache if successful
            return df
        except Exception as e:
            logging.warning(f"get_data failed for {symbol}: {e}")
            return read_cache_csv(symbol)
    return read_cache_csv(symbol)

def is_model_stale(path, max_age_hours=MAX_MODEL_AGE_HRS):
    if not os.path.exists(path):
        logging.warning(f"Missing model file: {path}")
        return True
    age_sec = time.time() - os.path.getmtime(path)
    return age_sec > max_age_hours * 3600

def load_ai_model(symbol):
    path = os.path.join(MODEL_DIR, f"{symbol}_model.pkl")
    if not os.path.exists(path):
        return None, path
    try:
        m = joblib.load(path)
        return m, path
    except Exception as e:
        logging.warning(f"Failed loading ai model {path}: {e}")
        return None, path

def load_spike_pack(symbol):
    path = os.path.join(SPIKE_DIR, f"{symbol}_spike_pack.pkl")
    if not os.path.exists(path):
        return None, path
    try:
        pack = joblib.load(path)
        return pack, path
    except Exception as e:
        logging.warning(f"Failed loading spike pack {path}: {e}")
        return None, path

def featurize_for_ai(df):
    """Return last row with features [open, high, low, close, volume]"""
    if df is None or df.empty:
        return None
    d = df.sort_values("timestamp").tail(1)
    try:
        X = d[["open", "high", "low", "close", "volume"]].astype(float).values
        return X, float(d["close"].iloc[-1])
    except Exception:
        # try with numeric coercion
        for c in ["open","high","low","close","volume"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna().tail(1)
        if d.empty:
            return None
        X = d[["open", "high", "low", "close", "volume"]].values
        return X, float(d["close"].iloc[-1])

def featurize_for_spike(df, features):
    """Build feature vector for spike model (features list from pack)."""
    if df is None or df.empty:
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)
    # compute required features (return, vol_change, rsi, macd)
    temp = df.copy()
    temp["close"] = pd.to_numeric(temp["close"], errors="coerce")
    temp["open"] = pd.to_numeric(temp["open"], errors="coerce")
    temp["high"] = pd.to_numeric(temp["high"], errors="coerce")
    temp["low"] = pd.to_numeric(temp["low"], errors="coerce")
    temp["volume"] = pd.to_numeric(temp.get("volume", 0), errors="coerce").fillna(0)
    temp["return"] = temp["close"].pct_change().fillna(0)
    temp["vol_change"] = temp["volume"].pct_change().fillna(0)
    temp["rsi"] = temp["close"].diff().ewm(alpha=1/14, adjust=False).mean().fillna(50)
    ema12 = temp["close"].ewm(span=12, adjust=False).mean()
    ema26 = temp["close"].ewm(span=26, adjust=False).mean()
    temp["macd"] = ema12 - ema26
    last = temp.tail(1)
    # ensure order matches features
    X = []
    for f in features:
        if f in last.columns:
            X.append(float(last[f].iloc[0]))
        else:
            X.append(0.0)
    return np.array(X).reshape(1, -1)

# -------------------------------
# Notification (Telegram primary, Email fallback)
# -------------------------------
def send_telegram(message_text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.info("Telegram not configured (token/chat_id missing).")
        return False, "telegram not configured"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message_text, "parse_mode": "Markdown"}
    try:
        import requests
        r = requests.post(url, json=payload, timeout=15)
        ok = r.status_code == 200 and r.json().get("ok", False)
        if not ok:
            logging.warning(f"Telegram send returned {r.status_code} {r.text}")
        return ok, r.text
    except Exception as e:
        logging.warning(f"Telegram send error: {e}")
        return False, str(e)

def send_email(subject, body):
    if not EMAIL_HOST or not EMAIL_USER or not EMAIL_PASS or not EMAIL_TO:
        logging.info("Email not configured fully; skipping email fallback.")
        return False, "email not configured"
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_TO
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=20) as s:
            s.starttls()
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:
        logging.warning(f"Email send failed: {e}")
        return False, str(e)

# -------------------------------
# Telemetry rotation (keep 3)
# -------------------------------
def rotate_and_write_telemetry(symbol, record):
    base = os.path.join(TELEMETRY_DIR, symbol)
    # names: symbol_0.json (most recent), _1, _2
    f0 = f"{base}_0.json"
    f1 = f"{base}_1.json"
    f2 = f"{base}_2.json"
    try:
        # remove f2, move f1->f2, f0->f1
        if os.path.exists(f1):
            if os.path.exists(f2):
                os.remove(f2)
            os.rename(f1, f2)
        if os.path.exists(f0):
            os.rename(f0, f1)
        # write new f0
        with open(f0, "w") as fh:
            json.dump(record, fh, default=str, indent=2)
        logging.info(f"Telemetry saved: {f0}")
    except Exception as e:
        logging.warning(f"Telemetry rotation failed for {symbol}: {e}")

# -------------------------------
# Fusion logic
# -------------------------------
def fuse_decision(ai_action, ai_conf, spike_forecast, spike_prob, spike_pct_est):
    """
    Return final_action: one of strong_buy, buy, strong_sell, sell, hold.
    Simple heuristic:
    - If AI buy & spike_prob strong (>=SPIKE_CONF_THRESH) -> strong_buy
    - If AI buy & spike small -> buy
    - Similar for sell
    - Edge conditions for conflicting signals -> hold
    """
    ai_action = (ai_action or "hold").lower()
    # treat spike forecast strings: likely_spike, likely_dip, none
    spike_dir = "none"
    if spike_forecast is not None:
        sf = str(spike_forecast).lower()
        if "spike" in sf or "up" in sf or "positive" in sf:
            spike_dir = "up"
        elif "dip" in sf or "down" in sf or "negative" in sf:
            spike_dir = "down"

    # decide
    if ai_action == "buy":
        if spike_dir == "up" and spike_prob >= SPIKE_CONF_THRESH:
            return "strong_buy"
        return "buy"
    if ai_action == "sell":
        if spike_dir == "down" and spike_prob >= SPIKE_CONF_THRESH:
            return "strong_sell"
        return "sell"
    # If spike alone (AI neutral), convert strong spike -> buy/sell depending direction
    if ai_action == "hold" or ai_action is None:
        if spike_dir == "up" and spike_prob >= SPIKE_CONF_THRESH:
            return "buy"
        if spike_dir == "down" and spike_prob >= SPIKE_CONF_THRESH:
            return "sell"
    return "hold"

# -------------------------------
# Main loop single-run
# -------------------------------
def run_cycle():
    logging.info(f"Starting alert cycle {utc_now_iso()}")
    send_items = []  # collect messages for symbols that require sending (buy/sell)
    telemetry_records = {}

    # load global config data for interval & spike window
    INTERVAL = CONF.get("interval", "1h")
    # interval in hours
    interval_hours = 1 if "h" in INTERVAL else int(''.join(filter(str.isdigit, INTERVAL))) if INTERVAL else 1
    spike_window = int(CONF.get("spike_window", 3))

    for sym in SYMBOLS:
        try:
            # ensure cache (will fetch if missing)
            df = ensure_cache(sym)
            if df is None or df.empty:
                df = read_cache_csv(sym)
            if df is None or df.empty:
                logging.warning(f"No data available for {sym}; skipping.")
                continue

            # AI model load + predict next price
            ai_model, ai_model_path = load_ai_model(sym)
            ai_action = "hold"
            ai_confidence = 0.0
            ai_pred_price = None
            last_close = None

            ai_feats = featurize_for_ai(df)
            if ai_feats is not None:
                X_ai, last_close = ai_feats
                if ai_model is not None:
                    try:
                        pred = ai_model.predict(X_ai)[0]
                        ai_pred_price = float(pred)
                        # compute relative confidence as (pred-last)/last scaled to 0..1 (clamped)
                        if last_close and last_close > 0:
                            rel = (ai_pred_price - last_close) / last_close
                            # convert rel into a pseudo-confidence in [0,1]
                            ai_confidence = max(0.0, min(1.0, abs(rel) * 5))  # scaling factor - tunable
                            ai_action = "buy" if rel > 0 else "sell" if rel < 0 else "hold"
                    except Exception as e:
                        logging.warning(f"AI prediction failed for {sym}: {e}")

            # Spike predictor
            spike_pack, spike_pack_path = load_spike_pack(sym)
            spike_forecast = None
            spike_prob = 0.0
            spike_pct_est = 0.0
            spike_duration_hours = spike_window * interval_hours

            if spike_pack is not None:
                features = spike_pack.get("features", ["return","vol_change","rsi","macd"])
                X_spike = featurize_for_spike(df, features)
                if X_spike is not None:
                    try:
                        rf_spike = spike_pack.get("spike")
                        if hasattr(rf_spike, "predict_proba"):
                            p = rf_spike.predict_proba(X_spike)[0]
                            # assume positive class is index 1
                            spike_prob = float(p[1]) if len(p) > 1 else float(p[0])
                        else:
                            spike_prob = float(rf_spike.predict(X_spike)[0])
                        # decide forecast label
                        if spike_prob >= SPIKE_CONF_THRESH:
                            spike_forecast = "likely_spike"
                        elif spike_prob <= (1 - SPIKE_CONF_THRESH):
                            spike_forecast = "likely_no_spike"
                        else:
                            spike_forecast = "uncertain"
                        # estimate magnitude (simple heuristic)
                        spike_pct_est = round(spike_prob * ASSUMED_SPIKE_PCT, 2)
                    except Exception as e:
                        logging.warning(f"Spike model predict failed for {sym}: {e}")

            # Fusion
            final_action = fuse_decision(ai_action, ai_confidence, spike_forecast, spike_prob, spike_pct_est)

            # Build telemetry record
            record = {
                "symbol": sym,
                "timestamp_utc": utc_now_iso(),
                "price": None if last_close is None else float(last_close),
                "hybrid_ai_signal": {
                    "action": ai_action,
                    "confidence": round(ai_confidence, 4),
                    "predicted_price": None if ai_pred_price is None else round(ai_pred_price, 6),
                    "model_path": ai_model_path
                },
                "spike_signal": {
                    "forecast": spike_forecast,
                    "probability": round(spike_prob, 4),
                    "expected_change_pct": spike_pct_est,
                    "duration_hours": spike_duration_hours,
                    "model_path": spike_pack_path
                },
                "final_action": final_action,
                "reason": {
                    "summary": "",
                    "details": {}
                },
                "sources": {
                    "ai_model": ai_model_path,
                    "spike_model": spike_pack_path
                }
            }

            # populate reason summary & details (simple explainability)
            reasons = []
            details = {}
            details["ai_rel_diff_pct"] = None
            if last_close and ai_pred_price is not None:
                rel_pct = (ai_pred_price - last_close) / last_close * 100.0
                details["ai_rel_diff_pct"] = round(rel_pct, 3)
                if rel_pct > 0:
                    reasons.append(f"AI expects +{round(rel_pct,2)}%")
                elif rel_pct < 0:
                    reasons.append(f"AI expects {round(rel_pct,2)}%")
            if spike_forecast and spike_prob:
                reasons.append(f"SpikeProb={round(spike_prob,2)} (est {spike_pct_est}%) over {spike_duration_hours}h")
                details["spike_prob"] = round(spike_prob, 4)
                details["spike_est_pct"] = spike_pct_est
            record["reason"]["summary"] = "; ".join(reasons) if reasons else "No strong signals"
            record["reason"]["details"] = details

            telemetry_records[sym] = record
            rotate_and_write_telemetry(sym, record)

            # if final_action is buy or sell (not hold) prepare to send
            if final_action in ("buy", "sell", "strong_buy", "strong_sell"):
                send_items.append(record)

        except Exception as e:
            logging.exception(f"Unhandled error for {sym}: {e}")

    # Prepare unified message (only buy/sell)
    if send_items:
        # Build a compact markdown message
        header = f"📈 Crypto AI Hybrid Alerts — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        parts = []
        for r in send_items:
            sym = r["symbol"]
            action = r["final_action"].upper().replace("_", " ")
            price = r.get("price")
            ai_conf = r["hybrid_ai_signal"]["confidence"]
            sp = r["spike_signal"]["probability"]
            sp_pct = r["spike_signal"]["expected_change_pct"]
            dur = r["spike_signal"]["duration_hours"]
            summary = r["reason"]["summary"]
            parts.append(
                f"*{sym}* → *{action}* @ {price:.6f}\n"
                f"AI Conf: {ai_conf:.2f} | Spike P: {sp:.2f} (est {sp_pct}% in {dur}h)\n"
                f"{summary}\n"
            )
        body = header + "\n".join(parts)
        # send telegram
        ok, resp = send_telegram(body)
        if ok:
            logging.info("Telegram alert sent successfully.")
        else:
            logging.warning("Telegram send failed; attempting email fallback...")
            sub = "Crypto AI Hybrid Alerts"
            ok2, resp2 = send_email(sub, body)
            if ok2:
                logging.info("Email fallback sent.")
            else:
                logging.error(f"Both Telegram and Email fallback failed. telegram:{resp} email:{resp2}")
    else:
        logging.info("No BUY/SELL signals this cycle. Nothing sent.")

    logging.info("Alert cycle complete.")

if __name__ == "__main__":
    run_cycle()
