#!/usr/bin/env python3
"""
crypto_ai_alert_v4.py - simplified alert runner
Loads model packs and sends Telegram alerts only for BUY/SELL.
"""
import os, json, joblib, requests, numpy as np, pandas as pd
from datetime import datetime

HERE = os.path.dirname(__file__)
CONF = json.load(open(os.path.join(HERE, "config.json")))
SYMBOLS = CONF.get("symbols", ["BTCUSDT"])
MODEL_DIR = os.path.join(HERE, CONF.get("model_dir","models"))
MODEL_VERSION = CONF.get("model_version","v4_hybrid")
TH_BUY = CONF.get("alert_probability_thresholds", {}).get("buy", 0.7)
TH_SELL = CONF.get("alert_probability_thresholds", {}).get("sell", 0.3)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") or CONF.get("telegram_token")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or CONF.get("telegram_chat_id")

BINANCE_API = "https://api.binance.com"

def fetch_latest_price_yf(symbol):
    mapping = {"BTCUSDT":"BTC-USD","XRPUSDT":"XRP-USD","GALAUSDT":"GALA-USD"}
    yf_sym = mapping.get(symbol)
    try:
        import yfinance as yf
        df = yf.download(yf_sym, period="1d", interval="1h", progress=False)
        if not df.empty:
            last = df.reset_index().iloc[-1]
            return float(last["Close"])
    except Exception as e:
        print("yfinance price fetch failed", e)
    return None

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

def build_features_for_alert(df):
    # minimal features matching training
    df = df.copy()
    df['rsi'] = df['close'].diff().fillna(0)  # placeholder, trainer provides real features
    return df

def send_telegram_message(text):
    token = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    if not token or "PASTE" in str(token):
        raise RuntimeError("Telegram token not configured")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload, timeout=20)
    return r.ok, r.text

def process_symbol(symbol):
    model_pack = load_model(symbol)
    if model_pack is None:
        print("Model missing for", symbol); return
    # fetch latest close price via yfinance
    price = fetch_latest_price_yf(symbol)
    # Build dummy features - in production compute exact same features as trainer
    X = np.array([[0.5]*len(model_pack.get("features"))])
    prob = float(ensemble_prob(model_pack, X)[0])
    signal = "HOLD"
    if prob >= TH_BUY:
        signal = "BUY"
    elif prob <= TH_SELL:
        signal = "SELL"
    if signal in ["BUY","SELL"]:
        message = f"🔔 {signal} Signal for {symbol}\nPrice: {price}\nProbability: {prob:.2f}\nTime: {datetime.utcnow().isoformat()} UTC"
        ok, resp = send_telegram_message(message)
        print(f"Telegram sent? {ok} resp={resp}")
    else:
        print(f"{symbol}: Hold - no alert")

def main():
    for s in SYMBOLS:
        try:
            process_symbol(s)
        except Exception as e:
            print("Error:", e)

if __name__ == '__main__':
    main()
