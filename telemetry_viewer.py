#!/usr/bin/env python3
import os, json, argparse
import pandas as pd
import streamlit as st
from datetime import datetime

LOG_DIR = "telemetry_logs"

def load_telemetry():
    data = []
    for file in os.listdir(LOG_DIR):
        if file.endswith("_0.json"):
            path = os.path.join(LOG_DIR, file)
            try:
                with open(path, "r") as f:
                    entry = json.load(f)
                    data.append(entry)
            except Exception as e:
                print(f"[WARN] Failed to read {file}: {e}")
    return pd.DataFrame(data) if data else pd.DataFrame()

def run_cli():
    df = load_telemetry()
    if df.empty:
        print("No telemetry data found.")
        return
    print("\nLatest Telemetry Summary:\n")
    for _, row in df.iterrows():
        print(f"{row['symbol']:8} | {row['final_action']:10} | conf={row['confidence']:.2f} | {row['timestamp_utc']}")

def run_streamlit():
    st.set_page_config(page_title="Telemetry Dashboard", layout="wide")
    st.title("📊 Crypto AI Telemetry Dashboard")
    st.caption("Live view of latest model signals")

    df = load_telemetry()
    if df.empty:
        st.warning("No telemetry logs found.")
        return

    st.dataframe(df[["symbol", "final_action", "confidence", "timestamp_utc", "spike_forecast"]])
    st.bar_chart(df.set_index("symbol")["confidence"])

    st.markdown("---")
    st.json(df.to_dict(orient="records"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telemetry Viewer (CLI/Streamlit)")
    parser.add_argument("--mode", choices=["cli", "ui"], default="cli")
    args = parser.parse_args()

    if args.mode == "ui":
        run_streamlit()
    else:
        run_cli()
