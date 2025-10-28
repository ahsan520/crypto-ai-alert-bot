#!/usr/bin/env python3
# telemetry_viewer.py
# -----------------------------------------------------
# View telemetry logs for all tracked crypto symbols.
# Can be run as:
#   python telemetry_viewer.py          → CLI summary
#   streamlit run telemetry_viewer.py   → Web dashboard
# -----------------------------------------------------

import os
import json
import streamlit as st
from datetime import datetime
import pandas as pd

LOG_DIR = "telemetry_logs"

def load_telemetry():
    all_data = []
    for file in sorted(os.listdir(LOG_DIR)):
        if not file.endswith(".json"):
            continue
        path = os.path.join(LOG_DIR, file)
        try:
            data = json.load(open(path))
            if isinstance(data, list):
                for entry in data:
                    entry["_file"] = file
                    all_data.append(entry)
            else:
                data["_file"] = file
                all_data.append(data)
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")
    return all_data

def cli_view():
    print("\n📊 Telemetry Viewer (CLI Mode)")
    data = load_telemetry()
    if not data:
        print("No telemetry data found.")
        return
    df = pd.DataFrame(data)
    print(df[["_file", "symbol", "timestamp", "hybrid_ai_signal", "spike_forecast", "final_action", "confidence"]].tail(10))

def dashboard_view():
    st.title("📈 Crypto AI Hybrid v13 Telemetry Dashboard")
    data = load_telemetry()
    if not data:
        st.warning("No telemetry logs found.")
        return
    df = pd.DataFrame(data)
    symbols = df["symbol"].unique().tolist()
    choice = st.sidebar.selectbox("Select Symbol", symbols)
    sub_df = df[df["symbol"] == choice].sort_values("timestamp", ascending=False)
    st.write(f"### Recent Telemetry for {choice}")
    st.dataframe(sub_df[["timestamp", "hybrid_ai_signal", "spike_forecast", "final_action", "confidence"]])
    st.line_chart(sub_df[["confidence"]], x="timestamp", y="confidence")

if __name__ == "__main__":
    if os.getenv("STREAMLIT_SERVER_RUNNING"):
        dashboard_view()
    else:
        cli_view()
