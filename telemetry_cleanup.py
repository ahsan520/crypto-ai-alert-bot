#!/usr/bin/env python3
import os, json
from datetime import datetime, timedelta

CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def cleanup():
    config = load_config()
    max_age_days = config.get("max_log_age_days", 30)
    log_dir = "telemetry_logs"
    if not os.path.exists(log_dir):
        print("[INFO] No telemetry_logs directory found.")
        return

    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    deleted = 0
    for file in os.listdir(log_dir):
        path = os.path.join(log_dir, file)
        if os.path.isfile(path):
            mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                os.remove(path)
                deleted += 1

    print(f"[CLEANUP] Removed {deleted} old telemetry files (>{max_age_days} days).")

if __name__ == "__main__":
    cleanup()
