#!/usr/bin/env python3
# telemetry_cleanup.py
# -----------------------------------------------------
# Clean up old telemetry logs, keeping last 3 per symbol.
# -----------------------------------------------------

import os
import json
from datetime import datetime

LOG_DIR = "telemetry_logs"
KEEP = 3

def cleanup():
    if not os.path.exists(LOG_DIR):
        print("[INFO] telemetry_logs folder missing, skipping cleanup.")
        return

    files_by_symbol = {}
    for f in sorted(os.listdir(LOG_DIR)):
        if not f.endswith(".json"):
            continue
        parts = f.split("_")
        if len(parts) < 2:
            continue
        symbol = parts[0]
        files_by_symbol.setdefault(symbol, []).append(f)

    for sym, files in files_by_symbol.items():
        if len(files) <= KEEP:
            continue
        to_delete = sorted(files)[:-KEEP]
        for f in to_delete:
            try:
                os.remove(os.path.join(LOG_DIR, f))
                print(f"[CLEAN] Removed old telemetry: {f}")
            except Exception as e:
                print(f"[WARN] Could not delete {f}: {e}")

    print(f"[DONE] Telemetry cleanup complete. Kept {KEEP} files per symbol.")

if __name__ == "__main__":
    cleanup()
