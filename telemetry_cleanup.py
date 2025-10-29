#!/usr/bin/env python3
# telemetry_cleanup.py
# -----------------------------------------------------
# 🔍 Purpose:
#   Clean up old telemetry logs, keeping the latest N per symbol.
#   Ensures the telemetry_logs directory doesn't grow indefinitely.
# -----------------------------------------------------

import os
from datetime import datetime

LOG_DIR = "telemetry_logs"
KEEP = 3  # Keep last 3 logs per symbol


def parse_timestamp(filename: str):
    """Try to extract a sortable timestamp from the filename."""
    try:
        # Example format: BTCUSDT_2025-10-28_08-30-15.json
        parts = filename.replace(".json", "").split("_")
        if len(parts) >= 3:
            date_str = "_".join(parts[1:3])
            return datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        pass
    # fallback: use file modified time
    return datetime.fromtimestamp(os.path.getmtime(os.path.join(LOG_DIR, filename)))


def cleanup():
    if not os.path.exists(LOG_DIR):
        print("[INFO] telemetry_logs folder missing, skipping cleanup.")
        return

    files_by_symbol = {}

    for f in os.listdir(LOG_DIR):
        if not f.endswith(".json"):
            continue
        parts = f.split("_")
        if len(parts) < 2:
            continue
        symbol = parts[0]
        files_by_symbol.setdefault(symbol, []).append(f)

    total_deleted = 0
    for sym, files in files_by_symbol.items():
        if len(files) <= KEEP:
            continue

        # Sort by timestamp (newest last)
        files_sorted = sorted(files, key=parse_timestamp)
        to_delete = files_sorted[:-KEEP]

        for f in to_delete:
            try:
                os.remove(os.path.join(LOG_DIR, f))
                total_deleted += 1
                print(f"[CLEAN] Removed old telemetry: {f}")
            except Exception as e:
                print(f"[WARN] Could not delete {f}: {e}")

    print(f"[DONE] Telemetry cleanup complete — kept last {KEEP} per symbol, removed {total_deleted} old logs.")


if __name__ == "__main__":
    cleanup()
