#!/usr/bin/env python3
# telemetry_cleanup.py
# -----------------------------------------------------
# 🔍 Purpose:
#   Clean up old telemetry logs, keeping the latest N per symbol.
#   Also trims spike_train_summar* logs to the latest N.
#   Ensures telemetry_logs directory doesn't grow indefinitely.
# -----------------------------------------------------

import os
from datetime import datetime

LOG_DIR = "telemetry_logs"
KEEP = 3  # Keep last 3 logs per symbol and spike summaries


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
    spike_files = []

    # 🧭 Classify files
    for f in os.listdir(LOG_DIR):
        if not f.endswith(".json"):
            continue

        if f.startswith("spike_train_summar"):
            spike_files.append(f)
            continue

        parts = f.split("_")
        if len(parts) < 2:
            continue
        symbol = parts[0]
        files_by_symbol.setdefault(symbol, []).append(f)

    total_deleted = 0

    # 🧹 Clean symbol-specific telemetry
    for sym, files in files_by_symbol.items():
        if len(files) <= KEEP:
            continue
        files_sorted = sorted(files, key=parse_timestamp)
        to_delete = files_sorted[:-KEEP]
        for f in to_delete:
            try:
                os.remove(os.path.join(LOG_DIR, f))
                total_deleted += 1
                print(f"[CLEAN] Removed old telemetry: {f}")
            except Exception as e:
                print(f"[WARN] Could not delete {f}: {e}")

    # 🧹 Clean spike_train_summar files
    if len(spike_files) > KEEP:
        spike_sorted = sorted(spike_files, key=parse_timestamp)
        to_delete = spike_sorted[:-KEEP]
        for f in to_delete:
            try:
                os.remove(os.path.join(LOG_DIR, f))
                total_deleted += 1
                print(f"[CLEAN] Removed old spike summary: {f}")
            except Exception as e:
                print(f"[WARN] Could not delete spike summary {f}: {e}")

    print(f"[DONE] Telemetry cleanup complete — kept last {KEEP} per symbol & spike summaries, removed {total_deleted} old logs.")


if __name__ == "__main__":
    cleanup()
