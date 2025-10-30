#!/usr/bin/env python3
# telemetry_cleanup.py
# -----------------------------------------------------
# 🔍 Purpose:
#   Clean up old telemetry logs and spike summaries.
#   - Delete any file older than 2 hours.
#   - Keep only the last N (default 3) per symbol.
#   - Keep only the last N spike summaries.
# -----------------------------------------------------

import os
from datetime import datetime, timedelta

LOG_DIR = "telemetry_logs"
KEEP = 3
TTL_HOURS = 2  # Applies to all files, including spike summaries


def parse_timestamp(filename: str):
    """Extract timestamp from known file naming patterns."""
    path = os.path.join(LOG_DIR, filename)
    try:
        name = filename.replace(".json", "")
        parts = name.split("_")

        # Case 1: spike_train_summary_20251028_044035.json
        if "spike" in parts and "summary" in parts:
            for i, p in enumerate(parts):
                if p.isdigit() and len(p) == 8 and i + 1 < len(parts):
                    nextp = parts[i + 1]
                    if nextp.isdigit() and len(nextp) == 6:
                        date_str = p + nextp
                        return datetime.strptime(date_str, "%Y%m%d%H%M%S")

        # Case 2: BTCUSDT_2025-10-28_08-30-15.json
        if len(parts) >= 3 and "-" in parts[1]:
            date_str = "_".join(parts[1:3])
            return datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")

    except Exception:
        pass

    # Fallback → file modified time
    return datetime.fromtimestamp(os.path.getmtime(path))


def cleanup():
    if not os.path.exists(LOG_DIR):
        print("[INFO] telemetry_logs folder missing — skipping cleanup.")
        return

    now = datetime.now()
    total_deleted = 0
    spike_files, files_by_symbol = [], {}

    # Classify files
    for f in os.listdir(LOG_DIR):
        if not f.endswith(".json"):
            continue
        if f.startswith("spike_train_summary"):
            spike_files.append(f)
        else:
            sym = f.split("_")[0]
            files_by_symbol.setdefault(sym, []).append(f)

    # Step 1: Delete files older than TTL_HOURS
    for f in os.listdir(LOG_DIR):
        fpath = os.path.join(LOG_DIR, f)
        if not f.endswith(".json"):
            continue
        try:
            ftime = parse_timestamp(f)
            age = now - ftime
            if age > timedelta(hours=TTL_HOURS):
                os.remove(fpath)
                total_deleted += 1
                print(f"[CLEAN] Deleted {f} ({age.total_seconds()/3600:.1f}h old)")
        except Exception as e:
            print(f"[WARN] Could not process {f}: {e}")

    # Step 2: Keep last N per symbol
    for sym, files in files_by_symbol.items():
        files_sorted = sorted(files, key=parse_timestamp)
        if len(files_sorted) > KEEP:
            for f in files_sorted[:-KEEP]:
                try:
                    os.remove(os.path.join(LOG_DIR, f))
                    total_deleted += 1
                    print(f"[CLEAN] Removed old telemetry for {sym}: {f}")
                except Exception as e:
                    print(f"[WARN] Could not delete {f}: {e}")

    # Step 3: Keep only last N spike summaries
    spike_sorted = sorted(spike_files, key=parse_timestamp)
    if len(spike_sorted) > KEEP:
        for f in spike_sorted[:-KEEP]:
            try:
                os.remove(os.path.join(LOG_DIR, f))
                total_deleted += 1
                print(f"[CLEAN] Trimmed spike summary: {f}")
            except Exception as e:
                print(f"[WARN] Could not delete spike summary {f}: {e}")

    print(f"[DONE] Cleanup complete — {total_deleted} files deleted. Kept last {KEEP} per symbol and spike summaries.")


if __name__ == "__main__":
    cleanup()
