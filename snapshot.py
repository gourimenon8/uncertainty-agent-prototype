"""
snapshot.py
───────────
Reads results.jsonl and maintains two files:

  live_predictions_snapshot.json  — first prediction per market (never overwritten)
  live_predictions_latest.json    — most recent prediction per market

Run manually:    python snapshot.py
Run on a cron:   */30 * * * * cd ~/Downloads/files/uncertainty_agent && source venv/bin/activate && python snapshot.py

The snapshot file is your research dataset. It records what the agent
predicted BEFORE outcomes were known — the core of your prospective validation.
"""

import json
from datetime import datetime
from pathlib import Path

RESULTS_FILE   = "results.jsonl"
SNAPSHOT_FILE  = "live_predictions_snapshot.json"   # first-seen, never overwritten
LATEST_FILE    = "live_predictions_latest.json"     # most recent per market

def load_existing_snapshot():
    if Path(SNAPSHOT_FILE).exists():
        return json.loads(Path(SNAPSHOT_FILE).read_text())
    return {}

def run():
    if not Path(RESULTS_FILE).exists():
        print("No results.jsonl found — is the agent running?")
        return

    snapshot = load_existing_snapshot()   # first-seen predictions (protected)
    latest   = {}                         # will be fully rewritten each run

    total_sweeps  = 0
    total_records = 0

    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sweep = json.loads(line)
            except Exception:
                continue

            total_sweeps += 1
            ts = sweep.get("timestamp", "")

            for m in sweep.get("markets", []):
                mid = m.get("id") or m.get("market_id", "")
                if not mid:
                    continue

                total_records += 1

                record = {
                    "question":     m.get("question", ""),
                    "market_prob":  m.get("market_prob"),
                    "p_hat":        m.get("p_hat"),
                    "ensemble_std": m.get("ensemble_std"),
                    "ci_lower":     m.get("ci_lower"),
                    "ci_upper":     m.get("ci_upper"),
                    "ci_width":     m.get("ci_width"),
                    "mismatch":     m.get("mismatch_score"),
                    "action":       m.get("action"),
                    "timestamp":    ts,
                }

                # Snapshot: only write if this market hasn't been seen before
                if mid not in snapshot:
                    snapshot[mid] = record

                # Latest: always overwrite with most recent
                latest[mid] = record

    # Save snapshot (append-only — never removes existing entries)
    Path(SNAPSHOT_FILE).write_text(json.dumps(snapshot, indent=2))

    # Save latest (fully rewritten)
    Path(LATEST_FILE).write_text(json.dumps(latest, indent=2))

    # Print summary
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC]")
    print(f"  Processed: {total_sweeps} sweeps, {total_records} market records")
    print(f"  Snapshot:  {len(snapshot)} unique markets (first predictions, protected)")
    print(f"  Latest:    {len(latest)} unique markets (most recent predictions)")
    print()

    # Print current top signals from latest
    actionable = [
        (mid, r) for mid, r in latest.items()
        if r.get("action") in ("INVESTIGATE", "MONITOR")
        and r.get("mismatch") is not None
    ]
    actionable.sort(key=lambda x: x[1].get("mismatch", 0), reverse=True)

    if actionable:
        print("  Top signals (from latest sweep):")
        for mid, r in actionable[:8]:
            q       = r.get("question", "")[:50]
            mkt     = r.get("market_prob", 0) or 0
            p_hat   = r.get("p_hat", 0) or 0
            mis     = r.get("mismatch", 0) or 0
            action  = r.get("action", "")
            print(f"    [{action}] {q}")
            print(f"             market={mkt:.1%}  model={p_hat:.1%}  mismatch={mis:+.3f}")

if __name__ == "__main__":
    run()
