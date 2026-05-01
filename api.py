"""
api.py
──────
FastAPI backend for the Uncertainty Agent dashboard.
Reads results.jsonl, calibration.json, and prospective_validation.json
and serves them as a JSON API.

Run locally:
    uvicorn api:app --reload --port 8000

Deploy to Railway/Render:
    Set start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Uncertainty Agent API", version="1.0.0")

# Allow GitHub Pages + localhost to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your GitHub Pages URL in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── File paths ────────────────────────────────────────────────────────────────
RESULTS_FILE     = os.environ.get("RESULTS_FILE",     "results.jsonl")
CALIB_FILE       = os.environ.get("CALIB_FILE",       "calibration.json")
VALIDATION_FILE  = os.environ.get("VALIDATION_FILE",  "prospective_validation.json")
SNAPSHOT_FILE    = os.environ.get("SNAPSHOT_FILE",    "live_predictions_snapshot.json")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_latest_sweep() -> Optional[dict]:
    """Return the most recent sweep from results.jsonl."""
    if not Path(RESULTS_FILE).exists():
        return None
    last = None
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except Exception:
                    pass
    return last


def load_all_sweeps() -> list:
    """Return all sweeps from results.jsonl."""
    if not Path(RESULTS_FILE).exists():
        return []
    sweeps = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sweeps.append(json.loads(line))
                except Exception:
                    pass
    return sweeps


def compute_signal_consistency(sweeps: list) -> list:
    """
    For each market, compute how many times it appeared and avg mismatch.
    Returns sorted list of consistent signals.
    """
    counts    = defaultdict(int)
    mismatches = defaultdict(list)
    questions  = {}
    actions    = {}
    p_hats     = defaultdict(list)
    mkt_probs  = defaultdict(list)

    for sweep in sweeps:
        for m in sweep.get("markets", []):
            mid = str(m.get("id") or m.get("market_id", ""))
            if not mid or m.get("p_hat") is None:
                continue
            counts[mid] += 1
            questions[mid] = m.get("question", "")
            if m.get("mismatch_score") is not None:
                mismatches[mid].append(m["mismatch_score"])
            if m.get("action"):
                actions[mid] = m["action"]
            p_hats[mid].append(m["p_hat"])
            if m.get("market_prob"):
                mkt_probs[mid].append(m["market_prob"])

    signals = []
    for mid, count in counts.items():
        avg_mis  = sum(mismatches[mid]) / len(mismatches[mid]) if mismatches[mid] else 0
        avg_phat = sum(p_hats[mid]) / len(p_hats[mid]) if p_hats[mid] else None
        avg_mkt  = sum(mkt_probs[mid]) / len(mkt_probs[mid]) if mkt_probs[mid] else None
        signals.append({
            "id":           mid,
            "question":     questions[mid],
            "n_sweeps":     count,
            "avg_mismatch": round(avg_mis, 4),
            "avg_p_hat":    round(avg_phat, 4) if avg_phat else None,
            "avg_market_prob": round(avg_mkt, 4) if avg_mkt else None,
            "action":       actions.get(mid, "PASS"),
        })

    signals.sort(key=lambda x: x["avg_mismatch"], reverse=True)
    return signals


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Uncertainty Agent API", "version": "1.0.0"}


@app.get("/api/sweep/latest")
def get_latest_sweep():
    """Most recent sweep — what the agent saw last."""
    sweep = load_latest_sweep()
    if not sweep:
        return {"error": "No sweep data found. Is the agent running?"}

    markets = sweep.get("markets", [])
    # Sort by mismatch score descending
    markets_sorted = sorted(
        markets,
        key=lambda m: m.get("mismatch_score") or 0,
        reverse=True
    )

    return {
        "timestamp":    sweep.get("timestamp"),
        "n_markets":    len(markets),
        "markets":      markets_sorted,
        "n_investigate": sum(1 for m in markets if m.get("action") == "INVESTIGATE"),
        "n_monitor":     sum(1 for m in markets if m.get("action") == "MONITOR"),
    }


@app.get("/api/signals")
def get_signal_consistency():
    """Signal consistency across all sweeps — which markets are persistently flagged."""
    sweeps = load_all_sweeps()
    if not sweeps:
        return {"error": "No data", "signals": []}

    signals = compute_signal_consistency(sweeps)
    return {
        "n_sweeps":       len(sweeps),
        "n_unique_markets": len(signals),
        "signals":        signals[:20],  # top 20
        "generated_at":   datetime.utcnow().isoformat(),
    }


@app.get("/api/calibration")
def get_calibration():
    """Conformal calibration parameters."""
    if not Path(CALIB_FILE).exists():
        return {"error": "calibration.json not found. Run: python agent.py --calibrate"}
    data = json.loads(Path(CALIB_FILE).read_text())
    return {
        "n_calibration": data.get("n_calibration"),
        "quantiles":     data.get("quantiles"),
        "mean_score":    data.get("mean_score"),
        "score_std":     data.get("score_std"),
    }


@app.get("/api/validation")
def get_validation():
    """Prospective validation results."""
    if not Path(VALIDATION_FILE).exists():
        return {"markets": [], "summary": {"n_correct": 0, "n_wrong": 0, "n_pending": 0}}

    data = json.loads(Path(VALIDATION_FILE).read_text())
    markets = data.get("markets", [])

    n_correct = sum(1 for m in markets if m.get("agent_correct") is True)
    n_wrong   = sum(1 for m in markets if m.get("agent_correct") is False)
    n_pending = sum(1 for m in markets if m.get("agent_correct") is None)

    return {
        "markets":  markets,
        "summary": {
            "n_correct":  n_correct,
            "n_wrong":    n_wrong,
            "n_pending":  n_pending,
            "hit_rate":   n_correct / (n_correct + n_wrong) if (n_correct + n_wrong) > 0 else None,
        }
    }


@app.get("/api/stats")
def get_stats():
    """Aggregate stats for the header strip."""
    sweeps  = load_all_sweeps()
    latest  = sweeps[-1] if sweeps else None
    calib   = json.loads(Path(CALIB_FILE).read_text()) if Path(CALIB_FILE).exists() else {}
    val     = json.loads(Path(VALIDATION_FILE).read_text()) if Path(VALIDATION_FILE).exists() else {}

    val_markets = val.get("markets", [])
    n_correct   = sum(1 for m in val_markets if m.get("agent_correct") is True)
    n_resolved  = sum(1 for m in val_markets if m.get("agent_correct") is not None)

    n_markets   = len(latest.get("markets", [])) if latest else 0
    n_inv       = sum(1 for m in (latest or {}).get("markets", []) if m.get("action") == "INVESTIGATE")

    # Compute fade ROI from validation data
    payoffs = []
    for m in val_markets:
        if m.get("agent_correct") is True and m.get("market_prob_at_prediction"):
            cost   = 1 - m["market_prob_at_prediction"]
            payoffs.append(1.0 - cost)  # won NO bet
        elif m.get("agent_correct") is False and m.get("market_prob_at_prediction"):
            cost = 1 - m["market_prob_at_prediction"]
            payoffs.append(0.0 - cost)  # lost NO bet

    fade_roi = round(sum(payoffs) / len(payoffs), 4) if payoffs else None

    return {
        "n_markets_monitored": n_markets,
        "n_investigate":       n_inv,
        "n_sweeps_total":      len(sweeps),
        "prospective_correct": n_correct,
        "prospective_resolved": n_resolved,
        "hit_rate":            f"{n_correct}/{n_resolved}" if n_resolved > 0 else "0/0",
        "fade_roi":            fade_roi,
        "calib_n":             calib.get("n_calibration"),
        "last_sweep":          latest.get("timestamp") if latest else None,
    }
