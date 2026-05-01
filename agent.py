"""
agent.py
────────
The live agent loop. Wires together:
  data/polymarket.py     → fetch markets
  models/ensemble.py     → run LLM ensemble per market
  models/conformal.py    → wrap in calibrated intervals
  models/mispricing.py   → score and rank opportunities

Run once to build the calibration set, then continuously poll live markets.

Usage:
    python agent.py --calibrate          # fit calibration set from resolved markets
    python agent.py --run                # start live monitoring loop
    python agent.py --calibrate --run    # do both sequentially
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()   # reads .env into os.environ before any API client is created

from data.polymarket   import fetch_live_crypto_markets, fetch_resolved_crypto_markets, Market
from models.ensemble   import run_ensemble, EnsembleResult
from models.conformal  import (
    fit_calibration, predict_interval, CalibrationSet,
    RollingCalibrator, coverage_report
)
from models.mispricing import rank_opportunities, kelly_fraction


CALIB_PATH    = "calibration.json"
RESULTS_PATH  = "results.jsonl"
POLL_INTERVAL = 300          # seconds between live sweeps (5 min)
ALPHA         = 0.20         # 80% coverage — tighter intervals than 90%, more actionable signals
TOP_N         = 10           # how many markets to surface per sweep


# ── Phase 0: Build calibration set ────────────────────────────────────────────

RECENT_MARKETS_FILE = "recent_resolved_markets.json"


def _load_recent_markets() -> list:
    """Load pre-fetched markets from fetch_recent.py cache."""
    data = json.loads(Path(RECENT_MARKETS_FILE).read_text())
    markets = []
    for m in data:
        markets.append(Market(
            id=m["id"],
            question=m["question"],
            category=m.get("category", ""),
            end_date=m.get("closed_date", ""),
            market_prob=m["market_prob"],
            volume_usd=m.get("volume_usd", 0),
            resolved=True,
            outcome=m["outcome"],
        ))
    return markets


def build_calibration(synthetic: bool = False, months_recent: int = 6, use_recent: bool = False):
    """
    Build calibration set from resolved markets, or synthetically.

    --synthetic:   skips API calls, uses simulated data
    --use-recent:  loads from recent_resolved_markets.json (run fetch_recent.py first)
    """
    print("=" * 60)
    print("PHASE 0: Building calibration set")
    print("=" * 60)

    if synthetic:
        return _build_synthetic_calibration()

    if use_recent:
        if not Path(RECENT_MARKETS_FILE).exists():
            print(f"  {RECENT_MARKETS_FILE} not found. Run: python fetch_recent.py")
            return None
        print(f"Loading pre-fetched markets from {RECENT_MARKETS_FILE}...")
        resolved = _load_recent_markets()
        print(f"  Loaded {len(resolved)} markets")
    else:
        print("Fetching resolved markets (all categories for calibration)...")
        resolved = fetch_resolved_crypto_markets(limit=500, months_recent=months_recent)
    print(f"  Found {len(resolved)} resolved crypto markets")

    # 70/30 split: calibration vs. held-out test set
    split_idx = int(len(resolved) * 0.70)
    calib_markets = resolved[:split_idx]
    test_markets  = resolved[split_idx:]

    print(f"  Calibration: {len(calib_markets)}  |  Test: {len(test_markets)}")
    print()

    # Run ensemble on calibration markets
    print("Running LLM ensemble on calibration set (this will take a while)...")
    calib_p_hats = []
    calib_outcomes = []

    for i, market in enumerate(calib_markets):
        print(f"  [{i+1}/{len(calib_markets)}] {market.question[:65]}...")
        try:
            result = run_ensemble(market.question)
            calib_p_hats.append(result.point_estimate)
            calib_outcomes.append(market.outcome)
        except Exception as e:
            print(f"    Skipped: {e}")

    # Fit the conformal calibration
    print(f"\nFitting calibration on {len(calib_p_hats)} samples...")
    calib = fit_calibration(calib_p_hats, calib_outcomes, alphas=[0.05, 0.10, 0.20])
    calib.save(CALIB_PATH)
    print(f"  Saved to {CALIB_PATH}")
    print(f"  Quantiles: {calib.quantiles}")

    # Validate on test set
    print("\nValidating coverage on held-out test set...")
    test_p_hats  = []
    test_outcomes = []

    for market in test_markets:
        try:
            result = run_ensemble(market.question)
            test_p_hats.append(result.point_estimate)
            test_outcomes.append(market.outcome)
        except Exception:
            continue

    test_intervals = [predict_interval(p, calib, ALPHA) for p in test_p_hats]
    report = coverage_report(test_intervals, test_outcomes)

    print(f"\n  Coverage report:")
    print(f"    Target:   {report['target_coverage']:.0%}")
    print(f"    Empirical:{report['empirical_coverage']:.0%}  {'✓' if report['calibration_ok'] else '✗ WARNING'}")
    print(f"    Mean CI width: {report['mean_width']:.3f}")

    if not report['calibration_ok']:
        print("\n  ⚠ Coverage is below target. Possible causes:")
        print("    - Calibration set too small (need ≥100 markets)")
        print("    - Distribution shift (calib markets too old)")
        print("    - LLM ensemble is systematically biased")

    return calib


# ── Phase 1: Live monitoring sweep ────────────────────────────────────────────

def _build_synthetic_calibration() -> CalibrationSet:
    """
    Build a calibration set from synthetic data when the API can't provide
    enough resolved markets. Uses plausible LLM forecast error distributions.

    Coverage guarantees are approximate rather than exact, but this lets
    you run the full pipeline while debugging the Polymarket API.
    """
    import numpy as np
    print("  Building synthetic calibration set (500 simulated markets)...")
    np.random.seed(42)
    n = 500

    # Simulate: true probs drawn from Beta(2,2), outcomes from Bernoulli(p)
    # LLM estimates add Gaussian noise + slight overconfidence bias
    true_probs = np.random.beta(2, 2, n)
    outcomes   = np.random.binomial(1, true_probs)
    noise      = np.random.normal(0, 0.08, n)
    p_hats     = np.clip(true_probs + noise, 0.02, 0.98).tolist()

    calib = fit_calibration(p_hats, outcomes.tolist(), alphas=[0.05, 0.10, 0.20])
    calib.save(CALIB_PATH)
    print(f"  Synthetic calibration saved → {CALIB_PATH}")
    print(f"  q_hat @ α=0.10: {calib.quantiles[0.10]:.4f}")
    print()
    print("  NOTE: Replace with real calibration when API markets are available.")
    print("        Run: python agent.py --calibrate  (without --synthetic)")
    return calib


# ── Phase 1: Live monitoring sweep ────────────────────────────────────────────

def run_sweep(calib: CalibrationSet):
    """
    Single monitoring sweep:
    1. Fetch live markets
    2. Run ensemble on each
    3. Apply conformal wrapper
    4. Score mispricing
    5. Surface top opportunities
    """
    ts = datetime.utcnow().isoformat()
    print(f"\n{'='*60}")
    print(f"SWEEP @ {ts}")
    print(f"{'='*60}")

    print("Fetching live markets...")
    markets = fetch_live_crypto_markets(limit=200)
    print(f"  Found {len(markets)} live crypto markets")

    ensemble_results = []   # list of EnsembleResult
    valid_markets = []

    for i, market in enumerate(markets):
        print(f"  [{i+1}/{len(markets)}] {market.question[:65]}...")
        try:
            result = run_ensemble(market.question)
            ensemble_results.append(result)
            valid_markets.append(market)
            print(f"    p̂={result.point_estimate:.3f}  σ={result.std_dev:.3f}  [{result.uncertainty_level}]")
        except Exception as e:
            print(f"    Skipped: {e}")

    # Apply conformal intervals
    p_hats    = [r.point_estimate for r in ensemble_results]
    intervals = [predict_interval(p, calib, ALPHA) for p in p_hats]

    # Score and rank
    print("\nScoring mispricing opportunities...")
    opportunities = rank_opportunities(valid_markets, intervals, top_n=TOP_N)

    # Display top opportunities
    print(f"\nTOP {TOP_N} OPPORTUNITIES (market overconfident vs our model):")
    for opp in opportunities:
        if opp.action == "PASS":
            continue
        print(opp)

        # Compute Kelly position size for top signals
        if opp.action == "INVESTIGATE":
            frac, direction = kelly_fraction(
                opp.interval.point_estimate,
                opp.interval,
                market_prob=opp.market.market_prob,
            )
            if frac > 0:
                print(f"  Kelly: {frac:.3f} ({frac*100:.1f}% of bankroll) → bet {direction}")

    # Persist results
    _save_results(ts, valid_markets, ensemble_results, intervals, opportunities)

    print(f"\nSweep complete. Next sweep in {POLL_INTERVAL}s.")
    return opportunities


# ── Persistence ────────────────────────────────────────────────────────────────

def _save_results(ts, markets, ensemble_results, intervals, opportunities):
    """Append sweep results to JSONL for later analysis / backtesting."""
    record = {
        "timestamp": ts,
        "markets": [
            {
                "id":              m.id,
                "question":        m.question,
                "market_prob":     m.market_prob,
                "volume_usd":      m.volume_usd,
                "p_hat":           er.point_estimate,
                "ensemble_std":    er.std_dev,
                "ensemble_n":      er.n_samples,
                "ci_lower":        iv.lower,
                "ci_upper":        iv.upper,
                "ci_width":        iv.width,
                "mismatch_score":  opp.mismatch_score if opp else None,
                "action":          opp.action if opp else None,
            }
            for m, er, iv, opp in zip(
                markets, ensemble_results, intervals,
                _match_opportunities(markets, opportunities)
            )
        ]
    }
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _match_opportunities(markets, opportunities):
    """Map markets back to their opportunity signal (or None)."""
    opp_by_id = {o.market.id: o for o in opportunities}
    return [opp_by_id.get(m.id) for m in markets]


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Epistemic Uncertainty Prediction Market Agent")
    parser.add_argument("--calibrate",   action="store_true", help="Build calibration set from resolved markets")
    parser.add_argument("--synthetic",   action="store_true", help="Use synthetic calibration (no API needed)")
    parser.add_argument("--use-recent",  action="store_true", help="Load markets from recent_resolved_markets.json (run fetch_recent.py first)")
    parser.add_argument("--months",      type=int, default=6,  help="Only use markets closed in last N months (default: 6)")
    parser.add_argument("--run",         action="store_true",  help="Start live monitoring loop")
    parser.add_argument("--once",        action="store_true",  help="Run a single sweep (no loop)")
    args = parser.parse_args()

    calib = None

    if args.calibrate:
        calib = build_calibration(
            synthetic=args.synthetic,
            months_recent=args.months,
            use_recent=args.use_recent,
        )

    if args.run or args.once:
        if calib is None:
            if not Path(CALIB_PATH).exists():
                print(f"No calibration file found at {CALIB_PATH}. Run with --calibrate first.")
                return
            print(f"Loading calibration from {CALIB_PATH}...")
            calib = CalibrationSet.load(CALIB_PATH)
            print(f"  n={calib.n_calibration}  quantiles={calib.quantiles}")

        if args.once:
            run_sweep(calib)
        else:
            print(f"\nStarting live loop (poll every {POLL_INTERVAL}s). Ctrl+C to stop.")
            while True:
                try:
                    run_sweep(calib)
                    time.sleep(POLL_INTERVAL)
                except KeyboardInterrupt:
                    print("\nStopped.")
                    break
                except Exception as e:
                    print(f"\nSweep failed: {e}. Retrying in 60s...")
                    time.sleep(60)


if __name__ == "__main__":
    main()
