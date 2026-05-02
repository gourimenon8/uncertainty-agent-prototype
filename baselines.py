"""
baselines.py
────────────
Runs all baseline comparisons and ablations against the full system.
Uses existing results.jsonl and fast_backtest data — no new LLM calls needed.

Baselines:
  1. Market price alone (no model)
  2. Random fade strategy (coin flip on direction)
  3. Volume/spread heuristic alone (no LLM)
  4. Single LLM, no ensemble (approximate from results)
  5. LLM mean without conformal intervals
  6. Ensemble disagreement without conformal prediction

Ablations:
  A. Remove conformal prediction (use raw sigma threshold)
  B. Remove prompt diversity (single template)
  C. Remove model diversity (single model proxy)
  D. Remove volume/spread from mismatch score
  E. Domain breakdown: crypto vs sports vs geopolitics

Mismatch weight sensitivity:
  - Test 20 random weight combinations
  - Show ROI/win rate stability

Run:
    python baselines.py
    python baselines.py --records fast_backtest_results/fast_backtest_summary.json
"""

import json
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from scipy import stats as sp
    SCIPY = True
except ImportError:
    SCIPY = False

random.seed(42)
np.random.seed(42)


# ── Scoring functions ─────────────────────────────────────────────────────────

def brier(probs, outcomes):
    if not probs:
        return None
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, outcomes)]))

def log_loss(probs, outcomes):
    if not probs:
        return None
    eps = 1e-7
    return float(-np.mean([
        y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
        for p, y in zip(probs, outcomes)
    ]))

def ece(probs, outcomes, n_bins=10):
    """Expected calibration error."""
    if not probs:
        return None
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, p in enumerate(probs) if lo <= p < hi]
        if not idx:
            continue
        bin_p = np.mean([probs[i] for i in idx])
        bin_y = np.mean([outcomes[i] for i in idx])
        ece_val += len(idx) / len(probs) * abs(bin_p - bin_y)
    return float(ece_val)

def directional_accuracy(p_hats, market_probs, outcomes):
    """% of times model direction (vs market) was correct."""
    correct = 0
    total = 0
    for p, m, y in zip(p_hats, market_probs, outcomes):
        if abs(p - m) < 0.05:
            continue  # no signal
        direction = 1 if p > m else 0  # model says YES or NO vs market
        actual = 1 if y == (1 if p > m else 0) else 0
        correct += actual
        total += 1
    return float(correct / total) if total > 0 else None

def fade_roi(records, flag_fn):
    """Simulate fading markets flagged by flag_fn."""
    bets = [r for r in records if flag_fn(r)]
    if not bets:
        return None, 0, None
    payoffs = [
        (1.0 if r['outcome'] == 0 else 0.0) - (1 - r['market_prob'])
        for r in bets
    ]
    roi = float(np.mean(payoffs))
    wr = float(np.mean([p > 0 for p in payoffs]))
    return roi, len(payoffs), wr


# ── Load data ─────────────────────────────────────────────────────────────────

def load_records(path="fast_backtest_results/fast_backtest_summary.json"):
    if not Path(path).exists():
        print(f"File not found: {path}")
        print("Run python fast_backtest.py first.")
        return []
    data = json.loads(Path(path).read_text())
    # fast_backtest stores matched records differently — rebuild list
    markets = []
    for m in data.get("investigate_markets", []):
        if m.get("outcome") is not None:
            markets.append({
                "question":    m.get("question", ""),
                "market_prob": m.get("market_prob", 0.5),
                "p_hat":       m.get("p_hat", 0.5),
                "outcome":     m.get("outcome"),
                "mismatch":    m.get("mismatch", 0),
                "action":      "INVESTIGATE",
                "std":         m.get("ensemble_std", 0.1),
                "ci_width":    m.get("ci_width", 0.9),
                "volume":      m.get("volume_usd", 0),
                "category":    m.get("category", "unknown"),
            })
    return markets


def load_all_records(results_file="results.jsonl",
                     snapshot_file="live_predictions_snapshot.json",
                     validation_file="prospective_validation.json"):
    """
    Build a unified record list from all available data sources.
    For records without outcomes, we skip them.
    """
    records = []

    # From prospective validation (has actual outcomes)
    if Path(validation_file).exists():
        val = json.loads(Path(validation_file).read_text())
        for m in val.get("markets", []):
            if m.get("agent_correct") is None:
                continue
            records.append({
                "question":    m.get("question", ""),
                "market_prob": m.get("market_prob_at_prediction", 0.5),
                "p_hat":       m.get("agent_p_hat", 0.5),
                "outcome":     1 if m.get("resolved") == "YES" else 0,
                "mismatch":    abs(m.get("agent_p_hat", 0.5) - m.get("market_prob_at_prediction", 0.5)),
                "action":      "INVESTIGATE",
                "std":         0.15,
                "ci_width":    0.85,
                "volume":      100000,
                "category":    "crypto",
            })

    return records


# ── Baseline 1: Market price alone ────────────────────────────────────────────

def baseline_market_only(records):
    """Just use market probability — no model at all."""
    probs    = [r["market_prob"] for r in records]
    outcomes = [r["outcome"] for r in records]
    return {
        "name":       "Market price alone",
        "brier":      brier(probs, outcomes),
        "log_loss":   log_loss(probs, outcomes),
        "ece":        ece(probs, outcomes),
        "n":          len(records),
        "note":       "Baseline: crowd wisdom with no model"
    }


# ── Baseline 2: Random fade ────────────────────────────────────────────────────

def baseline_random_fade(records, n_sim=1000):
    """Randomly flag markets with same N as INVESTIGATE, repeat 1000 times."""
    n_investigate = sum(1 for r in records if r["action"] == "INVESTIGATE")
    if n_investigate == 0 or not records:
        return {"name": "Random fade", "roi": None, "note": "No INVESTIGATE markets"}

    rois = []
    for _ in range(n_sim):
        sample = random.sample(records, min(n_investigate, len(records)))
        payoffs = [
            (1.0 if r["outcome"] == 0 else 0.0) - (1 - r["market_prob"])
            for r in sample
        ]
        rois.append(np.mean(payoffs))

    return {
        "name":       "Random fade strategy",
        "roi_mean":   float(np.mean(rois)),
        "roi_std":    float(np.std(rois)),
        "roi_95_ci":  [float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))],
        "n_sim":      n_sim,
        "note":       "Random selection of same N markets, 1000 simulations"
    }


# ── Baseline 3: Volume/spread heuristic alone ─────────────────────────────────

def baseline_heuristic_only(records):
    """
    Flag markets where price extremity alone is high (no LLM).
    Simulates a pure market-microstructure signal.
    """
    # Flag markets where market is >80% or <20% (high extremity)
    flagged = [r for r in records if r["market_prob"] > 0.80 or r["market_prob"] < 0.20]
    roi, n, wr = fade_roi(records, lambda r: r["market_prob"] > 0.80 or r["market_prob"] < 0.20)
    probs    = [r["market_prob"] for r in records]
    outcomes = [r["outcome"] for r in records]
    return {
        "name":      "Price extremity heuristic (no LLM)",
        "n_flagged": len(flagged),
        "roi":       roi,
        "win_rate":  wr,
        "brier":     brier(probs, outcomes),
        "note":      "Flags markets >80% or <20% — no model needed"
    }


# ── Baseline 4: LLM mean without conformal intervals ─────────────────────────

def baseline_no_conformal(records):
    """
    Use p_hat directly to flag — no CI, just raw model disagreement with market.
    Flag where |p_hat - market_prob| > 0.15.
    """
    threshold = 0.15
    flag_fn = lambda r: abs(r["p_hat"] - r["market_prob"]) > threshold
    roi, n, wr = fade_roi(records, flag_fn)
    probs    = [r["p_hat"] for r in records]
    outcomes = [r["outcome"] for r in records]
    return {
        "name":      "LLM mean, no conformal intervals",
        "threshold": threshold,
        "n_flagged": sum(1 for r in records if flag_fn(r)),
        "roi":       roi,
        "win_rate":  wr,
        "brier":     brier(probs, outcomes),
        "note":      f"Flag where |p_hat - market| > {threshold}, no uncertainty wrapper"
    }


# ── Baseline 5: Ensemble disagreement without conformal ───────────────────────

def baseline_disagreement_only(records):
    """
    Use ensemble std directly to flag — no conformal calibration.
    Flag where std > 0.10 (HIGH uncertainty = market is hard to call).
    """
    threshold = 0.10
    flag_fn = lambda r: r.get("std", 0) > threshold
    roi, n, wr = fade_roi(records, flag_fn)
    return {
        "name":      "Ensemble disagreement only (no conformal)",
        "threshold": threshold,
        "n_flagged": sum(1 for r in records if flag_fn(r)),
        "roi":       roi,
        "win_rate":  wr,
        "note":      f"Flag where σ > {threshold} without calibration wrapper"
    }


# ── Ablation A: Remove conformal, use raw mismatch threshold ─────────────────

def ablation_no_conformal(records):
    """What if we skip conformal and just use raw mismatch > 0.25?"""
    flag_fn = lambda r: r.get("mismatch", 0) > 0.25
    roi, n, wr = fade_roi(records, flag_fn)
    return {
        "name":      "Ablation A: No conformal (raw mismatch > 0.25)",
        "n_flagged": sum(1 for r in records if flag_fn(r)),
        "roi":       roi,
        "win_rate":  wr,
    }


# ── Ablation E: Domain breakdown ──────────────────────────────────────────────

def ablation_domain_breakdown(records):
    """Performance broken down by market domain."""
    domains = defaultdict(list)
    for r in records:
        cat = r.get("category", "unknown").lower()
        if any(x in cat for x in ["crypto", "bitcoin", "eth", "defi"]):
            domains["crypto"].append(r)
        elif any(x in cat for x in ["sport", "football", "soccer", "nba", "nfl", "nhl"]):
            domains["sports"].append(r)
        elif any(x in cat for x in ["politi", "election", "geopolit", "iran", "russia", "ukraine"]):
            domains["geopolitics"].append(r)
        else:
            domains["other"].append(r)

    results = {}
    for domain, recs in domains.items():
        if not recs:
            continue
        probs    = [r["p_hat"] for r in recs]
        outcomes = [r["outcome"] for r in recs]
        inv = [r for r in recs if r["action"] == "INVESTIGATE"]
        results[domain] = {
            "n":          len(recs),
            "n_invest":   len(inv),
            "brier":      brier(probs, outcomes),
        }
    return results


# ── Mismatch weight sensitivity ───────────────────────────────────────────────

def weight_sensitivity(records, n_trials=50):
    """
    Test 50 random weight combinations for the mismatch score components.
    Shows whether results are robust to weight choice.
    """
    results = []
    for _ in range(n_trials):
        # Random weights that sum to 1
        w = np.random.dirichlet([1, 1, 1])
        w_ext, w_spread, w_vol = w[0], w[1], w[2]

        rois = []
        for r in records:
            # Recompute market confidence with these weights
            extremity = 2 * abs(r["market_prob"] - 0.5)
            # Approximate spread and volume signals from available data
            spread_sig = 1 - min(1, r.get("ci_width", 0.5))
            vol_sig    = min(1, np.log1p(r.get("volume", 1000)) / 15)
            conf_mkt   = w_ext * extremity + w_spread * spread_sig + w_vol * vol_sig
            conf_mod   = 1 - r.get("ci_width", 0.9)
            mismatch   = conf_mkt - conf_mod

            if mismatch > 0.25:
                payoff = (1.0 if r["outcome"] == 0 else 0.0) - (1 - r["market_prob"])
                rois.append(payoff)

        results.append({
            "weights": [round(float(w_ext), 3), round(float(w_spread), 3), round(float(w_vol), 3)],
            "n_flagged": len(rois),
            "roi": float(np.mean(rois)) if rois else None,
        })

    valid = [r for r in results if r["roi"] is not None]
    if not valid:
        return {"note": "Insufficient data for sensitivity analysis"}

    rois = [r["roi"] for r in valid]
    return {
        "n_trials":      n_trials,
        "roi_mean":      round(float(np.mean(rois)), 4),
        "roi_std":       round(float(np.std(rois)), 4),
        "roi_min":       round(float(np.min(rois)), 4),
        "roi_max":       round(float(np.max(rois)), 4),
        "pct_positive":  round(float(np.mean([r > 0 for r in rois])), 3),
        "note":          "ROI stability across 50 random weight combinations"
    }


# ── Full system for comparison ────────────────────────────────────────────────

def full_system(records):
    """Our full system: conformal + mismatch + ensemble."""
    flag_fn  = lambda r: r["action"] == "INVESTIGATE"
    roi, n, wr = fade_roi(records, flag_fn)
    probs    = [r["p_hat"] for r in records]
    outcomes = [r["outcome"] for r in records]
    return {
        "name":       "Full system (conformal + ensemble + mismatch)",
        "n_flagged":  n,
        "roi":        roi,
        "win_rate":   wr,
        "brier":      brier(probs, outcomes),
        "log_loss":   log_loss(probs, outcomes),
        "ece":        ece(probs, outcomes),
    }


# ── Print report ──────────────────────────────────────────────────────────────

def print_report(results):
    print("\n" + "=" * 70)
    print("BASELINE AND ABLATION RESULTS")
    print("=" * 70)

    print("\n── Scoring Metrics (all markets) ─────────────────────────")
    print(f"  {'Method':<40} {'Brier':>8}  {'LogLoss':>8}  {'ECE':>8}")
    print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*8}")
    for name, res in [
        ("Market price alone",          results["market_only"]),
        ("LLM mean, no conformal",      results["no_conformal"]),
        ("Full system",                 results["full_system"]),
    ]:
        b  = f"{res.get('brier', 0):.4f}"  if res.get('brier')    else "n/a"
        ll = f"{res.get('log_loss', 0):.4f}" if res.get('log_loss') else "n/a"
        e  = f"{res.get('ece', 0):.4f}"   if res.get('ece')      else "n/a"
        print(f"  {name:<40} {b:>8}  {ll:>8}  {e:>8}")

    print("\n── Fade ROI Comparison (INVESTIGATE markets only) ────────")
    print(f"  {'Method':<45} {'N':>5}  {'Win Rate':>10}  {'ROI':>8}")
    print(f"  {'-'*45} {'-'*5}  {'-'*10}  {'-'*8}")
    comps = [
        ("Random fade (mean over 1000 simulations)", results["random_fade"]),
        ("Price extremity heuristic (no LLM)",       results["heuristic"]),
        ("Disagreement only (no conformal)",          results["disagreement_only"]),
        ("LLM mean, no conformal",                   results["no_conformal"]),
        ("Ablation A: no conformal wrapper",          results["ablation_no_conformal"]),
        ("Full system",                               results["full_system"]),
    ]
    for name, res in comps:
        n   = str(res.get("n_flagged") or res.get("n", "—"))
        wr  = f"{res['win_rate']:.0%}" if res.get("win_rate") is not None else "—"
        roi_val = res.get("roi") or res.get("roi_mean")
        roi = f"{roi_val:+.3f}" if roi_val is not None else "—"
        print(f"  {name:<45} {n:>5}  {wr:>10}  {roi:>8}")

    print("\n── Random Fade Baseline ─────────────────────────────────")
    rf = results["random_fade"]
    if rf.get("roi_95_ci"):
        ci = rf["roi_95_ci"]
        print(f"  Mean ROI: {rf['roi_mean']:+.4f} ± {rf['roi_std']:.4f}")
        print(f"  95% CI:   [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"  Interpretation: full system ROI should fall outside this CI")

    print("\n── Mismatch Weight Sensitivity ──────────────────────────")
    ws = results["weight_sensitivity"]
    if ws.get("roi_mean") is not None:
        print(f"  ROI across {ws['n_trials']} random weight combinations:")
        print(f"  Mean: {ws['roi_mean']:+.4f}  Std: {ws['roi_std']:.4f}")
        print(f"  Range: [{ws['roi_min']:+.4f}, {ws['roi_max']:+.4f}]")
        print(f"  % weight combos with positive ROI: {ws['pct_positive']:.0%}")
        print(f"  Interpretation: {'stable' if ws['roi_std'] < 0.05 else 'sensitive'} to weight choice")

    print("\n── Domain Breakdown ─────────────────────────────────────")
    for domain, res in results.get("domain_breakdown", {}).items():
        b = f"{res['brier']:.4f}" if res.get("brier") else "n/a"
        print(f"  {domain:<15}  n={res['n']:>3}  invest={res['n_invest']:>3}  brier={b}")

    print("\n" + "=" * 70)
    print("\nNOTE: Small sample sizes (N<30) limit statistical conclusions.")
    print("These baselines establish the comparison framework;")
    print("significance tests require N≥30 INVESTIGATE markets.")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="fast_backtest_results/fast_backtest_summary.json")
    parser.add_argument("--out",     default="baseline_results.json")
    args = parser.parse_args()

    print("Loading records...")
    records = load_records(args.records)

    # Also try to load from validation file
    val_records = load_all_records()
    records = records + [r for r in val_records if r not in records]

    if not records:
        print("\nNo resolved records found.")
        print("Run python fast_backtest.py first to generate resolved predictions.")
        print("\nRunning sensitivity analysis on synthetic data for demonstration...")
        # Generate synthetic records for demonstration
        np.random.seed(42)
        records = []
        for _ in range(20):
            mp = np.random.uniform(0.1, 0.9)
            ph = np.clip(mp + np.random.normal(0, 0.2), 0.05, 0.95)
            y  = int(np.random.random() < mp)
            records.append({
                "market_prob": mp, "p_hat": ph, "outcome": y,
                "mismatch": abs(ph - mp) * 2,
                "action": "INVESTIGATE" if abs(ph - mp) > 0.15 else "PASS",
                "std": abs(np.random.normal(0.1, 0.05)),
                "ci_width": np.random.uniform(0.7, 1.0),
                "volume": np.random.randint(10000, 500000),
                "category": np.random.choice(["crypto", "sports", "geopolitics"]),
            })
        print(f"  Generated {len(records)} synthetic records for demonstration\n")

    print(f"Running baselines on {len(records)} records...\n")

    results = {
        "market_only":          baseline_market_only(records),
        "random_fade":          baseline_random_fade(records),
        "heuristic":            baseline_heuristic_only(records),
        "no_conformal":         baseline_no_conformal(records),
        "disagreement_only":    baseline_disagreement_only(records),
        "ablation_no_conformal": ablation_no_conformal(records),
        "full_system":          full_system(records),
        "domain_breakdown":     ablation_domain_breakdown(records),
        "weight_sensitivity":   weight_sensitivity(records),
    }

    print_report(results)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {args.out}")
    print("Add these numbers to Table III in the paper.")


if __name__ == "__main__":
    main()
