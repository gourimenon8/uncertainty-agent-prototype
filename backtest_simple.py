"""
backtest_simple.py
──────────────────
Clean, fast retroactive backtest. Runs on 30-50 markets max.
Uses Ollama only (no rate limits). Finishes in 20-30 minutes.
Includes all statistical tests and baseline comparisons.

Run:
    python backtest_simple.py
    python backtest_simple.py --markets 50 --out backtest_results/
"""

import argparse, json, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

from data.polymarket   import fetch_resolved_markets
from models.ensemble   import run_ensemble
from models.conformal  import fit_calibration, predict_interval
from models.mispricing import infer_market_confidence, ci_width_to_confidence

try:
    from scipy import stats as sp
    SCIPY = True
except ImportError:
    SCIPY = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def brier(probs, outcomes):
    return float(np.mean([(p-y)**2 for p,y in zip(probs,outcomes)]))

def fade_simulation(records):
    inv = [r for r in records if r["action"] == "INVESTIGATE"]
    if not inv:
        return None, 0, None
    payoffs = [(1.0 if r["outcome"]==0 else 0.0) - (1 - r["market_prob"]) for r in inv]
    return float(np.mean(payoffs)), len(payoffs), float(np.mean([p>0 for p in payoffs]))

def statistical_tests(records, payoffs):
    out = {}
    if not payoffs:
        return out
    n    = len(payoffs)
    wins = sum(1 for p in payoffs if p > 0)

    if SCIPY and n >= 5:
        bt = sp.binomtest(wins, n, p=0.5, alternative="greater")
        out["binomial_p"]   = float(bt.pvalue)
        out["significant"]  = bt.pvalue < 0.05
        out["n_bets"]       = n
        out["win_rate"]     = wins / n

        arr  = np.array(payoffs)
        boot = [np.mean(np.random.choice(arr, n, replace=True)) for _ in range(10000)]
        ci   = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out["roi_95_ci"]          = ci
        out["roi_ci_excl_zero"]   = ci[0] > 0

    inv_err  = [abs(r["market_prob"]-r["outcome"]) for r in records if r["action"]=="INVESTIGATE"]
    pass_err = [abs(r["market_prob"]-r["outcome"]) for r in records if r["action"]=="PASS"]
    if SCIPY and len(inv_err) >= 3 and len(pass_err) >= 3:
        u, p = sp.mannwhitneyu(inv_err, pass_err, alternative="greater")
        out["mannwhitney_p"]    = float(p)
        out["mw_significant"]   = p < 0.05
        out["mean_err_inv"]     = float(np.mean(inv_err))
        out["mean_err_pass"]    = float(np.mean(pass_err))

    return out

def print_report(metrics):
    print("\n" + "="*65)
    print("BACKTEST RESULTS")
    print("="*65)
    print(f"\n  Markets:   {metrics['n_markets']} ({metrics['n_calib']} calib / {metrics['n_test']} test)")
    print(f"  Coverage:  {metrics['coverage']:.0%} (target {metrics['target']:.0%}) {'✓' if metrics['coverage'] >= metrics['target'] else '✗'}")
    print(f"  CI width:  {metrics['mean_ci_width']:.3f}")
    print(f"\n── Brier Scores (lower = better)")
    print(f"  Constant 0.5:  {metrics['brier_constant']:.4f}")
    print(f"  Market crowd:  {metrics['brier_market']:.4f}")
    print(f"  Our ensemble:  {metrics['brier_model']:.4f}")
    print(f"  vs market: {metrics['brier_improvement']:+.4f} {'✓' if metrics['brier_improvement']>0 else '✗'}")
    print(f"\n── Fade Simulation")
    if metrics.get("fade_roi") is not None:
        print(f"  INVESTIGATE bets: {metrics['n_investigate']}")
        print(f"  Win rate:         {metrics['fade_win_rate']:.0%}")
        print(f"  Mean ROI:         {metrics['fade_roi']:+.3f} {'✓' if metrics['fade_roi']>0 else '✗'}")
    else:
        print("  No INVESTIGATE markets in test set")
    print(f"\n── Statistical Tests")
    st = metrics.get("stats", {})
    if "binomial_p" in st:
        sig = "✓ significant" if st["significant"] else "✗ not significant"
        print(f"  Binomial (win rate vs 50%): p={st['binomial_p']:.3f}  {sig}")
        print(f"  Win rate: {st['win_rate']:.0%} on {st['n_bets']} bets")
    if "roi_95_ci" in st:
        ci  = st["roi_95_ci"]
        exc = "✓ excludes zero" if st["roi_ci_excl_zero"] else "✗ includes zero"
        print(f"  Bootstrap 95% CI on ROI: [{ci[0]:+.3f}, {ci[1]:+.3f}]  {exc}")
    if "mannwhitney_p" in st:
        sig = "✓ significant" if st["mw_significant"] else "✗ not significant"
        print(f"  Mann-Whitney INVESTIGATE({st['mean_err_inv']:.3f}) vs PASS({st['mean_err_pass']:.3f}): p={st['mannwhitney_p']:.3f}  {sig}")
    if not st:
        print("  Install scipy: pip install scipy")
    print("="*65)

# ── Main ──────────────────────────────────────────────────────────────────────

def run(n_markets=40, alpha=0.20, out_dir="backtest_results/", use_recent=False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print("="*65)
    print(f"BACKTEST — {n_markets} markets, Ollama only, alpha={alpha}")
    print("="*65)

    # 1. Fetch markets
    print(f"\n[1/4] Fetching {n_markets} resolved markets...")
    if use_recent and Path("recent_resolved_markets.json").exists():
        raw = json.loads(Path("recent_resolved_markets.json").read_text())
        from data.polymarket import Market
        all_markets = [Market(
            id=m["id"], question=m["question"], category=m.get("category",""),
            end_date=m.get("closed_date",""), market_prob=m["market_prob"],
            volume_usd=m.get("volume_usd",0), resolved=True, outcome=m["outcome"]
        ) for m in raw[:n_markets]]
    else:
        all_markets = fetch_resolved_markets(limit=n_markets)

    if len(all_markets) < 10:
        print(f"Only {len(all_markets)} markets found. Need at least 10.")
        return

    print(f"  Got {len(all_markets)} markets")

    split      = int(len(all_markets) * 0.65)
    calib_mkts = all_markets[:split]
    test_mkts  = all_markets[split:]
    print(f"  Calibration: {len(calib_mkts)}  Test: {len(test_mkts)}")

    # 2. Run ensemble on calibration
    print(f"\n[2/4] Running ensemble on {len(calib_mkts)} calibration markets...")
    calib_p, calib_y = [], []
    for i, m in enumerate(calib_mkts):
        print(f"  [{i+1}/{len(calib_mkts)}] {m.question[:55]}...", end=" ", flush=True)
        try:
            r = run_ensemble(m.question)
            calib_p.append(r.point_estimate)
            calib_y.append(m.outcome)
            print(f"p̂={r.point_estimate:.3f} σ={r.std_dev:.3f}")
        except Exception as e:
            print(f"SKIP ({e})")

    if len(calib_p) < 10:
        print(f"Only {len(calib_p)} calibration samples. Stopping.")
        return

    # 3. Fit calibration
    print(f"\n[3/4] Fitting conformal calibration on {len(calib_p)} samples...")
    calib = fit_calibration(calib_p, calib_y, alphas=[0.10, 0.20])
    print(f"  q_hat @ α={alpha}: {calib.quantiles[alpha]:.4f}")

    # 4. Run ensemble on test set
    print(f"\n[4/4] Running ensemble on {len(test_mkts)} test markets...")
    records = []
    for i, m in enumerate(test_mkts):
        print(f"  [{i+1}/{len(test_mkts)}] {m.question[:55]}...", end=" ", flush=True)
        try:
            r    = run_ensemble(m.question)
            iv   = predict_interval(r.point_estimate, calib, alpha)
            mc   = infer_market_confidence(m)
            modc = ci_width_to_confidence(iv)
            mis  = mc - modc
            act  = "INVESTIGATE" if mis > 0.25 else ("MONITOR" if mis > 0.10 else "PASS")
            records.append({
                "question":    m.question,
                "outcome":     m.outcome,
                "market_prob": m.market_prob,
                "p_hat":       r.point_estimate,
                "std":         r.std_dev,
                "ci_lower":    iv.lower,
                "ci_upper":    iv.upper,
                "ci_width":    iv.width,
                "ci_covered":  (iv.lower <= m.outcome <= iv.upper),
                "mismatch":    mis,
                "action":      act,
                "market_conf": mc,
            })
            print(f"p̂={r.point_estimate:.3f} [{act}]")
        except Exception as e:
            print(f"SKIP ({e})")

    if not records:
        print("No test records. Something went wrong.")
        return

    # Compute metrics
    p_hats    = [r["p_hat"]       for r in records]
    mkt_probs = [r["market_prob"] for r in records]
    outcomes  = [r["outcome"]     for r in records]

    coverage = float(np.mean([r["ci_covered"] for r in records]))
    roi, n_inv, wr = fade_simulation(records)

    inv_records = [r for r in records if r["action"] == "INVESTIGATE"]
    payoffs = []
    if inv_records:
        payoffs = [(1.0 if r["outcome"]==0 else 0.0) - (1-r["market_prob"]) for r in inv_records]

    corr = None
    merrs = [abs(p-y) for p,y in zip(mkt_probs, outcomes)]
    mismatches = [r["mismatch"] for r in records]
    if len(set(merrs)) > 1:
        c = float(np.corrcoef(mismatches, merrs)[0,1])
        if not np.isnan(c):
            corr = c

    metrics = {
        "n_markets":       len(records),
        "n_calib":         len(calib_p),
        "n_test":          len(records),
        "n_investigate":   n_inv or 0,
        "coverage":        coverage,
        "target":          1 - alpha,
        "mean_ci_width":   float(np.mean([r["ci_width"] for r in records])),
        "brier_model":     brier(p_hats, outcomes),
        "brier_market":    brier(mkt_probs, outcomes),
        "brier_constant":  brier([0.5]*len(outcomes), outcomes),
        "brier_improvement": round(brier(mkt_probs,outcomes) - brier(p_hats,outcomes), 4),
        "mismatch_corr":   corr,
        "fade_roi":        roi,
        "fade_win_rate":   wr,
        "stats":           statistical_tests(records, payoffs),
        "timestamp":       datetime.utcnow().isoformat(),
    }

    print_report(metrics)

    # Save
    Path(f"{out_dir}/summary.json").write_text(json.dumps(metrics, indent=2))
    with open(f"{out_dir}/records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved to {out_dir}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets",    type=int, default=40)
    parser.add_argument("--alpha",      type=float, default=0.20)
    parser.add_argument("--out",        default="backtest_results/")
    parser.add_argument("--use-recent", action="store_true")
    args = parser.parse_args()
    run(args.markets, args.alpha, args.out, args.use_recent)
