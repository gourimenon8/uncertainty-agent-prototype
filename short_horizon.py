"""
short_horizon.py
────────────────
Find Polymarket markets closing within N days, run the ensemble on them,
and flag INVESTIGATE opportunities. Results are appended to results.jsonl
so fast_backtest.py picks them up once markets resolve.

Usage:
    python short_horizon.py --find --days 14
    python short_horizon.py --find --days 7 --all-categories
    python short_horizon.py --find --days 30 --min-volume 5000
"""

import argparse
import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", message=".*LibreSSL.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

from data.polymarket   import Market, _is_crypto, _get_pre_resolution_prob
from models.ensemble   import run_ensemble
from models.conformal  import CalibrationSet, predict_interval
from models.mispricing import rank_opportunities, kelly_fraction

GAMMA_BASE   = "https://gamma-api.polymarket.com"
CALIB_PATH   = "calibration.json"
RESULTS_PATH = "results.jsonl"
ALPHA        = 0.20


# ── Fetch markets closing within N days ──────────────────────────────────────

def fetch_short_horizon_markets(days, all_categories, min_volume, limit=500):
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days)

    print(f"  Fetching active markets closing before {cutoff.strftime('%Y-%m-%d')}...")

    # Paginate: there are thousands of unresolved-but-expired markets sorted first.
    # We page through until we see end dates beyond the cutoff.
    raw_list   = []
    offset     = 0
    page_limit = 500
    max_pages  = 20  # safety cap: 10,000 markets max

    for _ in range(max_pages):
        resp = requests.get(
            f"{GAMMA_BASE}/markets",
            params={"active": "true", "closed": "false", "limit": page_limit,
                    "order": "endDate", "ascending": "true", "offset": offset},
            timeout=12,
        )
        resp.raise_for_status()
        page = resp.json()
        if not isinstance(page, list) or not page:
            break

        raw_list.extend(page)

        # Stop once this page's last market is beyond the cutoff
        last_end = (page[-1].get("endDate") or "")[:19].replace("Z", "")
        try:
            last_dt = datetime.fromisoformat(last_end).replace(tzinfo=timezone.utc)
            if last_dt > cutoff:
                break
        except Exception:
            pass

        if len(page) < page_limit:
            break
        offset += page_limit
        print(f"    ...paged to offset {offset} (last date seen: {last_end[:10]})")

    markets        = []
    skipped_date   = 0
    skipped_crypto = 0
    skipped_prob   = 0
    skipped_volume = 0

    for raw in raw_list:
        q        = raw.get("question", "")
        end_date = raw.get("endDate") or raw.get("endDateIso", "")

        if end_date:
            try:
                end_dt = datetime.fromisoformat(
                    end_date[:19].replace("Z", "")
                ).replace(tzinfo=timezone.utc)
                if end_dt > cutoff or end_dt <= now:
                    skipped_date += 1
                    continue
            except Exception:
                skipped_date += 1
                continue
        else:
            skipped_date += 1
            continue

        if not all_categories and not _is_crypto(q):
            skipped_crypto += 1
            continue

        vol = float(raw.get("volumeNum") or raw.get("volume") or 0)
        if vol < min_volume:
            skipped_volume += 1
            continue

        p = _get_pre_resolution_prob(raw)
        if p is None:
            skipped_prob += 1
            continue

        markets.append(Market(
            id=str(raw.get("id", "")),
            question=q,
            category=raw.get("category", ""),
            end_date=end_date,
            market_prob=p,
            volume_usd=vol,
            resolved=False,
            outcome=None,
        ))

    cat = "all-category" if all_categories else "crypto"
    print(f"  Found {len(markets)} {cat} markets closing within {days} days")
    print(f"  Skipped: {skipped_date} wrong date | {skipped_prob} no prob"
          + (f" | {skipped_crypto} non-crypto" if not all_categories else "")
          + (f" | {skipped_volume} low volume" if min_volume > 0 else ""))
    return markets


# ── Save to results.jsonl ─────────────────────────────────────────────────────

def _match_opps(markets, opportunities):
    opp_map = {o.market.id: o for o in opportunities}
    return [opp_map.get(m.id) for m in markets]


def save_results(markets, ensemble_results, intervals, opportunities):
    ts = datetime.utcnow().isoformat()
    record = {
        "timestamp": ts,
        "source":    "short_horizon",
        "markets": [
            {
                "id":             m.id,
                "question":       m.question,
                "end_date":       m.end_date,
                "market_prob":    m.market_prob,
                "volume_usd":     m.volume_usd,
                "p_hat":          er.point_estimate,
                "ensemble_std":   er.std_dev,
                "ensemble_n":     er.n_samples,
                "ci_lower":       iv.lower,
                "ci_upper":       iv.upper,
                "ci_width":       iv.width,
                "mismatch_score": opp.mismatch_score if opp else None,
                "action":         opp.action         if opp else None,
            }
            for m, er, iv, opp in zip(
                markets, ensemble_results, intervals,
                _match_opps(markets, opportunities)
            )
        ],
    }
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nAppended {len(markets)} predictions to {RESULTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--find",           action="store_true")
    parser.add_argument("--days",           type=int,   default=14)
    parser.add_argument("--all-categories", action="store_true")
    parser.add_argument("--min-volume",     type=float, default=0)
    parser.add_argument("--top",            type=int,   default=10)
    args = parser.parse_args()

    if not args.find:
        print("Usage: python short_horizon.py --find --days 14")
        return

    print("=" * 60)
    print(f"SHORT HORIZON FINDER  (closing within {args.days} days)")
    print("=" * 60)

    print("\n[1/3] Loading calibration...")
    calib = CalibrationSet.load(CALIB_PATH)
    print(f"  n={calib.n_calibration}  mean_score={calib.mean_score:.4f}")

    print(f"\n[2/3] Fetching markets...")
    markets = fetch_short_horizon_markets(
        days=args.days,
        all_categories=args.all_categories,
        min_volume=args.min_volume,
    )

    if not markets:
        print("\nNo markets found. Try --days 30 or --all-categories.")
        return

    print(f"\n[3/3] Running ensemble on {len(markets)} markets...")
    ensemble_results = []
    valid_markets    = []

    for i, m in enumerate(markets):
        end_str = m.end_date[:10] if m.end_date else "?"
        print(f"  [{i+1}/{len(markets)}] [{end_str}] {m.question[:55]}...", end=" ", flush=True)
        try:
            r = run_ensemble(m.question)
            ensemble_results.append(r)
            valid_markets.append(m)
            print(f"p̂={r.point_estimate:.3f}  σ={r.std_dev:.3f}")
        except Exception as e:
            print(f"skipped: {e}")

    if not valid_markets:
        print("No valid results.")
        return

    intervals     = [predict_interval(r.point_estimate, calib, ALPHA) for r in ensemble_results]
    opportunities = rank_opportunities(valid_markets, intervals, top_n=len(valid_markets))

    save_results(valid_markets, ensemble_results, intervals, opportunities)

    investigate = [o for o in opportunities if o.action == "INVESTIGATE"]
    print(f"\n{'='*60}")
    print(f"INVESTIGATE SIGNALS  ({len(investigate)} / {len(valid_markets)} markets)")
    print(f"{'='*60}")

    if not investigate:
        print("No INVESTIGATE signals — model and market agree on all.")
    else:
        for opp in sorted(investigate, key=lambda o: -abs(o.mismatch_score)):
            m  = opp.market
            iv = opp.interval
            days_left = ""
            if m.end_date:
                try:
                    end_dt    = datetime.fromisoformat(m.end_date[:19].replace("Z","")).replace(tzinfo=timezone.utc)
                    days_left = f"  closes in {(end_dt - datetime.now(timezone.utc)).days}d"
                except Exception:
                    pass
            print(f"\n  {m.question[:70]}")
            print(f"    market={m.market_prob:.0%}  model={iv.point_estimate:.0%}"
                  f"  CI=[{iv.lower:.0%},{iv.upper:.0%}]"
                  f"  mismatch={opp.mismatch_score:+.3f}{days_left}")
            frac, direction = kelly_fraction(iv.point_estimate, iv, market_prob=m.market_prob)
            if frac > 0:
                print(f"    Kelly: {frac*100:.1f}% bankroll → bet {direction}")

    print(f"\n{'='*60}")
    print("Run python fast_backtest.py after markets resolve to score predictions.")


if __name__ == "__main__":
    main()
