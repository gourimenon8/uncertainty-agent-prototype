"""
fetch_recent.py
───────────────
Polymarket's API returns oldest markets first. To get post-2025 resolved
markets (uncontaminated by LLM training data), we need to paginate far
into the API. This script finds the right offset and saves recent markets.

Run once:
    python fetch_recent.py

Saves: recent_resolved_markets.json
Then recalibrate: python agent.py --calibrate --use-recent
"""

import json
import time
import warnings
import requests
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message=".*LibreSSL.*")

GAMMA_BASE   = "https://gamma-api.polymarket.com"
TARGET_DATE  = datetime(2024, 6, 1)   # only markets closed after this (2 years back)
OUTPUT_FILE  = "recent_resolved_markets.json"
BATCH        = 200
MAX_OFFSET   = 100_000   # safety cap


def _fetch_page(offset: int) -> list:
    resp = requests.get(
        f"{GAMMA_BASE}/markets",
        params={"closed": "true", "limit": BATCH, "offset": offset},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _parse_date(raw: dict) -> datetime:
    for field in ["closedTime", "endDateIso", "endDate"]:
        val = raw.get(field, "")
        if val:
            try:
                return datetime.strptime(str(val)[:10], "%Y-%m-%d")
            except Exception:
                pass
    return datetime.min


def _infer_outcome(raw: dict):
    op = raw.get("outcomePrices")
    if not op:
        return None
    try:
        prices = json.loads(op) if isinstance(op, str) else op
        yes_price = float(prices[0])
        no_price  = float(prices[1]) if len(prices) > 1 else 0.0
        if yes_price == 0.0 and no_price == 0.0:
            return None
        return 1 if yes_price >= 0.5 else 0
    except Exception:
        return None


def _get_prob(raw: dict):
    ltp_raw = raw.get("lastTradePrice")
    ltp = None
    try:
        ltp = float(ltp_raw) if ltp_raw is not None else None
    except Exception:
        pass

    if ltp is not None:
        for field in ["oneDayPriceChange", "oneWeekPriceChange"]:
            change = raw.get(field)
            if change is not None:
                try:
                    prior = ltp - float(change)
                    if 0.02 < prior < 0.98:
                        return round(prior, 4)
                except Exception:
                    pass

    if ltp is not None and 0.02 < ltp < 0.98:
        return round(ltp, 4)

    bid = raw.get("bestBid")
    ask = raw.get("bestAsk")
    if bid and ask:
        try:
            mid = (float(bid) + float(ask)) / 2
            if 0.01 < mid < 0.99:
                return round(mid, 4)
        except Exception:
            pass

    return None


def find_recent_markets(target: int = 200) -> list:
    """
    Binary search for the offset where markets start being from 2025+,
    then collect until we have enough clean recent markets.
    """
    print(f"Searching for post-{TARGET_DATE.strftime('%Y-%m')} resolved markets...")
    print(f"Target: {target} clean markets\n")

    # Step 1: find approximate offset where 2025+ markets start
    # Sample at increasing offsets until we hit recent dates
    probe_offsets = [0, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
    start_offset  = 0

    for offset in probe_offsets:
        page = _fetch_page(offset)
        if not page:
            break
        dates = [_parse_date(m) for m in page[:5]]
        newest = max(dates)
        print(f"  Offset {offset:6d} → newest date: {newest.strftime('%Y-%m-%d')}")
        if newest >= TARGET_DATE:
            start_offset = max(0, offset - 5000)
            break
        time.sleep(0.3)

    print(f"\nStarting collection from offset ~{start_offset}...")
    print(f"Looking for markets closed after {TARGET_DATE.strftime('%Y-%m-%d')}\n")

    # Step 2: collect from start_offset forward
    markets  = []
    offset   = start_offset
    skipped  = {"too_old": 0, "no_outcome": 0, "no_prob": 0}

    while len(markets) < target and offset < MAX_OFFSET:
        page = _fetch_page(offset)
        if not page:
            print("  API returned empty page — stopping")
            break

        for raw in page:
            date = _parse_date(raw)

            if date < TARGET_DATE:
                skipped["too_old"] += 1
                continue

            outcome = _infer_outcome(raw)
            if outcome is None:
                skipped["no_outcome"] += 1
                continue

            prob = _get_prob(raw)
            if prob is None:
                skipped["no_prob"] += 1
                continue

            markets.append({
                "id":         str(raw.get("id", "")),
                "question":   raw.get("question", ""),
                "category":   raw.get("category", ""),
                "closed_date": date.strftime("%Y-%m-%d"),
                "market_prob": prob,
                "outcome":    outcome,
                "volume_usd": float(raw.get("volumeNum") or raw.get("volume") or 0),
            })

            if len(markets) >= target:
                break

        offset += BATCH
        print(f"  Offset {offset:6d} → collected {len(markets)}/{target} "
              f"(skipped: {sum(skipped.values())} total)")
        time.sleep(0.2)

    print(f"\nDone.")
    print(f"  Collected: {len(markets)} recent resolved markets")
    print(f"  Skipped: {skipped}")

    if markets:
        cats = {}
        for m in markets:
            cats[m["category"]] = cats.get(m["category"], 0) + 1
        top = sorted(cats.items(), key=lambda x: -x[1])[:5]
        print(f"  Categories: {top}")

        date_range = [m["closed_date"] for m in markets]
        print(f"  Date range: {min(date_range)} → {max(date_range)}")

    return markets


def main():
    markets = find_recent_markets(target=300)

    if len(markets) < 30:
        print("\n!! Found fewer than 30 markets.")
        print("   The API may not have enough post-2025 resolved markets yet.")
        print("   Try lowering TARGET_DATE to datetime(2025, 6, 1) and rerun.")
        return

    Path(OUTPUT_FILE).write_text(json.dumps(markets, indent=2))
    print(f"\nSaved {len(markets)} markets to {OUTPUT_FILE}")
    print("Now recalibrate: python agent.py --calibrate --use-recent")


if __name__ == "__main__":
    main()
