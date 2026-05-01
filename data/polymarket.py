"""
polymarket.py
─────────────
Fetches markets from Polymarket Gamma API.

Key design decision:
  - fetch_resolved_markets()      → ALL binary markets, used for calibration
  - fetch_live_crypto_markets()   → crypto-only, used for live monitoring

Using all markets for calibration is correct — the conformal predictor
only needs exchangeable binary outcome samples, not crypto-specific ones.
Crypto filtering only matters for what the agent actually monitors.
"""

import json
import time
import warnings
import requests
from collections import Counter

warnings.filterwarnings("ignore", message=".*LibreSSL.*")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

CRYPTO_KEYWORDS = [
    # Core assets
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "ripple",
    "dogecoin", "doge", "shiba", "avalanche", "avax", "polygon", "matic",
    "chainlink", "link", "uniswap", "uni", "aave", "compound", "maker",
    "usdc", "usdt", "dai", "stablecoin", "tether", "litecoin", "ltc",
    "cardano", "ada", "polkadot", "dot", "cosmos", "atom", "near",
    "arbitrum", "optimism", "base chain", "sui", "aptos", "sei",
    # Concepts
    "crypto", "cryptocurrency", "blockchain", "defi", "nft", "dao", "web3",
    "staking", "validator", "mining", "halving", "mempool", "gas fee",
    "wallet", "exchange", "dex", "cex", "layer 2", "l2", "rollup",
    "market cap", "all-time high", "ath", "satoshi", "gwei",
    # Companies / events
    "coinbase", "binance", "ftx", "celsius", "grayscale", "microstrategy",
    "blackrock bitcoin", "spot etf", "crypto etf", "sec crypto",
    "ledger", "metamask", "trezor", "hardware wallet",
    # Price milestone phrasing (catches "bitcoin hit $1m", "BTC above")
    "$1m", "$1 million", "$500k", "$100k", "$200k", "$50k",
    "1 million", "100,000", "100000",
    # GTA VI is commonly paired with BTC milestones on Polymarket
    "gta vi",
]


# ── Data model ─────────────────────────────────────────────────────────────────

class Market:
    def __init__(self, id, question, category, end_date,
                 market_prob, volume_usd, resolved, outcome, tags=None):
        self.id          = id
        self.question    = question
        self.category    = category
        self.end_date    = end_date
        self.market_prob = market_prob
        self.volume_usd  = volume_usd
        self.resolved    = resolved
        self.outcome     = outcome   # 1=YES, 0=NO, None=live
        self.tags        = tags or []

    def __repr__(self):
        return f"Market({self.question[:50]!r}, p={self.market_prob:.2f}, outcome={self.outcome})"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_crypto(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CRYPTO_KEYWORDS)


def _infer_outcome(raw: dict):
    """
    Infer YES/NO from outcomePrices after resolution.

    From debug output we know two patterns:
      ["0", "0"]              → cancelled/dead market, skip
      ["0.000001", "0.999"]  → resolved: YES near 0 = NO won, YES near 1 = YES won

    We use prices[0] > 0.5 → YES won, prices[0] < 0.5 → NO won.
    Skip only if both prices are exactly zero (cancelled).
    """
    op = raw.get("outcomePrices")
    if not op:
        return None
    try:
        prices = json.loads(op) if isinstance(op, str) else op
        yes_price = float(prices[0])
        no_price  = float(prices[1]) if len(prices) > 1 else 0.0

        # Both zero = cancelled market, not a real resolution
        if yes_price == 0.0 and no_price == 0.0:
            return None

        # Use 0.5 as the threshold — works for both near-0 and near-1 cases
        return 1 if yes_price >= 0.5 else 0

    except Exception:
        return None


def _get_pre_resolution_prob(raw: dict):
    """
    Get the probability implied by the market BEFORE resolution.

    lastTradePrice on closed markets is typically 0 or 1 (settlement price).
    We back-calculate using price change fields:
      prior_price = lastTradePrice - oneDayPriceChange

    If that fails we try bestBid/bestAsk and spread.
    """
    ltp_raw = raw.get("lastTradePrice")
    ltp = None
    try:
        ltp = float(ltp_raw) if ltp_raw is not None else None
    except Exception:
        pass

    # Strategy 1: back-calculate from oneDayPriceChange
    # lastTradePrice = 0 (NO settled), change = -0.65 → prior was 0.65
    # lastTradePrice = 1 (YES settled), change = +0.40 → prior was 0.60
    if ltp is not None:
        for field in ["oneDayPriceChange", "oneWeekPriceChange", "oneHourPriceChange"]:
            change = raw.get(field)
            if change is not None:
                try:
                    prior = ltp - float(change)
                    if 0.02 < prior < 0.98:
                        return round(prior, 4)
                except Exception:
                    pass

    # Strategy 2: lastTradePrice itself if it's not a binary settlement
    if ltp is not None and 0.02 < ltp < 0.98:
        return round(ltp, 4)

    # Strategy 3: bestBid / bestAsk mid
    bid = raw.get("bestBid")
    ask = raw.get("bestAsk")
    if bid and ask:
        try:
            mid = (float(bid) + float(ask)) / 2
            if 0.01 < mid < 0.99:
                return round(mid, 4)
        except Exception:
            pass

    # Strategy 4: spread back-calculation from bestAsk
    spread = raw.get("spread")
    if ask and spread:
        try:
            implied_bid = float(ask) - float(spread)
            mid = (implied_bid + float(ask)) / 2
            if 0.01 < mid < 0.99:
                return round(mid, 4)
        except Exception:
            pass

    # Strategy 5: use outcomePrices as last resort if not exactly 0/1
    # (works when market was still trading near resolution)
    op = raw.get("outcomePrices")
    if op:
        try:
            prices = json.loads(op) if isinstance(op, str) else op
            p = float(prices[0])
            if 0.02 < p < 0.98:
                return round(p, 4)
        except Exception:
            pass

    return None


def _fetch_page(params: dict) -> list:
    resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


# ── Public fetchers ────────────────────────────────────────────────────────────

def fetch_live_crypto_markets(limit: int = 100) -> list:
    """Fetch open crypto markets for live monitoring."""
    raw_list = _fetch_page({"active": "true", "closed": "false", "limit": limit})

    markets = []
    for raw in raw_list:
        q = raw.get("question", "")
        if not _is_crypto(q):
            continue
        p = _get_pre_resolution_prob(raw)
        if p is None:
            continue
        markets.append(Market(
            id=str(raw.get("id", "")),
            question=q,
            category=raw.get("category", "crypto"),
            end_date=raw.get("endDate", ""),
            market_prob=p,
            volume_usd=float(raw.get("volumeNum") or raw.get("volume") or 0),
            resolved=False,
            outcome=None,
        ))

    print(f"  [polymarket] {len(markets)} live crypto markets "
          f"(from {len(raw_list)} total active)")
    return markets


def fetch_resolved_markets(
    limit: int = 500,
    crypto_only: bool = False,
    months_recent: int = 6,       # only use markets closed in last N months
) -> list:
    """
    Fetch resolved binary markets for calibration.

    months_recent=6 is the key parameter. LLMs have training data up to
    early 2025, so they already know outcomes for older markets — that
    inflates calibration error and makes CI intervals too wide.
    Using only recent markets avoids this contamination.

    By default fetches ALL categories — conformal calibration only needs
    exchangeable binary outcome samples, not crypto-specific ones.
    """
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(days=30 * months_recent)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    print(f"  [polymarket] Fetching closed markets (closed after {cutoff_str})...")

    all_raw   = []
    batch     = 200
    offset    = 0
    max_fetch = limit * 8

    while len(all_raw) < max_fetch:
        page = _fetch_page({
            "closed":   "true",
            "limit":    batch,
            "offset":   offset,
        })
        if not page:
            break
        all_raw.extend(page)
        offset += batch
        if len(page) < batch:
            break

    print(f"  [polymarket] Processing {len(all_raw)} closed records...")

    markets         = []
    skipped_old     = 0
    skipped_crypto  = 0
    skipped_outcome = 0
    skipped_prob    = 0
    cat_counter     = Counter()

    for raw in all_raw:
        q   = raw.get("question", "")
        cat = raw.get("category", "")
        cat_counter[cat] += 1

        # ── Date filter: skip markets that closed before cutoff ───────────────
        closed_time = raw.get("closedTime") or raw.get("endDateIso") or raw.get("endDate", "")
        if closed_time:
            try:
                # Handle both "2025-11-01T00:00:00Z" and "2025-11-01" formats
                date_str = closed_time[:10]
                closed_date = datetime.strptime(date_str, "%Y-%m-%d")
                if closed_date < cutoff:
                    skipped_old += 1
                    continue
            except Exception:
                pass   # if we can't parse the date, keep the market

        if crypto_only and not _is_crypto(q):
            skipped_crypto += 1
            continue

        outcome = _infer_outcome(raw)
        if outcome is None:
            skipped_outcome += 1
            continue

        p = _get_pre_resolution_prob(raw)
        if p is None:
            skipped_prob += 1
            continue

        markets.append(Market(
            id=str(raw.get("id", "")),
            question=q,
            category=cat,
            end_date=closed_time,
            market_prob=p,
            volume_usd=float(raw.get("volumeNum") or raw.get("volume") or 0),
            resolved=True,
            outcome=outcome,
        ))

        if len(markets) >= limit:
            break

    print(f"  [polymarket] {len(markets)} recent resolved markets kept")
    print(f"               Skipped: {skipped_old} too old (pre-{cutoff_str}) | "
          f"{skipped_outcome} ambiguous outcome | {skipped_prob} no prob"
          + (f" | {skipped_crypto} non-crypto" if crypto_only else ""))

    if cat_counter:
        print(f"               Top categories: {cat_counter.most_common(5)}")

    if len(markets) < 30:
        print()
        if months_recent <= 48:
            print(f"  !! Only {len(markets)} recent markets found — API returns oldest first.")
            print(f"     Falling back to all available markets (no date filter)...")
            return fetch_resolved_markets(limit=limit, crypto_only=crypto_only, months_recent=9999)
        else:
            print(f"  !! Only {len(markets)} markets found total. Using what we have.")

    return markets


# Keep old name as alias so agent.py doesn't break
def fetch_resolved_crypto_markets(limit: int = 500, months_recent: int = 6) -> list:
    """Alias — fetches all binary markets (not just crypto) for calibration."""
    return fetch_resolved_markets(limit=limit, crypto_only=False, months_recent=months_recent)


def _diagnose_shortage(sample: list):
    """Print diagnostic info when we can't find enough resolved markets."""
    print()
    print("  !! Diagnostic — first 5 closed market samples:")
    for raw in sample[:5]:
        print(f"     q={raw.get('question','')[:55]!r}")
        print(f"       ltp={raw.get('lastTradePrice')}  "
              f"bid={raw.get('bestBid')}  ask={raw.get('bestAsk')}  "
              f"op={str(raw.get('outcomePrices',''))[:30]}")
    print()
    print("  Fix options:")
    print("  1. Run `python debug_api.py` and paste output to Claude")
    print("  2. Lower outcome threshold in _infer_outcome() from 0.95 to 0.90")
    print("  3. Use synthetic calibration: `python agent.py --calibrate --synthetic`")


def fetch_orderbook_spread(market_id: str):
    try:
        resp = requests.get(
            f"{CLOB_BASE}/book", params={"token_id": market_id}, timeout=5)
        if resp.status_code != 200:
            return None
        book = resp.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if not bids or not asks:
            return None
        return float(asks[0]["price"]) - float(bids[0]["price"])
    except Exception:
        return None


def with_retry(fn, retries=3, backoff=1.5):
    for attempt in range(retries):
        try:
            return fn()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)
