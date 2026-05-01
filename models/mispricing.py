"""
mispricing.py
─────────────
Compares our calibrated uncertainty intervals to the confidence implied
by the market itself — and scores the gap.

Core insight:
  - A liquid market at 60/40 with a tight order book is expressing high conviction
  - If our conformal interval is [0.30, 0.85], the market is wildly overconfident
  - That's the signal: our uncertainty is HIGHER than the market prices in

This file computes:
  1. Market-implied confidence from order book spread + volume
  2. The mismatch score between our CI width and market confidence
  3. A ranked list of opportunities

A positive mismatch score = market is more confident than our model.
These are the markets worth examining for edge.
"""

import math
from dataclasses import dataclass
from typing import Optional

from data.polymarket import Market, fetch_orderbook_spread
from models.conformal import ConformalInterval


# ── Market confidence inference ────────────────────────────────────────────────

def infer_market_confidence(
    market: Market,
    spread: Optional[float] = None,
) -> float:
    """
    Infer how confident the crowd is from observable market features.
    Returns a score in [0, 1]: higher = market is more certain.

    We use three signals:
    1. Distance of market_prob from 0.5 (extreme prob = more certain)
    2. Order book spread (tight spread = high liquidity = confident crowd)
    3. Volume (more traded = more information has been priced in)

    These are heuristics, not a formal model. The key insight is directional:
    high-confidence markets have high p, tight spreads, and high volume.
    """

    # Signal 1: probability extremity — how far from the coin-flip line
    # A market at 0.9 implies more certainty than one at 0.55
    prob_extremity = abs(market.market_prob - 0.5) * 2   # scales [0, 0.5] → [0, 1]

    # Signal 2: spread signal — tighter spread = more liquid = more confident
    # Normalize spread to [0,1] where 0 = very tight, 1 = very wide
    if spread is not None:
        spread_signal = max(0.0, 1.0 - (spread / 0.20))  # 0.20 is ~wide for crypto
    else:
        spread_signal = 0.5   # neutral if we couldn't fetch the book

    # Signal 3: volume signal — log-normalize across a reasonable range
    # Clip at $10k (very thin) and $10M (very liquid)
    vol_clipped = max(10_000, min(10_000_000, market.volume_usd))
    vol_signal  = (math.log(vol_clipped) - math.log(10_000)) / (math.log(10_000_000) - math.log(10_000))

    # Weighted combination (tunable)
    confidence = (
        0.40 * prob_extremity +
        0.35 * spread_signal  +
        0.25 * vol_signal
    )
    return float(max(0.0, min(1.0, confidence)))


def ci_width_to_confidence(ci: ConformalInterval) -> float:
    """
    Convert conformal interval width to an implied confidence score in [0,1].
    Width of 0 = perfectly confident (1.0). Width of 1 = maximally uncertain (0.0).
    """
    return 1.0 - ci.width


# ── Mispricing score ───────────────────────────────────────────────────────────

@dataclass
class MispricingSignal:
    market:              Market
    interval:            ConformalInterval
    market_confidence:   float      # inferred from spread + volume + prob
    model_confidence:    float      # 1 − CI width
    mismatch_score:      float      # market_confidence − model_confidence
                                    # positive = market overconfident (our signal)
    spread:              Optional[float]
    action:              str        # "INVESTIGATE", "PASS", "MONITOR"

    def __repr__(self):
        return (
            f"\n{'─'*60}\n"
            f"  {self.market.question[:70]}\n"
            f"  Market prob:  {self.market.market_prob:.3f}\n"
            f"  {self.interval}\n"
            f"  Market conf:  {self.market_confidence:.3f}\n"
            f"  Model conf:   {self.model_confidence:.3f}\n"
            f"  Mismatch:    {self.mismatch_score:+.3f}  →  {self.action}\n"
        )


def score_market(
    market:   Market,
    interval: ConformalInterval,
    fetch_spread: bool = True,
) -> MispricingSignal:
    """
    Compute the full mispricing signal for a single market.
    """
    spread = None
    if fetch_spread:
        try:
            spread = fetch_orderbook_spread(market.id)
        except Exception:
            pass    # fail gracefully; spread just contributes neutral signal

    market_conf = infer_market_confidence(market, spread)
    model_conf  = ci_width_to_confidence(interval)
    mismatch    = market_conf - model_conf

    # Action thresholds (tune with backtesting)
    if mismatch > 0.25:
        action = "INVESTIGATE"   # market is much more confident than our model
    elif mismatch > 0.10:
        action = "MONITOR"
    else:
        action = "PASS"

    return MispricingSignal(
        market=market,
        interval=interval,
        market_confidence=market_conf,
        model_confidence=model_conf,
        mismatch_score=mismatch,
        spread=spread,
        action=action,
    )


# ── Batch scoring + ranking ────────────────────────────────────────────────────

def rank_opportunities(
    markets:   list[Market],
    intervals: list[ConformalInterval],
    top_n:     int = 10,
    fetch_spread: bool = True,
) -> list[MispricingSignal]:
    """
    Score all markets and return the top_n by mismatch score.
    These are your highest-priority markets to examine.
    """
    assert len(markets) == len(intervals)

    signals = [
        score_market(m, iv, fetch_spread=fetch_spread)
        for m, iv in zip(markets, intervals)
    ]

    # Sort by mismatch descending: biggest gap first
    signals.sort(key=lambda s: s.mismatch_score, reverse=True)
    return signals[:top_n]


# ── Kelly position sizing ──────────────────────────────────────────────────────

def kelly_fraction(
    p_hat:       float,
    ci:          ConformalInterval,
    market_prob: float = None,    # if provided, determines bet direction
    b:           float = 1.0,     # net odds (1.0 = even money)
    max_frac:    float = 0.05,    # never risk more than 5% of bankroll
) -> tuple:
    """
    Kelly criterion adjusted for epistemic uncertainty.

    If model p̂ < market_prob: bet NO (fade the overconfident market)
    If model p̂ > market_prob: bet YES (market is underpricing this)

    Uncertainty scaling: multiply raw Kelly by (1 - CI_width).
    Wide CI = uncertain = smaller bet regardless of direction.

    Returns: (fraction, direction) where direction is 'YES', 'NO', or 'PASS'
    """
    uncertainty_scalar = max(0.0, 1.0 - ci.width)

    # Determine bet direction
    if market_prob is not None and p_hat < market_prob:
        # Fade the market — bet NO
        # Buying NO at price (1 - market_prob), wins if outcome = 0
        p_win  = 1.0 - p_hat          # model's prob that NO wins
        cost   = 1.0 - market_prob    # price of NO token
        if cost <= 0:
            return 0.0, "PASS"
        b_no   = (1.0 / cost) - 1.0  # net odds on NO bet
        kelly  = (b_no * p_win - (1.0 - p_win)) / b_no
        direction = "NO"
    else:
        # Bet YES
        p_win = p_hat
        kelly = (b * p_win - (1.0 - p_win)) / b
        direction = "YES"

    if kelly <= 0:
        return 0.0, "PASS"

    adjusted = kelly * uncertainty_scalar
    return min(adjusted, max_frac), direction
