"""
conformal.py
────────────
Split conformal prediction applied to LLM ensemble outputs.

This is the methodologically serious piece. It gives prediction intervals
with GUARANTEED marginal coverage — without assuming Gaussian outputs,
without assuming a well-specified model, and without assuming stationarity.

The only assumption: calibration and test markets are exchangeable
(drawn from the same distribution). For same-category crypto markets,
this is reasonable.

Reference: Angelopoulos & Bates (2022) "A Gentle Introduction to
Conformal Prediction and Distribution-Free Uncertainty Quantification"
https://arxiv.org/abs/2107.07511
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class CalibrationSet:
    """
    Stores fitted nonconformity score quantiles at multiple alpha levels.
    Serialize this and reuse — don't refit on every run.
    """
    n_calibration:   int
    quantiles:       dict          # alpha → q_hat (e.g. 0.10 → 0.18)
    mean_score:      float
    score_std:       float

    def save(self, path: str):
        Path(path).write_text(json.dumps({
            "n_calibration": self.n_calibration,
            "quantiles":     self.quantiles,
            "mean_score":    self.mean_score,
            "score_std":     self.score_std,
        }, indent=2))

    @classmethod
    def load(cls, path: str) -> "CalibrationSet":
        d = json.loads(Path(path).read_text())
        return cls(
            n_calibration=d["n_calibration"],
            quantiles={float(k): v for k, v in d["quantiles"].items()},
            mean_score=d["mean_score"],
            score_std=d["score_std"],
        )


@dataclass
class ConformalInterval:
    """
    A prediction interval for a single market with guaranteed coverage.
    """
    point_estimate:  float          # p̂ from ensemble mean
    lower:           float          # clipped to [0, 1]
    upper:           float          # clipped to [0, 1]
    alpha:           float          # miscoverage level used (e.g. 0.10 = 90% coverage)
    width:           float          # upper − lower; the uncertainty magnitude
    q_hat:           float          # nonconformity quantile used

    @property
    def coverage_level(self) -> float:
        return 1.0 - self.alpha

    def __repr__(self):
        return (
            f"p̂={self.point_estimate:.3f}  "
            f"CI=[{self.lower:.3f}, {self.upper:.3f}]  "
            f"width={self.width:.3f}  "
            f"({self.coverage_level*100:.0f}% coverage)"
        )


# ── Nonconformity score ────────────────────────────────────────────────────────

def nonconformity_score(p_hat: float, y: int) -> float:
    """
    Absolute residual score: |p̂ − y|.

    The simplest valid choice. Alternatives:
      - Interval-based: max(p̂ - y, y - p̂) (same for binary y)
      - CQR (conformalized quantile regression) for richer ensemble outputs
      - Adaptive scores that weight by ensemble variance

    For v1, the absolute residual is sufficient and well-understood.
    Larger score = model was more wrong = more nonconforming.
    """
    return abs(p_hat - y)


# ── Calibration fitting ────────────────────────────────────────────────────────

def fit_calibration(
    p_hats: list,
    outcomes: list,
    alphas: list = None,
) -> CalibrationSet:
    """
    Fit nonconformity score distribution on the calibration set.

    For each alpha (miscoverage level), we compute q_hat = the ceil((1-α)(n+1)/n)
    quantile of calibration scores. This gives the Bonferroni-corrected quantile
    that guarantees marginal coverage ≥ 1−α on test data.

    Args:
        p_hats:   list of ensemble mean probabilities on calibration markets
        outcomes: list of actual binary outcomes (same order)
        alphas:   list of miscoverage levels to precompute (default: 0.05, 0.10, 0.20)

    Returns:
        CalibrationSet ready to use for prediction intervals
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.20]   # 95%, 90%, 80% coverage

    assert len(p_hats) == len(outcomes), "Lengths must match"
    assert len(p_hats) >= 30, (
        f"Calibration set too small ({len(p_hats)}). "
        "Need ≥30 for reliable quantile estimates; aim for 100+."
    )

    scores = np.array([
        nonconformity_score(p, y)
        for p, y in zip(p_hats, outcomes)
    ])

    n = len(scores)
    quantiles = {}
    for alpha in alphas:
        # Conformal quantile: ceil((1-α)(n+1)/n)-th empirical quantile
        # This is the key formula from Vovk et al. / Angelopoulos & Bates
        level = min(1.0, np.ceil((1 - alpha) * (n + 1)) / n)
        q_hat = float(np.quantile(scores, level))
        quantiles[alpha] = q_hat

    return CalibrationSet(
        n_calibration=n,
        quantiles=quantiles,
        mean_score=float(scores.mean()),
        score_std=float(scores.std()),
    )


# ── Interval prediction ────────────────────────────────────────────────────────

def predict_interval(
    p_hat: float,
    calib: CalibrationSet,
    alpha: float = 0.10,
) -> ConformalInterval:
    """
    Given a point estimate and fitted calibration, return a conformal interval.

    The interval [p̂ − q̂, p̂ + q̂] has guaranteed marginal coverage ≥ 1−α,
    meaning: over many test markets, at least (1−α) fraction will contain
    the true outcome.

    This is NOT a frequentist confidence interval for a parameter.
    It's a predictive interval for the binary outcome — a different (stronger)
    claim: it covers the actual event, not just the mean.
    """
    if alpha not in calib.quantiles:
        raise ValueError(
            f"alpha={alpha} not in calibration set. "
            f"Available: {list(calib.quantiles.keys())}"
        )

    q_hat = calib.quantiles[alpha]
    lower = max(0.0, p_hat - q_hat)
    upper = min(1.0, p_hat + q_hat)

    return ConformalInterval(
        point_estimate=p_hat,
        lower=lower,
        upper=upper,
        alpha=alpha,
        width=upper - lower,
        q_hat=q_hat,
    )


# ── Batch prediction ───────────────────────────────────────────────────────────

def predict_intervals_batch(
    p_hats: list,
    calib: CalibrationSet,
    alpha: float = 0.10,
) -> list:
    return [predict_interval(p, calib, alpha) for p in p_hats]


def coverage_report(
    intervals: list,
    outcomes:  list,
) -> dict:
    """
    Empirically verify coverage on a held-out test set.
    Should be ≥ (1−alpha). If it's much higher, your intervals are conservative.
    If it's lower, something is wrong (distribution shift, or too-small calib set).
    """
    assert len(intervals) == len(outcomes)
    covered = [
        iv.lower <= y <= iv.upper
        for iv, y in zip(intervals, outcomes)
    ]
    empirical_coverage = sum(covered) / len(covered)
    target_coverage    = 1.0 - intervals[0].alpha

    return {
        "n_test":             len(intervals),
        "target_coverage":    target_coverage,
        "empirical_coverage": empirical_coverage,
        "coverage_gap":       empirical_coverage - target_coverage,
        "mean_width":         np.mean([iv.width for iv in intervals]),
        "calibration_ok":     empirical_coverage >= target_coverage,
    }


# ── Rolling recalibration ──────────────────────────────────────────────────────

class RollingCalibrator:
    """
    Recalibrate on a sliding window of the most recent N resolved markets.
    Important for crypto: the distribution shifts with regime (bull/bear/sideways).
    Stale calibration = invalid coverage guarantees.

    Usage:
        calibrator = RollingCalibrator(window=150)
        calibrator.add(p_hat=0.62, outcome=1)
        ...
        calib = calibrator.fit()
        interval = predict_interval(new_p_hat, calib)
    """
    def __init__(self, window: int = 150):
        self.window = window
        self._p_hats:   list[float] = []
        self._outcomes: list[int]   = []

    def add(self, p_hat: float, outcome: int):
        self._p_hats.append(p_hat)
        self._outcomes.append(outcome)
        # Trim to window
        if len(self._p_hats) > self.window:
            self._p_hats   = self._p_hats[-self.window:]
            self._outcomes = self._outcomes[-self.window:]

    def fit(self, alphas: list[float] = None) -> CalibrationSet:
        return fit_calibration(self._p_hats, self._outcomes, alphas)

    @property
    def n(self) -> int:
        return len(self._p_hats)
