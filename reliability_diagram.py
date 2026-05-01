"""
reliability_diagram.py
──────────────────────
Generates a reliability diagram (calibration curve) from backtest records.
Standard visual proof of calibration quality for the paper.

Usage:
    python reliability_diagram.py --records backtest_results/records.jsonl
    python reliability_diagram.py --records backtest_results/records.jsonl --show

Output: reliability_diagram.png
"""

import json
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL = True
except ImportError:
    MPL = False
    print("Install matplotlib: pip install matplotlib --break-system-packages")


def load_records(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def reliability_data(p_hats: list, outcomes: list, n_bins: int = 10):
    """
    Bin predictions and compute empirical frequencies.
    Returns bin centers, empirical frequencies, counts.
    """
    bins       = np.linspace(0, 1, n_bins + 1)
    centers    = []
    freqs      = []
    counts     = []
    mean_preds = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = [(lo <= p < hi) for p in p_hats]

        bin_p = [p for p, m in zip(p_hats, mask) if m]
        bin_y = [y for y, m in zip(outcomes, mask) if m]

        if len(bin_y) >= 2:
            centers.append((lo + hi) / 2)
            freqs.append(np.mean(bin_y))
            counts.append(len(bin_y))
            mean_preds.append(np.mean(bin_p))

    return np.array(centers), np.array(freqs), np.array(counts), np.array(mean_preds)


def plot_reliability(records: list, out_path: str = "reliability_diagram.png"):
    if not MPL:
        print("matplotlib not available — cannot plot")
        return

    # Extract data
    p_hats       = [r["p_hat"]       for r in records if r.get("p_hat") is not None]
    market_probs = [r["market_prob"]  for r in records if r.get("market_prob") is not None]
    outcomes     = [r["outcome"]      for r in records if r.get("outcome") is not None]

    # Align lengths
    n = min(len(p_hats), len(market_probs), len(outcomes))
    p_hats       = p_hats[:n]
    market_probs = market_probs[:n]
    outcomes     = outcomes[:n]

    if n < 10:
        print(f"Only {n} records — need at least 10 for a meaningful diagram")
        return

    # Compute reliability data for both model and market
    centers_m, freqs_m, counts_m, preds_m = reliability_data(p_hats,       outcomes)
    centers_c, freqs_c, counts_c, preds_c = reliability_data(market_probs, outcomes)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Reliability Diagram — Epistemic Uncertainty Agent", fontsize=14, y=1.02)

    for ax, centers, freqs, counts, preds, label, color in [
        (axes[0], centers_m, freqs_m, counts_m, preds_m, "LLM Ensemble (our model)", "#534AB7"),
        (axes[1], centers_c, freqs_c, counts_c, preds_c, "Market probability (crowd)", "#0F6E56"),
    ]:
        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")

        # Reliability curve
        ax.plot(preds, freqs, "o-", color=color, linewidth=2,
                markersize=6, label=label)

        # Shaded area showing deviation from perfect
        ax.fill_between(preds, preds, freqs, alpha=0.12, color=color)

        # Count annotations
        for p, f, c in zip(preds, freqs, counts):
            ax.annotate(f"n={c}", (p, f), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color="#888")

        # Formatting
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted probability", fontsize=11)
        ax.set_ylabel("Empirical frequency", fontsize=11)
        ax.set_title(label, fontsize=11, color=color)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # Brier score annotation
        bs = np.mean([(p - y) ** 2 for p, y in zip(
            market_probs if label.startswith("Market") else p_hats, outcomes
        )])
        ax.text(0.98, 0.04, f"Brier = {bs:.4f}", transform=ax.transAxes,
                ha="right", fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also print a text version for quick reading
    print("\nReliability summary (our model):")
    print(f"  {'Predicted':>12}  {'Empirical':>10}  {'Count':>6}  {'Gap':>8}")
    for p, f, c in zip(preds_m, freqs_m, counts_m):
        gap = f - p
        bar = "▲" if gap > 0.05 else ("▼" if gap < -0.05 else "≈")
        print(f"  {p:>12.2f}  {f:>10.2f}  {c:>6d}  {gap:>+7.3f} {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="backtest_results/records.jsonl")
    parser.add_argument("--out",     default="reliability_diagram.png")
    args = parser.parse_args()

    if not Path(args.records).exists():
        print(f"Records file not found: {args.records}")
        print("Run backtest first: python backtest.py --out backtest_results/")
        return

    records = load_records(args.records)
    print(f"Loaded {len(records)} backtest records")
    plot_reliability(records, args.out)


if __name__ == "__main__":
    main()
