# ── requirements.txt ─────────────────────────────────────────────────────────

openai>=1.30.0
anthropic>=0.25.0
numpy>=1.26.0
requests>=2.31.0
python-dotenv>=1.0.0


# ── .env.template ─────────────────────────────────────────────────────────────
# Copy to .env and fill in your keys. Never commit .env.

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...


# ── Project structure ─────────────────────────────────────────────────────────

uncertainty_agent/
├── agent.py                  ← entry point: --calibrate | --run | --once
├── requirements.txt
├── calibration.json          ← written by --calibrate, read by --run
├── results.jsonl             ← sweep-by-sweep output for backtesting
│
├── data/
│   └── polymarket.py         ← fetch live + resolved markets from CLOB/Gamma APIs
│
└── models/
    ├── ensemble.py           ← LLM ensemble: multiple models × prompt variants
    ├── conformal.py          ← split conformal calibration + interval prediction
    └── mispricing.py         ← market confidence inference + mismatch scoring


# ── README ────────────────────────────────────────────────────────────────────

## Epistemic Uncertainty Agent for Crypto Prediction Markets

An AI agent that surfaces markets where the crowd is overconfident relative
to what the evidence actually supports — by comparing calibrated LLM ensemble
uncertainty intervals to market-implied confidence.

### Core idea
Every prediction market gives one number: "68% chance ETH hits $5k."
But that collapses two very different situations:
  - A confident 68% (models agree, strong signals, liquid market)
  - An uncertain 68% (models disagree, thin signals, illiquid book)

This agent distinguishes them. Wide conformal interval + tight market odds = edge.

### Quick start

    pip install -r requirements.txt
    cp .env.template .env           # fill in API keys

    # Step 1: build calibration set (run once; costs ~$10-20 in API calls)
    python agent.py --calibrate

    # Step 2: run a single sweep to verify everything works
    python agent.py --once

    # Step 3: start the live loop
    python agent.py --run

### The conformal guarantee

Intervals from conformal.py have GUARANTEED marginal coverage ≥ 90%
(with default alpha=0.10) over the calibration distribution.
This means: on markets similar to your calibration set, at least 90%
of your intervals will contain the true outcome.

This is NOT a Gaussian confidence interval. It requires no distributional
assumptions about LLM outputs. The only requirement is exchangeability
between calibration and test markets — reasonable for same-category crypto markets.

### Tuning

Adjust in agent.py:
  ALPHA = 0.10        # 0.05 = tighter 95% intervals (more conservative)
                      # 0.20 = wider 80% intervals (more aggressive)
  POLL_INTERVAL = 300 # how often to sweep live markets (seconds)
  TOP_N = 10          # how many opportunities to surface per sweep

Adjust thresholds in mispricing.py:
  mismatch > 0.25 → INVESTIGATE
  mismatch > 0.10 → MONITOR

### Known limitations and next steps

1. Calibration drift: refit weekly using RollingCalibrator in conformal.py
2. LLM overconfidence: models have priors about crypto baked in.
   Experiment with prompt templates that explicitly elicit doubt.
3. Thin markets: market confidence inference degrades on illiquid books.
   Filter out markets below $50k volume.
4. Execution: this surfaces opportunities but doesn't place bets.
   Polymarket has a CLOB API for programmatic order placement.

### Research angle

Conformal prediction applied to LLM ensemble outputs on decentralized
prediction markets is a novel combination. The calibration validity result,
combined with backtesting on Polymarket's full history, is publishable.

Directly extends LLM uncertainty quantification work (cf. Dalal et al.)
into the prediction market domain.
