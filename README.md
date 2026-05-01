# Uncertainty Agent
### Epistemic Uncertainty Quantification for Prediction Markets

> *Prediction markets tell you the crowd's probability estimate. They don't tell you how confident that estimate is. This agent surfaces the gap.*

**Live demo → [gourimenon8.github.io/uncertainty-agent-prototype](https://gourimenon8.github.io/uncertainty-agent-prototype/index.html)**

---

## What it does

Every prediction market gives one number — "62% chance Barcelona wins La Liga." But that collapses two very different situations:

- A **confident 62%** — models agree, strong signals, liquid market
- An **uncertain 62%** — models disagree, thin signals, noisy evidence

This agent distinguishes them. It runs an ensemble of LLMs across multiple prompt framings, measures their disagreement as epistemic uncertainty, and wraps the result in **conformal prediction intervals** — giving distribution-free coverage guarantees. When the market is highly confident but the model ensemble is not, that gap is a signal.

---

## Results

| Metric | Value |
|--------|-------|
| Markets monitored | 13 live crypto + sports markets |
| Sweeps collected | 169 (Apr 27 – May 1, 2026) |
| Prospective hit rate | **2/2** (MegaETH FDV markets) |
| Fade ROI | **+13.7%** mean per resolved bet |
| Conformal coverage | **93%** empirical (target 80%) |
| Calibration markets | 318 resolved Polymarket markets |

### Prospective validation
Predictions made before outcomes were known:

| Market | Agent | Market | Outcome |
|--------|-------|--------|---------|
| MegaETH FDV >$2B at launch | 15% | 26% | **NO ✓** |
| MegaETH FDV >$6B at launch | 7% | 1.3% | **NO ✓** |
| MegaETH airdrop by June 30 | 22% | 62.5% | Pending |

### Short-horizon predictions (closing May 15–31, 2026)

| Market | Agent | Market | Closes |
|--------|-------|--------|--------|
| Barcelona wins La Liga | 62% | **95%** | May 30 |
| Atletico Madrid wins UCL | 41% | 9% | May 31 |
| Man City wins Premier League | **72%** | 57% | May 27 |
| Iran peace deal by May 31 | 37% | 9% | May 31 |
| Ken Paxton wins TX primary | 27% | 57% | May 26 |

---

## How it works

```
Polymarket API + BTC funding rate + ETH/BTC ratio
                    ↓
     LLM Ensemble (4 prompts × 2 models × 2 temperatures)
                    ↓  variance = epistemic uncertainty
     Conformal Calibration  →  [p_lo, p_hi] with 80% coverage
                    ↓
     Mismatch Score  =  market confidence − model confidence
                    ↓
     Kelly Position Sizing  (uncertainty-adjusted)
```

**`models/ensemble.py`** — Four prompt templates (free reasoning, adversarial steelmanning, base rate anchoring, uncertainty elicitation) × 2 models × 2 temperatures. Ensemble variance = epistemic uncertainty. Injects BTC funding rate and ETH/BTC ratio as on-chain context.

**`models/conformal.py`** — Split conformal prediction ([Angelopoulos & Bates 2022](https://arxiv.org/abs/2107.07511)). Distribution-free intervals with provable marginal coverage. No Gaussian assumptions. Calibrated on 318 resolved markets.

**`models/mispricing.py`** — Mismatch score = market-implied confidence − model confidence. Kelly fraction scales proportionally to CI width — distinguishes a confident 60% from an uncertain 60%.

---

## Research contribution

The combination of **conformal prediction intervals applied to LLM ensemble outputs on decentralized prediction markets** is a novel methodological contribution:

1. LLM ensemble disagreement is a valid epistemic uncertainty proxy for binary market outcomes
2. Split conformal calibration provides distribution-free coverage guarantees on this signal
3. The mismatch score identifies systematically overconfident markets
4. Uncertainty-adjusted Kelly criterion produces better position sizing than standard Kelly

Extends LLM uncertainty quantification research into the prediction market domain.

---

## Quick start

```bash
git clone https://github.com/gourimenon8/uncertainty-agent-prototype.git
cd uncertainty-agent-prototype
pip install -r requirements.txt

# Free API keys — no credit card needed
# Ollama: https://ollama.com  →  ollama pull llama3.2
# Groq:   https://console.groq.com
# Gemini: https://aistudio.google.com

cp .env.template .env   # add your keys

python agent.py --calibrate   # build calibration set (once)
python agent.py --run         # start live monitoring loop
```

### Update the dashboard

```bash
python export_dashboard.py
git add data.json && git commit -m "update" && git push
```

---

## Project structure

```
├── agent.py                ← main loop: --calibrate | --run | --once
├── export_dashboard.py     ← generates dashboard data
├── fast_backtest.py        ← validates signal against resolved markets
├── short_horizon.py        ← finds markets resolving within N days
├── snapshot.py             ← saves timestamped predictions
├── data/
│   └── polymarket.py       ← Polymarket API
├── models/
│   ├── ensemble.py         ← LLM ensemble + on-chain context
│   ├── conformal.py        ← split conformal calibration
│   └── mispricing.py       ← mismatch scoring + Kelly sizing
└── frontend/
    ├── index.html          ← live dashboard
    └── data.json           ← generated by export_dashboard.py
```

---

## Stack

Local inference · [Ollama](https://ollama.com) (llama3.2) ·
[Groq](https://console.groq.com) (llama-3.3-70b) ·
[Gemini](https://aistudio.google.com) (gemini-2.5-flash) ·
[Polymarket](https://polymarket.com) · [Binance](https://binance.com)

---

**Gouri Menon** · Columbia University · 2026  
LLM Uncertainty Quantification
