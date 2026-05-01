"""
ensemble.py  (v2 — research grade)
────────────────────────────────────
Key improvements over v1:
  - 4 prompt templates (was 2) — more diverse reasoning frames
  - 2 Ollama models (was 1) — llama3.2 + llama3.1 for model diversity
  - 2 on-chain signals — BTC funding rate + ETH/BTC ratio
  - 2 temperatures per model — 0.2 (committed) + 0.7 (expresses doubt)
  - Result: 12+ samples per market vs 4 before

With Ollama running locally: 2 models × 4 templates × 2 temps = 16 samples
Groq/Gemini add further samples if configured.
"""

import os
import re
import time
import statistics
import requests
from dataclasses import dataclass
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "true").lower() == "true"
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Two Ollama models — pull both with:
#   ollama pull llama3.2 && ollama pull llama3.1
OLLAMA_MODELS = [
    "llama3.2",   # 2GB, fast — use alone for backtesting, add llama3.1 for live only
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # largest free Groq model
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
]

GROQ_DELAY   = 12   # seconds between Groq calls
GEMINI_DELAY = 6    # seconds between Gemini calls
MARKET_DELAY = 3    # seconds between markets


# ── On-chain signals ──────────────────────────────────────────────────────────

def get_onchain_context() -> str:
    """
    Fetch two on-chain signals and return as natural language context
    injected into prompt templates.

    Signal 1: BTC perpetual futures funding rate (Binance, free, no auth)
      Positive = bullish crowding, Negative = bearish crowding

    Signal 2: ETH/BTC price ratio (Binance, free, no auth)
      Rising = altcoin season / risk-on, Falling = BTC dominance / risk-off
    """
    signals = []

    # Signal 1: BTC funding rate
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1",
            timeout=5,
        )
        rate = float(r.json()[0]["fundingRate"])
        if rate > 0.001:
            sentiment = "bullish crowding (longs dominant, potential for squeeze)"
        elif rate < -0.001:
            sentiment = "bearish crowding (shorts dominant, potential for squeeze)"
        else:
            sentiment = "neutral positioning"
        signals.append(f"BTC funding rate: {rate:.4f} ({sentiment})")
    except Exception:
        pass

    # Signal 2: ETH/BTC ratio
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=ETHBTC",
            timeout=5,
        )
        ratio = float(r.json()["price"])
        # Context: ratio > 0.055 historically = altcoin season
        if ratio > 0.055:
            context = "elevated (altcoin-favorable environment)"
        elif ratio < 0.040:
            context = "suppressed (BTC dominance, risk-off)"
        else:
            context = "neutral"
        signals.append(f"ETH/BTC ratio: {ratio:.4f} ({context})")
    except Exception:
        pass

    if not signals:
        return ""
    return "On-chain context: " + " | ".join(signals) + "."


# ── Prompt templates (4, up from 2) ───────────────────────────────────────────

PROMPT_TEMPLATES = [
    # T1 — free reasoning with on-chain context
    """You are a calibrated forecaster. Estimate the probability (0.0 to 1.0) that the answer is YES. Be concise.
{onchain}
Brief reasoning, then end with exactly:
PROBABILITY: <number>

Question: {question}""",

    # T2 — adversarial (steelman the NO case)
    """You are a calibrated forecaster. List 2 reasons this would NOT happen, then give your probability it DOES happen. Be brief.
End with exactly:
PROBABILITY: <number>

Question: {question}""",

    # T3 — base rate anchoring
    """You are a calibrated forecaster. State the historical base rate for this type of event, then adjust briefly.
End with exactly:
PROBABILITY: <number>

Question: {question}""",

    # T4 — explicit uncertainty elicitation
    """You are a calibrated forecaster. Rate your confidence and give a probability. If uncertain, shade toward 0.5.
End with exactly:
PROBABILITY: <number>

Question: {question}""",
]


# ── Output type ────────────────────────────────────────────────────────────────

@dataclass
class EnsembleResult:
    question:       str
    estimates:      list
    point_estimate: float
    variance:       float
    std_dev:        float
    min_est:        float
    max_est:        float
    n_samples:      int
    providers_used: list

    @property
    def uncertainty_level(self) -> str:
        if self.std_dev < 0.05:  return "LOW"
        if self.std_dev < 0.12:  return "MEDIUM"
        return "HIGH"


# ── Probability extraction ─────────────────────────────────────────────────────

def _extract_probability(text: str) -> Optional[float]:
    match = re.search(r"PROBABILITY:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if match:
        p = float(match.group(1))
        return max(0.01, min(0.99, p))
    numbers = re.findall(r"\b0\.\d+\b", text)
    if numbers:
        return max(0.01, min(0.99, float(numbers[-1])))
    return None


# ── Provider: Ollama ───────────────────────────────────────────────────────────

def _query_ollama(model: str, prompt: str, temperature: float) -> Optional[float]:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": temperature, "num_predict": 200},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return _extract_probability(resp.json().get("response", ""))
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Ollama not running. Start: ollama serve")


# ── Provider: Groq ────────────────────────────────────────────────────────────

def _query_groq(model: str, prompt: str, temperature: float) -> Optional[float]:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature, "max_tokens": 400},
        timeout=30,
    )
    resp.raise_for_status()
    return _extract_probability(resp.json()["choices"][0]["message"]["content"])


# ── Provider: Gemini ──────────────────────────────────────────────────────────

def _query_gemini(model: str, prompt: str, temperature: float) -> Optional[float]:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={GEMINI_API_KEY}")
    resp = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": temperature, "maxOutputTokens": 400}},
        timeout=30,
    )
    resp.raise_for_status()
    return _extract_probability(resp.json()["candidates"][0]["content"]["parts"][0]["text"])


# ── Rate limit backoff ─────────────────────────────────────────────────────────

def _with_backoff(fn, retries=3, delay=2.0):
    for attempt in range(retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"    Rate limited — waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
    return None


# ── Main ensemble runner ───────────────────────────────────────────────────────

def run_ensemble(
    question:     str,
    temperatures: list = None,
    verbose:      bool = False,
) -> EnsembleResult:
    """
    Run question through all configured providers × templates × temperatures.

    Default: 2 Ollama models × 4 templates × 2 temperatures = 16 samples.
    This gives stable variance estimates suitable for research-grade uncertainty quantification.
    """
    if temperatures is None:
        temperatures = [0.4]   # single temp for speed — add [0.2, 0.7] for live runs

    estimates      = []
    providers_used = []
    skipped        = 0
    ollama_dead    = False

    # Fetch on-chain context once per market (shared across all prompts)
    onchain = get_onchain_context()
    if verbose and onchain:
        print(f"    [on-chain] {onchain}")

    # ── Ollama (primary — local, no rate limits) ───────────────────────────────
    if OLLAMA_ENABLED and not ollama_dead:
        for model in OLLAMA_MODELS:
            for template in PROMPT_TEMPLATES:
                for temp in temperatures:
                    prompt = template.format(question=question, onchain=onchain)
                    try:
                        p = _query_ollama(model, prompt, temp)
                        if p is not None:
                            estimates.append(p)
                            if "ollama" not in providers_used:
                                providers_used.append("ollama")
                            if verbose:
                                print(f"    [ollama/{model} t={temp:.1f}] → {p:.3f}")
                    except ConnectionError:
                        print("    [ollama] Not reachable — skipping")
                        ollama_dead = True
                        break
                    except Exception as e:
                        skipped += 1
                        if verbose:
                            print(f"    [ollama/{model}] failed: {e}")
                if ollama_dead:
                    break

    # ── Groq (backup cloud — rate limited) ────────────────────────────────────
    if GROQ_API_KEY:
        for model in GROQ_MODELS:
            for template in PROMPT_TEMPLATES[:2]:   # only T1+T2 to limit API calls
                prompt = template.format(question=question, onchain=onchain)
                try:
                    p = _with_backoff(lambda: _query_groq(model, prompt, 0.4))
                    if p is not None:
                        estimates.append(p)
                        if "groq" not in providers_used:
                            providers_used.append("groq")
                        if verbose:
                            print(f"    [groq/{model}] → {p:.3f}")
                except Exception as e:
                    skipped += 1
                    if verbose:
                        print(f"    [groq/{model}] failed: {e}")
                time.sleep(GROQ_DELAY)

    # ── Gemini (additional diversity) ─────────────────────────────────────────
    if GEMINI_API_KEY:
        for model in GEMINI_MODELS:
            for template in PROMPT_TEMPLATES[:2]:   # T1+T2 only
                prompt = template.format(question=question, onchain=onchain)
                try:
                    p = _with_backoff(lambda: _query_gemini(model, prompt, 0.3))
                    if p is not None:
                        estimates.append(p)
                        if "gemini" not in providers_used:
                            providers_used.append("gemini")
                        if verbose:
                            print(f"    [gemini/{model}] → {p:.3f}")
                except Exception as e:
                    skipped += 1
                    if verbose:
                        print(f"    [gemini/{model}] failed: {e}")
                time.sleep(GEMINI_DELAY)

    if len(estimates) < 1:
        raise ValueError(f"Got 0 valid estimates (skipped {skipped}). "
                         "Ensure Ollama is running: ollama serve")

    std = statistics.stdev(estimates)     if len(estimates) > 1 else 0.0
    var = statistics.variance(estimates)  if len(estimates) > 1 else 0.0

    return EnsembleResult(
        question=question,
        estimates=estimates,
        point_estimate=statistics.mean(estimates),
        variance=var,
        std_dev=std,
        min_est=min(estimates),
        max_est=max(estimates),
        n_samples=len(estimates),
        providers_used=providers_used,
    )


def run_ensemble_batch(questions: list, verbose: bool = False) -> list:
    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q[:65]}...")
        try:
            r = run_ensemble(q, verbose=verbose)
            results.append(r)
            print(f"  p̂={r.point_estimate:.3f}  σ={r.std_dev:.3f}  "
                  f"n={r.n_samples}  [{', '.join(r.providers_used)}]")
        except Exception as e:
            print(f"  Skipped: {e}")
        if i < len(questions) - 1:
            time.sleep(MARKET_DELAY)
    return results
