"""
smoke_test.py
─────────────
Run this first. Verifies all dependencies and connections before
running the full agent. Costs nothing.

    python smoke_test.py
"""

import sys
import os

print("=" * 55)
print("Smoke test — uncertainty agent (free tier)")
print("=" * 55)

# ── 1. Python version ──────────────────────────────────────────
v = sys.version_info
assert v.major == 3 and v.minor >= 9, f"Need Python 3.9+, got {v.major}.{v.minor}"
print(f"  [OK] Python {v.major}.{v.minor}.{v.micro}")

# ── 2. Load .env ───────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("  [OK] dotenv loaded")
except ImportError:
    print("  [FAIL] python-dotenv not installed")
    print("         Run: pip install -r requirements.txt")
    sys.exit(1)

# ── 3. Check library imports ───────────────────────────────────
for lib in ["numpy", "requests"]:
    try:
        __import__(lib)
        print(f"  [OK] {lib}")
    except ImportError:
        print(f"  [FAIL] {lib} — run: pip install -r requirements.txt")
        sys.exit(1)

# ── 4. Check at least one provider is configured ──────────────
import requests

ollama_ok  = False
groq_ok    = False
gemini_ok  = False

OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "true").lower() == "true"
OLLAMA_BASE    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_KEY       = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY     = os.environ.get("GEMINI_API_KEY", "")

# Test Ollama
if OLLAMA_ENABLED:
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            print(f"  [OK] Ollama running — models: {', '.join(models[:3])}")
            ollama_ok = True
        else:
            print("  [WARN] Ollama running but no models pulled")
            print("         Run: ollama pull llama3.1")
    except Exception:
        print("  [WARN] Ollama not reachable")
        print("         Install: https://ollama.com then: ollama serve")

# Test Groq
if GROQ_KEY:
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            timeout=8,
        )
        resp.raise_for_status()
        print(f"  [OK] Groq API key valid")
        groq_ok = True
    except Exception as e:
        print(f"  [FAIL] Groq API key invalid: {e}")
else:
    print("  [--] Groq not configured (optional)")

# Test Gemini
if GEMINI_KEY:
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}",
            timeout=8,
        )
        resp.raise_for_status()
        print(f"  [OK] Gemini API key valid")
        gemini_ok = True
    except Exception as e:
        print(f"  [FAIL] Gemini API key invalid: {e}")
else:
    print("  [--] Gemini not configured (optional)")

if not any([ollama_ok, groq_ok, gemini_ok]):
    print()
    print("  [FAIL] No providers available. Set up at least one:")
    print("    Ollama:  brew install ollama && ollama pull llama3.1")
    print("    Groq:    get free key at console.groq.com, add to .env")
    print("    Gemini:  get free key at aistudio.google.com, add to .env")
    sys.exit(1)

# ── 5. Test Polymarket API ─────────────────────────────────────
try:
    resp = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={"active": "true", "closed": "false", "tag": "crypto", "limit": 2},
        timeout=8,
    )
    resp.raise_for_status()
    markets = resp.json()
    print(f"  [OK] Polymarket API — {len(markets)} markets fetched")
    if markets:
        print(f"       Sample: {markets[0].get('question','')[:55]}...")
except Exception as e:
    print(f"  [FAIL] Polymarket API: {e}")
    sys.exit(1)

# ── 6. Quick one-question ensemble test ───────────────────────
print()
print("  Running one test question through ensemble...")
try:
    from models.ensemble import run_ensemble
    result = run_ensemble(
        "Will Bitcoin be above $50,000 at the end of this month?",
        temperatures=[0.3],
        verbose=True,
    )
    print(f"  [OK] Ensemble worked!")
    print(f"       p̂={result.point_estimate:.3f}  σ={result.std_dev:.3f}")
    print(f"       n={result.n_samples} samples from {result.providers_used}")
except Exception as e:
    print(f"  [FAIL] Ensemble error: {e}")
    sys.exit(1)

# ── 7. Conformal sanity check ─────────────────────────────────
import numpy as np
from models.conformal import fit_calibration, predict_interval

fake_p   = list(np.random.uniform(0.2, 0.8, 80))
fake_y   = [1 if p > 0.5 else 0 for p in fake_p]
calib    = fit_calibration(fake_p, fake_y)
iv       = predict_interval(0.65, calib, alpha=0.10)
print(f"  [OK] Conformal — test interval: [{iv.lower:.3f}, {iv.upper:.3f}]")

# ── Done ───────────────────────────────────────────────────────
print()
print("=" * 55)
print("  All checks passed.")
print()
print("  Next steps:")
print("  python agent.py --calibrate   # build calibration set")
print("  python agent.py --once        # single live sweep")
print("  python agent.py --run         # start live loop")
print("  python backtest.py            # run full backtest")
print("=" * 55)
