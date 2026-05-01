"""
debug_api.py
────────────
Run this first to see what Polymarket actually returns.
It tests different param combinations and prints what works.

    python debug_api.py
"""

import requests, json

GAMMA = "https://gamma-api.polymarket.com"

print("Testing Polymarket Gamma API...\n")

# ── Test 1: what does a raw closed=true response look like? ──────────────────
print("Test 1: closed=true, limit=3")
r = requests.get(f"{GAMMA}/markets", params={"closed": "true", "limit": 3}, timeout=10)
data = r.json()
print(f"  Status: {r.status_code}  |  Records: {len(data) if isinstance(data, list) else type(data)}")
if isinstance(data, list) and data:
    m = data[0]
    print(f"  Keys: {list(m.keys())}")
    print(f"  question:    {m.get('question','')[:60]}")
    print(f"  resolution:  {m.get('resolution')}")
    print(f"  closed:      {m.get('closed')}")
    print(f"  active:      {m.get('active')}")
    print(f"  tags:        {m.get('tags', [])[:3]}")
    print(f"  outcomePrices: {m.get('outcomePrices')}")
print()

# ── Test 2: active markets with crypto tag ────────────────────────────────────
print("Test 2: active=true, tag=crypto, limit=3")
r = requests.get(f"{GAMMA}/markets",
    params={"active": "true", "closed": "false", "tag": "crypto", "limit": 3}, timeout=10)
data = r.json()
print(f"  Status: {r.status_code}  |  Records: {len(data) if isinstance(data, list) else 'err'}")
if isinstance(data, list) and data:
    m = data[0]
    print(f"  question: {m.get('question','')[:60]}")
    print(f"  tags: {m.get('tags', [])[:3]}")
print()

# ── Test 3: no tag filter, closed=true ────────────────────────────────────────
print("Test 3: closed=true, limit=5 (no tag filter)")
r = requests.get(f"{GAMMA}/markets",
    params={"closed": "true", "limit": 5}, timeout=10)
data = r.json()
print(f"  Status: {r.status_code}  |  Records: {len(data) if isinstance(data, list) else 'err'}")
if isinstance(data, list) and data:
    for m in data[:3]:
        res = m.get("resolution")
        tags = m.get("tags", [])
        q = m.get("question", "")[:55]
        print(f"  [{res}] tags={tags} | {q}")
print()

# ── Test 4: check what tags actually look like ────────────────────────────────
print("Test 4: active=true, limit=10 — inspecting tag format")
r = requests.get(f"{GAMMA}/markets",
    params={"active": "true", "limit": 10}, timeout=10)
data = r.json()
if isinstance(data, list) and data:
    tag_formats = set()
    for m in data:
        tags = m.get("tags", [])
        for t in tags:
            tag_formats.add(str(t)[:60])
    print(f"  Tag samples: {list(tag_formats)[:8]}")
print()

# ── Test 5: what does outcomePrices look like on resolved markets? ────────────
print("Test 5: looking at outcomePrices format on resolved markets")
r = requests.get(f"{GAMMA}/markets", params={"closed": "true", "limit": 10}, timeout=10)
data = r.json()
if isinstance(data, list):
    for m in data[:5]:
        op = m.get("outcomePrices")
        res = m.get("resolution")
        print(f"  resolution={res!r:5}  outcomePrices={str(op)[:50]}")

print("\nTest 6: inspect actual closed market fields (questions + prices)")
r = requests.get(f"{GAMMA}/markets", params={"closed": "true", "limit": 20, "offset": 0}, timeout=10)
data = r.json()
if isinstance(data, list):
    for m in data[:20]:
        q   = m.get("question", "")[:60]
        ltp = m.get("lastTradePrice")
        op  = m.get("outcomePrices")
        bid = m.get("bestBid")
        ask = m.get("bestAsk")
        cat = m.get("category", "")
        print(f"  q={q}")
        print(f"    category={cat!r}  lastTradePrice={ltp}  bestBid={bid}  bestAsk={ask}  outcomePrices={str(op)[:40]}")

print("\nTest 7: what categories exist?")
r = requests.get(f"{GAMMA}/markets", params={"closed": "true", "limit": 200}, timeout=10)
data = r.json()
if isinstance(data, list):
    from collections import Counter
    cats = Counter(m.get("category","(none)") for m in data)
    for cat, count in cats.most_common(15):
        print(f"  {count:4d}x  {cat!r}")

print("\nDone.")
