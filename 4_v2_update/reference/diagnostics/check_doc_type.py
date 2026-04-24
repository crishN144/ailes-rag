#!/usr/bin/env python3
"""
Verify doc_type distribution on local Qdrant.
Confirms the deep-dive finding that doc_type is only populated on statutes.
"""

import requests
from collections import Counter

URL = "http://localhost:6333"
COLLECTION = "uk_family_law_dense"

# Stream scroll through all points, counting doc_type values
next_offset = None
types = Counter()
scanned = 0

print("Scanning all points (may take 30s for 310k)...")
while True:
    payload = {"limit": 5000, "with_payload": ["doc_type"]}
    if next_offset is not None:
        payload["offset"] = next_offset
    r = requests.post(f"{URL}/collections/{COLLECTION}/points/scroll", json=payload)
    result = r.json()["result"]
    points = result.get("points", [])
    for p in points:
        dt = p["payload"].get("doc_type")
        types[dt if dt is not None else "__NULL__"] += 1
    scanned += len(points)
    next_offset = result.get("next_page_offset")
    if not next_offset:
        break

print(f"\nScanned {scanned:,} points\n")
print(f"{'doc_type':30s} {'count':>10s}  {'%':>6s}")
print("-" * 50)
for t, c in types.most_common():
    pct = 100 * c / scanned
    print(f"{t:30s} {c:>10,}  {pct:>5.1f}%")

null_pct = 100 * types.get("__NULL__", 0) / scanned
print(f"\n>>> {null_pct:.1f}% of chunks have doc_type=NULL")
if null_pct > 80:
    print(">>> CONFIRMED: doc_type is missing on most (judgment) chunks.")
    print(">>> Step 4 doc_type filter requires a payload backfill first.")
elif null_pct < 5:
    print(">>> doc_type is reliably populated. Step 4 filter can be applied directly.")
else:
    print(">>> Partial population. Audit before relying on doc_type filtering.")
