#!/usr/bin/env python3
"""
Diagnostic: is the canonical statute chunk even in the top-30 candidate pool
after retrieve() + cross-encoder rerank?

If YES → statute_boost can promote it. Ship the boost.
If NO  → boost is useless. Need to widen retrieval (larger top_k) or
         intervene earlier (at retrieval, not at rerank).

Runs on the 7 queries where eval showed the canonical chunk at rank > 10.
Prints the rank of the target chunk in the top-30 candidate list, both
pre-CE and post-CE.
"""

import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrieval"))
from retrieval_step4 import retrieve  # noqa

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

QDRANT_URL = "http://localhost:6333"
BM25_PKL = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index" / "bm25_model.pkl"

# Queries where the canonical statute didn't make top-10 in the eval
PROBES = [
    ("FR_04 conduct",       "When is conduct relevant in financial remedy proceedings, particularly where a party has dissipated assets or failed to make disclosure?",  ["mca-1973-section-25(2)(g)", "mca-1973-section-37"]),
    ("FR_05 clean break",   "When should the court impose a clean break and terminate financial obligations between former spouses after divorce?",  ["mca-1973-section-25A"]),
    ("FR_06 trust piercing","Can the court treat assets held in a family trust as a financial resource available to a spouse under Section 25?",  ["mca-1973-section-25-full"]),
    ("PLAIN_01 maintenance","My husband earns £100k and I stayed home with the kids for 15 years. What am I entitled to in the divorce?",  ["mca-1973-section-25-full", "mca-1973-section-23"]),
    ("PLAIN_04 DV",         "My partner is violent and I need him out of the house. What legal options do I have?",  ["fla-1996-section-33", "fla-1996-section-42", "daa-2021-section-1"]),
    ("FORM_E_01",           "Wife, primary carer, no income, 15-year marriage, husband earns £180k as company director, family home £850k with £200k mortgage, husband's business worth £2.5m, two children ages 12 and 14, wife needs housing.",  ["mca-1973-section-25-full", "mca-1973-section-23"]),
]


def find_target_ranks(chunk_ids, targets):
    """Return list of (target, rank-or-None)."""
    out = []
    for t in targets:
        rank = None
        for i, cid in enumerate(chunk_ids, 1):
            if cid == t or cid.startswith(t):
                rank = i
                break
        out.append((t, rank))
    return out


def main():
    print("Loading pipeline...")
    t0 = time.time()
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    with open(BM25_PKL, "rb") as f:
        bm25 = pickle.load(f)
    token_to_idx = {tok: idx for idx, tok in enumerate(bm25.idf.keys())}
    print(f"   ready in {time.time()-t0:.1f}s\n")

    for label, query, targets in PROBES:
        print(f"=== {label} ===")
        print(f"   target(s): {targets}")

        # Pull top-60 (wider than default 30) to see if widening would help
        candidates = retrieve(query, client, embedder, token_to_idx, top_k=60)
        pre_ce_ids = [(p.payload.get("chunk_id") or str(p.id)) for p in candidates]

        pre_ce_ranks = find_target_ranks(pre_ce_ids, targets)
        found_pre = [(t, r) for t, r in pre_ce_ranks if r is not None]

        if not found_pre:
            print(f"   ❌ NONE of the target chunks in top-60 AFTER retrieval (pre-CE)")
            print(f"      → statute_boost CANNOT help — chunks not in pool at all.")
            print(f"      → Fix: widen retrieval, or inject keyword at retrieval stage.\n")
            continue

        for t, r in found_pre:
            print(f"   ✓ {t} at rank {r} in pre-CE top-60")

        # Rerank with CE and see where they land
        pairs = [[query, (p.payload.get("text") or "")[:512]] for p in candidates]
        ce_scores = reranker.predict(pairs)
        ce_ranked = sorted(zip(candidates, ce_scores), key=lambda x: -float(x[1]))
        post_ce_ids = [(p.payload.get("chunk_id") or str(p.id)) for p, _ in ce_ranked]

        post_ce_ranks = find_target_ranks(post_ce_ids, targets)
        for (t, pre), (_, post) in zip(pre_ce_ranks, post_ce_ranks):
            if pre is None:
                continue
            # how much did CE demote it?
            delta = (post or 61) - pre
            tag = "▲ promoted" if delta < 0 else "▼ demoted" if delta > 0 else "= unchanged"
            boost_can_fix = "✅ boost-fixable" if post and post <= 30 else "❌ boost can't fix (outside top-30)"
            print(f"   {t:35s}  pre-CE #{pre:>2}  →  post-CE #{post if post else '60+':>2}  {tag}  {boost_can_fix}")
        print()


if __name__ == "__main__":
    main()
