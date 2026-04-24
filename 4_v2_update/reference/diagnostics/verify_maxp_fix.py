#!/usr/bin/env python3
"""
Verify MaxP reranking fixes the long-chunk failure mode on real queries.

Compares single-pass CE vs MaxP CE on the 5 golden queries where the
canonical statute chunk is long (>1500 chars):

  * FR_05 (s.25A)       — canonical 1852 chars → fits in single window,
                           included as control (MaxP shouldn't hurt)
  * FR_06 (s.25-full)   — canonical 3388 chars → expected to benefit
  * PLAIN_01 (s.25-full)— canonical 3388 chars → expected to benefit
  * FORM_E_01 (s.25-full)— canonical 3388 chars → expected to benefit
  * FR_07 (s.25-full)   — canonical 3388 chars → expected to benefit

For each query:
  1. Run retrieve() to get the candidate pool (same for both methods)
  2. Rerank with single-pass CE (current behaviour)
  3. Rerank with MaxP CE
  4. Report rank of canonical chunk in each

This is the empirical check before shipping MaxP to production.
"""

import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrieval"))
from retrieval_step4 import retrieve
from maxp_reranker import rerank_with_maxp

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

QDRANT_URL = "http://localhost:6333"
BM25_PKL = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index" / "bm25_model.pkl"

# The failing queries, with the statute_queries the new expander would emit
# (so retrieval has a fair chance of getting the chunk into the pool).
CASES = [
    {
        "label": "FR_06 trust_piercing",
        "user_query": "Can the court treat assets held in a family trust as a financial resource available to a spouse under Section 25?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 financial resources trust",
            "matrimonial causes act 1973 section 25(2)(a) financial resources",
        ],
        "target": "mca-1973-section-25-full",
        "target_len": 3388,
    },
    {
        "label": "PLAIN_01 maintenance",
        "user_query": "My husband earns £100k and I stayed home with the kids for 15 years. What am I entitled to in the divorce?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 matters court regard",
            "matrimonial causes act 1973 section 25(2)(f) contributions homemaker",
            "matrimonial causes act 1973 section 23 financial provision orders",
        ],
        "target": "mca-1973-section-25-full",
        "target_len": 3388,
    },
    {
        "label": "FORM_E_01",
        "user_query": "Wife, primary carer, no income, 15-year marriage, husband earns £180k as company director, family home £850k with £200k mortgage, husband's business worth £2.5m, two children ages 12 and 14, wife needs housing.",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 financial provision factors",
            "matrimonial causes act 1973 section 23 financial provision orders",
        ],
        "target": "mca-1973-section-25-full",
        "target_len": 3388,
    },
    {
        "label": "FR_07 inherited_assets",
        "user_query": "How should inherited or pre-marital non-matrimonial assets be treated in financial remedy proceedings?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 financial resources inheritance",
            "matrimonial causes act 1973 section 25(2)(a) property financial resources",
        ],
        "target": "mca-1973-section-25-full",
        "target_len": 3388,
    },
    {
        "label": "FR_05 clean_break (control)",
        "user_query": "When should the court impose a clean break and terminate financial obligations between former spouses after divorce?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25A clean break",
            "matrimonial causes act 1973 section 25A",
        ],
        "target": "mca-1973-section-25A",
        "target_len": 1852,
    },
]


def find_rank(chunks_or_ids, target):
    for i, c in enumerate(chunks_or_ids, 1):
        cid = c if isinstance(c, str) else c.get("chunk_id")
        if cid == target or (cid and cid.startswith(target)):
            return i
    return None


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

    wins = 0
    regressions = 0
    same = 0

    for case in CASES:
        print(f"=== {case['label']} ===")
        print(f"   target: {case['target']} ({case['target_len']} chars)")

        # Build candidate pool using the expander's statute_queries + user query
        all_cand = {}
        for sq in case["statute_queries"] + [case["user_query"]]:
            results = retrieve(sq, client, embedder, token_to_idx, top_k=60)
            for p in results:
                cid = p.payload.get("chunk_id") or str(p.id)
                if cid not in all_cand:
                    all_cand[cid] = p

        candidates = list(all_cand.values())
        print(f"   pool size after expander-merge: {len(candidates)}")

        # Check if target is even in the pool
        pool_ids = [p.payload.get("chunk_id") or str(p.id) for p in candidates]
        pool_rank = find_rank(pool_ids, case["target"])
        if pool_rank is None:
            print(f"   ❌ target not in pool — neither method can help")
            print()
            continue
        print(f"   target at pool rank #{pool_rank} (pre-CE)")

        # A) Single-pass CE rerank (current behaviour)
        t0 = time.time()
        pairs = [[case["user_query"], (p.payload.get("text") or "")[:512]]
                 for p in candidates]
        ce_scores = reranker.predict(pairs)
        sp_ranked = sorted(zip(candidates, ce_scores), key=lambda x: -float(x[1]))
        sp_ids = [p.payload.get("chunk_id") or str(p.id) for p, _ in sp_ranked]
        sp_time = time.time() - t0
        sp_rank = find_rank(sp_ids, case["target"])

        # B) MaxP CE rerank
        t0 = time.time()
        maxp_ranked = rerank_with_maxp(candidates, case["user_query"], reranker)
        maxp_time = time.time() - t0
        mp_rank = find_rank(maxp_ranked, case["target"])

        sp_str = f"#{sp_rank}" if sp_rank else "miss"
        mp_str = f"#{mp_rank}" if mp_rank else "miss"
        print(f"   single_pass: target at {sp_str:>6s}   time {sp_time:.2f}s")
        print(f"   maxp:        target at {mp_str:>6s}   time {maxp_time:.2f}s")

        sp_n = sp_rank if sp_rank else 999
        mp_n = mp_rank if mp_rank else 999
        if mp_n < sp_n:
            delta = sp_n - mp_n
            print(f"   ✅ MaxP improves rank by {delta} positions")
            wins += 1
        elif mp_n > sp_n:
            print(f"   ⚠️  MaxP regresses rank by {mp_n - sp_n}")
            regressions += 1
        else:
            print(f"   = MaxP no change")
            same += 1
        print()

    print("=" * 70)
    print(f"SUMMARY: {wins} wins, {regressions} regressions, {same} ties out of {len(CASES)}")
    print()
    if wins >= 3:
        print("VERDICT: MaxP is the right fix. Ship it.")
        print("        Integrate rerank_with_maxp() as the rerank step in production")
        print("        — replaces the single-pass CE call, no other changes.")
    elif wins > regressions:
        print("VERDICT: MaxP helps. Ship behind a flag, tune window size.")
    else:
        print("VERDICT: MaxP doesn't help on real retrieval output. Investigate.")


if __name__ == "__main__":
    main()
