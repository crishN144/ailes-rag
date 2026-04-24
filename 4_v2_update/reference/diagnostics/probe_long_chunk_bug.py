#!/usr/bin/env python3
"""
Final diagnostic: why do the 3 remaining failing queries (FR_05, PLAIN_01,
FORM_E_01) not surface their canonical statute chunks even when the expander
injects explicit keywords?

Hypothesis: long statute chunks (mca-1973-section-25-full at 3388 chars,
mca-1973-section-25A at 1852 chars) are penalised by:
  (a) dense embedding diffusion — BGE-large averages over long text, losing
      specificity, so cosine similarity to a focused query is low
  (b) cross-encoder truncation — CE only scores first 512 tokens, cutting
      off the meat of a long section

This script probes each failing query's statute_queries at both the
retrieval layer (pre-CE) and post-CE, with an expanded pool (top_k=100),
to tell us EXACTLY where each target chunk is.

Outcome informs the final fix:
  - If target is in top-60 pre-CE but demoted below top-10 post-CE  → statute_boost + wider pool = fix
  - If target is in top-100 pre-CE but not top-60                    → widen retrieval pool
  - If target is not in top-100 at all                                → BM25/dense can't find it with these keywords → need different keywords or chunking fix
"""

import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrieval"))
from retrieval_step4 import retrieve

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

QDRANT_URL = "http://localhost:6333"
BM25_PKL = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index" / "bm25_model.pkl"

FAILING_CASES = [
    {
        "label": "FR_05 clean break",
        "user_query": "When should the court impose a clean break and terminate financial obligations between former spouses after divorce?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25A clean break",
            "matrimonial causes act 1973 section 25A termination financial obligations",
            "matrimonial causes act 1973 section 25A",  # just the section name
        ],
        "target": "mca-1973-section-25A",
        "target_len": 1852,
    },
    {
        "label": "PLAIN_01 maintenance",
        "user_query": "My husband earns £100k and I stayed home with the kids for 15 years. What am I entitled to in the divorce?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 financial provision factors",
            "matrimonial causes act 1973 section 25 matters court regard",
            "matrimonial causes act 1973 section 25",
        ],
        "target": "mca-1973-section-25-full",
        "target_len": 3388,
    },
    {
        "label": "FORM_E_01",
        "user_query": "Wife, primary carer, 15-year marriage, husband earns £180k, £850k home, business worth £2.5m, two kids.",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 financial provision factors",
            "matrimonial causes act 1973 section 25",
        ],
        "target": "mca-1973-section-25-full",
        "target_len": 3388,
    },
]


def find_rank(chunk_ids, target):
    for i, cid in enumerate(chunk_ids, 1):
        if cid == target or cid.startswith(target):
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

    for case in FAILING_CASES:
        print(f"=== {case['label']} ===")
        print(f"   target: {case['target']}   (text_len={case['target_len']} chars)")
        print()

        # For each statute_query, check where target ranks in top-100 pre-CE
        best_pre_rank = None
        for sq in case["statute_queries"]:
            results = retrieve(sq, client, embedder, token_to_idx, top_k=100)
            ids = [p.payload.get("chunk_id") or str(p.id) for p in results]
            rank = find_rank(ids, case["target"])
            marker = "✓" if rank else "✗"
            rank_str = f"#{rank}" if rank else "not in top-100"
            print(f"   statute_query: {sq!r}")
            print(f"     {marker} target at {rank_str}")
            if rank and (best_pre_rank is None or rank < best_pre_rank):
                best_pre_rank = rank

        print()
        if best_pre_rank is None:
            print(f"   ❌ target is NOT in top-100 for ANY statute_query")
            print(f"   → Retrieval cannot find this chunk with these keywords.")
            print(f"   → Fix: inspect whether BM25/dense can reach this chunk at all")
        else:
            print(f"   ✓ Best pre-CE rank across statute_queries: #{best_pre_rank}")

            # Now rerank the merged candidate pool with USER query and see CE's verdict
            all_cand = {}
            for sq in case["statute_queries"]:
                for p in retrieve(sq, client, embedder, token_to_idx, top_k=100):
                    cid = p.payload.get("chunk_id") or str(p.id)
                    all_cand.setdefault(cid, p)
            merged = list(all_cand.values())

            pairs = [[case["user_query"], (p.payload.get("text") or "")[:512]] for p in merged]
            ces = reranker.predict(pairs)
            scored = sorted(zip(merged, ces), key=lambda x: -float(x[1]))
            post_ce_ids = [p.payload.get("chunk_id") or str(p.id) for p, _ in scored]
            post_rank = find_rank(post_ce_ids, case["target"])
            if post_rank is None:
                print(f"   post-CE rank: dropped out entirely")
            else:
                print(f"   post-CE rank: #{post_rank}")
                if post_rank <= 10:
                    print(f"   ✅ IN TOP-10 after full pipeline — this query IS fixed by expander")
                elif post_rank <= 30:
                    print(f"   ⚠️  in pool but demoted to {post_rank} — statute_boost CAN fix this")
                    print(f"      (widen top_k in retrieve() to at least {post_rank+5} for safety)")
                else:
                    print(f"   ❌ demoted to {post_rank} by CE — statute_boost insufficient")
                    print(f"      → cross-encoder is actively penalising this chunk")
                    print(f"      → may need bigger boost, or chunking fix (split long chunks)")
        print()


if __name__ == "__main__":
    main()
