#!/usr/bin/env python3
"""
Verify that the new expander prompt's doctrine→statute injection fixes
the 5/6 queries where statute_boost failed because the canonical chunk
was not in the candidate pool.

Logic: simulate what the new expander would emit for each failing query
(hand-crafted statute_queries matching the expander's injection table),
run each statute_query through retrieve() separately, merge results,
and check whether the canonical statute chunks now surface in top-10.

If YES → the expander prompt IS the fix. statute_boost is unnecessary
         (or optional belt-and-braces).
If NO  → something else is broken at the retrieval layer itself. Need
         to investigate whether the statute chunks are reachable at all
         by their canonical keywords.
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

# Simulated new-expander output — what Gemma 3 4B with the new prompt
# would emit for each query. Each list is the statute_queries field.
SIMULATED_EXPANDER = {
    "FR_05 clean break": {
        "user_query": "When should the court impose a clean break and terminate financial obligations between former spouses after divorce?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25A clean break",
            "matrimonial causes act 1973 section 25A termination financial obligations",
            "matrimonial causes act 1973 section 31 periodical payments clean break",
        ],
        "canonical_targets": ["mca-1973-section-25A"],
    },
    "FR_06 trust piercing": {
        "user_query": "Can the court treat assets held in a family trust as a financial resource available to a spouse under Section 25?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 financial resources trust",
            "matrimonial causes act 1973 section 25(2)(a) financial resources",
            "matrimonial causes act 1973 section 24 property adjustment trust",
        ],
        "canonical_targets": ["mca-1973-section-25-full", "mca-1973-section-25(2)(a)", "mca-1973-section-24"],
    },
    "PLAIN_01 maintenance": {
        "user_query": "My husband earns £100k and I stayed home with the kids for 15 years. What am I entitled to in the divorce?",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 financial provision factors",
            "matrimonial causes act 1973 section 25(2)(f) contributions homemaker",
            "matrimonial causes act 1973 section 23 periodical payments orders",
        ],
        "canonical_targets": ["mca-1973-section-25-full", "mca-1973-section-25(2)(f)", "mca-1973-section-23"],
    },
    "PLAIN_04 DV": {
        "user_query": "My partner is violent and I need him out of the house. What legal options do I have?",
        "statute_queries": [
            "family law act 1996 section 33 occupation order",
            "family law act 1996 section 42 non-molestation order",
            "domestic abuse act 2021 section 1 definition domestic abuse",
        ],
        "canonical_targets": ["fla-1996-section-33", "fla-1996-section-42", "daa-2021-section-1"],
    },
    "FORM_E_01": {
        "user_query": "Wife, primary carer, no income, 15-year marriage, husband earns £180k as company director, family home £850k with £200k mortgage, husband's business worth £2.5m, two children ages 12 and 14, wife needs housing.",
        "statute_queries": [
            "matrimonial causes act 1973 section 25 financial provision factors",
            "matrimonial causes act 1973 section 23 financial provision orders",
            "matrimonial causes act 1973 section 24 property adjustment order",
        ],
        "canonical_targets": ["mca-1973-section-25-full", "mca-1973-section-23", "mca-1973-section-24"],
    },
}


def find_rank(chunk_ids, targets):
    for i, cid in enumerate(chunk_ids, 1):
        if any(cid == t or cid.startswith(t) for t in targets):
            return i, cid
    return None, None


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

    successes = 0
    for label, spec in SIMULATED_EXPANDER.items():
        print(f"=== {label} ===")
        print(f"   canonical targets: {spec['canonical_targets']}")
        print()

        # BASELINE — current behaviour: run user query directly
        baseline = retrieve(spec["user_query"], client, embedder, token_to_idx, top_k=30)
        pairs = [[spec["user_query"], (p.payload.get("text") or "")[:512]] for p in baseline]
        ce = reranker.predict(pairs)
        baseline_top = sorted(zip(baseline, ce), key=lambda x: -float(x[1]))[:10]
        baseline_ids = [(p.payload.get("chunk_id") or str(p.id)) for p, _ in baseline_top]
        b_rank, b_cid = find_rank(baseline_ids, spec["canonical_targets"])

        # NEW — run each statute_query separately, merge, rerank with user query
        all_candidates = {}
        for sq in spec["statute_queries"]:
            results = retrieve(sq, client, embedder, token_to_idx, top_k=15)
            for p in results:
                cid = p.payload.get("chunk_id") or str(p.id)
                if cid not in all_candidates:
                    all_candidates[cid] = p

        merged = list(all_candidates.values())
        # Rerank by the USER query (not the statute query) — asking "which of
        # these chunks best answers the user's actual question"
        pairs = [[spec["user_query"], (p.payload.get("text") or "")[:512]] for p in merged]
        ce = reranker.predict(pairs)
        new_top = sorted(zip(merged, ce), key=lambda x: -float(x[1]))[:10]
        new_ids = [(p.payload.get("chunk_id") or str(p.id)) for p, _ in new_top]
        n_rank, n_cid = find_rank(new_ids, spec["canonical_targets"])

        # Report
        b_str = f"rank {b_rank} ({b_cid})" if b_rank else "NOT in top-10"
        n_str = f"rank {n_rank} ({n_cid})" if n_rank else "NOT in top-10"

        print(f"   BASELINE (current pipeline):        {b_str}")
        print(f"   WITH NEW EXPANDER (simulated):     {n_str}")

        if n_rank and not b_rank:
            print(f"   ✅ FIXED by expander alone — statute_boost unnecessary")
            successes += 1
        elif n_rank and b_rank and n_rank < b_rank:
            print(f"   ✅ IMPROVED from {b_rank} to {n_rank}")
            successes += 1
        elif not n_rank:
            print(f"   ❌ NOT fixed by expander — deeper retrieval issue")
        else:
            print(f"   = same")
        print()

    print("=" * 80)
    print(f"Summary: expander simulation fixed {successes} / {len(SIMULATED_EXPANDER)} queries")
    print()
    if successes == len(SIMULATED_EXPANDER):
        print("VERDICT: the new rag.expander prompt IS the fix.")
        print("         statute_boost.py becomes optional (belt-and-braces).")
        print("         SHIP THE EXPANDER. That's the whole win.")
    elif successes >= len(SIMULATED_EXPANDER) * 0.6:
        print("VERDICT: expander fixes most; residual cases need investigation.")
        print("         Ship expander AND keep statute_boost for the residual.")
    else:
        print("VERDICT: expander doesn't fully fix retrieval — investigate further.")
        print("         Likely need query decomposition or broader retrieval changes.")


if __name__ == "__main__":
    main()
