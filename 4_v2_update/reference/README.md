# Reference material — NOT for production

Nothing in this folder ships. It exists so the v2 design decisions are reproducible.

## `diagnostics/`

One-shot probe scripts I ran locally to prove where the v1 retrieval bugs lived. Each script answers a single question; together they justified the 5 changes that DO ship.

| Script | Question it answered |
|---|---|
| `compare_rerankers.py` | Is BGE-reranker-large worth the 8× latency vs MS-MARCO? (No — kept MS-MARCO.) |
| `check_candidate_pool.py` | Are the failing chunks even in the top-60 pool? (No — boost can't help; need expander fix.) |
| `verify_expander_fix.py` | Does the new expander prompt surface the missing statute chunks? (Yes for 2/5.) |
| `probe_long_chunk_bug.py` | Why does the cross-encoder demote `-full` statute chunks? (512-token truncation.) |
| `check_doc_type.py` | Is `doc_type` populated on judgment chunks? (No — null on 303k.) |
| `check_chunk_id_collisions.py` | How many chunk_id collisions in the corpus? (40k same-text, 115 different-text.) |
| `check_bge_prefix.py` / `benchmark_with_prefix.py` | Do BGE query/passage prefixes change retrieval? (Marginal.) |
| `verify_maxp_fix.py` | Does MaxP reranking rescue long statute chunks? (Yes but 10× latency, marginal quality — not shipping.) |

Re-run any of these against your Qdrant if you want to reproduce a specific finding.

## `experimental_archived/`

Two modules I built and then disqualified after testing. Kept here for posterity so future-you doesn't waste a week rediscovering they don't work.

- **`statute_boost.py`** — post-rerank metadata boost. Empirically can't help chunks that the cross-encoder pushed out of the top-30 candidate pool. Out-of-pool chunks stay out.
- **`maxp_reranker.py`** — Maximum-Passage reranking for long chunks. Improves the ranking but the target chunks still don't cross top-10, and latency goes 10× for marginal quality. Not worth it.

The real fix for long statute chunks is at the chunker layer (drop `-full` parents, keep per-subsection chunks), not at the rerank layer. Flagged in the v2 banner as known limitations.
