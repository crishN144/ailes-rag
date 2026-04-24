#!/usr/bin/env python3
"""
run_benchmark_v2.py — deterministic CI gate for AILES retrieval.

Runs the Step-4 + statute-boost pipeline on every query in
golden_queries_v2.json and computes four metrics:

    1. Strict Hit@10       — any chunk from (canonical ∪ acceptable) in top-10
    2. Coverage score      — for queries with coverage blocks, fraction of
                              blocks satisfied in top-10
    3. Forbidden hits       — count of chunks matching must_not_retrieve_prefixes
    4. Best canonical rank  — rank of the best canonical chunk (11 if none)

No LLM calls. Runs in ~3-5 minutes for 20 queries.

Pass/fail gate:
    PASS  ⟺  mean Hit@10 ≥ --strict-threshold (default 0.85)
             AND total forbidden hits ≤ --forbidden-budget (default 2)

CLI:
    python3 run_benchmark_v2.py
    python3 run_benchmark_v2.py --queries golden_queries_v2.json \\
                                --strict-threshold 0.85 \\
                                --forbidden-budget 2 \\
                                --disable-statute-boost
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

# ─── Path setup — import from neighbouring retrieval/ dir ───────────────
HERE = Path(__file__).resolve().parent
RETRIEVAL_DIR = HERE.parent / "retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from retrieval_step4 import retrieve                           # noqa: E402
from statute_boost import apply_statute_boost                  # noqa: E402

from qdrant_client import QdrantClient                         # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402


# ─── Defaults (override via CLI) ────────────────────────────────────────
QDRANT_URL = "http://localhost:6333"
COLLECTION = "uk_family_law_dense"
BM25_PKL = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index" / "bm25_model.pkl"
DEFAULT_QUERIES_FILE = HERE / "golden_queries_v2.json"
DEFAULT_OUTPUT = Path("/tmp/benchmark_v2_results.json")
TOP_K = 10


# ─── Scoring primitives ─────────────────────────────────────────────────
def _matches_any(chunk_id: str, target_ids: list[str]) -> bool:
    """Prefix / exact match against a list of target chunk IDs."""
    return any(chunk_id == t or chunk_id.startswith(t) for t in target_ids)


def _matches_forbidden(chunk_id: str, forbidden_prefixes: list[str]) -> bool:
    return any(chunk_id.startswith(p) for p in forbidden_prefixes)


def score_query(retrieved_ids: list[str], query_spec: dict) -> dict:
    canonical = query_spec.get("canonical_chunks", []) or []
    acceptable = query_spec.get("acceptable_chunks", []) or []
    forbidden = query_spec.get("must_not_retrieve_prefixes", []) or []
    coverage_blocks = query_spec.get("coverage") or []

    accept_set = canonical + acceptable

    # 1. Strict Hit@10
    hit = any(_matches_any(cid, accept_set) for cid in retrieved_ids[:10])

    # 2. Coverage score
    coverage_score = None
    coverage_detail = []
    if coverage_blocks:
        satisfied = []
        for block in coverage_blocks:
            any_of = block.get("any_of", [])
            got = any(_matches_any(cid, any_of) for cid in retrieved_ids[:10])
            satisfied.append(got)
            coverage_detail.append({"name": block["name"], "satisfied": got})
        coverage_score = sum(satisfied) / len(satisfied) if satisfied else None

    # 3. Forbidden hits
    forbidden_hits = sum(
        1 for cid in retrieved_ids[:10] if _matches_forbidden(cid, forbidden)
    )

    # 4. Best canonical rank
    best_rank = 11
    for i, cid in enumerate(retrieved_ids[:10], start=1):
        if _matches_any(cid, canonical):
            best_rank = i
            break

    return {
        "hit_at_10": bool(hit),
        "coverage_score": coverage_score,
        "coverage_detail": coverage_detail,
        "forbidden_hits": forbidden_hits,
        "best_canonical_rank": best_rank,
    }


# ─── Retrieval wrapper (with optional statute boost + CE rerank) ────────
def run_query(
    query_text: str,
    preserved_terms: list[str] | None,
    client, embedder, reranker, token_to_idx,
    disable_statute_boost: bool = False,
) -> list[str]:
    """Returns chunk_ids in rank order (top 10)."""
    candidates = retrieve(query_text, client, embedder, token_to_idx, top_k=30)
    if not candidates:
        return []

    pairs = [[query_text, (p.payload.get("text") or "")[:512]] for p in candidates]
    ce_scores = reranker.predict(pairs)

    # Normalize to dict form that statute_boost expects
    as_dicts = []
    for pt, ce in zip(candidates, ce_scores):
        d = dict(pt.payload or {})
        d["chunk_id"] = d.get("chunk_id") or str(pt.id)
        d["ce_score"] = float(ce)
        as_dicts.append(d)
    as_dicts.sort(key=lambda x: -x["ce_score"])

    if not disable_statute_boost and preserved_terms:
        as_dicts = apply_statute_boost(as_dicts, preserved_terms)

    return [c["chunk_id"] for c in as_dicts[:10]]


# ─── Main ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default=str(DEFAULT_QUERIES_FILE),
                   help=f"Path to golden_queries_v2.json (default: {DEFAULT_QUERIES_FILE})")
    p.add_argument("--strict-threshold", type=float, default=0.85,
                   help="Mean Hit@10 required to pass (default: 0.85)")
    p.add_argument("--forbidden-budget", type=int, default=2,
                   help="Max total forbidden hits allowed (default: 2)")
    p.add_argument("--disable-statute-boost", action="store_true",
                   help="Run without the post-rerank statute boost (A/B isolation)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"ERROR: queries file not found at {queries_path}")
        print("       (the agent should have produced golden_queries_v2.json by now)")
        sys.exit(2)

    queries = json.loads(queries_path.read_text())
    # Allow either top-level list or {"queries": [...]} shape
    if isinstance(queries, dict):
        queries = queries.get("queries", queries.get("per_query", []))
    print(f"Loaded {len(queries)} queries from {queries_path.name}\n")

    print("Loading pipeline (BGE + cross-encoder + BM25 + Qdrant)...")
    t0 = time.time()
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    with open(BM25_PKL, "rb") as f:
        bm25 = pickle.load(f)
    token_to_idx = {tok: idx for idx, tok in enumerate(bm25.idf.keys())}
    print(f"   ready in {time.time()-t0:.1f}s\n")

    if args.disable_statute_boost:
        print("⚠️  statute_boost DISABLED for this run (A/B isolation mode)\n")

    # ─── Per-query execution ────────────────────────────────────────────
    results = []
    hits, forbidden_total = 0, 0
    coverage_scores, best_ranks = [], []

    print(f"{'query_id':28s} {'Hit@10':>6s} {'Cov':>5s} {'Forb':>5s} {'Rank':>5s}")
    print("-" * 60)

    for q in queries:
        qid = q["id"]
        preserved = q.get("preserved_terms")  # optional; derived from expander in real pipeline
        # If the query doesn't carry preserved_terms, fall back to heuristic from canonical chunks
        if not preserved:
            preserved = _infer_preserved_terms(q)

        retrieved_ids = run_query(
            q["query"], preserved, client, embedder, reranker, token_to_idx,
            disable_statute_boost=args.disable_statute_boost,
        )
        s = score_query(retrieved_ids, q)

        hits += int(s["hit_at_10"])
        forbidden_total += s["forbidden_hits"]
        if s["coverage_score"] is not None:
            coverage_scores.append(s["coverage_score"])
        best_ranks.append(s["best_canonical_rank"])

        cov_str = f"{s['coverage_score']:.2f}" if s["coverage_score"] is not None else "  —"
        print(f"{qid:28s} {'✅' if s['hit_at_10'] else '❌':>6s} "
              f"{cov_str:>5s} {s['forbidden_hits']:>5d} "
              f"{s['best_canonical_rank']:>5d}")

        results.append({
            "id": qid,
            "query": q["query"],
            "retrieved_top10": retrieved_ids,
            "scores": s,
        })

    n = len(queries)
    mean_hit = hits / n if n else 0.0
    mean_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else None
    mean_best_rank = sum(best_ranks) / n if n else 0.0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mean Hit@10              : {mean_hit:.3f}  ({hits}/{n})")
    print(f"  Mean coverage score      : "
          f"{f'{mean_coverage:.3f}  (over {len(coverage_scores)} queries)' if mean_coverage is not None else 'n/a'}")
    print(f"  Total forbidden hits     : {forbidden_total}")
    print(f"  Mean best canonical rank : {mean_best_rank:.1f}  (lower = better, 11 = miss)")

    # ─── Gate ───────────────────────────────────────────────────────────
    passed = (mean_hit >= args.strict_threshold
              and forbidden_total <= args.forbidden_budget)
    if passed:
        print(f"\n✅  PASS  (Hit@10 ≥ {args.strict_threshold}, "
              f"forbidden ≤ {args.forbidden_budget})")
    else:
        why = []
        if mean_hit < args.strict_threshold:
            why.append(f"Hit@10 {mean_hit:.3f} below {args.strict_threshold}")
        if forbidden_total > args.forbidden_budget:
            why.append(f"{forbidden_total} forbidden hits exceed budget {args.forbidden_budget}")
        print(f"\n❌  FAIL  ({'; '.join(why)})")

    Path(args.output).write_text(json.dumps({
        "summary": {
            "mean_hit_at_10": mean_hit,
            "mean_coverage": mean_coverage,
            "total_forbidden": forbidden_total,
            "mean_best_canonical_rank": mean_best_rank,
            "passed": passed,
            "statute_boost_enabled": not args.disable_statute_boost,
        },
        "per_query": results,
    }, indent=2))
    print(f"\nFull results: {args.output}")

    sys.exit(0 if passed else 1)


def _infer_preserved_terms(q: dict) -> list[str]:
    """
    Heuristic fallback — in production the expander supplies preserved_terms
    directly. For the benchmark we don't want to depend on Gemma being up,
    so we approximate: use the query text's distinctive multi-word phrases.
    Strips common words, keeps legal-sounding 2-3 word phrases.
    """
    q_text = q["query"].lower()
    # very light heuristic: look for phrases like "clean break", "non-molestation",
    # "section 25", "matrimonial causes act", etc.
    candidates = []
    # multi-word doctrine patterns
    for phrase in [
        "clean break", "non-molestation", "occupation order",
        "welfare paramount", "welfare checklist", "financial provision",
        "financial resources", "matrimonial causes act", "children act",
        "family law act", "domestic abuse", "inheritance act",
        "pension sharing", "pension attachment", "conduct",
    ]:
        if phrase in q_text:
            candidates.append(phrase)
    # explicit section references
    import re
    for m in re.finditer(r"section\s+\d+[a-z]?(?:\(\d+\))?(?:\([a-z]\))?", q_text):
        candidates.append(m.group())
    return candidates


if __name__ == "__main__":
    main()
