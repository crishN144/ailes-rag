#!/usr/bin/env python3
"""
A/B test: ms-marco-MiniLM-L-12-v2 vs BAAI/bge-reranker-large on the
20 golden queries. Isolates whether swapping the cross-encoder fixes
the statute-demotion bug (FR_05, CW_01, FR_04) without needing
statute_boost.

Specifically tracks the rank of each known canonical chunk per query.

Runs the full Step-4 pipeline (dual-lane RRF retrieval + dedup + filter),
then reranks the same candidates with BOTH rerankers and compares per-query:
  * Hit@10 (strict)
  * Best canonical rank
  * Specifically: where does the canonical STATUTE chunk land?

Outputs a diff table and a verdict recommendation.

Usage:
    python3 compare_rerankers.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrieval"))
from retrieval_step4 import retrieve  # noqa: E402

from qdrant_client import QdrantClient  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

QDRANT_URL = "http://localhost:6333"
BM25_PKL = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index" / "bm25_model.pkl"
GOLDEN_FILE = HERE.parent.parent / "3_tests_and_results" / "golden_queries.json"
OUTPUT_JSON = Path("/tmp/reranker_comparison.json")

MINI_LM = "cross-encoder/ms-marco-MiniLM-L-12-v2"
BGE_RERANKER = "BAAI/bge-reranker-large"

# The specific chunks we want to track per query —
# these are the canonical statutes that should appear near top-1 but
# currently don't. If BGE fixes the bug, these should climb.
STATUTE_TRACKING = {
    "FR_01_s25_basic":           ["mca-1973-section-25-full", "mca-1973-section-23"],
    "FR_03_contributions_homemaker": ["mca-1973-section-25(2)(f)"],
    "FR_04_conduct_nondisclosure": ["mca-1973-section-25(2)(g)", "mca-1973-section-37"],
    "FR_05_clean_break":         ["mca-1973-section-25A", "mca-1973-section-28"],
    "FR_06_trust_piercing":      ["mca-1973-section-25-full"],
    "FR_07_inherited_assets":    ["mca-1973-section-25-full"],
    "FR_08_pension_sharing":     ["mca-1973-section-24B", "mca-1973-section-25B"],
    "CW_01_welfare_checklist":   ["ca-1989-section-1-full"],
    "CW_02_child_wishes":        ["ca-1989-section-1-full"],
    "DA_01_occupation_order":    ["fla-1996-section-33", "fla-1996-section-42"],
    "INH_01_reasonable_provision": ["i(pfd)a-1975-section-1", "i(pfd)a-1975-section-3-full"],
    "PLAIN_01_maintenance":      ["mca-1973-section-25-full"],
    "PLAIN_02_hiding_money":     ["mca-1973-section-37", "mca-1973-section-25(2)(g)"],
    "PLAIN_03_child_custody":    ["ca-1989-section-1-full"],
    "PLAIN_04_domestic_violence": ["fla-1996-section-33", "fla-1996-section-42", "daa-2021-section-1"],
    "PLAIN_05_inheritance":      ["i(pfd)a-1975-section-1", "i(pfd)a-1975-section-3-full"],
    "FORM_E_01":                 ["mca-1973-section-25-full", "mca-1973-section-23"],
}


def _matches(chunk_id: str, targets: list[str]) -> bool:
    return any(chunk_id == t or chunk_id.startswith(t) for t in targets)


def rank_of_targets(chunk_ids: list[str], targets: list[str]) -> int:
    """Return 1-based rank of first matching target, 11 if none found in top 10."""
    for i, cid in enumerate(chunk_ids[:10], 1):
        if _matches(cid, targets):
            return i
    return 11


def rerank_and_score(candidates, query: str, reranker, query_spec: dict) -> dict:
    """Rerank candidates with the given reranker, return metrics."""
    pairs = [[query, (p.payload.get("text") or "")[:512]] for p in candidates]
    t0 = time.time()
    scores = reranker.predict(pairs)
    rerank_time = time.time() - t0

    ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))[:10]
    chunk_ids = [(p.payload.get("chunk_id") or str(p.id)) for p, _ in ranked]

    # Strict Hit@10 against full expected list
    expected = (
        query_spec.get("expected_documents", {}).get("statutes", [])
        + query_spec.get("expected_documents", {}).get("judgments", [])
    )
    hit = any(_matches(cid, expected) for cid in chunk_ids) if expected else True

    # Best canonical rank (from expected list)
    best_exp_rank = 11
    for i, cid in enumerate(chunk_ids, 1):
        if _matches(cid, expected):
            best_exp_rank = i
            break

    # Statute-chunk tracking (the specific bug we're testing)
    tracked_statutes = STATUTE_TRACKING.get(query_spec["id"], [])
    tracked_rank = rank_of_targets(chunk_ids, tracked_statutes) if tracked_statutes else None

    return {
        "top10_chunk_ids": chunk_ids,
        "hit_at_10": hit,
        "best_expected_rank": best_exp_rank,
        "tracked_statute_rank": tracked_rank,
        "rerank_time_sec": rerank_time,
    }


def main():
    print("Loading golden queries...")
    queries = json.loads(GOLDEN_FILE.read_text())["queries"]
    print(f"   {len(queries)} queries\n")

    print("Loading pipeline (BGE dense embedder + BM25 + Qdrant)...")
    t0 = time.time()
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    with open(BM25_PKL, "rb") as f:
        bm25 = pickle.load(f)
    token_to_idx = {tok: idx for idx, tok in enumerate(bm25.idf.keys())}
    print(f"   ready in {time.time()-t0:.1f}s\n")

    print(f"Loading reranker A: {MINI_LM}")
    r_minilm = CrossEncoder(MINI_LM)
    print(f"Loading reranker B: {BGE_RERANKER}")
    print("  ⚠️  BGE-reranker-large is 560M params — may take 60-90s to download on first run")
    r_bge = CrossEncoder(BGE_RERANKER)
    print(f"   both ready\n")

    rows = []
    total_time_minilm = 0.0
    total_time_bge = 0.0
    wins_minilm = 0
    wins_bge = 0
    ties = 0

    print(f"{'query_id':30s} {'mm-hit':>7s} {'bge-hit':>8s} "
          f"{'mm-rank':>8s} {'bge-rank':>9s} {'mm-stat':>8s} {'bge-stat':>9s}")
    print("-" * 110)

    for q in queries:
        # Shared retrieval step
        candidates = retrieve(q["query"], client, embedder, token_to_idx, top_k=30)

        if not candidates:
            continue

        # Rerank with each
        m = rerank_and_score(candidates, q["query"], r_minilm, q)
        b = rerank_and_score(candidates, q["query"], r_bge, q)
        total_time_minilm += m["rerank_time_sec"]
        total_time_bge += b["rerank_time_sec"]

        # Who wins on best_expected_rank (lower is better)
        if b["best_expected_rank"] < m["best_expected_rank"]:
            wins_bge += 1
            winner = "B>"
        elif m["best_expected_rank"] < b["best_expected_rank"]:
            wins_minilm += 1
            winner = "M>"
        else:
            ties += 1
            winner = "=="

        mm_stat = m["tracked_statute_rank"] if m["tracked_statute_rank"] is not None else "—"
        bg_stat = b["tracked_statute_rank"] if b["tracked_statute_rank"] is not None else "—"

        print(f"{q['id']:30s} {'✅' if m['hit_at_10'] else '❌':>7s} "
              f"{'✅' if b['hit_at_10'] else '❌':>8s} "
              f"{m['best_expected_rank']:>8d} {b['best_expected_rank']:>9d} "
              f"{str(mm_stat):>8s} {str(bg_stat):>9s}  {winner}")

        rows.append({
            "id": q["id"],
            "minilm": m,
            "bge_reranker": b,
            "winner": winner,
            "tracked_statutes": STATUTE_TRACKING.get(q["id"], []),
        })

    n = len(rows)
    mm_hits = sum(1 for r in rows if r["minilm"]["hit_at_10"])
    bg_hits = sum(1 for r in rows if r["bge_reranker"]["hit_at_10"])
    mm_mean_rank = sum(r["minilm"]["best_expected_rank"] for r in rows) / n
    bg_mean_rank = sum(r["bge_reranker"]["best_expected_rank"] for r in rows) / n

    # Stat-tracking stats
    mm_stat_ranks = [r["minilm"]["tracked_statute_rank"] for r in rows if r["minilm"]["tracked_statute_rank"] is not None]
    bg_stat_ranks = [r["bge_reranker"]["tracked_statute_rank"] for r in rows if r["bge_reranker"]["tracked_statute_rank"] is not None]
    mm_stat_in_top5 = sum(1 for r in mm_stat_ranks if r <= 5)
    bg_stat_in_top5 = sum(1 for r in bg_stat_ranks if r <= 5)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'metric':40s} {'ms-marco':>12s} {'bge-reranker':>14s}")
    print("-" * 80)
    print(f"{'Hit@10':40s} {mm_hits:>11d}/{n} {bg_hits:>13d}/{n}")
    print(f"{'Mean best-expected rank (lower=better)':40s} {mm_mean_rank:>12.2f} {bg_mean_rank:>14.2f}")
    print(f"{'Tracked statutes in top-5':40s} "
          f"{mm_stat_in_top5:>11d}/{len(mm_stat_ranks)} "
          f"{bg_stat_in_top5:>13d}/{len(bg_stat_ranks)}")
    print(f"{'Total rerank time':40s} {total_time_minilm:>11.1f}s {total_time_bge:>13.1f}s")
    print(f"{'Mean rerank time per query':40s} "
          f"{total_time_minilm/n:>11.3f}s {total_time_bge/n:>13.3f}s")

    print()
    print(f"Per-query winner: BGE better: {wins_bge}  MiniLM better: {wins_minilm}  Ties: {ties}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if bg_stat_in_top5 > mm_stat_in_top5 + 3 and bg_mean_rank < mm_mean_rank - 0.5:
        verdict = "SWAP — BGE-reranker-large materially fixes statute demotion"
        print(verdict)
        print("  Recommend: swap reranker in production; drop statute_boost.py")
    elif bg_stat_in_top5 > mm_stat_in_top5 and bg_hits >= mm_hits:
        verdict = "MIXED — BGE improves slightly; keep statute_boost as belt-and-braces"
        print(verdict)
        print("  Recommend: swap reranker AND keep statute_boost (defence in depth)")
    elif bg_hits < mm_hits - 1 or bg_mean_rank > mm_mean_rank + 0.5:
        verdict = "KEEP — MS-MARCO performs as well or better; don't swap"
        print(verdict)
        print("  Recommend: keep ms-marco-MiniLM; use statute_boost to fix ranking bugs")
    else:
        verdict = "INCONCLUSIVE — similar performance; prefer smaller/faster model"
        print(verdict)
        print("  Recommend: keep ms-marco-MiniLM (17× smaller, 2.6× faster); ship statute_boost")

    OUTPUT_JSON.write_text(json.dumps({
        "verdict": verdict,
        "summary": {
            "minilm": {"hit10": mm_hits, "mean_best_rank": mm_mean_rank,
                       "stat_in_top5": mm_stat_in_top5, "total_time_sec": total_time_minilm},
            "bge": {"hit10": bg_hits, "mean_best_rank": bg_mean_rank,
                    "stat_in_top5": bg_stat_in_top5, "total_time_sec": total_time_bge},
            "n_queries": n, "wins_bge": wins_bge, "wins_minilm": wins_minilm, "ties": ties,
        },
        "per_query": rows,
    }, indent=2))
    print(f"\nDetail: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
