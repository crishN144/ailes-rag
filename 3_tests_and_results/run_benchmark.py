#!/usr/bin/env python3
"""
AILES RAG Benchmark Runner
Loads golden queries from retrieval_benchmark_v2.json, runs each through
the hybrid pipeline (BGE + BM25 + RRF + Cross-Encoder), compares results
against expected documents, and computes Recall@K, MRR, Hit Rate.

Usage:
    python3 run_benchmark.py
    python3 run_benchmark.py --benchmark retrieval_benchmark_v2.json
    python3 run_benchmark.py --category financial_remedy
    python3 run_benchmark.py --query-type plain_english
    python3 run_benchmark.py --save results_2025_12_21.json

Requirements:
    pip install qdrant-client sentence-transformers
    Environment: QDRANT_URL, QDRANT_API_KEY
    Local: ~/Downloads/hpc_outputs/bm25_index/bm25_model.pkl
"""

import os
import sys
import json
import time
import pickle
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

# ─── Config ──────────────────────────────────────────────────────────────────

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://1f674c1b-78e4-4df7-82d6-12d4bc1fad52.europe-west3-0.gcp.cloud.qdrant.io"
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "uk_family_law_dense"
BM25_DIR = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index"
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RRF_K = 60


# ─── Metrics ─────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[str], expected_ids: Set[str], k: int) -> float:
    """Fraction of expected documents found in top-k results."""
    if not expected_ids:
        return 1.0  # negative test — no expected docs means perfect recall
    hits = sum(1 for rid in retrieved_ids[:k] if any(rid.startswith(eid) for eid in expected_ids))
    return hits / len(expected_ids)


def mrr(retrieved_ids: List[str], expected_ids: Set[str]) -> float:
    """Mean Reciprocal Rank — rank of first expected document."""
    if not expected_ids:
        return 1.0
    for rank, rid in enumerate(retrieved_ids, 1):
        if any(rid.startswith(eid) for eid in expected_ids):
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved_ids: List[str], expected_ids: Set[str], k: int = 10) -> float:
    """1 if any expected document appears in top-k, else 0."""
    if not expected_ids:
        return 1.0
    for rid in retrieved_ids[:k]:
        if any(rid.startswith(eid) for eid in expected_ids):
            return 1.0
    return 0.0


def precision_at_k(retrieved_ids: List[str], expected_ids: Set[str], k: int) -> float:
    """Fraction of top-k results that match expected documents."""
    if not retrieved_ids[:k]:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if any(rid.startswith(eid) for eid in expected_ids))
    return hits / min(k, len(retrieved_ids))


def check_must_not_retrieve(retrieved_ids: List[str], must_not: List[str]) -> List[str]:
    """Return any must_not_retrieve prefixes found in results."""
    violations = []
    for rid in retrieved_ids:
        for prefix in must_not:
            if prefix in rid:
                violations.append(f"{rid} matched must_not '{prefix}'")
    return violations


# ─── Hybrid Retrieval ────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self):
        print("Loading models...")

        # Qdrant
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        print(f"  Qdrant: connected to {QDRANT_URL}")

        # BGE embedder
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.embedder = SentenceTransformer(DENSE_MODEL)
        print(f"  BGE: {DENSE_MODEL} loaded")

        # Cross-encoder
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print(f"  Reranker: {RERANKER_MODEL} loaded")

        # BM25
        with open(BM25_DIR / "bm25_model.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.token_to_idx = {tok: idx for idx, tok in enumerate(self.bm25.idf.keys())}
        print(f"  BM25: {len(self.token_to_idx):,} tokens loaded")

        print()

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Run full hybrid pipeline: Dense + Sparse → RRF → Cross-Encoder rerank."""
        from qdrant_client.models import Prefetch, SparseVector, FusionQuery

        # Dense embedding
        dense_vec = self.embedder.encode(query, normalize_embeddings=True)

        # Sparse vector
        tokens = re.sub(r"[^\w\s]", "", query.lower()).split()
        sp_indices, sp_values = [], []
        for tok in tokens:
            if tok in self.token_to_idx:
                sp_indices.append(self.token_to_idx[tok])
                sp_values.append(float(self.bm25.idf.get(tok, 1.0)))

        sparse_vec = SparseVector(indices=sp_indices, values=sp_values)

        # Hybrid search via Qdrant prefetch + RRF
        results = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(using="dense", query=dense_vec.tolist(), limit=20),
                Prefetch(using="sparse", query=sparse_vec, limit=20),
            ],
            query=FusionQuery(fusion="rrf"),
            limit=top_k + 5,  # fetch extra for reranking buffer
            with_payload=True,
        )

        candidates = results.points

        # Cross-encoder rerank
        if candidates:
            pairs = [[query, (p.payload.get("text") or "")[:512]] for p in candidates]
            ce_scores = self.reranker.predict(pairs)

            scored = []
            for i, pt in enumerate(candidates):
                scored.append({
                    "chunk_id": pt.payload.get("chunk_id", str(pt.id)),
                    "score": float(ce_scores[i]),
                    "doc_type": pt.payload.get("doc_type", "unknown"),
                    "text": (pt.payload.get("text") or "")[:200],
                    "case_citation": pt.payload.get("case_citation"),
                    "statute_title": pt.payload.get("statute_title") or pt.payload.get("act_name"),
                    "section": pt.payload.get("section_number"),
                    "topics": pt.payload.get("topics", []),
                })

            scored.sort(key=lambda x: -x["score"])
            return scored[:top_k]

        return []


# ─── Benchmark Runner ────────────────────────────────────────────────────────

def run_benchmark(benchmark_path: str, retriever: HybridRetriever,
                  category_filter: str = None, query_type_filter: str = None) -> Dict:
    """Run all golden queries and compute metrics."""

    with open(benchmark_path) as f:
        benchmark = json.load(f)

    queries = benchmark["queries"]

    # Apply filters
    if category_filter:
        queries = [q for q in queries if q["category"] == category_filter]
    if query_type_filter:
        queries = [q for q in queries if q["query_type"] == query_type_filter]

    print(f"Running {len(queries)} queries...")
    print("=" * 100)

    all_results = []
    total_start = time.time()

    for i, q in enumerate(queries, 1):
        qid = q["id"]
        query_text = q["query"]
        expected_statutes = set(q["expected_documents"]["statutes"])
        expected_judgments = set(q["expected_documents"]["judgments"])
        expected_all = expected_statutes | expected_judgments
        must_not = q.get("must_not_retrieve", [])
        is_negative = q["query_type"] == "out_of_scope"

        print(f"\n[{i}/{len(queries)}] {qid} ({q['difficulty']}, {q['query_type']})")
        print(f"  Query: \"{query_text[:80]}{'...' if len(query_text) > 80 else ''}\"")

        t0 = time.time()
        results = retriever.search(query_text, top_k=10)
        elapsed = (time.time() - t0) * 1000

        retrieved_ids = [r["chunk_id"] for r in results]

        # Compute metrics
        if is_negative:
            # For negative tests: check that top scores are low
            top_score = results[0]["score"] if results else 0
            passed = top_score < 5.0  # cross-encoder scores above 5 = too confident
            r5 = r10 = m = hr = 1.0 if passed else 0.0
            p10 = 0.0
        else:
            r5 = recall_at_k(retrieved_ids, expected_all, 5)
            r10 = recall_at_k(retrieved_ids, expected_all, 10)
            m = mrr(retrieved_ids, expected_all)
            hr = hit_rate(retrieved_ids, expected_all, 10)
            p10 = precision_at_k(retrieved_ids, expected_all, 10)

        violations = check_must_not_retrieve(retrieved_ids, must_not)

        # Print per-query results
        status = "PASS" if (r10 >= 0.5 and hr >= 1.0 and not violations) else "FAIL"
        if is_negative:
            top_score = results[0]["score"] if results else 0
            status = "PASS" if top_score < 5.0 else "FAIL"
            print(f"  NEGATIVE TEST | Top score: {top_score:.2f} | {'Below threshold' if top_score < 5.0 else 'TOO HIGH'} | {status}")
        else:
            print(f"  Recall@5: {r5:.2f} | Recall@10: {r10:.2f} | MRR: {m:.3f} | Hit: {hr:.0f} | P@10: {p10:.2f} | {elapsed:.0f}ms | {status}")

            # Show what was found
            found_statutes = [rid for rid in retrieved_ids if any(rid.startswith(s) for s in expected_statutes)]
            found_judgments = [rid for rid in retrieved_ids if any(rid.startswith(j) for j in expected_judgments)]
            missed_statutes = [s for s in expected_statutes if not any(rid.startswith(s) for rid in retrieved_ids)]
            missed_judgments = [j for j in expected_judgments if not any(rid.startswith(j) for rid in retrieved_ids)]

            if found_statutes:
                print(f"  Found statutes:  {found_statutes[:3]}")
            if found_judgments:
                print(f"  Found judgments: {found_judgments[:3]}")
            if missed_statutes:
                print(f"  MISSED statutes: {missed_statutes}")
            if missed_judgments:
                print(f"  MISSED judgments: {missed_judgments}")

        if violations:
            print(f"  VIOLATIONS: {violations}")

        # Show top 3 retrieved
        print(f"  Top 3 retrieved:")
        for j, r in enumerate(results[:3], 1):
            label = r.get("statute_title") or r.get("case_citation") or r["chunk_id"]
            print(f"    {j}. [{r['score']:.2f}] {r['chunk_id'][:50]}  ({label})")

        all_results.append({
            "id": qid,
            "category": q["category"],
            "difficulty": q["difficulty"],
            "query_type": q["query_type"],
            "recall_at_5": r5,
            "recall_at_10": r10,
            "mrr": m,
            "hit_rate": hr,
            "precision_at_10": p10,
            "latency_ms": elapsed,
            "status": status,
            "violations": violations,
            "retrieved_top_5": [r["chunk_id"] for r in results[:5]],
        })

    total_elapsed = time.time() - total_start

    # ── Aggregate Metrics ──
    print()
    print("=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    n = len(all_results)
    avg_r5 = sum(r["recall_at_5"] for r in all_results) / n
    avg_r10 = sum(r["recall_at_10"] for r in all_results) / n
    avg_mrr = sum(r["mrr"] for r in all_results) / n
    avg_hr = sum(r["hit_rate"] for r in all_results) / n
    avg_p10 = sum(r["precision_at_10"] for r in all_results) / n
    avg_lat = sum(r["latency_ms"] for r in all_results) / n
    pass_count = sum(1 for r in all_results if r["status"] == "PASS")
    fail_count = n - pass_count

    print(f"""
   Queries run:       {n}
   Passed:            {pass_count}/{n} ({pass_count/n*100:.0f}%)
   Failed:            {fail_count}/{n}

   METRIC             SCORE     TARGET
   ─────────────────  ────────  ────────
   Recall@5           {avg_r5:.3f}     >= 0.50
   Recall@10          {avg_r10:.3f}     >= 0.70
   MRR                {avg_mrr:.3f}     >= 0.50
   Hit Rate           {avg_hr:.3f}     >= 0.90
   Precision@10       {avg_p10:.3f}     >= 0.30

   Avg Latency:       {avg_lat:.0f}ms
   Total Time:        {total_elapsed:.1f}s
""")

    # ── Breakdown by Category ──
    print("   BY CATEGORY:")
    categories = sorted(set(r["category"] for r in all_results))
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        cat_r10 = sum(r["recall_at_10"] for r in cat_results) / len(cat_results)
        cat_hr = sum(r["hit_rate"] for r in cat_results) / len(cat_results)
        cat_pass = sum(1 for r in cat_results if r["status"] == "PASS")
        print(f"   {cat:<25} R@10: {cat_r10:.2f}  HR: {cat_hr:.2f}  Pass: {cat_pass}/{len(cat_results)}")

    # ── Breakdown by Difficulty ──
    print()
    print("   BY DIFFICULTY:")
    for diff in ["easy", "medium", "hard", "n/a"]:
        diff_results = [r for r in all_results if r["difficulty"] == diff]
        if not diff_results:
            continue
        diff_r10 = sum(r["recall_at_10"] for r in diff_results) / len(diff_results)
        diff_pass = sum(1 for r in diff_results if r["status"] == "PASS")
        print(f"   {diff:<25} R@10: {diff_r10:.2f}  Pass: {diff_pass}/{len(diff_results)}")

    # ── Breakdown by Query Type ──
    print()
    print("   BY QUERY TYPE:")
    for qt in ["legal", "plain_english", "form_e", "out_of_scope"]:
        qt_results = [r for r in all_results if r["query_type"] == qt]
        if not qt_results:
            continue
        qt_r10 = sum(r["recall_at_10"] for r in qt_results) / len(qt_results)
        qt_pass = sum(1 for r in qt_results if r["status"] == "PASS")
        print(f"   {qt:<25} R@10: {qt_r10:.2f}  Pass: {qt_pass}/{len(qt_results)}")

    # ── Failed Queries ──
    failed = [r for r in all_results if r["status"] == "FAIL"]
    if failed:
        print()
        print("   FAILED QUERIES:")
        for r in failed:
            print(f"   {r['id']:<30} R@10: {r['recall_at_10']:.2f}  MRR: {r['mrr']:.3f}")

    print()
    print("=" * 100)

    # Overall verdict
    if avg_r10 >= 0.7 and avg_hr >= 0.9 and avg_mrr >= 0.5:
        print("VERDICT: PASS — Retrieval meets production targets")
    elif avg_r10 >= 0.5 and avg_hr >= 0.7:
        print("VERDICT: MARGINAL — Retrieval needs tuning (check failed queries)")
    else:
        print("VERDICT: FAIL — Retrieval needs significant improvement")

    print("=" * 100)

    return {
        "timestamp": datetime.now().isoformat(),
        "benchmark_file": benchmark_path,
        "queries_run": n,
        "passed": pass_count,
        "failed": fail_count,
        "aggregate": {
            "recall_at_5": round(avg_r5, 4),
            "recall_at_10": round(avg_r10, 4),
            "mrr": round(avg_mrr, 4),
            "hit_rate": round(avg_hr, 4),
            "precision_at_10": round(avg_p10, 4),
            "avg_latency_ms": round(avg_lat),
        },
        "per_query": all_results,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AILES RAG Benchmark Runner")
    parser.add_argument("--benchmark", type=str, default="retrieval_benchmark_v2.json",
                        help="Path to benchmark JSON")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category (financial_remedy, children_welfare, etc.)")
    parser.add_argument("--query-type", type=str, default=None,
                        help="Filter by query type (legal, plain_english, form_e, out_of_scope)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Resolve benchmark path
    bench_path = args.benchmark
    if not os.path.exists(bench_path):
        bench_path = os.path.join(os.path.dirname(__file__), args.benchmark)
    if not os.path.exists(bench_path):
        print(f"Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    print()
    print("=" * 100)
    print("AILES RAG BENCHMARK RUNNER")
    print(f"Benchmark: {bench_path}")
    print(f"Qdrant:    {QDRANT_URL}")
    print(f"Models:    {DENSE_MODEL} + {RERANKER_MODEL}")
    print("=" * 100)
    print()

    retriever = HybridRetriever()
    results = run_benchmark(bench_path, retriever, args.category, args.query_type)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
