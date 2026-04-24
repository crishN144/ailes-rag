#!/usr/bin/env python3
"""
A/B test: run the Step 4 benchmark WITH BGE query prefix and compare against
the no-prefix baseline (benchmark_step4_results.json).

BGE-large-en-v1.5 is asymmetric: queries should be prefixed with
    "Represent this sentence for searching relevant passages: "
Passages (already indexed) are NOT prefixed. The check_bge_prefix.py diagnostic
confirmed our passages were indexed without prefix (standard) and that adding
the query prefix changes top-10 by 4/10 on a probe query.

This script measures the actual R@5 / R@10 / MRR / Hit@10 impact across all
20 golden queries.

Usage:
    python3 benchmark_with_prefix.py
"""

import json
import pickle
import re
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "uk_family_law_dense"
BM25_PKL = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index" / "bm25_model.pkl"
GOLDEN_FILE = Path(__file__).parent / "golden_queries.json"
BASELINE_JSON = Path("/tmp/benchmark_step4_results.json")
OUTPUT_JSON = Path("/tmp/benchmark_with_prefix_results.json")

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K_PER_LANE = 40
TOP_K_AFTER_RRF = 40
TOP_K_FINAL = 10
MIN_TEXT_LEN = 80


def retrieve(query: str, client, embedder, reranker, token_to_idx, use_prefix: bool):
    # Dense vector — with or without BGE query prefix
    if use_prefix:
        dense_vec = embedder.encode(
            BGE_PREFIX + query, normalize_embeddings=True
        ).tolist()
    else:
        dense_vec = embedder.encode(query, normalize_embeddings=True).tolist()

    # Sparse vector — always plain (BM25 is symmetric)
    tokens = re.sub(r"[^\w\s\-]", "", query.lower()).split()
    tf_map = {}
    for tok in tokens:
        if tok in token_to_idx:
            idx = token_to_idx[tok]
            tf_map[idx] = tf_map.get(idx, 0.0) + 1.0
    sparse_vec = SparseVector(
        indices=list(tf_map.keys()), values=list(tf_map.values())
    )

    statute_filter = Filter(
        must=[FieldCondition(key="doc_type", match=MatchValue(value="statute"))]
    )

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(using="dense", query=dense_vec, limit=TOP_K_PER_LANE),
            Prefetch(using="sparse", query=sparse_vec, limit=TOP_K_PER_LANE),
            Prefetch(
                using="dense", query=dense_vec, limit=TOP_K_PER_LANE,
                filter=statute_filter,
            ),
            Prefetch(
                using="sparse", query=sparse_vec, limit=TOP_K_PER_LANE,
                filter=statute_filter,
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=TOP_K_AFTER_RRF,
        with_payload=True,
    )

    # Dedup by chunk_id + content-length filter
    seen = set()
    filtered = []
    for p in results.points:
        text = (p.payload.get("text") or "").strip()
        if len(text) < MIN_TEXT_LEN:
            continue
        cid = p.payload.get("chunk_id") or str(p.id)
        if cid in seen:
            continue
        seen.add(cid)
        filtered.append(p)

    if not filtered:
        return []

    pairs = [[query, (p.payload.get("text") or "")[:512]] for p in filtered]
    ce_scores = reranker.predict(pairs)
    reranked = sorted(zip(filtered, ce_scores), key=lambda x: -x[1])[:TOP_K_FINAL]

    return [{
        "chunk_id": pt.payload.get("chunk_id") or str(pt.id),
        "ce_score": float(ce),
        "doc_type": pt.payload.get("doc_type"),
    } for pt, ce in reranked]


def matches(retrieved_id, expected_ids):
    return any(retrieved_id == eid or retrieved_id.startswith(eid) for eid in expected_ids)


def score(retrieved, expected):
    if not expected:
        return {"recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0, "hit@10": 1.0}
    ids = [r["chunk_id"] for r in retrieved]
    hits5 = sum(1 for r in ids[:5] if matches(r, expected))
    hits10 = sum(1 for r in ids[:10] if matches(r, expected))
    mrr = 0.0
    for rank, rid in enumerate(ids, 1):
        if matches(rid, expected):
            mrr = 1.0 / rank
            break
    return {
        "recall@5": hits5 / len(expected),
        "recall@10": hits10 / len(expected),
        "mrr": mrr,
        "hit@10": 1.0 if hits10 > 0 else 0.0,
    }


def main():
    with open(GOLDEN_FILE) as f:
        queries = json.load(f)["queries"]
    print(f"Loaded {len(queries)} golden queries\n")

    print("Loading pipeline...")
    t0 = time.time()
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    with open(BM25_PKL, "rb") as f:
        bm25 = pickle.load(f)
    token_to_idx = {tok: idx for idx, tok in enumerate(bm25.idf.keys())}
    print(f"   ready in {time.time()-t0:.1f}s\n")

    # Load baseline for side-by-side comparison
    baseline_by_id = {}
    if BASELINE_JSON.exists():
        baseline = json.loads(BASELINE_JSON.read_text())
        for r in baseline["per_query"]:
            baseline_by_id[r["id"]] = r["scores"]

    all_results = []
    sum_p = {"recall@5": 0, "recall@10": 0, "mrr": 0, "hit@10": 0}

    print(f"{'query_id':28s}  {'no_prefix R@5':>13s}  {'prefix R@5':>11s}  "
          f"{'ΔR@5':>7s}  {'prefix Hit@10':>14s}")
    print("-" * 100)

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        expected = (
            q.get("expected_documents", {}).get("statutes", [])
            + q.get("expected_documents", {}).get("judgments", [])
        )

        retrieved = retrieve(query_text, client, embedder, reranker, token_to_idx, use_prefix=True)
        s = score(retrieved, expected)

        baseline_r5 = baseline_by_id.get(qid, {}).get("recall@5", None)
        delta_r5 = (s["recall@5"] - baseline_r5) if baseline_r5 is not None else None

        delta_str = f"{delta_r5:+.2f}" if delta_r5 is not None else "  —  "
        base_str = f"{baseline_r5:.2f}" if baseline_r5 is not None else "  —  "
        mark = "↑" if (delta_r5 or 0) > 0.001 else ("↓" if (delta_r5 or 0) < -0.001 else " ")

        print(f"{qid:28s}  {base_str:>13s}  {s['recall@5']:>11.2f}  "
              f"{delta_str:>7s} {mark} {int(s['hit@10']):>14d}")

        for k in sum_p:
            sum_p[k] += s[k]

        all_results.append({
            "id": qid,
            "query": query_text,
            "expected": expected,
            "retrieved_top10": retrieved,
            "scores_with_prefix": s,
            "scores_no_prefix_baseline": baseline_by_id.get(qid),
        })

    n = len(queries)
    print("\n" + "=" * 100)
    print(f"SUMMARY  ({n} queries)")
    print("=" * 100)

    if baseline_by_id:
        base_means = {
            k: sum(baseline_by_id[r["id"]][k] for r in all_results) / n
            for k in ("recall@5", "recall@10", "mrr", "hit@10")
        }
        print(f"  {'metric':10s}  {'no_prefix':>10s}  {'with_prefix':>12s}  {'delta':>8s}")
        for k in ("recall@5", "recall@10", "mrr", "hit@10"):
            base = base_means[k]
            new = sum_p[k] / n
            d = new - base
            sign = "↑" if d > 0.001 else ("↓" if d < -0.001 else " ")
            print(f"  {k:10s}  {base:>10.3f}  {new:>12.3f}  {d:+8.3f} {sign}")

    OUTPUT_JSON.write_text(json.dumps({
        "summary_with_prefix": {k: v / n for k, v in sum_p.items()},
        "n_queries": n,
        "per_query": all_results,
    }, indent=2))
    print(f"\nFull results: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
