#!/usr/bin/env python3
"""
Dump Step-4 retrieval output with FULL text + split dense/BM25/CE scores
for manual (lawyer-grade) evaluation.

Runs three passes per query:
  1. Step 4 pipeline (dual-lane + dedup + content filter + CE rerank) → top 10 final
  2. Dense-only lookup over top 200 → map chunk_id → dense rank + score
  3. Sparse-only lookup over top 200 → map chunk_id → sparse rank + score

For each top-10 chunk, reports:
  rank, chunk_id, doc_type, full text, dense_rank+score, bm25_rank+score, ce_score

Outputs:
  /tmp/eval_for_grading.json  — machine-readable per-query dump
  /tmp/eval_for_grading.txt   — human-readable, paste-to-Claude format

Usage:
    python3 dump_for_grading.py
"""

import json
import pickle
import re
import sys
import time
from pathlib import Path

# Import the Step-4 pipeline function we've already validated
sys.path.insert(0, str(Path(__file__).parent))
from run_benchmark_step4 import (  # noqa: E402
    BM25_PKL,
    COLLECTION,
    QDRANT_URL,
    retrieve_step4,
    build_query_vectors,
)

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import SparseVector  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

GOLDEN_FILE = Path(__file__).parent / "golden_queries.json"
OUT_JSON = Path("/tmp/eval_for_grading.json")
OUT_TXT = Path("/tmp/eval_for_grading.txt")
COMPONENT_LOOKUP_DEPTH = 200  # how deep to look into dense/sparse lanes for per-chunk scores


def component_lookup(client, dense_vec, sparse_vec):
    """Return two dicts mapping chunk_id -> (rank, score) for dense and sparse lanes."""
    # Dense-only
    dense_results = client.query_points(
        collection_name=COLLECTION,
        query=dense_vec,
        using="dense",
        limit=COMPONENT_LOOKUP_DEPTH,
        with_payload=["chunk_id"],
    )
    dense_map = {}
    for rank, p in enumerate(dense_results.points, 1):
        cid = p.payload.get("chunk_id") or str(p.id)
        if cid not in dense_map:
            dense_map[cid] = (rank, float(p.score))

    # Sparse-only
    sparse_results = client.query_points(
        collection_name=COLLECTION,
        query=sparse_vec,
        using="sparse",
        limit=COMPONENT_LOOKUP_DEPTH,
        with_payload=["chunk_id"],
    )
    sparse_map = {}
    for rank, p in enumerate(sparse_results.points, 1):
        cid = p.payload.get("chunk_id") or str(p.id)
        if cid not in sparse_map:
            sparse_map[cid] = (rank, float(p.score))

    return dense_map, sparse_map


def fetch_full_text(client, chunk_ids):
    """Fetch the full payload for a set of chunk_ids (limit 10 per call is cheap)."""
    # We need to scroll or retrieve. Easiest: use scroll with chunk_id filter.
    # But since chunk_ids aren't Qdrant's internal `id`, use scroll with payload filter.
    # For simplicity, re-query each chunk via scroll on the chunk_id.
    out = {}
    for cid in chunk_ids:
        r = client.scroll(
            collection_name=COLLECTION,
            scroll_filter={"must": [{"key": "chunk_id", "match": {"value": cid}}]},
            limit=1,
            with_payload=True,
        )
        if r[0]:
            out[cid] = r[0][0].payload
    return out


def dump_query(q, client, embedder, reranker, token_to_idx):
    query_text = q["query"]
    qid = q["id"]
    expected_statutes = q.get("expected_documents", {}).get("statutes", [])
    expected_judgments = q.get("expected_documents", {}).get("judgments", [])

    # Step 4 retrieval (top 10, ce-ranked)
    top10 = retrieve_step4(query_text, client, embedder, reranker, token_to_idx)

    # Component-score lookups
    dense_vec, sparse_vec = build_query_vectors(query_text, embedder, token_to_idx)
    dense_map, sparse_map = component_lookup(client, dense_vec, sparse_vec)

    # Full text for each top-10 chunk (Step 4 returns a 120-char preview)
    full_payloads = fetch_full_text(client, [r["chunk_id"] for r in top10])

    enriched = []
    for rank, r in enumerate(top10, 1):
        cid = r["chunk_id"]
        payload = full_payloads.get(cid, {})
        dense = dense_map.get(cid, (None, None))
        sparse = sparse_map.get(cid, (None, None))
        enriched.append({
            "rank": rank,
            "chunk_id": cid,
            "doc_type": payload.get("doc_type") or r.get("doc_type"),
            "statute_title": payload.get("statute_title"),
            "section_number": payload.get("section_number"),
            "case_citation": payload.get("case_citation"),
            "court": payload.get("court"),
            "paragraph_type": payload.get("paragraph_type"),
            "ce_score": r["ce_score"],
            "dense_rank": dense[0],
            "dense_score": dense[1],
            "bm25_rank": sparse[0],
            "bm25_score": sparse[1],
            "text": payload.get("text") or r.get("text_preview", ""),
        })

    return {
        "id": qid,
        "query": query_text,
        "category": q.get("category"),
        "difficulty": q.get("difficulty"),
        "expected_statutes": expected_statutes,
        "expected_judgments": expected_judgments,
        "top10": enriched,
    }


def format_txt(entry, n, total):
    lines = []
    lines.append("=" * 100)
    lines.append(f"[Q{n} of {total}]  {entry['id']}   "
                 f"(category={entry['category']}, difficulty={entry['difficulty']})")
    lines.append("=" * 100)
    lines.append(f"QUERY: {entry['query']}")
    lines.append("")
    lines.append(f"EXPECTED (from golden_queries.json):")
    if entry["expected_statutes"]:
        lines.append(f"  statutes:  {entry['expected_statutes']}")
    if entry["expected_judgments"]:
        lines.append(f"  judgments: {entry['expected_judgments']}")
    if not entry["expected_statutes"] and not entry["expected_judgments"]:
        lines.append("  (empty — negative/out-of-scope test)")
    lines.append("")
    lines.append("--- RETRIEVED TOP 10 ---")
    for r in entry["top10"]:
        cid_short = r["chunk_id"][:75]
        ident = r["statute_title"] or r["case_citation"] or "—"
        if r["section_number"]:
            ident = f"{ident}  s.{r['section_number']}"
        elif r["paragraph_type"]:
            ident = f"{ident}  [{r['paragraph_type']}]"

        dr = f"{r['dense_rank']}" if r["dense_rank"] is not None else "—"
        ds = f"{r['dense_score']:.3f}" if r["dense_score"] is not None else "—"
        br = f"{r['bm25_rank']}" if r["bm25_rank"] is not None else "—"
        bs = f"{r['bm25_score']:.2f}" if r["bm25_score"] is not None else "—"

        lines.append("")
        lines.append(
            f"#{r['rank']:2}  [{r['doc_type']}]  {cid_short}"
        )
        lines.append(
            f"    {ident}"
        )
        lines.append(
            f"    ce={r['ce_score']:+.3f}   "
            f"dense: rank={dr} score={ds}   "
            f"bm25: rank={br} score={bs}"
        )
        text = (r["text"] or "").replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        # Show up to ~800 chars of text for grading
        if len(text) > 800:
            text = text[:800] + "…"
        lines.append(f"    TEXT: {text}")
    lines.append("")
    return "\n".join(lines)


def main():
    with open(GOLDEN_FILE) as f:
        data = json.load(f)
    queries = data["queries"]
    print(f"Loaded {len(queries)} golden queries\n")

    print("Loading pipeline (BGE + CE + BM25 + Qdrant)...")
    t0 = time.time()
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    with open(BM25_PKL, "rb") as f:
        bm25 = pickle.load(f)
    token_to_idx = {tok: idx for idx, tok in enumerate(bm25.idf.keys())}
    print(f"   ready in {time.time()-t0:.1f}s\n")

    all_entries = []
    txt_blocks = []

    for i, q in enumerate(queries, 1):
        print(f"[{i:2}/{len(queries)}] {q['id']}: {q['query'][:70]}")
        entry = dump_query(q, client, embedder, reranker, token_to_idx)
        all_entries.append(entry)
        txt_blocks.append(format_txt(entry, i, len(queries)))

    OUT_JSON.write_text(json.dumps(all_entries, indent=2))
    OUT_TXT.write_text("\n".join(txt_blocks))
    print(f"\nJSON → {OUT_JSON}  ({OUT_JSON.stat().st_size/1024:.0f} KB)")
    print(f"TXT  → {OUT_TXT}   ({OUT_TXT.stat().st_size/1024:.0f} KB)")

    # Print first 2 blocks so user can confirm format
    print("\n" + "=" * 100)
    print("PREVIEW — first 2 query blocks (user: confirm format before grading)")
    print("=" * 100 + "\n")
    print(txt_blocks[0])
    print(txt_blocks[1])


if __name__ == "__main__":
    main()
