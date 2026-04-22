#!/usr/bin/env python3
"""
Sanity-check retrieval on the Qdrant VM without needing internet access
(this VM can't reach huggingface.co to download BGE, so we can't do dense
queries here). Uses the BM25 pickle that's already on disk to run a REAL
sparse query from a natural-language string and prints the top matches.

Usage:
    python3 test_query.py "what factors does the court consider for child welfare"
"""

import json
import pickle
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

QDRANT_URL = "http://localhost:6333"
COLLECTION = "uk_family_law_dense"
BM25_PKL = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index" / "bm25_model.pkl"


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 test_query.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\nQuery: {query!r}\n")

    print("Loading BM25 vocab from pickle...")
    with open(BM25_PKL, "rb") as f:
        bm25 = pickle.load(f)
    token_to_idx = {token: idx for idx, token in enumerate(bm25.idf.keys())}
    print(f"  Vocab size: {len(token_to_idx):,}\n")

    # Build sparse query vector - raw TFs, server applies IDF (Modifier.IDF)
    query_tokens = query.lower().split()
    index_scores = {}
    matched = []
    unmatched = []
    for token in query_tokens:
        if token in token_to_idx:
            idx = token_to_idx[token]
            index_scores[idx] = index_scores.get(idx, 0.0) + 1.0
            matched.append(token)
        else:
            unmatched.append(token)

    print(f"Matched tokens ({len(matched)}): {matched}")
    if unmatched:
        print(f"Not in vocab ({len(unmatched)}): {unmatched}")
    print()

    if not index_scores:
        print("No query tokens matched BM25 vocab. Try different words.")
        sys.exit(1)

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    results = client.query_points(
        collection_name=COLLECTION,
        query=SparseVector(
            indices=list(index_scores.keys()),
            values=list(index_scores.values()),
        ),
        using="sparse",
        limit=5,
        with_payload=True,
    )

    print("=" * 100)
    print(f"Top 5 results (sparse BM25):")
    print("=" * 100)
    for i, point in enumerate(results.points, 1):
        p = point.payload or {}
        print(f"\n#{i}  score={point.score:.4f}  id={point.id}")
        print(f"    chunk_id:  {p.get('chunk_id')}")
        print(f"    doc_type:  {p.get('doc_type')}")
        print(f"    citation:  {p.get('case_citation') or p.get('statute_title')}")
        print(f"    section:   {p.get('section_number')}")
        print(f"    text:      {(p.get('text') or '')[:300]}...")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
