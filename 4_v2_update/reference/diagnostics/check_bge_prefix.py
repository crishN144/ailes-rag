#!/usr/bin/env python3
"""
BGE-large-en-v1.5 asymmetric-retrieval check.

BGE is asymmetric: queries should be prefixed with
    "Represent this sentence for searching relevant passages: "
while passages are NOT prefixed. Using the wrong convention costs 5-15% recall.

This script:
 1. Picks a known statute chunk.
 2. Embeds its text two ways (with prefix / without prefix).
 3. Compares both to the stored Qdrant vector.
 4. Tells you which convention was used at indexing time.

 5. Runs a real query through the pipeline with/without prefix and compares
    top-10 overlap to see if adding the query prefix changes results.

Usage:
    python3 check_bge_prefix.py
"""

import re
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "uk_family_law_dense"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

PROBE_CHUNK_ID = "mca-1973-section-25-full"
TEST_QUERY = "section 25 matrimonial causes act 1973 factors"


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("Loading BGE...")
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    # 1. Fetch the probe chunk from Qdrant
    print(f"\nFetching probe chunk: {PROBE_CHUNK_ID}")
    r = client.scroll(
        collection_name=COLLECTION,
        scroll_filter={"must": [{"key": "chunk_id", "match": {"value": PROBE_CHUNK_ID}}]},
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    if not r[0]:
        print(f"ERROR: chunk {PROBE_CHUNK_ID} not found")
        return

    point = r[0][0]
    stored_vec = point.vector["dense"]
    text = point.payload["text"]
    print(f"  text length: {len(text)}")
    print(f"  stored vec dim: {len(stored_vec)}")
    print(f"  stored vec norm: {np.linalg.norm(stored_vec):.4f}"
          f"   (≈1.0 means normalized, good)")

    # 2. Re-embed the text two ways
    print("\nRe-embedding chunk text two ways:")
    vec_noprefix = embedder.encode(text, normalize_embeddings=True)
    vec_withprefix = embedder.encode(BGE_PREFIX + text, normalize_embeddings=True)

    sim_noprefix = cosine(vec_noprefix, stored_vec)
    sim_withprefix = cosine(vec_withprefix, stored_vec)

    print(f"  cos(stored, re-embed WITHOUT prefix): {sim_noprefix:.4f}")
    print(f"  cos(stored, re-embed WITH    prefix): {sim_withprefix:.4f}")

    diff = sim_noprefix - sim_withprefix
    if sim_noprefix > 0.99:
        convention = "NO PREFIX (passages indexed without prefix — standard BGE)"
    elif sim_withprefix > 0.99:
        convention = "WITH PREFIX (passages indexed with prefix — non-standard)"
    elif abs(diff) < 0.01:
        convention = "AMBIGUOUS (both match within noise — unusual)"
    elif diff > 0:
        convention = "NO PREFIX more likely"
    else:
        convention = "WITH PREFIX more likely"
    print(f"\n  VERDICT: passages were indexed {convention}")

    # 3. Test whether adding the QUERY prefix changes retrieval
    print("\n" + "=" * 80)
    print(f"Testing query prefix effect on retrieval:")
    print(f"  Query: {TEST_QUERY!r}")
    print("=" * 80)

    q_noprefix = embedder.encode(TEST_QUERY, normalize_embeddings=True).tolist()
    q_withprefix = embedder.encode(BGE_PREFIX + TEST_QUERY, normalize_embeddings=True).tolist()

    results_noprefix = client.query_points(
        collection_name=COLLECTION,
        query=q_noprefix,
        using="dense",
        limit=10,
        with_payload=["chunk_id", "doc_type"],
    ).points

    results_withprefix = client.query_points(
        collection_name=COLLECTION,
        query=q_withprefix,
        using="dense",
        limit=10,
        with_payload=["chunk_id", "doc_type"],
    ).points

    ids_noprefix = [p.payload.get("chunk_id", str(p.id)) for p in results_noprefix]
    ids_withprefix = [p.payload.get("chunk_id", str(p.id)) for p in results_withprefix]

    overlap = set(ids_noprefix) & set(ids_withprefix)
    print(f"\n  top-10 overlap: {len(overlap)} / 10")
    print(f"  top-1 match: {ids_noprefix[0] == ids_withprefix[0]}")

    print(f"\n  NO PREFIX top-5:")
    for i, (cid, p) in enumerate(zip(ids_noprefix, results_noprefix), 1):
        mark = " " if cid in ids_withprefix else "*"
        print(f"    {mark}#{i}  score={p.score:.4f}  {cid}")

    print(f"\n  WITH PREFIX top-5:")
    for i, (cid, p) in enumerate(zip(ids_withprefix, results_withprefix), 1):
        mark = " " if cid in ids_noprefix else "*"
        print(f"    {mark}#{i}  score={p.score:.4f}  {cid}")

    print("\n  * = appears only in that column")

    # Verdict
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if sim_noprefix > 0.99 and len(overlap) >= 8 and ids_noprefix[0] == ids_withprefix[0]:
        print("  Passages indexed without prefix (standard BGE).")
        print("  Query prefix barely moves results — current pipeline is OK as-is.")
        print("  Marginal lift possible from adding prefix; test on the full 20-query")
        print("  golden benchmark to quantify.")
    elif sim_noprefix > 0.99 and len(overlap) < 8:
        print("  Passages indexed without prefix (standard BGE).")
        print("  Adding prefix to QUERIES changes top-10 meaningfully —")
        print(f"  only {len(overlap)}/10 overlap. Worth re-running the benchmark with")
        print("  BGE_PREFIX + query at encode time. This could be the 5-15% free lift.")
    elif sim_withprefix > 0.99:
        print("  Passages were indexed WITH the prefix (non-standard).")
        print("  Current queries (no prefix) may be mis-asymmetric.")
        print("  Either re-embed passages without prefix OR add prefix to queries.")
    else:
        print("  Ambiguous — neither re-embed matches the stored vector exactly.")
        print("  Possible causes: different BGE version, different normalization,")
        print("  text preprocessing differs. Investigate before acting.")


if __name__ == "__main__":
    main()
