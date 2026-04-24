"""
retrieval_step4.py — drop-in Step-4 retrieval for AILES RAG.

Implements the three query-time patches that fixed the visible retrieval
failures in our evaluation:

  1. Dual-lane hybrid retrieval — one unfiltered lane (catches judgments)
     plus one statute-filtered lane (forces statute chunks into the candidate
     pool even when judgments semantically dominate). Unioned via RRF.

  2. Chunk-ID dedup — upstream chunker has ~40k chunk_id collisions
     (confirmed in Phase 3 corpus audit). Dedup at query time eliminates
     top-10 duplicates.

  3. Content-length filter — drops chunks with text < MIN_TEXT_LEN chars.
     Removes repealed statute sections stored as ". . . . . ." placeholders.

Usage:
    from retrieval_step4 import retrieve

    candidates = retrieve(
        query_text=user_query,
        qdrant_client=client,
        embedder=sentence_transformer_model,
        bm25_model=bm25okapi_instance,
        token_to_idx=token_to_idx_dict,
        top_k=30,   # send this many to the cross-encoder
    )
    # ... run cross-encoder rerank on `candidates` ...
    # ... then call statute_boost.apply_statute_boost(...) if desired ...

Returns a list of Qdrant PointStruct-like objects with full payload and
.score (RRF-fused score). No cross-encoder rerank is applied here — that's
the caller's responsibility (so it can be swapped without touching this
module).
"""

from __future__ import annotations

import re
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

# ─── Constants (tune these in one place) ────────────────────────────────
COLLECTION = "uk_family_law_dense"
TOP_K_PER_LANE = 40       # candidates per prefetch lane
TOP_K_AFTER_RRF = 40      # candidates returned to caller
MIN_TEXT_LEN = 80         # drop repealed ". . ." sections below this length


# ─── Query-vector construction ──────────────────────────────────────────
def build_query_vectors(query: str, embedder, token_to_idx: dict[str, int]):
    """
    Build dense (BGE) + sparse (BM25, raw TFs for Modifier.IDF) vectors.

    Returns (dense_list, SparseVector).

    Sparse values are RAW term frequencies. The server applies IDF via
    Modifier.IDF (set at collection creation). Do NOT pre-multiply by IDF.
    """
    dense_vec = embedder.encode(query, normalize_embeddings=True).tolist()

    tokens = re.sub(r"[^\w\s\-]", "", query.lower()).split()
    tf_map: dict[int, float] = {}
    for tok in tokens:
        if tok in token_to_idx:
            idx = token_to_idx[tok]
            tf_map[idx] = tf_map.get(idx, 0.0) + 1.0

    sparse_vec = SparseVector(
        indices=list(tf_map.keys()),
        values=list(tf_map.values()),
    )
    return dense_vec, sparse_vec


# ─── Main retrieval function ────────────────────────────────────────────
def retrieve(
    query_text: str,
    qdrant_client: QdrantClient,
    embedder: Any,
    token_to_idx: dict[str, int],
    top_k: int = TOP_K_AFTER_RRF,
    collection: str = COLLECTION,
    min_text_len: int = MIN_TEXT_LEN,
) -> list:
    """
    Dual-lane hybrid retrieval with dedup and content filter.

    Parameters
    ----------
    query_text : str
        User query (no prefix; BGE asymmetric prefix tested, no measurable
        lift end-to-end — see tonight's eval).
    qdrant_client : QdrantClient
        Connected Qdrant client.
    embedder : SentenceTransformer
        BGE-large-en-v1.5 instance (or compatible).
    token_to_idx : dict
        Maps BM25 vocab tokens to integer indices (matches upload-time map).
    top_k : int
        Number of candidates to return after dedup + content filter.
    collection : str
        Qdrant collection name.
    min_text_len : int
        Drop chunks with `len(text) < min_text_len`. Eliminates repealed
        sections.

    Returns
    -------
    list of Qdrant points with .payload and .score (RRF-fused).
    Caller should cross-encoder rerank this list then optionally boost.
    """
    dense_vec, sparse_vec = build_query_vectors(query_text, embedder, token_to_idx)

    statute_filter = Filter(
        must=[FieldCondition(key="doc_type", match=MatchValue(value="statute"))]
    )

    # Dual-lane: 2 unfiltered + 2 statute-filtered prefetches, RRF-fused
    results = qdrant_client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(using="dense",  query=dense_vec,  limit=TOP_K_PER_LANE),
            Prefetch(using="sparse", query=sparse_vec, limit=TOP_K_PER_LANE),
            Prefetch(using="dense",  query=dense_vec,  limit=TOP_K_PER_LANE,
                     filter=statute_filter),
            Prefetch(using="sparse", query=sparse_vec, limit=TOP_K_PER_LANE,
                     filter=statute_filter),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k * 2,  # fetch 2× to give dedup/filter headroom
        with_payload=True,
    )

    # Dedup by chunk_id + content-length filter
    seen: set[str] = set()
    filtered = []
    for p in results.points:
        text = (p.payload.get("text") or "").strip()
        if len(text) < min_text_len:
            continue
        cid = p.payload.get("chunk_id") or str(p.id)
        if cid in seen:
            continue
        seen.add(cid)
        filtered.append(p)
        if len(filtered) >= top_k:
            break

    return filtered
