"""
retrieve_from_expander.py — the glue between the Gemma expander's JSON output
and the dual-lane retrieval in retrieval_step4.

The expander emits 3-5 statute_queries + 3-5 judgment_queries per user turn.
Each is designed to hit different vocabulary angles on the same legal concept.
This module iterates over every expander query, runs retrieve() per query,
unions the candidate pool with dedup, and returns it — ready for the caller's
cross-encoder rerank against the ORIGINAL user query.

Why rerank against the USER query and not the expander queries:
  * Expander queries are vocabulary-shifted to match corpus text — they're
    "retrieval anchors" not user intent.
  * The cross-encoder job is "which chunk answers what the user actually
    asked" — that answer lives in the user query, not in the BM25-friendly
    reformulations.

Integration (this is what production RAG service must do):

    from retrieve_from_expander import retrieve_from_expander

    # 1. Call Gemma with the rag.expander prompt, parse JSON
    expander_out = call_gemma_expander(user_query, conversation_history)

    # 2. If needs_retrieval is false, skip retrieval entirely
    if not expander_out.get("needs_retrieval"):
        return handle_conversational(user_query)

    # 3. Run dual-lane retrieval per expander query, union results
    candidates = retrieve_from_expander(
        expander_out,
        qdrant_client, embedder, token_to_idx,
        candidates_per_query=15,
        total_cap=60,
    )

    # 4. Cross-encoder rerank against the USER query
    pairs = [[user_query, c.payload["text"][:512]] for c in candidates]
    ce_scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: -x[1])[:10]

    # 5. Citation validator on the LLM response later
"""

from __future__ import annotations

from typing import Any

from retrieval_step4 import retrieve


def retrieve_from_expander(
    expander_output: dict,
    qdrant_client: Any,
    embedder: Any,
    token_to_idx: dict[str, int],
    *,
    candidates_per_query: int = 15,
    total_cap: int = 60,
    collection: str = "uk_family_law_dense",
    min_text_len: int = 80,
) -> list:
    """
    Consume the Gemma expander's JSON output and return a deduplicated
    candidate pool ready for cross-encoder reranking.

    Parameters
    ----------
    expander_output : dict
        Parsed JSON from rag.expander. Must contain `statute_queries` and
        `judgment_queries` as lists. `corpora` is respected — if it's
        `["statute"]` only, judgment_queries are ignored even if populated
        (and vice versa). If `needs_retrieval=false`, returns empty list.
    candidates_per_query : int
        How many candidates to pull per expander query. 15 is a reasonable
        default (3-5 queries × 15 = 45-75 candidates pre-dedup).
    total_cap : int
        Maximum total candidates returned after dedup. Caller passes this
        to the cross-encoder. 60 gives healthy headroom.

    Returns
    -------
    list
        Deduplicated list of Qdrant Points with .payload. Order preserved
        by first-seen (which is rank in whichever query surfaced the chunk
        first). Caller reranks with CE against the user query.
    """
    if not expander_output.get("needs_retrieval", False):
        return []

    corpora = expander_output.get("corpora", ["statute", "judgment"])
    want_statute = "statute" in corpora
    want_judgment = "judgment" in corpora

    queries_to_run: list[str] = []
    if want_statute:
        queries_to_run.extend(expander_output.get("statute_queries") or [])
    if want_judgment:
        queries_to_run.extend(expander_output.get("judgment_queries") or [])

    if not queries_to_run:
        return []

    seen: dict[str, Any] = {}
    for sub_query in queries_to_run:
        if not sub_query or not sub_query.strip():
            continue
        try:
            results = retrieve(
                query_text=sub_query,
                qdrant_client=qdrant_client,
                embedder=embedder,
                token_to_idx=token_to_idx,
                top_k=candidates_per_query,
                collection=collection,
                min_text_len=min_text_len,
            )
        except Exception as exc:
            # One bad query shouldn't kill the pool — log and continue.
            print(f"   [retrieve_from_expander] sub-query failed: {sub_query!r} — {exc}")
            continue

        for p in results:
            cid = (p.payload.get("chunk_id") if p.payload else None) or str(p.id)
            if cid not in seen:
                seen[cid] = p
            if len(seen) >= total_cap:
                break
        if len(seen) >= total_cap:
            break

    return list(seen.values())


# ─── Smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated expander output for "What is a clean break in divorce?"
    mock_output = {
        "query_type": "doctrine",
        "needs_retrieval": True,
        "corpora": ["statute", "judgment"],
        "preserved_terms": ["clean break"],
        "statute_queries": [
            "matrimonial causes act 1973 section 25a clean break",
            "matrimonial causes act 1973 section 25a exercise court powers",
            "matrimonial causes act 1973 section 31 variation periodical payments",
        ],
        "judgment_queries": [
            "clean break principle divorce financial remedy",
            "complete the clean break periodical payments dismissed",
        ],
        "reasoning": "Clean-break doctrine — statutory home is MCA 1973 s.25A.",
    }

    import pickle
    from pathlib import Path
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    print("Loading infra...")
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    bm25_pkl = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index" / "bm25_model.pkl"
    with open(bm25_pkl, "rb") as f:
        bm25 = pickle.load(f)
    token_to_idx = {tok: idx for idx, tok in enumerate(bm25.idf.keys())}

    candidates = retrieve_from_expander(
        mock_output,
        qdrant_client=client,
        embedder=embedder,
        token_to_idx=token_to_idx,
        candidates_per_query=15,
        total_cap=60,
    )

    print(f"\nCandidate pool size: {len(candidates)}")
    print("Top 5 chunk_ids (pre-rerank):")
    for i, p in enumerate(candidates[:5], 1):
        cid = (p.payload.get("chunk_id") if p.payload else None) or str(p.id)
        doc_type = (p.payload.get("doc_type") if p.payload else None) or "—"
        print(f"   #{i}  [{doc_type}]  {cid}")

    # Check: did s.25A make the pool?
    pool_ids = [(p.payload.get("chunk_id") if p.payload else None) or str(p.id)
                for p in candidates]
    s25a_in_pool = any(cid.startswith("mca-1973-section-25A") for cid in pool_ids)
    print(f"\nmca-1973-section-25A in pool: {'✅ YES' if s25a_in_pool else '❌ NO'}")
    assert s25a_in_pool, "Clean-break expander output should surface s.25A in pool"
    print("Smoke test passed.")
