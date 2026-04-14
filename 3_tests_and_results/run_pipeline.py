#!/usr/bin/env python3
"""
AILES RAG Pipeline — Full End-to-End Retrieval
Connects to: Qdrant Cloud, Gemini Flash API, local BM25 + BGE + Cross-Encoder

Usage:
    python3 run_full_pipeline.py
    python3 run_full_pipeline.py --query "How are child welfare considerations assessed?"
    python3 run_full_pipeline.py --query "Recognition of homemaker contributions in financial settlements"

Requirements:
    pip install qdrant-client sentence-transformers google-generativeai
"""

import os
import sys
import json
import time
import pickle
import re
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://1f674c1b-78e4-4df7-82d6-12d4bc1fad52.europe-west3-0.gcp.cloud.qdrant.io"
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "uk_family_law_dense"
BM25_DIR = Path.home() / "Downloads" / "hpc_outputs" / "bm25_index"
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GEMINI_MODEL = "gemini-2.0-flash"

RRF_K = 60
DENSE_LIMIT = 20
SPARSE_LIMIT = 20
RERANK_CANDIDATES = 15
FINAL_TOP_K = 10


# ─── Helpers ─────────────────────────────────────────────────────────────────

def sep(char="=", width=100):
    print(char * width)

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def step_header(num, title):
    print()
    sep()
    print(f"[{ts()}] STEP {num}: {title}")
    sep()
    print()


# ─── STEP 2: Query Translation (Gemini Flash) ───────────────────────────────

def translate_query(original_query: str) -> Dict:
    step_header(2, "QUERY TRANSLATION (Gemini Flash)")
    print(f'   Original query: "{original_query}"')
    print()

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""You are a UK family law legal query translator. Rewrite the user query using precise UK family law terminology.

User query: "{original_query}"

Return ONLY valid JSON (no markdown fences):
{{
  "translated_query": "<rewritten query using precise legal terms>",
  "legal_concepts_identified": ["<concept 1>", "<concept 2>", ...],
  "search_keywords": ["<keyword1>", "<keyword2>", ...]
}}

Rules:
- Expand abbreviations to full legal terms
- Add relevant statutory references (e.g., MCA 1973 s.25)
- Include related legal concepts
- Keep search_keywords as individual terms for BM25 matching"""

    print(f"   Calling {GEMINI_MODEL}...")
    t0 = time.time()
    response = model.generate_content(prompt)
    elapsed = (time.time() - t0) * 1000

    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    print(f"   Translated: \"{result['translated_query']}\"")
    print()
    print(f"   Legal concepts identified:")
    for c in result["legal_concepts_identified"]:
        print(f"     - {c}")
    print()
    print(f"   Search keywords: {result['search_keywords']}")
    print(f"   +{elapsed:.0f}ms")
    return result, elapsed


# ─── STEP 3: Dual Embedding Generation ──────────────────────────────────────

def generate_embeddings(translated_query: str, bm25_model, token_to_idx: Dict):
    step_header(3, "DUAL EMBEDDING GENERATION")

    # 3A: Dense
    print(f"   --- 3A: DENSE EMBEDDING ({DENSE_MODEL}) ---")
    print()
    print(f'   Input: "{translated_query[:80]}..."')

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(DENSE_MODEL)

    t0 = time.time()
    dense_vector = embedder.encode(translated_query, normalize_embeddings=True)
    dense_ms = (time.time() - t0) * 1000

    preview = ", ".join(f"{v:.4f}" for v in dense_vector[:5])
    print(f"   Vector: [{preview}, ..., {dense_vector[-1]:.4f}]")
    print(f"           ({len(dense_vector)} floats, L2 normalized)")
    print(f"   +{dense_ms:.0f}ms")
    print()

    # 3B: Sparse
    print(f"   --- 3B: SPARSE VECTOR (BM25 Tokenization) ---")
    print()

    t0 = time.time()
    query_tokens = translated_query.lower().split()
    sparse_indices = []
    sparse_values = []
    matched_tokens = []
    unmatched_tokens = []

    for token in query_tokens:
        clean = re.sub(r"[^\w]", "", token)
        if not clean:
            continue
        if clean in token_to_idx:
            idx = token_to_idx[clean]
            idf = bm25_model.idf.get(clean, 1.0)
            sparse_indices.append(idx)
            sparse_values.append(float(idf))
            matched_tokens.append((clean, idx, float(idf)))
        else:
            unmatched_tokens.append(clean)

    sparse_ms = (time.time() - t0) * 1000

    vocab_size = len(token_to_idx)
    print(f"   Tokens: {[t for t, _, _ in matched_tokens]}")
    print(f"   Vocabulary matches: {len(matched_tokens)}/{len(matched_tokens) + len(unmatched_tokens)} "
          f"found in BM25 index ({vocab_size:,} terms)")
    if unmatched_tokens:
        print(f"   Unmatched: {unmatched_tokens}")
    print()

    print(f"   Sparse vector (non-zero entries):")
    print(f"   {'Token':<22} {'Index':>8} {'IDF Score':>10}")
    print(f"   {'-'*22} {'-'*8} {'-'*10}")
    for tok, idx, idf in sorted(matched_tokens, key=lambda x: -x[2])[:15]:
        print(f"   {tok:<22} {idx:>8,} {idf:>10.3f}")
    print()
    print(f"   {len(sparse_indices)} non-zero dimensions out of {vocab_size:,} vocab")
    print(f"   +{sparse_ms:.0f}ms")

    total_ms = dense_ms + sparse_ms
    print(f"\n   Total embedding: +{total_ms:.0f}ms")

    return dense_vector, sparse_indices, sparse_values, embedder, total_ms


# ─── STEP 4: Qdrant Hybrid Search ───────────────────────────────────────────

def qdrant_hybrid_search(client, dense_vector, sparse_indices, sparse_values):
    from qdrant_client.models import Prefetch, SparseVector, FusionQuery

    step_header(4, "QDRANT HYBRID SEARCH")
    print(f"   Qdrant: {QDRANT_URL}")

    # Get collection info
    import requests
    headers = {"api-key": QDRANT_API_KEY} if QDRANT_API_KEY else {}
    try:
        resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}", headers=headers, timeout=10)
        info = resp.json()
        point_count = info.get("result", {}).get("points_count", "unknown")
    except:
        point_count = "unknown"

    print(f"   Collection: {COLLECTION_NAME} ({point_count:,} points)" if isinstance(point_count, int)
          else f"   Collection: {COLLECTION_NAME} ({point_count} points)")
    print()

    # 4A: Dense search
    print(f"   --- 4A: DENSE SEARCH (Semantic) ---")
    print()
    print(f"   query_points(collection='{COLLECTION_NAME}', using='dense', limit={DENSE_LIMIT})")

    t0 = time.time()
    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(using="dense", query=dense_vector.tolist(), limit=DENSE_LIMIT)
        ],
        query=FusionQuery(fusion="rrf"),
        limit=DENSE_LIMIT,
        with_payload=True,
    )
    dense_ms = (time.time() - t0) * 1000

    dense_hits = dense_results.points
    print(f"   Dense search returned {len(dense_hits)} candidates (+{dense_ms:.0f}ms)")
    print()
    print(f"   Top 5 dense hits:")
    print(f"   {'#':>4} {'Score':>8} {'Chunk ID'}")
    print(f"   {'-'*4} {'-'*8} {'-'*50}")
    for i, pt in enumerate(dense_hits[:5], 1):
        cid = pt.payload.get("chunk_id", f"point_{pt.id}")
        print(f"   {i:>4} {pt.score:>8.3f} {cid}")
    print()

    # 4B: Sparse search
    print(f"   --- 4B: SPARSE SEARCH (BM25 Keyword) ---")
    print()

    query_sparse = SparseVector(indices=sparse_indices, values=sparse_values)
    print(f"   query_points(collection='{COLLECTION_NAME}', using='sparse', limit={SPARSE_LIMIT})")

    t0 = time.time()
    sparse_results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(using="sparse", query=query_sparse, limit=SPARSE_LIMIT)
        ],
        query=FusionQuery(fusion="rrf"),
        limit=SPARSE_LIMIT,
        with_payload=True,
    )
    sparse_ms = (time.time() - t0) * 1000

    sparse_hits = sparse_results.points
    print(f"   Sparse search returned {len(sparse_hits)} candidates (+{sparse_ms:.0f}ms)")
    print()
    print(f"   Top 5 sparse hits:")
    print(f"   {'#':>4} {'Score':>8} {'Chunk ID'}")
    print(f"   {'-'*4} {'-'*8} {'-'*50}")
    for i, pt in enumerate(sparse_hits[:5], 1):
        cid = pt.payload.get("chunk_id", f"point_{pt.id}")
        print(f"   {i:>4} {pt.score:>8.2f} {cid}")

    total_ms = dense_ms + sparse_ms
    print(f"\n   Total Qdrant: +{total_ms:.0f}ms")

    return dense_hits, sparse_hits, total_ms


# ─── STEP 5: RRF Fusion ─────────────────────────────────────────────────────

def rrf_fusion(dense_hits, sparse_hits):
    step_header(5, "RECIPROCAL RANK FUSION (RRF)")
    print(f"   Merging Dense ({len(dense_hits)}) + Sparse ({len(sparse_hits)}) results...")
    print(f"   Formula: RRF_score(d) = sum( 1 / (k + rank_i(d)) )  where k = {RRF_K}")
    print()

    t0 = time.time()
    scores = defaultdict(lambda: {"rrf": 0.0, "payload": None, "sources": set()})

    for rank, pt in enumerate(dense_hits, 1):
        cid = pt.payload.get("chunk_id", str(pt.id))
        scores[cid]["rrf"] += 1.0 / (RRF_K + rank)
        scores[cid]["payload"] = pt.payload
        scores[cid]["sources"].add("Dense")

    for rank, pt in enumerate(sparse_hits, 1):
        cid = pt.payload.get("chunk_id", str(pt.id))
        scores[cid]["rrf"] += 1.0 / (RRF_K + rank)
        if scores[cid]["payload"] is None:
            scores[cid]["payload"] = pt.payload
        scores[cid]["sources"].add("Sparse")

    ranked = sorted(scores.items(), key=lambda x: -x[1]["rrf"])

    both = sum(1 for _, v in ranked if len(v["sources"]) == 2)
    dense_only = sum(1 for _, v in ranked if v["sources"] == {"Dense"})
    sparse_only = sum(1 for _, v in ranked if v["sources"] == {"Sparse"})

    elapsed = (time.time() - t0) * 1000

    print(f"   Unique candidates: {len(ranked)}")
    print(f"   Both:        {both} chunks")
    print(f"   Dense only:  {dense_only} chunks")
    print(f"   Sparse only: {sparse_only} chunks")
    print()

    top_n = ranked[:RERANK_CANDIDATES]
    print(f"   RRF Ranked (top {RERANK_CANDIDATES} passed to reranker):")
    print(f"   {'#':>4} {'RRF Score':>10} {'Chunk ID':<50} {'Source'}")
    print(f"   {'-'*4} {'-'*10} {'-'*50} {'-'*10}")
    for i, (cid, data) in enumerate(top_n, 1):
        src = "BOTH" if len(data["sources"]) == 2 else list(data["sources"])[0]
        print(f"   {i:>4} {data['rrf']:>10.5f} {cid:<50} {src}")

    print(f"\n   +{elapsed:.0f}ms")
    return top_n, elapsed


# ─── STEP 6: Cross-Encoder Reranking ────────────────────────────────────────

def cross_encoder_rerank(query: str, candidates: List[Tuple[str, Dict]]):
    from sentence_transformers import CrossEncoder

    step_header(6, "CROSS-ENCODER RERANKING")
    print(f"   Model: {RERANKER_MODEL}")
    print(f"   Scoring {len(candidates)} (query, chunk) pairs...")
    print()

    reranker = CrossEncoder(RERANKER_MODEL)

    pairs = []
    for cid, data in candidates:
        text = data["payload"].get("text", "")[:512]
        pairs.append([query, text])

    t0 = time.time()
    ce_scores = reranker.predict(pairs)
    elapsed = (time.time() - t0) * 1000

    scored = []
    for i, (cid, data) in enumerate(candidates):
        scored.append((cid, data, float(ce_scores[i]), i + 1))

    scored.sort(key=lambda x: -x[2])
    top = scored[:FINAL_TOP_K]

    print(f"   Cross-encoder scores (sorted, top {FINAL_TOP_K} kept):")
    print(f"   {'#':>4} {'CE Score':>10} {'Chunk ID':<50} {'Movement'}")
    print(f"   {'-'*4} {'-'*10} {'-'*50} {'-'*12}")
    for new_rank, (cid, data, score, old_rank) in enumerate(top, 1):
        if new_rank < old_rank:
            mv = f"was #{old_rank} UP"
        elif new_rank > old_rank:
            mv = f"was #{old_rank} DOWN"
        else:
            mv = f"was #{old_rank}"
        print(f"   {new_rank:>4} {score:>10.3f} {cid:<50} {mv}")

    dropped = len(scored) - FINAL_TOP_K
    if dropped > 0:
        print(f"\n   ({dropped} candidates below threshold dropped)")
    print(f"\n   +{elapsed:.0f}ms")

    return top, elapsed


# ─── STEP 7: Final Results ──────────────────────────────────────────────────

def display_results(query: str, results: List):
    step_header(7, "FINAL RESULTS")
    print(f'   TOP {len(results)} RESULTS -- Query: "{query}"')
    print(f"   " + "-" * 90)

    for i, (cid, data, score, _) in enumerate(results, 1):
        payload = data["payload"]
        doc_type = payload.get("doc_type", "unknown")
        label = "STATUTE" if doc_type == "statute" else "JUDGMENT"
        print()
        print(f"   {i:>2}. SCORE: {score:.3f} | {label}")
        print(f"       " + "-" * 30)

        if doc_type == "statute":
            print(f"       Statute:  {payload.get('statute_title') or payload.get('act_name', 'N/A')}")
            print(f"       Section:  {payload.get('section_number', 'N/A')}")
        else:
            court = payload.get("court", "N/A")
            year = payload.get("year", "N/A")
            cite = payload.get("case_citation", "N/A")
            print(f"       Court:    {court} ({year})")
            print(f"       Citation: {cite}")

        topics = payload.get("topics", [])
        if topics:
            print(f"       Topics:   {', '.join(topics[:5])}")

        text = payload.get("text", "")
        if text:
            preview = text[:250].replace("\n", " ")
            if len(text) > 250:
                preview += "..."
            print(f"       Text:     {preview}")
        print(f"   " + "-" * 90)

    statutes = sum(1 for _, d, _, _ in results if d["payload"].get("doc_type") == "statute")
    judgments = len(results) - statutes
    print(f"\n   Summary: {statutes} statutes + {judgments} judgments")


# ─── STEP 8: Context Packaging ──────────────────────────────────────────────

def package_context(query: str, results: List) -> str:
    step_header(8, "CONTEXT PACKAGING FOR LLM")

    system_prompt = (
        "You are a UK family law legal research assistant. Answer questions using "
        "ONLY the provided legal sources. Cite specific sections and case references. "
        "If the sources do not contain sufficient information, say so."
    )

    chunks_text = ""
    for i, (cid, data, score, _) in enumerate(results, 1):
        p = data["payload"]
        doc_type = p.get("doc_type", "unknown")
        if doc_type == "statute":
            ref = f"{p.get('statute_title', '')} {p.get('section_number', '')}"
        else:
            ref = p.get("case_citation", cid)
        text = p.get("text", "")[:500]
        chunks_text += f"\n[Source {i}] ({ref})\n{text}\n"

    sys_tokens = len(system_prompt.split()) * 1.3
    query_tokens = len(query.split()) * 1.3
    chunk_tokens = len(chunks_text.split()) * 1.3

    print(f"   System prompt:     ~{int(sys_tokens)} tokens")
    print(f"   User query:        ~{int(query_tokens)} tokens")
    print(f"   Retrieved chunks:  ~{int(chunk_tokens)} tokens ({len(results)} chunks)")
    print(f"   Output budget:     ~500 tokens")
    print(f"   Total:             ~{int(sys_tokens + query_tokens + chunk_tokens + 500)} tokens")

    full_prompt = f"{system_prompt}\n\n# USER QUERY\n{query}\n\n# LEGAL SOURCES\n{chunks_text}\n\n# ANSWER"
    print(f"\n   Context package ready")
    return full_prompt


# ─── STEP 9: LLM Answer Generation ──────────────────────────────────────────

def generate_answer(prompt: str):
    import google.generativeai as genai

    step_header(9, f"LLM ANSWER GENERATION ({GEMINI_MODEL})")
    print(f"   Generating answer from retrieved chunks...")
    print()

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    t0 = time.time()
    response = model.generate_content(prompt)
    elapsed = (time.time() - t0) * 1000

    answer = response.text.strip()
    print(f"   --- GENERATED ANSWER ---")
    print()
    for line in answer.split("\n"):
        print(f"   {line}")
    print()
    print(f"   --- END ANSWER ---")
    print(f"\n   +{elapsed:.0f}ms")
    return answer, elapsed


# ─── STEP 10: API Response ──────────────────────────────────────────────────

def format_response(query, answer, results, timings):
    step_header(10, "API RESPONSE")

    sources = []
    for i, (cid, data, score, _) in enumerate(results, 1):
        p = data["payload"]
        sources.append({
            "rank": i,
            "score": round(score, 3),
            "type": p.get("doc_type", "unknown"),
            "title": p.get("statute_title") or p.get("act_name"),
            "section": p.get("section_number"),
            "citation": p.get("case_citation"),
        })

    response = {
        "query": query,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "result_count": len(results),
        "sources": sources[:3],
        "pipeline": {
            "query_translation_ms": round(timings["translation"]),
            "embedding_ms": round(timings["embedding"]),
            "qdrant_search_ms": round(timings["qdrant"]),
            "rrf_fusion_ms": round(timings["rrf"]),
            "cross_encoder_ms": round(timings["rerank"]),
            "llm_generation_ms": round(timings["generation"]),
            "total_ms": round(timings["total"]),
        },
        "metadata": {
            "corpus_size": 310177,
            "dense_model": DENSE_MODEL,
            "sparse_model": "BM25Okapi",
            "reranker": RERANKER_MODEL,
            "generator": GEMINI_MODEL,
            "fusion": "reciprocal_rank_fusion",
        },
    }

    print(f"   Results: {len(results)} chunks retrieved")
    print(f"   Latency: {round(timings['total'])}ms")
    print()
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return response


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(timings):
    print()
    sep()
    print("PIPELINE SUMMARY")
    sep()
    steps = [
        ("1.  API request received", 0),
        ("2.  Query translation (Gemini Flash)", timings["translation"]),
        ("3.  Embedding generation (Dense + Sparse)", timings["embedding"]),
        ("4.  Qdrant hybrid search", timings["qdrant"]),
        ("5.  RRF fusion", timings["rrf"]),
        ("6.  Cross-encoder reranking", timings["rerank"]),
        ("7.  Results formatted", timings.get("format", 1)),
        ("8.  Context packaging", timings.get("package", 1)),
        ("9.  LLM generation (Gemini Flash)", timings["generation"]),
        ("10. API response sent", 1),
    ]

    cumulative = 0
    print(f"\n   {'STEP':<45} {'TIME':>8} {'CUMUL':>8}")
    print(f"   {'-'*45} {'-'*8} {'-'*8}")
    for label, ms in steps:
        cumulative += ms
        print(f"   {label:<45} {ms:>5.0f}ms {cumulative:>5.0f}ms")

    retrieval = timings["translation"] + timings["embedding"] + timings["qdrant"] + timings["rrf"] + timings["rerank"]
    generation = timings["generation"]
    print(f"\n   Retrieval (steps 2-6):  {retrieval:,.0f}ms")
    print(f"   Generation (step 9):    {generation:,.0f}ms")
    print(f"   Total:                  {cumulative:,.0f}ms ({cumulative/1000:.2f}s)")
    print()
    sep()
    print("PIPELINE COMPLETE")
    sep()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AILES RAG Full Pipeline")
    parser.add_argument("--query", type=str,
                        default="What maintenance would court award for £750k assets?")
    args = parser.parse_args()
    query = args.query

    total_start = time.time()

    # Banner
    print()
    sep("=")
    print("AILES RAG PIPELINE — FULL END-TO-END RETRIEVAL")
    print(f"Corpus: 310,177 UK Family Law chunks | Search: Hybrid BGE + BM25 + RRF + Cross-Encoder")
    sep("=")
    print(f'\nQuery: "{query}"\n')

    timings = {}

    # ── Step 1: Request
    step_header(1, "QUERY RECEIVED")
    print(f'   Query: "{query}"')
    print(f"   Mode: hybrid | Top-K: {FINAL_TOP_K} | Reranker: {RERANKER_MODEL}")
    print(f"   Request validated")

    # ── Step 2: Query Translation
    translation_result, t_ms = translate_query(query)
    timings["translation"] = t_ms
    translated = translation_result["translated_query"]

    # ── Load BM25
    print(f"\n   Loading BM25 model from {BM25_DIR}...")
    with open(BM25_DIR / "bm25_model.pkl", "rb") as f:
        bm25 = pickle.load(f)
    token_to_idx = {token: idx for idx, token in enumerate(bm25.idf.keys())}
    print(f"   BM25 loaded: {len(token_to_idx):,} tokens in vocabulary")

    # ── Step 3: Embeddings
    dense_vec, sp_idx, sp_vals, embedder, emb_ms = generate_embeddings(translated, bm25, token_to_idx)
    timings["embedding"] = emb_ms

    # ── Step 4: Qdrant
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    dense_hits, sparse_hits, q_ms = qdrant_hybrid_search(client, dense_vec, sp_idx, sp_vals)
    timings["qdrant"] = q_ms

    # ── Step 5: RRF
    fused, rrf_ms = rrf_fusion(dense_hits, sparse_hits)
    timings["rrf"] = rrf_ms

    # ── Step 6: Cross-Encoder
    reranked, ce_ms = cross_encoder_rerank(translated, fused)
    timings["rerank"] = ce_ms

    # ── Step 7: Display
    display_results(query, reranked)

    # ── Step 8: Package
    prompt = package_context(query, reranked)

    # ── Step 9: Generate
    answer, gen_ms = generate_answer(prompt)
    timings["generation"] = gen_ms

    # ── Step 10: Response
    timings["total"] = (time.time() - total_start) * 1000
    format_response(query, answer, reranked, timings)

    # Summary
    print_summary(timings)


if __name__ == "__main__":
    main()
