"""
Prep script: header + keyword enrichment for the 17 curated factor chunks.

Outputs (in /tmp/factor_enrichment_out/):
  bm25_model.pkl              — new BM25 model with extended vocab (Aslam: swap on backend)
  chunk_ids.pkl               — unchanged but copied for completeness
  factor_dense_vectors.npy    — (17, 1024) new BGE embeddings for the 17 chunks
  factor_chunk_ids.json       — order-aligned chunk_id list for the dense matrix
  factor_sparse_vectors.json  — {chunk_id: {indices: [...], values: [...]}} for re-upload
  factor_payloads.json        — {chunk_id: {full payload incl. enriched bm25_text}} for re-upload
  upsert_17.py                — runnable script Aslam executes against prod Qdrant

Design choices:
  * Vocab indices are STABLE — old tokens keep their original index. New tokens
    introduced by the 17 chunks' keyword folding get appended at the end.
    This means existing 310,160 sparse vectors in Qdrant remain valid.
  * IDF values shift slightly (corpus stats changed for the 17 docs). Qdrant
    applies IDF server-side via Modifier.IDF, so it'll use its own corpus stats
    after re-upload of the 17 — backend pkl IDF only matters for backend-side
    BM25 score computation (if any). Safe.
  * Dense vectors fully recomputed for the 17 only.
  * Payload re-write includes enriched keywords/metadata so future queries can
    rely on them.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE = Path("/Users/crishnagarkar/Downloads/hpc_outputs")
CHUNKS_PATH = BASE / "merged_all_chunks.json"
BM25_DIR = BASE / "bm25_index"
EMB_PATH = BASE / "embeddings" / "bge_embeddings.npy"
OUT_DIR = Path("/tmp/factor_enrichment_out")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def build_enriched_texts(chunk: dict) -> tuple[str, str]:
    """
    Returns (new_bm25_text, new_embed_text).

    Header strategy: prepend full statute name + section + factor name + section
    title. Plus fold keywords into bm25_text for synonym hits. Embed text gets
    keywords as a comma-separated phrase (BGE handles natural-language hints
    better than token soup).
    """
    statute_title = chunk.get("statute_title", "")
    section_number = chunk.get("section_number", "")
    factor_letter = chunk.get("factor_letter", "")
    factor_name = (chunk.get("factor_name") or "").replace("_", " ")
    section_title = chunk.get("section_title", "")
    keywords = chunk.get("keywords") or []
    text = chunk.get("text", "")
    bm25_text_orig = chunk.get("bm25_text") or ""

    header = (
        f"{statute_title} section {section_number}{factor_letter} "
        f"{factor_name} {section_title}"
    ).strip()

    new_bm25_text = (
        header.lower()
        + " "
        + bm25_text_orig
        + " "
        + " ".join(k.lower() for k in keywords)
    ).strip()

    new_embed_text = (
        f"{header}. "
        f"Keywords: {', '.join(keywords)}. "
        f"{text}"
    ).strip()

    return new_bm25_text, new_embed_text


def main() -> None:
    print("Loading chunks...")
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)["chunks"]
    print(f"  {len(chunks):,} chunks")

    print("Loading bm25 + chunk_ids...")
    with open(BM25_DIR / "chunk_ids.pkl", "rb") as f:
        chunk_ids: list[str] = pickle.load(f)
    with open(BM25_DIR / "bm25_model.pkl", "rb") as f:
        bm25 = pickle.load(f)
    print(f"  vocab={len(bm25.idf):,}  docs={bm25.corpus_size:,}")

    cid_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

    # Identify the 17 factor chunks
    factor_chunks = [c for c in chunks if c.get("is_factor")]
    print(f"\nFound {len(factor_chunks)} factor chunks")
    for c in factor_chunks:
        print(f"  {c['chunk_id']}  ({c.get('statute_short')})  {len(c.get('text') or '')} chars")

    # Build enriched texts and new tokenizations
    print("\nBuilding enriched texts...")
    enriched: dict[str, dict] = {}
    for c in factor_chunks:
        cid = c["chunk_id"]
        new_bm25_text, new_embed_text = build_enriched_texts(c)
        enriched[cid] = {
            "chunk": c,
            "new_bm25_text": new_bm25_text,
            "new_embed_text": new_embed_text,
            "new_tokens": new_bm25_text.split(),
        }
        print(
            f"  {cid}: "
            f"bm25 {len((c.get('bm25_text') or '').split())}t -> {len(enriched[cid]['new_tokens'])}t"
        )

    # ---- Vocab maintenance: preserve old token_to_idx, append new tokens ----
    print("\nUpdating vocab with stable indices...")
    # Old vocab (preserved order from idf.keys() insertion order)
    old_token_to_idx = {tok: i for i, tok in enumerate(bm25.idf.keys())}
    print(f"  old vocab size: {len(old_token_to_idx):,}")

    # Find genuinely-new tokens introduced by the 17 enriched chunks
    new_tokens_seen: set[str] = set()
    for info in enriched.values():
        for t in info["new_tokens"]:
            if t and t not in old_token_to_idx and t not in new_tokens_seen:
                new_tokens_seen.add(t)
    print(f"  brand-new tokens to append: {len(new_tokens_seen)}")
    if new_tokens_seen:
        sample = list(new_tokens_seen)[:20]
        print(f"  sample: {sample}")

    # Extended vocab: old tokens keep their indices, new tokens appended
    token_to_idx = dict(old_token_to_idx)
    for tok in sorted(new_tokens_seen):  # deterministic order
        token_to_idx[tok] = len(token_to_idx)
    print(f"  new vocab size: {len(token_to_idx):,}")

    # ---- Update BM25 model in-place: doc_freqs, doc_len, idf, avgdl ----
    print("\nUpdating BM25 model state...")
    for cid, info in enriched.items():
        i = cid_to_idx[cid]
        # New token-frequency dict for this chunk
        from collections import Counter
        tok_counts = Counter(info["new_tokens"])
        bm25.doc_freqs[i] = dict(tok_counts)
        bm25.doc_len[i] = sum(tok_counts.values())

    # avgdl
    bm25.avgdl = sum(bm25.doc_len) / len(bm25.doc_len)
    print(f"  new avgdl: {bm25.avgdl:.2f}")

    # IDF: recompute from scratch using updated doc_freqs.
    # rank_bm25 stores idf as dict[token, idf_value]. To preserve order:
    # iterate token_to_idx (already in stable old+appended order) and compute IDF.
    print("  recomputing IDF (full vocab pass)...")
    # First count document frequency per token
    from collections import defaultdict
    df = defaultdict(int)
    for doc_freqs in bm25.doc_freqs:
        for tok in doc_freqs:
            df[tok] += 1
    N = len(bm25.doc_freqs)
    new_idf: dict[str, float] = {}
    import math
    # BM25Okapi uses: idf(qi) = log((N - df + 0.5) / (df + 0.5) + 1)
    # We use the same formula it uses internally — see rank_bm25 source.
    eps = bm25.epsilon
    avg_idf_acc = 0.0
    for tok in token_to_idx:  # preserve order
        df_t = df.get(tok, 0)
        # rank_bm25's BM25Okapi computes:
        #   freq = (N - df + 0.5) / (df + 0.5)
        #   idf  = math.log(freq + 1)
        # then floors negative idfs at eps * average_idf
        score = math.log(((N - df_t + 0.5) / (df_t + 0.5)) + 1)
        new_idf[tok] = score
        avg_idf_acc += score
    bm25.idf = new_idf
    bm25.average_idf = avg_idf_acc / len(new_idf)
    bm25.corpus_size = N
    # Floor negative IDFs as BM25Okapi does
    for tok, v in list(bm25.idf.items()):
        if v < 0:
            bm25.idf[tok] = eps * bm25.average_idf
    print(f"  new average_idf: {bm25.average_idf:.4f}")

    # ---- Save updated BM25 model ----
    out_pkl = OUT_DIR / "bm25_model.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(bm25, f)
    print(f"\n  Saved {out_pkl} ({out_pkl.stat().st_size/1e6:.1f} MB)")

    # Copy chunk_ids unchanged (Aslam may want both files together)
    out_cids = OUT_DIR / "chunk_ids.pkl"
    with open(out_cids, "wb") as f:
        pickle.dump(chunk_ids, f)

    # ---- Encode 17 chunks with BGE ----
    print("\nEncoding 17 chunks with BGE...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    embed_texts = [enriched[c["chunk_id"]]["new_embed_text"] for c in factor_chunks]
    new_embs = model.encode(embed_texts, normalize_embeddings=True, show_progress_bar=False)
    print(f"  encoded shape: {new_embs.shape}")

    # Save dense vectors aligned to factor_chunk_ids order
    factor_chunk_ids = [c["chunk_id"] for c in factor_chunks]
    np.save(OUT_DIR / "factor_dense_vectors.npy", new_embs.astype(np.float32))
    with open(OUT_DIR / "factor_chunk_ids.json", "w") as f:
        json.dump(factor_chunk_ids, f, indent=2)
    print(f"  Saved factor_dense_vectors.npy ({new_embs.nbytes/1024:.1f} KB)")

    # ---- Build sparse vectors for 17 chunks using new vocab ----
    print("\nBuilding sparse vectors for 17 chunks...")
    sparse_out: dict[str, dict] = {}
    for cid in factor_chunk_ids:
        i = cid_to_idx[cid]
        idx_scores: dict[int, float] = {}
        for tok, freq in bm25.doc_freqs[i].items():
            tidx = token_to_idx[tok]
            idx_scores[tidx] = idx_scores.get(tidx, 0.0) + float(freq)
        sparse_out[cid] = {
            "indices": list(idx_scores.keys()),
            "values": list(idx_scores.values()),
        }
    with open(OUT_DIR / "factor_sparse_vectors.json", "w") as f:
        json.dump(sparse_out, f, indent=2)
    print(f"  Saved factor_sparse_vectors.json")

    # ---- Build enriched payloads ----
    print("\nBuilding payloads...")
    payloads: dict[str, dict] = {}
    for c in factor_chunks:
        cid = c["chunk_id"]
        info = enriched[cid]
        payloads[cid] = {
            "chunk_id": cid,
            "text": c.get("text"),
            "bm25_text": info["new_bm25_text"],  # enriched
            "doc_type": c.get("doc_type", "statute"),
            "court": c.get("court"),
            "year": c.get("statute_year"),
            "case_citation": c.get("case_citation"),
            "statute_title": c.get("statute_title"),
            "statute_short": c.get("statute_short"),
            "section_number": c.get("section_number"),
            "section_title": c.get("section_title"),
            "section_id": c.get("section_id"),
            "paragraph_type": c.get("paragraph_type"),
            "topics": c.get("topics", []),
            # NEW METADATA fields for future boost code:
            "factor_letter": c.get("factor_letter"),
            "factor_name": c.get("factor_name"),
            "factor_type": c.get("factor_type"),
            "is_factor": c.get("is_factor", False),
            "importance": c.get("importance"),
            "keywords": c.get("keywords", []),
            "parent_chunk_id": c.get("parent_chunk_id"),
            "indexed_at_v2": "2026-04-25-factor-enrichment",
        }
    with open(OUT_DIR / "factor_payloads.json", "w") as f:
        json.dump(payloads, f, indent=2)
    print(f"  Saved factor_payloads.json")

    # ---- Write the upsert script for Aslam ----
    upsert_script = '''#!/usr/bin/env python3
"""
Targeted re-upload for the 17 factor-enriched chunks.

Run this AFTER swapping bm25_model.pkl on the backend.

Usage:
  QDRANT_URL=...  QDRANT_API_KEY=...  python upsert_17.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector


HERE = Path(__file__).parent
COLLECTION = "uk_family_law_dense"

dense = np.load(HERE / "factor_dense_vectors.npy")
chunk_ids = json.loads((HERE / "factor_chunk_ids.json").read_text())
sparse = json.loads((HERE / "factor_sparse_vectors.json").read_text())
payloads = json.loads((HERE / "factor_payloads.json").read_text())

assert len(chunk_ids) == dense.shape[0] == len(sparse) == len(payloads), "size mismatch"

url = os.environ["QDRANT_URL"]
key = os.environ.get("QDRANT_API_KEY")
client = QdrantClient(url=url, api_key=key, timeout=120)

# Map chunk_id -> point_id by scrolling existing collection.
# (point IDs are integers from the original upload; we need the same id.)
print("Looking up existing point_ids for the 17 chunks...")
cid_to_pid = {}
for cid in chunk_ids:
    res, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter={"must": [{"key": "chunk_id", "match": {"value": cid}}]},
        limit=1,
        with_payload=False,
    )
    if not res:
        print(f"  WARN: {cid} not found in collection!")
        continue
    cid_to_pid[cid] = res[0].id

print(f"  resolved {len(cid_to_pid)}/{len(chunk_ids)} point IDs")

# Build PointStructs for upsert
points = []
for i, cid in enumerate(chunk_ids):
    if cid not in cid_to_pid:
        continue
    points.append(
        PointStruct(
            id=cid_to_pid[cid],
            vector={
                "dense": dense[i].tolist(),
                "sparse": SparseVector(
                    indices=sparse[cid]["indices"],
                    values=sparse[cid]["values"],
                ),
            },
            payload=payloads[cid],
        )
    )

print(f"Upserting {len(points)} points...")
client.upsert(collection_name=COLLECTION, points=points, wait=True)
print("Done.")

# Verify
for cid in chunk_ids[:3]:
    if cid not in cid_to_pid:
        continue
    pt = client.retrieve(collection_name=COLLECTION, ids=[cid_to_pid[cid]], with_payload=True)
    print(f"  {cid} -> payload keys: {sorted((pt[0].payload or {}).keys())[:8]}...")
'''
    upsert_path = OUT_DIR / "upsert_17.py"
    upsert_path.write_text(upsert_script)
    upsert_path.chmod(0o755)
    print(f"\n  Saved {upsert_path}")

    # ---- Manifest ----
    manifest = {
        "generated_at": "2026-04-25",
        "scope": "17 curated factor chunks across MCA 1973, CA 1989, IPFDA 1975",
        "chunk_ids": factor_chunk_ids,
        "vocab_size_before": len(old_token_to_idx),
        "vocab_size_after": len(token_to_idx),
        "new_tokens_added": len(new_tokens_seen),
        "files": {
            "bm25_model.pkl": "Replace on backend GCS, restart server",
            "chunk_ids.pkl": "Unchanged copy (replace if backend reads it)",
            "factor_dense_vectors.npy": "Used by upsert_17.py",
            "factor_chunk_ids.json": "Order-aligned to factor_dense_vectors.npy",
            "factor_sparse_vectors.json": "Used by upsert_17.py",
            "factor_payloads.json": "Used by upsert_17.py",
            "upsert_17.py": "Run against prod Qdrant after pkl swap",
        },
    }
    with open(OUT_DIR / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Saved MANIFEST.json")

    print("\n" + "=" * 60)
    print(f"DONE. Output in {OUT_DIR}/")
    print("=" * 60)
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name:<35s}  {p.stat().st_size:>10,} bytes")


if __name__ == "__main__":
    main()
