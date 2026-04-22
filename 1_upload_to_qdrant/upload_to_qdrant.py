#!/usr/bin/env python3
"""
Step 3: Upload to Qdrant with Hybrid Search (Sparse + Dense)

Bulk-upload-safe version (for 310K+ chunks on a 16GB VM):
- Streams points via a generator + client.upload_points(parallel=4) so RAM stays flat
- np.load(..., mmap_mode="r") so embeddings don't copy into RAM
- Real BM25 vocab + IDF modifier (Qdrant applies IDF server-side)
- HNSW disabled during upload (m=0), then re-enabled after
- RRF hybrid test query at the end
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    Modifier,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm


BATCH_SIZE = 1000      # Qdrant docs recommend 1000–10000 for bulk loads
# parallel=1 avoids qdrant-client issue #459 (generator + parallel>1 can
# raise "cannot pickle 'generator' object"). Streaming + batch=1000 is
# fast enough — no multiprocessing risk.
PARALLEL = 1


def main():
    print("\n" + "=" * 100)
    print("UPLOADING TO QDRANT WITH HYBRID SEARCH (streaming)")
    print("=" * 100 + "\n")

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    base_dir = Path.home() / "Downloads" / "hpc_outputs"
    merged_file = base_dir / "merged_all_chunks.json"
    embeddings_file = base_dir / "embeddings" / "bge_embeddings.npy"
    bm25_dir = base_dir / "bm25_index"

    collection_name = "uk_family_law_dense"

    print("Configuration:")
    print(f"  Qdrant URL: {qdrant_url}")
    print(f"  Collection: {collection_name}")
    print(f"  Batch size: {BATCH_SIZE}  Parallel: {PARALLEL}\n")

    # Connect
    print("Connecting to Qdrant...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)
    print("   Connected\n")

    # Load data
    print("Loading data...")
    with open(merged_file, "r") as f:
        data = json.load(f)
    chunks = data["chunks"]
    del data  # free ~800MB outer dict; we only need `chunks`
    print(f"   Loaded {len(chunks):,} chunks")

    # mmap the embeddings - does NOT copy the full array into RAM
    embeddings = np.load(embeddings_file, mmap_mode="r")
    print(f"   Loaded embeddings (mmap): {embeddings.shape}")

    with open(bm25_dir / "bm25_model.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(bm25_dir / "chunk_ids.pkl", "rb") as f:
        chunk_ids = pickle.load(f)
    print(f"   Loaded BM25 index\n")

    assert len(chunks) == len(embeddings) == len(chunk_ids), "Data size mismatch!"

    # Create collection
    print(f"Setting up Qdrant collection: {collection_name}")
    try:
        client.delete_collection(collection_name=collection_name)
        print("   Deleted existing collection")
    except (UnexpectedResponse, ValueError):
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=embeddings.shape[1],
                distance=Distance.COSINE,
                on_disk=True,  # keep dense vectors on disk during bulk load
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,  # apply IDF server-side
            )
        },
        # Disable HNSW construction during upload (m=0), re-enable after
        hnsw_config=HnswConfigDiff(m=0),
    )
    print("   Created collection (HNSW disabled, on_disk dense, IDF modifier)\n")

    # Build BM25 vocab -> stable integer index
    print("Building BM25 vocabulary index...")
    token_to_idx = {token: idx for idx, token in enumerate(bm25.idf.keys())}
    print(f"   Vocab size: {len(token_to_idx):,} tokens\n")

    # Streaming point generator - no full list built in RAM.
    # Wrapped in tqdm so we see progress during the upload_points call.
    def point_generator():
        for i, chunk in enumerate(tqdm(chunks, desc="Streaming", unit="pt")):
            # Raw term frequencies (Qdrant applies IDF via Modifier.IDF)
            index_scores = {}
            for token, freq in bm25.doc_freqs[i].items():
                idx = token_to_idx[token]
                index_scores[idx] = index_scores.get(idx, 0.0) + float(freq)

            yield PointStruct(
                id=i,
                vector={
                    "dense": embeddings[i].tolist(),
                    "sparse": SparseVector(
                        indices=list(index_scores.keys()),
                        values=list(index_scores.values()),
                    ),
                },
                payload={
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text"),
                    "bm25_text": chunk.get("bm25_text"),
                    "doc_type": chunk.get("doc_type", chunk.get("court", "statute")),
                    "court": chunk.get("court"),
                    "year": chunk.get("year", chunk.get("statute_year")),
                    "case_citation": chunk.get("case_citation"),
                    "statute_title": chunk.get("statute_title"),
                    "section_number": chunk.get("section_number"),
                    "paragraph_type": chunk.get("paragraph_type"),
                    "importance_score": chunk.get("importance_score"),
                    "topics": chunk.get("topics", []),
                },
            )

    # Streaming bulk upload - handles batching, retries, parallelism
    print(f"Uploading {len(chunks):,} points (streaming, parallel={PARALLEL})...")
    client.upload_points(
        collection_name=collection_name,
        points=point_generator(),
        batch_size=BATCH_SIZE,
        parallel=PARALLEL,
        max_retries=3,
        wait=True,
    )
    print("   Upload complete\n")

    # Re-enable HNSW now that all points are in
    print("Re-enabling HNSW indexing...")
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=HnswConfigDiff(m=16),
    )
    print("   HNSW m=16 set - Qdrant will build the index in the background\n")

    # Verify
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"Collection point count: {collection_info.points_count:,}")
    print(f"Expected: {len(chunks):,}")
    print(f"Match: {'YES' if collection_info.points_count == len(chunks) else 'NO'}\n")

    # Hybrid RRF test
    print("\n" + "=" * 100)
    print("TEST HYBRID SEARCH (RRF)")
    print("=" * 100 + "\n")

    # Lazy-import sentence_transformers so the upload itself doesn't fail
    # if the library is missing on the VM.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    test_query = "What factors does the court consider for child welfare?"
    query_embedding = model.encode([test_query], normalize_embeddings=True)[0].tolist()

    # Query-side sparse: raw TFs (Qdrant applies IDF via the modifier)
    query_tokens = test_query.lower().split()
    query_index_scores = {}
    for token in query_tokens:
        if token in token_to_idx:
            idx = token_to_idx[token]
            query_index_scores[idx] = query_index_scores.get(idx, 0.0) + 1.0

    query_sparse = SparseVector(
        indices=list(query_index_scores.keys()),
        values=list(query_index_scores.values()),
    )

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=query_embedding, using="dense", limit=20),
            Prefetch(query=query_sparse, using="sparse", limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=5,
        with_payload=True,
    )

    print(f"Query: '{test_query}'\n")
    print("Top 5 results (hybrid, RRF fusion):")
    for i, result in enumerate(results.points, 1):
        payload = result.payload
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   ID: {payload.get('chunk_id')}")
        print(f"   Type: {payload.get('doc_type')}")
        print(f"   Text: {payload.get('text', '')[:150]}...")

    print("\n" + "=" * 100)
    print("QDRANT UPLOAD COMPLETE - HYBRID SEARCH READY")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
