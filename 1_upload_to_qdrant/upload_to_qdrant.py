#!/usr/bin/env python3
"""
Step 3: Upload to Qdrant with Hybrid Search (Sparse + Dense)
- Uploads both BM25 sparse vectors and BGE dense embeddings
- Enables RRF (Reciprocal Rank Fusion) for hybrid retrieval
- Batch uploads for 310K+ chunks
"""

import json
import numpy as np
import pickle
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams, SparseVector,
    Prefetch, FusionQuery, Fusion,
)
from tqdm import tqdm
import os

def main():
    print("\n" + "="*100)
    print("UPLOADING TO QDRANT WITH HYBRID SEARCH")
    print("="*100 + "\n")

    # Configuration
    QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

    base_dir = Path.home() / 'Downloads' / 'hpc_outputs'
    merged_file = base_dir / 'merged_all_chunks.json'
    embeddings_file = base_dir / 'embeddings' / 'bge_embeddings.npy'
    bm25_dir = base_dir / 'bm25_index'

    collection_name = 'uk_family_law_dense'
    batch_size = 100  # Qdrant batch size

    print(f"Configuration:")
    print(f"  Qdrant URL: {QDRANT_URL}")
    print(f"  Collection: {collection_name}")
    print(f"  Batch size: {batch_size}\n")

    # Connect to Qdrant
    print(f"🔗 Connecting to Qdrant...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
    print(f"   ✅ Connected\n")

    # Load data
    print(f"📂 Loading data...")

    # Load merged chunks
    with open(merged_file, 'r') as f:
        data = json.load(f)
    chunks = data['chunks']
    print(f"   ✅ Loaded {len(chunks):,} chunks")

    # Load BGE embeddings
    embeddings = np.load(embeddings_file)
    print(f"   ✅ Loaded embeddings: {embeddings.shape}")

    # Load BM25 index
    with open(bm25_dir / 'bm25_model.pkl', 'rb') as f:
        bm25 = pickle.load(f)
    with open(bm25_dir / 'chunk_ids.pkl', 'rb') as f:
        chunk_ids = pickle.load(f)
    print(f"   ✅ Loaded BM25 index\n")

    # Verify counts match
    assert len(chunks) == len(embeddings) == len(chunk_ids), "Data size mismatch!"

    # Create or recreate collection
    print(f"🏗️  Setting up Qdrant collection: {collection_name}")

    # Delete if exists (ignore "not found" — collection just didn't exist yet)
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"   Deleted existing collection")
    except (UnexpectedResponse, ValueError):
        pass

    # Create collection with dense + sparse vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=embeddings.shape[1],  # BGE dimension (1024)
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,  # Keep in memory for speed
                )
            )
        }
    )
    print(f"   ✅ Created collection with dense + sparse vectors\n")

    # Build token -> index mapping from the BM25 vocabulary.
    # This gives each token in the BM25 vocab a stable unique integer id,
    # which Qdrant requires for sparse vectors (no collisions allowed).
    print(f"🔧 Building BM25 vocabulary index...")
    token_to_idx = {token: idx for idx, token in enumerate(bm25.idf.keys())}
    print(f"   ✅ Vocabulary size: {len(token_to_idx):,} tokens\n")

    # Prepare points for upload
    print(f"📦 Preparing points for upload...")

    points = []
    for i, chunk in enumerate(tqdm(chunks, desc="Preparing")):
        # Dense vector (BGE embedding)
        dense_vector = embeddings[i].tolist()

        # Sparse vector (BM25 TF-IDF)
        # Use the actual BM25 model's per-doc frequencies and IDF scores.
        # Accumulate into a dict keyed by vocab index so duplicate
        # tokens (or index collisions) are summed, not duplicated —
        # Qdrant rejects sparse vectors with duplicate indices (422).
        # doc_freqs tokens are always a subset of the BM25 vocab (token_to_idx
        # is built from that same vocab), so we can skip an `if token in ...`
        # guard. Using .get() on index_scores handles any rare duplicate.
        index_scores = {}
        doc_freqs = bm25.doc_freqs[i]  # {token: freq in this doc}
        for token, freq in doc_freqs.items():
            idx = token_to_idx[token]
            idf = bm25.idf[token]
            score = freq * idf  # TF-IDF
            index_scores[idx] = index_scores.get(idx, 0.0) + score

        sparse_indices = list(index_scores.keys())
        sparse_values = list(index_scores.values())

        # Create point
        point = PointStruct(
            id=i,
            vector={
                "dense": dense_vector,
                "sparse": {
                    "indices": sparse_indices,
                    "values": sparse_values
                }
            },
            payload={
                "chunk_id": chunk.get('chunk_id'),
                "text": chunk.get('text'),
                "bm25_text": chunk.get('bm25_text'),
                "doc_type": chunk.get('doc_type', chunk.get('court', 'statute')),
                "court": chunk.get('court'),
                "year": chunk.get('year', chunk.get('statute_year')),
                "case_citation": chunk.get('case_citation'),
                "statute_title": chunk.get('statute_title'),
                "section_number": chunk.get('section_number'),
                "paragraph_type": chunk.get('paragraph_type'),
                "importance_score": chunk.get('importance_score'),
                "topics": chunk.get('topics', []),
            }
        )

        points.append(point)

    print(f"   ✅ Prepared {len(points):,} points\n")

    # Upload in batches
    print(f"📤 Uploading to Qdrant (batch size: {batch_size})...")

    num_batches = (len(points) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(points), batch_size), desc="Uploading", total=num_batches):
        batch = points[i:i+batch_size]

        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True
        )

    print(f"\n   ✅ Upload complete!\n")

    # Verify upload
    print(f"🔍 Verifying upload...")
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"   Collection point count: {collection_info.points_count:,}")
    print(f"   Expected: {len(points):,}")
    print(f"   Match: {'✅ YES' if collection_info.points_count == len(points) else '❌ NO'}\n")

    # Test hybrid search
    print(f"\n" + "="*100)
    print(f"🧪 TEST HYBRID SEARCH")
    print("="*100 + "\n")

    from sentence_transformers import SentenceTransformer

    # Load model for test query
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    test_query = "What factors does the court consider for child welfare?"
    query_embedding = model.encode([test_query], normalize_embeddings=True)[0].tolist()

    # Build a sparse query vector from the BM25 vocab (same mapping as upload)
    query_tokens = test_query.lower().split()
    query_index_scores = {}
    for token in query_tokens:
        if token in token_to_idx:
            idx = token_to_idx[token]
            # Query-side weight is just the token's IDF
            query_index_scores[idx] = bm25.idf[token]

    query_sparse = SparseVector(
        indices=list(query_index_scores.keys()),
        values=list(query_index_scores.values()),
    )

    # Hybrid search: dense + sparse fused with RRF, server-side
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
    print(f"Top 5 results (hybrid, RRF fusion):")
    for i, result in enumerate(results.points, 1):
        payload = result.payload
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   ID: {payload.get('chunk_id')}")
        print(f"   Type: {payload.get('doc_type')}")
        print(f"   Text: {payload.get('text', '')[:150]}...")

    print(f"\n" + "="*100)
    print(f"✅ QDRANT UPLOAD COMPLETE - HYBRID SEARCH READY!")
    print("="*100 + "\n")

    print(f"Collection details:")
    print(f"  Name: {collection_name}")
    print(f"  Points: {collection_info.points_count:,}")
    print(f"  Vectors: Dense (BGE) + Sparse (BM25)")
    print(f"  Search modes: Vector, BM25, Hybrid (RRF)")

if __name__ == '__main__':
    main()
