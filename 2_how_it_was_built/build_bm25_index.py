#!/usr/bin/env python3
"""
Step 1: Build BM25 index from merged chunks
- Uses rank_bm25 library for sparse retrieval
- Indexes the 'bm25_text' field from all 310K+ chunks
- Saves index for fast lexical search
"""

import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def main():
    print("\n" + "="*100)
    print("BUILDING BM25 INDEX")
    print("="*100 + "\n")

    # Paths
    base_dir = Path.home() / 'Downloads' / 'hpc_outputs'
    merged_file = base_dir / 'merged_all_chunks.json'
    output_dir = base_dir / 'bm25_index'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load merged data
    print(f"📂 Loading merged chunks from: {merged_file}")
    with open(merged_file, 'r') as f:
        data = json.load(f)

    chunks = data['chunks']
    total_chunks = len(chunks)
    print(f"   Total chunks: {total_chunks:,}\n")

    # Extract bm25_text and tokenize
    print("🔤 Tokenizing bm25_text for all chunks...")
    tokenized_corpus = []
    chunk_ids = []
    metadata = []

    for chunk in tqdm(chunks, desc="Tokenizing"):
        bm25_text = chunk.get('bm25_text', '')
        chunk_id = chunk.get('chunk_id', 'unknown')

        # Simple whitespace tokenization (BM25 handles this)
        tokens = bm25_text.split()

        tokenized_corpus.append(tokens)
        chunk_ids.append(chunk_id)

        # Store minimal metadata for retrieval
        metadata.append({
            'chunk_id': chunk_id,
            'doc_type': chunk.get('doc_type', chunk.get('court', 'statute')),
            'year': chunk.get('year', chunk.get('statute_year', 'unknown'))
        })

    print(f"   Tokenized {len(tokenized_corpus):,} documents\n")

    # Build BM25 index
    print("🏗️  Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"   ✅ BM25 index built\n")

    # Save BM25 index
    print("💾 Saving BM25 index components...")

    # Save BM25 model
    bm25_file = output_dir / 'bm25_model.pkl'
    with open(bm25_file, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"   ✅ Saved BM25 model: {bm25_file}")

    # Save chunk IDs
    ids_file = output_dir / 'chunk_ids.pkl'
    with open(ids_file, 'wb') as f:
        pickle.dump(chunk_ids, f)
    print(f"   ✅ Saved chunk IDs: {ids_file}")

    # Save metadata
    metadata_file = output_dir / 'metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✅ Saved metadata: {metadata_file}")

    # Statistics
    print(f"\n" + "="*100)
    print(f"📊 BM25 INDEX STATISTICS")
    print("="*100 + "\n")

    print(f"Total documents indexed: {len(tokenized_corpus):,}")
    print(f"Average tokens per document: {sum(len(doc) for doc in tokenized_corpus) / len(tokenized_corpus):.1f}")
    print(f"Total tokens: {sum(len(doc) for doc in tokenized_corpus):,}")
    print(f"\nIndex components saved to: {output_dir}")

    # Test search
    print(f"\n" + "="*100)
    print(f"🧪 TEST SEARCH")
    print("="*100 + "\n")

    test_queries = [
        "financial provision section 25",
        "child welfare best interests",
        "domestic abuse protection order"
    ]

    for query in test_queries:
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        # Get top 3 results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

        print(f"Query: '{query}'")
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. Score: {scores[idx]:.4f} | ID: {chunk_ids[idx]} | Type: {metadata[idx]['doc_type']}")
        print()

    print("="*100)
    print("✅ BM25 INDEX READY FOR USE")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
