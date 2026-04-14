#!/usr/bin/env python3
"""
Step 2: Generate BGE embeddings for all chunks
- Uses BGE-large model for dense retrieval
- Generates embeddings from 'text' field
- Processes in batches to handle 310K+ chunks
- Saves embeddings as numpy array
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

def main():
    print("\n" + "="*100)
    print("GENERATING BGE EMBEDDINGS")
    print("="*100 + "\n")

    # Configuration
    base_dir = Path.home() / 'Downloads' / 'hpc_outputs'
    merged_file = base_dir / 'merged_all_chunks.json'
    output_dir = base_dir / 'embeddings'
    output_dir.mkdir(exist_ok=True, parents=True)

    model_name = 'BAAI/bge-large-en-v1.5'  # Best BGE model for English
    batch_size = 32  # Adjust based on GPU memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}\n")

    # Load model
    print(f"📥 Loading BGE model...")
    model = SentenceTransformer(model_name, device=device)
    print(f"   ✅ Model loaded on {device}\n")

    # Load merged data
    print(f"📂 Loading merged chunks from: {merged_file}")
    with open(merged_file, 'r') as f:
        data = json.load(f)

    chunks = data['chunks']
    total_chunks = len(chunks)
    print(f"   Total chunks: {total_chunks:,}\n")

    # Extract text from all chunks
    print("📝 Extracting text from all chunks...")
    texts = []
    chunk_ids = []

    for chunk in tqdm(chunks, desc="Extracting"):
        text = chunk.get('text', '')
        chunk_id = chunk.get('chunk_id', 'unknown')

        texts.append(text)
        chunk_ids.append(chunk_id)

    print(f"   Extracted {len(texts):,} texts\n")

    # Generate embeddings in batches
    print(f"🧠 Generating embeddings (batch size: {batch_size})...")
    print(f"   This will take some time for {total_chunks:,} chunks...\n")

    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", total=num_batches):
        batch_texts = texts[i:i+batch_size]

        # Encode batch
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        all_embeddings.append(batch_embeddings)

        # Progress update every 100 batches
        if (i // batch_size) % 100 == 0:
            current_batch = i // batch_size
            print(f"   Processed {current_batch}/{num_batches} batches ({current_batch/num_batches*100:.1f}%)")

    # Concatenate all embeddings
    print(f"\n🔗 Concatenating embeddings...")
    embeddings = np.vstack(all_embeddings)
    print(f"   ✅ Final shape: {embeddings.shape}\n")

    # Save embeddings
    print(f"💾 Saving embeddings...")

    # Save as numpy array
    embeddings_file = output_dir / 'bge_embeddings.npy'
    np.save(embeddings_file, embeddings)
    print(f"   ✅ Saved embeddings: {embeddings_file}")

    # Save chunk IDs
    ids_file = output_dir / 'chunk_ids.json'
    with open(ids_file, 'w') as f:
        json.dump(chunk_ids, f)
    print(f"   ✅ Saved chunk IDs: {ids_file}")

    # Save embedding metadata
    metadata = {
        'model': model_name,
        'num_chunks': len(texts),
        'embedding_dim': embeddings.shape[1],
        'normalized': True,
        'device': device,
        'batch_size': batch_size
    }

    metadata_file = output_dir / 'embedding_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Saved metadata: {metadata_file}")

    # Statistics
    print(f"\n" + "="*100)
    print(f"📊 EMBEDDING STATISTICS")
    print("="*100 + "\n")

    print(f"Total embeddings: {len(embeddings):,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Data type: {embeddings.dtype}")
    print(f"File size: {embeddings_file.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Average text length: {sum(len(t) for t in texts) / len(texts):.1f} characters")

    # Test similarity
    print(f"\n" + "="*100)
    print(f"🧪 TEST SIMILARITY SEARCH")
    print("="*100 + "\n")

    test_query = "What factors does the court consider for financial provision?"
    query_embedding = model.encode([test_query], normalize_embeddings=True)[0]

    # Compute cosine similarities
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-5:][::-1]

    print(f"Query: '{test_query}'\n")
    print(f"Top 5 most similar chunks:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Score: {similarities[idx]:.4f} | ID: {chunk_ids[idx]}")
        print(f"     Text preview: {texts[idx][:150]}...\n")

    print("="*100)
    print("✅ BGE EMBEDDINGS READY FOR UPLOAD")
    print("="*100 + "\n")

if __name__ == '__main__':
    main()
