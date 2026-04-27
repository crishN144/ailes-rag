#!/usr/bin/env python3
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
