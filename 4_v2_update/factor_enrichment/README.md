# Factor-Chunk BM25 + Dense Enrichment

Targeted retrieval fix for the 17 curated UK family-law factor chunks
(MCA 1973 s.25(2)(a)–(h), CA 1989 s.1(3)(a)–(g), IPFDA 1975 s.3(2)(a)–(b)).

## Why

Short statute factor chunks (70–400 chars) were getting buried in retrieval
because longer case-law paragraphs that quoted them won on both BM25 (more
term occurrences) and dense (richer embedding signal). Result: the headline
statute never reached the cross-encoder.

Verified locally on realistic user queries:

| Metric                   | Baseline | Enriched |
|--------------------------|----------|----------|
| RRF median rank          | 101      | **15**   |
| Chunks reaching top-15   | 2/17     | **9/17** |
| Chunks reaching top-30   | 6/17     | **11/17**|

Net positive across 17 chunks. Five queries with no statute-vocab overlap
still miss top-15 — those need expander prompt fixes, not retrieval changes.

## What's in this folder

| File                          | Size   | Purpose                                        |
|-------------------------------|--------|------------------------------------------------|
| `prep_factor_enrichment.py`   | 15 KB  | Generator script — produces all artifacts      |
| `verify_enrichment.py`        | 4.6 KB | Local rank-delta verification                  |
| `upsert_17.py`                | 2.4 KB | **Aslam runs this** against prod Qdrant       |
| `factor_chunk_ids.json`       | 521 B  | Order-aligned chunk_id list (17 entries)       |
| `factor_dense_vectors.npy`    | 68 KB  | (17, 1024) BGE-large vectors                   |
| `factor_sparse_vectors.json`  | 18 KB  | Sparse vectors per chunk (token_id → raw freq) |
| `factor_payloads.json`        | 27 KB  | Full enriched payload per chunk                |
| `MANIFEST.json`               | 1.2 KB | Build provenance                               |

**Not in this folder (too big for GitHub, ~182MB):**
- `bm25_model.pkl` — uploaded directly to GCS (see deployment instructions)

## Deployment (Aslam)

### 1. Replace BM25 model on backend

I'll upload the new `bm25_model.pkl` to:
```
gs://ailes-prompts-config/bm25_model.pkl
```

(or wherever Aslam reads it from — confirm the bucket path)

After upload, restart the backend service so it reloads the pkl.

### 2. Upsert the 17 enriched chunks in Qdrant

```bash
cd 4_v2_update/factor_enrichment/
QDRANT_URL=<prod-url> QDRANT_API_KEY=<key> python upsert_17.py
```

This calls Qdrant's `upsert` API for the 17 specific point_ids that
correspond to the factor chunks. Updates their dense + sparse + payload.
Other 310,160 chunks are untouched. Takes ~5 minutes.

### Stable-vocab guarantee

Vocab indices are **backwards-compatible**:
- 285,928 existing tokens keep their original indices
- 9 brand-new tokens (mostly punctuation: `s.`, `1(g)`, etc.) appended at end

This means existing 310,160 chunks' sparse vectors in Qdrant remain valid
without re-upload. Qdrant computes IDF server-side from corpus stats, so it
adjusts naturally to the slightly-shifted statistics from the 17 changed docs.

## Test after deployment

Run any of these queries — should now retrieve the matching factor chunk:

| Query                                                  | Should retrieve                  |
|--------------------------------------------------------|----------------------------------|
| "What financial resources will the court consider?"    | mca-1973-section-25(2)(a)        |
| "Pension benefits lost from divorce?"                  | mca-1973-section-25(2)(h)        |
| "What does my child want?"                             | ca-1989-section-1(3)(a)          |
| "Has my child suffered any harm?"                      | ca-1989-section-1(3)(e)          |

Check Langfuse trace → `rag.rag_rerank` → Output → look for chunk_id in top 10.

## Reproducibility

To regenerate all artifacts from scratch:

```bash
cd 4_v2_update/factor_enrichment/
python prep_factor_enrichment.py    # produces all output files
python verify_enrichment.py         # runs local rank-delta validation
```

Both scripts read from `~/Downloads/hpc_outputs/` — point them at your local
chunks/embeddings/bm25 paths if different.
