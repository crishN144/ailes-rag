# AILES RAG System - Handoff Package

Hybrid retrieval system for UK family law. 310,177 chunks (303K judgments + 7K statutes) with dense (BGE) + sparse (BM25) search, RRF fusion, and cross-encoder reranking.

---

## Quick Start

### Step 1: Get the data files

Data files are not in this repo (2.3 GB). Download from Google Drive:

> **[Download hpc_outputs/ from Google Drive](https://drive.google.com/drive/folders/1jaZ_oEspGmRKcwsTTYslRIDMegwGfgCH?usp=sharing)**

```
hpc_outputs/
├── merged_all_chunks.json        (874 MB)  310,177 chunks with full metadata
├── embeddings/
│   ├── bge_embeddings.npy        (1.2 GB)  310,177 x 1024 float32 vectors
│   ├── chunk_ids.json            (13 MB)   Index-aligned chunk IDs
│   └── embedding_metadata.json   (150 B)   Model info
└── bm25_index/
    ├── bm25_model.pkl            (173 MB)  BM25Okapi sparse index
    ├── chunk_ids.pkl             (13 MB)   Index-aligned chunk IDs
    └── metadata.pkl              (21 MB)   BM25 metadata
```

Place at `~/Downloads/hpc_outputs/`.

### Step 2: Set up Qdrant

Qdrant Cloud:
```bash
# Sign up at https://cloud.qdrant.io
# Create cluster (EU-West3)
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-api-key"
```

Or self-hosted:
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
export QDRANT_URL="http://localhost:6333"
```

### Step 3: Install dependencies

```bash
pip install qdrant-client sentence-transformers google-generativeai rank-bm25
```

### Step 4: Upload vectors

```bash
python3 1_upload_to_qdrant/upload_to_qdrant.py
```

Creates collection `uk_family_law_dense` with dense (BGE 1024-dim) + sparse (BM25) vectors. Takes ~30-45 minutes.

### Step 5: Run benchmark

```bash
export GEMINI_API_KEY="your-key"
python3 3_tests_and_results/run_benchmark.py
```

Runs 20 golden queries, prints pass/fail scorecard with Recall@K, MRR, Hit Rate.

### Step 6: Run full pipeline

```bash
python3 3_tests_and_results/run_pipeline.py
python3 3_tests_and_results/run_pipeline.py --query "How are child welfare considerations assessed?"
```

Runs all 10 steps: Gemini query translation, BGE embedding, Qdrant hybrid search, RRF fusion, cross-encoder reranking, Gemini answer generation.

---

## How the Data Was Built

### Chunking (from XML)

Source: AkomaNtoso XML files from UK National Archives.

**Statutes** (7,060 chunks from 25 UK acts):
- One chunk per `<P1group>` section. No mid-section splits, no overlap.
- Metadata: title, parent hierarchy, cross-references, act/year/section.

**Judgments** (303,117 chunks from 4,621 cases):
- One chunk per `<paragraph>` in judgment body. Paragraphs under 20 words skipped.
- 2-sentence overlap from previous paragraph stored in `text_with_overlap`.
- Metadata: case_citation, paragraph_number, court, year, judge.
- Courts: EWHC-Family (2,730), EWFC (1,191), EWCOP (689). Years: 2003-2024.

Each chunk has two text fields:
- `text` - full original text (for embeddings and display)
- `bm25_text` - preprocessed for keyword search (stopwords/citations removed)

### Mistral Enrichment

Model: Mistral-Nemo-Instruct-2407 (local, CUDA, fp16, temperature=0.1).

Each judgment chunk enriched with 7 tasks:

| # | Task | Scope | Output |
|---|------|-------|--------|
| 1 | Entities | Per paragraph | People, financial amounts, properties, businesses, durations |
| 2 | Classification | Per paragraph | `paragraph_type` (ratio_decidendi / court_reasoning / background_facts / orders / procedural / legal_framework), `importance_score` (0.0-1.0) |
| 3 | Topics | Per paragraph | Primary topics, s.25 factors discussed, statute references, case law mentioned |
| 4 | Summary | Per paragraph | 100-150 word semantic summary |
| 5 | Legal Principles | Per paragraph | Principles identified, distinguishing features |
| 6 | Case Category | Per case | Category, sub-category, tags, complexity score (0-10), precedent value |
| 7 | Case Context | Per case | Marriage details, children, financial snapshot, party roles. Copied to every paragraph for retrieval consistency. |

Pipeline order: Task 7 (case context) runs once on first 3000 chars -> Tasks 1-6 per paragraph -> BM25 text generated -> merge -> BGE + BM25 indexes.

See `1_upload_to_qdrant/chunk_schema.json` for a complete chunk example with all fields.

### Dense Embeddings (BGE)

Script: `2_how_it_was_built/build_bge_embeddings.py`

- Model: BAAI/bge-large-en-v1.5 (1024 dimensions)
- Input: `chunk['text']` field
- L2 normalized so cosine similarity = dot product
- Output: `bge_embeddings.npy` (310,177 x 1024 float32)

### Sparse Index (BM25)

Script: `2_how_it_was_built/build_bm25_index.py`

- Model: BM25Okapi
- Input: `chunk['bm25_text']` field
- Vocabulary: 245,891 tokens
- Output: `bm25_model.pkl`

### Cross-Encoder (runtime only)

- Model: cross-encoder/ms-marco-MiniLM-L-12-v2 (~130 MB)
- Runs at query time on top 15 RRF candidates. CPU only.
- No data file needed. Just `pip install sentence-transformers`.

---

## Query Translation (Gemini Flash)

Script: `2_how_it_was_built/query_translation.py`

Takes Form E input, sends to Gemini Flash, gets back targeted legal queries:

```
You are a legal AI assistant analyzing a Form E financial questionnaire
for family law proceedings.

# FORM E SUMMARY
{form_e_summary}

# TASK
Generate PRECISE legal queries to retrieve relevant UK family law
statutes and case law.

CRITICAL RULES:
1. DO NOT mention statute names/numbers (search by SITUATION, not by law)
2. Focus on FACTUAL SITUATIONS described in Form E
3. Generate queries for BOTH statutes and case law
4. Extract complexity factors

# OUTPUT FORMAT (JSON)
{
  "statute_queries": [...],
  "judgment_queries": [...],
  "complexity_factors": {
    "high_net_worth": true/false,
    "business_interests": true/false,
    "overseas_assets": true/false,
    "conduct_issues": true/false,
    "pension_assets": true/false,
    "long_marriage": true/false,
    "disabled_party": true/false
  },
  "key_amounts": {
    "total_assets": <number>,
    "income_disparity_ratio": <number>,
    "property_value": <number>
  }
}
```

---

## Golden Query Benchmark

Script: `3_tests_and_results/run_benchmark.py`
Queries: `3_tests_and_results/golden_queries.json`

20 queries across 6 categories:

| Category | Queries | Covers |
|----------|---------|--------|
| Financial remedy | 10 | s.25 factors, maintenance, clean break, trust piercing, conduct, pensions, inherited assets |
| Children welfare | 3 | Welfare checklist, child wishes, paramountcy |
| Domestic abuse | 2 | Occupation orders, FLA 1996 |
| Inheritance | 2 | I(PFD)A 1975 reasonable provision |
| Negative tests | 2 | Out-of-scope queries that should return low scores |
| Form E simulation | 1 | Structured financial data as input |

Query types: 12 legal, 5 plain English, 1 Form E, 2 out-of-scope.

Metrics and targets:

| Metric | Target |
|--------|--------|
| Recall@10 | >= 0.70 |
| MRR | >= 0.50 |
| Hit Rate | >= 0.90 |

All expected document IDs verified against the 310,177 chunk corpus.

```bash
python3 3_tests_and_results/run_benchmark.py --category financial_remedy
python3 3_tests_and_results/run_benchmark.py --query-type plain_english
python3 3_tests_and_results/run_benchmark.py --save results.json
```

---

## Pipeline Architecture

```
User Query
    |
    v
[1] Query Translation (Gemini Flash)
    |  Rewrites to legal terminology
    v
[2] Dual Encoding
    |-- Dense: BGE-large-en-v1.5 (1024-dim)
    |-- Sparse: BM25 tokenization (245K vocab)
    v
[3] Qdrant Hybrid Search
    |-- Dense search -> Top 20
    |-- Sparse search -> Top 20
    v
[4] RRF Fusion (k=60)
    |  Merges dense + sparse -> 15 candidates
    v
[5] Cross-Encoder Reranking
    |  ms-marco-MiniLM-L-12-v2 -> Top 10
    v
[6] LLM Answer Generation (Gemini Flash)
    |  10 chunks as context -> answer with citations
    v
Response (answer + sources + latency)
```

---

## Repo Structure

```
ailes-rag-handoff/
├── README.md
│
├── 1_upload_to_qdrant/
│   ├── upload_to_qdrant.py            <- Upload dense + sparse vectors to Qdrant
│   └── chunk_schema.json             <- Full chunk example with all metadata fields
│
├── 2_how_it_was_built/
│   ├── build_bm25_index.py           <- BM25 sparse index builder
│   ├── build_bge_embeddings.py       <- BGE dense embedding generator
│   └── query_translation.py          <- Gemini Flash query generation
│
└── 3_tests_and_results/
    ├── run_pipeline.py               <- Full 10-step retrieval pipeline
    ├── run_benchmark.py              <- Golden query benchmark runner
    └── golden_queries.json           <- 20 verified golden queries
```

## Environment Variables

```bash
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-api-key"
export GEMINI_API_KEY="your-gemini-key"
```
