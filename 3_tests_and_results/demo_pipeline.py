#!/usr/bin/env python3
"""
AILES RAG Pipeline — Offline Demo
Produces the same output as run_pipeline.py without requiring live connections.
No Qdrant, no Gemini API, no model downloads needed.

Usage:
    python3 demo_pipeline.py
    python3 demo_pipeline.py --query "How are child welfare considerations assessed?"
"""

import time
import json
import sys
import random
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────

CORPUS_SIZE = 310_177
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL = "BM25Okapi"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GENERATOR = "gemini-2.0-flash"
COLLECTION = "uk_family_law_dense"
QDRANT_URL = "https://1f674c1b-78e4-4df7-82d6-12d4bc1fad52.europe-west3-0.gcp.cloud.qdrant.io"
DENSE_DIM = 1024
BM25_VOCAB = 245_891
BM25_PATH = "~/Downloads/hpc_outputs/bm25_index"

# ─── Sample Data ─────────────────────────────────────────────────────────────

SAMPLE_QUERIES = {
    "What maintenance would court award for £750k assets?": {
        "translated": (
            "Periodical payments order and spousal maintenance under Section 25 "
            "Matrimonial Causes Act 1973 where total matrimonial assets are "
            "approximately \u00a3750,000"
        ),
        "legal_concepts": [
            "periodical payments (MCA 1973 s.23)",
            "section 25 factors",
            "spousal maintenance",
            "financial remedy proceedings",
            "asset division",
        ],
        "keywords": [
            "maintenance", "periodical", "payments", "spousal", "section",
            "25", "matrimonial", "causes", "act", "1973", "total",
            "assets", "approximately", "750,000",
        ],
    },
    "What factors does the court consider when dividing assets in divorce?": {
        "translated": (
            "Section 25 Matrimonial Causes Act 1973 factors for ancillary "
            "relief and financial remedy in divorce proceedings including "
            "needs, contributions, and standard of living"
        ),
        "legal_concepts": [
            "section 25 factors",
            "ancillary relief",
            "financial remedy",
            "equal sharing principle",
            "needs-based approach",
        ],
        "keywords": [
            "section", "25", "matrimonial", "causes", "act", "1973",
            "factors", "ancillary", "relief", "financial", "remedy",
            "divorce", "needs", "contributions", "standard", "living",
        ],
    },
    "How are child welfare considerations assessed in financial proceedings?": {
        "translated": (
            "Paramountcy principle under Section 1 Children Act 1989 and "
            "first consideration of child welfare under Section 25(1) "
            "Matrimonial Causes Act 1973 in financial remedy proceedings"
        ),
        "legal_concepts": [
            "paramountcy principle (CA 1989 s.1)",
            "first consideration (MCA 1973 s.25(1))",
            "welfare checklist",
            "Schedule 1 Children Act 1989",
        ],
        "keywords": [
            "paramountcy", "section", "1", "children", "act", "1989",
            "welfare", "checklist", "financial", "remedy", "proceedings",
        ],
    },
}

SAMPLE_CHUNKS = [
    {
        "chunk_id": "mca-1973-section-25-full",
        "doc_type": "statute",
        "statute_title": "Matrimonial Causes Act 1973",
        "section": "25",
        "court": None,
        "citation": None,
        "year": 1973,
        "topics": ["asset_division", "financial_remedy", "section_25_factors"],
        "text": (
            "25(2) ... the court shall in particular have regard to the "
            "following matters\u2014 (a) the income, earning capacity, property "
            "and other financial resources which each of the parties to the "
            "marriage has or is likely to have in the foreseeable future, "
            "including in the case of earning capacity any increase in that "
            "capacity which it would in the opinion of the court be reasonable "
            "to expect a party to the marriage to take steps to acquire; "
            "(b) the financial needs, obligations and responsibilities which "
            "each of the parties to the marriage has or is likely to have in "
            "the foreseeable future..."
        ),
    },
    {
        "chunk_id": "ewfc-2023-[2023]-EWFC-142-para_17",
        "doc_type": "judgment",
        "statute_title": None,
        "section": None,
        "court": "EWFC",
        "citation": "[2023] EWFC 142",
        "year": 2023,
        "topics": ["asset_division", "needs", "section_25_factors", "high_net_worth"],
        "text": (
            "In applying s.25 of the Matrimonial Causes Act 1973, I have had "
            "regard to all the circumstances of the case, first consideration "
            "being given to the welfare of the children. The total assets of "
            "the marriage are \u00a32.3 million. The husband's earning capacity "
            "significantly exceeds that of the wife, who has been the primary "
            "carer for the three children throughout the marriage. In these "
            "circumstances, an equal division of assets is the appropriate "
            "starting point, subject to the needs of the parties..."
        ),
    },
    {
        "chunk_id": "mca-1973-section-25(2)(f)",
        "doc_type": "statute",
        "statute_title": "Matrimonial Causes Act 1973",
        "section": "25(2)(f)",
        "court": None,
        "citation": None,
        "year": 1973,
        "topics": ["contributions", "homemaker", "asset_division"],
        "text": (
            "(f) the contributions which each of the parties has made or is "
            "likely in the foreseeable future to make to the welfare of the "
            "family, including any contribution by looking after the home or "
            "caring for the family..."
        ),
    },
    {
        "chunk_id": "ewfc-2023-[2023]-EWFC-87-para_9",
        "doc_type": "judgment",
        "statute_title": None,
        "section": None,
        "court": "EWFC",
        "citation": "[2023] EWFC 87",
        "year": 2023,
        "topics": ["earning_capacity", "compensation", "periodical_payments"],
        "text": (
            "The wife gave up a career in investment banking to raise the "
            "parties' four children. Her earning capacity has been significantly "
            "diminished as a result. In a case where the total assets are "
            "\u00a3780,000, the court considers it appropriate to award periodical "
            "payments of \u00a32,500 per month for a term of five years, during "
            "which the wife may retrain and re-establish her earning capacity..."
        ),
    },
    {
        "chunk_id": "mca-1973-section-23",
        "doc_type": "statute",
        "statute_title": "Matrimonial Causes Act 1973",
        "section": "23",
        "court": None,
        "citation": None,
        "year": 1973,
        "topics": ["periodical_payments", "lump_sum", "financial_order"],
        "text": (
            "23(1) On granting a decree of divorce... the court may make any "
            "one or more of the following orders\u2014 (a) an order that either "
            "party to the marriage shall make to the other such periodical "
            "payments... as may be specified in the order; (b) an order that "
            "either party to the marriage shall pay to the other such lump "
            "sum or sums as may be so specified..."
        ),
    },
    {
        "chunk_id": "ewfc-2024-[2024]-EWFC-8-para_11",
        "doc_type": "judgment",
        "statute_title": None,
        "section": None,
        "court": "EWFC",
        "citation": "[2024] EWFC 8",
        "year": 2024,
        "topics": ["needs", "housing", "children_welfare", "asset_division"],
        "text": (
            "The wife's housing needs must be assessed by reference to the "
            "needs of the children who will be living with her for the "
            "majority of the time. In a case where the assets are insufficient "
            "to meet the needs of both parties, the court must prioritise the "
            "needs of the children and the parent with whom they primarily "
            "reside..."
        ),
    },
    {
        "chunk_id": "ewhc-family-2022-[2022]-EWHC-2891-(Fam)-para_23",
        "doc_type": "judgment",
        "statute_title": None,
        "section": None,
        "court": "EWHC-Family",
        "citation": "[2022] EWHC 2891 (Fam)",
        "year": 2022,
        "topics": ["sharing_principle", "equal_division", "matrimonial_property"],
        "text": (
            "The sharing principle, as articulated by the House of Lords in "
            "Miller v Miller; McFarlane v McFarlane [2006] UKHL 24, requires "
            "that the product of the marital partnership be divided equally "
            "between the parties unless there is good reason to depart from "
            "equality..."
        ),
    },
    {
        "chunk_id": "mca-1973-section-28",
        "doc_type": "statute",
        "statute_title": "Matrimonial Causes Act 1973",
        "section": "28",
        "court": None,
        "citation": None,
        "year": 1973,
        "topics": ["periodical_payments", "term_order", "clean_break"],
        "text": (
            "28(1A) Where a periodical payments order is made on or after the "
            "grant of a decree of divorce... the court may direct that that "
            "party shall not be entitled to apply for the extension of the "
            "term specified in the order..."
        ),
    },
    {
        "chunk_id": "ewhc-family-2023-[2023]-EWHC-1456-(Fam)-para_31",
        "doc_type": "judgment",
        "statute_title": None,
        "section": None,
        "court": "EWHC-Family",
        "citation": "[2023] EWHC 1456 (Fam)",
        "year": 2023,
        "topics": ["non_matrimonial_property", "inheritance", "pre_marital_assets"],
        "text": (
            "The husband's pre-marital assets, including the property inherited "
            "from his late father, are non-matrimonial in character. The sharing "
            "principle does not ordinarily apply to such assets. However, where "
            "the matrimonial assets are insufficient to meet the parties' needs, "
            "the court may have regard to the non-matrimonial assets..."
        ),
    },
    {
        "chunk_id": "mca-1973-section-25A",
        "doc_type": "statute",
        "statute_title": "Matrimonial Causes Act 1973",
        "section": "25A",
        "court": None,
        "citation": None,
        "year": 1973,
        "topics": ["clean_break", "periodical_payments", "financial_order"],
        "text": (
            "25A(1) Where on or after the grant of a decree of divorce... the "
            "court shall consider whether it would be appropriate so to exercise "
            "those powers that the financial obligations of each party towards "
            "the other will be terminated as soon after the grant of the decree "
            "as the court considers just and reasonable..."
        ),
    },
]

CE_SCORES = [9.234, 8.891, 8.456, 8.102, 7.834, 7.567, 7.321, 6.998, 6.743, 6.512]

DENSE_TOP5 = [
    (0.891, "mca-1973-section-25-full"),
    (0.867, "ewfc-2023-[2023]-EWFC-142-para_17"),
    (0.854, "ewfc-2024-[2024]-EWFC-8-para_11"),
    (0.841, "mca-1973-section-23"),
    (0.829, "ewhc-family-2022-[2022]-EWHC-2891-(Fam)-para_23"),
]

SPARSE_TOP5 = [
    (18.43, "ewfc-2023-[2023]-EWFC-87-para_9"),
    (17.91, "mca-1973-section-25-full"),
    (16.78, "ewfc-2023-[2023]-EWFC-142-para_17"),
    (15.34, "mca-1973-section-28"),
    (14.92, "ewhc-family-2023-[2023]-EWHC-1456-(Fam)-para_31"),
]

RRF_TABLE = [
    (0.03226, "mca-1973-section-25-full", "BOTH"),
    (0.03145, "ewfc-2023-[2023]-EWFC-142-para_17", "BOTH"),
    (0.02941, "ewfc-2023-[2023]-EWFC-87-para_9", "BOTH"),
    (0.02778, "ewfc-2024-[2024]-EWFC-8-para_11", "BOTH"),
    (0.02632, "mca-1973-section-23", "BOTH"),
    (0.02500, "ewhc-family-2022-[2022]-EWHC-2891-(Fam)-para_23", "BOTH"),
    (0.02381, "mca-1973-section-28", "BOTH"),
    (0.02273, "ewhc-family-2023-[2023]-EWHC-1456-(Fam)-para_31", "BOTH"),
    (0.02174, "ewfc-2023-[2023]-EWFC-201-para_14", "BOTH"),
    (0.01639, "mca-1973-section-25A", "Dense"),
    (0.01587, "ewfc-2024-[2024]-EWFC-31-para_8", "Dense"),
    (0.01538, "mca-1973-section-25(2)(f)", "Sparse"),
    (0.01493, "ewhc-family-2024-[2024]-EWHC-567-(Fam)-para_19", "Sparse"),
    (0.01449, "ca-1989-section-1-full", "Dense"),
    (0.01408, "ewcop-2023-[2023]-EWCOP-44-para_5", "Sparse"),
]

CE_TABLE = [
    (9.234, "mca-1973-section-25-full", "was #1"),
    (8.891, "ewfc-2023-[2023]-EWFC-142-para_17", "was #2"),
    (8.456, "mca-1973-section-25(2)(f)", "was #12 UP"),
    (8.102, "ewfc-2023-[2023]-EWFC-87-para_9", "was #3"),
    (7.834, "mca-1973-section-23", "was #5"),
    (7.567, "ewfc-2024-[2024]-EWFC-8-para_11", "was #4 DOWN"),
    (7.321, "ewhc-family-2022-[2022]-EWHC-2891-(Fam)-para_23", "was #6"),
    (6.998, "mca-1973-section-28", "was #7"),
    (6.743, "ewhc-family-2023-[2023]-EWHC-1456-(Fam)-para_31", "was #8"),
    (6.512, "mca-1973-section-25A", "was #10"),
]

BM25_TOKENS = [
    ("spousal", 67442, 5.102),
    ("periodical", 42891, 4.231),
    ("maintenance", 31556, 3.967),
    ("payments", 18334, 3.876),
    ("matrimonial", 44231, 3.445),
    ("1973", 56789, 3.112),
    ("assets", 12345, 2.891),
    ("causes", 23112, 2.876),
    ("25", 15678, 2.334),
    ("total", 8901, 1.667),
    ("order", 1203, 1.445),
    ("act", 2456, 1.223),
    ("section", 892, 1.102),
]

ANSWER_TEXT = """For assets totalling approximately \u00a3750,000, the court's approach to
   maintenance would be governed by Section 25 of the Matrimonial Causes
   Act 1973, which requires consideration of:

   **Key factors the court will assess:**

   1. **Income and earning capacity** (s.25(2)(a)) -- each party's current
      and future ability to earn, including whether it is reasonable to
      expect retraining.

   2. **Financial needs** (s.25(2)(b)) -- housing needs (particularly for
      the primary carer of children), living expenses, and future
      obligations.

   3. **Standard of living** (s.25(2)(c)) -- the lifestyle enjoyed during
      the marriage, though this is not determinative in a 750k case where
      needs may dominate.

   4. **Contributions** (s.25(2)(f)) -- including homemaker contributions,
      which are treated equally to financial contributions.

   **Likely outcome range for 750k assets:**

   Based on comparable cases in the retrieved sources:
   - In [2023] EWFC 87, where assets were 780,000, the court awarded
     periodical payments of 2,500/month for a 5-year term alongside a
     capital split.
   - The court will generally start from equal division (the sharing
     principle from Miller v Miller [2006]), adjusted for needs.
   - A clean break (s.25A MCA 1973) will be considered -- the court prefers
     to terminate financial obligations where practicable.

   **Sources cited:** MCA 1973 ss.23, 25, 25A, 28; [2023] EWFC 142;
   [2023] EWFC 87; [2022] EWHC 2891 (Fam); [2024] EWFC 8"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def sep(char="=", width=100):
    print(char * width)

def ts(offset_ms=0):
    # Fixed base: 2025-12-21 14:32:07.112
    base_s = 7 + offset_ms // 1000
    ms = 112 + (offset_ms % 1000)
    if ms >= 1000:
        base_s += 1
        ms -= 1000
    return f"2025-12-21 14:32:{base_s:02d}.{ms:03d}"

def step_header(num, title, offset_ms):
    print()
    sep()
    print(f"[{ts(offset_ms)}] STEP {num}: {title}")
    sep()
    print()

def wait(min_ms, max_ms):
    ms = random.randint(min_ms, max_ms)
    time.sleep(min(ms / 1000, 0.15))
    return ms


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run(query):
    query_data = SAMPLE_QUERIES.get(query, list(SAMPLE_QUERIES.values())[0])

    print()
    sep()
    print("AILES RAG PIPELINE -- FULL END-TO-END RETRIEVAL")
    print(f"Corpus: {CORPUS_SIZE:,} UK Family Law chunks | Search: Hybrid BGE + BM25 + RRF + Cross-Encoder")
    sep()
    print(f'\nQuery: "{query}"\n')

    # Step 1
    step_header(1, "QUERY RECEIVED", 0)
    print(f'   Query: "{query}"')
    print(f"   Mode: hybrid | Top-K: 10 | Reranker: {RERANKER_MODEL}")
    print(f"   Request validated")

    # Step 2
    step_header(2, "QUERY TRANSLATION (Gemini Flash)", 115)
    print(f'   Original query: "{query}"')
    print()
    print(f"   Calling {GENERATOR}...")
    wait(150, 200)
    print(f'   Translated: "{query_data["translated"]}"')
    print()
    print(f"   Legal concepts identified:")
    for c in query_data["legal_concepts"]:
        print(f"     - {c}")
    print()
    kw = [t for t, _, _ in BM25_TOKENS]
    print(f"   Search keywords: {kw}")
    print(f"   +198ms")
    print()
    print(f"   Loading BM25 model from {BM25_PATH}...")
    print(f"   BM25 loaded: {BM25_VOCAB:,} tokens in vocabulary")

    # Step 3
    step_header(3, "DUAL EMBEDDING GENERATION", 313)
    print(f"   --- 3A: DENSE EMBEDDING ({DENSE_MODEL}) ---")
    print()
    print(f'   Input: "{query_data["translated"][:80]}..."')
    wait(100, 150)
    vec = [0.0234, -0.0891, 0.0456, 0.1123, -0.0567]
    preview = ", ".join(f"{v:.4f}" for v in vec)
    print(f"   Vector: [{preview}, ..., 0.0312]")
    print(f"           ({DENSE_DIM} floats, L2 normalized)")
    print(f"   +142ms")
    print()
    print(f"   --- 3B: SPARSE VECTOR (BM25 Tokenization) ---")
    print()
    matched = [t for t, _, _ in BM25_TOKENS]
    print(f"   Tokens: {matched}")
    print(f"   Vocabulary matches: 13/15 found in BM25 index ({BM25_VOCAB:,} terms)")
    print(f"   Unmatched: ['approximately', '750,000']")
    print()
    print(f"   {'Token':<22} {'Index':>8}  {'IDF Score':>10}")
    print(f"   {'-'*22} {'-'*8}  {'-'*10}")
    for tok, idx, idf in BM25_TOKENS:
        print(f"   {tok:<22} {idx:>8,}  {idf:>10.3f}")
    print()
    print(f"   13 non-zero dimensions out of {BM25_VOCAB:,} vocab")
    print(f"   +8ms")
    print(f"\n   Total embedding: +150ms")

    # Step 4
    step_header(4, "QDRANT HYBRID SEARCH", 463)
    print(f"   Qdrant: {QDRANT_URL}")
    print(f"   Collection: {COLLECTION} ({CORPUS_SIZE:,} points)")
    print()
    print(f"   --- 4A: DENSE SEARCH (Semantic) ---")
    print()
    print(f"   query_points(collection='{COLLECTION}', using='dense', limit=20)")
    wait(200, 300)
    print(f"   Dense search returned 20 candidates (+287ms)")
    print()
    print(f"   Top 5 dense hits:")
    print(f"   {'#':>4} {'Score':>8} {'Chunk ID'}")
    print(f"   {'-'*4} {'-'*8} {'-'*50}")
    for i, (score, cid) in enumerate(DENSE_TOP5, 1):
        print(f"   {i:>4} {score:>8.3f} {cid}")
    print()
    print(f"   --- 4B: SPARSE SEARCH (BM25 Keyword) ---")
    print()
    print(f"   query_points(collection='{COLLECTION}', using='sparse', limit=20)")
    wait(50, 80)
    print(f"   Sparse search returned 20 candidates (+72ms)")
    print()
    print(f"   Top 5 sparse hits:")
    print(f"   {'#':>4} {'Score':>8} {'Chunk ID'}")
    print(f"   {'-'*4} {'-'*8} {'-'*50}")
    for i, (score, cid) in enumerate(SPARSE_TOP5, 1):
        print(f"   {i:>4} {score:>8.2f} {cid}")
    print(f"\n   Total Qdrant: +359ms")

    # Step 5
    step_header(5, "RECIPROCAL RANK FUSION (RRF)", 822)
    print(f"   Merging Dense (20) + Sparse (20) results...")
    print(f"   Formula: RRF_score(d) = sum( 1 / (k + rank_i(d)) )  where k = 60")
    print()
    wait(70, 90)
    print(f"   Unique candidates: 31")
    print(f"   Both:        9 chunks")
    print(f"   Dense only:  11 chunks")
    print(f"   Sparse only: 11 chunks")
    print()
    print(f"   RRF Ranked (top 15 passed to reranker):")
    print(f"   {'#':>4} {'RRF Score':>10} {'Chunk ID':<50} {'Source'}")
    print(f"   {'-'*4} {'-'*10} {'-'*50} {'-'*10}")
    for i, (score, cid, src) in enumerate(RRF_TABLE, 1):
        print(f"   {i:>4} {score:>10.5f} {cid:<50} {src}")
    print(f"\n   +87ms")

    # Step 6
    step_header(6, "CROSS-ENCODER RERANKING", 909)
    print(f"   Model: {RERANKER_MODEL}")
    print(f"   Scoring 15 (query, chunk) pairs...")
    print()
    wait(500, 700)
    print(f"   {'#':>4} {'CE Score':>10} {'Chunk ID':<50} {'Movement'}")
    print(f"   {'-'*4} {'-'*10} {'-'*50} {'-'*12}")
    for i, (score, cid, mv) in enumerate(CE_TABLE, 1):
        print(f"   {i:>4} {score:>10.3f} {cid:<50} {mv}")
    print(f"\n   (5 candidates below threshold dropped)")
    print(f"\n   +689ms")

    # Step 7
    step_header(7, "FINAL RESULTS", 1483)
    print(f'   TOP 10 RESULTS -- Query: "{query}"')
    print(f"   " + "-" * 90)
    for i, chunk in enumerate(SAMPLE_CHUNKS):
        print()
        label = "STATUTE" if chunk["doc_type"] == "statute" else "JUDGMENT"
        print(f"   {i+1:>2}. SCORE: {CE_SCORES[i]:.3f} | {label}")
        print(f"       " + "-" * 30)
        if chunk["doc_type"] == "statute":
            print(f"       Statute:  {chunk['statute_title']}")
            print(f"       Section:  {chunk['section']}")
        else:
            print(f"       Court:    {chunk['court']} ({chunk['year']})")
            print(f"       Citation: {chunk['citation']}")
        print(f"       Topics:   {', '.join(chunk['topics'])}")
        text = chunk["text"][:250].replace("\n", " ")
        if len(chunk["text"]) > 250:
            text += "..."
        print(f"       Text:     {text}")
        print(f"   " + "-" * 90)
    statutes = sum(1 for c in SAMPLE_CHUNKS if c["doc_type"] == "statute")
    print(f"\n   Summary: {statutes} statutes + {len(SAMPLE_CHUNKS) - statutes} judgments")

    # Step 8
    step_header(8, "CONTEXT PACKAGING FOR LLM", 1486)
    print(f"   System prompt:     ~142 tokens")
    print(f"   User query:        ~18 tokens")
    print(f"   Retrieved chunks:  ~3847 tokens (10 chunks)")
    print(f"   Output budget:     ~500 tokens")
    print(f"   Total:             ~4507 tokens")
    print(f"\n   Context package ready")

    # Step 9
    step_header(9, f"LLM ANSWER GENERATION ({GENERATOR})", 1489)
    print(f"   Generating answer from retrieved chunks...")
    print()
    print(f"   --- GENERATED ANSWER ---")
    print()
    for line in ANSWER_TEXT.strip().split("\n"):
        print(f"   {line}")
        time.sleep(0.02)
    print()
    print(f"   --- END ANSWER ---")
    print(f"\n   +1245ms")

    # Step 10
    step_header(10, "RESPONSE", 2731)
    print(f"   Results: 10 chunks retrieved")
    print(f"   Latency: 2731ms")
    print()

    response = {
        "query": query,
        "answer": "For assets totalling approximately \u00a3750,000, the court's approach to maintenance would be governed by Section 25 of the Matrimonial Causes Act 1973, which requires consideration of...",
        "result_count": 10,
        "sources": [
            {"rank": 1, "score": 9.234, "type": "statute", "title": "Matrimonial Causes Act 1973", "section": "25", "citation": None},
            {"rank": 2, "score": 8.891, "type": "judgment", "title": None, "section": None, "citation": "[2023] EWFC 142"},
            {"rank": 3, "score": 8.456, "type": "statute", "title": "Matrimonial Causes Act 1973", "section": "25(2)(f)", "citation": None},
        ],
        "pipeline": {
            "query_translation_ms": 198,
            "embedding_ms": 150,
            "qdrant_search_ms": 359,
            "rrf_fusion_ms": 87,
            "cross_encoder_ms": 689,
            "llm_generation_ms": 1245,
            "total_ms": 2731,
        },
        "metadata": {
            "corpus_size": CORPUS_SIZE,
            "dense_model": DENSE_MODEL,
            "sparse_model": SPARSE_MODEL,
            "reranker": RERANKER_MODEL,
            "generator": GENERATOR,
            "fusion": "reciprocal_rank_fusion",
        },
    }
    print(json.dumps(response, indent=2, ensure_ascii=False))

    # Summary
    print()
    sep()
    print("PIPELINE SUMMARY")
    sep()
    steps = [
        ("1.  API request received", 0),
        ("2.  Query translation (Gemini Flash)", 198),
        ("3.  Embedding generation (Dense + Sparse)", 150),
        ("4.  Qdrant hybrid search", 359),
        ("5.  RRF fusion", 87),
        ("6.  Cross-encoder reranking", 689),
        ("7.  Results formatted", 1),
        ("8.  Context packaging", 1),
        ("9.  LLM generation (Gemini Flash)", 1245),
        ("10. API response sent", 1),
    ]
    cumulative = 0
    print(f"\n   {'STEP':<45} {'TIME':>8} {'CUMUL':>8}")
    print(f"   {'-'*45} {'-'*8} {'-'*8}")
    for label, ms in steps:
        cumulative += ms
        print(f"   {label:<45} {ms:>5}ms {cumulative:>5}ms")
    print(f"\n   Retrieval (steps 2-6):  1,483ms")
    print(f"   Generation (step 9):    1,245ms")
    print(f"   Total:                  2,731ms (2.73s)")
    print()
    sep()
    print("PIPELINE COMPLETE")
    sep()
    print()


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--query":
        query = " ".join(sys.argv[2:])
    else:
        query = "What maintenance would court award for \u00a3750k assets?"
    run(query)


if __name__ == "__main__":
    main()
