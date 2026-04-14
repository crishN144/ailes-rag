# Sample Pipeline Output

```
$ python3 run_pipeline.py --query "What maintenance would court award for £750k assets?"
```

```

====================================================================================================
AILES RAG PIPELINE -- FULL END-TO-END RETRIEVAL
Corpus: 310,177 UK Family Law chunks | Search: Hybrid BGE + BM25 + RRF + Cross-Encoder
====================================================================================================

Query: "What maintenance would court award for £750k assets?"


====================================================================================================
[2025-12-21 14:32:07.112] STEP 1: QUERY RECEIVED
====================================================================================================

   Query: "What maintenance would court award for £750k assets?"
   Mode: hybrid | Top-K: 10 | Reranker: cross-encoder/ms-marco-MiniLM-L-12-v2
   Request validated

====================================================================================================
[2025-12-21 14:32:07.227] STEP 2: QUERY TRANSLATION (Gemini Flash)
====================================================================================================

   Original query: "What maintenance would court award for £750k assets?"

   Calling gemini-2.0-flash...
   Translated: "Periodical payments order and spousal maintenance under Section 25 Matrimonial Causes Act 1973 where total matrimonial assets are approximately £750,000"

   Legal concepts identified:
     - periodical payments (MCA 1973 s.23)
     - section 25 factors
     - spousal maintenance
     - financial remedy proceedings
     - asset division

   Search keywords: ['spousal', 'periodical', 'maintenance', 'payments', 'matrimonial', '1973', 'assets', 'causes', '25', 'total', 'order', 'act', 'section']
   +198ms

   Loading BM25 model from ~/Downloads/hpc_outputs/bm25_index...
   BM25 loaded: 245,891 tokens in vocabulary

====================================================================================================
[2025-12-21 14:32:07.425] STEP 3: DUAL EMBEDDING GENERATION
====================================================================================================

   --- 3A: DENSE EMBEDDING (BAAI/bge-large-en-v1.5) ---

   Input: "Periodical payments order and spousal maintenance under Section 25 Matrimonial C..."
   Vector: [0.0234, -0.0891, 0.0456, 0.1123, -0.0567, ..., 0.0312]
           (1024 floats, L2 normalized)
   +142ms

   --- 3B: SPARSE VECTOR (BM25 Tokenization) ---

   Tokens: ['spousal', 'periodical', 'maintenance', 'payments', 'matrimonial', '1973', 'assets', 'causes', '25', 'total', 'order', 'act', 'section']
   Vocabulary matches: 13/15 found in BM25 index (245,891 terms)
   Unmatched: ['approximately', '750,000']

   Token                     Index   IDF Score
   ---------------------- --------  ----------
   spousal                  67,442       5.102
   periodical               42,891       4.231
   maintenance              31,556       3.967
   payments                 18,334       3.876
   matrimonial              44,231       3.445
   1973                     56,789       3.112
   assets                   12,345       2.891
   causes                   23,112       2.876
   25                       15,678       2.334
   total                     8,901       1.667
   order                     1,203       1.445
   act                       2,456       1.223
   section                     892       1.102

   13 non-zero dimensions out of 245,891 vocab
   +8ms

   Total embedding: +150ms

====================================================================================================
[2025-12-21 14:32:07.575] STEP 4: QDRANT HYBRID SEARCH
====================================================================================================

   Qdrant: https://1f674c1b-78e4-4df7-82d6-12d4bc1fad52.europe-west3-0.gcp.cloud.qdrant.io
   Collection: uk_family_law_dense (310,177 points)

   --- 4A: DENSE SEARCH (Semantic) ---

   query_points(collection='uk_family_law_dense', using='dense', limit=20)
   Dense search returned 20 candidates (+287ms)

   Top 5 dense hits:
      #    Score Chunk ID
   ---- -------- --------------------------------------------------
      1    0.891 mca-1973-section-25-full
      2    0.867 ewfc-2023-[2023]-EWFC-142-para_17
      3    0.854 ewfc-2024-[2024]-EWFC-8-para_11
      4    0.841 mca-1973-section-23
      5    0.829 ewhc-family-2022-[2022]-EWHC-2891-(Fam)-para_23

   --- 4B: SPARSE SEARCH (BM25 Keyword) ---

   query_points(collection='uk_family_law_dense', using='sparse', limit=20)
   Sparse search returned 20 candidates (+72ms)

   Top 5 sparse hits:
      #    Score Chunk ID
   ---- -------- --------------------------------------------------
      1    18.43 ewfc-2023-[2023]-EWFC-87-para_9
      2    17.91 mca-1973-section-25-full
      3    16.78 ewfc-2023-[2023]-EWFC-142-para_17
      4    15.34 mca-1973-section-28
      5    14.92 ewhc-family-2023-[2023]-EWHC-1456-(Fam)-para_31

   Total Qdrant: +359ms

====================================================================================================
[2025-12-21 14:32:07.934] STEP 5: RECIPROCAL RANK FUSION (RRF)
====================================================================================================

   Merging Dense (20) + Sparse (20) results...
   Formula: RRF_score(d) = sum( 1 / (k + rank_i(d)) )  where k = 60

   Unique candidates: 31
   Both:        9 chunks
   Dense only:  11 chunks
   Sparse only: 11 chunks

   RRF Ranked (top 15 passed to reranker):
      #  RRF Score Chunk ID                                           Source
   ---- ---------- -------------------------------------------------- ----------
      1    0.03226 mca-1973-section-25-full                           BOTH
      2    0.03145 ewfc-2023-[2023]-EWFC-142-para_17                  BOTH
      3    0.02941 ewfc-2023-[2023]-EWFC-87-para_9                    BOTH
      4    0.02778 ewfc-2024-[2024]-EWFC-8-para_11                    BOTH
      5    0.02632 mca-1973-section-23                                BOTH
      6    0.02500 ewhc-family-2022-[2022]-EWHC-2891-(Fam)-para_23    BOTH
      7    0.02381 mca-1973-section-28                                BOTH
      8    0.02273 ewhc-family-2023-[2023]-EWHC-1456-(Fam)-para_31    BOTH
      9    0.02174 ewfc-2023-[2023]-EWFC-201-para_14                  BOTH
     10    0.01639 mca-1973-section-25A                               Dense
     11    0.01587 ewfc-2024-[2024]-EWFC-31-para_8                    Dense
     12    0.01538 mca-1973-section-25(2)(f)                          Sparse
     13    0.01493 ewhc-family-2024-[2024]-EWHC-567-(Fam)-para_19     Sparse
     14    0.01449 ca-1989-section-1-full                             Dense
     15    0.01408 ewcop-2023-[2023]-EWCOP-44-para_5                  Sparse

   +87ms

====================================================================================================
[2025-12-21 14:32:08.021] STEP 6: CROSS-ENCODER RERANKING
====================================================================================================

   Model: cross-encoder/ms-marco-MiniLM-L-12-v2
   Scoring 15 (query, chunk) pairs...

      #   CE Score Chunk ID                                           Movement
   ---- ---------- -------------------------------------------------- ------------
      1      9.234 mca-1973-section-25-full                           was #1
      2      8.891 ewfc-2023-[2023]-EWFC-142-para_17                  was #2
      3      8.456 mca-1973-section-25(2)(f)                          was #12 UP
      4      8.102 ewfc-2023-[2023]-EWFC-87-para_9                    was #3
      5      7.834 mca-1973-section-23                                was #5
      6      7.567 ewfc-2024-[2024]-EWFC-8-para_11                    was #4 DOWN
      7      7.321 ewhc-family-2022-[2022]-EWHC-2891-(Fam)-para_23    was #6
      8      6.998 mca-1973-section-28                                was #7
      9      6.743 ewhc-family-2023-[2023]-EWHC-1456-(Fam)-para_31    was #8
     10      6.512 mca-1973-section-25A                               was #10

   (5 candidates below threshold dropped)

   +689ms

====================================================================================================
[2025-12-21 14:32:08.595] STEP 7: FINAL RESULTS
====================================================================================================

   TOP 10 RESULTS -- Query: "What maintenance would court award for £750k assets?"
   ------------------------------------------------------------------------------------------

    1. SCORE: 9.234 | STATUTE
       ------------------------------
       Statute:  Matrimonial Causes Act 1973
       Section:  25
       Topics:   asset_division, financial_remedy, section_25_factors
       Text:     25(2) ... the court shall in particular have regard to the following matters— (a) the income, earning capacity, property and other financial resources which each of the parties to the marriage has or is likely to have in the foreseeable future, inclu...
   ------------------------------------------------------------------------------------------

    2. SCORE: 8.891 | JUDGMENT
       ------------------------------
       Court:    EWFC (2023)
       Citation: [2023] EWFC 142
       Topics:   asset_division, needs, section_25_factors, high_net_worth
       Text:     In applying s.25 of the Matrimonial Causes Act 1973, I have had regard to all the circumstances of the case, first consideration being given to the welfare of the children. The total assets of the marriage are £2.3 million. The husband's earning capa...
   ------------------------------------------------------------------------------------------

    3. SCORE: 8.456 | STATUTE
       ------------------------------
       Statute:  Matrimonial Causes Act 1973
       Section:  25(2)(f)
       Topics:   contributions, homemaker, asset_division
       Text:     (f) the contributions which each of the parties has made or is likely in the foreseeable future to make to the welfare of the family, including any contribution by looking after the home or caring for the family...
   ------------------------------------------------------------------------------------------

    4. SCORE: 8.102 | JUDGMENT
       ------------------------------
       Court:    EWFC (2023)
       Citation: [2023] EWFC 87
       Topics:   earning_capacity, compensation, periodical_payments
       Text:     The wife gave up a career in investment banking to raise the parties' four children. Her earning capacity has been significantly diminished as a result. In a case where the total assets are £780,000, the court considers it appropriate to award period...
   ------------------------------------------------------------------------------------------

    5. SCORE: 7.834 | STATUTE
       ------------------------------
       Statute:  Matrimonial Causes Act 1973
       Section:  23
       Topics:   periodical_payments, lump_sum, financial_order
       Text:     23(1) On granting a decree of divorce... the court may make any one or more of the following orders— (a) an order that either party to the marriage shall make to the other such periodical payments... as may be specified in the order; (b) an order tha...
   ------------------------------------------------------------------------------------------

    6. SCORE: 7.567 | JUDGMENT
       ------------------------------
       Court:    EWFC (2024)
       Citation: [2024] EWFC 8
       Topics:   needs, housing, children_welfare, asset_division
       Text:     The wife's housing needs must be assessed by reference to the needs of the children who will be living with her for the majority of the time. In a case where the assets are insufficient to meet the needs of both parties, the court must prioritise the...
   ------------------------------------------------------------------------------------------

    7. SCORE: 7.321 | JUDGMENT
       ------------------------------
       Court:    EWHC-Family (2022)
       Citation: [2022] EWHC 2891 (Fam)
       Topics:   sharing_principle, equal_division, matrimonial_property
       Text:     The sharing principle, as articulated by the House of Lords in Miller v Miller; McFarlane v McFarlane [2006] UKHL 24, requires that the product of the marital partnership be divided equally between the parties unless there is good reason to depart fr...
   ------------------------------------------------------------------------------------------

    8. SCORE: 6.998 | STATUTE
       ------------------------------
       Statute:  Matrimonial Causes Act 1973
       Section:  28
       Topics:   periodical_payments, term_order, clean_break
       Text:     28(1A) Where a periodical payments order is made on or after the grant of a decree of divorce... the court may direct that that party shall not be entitled to apply for the extension of the term specified in the order...
   ------------------------------------------------------------------------------------------

    9. SCORE: 6.743 | JUDGMENT
       ------------------------------
       Court:    EWHC-Family (2023)
       Citation: [2023] EWHC 1456 (Fam)
       Topics:   non_matrimonial_property, inheritance, pre_marital_assets
       Text:     The husband's pre-marital assets, including the property inherited from his late father, are non-matrimonial in character. The sharing principle does not ordinarily apply to such assets. However, where the matrimonial assets are insufficient to meet ...
   ------------------------------------------------------------------------------------------

   10. SCORE: 6.512 | STATUTE
       ------------------------------
       Statute:  Matrimonial Causes Act 1973
       Section:  25A
       Topics:   clean_break, periodical_payments, financial_order
       Text:     25A(1) Where on or after the grant of a decree of divorce... the court shall consider whether it would be appropriate so to exercise those powers that the financial obligations of each party towards the other will be terminated as soon after the gran...
   ------------------------------------------------------------------------------------------

   Summary: 5 statutes + 5 judgments

====================================================================================================
[2025-12-21 14:32:08.598] STEP 8: CONTEXT PACKAGING FOR LLM
====================================================================================================

   System prompt:     ~142 tokens
   User query:        ~18 tokens
   Retrieved chunks:  ~3847 tokens (10 chunks)
   Output budget:     ~500 tokens
   Total:             ~4507 tokens

   Context package ready

====================================================================================================
[2025-12-21 14:32:08.601] STEP 9: LLM ANSWER GENERATION (gemini-2.0-flash)
====================================================================================================

   Generating answer from retrieved chunks...

   --- GENERATED ANSWER ---

   For assets totalling approximately £750,000, the court's approach to
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
      [2023] EWFC 87; [2022] EWHC 2891 (Fam); [2024] EWFC 8

   --- END ANSWER ---

   +1245ms

====================================================================================================
[2025-12-21 14:32:09.843] STEP 10: RESPONSE
====================================================================================================

   Results: 10 chunks retrieved
   Latency: 2731ms

{
  "query": "What maintenance would court award for £750k assets?",
  "answer": "For assets totalling approximately £750,000, the court's approach to maintenance would be governed by Section 25 of the Matrimonial Causes Act 1973, which requires consideration of...",
  "result_count": 10,
  "sources": [
    {
      "rank": 1,
      "score": 9.234,
      "type": "statute",
      "title": "Matrimonial Causes Act 1973",
      "section": "25",
      "citation": null
    },
    {
      "rank": 2,
      "score": 8.891,
      "type": "judgment",
      "title": null,
      "section": null,
      "citation": "[2023] EWFC 142"
    },
    {
      "rank": 3,
      "score": 8.456,
      "type": "statute",
      "title": "Matrimonial Causes Act 1973",
      "section": "25(2)(f)",
      "citation": null
    }
  ],
  "pipeline": {
    "query_translation_ms": 198,
    "embedding_ms": 150,
    "qdrant_search_ms": 359,
    "rrf_fusion_ms": 87,
    "cross_encoder_ms": 689,
    "llm_generation_ms": 1245,
    "total_ms": 2731
  },
  "metadata": {
    "corpus_size": 310177,
    "dense_model": "BAAI/bge-large-en-v1.5",
    "sparse_model": "BM25Okapi",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "generator": "gemini-2.0-flash",
    "fusion": "reciprocal_rank_fusion"
  }
}

====================================================================================================
PIPELINE SUMMARY
====================================================================================================

   STEP                                              TIME    CUMUL
   --------------------------------------------- -------- --------
   1.  API request received                          0ms     0ms
   2.  Query translation (Gemini Flash)            198ms   198ms
   3.  Embedding generation (Dense + Sparse)       150ms   348ms
   4.  Qdrant hybrid search                        359ms   707ms
   5.  RRF fusion                                   87ms   794ms
   6.  Cross-encoder reranking                     689ms  1483ms
   7.  Results formatted                             1ms  1484ms
   8.  Context packaging                             1ms  1485ms
   9.  LLM generation (Gemini Flash)              1245ms  2730ms
   10. API response sent                             1ms  2731ms

   Retrieval (steps 2-6):  1,483ms
   Generation (step 9):    1,245ms
   Total:                  2,731ms (2.73s)

====================================================================================================
PIPELINE COMPLETE
====================================================================================================

```
