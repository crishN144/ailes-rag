"""
[SUPERSEDED by v2]
This v1 query-translation module has been replaced for the v2 deploy by:
  - 4_v2_update/prompts/prompts_v9.yaml   (rag.expander block)
  - 4_v2_update/retrieval/retrieve_from_expander.py
Do NOT use this file for v2. Kept here for historical reference only.
See the v2 banner in the top-level README for the full ship list.

---

PRODUCTION-READY HYBRID RETRIEVAL SYSTEM
Based on: "Why Your RAG System is Failing in Production" blog

Blog insights:
1. Hybrid search (BM25 + Dense) = 87% recall vs 62% dense-only
2. Cross-encoder reranking = Critical for precision
3. Query intelligence layer = 89% query understanding vs 61%
4. Multi-stage retrieval = Progressive refinement

Your question: "Which model for query generation?"
Answer: DON'T use fine-tuned Llama-8B (that's for report generation)
        USE: Gemini 1.5 Flash (FREE, fast, excellent instruction following)
        Alternative: GPT-4o-mini (cheap: $0.15/1M input tokens)
"""

import os
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
import google.generativeai as genai
import asyncio
import aiohttp

# ============================================================================
# 1. QUERY GENERATION MODEL (NOT FINE-TUNED LLAMA-8B!)
# ============================================================================

class FormEQueryGenerator:
    """
    Blog lesson: "Users don't know how to query RAG systems"

    Your question: Which model for query generation?

    WRONG: Fine-tuned Llama-8B (that's for final report generation)

    RIGHT OPTIONS:
    1. Gemini 1.5 Flash (FREE up to 1500 requests/day, fast, good)
    2. GPT-4o-mini ($0.15/1M tokens, very cheap)
    3. Other fast instruction-following models (paid)

    I'll use Gemini Flash since it's FREE and excellent for this task.
    """

    def __init__(self, gemini_api_key: str):
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_queries_from_form_e(self, form_e_sections: Dict) -> Dict:
        """
        Generate targeted legal queries from Form E

        Blog lesson: "Generate query variations" + "Extract filters"

        Returns:
        {
            'statute_queries': [...],  # For statute retrieval
            'judgment_queries': [...], # For case law retrieval
            'filters': {...},          # For pre-filtering
            'complexity_score': 0-10   # For routing decisions
        }
        """

        # Build comprehensive Form E summary
        form_e_summary = self._summarize_form_e(form_e_sections)

        prompt = f"""You are a legal AI assistant analyzing a Form E financial questionnaire for family law proceedings.

# FORM E SUMMARY
{form_e_summary}

# TASK
Generate PRECISE legal queries to retrieve relevant UK family law statutes and case law.

CRITICAL RULES:
1. DO NOT mention statute names/numbers (we're searching by SITUATION, not by law)
2. Focus on FACTUAL SITUATIONS described in Form E
3. Generate queries for BOTH statutes and case law
4. Extract complexity factors

# OUTPUT FORMAT (JSON)
{{
  "statute_queries": [
    // Situation-based queries for statutes
    // Example: "financial provision when one party is homemaker and other has high income"
    // NOT: "section 25 factors"
  ],
  "judgment_queries": [
    // Situation-based queries for case law
    // Example: "homemaker contributions divorce settlement high net worth"
  ],
  "complexity_factors": {{
    "high_net_worth": true/false,
    "business_interests": true/false,
    "overseas_assets": true/false,
    "conduct_issues": true/false,
    "pension_assets": true/false,
    "long_marriage": true/false,
    "disabled_party": true/false
  }},
  "key_amounts": {{
    "total_assets": <number>,
    "income_disparity_ratio": <number>,
    "property_value": <number>
  }}
}}

Generate queries that will find relevant law based on the SITUATION, not legal terms."""

        # Call Gemini
        response = self.model.generate_content(prompt)

        # Parse JSON response
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fallback parsing
            result = {
                'statute_queries': self._fallback_statute_queries(form_e_sections),
                'judgment_queries': self._fallback_judgment_queries(form_e_sections),
                'complexity_factors': {},
                'key_amounts': {}
            }

        # Calculate complexity score
        complexity_score = self._calculate_complexity(result['complexity_factors'])
        result['complexity_score'] = complexity_score

        return result

    def _summarize_form_e(self, sections: Dict) -> str:
        """Create concise Form E summary"""
        summary_parts = []

        for section_name, section_content in sections.items():
            if section_content.strip():
                # Take first 200 chars of each section
                snippet = section_content[:200].strip()
                summary_parts.append(f"**{section_name.upper()}**: {snippet}...")

        return "\n".join(summary_parts)

    def _calculate_complexity(self, factors: Dict) -> int:
        """Calculate case complexity score (0-10)"""
        score = 0

        complexity_weights = {
            'high_net_worth': 2,
            'business_interests': 2,
            'overseas_assets': 2,
            'conduct_issues': 1,
            'pension_assets': 1,
            'long_marriage': 1,
            'disabled_party': 1
        }

        for factor, weight in complexity_weights.items():
            if factors.get(factor, False):
                score += weight

        return min(score, 10)

    def _fallback_statute_queries(self, sections: Dict) -> List[str]:
        """Fallback queries if Gemini fails"""
        queries = [
            "financial provision matrimonial proceedings",
            "property adjustment orders divorce"
        ]

        # Add specific queries based on sections
        if 'contributions' in sections:
            queries.append("homemaker contributions welfare of family")

        if 'assets' in sections:
            queries.append("property capital assets division divorce")

        return queries

    def _fallback_judgment_queries(self, sections: Dict) -> List[str]:
        """Fallback judgment queries"""
        return [
            "financial provision divorce settlement",
            "section 25 factors application"
        ]


# ============================================================================
# 2. BM25 SETUP (KEYWORD SEARCH)
# ============================================================================

class LegalBM25Index:
    """
    Blog lesson: "Hybrid search > pure vector search"

    BM25 catches exact keyword matches that embeddings might miss
    Example: "Section 25(2)(f)" exact match
    """

    def __init__(self):
        self.bm25_statutes = None
        self.bm25_judgments = None
        self.statute_corpus = []
        self.judgment_corpus = []
        self.statute_ids = []
        self.judgment_ids = []

    def build_indexes(self, statute_chunks: List[Dict], judgment_chunks: List[Dict]):
        """Build BM25 indexes from chunks"""

        print("Building BM25 indexes...")

        # Statutes
        self.statute_corpus = [chunk['bm25_text'] for chunk in statute_chunks]
        self.statute_ids = [chunk['chunk_id'] for chunk in statute_chunks]

        # Tokenize for BM25
        tokenized_statutes = [doc.split() for doc in self.statute_corpus]
        self.bm25_statutes = BM25Okapi(tokenized_statutes)

        # Judgments
        self.judgment_corpus = [chunk['bm25_text'] for chunk in judgment_chunks]
        self.judgment_ids = [chunk['chunk_id'] for chunk in judgment_chunks]

        tokenized_judgments = [doc.split() for doc in self.judgment_corpus]
        self.bm25_judgments = BM25Okapi(tokenized_judgments)

        print(f"✅ BM25 indexes built: {len(self.statute_ids)} statutes, {len(self.judgment_ids)} judgments")

    def search_statutes(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """BM25 search in statutes"""
        tokenized_query = query.split()
        scores = self.bm25_statutes.get_scores(tokenized_query)

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.statute_ids[idx], scores[idx])
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results

    def search_judgments(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """BM25 search in judgments"""
        tokenized_query = query.split()
        scores = self.bm25_judgments.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.judgment_ids[idx], scores[idx])
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results


# ============================================================================
# 3. CROSS-ENCODER RERANKING (CRITICAL!)
# ============================================================================

class LegalCrossEncoderReranker:
    """
    Blog lesson: "Cross-encoder reranking is the secret sauce"

    From blog: "Retrieval recall: 62% → 87% with reranking"

    Model choice:
    - Blog uses: cross-encoder/ms-marco-MiniLM-L-6-v2
    - For legal: cross-encoder/ms-marco-MiniLM-L-12-v2 (larger, more accurate)
    """

    def __init__(self):
        # Blog recommendation: ms-marco cross-encoder
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank candidates with cross-encoder

        Blog lesson: "Score query-document pairs for precise relevance"
        """

        if not candidates:
            return []

        # Prepare pairs
        pairs = []
        for candidate in candidates:
            # Use section_text for statutes, paragraph_text for judgments
            text = candidate.get('section_text') or candidate.get('paragraph_text', '')
            pairs.append([query, text])

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Combine with original scores
        for i, candidate in enumerate(candidates):
            candidate['cross_encoder_score'] = float(scores[i])

            # Weighted combination: 60% cross-encoder, 40% original
            original_score = candidate.get('fusion_score', 0.5)
            candidate['final_score'] = 0.6 * scores[i] + 0.4 * original_score

        # Sort by final score
        reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)

        return reranked[:top_k]


# ============================================================================
# 4. COMPLETE HYBRID RETRIEVAL SYSTEM
# ============================================================================

class ProductionLegalRetriever:
    """
    Blog lesson: "Three-stage retrieval with progressive refinement"

    Stage 1: Hybrid search (BM25 + Dense) → 50 candidates
    Stage 2: Cross-encoder rerank → 20 results
    Stage 3: Intelligent fusion → 10 final (with Section 25 guarantee)
    """

    def __init__(self, qdrant_url: str, gemini_api_key: str):
        # Qdrant client
        self.qdrant = QdrantClient(url=qdrant_url)

        # BGE embedder (answer: YES, BGE is excellent!)
        self.embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')

        # Query generator (Gemini Flash - FREE!)
        self.query_gen = FormEQueryGenerator(gemini_api_key)

        # BM25 indexes (will be built after indexing)
        self.bm25 = LegalBM25Index()

        # Cross-encoder reranker
        self.reranker = LegalCrossEncoderReranker()

        # Min score threshold (blog: 0.65)
        self.min_score_threshold = 0.65

    async def retrieve(self, form_e_sections: Dict, top_k: int = 10) -> Dict:
        """
        Complete hybrid retrieval pipeline

        Returns:
        {
            'statutes': [...],  # Top statute chunks
            'judgments': [...], # Top judgment chunks
            'query_info': {...},
            'validation': {...}
        }
        """

        # ===== STAGE 0: QUERY INTELLIGENCE =====
        print("Stage 0: Generating queries from Form E...")
        query_info = self.query_gen.generate_queries_from_form_e(form_e_sections)

        statute_queries = query_info['statute_queries']
        judgment_queries = query_info['judgment_queries']

        print(f"Generated {len(statute_queries)} statute queries, {len(judgment_queries)} judgment queries")

        # ===== STAGE 1: HYBRID SEARCH (PARALLEL) =====
        print("\nStage 1: Hybrid search (BM25 + Dense)...")

        statute_candidates = await self._hybrid_search_statutes(
            statute_queries,
            limit=50
        )

        judgment_candidates = await self._hybrid_search_judgments(
            judgment_queries,
            limit=50
        )

        print(f"Retrieved {len(statute_candidates)} statute candidates, {len(judgment_candidates)} judgment candidates")

        # ===== STAGE 2: CROSS-ENCODER RERANKING =====
        print("\nStage 2: Cross-encoder reranking...")

        # Combine all queries for reranking
        all_statute_query = " ".join(statute_queries)
        all_judgment_query = " ".join(judgment_queries)

        statute_reranked = self.reranker.rerank(
            all_statute_query,
            statute_candidates,
            top_k=20
        )

        judgment_reranked = self.reranker.rerank(
            all_judgment_query,
            judgment_candidates,
            top_k=20
        )

        print(f"Reranked to {len(statute_reranked)} statutes, {len(judgment_reranked)} judgments")

        # ===== STAGE 3: VALIDATION & FUSION =====
        print("\nStage 3: Validation & fusion...")

        # Blog lesson: "Retrieval quality gates"
        validation = self._validate_retrieval(statute_reranked, judgment_reranked)

        if not validation['valid']:
            print(f"⚠️ Retrieval validation failed: {validation['reason']}")

            # Force-retrieve Section 25 (blog lesson: critical sections always included)
            section_25 = self._force_retrieve_section_25()
            if section_25:
                statute_reranked.insert(0, section_25)

        # Final selection
        final_statutes = statute_reranked[:top_k]
        final_judgments = judgment_reranked[:max(1, top_k // 2)]  # Fewer judgments

        print(f"✅ Final: {len(final_statutes)} statutes, {len(final_judgments)} judgments")

        return {
            'statutes': final_statutes,
            'judgments': final_judgments,
            'query_info': query_info,
            'validation': validation,
            'complexity_score': query_info['complexity_score']
        }

    async def _hybrid_search_statutes(self, queries: List[str], limit: int) -> List[Dict]:
        """
        Hybrid search: BM25 + Dense vector search

        Blog lesson: "Pure vector search failed 34% of keyword queries"
        """

        all_results = {}

        for query in queries:
            # Dense vector search
            query_embedding = self.embedder.encode(query, normalize_embeddings=True)

            dense_results = self.qdrant.search(
                collection_name="uk_statutes",
                query_vector=query_embedding.tolist(),
                limit=limit,
                with_payload=True
            )

            # BM25 sparse search
            bm25_results = self.bm25.search_statutes(query, top_k=limit)

            # Reciprocal Rank Fusion (blog: RRF for hybrid)
            for rank, result in enumerate(dense_results, 1):
                chunk_id = result.id
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        **result.payload,
                        'chunk_id': chunk_id,
                        'dense_score': result.score,
                        'bm25_score': 0,
                        'rrf_score': 0
                    }

                # RRF: 1 / (k + rank)
                all_results[chunk_id]['rrf_score'] += 1 / (60 + rank)

            # Add BM25 results
            for chunk_id, bm25_score in bm25_results:
                if chunk_id not in all_results:
                    # Retrieve from Qdrant
                    retrieved = self.qdrant.retrieve(
                        collection_name="uk_statutes",
                        ids=[chunk_id]
                    )
                    if retrieved:
                        all_results[chunk_id] = {
                            **retrieved[0].payload,
                            'chunk_id': chunk_id,
                            'dense_score': 0,
                            'bm25_score': bm25_score,
                            'rrf_score': 0
                        }

                # Find rank in BM25 results
                bm25_rank = next((i for i, (cid, _) in enumerate(bm25_results, 1) if cid == chunk_id), 100)
                all_results[chunk_id]['rrf_score'] += 1 / (60 + bm25_rank)
                all_results[chunk_id]['bm25_score'] = max(all_results[chunk_id]['bm25_score'], bm25_score)

        # Sort by RRF score
        candidates = sorted(all_results.values(), key=lambda x: x['rrf_score'], reverse=True)

        # Add fusion_score for reranking
        for candidate in candidates:
            candidate['fusion_score'] = candidate['rrf_score']

        return candidates

    async def _hybrid_search_judgments(self, queries: List[str], limit: int) -> List[Dict]:
        """Hybrid search for judgments"""

        all_results = {}

        for query in queries:
            # Dense vector search
            query_embedding = self.embedder.encode(query, normalize_embeddings=True)

            dense_results = self.qdrant.search(
                collection_name="uk_judgments",
                query_vector=query_embedding.tolist(),
                limit=limit,
                with_payload=True
            )

            # BM25 search
            bm25_results = self.bm25.search_judgments(query, top_k=limit)

            # RRF fusion
            for rank, result in enumerate(dense_results, 1):
                chunk_id = result.id
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        **result.payload,
                        'chunk_id': chunk_id,
                        'dense_score': result.score,
                        'bm25_score': 0,
                        'rrf_score': 0
                    }

                all_results[chunk_id]['rrf_score'] += 1 / (60 + rank)

            for chunk_id, bm25_score in bm25_results:
                if chunk_id not in all_results:
                    retrieved = self.qdrant.retrieve(
                        collection_name="uk_judgments",
                        ids=[chunk_id]
                    )
                    if retrieved:
                        all_results[chunk_id] = {
                            **retrieved[0].payload,
                            'chunk_id': chunk_id,
                            'dense_score': 0,
                            'bm25_score': bm25_score,
                            'rrf_score': 0
                        }

                bm25_rank = next((i for i, (cid, _) in enumerate(bm25_results, 1) if cid == chunk_id), 100)
                all_results[chunk_id]['rrf_score'] += 1 / (60 + bm25_rank)
                all_results[chunk_id]['bm25_score'] = max(all_results[chunk_id]['bm25_score'], bm25_score)

        candidates = sorted(all_results.values(), key=lambda x: x['rrf_score'], reverse=True)

        for candidate in candidates:
            candidate['fusion_score'] = candidate['rrf_score']

        return candidates

    def _validate_retrieval(self, statutes: List[Dict], judgments: List[Dict]) -> Dict:
        """
        Blog lesson: "Retrieval quality gates - don't generate if retrieval is poor"
        """

        if not statutes:
            return {
                'valid': False,
                'reason': 'No statutes retrieved',
                'confidence': 'low'
            }

        # Check top score
        top_score = statutes[0].get('final_score', 0)

        if top_score < self.min_score_threshold:
            return {
                'valid': False,
                'reason': f'Low relevance score: {top_score:.2f}',
                'confidence': 'low'
            }

        # Check if Section 25 is present (CRITICAL)
        section_numbers = [s.get('section_number') for s in statutes]

        if '25' not in section_numbers:
            return {
                'valid': False,
                'reason': 'Section 25 MCA 1973 not retrieved (critical)',
                'confidence': 'medium'
            }

        return {
            'valid': True,
            'confidence': 'high' if top_score > 0.8 else 'medium',
            'top_score': top_score
        }

    def _force_retrieve_section_25(self) -> Optional[Dict]:
        """Always include Section 25 MCA 1973"""

        results = self.qdrant.retrieve(
            collection_name="uk_statutes",
            ids=["mca-1973-section-25"]
        )

        if results:
            return {
                **results[0].payload,
                'chunk_id': results[0].id,
                'final_score': 1.0,
                'forced': True
            }

        return None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage"""

    # Initialize
    retriever = ProductionLegalRetriever(
        qdrant_url="http://localhost:6333",
        gemini_api_key=os.getenv("GEMINI_API_KEY")  # FREE API key
    )

    # Build BM25 indexes (do this AFTER indexing to Qdrant)
    # You need to pass the chunks from indexing
    # retriever.bm25.build_indexes(statute_chunks, judgment_chunks)

    # Example Form E
    form_e = {
        'income': 'Applicant: £45,000/year salary. Respondent: £180,000/year as company director.',
        'assets': 'Family home valued £850,000 with £200,000 mortgage. Respondent has business worth £2.5M.',
        'contributions': 'Applicant was primary caregiver for 15 years, managed household while Respondent built business.',
        'needs': 'Applicant needs housing for self and two children (ages 12, 14).',
    }

    # Retrieve
    results = await retriever.retrieve(form_e, top_k=10)

    print("\n" + "="*60)
    print("RETRIEVAL RESULTS")
    print("="*60)

    print(f"\nStatutes retrieved: {len(results['statutes'])}")
    for i, statute in enumerate(results['statutes'][:3], 1):
        print(f"{i}. Section {statute['section_number']}: {statute['section_heading']}")
        print(f"   Score: {statute['final_score']:.3f}")

    print(f"\nJudgments retrieved: {len(results['judgments'])}")
    for i, judgment in enumerate(results['judgments'][:3], 1):
        print(f"{i}. {judgment['case_name']}")
        print(f"   Score: {judgment['final_score']:.3f}")

    print(f"\nComplexity score: {results['complexity_score']}/10")


if __name__ == "__main__":
    asyncio.run(main())
