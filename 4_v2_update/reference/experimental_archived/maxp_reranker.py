"""
maxp_reranker.py — MaxP (Maximum Passage) reranking wrapper for long chunks.

The 512-token cross-encoder truncation destroys scores for long statute chunks
(e.g. `mca-1973-section-25-full` at 3388 chars ≈ 850 tokens). The first 512
tokens contain boilerplate statute header + subsection (a), cutting off the
subsection that actually matches the query.

MaxP (used by Cohere rerank-v4.0, documented in their best-practices) fixes
this by:

  1. Detecting chunks whose token count exceeds the CE's window
  2. Splitting such chunks into overlapping 512-token windows
  3. Scoring each window independently with the cross-encoder
  4. Taking the MAXIMUM score as the chunk's final score

For a 3388-char statute chunk with the relevant subsection in the back half,
this recovers the correct score — the window containing the relevant
subsection gets high CE score, MaxP takes that max, chunk ranks correctly.

Integration:

    from maxp_reranker import rerank_with_maxp

    reranked = rerank_with_maxp(
        candidates=retrieval_output,   # list of Qdrant points
        query=user_query,
        reranker=cross_encoder_instance,
    )

Latency impact: on top-30 candidates, if ~3 are long (>512 tokens), adds
~5–10 extra CE calls per query. Total rerank stays under 1 second for MS-MARCO.
"""

from __future__ import annotations

from typing import Any

# Tuned for ms-marco-MiniLM-L-12-v2 (512 token window, ~4 chars/token English)
# Keep query budget (~64 tokens) and overlap modest.
MAX_CHARS_SINGLE_PASS = 1500        # Chunks under this get standard single-pass CE
WINDOW_SIZE_CHARS = 1400            # ~350 tokens chunk text + 64-tok query + overhead
WINDOW_STRIDE_CHARS = 1000          # 400 char overlap between windows — catches boundary-straddling content


def _split_into_windows(text: str,
                        window_size: int = WINDOW_SIZE_CHARS,
                        stride: int = WINDOW_STRIDE_CHARS) -> list[str]:
    """Sliding-window split. Overlap ensures no boundary-cut content is missed."""
    if len(text) <= window_size:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        windows.append(text[start:end])
        if end >= len(text):
            break
        start += stride
    return windows


def _get_chunk_text(chunk: Any) -> str:
    """Accept Qdrant Point or plain dict."""
    if isinstance(chunk, dict):
        return chunk.get("text") or ""
    payload = getattr(chunk, "payload", None) or {}
    return payload.get("text") or ""


def rerank_with_maxp(
    candidates: list,
    query: str,
    reranker: Any,
    *,
    max_chars_single_pass: int = MAX_CHARS_SINGLE_PASS,
    window_size: int = WINDOW_SIZE_CHARS,
    stride: int = WINDOW_STRIDE_CHARS,
    top_k: int | None = None,
) -> list[dict]:
    """
    Score each candidate with the cross-encoder using MaxP for long chunks.

    For each candidate:
      * If chunk text ≤ max_chars_single_pass: standard single-pass scoring
        (matches existing behaviour — no regression on short chunks)
      * If chunk text > max_chars_single_pass: split into overlapping windows,
        score each window, take max. Also records which window won (for
        diagnostics).

    Returns a list of dicts (flattened from Qdrant payload + ce_score), sorted
    by ce_score descending, truncated to top_k if provided.

    Each returned dict includes:
      * ce_score           — final MaxP (or single-pass) score
      * ce_score_method    — "single_pass" | "maxp"
      * ce_score_windows   — list of per-window scores (if maxp)
      * ce_score_best_window — 0-indexed window that won (if maxp)
    """
    # Build (query, text_or_window) pairs, tagged with chunk index + window
    pair_plan: list[tuple[int, int, str]] = []  # (chunk_idx, window_idx, text)
    for i, cand in enumerate(candidates):
        text = _get_chunk_text(cand).strip()
        if not text:
            pair_plan.append((i, 0, ""))
            continue
        if len(text) <= max_chars_single_pass:
            pair_plan.append((i, 0, text))
        else:
            windows = _split_into_windows(text, window_size, stride)
            for wi, window_text in enumerate(windows):
                pair_plan.append((i, wi, window_text))

    # Score all (query, text) pairs in one batch — maximises CE throughput
    pairs_for_ce = [[query, text] for _, _, text in pair_plan]
    if not pairs_for_ce:
        return []
    all_scores = reranker.predict(pairs_for_ce)

    # Aggregate window scores per chunk with MaxP
    per_chunk_scores: dict[int, list[tuple[int, float]]] = {}
    for (chunk_idx, window_idx, _), score in zip(pair_plan, all_scores):
        per_chunk_scores.setdefault(chunk_idx, []).append((window_idx, float(score)))

    # Build output dicts with diagnostic fields
    out = []
    for i, cand in enumerate(candidates):
        window_scores = per_chunk_scores.get(i, [(0, 0.0)])
        if len(window_scores) == 1:
            method = "single_pass"
            best_score = window_scores[0][1]
            best_window = 0
            windows_detail = None
        else:
            method = "maxp"
            best_window, best_score = max(window_scores, key=lambda x: x[1])
            windows_detail = [s for _, s in window_scores]

        # Flatten payload into dict
        if isinstance(cand, dict):
            d = dict(cand)
        else:
            d = dict(getattr(cand, "payload", {}) or {})
            if "chunk_id" not in d:
                d["chunk_id"] = str(getattr(cand, "id", ""))

        d["ce_score"] = best_score
        d["ce_score_method"] = method
        if windows_detail is not None:
            d["ce_score_windows"] = windows_detail
            d["ce_score_best_window"] = best_window

        out.append(d)

    out.sort(key=lambda x: x["ce_score"], reverse=True)
    if top_k is not None:
        out = out[:top_k]
    return out


# ─── Smoke tests ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Simulate a long statute chunk where the relevant content is in the SECOND half
    # (exactly the failure mode we diagnosed)
    long_statute_text = (
        "Matrimonial Causes Act 1973, Section 25: Matters to which court is to have regard. "
        "In deciding what order to make under sections 23, 24, 24A, 24B or 24E, it shall be "
        "the duty of the court to have regard to all the circumstances of the case. "
        + "Boilerplate about sections cross-references and procedural notes. " * 20
        + "The court shall in particular have regard to the following matters: "
        "(a) the income, earning capacity, property and other financial resources; "
        "(b) the financial needs, obligations and responsibilities; "
        "(c) the standard of living enjoyed by the family; "
        "(d) the age of each party and duration of the marriage; "
        "(e) any physical or mental disability of either of the parties; "
        "(f) the contributions which each of the parties has made or is likely in the "
        "foreseeable future to make to the welfare of the family, including any "
        "contribution by looking after the home or caring for the family; "
        "(g) the conduct of each of the parties, if that conduct is such that it would "
        "in the opinion of the court be inequitable to disregard it."
    )
    short_unrelated_text = (
        "Family Procedure Rules 2010, Section 2: Interpretation. "
        "'the 1958 Act' means the Maintenance Orders Act 1958."
    )

    query = "what factors must the court consider for financial provision — conduct"

    candidates = [
        {"chunk_id": "fpr-2010-s-2",          "text": short_unrelated_text},
        {"chunk_id": "mca-1973-section-25",   "text": long_statute_text},
    ]

    print("Test 1 — single-pass baseline (existing behaviour):")
    pairs = [[query, c["text"][:512]] for c in candidates]
    baseline_scores = ce.predict(pairs)
    for c, s in zip(candidates, baseline_scores):
        print(f"    single_pass(truncated): {c['chunk_id']:28s}  ce={s:+.3f}")

    print("\nTest 2 — MaxP reranking:")
    reranked = rerank_with_maxp(candidates, query, ce)
    for r in reranked:
        method = r["ce_score_method"]
        extra = ""
        if method == "maxp":
            extra = f"  windows={r['ce_score_windows']}, best=#{r['ce_score_best_window']}"
        print(f"    {method:12s}: {r['chunk_id']:28s}  ce={r['ce_score']:+.3f}{extra}")

    # Assertion: MaxP should surface the long statute chunk
    assert reranked[0]["chunk_id"] == "mca-1973-section-25", \
        f"MaxP should rank the long statute #1 (contains relevant subsection) — got {reranked[0]['chunk_id']}"
    assert reranked[0]["ce_score_method"] == "maxp", \
        "MaxP method should have been applied to the long chunk"
    print("\n✅ MaxP correctly surfaces long statute chunk where single-pass truncation would miss it")
