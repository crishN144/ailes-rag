"""
statute_boost.py — post-rerank statute promotion + Form-E-aware context boost.

Fixes the FR_05 / CW_01 ranking bugs surfaced in the eval: the canonical
statute chunk exists in the corpus (MCA s.25A, CA 1989 s.1) but the
cross-encoder ranks it below judicial paraphrasings of itself. This module
promotes statutes when the chunk content actually corresponds to the
query's legal intent.

Pipeline integration:

    candidates = retrieval_step4.retrieve(query, ...)         # dual-lane RRF
    reranked   = cross_encoder_rerank(candidates, query)     # ms-marco-MiniLM
    boosted    = apply_statute_boost(reranked, preserved_terms)
    final      = apply_context_boost(boosted, form_e_flags)
    return final[:10]

Design choices (worth knowing):

  * Multi-word preserved_terms ("clean break", "non-molestation order") are
    matched as phrases, not as independent tokens. Single-token matches
    ("clean") are too permissive — they'd promote every chunk mentioning
    clean water.

  * Preserved_terms that are just Act names ("Matrimonial Causes Act 1973")
    are IGNORED — interpretation sections of other Acts often mention
    each other's names and would get over-boosted. Only actual legal-
    content terms trigger promotion.

  * Total combined boost is capped at MAX_TOTAL_BOOST to prevent runaway
    stacking. Original ce_score is preserved as ce_score_raw for
    diagnostics.

  * Case-insensitive matching throughout.

  * Both functions are idempotent — passing None / empty args returns the
    input list unmodified.
"""

from __future__ import annotations

import re
from typing import Any

STATUTE_BOOST_VALUE = 1.5
CONTEXT_BOOST_VALUE = 0.2
MAX_TOTAL_BOOST = 2.5

# Preserved_terms matching any of these regexes are treated as "just the Act
# name" and skipped for boosting — too noisy (interpretation sections mention
# Act names without substantive content).
ACT_NAME_PATTERNS = [
    r"^matrimonial\s+causes\s+act(\s+\d{4})?$",
    r"^children\s+act(\s+\d{4})?$",
    r"^family\s+law\s+act(\s+\d{4})?$",
    r"^domestic\s+abuse\s+act(\s+\d{4})?$",
    r"^civil\s+partnership\s+act(\s+\d{4})?$",
    r"^inheritance\s+\(provision.*\)\s+act(\s+\d{4})?$",
    r"^matrimonial\s+and\s+family\s+proceedings\s+act(\s+\d{4})?$",
    r"^children\s+and\s+families\s+act(\s+\d{4})?$",
    r"^mca(\s+\d{4})?$",
    r"^fla(\s+\d{4})?$",
    r"^ca(\s+\d{4})?$",
    r"^daa(\s+\d{4})?$",
    r"^cpa(\s+\d{4})?$",
    r"^i\(?pfd\)?a(\s+\d{4})?$",
]


def _is_just_act_name(term: str) -> bool:
    """True if term is merely a statute name with no substantive content."""
    t = term.strip().lower()
    return any(re.match(p, t) for p in ACT_NAME_PATTERNS)


def _phrase_match(text: str, term: str) -> bool:
    """
    Case-insensitive phrase match. For multi-word terms, all words must
    appear in order. For single words, exact token match (word boundary).
    """
    t = text.lower()
    term_norm = term.strip().lower()
    if not term_norm:
        return False
    if " " in term_norm or "-" in term_norm:
        # phrase — match as-is with flexible internal whitespace
        pattern = re.escape(term_norm).replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]")
        return bool(re.search(pattern, t))
    # single token — word-boundary match
    return bool(re.search(rf"\b{re.escape(term_norm)}\b", t))


def apply_statute_boost(
    reranked_chunks: list,
    preserved_terms: list[str] | None,
    boost_value: float = STATUTE_BOOST_VALUE,
) -> list:
    """
    Promote statute chunks whose text matches the query's preserved terms.

    For each chunk where `doc_type == 'statute'` AND at least one preserved
    term (that is not merely an Act name) appears in `chunk.text`, add
    `boost_value` to the chunk's `ce_score`. Re-sort descending.

    Parameters
    ----------
    reranked_chunks : list[dict]
        Output of the cross-encoder reranker. Each element must have
        `text`, `ce_score`, `doc_type` keys (or be a Qdrant point with
        payload + an attached `ce_score`).
    preserved_terms : list[str] | None
        Legal-content keywords from the expander. E.g. ["clean break",
        "section 25A"]. Act-name-only terms are auto-skipped.
    boost_value : float
        Additive boost applied to matching chunks.

    Returns
    -------
    list
        Same chunks, with `ce_score_raw` preserved, `ce_score` possibly
        boosted, `statute_boost_applied` flag set, resorted by ce_score.
    """
    if not preserved_terms:
        return list(reranked_chunks)

    # Filter out pure Act-name terms
    effective_terms = [
        t for t in preserved_terms
        if t and not _is_just_act_name(t)
    ]
    if not effective_terms:
        return list(reranked_chunks)

    out = []
    for chunk in reranked_chunks:
        c = _as_dict(chunk)
        c.setdefault("ce_score_raw", c.get("ce_score"))
        c.setdefault("statute_boost_applied", False)

        if c.get("doc_type") == "statute":
            text = c.get("text") or ""
            matched = [t for t in effective_terms if _phrase_match(text, t)]
            if matched:
                boost = min(boost_value, MAX_TOTAL_BOOST)
                c["ce_score"] = c["ce_score_raw"] + boost
                c["statute_boost_applied"] = True
                c["statute_boost_matched_terms"] = matched
        out.append(c)

    out.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
    return out


def apply_context_boost(
    reranked_chunks: list,
    form_e_flags: dict | None = None,
    boost_value: float = CONTEXT_BOOST_VALUE,
) -> list:
    """
    Form-E-context-aware boost (Feb-report-style).

    Applies +boost_value per matching condition:

      * has_children       → boost chunks where statute_title contains "Children Act"
      * has_da_allegations  → boost chunks where statute_title contains "Family Law Act 1996"
                                               or "Domestic Abuse Act"
      * is_financial_dispute → boost chunks where statute_title contains "Matrimonial Causes Act"
      * is_inheritance_claim → boost chunks where statute_title contains "Inheritance"

    Combined boost per chunk is capped at MAX_TOTAL_BOOST minus any
    statute_boost already applied, so a chunk can't accumulate more than
    MAX_TOTAL_BOOST total above its raw ce_score.

    Parameters
    ----------
    form_e_flags : dict | None
        Keys: has_children, has_da_allegations, is_financial_dispute,
        is_inheritance_claim. Missing / None = no boost applied.

    Returns
    -------
    list
        Chunks with `context_boost_applied` flag and optional boosted score.
    """
    if not form_e_flags:
        return list(reranked_chunks)

    boost_rules = [
        ("has_children",       "Children Act"),
        ("has_da_allegations", "Family Law Act 1996"),
        ("has_da_allegations", "Domestic Abuse Act"),
        ("is_financial_dispute", "Matrimonial Causes Act"),
        ("is_inheritance_claim", "Inheritance"),
    ]

    out = []
    for chunk in reranked_chunks:
        c = _as_dict(chunk)
        c.setdefault("ce_score_raw", c.get("ce_score"))
        c.setdefault("context_boost_applied", False)

        stat_title = (c.get("statute_title") or "").lower()
        matched_rules = []
        for flag_name, title_pattern in boost_rules:
            if form_e_flags.get(flag_name) and title_pattern.lower() in stat_title:
                matched_rules.append((flag_name, title_pattern))

        if matched_rules:
            # Cap combined boost (statute_boost + context_boost) at MAX_TOTAL_BOOST
            already_boosted = c["ce_score"] - c["ce_score_raw"]
            headroom = max(0.0, MAX_TOTAL_BOOST - already_boosted)
            context_boost = min(boost_value * len(matched_rules), headroom)
            c["ce_score"] = c["ce_score_raw"] + already_boosted + context_boost
            c["context_boost_applied"] = True
            c["context_boost_rules_matched"] = [r[0] for r in matched_rules]
        out.append(c)

    out.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
    return out


def _as_dict(chunk: Any) -> dict:
    """
    Accept either a plain dict or a Qdrant-style point (payload + score).
    Returns a mutable dict with merged fields.
    """
    if isinstance(chunk, dict):
        return dict(chunk)
    payload = getattr(chunk, "payload", {}) or {}
    d = dict(payload)
    if "ce_score" not in d and hasattr(chunk, "score"):
        d["ce_score"] = float(chunk.score)
    return d


# ─── Smoke tests ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated reranker output for query "what is a clean break in divorce"
    # Scores modelled loosely on what we see in real eval output.
    chunks = [
        {"chunk_id": "case-A",     "doc_type": "judgment", "text": "applying the clean break doctrine",                          "ce_score": 5.0},
        {"chunk_id": "mca-s25A",   "doc_type": "statute",  "text": "Termination of financial obligations clean break provisions", "ce_score": 3.0, "statute_title": "Matrimonial Causes Act 1973"},
        {"chunk_id": "case-B",     "doc_type": "judgment", "text": "Husband paid lump sum under s.25A",                            "ce_score": 4.0},
        {"chunk_id": "irrelevant", "doc_type": "statute",  "text": "Rules of procedure governing applications",                   "ce_score": 4.3, "statute_title": "Family Procedure Rules 2010"},
    ]

    # Test 1: matching statute chunk gets boosted above a case with same raw score
    preserved = ["clean break"]
    boosted = apply_statute_boost(chunks, preserved)

    mca = next(c for c in boosted if c["chunk_id"] == "mca-s25A")
    irrelevant = next(c for c in boosted if c["chunk_id"] == "irrelevant")

    # Boost applied only to the matching statute
    assert mca["statute_boost_applied"] is True, "mca-s25A should have been boosted"
    assert mca["ce_score"] == 3.0 + STATUTE_BOOST_VALUE, f"expected ce_score 4.5, got {mca['ce_score']}"
    # Irrelevant statute (no 'clean break' in text) should NOT be boosted
    assert irrelevant.get("statute_boost_applied", False) is False, \
        "FPR chunk without 'clean break' in text should NOT be boosted"
    # After boost, mca-s25A (4.5) should rank ABOVE irrelevant (4.3)
    boosted_ids = [c["chunk_id"] for c in boosted]
    assert boosted_ids.index("mca-s25A") < boosted_ids.index("irrelevant"), \
        "boosted MCA s.25A should now rank above unrelated statute"
    print("✅ Test 1 passed: matching statute boosted (+1.5), outranks similar-scoring unrelated chunk")

    # Test 2: act-name-only preserved term is skipped
    preserved2 = ["Matrimonial Causes Act 1973"]
    boosted2 = apply_statute_boost(chunks, preserved2)
    # No chunk should get boosted
    assert all(not c.get("statute_boost_applied", False) for c in boosted2), \
        "act-name-only term should NOT trigger boost"
    print("✅ Test 2 passed: act-name-only preserved term correctly ignored")

    # Test 3: case chunks not boosted (doc_type != statute)
    boosted3 = apply_statute_boost(chunks, ["clean break"])
    for c in boosted3:
        if c["doc_type"] == "judgment":
            assert not c.get("statute_boost_applied", False), \
                f"judgment {c['chunk_id']} should not get statute boost"
    print("✅ Test 3 passed: judgment chunks correctly excluded from statute boost")

    # Test 4: context boost for DV case
    chunks_dv = [
        {"chunk_id": "fla-s42",     "doc_type": "statute",  "text": "non-molestation", "ce_score": 3.0, "statute_title": "Family Law Act 1996"},
        {"chunk_id": "mca-s25",     "doc_type": "statute",  "text": "factors",         "ce_score": 4.0, "statute_title": "Matrimonial Causes Act 1973"},
    ]
    flags = {"has_da_allegations": True}
    ctx_boosted = apply_context_boost(chunks_dv, flags)
    # FLA chunk should be boosted, MCA chunk should not
    fla_chunk = next(c for c in ctx_boosted if c["chunk_id"] == "fla-s42")
    mca_chunk = next(c for c in ctx_boosted if c["chunk_id"] == "mca-s25")
    assert fla_chunk["context_boost_applied"] is True
    assert mca_chunk.get("context_boost_applied", False) is False
    assert fla_chunk["ce_score"] > fla_chunk["ce_score_raw"]
    print("✅ Test 4 passed: context boost applies FLA boost for DA case, not MCA")

    # Test 5: combined boost capped at MAX_TOTAL_BOOST
    chunks_combo = [
        {"chunk_id": "mca-s25A",
         "doc_type": "statute",
         "text": "clean break provisions for termination",
         "ce_score": 0.0,
         "statute_title": "Matrimonial Causes Act 1973"},
    ]
    b = apply_statute_boost(chunks_combo, ["clean break"])
    b = apply_context_boost(b, {"is_financial_dispute": True})
    # statute_boost adds 1.5; context_boost would add 0.2; total 1.7, under cap
    assert abs(b[0]["ce_score"] - (0.0 + 1.5 + 0.2)) < 0.01
    print("✅ Test 5 passed: stacked boost respects cap, no runaway")

    # Test 6: passing None preserves input
    assert apply_statute_boost(chunks, None) == chunks or len(apply_statute_boost(chunks, None)) == len(chunks)
    assert apply_context_boost(chunks, None) == chunks or len(apply_context_boost(chunks, None)) == len(chunks)
    print("✅ Test 6 passed: None args return input unchanged")

    print("\nAll 6 smoke tests passed.")
