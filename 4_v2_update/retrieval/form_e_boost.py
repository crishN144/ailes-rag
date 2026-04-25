"""
form_e_boost.py — case-context-aware metadata boost.

Sits between cross-encoder rerank and context assembly. When the user has an
active case with a parsed Form E, applies small additive boosts to chunks
that are *more* relevant given the case facts, then re-sorts.

Design principles
─────────────────
1. Boosts are TIE-BREAKERS on already-relevant chunks. They never rescue
   chunks the cross-encoder pushed out of the top pool — that experiment
   (`reference/experimental_archived/statute_boost.py`) was empirically
   disqualified.
2. Magnitudes are deliberately small (+0.10 to +0.20). The cross-encoder
   score range we observe in production is roughly [-11, +3] (per Langfuse
   traces), so a +0.20 nudge meaningfully reorders chunks within ±0.5 of
   each other but cannot override a strong CE signal.
3. If there is no active case (case_context is None / empty), the function
   is a pass-through. We do NOT try to infer Form-E-like signals from
   conversation history.
4. Court-type boost (EWFC / EWHC-Fam) is INTENTIONALLY OMITTED. The corpus
   is already ~99% from those courts, so the boost only shuffles within
   the same bucket.

Integration
───────────
Wire post-rerank, pre-context-assembly:

    reranked = cross_encoder_rerank(candidates, user_query)
    boosted  = apply_form_e_boost(reranked, case_context)
    context_block = assemble_context(boosted[:10])

Expected case_context shape (from the Form E parser):

    {
        "has_children": bool,           # any minor children of the family
        "has_business_assets": bool,    # privately-held business interests
        "marriage_years": int,          # length of marriage in years
    }

Missing keys are treated as False / 0 — function is robust to a partially
populated dict.
"""

from __future__ import annotations

import re
from typing import Any

# ─── Boost magnitudes ────────────────────────────────────────────────────
# Tuned for the production CE-score range. Sum of all three boosts hitting
# simultaneously caps at +0.45, which is enough to flip a chunk that was
# half-a-point behind but cannot mask a clearly weak CE signal.
BOOST_CHILDREN_ACT = 0.20      # Children Act 1989 when has_children
BOOST_S25_2A_BUSINESS = 0.15   # s.25(2)(a) (financial resources) when business assets
BOOST_SHARING_PRINCIPLE = 0.10 # Sharing-principle precedents when long marriage
LONG_MARRIAGE_YEARS = 10       # threshold for sharing-principle boost

# Phrases that indicate a chunk is about the sharing principle. Drawn from
# the canonical UK family-law authorities (White, Miller/McFarlane, Charman).
SHARING_PRINCIPLE_MARKERS = (
    "sharing principle",
    "white v white",
    "miller v miller",
    "mcfarlane v mcfarlane",
    "charman v charman",
    "equal sharing",
    "yardstick of equality",
)


# ─── Helpers ─────────────────────────────────────────────────────────────
def _is_children_act_chunk(chunk: dict) -> bool:
    """True if the chunk is from the Children Act 1989."""
    cid = (chunk.get("chunk_id") or "").lower()
    if cid.startswith("ca-1989-") or cid.startswith("ca1989-"):
        return True
    statute = (chunk.get("statute_title") or chunk.get("act_name") or "").lower()
    return "children act 1989" in statute


_S25_2A_RX = re.compile(r"section[-_\s]*25\s*\(?\s*2\s*\)?\s*\(?\s*a\s*\)?", re.I)


def _is_s25_2a_chunk(chunk: dict) -> bool:
    """
    True if the chunk is MCA 1973 s.25(2)(a) — income, earning capacity,
    property and other financial resources. Tolerates a few chunk_id formats.
    """
    cid = (chunk.get("chunk_id") or "").lower()
    if "section-25(2)(a)" in cid or "section-25-2-a" in cid:
        return True
    section = (chunk.get("section_label") or chunk.get("section_number") or "")
    return bool(_S25_2A_RX.search(str(section)))


def _is_sharing_principle_chunk(chunk: dict) -> bool:
    """
    True if the chunk text or citation references the sharing principle
    or one of its canonical authorities.
    """
    haystack = " ".join(
        str(chunk.get(k, "") or "")
        for k in ("text", "case_citation", "topics", "summary", "legal_principles")
    ).lower()
    return any(marker in haystack for marker in SHARING_PRINCIPLE_MARKERS)


# ─── Main entry point ────────────────────────────────────────────────────
def apply_form_e_boost(
    reranked_chunks: list[dict],
    case_context: dict | None,
    *,
    score_field: str = "ce_score",
) -> list[dict]:
    """
    Apply Form-E-conditional metadata boosts to a reranked chunk list.

    Parameters
    ----------
    reranked_chunks
        List of chunk dicts already ordered by cross-encoder score (descending).
        Each chunk must carry a numeric `score_field` (default "ce_score").
        Chunks may carry any of: chunk_id, statute_title, act_name,
        section_label, text, case_citation, topics — only used for matching.
    case_context
        Form E summary dict (may be None for general chat / no active case).
        Recognised keys: has_children (bool), has_business_assets (bool),
        marriage_years (int). Missing keys default to False / 0.
    score_field
        Name of the field on each chunk holding the cross-encoder score.

    Returns
    -------
    list[dict]
        New list of chunks (same objects, mutated in place with two new keys:
        `final_score` and `boost_reasons`) sorted descending by final_score.
        If case_context is None / empty, final_score == ce_score and
        boost_reasons is an empty list.
    """
    # Pass-through if there's no active case.
    if not case_context:
        for c in reranked_chunks:
            c["final_score"] = float(c.get(score_field, 0.0))
            c["boost_reasons"] = []
        return reranked_chunks

    has_children = bool(case_context.get("has_children", False))
    has_business = bool(case_context.get("has_business_assets", False))
    marriage_years = int(case_context.get("marriage_years", 0) or 0)
    long_marriage = marriage_years >= LONG_MARRIAGE_YEARS

    for c in reranked_chunks:
        ce_score = float(c.get(score_field, 0.0))
        boost = 0.0
        reasons: list[str] = []

        if has_children and _is_children_act_chunk(c):
            boost += BOOST_CHILDREN_ACT
            reasons.append(f"children_act+{BOOST_CHILDREN_ACT:.2f}")

        if has_business and _is_s25_2a_chunk(c):
            boost += BOOST_S25_2A_BUSINESS
            reasons.append(f"s25_2a_business+{BOOST_S25_2A_BUSINESS:.2f}")

        if long_marriage and _is_sharing_principle_chunk(c):
            boost += BOOST_SHARING_PRINCIPLE
            reasons.append(f"sharing_principle+{BOOST_SHARING_PRINCIPLE:.2f}")

        c["final_score"] = ce_score + boost
        c["boost_reasons"] = reasons

    return sorted(reranked_chunks, key=lambda x: -x["final_score"])


# ─── Smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated post-rerank list (CE scores typical of production range).
    chunks = [
        {
            "chunk_id": "mca-1973-section-25(2)(a)",
            "statute_title": "Matrimonial Causes Act 1973",
            "ce_score": 2.10,
            "text": "Income, earning capacity, property and other financial resources...",
        },
        {
            "chunk_id": "ca-1989-section-1",
            "statute_title": "Children Act 1989",
            "ce_score": 1.85,
            "text": "When a court determines any question with respect to the upbringing of a child...",
        },
        {
            "chunk_id": "ewhc-family-2000-[2000]-UKHL-54-para_42",
            "case_citation": "[2000] UKHL 54",
            "ce_score": 1.50,
            "text": "The yardstick of equality is to be departed from only for good reason. White v White...",
        },
        {
            "chunk_id": "fla-1996-section-42",
            "statute_title": "Family Law Act 1996",
            "ce_score": 0.40,
            "text": "Non-molestation orders.",
        },
    ]

    print("=== Case 1: no Form E (general chat) — pass-through ===")
    out = apply_form_e_boost([dict(c) for c in chunks], None)
    for c in out:
        print(f"  {c['final_score']:>6.3f}  {c['chunk_id']:<60s} {c['boost_reasons']}")
    assert all(c["boost_reasons"] == [] for c in out), "pass-through must apply no boosts"

    print()
    print("=== Case 2: family with kids, business assets, 15-yr marriage ===")
    out = apply_form_e_boost(
        [dict(c) for c in chunks],
        {"has_children": True, "has_business_assets": True, "marriage_years": 15},
    )
    for c in out:
        print(f"  {c['final_score']:>6.3f}  {c['chunk_id']:<60s} {c['boost_reasons']}")

    # Children Act chunk should now be at the top: 1.85 + 0.20 = 2.05 > 2.10? No, still under.
    # Actually CE 2.10 (s.25(2)(a)) gets +0.15 = 2.25; CA 1989 gets 1.85 + 0.20 = 2.05.
    # Sharing chunk gets 1.50 + 0.10 = 1.60. Order: 2.25, 2.05, 1.60, 0.40.
    assert out[0]["chunk_id"].startswith("mca-1973-section-25(2)(a)"), "s.25(2)(a) should top with business boost"
    assert out[1]["chunk_id"].startswith("ca-1989-"), "Children Act should be #2 with kids boost"
    assert out[2]["case_citation"] == "[2000] UKHL 54", "Sharing-principle chunk should be #3"
    assert out[3]["chunk_id"].startswith("fla-1996-"), "FLA chunk gets no boost, stays last"

    print()
    print("=== Case 3: childless, no business, short marriage — no boosts trigger ===")
    out = apply_form_e_boost(
        [dict(c) for c in chunks],
        {"has_children": False, "has_business_assets": False, "marriage_years": 3},
    )
    for c in out:
        print(f"  {c['final_score']:>6.3f}  {c['chunk_id']:<60s} {c['boost_reasons']}")
    assert all(c["boost_reasons"] == [] for c in out), "no conditions met → no boosts"
    # Order should match original CE ordering
    assert [c["ce_score"] for c in out] == sorted([c["ce_score"] for c in out], reverse=True)

    print()
    print("=== Case 4: kids only — only Children Act boost fires ===")
    out = apply_form_e_boost(
        [dict(c) for c in chunks],
        {"has_children": True, "has_business_assets": False, "marriage_years": 5},
    )
    boosted = [c for c in out if c["boost_reasons"]]
    assert len(boosted) == 1, "exactly one chunk should be boosted"
    assert boosted[0]["chunk_id"].startswith("ca-1989-"), "only Children Act chunk should be boosted"

    print()
    print("All smoke tests passed.")
