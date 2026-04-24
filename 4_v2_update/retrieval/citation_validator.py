"""
citation_validator.py
=====================

Citation / hallucination validator for the AILES UK family-law RAG system.

Given an LLM output and the retrieved chunks that were fed into the LLM,
this module detects likely hallucinated legal citations (cases and statutes)
and returns a tiered confidence score 0..1.

Design principles
-----------------
* High precision over high recall. A false positive (flagging a real
  citation) erodes user trust faster than a false negative (missing a
  fake one). When borderline, we do NOT flag.
* Tiered, independently enable-able checks:
    - Tier 1: case citations must appear in retrieved chunks (text or
      case_citation field). Highest signal, highest confidence.
    - Tier 2 (opt-in): statute (act, section) tuple must appear in
      retrieved chunks. Off by default because the LLM often discusses
      statutes the chunks reference only by name.
    - Tier 3: statute year whitelist — cheap sanity check that catches
      wrong-year hallucinations (e.g. "Matrimonial Causes Act 1857"
      when only 1973 exists in the corpus). High precision.
    - Tier 4 (opt-in, advanced): cross-reference check — does the
      retrieved chunk's text contain key content terms from the LLM's
      assertion near the citation? Experimental.
* Returns a score in [0, 1]; the caller chooses its own serve threshold.

Corpus-derived constants
------------------------
STATUTE_ALIASES and STATUTE_YEAR_WHITELIST are built from the 25 unique
statute_title values in merged_all_chunks.json (Phase A inspection).

Author: Crish Nagarkar
Date: 2026-04-23
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =============================================================================
# CONSTANTS (derived from Phase A corpus inspection)
# =============================================================================

# Case citation regex — matches UK neutral citations seen in corpus:
#   [2024] EWFC 187
#   [2023] EWFC 52 (B)
#   [2020] EWHC 545 (Fam)
#   [2024] EWCOP 12
#   [2023] EWCOP 74 (T2)
#   [2006] UKHL 24           (older House of Lords)
#   [2001] UKSC 5            (Supreme Court, cited by judgments)
#   [2010] EWCA Civ 1171     (Court of Appeal, cited by judgments)
CASE_CITATION_REGEX = re.compile(
    r"\[(?P<year>\d{4})\]\s+"
    r"(?P<court>EWFC|EWHC|EWCOP|UKSC|UKHL|EWCA(?:\s+Civ)?)"
    r"\s+(?P<num>\d+)"
    r"(?:\s+\((?P<suffix>Fam|B|T1|T2|T3)\))?"
)

# Canonical statute titles seen in corpus (25 of them).
# These are the ONLY valid canonical names.
_CANONICAL_STATUTES: Dict[str, List[int]] = {
    # canonical_title : [year, ...]  (single-element list for most)
    "Civil Partnership Act 2004": [2004],
    "The Family Procedure Rules 2010": [2010],
    "Children Act 1989": [1989],
    "Children and Families Act 2014": [2014],
    "Mental Capacity Act 2005": [2005],
    "Mental Health Act 1983": [1983],
    "Adoption and Children Act 2002": [2002],
    "Domestic Violence, Crime and Victims Act 2004": [2004],
    "Human Fertilisation and Embryology Act 2008": [2008],
    "Family Law Act 1996": [1996],
    "Children and Social Work Act 2017": [2017],
    "Child Support Act 1991": [1991],
    "Domestic Abuse Act 2021": [2021],
    "Gender Recognition Act 2004": [2004],
    "Marriage Act 1949": [1949],
    "Matrimonial Causes Act 1973": [1973],
    "Matrimonial and Family Proceedings Act 1984": [1984],
    "Family Law Act 1986": [1986],
    "Divorce, Dissolution and Separation Act 2020": [2020],
    "Trusts of Land and Appointment of Trustees Act 1996": [1996],
    "Human Rights Act 1998": [1998],
    "Contempt of Court Act 1981": [1981],
    "Child Abduction and Custody Act 1985": [1985],
    "Inheritance (Provision for Family and Dependants) Act 1975": [1975],
    "Child Abduction Act 1984": [1984],
}

STATUTE_YEAR_WHITELIST: Dict[str, List[int]] = dict(_CANONICAL_STATUTES)


def _build_aliases(canonical: Dict[str, List[int]]) -> Dict[str, str]:
    """Build alias map → canonical title.

    Pattern: lowercase strings that the LLM might use are mapped to the
    canonical title in the corpus. Includes:
      - lowercase full title
      - initialism without year ('MCA', 'FLA 1996', 'CA 1989')
      - 'Matrimonial Causes Act' → 'Matrimonial Causes Act 1973'
    """
    aliases: Dict[str, str] = {}
    for title in canonical:
        aliases[title.lower()] = title
        # strip trailing year to get the "root"
        m = re.match(r"^(.+?)\s+(\d{4})\s*$", title)
        if m:
            root = m.group(1).strip()
            year = m.group(2)
            aliases[root.lower()] = title
            # initialism (first letters of content words)
            words = re.findall(r"[A-Z][a-z]*", root)
            if len(words) >= 2:
                initials = "".join(w[0] for w in words)
                aliases[initials.lower()] = title
                aliases[f"{initials} {year}".lower()] = title
    # hand-curated common abbreviations (still validated against canonical list)
    manual = {
        "mca": "Matrimonial Causes Act 1973",
        "mca 1973": "Matrimonial Causes Act 1973",
        "fla": "Family Law Act 1996",        # default to 1996 (more common)
        "fla 1996": "Family Law Act 1996",
        "fla 1986": "Family Law Act 1986",
        "ca": "Children Act 1989",            # default (not great but common)
        "ca 1989": "Children Act 1989",
        "ca 1984": "Child Abduction Act 1984",
        "ca 2002": "Adoption and Children Act 2002",
        "daa": "Domestic Abuse Act 2021",
        "daa 2021": "Domestic Abuse Act 2021",
        "hra": "Human Rights Act 1998",
        "hra 1998": "Human Rights Act 1998",
        "i(pfd)a": "Inheritance (Provision for Family and Dependants) Act 1975",
        "i(pfd)a 1975": "Inheritance (Provision for Family and Dependants) Act 1975",
        "ia 1975": "Inheritance (Provision for Family and Dependants) Act 1975",
        "inheritance act 1975": "Inheritance (Provision for Family and Dependants) Act 1975",
        "mfpa": "Matrimonial and Family Proceedings Act 1984",
        "mfpa 1984": "Matrimonial and Family Proceedings Act 1984",
        "mhc": "Mental Capacity Act 2005",
        "mca 2005": "Mental Capacity Act 2005",
        "cpa": "Civil Partnership Act 2004",
        "cpa 2004": "Civil Partnership Act 2004",
        "cs act 1991": "Child Support Act 1991",
        "csa 1991": "Child Support Act 1991",
        "fpr": "The Family Procedure Rules 2010",
        "fpr 2010": "The Family Procedure Rules 2010",
        "family procedure rules": "The Family Procedure Rules 2010",
        "family procedure rules 2010": "The Family Procedure Rules 2010",
        "tlata": "Trusts of Land and Appointment of Trustees Act 1996",
        "tlata 1996": "Trusts of Land and Appointment of Trustees Act 1996",
        "dds 2020": "Divorce, Dissolution and Separation Act 2020",
    }
    aliases.update(manual)
    return aliases


STATUTE_ALIASES: Dict[str, str] = _build_aliases(_CANONICAL_STATUTES)


# -- Case citation: normalisation helpers --

def _normalise_citation(raw: str) -> str:
    """Collapse whitespace inside a neutral citation."""
    return re.sub(r"\s+", " ", raw.strip())


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_case_citations(text: str) -> List[Dict[str, Any]]:
    """Extract neutral citations like '[2024] EWFC 187' or '[2020] EWHC 545 (Fam)'."""
    out = []
    for m in CASE_CITATION_REGEX.finditer(text):
        out.append({
            "raw": m.group(0),
            "normalised": _normalise_citation(m.group(0)),
            "year": int(m.group("year")),
            "court": re.sub(r"\s+", " ", m.group("court")).upper(),
            "number": int(m.group("num")),
            "suffix": m.group("suffix"),
            "span": (m.start(), m.end()),
        })
    return out


# Statute pattern — matches things like:
#   "Matrimonial Causes Act 1973, Section 25(2)(f)"
#   "section 33 of the Family Law Act 1996"
#   "s.25A MCA 1973"
#   "s. 42 FLA 1996"
_STATUTE_PATTERNS = [
    # "<Title year>, Section <X>"  or  "<Title year>, s <X>"
    re.compile(
        r"(?P<title>[A-Z][A-Za-z()',\- ]+?\s+(?P<year>\d{4}))\s*,?\s*"
        r"(?:section|s\.?|sec\.?)\s*(?P<section>[\w()]+)",
        re.IGNORECASE,
    ),
    # "section <X> of the <Title year>"
    re.compile(
        r"(?:section|s\.?|sec\.?)\s*(?P<section>[\w()]+)\s+of\s+(?:the\s+)?"
        r"(?P<title>[A-Z][A-Za-z()',\- ]+?\s+(?P<year>\d{4}))",
        re.IGNORECASE,
    ),
    # "s.25A MCA 1973" / "s 33 FLA 1996" — short form (abbreviation)
    re.compile(
        r"(?:section|s\.?|sec\.?)\s*(?P<section>[\w()]+)\s+(?P<title>[A-Z()]{2,10}(?:\s+\d{4})?)",
    ),
]


def _lookup_alias(raw_title: str) -> Tuple[Optional[str], Optional[int]]:
    """Map raw title text → (canonical, canonical_year)."""
    key = raw_title.strip().lower()
    if key in STATUTE_ALIASES:
        canon = STATUTE_ALIASES[key]
        return canon, STATUTE_YEAR_WHITELIST[canon][0]
    # try stripping 'the '
    if key.startswith("the "):
        k2 = key[4:]
        if k2 in STATUTE_ALIASES:
            canon = STATUTE_ALIASES[k2]
            return canon, STATUTE_YEAR_WHITELIST[canon][0]
    # try matching on the root (strip trailing year)
    m = re.match(r"^(.+?)\s+(\d{4})\s*$", key)
    if m and m.group(1) in STATUTE_ALIASES:
        canon = STATUTE_ALIASES[m.group(1)]
        try:
            yr = int(m.group(2))
        except ValueError:
            yr = None
        return canon, yr
    return None, None


def extract_statute_citations(text: str) -> List[Dict[str, Any]]:
    """Return list of {'raw','act_canonical','section','year','span'}."""
    seen_spans: set = set()
    out: List[Dict[str, Any]] = []
    for pat in _STATUTE_PATTERNS:
        for m in pat.finditer(text):
            span = (m.start(), m.end())
            # dedupe overlapping
            if any(abs(span[0] - s[0]) < 5 for s in seen_spans):
                continue
            seen_spans.add(span)
            raw_title = m.group("title")
            raw_year = m.groupdict().get("year")
            canonical, canon_year = _lookup_alias(raw_title)
            year = None
            if raw_year:
                try:
                    year = int(raw_year)
                except ValueError:
                    year = None
            elif canon_year:
                year = canon_year
            out.append({
                "raw": m.group(0),
                "act_raw": raw_title,
                "act_canonical": canonical,  # None if unknown
                "section": str(m.group("section")).strip(),
                "year": year,
                "span": span,
            })
    return out


# =============================================================================
# TIER 1 — case citation validation against retrieved chunks
# =============================================================================

def _collect_retrieved_case_strings(retrieved_chunks: Iterable[Dict[str, Any]]) -> Tuple[set, str]:
    """Collect (a) set of normalised case_citations from chunk fields,
       and (b) a concatenated text blob of chunk text."""
    citations = set()
    text_parts = []
    for ch in retrieved_chunks:
        cc = ch.get("case_citation")
        if cc:
            citations.add(_normalise_citation(cc))
        t = ch.get("text") or ""
        text_parts.append(t)
    return citations, " \n ".join(text_parts)


def validate_case_citations(
    extracted: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Tier 1. For every extracted citation, check it appears in the
    retrieved chunks (case_citation field OR anywhere in chunk text)."""
    cite_set, text_blob = _collect_retrieved_case_strings(retrieved_chunks)
    # Also build a set of normalised citations found in chunk text
    text_cites = {c["normalised"] for c in extract_case_citations(text_blob)}
    available = cite_set | text_cites

    valid, halluc = [], []
    for ext in extracted:
        if ext["normalised"] in available:
            valid.append(ext)
        else:
            halluc.append(ext)
    n = len(extracted)
    score = 1.0 if n == 0 else len(valid) / n
    return {
        "valid": valid,
        "hallucinated": halluc,
        "validation_score": score,
        "n_extracted": n,
    }


# =============================================================================
# TIER 2 — statute (act, section) validation (opt-in strict)
# =============================================================================

def validate_statute_citations(
    extracted: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
    enable_strict: bool = False,
) -> Dict[str, Any]:
    """Tier 2. When enable_strict=False, skip (defer to Tier 3).
       When True, verify each (act_canonical, section) appears in chunks'
       statute_title + section_number OR in any chunk text."""
    if not enable_strict:
        return {"enabled": False, "valid": [], "hallucinated": [],
                "validation_score": 1.0, "n_extracted": 0}

    # Build (canonical_title, parent_section) tuples from retrieved chunks
    available_tuples: set = set()
    text_blob_parts = []
    for ch in retrieved_chunks:
        title = ch.get("statute_title")
        section = ch.get("section_number")
        if title and section is not None:
            available_tuples.add((title, str(section)))
            # also parent-section from chunk_id (e.g. 25(2)(f) → 25)
        t = ch.get("text") or ""
        text_blob_parts.append(t)
    text_blob = " \n ".join(text_blob_parts).lower()

    valid, halluc = [], []
    for ext in extracted:
        canon = ext["act_canonical"]
        sec = ext["section"]
        if canon is None:
            # Can't normalise — be conservative, don't flag
            valid.append(ext)
            continue
        # Parent section for coarse match (e.g. '25(2)(f)' → '25')
        parent = re.match(r"^(\d+[A-Z]?)", sec)
        parent_sec = parent.group(1) if parent else sec
        matched = (
            (canon, parent_sec) in available_tuples
            or (canon, sec) in available_tuples
            or (canon.lower() in text_blob and f"section {sec.lower()}" in text_blob)
            or (canon.lower() in text_blob and f"s.{sec.lower()}" in text_blob)
            or (canon.lower() in text_blob and f"s {sec.lower()}" in text_blob)
        )
        if matched:
            valid.append(ext)
        else:
            halluc.append(ext)
    n = len(extracted)
    score = 1.0 if n == 0 else len(valid) / n
    return {
        "enabled": True,
        "valid": valid,
        "hallucinated": halluc,
        "validation_score": score,
        "n_extracted": n,
    }


# =============================================================================
# TIER 3 — year whitelist (cheap, high-precision)
# =============================================================================

def validate_year_whitelist(extracted: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each extracted statute, if the LLM's year conflicts with the
    canonical year in the whitelist, flag it."""
    valid, halluc = [], []
    for ext in extracted:
        canon = ext["act_canonical"]
        year = ext["year"]
        if canon is None:
            # Unknown statute — not a whitelist failure; don't flag here
            valid.append(ext)
            continue
        valid_years = STATUTE_YEAR_WHITELIST.get(canon, [])
        if year is None or year in valid_years:
            valid.append(ext)
        else:
            halluc.append({**ext,
                           "reason": f"year {year} not in whitelist for "
                                     f"{canon!r} (expected {valid_years})"})
    n = len(extracted)
    score = 1.0 if n == 0 else len(valid) / n
    return {
        "valid": valid,
        "hallucinated": halluc,
        "validation_score": score,
        "n_extracted": n,
    }


# =============================================================================
# TIER 4 — cross-reference check (experimental, opt-in)
# =============================================================================

_STOPWORDS = {"the", "a", "an", "of", "to", "and", "in", "is", "that",
              "for", "on", "with", "by", "at", "from", "this", "it", "as",
              "be", "or", "are", "section", "s", "act", "court", "case"}


def _content_terms(text: str) -> set:
    toks = re.findall(r"[A-Za-z]{4,}", text.lower())
    return {t for t in toks if t not in _STOPWORDS}


def cross_reference_check(
    extracted: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
    window: int = 200,
    min_overlap: int = 3,
    llm_output: Optional[str] = None,
) -> Dict[str, Any]:
    """Tier 4 (advanced, opt-in).

    For each extracted citation in llm_output, grab a window of text
    around it, extract content terms, and verify those terms overlap
    with the text of *some* retrieved chunk. If no chunk's text has
    >= min_overlap shared content terms, flag.

    Low confidence — keyword overlap is a rough proxy.
    """
    if llm_output is None:
        # Nothing to cross-reference against; return neutral
        return {"enabled": True, "supported": [], "unsupported": [],
                "validation_score": 1.0, "n_extracted": 0}

    chunk_term_sets = [_content_terms(ch.get("text") or "") for ch in retrieved_chunks]

    supported, unsupported = [], []
    for ext in extracted:
        s, e = ext["span"]
        ctx_start = max(0, s - window)
        ctx_end = min(len(llm_output), e + window)
        ctx = llm_output[ctx_start:ctx_end]
        ctx_terms = _content_terms(ctx)
        # Remove the citation tokens themselves
        ctx_terms -= _content_terms(ext["raw"])
        if not ctx_terms:
            supported.append(ext)
            continue
        best = max((len(ctx_terms & ts) for ts in chunk_term_sets), default=0)
        if best >= min_overlap:
            supported.append({**ext, "overlap": best})
        else:
            unsupported.append({**ext, "overlap": best})
    n = len(extracted)
    score = 1.0 if n == 0 else len(supported) / n
    return {
        "enabled": True,
        "supported": supported,
        "unsupported": unsupported,
        "validation_score": score,
        "n_extracted": n,
    }


# =============================================================================
# OUTPUT POST-PROCESSING
# =============================================================================

def strip_or_flag_hallucinations(
    llm_output: str,
    validation_results: Dict[str, Any],
    mode: str = "flag",
) -> str:
    """Apply a post-processing action to the LLM output.

    mode='passthrough'  — return unchanged.
    mode='flag'         — append a footer listing flagged citations.
    mode='strip'        — remove sentences containing hallucinated citations.
    """
    if mode == "passthrough":
        return llm_output

    flagged: List[Tuple[Tuple[int, int], str]] = []
    for key in ("tier1_case_validation", "tier2_statute_validation",
                "tier3_year_validation"):
        block = validation_results.get(key)
        if not block:
            continue
        for h in block.get("hallucinated", []):
            flagged.append((tuple(h["span"]), h.get("raw", "")))

    if mode == "flag":
        if not flagged:
            return llm_output
        footer_lines = ["", "", "---", "[Citation warnings — verify before relying on these]:"]
        for span, raw in flagged:
            footer_lines.append(f"  • {raw!r} could not be validated against retrieved chunks.")
        return llm_output + "\n".join(footer_lines)

    if mode == "strip":
        if not flagged:
            return llm_output
        # Build sentence spans in the output and drop sentences with any flagged span inside.
        spans = sorted({s for s, _ in flagged})
        keep = list(llm_output)
        # Simple strategy: blank out each sentence containing a flagged citation
        out_parts = []
        sentences = re.split(r"(?<=[.!?])\s+", llm_output)
        idx = 0
        for sent in sentences:
            s_start = idx
            s_end = idx + len(sent)
            idx = s_end + 1
            if any(s_start <= fs[0] < s_end for fs in spans):
                continue  # drop this sentence
            out_parts.append(sent)
        return " ".join(out_parts)

    raise ValueError(f"unknown mode: {mode!r}")


# =============================================================================
# TOP-LEVEL ENTRY
# =============================================================================

# Weights for combining tier scores. Weights sum to 1 only over enabled tiers.
_TIER_WEIGHTS = {
    "tier1_case_validation": 0.55,   # highest-signal
    "tier3_year_validation": 0.30,
    "tier2_statute_validation": 0.10,
    "tier4_cross_reference": 0.05,
}


def validate_llm_output(
    llm_output: str,
    retrieved_chunks: List[Dict[str, Any]],
    enable_strict_statute: bool = False,
    enable_cross_reference: bool = False,
    mode: str = "flag",
    serve_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Top-level entry point. See module docstring for output schema."""
    # 1. Extract
    cases = extract_case_citations(llm_output)
    statutes = extract_statute_citations(llm_output)

    # 2. Validate each tier
    t1 = validate_case_citations(cases, retrieved_chunks)
    t3 = validate_year_whitelist(statutes)
    t2 = validate_statute_citations(statutes, retrieved_chunks,
                                    enable_strict=enable_strict_statute)
    t4 = (cross_reference_check(cases + statutes, retrieved_chunks,
                                llm_output=llm_output)
          if enable_cross_reference else None)

    # 3. Compute weighted confidence score over ENABLED tiers
    enabled = {"tier1_case_validation": t1["validation_score"],
               "tier3_year_validation": t3["validation_score"]}
    if enable_strict_statute:
        enabled["tier2_statute_validation"] = t2["validation_score"]
    if enable_cross_reference and t4 is not None:
        enabled["tier4_cross_reference"] = t4["validation_score"]

    total_w = sum(_TIER_WEIGHTS[k] for k in enabled)
    confidence = (sum(_TIER_WEIGHTS[k] * v for k, v in enabled.items()) / total_w
                  if total_w > 0 else 1.0)

    # 4. Count flagged
    total_flagged = (len(t1["hallucinated"])
                     + len(t3["hallucinated"])
                     + (len(t2["hallucinated"]) if enable_strict_statute else 0))

    # 5. Build intermediate results for post-processing
    interim = {
        "tier1_case_validation": t1,
        "tier3_year_validation": t3,
        "tier2_statute_validation": t2 if enable_strict_statute else None,
        "tier4_cross_reference": t4,
    }
    modified = strip_or_flag_hallucinations(llm_output, interim, mode=mode)

    return {
        "modified_output": modified,
        "tier1_case_validation": t1,
        "tier2_statute_validation": t2 if enable_strict_statute else None,
        "tier3_year_validation": t3,
        "tier4_cross_reference": t4,
        "total_flagged": total_flagged,
        "confidence_score": round(confidence, 3),
        "safe_to_serve": confidence >= serve_threshold,
    }


# =============================================================================
# SMOKE TESTS
# =============================================================================

def _make_chunk(chunk_id="x", text="", case_citation=None,
                statute_title=None, section_number=None):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "case_citation": case_citation,
        "statute_title": statute_title,
        "section_number": section_number,
    }


def _run_smoke_tests() -> Tuple[int, int]:
    tests = []

    # --- Test 1: VALID case citation present in retrieved chunks ---
    llm1 = ("Under the reasoning in OG v AG [2020] EWFC 52, the court set out "
            "four categories of conduct.")
    chunks1 = [_make_chunk(
        chunk_id="ewfc-2020-[2020]-EWFC-52-para_34",
        text=("Conduct rears its head in financial remedy cases in four distinct "
              "scenarios (see OG v AG [2020] EWFC 52 at para 34)."),
        case_citation="[2020] EWFC 52",
    )]
    r1 = validate_llm_output(llm1, chunks1)
    tests.append(("1. valid case citation",
                  r1["tier1_case_validation"]["validation_score"] == 1.0
                  and r1["total_flagged"] == 0
                  and r1["safe_to_serve"]))

    # --- Test 2: HALLUCINATED case citation absent from retrieved chunks ---
    llm2 = ("As established in Smith v Smith [2024] EWFC 999, non-disclosure "
            "warrants adverse inferences.")
    chunks2 = [_make_chunk(
        chunk_id="ewfc-2020-[2020]-EWFC-52-para_34",
        text="OG v AG [2020] EWFC 52 sets out four categories of conduct.",
        case_citation="[2020] EWFC 52",
    )]
    r2 = validate_llm_output(llm2, chunks2)
    tests.append(("2. hallucinated case citation",
                  len(r2["tier1_case_validation"]["hallucinated"]) == 1
                  and r2["total_flagged"] >= 1
                  and "[2024] EWFC 999" in r2["modified_output"]))

    # --- Test 3: YEAR HALLUCINATION — wrong year for MCA ---
    llm3 = ("Section 25 of the Matrimonial Causes Act 1857 requires the court "
            "to have regard to all the circumstances.")
    chunks3 = [_make_chunk(
        chunk_id="mca-1973-section-25-full",
        text="Matrimonial Causes Act 1973, Section 25: ...",
        statute_title="Matrimonial Causes Act 1973",
        section_number="25",
    )]
    r3 = validate_llm_output(llm3, chunks3)
    tests.append(("3. year hallucination (MCA 1857)",
                  len(r3["tier3_year_validation"]["hallucinated"]) >= 1
                  and r3["total_flagged"] >= 1))

    # --- Test 4: ABBREVIATION that should normalise ---
    llm4 = "Under s.25A MCA 1973, the clean-break duty applies."
    extracted = extract_statute_citations(llm4)
    normalised_ok = any(
        e["act_canonical"] == "Matrimonial Causes Act 1973" for e in extracted
    )
    tests.append(("4. abbreviation MCA → Matrimonial Causes Act 1973",
                  normalised_ok and len(extracted) >= 1))

    # --- Test 5: Real statute, wrong section (strict mode) ---
    llm5 = "Section 999 of the Matrimonial Causes Act 1973 is the relevant provision."
    chunks5 = [_make_chunk(
        chunk_id="mca-1973-section-25-full",
        text="Matrimonial Causes Act 1973, Section 25: ...",
        statute_title="Matrimonial Causes Act 1973",
        section_number="25",
    )]
    r5 = validate_llm_output(llm5, chunks5, enable_strict_statute=True)
    tests.append(("5. real statute, wrong section (strict)",
                  r5["tier2_statute_validation"] is not None
                  and len(r5["tier2_statute_validation"]["hallucinated"]) >= 1
                  and r5["total_flagged"] >= 1))

    passed = sum(1 for _, ok in tests if ok)
    print("-" * 60)
    print("SMOKE TESTS")
    print("-" * 60)
    for name, ok in tests:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    return passed, len(tests)


def _print_design_summary() -> None:
    print("\n" + "=" * 72)
    print("DESIGN SUMMARY")
    print("=" * 72)
    print("""\
Tier 1 — Case citation validation       [default: ON]
  Validates every [YEAR] COURT NUM (SUFFIX?) pattern in the LLM output is
  either (a) in a retrieved chunk's case_citation field, or (b) present in
  any retrieved chunk's text. Confidence: HIGH (exact-string match).

Tier 2 — Statute (act, section) validation   [default: OFF — opt-in strict]
  Normalises act to canonical title via STATUTE_ALIASES. Verifies
  (canonical_title, parent_section) against retrieved chunks' metadata
  and text. Kept off by default because the LLM legitimately discusses
  statutes not cited in the chunks. Confidence: MEDIUM.

Tier 3 — Statute year whitelist          [default: ON]
  For every statute mention whose act normalises to a canonical title,
  checks the year matches STATUTE_YEAR_WHITELIST. Flags cases like
  'Matrimonial Causes Act 1857'. Cheap, high precision. Confidence: HIGH.

Tier 4 — Cross-reference check           [default: OFF — experimental]
  Keyword-overlap between a window around the citation in the LLM output
  and the retrieved chunks' text. Rough signal; false-positive prone on
  short assertions. Confidence: LOW.

Default weights when computing confidence_score:
  t1=0.55, t3=0.30, t2=0.10 (if enabled), t4=0.05 (if enabled).
Weights renormalise over enabled tiers.

Safe-to-serve default: confidence_score >= 0.7.

Known limitations
-----------------
1. Sub-paragraph sections (25(2)(f)) are only validated at parent-section
   granularity by Tier 2 when chunks store only the parent section_number.
2. Citation normalisation is whitespace-insensitive but case-sensitive on
   the court token (case-upper only). OK for UK neutral citations.
3. Aliases like 'CA' are genuinely ambiguous across CA 1989 / CA 2002 /
   CA 1984. Defaults are documented; callers needing precision should
   disable aliases or use full titles in prompts.
4. Year whitelist is 1:1 (one canonical year per act). If new post-2024
   amendments are ingested, the whitelist needs regenerating from corpus.
5. Corpus note for data team: judgment chunks have doc_type=null in
   merged_all_chunks.json. The 'court' field is populated, but the
   handoff brief's claim that doc_type='EWHC-Family' etc. is incorrect.

How to enable / disable each tier
---------------------------------
validate_llm_output(
    llm_output, retrieved_chunks,
    enable_strict_statute=True,     # turn Tier 2 ON
    enable_cross_reference=True,    # turn Tier 4 ON
    mode='flag',                    # 'flag' | 'strip' | 'passthrough'
    serve_threshold=0.7,            # raise/lower the pass bar
)
""")


if __name__ == "__main__":
    passed, total = _run_smoke_tests()
    _print_design_summary()
    if passed != total:
        raise SystemExit(1)
