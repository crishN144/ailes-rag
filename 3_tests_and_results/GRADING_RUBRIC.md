# AILES Retrieval Grading — Claude Code Rubric

**How to use this:** paste this whole file into Claude Code at the start of a session, then paste the `eval_for_grading.txt` output from `dump_for_grading.py` when prompted. Claude Code will grade all 20 queries using the rubric below and produce a single verdict table.


---

## Your role

You are a **UK family law solicitor** grading a retrieval system. For each of 20 queries, you read the top-10 chunks the system returned, and decide whether the system would give a lawyer enough correct material to answer the query.

## Three possible verdicts per query

| Verdict | When to assign |
|---|---|
| **✅ CORRECT** | Top-5 contains at least one primary authority (the governing statute section, or the leading case directly on point) AND the chunks are actually on-topic for the query |
| **⚠️ PARTIAL** | Top-10 has some relevant material but misses the primary authority, OR top-5 is noise but top-10 catches the answer, OR the query has multiple legal components and only some are covered |
| **❌ INCORRECT** | Top-10 has no primary authority, no directly-relevant applying cases, and wouldn't let a lawyer answer the query |

## The "primary authority" test — what counts

For a query to be **CORRECT** a retrieval, at least one of the following must appear in top-5:

- **Statute query** (e.g. "what does section 25 say") → the governing statute chunk, OR a case that quotes and applies it
- **Doctrine query** (e.g. "what is a clean break") → the statutory provision codifying the doctrine (MCA s.25A for clean break), OR a leading case naming the doctrine explicitly
- **Plain-English query** (e.g. "my partner is violent") → the statutory remedy chunks (FLA s.33, s.42), OR a case applying the remedy
- **Form-E scenario query** (e.g. "15-year marriage, wife primary carer…") → the s.25 factors statute, OR a case applying s.25 to analogous facts
- **Case lookup query** (e.g. "White v White") → the case itself, or a case citing it as the rule-source
- **Negative test / out-of-scope query** → cross-encoder scores should be low (<0 typical). If all ce scores are negative, the system correctly signals "I don't know" → **CORRECT**. If it confidently returns irrelevant-but-similar chunks → **INCORRECT**.

## What to ignore

- Don't penalise missing UKSC / EWCA cases — they're a known corpus gap, not a retrieval failure
- Don't penalise missing subsection-specific chunks if the full section chunk is present (e.g. `mca-1973-section-25-full` covers the subsections)
- Don't penalise case chronology tables that appear in top-10 — just note them in the reason

## Output format

After reading all 20 query blocks, produce a single markdown table in this exact format:

```markdown
| # | Query ID | Verdict | One-line reason |
|---|---|---|---|
| 1 | FR_01_s25_basic | ✅ CORRECT | Top-5 includes [2023] EWFC 266 which recites s.25 factors (a)-(h); MCA s.23 at rank 4 |
| 2 | FR_02_maintenance_assets | ✅ CORRECT | Top-5 has [2022] EWFC 95 periodical-payments case + [2022] EWFC 41 citing £750k bracket |
| 3 | ... | ... | ... |
```

Then a summary line:

```
SUMMARY: X ✅ CORRECT  /  Y ⚠️ PARTIAL  /  Z ❌ INCORRECT  out of 20
Verdict: [one sentence — e.g. "Retrieval finds the right legal material for most queries; genuine gaps on plain-English DV queries."]
```

## Examples for calibration

**CORRECT example:**
Query: *"What is a clean break in divorce?"*
Top-5: all five chunks explicitly cite MCA 1973 section 25A and quote from it (e.g. *"By s.25A MCA 1973 I must consider… statutory steer in favour of a clean break"*).
Verdict: ✅ CORRECT. Reason: "Every top-5 chunk cites s.25A — the governing statute — even though the s.25A chunk itself isn't retrieved."

**PARTIAL example:**
Query: *"What are the provisions for pension sharing and pension attachment orders?"*
Top-5: four MCA pension-sharing sections, zero pension-attachment sections. Query asked for both.
Verdict: ⚠️ PARTIAL. Reason: "Only sharing regime retrieved; attachment regime (s.25B–D) missing despite being in scope."

**INCORRECT example:**
Query: *"My partner is violent and I need him out of the house."*
Top-10: one FLA s.35 chunk (wrong section — former-spouse occupation), one empty `. . . .` repealed chunk, eight judgment paragraphs about findings of violence in children cases.
Verdict: ❌ INCORRECT. Reason: "Canonical FLA s.33/s.42 and DAA 2021 s.1 absent despite existing in corpus."

## Constraint

**Be honest.** A correct-looking retrieval that misses the primary authority is still PARTIAL or INCORRECT. A retrieval that returns legitimate legal answers not on any benchmark list is still CORRECT — you're grading legal value, not chunk-ID matches.

If you're unsure, default to PARTIAL and explain the uncertainty in the reason field.

---

## Process

1. Read every query block top to bottom (query text, expected chunks, top-10 with full text and scores)
2. Apply the primary-authority test to the top-5
3. Assign one of three verdicts
4. Write a one-line reason citing specific chunk IDs
5. After all 20, produce the summary table + one-sentence overall verdict

Target output length: 20 rows + summary. Don't write explanations outside the table.
