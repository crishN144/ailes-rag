#!/usr/bin/env python3
"""
One-command evaluation runner.

Runs the Step-4 retrieval pipeline on all golden queries and produces
a single file `/tmp/ailes_eval_output.txt` that contains EVERYTHING Claude
Code needs to grade: the rubric + instructions + per-query top-10 output.

The person running it just:

  1. Run this script   (~5 minutes — loads BGE + cross-encoder + Qdrant)
  2. Open Claude Code
  3. Paste ONE file: /tmp/ailes_eval_output.txt
  4. Claude Code returns a 20-row table: ✅ / ⚠️ / ❌ per query

No second paste. No separate rubric file. Everything is inlined.

Queries sourced from:
  3_tests_and_results/golden_queries.json (20 queries, the original benchmark)
  — same file the Claude Code grading in tonight's eval used.
  — To use a different benchmark, edit GOLDEN_FILE in dump_for_grading.py.

Usage:
    python3 run_eval.py
"""

from pathlib import Path
import sys
import subprocess

HERE = Path(__file__).parent
DUMP_SCRIPT = HERE / "dump_for_grading.py"
RUBRIC = HERE / "GRADING_RUBRIC.md"
GOLDEN = HERE / "golden_queries.json"
DUMP_OUTPUT = Path("/tmp/eval_for_grading.txt")
FINAL_OUTPUT = Path("/tmp/ailes_eval_output.txt")

HEADER = """\
================================================================================
AILES RAG RETRIEVAL — CLAUDE CODE EVALUATION PACKAGE
================================================================================

This is a SINGLE-PASTE evaluation.

  1. Open a Claude Code session.
  2. Paste this ENTIRE file as a single message.
  3. Claude Code will apply the grading rubric below to the 20 query blocks
     and return a single verdict table with ✅ CORRECT / ⚠️ PARTIAL / ❌ INCORRECT
     per query, plus a one-line summary.

The table answers ONE question per query in plain English:
  "Would a UK family law solicitor consider this retrieval correct?"

No recall@5. No MRR. No coverage scores. No API keys. Just: is each query's
top-10 good enough for a lawyer to answer the query?

If the summary says e.g. "16 ✅ / 2 ⚠️ / 2 ❌" — production-acceptable.
If it says "6 ✅ / 14 ❌" — something is broken and the `reason` field per
row tells you where.

================================================================================
"""


def main():
    # Validate inputs exist
    if not RUBRIC.exists():
        print(f"ERROR: rubric not found at {RUBRIC}")
        sys.exit(1)
    if not GOLDEN.exists():
        print(f"ERROR: golden queries not found at {GOLDEN}")
        sys.exit(1)

    print(f"Sources")
    print(f"  golden queries : {GOLDEN}")
    print(f"  grading rubric : {RUBRIC}")
    print()
    print(f"Step 1: running Step-4 retrieval pipeline on all 20 queries...")
    print(f"         (loads BGE + cross-encoder + BM25 + queries Qdrant — ~5 min)")
    print()

    # Re-run the dump so the raw retrieval output is fresh
    result = subprocess.run(
        [sys.executable, str(DUMP_SCRIPT)],
        cwd=str(HERE.parent),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("ERROR: dump_for_grading.py failed")
        print(result.stderr[-2000:])
        sys.exit(1)
    if not DUMP_OUTPUT.exists():
        print(f"ERROR: expected dump at {DUMP_OUTPUT} but file missing")
        sys.exit(1)

    dump_text = DUMP_OUTPUT.read_text()
    rubric_text = RUBRIC.read_text()

    # Assemble the single-paste bundle
    composite = (
        HEADER
        + "\n"
        + "=" * 80 + "\n"
        + "PART 1 — GRADING RUBRIC (apply this to the query blocks in Part 2)\n"
        + "=" * 80 + "\n\n"
        + rubric_text
        + "\n\n"
        + "=" * 80 + "\n"
        + "PART 2 — RETRIEVAL OUTPUT FOR 20 QUERIES (apply the rubric to each)\n"
        + "=" * 80 + "\n\n"
        + dump_text
    )

    FINAL_OUTPUT.write_text(composite)

    size_kb = FINAL_OUTPUT.stat().st_size // 1024
    print(f"\n✅ Done.")
    print(f"\nOutput file: {FINAL_OUTPUT}  ({size_kb} KB)")
    print()
    print("=" * 72)
    print("NEXT STEPS (single paste):")
    print("=" * 72)
    print(f"  1. Open Claude Code")
    print(f"  2. Paste: {FINAL_OUTPUT}")
    print(f"  3. Read the 20-row verdict table Claude Code returns")
    print()
    print("No second file to paste. Rubric is included in the bundle.")


if __name__ == "__main__":
    main()
