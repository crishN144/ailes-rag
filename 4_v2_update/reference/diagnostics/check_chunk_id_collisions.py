#!/usr/bin/env python3
"""
Diagnose the 40k duplicate chunk_id finding from the corpus deep-dive.

Two possible root causes, very different implications:

  A. HARMLESS — the text is actually identical between duplicates
     (upstream chunker legitimately produced the same chunk twice,
     or same paragraph exists in near-identical judgments).
     Consequence: query-time dedup is sufficient; no fix needed upstream.

  B. REAL BUG — the chunk_id collides between chunks with DIFFERENT text.
     Consequence: ANY code that looks up a chunk by chunk_id gets the
     wrong result silently. Citation validation, payload filters, any
     future chunk-level operation breaks. Must fix the ID generator.

This script streams the full 310k corpus, groups by chunk_id, and reports
the split between (A) and (B) with examples.

Usage:
    python3 check_chunk_id_collisions.py
"""

import hashlib
from collections import defaultdict
from pathlib import Path

import ijson

CORPUS = Path.home() / "Downloads" / "hpc_outputs" / "merged_all_chunks.json"


def text_hash(t):
    return hashlib.md5((t or "").encode("utf-8")).hexdigest()[:16]


def main():
    print(f"Streaming {CORPUS} (streaming, ~60s)...")
    by_id = defaultdict(list)  # chunk_id -> list of (row_index, text_hash, text_preview)

    n_total = 0
    with open(CORPUS, "rb") as f:
        for i, chunk in enumerate(ijson.items(f, "chunks.item")):
            n_total += 1
            cid = chunk.get("chunk_id")
            text = chunk.get("text", "")
            by_id[cid].append((i, text_hash(text), text[:80]))
            if n_total % 50000 == 0:
                print(f"  scanned {n_total:,}")

    print(f"\nTotal chunks: {n_total:,}")
    print(f"Distinct chunk_ids: {len(by_id):,}")
    dupes = {cid: occs for cid, occs in by_id.items() if len(occs) > 1}
    print(f"chunk_ids with >1 occurrence: {len(dupes):,}")

    if not dupes:
        print("\nNo duplicates found. All chunk_ids are unique.")
        return

    # Classify each duplicate group
    harmless = 0  # all occurrences have identical text
    real_bug = 0  # occurrences have different text under same id
    harmless_examples = []
    bug_examples = []

    total_dupe_rows = 0
    for cid, occs in dupes.items():
        total_dupe_rows += len(occs)
        hashes = {h for _, h, _ in occs}
        if len(hashes) == 1:
            harmless += 1
            if len(harmless_examples) < 3:
                harmless_examples.append((cid, occs))
        else:
            real_bug += 1
            if len(bug_examples) < 5:
                bug_examples.append((cid, occs))

    print(f"\nDuplicate rows total:  {total_dupe_rows:,}")
    print(f"  (chunk_id seen N times contributes N rows)")
    print()
    print("CLASSIFICATION OF DUPLICATE chunk_ids")
    print("=" * 80)
    print(f"  (A) HARMLESS — same id, identical text:   {harmless:,} ids")
    print(f"  (B) REAL BUG — same id, different text:   {real_bug:,} ids")

    if harmless_examples:
        print("\n--- Sample (A) harmless duplicates (safe to leave) ---")
        for cid, occs in harmless_examples:
            print(f"\n  chunk_id: {cid}")
            print(f"  seen {len(occs)} times with identical text")
            print(f"  text preview: {occs[0][2]!r}")

    if bug_examples:
        print("\n" + "=" * 80)
        print("--- Sample (B) REAL COLLISIONS — SAME chunk_id, DIFFERENT text ---")
        print("=" * 80)
        for cid, occs in bug_examples:
            print(f"\n  chunk_id: {cid}")
            print(f"  seen {len(occs)} times with {len({h for _, h, _ in occs})} distinct texts")
            for row_i, h, preview in occs[:4]:
                print(f"    row {row_i:>6}  hash={h}  text: {preview!r}")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if real_bug == 0:
        print("  All 'duplicates' are identical text under the same chunk_id.")
        print("  Upstream chunker is legitimately emitting repeated chunks")
        print("  (same paragraph, multiple judgments). Query-time dedup is")
        print("  sufficient. NO UPSTREAM FIX NEEDED.")
    else:
        print(f"  {real_bug:,} chunk_ids collide on DIFFERENT text.")
        print("  This is a real data-integrity bug in the ID generator.")
        print("  Any chunk-level lookup (payload filter, citation validation,")
        print("  future re-ranking by chunk_id) silently returns wrong results.")
        print()
        print("  MUST FIX before next ingest. Suggested fix: replace current")
        print("  chunk_id generator with deterministic hash of (source_file,")
        print("  paragraph_index) or similar unique composite key.")


if __name__ == "__main__":
    main()
