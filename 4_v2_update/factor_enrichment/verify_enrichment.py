"""Verify the prep output: re-run rank test using new pkl + new dense vectors."""
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE = Path("/Users/crishnagarkar/Downloads/hpc_outputs")
OUT = Path("/tmp/factor_enrichment_out")

# Load OLD setup
with open(BASE / "merged_all_chunks.json") as f:
    chunks = json.load(f)["chunks"]
with open(BASE / "bm25_index/chunk_ids.pkl", "rb") as f:
    chunk_ids = pickle.load(f)
with open(BASE / "bm25_index/bm25_model.pkl", "rb") as f:
    bm25_old = pickle.load(f)
emb_old = np.load(BASE / "embeddings/bge_embeddings.npy", mmap_mode="r")

# Load NEW setup
with open(OUT / "bm25_model.pkl", "rb") as f:
    bm25_new = pickle.load(f)
factor_chunk_ids = json.loads((OUT / "factor_chunk_ids.json").read_text())
new_dense = np.load(OUT / "factor_dense_vectors.npy")

cid_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
factor_chunks = [c for c in chunks if c.get("is_factor")]

# Realistic user queries (NOT containing factor_name verbatim)
QUERIES = {
    "ca-1989-section-1(3)(a)": ["what does the child want?"],
    "ca-1989-section-1(3)(b)": ["what does my child need physically and emotionally?"],
    "ca-1989-section-1(3)(c)": ["what happens if we change the child's circumstances?"],
    "ca-1989-section-1(3)(d)": ["does the child's age matter?"],
    "ca-1989-section-1(3)(e)": ["has my child suffered any harm?"],
    "ca-1989-section-1(3)(f)": ["am I capable of meeting my child's needs?"],
    "ca-1989-section-1(3)(g)": ["what powers does the court have?"],
    "i(pfd)a-1975-section-3(2)(a)": ["how long was the marriage in inheritance claim?"],
    "i(pfd)a-1975-section-3(2)(b)": ["what was my contribution to the family?"],
    "mca-1973-section-25(2)(a)": ["What financial resources will the court consider?"],
    "mca-1973-section-25(2)(b)": ["What about financial needs of each party?"],
    "mca-1973-section-25(2)(c)": ["Standard of living before breakdown?"],
    "mca-1973-section-25(2)(d)": ["How long was our marriage?"],
    "mca-1973-section-25(2)(e)": ["What if one of us has a disability?"],
    "mca-1973-section-25(2)(f)": ["Contribution as homemaker raising children?"],
    "mca-1973-section-25(2)(g)": ["What if my partner behaved badly?"],
    "mca-1973-section-25(2)(h)": ["Pension benefits lost from divorce?"],
}

def rank(scores, t):
    return int((scores > scores[t]).sum()) + 1


def rrf(d, b, t, k=60, pf=100):
    d_top = np.argpartition(-d, pf)[:pf]
    b_top = np.argpartition(-b, pf)[:pf]
    d_o = d_top[np.argsort(-d[d_top])]
    b_o = b_top[np.argsort(-b[b_top])]
    dr = {i: r+1 for r,i in enumerate(d_o)}
    br = {i: r+1 for r,i in enumerate(b_o)}
    u = set(dr) | set(br)
    s = {i: 1/(k+dr.get(i,pf+1)) + 1/(k+br.get(i,pf+1)) for i in u}
    f = sorted(u, key=lambda x: -s[x])
    return f.index(t)+1 if t in f else pf+1


print("Loading BGE...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Patch in new dense vectors as a dict (won't materialize full new matrix)
new_emb = {cid_to_idx[cid]: new_dense[i] for i, cid in enumerate(factor_chunk_ids)}

def dense_scores_new(q_emb):
    s = emb_old @ q_emb
    for tidx, e in new_emb.items():
        s[tidx] = float(np.dot(e, q_emb))
    return s


print(f"\n{'chunk':<35s} {'query':<55s} {'BM25':>14s}  {'Dense':>14s}  {'RRF':>14s}")
print("-"*140)

baseline_rrf, enriched_rrf = [], []
for cid, queries in QUERIES.items():
    if cid not in cid_to_idx:
        continue
    t = cid_to_idx[cid]
    for q in queries:
        # Old
        b_old = bm25_old.get_scores(q.lower().split())
        q_emb = model.encode([q], normalize_embeddings=True)[0]
        d_old = emb_old @ q_emb
        rb_old = rank(b_old, t); rd_old = rank(d_old, t); rr_old = rrf(d_old, b_old, t)

        # New
        b_new = bm25_new.get_scores(q.lower().split())
        d_new = dense_scores_new(q_emb)
        rb_new = rank(b_new, t); rd_new = rank(d_new, t); rr_new = rrf(d_new, b_new, t)

        baseline_rrf.append(rr_old); enriched_rrf.append(rr_new)
        print(f"{cid:<35s} {q[:55]:<55s} {rb_old:>5d}->{rb_new:<5d}  {rd_old:>5d}->{rd_new:<5d}  {rr_old:>5d}->{rr_new:<5d}")

print("\n--- SUMMARY ---")
print(f"RRF median: {int(np.median(baseline_rrf))} -> {int(np.median(enriched_rrf))}")
print(f"In top-15:  {sum(1 for r in baseline_rrf if r<=15)}/{len(baseline_rrf)} -> {sum(1 for r in enriched_rrf if r<=15)}/{len(enriched_rrf)}")
print(f"In top-30:  {sum(1 for r in baseline_rrf if r<=30)}/{len(baseline_rrf)} -> {sum(1 for r in enriched_rrf if r<=30)}/{len(enriched_rrf)}")
