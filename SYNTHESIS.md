# AILES Fine-tune Retune — SYNTHESIS

Companion docs: [RESEARCH.md](RESEARCH.md) (Track A, local audit), [RESEARCH_TRACK_B.md](RESEARCH_TRACK_B.md) (Track B, online research). This doc is the decision layer — every claim here is grounded in one or both of those.

---

## The one binary decision you need to make to unblock

**Q: Do you ship the retune tomorrow, or do you ship it Monday?**

That binary controls everything else. Write down an answer before reading the recommendation section — the right plan is very different for each.

- **Tomorrow** → Track C below ("minimum-viable retune"). Accepts known technical debt to hit the deadline; most of the data-quality work deferred to v2.1.
- **Monday / 3–4 days** → Track D below ("properly rebuilt retune"). Fixes the root causes instead of papering over them. This is what the evidence actually supports.

Everything in this document flows from that choice.

---

## Top-3 root-cause findings

### 1. Training-serving distribution mismatch (the dominant finding)

**The model has never been trained on the retrieved-context prompt shape that v2 production uses.**

- Track A (RESEARCH.md §A3): zero of 24,674 rows contain any retrieval marker (`[CONTEXT]`, `[RETRIEVED`, `chunk_id`, etc.).
- Track B (§B3): RAFT (Berkeley, 2024) shows +28.9% on HotpotQA vs domain-SFT-alone specifically by training rows in the shape the model sees at inference — oracle passage + 4 distractors + CoT answer with inline `[chunk_id]` citations. Stanford RegLab (2024) documents ≥17% hallucination rate even in RAG-backed commercial legal tools, explicitly because they were not trained on grounded generation.
- Inference layer: `prompts (9).yaml` has no `rag.grounded_answer` prompt block. Nobody has specified the retrieval-context shape on the serving side either.

This is not a prompt bug, a hyperparam bug, or a model-choice bug. The entire training pipeline treated AILES as a closed-book domain expert; the entire serving pipeline wants an open-book retrieval-grounded reasoner. These are different models.

### 2. Correction oversampling is drilling, not teaching

**47 unique correction texts were duplicated 10×–150× each; the four most-drilled rows (s.25(2)(f) homemaker corrections) were seen by the model ≈1,260 times across 8.4 epochs.**

- Track A (§A3): "369 × 5" claim in `FINAL_ACCURATE_TRAINING_DATA.md` is factually wrong. Real duplication distribution (by full-text hash): 33 texts × 10, 4 × 50, 4 × 150, 3 × 105, 2 × 135, 1 × 130.
- Track B (§B9): LIMA / LIMIT / Raschka all agree that heavy duplication of a small correction set is "known-bad" — targeted overfitting that memorises surface wording rather than teaching the correction principle. Published recipes cap per-row duplication at 3–5×.
- Failure mode implication: the model has verbatim recall of ~20 specific corrections and will cite them confidently even when retrieved context contradicts them. This directly fights `citation_validator.py`.

### 3. System-prompt inconsistency across training + prod

**The model was trained on three AILES identities; production uses a fourth variant it has never seen.**

- Track A (§A3): four distinct system prompts identified — 17,829 rows use variant A, 1,845 rows (the corrections) use variant B, 3,000 UltraChat rows use variant C, prod `base.system_message` is variant D.
- Track B (§B8): catastrophic-forgetting literature shows that replay data only protects general capability when it matches the production template. Mismatch dilutes the replay benefit and is a plausible contributor to the "hi" failure mode.
- Track B (§B2): Unsloth docs and Tülu 3 both emphasise that the model's exact chat template must be used unchanged across training and inference.

---

## Go / no-go on reusing `phase1_v2.jsonl`

**No-go for direct reuse. Partial-go for reusing the underlying Q&A content after a structured rebuild.**

Reasoning:

| Component of phase1_v2.jsonl | Keep? | Reason |
|---|---|---|
| 16,639 judgment Q&A pairs | **Transform, don't copy.** Restructure into RAFT rows with oracle chunks from Qdrant. | The Q&A content is reasonable (per Track A §A1); what's wrong is the prompt shape it's wrapped in. |
| 1,190 statute Q&A | **Same — transform.** Plus add verbatim-quote supervision for statute sections. | Highest-stakes content; needs strict faithfulness discipline. |
| 1,845 "correction" rows | **Keep the 47 unique texts. Paraphrase each into 3–5 variants. Drop duplication.** | Track B §B9 is explicit: cap duplication at 3–5× per unique text. |
| 3,000 UltraChat | **Keep a re-templated subset (1–2k) under the production system prompt.** | Replay for general chat capability. Re-templating is mandatory. |
| 2,000 OpenOrca | **Drop or reduce to ~500.** | Instruct base already handles generic instruction-following from RLHF. 2k is unnecessary. |

**Critical precondition for any of the above:** the XML→Q&A fidelity audit I flagged in Track A §A3.5 (did Mistral fabricate facts when generating the Q&A pairs?) has not been done, because the source XMLs are on HPC and I don't have them. Without that audit, we are rebuilding on potentially corrupted ground truth. This is a real risk — if Mistral got 5% of the answers wrong at Q&A generation time, you'll fine-tune on ~1,200 fabrications and the same confident-but-wrong pattern will carry over to the Instruct retune.

**Recommendation on the fidelity audit:** run a 100-row sample fidelity check on HPC before any retrain starts. 4 hours of scripted work. If hit rate is >95%, proceed. If <90%, the dataset is compromised and needs regeneration before retraining. I have no way to predict which it will be.

---

## Recipe recommendation

Two tracks; pick based on the binary above.

### Track C — Minimum-viable retune (tomorrow ship)

**What you change:**
- Base model: `meta-llama/Llama-3.1-8B-Instruct` (not Base). *Evidence: Track B §B2.*
- LoRA config unchanged: r=64, α=128, all 7 modules, QLoRA nf4, bf16 compute, dropout 0.05. *Evidence: Track B §B1 — current config is within 2025 norms.*
- Learning rate: **1e-4** (down from 2e-4). *Evidence: Track B §B1 + §B2 — lower LR is the safer default for Instruct bases; reduces forgetting.*
- Epochs: **1**. Single pass, no phase 2 resume. *Evidence: Track B §B9 — multi-epoch past 3 on <25k rows overfits.*
- Max steps: ~1,500 at eff-batch 16, seq_len 3072. *Arithmetic below.*
- **Data normalisation before training:** re-template every row to use the single production system prompt from `prompts (9).yaml` `base.system_message`. *Evidence: Track A §A3 + Track B §B8.*
- **Correction deduplication:** collapse the 1,845 duplicated correction rows to 47 unique texts, each repeated exactly 3× (141 total rows). Net dataset size after dedup: 22,876 rows. *Evidence: Track B §B9.*
- Data content otherwise unchanged from phase1_v2.jsonl.

**What you do NOT change for Track C:**
- Data shape stays single-turn Q&A from memory. No RAFT. No retrieved context in rows. No distractor training. No abstention rows.

**Expected outcome:**
- ✅ Fixes conversational hallucination on "hi" / "hello" (Instruct base has RLHF turn-boundary training).
- ✅ Fixes the Cyrillic/token-instability bleed from over-training (1 epoch vs 8.4).
- ✅ Partial fix for identity drift (one consistent system prompt).
- ❌ Does NOT fix RAG grounding. Model will still ignore retrieved context and answer from memory. `citation_validator.py` will flag hallucinations frequently.
- ❌ Does NOT fix out-of-jurisdiction refusal.
- ❌ No eval suite means you cannot measure whether it actually improved.

**Track C is a regression fix, not an architecture fix.** It ships tomorrow because it leaves the root cause (finding #1) untouched. Plan to iterate within 2 weeks.

### Track D — Properly rebuilt retune (Monday ship)

**Step 1 — audit (day 0, parallel):**
- XML→Q&A fidelity audit on 150 rows (§A3.5 above). 4 hours.
- Build ~200-question UK family-law eval suite: 100 verifiable-fact queries + 50 abstention queries + 50 citation-validity queries. *Evidence: Track B §B5.* 1 day.

**Step 2 — rebuild data (day 1):**
- Start from the 47 unique correction texts + 16,639 judgment Q&A + 1,190 statute Q&A = ~17,876 source items.
- For each source item, construct a RAFT training row:
  - **Oracle:** the Qdrant chunk that the original Q&A answer actually draws from (you have chunk_id → text in Qdrant).
  - **Distractors:** top-4 hard negatives from Qdrant BM25+BGE for the same query (not random — must be topically related but factually unhelpful).
  - **Answer:** rewrite the original answer to include inline `[chunk_id]` citations and verbatim quotes from the oracle wrapped in `##begin_quote##…##end_quote##` (the RAFT convention).
  - **P%:** 70% of rows include the oracle; 20% of rows contain only distractors (gold answer is abstention); 10% are pure general-chat replay (re-templated UltraChat subset). *Evidence: Track B §B3 + §B7.*
- Normalise every row to the exact production Llama 3.1 chat template. Apply loss-masking: labels = -100 for everything before the assistant turn. *Evidence: Track B §B2.*
- Cap correction duplication at 3× per unique text. Paraphrase into 5 variants each. *Evidence: Track B §B9.*
- Target dataset size: ~18,000–20,000 high-quality rows (LIMA heuristic — quality beats quantity).
- Cap total row length at 6,000 tokens to stay within L4 inference envelope. *Evidence: Track B §B10.*

**Step 3 — train (day 2):**
- Base: Llama 3.1 8B Instruct.
- LoRA: r=64 α=128 all 7 modules QLoRA nf4 bf16 dropout 0.05.
- Optimiser: paged_adamw_8bit, LR 1e-4 cosine, warmup 10 steps, weight_decay 0.01, max_grad_norm 1.0.
- Epochs: 1. Max steps ~1,200 at eff-batch 16, seq_len 6144.
- Gradient checkpointing on.
- **Explicit:** `tokenizer.apply_chat_template()` at training time AND inference time; no manual template string building.

**Step 4 — eval (day 3):**
- Run the 200-question eval suite. Report hallucination rate, citation validity rate, abstention precision/recall, length distribution.
- Run `run_eval.py` (Claude-as-judge) on a 20-query sample.
- If eval regresses vs the current model on substantive legal reasoning (> 2pp drop on citation accuracy), investigate before shipping.

**Expected outcome:**
- ✅ Fixes all three top-3 findings.
- ✅ Grounds generation in retrieved context (first time AILES has been trained to do this).
- ✅ Teaches abstention (first time).
- ✅ Measurable against an explicit eval suite.
- ❌ Does not include continued pretraining on BAILII/TNA (SaulLM pattern) — defer to iteration 3.
- ❌ Still 8B base — not going to match GPT-4-class reasoning.

---

## Time + compute arithmetic

Single L40S (48 GB), Llama 3.1 8B Instruct, QLoRA nf4 r=64 all 7 modules:

### Track C
| Phase | Rows | Steps | Wall time |
|---|---|---|---|
| Data dedup + re-template | 22,876 out → 22,876 in | — | 1 h (scripted) |
| Training (1 epoch, eff-bs 16, seq_len 3072) | 22,876 / 16 = 1,430 | ~1,430 steps | **6–10 h** |
| Merge LoRA into base | — | — | 30 min |
| Upload to GCS + Aslam swap | — | — | 30 min |
| **Total** | | | **8–12 h** |

Tight but feasible for tomorrow morning ship if kicked off tonight. Accept the root-cause gap.

### Track D
| Phase | Work | Wall time |
|---|---|---|
| XML fidelity audit (HPC) | 150-row sample check | 4 h |
| Build eval suite (200 items) | Hand-written UK family-law questions + ground truth | 1 day |
| Build RAFT training data | 17,876 source items → 18k–20k RAFT rows with oracle+distractors from Qdrant | 1 day |
| Train (1 epoch, eff-bs 16, seq_len 6144) | ~1,200 steps | **8–12 h** (longer seq → slower) |
| Run eval suite | 200 queries × 2 runs (before/after) | 2 h |
| Merge + upload + swap | | 1 h |
| **Total** | | **3–4 days** |

Lines up with a Monday ship if you start tomorrow morning.

---

## Go/no-go question that decides the plan

Answer this out loud before committing:

> **"Is it more important that AILES has a working conversational model in production by tomorrow morning, or that the retune actually fixes the RAG-grounding root cause?"**

- If conversational-fix-in-prod-tomorrow is more important: **Track C.** Accept that `citation_validator.py` will reject many outputs for the first 2 weeks; iterate.
- If root-cause-fix is more important and you can negotiate 2–3 extra days: **Track D.** This is what the evidence actually justifies.

There is no middle path worth recommending. Doing Track C and "adding RAG data later" is real — but the published RAFT evidence shows that an additional small phase after SFT-only is weaker than doing RAFT from the start, because the SFT-only phase teaches the model "ignore the context and answer from memory" and that prior has to be unlearned.

---

## What I need from you to execute Track D

If you pick Track D, the critical-path inputs are:

1. **HPC access** to `/users/bgxp240/ailes_legal_ai/data/` for the fidelity audit + to run the retrain. (You already have this.)
2. **Qdrant access from your laptop or HPC** — needed to pull oracle chunks + generate distractors during RAFT data construction. Adi set this up with you locally; same path.
3. **Agreement on the production prompt shape for RAG context** — this is a design decision that touches both `prompts (9).yaml` and the backend. Suggest: `user = "[CONTEXT]\n{chunks_with_ids}\n[/CONTEXT]\n\n{query}"`. If different, the training data has to match. Decide this before generating the RAFT rows.
4. **Confirmation that building a 200-question eval suite is acceptable scope** — someone has to write the questions + ground truth. You're probably that someone. Budget 1 day.

If you pick Track C, the only input needed is: "go" — I can start the data re-template + dedup script right now.

---

## Unresolved evidence gaps (what Track B flagged as weak)

- Exact % of abstention rows for legal specifically — Track B uses RAFT's 20% default; legal-specific calibration not published.
- Expected citation-lift magnitude at 8B for UK family law specifically — the +28.9% HotpotQA figure is the strongest published data point, but legal-specific lift numbers at 8B are weak. Treat as hypothesis, not commitment.
- L4 long-context ceiling under real AILES traffic — 8k tokens is the rule of thumb; untested against real concurrent load.
- **The XML→Q&A fidelity question** — can only be answered on HPC. If Mistral hallucinated, both tracks C and D build on a shaky foundation and need regeneration first.
