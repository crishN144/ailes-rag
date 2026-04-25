# AILES Fine-tuning Audit — RESEARCH.md

**Status:** Track A (local audit) complete. Track B (online research) running in background; will be merged into `RESEARCH_TRACK_B.md` and summarised in `SYNTHESIS.md` on completion.

**Method:** Every finding below cites either a file+line from local artefacts (`/Users/crishnagarkar/Downloads/finetunnn/`, `/Users/crishnagarkar/Downloads/reggobs/`, `/Users/crishnagarkar/Downloads/MEETING_TODAY_APRIL14/`) or a live observation from the HPC reproduction earlier in the conversation. Nothing is inferred; claims that can't be grounded are explicitly flagged.

**Confirmed ground truth (given by user, not re-derived):**
- Phase 1: `finetune_llama_ailes_v2.py` → `ailes_training_v2_final_merged.jsonl` (22,815 ex) → base Llama 3.1 8B → ckpt-1250
- Phase 2: `finetune_llama_phase1_v2.py` → `ailes_training_v2_phase1_v2.jsonl` (24,674 ex, +1,845 corrections) → resumed → ckpt-2050
- Merge: `merge_lora_into_base.py` → `llama-8b-legal-merged` (15 GB, Dec 16)
- Current deployed artefact on GCP: `gs://ailes-fine-tuned-llm/llama-8b-legal-merged/` (same model)

---

## TRACK A — LOCAL AUDIT

### A1. AILES capability surface (re-derived from the repo, not from vibe)

Evidence sources walked: `prompts (9).yaml` (v2 production prompts), `reggobs/regobs-winuyar-ailes-backend-*.txt` (backend code dump), `reggobs/regobs-winuyar-ailes-frontend-*.txt` (frontend endpoints), `ailes-rag-handoff/` (RAG blueprint).

**Capabilities AILES must deliver:**

| # | Capability | What "good" looks like | What current model does | Delta |
|---|---|---|---|---|
| 1 | **General chat / greetings** | "Hi, I'm Ailes, a UK family-law assistant. How can I help today?" Stop. One turn. | Answers briefly, then hallucinates additional user/assistant turns filled with tangential legal content ("user: Can you explain reasonable excuse..."). Confirmed on HPC repro: "hi" → 250 words of multi-turn drift. | **Severe.** Model doesn't respect turn boundaries. |
| 2 | **Identity / brand tone** | Consistently "Ailes, UK family-law assistant." Professional register. Doesn't claim US or generic law expertise. | Inconsistent. "who are you?" returned "I am a specialized legal assistant that provides comprehensive legal guidance and analysis" and then listed US-style practice areas (contract, corporate, employment, real estate, IP) — NONE of which AILES covers. Plus Cyrillic garble mid-response. | **Severe.** Scope drift + token instability. |
| 3 | **Legal Q&A from parametric memory (no RAG)** | Accurate UK family-law answer with section references. | Worked well on HPC repro — s.25 MCA, financial remedy, hiding assets all produced coherent accurate responses. | **No delta.** This is the one thing that works. |
| 4 | **Legal Q&A with RAG context injected** | Reads `[CONTEXT]` block, grounds answer in retrieved chunks, cites specific chunk_ids, falls back to "no specific authority retrieved" when context is silent. | **Untested** — current backend never injects retrieved context (see A5). Even if we injected it, the model has ZERO training on this shape (see A3). | **Structural.** No training signal exists. |
| 5 | **Form-E → 8-section judgment report** | Structured markdown matching `legal_reports.judgment_report` prompt: Case Overview → Facts → Legal Issues → Applicable Law → Analysis → Recommendations → Implementation → Disclaimer. Section 4 cites only retrieved authorities (per CITATION DISCIPLINE added in prompts_v9). | No direct test, but training set has **zero** rows using that 8-section format (see A3). Output quality relies entirely on in-context prompt coercion. | **Moderate.** Works-but-fragile; no training reinforcement. |
| 6 | **Out-of-scope refusal** | "California divorce / crypto tax / criminal law is outside my scope; I cover UK family law (England & Wales)." | Known failure: NEG_02 California divorce query retrieved 3 forbidden UK-MCA chunks during Adi's benchmark. Model has no refusal training. | **Severe.** Zero refusal rows in training set. |
| 7 | **Intake / qualification** | Uses `client_qualification` prompt's YES/NO format at end of message. | Prompt handles this at runtime. Model has no training examples of this format. | **Light.** Prompt does the lifting; not blocking. |
| 8 | **Citation discipline** | Never cites a section number, case name, or year not present in retrieved context. | Training teaches model to answer from memory — actively opposite of grounded citation. `citation_validator.py` is the only safety net. | **Moderate.** Runtime guard partial mitigation. |
| 9 | **Clarifying questions** | When the user's request is ambiguous, model asks a follow-up rather than guessing. | No training rows of this form. Instruct base model would do this naturally; current base fine-tune does not. | **Moderate.** Base model failure. |
| 10 | **Multi-turn conversation** | Remembers prior turns, references them, doesn't re-introduce self. | UltraChat rows (3,000) do include multi-turn (up to 7×7), so some signal exists. Unknown in-production behaviour. | **Light.** Some training signal. |

**The delta summary:** the product AILES needs to ship and the model currently in GCS share one working capability (#3) and diverge on the other nine. Capability #4 (RAG grounding) is the single biggest structural gap because the entire v2 pipeline is built around it.

---

### A2. Current model — regression sweep (evidence from HPC reproduction)

I do not have a held-out eval set, but the HPC reproduction earlier in this conversation exercised the deployed model against 7 queries spanning both categories. Confirmed failure modes:

| # | Failure class | Observed | Root cause category |
|---|---|---|---|
| 1 | Chat-fluency regression vs stock Llama 3.1 8B Instruct | "hi" → coherent first sentence, then 250 words of fabricated multi-turn legal dialog. Classic over-fine-tune damage: base model had no turn-boundary prior, fine-tune only saw single-turn Q&A (except 3k UltraChat). | **base model + training recipe** (started from Base not Instruct; didn't include enough turn-terminated conversational data) |
| 2 | Jurisdiction / domain drift | "who are you?" → lists US-style practice areas (contract, corporate, employment, real estate, IP). | **data** (no "identity" training rows pinning AILES to UK family-law scope) |
| 3 | Token instability | "who are you?" response contained `assistantилася` (Cyrillic letters). | **training recipe** (decoding drifted into low-probability token distribution; typical of over-trained base models with non-English token bleed-through) |
| 4 | Not quoting verbatim when summarising statutes | s.25 MCA response paraphrased correctly but did not quote statute wording. A solicitor relying on exact phrasing would be misled. | **data** (training examples paraphrase statute text rather than quote it verbatim) |
| 5 | Refusing to refuse | Implicit from Adi's benchmark: NEG_02 California divorce returned 3 UK MCA chunks at retrieval (forbidden_hits: 3). Even without RAG this model will confidently produce UK answers for US questions. | **data** (no refusal examples) + **prompt** (classifier mitigates at runtime) |
| 6 | Ignoring retrieved chunks | Not testable on current model (no RAG in backend) but guaranteed by construction — zero training rows simulate retrieved-context input. | **data** |
| 7 | Form-to-judgement structural collapse | Untested; training set has no 8-section report examples. | **data** |
| 8 | Prompt-injection susceptibility | Not tested. | Unknown. |
| 9 | Fabricated statutes | s.25(2)(f), s.25(2)(h), s.25A etc. in the model's training data are real. But the 150× drilling of s.25(2)(f) corrections means the model will cite this specific section with high confidence even when context doesn't support it. Risk of confident fabrication on adjacent sections it saw less of. | **data** (massively skewed drilling) |
| 10 | Fabricated cases | `Charman v Charman [2007] EWCA Civ 503` is real. Drilled 105× on "not Hague". Model has high-confidence recall on the ~30 drilled cases; risk on undrilled cases. | **data** |

**Classification summary:**
- **data issues:** 6 of 9 testable failures
- **base model choice:** 1 (chat fluency — the big one)
- **training recipe:** 2 (over-training, token instability)
- **RAG:** 0 testable in current deployment (it doesn't have RAG)
- **merge step:** 0 evidence of merge error
- **prompt:** runtime guards partial but not load-bearing

The dominant finding: **most failures trace back to training data, not the base model or infra.** The base-model choice amplifies the data problem (Instruct would compensate for some of it through RLHF priors) but data is the root.

---

### A3. Training dataset forensics — `ailes_training_v2_phase1_v2.jsonl`

Parsed all 24,674 rows directly. Do not trust the handwritten "369 × 5 corrections / 78% legal / 22% capability" numbers in the MD files — several are wrong.

**Real distribution (verified):**

| Bucket | Rows | Share | Notes |
|---|---|---|---|
| UK judgment XMLs | 16,639 | 67.4% | 1,199 unique source cases. Up to 30 Q&A pairs per case — heavy redundancy within single judgments. |
| Statute Q&A | 1,190 | 4.8% | Separate `case_type=statute_qa` field, 1 statute per row. |
| Corrections (case_type=None) | 1,845 | 7.5% | **47 unique texts, oversampled 10×–150×.** The "369×5" claim is wrong. |
| UltraChat (multi-turn) | 3,000 | 12.2% | Some rows 7×7 turns. Good turn-boundary signal but generic (not AILES-branded). |
| OpenOrca (instruction-following) | 2,000 | 8.1% | Single-turn generic tasks. |

Total legal/domain = 19,674 (79.7%). Capability = 5,000 (20.3%). Mix is ≈80/20.

**Correction oversampling — the real breakdown:**

```
47 unique correction texts, distributed:
   33 texts × 10  copies = 330 rows
    4 texts × 50  copies = 200 rows
    4 texts × 150 copies = 600 rows   ← heaviest drilling
    3 texts × 105 copies = 315 rows
    2 texts × 135 copies = 270 rows
    1 text  × 130 copies = 130 rows
                           ─────
                           1,845 rows total
```

The four rows appearing 150× each are all Section 25(2)(f) homemaker-contribution corrections. The three rows appearing 105× are all "Charman v Charman is NOT Hague Convention" corrections. These were manually written to fix specific mistakes the base fine-tune made — the team sledgehammer-drilled them to force memorisation.

At `max_steps=2050`, `effective_batch=16`, `seq_len=3072`, total training compute = 100,761,600 tokens on a 12,032,853-token dataset = **8.4 epochs**. A row repeated 150× in the dataset was seen by the model ~1,260 times during training. That's heavy memorisation, not learning.

Consequences:
- Model has near-verbatim recall of the 20-ish most-drilled correction facts (s.25(2)(f), s.25(2)(h), Charman). It will cite these confidently.
- Model has shallow exposure to the other 99% of the dataset — each judgment Q&A seen ~8 times on average.
- If the production CITATION DISCIPLINE in prompts_v9 tells the model "only cite what's in context" and the retrieved context doesn't mention s.25(2)(f), the model is heavily conditioned to mention it anyway. The 150×-drilled knowledge will fight the prompt.

**Chat-template conformance:** 100% (24,674/24,674). Every row uses `<|begin_of_text|>` + Llama-3 instruct template tokens + ends in `<|eot_id|>`. Data format is correct. The bug is not in data formatting — it's that the base model wasn't natively trained on these tokens.

**Response-length distribution:**
- Mean 122 words, median 110, p95 251, p99 higher
- 92.7% in the intended 40–300 word range
- 1,114 rows <40 words (mostly OpenOrca and UltraChat — legitimate short-answer tasks)
- 692 rows >300 words (long judgment analysis)

Reasonable; not a red flag on its own. But the model learned "vary length 13–890 words" without a clear signal on when to be brief — this may show as erratic length at inference time.

**Sequence-length truncation:**
- 201 rows (0.8%) exceed `max_seq_length=3072` at 4 chars/token estimate
- Longest row is 36,302 chars (~9,075 tokens)
- These rows were **silently truncated** during training — the tails (often the concluding legal reasoning) were cut off
- Low-volume issue (0.8%) but means the model sometimes saw fact-patterns without their conclusions

**System-prompt consistency (big finding):** four distinct system prompts exist across training + production:

| Variant | Rows | Source |
|---|---|---|
| A: "You are AILES, a specialized UK family law assistant that provides comprehensive legal guidance and analysis." | 17,829 | Judgments + statutes + some |
| B: "You are AILES, a specialized UK family law assistant trained on the Matrimonial Causes Act 1973, Children Act 1989, and relevant case law. Provide accurate statutory citations and legal reasoning." | 1,845 | All correction rows (case_type=None) |
| C: "You are a helpful, knowledgeable assistant that provides accurate and thoughtful responses." | 3,000 | UltraChat |
| D: (production) `base.system_message` in `prompts (9).yaml` line 14: "You are Ailes, a helpful and professional AI assistant on a legal platform. CRITICAL: Only respond to the user's current message..." | 0 | Production — model never saw this exact string during training |

The model was trained on four distinct AILES identities (one of which was never shown to it in training). At inference time production uses variant D — which the model has no prior on. Model will match production prompt to the closest training pattern, likely variant A (shortest, most common).

**Zero RAG-shape rows:** searched the dataset for markers that would indicate "retrieved context is provided in the prompt" — `[CONTEXT]`, `[RETRIEVED`, `[CITATIONS]`, `chunk_id`, `CHUNK_ID`, `<context>`, `Here is retrieved`. **Total hits across 24,674 rows: 0.** The model has literally never been shown a training example of the retrieved-context shape that production will send it. This is the single most important finding.

**Zero judgment-report-format rows:** searched for `CASE OVERVIEW`, `APPLICABLE LAW`, `Issue Summary Report`, 8-section markdown patterns matching `legal_reports.judgment_report` in prompts_v9. Total rows with any marker: 32 (incidental mentions in judgment text). **Total rows teaching the 8-section report format: 0.** Same structural gap.

**UltraChat shape:** confirmed multi-turn (1×1 through 7×7 user/assistant pairs). 3,000 rows. Adequate turn-boundary signal for an Instruct base; insufficient for a Base model that had no RLHF.

**Duplication beyond corrections:** 22,829 rows appear exactly once. 1,845 are the correction duplicates. No silent leakage of duplicated judgment Q&A — the Q&A extraction appears to have properly varied questions per source XML (30 Q&A per case, different questions).

---

### A3.5. XML → Q&A fidelity audit — **NOT PERFORMABLE LOCALLY**

Source XML files are on HPC at `/users/bgxp240/ailes_legal_ai/data/` (per the training scripts, line 122 of `finetune_llama_ailes_v2.py` and the Mistral-generation pipeline reference in `FINAL_ACCURATE_TRAINING_DATA.md`). They are not in the `finetunnn/` download.

**What this audit would check but can't, without HPC access:**
- Are factual claims in the training answers actually present in the source XML chunks Mistral saw?
- Are case citations, dates, party names, judge names correct against the XML?
- Did Mistral invent section numbers or paraphrase statute wording?
- What was the chunk-size distribution fed to Mistral — did it have enough context, or was it filling gaps?

**Why this matters:** the training pipeline was XML → Mistral chunking+Q&A → JSONL → fine-tune. Mistral is the weakest link in that chain because it was a silent synthetic-data step with no human review at 24k scale. If Mistral got factual holdings wrong or invented section numbers on even 5% of rows, you've trained the model on ~1,200 confident fabrications. Every one becomes a hallucination source at inference.

**Action required to close this gap:** fetch 100 random judgment rows + 50 random statute rows from `phase1_v2.jsonl`, locate their source XMLs on HPC, and verify. This is ~4 hours of scripted work and is the single highest-value Track A task remaining.

Flagged as the #1 unverifiable. If you want me to prepare a script to run on HPC, say so — I can write a fidelity-audit script that reads `source_file` field, opens the corresponding XML, and diffs claims. Takes 30 min to draft.

---

### A4. Base vs Instruct — which was actually used (confirmed)

**Phase 1 script** (`finetune_llama_ailes_v2.py`):
- Line 126: `MODEL_NAME = "/mnt/scratch/bgxp240/models/meta-llama-3.1-8b"` — **BASE** (no `-instruct` suffix)
- Line 127: `MAX_SEQ_LENGTH = 3072`
- Lines 130–134: LoRA r=64, α=128, dropout=0.0, target_modules = `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (all 7 attn+mlp)
- Lines 137–143: LR 2e-4, per-device-bs=4, grad-accum=4, eff-bs=16, 2 epochs, warmup 10, max_grad_norm 0.3, weight_decay 0.01
- Lines 197–200: Adam optimiser via `paged_adamw_8bit`, cosine LR schedule

**Phase 2 script** (`finetune_llama_phase1_v2.py`):
- Line 59: `BASE_MODEL = "/mnt/scratch/bgxp240/models/meta-llama-3.1-8b"` — **BASE**, same
- Lines 57, 146: Loads adapter from checkpoint-1250
- Lines 62–68: per-device-bs=2, grad-accum=8, eff-bs=16 (same as Phase 1), LR 2e-4, warmup 20, max_steps=2050
- Lines 71–73: Same LoRA config (r=64, α=128)
- Line 126–131: QLoRA 4-bit, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16
- Line 192: bf16=True
- Line 193: gradient_checkpointing=True
- Line 197: paged_adamw_8bit optimiser

**Tokenizer handling (both scripts):**
- Line 152–154 (Phase 2): `tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "right"`
- Neither script applies a chat template via `tokenizer.apply_chat_template`. Training relies on the raw `text` field of the JSONL, which has the Llama-3 instruct tokens hard-coded in every row.

**SFTTrainer config (both scripts):**
- `dataset_text_field="text"`, `packing=False` — rows are trained individually, not packed into 3072-token sequences
- No explicit loss masking on prompt tokens (SFTTrainer default will compute loss over the whole sequence including system+user tokens). This matters because the model is being trained to predict system-prompt tokens as well as assistant response — dilutes the "learn to respond as assistant" signal.

**Verdict:** confirmed Llama 3.1 8B BASE + QLoRA 4-bit + LoRA r=64 α=128 on all 7 modules + bf16 + gradient checkpointing + paged AdamW 8-bit + LR 2e-4 cosine + ~8.4 epochs of effective exposure. No instruction-following prior; no RLHF; no chat template application at training time — relied entirely on template tokens being present in the text.

**What this implies for the Instruct re-tune:**
- Different base weights. Instruct has RLHF-trained prior for turn boundaries, refusal, politeness, helpfulness. Fine-tune should be **lighter touch** to avoid erasing that prior.
- Specifically: lower LR (1e-4 not 2e-4), fewer steps (800-1000 not 2050), possibly lower rank (r=32 not 64) — Track B will weigh in on this.
- With Instruct you can use `tokenizer.apply_chat_template(messages, ...)` at training time AND at inference time (both tokenisers have the chat template). This is what fixes the "manual template building in vertex_ai.py" problem long-term.

---

### A5. RAG alignment audit — training data vs production prompt shape

**Production prompt shape (v2, as specified in `prompts (9).yaml` + backend dump):**

The `prompts (9).yaml` defines prompt blocks but **does NOT define a prompt shape for RAG-grounded chat.** Specifically:
- `base.system_message` (line 14): the bare AILES identity, no context slot.
- `rag.classifier` (line 310): decides needs_rag yes/no. No context itself.
- `rag.expander` (line 341): produces query plan JSON. No context.
- `legal_reports.judgment_report` (line 148): takes Form E + case info, not retrieved chunks directly.

**There is no `rag.grounded_answer` prompt block.** Nobody has decided how retrieved chunks flow into the final LLM call. In the December POC, the backend concatenated chunk texts into a pseudo-system message — but the current production backend (per `reggobs/regobs-winuyar-ailes-backend`) has no RAG integration at all, per the earlier repo audit. So the production prompt shape for RAG-grounded chat is **undefined** as of today.

**Practical implication:** when Aslam wires RAG into the backend, he'll have to decide on a prompt shape. The natural options are:

```
Option A: append to system message
  system: "You are Ailes... [RETRIEVED CONTEXT]\n{chunks}\n[/RETRIEVED CONTEXT]"
  user:   "{query}"

Option B: prepend to user message
  system: "You are Ailes... When context is provided, use it..."
  user:   "[CONTEXT]\n{chunks}\n[/CONTEXT]\n\n{query}"

Option C: structured messages (assistant-simulated retrieval)
  system: "You are Ailes..."
  user:   "{query}"
  assistant: "I'll search my knowledge base. Here's what I found:"
  tool_result: "{chunks}"
  user:   "Now please answer using that context."
```

Option B is the RAFT convention (Berkeley 2024) and is most common in legal-RAG literature. Option A is simpler and what the Dec POC used. Option C is closer to tool-calling patterns.

**Training data support for any of these: ZERO rows.** As shown in A3, no training example includes any of these markers. Whatever shape Aslam picks in v2, the current model has been trained to ignore it — it will default to parametric-memory answering and possibly trip the citation validator.

**Refusal/low-confidence shape:** also absent. No training rows of the form "Given [CONTEXT], I can't answer this because the context is silent on X." The model has no prior for "context exists but doesn't help." Combined with the 150×-drilled corrections, the failure mode will be: retrieval returns weak chunks, model ignores them, confidently answers from memorised Section 25(2)(f) correction.

**Net alignment finding:** training data and production pipeline are **architecturally disconnected.** The model was trained as a closed-book domain expert; the production pipeline wants an open-book retrieval-grounded reasoner. Without retraining on the retrieval-grounded shape (or at a minimum augmenting the existing training set with ~2k RAG-shape examples), `citation_validator.py` will reject the model's outputs more often than not.

---

### A6. Training recipe summary (one-page reference)

| Aspect | Phase 1 value | Phase 2 value | Notes |
|---|---|---|---|
| Base model | Llama 3.1 8B **BASE** | same (resumes) | NOT instruct |
| Quantisation | 4-bit QLoRA, nf4, double-quant | same | Compute dtype bf16 |
| LoRA rank | 64 | 64 | On all 7 modules (attn + mlp) |
| LoRA alpha | 128 | 128 | 2× rank |
| LoRA dropout | 0.0 | 0.05 | Slight increase in phase 2 |
| Learning rate | 2e-4 | 2e-4 | Cosine schedule |
| Warmup steps | 10 | 20 | |
| Per-device batch | 4 | 2 | |
| Grad accumulation | 4 | 8 | Both: eff-bs=16 |
| Max steps | 2 epochs (~1250) | 2050 total (800 additional from ckpt-1250) | |
| Max seq length | 3072 | 3072 | 201 rows truncated |
| Optimiser | paged_adamw_8bit | same | |
| Gradient checkpointing | on | on | |
| bf16 | on | on | |
| Weight decay | 0.01 | 0.01 | |
| Max grad norm | 0.3 | 1.0 | Loosened in phase 2 |
| Packing | off | off | Rows padded to fixed length; compute inefficient |
| Loss mask on prompt tokens | not applied | not applied | Default SFTTrainer behaviour = compute loss over whole sequence |

**Compute cost estimate (Phase 2 only on L40S, for reference):**
- 800 steps × 49,152 tokens/step = 39.3M tokens processed
- At ~1-2 steps/min on L40S for this config → 400-800 minutes → **7-14 hours**
- Phase 1 (1250 steps from scratch) similar → **12-20 hours** for full 2-phase re-run

---

## ARTEFACTS I STILL NEED TO CLOSE THE AUDIT

1. **XML source files on HPC** (for A3.5 fidelity check). Path: `/users/bgxp240/ailes_legal_ai/data/` — judgments and statutes. Need ~100 judgment XMLs + ~50 statute XMLs plus the Mistral prompt used for Q&A extraction.

2. **Real user chat logs from dev/staging** (for A1 and A2 — ground what the model actually sees). If the backend has Postgres `chat_history` rows from internal testers, a sample of 50 with the query → generated response pairs would let me test hypotheses from A2 against real behaviour instead of the 7-query HPC synthetic set.

3. **The Mistral Q&A generation prompt itself** (for A3.5). If Mistral was told "generate 30 Q&A from this XML," the prompt is probably on HPC too. That prompt is the primary source of any fabrication risk.

4. **Held-out eval set.** The golden queries v2 (20 items) are a retrieval benchmark, not a generation benchmark. A proper eval set should have ~50-100 (query, gold-answer, forbidden-facts, coverage-concepts) tuples. Doesn't exist yet; probably has to be built.

---

## TRACK B — ONLINE RESEARCH (running in background)

A research agent is live-searching for 2024-2026 best practice on: Llama 3.1 8B fine-tune recipes, Instruct vs Base, RAFT/grounded-generation training, legal LLM landscape (SaulLM, Lawyer-LLaMA, Harvey), evaluation benchmarks (LegalBench, LexGLUE, Stanford RegLab 2024), refusal training, catastrophic forgetting mix ratios, synthetic-data quality, and vLLM LoRA deployment. Output will land in `RESEARCH_TRACK_B.md`.

When it completes, I'll merge findings with Track A and produce `SYNTHESIS.md` with:
1. Top-3 root-cause findings
2. Go / no-go on reusing phase1_v2.jsonl
3. Recipe recommendation (every number justified by either Track A evidence or a Track B citation)
4. Time+compute arithmetic
5. The one binary decision you need to make to unblock the re-tune
