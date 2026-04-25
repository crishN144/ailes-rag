# RESEARCH_TRACK_B — External Evidence Audit for AILES Retune

**Scope:** Online research audit to support the decision on whether to retune AILES on Llama 3.1 8B Instruct with current data, augmented data, or a rebuild. All claims are cited with URL + publication date. Dates are as reported by the source; where unclear I state so.

---

## One-sentence pilot recommendation

**If AILES only changes one thing, it should rebuild the training data in RAFT format (oracle passage + ~4 distractors + CoT answer with inline chunk-id citations) for at least ~70% of rows, because the current corpus contains zero rows matching the production retrieved-context prompt shape and this is the single largest documented cause of the failure mode the team is seeing.** Justification:

- RAFT (Zhang & Patil, Berkeley, Mar 2024) reports up to **+35.25%** on HotpotQA vs. base Llama-2 + RAG and **+28.9%** vs. domain-SFT alone — the published lift from training-shape matching the inference-shape is large and consistent. <https://arxiv.org/abs/2403.10131>
- Stanford RegLab (Dahl et al., Journal of Legal Analysis, 2024) documents that off-the-shelf LLMs hallucinate case law **69–88%** of the time and that RAG-augmented legal tools still hallucinate in more than **1 of 6** queries, so the model must be explicitly trained to ground on retrieved passages and to abstain when they are silent. <https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries>
- Observation specific to AILES: the stated training mix has 72.2% single-turn Q&A with **no retrieved context in the prompt**, 12.2% UltraChat conversational (explains the hallucinated-continuation bug when a conversational prompt arrives), and 0% rows in prod prompt shape — this is a training/serving distribution mismatch, not a hyperparameter problem.

The rest of this document is the evidence base.

---

## B1. Fine-tuning Llama 3.1 8B for a specialised domain — current best practice

**Summary.** The 2025 consensus for single-GPU domain adaptation of Llama 3.1 8B is LoRA on all seven attention + MLP projections (q,k,v,o,gate,up,down) with rank 16–64 and α = rank or 2·rank, 1–3 epochs, LR 1–2e-4 cosine, and QLoRA 4-bit nf4 if VRAM-bound. Production legal LLMs (SaulLM family) are trained via **continued pretraining on domain tokens, then IFT, then preference alignment** rather than SFT-only; SaulLM-7B used 30B legal tokens of CPT before IFT. QLoRA 4-bit costs roughly **1 MMLU point** vs 16-bit LoRA and ~30–40% longer wall-time, which is the accepted trade-off at 8B.

**Evidence.**
- Equall, "Saul – The first family of open models for law" (blog, 2024; paper arXiv:2403.03883, Mar 2024): Saul-7B used 30B legal tokens CPT → IFT → DSO; +6% LegalBench avg. over Mistral-7B. <https://arxiv.org/abs/2403.03883> and <https://blog.equall.com/saul>
- Dettmers et al., "QLoRA" (arXiv:2305.14314, May 2023) and follow-ups: QLoRA NF4 lags 16-bit LoRA by ~1 pp MMLU; memory cut ~4×. <https://arxiv.org/pdf/2305.14314>
- Schmid, "How to fine-tune open LLMs in 2025 with Hugging Face" (philschmid.de, 2025): recommends LoRA on all linear modules, r=16–64, α=rank, LR 1–2e-4, 1–3 epochs as the default recipe. <https://www.philschmid.de/fine-tune-llms-in-2025>
- Spheron, "Axolotl vs Unsloth vs TorchTune" (2026 update of 2025 piece): Unsloth 3.2 h vs Axolotl 5.8 h on single-A100 Llama-3.1-8B with identical configs; LLaMA-Factory uses Unsloth as an acceleration backend. <https://www.spheron.network/blog/axolotl-vs-unsloth-vs-torchtune/>
- Unsloth docs, "What model should I use" (2025): defaults of r=16, α=16, LR 2e-4, 3 epochs; target all 7 modules for domain adaptation. <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/what-model-should-i-use>

**Implication for AILES.** Current config (r=64 α=128 all 7 modules, QLoRA nf4) is well within normal 2025 practice — it is not the cause of the bug. The missing step relative to shipped legal LLMs (SaulLM) is **continued pretraining on UK legal tokens before SFT**; with 310k Qdrant chunks available, even 1 epoch of CPT on a deduplicated judgment subset before the SFT phases would bring AILES closer to the SaulLM recipe. If time-pressed, skip CPT and fix the data shape first (see B3).

---

## B2. Instruct vs Base — the chat-template question

**Summary.** Current consensus in 2025: start from **Instruct** when your domain corpus is small-to-medium (<~50k rows) and you need conversational + instruction-following behaviour out of the box; start from Base only when you have enough data to rebuild those capabilities or when you deliberately want a blank template. Training on Instruct with the wrong chat template — or without loss-masking the prompt tokens — demonstrably degrades instruction-following. The safe recipe is: apply the model's **exact** Llama 3.1 chat template (`<|begin_of_text|><|start_header_id|>system|...`), mask prompt/system tokens from the loss (labels = -100), preserve the EOS (`<|eot_id|>`), and keep domain:general replay at a non-trivial ratio.

**Evidence.**
- Unsloth, "What model should I use for Fine-tuning?" (docs, 2025): explicit rule — "For smaller datasets (<300 rows) the instruct model is typically the better choice"; base models have no chat template and require you to define one. <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/what-model-should-i-use>
- Shi et al., "Instruction Fine-Tuning: Does Prompt Loss Matter?" (arXiv:2401.13586): prompt-loss-weight has a significant quadratic effect on short-completion data; PLW=0 (full masking) is the safe default for SFT. <https://arxiv.org/html/2401.13586v2/>
- Gottesman, "Mask Your User Tokens" (May 2024): practical walkthrough of setting labels = -100 for prompt tokens in Llama chat templates. <https://yonigottesman.github.io/2024/05/13/mask-user-tokens.html>
- AI2, "Tülu 3 Technical Blog" (Nov 2024): full open recipe for layering instruction + preference training on Llama 3.1 base; demonstrates that correct template + mixed-capability data beats DeepSeek V3 on some suites. <https://allenai.org/blog/tulu-3-technical>
- "Scaling Laws for Forgetting When Fine-Tuning LLMs" (arXiv:2401.05605): fine-tuning on narrow domain data produces predictable forgetting of general capabilities proportional to update magnitude; motivates replay mixture. <https://arxiv.org/html/2401.05605v1>

**Implication for AILES.** Starting from **Llama 3.1 8B Instruct** is the right call for AILES given the ~25k training rows and the requirement to handle conversational turns. The current hallucinated-continuation bug — substantive queries work, conversational ones produce fake continuations — is the textbook failure mode of training Base on single-turn Q&A: the model never learned a turn-termination policy. Three operational rules: (i) use the exact Llama 3.1 chat template unchanged, not a fourth custom AILES variant; (ii) loss-mask everything before the assistant turn; (iii) verify `<|eot_id|>` appears as the label for end-of-response in every row, because a missing EOS is the most common cause of "continuation hallucination."

---

## B3. RAG-aware fine-tuning — THE CRITICAL SECTION

**Summary.** RAFT (Berkeley, Mar 2024) is the published state-of-the-art recipe for domain RAG SFT and the most directly applicable to AILES. Each training row is: `{question, [oracle_doc, distractor_1..k], CoT answer that quotes verbatim from the oracle and cites it}`. Published ablations: **k = 4 distractors** is optimal; the fraction of rows that include the oracle (**P%**) should be ~60–80% — the remaining 20–40% contain only distractors so the model learns to refuse/abstain. CoT with inline verbatim quotes adds **+9.66%** on HotpotQA over non-CoT. RAFT beats DSF+RAG by **+28.9%** on HotpotQA; gains are smaller on binary-QA datasets like PubMedQA. Self-RAG and CRAG are complementary — Self-RAG adds reflection/critique tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`); CRAG adds a lightweight retrieval-quality evaluator at inference. For legal, inline structured citations (`[chunk_id:span]` or `[[BAILII-URL#para-N]]`) generalise better than free-form footnotes because they are mechanically verifiable by a post-generation validator.

**Evidence.**
- Zhang, Patil et al., "RAFT: Adapting Language Model to Domain Specific RAG" (arXiv:2403.10131, Mar 2024 / ICLR 2025): full schema, k=4 distractors optimal, P% = 60–100% sweep with 80% robust, +35.25% vs Llama-2+RAG on HotpotQA, CoT adds 9.66%. <https://arxiv.org/abs/2403.10131> and Berkeley Gorilla blog: <https://gorilla.cs.berkeley.edu/blogs/9_raft.html>
- Asai et al., "Self-RAG" (arXiv:2310.11511, Oct 2023; ICLR 2024): training format with reflection tokens; improves citation accuracy for long-form generation over ChatGPT + Llama-2-chat+RAG. <https://arxiv.org/abs/2310.11511>
- Yan et al., "CRAG: Corrective Retrieval Augmented Generation" (arXiv:2401.15884, Jan 2024): lightweight retrieval evaluator gates generation on retrieval quality. <https://arxiv.org/abs/2401.15884>
- Microsoft, "RAFT: A new way to teach LLMs to be better at RAG" (Azure AI Foundry blog, Mar 2024): production framing of RAFT; confirms 4-distractor default and CoT-with-quotes. <https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/4084674>
- SuperAnnotate, "RAFT: Combining RAG with fine-tuning" (2024): restates the P% = 80% finding and the `##begin_quote##...##end_quote##` verbatim citation convention. <https://www.superannotate.com/blog/raft-retrieval-augmented-fine-tuning>

**Implication for AILES.** This is where AILES is failing. Zero of the 24,674 rows in phase-2 match production prompt shape; the model has literally never been trained to condition on a retrieved-context block. Concrete prescription drawn from the RAFT paper, adapted to AILES: target mix roughly **70% oracle+distractor rows / 20% distractor-only (train abstention) / 10% pure general-chat replay**; k = 4 distractors from Qdrant's BM25+BGE neighbours of the oracle (use hard negatives, not random); CoT answers must inline-cite the oracle's `chunk_id` and quote verbatim spans; preserve the existing citation validator as a post-generation check. Because Qdrant already holds 310k chunks, you can auto-generate distractors at training time rather than hand-curate.

---

## B4. Legal LLM landscape (2024–2026) — who shipped what

**Summary.** SaulLM (Equall, Mar 2024, extended to 54B/141B in Aug 2024) is the clearest published recipe: CPT on a large legal corpus (30B tokens for 7B, 540B for 141B) → IFT on legal instruction data → DSO preference alignment on Mistral base. Harvey (OpenAI partnership) runs a custom-trained case-law model plus an agent framework built on the OpenAI Agent SDK with a "lawyers-in-the-loop" eval process and zero training on customer data. Chinese-legal models (ChatLaw, DISC-LawLLM, Lawyer-LLaMA) broadly follow the pattern base+CPT+IFT+RAG; documented failure modes include jurisdiction drift, distractor sensitivity (ChatLaw explicitly noted as poor at ignoring distractors), and low absolute benchmark scores (best Lawyer-LLaMA-13B-v2 ≈ 0.29 on the cited benchmark). UK-specific open corpora: **BAILII** (comprehensive UK/IE case law and legislation) and **The National Archives Find Case Law** service (bulk API, LegalDocML XML, explicitly cleared for AI training).

**Evidence.**
- Colombo et al., "SaulLM-7B: A pioneering Large Language Model for Law" (arXiv:2403.03883, Mar 2024). <https://arxiv.org/abs/2403.03883>
- BigDATAwire, "Equall Introduces Expanded Saul Family of Legal LLMs with 54b and 141b Models" (Aug 2024): 540B legal tokens for the 141B. <https://www.bigdatawire.com/this-just-in/equall-introduces-expanded-saul-family-of-legal-llms-with-54b-and-141b-models/>
- OpenAI, "Customizing models for legal professionals" (openai.com/index/harvey, 2023/2024): 10B-token Delaware+US case-law custom model for Harvey. <https://openai.com/index/harvey/>
- Harvey, "Harvey is building legal agents and workflows with OpenAI o1" (harvey.ai blog, 2024): agent-first architecture. <https://www.harvey.ai/blog/harvey-building-legal-agents-and-workflows-with-openai-s-o1>
- "Large language models in law: A survey" (ScienceDirect, 2024): summarises Lawyer-LLaMA, ChatLaw, DISC-LawLLM failure modes including distractor sensitivity. <https://www.sciencedirect.com/science/article/pii/S2666651024000172>
- Transparency Project, "All about BAILII" (2021–2024 series); TNA Find Case Law coverage announced via Law Gazette, "Government considers plans to create national hub for court judgments." <https://www.lawgazette.co.uk/news/government-considers-plans-to-create-national-hub-for-court-judgments/5108426.article>

**Implication for AILES.** Precedent for "CPT on jurisdiction-specific corpus → SFT → alignment" is well-established at 7B for legal. AILES' 310k UK judgment/statute chunks are already the right corpus; the question is whether to spend a day on light CPT before the SFT retune or to skip it. For a tomorrow-morning deadline: skip CPT, fix the SFT shape (B3). For the next iteration: do 1 epoch of CPT on deduplicated BAILII+TNA text before SFT — this is the differentiator between "fine-tuned chatbot" and "SaulLM-class domain model." Family-law-specific: there is no published open family-law LLM as of April 2026 from this search; AILES has first-mover position if executed well.

---

## B5. Legal LLM evaluation — benchmarks + hallucination protocols

**Summary.** The canonical benchmarks are **LegalBench** (162 tasks, 6 reasoning categories; US-leaning but the reasoning-category framework transfers), **LexGLUE** (EU + US legal-language understanding; mostly classification), and the newer **LEXam** (May 2025, 7,537 law-exam questions in English and German across 340 exams — closest to exam-style legal reasoning AILES would face). The reproducible hallucination protocol is Dahl et al.'s Stanford RegLab methodology: generate a controlled set of verifiable queries (case existence, holding, precedential relationship, court of origin), score against a ground-truth database, and report hallucination rate by query type and jurisdiction. For RAG-specific legal hallucination, the 2024 RegLab/Stanford Law study on Westlaw / Lexis+ / Ask Practical Law AI found hallucination **even in RAG-assisted tools** in **≥17%** of queries, establishing that RAG alone does not solve the problem — grounded-generation training does.

**Evidence.**
- Guha et al., "LegalBench" (NeurIPS 2023 / GitHub HazyResearch/legalbench). <https://github.com/HazyResearch/legalbench/>
- LexGLUE dataset/benchmark (Chalkidis et al.) summarised in LEXam paper. <https://arxiv.org/html/2505.12864v1>
- Fan et al., "LEXam: Benchmarking Legal Reasoning on 340 Law Exams" (arXiv:2505.12864, May 2025). <https://lexam-benchmark.github.io/>
- Dahl, Magesh, Suzgun, Ho, "Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models" (Journal of Legal Analysis 16, 2024; arXiv 2024). <https://reglab.stanford.edu/publications/hlarge-legal-fictions-profiling-legal-hallucinations-in-large-language-models/>
- Magesh, Surani, Dahl et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools" (Stanford RegLab, May 2024; J. Empirical Legal Studies 2025): documented ≥17% hallucination in RAG-backed commercial legal tools. <https://reglab.stanford.edu/publications/hallucination-free-assessing-the-reliability-of-leading-ai-legal-research-tools/>

**Implication for AILES.** Build a small AILES-specific eval suite before the retune starts: (1) ~200 hand-written UK family-law questions with ground-truth answers and required citations; (2) a Dahl-style verifiable-query set (does this case exist? what was the holding? which court?); (3) an abstention set where retrieval deliberately returns irrelevant chunks and the correct answer is "I cannot answer from the provided materials." Track hallucination rate, citation validity rate, and abstention precision/recall. Without this, you cannot tell whether the retune helped.

---

## B6. Grounded generation + citation training

**Summary.** Two complementary levers: (i) **training-time** — supervise fine-grained citation quotes, as in FRONT/FARD-style approaches that supervise the model with verbatim supporting quotes from the oracle; (ii) **inference-time** — constrained decoding restricted to valid `chunk_id`s from the retrieved set, guaranteeing every citation resolves. FActScore (EMNLP 2023) remains the reference automated factuality metric for long-form generation. Published realistic lift for an 8B + LoRA team: SaulLM recipe (CPT+IFT+DSO) adds ~6 pp on LegalBench; RAFT adds up to ~29 pp on open-domain multi-hop. For AILES, a realistic expectation after a proper RAG-aware retune is **10–25 pp improvement in citation-validity rate** and a meaningful drop in fabricated-case-name incidents, not a miracle on raw reasoning.

**Evidence.**
- Min et al., "FActScore" (EMNLP 2023, arXiv:2305.14251). <https://arxiv.org/abs/2305.14251>
- Survey: "Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with LLMs" (arXiv:2508.15396, Aug 2025): covers FRONT, FARD, and citation-supervised SFT methods. <https://arxiv.org/html/2508.15396v1>
- Aidan Cooper, "A Guide to Structured Outputs Using Constrained Decoding" (aidancooper.co.uk, 2024). <https://www.aidancooper.co.uk/constrained-decoding/>
- vLLM docs / Nexastack, "Structured Decoding with vLLM" (2024/2025): grammar-constrained decoding is a first-class vLLM feature now usable to force well-formed citation tokens. <https://www.nexastack.ai/blog/structured-decoding-with-vllm>

**Implication for AILES.** Combine: train with inline `[chunk_id]` citations in the RAFT rows (B3), then at inference use vLLM grammar-constrained decoding to restrict generated `chunk_id` tokens to members of the retrieved set. The existing citation validator becomes a safety net rather than the primary guarantee. Evidence for the exact percentage lift at 8B is weak for legal specifically — **treat the 10–25 pp figure as a hypothesis, not a commitment.**

---

## B7. Refusal training and uncertainty calibration

**Summary.** R-Tuning (NAACL 2024 Outstanding Paper) shows that adding a modest fraction of refusal-aware examples — constructed by partitioning training data into "known" and "unknown" by probing the base model — teaches reliable abstention that generalises to OOD. The 2025 follow-ups (AbstentionBench, Abstain-R1) emphasise that over-refusal is the failure mode of naive refusal training and that RL with verifiable rewards or calibrated rewards is needed for production. Published recipes do not pin down an exact percentage, but RAFT's own finding — that 20–40% of rows containing only distractors (no oracle) is optimal for in-domain robustness — effectively doubles as a refusal-training signal when the gold answer on those rows is "the provided materials do not answer this."

**Evidence.**
- Zhang, Diao et al., "R-Tuning: Instructing LLMs to Say 'I Don't Know'" (NAACL 2024 Outstanding Paper, arXiv:2311.09677). <https://arxiv.org/abs/2311.09677>
- "Know Your Limits: A Survey of Abstention in LLMs" (TACL 2024). <https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754/131566/Know-Your-Limits-A-Survey-of-Abstention-in-Large>
- "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions" (arXiv:2506.09038, Jun 2025). <https://arxiv.org/html/2506.09038v1>
- RAFT paper (B3) — the "distractor-only" rows as implicit abstention training.

**Implication for AILES.** Allocate ~20% of the RAFT-format rows to explicit "retrieval is insufficient" training with the gold answer being a concise abstention ("Based on the materials provided, I cannot determine X. Relevant authorities would include …"). This directly addresses the production failure mode where the model confabulates rather than abstaining. Evidence for the exact percentage is moderate; 20% is the RAFT default, not a legal-specific calibration — monitor over-refusal on the eval set from B5.

---

## B8. Catastrophic forgetting + capability preservation

**Summary.** 2025 consensus: LoRA "learns less and forgets less" than full fine-tuning, but forgetting is non-zero and scales with update magnitude and dataset narrowness. A domain:general replay ratio of roughly **80:20 to 70:30** is the accepted working range; pure-domain fine-tuning reliably kills instruction-following on Instruct bases. Replay with real pre-training / instruction data is preferred over self-synthesised.

**Evidence.**
- Biderman et al., "Scaling Laws for Forgetting When Fine-Tuning LLMs" (arXiv:2401.05605, Jan 2024). <https://arxiv.org/html/2401.05605v1>
- "How to Alleviate Catastrophic Forgetting in LLMs Finetuning?" (arXiv:2501.13669, Jan 2025). <https://arxiv.org/pdf/2501.13669>
- "Conditions for Catastrophic Forgetting in Multilingual translation" (ACL MRL 2025): LoRA forgets less but not zero. <https://aclanthology.org/2025.mrl-main.23.pdf>
- "Replaying pre-training data improves fine-tuning" (arXiv:2603.04964, 2026). <https://arxiv.org/html/2603.04964>
- SMoLoRA (ICCV 2025, arXiv:2411.13949): separable routing to preserve instruction-following in continual LoRA. <https://arxiv.org/pdf/2411.13949>

**Implication for AILES.** Current mix: ~72% legal Q&A + 7.5% "corrections" + 12.2% UltraChat + 8.1% OpenOrca = **~80:20 domain:general**, which is in the accepted range. Forgetting is **not** the primary problem. Critique: 3,000 UltraChat rows is enough to preserve general chat if they match the production template, but if they use a different system prompt (as the four-prompt variance suggests), the model sees them as a different distribution and the replay benefit is diluted. **Normalise all rows to one system prompt — the production one — before retraining.** This alone probably explains a meaningful fraction of the conversational-hallucination bug.

---

## B9. Data quality: synthetic data + correction weighting + duplication

**Summary.** LIMA (Meta, 2023) and its 2024–2025 successors establish firmly that **1,000–10,000 high-quality, diverse examples beat 50k–100k mixed-quality examples** for instruction alignment; ablations show diminishing returns past ~10k when quality is held constant. Multi-epoch training past 3 epochs hurts most instruction datasets. On duplication specifically: heavy oversampling of a small set of "correction" rows (AILES: ~20 unique texts repeated 10×–150×) is a known-bad pattern — it functions as targeted overfitting and the model memorises the exact wording rather than the underlying correction principle. Synthetic self-distillation is now standard (Tülu 3 used 43% synthetic prompts) but requires quality filtering.

**Evidence.**
- Zhou et al., "LIMA: Less Is More for Alignment" (arXiv:2305.11206, 2023). <https://arxiv.org/pdf/2305.11206>
- "When Scaling Meets LLM Finetuning" (ICLR 2024): diminishing returns beyond curated moderate-sized sets. <https://openreview.net/pdf?id=5HCnKDeTws>
- Raschka, "Practical Tips for Finetuning LLMs Using LoRA": performance drop when 2× training iterations on instruction data; overfitting at 3–5 epochs on <5k datasets. <https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms>
- Databricks, "LIMIT: Less Is More for Instruction Tuning" (2023/2024): independent confirmation of LIMA. <https://www.databricks.com/blog/limit-less-more-instruction-tuning>
- AI2, "Tülu 3" (Nov 2024): 43% synthetic prompts via persona-driven generation, with explicit quality curation pipeline. <https://allenai.org/blog/tulu-3-technical>

**Implication for AILES.** The 150×-oversampled correction rows are probably actively harmful: the model is memorising ~20 specific texts. Recommend (a) cap duplication at 3–5× per unique text, (b) rewrite the 20 unique corrections into 100–300 paraphrased variants instead of duplicating, (c) if time allows, drop the phase-1/phase-2 split and retrain in one pass on a cleaned 15k–20k set of RAFT-format rows + 3k replay — evidence says this will outperform the current 24.6k mixed-quality mix. **Quality beats quantity is the single strongest, most-cited finding in this audit.**

---

## B10. Deployment considerations that feed back into training

**Summary.** vLLM supports first-class LoRA hot-swap via `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` and the `/v1/load_lora_adapter` endpoint, with per-adapter load/unload latency small enough for iteration cycles. Merging into base gives slightly higher throughput but kills iteration speed; keep adapters unmerged during the rapid-iteration phase, merge before a production pin. Latency: Llama 3.1 8B on an L4 (24 GB, 300 GB/s) is fine for single-digit concurrency at modest context; above that, A100 or H100 is strongly preferred. MLPerf 5.1 (Sep 2025) benchmarks Llama 3.1 8B as its small-LLM reference on A100/H100/L40S. Practical long-context cap on a single L4 at acceptable latency is typically ~8k tokens; beyond that, KV-cache latency dominates.

**Evidence.**
- vLLM docs, "LoRA Adapters" (docs.vllm.ai, 2025). <https://docs.vllm.ai/en/latest/features/lora/>
- Unsloth, "LoRA Hot Swapping Guide" (2025). <https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide>
- MLCommons, "MLPerf Inference 5.1: Benchmarking Small LLMs with Llama3.1-8B" (Sep 2025). <https://mlcommons.org/2025/09/small-llm-inference-5-1/>
- Microsoft, "Inference performance of Llama 3.1 8B using vLLM across various GPUs and CPUs" (2024/2025): L4 memory-bandwidth constraint flagged explicitly. <https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/inference-performance-of-llama-3-1-8b-using-vllm-across-various-gpus-and-cpus/4448420>
- Ori, "Benchmarking Llama 3.1 8B Instruct on H100 and A100 with vLLM" (2024). <https://www.ori.co/blog/benchmarking-llama-3.1-8b-instruct-on-nvidia-h100-and-a100-chips-with-the-vllm-inferencing-engine>
- "Improving the Serving Performance of Multi-LoRA LLMs" (arXiv:2505.03756, May 2025). <https://arxiv.org/html/2505.03756v1>

**Implication for AILES.** Running Vertex AI on L4 constrains usable context. If top-k=8 chunks × ~500 tokens = 4k retrieved + prompt + system + response, you are close to the L4 comfort zone (~8k). This bounds RAFT training rows: **cap training-row total length at ~6–7k tokens** so the model never sees a distribution that cannot be reproduced at inference. Also, plan to keep LoRA adapters hot-swappable in vLLM until you have a production-pinned eval pass; the iteration-speed gain is worth more than the ~5–15% merged-adapter throughput advantage at the current stage. If the model moves to A100 in europe-west2, revisit.

---

## Cross-cutting recommendations (condensed)

1. **Start from Llama 3.1 8B Instruct.** Evidence: B2.
2. **Rebuild the SFT data in RAFT format** with one production chat template: ~70% oracle+4-distractor+CoT+inline `[chunk_id]` citations, ~20% distractor-only abstention rows, ~10% general replay (UltraChat subset only, re-templated). Evidence: B3, B7.
3. **Cap correction-row duplication at 3–5×;** rewrite the ~20 unique texts into paraphrased variants. Evidence: B9.
4. **Normalise every row to the exact production Llama 3.1 chat template** with EOS on every assistant turn and loss-masking on everything before it. Evidence: B2.
5. **Keep LoRA config essentially as-is** (r=64 α=128 all 7 modules QLoRA nf4); the problem is data, not hyperparams. 1–2 epochs, LR 1–2e-4 cosine. Evidence: B1, B8.
6. **Build a ~200-question UK family-law eval suite before retraining** (Dahl-style verifiable queries + abstention set + citation-validity scoring). Evidence: B5.
7. **Defer continued pretraining on BAILII/TNA to iteration 2;** the deadline does not support it. Evidence: B4.
8. **At inference, add vLLM grammar-constrained decoding on citation tokens** to guarantee every `[chunk_id]` resolves. Evidence: B6.

Sections where evidence is weak and recommendations should be treated as hypotheses: exact refusal percentage (B7), exact citation-lift magnitude at 8B for legal specifically (B6), L4 long-context ceiling under real AILES traffic (B10).
