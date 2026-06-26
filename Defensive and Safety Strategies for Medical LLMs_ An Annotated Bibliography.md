# Defensive & Safety Strategies for Medical LLMs: A Curated, Annotated Bibliography

## TL;DR
- The strongest entry-point surveys are **Aljohani et al., "A Comprehensive Survey on the Trustworthiness of LLMs in Healthcare"** (EMNLP Findings 2025, arXiv:2502.15871) and **Wang et al., "Trustworthy Medical Question Answering: An Evaluation-Centric Survey"** (arXiv:2506.03659) — both organize the medical-LLM safety literature into mineable taxonomies (truthfulness, privacy, safety, robustness, fairness, explainability) and cite most primary papers below.
- For a code-first experiment comparing defense combinations, the most implementable building blocks are **MedSafetyBench** (NeurIPS 2024, code), **MedHallu/Med-HALT** (hallucination benchmarks, code), **CARES** (clinical adversarial-robustness benchmark), **Llama Guard 3/4 + NeMo Guardrails** (guardrails — the medical "L2M3" paper shows the exact stack), **RMU/Circuit Breakers** (representation interventions, code), and the new **MedForget/MedEditBench** medical unlearning repos.
- The evidence is clear that general-domain safety tuning does not transfer to medicine, that medical-specialized models are *not* safer (often worse on hallucination), and that layered defenses (guard model + RAG grounding + representation/unlearning intervention) are required; you should benchmark each layer against MedSafetyBench + CARES + MedHallu and report safety, utility, and false-refusal jointly.

## Key Findings
- **Surveys come first and are the best bibliographies.** Aljohani et al. screened an initial 30,595 papers and narrowed to a curated set of medical-trust papers (the arXiv §2 text states "we identified a total of 30,595 papers… we narrowed the focus to 69 papers"; the published EMNLP version reports a final set of 62 papers plus 38 datasets). Use its Figure 1 taxonomy as your reading map.
- **Medical safety ≠ general safety.** MedSafetyBench shows publicly available medical LLMs fail AMA-ethics-based safety, and fine-tuning on it improves safety without hurting medical performance.
- **Medical-specialized models are often LESS safe on hallucinations.** Kim et al. (2025) found general-purpose models had a median 76.6% hallucination-free responses vs 51.3% for medical-specialized ones (difference 25.2%, 95% CI 18.7–31.3%, p = 0.012); chain-of-thought reasoning significantly reduced hallucinations in 86.4% of tested comparisons (FDR q < 0.05). Gemini-2.5 Pro exceeded 97% accuracy with CoT (base 87.6%), while MedGemma ranged 28.6–61.9%.
- **Guardrails are the most plug-and-play layer.** Llama Guard 3/4, ShieldGemma, Aegis, and NeMo Guardrails are downloadable and runnable today; the L2M3 paper demonstrates the literal "guard model in front of an LLM" pattern for medicine.
- **Representation interventions (Circuit Breakers/RMU) and unlearning** are the deepest defenses, with released code, and are directly applicable to removing hazardous biomedical knowledge.
- **RAG is a safety mechanism, not just a quality one** — but it introduces a new attack surface (knowledge-base poisoning), so RAG defenses must be tested adversarially.

## Details

### 1. Review / Survey papers (entry points to mine)

**A Comprehensive Survey on the Trustworthiness of LLMs in Healthcare** — Manar Aljohani, Jun Hou, Sindhura Kommu, Xuan Wang (Virginia Tech), EMNLP Findings 2025; arXiv:2502.15871. Category: survey (all). Organizes the field into six dimensions and cites the primary papers in each (Med-HALT, MedSafetyBench, de-identification, misinformation attacks, BiasMedQA, EquityMedQA, etc.). This is the single best starting bibliography. No code; it is a literature map. (aclanthology.org/2025.findings-emnlp.356)

**Trustworthy Medical Question Answering: An Evaluation-Centric Survey** — Yinuo Wang et al., arXiv:2506.03659 (2025). Category: survey (all). Examines six trust dimensions (Factuality, Robustness, Fairness, Safety, Explainability, Calibration) and explicitly reviews evaluation-guided mitigation techniques: retrieval-augmented grounding, adversarial fine-tuning, safety alignment. Best for connecting defenses to the benchmarks that test them.

**Medical Hallucinations in Foundation Models and Their Impact on Healthcare** — Yubin Kim et al. (MIT Media Lab and collaborators), arXiv:2503.05777 / medRxiv 2025.02.28.25323115 (2025). Category: survey + benchmark. Defines medical hallucination, provides a taxonomy, benchmarks 11 models on 7 tasks, and runs a clinician survey (91.8% had encountered medical hallucinations; 84.7% said they could cause harm). Code/resources: github.com/mitmedialab/medical_hallucination.

**Ensuring Safety and Trust: Analyzing the Risks of LLMs in Medicine (MedGuard)** — arXiv:2411.14487 / PMC12478422. Category: survey + benchmark. Proposes a 5-principle framework (Truthfulness, Resilience, Fairness, Robustness, Privacy) and MedGuard-Bench (1,000 expert-verified questions, 100 per aspect). Found 11 LLMs perform poorly regardless of safety alignment vs. human physicians. Implementable as an evaluation suite.

Supporting general-domain surveys worth citing for method context: **SoK: Evaluating Jailbreak Guardrails for LLMs** (arXiv:2506.10597), **Representation Engineering for LLMs: Survey** (arXiv:2502.17601), **Large Language Model Safety: A Holistic Survey** (arXiv:2412.17686).

### 2. Guardrails (external guard models / input-output filtering)

**Llama Guard** — Inan et al., Meta, arXiv:2312.06674 (2023). Foundational external input/output safeguard; Llama2-7B fine-tuned classifier with a customizable taxonomy. Code/weights: huggingface.co/meta-llama/LlamaGuard-7b; PurpleLlama repo. **Llama Guard 3** (8B/1B, Llama-3.1; MLCommons taxonomy, includes an S6 "Specialized Advice" medical category) and **Llama Guard 4** (12B multimodal) are drop-in upgrades. This is the canonical "guard model in front of the LLM." Note: its base taxonomy is not medical; a Medium tutorial shows it fails on custom medical categories without fine-tuning.

**Enhancing Guardrails for Safe and Secure Healthcare AI (L2M3)** — arXiv:2409.17190 (2024). Category: guardrail (medical). The most directly relevant paper to the user's goal: integrates **Llama Guard 3 + NVIDIA NeMo Guardrails + a medical-tuned model (L2M3)**, with Llama Guard 3 doing input validation/jailbreak detection, NeMo doing further checks/retrieval, and L2M3 generating. Evaluated on Med-HALT plus a synthetic fabricated-drug dataset. This is a literal blueprint for a layered medical guard pipeline.

**Aegis / Aegis 2.0** — Ghosh et al. (NVIDIA), arXiv:2404.05993 and arXiv:2501.09004 (NAACL 2025). LoRA fine-tunes of Llama Guard with Defensive/Permissive variants over 13 risk categories; Aegis 2.0 releases an open safety dataset and beats Llama Guard 3-8B. Open weights + data — good for adapting a guard to a medical taxonomy.

**ShieldGemma** — Zeng et al. (Google), arXiv:2407.21772 (2024). Gemma-2-based (2B/9B/27B) content-moderation models; runnable locally via Ollama (`shieldgemma:2b`). Covers 4 harm categories; open-weight.

**A Causal Explainable Guardrails for LLMs (LLMGuardrail)** — arXiv:2405.04160. Category: guardrail + representation. Uses causal analysis on internal representations to defend against jailbreaks, reducing attack success rate by an average of 44.11% across four attacks (GCG, AutoDAN, PAIR, DeepInception) on Vicuna-7B. Implementable; explains guardrail behavior in representation space.

Useful guardrail context for benchmarking: **GuardBench** (EMNLP 2024), **SoK: Evaluating Jailbreak Guardrails** (arXiv:2506.10597, with a Security-Efficiency-Utility framework for comparing defense combinations — directly matches the user's goal), and open frameworks **NeMo Guardrails**, **Guardrails AI**, **LLM Guard**.

A specialized-domain example: **AI Content Moderation in Therapy Conversations** (arXiv:2605.25454) found a fine-tuned Mistral judge (recall 0.5143) outperformed Llama Guard (0.3429), ShieldGemma (0.0286), and OpenAI Omni Moderation (0.0860) on clinically inappropriate mental-health responses — evidence that small domain-tuned guards beat large general ones in medicine.

### 3. Machine unlearning (removing hazardous/private/specific knowledge)

**The WMDP Benchmark + RMU** — Li et al., ICML 2024; arXiv:2403.03218. Category: unlearning + representation. WMDP is 3,668 MCQs on hazardous biosecurity/cyber/chemical knowledge; **RMU** (Representation Misdirection for Unlearning) controls model representations to reduce WMDP-Bio performance while preserving MMLU. Code: github.com/centerforaisafety/wmdp. The most directly relevant foundational unlearning method for removing dangerous biomedical knowledge; the bio split is medicine-adjacent.

**MLLMU-Med** — Dunyuan Xu et al. (CUHK), arXiv:2508.04192 (2025). Category: unlearning (medical, multimodal). The first biomedical multimodal unlearning benchmark; injects synthetic PHI and incorrect clinical facts into LLaVA-Med then benchmarks five unlearning methods (Gradient Ascent, Gradient Difference, KL-Min, IDK-tuning, LLMU) using a novel Unlearning Efficiency Score. Best method (IDK) cut forget-set ROUGE by 45.6% while preserving utility; incorrect-fact removal remained unsolved (best UES only 0.58). Demonstrates current unlearning is weak in medicine. (Code association: CUHK group; verify exact repo.)

**MedForget / CHIP** — Fengli Wu, Vaidehi Patil, Jaehong Yoon, Yue Zhang, Mohit Bansal (UNC Chapel Hill), arXiv:2512.09867 (2025). Category: unlearning (medical, multimodal). HIPAA/GDPR-aligned benchmark modeling hospital data as a nested hierarchy (institution → patient → study); proposes CHIP (Cross-modal Hierarchy-Informed Projection), a **training-free** method that removes target-specific weight subspaces while preserving sibling-shared info. Achieves the highest forget–retain gap across hierarchy levels while keeping competitive medical utility. Code: github.com/fengli-wu/MedForget. Code available + credible lab — a strong implementable entry.

**DuoLearn** — Yi Zhang et al. (clinically authored; Peking Union Medical College Hospital, Qilu Hospital), arXiv:2511.19498 (2025). Category: unlearning (medical, text). Fisher-information-guided orthogonal-projection gradient updates + concept-aware token interventions over a 4-level UMLS/MetaMap medical concept hierarchy, with DP-LoRA modifying ~0.1% of parameters, targeting GDPR right-to-be-forgotten. Reports 82.7% forgetting rate and 88.5% knowledge preservation on MedMCQA surgical-knowledge unlearning (surgical accuracy 89.2%→17.3% with <3.2% drop on other specialties; MIA resistance 0.89 at ε=4.0). No code released.

**MedEditBench / SGR-Edit** — Shigeng Chen, Linhao Luo, Zhangchi Qiu, Yanan Cao, Carl Yang, Shirui Pan (Griffith/Emory/CAS), arXiv:2506.03490, published at EACL 2026 (ACL Anthology 2026.eacl-long.219). Category: knowledge editing (medical). Peer-reviewed; builds a rigorous medical knowledge-editing benchmark and proposes Self-Generated Rationale Editing (using the model's own chain-of-thought as the edit target, so corrections are internalized rather than memorized). Even the strongest baseline (AlphaEdit) scored only 53.9% post-edit; SGR-Edit gives +8.6 pts (LLaMA-8B) / +12.6 pts (LLaMA-3B) over standard editing. Code: github.com/Aries-chen/MedEditBench. Best for correcting medical misinformation in-weights.

Context: **Does Machine Unlearning Truly Remove Knowledge?** (Chen et al., arXiv:2505.23270, NeurIPS 2025 workshop) audits unlearning robustness — important caveat that "forgotten" knowledge can be recovered. Additional medical knowledge-editing benchmarks: **MedMKEB** (arXiv:2508.05083), **MultiMedEdit** (arXiv:2508.07022), **MedLaSA** (arXiv:2402.18099), and retrieval-based **MedEdit** (arXiv:2309.16035).

### 4. Representation-level interventions / activation steering / circuit breakers

**Improving Alignment and Robustness with Circuit Breakers** — Zou et al. (Gray Swan AI), NeurIPS 2024; arXiv:2406.04313. Category: representation intervention. Representation Rerouting (RR) remaps harmful internal representations to refusal/incoherent states; attack-agnostic, reduces harmful outputs by roughly two orders of magnitude under strong attacks on Llama-3-8B/Mistral while preserving MT-Bench/MMLU. Code: github.com/GraySwanAI/circuit-breakers. The premier representation defense; applicable to medicine for refusing harmful clinical requests.

**Representation Bending (RepBend)** — Yousefpour et al., ACL 2025 (aclanthology.org/2025.acl-long.1173). Category: representation intervention. Activation-difference loss (akin to activation steering) that disrupts unsafe representation space; generalizes circuit breakers/RMU.

**Improving LLM Safety with Contrastive Representation Learning** — Simko et al., arXiv:2506.11938 (2025). Category: representation intervention. Triplet-loss extension of circuit breaking with adversarial hard-negative mining; cuts Llama-3-8B embedding-attack ASR from 29%→5% and REINFORCE-GCG from 14%→0%. Code: github.com/samuelsimko/crl-llm-defense.

**Adaptive Activation Steering (ACT)** — Wang et al., arXiv:2406.00034 (2024). Category: representation intervention (truthfulness). Tuning-free inference-time steering toward a "truthful" direction using clustered steering vectors; up to ~140% truthfulness improvement on TruthfulQA across LLaMA/Alpaca/Vicuna. Directly relevant to reducing medical hallucination via steering; needs only ~40 samples.

Note: **Semantic Entropy Probes (SEPs)** (Han et al. 2024, cited in the healthcare survey) approximate semantic entropy from hidden states to detect medical hallucinations cheaply — a representation-level detector.

### 5. Prompt-level / policy / system-prompt defenses

**Med-HEAL: Hallucination-Aware In-Context Learning** — arXiv:2606.01301. Category: prompt defense. Builds a hallucination dataset from EHRNoteQA (MIMIC-IV discharge summaries) and mitigates via hallucination-aware in-context learning — a pure prompt-level mitigation testable in code.

**Chain-of-thought as a prompt-level safety mechanism** — documented in Kim et al. (arXiv:2503.05777): CoT prompting raised Gemini-2.5 Pro to >97% accuracy (from 87.6% base) and significantly reduced hallucinations in 86.4% of comparisons. Implementable as a prompt template.

**Red-Teaming Medical AI (medRxiv 2026.02.26.26347212)** — Category: red-team + prompt-defense limits. Of 160 adversarial prompts against Claude Sonnet 4.5 under a standard medical-assistant system prompt, 11 (6.9%) elicited clinically significant harm (harm level ≥3); the model fully refused in 86.2% of cases. **Authority Impersonation was the dominant attack vector (45.0% success), with the "Educational Authority" sub-strategy (framing as a medical-student question) at 83.3% success, while multi-turn escalation achieved 0% (0/20).** Quantifies the limits of system-prompt-only defenses and the "weak caveat" failure pattern; open attack taxonomy.

Foundational alignment methods applicable via prompting/training: **Constitutional AI / RLAIF** (Bai et al. 2022) — explicit written principles map well to medicine (AMA ethics); the multi-agent medical evaluation-loop paper "Improving the Safety and Trustworthiness of Medical AI via Multi-Agent Evaluation Loops" (arXiv:2601.13268) applies constitutional-style critique to clinical outputs.

### 6. Retrieval restrictions / RAG-based safety

**Retrieval-Augmented Generation in Healthcare: A Comprehensive Review** — MDPI AI 2025 (mdpi.com/2673-2688/6/9/226). PRISMA review of 30 studies on clinical RAG (diagnostic support, EHR summarization, medical QA); catalogs naive/advanced/modular architectures and challenges (retrieval noise, domain shift, explainability). Best RAG-safety survey.

**Evaluating RAG Variants for Clinical Decision Support** — MDPI Electronics 2025 (14/21/4227). Tested 12 RAG types on 250 de-identified vignettes; self-reflective RAG lowered hallucinations to 5.8%; best retrieval accuracy (P@5 ≥ 0.68) from a Haystack DPR+BM25+cross-encoder pipeline. Proposes a hallucination-mitigation framework with retrieval-confidence thresholds, CoT verification, and external fact-checking, plus on-prem deployment with encryption/provenance/audit trails. Highly implementable safety-RAG recipe.

**RAG elevates local LLM quality in radiology contrast-media consultation** — PMC12223273 (2025). A retrieval-augmented local LLM eliminated hallucinations (8%→0%) in a safety-critical radiology use case while preserving privacy via local deployment. Concrete clinical RAG-safety result.

**Knowledge Poisoning Attacks on Medical Multimodal RAG** — arXiv:2605.10253 (2025). Category: RAG attack (motivates defenses). Shows RAG is a new attack surface — injected misinformation in the knowledge base dominates generated diagnoses. Essential to test any RAG defense against.

### 7. Other defenses (adversarial/continual fine-tuning, jailbreak defense, attacks-as-tests)

**MedSafetyBench** — Tessa Han et al., NeurIPS 2024 Datasets & Benchmarks; arXiv:2403.03744. Category: benchmark + fine-tuning defense. First medical-safety benchmark (1,800 harmful medical requests + safe responses, grounded in AMA Principles of Medical Ethics); fine-tuning on it improves medical safety while preserving performance. Code: github.com/AI4LIFE-GROUP/med-safety-bench. The cornerstone benchmark and a fine-tuning defense in one.

**CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs** — Sijia Chen, Xiaomin Li et al. (Harvard), arXiv:2505.11413 (2025). Category: benchmark. Over 18,000 prompts spanning eight medical safety principles, four harm levels, and four prompting styles (direct, indirect, obfuscated, role-play); introduces a three-way response protocol (Accept, Caution, Refuse) and a fine-grained Safety Score that also penalizes over-cautious refusals. The most comprehensive medical jailbreak + false-refusal benchmark — ideal for comparing defenses.

**Towards Safe AI Clinicians: LLM Jailbreaking in Healthcare** — arXiv:2501.18632 / PMC12919412 (2025). Category: attack + fine-tuning defense. Tests 7 LLMs against 3 black-box jailbreaks with an agentic medical evaluation pipeline; shows **Continual Fine-Tuning (CFT)** as a defense reducing attack success by an average of 62.7% on Llama-3.1-8B.

**Medical LLMs are susceptible to targeted misinformation attacks** — Han et al., npj Digital Medicine 2024 (10.1038/s41746-024-01282-7); see also arXiv:2309.17007. Category: attack (motivates defenses). Editing ~1.1% of weights injects persistent false biomedical facts; raises jailbreak success 2%→58% on Llama-3. The threat model that unlearning/editing defenses must counter.

**Medical MLLM is Vulnerable (3MAD / MCM)** — arXiv:2405.20775 (2024). Category: multimodal attack + dataset. 3MAD dataset across medical image modalities; cross-modality jailbreak. Code: github.com/dirtycomputer/O2M_attack. For testing multimodal medical defenses.

**MedHallu** — Pandit et al., arXiv:2502.14302 (EMNLP 2025 main). Category: hallucination-detection benchmark. 10,000 QA pairs from PubMedQA with controlled hallucinations; state-of-the-art LLMs (incl. GPT-4o, Llama-3.1, medically fine-tuned UltraMedical) struggle, with the best model reaching an F1 as low as 0.625 on "hard" cases; incorporating domain knowledge and a "not sure" category improves precision and F1 by up to 38% relative to baselines. Site/code: medhallu.github.io. **Med-HALT** (Pal et al. 2023; github.com/medhalt/medhalt) and **MedHallBench** (arXiv:2412.18947) are complementary hallucination benchmarks.

### Privacy / PHI de-identification (a major medical safety concern)

**LPPA: LLM-Empowered Privacy-Protected PHI Annotation** — Guanchen Wu et al. (Emory), arXiv:2504.18569 (2025). Fine-tunes LLMs locally on synthetic notes for PHI de-identification without exposing real PHI. Implementable privacy defense.

**LLM-Anonymizer** (Llama-3 70B local, NEJM AI 2025; 99.24% PHI removal) and **RedactOR** (Oracle Health, ACL 2025, multimodal incl. clinical audio) are additional implementable de-identification systems. Context: de-identified notes remain vulnerable to membership-inference re-identification, motivating layered privacy defenses.

## Recommendations

**Stage 1 — Establish the evaluation harness first (week 1).** Stand up three benchmarks in code before building defenses: **MedSafetyBench** (harmful-request refusal), **CARES** (jailbreak robustness + false-refusal Safety Score, using its Accept/Caution/Refuse protocol), and **MedHallu + Med-HALT** (factual/hallucination safety). Add **MedGuard-Bench** for privacy/fairness coverage. These give you the safety/utility/false-refusal axes to compare every defense. Benchmark a baseline (e.g., an open medical LLM like Meditron or a general model) with no defenses.

**Stage 2 — Implement defenses as composable layers (weeks 2–4), in increasing depth:**
1. *Prompt layer*: CoT + a constitutional/AMA-ethics system prompt; measure against CARES (expect partial gains — recall the 45% authority-impersonation bypass, and that "Educational Authority" framing hit 83.3%).
2. *Guardrail layer*: replicate the **L2M3 pattern** — Llama Guard 3 (input) → NeMo Guardrails → generator; also try ShieldGemma and an **Aegis 2.0**-style guard fine-tuned to a medical taxonomy. Measure latency and false-refusals, not just blocks.
3. *Retrieval layer*: add self-reflective RAG with retrieval-confidence thresholds and provenance, and test it against the **knowledge-poisoning attack** (arXiv:2605.10253).
4. *Representation/weights layer*: apply **Circuit Breakers (RR)** and/or **RMU** for hazardous-knowledge refusal; for privacy/misinformation removal, try **MedForget/CHIP** or **MedEditBench/SGR-Edit** (both have code).

**Stage 3 — Compare combinations and report jointly.** Use the SoK Security-Efficiency-Utility framing (arXiv:2506.10597) to score each combination on (a) attack-success-rate reduction (CARES/jailbreak benchmarks), (b) medical utility retention (MedQA/MMLU-med), (c) false-refusal rate (CARES Safety Score), and (d) latency.

**Decision thresholds that should change your approach:**
- If a guardrail-only stack leaves CARES jailbreak ASR >10% (likely, given the literature), escalate to representation-level defenses (Circuit Breakers/RMU).
- If utility drops >3–5 points on MedQA after a weights-level intervention, prefer an external guard + RAG combination instead of unlearning.
- If false-refusal rate rises materially (over-blocking benign clinical queries), loosen the guard (Aegis "Permissive") or move filtering to output-only.
- If your use case involves PHI, treat local deployment + de-identification (LPPA/LLM-Anonymizer) as mandatory, since cloud APIs are unsuitable for PHI.

## Caveats
- **Recency/peer-review status:** Several key items are 2025–2026 preprints (MedForget, DuoLearn, MLLMU-Med, several medRxiv/arXiv entries) not yet peer-reviewed; MedSafetyBench, WMDP/RMU, Circuit Breakers, MedEditBench, MedHallu, and the trustworthiness survey are peer-reviewed. Treat preprint numbers as provisional.
- **Unlearning is not bulletproof:** auditing work (arXiv:2505.23270) shows "forgotten" knowledge can be recovered via relearning/perturbation; MLLMU-Med shows current methods struggle with incorrect-fact removal in medicine.
- **Guardrails can become attack vectors:** some guard models generated harmful content when prompt-injected (arXiv:2511.22047); guards must themselves be red-teamed (Garak, PyRIT, HarmBench).
- **Domain transfer fails:** general safety fine-tuning and even medical fine-tuning do not reliably improve medical safety/hallucination; medical-specialized models can be *worse* on hallucination than general models.
- **Multimodal vs text:** several medical unlearning/attack papers (MLLMU-Med, MedForget, 3MAD) are multimodal (VQA); if your experiments are text-only, prioritize DuoLearn, MedEditBench, RMU, and Circuit Breakers.
- **Some metrics are composite** (UES, hierarchy gaps, CARES Safety Score) and not directly comparable across papers; normalize within your own harness.
- **Tooling note:** "Perplexity" (recommended to the user) is a retrieval/search assistant — useful as a literature-discovery aid, but it is a product, not a defense method, and should not be confused with the perplexity-based input filter that appears in some jailbreak-defense benchmarks.