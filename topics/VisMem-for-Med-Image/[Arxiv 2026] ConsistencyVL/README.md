# Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models

**Authors:** Logan Mann, Yi Xia, Ajit Saravanan, Ishan Dave, Saadullah Ismail, Shikhar Shiromani, Emily Huang, Ruizhe Li, Kevin Zhu

**Affiliations:** UC Santa Barbara, Algoverse AI Research, UC Berkeley, Independent Researcher

**Venue:** arXiv 2026 (2606.17389)

**Code:** https://github.com/itsloganmann/VLM-Reliability-Probe

---

## One-Sentence Summary

This paper systematically demonstrates that spatial attention patterns in Vision-Language Models (VLMs) have near-zero correlation with output correctness (R ~ 0.001), and instead, reliability signals are best captured by generation-time dynamics (Self-Consistency, R = 0.429) and hidden-state probes (AUROC > 0.95), revealing a fundamental "Symbolic Detachment" between visual grounding and truthful generation.

---

## Key Contributions

1. **"Cluster Failure" Discovery:** Spatial attention metrics (cluster count C_k, spatial entropy H_s) show near-zero correlation (R ~ 0.001) with correctness across three VLM families (LLaVA-1.5, PaliGemma, Qwen2-VL), directly refuting the widely-held "Attention-Confidence Assumption."

2. **"Symbolic Detachment" Mechanism:** Layer-wise attention evolution analysis reveals "Early Locking" -- models sharpen visual attention early but later diffuse it, severing the link between perception and generation. This explains why attention maps are statistically orthogonal to truth.

3. **Hidden-State Probes as Reliability Detectors:** Trained probes on internal hidden states achieve AUROC > 0.95 for predicting answer correctness in single-pass inference, drastically outperforming attention-based metrics (AUROC ~ 0.50) and output confidence (AUROC ~ 0.54).

4. **Architectural Divergence in Causal Robustness:** Large-scale ablation reveals that LLaVA relies on fragile, localized late-stage bottleneck neurons (ablation of just 5 neurons drops object-ID accuracy by 8.3pp), while PaliGemma and Qwen2-VL distribute reliability globally, remaining robust even when > 50% of neurons in predictive layers are destroyed.

5. **Self-Consistency as Gold Standard Behavioral Signal:** Agreement across K=10 sampled reasoning paths (R = 0.429, AUROC = 0.78-0.81) emerges as the strongest behavioral reliability signal, though at 10x inference cost. Hidden-state probes achieve higher AUROC (up to 0.971) at single-pass cost.

---

## Section Navigation

| Section | File | Description |
|---------|------|-------------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | Full abstract with annotations |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | Problem motivation and the Attention-Confidence Assumption |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | VLMs, hallucination, interpretability, language prior |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | VRP framework: Structural vs. Consistency hypotheses |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | Full results: attention failure, logit lens, sparse circuits, reliability prediction |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | Summary, limitations, future work |

---

## Key Numbers

| Metric | Value | Context |
|--------|-------|---------|
| R(C_k, y) -- Cluster Count vs. Correctness | 0.001 (95% CI [-0.034, 0.036]) | Near-zero correlation, p > 0.05 |
| R(H_s, y) -- Spatial Entropy vs. Correctness | -0.012 (95% CI [-0.047, 0.024]) | Statistically indistinguishable from noise |
| Self-Consistency (SC) R | 0.429 | Strongest behavioral reliability signal |
| Precision at SC=1 | 90.8% (95% CI [88.4, 92.8]%) | High precision when all 10 samples agree |
| Hidden-State Probe AUROC (LLaVA) | 0.956 (POPE) | Near-perfect reliability discrimination |
| Hidden-State Probe AUROC (Qwen2-VL) | 0.971 (POPE) | Best single-pass reliability detector |
| Supervised Attention Probe AUROC | 0.725 | Attention carries limited signal |
| Spatial Attention AUROC | 0.50 | Random-chance level |
| Logit Entropy AUROC | 0.50-0.52 | Poor calibration baseline |
| Output Confidence AUROC | 0.54 | Marginally above random |
| MLP Contribution to Margin Growth (LLaVA) | 82.1% | Reliability driven by feature processing, not routing |
| LLaVA Accuracy Drop (top-5 neurons ablated, Object ID) | -8.3pp | Fragile localized bottleneck |
| PaliGemma Accuracy Drop (1000/2048 neurons ablated) | -1.0pp | Highly distributed, robust |
| Qwen2-VL Accuracy Change (2000/3584 neurons ablated) | +2.0pp (within noise) | Extreme resilience |
| Qwen2-VL on POPE | 28.8% accuracy | Severely miscalibrated (caution: low model accuracy may inflate probe AUROC) |
| Pooled Structural-Analysis Set | n = 3,090 | POPE + LLaVA-Bench + custom tasks |
| Self-Consistency Cost | 10x inference | K=10 samples with nucleus sampling (p=0.9, T=0.7) |

---

## Data Flow: Input -> Intermediate -> Output

```
[Image + Question]
       |
       v
[Stage 1: Structural Metrics]
  - Extract cross-attention maps A^{(l,h)} from visual encoder
  - Average over heads & answer-token positions -> per-layer spatial vector m^{(l)} in R^S
  - Compute: Cluster Count (C_k), Spatial Entropy (H_s), Attention Evolution (ΔH_s)
  - Output: Structural reliability scores (FAIL: R^2 < 0.08)
       |
       v
[Stage 2: Mechanistic Probes]
  - Logit Lens: Project hidden state h_l to vocabulary space
  - Compute Truth Margin ΔM_l = logit(correct) - logit(top incorrect)
  - Train: Dense MLP probes + Sparse L1-logistic probes on hidden states
  - Identify predictive neurons (success/failure neurons)
  - Causal ablation: Zero out identified neurons, measure accuracy impact
  - Output: Reliability prediction scores (SUCCEED: AUROC up to 0.971)
       |
       v
[Stage 3: Behavioral Metrics]
  - Sample K=10 reasoning paths (nucleus sampling p=0.9, T=0.7)
  - Compute Self-Consistency = agreement rate across samples
  - Output: Behavioral reliability score (SUCCEED: R = 0.429, AUROC 0.78-0.81)
       |
       v
[Final Prediction: Is this answer reliable?]
  - Best single-pass: Hidden-state probe (AUROC 0.95+)
  - Best behavioral: Self-Consistency (10x cost, R = 0.429)
  - Do NOT use: Attention map sharpness / cluster count (AUROC = 0.50)
```

---

## Pros & Cons

### Pros
- **Rigorous cross-family design:** Tests three architecturally diverse VLM families (prefix-based LLaVA, early-fusion PaliGemma, native-multimodal Qwen2-VL), strengthening generalizability claims
- **Multi-level analysis:** Moves from correlation (attention vs. correctness) to causation (neuron ablation) to mechanism (logit lens, sparse circuits), forming a complete scientific narrative
- **Practical implications clear:** Hidden-state probes offer single-pass reliability at near-zero overhead, directly actionable for deployment
- **Negative results well-documented:** "Cluster Failure" and attention failures are statistically validated with confidence intervals and supervised stress tests
- **Honest positioning:** Authors explicitly state that attention-failure and self-consistency are prior findings; novelty is in the unified cross-family reliability study and hidden-state probes

### Cons
- **Mid-scale models only (7B, 3B, 7B):** Larger models (LLaVA-34B, GPT-4V) may exhibit different attention-reliability relationships due to better RLHF
- **Qwen2-VL's low accuracy (28.8% on POPE) confounds probe AUROC:** High probe AUROC (0.971) on a model that is mostly wrong raises questions about what the probe is actually detecting
- **Effect size of neuron ablation is modest (-8.3pp on Object ID):** The causal evidence shows correlation but limited mechanistic control; neurons are "contributors" not "truth units"
- **Limited benchmark diversity in main analysis:** POPE and LLaVA-Bench dominate; VQA v2 and TextVQA results are more mixed
- **Self-Consistency requires 10x inference:** Impractical for real-time applications; distillation proposed but not implemented
- **No investigation of calibration techniques:** Finetuning or RLHF could potentially align attention with reliability
- **Probe requires labeled correctness data for training:** Not zero-shot; needs in-distribution calibration per model family

---

## Q&A Record

> **Q1:** If attention is causally necessary (masking top 30% attended patches drops accuracy by 8-11pp), why is it not correlated with correctness?
> **A:** The paper draws a key distinction: attention enables *feature extraction* (causally necessary for task performance) but does not encode *uncertainty about those features* (not correlated with correctness). Think of it as: attention is the "where to look" mechanism, but knowing "where you looked" does not tell you "whether you interpreted what you saw correctly." The Symbolic Detachment phenomenon (Early Locking + Late Diffusion) means that early attention patterns become stale by the time the LLM decoder makes its decision.

> **Q2:** Why does Qwen2-VL have such low POPE accuracy (28.8%) but the highest probe AUROC (0.971)?
> **A:** This is a potential confounding factor the paper does not fully address. When a model is wrong most of the time, a probe that learns to detect "when the model goes against its own bias" might achieve artificially high AUROC simply by detecting rare correct answers. The authors note this as a limitation (model scale / architecture-specific effects), but the high AUROC should be interpreted with caution given the low base accuracy.

> **Q3:** How is "self-consistency" different from simple majority voting?
> **A:** Self-consistency (SC) here is the agreement rate across K=10 sampled reasoning paths using nucleus sampling (p=0.9, T=0.7). It is essentially majority voting with temperature-based diversity. The key insight is that when all 10 diverse samples agree, the answer is highly reliable (90.8% precision at SC=1).

> **Q4:** Why does PaliGemma have lower probe AUROC (0.738) compared to LLaVA/Qwen2-VL?
> **A:** PaliGemma integrates visual evidence earlier (peak at L14) and has a shallower decoder (18 layers), leaving less late-layer separation between correct and hallucinated trajectories. This weakens probe margin contrast. LLaVA delays integration to L24-31, creating a large separation gap that probes can exploit.

> **Q5:** Can we use hidden-state probes in a zero-shot manner across different models?
> **A:** No. The paper recommends architecturally adaptive probing (different layer selection and probe capacity per family). Cross-family generalization of probes is not tested.

> **Q6:** What is the Counting Anomaly and why is it significant?
> **A:** The Counting Anomaly is a case where the visual encoder correctly identifies 3 distinct clusters in an image, but the LLM decoder outputs "Four" with 92% confidence. This vividly illustrates Symbolic Detachment: the visual system works correctly, but the linguistic projection fails. Token probability reflects fluency, not grounding.

---

## Citation Landscape

Connected Papers: https://www.connectedpapers.com/main/2606.17389

Key related works cited:
- **CLIP** (Radford et al., 2021): Foundation for vision-language alignment
- **LLaVA** (Liu et al., 2023): Visual instruction tuning, prefix-based VLM architecture
- **PaliGemma** (Beyer et al., 2024): Google's 3B vision-language model with SigLIP encoder
- **Qwen2-VL** (Wang et al., 2024): Alibaba's native multimodal architecture
- **Self-Consistency** (Wang et al., 2022): Agreement across sampled reasoning paths
- **Logit Lens** (Nostalgebraist, 2020): Hidden state projection to vocabulary
- **"Attention is not Explanation"** (Jain & Wallace, 2019): NLP interpretability debate
- **POPE** (Li et al., 2023b): Object hallucination evaluation benchmark
- **VIP/TVI** (Long et al., 2025): Visual Integration Point for LVLMs, measuring representational shift from images
- **"See but not believe"** (Liu et al., 2025): Correct localization without correct reasoning in VLMs
- **MME, SEED-Bench, MM-Vet:** Broader multimodal evaluation suites
