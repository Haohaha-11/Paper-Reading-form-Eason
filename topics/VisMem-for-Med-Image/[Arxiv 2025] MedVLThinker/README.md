# MedVLThinker: Simple Baselines for Multimodal Medical Reasoning

## Paper Metadata

| Field | Value |
|-------|-------|
| **Title** | MedVLThinker: Simple Baselines for Multimodal Medical Reasoning |
| **Authors** | Xiaoke Huang (UC Santa Cruz), Juncheng Wu (UC Santa Cruz), Hui Liu (Amazon Research), Xianfeng Tang (Amazon Research), Yuyin Zhou (UC Santa Cruz) |
| **Venue** | arXiv 2025 (also accepted at ML4H 2025) |
| **ArXiv ID** | 2508.02669 |
| **Link** | https://arxiv.org/abs/2508.02669 |
| **Code & Data** | https://github.com/UCSC-VLAA/MedVLThinker |
| **Citations** | 17 (2 influential), 49 references |

## One-Sentence Summary

MedVLThinker provides the first fully open-source recipe for building multimodal medical reasoning models, demonstrating that RLVR on text-only data outperforms SFT on multimodal data and that a 32B open model can match proprietary GPT-4o performance.

## Core Contributions

1. **First fully open-source recipe** for medical vision-language reasoning models, including data curation, training code, model weights, and evaluation pipelines.
2. **Systematic comparison of SFT vs. RLVR**: RLVR (GRPO-based) consistently and significantly outperforms supervised fine-tuning on distilled CoT traces across all model scales and benchmarks.
3. **Counter-intuitive finding on data modality**: Text-only training data (m23k) yields better results than image-text data (PMC-VQA), highlighting a critical data quality gap in current multimodal medical corpora.
4. **State-of-the-art open-source performance**: MedVLThinker-7B achieves 54.88% average accuracy across six benchmarks, surpassing all prior open-source medical LMMs; MedVLThinker-32B (63.12%) matches GPT-4o (63.74%).
5. **Difficulty-based data filtering**: A novel curriculum-aware filtering strategy using pass-count statistics from a 3B model to select medium-difficulty questions for optimal training.

## Section Navigation

| Section | File | Description |
|---------|------|-------------|
| Abstract & Overview | [sections/00-abstract.md](sections/00-abstract.md) | Abstract, Figure 1 (performance overview), Figure 2 (pipeline) |
| Introduction | [sections/01-introduction.md](sections/01-introduction.md) | Motivation, problem statement, contributions overview |
| Related Work | [sections/02-related-work.md](sections/02-related-work.md) | LRMs & medical adaptation (2.1), Multimodal medical LMMs (2.2), Concurrent works (2.3) |
| Methods | [sections/03-methods.md](sections/03-methods.md) | Data curation & filtering, SFT (3.1), RLVR/GRPO (3.1) |
| Experiments | [sections/04-experiments.md](sections/04-experiments.md) | Implementation (4.1), Evaluation (4.2), Results (4.3) with Tables 1-2 |
| Conclusion | [sections/05-conclusion.md](sections/05-conclusion.md) | Summary, limitations, future work |

## Key Numbers

| Metric | Value |
|--------|-------|
| **Base Models** | Qwen2.5-VL (3B, 7B, 32B) |
| **Text-only training data** | m23k: 23,493 questions (filtered to 16,512) |
| **Image-text training data** | PMC-VQA: 176,948 QA pairs (filtered to 115,456) |
| **Filtering criteria** | Pass count = 0 or >= 7 filtered out (3B model, 16 trials) |
| **SFT epochs** | 3 epochs, lr=1e-4, batch=32 |
| **RLVR epochs** | 5 epochs (text), 1 epoch (image-text), lr=1e-6, batch=128/256 |
| **RLVR config** | GRPO, N=8 rollouts, binary reward (+1/-1) |
| **GPUs** | 8x H100 (3B/7B), 32x H100 (32B) |
| **Benchmarks** | 6: PMC-VQA, MMMU-Health, MedXpert-MM, PathVQA, SLAKE, VQA-Rad |

### Performance Summary

| Model | Avg Accuracy | vs Base |
|-------|-------------|---------|
| Qwen2.5-VL-3B-Instruct (base) | 49.14% | -- |
| MedVLThinker-3B RL(text) | 53.19% | +4.05% |
| Qwen2.5-VL-7B-Instruct (base) | 53.50% | -- |
| MedVLThinker-7B RL(text) | 54.88% | +1.38% |
| Qwen2.5-VL-32B-Instruct (base) | 60.20% | -- |
| MedVLThinker-32B RL(text) | **63.12%** | +2.92% |
| GPT-4o (closed-source) | 63.74% | reference |
| GPT-4o-mini (closed-source) | 58.24% | reference |

### Key Counter-Results

| Training Recipe | 3B Avg | 7B Avg | Note |
|-----------------|--------|--------|------|
| SFT(text) | 32.80% | 43.83% | **Worse than base!** (-16.34% / -9.67%) |
| SFT(image-text) | 50.16% | 50.72% | Similar to base |
| RL(text) | **53.19%** | **54.88%** | Best overall |
| RL(image-text) | 52.28% | 53.66% | Improves, but less than text |
| RL(text)+RL(image) | 49.72% | 49.77% | Combining hurts performance |

## Data Flow: Input to Intermediate to Output

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA CURATION STAGE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│    ├── m23k (23,493 text-only medical QA, from MedQA/        │
│    │        MedMCQA/HeadQA training splits)                   │
│    └── PMC-VQA (176,948 image-text QA pairs, from PubMed)    │
│                                                              │
│  Intermediate: Difficulty Probing                            │
│    ├── Qwen2.5-VL-Instruct (3B/7B/32B) answers each          │
│    │   question 16 times (nucleus sampling, T=1.0)           │
│    ├── Pass count recorded for each question                 │
│    └── Filter: remove pass=0 or pass>=7 (using 3B results)   │
│                                                              │
│  Intermediate: CoT Generation                                │
│    ├── Text-only: DeepSeek-R1 generates reasoning chains     │
│    └── Image-text: GPT-4o generates reasoning chains         │
│                                                              │
│  Output:                                                     │
│    ├── Filtered m23k: 16,512 questions (with CoTs)           │
│    └── Filtered PMC-VQA: 115,456 questions (with CoTs)       │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     TRAINING STAGE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Base Model: Qwen2.5-VL-Instruct (3B / 7B / 32B)            │
│                                                              │
│  Path A: SFT (Supervised Fine-Tuning)                        │
│    ├── Input: (question, image?) → teacher CoT + answer      │
│    ├── Loss: token-level cross-entropy on CoT + answer       │
│    ├── Epochs: 3, lr=1e-4, batch=32                          │
│    └── Output: SFT model                                     │
│                                                              │
│  Path B: RLVR (Reinforcement Learning with Verifiable        │
│          Rewards) via GRPO                                    │
│    ├── Input: (question, image?) → model generates 8 traces   │
│    ├── Verifier: check format (<think>...</think>            │
│    │   <answer>...</answer>) + exact answer match             │
│    ├── Reward: +1 (correct) or -1 (incorrect)                │
│    ├── Advantage: whitened across group (N=8)                │
│    ├── Update: PPO-style clipped objective + KL penalty      │
│    ├── Epochs: 5 (text) / 1 (image-text), lr=1e-6            │
│    └── Output: RLVR model                                    │
│                                                              │
│  Training Variants Evaluated:                                │
│    ├── SFT(text), SFT(image-text)                            │
│    ├── RL(text), RL(image-text)                               │
│    ├── SFT(text) + RL(image-text)                             │
│    └── RL(text) + RL(image-text)                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION STAGE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Benchmarks (6 total):                                        │
│    General: PMC-VQA (test), MMMU-Health (val),               │
│             MedXpert-MM (challenging)                         │
│    Modality-specific: PathVQA (pathology), SLAKE             │
│                       (ophthalmology), VQA-Rad (radiology)    │
│                                                              │
│  Decoding: Greedy (T=0), 3 runs averaged                     │
│  Metric: Accuracy (%)                                        │
│                                                              │
│  Output: Table 1 (ablation), Table 2 (SOTA comparison)       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Pros / Cons & Future Work

### Strengths

1. **Fully open-source**: Code, data, models all released -- the first complete reproducible recipe for multimodal medical reasoning.
2. **Rigorous ablation**: Systematic comparison across training paradigms (SFT vs RLVR), data modalities (text vs image-text), model scales (3B/7B/32B), and training stages (single vs combined).
3. **Practical insight**: RLVR is simpler (no need for teacher CoTs) yet more effective than SFT for reasoning tasks, and text-only data can be sufficient.
4. **Strong empirical results**: SOTA among open-source medical LMMs, 32B matches GPT-4o.
5. **Difficulty-based filtering**: Novel curriculum-aware data selection strategy that is simple and effective.

### Weaknesses

1. **PMC-VQA data quality**: The image-text training data is GPT-3.5-generated and noisy, limiting multimodal training gains. This is acknowledged but not solved.
2. **Static difficulty filtering**: The same filter (based on 3B pass counts) applied to all model sizes; larger models may benefit from different thresholds.
3. **QA-only scope**: Training and evaluation limited to multiple-choice QA; no multi-turn dialogue, visual grounding, or temporal reasoning tasks.
4. **Short RLVR training**: Only 1-5 epochs of RL, which may not fully exploit RLVR's potential.
5. **SFT on text-only CoTs degrades performance**: The paper observes this but only hypothesizes about cause (mismatch between text-only rationales and multimodal model); no controlled experiment to isolate the root cause.
6. **No direct multimodal RLVR benefit**: Text-only + image-text combination underperforms text-only alone -- the paper does not resolve how to effectively leverage multimodal data.

### Future Work (from paper)

1. Improve multimodal training data quality (human-curated reports, better synthetic generation).
2. Dynamic/adaptive curriculum learning tailored per model scale.
3. Extend beyond QA to visual grounding, multi-turn dialogue, and temporal reasoning.
4. Scale up RLVR with larger verification datasets and more compute.
5. Extend RLVR to open-ended generation and multi-step clinical reasoning (non-MCQ reward shaping).

## Reading Q&A Record

### Q1: Why does SFT on text-only CoT data hurt performance so dramatically?

**A**: The paper hypothesizes (Section 4.3, "Impact of Training Data") that the long, text-only rationales distilled from DeepSeek-R1 (a text-based LRM) may not align well with the needs of a multimodal model that must also interpret images. The SFT step "overloads" the model with long, possibly mismatched rationales. However, this is only a hypothesis -- no controlled experiment isolates whether the issue is (a) CoT length, (b) domain mismatch (text vs vision reasoning), or (c) teacher-student capacity gap. This is a key open question for reproducibility.

### Q2: If RLVR on text-only data works best, why bother with multimodal data at all?

**A**: This is the paper's most striking finding. While RLVR on text-only data is sufficient for MCQ-style benchmarks, real clinical applications require visual understanding. The poor performance of multimodal training is attributed to data quality (PMC-VQA's GPT-3.5-generated questions are noisy). The paper argues this highlights the need for better multimodal data, not that multimodal data is inherently useless (see Appendix C Discussion).

### Q3: Is the difficulty filtering really necessary? What happens without it?

**A**: The paper does not include an ablation of the filtering strategy itself. This is a notable gap -- we don't know if filtering is essential or if similar results could be achieved with all data. The motivation comes from Muennighoff et al. (2025)'s curriculum learning insights, but direct evidence is missing.

### Q4: How does GRPO's group-based advantage computation differ from standard PPO?

**A**: GRPO (Shao et al. 2024) eliminates the critic/value network by computing advantages from a group of N sampled outputs per question. Rewards are whitened (normalized) across the group. The update uses a PPO-style clipped objective with KL regularization against the reference policy. This is described in Section 3.1 and Appendix A.

### Q5: Can MedVLThinker-32B really "match GPT-4o" given the benchmark suite?

**A**: The 32B model achieves 63.12% vs GPT-4o's 63.74% average across the six benchmarks -- a difference of only 0.62%. However, the benchmarks are all multiple-choice medical QA, which is a narrow evaluation. The paper acknowledges this limitation (QA-only scope) but the claim is quantitatively supported by Table 2.

### Q6: Why does sequential training (text then image-text) hurt performance?

**A**: Section 4.3 reports that adding image-text data after text-only optimization (via either SFT+RL or RL+RL) degrades performance. The paper speculates that the noisy image-text data may "hinder the reasoning capability" already acquired. This suggests catastrophic forgetting or interference between modalities, but no mechanistic analysis is provided.

## Citation Landscape

- **References**: 49 papers
- **Citations received**: 17 total, 2 influential
- **Semantic Scholar TLDR**: "This paper presents MedVLThinker, a suite of simple yet strong baselines for building reasoning-centric medical LMMs, and establishes a new state-of-the-art on existing public VQA benchmarks, surpassing all previous open-source medical LMMs."

### Key Related Works

| Paper | Relevance | Relationship |
|-------|-----------|-------------|
| HuatuoGPT-o1 (Chen et al. 2024a) | Text-only medical reasoning LLM | Uses PPO with external reward model; MedVLThinker extends to multimodal |
| MedVLM-R1 (Pan et al. 2025) | Multimodal medical RLVR | Small-scale (<1K samples), not general; MedVLThinker is larger-scale and general |
| Med-R1 (Lai et al. 2025) | Modality-specific RLVR | Trained on separate datasets per domain; MedVLThinker trains on general data |
| GMAI-VL-R1 (Su et al. 2025) | General multimodal medical RLVR | Closed-source; MedVLThinker is fully open |
| M1 (Huang et al. 2025) | Text-only CoT distillation | Distills GPT-4 reasoning into smaller model; MedVLThinker's text-only SFT follows similar idea |
| DeepSeek-R1 (Guo et al. 2025) | Foundation LRM | Used as CoT teacher for text-only data; GRPO algorithm origin |

**Connected Papers**: https://www.connectedpapers.com/main/2508.02669
