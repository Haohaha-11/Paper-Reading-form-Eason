# MedMO: Grounding and Understanding Multimodal Large Language Model for Medical Images

**Authors**: Ankan Deria\*, Komal Kumar\*, Adinath Madhavrao Dukre, Eran Segal, Salman Khan, Imran Razzak

**Affiliation**: Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)

**Venue**: arXiv 2026 (arXiv:2602.06965)

**Resources**: [Models](https://huggingface.co/collections/MBZUAI/medmo) | [GitHub](https://github.com/genmilab/MedMO) | [Project Page](https://genmilab.github.io/MedMO-Page)

---

## One-Sentence Summary

MedMO is a fully open-source medical multimodal foundation model built on Qwen3-VL, post-trained through a four-stage progressive pipeline (large-scale alignment, high-resolution fine-tuning, instruction tuning, and GRPO-based RL with bounding-box verifiable reward) on **26M+ samples from 45 multimodal medical datasets**, achieving SOTA performance across medical VQA, text QA, report generation, and grounding tasks while supporting spatial localization with bounding boxes.

---

## Contributions

1. **Open-source Medical Foundation Model**: Developed MedMO (4B and 8B variants), a post-trained multimodal VLM that unifies visual grounding, clinical reasoning, and language understanding across radiology, pathology, ophthalmology, dermatology, CT, MRI, ultrasound, and surgical videos.

2. **Large-scale Data Curation**: Assembled over **26M multimodal medical and biomedical samples** from **45 diverse open-source datasets**, spanning multiple imaging modalities and biological systems, coupled with a dedicated Cell detection benchmark from open-source microscopy images.

3. **Multi-stage Post-training Pipeline**: Designed a four-stage progressive training recipe: (i) General SFT on 18.5M pairs at 768x768, (ii) High-resolution SFT on 3M samples at 1280x1280 for grounding, (iii) Instruction tuning on 4.3M multimodal QA/reasoning pairs, (iv) GRPO-based RL with a novel **Bounding Box Verifiable Reward** that combines Hungarian-matched GIoU + normalized L1 with FP/FN penalties.

4. **State-of-the-Art Results**: MedMO-8B-Next surpasses Fleming-VL-8B by **+6.6% VQA average**, **+14.4% text QA average**, **+6.7% CIDEr on MIMIC-CXR**, and **+47.0 IoU on Bacteria grounding**. MedMO-4B-Next is competitive with 8B-scale baselines.

5. **Comprehensive Ablation Framework**: Provides open, reproducible stage-wise ablation studies and bounding-box reward analyses, establishing benchmarks and training recipes for future medical MLLM research.

---

## Section Navigation

| Section | File | Description |
|---------|------|-------------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | Paper abstract with batch-reading annotations |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | Motivation, gap analysis, and contributions |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | Medical MLLMs and grounding with multimodal models |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | Four-stage training pipeline, SFT, RL, bounding-box reward |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | Setup, datasets, SOTA results, ablation studies |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | Summary, limitations, and future work |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Base architecture | Qwen3-VL-8B-Instruct |
| Model variants | MedMO-4B, MedMO-4B-Next, MedMO-8B, MedMO-8B-Next |
| Total training data | **26M+ samples** from **45 datasets** |
| Training compute | 64x AMD Instinct MI210 (64 GB), **25 days** |
| Stage 1 (General SFT) | 18.5M pairs, 768x768, BS=10, LR=1e-5, **225 hours** |
| Stage 2 (High-res SFT) | 3M samples, 1280x1280, BS=2, LR=8e-6, **155 hours** |
| Stage 3 (Instruction Tuning) | 4.3M instruction pairs, BS=10, LR=5e-6, **110 hours** |
| Stage 4 (RL/DAPO) | 300K samples, 8 generations/prompt, **98 hours** |
| VQA Avg (MedMO-8B-Next) | **72.7%** (+6.6% vs Fleming-VL-8B) |
| QA Avg (MedMO-8B-Next) | **60.1%** (+14.4% vs Fleming-VL-8B) |
| MIMIC-CXR CIDEr (MedMO-8B-Next) | **143.4** |
| Grounding Avg IoU (MedMO-8B-Next) | **56.8%** |
| Bacteria IoU (MedMO-8B-Next) | **56.1** (+47.0 vs Fleming-VL-8B) |

---

## Data Flow: Input -> Intermediate -> Output

```
[Input] Multimodal Medical Images (X-ray, CT, MRI, ultrasound, pathology, OCT, fundus, surgical, etc.)
  + Text Queries (VQA questions, clinical prompts, report requests)
    │
    ▼
[Stage 1: General Medical SFT] 18.5M image-text pairs @ 768×768
  • Vision Encoder (ViT) → Vision-Language Adapter (DeepStack) → LLM Decoder
  • Tasks: captioning, VQA, general multimodal alignment
  • Output: Base model with foundational medical knowledge
    │
    ▼
[Stage 2: High-Resolution Medical Image + Grounding SFT] 3M samples @ 1280×1280
  • High-res expert-annotated image-text pairs
  • Introduces bounding-box prediction for spatial grounding
  • Tasks: captioning + VQA + supervised grounding signals
  • Output: Spatial-aware model with localization capability
    │
    ▼
[Stage 3: Instruction Tuning] 4.3M instruction-response pairs
  • Medical QA, reasoning, report summarization, retrieval
  • Aligns responses with human-style medical reasoning
  • Output: Clinically-aligned instruction-following model
    │
    ▼
[Stage 4: RL with Verifiable Rewards] GRPO/DAPO, 300K samples
  • 4 reward signals: label accuracy, bounding-box GIoU, tag count, soft-overlap penalty
  • Hungarian matching for box assignment
  • Bounding Box Reward = clip(base - penalty)
  • Output: MedMO (base) / MedMO-Next (with RL)
    │
    ▼
[Output] Text Responses + Bounding Box Coordinates
  • Medical VQA answers, diagnostic reports, spatial localization
  • Benchmark: VQA, Text QA, Report Generation (ROUGE-L, CIDEr, RaTE, Semb), Grounding (IoU)
```

---

## Pros and Cons

### Pros

- **Fully open-source**: All models, datasets, and training recipes are publicly released.
- **Comprehensive multi-modality coverage**: Handles radiology, pathology, ophthalmology, dermatology, CT, MRI, ultrasound, and surgical videos -- much broader than most medical MLLMs.
- **Built-in visual grounding**: Native bounding-box localization capability via the novel GRPO bounding-box reward, unlike most medical MLLMs that only do VQA/captioning.
- **Efficient scaling**: MedMO-4B-Next surpasses Fleming-VL-8B on many benchmarks, demonstrating strong performance even at smaller scales.
- **Transparent ablation**: Full stage-wise ablation studies show the contribution of each training phase.
- **Progressive curriculum**: The four-stage pipeline is designed as a scalable roadmap from general alignment to fine-grained spatial reasoning.

### Cons

- **Catastrophic forgetting**: Stage-wise training introduces minor task-level performance shifts (e.g., Stage 1 improves on MedTrinity but degrades slightly on other datasets).
- **IU-Xray not best**: On IU-Xray report generation, Fleming-VL-8B still leads (CIDEr 198.6 vs MedMO-8B-Next 171.9).
- **RL overhead**: The Next variants add 98 hours of RL training, and the gains from RL are modest on some benchmarks (e.g., Qwen3VL-8B already achieves strong QA scores without RL).
- **Limited ablation for RL reward components**: Only bounding-box reward ablation is shown; the relative contribution of tag count and soft-overlap penalty is not isolated.
- **English-only**: No explicit multilingual evaluation for non-English clinical settings.
- **25-day training**: Training requires substantial GPU resources (64x MI210), somewhat limiting accessibility for smaller labs.

---

## Q&A Record

> **Q1**: Why does MedMO use Qwen3-VL as the base instead of other VLMs?
>
> MedMO builds on Qwen3-VL-8B-Instruct. The authors do not explicitly ablate the base model choice, but Qwen3-VL provides native dynamic-resolution processing and a DeepStack vision-language fusion mechanism, which likely facilitate the multi-scale feature alignment needed for medical grounding tasks. The strong baseline performance of Qwen3VL-8B on Text QA (53.6% avg, close to MedMO-8B at 61.3%) also suggests it is a capable starting point.
>
> > **Q1 追问**: Could other strong VLMs like InternVL3 be an even better starting point?
> >
> > InternVL3-8B shows solid VQA performance (57.4% avg on VQA) but notably scores 0.00 IoU on both DeepLesion and near-zero (0.7) on Bacteria grounding (Table 3), suggesting its spatial localization capability is fundamentally broken for medical detection. Qwen3VL-8B achieves 16.4 IoU on NIH and 9.16 on Bacteria, indicating it already possesses emergent grounding capability before fine-tuning, making it a more suitable starting point for spatial grounding tasks.

> **Q2**: What is the difference between "MedMO" and "MedMO-Next"?
>
> The "Next" suffix denotes models that underwent Stage 4 (Reinforcement Learning with GRPO/DAPO and the bounding-box verifiable reward). The plain MedMO-4B/8B are the Stage 3 checkpoint. Interestingly, for some metrics (e.g., MedQA where MedMO-8B scores 84.3% vs Next's 83.8%, and MedMCQA where base scores 65.0% vs Next's 62.0%), the base variant actually performs better, suggesting that RL fine-tuning can degrade certain text QA capabilities while improving grounding.

> **Q3**: Why does the bounding-box reward use Hungarian matching instead of simpler greedy matching?
>
> The Hungarian algorithm finds the globally optimal one-to-one matching between predicted and ground-truth boxes, which is critical when multiple objects are present in an image (e.g., multiple lesions, multiple bacterial cells). Greedy matching can produce suboptimal pairings and lead to incorrect credit assignment during RL training. The cost matrix combines a weighted L1 distance (w=5) and GIoU (1 - GIoU, w=2), prioritizing spatial proximity over overlap initially, consistent with how detection evaluation metrics judge localization.

> **Q4**: Is MedMO suitable for deployment in real clinical settings?
>
> The paper does not report clinical validation or FDA/CE clearance. While MedMO achieves strong benchmarks, the authors acknowledge catastrophic forgetting as a limitation and note "minor task-level performance shifts." There is no discussion of bias/fairness evaluation, inference latency, or clinical workflow integration. MedMO is primarily positioned as a research foundation model for open scientific development, not a clinically certified diagnostic tool.

> **Q5**: How does the Cell Benchmark Dataset differ from existing microscopy benchmarks?
>
> MedMO introduces a Cell dataset constructed from open-source microscopy images (DeepCell, Bacteria), covering varying cell sizes, shapes, and densities. The key novelty is that it is designed specifically to evaluate VLM performance on detection tasks (IoU-based), bridging a gap in current medical VLM benchmarks which focus almost exclusively on VQA and text QA rather than spatial detection accuracy. This allows evaluating whether VLMs can correctly locate objects rather than just answer questions about images.

---

## Citation Landscape

[Connected Papers: MedMO](https://www.connectedpapers.com/main/2602.06965)

MedMO sits at the intersection of several active research directions:

1. **Medical MLLMs**: Builds on and substantially improves over LLaVA-Med, HuatuoGPT-Vision, Med-Flamingo, BioMedGPT, GMAI-VL, Lingshu, and Fleming-VL. The key differentiator is the combined VQA + QA + grounding + report generation capability.

2. **Reinforcement Learning for Reasoning**: Adopts GRPO (from DeepSeekMath/DeepSeek-R1) and DAPO for preference optimization with verifiable rewards, extending the RLVR paradigm from math/CS (e.g., RLVR-tuned code models) into medical vision-language tasks.

3. **Visual Grounding**: Extends detection-oriented grounding (Grounding-DINO) and general-domain VLM grounding (Qwen2.5-VL) into the medical domain, validated on MedSG-Bench for sequential, multi-view, and referring expression grounding.

4. **Multi-modal Medical Datasets**: Leverages MedTrinity-25M as the cornerstone (18.5M pairs) and supplements with 26 additional datasets spanning report generation, VQA, and text QA.

[← 返回 README](README.md)
