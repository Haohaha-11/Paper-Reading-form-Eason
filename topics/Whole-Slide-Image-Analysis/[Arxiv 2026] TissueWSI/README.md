# Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning

## Metadata

| Field | Value |
|---|---|
| **Authors** | Wentao Huang, Weimin Lyu, Peiliang Lou, Qingqiao Hu, Xiaoling Hu, Shahira Abousamra, Wenchao Han, Ruifeng Guo, Jiawei Zhou, Chao Chen, Chen Wang |
| **Affiliations** | Stony Brook University; Mayo Clinic; Harvard Medical School; Stanford University |
| **Venue** | arXiv 2026 (arXiv:2603.00667) |
| **Code** | https://github.com/winston52/HistoSelect |
| **Paper PDF** | [paper.pdf](./paper.pdf) |

## One-Sentence Summary

HistoSelect is a question-guided, tissue-aware, coarse-to-fine patch selection framework for pathology VQA that reduces visual token usage by **70%** while improving accuracy across three benchmarks, by mimicking how pathologists first identify relevant tissue regions and then zoom into critical patches for diagnosis.

## Contributions

1. **Pathologist-Collaborative Tissue Prompts**: Collaborated with clinical pathologists to design a set of fundamental tissue type prompts, enabling automatic semantic partitioning of WSIs into distinct tissue regions (e.g., tumor, stroma, lymphocyte).

2. **Hierarchical Question-Guided Selection (HistoSelect)**: Introduced a two-stage, Information Bottleneck (IB)-theoretic selection framework consisting of a group sampler (coarse tissue-level) and a patch selector (fine-grained patch-level), which prunes question-irrelevant visual tokens and increases the signal-to-noise ratio for the downstream LLM.

3. **Clinical Pathologist Evaluation**: Conducted rigorous human evaluation with two independent pathologists on both tissue segmentation accuracy and patch selection relevance, using a custom interactive survey tool, achieving average ratings above 3.5/5.0 across all evaluation dimensions.

4. **State-of-the-Art Performance**: Achieved the best overall results on SlideBench-VQA (83.80% avg accuracy), WSI-Bench (best on 5/6 open-ended metrics and all close-ended categories), and an in-house Ovarian dataset (73.33% accuracy).

## Section Navigation

| Section | File | Description |
|---|---|---|
| Abstract | [00-abstract.md](sections/00-abstract.md) | Problem, method overview, and results summary |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | Motivation, pre-analysis, and contributions |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | WSI analysis, multimodal pathology models, IB in pathology |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | Tissue segmentation, group sampler, patch selector, HIB objective |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | Quantitative, qualitative, ablation, and supplementary results |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | Summary and future directions |

## Key Numbers

| Metric | Value |
|---|---|
| **Total QA pairs evaluated** | 356,000 |
| **Token reduction** | 70% (5k optimal from ~17k) |
| **Close-ended avg accuracy** | 83.80% (vs. SlideChat 80.88%) |
| **Open-ended BLEU-4** | 0.221 (vs. Quilt-LLaVA 0.216) |
| **Open-ended ROUGE-L** | 0.463 (vs. Quilt-LLaVA 0.455) |
| **Datasets** | SlideBench-VQA (4,560 WSIs, 176K QA), WSI-Bench (9,850 WSIs, 180K QA), In-house Ovarian (375 WSIs, 375 QA) |
| **Base LLM** | Qwen2.5-7B-Instruct |
| **Vision Encoder** | CONCH |
| **Pathologist rating (tissue seg.)** | 4.17 / 3.67 (P1 / P2, 5-point Likert) |
| **Pathologist rating (patch relevance)** | 4.80 / 3.87 (P1 / P2) |
| **Optimal token budget** | 5,000 patches (peak accuracy; 10k degrades performance) |
| **Hyperparameters** | βg = 0.2, βp = 0.1 (warmup from 0 over 5k iterations) |

## Data Flow: Input → Intermediate → Output

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT                                                                 │
│  • WSI (gigapixel) → N non-overlapping patches (224×224)             │
│  • Question Q (natural language, e.g., "What is the tumor subtype?")  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: TISSUE SEGMENTATION (coarse, question-agnostic)             │
│  • CONCH visual encoder → patch features X = {x₁, ..., x_N}          │
│  • CONCH text encoder → M tissue prompt features T = {t₁, ..., t_M}  │
│  • Cosine similarity → tissue label lᵢ = argmax(xᵢ·tⱼ / |xᵢ||tⱼ|) │
│  • Output: M tissue groups (tumor, stroma, lymphocyte, etc.)          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: HIERARCHICAL SELECTOR (coarse→fine, question-guided)        │
│                                                                       │
│  Group Sampler:                                                       │
│    • Compute group prototype gⱼ = avg({xᵢ | i ∈ Tⱼ})                 │
│    • Predict sampling rate rⱼ = σ(F_group([gⱼ; q]))                   │
│    • Determine tokens per group: kⱼ = ⌈rⱼ · Nⱼ⌉                      │
│                                                                       │
│  Patch Selector:                                                      │
│    • Predict selection prob sᵢ = σ(F_patch([xᵢ; q]))                  │
│    • Within each group j, rank by sᵢ, select top-kⱼ patches Zⱼ       │
│    • Final selected set: Z = ∪ⱼ Zⱼ                                    │
│                                                                       │
│  Training Objective: L_final = L_VQA + βg·L_group + βp·L_patch       │
│    • L_VQA: negative log-likelihood of answer sequence                │
│    • L_group: KL(B(rⱼ) || B(pⱼ^g)), pⱼ^g = cos(gⱼ, q)               │
│    • L_patch: KL(B(sᵢ) || B(pᵢ^p)), pᵢ^p = cos(xᵢ, q)               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                                │
│  • Compact, question-aligned visual tokens Z (5k patches, ~30%)      │
│  • Interpretable: selected patches highlight relevant tissue regions  │
│  • Generated answer: Y = LLM(Z, Q)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Pros and Cons

### Pros

- **Efficiency**: 70% token reduction while maintaining or improving accuracy; peak performance at 5k tokens with no gain from 10k, confirming that WSIs are highly redundant for specific questions.
- **Interpretability**: The selection process is transparent — selected patches can be visualized and verified by pathologists, providing attributable evidence for predictions.
- **Clinically Validated**: Pathologist evaluation confirms both tissue segmentation accuracy and patch selection relevance (avg ratings > 3.5/5.0 on all dimensions).
- **Theoretically Grounded**: The hierarchical IB formulation provides a principled framework for balancing compression and relevance at both group and patch levels.
- **Model-Agnostic**: The selector module improves performance even when paired with a different base model (Gemini 3 Flash), demonstrating its independent utility.
- **Adaptive Sampling Patterns**: t-SNE analysis reveals four distinct question-driven sampling clusters (Tumor Classification, Cellular Morphology, Tissue Architecture, Tumor Infiltration), confirming the model dynamically adjusts sampling based on question semantics.

### Cons

- **Limited Dataset Diversity**: Evaluation is primarily on TCGA and one private ovarian dataset; generalizability to other cancer types, organs, and scanning protocols remains to be validated (authors acknowledge this and mention BCNB as future work).
- **No Explicit Textual Reasoning**: While the model provides visual interpretability (selected patches), it does not generate natural language explanations for why specific patches were selected — a gap between visual attention and semantic reasoning.
- **Tissue Type Granularity**: Relies on a fixed set of M pathologist-defined tissue prompts; may not capture rare or dataset-specific tissue types not included in the prompt set.
- **Two-Stage Training**: Requires a two-stage training process with careful hyperparameter tuning (warmup schedule for βg and βp), which adds complexity to deployment.
- **Slide-Level Only**: The method is designed for slide-level VQA; performance on patch-level or region-level tasks is not explored.

## Q&A Record

> 💡 **Q1**: Why is question-guided selection necessary for WSI VQA? Why not just use all patches?
>
> **A1**: A single WSI contains tens of thousands of patches, most of which are background or unrelated to a specific clinical question. Feeding all patches indiscriminately overwhelms the LLM with irrelevant information and hits token limits. The authors' pre-analysis (Figure 2d) shows that question-guided (cosine similarity) sampling dramatically outperforms diversity-based and random sampling in retrieving tumor-relevant patches. The IB framework further improves upon simple similarity by learning a task-optimized selection policy.

> 💡 **Q2**: What is the relationship between the group sampler and patch selector? Could one work without the other?
>
> **A2**: The ablation in Table 5 answers this directly. Removing the patch selector (relying only on coarse group selection) drops performance (e.g., Diagnosis: 85.79→81.32). Removing the group sampler (forcing the patch selector to search globally) also drops performance (Diagnosis: 85.79→81.82). The two work synergistically: the group sampler narrows the search space, and the patch selector performs fine-grained selection within that narrowed space.

> 💡 **Q3**: How does the IB objective differ from standard VIB in this hierarchical setting?
>
> **A3**: Standard VIB uses a single β for the compression term. Here, the compression is decomposed hierarchically: I(Z; X | q) = I(Zg; X | q) + I(Zp; X | Zg, q). This allows independent Lagrange multipliers βg and βp to regulate information flow at group and patch granularities separately. The pseudo-priors pg_j and pp_i are derived from question-patch cosine similarity, providing a semantically grounded prior for the selection.

> 💡 **Q4**: Why does increasing token budget from 5k to 10k slightly degrade performance?
>
> **A4**: This counterintuitive result (Table 6) validates the core thesis: a large portion of WSI patches are redundant for a specific question. Adding more tokens beyond the optimal 5k introduces redundant information that acts as noise for the LLM, potentially distracting it from the truly informative patches. This is consistent with the IB principle — more information about the input X can harm the model if it does not carry additional information about the target Y.

> 💡 **Q5**: How does the differentiable hard selection work, since patch selection is a discrete operation?
>
> **A5**: The Straight-Through Estimator (STE) is used. During the forward pass, a hard binary mask selects the top-kⱼ patches per group (discrete). During the backward pass, gradients flow through the soft probabilities (rⱼ and sᵢ), bypassing the discrete step. This allows end-to-end gradient-based optimization of the entire pipeline including both selectors.

> 💡 **Q6**: How is the tissue segmentation different from standard tissue classification?
>
> **A6**: Instead of training a classifier, the authors use zero-shot CLIP-style matching with M pathologist-designed text prompts (e.g., "tumor region", "stromal region", "lymphocyte region"). Each patch is assigned to the tissue type with highest cosine similarity between patch embedding and prompt embedding. This is question-agnostic and serves as the structural basis for subsequent question-guided selection.

> 💡 **Q7**: Are the βg and βp hyperparameters sensitive?
>
> **A7**: The supplementary ablation (Tables 9-10) shows moderate sensitivity. βg=0.2 and βp=0.1 are optimal. Setting βg=0 (no group regularization) loses ~2.8 points on Diagnosis. Setting βg=0.3 (overly aggressive) loses ~2.5 points. The authors use a linear warmup schedule for the first 5k iterations to stabilize training. The optimal values suggest that patch-level compression needs to be less aggressive (βp=0.1) than group-level (βg=0.2), which makes intuitive sense — you want to be more selective about which tissue groups to attend to, but within a relevant group you want broader patch coverage.

## Citation Landscape

- **Connected Papers**: https://www.connectedpapers.com/main/2603.00667
- **Key References**: CONCH [26], SlideChat [9], WSI-LLaVA [22], VIB [2, 41], CLAM [25], Quilt-LLaVA [32], LLaVA-Med [20]
- **Venue Context**: This paper sits at the intersection of computational pathology, vision-language models, and information theory. It builds on foundational WSI analysis (MIL-based classification [17, 19, 31]) and extends recent slide-level VQA methods (SlideChat [9], WSI-LLaVA [22]) by introducing hierarchical question-guided selection grounded in IB theory.

---

[← 返回目录](../)
