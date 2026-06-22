# Segment Anything in Pathology Images with Natural Language (SAMPath)

> **Authors**: Zhixuan Chen, Junlin Hou, Liqi Lin, Yihui Wang, Yequan Bie, Xi Wang, Yanning Zhou, Ronald Cheong Kin Chan, Hao Chen
> **Affiliation**: HKUST, USTC, CUHK, Tencent
> **Venue**: arXiv 2025 (2506.20988)
> **Code**: https://anonymous.4open.science/r/PathSegmentor-3166

---

## One-Sentence Summary

PathSegmentor is the **first text-prompted segmentation foundation model for pathology images**, built on a new 275k-sample PathSeg dataset spanning 160 hierarchical categories (20 anatomical regions x 3 histological structures x 61 object types), enabling semantic segmentation via natural language descriptions without spatial prompts.

---

## Core Contributions

1. **PathSeg Dataset**: The largest and most comprehensive pathology semantic segmentation benchmark -- 275k image-mask-label triples from 21 public datasets, with a three-level hierarchical label taxonomy (anatomical region, histological structure, object type) resolving semantic ambiguity in pathology labeling.

2. **PathSegmentor Model**: A Transformer encoder-decoder foundation model with a joint feature interaction module (cross-attention + self-attention) that fuses visual features (FocalNet) and textual features (PubMedBERT) through learnable queries to predict semantic masks from text prompts alone.

3. **Comprehensive Empirical Validation**: PathSegmentor outperforms specialized models (nnU-Net, DeepLabV3+, SAM-Path), spatial-prompted foundation models (MedSAM, SAM-Med2D), and text-prompted models (BiomedParse) on both internal (16 datasets) and external (5 datasets) evaluations.

4. **Intricate Object Robustness**: Demonstrates superior segmentation of irregular shapes, small instances, and high-density objects compared to spatial-prompted models, while requiring only 1 text prompt vs. ~15 spatial prompts per mask.

5. **Explainable Cancer Diagnosis Pipeline**: Bidirectional integration of segmentation and classification for object-based feature importance estimation and imaging biomarker discovery, enabling interpretable diagnostic decision support.

---

## Section Navigation

| Section | File | Core Content |
|---------|------|-------------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | Problem motivation, PathSeg + PathSegmentor overview |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | Background, limitations of existing methods, contributions |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | SAM family, medical segmentation foundation models, pathology segmentation |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | PathSeg dataset construction, PathSegmentor architecture, explainable diagnosis pipeline |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | Internal/external validation, intricate object analysis, qualitative results, explainability |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | Discussion of key advantages, limitations, and future work |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| PathSeg dataset size | 275k image-mask-label triples |
| Source datasets | 21 public datasets (16 internal + 5 external) |
| Hierarchical categories | 160 (20 AR x 3 HS x 61 OT) |
| Anatomical regions | 20 |
| Histological structures | 3 (tissue, cell, nuclei) |
| Object types | 61 |
| PathSegmentor model size | 0.45B parameters |
| Specialized model group size (16 x SAM-Path) | 1.86B parameters (75% reduction) |
| Overall Dice (PathSegmentor) | 0.671 |
| Overall Dice (nnU-Net, best specialized) | 0.502 |
| Improvement over MedSAM | +0.145 Dice |
| Improvement over BiomedParse | +0.429 Dice |
| Prompt efficiency (text vs. instance boxes) | 1 prompt vs. ~15 prompts per mask |
| Max instances per mask in PathSeg | >800 |
| Classification AUC (standard model) | 0.936 (macro AUC) |
| Classification AUC (object-aware model) | 0.953 (macro AUC) |

---

## Data Flow: Input -> Intermediate -> Output

```
Input: Pathology Image (1024x1024, H&E stained)
  +
Input: Text Prompt ("tissue-level tumor in breast pathology")
  │
  ▼
Image Encoder (FocalNet) ──> F_image ∈ R^{m×d} (visual features)
  │
  ▼
Text Encoder (PubMedBERT) ──> F_text ∈ R^{L×d} (semantic features)
  │
  ▼
Joint Feature Interaction Module:
  ├── Learnable Queries q ∈ R^{n×d}
  ├── Cross-Attention(q, F_image) ──> q' (vision-enhanced queries)
  ├── Self-Attention([q' || F_text]) ──> F_joint
  ├── Feed-Forward Network ──> q'' (semantic-aware queries)
  ├── Mask Projector ──> E_mask (candidate masks via dot-product)
  └── Class Projector ──> E_cls (category embeddings)
  │
  ▼
Mask Selection: argmax cosine similarity(E_cls^i, F_text')
  │
  ▼
Output: Binary Segmentation Mask ŷ ∈ {0,1}^{H×W}
```

For the explainable diagnosis extension:

```
Input: Whole Slide Image (WSI)
  ├── [Classification→Segmentation Pipeline]
  │   Patch features → Slide aggregation → Classifier → Prediction
  │                                          ↓
  │   PathSegmentor masks → Object-based perturbation → Feature importance
  │
  └── [Segmentation→Classification Pipeline]
      Patch features × PathSegmentor object masks → Object-aware features
      → Per-object aggregation → Unified slide feature → Classifier
      → Object-aware CAM (Class Activation Map) with semantic labels
```

---

## Pros & Cons

### Pros

- **Unified architecture**: One model replaces 16 specialized models while achieving better overall Dice (0.671 vs. 0.502 for nnU-Net group).
- **Semantic awareness**: Text prompts encode hierarchical pathology knowledge (anatomical region + histological structure + object type), resolving ambiguity in segmentation tasks.
- **Highly efficient prompting**: Single text prompt replaces ~15 spatial prompts per mask, crucial for clinical workflows (annotating millions of objects in WSIs is prohibitive).
- **Robustness to intricate objects**: Stable performance on irregular shapes, tiny instances, and high-density regions where spatial-prompted models fail significantly.
- **Explainability integration**: Bidirectional coupling with classification models provides both feature importance estimation and object-aware CAMs, directly supporting clinical decision-making.

### Cons

- **Dataset scale still limited**: 275k samples vs. millions in non-semantic datasets; semantic annotation in pathology is resource-intensive.
- **Novel categories performance**: Text-prompted models may degrade on semantically unseen object types; spatial prompts provide complementary localization for such cases.
- **Endothelial cell weakness**: Both PathSegmentor and BiomedParse underperform spatial-prompted models on sparse, clustered cells (e.g., endothelial cells in CoNSeP) -- text lacks precise spatial cues for few-and-concentrated instances.
- **No multi-prompt support**: Current architecture uses text-only prompting; combining text + spatial prompts could improve robustness for unseen categories.
- **Real-world clinical validation pending**: No multi-center clinical trials reported; pathologist feedback integration is planned but not yet executed.
- **Initialized from BiomedParse**: Inherits weights from a general medical model; full pathology-specific pretraining from scratch may yield further gains.

---

## Q&A Record

> **Q1: Why not just use BiomedParse directly for pathology? What does PathSegmentor add?**
>
> BiomedParse is trained on multimodal biomedical data with ~15k pathology samples (vs. PathSeg's 275k). Its text template `[object type] in [anatomical region] pathology` omits histological structure information (tissue/cell/nuclei), which is critical for resolving the semantic ambiguity in pathology (e.g., "tumor" can mean tumor tissue or tumor cells). PathSegmentor's three-level hierarchy `[histological structure]-level [object type] in [anatomical region] pathology` explicitly encodes this multi-scale context.

> **Q2: Why does PathSegmentor outperform spatial-prompted models on high-density objects?**
>
> Spatial-prompted models using a single union box lack sufficient spatial information to resolve individual instances in crowded regions -- the box simply encompasses everything. Instance boxes would provide more precision but require ~15x more prompts per mask. Text prompts implicitly encode category-level shape and distribution priors learned from large-scale training, allowing PathSegmentor to segment dense instances without explicit spatial localization.

> **Q3: What is the role of learnable queries in the joint feature interaction module?**
>
> Learnable queries act as adaptive filters that first aggregate critical visual context via cross-attention with image features, then fuse with textual semantics via self-attention. The resulting semantic-aware queries encode both spatial localization (where) and semantic category (what), enabling mask and class embedding generation. This is similar to the query-based detection paradigm (DETR) but adapted for text-prompted segmentation.

> **Q4: Is the three-level hierarchy really necessary, or is it just label engineering?**
>
> The paper provides evidence that it matters: BiomedParse's simpler template (omitting histological structure) achieves only 0.242 Dice on PathSeg vs. PathSegmentor's 0.671. The histological structure level (tissue/cell/nuclei) is critical because the same object type (e.g., "tumor") at different scales requires fundamentally different segmentation strategies -- tissue-level tumors are large irregular regions, while nuclei-level tumors are tiny dense objects. The hierarchy encodes this prior explicitly.

> **Q5: What happens on categories not seen during training?**
>
> The paper acknowledges this limitation. For external evaluation, they only tested categories where the [object type] aligned with training categories. For truly novel semantic categories, text-prompted performance would likely degrade. The authors propose combining text + spatial prompts in future work to address this -- spatial prompts provide localization, text provides semantics for known categories.

> **Q6: How does object-aware CAM differ from vanilla CAM?**
>
> Vanilla CAM highlights discriminative image regions but cannot specify what pathological objects those regions contain. Object-aware CAM decomposes the classification signal into per-object contributions, associating each highlighted region with a specific pathological category (e.g., "breast-tissue-tumor" rather than just "warm region"). This bridges the semantic gap between saliency maps and clinical interpretation.

---

## Citation Landscape

The paper sits at the intersection of three research threads:
- **Pathology image analysis** (nnU-Net, HoVer-Net, SAM-Path, SegAnyPath)
- **Segmentation foundation models** (SAM, MedSAM, SAM-Med2D, BiomedParse, SEEM)
- **Explainable AI in medicine** (CAM, RISE, feature importance estimation)

Connected Papers: [https://www.connectedpapers.com/main/2506.20988](https://www.connectedpapers.com/main/2506.20988)

Key related works cited:
- SAM [13]: Original prompt-driven segmentation paradigm
- SEEM [16]: Multi-prompt (point, box, text, scribble) unified framework -- architectural backbone for PathSegmentor
- MedSAM [19]: SAM adapted for medical imaging via 1.5M image-mask pairs
- SAM-Med2D [20]: Extended medical SAM with 19.7M masks
- BiomedParse [21]: Text-prompted biomedical segmentation across 9 modalities
- SAM-Path [10]: Pathology-specific SAM adaptation with class prompts (specialized, not foundation)
- SegAnyPath [31]: Spatial-prompted pathology foundation model (no semantic awareness)
- nnU-Net [9]: Self-configuring medical segmentation (specialized per-dataset baseline)

---

*Last updated: 2025-06-22*
