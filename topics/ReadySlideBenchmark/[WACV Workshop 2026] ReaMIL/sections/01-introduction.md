[← 返回 README](../README.md)

## 1. Introduction

> 💡 **引言路线（claude 批注）**: 作者先区分 bag-level 正确与 tile-level 证据可靠，再把 foundation encoder、MIL consumer 和 evidence selector 拆成三层，最后用 keep/drop 干预给“证据”一个可检验定义。

Whole-slide histopathology has become a standard testbed for weakly supervised learning [7]. Modern scanners produce gigapixel slides, but in most clinical datasets only slide-level labels are available: tumor subtype, grade, or outcome, without any pixel- or patch-level annotations [12]. Multiple instance learning (MIL) provides a natural framework for this setting, treating each slide as a bag of tiles that are encoded and aggregated into a single prediction [2, 10, 13]. Despite the weak supervision, these models can reach pathologist-level performance on some benchmarks and are now being deployed in early-stage clinical decision support tools.

However, standard MIL training focuses on bag-level accuracy: the model is rewarded for predicting the correct slide label, with no explicit notion of which tiles actually constitute the “evidence” for that prediction. Attention weights are often interpreted as explanations, but they are a side effect of training, not a primary objective [15, 16]. This gap between slide-level performance and tile-level reasoning becomes critical when models are meant to support clinical decisions. In practice, pathologists justify diagnoses by pointing to specific regions—glands with certain architecture, nests of atypical cells, or characteristic tumor– stroma interfaces. Computational models should ideally do the same: highlight a compact set of tiles sufficient to support the predicted label, while showing that the rest of the slide does not drive the decision.

> 💡 **问题动机（claude 批注）**: 这里直接限制了“首次发现强诊断模型 attention 不可靠”的表述。ReaMIL 的关键推进不是再画一张 attention 图，而是要求两个反事实同时成立：keep 后预测仍成立，drop 后预测不再成立。只满足前者可能只是冗余证据，只有两者同时满足才更接近必要且充分的 rationale。

Recent advances in representation learning have shifted the landscape toward foundation models pretrained on millions of tiles across sites and organs [4, 6]. We leverage pre-extracted UNI2-h [4] features as patch-level representations, allowing us to focus on the reasoning layer. On top of these frozen features, transformer-based MIL backbones such as TransMIL [17] already achieve competitive performance on multiple WSI benchmarks. Yet this “foundation MIL” stack does not address interpretability [3]: we have powerful encoders and backbones, but how they use evidence inside the bag remains opaque.

Our work treats evidence selection as a first-class objective in MIL rather than an afterthought. We attach a lightweight selection head on top of a strong MIL backbone to produce soft selection scores over tiles. These scores define three views of each slide: a full bag, a keep bag retaining only evidence tiles, and a drop bag containing the complement. By feeding these three bags through a shared backbone, we explicitly shape how the model uses evidence through a budgeted sufficiency objective: the keep bag should reach a target confidence τ for the true class while the drop bag does not support the true label (its trueclass probability remains low). We regularize evidence to be spatially compact and penalize selecting too many tiles, yielding four concrete properties: sufficiency, exclusion, contiguity, and budget. We call this framework ReaMIL: reasoning- and evidence-aware MIL.

> 💡 **Selector–Consumer 拆解（claude 批注）**: selector 是每 tile 的轻量 MLP+Concrete gate，consumer 是共享的 TransMIL。三个 bag 都经过同一 consumer，避免“不同模型导致的置信度不可比”。但这也意味着 selection utility 只在一个 consumer 内定义；换成 ABMIL、MambaMIL 或另一病理 FM 后排序是否仍有效，本文没有回答。

To measure these properties, we introduce diagnostics that probe how the model’s true-class probability grows as we reveal more top-scoring tiles. The area under this “Kcurve” (AUKC) and the minimal sufficient K (MSK) at a chosen confidence threshold summarize how quickly the model’s belief saturates. Across TCGA-NSCLC, TCGA-BRCA [8, 18], and PANDA [1], we show that ReaMIL preserves or improves baseline AUC while substantially reducing MSK and improving AUKC, indicating that highconfidence decisions can be supported by small, spatially compact sets of tiles.

In summary, the main contributions of this work are summarized as follows:

• We present ReaMIL, a reasoning- and evidence-aware MIL framework that integrates sufficiency, exclusion, spatial contiguity, and evidence sparsity.

• We introduce quantitative evidence-efficiency metrics, including minimal sufficient K (MSK) and the area under the K-curve (AUKC), which measure how quickly confidence emerges as diagnostic tiles are revealed.

• We demonstrate that our ReaMIL maintains or even improves slide-level performance while producing highly compact and spatially coherent evidence sets across TCGA-NSCLC, TCGA-BRCA, and PANDA.

> 💡 **对 ReadySlide 的 novelty 审计（claude 批注）**: 不安全的首创表述包括“冻结病理 FM 后加轻量 selector”“用 keep/drop 测 evidence”“用最小充分 patch 数量衡量选择”。更稳的空白是：多个 selector FM × consumer FM × budget 的完整交叉实验，以及严格冻结 consumer 时能否只修 selector 而不改变 full-bag diagnostic ability。

## 🔖 Section 总结

1. 诊断性能与证据质量是两个独立轴。
2. ReaMIL 用 full/keep/drop 三视图把 evidence 变成可干预对象。
3. 跨 consumer 迁移和严格 frozen-consumer 修复仍未覆盖。
