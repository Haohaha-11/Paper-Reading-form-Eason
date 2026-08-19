[← 返回 README](../README.md)

> 💡 **claude 批注｜本节预览**: 摘要先给出核心诊断：分类注意力并不自动等于可干预证据；GCE-MIL 用 S/N/R 三个目标把“预测对”与“证据对”拆开评估。

# GCE-MIL: Faithful and Recoverable Evidence for Multiple Instance Learning in Whole-Slide Imaging

Xiangyu Li College of Intelligence and Computing Tianjin University xiangyuli@tju.edu.cn

Ran Su<sup>∗</sup> College of Intelligence and Computing Tianjin University ran.su@tju.edu.cn

## Abstract

Multiple instance learning (MIL) is the standard approach for whole-slide image (WSI) classification and survival prediction, where attention-based models aggregate patch features into slide-level predictions. These models treat attention weights as evidence for their predictions, but attention is optimized for classification, not for identifying which patches actually support the diagnosis. This conflation leads to three failures: selected patches are insufficient (keeping them alone drops Macro-F1 by 0.078), unnecessary (removing them barely changes the prediction), and unrecoverable (continuous attention scores disagree with discrete patch subsets used at inference). The central premise is that evidence quality should be optimized directly through explicit criteria—Sufficiency, Necessity, and Recoverability (S/N/R)—rather than inherited as a byproduct of classification. GCE-MIL is a backbone-agnostic wrapper implemented through three injection modes and three evidence components: a grounding mechanism that aligns selection with domain-specific concepts, noisy-OR coverage that acts as a differentiable proxy for interventional evidence search, and threshold-plus-repair recovery that converts continuous selectors into discrete subsets through marginal-guided repair. Across 9 backbones × 9 datasets (81 configurations), GCE-MIL improves average Macro-F1 by 0.024 and C-index by 0.014, reduces the continuous-discrete gap by 4–7×, and increases complement degradation by 2–4×. With optional tile prefiltering after discrete recovery, inference runs up to 5× faster while retaining 0.989× full-bag utility.

> 💡 **claude 批注｜摘要证据链**: 输入是 WSI patch bag，GCE 输出 slide prediction 与离散 evidence subset。9 backbone × 9 dataset 的 81 配置主要证明预测主结果的兼容性；0.078 keep-only、C-D gap 与 complement degradation 来自专门的证据诊断范围，并非每个 81 配置都同时做了稳定性、定位、消融和成本检查。5× 速度仅适用于可选 tile 预筛，不应与缓存特征下的标准验证混为一谈。

> 💡 **claude 批注｜本节小结**: 论文的核心不是把 attention 画得更漂亮，而是让证据成为训练目标、干预对象和可离散恢复的模型输出。
