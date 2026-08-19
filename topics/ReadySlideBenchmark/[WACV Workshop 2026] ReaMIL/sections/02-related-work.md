[← 返回 README](../README.md)

## 2. Related Work

> 💡 **相关工作预览（claude 批注）**: 三条来源分别提供 consumer、失败诊断和 selector 训练工具：MIL/TransMIL 负责 slide prediction，attention interpretability 指出热图不等于因果证据，rationalization 与 selective prediction 提供预算化可微选择。

## 2.1. Multiple instance learning for whole-slide histopathology

Multiple instance learning (MIL) treats a digital slide as a bag of tiles with a single slide-level label and no supervision for individual tiles. Attention-based pooling, introduced by Ilse et al. [10], replaced fixed max- or mean-pooling with a learned weighted combination of tile features and became the standard aggregation strategy. Subsequent architectures incorporated class-specific attention and clustering constraints (CLAM [13]) or transformer-based selfattention to model long-range context between tiles (Trans-MIL [17]). More recently, feature extraction has been decoupled from MIL aggregation: large self-supervised or multimodal encoders pre-trained on millions of histology tiles are frozen, and MIL models operate on pre-extracted features [5]. This reduces training cost and improves robustness across cohorts. We follow this strategy, using UNI2-h as the feature backbone while the MIL component focuses on aggregating and selecting evidence at the tile level.

## 2.2. Interpretability of attention in MIL

Interpretability in MIL for pathology has largely relied on visualizing attention weights as heatmaps or displaying top-attended tiles [10, 13]. However, attention as explanation has well-known limitations [16]: attention scores are shaped by end-to-end training and may not reflect causal importance; high-attention tiles can be redundant or partially spurious; and there is no guarantee that the attended subset alone suffices to recover the correct prediction, nor that the complement is non-predictive. Various remedies—instance-level regularization, auxiliary classifiers, multiple attention heads, or region proposals from slide labels—can make heatmaps more visually convincing, but they typically lack a quantitative framework for measuring how much evidence is actually needed for a decision.

## 2.3. Budgeted evidence and selective prediction

The idea of constraining a model to rely on a small subset of inputs appears in selective prediction [9], budgeted or early-exit models, and rationalization methods that train differentiable selectors to pick a few tokens or patches so that a downstream predictor matches the full model using only the selected subset. Our work adapts this perspective to MIL: we attach a small selection head on top of a fixed MIL encoder and train it with losses enforcing sufficiency of the kept bag, exclusion of the dropped bag, spatial contiguity, and an explicit budget on selection rate (normalized selection mass). We quantify the resulting behavior with K-curves, minimal sufficient K (MSK), and area under the K-curve (AUKC)—metrics that capture how quickly confidence rises as diagnostic regions are added and how small a subset suffices for a diagnosis.

> 💡 **术语核对（claude 批注）**: 文中此处称 “fixed MIL encoder”，但实验流程是 frozen UNI2-h feature encoder + 从 baseline checkpoint warm-start 并继续参与联合目标的 TransMIL consumer。因此 selector 与 consumer 会共同适应，本文不能等同于 strict frozen-consumer repair。

## 🔖 Section 总结

- ReaMIL 不是新的病理 FM，而是 rationalization 思路在 WSI-MIL 上的实例化。
- 它比注意力可视化多了 sufficiency、exclusion、budget、contiguity 四个显式约束。
- 与 ReadySlide 的差异应写成跨模型角色 benchmark，而不是泛化地声称“首次诊断—选择解耦”。
