[← 返回 README](../README.md)

## 📌 批读预览

本节把 FOCI 放在 WSI-MIL、解释忠实性、稀疏 rationalization 三条线之间，并明确与端到端 evidence-aware MIL 的差别。

## 2 Related Work

## 2.1 Multiple instance learning for whole-slide images

The standard MIL recipe treats each slide as a bag of patch features with a single slide-level label and no per-patch annotation [14, 1]. Attention-based pooling [2] became the default aggregator, with CLAM [3] adding class-specific attention branches and instance-level clustering. Later WSI-MIL backbones replace or augment this pooling mechanism with transformer self-attention [15], hierarchical representations [16], hard instance mining [17], attribution-based selection [18], or multibranch masked attention [19]. In parallel, frozen pathology foundation encoders such as UNI [4], CONCH [20], and Prov-GigaPath [21] now provide patch features for slide-level MIL. FOCI fits this frozen-feature pipeline: the encoder and MIL backbone remain fixed, and only a lightweight rationale-readout module is trained to score which tiles to keep.

## 2.2 Interpretability and faithfulness in MIL

Attention weights are commonly reused as explanations in MIL [2, 3, 22], but attention scores do not directly answer whether a compact tile subset can recover the model output [7, 8]. Other explanation approaches include instance-level classifiers in CLAM [3], concept-based models [23], and gradient-based localization such as GradCAM [24]. These methods surface regions or concepts, but they typically do not report the operating-point question central to our study: how many tiles are sufficient for the frozen model to recover its prediction?

Interpretable-by-design MIL methods address related goals from a different angle. Additive MIL [25] decomposes slide predictions into region-wise additive contributions, and SI-MIL [26] introduces a self-interpretable MIL framework with feature-level explanations. Rather than designing a new intrinsically interpretable MIL architecture, we audit frozen WSI-MIL classifiers that expose per-tile features, and we report where post-hoc rationale highlighting has selection headroom rather than claiming superiority over intrinsically interpretable models. Perturbation-based evaluation [27] and MIL-specific patch-dropping metrics such as xMIL/AUPC [28] measure how predictions change when ranked regions are removed. Our SRP is complementary: it evaluates the insertion direction, measuring how quickly confidence is recovered as ranked tiles are progressively revealed.

> 💡 **claude 批注｜与解释方法的边界**: FOCI 不提供可加性归因，也不声称学习到病理概念；其主排序接受真标签导向 keep/drop 训练，插入 SRP 问少量 tile 能否让 frozen consumer 正确支持 $y$，删除则问高分 tile 是否负载真类证据。两者都不同于 full-bag prediction fidelity，也不能合成单一“faithfulness”分数；Appendix N 的 predicted-class SRP 只是另一个评估 target。

## 2.3 Token selection and frozen rationale readouts

Selecting inputs that support a prediction has been studied extensively in NLP rationalization [29]. Differentiable selectors identify token subsets that recover the full-input prediction [30, 31, 32]. Related work on selective prediction [33], early-exit networks [34], and cooperative rationalization [35] studies adjacent questions of confidence, computation, and complement control under different training assumptions. In vision and MIL, straight-through estimators [36] enable hard sparse selection, and ASMIL [37] jointly trains a selector-like mechanism with the MIL backbone.

FOCI differs from joint selector-training approaches in its frozen setting. The MIL classifier is already trained and remains fixed; the selector is a post-hoc readout head over a stable feature space, trained with keep/drop sufficiency and exclusion losses and evaluated through insertion-style SRP. This also separates FOCI from ReaMIL [38], a concurrent evidence-aware MIL training method in whole-slide histopathology. In ReaMIL, the selector and backbone share gradient flow after warmup to train a compact-rationale classifier. FOCI does not train a new evidence-aware classifier. It asks whether the decisions of already-trained MIL backbones are post-hoc readable from compact tile subsets, and uses this readout to measure selection headroom and architecture-dependent failure modes.

> 💡 **claude 批注｜诊断—选择解耦已有先例**: 这里已经把“先训诊断器、再冻结、只训练 rationale readout”写成核心差异。ReadySlide 应把两个问题拆开：其一，在固定 consumer、候选池、真标签 target、阈值与可行子集空间下做组合搜索，得到 consumer-optimal 最小 K 并量化 learned gap；其二，用 tumor/region/reader annotation 测临床定位对齐。后者不必保存 consumer 证据，因此不是前者的 performance upper bound。

## 🔖 本节总结

- FOCI 与 intrinsic MIL 的目标不同：前者审计既有模型，后者重构模型本身。
- 与 ReaMIL 的关键差异是 backbone 是否共享 selector 梯度。
- ReadySlide 的相关工作定位应承认 frozen rationale readout 已被 FOCI 明确提出。
