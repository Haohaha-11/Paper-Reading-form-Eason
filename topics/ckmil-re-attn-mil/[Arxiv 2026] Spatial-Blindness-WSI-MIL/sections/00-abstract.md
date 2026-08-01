[← 返回 README](../README.md)

# 00 Abstract

> 💡 **Hao 批注 - 一句话概要**: 本文的核心主张是：现有 WSI MIL 模型虽然加了 Graph/Transformer/SSM 等"空间感知"模块，但联合训练导致模型实际依赖的是组合统计(composition/visual-word counting)而非组织拓扑(topology)。ResTopoMIL 通过"先学组合、再学拓扑残差"的两阶段训练 + 坐标打乱约束来强制空间分支学习真正的拓扑信息。

> 💡 **Hao 批注 - 关键洞察**: "spatial blindness"不是架构问题，是优化问题。这个诊断角度新颖——不是说你没加空间模块，而是加了也没用上。与 shortcut learning / simplicity bias / gradient starvation 文献一脉相承，但落实到 WSI MIL 的具体形式上。

> 💡 **Hao 批注 - 方法定位**: ResTopoMIL 的核心贡献不在于设计更复杂的图网络，而在于改变了空间分支的**训练问题**——用一个已冻结的组合锚点提供稳定的残差目标，用 shuffle loss 确保残差是真正的空间信号。这种"方法即训练策略"的思路与 ReadySlide 的 budgeted progressive coding 有类似的方法论气质。

> 💡 **Hao 批注 - 实验规模**: 9 个公开 WSI benchmark + Spatial-MNIST-Bag 受控诊断 + CAMELYON-16 定位，覆盖面广。但需要注意的是，所有方法都固定 UNI encoder，所以结论限定在"slide-level aggregation"层面。

---

**原文 Abstract:**

Whole-slide MIL models are often called context-aware once graphs, Transformers, or state-space modules are placed above patch embeddings. We show that this label can be deceptive. On pathology tasks where tissue architecture is part of the diagnostic signal, several strong MIL baselines retain nearly unchanged slide-level AUC after patch coordinates are permuted. Their predictions are accurate, but largely compositional. We refer to this failure mode as spatial blindness. Our explanation is optimization-based: dense appearance statistics are learned early under slide-level supervision, leaving weak gradients for sparse spatial relations. ResTopoMIL addresses the issue by first fitting a permutation-invariant prototype histogram and then freezing it while a lightweight graph branch learns the residual under a coordinate-shuffling constraint. The architecture is simple by design; the intervention is in how the spatial branch is trained. Across 9 public WSI benchmarks, ResTopoMIL improves classification and survival prediction with 1.15M parameters, restores sensitivity to coordinate perturbation, and gives stronger localization evidence on CAMELYON-16.
