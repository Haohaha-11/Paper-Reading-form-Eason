[← 返回 README](../README.md)

# 01 Introduction

> 💡 **Hao 批注 - 问题设定**: WSI MIL 的标准假设是 patches 作为 instances，slide 作为 bag。但很多病理标签不仅取决于"出现了哪些组织成分"，还取决于"它们如何排列"——Gleason 分级依赖腺体形成，乳腺癌亚型区分导管生长和单行浸润，预后反映浸润前沿/免疫聚集/乳头核心等空间结构。

> 💡 **Hao 批注 - 核心矛盾**: 近年 context-aware MIL (Graph/Transformer/Hierarchical/SSM) 声称解决了空间建模问题，但**架构存在不等于被使用**。作者提出的 coordinate-shuffling stress test 直接戳穿了这个假象——把坐标打乱，AUC 几乎不变，说明模型根本没用到拓扑。

> 💡 **Hao 批注 - 与 gradient starvation 的关联**: 作者的诊断是优化层面的——composition 信号密集、易学、梯度强；topology 信号稀疏、难对齐 slide label、梯度弱。联合训练下网络会先拟合组合信号，一旦 loss 降下来，留给空间分支的梯度几乎为零。这个解释与 gradient starvation [Pezeshki et al., 2021] 和 simplicity bias [Shah et al., 2020] 一致。

> 💡 **Hao 批注 - 方法思路**: 不是让一个网络同时发现 composition 和 topology，而是显式地把组合信号学出来并冻结，然后在残差上训练拓扑分支。这种方法论与"residual learning"的哲学一致——让容易的部分先被解释掉，困难的部分作为残差被单独处理。

> 💡 **Hao 批注 - 立意**: 把评价标准从"有没有加空间模块"转向"决策是否真的依赖空间信息"——这个区分很重要，也是这篇论文的核心贡献所在。

---

**原文 Introduction:**

Computational pathology increasingly learns diagnostic and prognostic models directly from digitized whole-slide images (WSIs), where cellular morphology, tissue architecture, and long-range context are preserved at gigapixel scale [Pantanowitz et al., 2011, Verghese et al., 2023, Song et al., 2023]. The weak supervision problem is severe: a slide may contain tens of thousands of relevant and irrelevant regions, but most clinical datasets provide only a slide-level label. Multiple instance learning (MIL) is therefore the standard formulation for WSI analysis [Dietterich et al., 1997, Maron and Lozano-Perez, 1997, Campanella et al., 2019, Lu et al., 2021]: patches are treated as instances, the slide as a bag, and the model predicts the bag label without patch-level annotation.

MIL works remarkably well in pathology, but its success can hide a limitation. Many labels are not determined by local appearance alone. Gleason grading depends on gland formation; breast cancer subtyping distinguishes ductal growth from single-file strands; prognosis often reflects invasive fronts, immune aggregates, papillary cores, solid growth, or tumor-stroma organization [Quail and Joyce, 2013, Cheng et al., 2021, Wang et al., 2021]. These are spatial statements. They depend not just on which tissue components appear, but on how they are arranged.

![](../images/96c6ce68f96152ed47fbcf9987b31bd590d9823b7346983151900d5ed0271c70.jpg)

> 💡 **Hao 批注 - Figure 1**: ResTopoMIL 概念图。(a) 标准 MIL 在坐标置换前后预测相似，说明主要用组合信息。(b) ResTopoMIL 将问题分为统计流和拓扑流。(c) 统计流提供基础预测，拓扑流从空间组织中学习残差修正。

At first glance, recent context-aware MIL methods should address this issue. Graph networks, Transformers, hierarchical models, and state-space models all process a slide as more than an unordered bag [Chen et al., 2021b, Pati et al., 2022, Adnan et al., 2020, Vaswani et al., 2017, Shao et al., 2021, Chen et al., 2022, Gu et al., 2021, Yang et al., 2024, Zhang et al., 2025a]. Architecture alone, however, does not tell us what the trained predictor uses. We use a simple stress test: keep every patch embedding fixed, and randomly permute the coordinates used to build spatial context. On tasks where architecture is label-relevant, a topology-using model should suffer. Several strong context-aware baselines barely move. They have spatial machinery, but their learned decision rules behave much like bag-of-visual-words classifiers.

The question is then not whether a model contains a spatial operator, but why such an operator can remain unused. Our explanation is optimization-based. Tissue composition provides a dense, early signal: many patches contribute to the same slide label. Topological evidence is sparser and harder to align with slide-level supervision. Under joint training, the network can reduce the loss by fitting composition first; once that happens, little useful gradient remains for the spatial branch. We call this behavior optimization laziness, in the descriptive sense that the easiest explanatory signal dominates training while the harder structural signal is left undertrained. The phenomenon is related to simplicity bias, texture bias, and gradient starvation [Shah et al., 2020, Geirhos et al., 2018, Noroozi and Favaro, 2016, Pezeshki et al., 2021].

ResTopoMIL follows this diagnosis. Rather than asking one network to discover composition and topology at the same time, it learns the compositional explanation explicitly. A permutation-invariant statistical stream is trained first and frozen. A lightweight graph stream is then trained on the residual, with a shuffle-based loss that asks it to distinguish real tissue topology from coordinate-permuted topology. The graph module is intentionally small. The point is not to add a heavier context block, but to change the training problem faced by the spatial branch.

This shifts the evaluation away from architecture labels. "Context-aware" should mean that a model's decision changes when clinically relevant spatial organization is removed. We therefore test topology-destroying coordinate perturbations, separate pure composition from pure topology in a controlled benchmark, and ask whether residual training changes both accuracy and spatial behavior. Topology-preserving transformations are discussed only as expected graph invariances, not as an additional measured benchmark.

The paper makes four contributions:

- Spatial blindness is defined and tested as insensitivity to coordinate perturbations on structure-dependent MIL tasks.
- A controlled composition-topology diagnostic benchmark shows that strong MIL models can solve compositional tasks while failing on pure topology.
- ResTopoMIL learns composition first and topology as a residual correction, with design analysis deferred to the appendix.
- Experiments on 9 public pathology benchmarks show gains in prediction, spatial sensitivity, and localization quality with a compact 1.15M-parameter model.
