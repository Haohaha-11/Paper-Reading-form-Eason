[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

DeepSets 是**集合函数（set function）的奠基性理论工作**：证明任何**排列不变（permutation-invariant）**的集合函数都能分解为 $\rho\left(\sum_x \phi(x)\right)$ 的形式——先对每个元素做变换 $\phi$、求和聚合、再用 $\rho$ 输出。这正是 WSI MIL 里 **MeanPool / SumPool baseline 的理论根据**：把 WSI 当作 patch 的集合，`patch → φ(FM 特征) → Σ/mean 聚合 → ρ(分类头)`。

---

## Abstract

We study the problem of designing models for machine learning tasks defined on sets. In contrast to traditional approach of operating on fixed dimensional vectors, we consider objective functions defined on sets that are invariant to permutations. Such problems are widespread, ranging from estimation of population statistics, to anomaly detection in piezometer data of embankment dams, to cosmology. Our main theorem characterizes the permutation invariant functions and provides a family of functions to which any permutation invariant objective function must belong. This family of functions has a special structure which enables us to design a deep network architecture that can operate on sets and which can be deployed on a variety of scenarios including both unsupervised and supervised learning tasks. We also derive the necessary and sufficient conditions for permutation equivariance in deep models. We demonstrate the applicability of our method on population statistic estimation, point cloud classification, set expansion, and outlier detection.

> 💡 **问题动机 + 为何是 MeanPool 的理论根据**（Hao 批注）：DeepSets 本身不是 MIL 论文，但它是 WSI MIL baseline set 里 **MeanPool 这个 sanity control 的理论出处**。核心结论（Theorem 2）：**任何排列不变的集合函数 $f(X)$ 都能写成 $\rho\left(\sum_{x\in X}\phi(x)\right)$**。映射到 WSI：
> - **集合 $X$** = 一张 WSI 的所有 patch；
> - **$\phi$** = patch encoder（FM 时代就是冻结的 pathology FM，把 patch 变成特征）；
> - **$\sum$（或 mean）** = 排列不变的聚合；
> - **$\rho$** = slide-level 分类头。
>
> 所以 **`Frozen FM [N,D] → MeanPool → Linear` 不是随手拍脑袋的弱基线，而是"排列不变集合函数的通用形式"的最简实例**。这解释了为什么 MeanPool 在强 FM 特征下往往出奇地强（见 [SiMLP](../../../ckmil-re-attn-mil/) 的 mean>>max 观察）——它落在了集合函数的正确函数族里。

> 💡 **机制拆解（invariance vs equivariance）**（Hao 批注）：DeepSets 给两类结构：
> - **排列不变（invariant）**：输出不随元素顺序变 → 对应 **bag-level 分类**（WSI → 一个 label）。ABMIL/MeanPool 都属此类。
> - **排列等变（equivariant，Lemma 3）**：输入置换则输出同样置换 → 对应 **instance-level 预测**（每个 patch 一个输出）。等变层要求权重矩阵 $\Theta=\lambda I+\gamma(\mathbf{11}^T)$（对角相等、非对角相等）。
>
> WSI MIL 主要用不变形式（slide 分类），但等变层是理解"注意力/MoE 路由为何要 permutation-aware"的理论背景。
