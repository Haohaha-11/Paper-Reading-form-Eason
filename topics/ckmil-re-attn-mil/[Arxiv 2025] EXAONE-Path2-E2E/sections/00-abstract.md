[← 返回 README](../README.md)

# 00 Abstract

## 原文

In digital pathology, whole-slide images (WSIs) are often difficult to handle due to their gigapixel scale, so most approaches train patch encoders via self-supervised learning (SSL) and then aggregate the patch-level embeddings via multiple instance learning (MIL) or slide encoders for downstream tasks. However, patch-level SSL may overlook complex domain-specific features that are essential for biomarker prediction, such as mutation status and molecular characteristics, as SSL methods rely only on basic augmentations selected for natural image domains on small patch-level area. Moreover, SSL methods remain less data efficient than fully supervised approaches, requiring extensive computational resources and datasets to achieve competitive performance. To address these limitations, we present EXAONE Path 2.0, a pathology foundation model that learns patch-level representations under direct slide-level supervision. Using only 37k WSIs for training, EXAONE Path 2.0 achieves state-of-the-art average performance across 10 biomarker prediction tasks, demonstrating remarkable data efficiency.

---

> 💡 **Hao 批注：文章的底层论证逻辑**
>
> 这篇文章的论证链条如下：
>
> **问题链**:
> 1. WSI 太大 → 无法直接全分辨率训练 → 主流做法是 patch SSL + MIL（两阶段）
> 2. patch SSL 的问题：SSL 的增强策略（旋转/颜色抖动/crop）是自然图像领域设计的 → 无法帮助模型学习 patch 级别的临床相关特征（如突变导致的组织形态变化）
> 3. SSL 数据效率低：需要海量数据才能学到好的表示 → 而带 slide-level 标签的数据虽然少，但标签信息量密度更高
>
> **解决方案**:
> - 用 slide-level 监督信号直接训练 patch 编码器（E2E）
> - 具体做法：三层 HIPT 架构让梯度从 slide-level loss 流回 patch 编码器
> - Curriculum learning 解决全分辨率 E2E 训练的计算成本问题
>
> **关键假设**:
> - Slide-level 的分子标签（如 EGFR 突变状态）可以为 patch 编码器提供比 SSL 更强的学习信号
> - 这个假设的前提是：突变状态会体现在组织的形态学变化上，且这些变化是可学习的
>
> **与 Revisiting-E2E 的区别**:
> - Revisiting-E2E 的核心问题是"E2E 中 MIL 不好好优化怎么办"——关注的是训练过程的稳定性
> - EXAONE Path 2.0 的核心问题是"SSL 特征学不到 biomarker 相关信息怎么办"——关注的是训练信号的质量
> - 两者都做 E2E，但关注的问题层次不同

---

> 💡 **Hao 批注：图 1 的关键信息**
>
> 图 1(a) 和 (b) 展示了 EXAONE Path 2.0 在两个维度上的效率优势：
> - **参数效率**: 比 GigaPath (1.1B params) 小很多但性能更高
> - **数据效率**: 用 37K WSIs 超越用 170K+ WSIs 的模型
>
> 这支持文章的核心主张："supervision signal quality > data quantity"。但需要注意，EXAONE Path 2.0 的训练需要多任务 slide-level 标签（33 癌种 + 12 器官 + 多个 biomarker），这些标签本身获取成本也不低。

---

> 💡 **Hao 批注：Method 快速索引**
>
> | 模块 | 作用 | 对应章节 |
> |------|------|----------|
> | HIPT 三层架构 | 分层处理 gigapixel WSI，降低计算复杂度 | Sec 2.1 |
> | Curriculum Learning | 分阶段提升分辨率，避免全分辨率全程训练 | Sec 2.1 |
> | Memory Management | Activation checkpointing + CPU offloading | Sec 2.1 |
> | Multi-Task Learning | 33 癌种 + 12 器官 + biomarker 联合优化 | Sec 2.2 |
> | Early Exit Strategy | 下游只用第一层 ViT + CLAM，避免过拟合 | Sec 2.2 |
