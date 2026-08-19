[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

**EAGLE**（Efficient Approach for Guided Local Examination）模仿病理学家"只看关键区域"——用 task-agnostic 的 **CHIEF 选出 25 个最信息量 tile**，再用 **Virchow2 精提特征**、平均成一个 slide embedding。43 任务/9 癌种上超 patch 聚合法最多 23%、总体最优；单 slide 2.27 秒（省 >99% 算力）；且能审计"预测用了哪 25 个 tile"。

---

Artificial intelligence has transformed digital pathology by enabling biomarker prediction from high-resolution whole-slide images. However, current methods are computationally inefficient, processing thousands of redundant tiles per slide and requiring complex aggregation models. We introduce EAGLE (Efficient Approach for Guided Local Examination), a deep learning framework that emulates pathologists by selectively analyzing informative regions. EAGLE combines task-agnostic tile selection with detailed feature extraction and is benchmarked against leading slide- and tile-level foundation models across 43 tasks from nine cancer types spanning morphology, biomarker prediction, treatment response and prognosis. EAGLE outperforms patch aggregation methods by up to 23% and achieves the highest overall classification performance. It processes one slide in 2.27 s, reducing computational time by more than 99% compared with existing models. This efficiency supports rapid and auditable workflows by enabling review of the exact tiles used for each prediction and reducing dependence on high-performance computing. By reliably identifying informative regions and minimizing artifacts, EAGLE provides robust and auditable outputs, supported by systematic negative controls and attention concentration analyses. Its unified embedding enables rapid slide search, integration into multi-omics pipelines and emerging clinical foundation models.

> 💡 **问题动机 + 对 ReadySlide 的直接相关性**（claude 批注）：这篇是本主题里与 ReadySlide（WSI 压缩/保留）**最直接相关**的论文——EAGLE 本质是**极端 retention**：一张 WSI 平均 ~18,000 个 tile，EAGLE **只保留 25 个**（~0.1-2%），却在 43 个任务上超过"处理全部 tile"的方法。这是"retention 是杠杆、大部分 patch 冗余"这一 ReadySlide 主结论的**最强外部证据**。
> - **关键设计**：两阶段——CHIEF（在 6 万+ slide 上预训练的 task-agnostic 慢速 ABMIL）先在便宜的 CTransPath 特征上排序选 tile，Virchow2（强 tile encoder）只对选中的 25 个 tile 精提特征。**便宜的粗筛 + 昂贵的精提只用在少数 tile 上**——这正是 ReadySlide 的 allocator 思路。
> - **可审计性**：只用 25 个 tile → 能明确列出"预测依据的确切区域"，比扩散的注意力热图可审计得多。

> 💡 **机制拆解**（EAGLE 为什么能"少即是多"）（claude 批注）：Discussion 里作者给了原理性解释——这**不是启发式截断，而是弱监督下的偏差-方差权衡**：当预测信号相对总组织面积**空间稀疏**时，把推理限制在一个可复现的高显著性子集上能**改善统计条件（statistical conditioning）**。换言之：处理全部 tile 会引入大量冗余噪声（稀释判别信号），只留高显著 tile 反而降方差。这与 [ACMIL](../../../Whole-Slide-Image-Analysis/%5BECCV%202024%5D%20ACMIL/)"Top-10 占 85% 注意力"、[PIBD](../../../Whole-Slide-Image-Analysis/%5BICLR%202024%5D%20PIBD/)"留 25-40% patch 即可"是同一洞察的不同侧面。**对压缩研究**：这是"信息稀疏 → 激进保留有益"的又一确证，且给出了 task-agnostic 的选择器（CHIEF）而非依赖任务标签。
