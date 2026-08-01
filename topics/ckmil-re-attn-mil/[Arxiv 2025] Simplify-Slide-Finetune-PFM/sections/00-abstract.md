[← 返回 README](../README.md)

# 00 — Abstract

> **原文**:

The emergence of foundation models in computational pathology has transformed histopathological image analysis, with whole slide imaging (WSI) diagnosis being a core application. Traditionally, weakly supervised fine-tuning via multiple instance learning (MIL) has been the primary method for adapting foundation models to WSIs. However, in this work we present a key experimental finding: a simple nonlinear mapping strategy combining mean pooling and a multilayer perceptron, called SiMLP, can effectively adapt patch-level foundation models to slide-level tasks without complex MIL-based learning. Through extensive experiments across diverse downstream tasks, we demonstrate the superior performance of SiMLP with state-of-the-art methods. For instance, on a large-scale pan-cancer classification task, SiMLP surpasses popular MIL-based methods by 3.52%. Furthermore, SiMLP shows strong learning ability in few-shot classification and remaining highly competitive with slide-level foundation models pretrained on tens of thousands of slides. Finally, SiMLP exhibits remarkable robustness and transferability in lung cancer subtyping. Overall, our findings challenge the conventional MIL-based fine-tuning paradigm, demonstrating that a task-agnostic representation strategy alone can effectively adapt foundation models to WSI analysis. These insights offer a unique and meaningful perspective for future research in digital pathology, paving the way for more efficient and broadly applicable methodologies.

> 💡 **核心论点**: Hao 批注 — 本文的核心论点是"在 PFM 时代，MIL-based fine-tuning 可能是不必要的复杂度"。Mean pooling + MLP（SiMLP）在所有任务上超越了 ABMIL、DTFD-MIL、ACMIL、RRTMIL 等复杂 MIL 方法——最高 +3.52%。这不仅是 performance improvement，更是 paradigm challenge：如果简单方法更好，为什么要用复杂的？

> 💡 **三个层次的评估**: Hao 批注 — 摘要中隐含了三个递进的评估维度：(1) 标准 slide-level 分类——SiMLP 最好；(2) few-shot——SiMLP 学习效率最高；(3) 迁移性——SiMLP 最稳定。这三个维度共同支撑了"task-agnostic representation is sufficient"的核心论点——如果 task-specific 微调真有必要，它应该在少样本和迁移场景中表现更好（因为针对性更强），但事实相反。
