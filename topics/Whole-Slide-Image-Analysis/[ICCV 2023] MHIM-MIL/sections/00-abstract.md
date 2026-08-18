[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

MHIM-MIL 的反直觉命题：**只盯"显著 instance"（易分样本）对训练 MIL 分类边界并非最优——应挖掘"难分 instance"**。做法：Siamese 结构（动量 Teacher-Student）+ 基于注意力分数的多种遮蔽策略，用动量 teacher 隐式挖掘 hard instance 喂给 student，再用 EMA 反过来更新 teacher，配一致性约束稳定优化。

---

## Abstract

The whole slide image (WSI) classification is often formulated as a multiple instance learning (MIL) problem. Since the positive tissue is only a small fraction of the gigapixel WSI, existing MIL methods intuitively focus on identifying salient instances via attention mechanisms. However, this leads to a bias towards easy-to-classify instances while neglecting hard-to-classify instances. Some literature has revealed that hard examples are beneficial for modeling a discriminative boundary accurately. By applying such an idea at the instance level, we elaborate a novel MIL framework with masked hard instance mining (MHIM-MIL), which uses a Siamese structure (Teacher-Student) with a consistency constraint to explore the potential hard instances. With several instance masking strategies based on attention scores, MHIM-MIL employs a momentum teacher to implicitly mine hard instancesfor training the student model, which can be any attention-based MIL model. This counter-intuitive strategy essentially enables the student to learn a better discriminating boundary. Moreover, the student is used to update the teacher with an exponential moving average (EMA), which in turn identifies new hard instances for subsequent training iterations and stabilizes the optimization. Experimental results on the CAMELYON-16 and TCGA Lung Cancer datasets demonstrate that MHIM-MIL outperforms other latest methods in terms of performance and training cost. The code is available at: https://github.com/DearCaat/MHIM-MIL.

> 💡 **问题动机**（Hao 批注）：本文与同目录 [ACMIL](../../%5BECCV%202024%5D%20ACMIL/) 是"表亲"——都发现"注意力只盯少数显著 instance 伤害泛化"，都用"遮蔽显著 instance"来治。但立论视角不同：ACMIL 从"注意力熵/过拟合"切入（分散注意力），MHIM 从"hard example mining"切入（SVM 边界样本、人脸/ReID 里难样本更有用的经典思想迁移到 instance 级）。**核心机制差异**：MHIM 需要一个**动量 teacher**来打分并遮蔽 → student 只看难 instance；teacher 由 student 的 EMA 更新，形成迭代。ACMIL 的 STKIM 则无需 teacher。

> 💡 **机制拆解**（为什么"遮掉显著、逼看困难"能提升）（Hao 批注）：
> - **easy instance = 高注意力 instance**：在测试时它们对分类有用，但训练时模型只靠它们就能降 loss → 边界建得粗糙、泛化差。
> - **hard instance = 被遮后剩下的**：靠近类别边界、更难分，但对刻画判别边界更有信息量（SVM 支持向量的直觉）。
> - **无 instance 标签的难点**：传统 hard mining 需样本标签，MIL 只有 bag 标签。MHIM 的巧解 = **用注意力分数"间接"定位难样本**（遮掉高注意力的，剩下的就当难样本），绕开对 instance 标签的依赖。

> 💡 **定位 + 与竞品对比**（Hao 批注）：相比 DTFD-MIL 等"级联梯度更新"的复杂框架，MHIM 用 Siamese + 动量 teacher（**无额外可学习参数**），更简单、更稳、训练更省（Tab. 2：TransMIL 基线上 -24% 时间、-48% 显存）。这是它相对 [ACMIL](../../%5BECCV%202024%5D%20ACMIL/) 的权衡点：ACMIL 的 STKIM 训练开销≈ABMIL（最省），MHIM 需 teacher 前向（比 ACMIL 略贵但仍比 DTFD 省），换来"可插任意注意力 MIL + 一致性正则"的通用性。
