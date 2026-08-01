# Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology

> **arXiv**: 2506.02408 | **年份**: 2025 | **引用数**: -- (S2 rate-limited)
>
> **作者**: Wenhao Tang<sup>1,2</sup>, Rong Qin<sup>1,2</sup>, Heng Fang<sup>3</sup>, Fengtao Zhou<sup>4</sup>, Hao Chen<sup>4</sup>, Xiang Li<sup>1,2</sup>, Ming-Ming Cheng<sup>1,2</sup>
>
> **机构**: <sup>1</sup>Nankai International Advanced Research Institute (Shenzhen Futian), <sup>2</sup>VCIP, Nankai University, <sup>3</sup>HUST, <sup>4</sup>HKUST

---

## 一句话概述

首次系统揭示 E2E 学习中稀疏注意力 MIL 引发的优化坍塌问题，提出 ABMILX（多头部注意力 + 全局相关性注意力增强），配合多尺度随机采样，用 ImageNet 预训练的 ResNet 超越 SOTA 病理基础模型。

---

## 核心贡献

1. **问题揭示**: 首次阐明 E2E 学习中的优化风险——稀疏注意力 MIL 在 E2E 训练中会形成恶性循环（劣质特征 → 错误注意力 → 劣质梯度 → 更劣质特征）
2. **方法创新**: 提出 ABMILX，含多头部注意力（MHLA，打破单点过度稀疏）和全局注意力增强（A+，利用 patch 间特征相似性传播注意力）
3. **实证突破**: E2E 训练的 ResNet-50 + ABMILX 在 PANDA 上超 UNI +14pp 准确率，在 BRCA 上超 GigaPath，训练成本仅 <10 RTX3090 GPU hours

---

## 关键洞见 (Hao 批注)

1. **MIL 是 E2E 瓶颈，不是采样策略**: 复杂采样策略（注意力采样）耗时 68h 只提升 ~0.5pp，而换 MIL（ABMIL→ABMILX）9h 提升 4.7pp —— MIL aggregator 才是 E2E 的核心杠杆
2. **稀疏性需保持但要合理**: 过度稀疏（ABMIL sparsity=80）→ 优化崩溃；全局注意力（TransMIL sparsity=13）→ 被冗余 patch 分散；合理稀疏（ABMILX sparsity=36）→ 最优
3. **FM 在经典任务（CAMELYON/NSCLC）已饱和**: 但挑战性任务（BRCA 亚型、生存分析、PANDA）仍有瓶颈——编码器未适配下游是关键
4. **与 ReadySlide 的关联**: 本文论证了 E2E 训练可让编码器学到任务特异性特征，但用的是 ResNet 轻量模型；ReadySlide 如果做 E2E fine-tuning，可能让压缩后的特征更好适配下游任务
5. **成本优势**: E2E ResNet 推理 1.7s/slide vs UNI 25s/slide vs GigaPath 83s/slide，对临床部署有实际意义

---

## 章节导航

| 章节 | 内容 |
|------|------|
| [00 Abstract](sections/00-abstract.md) | 摘要与核心发现 |
| [01 Introduction](sections/01-introduction.md) | 问题背景与研究动机 |
| [02 Method](sections/02-method.md) | ABMILX 设计与 E2E 框架 |
| [03 Experiments](sections/03-experiments.md) | 实验设置与结果分析 |
| [04 Discussion](sections/04-discussion.md) | 讨论、局限与未来方向 |

---

## 方法概览

```mermaid
flowchart TD
    WSI[Whole Slide Image] --> SAMPLE[Multi-scale Random<br>Instance Sampling]
    SAMPLE --> ENC[ResNet Encoder<br>θ - ImageNet预训练]
    ENC --> FEATS[Instance Features<br>E ∈ R^{s×D}]
    FEATS --> MHLA[Multi-Head Local Attention<br>m个独立注意力头]
    MHLA --> APLUS[Global Attention Plus<br>U·A 相关性传播]
    APLUS --> AGG[Head-level Aggregation<br>concat Z¹...Zᵐ]
    AGG --> HEAD[Task Head]
    HEAD --> LOSS[Slide-level Loss<br>CE / Cox]
    LOSS -.梯度回传.-> ENC
    LOSS -.梯度回传.-> MHLA
    
    style MHLA fill:#f9f,stroke:#333,stroke-width:2px
    style APLUS fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 实验结果速览

| 任务 | 数据集 | Best Two-Stage (R50) | E2E ABMILX (R18) | E2E ABMILX (R50) | Best FM |
|------|--------|---------------------|-------------------|-------------------|---------|
| Grading | PANDA | 62.72 | **78.34** | **78.83** | 76.37 (UNI) |
| Sub-typing | BRCA | 89.35 | 93.97 | **95.17** | 94.82 (GIGAP) |
| Sub-typing | NSCLC | 95.21 | 97.09 | 97.06 | 97.88 (UNI) |
| Survival | BRCA | 64.93 | **67.78** | 67.20 | 67.95 (CHIEF) |

---

## 核心公式：优化风险

$$\mathcal{R} = O\left(\max_{i \in \mathcal{N}} \hat{a}_i\right)$$

其中 $\mathcal{N}$ 为噪声实例集，$\hat{a}_i$ 为 softmax 归一化注意力。ABMILX 同时从 multi-head（方差降低）和 A+（相关性传播抑制噪声 attention）两个方向降低 $\mathcal{R}$。

---

## 与 EXAONE Path 2.0 的对比

| 维度 | Revisiting-E2E | EXAONE Path 2.0 |
|------|---------------|-----------------|
| 核心问题 | MIL 引发 E2E 优化风险 | SSL patch 不捕获 biomarker 特征 |
| 解决方案 | 改良 MIL (ABMILX) | HIPT + curriculum + 多任务 |
| 编码器 | ResNet (26M, ImageNet) | ViT (HIPT 三层) |
| 训练数据 | ~1K WSIs (下游数据) | 37K WSIs (多任务标签) |
| 典型提升 | PANDA +20pp vs R50 | 平均 AUROC 0.784 (SOTA) |
