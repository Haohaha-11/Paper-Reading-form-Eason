[← 返回 README](../README.md)

# 01 Introduction

## 原文

Digital pathology has emerged as a critical domain for AI-driven healthcare applications, with whole-slide images (WSIs) presenting unique computational challenges due to their gigapixel scale [2, 13, 16]. Current approaches typically follow a two-stage paradigm: training patch-level encoders through self-supervised learning methods such as DINO [1] and DINOv2 [12], then aggregating patch-level embeddings using multiple-instance learning (MIL) or slide-level encoders for downstream prediction tasks [4, 8, 13, 16].

Although this paradigm has shown promise, it has fundamental limitations in the digital pathology field. Self-supervised patch-level pretraining does not guarantee to capture complex domain-specific features that are essential for biomarker prediction, such as mutation status or other molecular characteristics, as self-supervised learning (SSL) methods rely only on basic augmentations selected for natural image domains on small patch-level area. Moreover, these approaches demonstrate inferior data efficiency compared to fully supervised methods, requiring extensive computational resources and large datasets to achieve competitive performance [7, 14].

To address these limitations, we introduce EXAONE Path 2.0, a pathology foundation model that learns patch-level representations under direct slide-level supervision. Our approach fundamentally differs from existing methods by incorporating multiple slide-level labels during patch encoder training, enabling the model to learn clinically relevant features more effectively.

Our results demonstrate that EXAONE Path 2.0 achieves superior average performance across all evaluated tasks while requiring substantially fewer training samples than competing methods, marking a significant advancement in computational pathology.

---

> 💡 **Hao 批注：Introduction 的论证策略分析**
>
> 这篇文章的 Introduction 非常简洁（仅 4 段），采用了"直奔问题"的策略：
>
> 1. **段 1（现状）**: WSI 难处理 → 主流用 SSL + MIL 两阶段
> 2. **段 2（问题）**: SSL 学不到 biomarker 特征 + 数据效率低
> 3. **段 3（方案）**: 用 slide-level 监督直接训 patch encoder
> 4. **段 4（结果）**: 更少数据，更好性能
>
> 与 Revisiting-E2E 相比，缺少对相关工作的详细讨论——这个在 Introduction 里没有展开，而是放在方法部分隐式地对比。
>
> **值得注意的未充分论证点**:
> - "SSL 学不到 biomarker 特征" 这个论断虽然直观合理，但文章没有在 Introduction 中提供直接的实验证据来证明 SSL 特征在 biomarker 任务上的不足（实验部分也只有最终 AUROC 对比，缺乏对 SSL vs supervised 特征空间的分析）
> - 文章没有讨论"为什么 slide-level 标签可以让 patch 编码器学到更好的特征"的机制——梯度是如何从 slide-level 传递到 patch 级别的？这部分直到 Method 部分才解释

---

> 💡 **Hao 批注：与 Revisiting-E2E Introduction 的对比**
>
> | 维度 | EXAONE Path 2.0 | Revisiting-E2E |
> |------|-----------------|---------------|
> | Introduction 长度 | ~4 段 | ~6 段 + 图 |
> | 问题定位的精确度 | 一般（SSL 不行→用监督） | 精确（MIL 稀疏注意力→优化坍塌） |
> | 对前人工作的批评 | 间接（SSL data inefficient） | 直接（前人忽略了 MIL 优化风险） |
> | 理论深度 | 浅（没有定义问题机制） | 深（定义了优化风险 R） |
> | 图的作用 | 展示效率优势 | 展示问题机制和 MIL 对比 |
>
> EXAONE Path 2.0 的 Introduction 更像工程报告，Revisiting-E2E 的 Introduction 更像研究论文。两者反映了不同的写作风格和侧重点。

---

> 💡 **Hao 批注：技术背景补充 —— HIPT 是什么**
>
> HIPT (Hierarchical Image Pyramid Transformer) 是由 Chen et al. (CVPR 2022) 提出的分层 ViT 架构，专门为 gigapixel 病理图像设计：
>
> ```mermaid
> flowchart TD
>     WSI[Whole Slide Image] --> P256[256×256 Patches]
>     P256 --> V1[ViT Stage-1<br>每个 patch → 1 token]
>     V1 --> R1024[1024×1024 Region<br>= 4×4 patch tokens]
>     R1024 --> V2[ViT Stage-2<br>每个 region → 1 token]
>     V2 --> R4096[4096×4096 Region<br>= 4×4 region tokens]
>     R4096 --> V3[ViT Stage-3<br>全图 → 1 token]
>     V3 --> PRED[Slide-level Prediction]
> ```
>
> 核心思路：通过逐层抽象（16×16 像素 → 256×256 → 1024×1024 → 4096×4096），避免直接处理全分辨率图像。原始 HIPT 用 SSL 训练，EXAONE Path 2.0 将其改造为 E2E 监督训练。
