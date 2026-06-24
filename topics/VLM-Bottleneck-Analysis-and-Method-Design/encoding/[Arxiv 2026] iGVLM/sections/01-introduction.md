[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文从 VLM 的发展历程出发，指出核心瓶颈：视觉特征被无差别复用于不同的文本查询。通过 Grad-CAM 可视化（Figure 1）直观展示了指令无关编码器的局限，然后对比了 QA-ViT（部分注入）和 DyFo（MCTS 搜索）等现有方案，最终引出 iGVLM 的解耦双分支设计理念。

---

## 二、原始文本

In recent years, advances in computer vision (Zhang et al., 2024; Liu et al., 2021) and natural language processing (Vaswani et al., 2017; Radford et al., 2019; Brown et al., 2020) have driven remarkable progress in Vision–Language Models (VLMs) (Chen et al., 2024a; Lu et al., 2024; Chen et al., 2023; Jiang et al., 2024). By jointly modeling visual perception and linguistic understanding, these models achieve strong performance on multimodal tasks such as image captioning, visual question answering, and grounded dialogue, representing an important step toward generalpurpose multimodal intelligence. Despite this progress, a fundamental challenge remains: how to condition visual perception on task-specific linguistic instructions in a principled and efficient manner.

> 💡 **开篇格局**: 标准的"大背景+核心问题"三段论开篇。值得注意的写法是最后一句被加粗强调的核心问题——它不是笼统的"how to improve VLMs"，而是非常精确地定位到"how to condition visual perception on task-specific linguistic instructions"。这表明文章的核心贡献点在**条件化机制**而非单纯的性能提升。

Most existing VLMs rely on static, instruction-agnostic vision encoders, such as CLIP-ViT (Radford et al., 2021), which extract visual representations independently of the downstream textual query. As a result, visual features are reused across different instructions in an invariant manner, limiting the model's ability to emphasize task-relevant cues and perform fine-grained, question-aware reasoing. This limitation is qualitatively illustrated in Figure 1, where static visual representations fail to highlight instruction-dependent regions that are critical for answering different questions grounded in the same image. These observations suggest that the core difficulty lies not in relearning visual perception itself, but in conditioning the utilization of visual features on linguistic instructions.

> 💡 **精确的问题诊断**: "The core difficulty lies **not in relearning visual perception itself**, but in **conditioning the utilization** of visual features on linguistic instructions." 这句话非常关键——它明确定义了问题的边界：视觉感知能力已有的预训练编码器已经具备，不需要重新学习；真正的问题是"在什么情况下使用哪些视觉特征"。这一定位直接决定了 iGVLM 的设计哲学：保留预训练视觉能力（frozen），在其之上叠加条件化机制。

Consequently, recent work has explored lightweight mechanisms to introduce instruction awareness while preserving the perceptual strength of pretrained vision encoders. QA-ViT (Ganz et al., 2024) injects textual representations into upper layers of a frozen vision transformer, enabling limited instruction-dependent adaptation with high efficiency. However, such partial integration provides relatively weak conditioning and may still perturb pretrained visual representations. In contrast, DyFo (Li et al., 2025) formulates visual reasoing as a sequential decision process guided by external expert models and Monte Carlo Tree Search, allowing more flexible, instruction-aware attention shifts at the cost of substantial inference overhead and reliance on expert quality. Taken together, existing approaches highlight the challenge of achieving effective instruction conditioning while maintaining both computational efficiency and representation stability.

> 💡 **机制拆解 — 两种现有方法的取舍**:
>
> | 方法 | 核心机制 | 优点 | 缺点 |
> |------|---------|------|------|
> | QA-ViT (CVPR'24) | 将文本表征注入 frozen ViT 高层 | 高效 | 条件化弱，可能扰动预训练表征 |
> | DyFo (CVPR'25) | 外部专家 + MCTS 引导视觉搜索 | 灵活，指令感知的注意力转移 | 推理开销巨大（20x+），依赖专家质量 |

> 💡 **写作技巧**: 这段 related work 的写法值得学习——不是简单列举方法，而是用"However...In contrast...Taken together..."将两种极端方案（高效但弱条件化 vs. 强条件化但低效）组织成一个连续的对比论证，自然地引出 iGVLM 的"平衡点"定位。

Motivated by this observation, we propose iGVLM, a decoupled instruction-guided vision encoder for Vision–Language Models. iGVLM adopts a dual-branch architecture that separates static and dynamic perception pathways: a frozen static branch preserves task-agnostic visual representations learned during pre-training, while a dynamic branch integrates lightweight, instruction-conditioned adapter modules that modulate feature utilization under textual guidance. This design enables flexible, instruction-aware visual reasoing without retraining the backbone, achieving a favorable balance between adaptability, efficiency, and representation stability. We evaluate iGVLM on the MMStar (Chen et al., 2024b) benchmark for fine-grained multimodal reasoing, and further introduce MM4, a controlled diagnostic benchmark for assessing question-aware visual reasoing under multi-query, multi-instruction settings.

> 💡 **iGVLM 定位**: 在 QA-ViT（高效但弱条件化）和 DyFo（强条件化但低效）之间的"甜点"。关键设计词是 **decoupled**（解耦）——不是让一个编码器同时做两件事，而是用两个分支各司其职：一个"记住怎么看"，一个"决定看什么"。

> 💡 **关于 MM4 的必要性**: 为什么需要一个新的 benchmark？现有 benchmark（如 MMStar）每个问题独立评估，不评估"同一个图像对不同问题的视觉感知一致性"。MM4 强制模型在同一个图像上回答 4 个语义不同的问题，并根据 n-out-of-4 正确数计分，直接测量 instruction-conditioned perception 的质量。

Our main contributions are summarized as follows:

- We propose iGVLM, a decoupled instruction-guided vision encoder that separates representation preservation from instruction-conditioned adaptation via a dualbranch architecture.
- We introduce MM4, a controlled diagnostic benchmark for evaluating question-aware visual perception under multi-query, multi-instruction scenarios.
- We demonstrate through extensive experiments on MMStar and other multimodal benchmarks that iGVLM improves instruction sensitivity and finegrained reasoing while maintaining efficiency and general-purpose multimodal capability.

---

## 三、Summary

| 维度 | 内容 |
|------|------|
| **核心问题** | 如何高效地将视觉感知条件化于任务特定的语言指令 |
| **问题诊断** | 核心难点不是重新学习视觉感知，而是调控视觉特征的**利用方式** |
| **现有方法光谱** | QA-ViT（高效但弱条件化） ← iGVLM（平衡点） → DyFo（强条件化但低效） |
| **iGVLM 方案** | 解耦双分支：冻结分支保留表征 + 动态分支通过 AdaLN 实现指令调制 |
| **三大贡献** | (1) iGVLM 框架 (2) MM4 诊断基准 (3) 全面的跨 backbone 实验验证 |
