[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文从 MLLM 的广泛应用需求（CUA、VLA 等）出发，指出核心挑战在于将感知信息整合到多步逻辑推理中。通过对比 LLM 的 test-time scaling law 与 MLLM 视觉信号衰减之间的矛盾，引出核心动机。随后批评了现有两种解决方案（文本增强推理 vs. 静态视觉重注入）的不足，提出 VaLR 的"动态视觉检查点"方案。

---

## 二、原始文本

Multi-modal Large Language Models (MLLMs) have achieved remarkable success in various multi-modal tasks such as image captioning (Zhang et al., 2024; Cheng et al., 2025) and visual question answering (Manmadhan & Kovoor, 2020; Huynh et al., 2025). Beyond these tasks, there is a growing demand to deploy MLLMs in more complex applications that require multi-step reasoing and long-horizon planning, such as computer-use agents (CUA) (Anthropic, 2024a;b) and Vision-Language-Action (VLA) models (Kim et al., 2024; Black et al., 2024; Lee et al., 2025; Bjorck et al., 2025). A core challenge in such applications lies in integrating perceptual information into multi-step logical reasoing within MLLM architectures.

> 💡 **应用背景**: 文中特别点名了 CUA（Computer-Use Agent）和 VLA（Vision-Language-Action）模型作为 MLLM 的下一个前沿应用场景，这两个场景都需要长时域的感知-推理闭环。这为本文的"长上下文视觉推理"问题提供了强烈的实践动机。

In the language domain, Chain-of-Thought (CoT) (Wei et al., 2022) has emerged as a cornerstone for improving reasoing capabilities of LLMs, enabling LLMs to decompose intricate tasks into intermediate linguistic steps. Building on the success of CoT, recent studies (Zheng et al., 2025b; Li et al., 2025e) have extended this approach from LLMs to MLLMs. However, in contrast to the test-time scaling law (Snell et al., 2024) of LLMs, MLLMs frequently struggle with long-context reasoing due to the attenuation of visual signals as the generated sequence length increases.

> 💡 **核心矛盾**: 这段话点出了本文最核心的问题：LLM 有 test-time scaling law（更长推理 = 更好性能），但 MLLM 没有——因为视觉信号随生成长度衰减。这个观察是整篇论文的出发点。作者用 "attenuation of visual signals" 精确描述了这个问题本质。

To address this issue, recent research in MLLMs focuses on enhancing the long-context reasoing of MLLMs. For instance, a line of work strengthens text reasoing of MLLMs via supervised fine-tuning (Yue et al., 2023; Yu et al., 2023) or reinforcement learning (Wang et al., 2024b; Havrilla et al., 2024; Shao et al., 2024b; Yu et al., 2024). While these text-centric methods have shown significant progress, they still suffer from diminishing visual signals when generating long text sequences. Alternatively, several studies explicitly re-introduce visual information by interleaving visual tokens (Zheng et al., 2025b; Yang et al., 2025d; Yoon et al., 2025) or generating images (Wang et al., 2025a; Li et al., 2025e). Yet, these approaches rely on static single-view visual features and use them only as a fixed initial context. Throughout this work, we demonstrate that utilizing static visual features leads to the gradual loss of visual context, whereas dynamically allocating visual details at each reasoing stage ensures information preservation, thereby enabling robust long-context reasoing in MLLMs.

> 💡 **机制拆解 — 现有方法的两条路线及其缺陷**:
>
> | 路线 | 代表方法 | 核心机制 | 致命缺陷 |
> |------|---------|---------|---------|
> | 文本推理增强 | SFT/RL 增强 CoT 推理 | 通过监督微调或强化学习强化文本推理链 | 视觉信号随文本生成逐渐衰减，治标不治本 |
> | 静态视觉重注入 | 交错视觉 token（DeepEyes）；生成图像（MVoT） | 在推理中插入视觉信息或生成中间图像 | 使用静态单视图特征，仅作为固定初始上下文 |
>
> 作者的立场非常明确：**静态视觉特征的利用方式必然导致视觉上下文的逐渐丢失**——因为推理过程中模型看到的始终是同一批视觉特征，随着推理步数增加，这些特征的相对影响力自然衰减。唯一的解决方案是**动态地在每个推理阶段重新注入视觉信息**。

In this paper, we introduce Vision-aligned Latent Reasoing (VaLR), a novel multi-modal reasoing framework that generates vision-aligned latent tokens during the reasoing process, which is inspired by the latent reasoing LLM approach (Hao et al., 2024b). The core idea of VaLR is to inject learnable latent tokens before each text-based reasoing step, creating "visual checkpoints" that keep the reasoing process grounded in image details. Unlike standard text tokens, these latent tokens are explicitly supervised to learn consistency with the dense visual representations of the input image which is highly correlated with the subsequent reasoing step. Specifically, we introduce a two-stage curriculum learning framework to gradually equip MLLMs with latent reasoing capabilities. The first stage involves supervised fine-tuning on general vision question-answering datasets to learn fundamental multi-modal reasoing ability. In the second stage, we incorporate a new group of latent tokens before every CoT step. We then apply representation alignment (Yu et al., 2025) to these latent tokens with dense features extracted from the corresponding image frame by vision encoders, e.g., DINOv2/v3 (Oquab et al., 2023; Simeoni et al., 2025), CLIP (Radford et al., 2021), and SigLIPv2 (Tschannen et al., 2025).

> 💡 **机制拆解 — VaLR 三要素**:
> 1. **Latent Tokens 作为"视觉检查点"**: 在每步文本推理前插入 K=16 个 latent tokens，在潜空间中保持视觉信息。这个概念非常直观——就像在长文档中插入书签。
> 2. **Representation Alignment (REPA)**: 将 MLLM 中间层的 hidden states 与外部视觉编码器的 patch 级稠密特征进行余弦相似度对齐。这是让 latent tokens "承载视觉信息"的关键监督信号。
> 3. **两阶段课程学习**: Stage 1 先建立文本推理基础，Stage 2 再引入 latent tokens + REPA。这种渐进式策略避免了直接学习潜推理的难度。

> 💡 **关键设计选择 — 为什么用 REPA 而非其他视觉注入方式？**:
> - 对比"将视觉特征作为额外的 input token"：REPA 在推理时不需要外部编码器，更高效（详见 Appendix C.4 的对比实验）
> - 对比"用 MLLM 原生 encoder 特征对齐"：外部编码器（DINOv3 等）提供更丰富的视觉表征，但 VaLR 也可以用 MLLM 原生 encoder（Table 3 验证）
> - 对比"生成图像"：潜空间操作比图像生成更高效、更可控

We demonstrate the effectiveness of VaLR through extensive evaluations on multiple Vision Question-Answering (VQA) datasets. Overall, VaLR exhibits superior performance over existing baselines on multiple VQA benchmarks. Specifically, on VSI-Bench (Yang et al., 2025b), VaLR boosts the accuracy of Qwen2.5-VL from 33.0% to 52.9%. Notably, as shown in Figure 2, VaLR successfully follows the test-time scaling law: the performance of VaLR improves in cases requiring longer reasoing, whereas baselines degrade under similar conditions. Furthermore, ablation studies suggest that VaLR can be used agnostically on several vision encoders, e.g., DINO, SigLIP, CLIP and even works with the standalone vision encoders of the original MLLM, i.e., Qwen2.5-VL encoder (Bai et al., 2025).

> 💡 **实验结果亮点**: VSI-Bench 上 +19.9%p 是本文最重的实验结果。这是一个多视图 3D 空间推理 benchmark，天然需要长上下文视觉推理，恰好验证了 VaLR 的设计目标。另外，"与原始 MLLM encoder 也能工作"说明 VaLR 的 latent reasoing 机制本身有效，外部编码器是锦上添花。

---

## 三、Summary

- **应用需求**: CUA 和 VLA 等应用需要 MLLM 具备长时域感知-推理闭环能力。
- **核心矛盾**: LLM 遵循 test-time scaling law，但 MLLM 因视觉信号衰减反而在长推理时性能下降。
- **现有方案不足**: 文本推理增强（治标不治本）+ 静态视觉重注入（固定的初始上下文，必然衰减）。
- **VaLR 方案**: 动态生成视觉对齐 latent tokens（"视觉检查点"）+ REPA 表征对齐 + 两阶段课程学习。
- **关键结果**: VSI-Bench +19.9%p，首个 MLLM test-time scaling，编码器无关性。
