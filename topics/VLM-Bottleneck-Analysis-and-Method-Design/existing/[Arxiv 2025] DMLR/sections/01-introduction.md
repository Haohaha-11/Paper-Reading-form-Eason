[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文从三个推理范式出发，揭示现有方法的局限性，并通过人类认知类比引出核心动机：推理不是固定的线性过程，而是在遇到不确定性时动态回顾视觉信息。

---

## 二、原始文本

Multimodal Large Language Models (MLLMs) [1, 2, 3, 4] have achieved remarkable breakthroughs in integrating visual and linguistic information. This progress has facilitated the incorporation of Chain-of-Thought (CoT) reasoning into multimodal tasks, enabling models to construct structured reasoning paths across visual and textual modalities. Current multimodal reasoning approaches can be broadly categorized into three types: (1) Textual-only Reasoning [5, 6, 7], which generates intermediate reasoning steps in the sematic space. Such methods explicitly express reasoning logic through language generation but often suffer from language bias and insufficient visual grounding, as shown in Figure. 1(a). (2) Think with Images attempts to directly manipulate or augment images during reasoning, such as local zooming [8, 9], region highlighting [10, 11], or generating intermediate reasoning steps via diffusion models [12, 13] to enhance visual alignment. Despite their effectiveness in improving reasoning to a certain extent, they still face challenges such as unstable tool invocation and high inference overhead, as reflected in Figure 1(b). Recently, latent-space reasoning has emerged as a promising paradigm for enhancing reasoning capabilities in large language models, as exemplified by approaches such as CoCoNut [14] and LatentSeek [15]. Its core idea is to perform implicit reasoning in the latent space, replacing explicit textual steps with latent vectors to reduce redundant generation and capture more compact information. However, recent studies [16, 17, 18, 19] still rely on extra training to enforce latent reasoning triggered at fixed positions (via special tokens). This rigidity prevents the model from adaptively allocating reasoning effort.

> 💡 **机制拆解 — 三类多模态推理范式**:
>
> | 范式 | 代表方法 | 核心机制 | 局限性 |
> |------|---------|---------|--------|
> | Textual-only Reasoning | CCoT, Kam-CoT | 在语义空间生成显式推理步骤 | Visual grounding 不足，语言偏置 |
> | Think with Images | Grit, DeepEyes, Pixel Reasoner | 操作/增强图像辅助推理 | 外部工具调用不稳定，推理开销大 |
> | Latent-Space Reasoning | CoCoNut, LatentSeek | 潜空间内隐式推理 | 需要额外训练，固定在特定位置触发 |

Inspired by human cognition, we argue that reasoning is not fixed. Instead, humans dynamically revisit visual information, specifically when they encounter uncertainty. Drawing on this intuition, we empirically analyze the interplay between the model's visual reliance and its internal confidence. Our analysis reveals two key phenomena: (i) Visual information is used only at a few specific stages of the reasoning process rather than at fixed positions, and (ii) Internal confidence serves as a natural indicator for the need of visual grounding as it strongly correlates with reasoning correctness. These findings suggest that effective multimodal reasoning relies on dynamic visual usage guided by internal confidence.

> 💡 **问题动机 — 两个关键实证发现**: (i) 视觉信息只在推理过程的少数特定阶段被使用，而不是固定位置；(ii) 模型内部置信度 (internal confidence) 是视觉 grounding 需求的天然指标，与推理正确性高度相关。这两个发现为 DMLR 的核心设计提供了直接的实证依据：**用置信度作为信号来动态决定何时、如何注入视觉信息**。

In light of these observations, we propose DMLR, a Test-time Dynamic Multimodal Latent Reasoning Framework, as shown in Figure 1(c). Specifically, it introduces optimizable latent think tokens to serve as a mental draft, which are iteratively refined through confidence-guided policy gradient updates. Crucially, we design a confidence-driven dynamic visual injection strategy. At each step, the model autonomously determines whether to revisit visual information and which contents to select (ranging from none to a few specific patches). This mechanism allows the model to naturally skip visual injection when internal confidence is sufficient, or actively integrate targeted visual clues when necessary, all driven by the objective of maximizing reasoning confidence, effectively mimicking the human cognitive process of checking visual clues to build confidence. After several iterations, the optimized latent tokens are decoded with the input without extra inference cost. Extensive experiments demonstrate that DMLR consistently outperforms existing methods across diverse architectures and tasks while maintaining high efficiency.

> 💡 **机制拆解 — DMLR 三组件**:
> 1. **Latent Think Tokens**: 可优化的潜向量，作为"思维草稿"在潜空间中迭代优化
> 2. **Confidence-Guided Policy Gradient**: 以最大化推理置信度为目标，通过 REINFORCE 类方法优化潜令牌
> 3. **Dynamic Visual Injection**: 基于置信度动态选择是否注入视觉 patch，以及注入哪些 patch（从 none 到 few specific patches）
>
> 与 baseline 的关键差异：不需要额外训练（training-free），不像 CoCoNut/LatentSeek 需要在特定位置触发；视觉注入是动态自适应的，不像 ICoT 注入大量冗余视觉信息。

The main contributions can be summarized as follows:

- We reveal two key phenomena: Visual information contributes only at specific reasoning steps; and confidence reflects both reasoning quality and visual grounding.
- We propose DMLR, a test-time framework for multimodal latent reasoning that integrates confidence-guided latent optimization with dynamic visual injection.
- Extensive evaluations show that DMLR consistently outperforms other methods across diverse architectures and multimodal tasks, while maintaining high efficiency.

---

## 三、Summary

- **问题定义**: 多模态推理中感知与推理的静态分离
- **核心洞察**: 视觉信息仅在特定推理步骤关键 + 置信度是视觉需求的自然指标
- **方案**: DMLR = Training-free 潜空间推理 + 置信度引导优化 + 动态视觉注入
- **贡献**: 两个现象揭示 + 新框架提出 + 全面的实验验证
