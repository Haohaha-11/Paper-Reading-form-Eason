[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文从 VLM 推理中"越生成越不看图"的 architecture-level 瓶颈出发，对比现有工具增强方法的两个根本性缺陷（推理连续性被打断、训练数据需求大），提出一个激进假设——"不需要生成新图像 view"，并通过 V\* benchmark 上的初步验证 (3% gain with manual region injection) 为 SIEVE 框架提供实证动机。最终提炼三个核心贡献。

---

## 二、原始文本

Vision–Language Models (VLMs) have demonstrated strong reasoning performance on multimodal question answering, often supported by long chain-of-thought (CoT) reasoning Team et al. [2025a], Guo et al. [2025], Bai et al. [2025a], Team et al. [2025b]. However, their reasoning pipeline remains largely text-centric. In a standard VLM inference pipeline, the image is encoded into a fixed set of visual tokens that serve as static context, while reasoning unfolds autoregressively in text. As generation proceeds, the model's conditioning gradually shifts toward the growing history of generated text tokens, reducing the relative influence of visual evidence Li et al. [2025a], Fang et al. [2025]. Consequently, the model rarely revisits the image in a targeted, step-dependent way, and visual information is often underutilized in long-horizon reasoning.

> 💡 **问题定位 — VLM 的"视觉遗忘"问题**: 这是全文最核心的动机。VLM 自回归生成中，image token 是**静态前缀**，而生成的 text token 不断增加。随着序列变长，cross-attention 或 causal attention 中 image token 的相对权重自然衰减。这不是模型能力的缺陷，而是 architecture 的固有属性——模型被设计成"先看一遍图，然后闭眼写答案"。这个 insight 与 Li et al. [2025a] ("The Hidden Life of Tokens") 和 Fang et al. [2025] 的发现一致，但 SIEVE 的解法比这两篇更结构化（前者做 attention steering，后者做 decoding calibration，而 SIEVE 直接注入 region embedding）。

To address this limitation, recent work has begun to explicitly integrate visual evidence into the reasoning trajectory, drawing inspiration from human cognition Shao et al. [2024a]. Models repeatedly consult the image during CoT to improve grounded reasoning. After the release of OpenAI's o3 model OpenAI [2025], a common implementation has operationalized this idea through external visual operations such as zooming and cropping, generating sub-images for targeted inspection Huang et al. [2025a], Bai et al. [2025b]. Beyond predefined operations, other approaches allow VLMs to generate executable code for more flexible, programmatic image manipulation Lee et al. [2025], Mallis et al. [2025].

> 💡 **背景 — "Thinking with Images" 潮流的兴起**: OpenAI o3 的发布是一个分水岭，引发了大量"thinking with images"的方法涌现。但值得注意的是，这些方法的共同特征是将视觉回访**实例化为显式的图像操作**（crop, zoom, 甚至 code generation）。SIEVE 想挑战的正是这个隐式前提：回访视觉信息真的需要通过操作图像来实现吗？

Despite their effectiveness, existing methods are constrained by their reliance on external tools or agents, which introduces two key drawbacks. First, existing methods often disrupt the continuity of the reasoning chain during multi-turn interaction. Specifically, these approaches typically rely on an external module to generate a new view of the image, which is then appended to the original image input rather than being directly integrated into the CoT reasoning process. As a result, the extracted visual view is not inserted into the corresponding positions of the CoT; instead, it is appended to the input as additional images before the input and current output text. Second, enabling the model to repeatedly invoke external tools requires constructing a large amount of training data and designing complex training pipelines for the VLM to learn such capabilities.

> 💡 **现有方法的两个致命伤 — SIEVE 的差异化切入点**:
> 1. **推理连续性被打断**: 外部工具产生的 new view 是"追加"到输入序列前端的，而不是"插入"到 CoT 对应的推理步骤中。这就像你在写数学证明时，中途重新看了一遍题目——但你的笔记里题目被塞到了证明文本的最前面，而不是你"回头看"的那个步骤的旁边。这种语义位置的错位是 SIEVE 想用 "原位注入" 来解决的。
> 2. **训练数据壁垒**: 教 VLM 学会"何时调用哪些工具"需要海量标注数据（工具调用轨迹 + 对应的正确答案）。SIEVE 只需要 ~1.5k 样本——因为它的 action space 极简（插入 evidence 或 不插入），不需要复杂的 tool API learning。

In this paper, we challenge the prevailing paradigm and ask a simple question: do we truly need to generate new image views through external operations to let model revist the image information during inference? Our hypothesis is that the original visual embeddings already contain sufficient information for grounded reasoning, and that the main bottleneck is the model's limited ability to selectively reuse relevant visual evidence as generation unfolds. Instead of cropping, zooming, and reencoding additional images, we propose to directly extract task-relevant visual embeddings and insert them into the reasoning chain. To validate this idea, we conduct a preliminary analysis by manually identifying salient regions and directly injecting their visual embeddings into intermediate reasoning. This simple intervention consistently improves performance on the V\* benchmark, yielding a 3% accuracy gain without any additional training.

> 💡 **关键验证实验 — "如果手动帮你指出来，你能做对吗？"**: 这个 preliminary analysis 是全文最精妙的设计之一。作者手动标注了任务相关区域，提取它们的 embedding，然后直接注入推理——结果 +3%。这个实验的意义在于：(1) 证明了 "原始 embedding 信息充足" 的假设是成立的；(2) 提供了一个**性能上界**的信号——如果 evidence 发现做得完美，至少能提升 3%；(3) 说明模型确实"看得到但用不起来"，瓶颈在 retrieval/reuse 而非 encoding。
>
> **批判性思考**: 3% 的提升绝对值不算大，是否意味着 evidence 的增益有上限？还是说手动标注的区域不够精准（人认为重要的区域不一定等于模型需要的信息）？这留给了 RL-based 自动证据发现来回答。

Inspired by this observation, we introduce SIEVE, a framework that enables VLMs to revisit visual evidence without external tools or agentic image operations. As shown in Figure 1, when the model signals a need for additional visual grounding during inference, SIEVE retrieves embeddings for the relevant image regions and inserts them into the current reasoning chain, rather than localizing regions with an external tool and re-encoding new views. By reusing already encoded region features, SIEVE preserves access to fine-grained, localized visual cues for grounded multi-step reasoning while avoiding redundant vision re-encoding. We further develop a visually grounded RL training pipeline that enables the model to learn when and how to effectively retrieve and insert region embeddings. This RL training is highly data-efficient: SIEVE requires only a small dataset (approximately 1.5k samples) to acquire the capability. We evaluate SIEVE on Qwen3-based VLMs (4B and 8B) [Yang et al., 2025a] across multiple benchmarks. Results show consistent improvements over inference time tool-augmented baselines, indicating that much of the benefit typically attributed to explicit visua re-inspection can be achieved through retrieval and reuse of the embeddings of visual evidence. Our contributions are:

> 💡 **SIEVE 核心设计哲学**: "从重新看到重新用"——这一表述概括了整个方法的哲学转向。工具增强派在问"怎么让模型重新看"，而 SIEVE 在问"怎么让模型重新用已经看过的东西"。这是一个从 "generation" 到 "retrieval" 的范式转换。

• We propose SIEVE, a framework that lets VLMs revisit fine-grained visual evidence by retrieving and reinserting region-level visual embeddings from the original encoding, avoiding external crop/zoom tools and any re-encoding.

• We design a saliency-based mechanism to identify the embeddings of visual evidence that could be critical to reasoning semantics. Building on this, we develop a visually grounded RL training pipeline that teaches the model when to retrieve these visual embeddings during reasoning.

• We validate SIEVE across multiple model scales and benchmarks, where we show that using only a small training set (approximately 1.5k samples), SIEVE can demonstrate consistent gains in grounded multimodal performance (up to 8% on average across several benchmarks).

> 💡 **贡献解读**: 三个贡献分别对应 "框架"、"训练方法"、"验证" 的经典论文结构。但最值得关注的是贡献 2 中的 details——"基于显著性的视觉证据识别机制"——这是 SIEVE 最原创的部分：不需要外部模型或标注来告诉你"哪里重要"，而是让模型自身的梯度告诉你。

---

## 三、Summary

- **核心问题**: VLM 自回归推理中视觉信息逐渐衰减，静态 image token 被膨胀的 text token 淹没
- **现有方案缺陷**: 工具增强方法打断推理连续性 + 训练数据需求大
- **SIEVE 的激进假设**: 原始 embedding 足够，不需要重新生成图像 view
- **初步验证**: V\* 上手动的 region embedding 注入 +3%，证明假设成立
- **三个贡献**: (1) 免外部工具的 self-revisit 框架；(2) 梯度显著性 + 跨模态匹配的证据发现 + GRPO RL 训练；(3) 多模型多 benchmark 一致验证
