[← 返回 README](../README.md)

# Abstract & Figure 1

## 一、Preview

本文提出 HIVE（HIerarchical Visual cuEs injection），将 loop transformer 架构扩展到多模态场景，并首次在潜空间推理过程中注入层级化视觉线索（从全局场景到细粒度区域）。核心 insight：稳健的推理应在隐空间内演进并无缝融合多模态信号，而非依赖显式文本 CoT。HIVE 通过在递归 transformer blocks 内注入多尺度视觉信息，实现完全在 aligned 隐空间中的 grounded 多步推理。

Table 1 是理解本文定位的关键：将潜空间推理方法分为"训练诱导型"（Coconut/Heima，通过压缩 CoT 到 learnable token 来模拟隐式推理）和"结构递归型"（Huginn/HIVE，通过 loop transformer 天然支持迭代推理），HIVE 在此基础上首次加入层级视觉特征。

---

## 二、原始文本

### Abstract

The advancement of multimodal large language models (MLLMs) has enabled impressive perception capabilities. However, their reasoning process often remains a "fast thinking" paradigm, reliant on end-to-end generation or explicit, language-centric chains of thought (CoT), which can be inefficient, verbose, and prone to hallucination. This work posits that robust reasoning should evolve within a latent space, integrating multimodal signals seamlessly. We propose multimodal latent reasoning via HIerarchical Visual cuEs injection (HIVE), a novel framework that instills deliberate, "slow thinking" without depending on superficial textual rationales. Our method recursively extends transformer blocks, creating an internal loop for iterative reasoning refinement. It further injects hierarchical visual cues, from global scene context to fine-grained regional details, into the model's latent representations, showing that this strategy remains effective in a loop-transformer reasoning framework. This enables the model to perform grounded, multistep inference entirely in the aligned latent space. Extensive evaluations demonstrate that test-time scaling remains effective when incorporating vision knowledge, and that hierarchical visual cue injection can be effectively integrated into the loop-transformer framework for improved understanding of complex scenes.

> 💡 **一句话概括**: HIVE 通过递归扩展 transformer blocks 创建内部推理循环，并在循环中注入从全局到局部的层级化视觉线索，实现完全在 aligned 潜空间中的 grounded 多步推理——不做显式 CoT，不做外部工具调用，纯粹的隐空间"慢思考"。

> 💡 **机制拆解 — 两个关键词**:
> - **"Loop Transformer Latent Reasoning"**: 不是 Coconut/Heima 那种"训练一个行为让模型模仿隐式推理"，而是通过 loop transformer 的**结构性递归**天然支持隐空间迭代 refined thinking。
> - **"Hierarchical Visual Cues Injection"**: 不是注入单一的最后一层 ViT 特征，而是从 ViT 的多层（浅→深）分别提取 visual features，在递归迭代中按"课程"顺序注入。

### Figure 1 (Architecture Overview)

> **Figure 1 原始描述**: Visualization of traditional MLLMs, visual features extracted from a vision tower are projected into the language space and directly concatenated with text tokens. This combined sequence is then fed into a stack of transformer decoder blocks. HIVE is built upon Huginn, a recursive architecture that iteratively processes token representations through a unified set of layers to enhance feature depth. We have extended this by incorporating the visual modality and, for the first time, introducing hierarchical visual information into latent space reasoning.

> 💡 **Figure 1 批读**: 
> - **(A) Traditional MLLM**: ViT 提取特征 → Projector 对齐到 LLM embedding space → concat 到 text tokens → 一次性前向解码。这是"System 1"快思考范式。
> - **(B) HIVE**: 在 Huginn 的三元结构（Embedding Blocks E → Recurrent Block R × N 次迭代 → Language Head H）基础上，加入 ViT 并从其不同深度层级提取多尺度特征，通过 Patch Merger 投影后在 R-Block 的前几次迭代中按"浅层→深层"顺序注入。
> - 核心差异：推理不只发生在一次 forward pass 中，而是通过同一组参数的多次递归计算逐步精炼。视觉信息不是"一次性喂入"，而是"分层渐次注入"。

### Table 1 (Latent Space Reasoning Approaches Comparison)

> **Table 1 原始文本**:
> | Method | Visual | Text | Hierarchical | CoT Data Requirement |
> |--------|--------|------|-------------|---------------------|
> | **Training-induced Recurrence** | | | | |
> | Coconut | X | √ | √ | High |
> | Heima | √ | √ | X | High |
> | **Loop Transformer Recurrence** | | | | |
> | Huginn | X | √ | √ | Low |
> | **HIVE (Ours)** | √ | √ | √ | Low |

> 💡 **Table 1 批读 — 潜空间推理方法谱系**:
>
> | 维度 | 训练诱导型 (Coconut/Heima) | 结构递归型 (Huginn/HIVE) |
> |------|--------------------------|------------------------|
> | 机制 | 行为层面模拟：在训练时逐步将显式 CoT 替换为 continuous thought token，模型"学会"在潜空间中思考 | 架构层面支持：loop transformer 天然具备递归能力，不依赖 CoT 数据即可迭代 refine hidden state |
> | CoT 数据依赖 | High（需要大量 CoT 标注数据来训练压缩和替换过程） | Low（不需要显式 CoT 数据，推理能力来自结构递归本身） |
> | 视觉模态 | Heima 是首个加入的多模态，但视觉信息仅作为压缩目标的一部分，未真正集成到推理机制中 | HIVE 将层级视觉特征直接注入 recurrent block，视觉信息驱动推理而非仅作为被压缩对象 |
> | 层级特征 | Coconut 和 Heima 均不支持 | HIVE 首次从 ViT 多层提取特征并分层注入 |

---

## 三、Summary

- **核心问题**: MLLM 的推理仍停留在"快思考"范式（单次前向），显式 CoT 低效、冗长且易幻觉
- **核心主张**: 稳健推理应在潜空间内演进并无缝融合多模态信号
- **核心方案**: HIVE = Loop Transformer（结构递归）× 层级视觉线索注入（多尺度感知）
- **核心优势**: 不依赖 CoT 文本监督，视觉信息驱动推理而非仅作为输入，推理深度可动态调整
- **方法定位**: Table 1 中的唯一方法——同时具备 loop transformer 结构递归、多模态输入、层级视觉特征三者
