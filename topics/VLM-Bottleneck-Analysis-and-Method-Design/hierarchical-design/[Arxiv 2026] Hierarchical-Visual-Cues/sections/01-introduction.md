[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

Introduction 沿着"大模型演进 → System1/System2 → 潜空间推理的两种路径 → 本文定位"的逻辑线展开。核心洞察：现有 MLLM 的"慢思考"方案（LLaVA-CoT, Vision-R1, Mulberry）都依赖显式文本 rationale 作为推理 scaffold，这可能带来语言偏置和幻觉风险。HIVE 另辟蹊径——在结构递归（loop transformer）的基础上通过层级视觉注入来实现视觉驱动的潜空间推理，不依赖 CoT 文本。

---

## 二、原始文本

The rapid evolution of large-scale pre-trained models (Brown et al., 2020; Kaplan et al., 2020) has fundamentally transformed the landscape of artificial intelligence. Beginning with breakthroughs in natural language processing, models such as GPT-3 (Brown et al., 2020) demonstrated unprecedented capabilities in understanding and generating human-like text. This progress soon expanded into the multimodal domain, where systems like GPT-4 (OpenAI, 2023) and Qwen-VL (Wang et al., 2024b) have set new benchmarks by integrating and aligning information across vision, language, and beyond. These Multimodal Large Language Models (MLLMs) (OpenAI, 2023; Liu et al., 2023d) excel in multimodal tasks such as visual question answering, image captioning, and cross-modal retrieval. Their success marks a paradigm shift from unimodal intelligence toward more holistic, human-like understanding, enabling richer interactions and more robust applications in real-world scenarios.

> 💡 **背景铺垫**: 从 GPT-3（纯语言）→ GPT-4/Qwen-VL（多模态）的范式演进。这部分是标准的 Introduction 开篇，无新信息，但为后续"但 MLLM 的推理仍不够深"做铺垫。

Building upon these advancements, the prevailing reasoing paradigm in most existing MLLMs can be characterized as a form of "System 1" or fast thinking which is a rapid, intuitive, and associative processing of multimodal inputs to generate direct, end-to-end responses. While effective for many pattern recognition and simple descriptive tasks, this single-step generation process often struggles with complex, compositional questions that demand deeper logical inference, sequential deliberation, or multi-faceted analysis. This limitation has spurred recent research aimed at instilling models with "System 2" or slow thinking capabilities, which involve explicit, structured, and often iterative reasoing steps. Representative efforts in this direction include LLaVA-CoT (Xu et al., 2025b), which applies chain-of-thought prompting to visual questions; Vision-r1 (Huang et al., 2025), which uses outcome-supervised rewards to incentivize faithful reasoing; and Mulberry (Yao et al., 2024), which employs collective tree search for deliberate planning. These works collectively highlight a critical shift towards more deliberate reasoing processes.

> 💡 **机制拆解 — System1 → System2 的过渡**:
>
> | 范式 | 特征 | 代表方法 | 问题 |
> |------|------|---------|------|
> | System 1（快思考） | 单次前向，端到端生成 | 大多数基础 MLLM | 复杂组合问题、多步推理吃力 |
> | System 2（慢思考） | 显式、结构化、迭代推理 | LLaVA-CoT, Vision-R1, Mulberry | 依赖显式文本 rationale，语言偏置 + 幻觉风险 |

However, a common reliance on explicitly generated textual rationales as the primary scaffold for "slow thinking" may introduce inefficiencies and remain susceptible to the very language biases and hallucinatory pitfalls that deeper reasoing seeks to mitigate. This highlights the need for exploring more fundamental, latent, and modality-synchronized reasoing structures beyond surface-level linguistic chains. A series of works in LLMs focus on thinking in the latent space (Chen et al., 2025b; Hao et al., 2024; Li et al., 2025b), where the model performs computations entirely within its continuous hidden state. Recently, Heima (Shen et al., 2025) extends the latent-space reasoing paradigm in LLMs to multimodal settings by introducing a set of Heima Encoder/Decoders. During the training stage, the encoder learns to compress CoT into predefined tokens. At inference time, Heima employs independently trained decoders to decode these abstract compressed tokens. However, the reasoing process is still driven by textual CoT supervision, rather than being grounded in or induced by visual representations. As a result, visual information is not truly integrated into the model's reasoing mechanism.

> 💡 **问题动机 — 现有"慢思考"方案的根本缺陷**: 显式文本 rationale 作为推理 scaffold，本质上是把推理问题转化为文本生成问题——这会继承 LLM 的语言偏置和幻觉倾向。真正的"慢思考"应该发生在更深层、更 fundamental 的表示空间。

> 💡 **机制拆解 — Heima 的局限**: Heima 虽然是首个多模态潜空间推理方法，但其推理过程仍然由**文本 CoT 监督**驱动：
> 1. 训练阶段：Encoder 学习将显式 CoT 压缩进 predefined think tokens
> 2. 推理阶段：Decoder 将这些压缩 token 解码
> 3. 根本问题：视觉信息只是被压缩和存储，而非**驱动推理**。推理的信号来源仍然是文本 CoT，视觉信息未被真正集成到推理机制中。

> 💡 **逻辑链追踪 — 从 Heima 到 HIVE 的关键差异**:
> - Heima：visual as data to compress → reasoing driven by TEXT CoT → visual not in reasoing loop
> - HIVE：visual as driver of reasoing → hierarchical cues injected INTO recurrent blocks → visual IS the reasoing signal

In this work, we propose HIVE, the first multimodal latent reasoing framework that enhances reasoing capability through recursive extension of transformer blocks and hierarchical injection of visual cues, as illustrated in Figure 1. The core of our framework is a looped transformer architecture, which performs test-time scaling via recurrent blocks, enabling iterative refinement of latent representations. We incorporate hierarchical visual information, from coarse scene-level semantics to fine-grained regional details, into the latent space of the model. This design demonstrates that multi-scale visual cue injection remains effective when integrated with loop-transformer-based latent reasoing. Our contributions in this paper are summarized as follows:

- We propose HIVE, the first MLLM that leverages loop transformers to enable recursive reasoing within the latent space, moving beyond the limitations of purely feed-forward architectures.
- We introduce hierarchical visual cue injection into the recurrent blocks, which allows the model to perform iterative reasoing guided by structural visual information.
- Extensive evaluations demonstrate that test-time scaling remains effective when incorporating vision knowledge, and that hierarchical visual cue injection works effectively within the loop-transformer framework on complex scene understanding.

> 💡 **贡献精读 — 三个贡献的层级关系**:
> 1. **架构贡献**（Architecture）：首个将 loop transformer 用于 MLLM 潜空间推理——这是从 0 到 1
> 2. **方法贡献**（Method）：层级视觉注入策略——这是从 1 到 N（在 loop transformer 框架上叠加视觉能力）
> 3. **实证贡献**（Empirical）：验证 test-time scaling 在多模态场景下依然有效 + 层级视觉注入在 loop 框架中 works

---

## 三、Summary

- **范式问题**: 现有 MLLM 的 System2 方案（LLaVA-CoT/Vision-R1/Mulberry）依赖显式文本 rationale，继承语言偏置
- **Heima 的局限**: 首个多模态潜空间推理，但推理仍由文本 CoT 监督驱动，视觉信息未被真正集成
- **本文方案**: HIVE = Loop Transformer（结构递归）+ 层级视觉注入（多尺度感知驱动推理）
- **三个贡献**: 架构 × 方法 × 实证
- **独特定位**: 目前唯一将 loop transformer + 多模态 + 层级视觉三者结合的方法
