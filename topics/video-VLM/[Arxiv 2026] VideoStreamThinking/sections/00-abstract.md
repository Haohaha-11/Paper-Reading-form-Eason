[← 返回 README](../README.md)

# 00 - Abstract

📌 **Preview**: 本文提出 Video Streaming Thinking (VST)，一种在视频流播放过程中边看边思考的范式。通过将推理成本摊销到查询前阶段，VST 在保持低响应延迟的同时实现显式 CoT 推理，并通过 VST-SFT 和 VST-RL 两阶段训使得 offline VideoLLM 适配流式场景。

---

## Abstract

Online Video Large Language Models (VideoLLMs) play a critical role in supporting responsive, real-time interaction. Existing methods focus on streaming perception, lacking a synchronized logical reasoning stream. However, directly applying test-time scaling methods incurs unacceptable response latency. To address this trade-off, we propose **Video Streaming Thinking (VST)**, a novel paradigm for streaming video understanding. It supports a *thinking while watching* mechanism, which activates reasoning over incoming video clips during streaming. This design improves timely comprehension and coherent cognition while preserving real-time responsiveness by amortizing LLM reasoning latency over video playback.

Furthermore, we introduce a comprehensive post-training pipeline that integrates **VST-SFT**, which structurally adapts the offline VideoLLM to causal streaming reasoning, and **VST-RL**, which provides end-to-end improvement through self-exploration in a multi-turn video interaction environment. Additionally, we devise an automated training-data synthesis pipeline that uses video knowledge graphs to generate high-quality streaming QA pairs, with an entity--relation grounded streaming Chain-of-Thought to enforce multi-evidence reasoning and sustained attention to the video stream.

Extensive evaluations show that VST-7B performs strongly on online benchmarks, e.g. **79.5%** on StreamingBench and **59.3%** on OVO-Bench. Meanwhile, VST remains competitive on offline long-form or reasoning benchmarks. Compared with Video-R1, VST responds **15.7x faster** and achieves **+5.4%** improvement on VideoHolmes, demonstrating higher efficiency and strong generalization across diverse video understanding tasks. Code, data, and models will be released at https://github.com/1ranGuan/VST.

**Keywords**: Streaming Video Understanding, CoT, VideoLLM

---

## Annotations

> 💡 **问题动机 批读**: 现有 online VideoLLM 的核心矛盾在于：流式感知仅处理了"看"的问题，却缺乏同步的"想"的能力。如果照搬 offline 场景中的 test-time scaling（即 query 后再 CoT），响应延迟不可接受。作者抓住了这个 trade-off，提出将 CoT 推理前移到视频播放阶段，本质上是"用空间（预推理时间）换时间（回答延迟）"。

> 💡 **机制拆解 批读**: Abstract 中明确了三项技术贡献的递进关系：
> 1. **VST Paradigm**（想法层面）：边看边思考，amortize 推理成本到播放阶段
> 2. **Post-training Pipeline**（训练层面）：VST-SFT 做格式对齐 + VST-RL 做端到端优化
> 3. **Data Synthesis**（数据层面）：基于 video knowledge graph 自动生成流式 QA 数据
> 这三层递进非常清晰：没有范式，训练无从谈起；没有数据，范式无法落地。

> 💡 **关键数字 批读**:
> - StreamingBench: 79.5%（超越 GPT-4o 的 73.3% 和 Gemini 1.5 pro 的 75.7%）
> - OVO-Bench: 59.3%（接近 GPT-4o 的 59.5%）
> - 对比 Video-R1: 速度快 15.7x，同时 VideoHolmes 高 5.4%
> - 这组数字说明：VST 不是以牺牲质量换取速度，而是质量更高、速度更快。

> 💡 **Q&A 批注记录**:
> - Q: VST 的"thinking while watching"和 human cognition 有何关联？
> - A: 作者在 Introduction 中引用了神经耦合（neural coupling）[16,36] 的研究——大脑在接收外部信息时，逻辑流与信息流入同步。VST 模仿了这一机制：在处理 incoming video clips 的同时生成 intermediate thoughts。

🔖 **Summary**: VST is a paradigm that introduces synchronized logical reasoning into streaming video understanding by front-loading CoT generation to the pre-query phase, achieving both strong reasoning performance and low response latency.
