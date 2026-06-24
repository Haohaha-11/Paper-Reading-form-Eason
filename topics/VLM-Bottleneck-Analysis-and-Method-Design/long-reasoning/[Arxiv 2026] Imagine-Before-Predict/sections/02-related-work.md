[← 返回 README](../README.md)

# 2. Related Work

## 一、Preview

Related Work 沿着三条脉络展开：(1) MLLM 的后训练推理能力发展——从识别到推理；(2) 潜空间推理范式——从 COCONUT 到多模态潜视觉推理；(3) VEP 任务的演变——从低层动作预测到高层语义推理。三条脉络的交汇点为本文的定位提供了清晰的学术坐标系。

---

## 二、原始文本

**Multimodal Large Language Models.** Multimodal large language models (MLLMs) connect visual encoders with strong LLM backbones and have become the mainstream framework for visual understanding (Bai et al., 2025a; Team et al., 2026; Hong et al., 2026; Xiao et al., 2026; An et al., 2026). For video understanding, recent MLLMs extend image-based models with temporal frame sampling, video instruction tuning, longer-context modeling, and large-scale video-text corpora (Wang et al., 2024; Zhang et al., 2024c; Wang et al., 2025a), substantially improving performance on diverse benchmarks (Li et al., 2024; Fu et al., 2024;Xu et al., 2025; Shi et al., 2026). Beyond perception and recognition, reasoning-oriented post-training has been applied to MLLMs, including chain-of-thought supervision (Han et al., 2025) and reinforcement learning (Li et al., 2025d). More recently, paradigms that encourage models to think with images or videos move beyond purely textual rationales by retrieving visual evidence (Zheng et al., 2025b; Zeng et al., 2026) with intermediate visual traces, motivating non-textual intermediate representations for visual reasoning.

> 💡 **脉络梳理 — MLLM 推理能力的演进轨迹**:
>
> | 阶段 | 代表工作 | 核心范式 | 推理媒介 |
> |------|---------|---------|---------|
> | 1. 基础视频理解 | InternVideo2, Qwen2.5-VL | 识别 + 描述 | 文本输出 |
> | 2. 视频推理后训练 | Video-R1, VideoChat-R1, VideoEspresso | SFT/RL 增强文本推理链 | 文本 CoT |
> | 3. 视觉辅助推理 | DeepEyes, Video-o3 | 在推理中检索/引用视觉证据 | 文本为主 + 视觉引用 |
> | 4. 非文本中间表示 | **(本文定位)** | 潜空间视觉推理 | 连续潜状态 |
>
> 本文处于第 4 阶段，与前人（visual retrieval / visual trace）的差异在于：**不显式生成图像，也不仅引用已有帧，而是在潜空间中内部化未来视觉信息**。

**Reasoning in Latent Space.** Latent reasoning (Yu et al., 2026b) replaces discrete textual reasoning tokens with continuous hidden states fed back into the LLM, compressing chain-of-thought into a compact thinking space. Coconut (Hao et al., 2024) first showed that an LLM can reason in its own embedding space, and CODI (Shen et al., 2025) and SIM-CoT (Wei et al., 2025) subsequently distilled or supervised these latent steps to close the gap to explicit textual CoT. This paradigm has also been adopted by MLLMs through visual supervision: Mirage (Yang et al., 2025b) and LVR (Li et al., 2025a) align latent slots with embeddings of helper images that hint at the answer, and LaViT (Wu et al., 2026) further constrains latent visual thoughts with teacher-guided attention. More flexible designs allow models to alternate between textual tokens and continuous visual states during reasoning, as in Monet (Wang et al., 2025c), SkiLa (Tong et al., 2025), and SwimBird (Tong et al., 2026). However, these methods largely anchor latent thoughts to static images, such as helper images, sketches, or scenes already given to the model. Video event prediction instead requires reasoning over dynamic future frames that are not yet observed, where above studies have not explored. FUTURE-L1 accordingly grounds latent thoughts in future information rather than static visual hints.

> 💡 **机制拆解 — 潜空间推理的谱系与本文定位**:
>
> | 维度 | LLM 潜推理 | MLLM 潜视觉推理 | **FUTURE-L1** |
> |------|-----------|---------------|-------------|
> | 代表工作 | COCONUT, CODI, SIM-CoT | LVR, Mirage, Monet, SwimBird | **本文** |
> | 潜状态语义 | 压缩的文本推理 | 辅助图像/草图的视觉特征 | **未来帧的视觉语义** |
> | 锚定对象 | 文本 CoT 的步骤 | 静态图像 (already given) | **未观察到的未来帧 (future)** |
> | 时序结构 | 无（单次潜推理块） | 无（静态视觉提示） | **有（时序演变的多 span）** |
> | 推理任务 | 文本 QA, 数学 | 多模态 QA | **视频事件预测** |
>
> 核心区分点：所有前人的潜视觉推理都锚定在**已知/静态**的视觉信息上（辅助图像、已有帧、草图），而本文是**首次**将潜视觉思维锚定在**未来的、尚未观察到的**视觉信息上。这带来了独特的训练挑战（未来帧不可用于推理，只能作为训练监督信号）。

> 💡 **关键词对照**:
> - "helper image"：在训练时可用的额外视觉信息，用于提供答案线索——相当于给模型"偷看答案"
> - "future-frame embedding"：本文的锚定目标——在训练时可用（因为是从完整视频中取出的后续帧），但在推理时不可用（实际预测未来时看不到）
> - 这种"训练时有、推理时无"的监督信号不对称是本文训练设计的核心挑战

**Video Event Prediction.** Unlike standard video understanding benchmarks (Li et al., 2024; Fu et al., 2024; Liu et al., 2024) that focus on visible content, video event prediction requires models to infer unobserved future events from a video prefix. This future-oriented setting spans low-level action anticipation (Lan et al., 2014; Gammulle et al., 2019), future-frame prediction (Ranzato et al., 2014; Vondrick et al., 2016b), and high-level semantic next-event prediction (Lei et al., 2020; Jiang et al., 2025; Liang et al., 2025; Su et al., 2025). Most VEP methods remain text-output oriented (Cheng et al., 2025a; Wang et al., 2025b); for example, Video-CoE (Su et al., 2026) structures the reasoning trace as a long textual chain of historical events. Video-as-Answer (Cheng et al., 2025b) instead moves the answer modality from text to generated video explicitly. FUTURE-L1 differs from these routes: rather than verbalizing every intermediate event or synthesizing full videos, it represents intermediate future states in an interleaved latent visual channel supervised by future-frame embeddings.

> 💡 **机制拆解 — VEP 的三条技术路线与本文的差异**:
>
> | 路线 | 代表工作 | 输出模态 | 中间推理模态 | 局限性 |
> |------|---------|---------|------------|--------|
> | 文本推理 + 文本输出 | NEP, Video-CoE, TEMPURA | 文本 | 文本（事件链描述） | 文本化丢失视觉动态 |
> | 文本推理 + 视频输出 | Video-as-Answer | 生成的视频帧 | 文本 → 扩散模型 → 视频 | 像素生成计算代价极高 |
> | **潜视觉推理 + 文本输出** | **FUTURE-L1 (本文)** | **文本** | **潜空间视觉状态** | **潜状态不可直接解释** |
>
> 本文被定位为第三条路线：答案仍是文本（可评估、可对标），但中间推理使用潜空间（保留视觉语义、高效）。这是介于"全文本"和"全视频"之间的第三条路。

---

## 三、Summary

- **MLLM 脉络**: 识别 → 推理 (CoT/RL) → 视觉辅助推理 → 非文本中间表示 (本文位置)
- **潜空间推理脉络**: COCONUT (LLM 潜推理) → 静态潜视觉推理 (Mirage, LVR, Monet) → **动态未来潜视觉推理 (本文)**
- **VEP 脉络**: 低层预测 (action/frame) → 高层语义 (next-event) → 文本推理 (Video-CoE) → 视频生成 (Video-as-Answer) → **潜空间推理 (本文)**
- **关键区分**: 前人的潜视觉思维锚定在已知静态视觉，本文首次锚定在**未来/动态**视觉状态
