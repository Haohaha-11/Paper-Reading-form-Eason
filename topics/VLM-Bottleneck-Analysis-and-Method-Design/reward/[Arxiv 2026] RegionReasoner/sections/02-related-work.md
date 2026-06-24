[← 返回 README](../README.md)

# 2. Related Work

## 一、Preview

本文的相关工作分成三条线：(1) VLM 的 post-training 技术（指令微调 + RL）；(2) 面向多模态推理的 RL 方法；(3) 多轮视觉理解。每条线都指出了与 RegionReasoner 的核心差异——要么是单轮/纯文本推理，要么缺少显式空间 grounding 和多轮一致性机制。

---

## 二、原始文本

**Post-training for vision-language models.** Post-training techniques, including instruction tuning and reinforcement learning (RL), have become essential for adapting large Vision-Language Models (VLMs) to complex multimodal reasoning tasks. Early efforts such as LLaVA, LLaVA-OV, Infinity-MM, MAmmoTH-VL, LISA, PixelLM, and GLAMM demonstrate that scaling instruction-tuning datasets and diversifying task formats can significantly improve generalization across multimodal benchmarks. More recent work, such as VL-Rethinker, further explores post-training for reasoning, introducing techniques like selective sample replay to address instability in RL optimization. Unlike these approaches, which mainly focus on single-pass or text-only reasoning, our work enforces explicit spatial grounding and global--local consistency within multi-round visual reasoning.

> 💡 **与 Post-training 工作的差异**:
> - 早期 post-training (LLaVA, LISA, PixelLM 等): 单轮/纯文本推理，不涉及显式空间 grounding
> - VL-Rethinker: 虽然用 RL 做推理后训练，但 focus 是单轮推理的稳定性优化
> - **RegionReasoner 的独特之处**: 在**多轮**视觉推理中同时强制执行**显式空间 grounding** 和**全局-局部语义一致性**

**Reinforcement learning for multimodal reasoning.** RL has emerged as a powerful tool for enhancing the reasoning and decision-making of VLMs. Vision-R1 and Video-R1 integrate RL to improve spatial grounding and temporal reasoning, respectively, while VLM-R1 applies RL to fine-grained grounding tasks. Pixel Reasoner further incentivizes pixel-space reasoning with curiosity-driven exploration. Visionary-R1 mitigates shortcut behaviors in visual reasoning with explicit RL signals, and the Self-Rewarding VLM adopts a reasoning-decomposition strategy where the model first generates image captions before deriving answers. Other efforts, such as OpenVLThinker and LMM-R1, adopt policy optimization methods like PPO to train VLMs as interactive decision-makers. Despite these advances, most RL-based approaches focus on single-pass reasoning or rely on textualized visual inputs, limiting their ability to enforce explicit spatial grounding or multi-step consistency. In contrast, RegionReasoner leverages RL to jointly optimize multi-round reasoning accuracy, region-level grounding fidelity, and global--local semantic alignment, providing a more structured training signal than prior RL-based methods.

> 💡 **机制拆解 — RL for Multimodal Reasoning 全景对比**:
>
> | 方法 | RL 目标 | 空间 Grounding | 多轮 | 本文差异 |
> |------|---------|---------------|------|---------|
> | Vision-R1 | 空间定位 | 有 (单轮) | 无 | 本文: 多轮 + 引用传播 |
> | Video-R1 | 时序推理 | 视频时序 | 时序非对话 | 本文: 对话式多轮 |
> | VLM-R1 | 细粒度 grounding | 有 (单轮) | 无 | 本文: 多轮交互 |
> | Pixel Reasoner | 像素空间推理 | curiosity-driven | 无 | 本文: explicit citation |
> | Visionary-R1 | 反 shortcut | 显式 RL 信号 | 无 | 本文: consistency |
> | Self-Rewarding VLM | caption→answer 分解 | 无 | 无 | 本文: 结构化 4-tag |
> | OpenVLThinker / LMM-R1 | PPO 交互决策 | 文本化视觉输入 | 无 | 本文: 区域级 grounding |
>
> **核心区别**: 现有 RL 方法聚焦于单轮或文本化输入，缺少显式空间 grounding 和多步一致性机制。RegionReasoner 通过联合优化多轮推理精度 + 区域级定位保真度 + 全局-局部语义对齐，提供了比已有方法更强的结构化训练信号。

**Multi-round visual understanding.** SegLLM explores multi-round interaction for referring segmentation and shows the value of dialogue-style supervision and evaluation, but it does not model explicit reasoning trajectories or incorporate RL signals, making it difficult to verify evidence use or enforce global--local semantic coherence. VisionReasoner provides structured, reward-shaped perception--reasoning in a single-turn setting without reference propagation across rounds. In this context, SegLLM also releases a multi-round segmentation benchmark; our RegionDial-Bench complements it by adding explicit reasoning-oriented design and per-turn evaluation for both referring detection and referring segmentation, enabling analysis of reasoning accuracy, grounding fidelity, and global--local alignment under iterative interaction.

> 💡 **与 SegLLM 和 VisionReasoner 的定位**:
> - **SegLLM**: 多轮交互分割，但无显式推理轨迹、无 RL 信号、无 global--local 一致性机制、限于分割任务
> - **VisionReasoner**: 单轮结构化推理，有 base rewards，但无跨轮引用传播
> - **RegionReasoner**: 填补两者间的空白——在多轮设定中引入显式推理 + 引用传播 + 一致性 reward + RL 优化
> - **RegionDial-Bench vs SegLLM benchmark**: SegLLM 的 benchmark 只覆盖分割；RegionDial-Bench 同时覆盖检测和分割，且包含推理导向的设计和逐轮评估

---

## 三、Summary

- **Line 1 (Post-training)**: 单轮或纯文本推理为主 → RegionReasoner 引入多轮 + 显式空间 grounding
- **Line 2 (RL for MLLM)**: RL 用于单轮定位/推理 → RegionReasoner 联合优化多轮推理精度 + grounding 保真度 + 语义对齐
- **Line 3 (Multi-round)**: SegLLM (无推理过程) + VisionReasoner (无多轮) → RegionReasoner 填补中间空白
