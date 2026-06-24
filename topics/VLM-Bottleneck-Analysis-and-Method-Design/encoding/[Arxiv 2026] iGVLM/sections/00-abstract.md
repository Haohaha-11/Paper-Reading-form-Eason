[← 返回 README](../README.md)

# Abstract

## 一、Preview

本文指出当前 LVLM 的核心瓶颈：静态、指令无关的视觉编码器导致视觉表征在不同文本任务中被无差别复用，限制了细粒度推理。为此提出 iGVLM —— 解耦双分支架构，冻结分支保留任务无关的视觉先验，动态分支通过 AdaLN 实现指令引导的视觉特征调制。同时引入 MM4 诊断基准，用于量化多查询多指令场景下的逻辑一致性。

---

## 二、原始文本

Despite the success of Large Vision–Language Models (LVLMs), most existing architectures suffer from a representation bottleneck: they rely on static, instruction-agnostic vision encoders whose visual representations are utilized in an invariant manner across different textual tasks. This rigidity hinders fine-grained reasoning where taskspecific visual cues are critical. To address this issue, we propose iGVLM, a general framework for instruction-guided visual modulation. iGVLM introduces a decoupled dual-branch architecture: a frozen representation branch that preserves taskagnostic visual representations learned during pretraining, and a dynamic conditioning branch that performs affine feature modulation via Adaptive Layer Normalization (AdaLN). This design enables a smooth transition from general-purpose perception to instruction-aware reasoning while maintaining the structural integrity and stability of pre-trained visual priors. Beyond standard benchmarks, we introduce MM4, a controlled diagnostic probe for quantifying logical consistency under multi-query, multi-instruction settings. Extensive results show that iGVLM consistently enhances instruction sensitivity across diverse language backbones, offering a plug-andplay paradigm for bridging passive perception and active reasoning.

> 💡 **一句话概括**: iGVLM 提出解耦双分支架构（冻结静态分支 + 动态指令调制分支），通过 AdaLN 实现指令引导的视觉特征调制，使视觉表征从"被动感知"平滑过渡到"主动推理"，同时保持预训练视觉先验的稳定性。此外提出的 MM4 基准专门评测多查询多指令下的逻辑一致性。

> 💡 **核心洞察 — 三个关键词**:
> 1. **Representation Bottleneck**: 现有 LVLM 的视觉编码器是静态的、与指令无关的，同一张图无论问什么，视觉特征都一样。这是本文要解决的核心问题。
> 2. **Decoupled Dual-Branch**: 不是替换静态编码器，而是"保留 + 增强"——冻结分支保留预训练感知能力，动态分支负责指令条件化调制。关键设计理念：分离（separation）而非替代（replacement）。
> 3. **Smooth Transition**: 从通用感知到指令感知不是二值的开关切换，而是通过 Zero-FFN（零初始化的线性投影）实现的渐进式、平滑过渡。

> 💡 **问题动机拆解**: 现有 LVLM 的视觉编码（如 CLIP-ViT）是在"看图说话"任务上预训练的，学到的是任务无关的通用视觉特征。但在 VQA 场景中，同一个图像的不同问题需要关注不同的视觉细节。静态编码器的本质问题在于：它把"提取视觉特征"和"利用视觉特征"两个阶段强行分离了。iGVLM 的核心思路是打破这种分离：让"利用视觉特征"的方式受"指令"的调控。

---

## 三、Summary

| 维度 | 内容 |
|------|------|
| **核心问题** | 静态视觉编码器导致表征瓶颈：同一图像对不同问题产生相同视觉特征 |
| **核心方案** | 解耦双分支架构：冻结分支保留通用感知 + AdaLN 动态分支实现指令调制 |
| **关键创新点** | Zero-FFN 融合机制，实现从预训练先验到指令感知的平滑过渡 |
| **评测贡献** | MM4 基准：180 图 x 4 问 = 720 QA，评测多查询多指令下的逻辑一致性 |
| **核心优势** | plug-and-play，跨语言 backbone 泛化，微小推理开销 |
