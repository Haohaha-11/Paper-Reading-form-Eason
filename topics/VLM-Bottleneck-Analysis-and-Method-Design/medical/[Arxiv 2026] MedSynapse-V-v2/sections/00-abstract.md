[← 返回 README](../README.md)

# Abstract

## 一、论文信息速览

| 项目 | 内容 |
|------|------|
| **标题** | MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution |
| **作者** | Chunzheng Zhu, Jiaqi Zeng, Junyu Jiang, Jianxin Lin*, Yijun Wang |
| **单位** | Hunan University, Changsha, China |
| **发表** | arXiv 2026 |
| **关键词** | VLMs, Implicit Diagnostic Memory, Latent Space Memory, Causal Counterfactual, Memory Distillation |

---

## 二、原始文本

High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose MedSynapse-V, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model's hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR) which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privileged-autonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets demonstrate that MedSynapse-V, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-the-art methods, particularly Chain-of-Thought (CoT) paradigms, in diagnostic accuracy and multi-dataset generalization without compromising the inference efficiency of standard VLMs.

> **一句话概括**: MedSynapse-V 通过三阶段渐进式 latent memory evolution（解剖先验压缩 -> 因果反事实精炼 -> 特权-自主双分支蒸馏），将临床专家的"隐性诊断记忆"激活机制建模为模型隐空间中的动态记忆合成与演化过程，在保持标准 VLM 推理效率的同时显著超越 CoT 等 SOTA 方法的诊断精度。

---

![Figure 1](../images/38265a02629d8413a4f7022ef1aafe8f2320fedc0ac407edfa41c5fbd72835c9.jpg)

*Fig. 1: Existing medical VLMs suffer from coarse symbolic granularity and long-range information dissipation in discrete reasoing. MedSynapse-V addresses this by evolving diagnostic implicit memory in latent space via anatomical prior condensation, causal counterfactual refinement, and autonomous latent memory internalization.*

> **Figure 1 批读**: 这张图是论文的核心 motivation 图示。左侧展示了现有医学 VLM 在离散 token 空间中进行推理的两个根本性问题：(1) **粗粒度符号表示**——固定词汇表无法精确表示病灶密度渐变、纹理异质性等连续病理特征；(2) **长程信息消散**——自回归解码导致视觉证据在长推理链中逐步衰减。右侧展示了 MedSynapse-V 的方案：在连续隐空间中通过 (A) 解剖先验压缩、(B) 因果反事实精炼、(C) 自主隐记忆内化三个阶段来 evolve 诊断记忆，最终实现对临床直觉的模拟。

> **问题动机**: 资深诊断专家在解读影像时并非进行逐步逻辑推理，而是激活"隐性诊断记忆 (Implicit Diagnostic Memory)"——一种基于积累病例知识的近乎即时的模式识别能力。然而现有医学 VLM 依赖离散 token 的自回归生成，导致三个层面的认知错位 (Cognitive Misalignment)：(1) **量化损失**——连续病理特征被离散符号粗粒度近似；(2) **长程信息消散**——视觉证据在长推理链中逐步衰减；(3) **缺乏病例自适应专业能力**——离散符号偏向通用语言先验，而非动态解剖上下文。

---

## 三、Summary

- **核心问题**: 医学 VLM 的离散 tokenization 造成 Cognitive Misalignment——连续病理特征无法被固定词汇表充分表示，视觉证据在长推理中逐步消散，离散符号难以编码动态解剖上下文。
- **核心方案**: MedSynapse-V 三阶段框架：(I) Meta Query for Prior Memorization——从冻结解剖编码器中提取结构化先验，压缩为 diagnostic implicit memory；(II) Causal Counterfactual Refinement——用 RL + 因果反事实 reward 精炼 memory，剪除因果无关成分；(III) Intrinsic Memory Transition——双分支 JSD 蒸馏将 external memory 内化为 autonomous memory。
- **核心优势**: 将外部专业知识转化为内生参数，在推理时完全移除辅助编码器，以近乎标准 VLM 的开销实现超越 CoT 的诊断精度和多数据集泛化能力。
