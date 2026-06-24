[← 返回 README](../README.md)

# Abstract

## 一、Preview

本文摘概括了核心问题、方法与结果：RLVR 的 outcome-level supervision 过于粗粒度，难以诊断和纠正推理链内的错误。Perceval 通过 token-level error grounding 来检测 image-text misalignment，并将检测结果注入 GRPO 的 token-level advantage 重分配，同时支持 test-time 迭代精炼。

---

## 二、原始文本

### Abstract

Recent advancements in reinforcement learning with verifiable rewards (RLVR) have significantly improved the complex reasoning ability of vision-language models (VLMs). However, its outcome-level supervision is too coarse to diagnose and correct errors within the reasoning chain. To this end, we propose Perceval, a process reward model (PRM) that enables token-level error grounding, which can extract image-related claims from the response and compare them one by one with the visual evidence in the image, ultimately returning claims that contain perceptual errors. Perceval is trained with perception-intensive supervised training data. We then integrate Perceval into the RL training process to train the policy models. Specifically, compared to traditional GRPO, which applies sequence-level advantages, we apply token-level advantages by targeting penalties on hallucinated spans identified by Perceval, thus enabling fine-grained supervision signals. In addition to augmenting the training process, Perceval can also assist VLMs during the inference stage. Using Perceval, we can truncate the erroneous portions of the model's response, and then either have the model regenerate the response directly or induce the model to reflect on its previous output. This process can be repeated multiple times to achieve test-time scaling. Experiments show significant improvements on benchmarks from various domains across multiple reasoning VLMs trained with RL, highlighting the promise of perception-centric supervision as a general-purpose strategy. For test-time scaling, it also demonstrates consistent performance gains over other strategies, such as major voting. Our code and data will be publicly released at https://github.com/RUCAIBox/Perceval.

> 💡 **一句话概括**: Perceval 是一个以感知为中心的过程奖励模型 (PRM)，能做三件事：训练时为 GRPO 提供 token-level 的精细信用分配（惩罚幻觉 span），推理时驱动截断-重生成的 test-time scaling。

> 💡 **机制拆解 — Perceval 的核心工作流**:
>
> ```
> [Policy 模型生成回复 $o_{i}$]
>         ↓
> [Perceval 提取 image-related claims，逐条与图像比对比对]  ← 核心步骤：claim-by-claim 校验
>         ↓
> [输出包含感知错误的 exact substrings]
>         ↓
>     ┌───────────────┬────────────────┐
>     ↓ (训练阶段)     ↓ (推理阶段)      ↓
> [Token-level mask   [Truncate-then-    Reflection-guided
>  构建 → advantage    Regenerate 迭代]   regeneration]
>  重分配 → GRPO]
> ```
>
> 关键设计：Perceval 不做粗粒度打分，而是做精确的错误提取和定位。这使它能同时服务于训练和推理，比传统标量 PRM 的"好坏判断"更进一步——它告诉模型"哪里错了"。

> 💡 **问题动机**: RLVR 的核心矛盾——outcome-level supervision 只能告诉模型"最终答案对不对"，无法知道"推理链中哪一步的哪个感知出了问题"。这导致了一个 hard credit-assignment problem：一个包含 200 个 token 的推理链中只有 3 个 token 是幻觉，但 GRPO 对所有 token 分配相同的 advantage。本文的方案正是将这种"均等信用分配"改为"精确责任追责"。

---

## 三、Summary

- **核心问题**: RLVR 的 outcome-level supervision 过于粗粒度，无法在推理链内做误差定位和信用分配。
- **核心方案**: Perceval (Perception-centric PRM) — 从模型回复中提取每个 image-related claim，与视觉证据逐一比对，输出包含错误的 exact substrings。
- **训练阶段使用**: 将错误 span 转化为 token-level mask，对 GRPO 的 advantage 做重分配 → 精确惩罚幻觉 token。
- **推理阶段使用**: Truncate-then-Regenerate 迭代精炼策略，截断错误部分后重新生成。
- **核心优势**: 粗粒度监督 (outcome-level) → 细粒度监督 (token-level)，一模型两用（训练+推理）。
