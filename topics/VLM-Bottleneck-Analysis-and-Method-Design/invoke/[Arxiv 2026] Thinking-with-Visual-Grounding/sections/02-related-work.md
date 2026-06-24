[← 返回 README](../README.md)

# 2. Related Work

## 一、Preview

Related Work 按时间线梳理了 visually grounded thinking 的演进脉络：从早期的 "region-of-interest selection"（选择关键区域）到 "tightly coupled grounding + reasoning"（grounding 与推理链深度绑定），再到 "grounding as active behavior via RL"（通过强化学习训练动态 grounding 行为）。本文在第三条线上进一步加入 explicit grounding reward。

---

## 二、原始文本

Early work on visually grounded thinking mainly uses grounding to locate the image regions needed for answering a question. Visual CoT (Shao et al., 2024a) introduces intermediate bounding boxes that highlight key regions, while UV-CoT (Zhao et al., 2025) reduces the need for human box annotations by learning from preferences over model-generated regions.

> 💡 **第一阶段: Region-of-Interest Selection**
>
> - **Visual CoT**: 在推理链中插入中间 bounding box 来高亮关键区域。本质是 "show where to look"，但 box 和 reasoning text 的关系仍然松散
> - **UV-CoT**: 通过偏好学习 (preference optimization) 减少对人工 box 标注的依赖。但 region selection 仍与推理步骤相对独立

Later work more tightly couples grounding with the reasoning trace. GCoT (Wu et al., 2025), Xia et al. (2025), and Argus (Man et al., 2025) generate grounding coordinates as step-level visual evidence, aiming to make the reasoning more faithful to the image and easier to check.

> 💡 **第二阶段: Grounding + Reasoning 深度耦合**
>
> - **GCoT** (Grounded Chain-of-Thought): 将 grounding 坐标直接作为推理步骤的 visual evidence，目标是让推理 "faithful to the image"
> - **Argus**: Vision-centric reasoning with grounded CoT，强调 grounding 使推理 "easier to check"
> - 这阶段的核心理念转变: grounding 不只是辅助定位，而是推理的证据基础

More recent work further treats grounding as an active behavior: GRIT (Fan et al., 2025) and ViGoRL (Sarch et al., 2025) train models to interleave natural language with visual coordinates through RL, and VGR (Wang et al., 2025) uses predicted regions for visual replay during inference.

> 💡 **第三阶段: Grounding as Active Behavior (via RL)**
>
> - **GRIT**: Teaching MLLMs to think with images -- 通过 RL 训练模型在思考中交错自然语言和视觉坐标
> - **ViGoRL**: Grounded reinforcement learning for visual reasoning -- 将 RL 用于训练 grounding 行为
> - **VGR**: Visual Grounded Reasoning -- 利用预测的 region 在推理时做 visual replay
> - 共同趋势: grounding 从 "静态标注" 变成 "模型主动生成的动态行为"

Our work follows this shift from region-of-interest selection to visually grounded thinking, and extends it with an explicit grounding reward that directly scores the visual grounding produced during thinking.

> 💡 **本文的定位**:
>
> ```
> Region Selection  →  Coupled Grounding+Reasoning  →  Active RL Grounding
> (Visual CoT,        (GCoT, Argus, Xia et al.)        (GRIT, ViGoRL, VGR)
>  UV-CoT)                                                  │
>                                                           │  + Explicit Grounding Reward
>                                                           ▼
>                                                   This Work:
>                                                   Visually Grounded Thinking
>                                                   + Box IoU / Point F1 Reward
> ```
>
> **本文的核心增量**:
> 1. **Explicit grounding reward**: 不是只让模型"学会产生 grounding"，而是在 RL 中直接评估 grounding 的质量（和 ground truth 的几何匹配度）
> 2. **Box vs Point 的对比分析**: 在受控条件下比较两种 grounding 接口的差异化优势
> 3. **SAM3-based 自动合成**: 不需要人工标注的 scalable pipeline

---

## 三、Summary

- **演进脉络**: Region selection → Coupled grounding+reasoning → Active RL grounding → +Explicit grounding reward
- **本文增量**: 在 GRIT/ViGoRL 的 active grounding RL 基础上，加入 explicit grounding quality reward (box IoU / point F1)，直接监督 rollout 中 grounding 的几何精度
- **差异化设计**: Box 和 Point 两种 grounding 接口的对比研究是之前工作没有的
