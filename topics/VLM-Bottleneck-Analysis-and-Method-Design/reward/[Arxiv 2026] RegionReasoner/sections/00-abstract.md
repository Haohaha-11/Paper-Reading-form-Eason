[← 返回 README](../README.md)

# Abstract & Figure 1

## 一、论文信息速览

| 项目 | 内容 |
|------|------|
| **标题** | RegionReasoner: Region-Grounded Multi-Round Visual Reasoning |
| **作者** | Wenfang Sun\*1, Hao Chen\*2, Yingjun Du1, Yefeng Zheng†3, Cees G. M. Snoek1 |
| **单位** | 1 University of Amsterdam, 2 Anhui University, 3 Westlake University |
| **发表** | arXiv 2026 |
| **\*** | Equal contribution. † Corresponding author. |

---

## 二、原始文本

**Abstract**: Large vision-language models have achieved remarkable progress in visual reasoning, yet most existing systems rely on single-step or text-only reasoning, limiting their ability to iteratively refine understanding across multiple visual contexts. To address this limitation, we introduce a new multi-round visual reasoning benchmark with training and test sets spanning both detection and segmentation tasks, enabling systematic evaluation under iterative reasoning scenarios. We further propose RegionReasoner, a reinforcement learning framework that enforces grounded reasoning by requiring each reasoning trace to explicitly cite the corresponding reference bounding boxes, while maintaining semantic coherence via a global--local consistency reward. This reward extracts key objects and nouns from both global scene captions and region-level captions, aligning them with the reasoning trace to ensure consistency across reasoning steps. RegionReasoner is optimized with structured rewards combining grounding fidelity and global--local semantic alignment. Experiments on detection and segmentation tasks show that RegionReasoner-7B, together with our newly introduced benchmark RegionDial-Bench, considerably improves multi-round reasoning accuracy, spatial grounding precision, and global--local consistency, establishing a strong baseline for this emerging research direction.

> 💡 **一句话概括**: RegionReasoner 提出了一种面向多轮视觉推理的 RL 框架，核心是两条互补奖励信号：强制推理显式引用参考框坐标（reference citation）和全局-局部语义对齐（global--local consistency），在检测和分割任务上大幅提升多轮推理的空间定位精度和语义连贯性。

---

![](../images/f0b08bb74fad6be4feb4d53a054a17b8a77baac7ece303a1b234e232fb98566b.jpg)

*Figure 1: RegionReasoner in a three-round, region-grounded dialogue. At round t, the user query may refer to a region localized earlier (R1/R2). For each turn, RegionReasoner produces a structured trajectory: <scene> (global context), <focus> (caption restricted to the referenced region with serialized coordinates, e.g., bbox=[x1,y1,x2,y2]), <think> (reasoning that explicitly cites the reference and the required spatial relation), and <answer> (final localization).*

> 💡 **Figure 1 批读**: 这张图是理解 RegionReasoner 最关键的一页。展示了一个三轮对话的例子：
>
> **每轮输出的结构化轨迹**:
> - `<scene>`: 全局场景描述
> - `<focus>`: 参考区域的局部描述（包含序列化坐标 bbox=[x1,y1,x2,y2]）
> - `<think>`: 推理过程，**必须显式引用参考框和所需的空间关系**（这是与 VisionReasoner 的核心区别之一）
> - `<answer>`: 最终定位输出的 JSON（如 bbox 坐标或分割 points）
>
> **跨轮参考传播**: Round 1 定位的物体 (R1/R2) 被后续轮次作为 reference bbox 引用，形成 "behind the R1 on the left"、"next to the R2" 等空间关系查询。核心设计理念是：**显式引用 + 全局-局部一致性 → 随对话深入保持定位稳定**。

> 💡 **问题动机**: 现有 VLM 推理系统依赖单步或纯文本推理，无法在多视觉上下文中迭代精炼理解。核心挑战在于：(1) 跨轮定位误差如何避免累积？(2) 推理的语义一致性如何随着对话上下文增长而保持？RegionReasoner 通过结构化推理轨迹 (4-tag) + 两条互补奖励信号来解决。

---

## 三、Summary

- **核心问题**: 多轮视觉推理中的跨轮参考传播不稳定和语义漂移
- **核心方案**: RegionReasoner = Structured 4-tag Trajectory + Reference Citation Reward + Global--Local Consistency Reward
- **核心基准**: RegionDial-Bench (RefCOCO+ Multi-turn + RefCOCOg Multi-turn, 检测 + 分割)
- **核心结果**: 显著提升多轮推理精度，尤其在后期轮次（R5-R7）增益最大
