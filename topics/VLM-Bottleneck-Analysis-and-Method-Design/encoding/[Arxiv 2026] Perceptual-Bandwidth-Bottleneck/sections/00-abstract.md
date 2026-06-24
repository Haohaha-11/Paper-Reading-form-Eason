[← 返回 README](../README.md)

# Abstract

## 一、论文信息速览

| 项目 | 内容 |
|------|------|
| **标题** | The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design |
| **作者** | Anjie Liu\*, Ziqin Gong\*, Yan Song, Yuxiang Chen, Xiaolong Liu, Hengtong Lu, Kaike Zhang, Chen Wei, Jun Wang |
| **单位** | HKUST(GZ), UCL, ShanghaiTech, AI Lab Yangtze River Delta, Li Auto |
| **发表** | arXiv 2026 (2605.01345v3) |
| **代码** | https://github.com/iamlilAJ/active-vlm |

---

## 二、原始文本

Visual perception in modern Vision-Language Models (VLMs) is constrained by a perceptual bandwidth bottleneck: a broad field of view preserves global context but sacrifices the finegrained details required for complex reasoning. We argue that high-resolution visual reasoning is therefore not only semantic reasoning but also task-relevant evidence acquisition under limited perceptual bandwidth. Inspired by active vision and information foraging, we formalise this process as sequential Bayesian optimal experimental design (S-BOED), where an agent decides which visual evidence to acquire before answering. Since exact Bayesian inference is intractable in continuous gigapixel spaces, we derive a tractable coverage–resolution objective as a proxy for task-relevant information gain. We instantiate this framework with FOVEA, a trainingfree procedure that refines VLM crop proposals through evidence-oriented probing. Experiments on high-resolution benchmarks show consistent gains over direct and ReAct-style baselines, with particularly strong improvements in searchdominated remote-sensing settings.

> 💡 **一句话概括**: 将高分辨率 VLM 推理重新定义为"感知带宽约束下的主动视觉证据采集"问题，用 S-BOED 框架形式化，推导出 Coverage-Resolution 乘积作为可计算的 crop 选择目标，实例化为 training-free 的 FOVEA 推理框架。

---

![Figure 1](../images/6cf1d798eec2fff88dd1768ad8da8b83d492a5a7dcc2ecbc1b9c7ec248ff175e.jpg)

*Figure 1: S-BOED-guided active visual reasoning. Under the perceptual bandwidth bottleneck, FOVEA iteratively refines VLM crop proposals to acquire task-relevant evidence. Candidate crops are scored by a coverage–resolution utility estimated through resolvability probing, and selected views update the interaction history for subsequent search.*

> 💡 **Figure 1 批读**: 该图展示了 FOVEA 的完整推理管线。核心循环：VLM 提出初始 crop proposal → FOVEA 生成候选池 → 通过 resolvability probing 为每个候选打分 → 选择 utility 最高的 crop 执行 → 新观察加入交互历史 → 下一轮搜索基于更新后的历史重新调整空间信念。关键洞察：FOVEA 将 VLM 的初始 crop 视为"噪声空间先验 (noisy spatial prior)"而非 ground truth，在周围采样候选并进行 evidence-oriented 的验证，而非盲目信任 VLM 的第一建议。

> 💡 **问题动机 （双核驱动）**: 
> 1. **感知带宽瓶颈**: ViT encoder 将任意分辨率图像投影到固定 token 数（如 576 个），全局编码时每个 token 承载过大的 spatial 面积，导致小目标/文字在进入 LLM 之前就"消失"了。
> 2. **被动感知 vs 主动信息采集**: 现有 VLM 是"被动观察者"——给定一张固定分辨率的图像，一次性编码并回答。但高分辨率推理需要 agent "主动决定在哪里花费宝贵的视觉带宽"来采集任务相关证据。这是一类 information foraging 问题。

> 💡 **核心逻辑链**: 

Perceptual Bandwidth Bottleneck  →  需要主动信息采集  →  形式化为 S-BOED (decision-theoretic)  →  EIG 计算 intractable  →  推导 Coverage-Resolution Objective (tractable proxy)  →  实例化 FOVEA (training-free)  →  实证验证

---

## 三、Summary

- **核心问题**: VLM 的固定 token budget 造成 field-of-view vs. resolution 的 trade-off，高分辨率推理需要主动的证据采集策略而非被动编码。
- **核心方案**: S-BOED 形式化 → Coverage-Resolution Objective (proxy for EIG) → FOVEA (training-free crop refinement)。
- **核心优势**: Training-free, 通用（多种 backbone scale 有效），理论上 principled（决策论而非启发式），实践中搜索密集型场景提升显著。
- **关键创新点**: (1) 将 VLM 的 crop 操作从启发式预处理提升为 principled 的 S-BOED 实验设计；(2) 推导出有明确物理含义的 Coverage-Resolution 目标；(3) 用 resolvability probing 作为可操作的 empirical surrogate。
