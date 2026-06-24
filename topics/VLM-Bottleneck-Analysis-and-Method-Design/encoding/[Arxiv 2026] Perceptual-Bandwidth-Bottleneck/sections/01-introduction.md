[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文首先诊断 VLM 的"感知盲区 (perceptual blindness)"本质上是感知带宽瓶颈导致的证据采集失败，而非语义推理失败。随后论证被动扫描策略不可行，需要主动信息采集策略，进而将问题形式化为 S-BOED，引出 Coverage-Resolution Objective 和 FOVEA 框架。

---

## 二、原始文本

Vision-Language Models (VLMs) have significantly advanced general visual understanding, demonstrating a remarkable ability to reaso about holistic scene context (Bai et al., 2025; Comanici et al., 2025). However, a critical performance gap remains: despite their high-level reasoing capabilities, these models often exhibit "perceptual blindness" in tasks requiring fine-grained resolution (Campbell et al., 2024; Li et al., 2025b). Current state-of-the-art models frequently struggle with small-scale object counting, optical character recognition (OCR), and precise spatial localisation, failing even when the underlying logic of the task is straightforward (Zhang et al., 2024; Tong et al., 2024). We argue that such failures are not only failures of semantic reasoing but also failures of evidence acquisition under limited perceptual bandwidth.

> 💡 **核心诊断**: VLM 在高分辨率任务上的失败有两层含义：(1) 传统认知——模型"不够聪明"（语义推理失败）；(2) 本文视角——模型"没看到足够的信息"（证据采集失败）。关键区分：当任务的底层逻辑很简单（如计数），但模型仍然失败，这个失败更可能来自证据层面而非推理层面。这一区分是整个工作的逻辑起点。

The Perceptual Bandwidth Bottleneck. We identify this limitation as a perceptual bandwidth bottleneck. Most standard vision encoders, such as ViT-based models, project an input image into a fixed number of visual tokens regardless of its original resolution (Dosovitskiy, 2020; Liu et al., 2023). This fixed budget induces an unavoidable field-ofview–resolution trade-off: a global view preserves broad spatial context but compresses fine-grained details, while a local crop preserves details but sacrifices coverage. When processing a high-resolution scene globally, each token must aggregate a large spatial area, causing small objects, text, and local spatial relations to vanish before reasoing begins. Consequently, the model cannot reaso about evidence that is absent from its visual representation.

> 💡 **机制拆解 — 感知带宽瓶颈的物理本质**:

> | 因素 | 全局视图 | 局部裁切 |
> |------|---------|---------|
> | 空间覆盖 | 完整场景上下文 | 仅局部区域 |
> | 分辨率/信息密度 | 低（每个 token 聚合大面积像素） | 高（每个 token 聚焦小面积像素） |
> | 小目标可见性 | 被压缩消失 | 清晰可见 |
> | 适用场景 | 场景理解、空间关系推理 | OCR、小目标识别、计数 |

> 本质矛盾：encoder 的 token budget B 是固定的（如 ViT 的 576 个 visual tokens），信息密度 ρ = B / A(d)，大面积 crop → 低 ρ → 小目标信号被 pooling 消灭。这解释了为什么 VLM 在简单的计数任务上也会失败——目标像素在进入 LLM 之前就"物理消失"了，模型无从推理。

The Need for an Active Strategy. Alleviating this bottleneck requires the model to act, not merely to perceive. Instead of passively encoding a single downsampled image, the agent must engage in information foraging (Pirolli & Card, 1999): it must decide where to allocate highresolution visual bandwidth in order to acquire task-relevant evidence. Passive scanning strategies, such as sliding windows, are computationally prohibitive and introduce large amounts of distractor evidence. Recent latent Chain-of-Thought (Li et al., 2025a; Sun et al., 2025) and tool-based methods (Ma et al., 2025; Zhang et al., 2025b; Su et al., 2025a; Gao et al., 2025) show that visual agents can benefit from iterative perception, but their crop or tool-selection policies often remain heuristic. They lack a decisiontheoretic objective for deciding which observation is most valuable when the target is not immediately visible.

> 💡 **机制拆解 — 现有方案与本文方案的差异**:
>
> | 方案类型 | 代表方法 | 核心机制 | 局限性 |
> |---------|---------|---------|--------|
> | 被动编码 (Direct) | 标准 VLM | 一次性 down-sample 全局图像 | 分辨率信息丢失，无法处理 fine-grained |
> | 被动扫描 (Sliding Window) | SAHI (Akyon et al., 2022) | 滑窗遍历全图 | 计算昂贵 + 大量 distractor evidence |
> | 启发式工具使用 (ReAct) | Thyme, LATTE, OpenThinkImg | VLM 提议 crop → 工具执行 → 观察 → 循环 | crop 选择策略是启发的，无决策论目标 |
> | 潜空间推理 | Latent CoT (Li et al., 2025a) | 在 latent space 中迭代推理 | 不直接解决感知带宽问题 |
> | **S-BOED (本文)** | **FOVEA** | **以信息增益最大化目标选择视觉观测** | **需额外 probing 计算 cost** |

Our Approach: Active Visual Reasoing as S-BOED. We formalise active visual information acquisition as a sequential Bayesian optimal experimental design (S-BOED) problem (Lindley, 1956; Chaloner & Verdinelli, 1995; Rainforth et al., 2024). Analogous to a scientist choosing experiments to reduce uncertainty about hidden hypotheses, a VLM agent selects foveation actions to reduce uncertainty about the user's query, as illustrated in Figure 1.

> 💡 **类比**: 科学家做实验来减少关于未知假设的不确定性 → VLM agent 选择 foveation（注视）动作来减少关于用户 query 答案的不确定性。这个类比很精妙：实验设计 (BOED) 的核心是"选择哪个实验能获得最有价值的观测"，在视觉上下文中，就是"把高分辨率的视觉带宽 (crop) 分配到哪里"。

This formulation exposes a key challenge overlooked in prior work: active visual reasoing is not just discrete visual tool selection, but continuous visual foraging under a bandwidth constraint. While BOED has recently been applied to discrete information-gathering tasks such as question selection (Kobalczyk et al., 2025; Choudhury et al., 2025), highresolution visual reasoing requires selecting continuous foveation actions over large image spaces. The perceptual bandwidth bottleneck creates an Information Cliff : a wide view offers context but too little resolution, while a random zoom offers resolution but may miss the target. As a result, individual observations can have near-zero value until a critical coverage–resolution threshold is reached, motivating non-myopic planning.

> 💡 **关键区分 — 离散 vs 连续 BOED**: 现有 BOED for LLMs (如 Kobalczyk et al., 2025) 处理的是离散动作空间（"选哪个问题问"），有枚举的候选集。而 visual foraging 的动作空间是连续的 [0,1]^4（crop 坐标），且信息动态是 super-additive 的（信息悬崖），不能简单地贪心。这是本文最主要的 theoretical contribution——将 BOED 从离散文本空间扩展到连续视觉空间。

> 💡 **Information Cliff（信息悬崖）的本质**: 宽视图定位了文本但不能读（信息增益=0）；随机缩放能读但目标不在 crop 内（信息增益=0）。单独看都是零增益，但两者顺序组合后才产生高信息增益。这种 super-additive 的信息结构是视觉推理独有的，在离散文本 BOED 中不存在。它直接要求 non-myopic（前瞻）规划——这也是 FOVEA 设计 Lookahead variant 的理论动机。

Since exact Bayesian inference and exact expected information gain are intractable in continuous gigapixel spaces, we derive a tractable Coverage–Resolution Objective as a proxy for task-relevant information gain. We then instantiate the framework with FOVEA, a training-free inference-time procedure for Foveated Observation and Visual Evidence Acquisition. FOVEA treats the VLM's initial crop proposal as a noisy spatial prior, generates candidate foveations, probes their query-relevant resolvability, and selects the design that maximises the coverage–resolution objective. Different optimisation strategies, including greedy sampling, MCMCstyle refinement, and look-ahead planning, can be plugged into the same S-BOED-guided template.

> 💡 **FOVEA 的核心设计理念**:
> 1. **将 VLM 的初始 crop 视为 noisy spatial prior** —— 不信任 VLM 的第一次提议，但以其为搜索中心。
> 2. **Resolvability Probing** —— 询问 VLM 一个 Yes/No 问题："这个 crop 包含足够信息回答原始 query 吗？"，用 Monte Carlo 平滑作为 crop utility 的 empirical surrogate。
> 3. **策略解耦** —— objective (Coverage-Resolution) 与 optimizer (Greedy/MCMC/Lookahead) 可以独立变化，构成 compute-accuracy trade-off 族。

The main contributions are: (1) Problem formulation. We identify the perceptual bandwidth bottleneck as a central obstacle in high-resolution VLM reasoing and formulate active visual reasoing as an S-BOED problem. (2) Objective and instantiation. We derive a tractable Coverage–Resolution Objective as a proxy for task-relevant information gain, and instantiate it with FOVEA, a training-free crop-refinement procedure. (3) Empirical validation. We show consistent gains over direct and ReAct-style baselines on high-resolution benchmarks, with further analysis of remote-sensing search, oracle gaps, proposal-limited failures, and compute–accuracy trade-offs.

---

## 三、Summary

- **问题定义**: VLM 的固定 visual token budget 造成 field-of-view vs. resolution 的根本性 trade-off，高分辨率推理失败本质上是证据采集失败而非语义推理失败。
- **核心洞察**: (1) 需要主动信息采集策略（information foraging），而非被动全局编码；(2) 视觉信息采集应形式化为 S-BOED（决策论），而非启发式工具调用；(3) 视觉空间是连续的且信息动态是 super-additive 的（Information Cliff），需要前瞻规划。
- **方案**: Active Visual Reasoing as S-BOED → Coverage-Resolution Objective (tractable proxy) → FOVEA (training-free instantiation)。
- **贡献**: Problem formulation (perceptual bandwidth) + Theoretical derivation (Coverage-Resolution) + Practical instantiation (FOVEA) + Empirical validation。
