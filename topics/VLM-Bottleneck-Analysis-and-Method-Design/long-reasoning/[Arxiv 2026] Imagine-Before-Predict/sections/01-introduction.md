[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文从 VEP 任务的独特挑战出发，指出文本化中间推理的两个根本性不足——信息丢失和模态错配——并通过 Figure 1 的三角 trade-off 引出潜空间表示作为最优折中方案。最后用三句话概括全部贡献。

---

## 二、原始文本

Video event prediction (VEP) asks a model to infer what will happen next from a partially observed video (Koppula and Saxena, 2016; Vondrick et al., 2016a; Lei et al., 2020; Wang et al., 2025b; Su et al., 2026). Unlike standard video understanding, whose answers can usually be grounded in visible frames, VEP requires constructing an internal hypothesis about unobserved dynamic visual states: where objects will move, whether entities will interact, and how a scene will evolve. Although recent multimodal large language models (MLLMs) have made rapid progress on retrospective video tasks (Bai et al., 2025b,a; Wang et al., 2024; Li et al., 2024; Fu et al., 2024; Li et al., 2025c), future-oriented reasoing remains less explored.

> 💡 **问题定位 — VEP 的独特挑战**: 标准的视频理解任务（如 Video-MME, MVBench）关注**可见帧**中的内容，答案可以直接从观察到的视觉证据中获得。但 VEP 不同——它要求模型建立关于**尚未观察到的动态视觉状态**的内部假设（物体将移向何处、实体是否会交互、场景将如何演变）。这意味着 VEP 本质上是一个 **predictive/anticipatory reasoing** 任务，而非 recognition 任务。这种"从已见到未见"的推理要求是本文所有后续设计决策的根本出发点。

Existing video MLLMs usually verbalize intermediate future reasoing in text space (Zhang et al., 2023; Han et al., 2025; Feng et al., 2026; Li et al., 2025d; Su et al., 2026). This is convenient for explanation, but it creates a poor interface for dynamic visual prediction: once visual evidence is converted into words, fine-grained motion, geometry, relative position, and interaction can be lost. The resulting reasoing may sound plausible while drifting away from visual semantics, especially when the correct answer depends on subtle future dynamics. Recent latent visual reasoing methods avoid part of this bottleneck by using continuous visual states (Li et al., 2025b; Pham and Ngo, 2025; Qin et al., 2025; Cheng et al., 2026; Li et al., 2025a; Yang et al., 2025b; Lu et al., 2026a), but most treat latent thoughts as static helper images or one-shot visual hints. VEP instead calls for a temporally organized latent process that can update imagined dynamic visual states over multiple reasoing steps.

> 💡 **机制拆解 — 两类现有方法的不足**:
>
> | 方法类别 | 代表工作 | 机制 | 对 VEP 的不足 |
> |---------|---------|------|-------------|
> | 文本化推理 (Textual CoT) | Video-R1, Video-CoE, NEP | 用自然语言描述每一步推理 | 运动、几何、位置关系等细粒度视觉信息在文本化过程中丢失 → "听起来合理但视觉上无根据"的幻觉（plausible but visually ungrounded） |
> | 静态潜视觉推理 | LVR, Monet, SwimBird, Mirage | 用辅助图像/草图的 embedding 作为潜空间视觉线索 | 潜视觉思维锚定在**已有/静态**图像上，不支持**时序演变的动态**未来视觉状态更新 |
>
> 关键洞察：VEP 需要的不是一个或几个静态视觉"提示"，而是一个**时序组织的潜空间过程**，能够在多个推理步骤中持续更新想象的动态视觉状态。

We introduce FUTURE-L1, a framework that equips MLLMs with interleaved latent visual reasoing for VEP. During autoregressive decoding, FUTURE-L1 alternates between textual tokens and continuous latent visual spans, allowing language to organize the reasoing while latent states preserve intermediate dynamic visual structure. Training proceeds in two stages. First, we construct FUTURE-L1-50K from TwiFF-style trajectories using visual-gain data curation, selecting examples where intermediate future visual hints measurably help prediction. Supervised fine-tuning then teaches the model when to invoke latent spans and aligns their hidden states with future-frame embeddings. Second, we apply LA-DAPO, a latent-aware RL objective that optimizes sampled latent trajectories with outcome-contrastive and temporal-diversity rewards, encouraging successful latent futures while discouraging repeated visual thoughts.

> 💡 **机制拆解 — 两阶段训练设计哲学**:
>
> **阶段 1 (SFT on FUTURE-L1-50K)**：冷启动 + 语义锚定
> - "教会模型在何时使用潜视觉 span"（行为层面）
> - "教会模型潜状态应该表达什么"（语义层面——通过未来帧 embedding 的 MSE 对齐）
> - 没有这一阶段，模型既不知道何时触发潜视觉 span，也不知道潜状态应该在什么语义 manifold 上
>
> **阶段 2 (LA-DAPO RL)**：轨迹优化
> - SFT 使用的是 teacher-forcing——每个潜状态精确匹配到了指定未来帧
> - 但实际推理时，模型是采样解码的——潜轨迹可能偏离最优路径
> - LA-DAPO 通过 outcome-contrastive 和 temporal-diversity 奖励来优化采样轨迹
> - 注意：RL 阶段**不需要**中间帧标注——signal 仅来自 answer correctness 和潜状态结构

> 💡 **设计亮点 — 两阶段的分工与互补**:
> - SFT 解决"能不能用"的问题（capability acquisition）
> - RL 解决"用得好不好"的问题（quality optimization）
> - 没有 SFT → 模型不会使用潜空间；没有 RL → 潜轨迹未针对预测正确性优化
> - 这种"先教后优化"的范式类似于 RLHF 中的 SFT → PPO，但在 latent reasoing 领域有独特的挑战（潜状态没有 language prior）

> 💡 **Figure 1 对照解读**（见 00-abstract.md 的 Figure 1 分析）:
> 论文通过 Figure 1 建立了 Text-CoT（高效但信息丢失）→ Pixel-space simulation（信息全但代价大）→ Latent visual span（最优折中）的三角对比，精确定位了本文的设计空间。

Experiments show that latent visual reasoing is substantially more effective than text-only reasoing for VEP. On FutureBench, FUTURE-L1-RL improves Qwen3-VL-8B from 61.0 to 85.4, exceeding the previous best Video-CoE by 10.4 points. On TwiFF-Bench, it improves the average score from 2.44 to 3.04. Under the same curated data source, text-only SFT reaches only 65.0 on FutureBench, whereas interleaved latent SFT reaches 73.2, indicating that the gain is not merely from additional supervision but from reasoing through a modality better matched to future visual structure.

> 💡 **关键消隐控制**: 文本-only SFT (65.0) vs 交错潜空间 SFT (73.2) —— 两者使用**完全相同的数据源**（FUTURE-L1-50K），但格式不同（纯文本轨迹 vs 文本-潜视觉交错轨迹）。这 +8.2 的差距直接证明了 **modality effect**（模态匹配效应）——增益不是来自更多监督数据，而是来自用更适合未来视觉结构的模态（潜空间 vs 文本）进行推理。

Our contributions are threefold:

1. We propose visual-gain data curation and construct FUTURE-L1-50K, a high-utility corpus for supervising latent future visual reasoing.

2. We introduce interleaved latent visual reasoing for VEP, enabling autoregressive models to alternate between language and continuous future visual states.

3. We develop LA-DAPO, a latent-aware RL method that improves sampled latent trajectories and achieves state-of-the-art results on FutureBench and TwiFF-Bench.

> 💡 **贡献结构分析**: 三个贡献呈"数据—方法—优化"递进关系：
> 1. **数据 (visual-gain)**：解决了"什么样的样本值得用来训练潜空间推理"的问题
> 2. **方法 (interleaved reasoing)**：解决了"如何让自回归模型交替进行文本和潜视觉推理"的问题
> 3. **优化 (LA-DAPO)**：解决了"如何在没有中间标注的情况下优化潜轨迹质量"的问题
>
> 三个贡献层层递进，缺一不可。

---

## 三、Summary

- **问题定义**: VEP 是预测未观察到的动态未来视觉状态，本质是 predictive reasoing 而非 recognition——文本化推理在此任务中特别脆弱（丢失运动/几何/交互信息）。
- **核心洞察**:
  - 文本化 = 离散 + 符号 = 丢失连续视觉动态
  - 像素生成 = 精确 + 完整 = 计算代价过高
  - 潜空间 = 连续 + 紧凑 = 保留视觉语义 + 高效 —— 最优折中
- **设计空间**: 三个层次的"数据—方法—优化"递进构建
  - 数据层：visual-gain = 筛选有可测量预测效用的样本
  - 方法层：interleaved reasoing = 文本组织推理 + 潜空间保留视觉
  - 优化层：LA-DAPO = outcome-contrastive + temporal-diversity
- **关键证据**: 文本-only SFT vs 潜空间 SFT 的 +8.2 差距 = modality effect，证明增益来自模态匹配而非数据量
