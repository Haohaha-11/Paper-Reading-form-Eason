# Improving Visual Reasoning with Iterative Evidence Refinement (SIEVE)

## Paper Metadata

| 项目 | 内容 |
|------|------|
| **Title** | Improving Visual Reasoning with Iterative Evidence Refinement |
| **Authors** | Zeru Shi\*, Kai Mei\*, Yihao Quan, Dimitris N. Metaxas, Ruixiang Tang† |
| **Affiliations** | Department of Computer Science, Rutgers University |
| **Venue** | arXiv 2026 |
| **Base Models** | Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct |
| **Training Paradigm** | GRPO (Group Relative Policy Optimization) |
| **Training Data** | ~1.5k samples from COCO 2017 |
| **GPU** | 2x NVIDIA H200 |

## One-Sentence Summary

SIEVE 提出了一种端到端的自回访 (self-revisit) 框架，完全摒弃外部工具调用与图像重编码，通过挖掘 VLM 内部隐状态中的梯度显著性锚点来发现与定位视觉证据 (visual evidence)，并在 GRPO 强化学习训练中学习"何时"及"如何"将这些 region embedding 注入推理链 (reasoning chain)，从而在无需任何外部工具的前提下实现 grounded visual reasoning。

> 💡 **核心定位**: SIEVE 的核心理念是"模型已经看到了，只是没看对地方"——VLM 在首次编码时已经提取了足够丰富的视觉信息，问题不在于信息缺失，而在于模型在长链推理中不会**回访** (revisit) 这些信息。SIEVE 用训练的方式教会模型如何从自己的隐状态中"翻找"关键视觉证据并动态注入。

## Core Contributions

1. **提出 SIEVE 框架** (Section 3): 首次将"内部 region embedding 检索与注入"作为视觉推理的核心机制，完全替代外部 crop/zoom 工具。核心组件包括：梯度显著性引导的 textual anchor 发现、中间层跨模态匹配的视觉证据定位、以及基于 GRPO 的 visually-grounded RL 训练。

2. **Self-Guided Visual Evidence Discovery** (Section 3.2): 基于梯度重要性的锚点发现 + 中间层跨模态相似性匹配 + 空间一致性区域扩充，无需任何人工标注或外部模型即可自动定位任务关键区域。

3. **Visually-Grounded RL Training** (Section 3.3): 设计四维 reward 函数（format + result + embedding + action），用仅 ~1.5k 样本教会模型何时调用视觉证据、选择哪些 region embedding、以及何时停止。数据效率极高。

4. **全面的实验验证** (Section 4): 在 2 个模型规模 (4B/8B) 和多个 benchmark 上一致提升，平均 +8%，尤其是在高分辨率视觉推理 (V\*, HR-Bench) 和感知任务 (MME-Real-Lite) 上提升显著。

## 📖 Section Navigation

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| Abstract & Figure 1 | [00-abstract.md](sections/00-abstract.md) | 论文概述、工具增强 vs SIEVE 的范式对比 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 动机：VLM 推理中的视觉证据衰减、初步验证、三个贡献 |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | 工具增强推理 + 潜空间推理两条线 |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | SIEVE 全流程：证据发现 Algorithm 1 + GRPO RL 训练 |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | 主实验、消融 (embedding 价值 / 层选择 / action reward) |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | 总结：内部信号替代外部工具是可行且高效的 |

## Key Numbers

| 指标 | 数值 |
|------|------|
| Base Models | Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct |
| Training Data | ~1,500 images from COCO 2017 |
| RL Algorithm | GRPO, 60 rollout steps |
| Batch Config | 16 prompts, 8 rollouts/prompt |
| Max Response Length | 8,192 tokens |
| Evaluate Benchmarks | V\* Bench, HR-Bench (4K/8K), MME-Real-Lite, RealWorldQA, MathVista, LogicVista, WeMath, HallusionBench |
| SIEVE 4B avg gain over Vanilla | +7.85% (V\*), +3.50% (HR-Bench 4K) |
| SIEVE 8B avg gain over Vanilla | +5.24% (V\*), +4.00% (HR-Bench 8K) |
| MME-Real-Lite 4B gain | +17.58% |
| MME-Real-Lite 8B gain | +19.30% |
| WeMath 8B gain | +20.65% |
| GPU | 2x NVIDIA H200 |

## Data Flow: Input → Intermediate → Output

```
┌─────────────────────────────────────────────────────────────────┐
│                        SIEVE Data Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Phase I: Visual Evidence Discovery (pre-training)]              │
│    │                                                              │
│    ├── 1. Forward pass: Model(I, question) → logits + hidden states
│    │                                                              │
│    ├── 2. Gradient Saliency: Sal(i) = ||∇_hi(s) ⊙ hi||_2        │
│    │      → Select top-k salient tokens as textual anchors A      │
│    │                                                              │
│    ├── 3. Cross-modal Matching:                                   │
│    │      H̄ = mean of middle-layer hidden states                 │
│    │      For each anchor q_i ∈ A:                               │
│    │        s_ij = cos(x̂_j, q̂_i)  →  w_ij = softmax(s_ij/τ)    │
│    │                                                              │
│    ├── 4. Region Selection: score blocks → TopK → merge → expand │
│    │                                                              │
│    └── 5. Cache: E_i = Concat(patch embeddings in region R_i)    │
│                                                                   │
│  [Phase II: Visually-Grounded RL Training (rollouts)]             │
│    │                                                              │
│    ├── Per rollout step t:                                        │
│    │   a_t ~ π_θ(· | s_t), s_t = I || (x_1||E_1) || ... || (x_{t-1}||E_{t-1})
│    │   │                                                          │
│    │   ├── Action: produce text OR trigger embedding insertion    │
│    │   ├── If trigger: inject cached E into reasoning stream      │
│    │   └── Terminate: final answer or max turns reached           │
│    │                                                              │
│    ├── Reward: R(τ) = 0.6·R_res + 0.3·R_fmt + 0.5·R_emb + 0.2·R_act
│    │                                                              │
│    └── On failure w/ evidence: re-discover evidence with updated model
│                                                                   │
│  [Output]                                                         │
│    └── Model learns to: WHEN to insert, WHICH region to insert,  │
│        produce correct answer with visual evidence support         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Pros/Cons & Future Work

### Strengths

1. **无需外部工具**: 彻底摆脱 crop/zoom/re-encode 的依赖，完全在 VLM 内部隐空间中操作，不引入额外的视觉编码器或工具调用开销
2. **数据效率极高**: 仅需 ~1.5k COCO 样本即可训练出"何时调用视觉证据"的能力，远超现有 tool-augmented 方法的数据需求
3. **多 benchmark 一致性**: 在高分辨率推理 (V\*, HR-Bench)、感知 (MME-Real-Lite)、推理 (MathVista, LogicVista, WeMath)、幻觉 (HallusionBench) 四个维度上均有提升
4. **推理时间合理**: 相比 ZoomRefine 等工具增强方法，SIEVE 的推理时间增加有限，提供了效率与效果的最佳 trade-off
5. **中间层设计的实证支撑**: 通过 IHR (Information Hit Ratio) 验证了中间层表征在跨模态匹配中的优越性，与现有文献的中间层语义丰富性发现一致

### Weaknesses / Limitations

1. **证据发现依赖初始模型质量**: 梯度显著性锚点发现的质量高度依赖基础模型的预测校准度；如果模型本身 poorly calibrated（尤其在训练初期），锚点可能是噪声，导致 evidence cache 质量差
2. **训练与推理的双重成本**: 虽然推理时不需外部工具，但训练需要反复进行 evidence re-discovery（模型更新后重新提取嵌入缓存），这在 early training stage 会增加算力开销
3. **Region 粒度的局限**: 目前取 top-K=1 的块并进行 bounding box 扩张，这依赖于 Qwen-VL 的 patch segmentation 机制；不同 VLM 的 patch 粒度不同可能导致区域质量差异
4. **未与更强工具增强方法对比**: 对比的 baselines 中 ZoomEye/ZoomRefine 的表现波动较大（8B 上 ZoomEye 反而低于 Vanilla），缺乏与更近期的 RL-based 工具增强方法（如 DeepEyes, OpenThinkImg）的对比
5. **仅验证 Qwen3 系列**: 所有实验均在 Qwen3-VL 上进行，未验证方法在如 InternVL、LLaVA 等其他 VLM 架构上的迁移性

### Future Work

1. 将 SIEVE 的证据发现机制与更细粒度的 multi-scale region 选择结合（当前仅 K=1 全局最优，但某些任务可能需要多区域联合推理）
2. 探索在训练阶段将 evidence discovery 与 policy learning 完全联合优化（当前是交替更新：先固定 evidence 训 policy，policy 变好后重新提取 evidence）
3. 扩展到视频理解、多图推理等需要跨帧/跨图视觉回访的场景
4. 与 FlashAttention 等高效 attention 后端兼容性验证（证据发现需要访问中间层 hidden states）
5. 在闭源 API 模型上探索是否能仅通过 logits 层面的信号近似锚点发现（无需 hidden state 访问）

## Reading Q&A Record

| # | 问题 | 答案位置 | 解答 |
|---|------|---------|------|
| 1 | SIEVE 为什么不像其他方法一样直接 crop/zoom，而要选择"检索已有 embedding"？ | Section 1, para 4 | Crop/zoom 需要 (1) 额外的图像重编码，打断推理连续性；(2) 大量训练数据和复杂训练管线来学习工具调用。作者认为原始 embedding 已有足够信息，瓶颈在于"选择性复用"而非"信息不足"。 |
| 2 | 梯度显著性锚点为什么比手工关键词更好？ | Section 3.2.1 | 手工关键词依赖外部标注或规则，泛化差。梯度显著性直接从模型的**预测动力学**中提取——模型认为哪些 token 对预测最重要，这些就是该任务的关键语义锚点。这种自引导 (self-guided) 方式不需要任何 task-specific 先验。 |
| 3 | 为什么选择中间层而非浅层或深层的 hidden states 做跨模态匹配？ | Section 3.2.2 + Section 4.4.2 | 浅层表征语义抽象不够，深层表征过于任务特化（偏向输出预测）。中间层在语义抽象和空间保真之间取得最佳平衡。IHR 实验显示中间层的信息命中率显著高于浅层和深层。 |
| 4 | RL 训练中 Embedding Reward 的作用是什么？ | Section 3.3, Eq.4 | 它只在模型**答对且使用了 embedding**时激活。目的是防止模型学会"不使用 evidence 也能答对"的捷径——鼓励模型在 evidence 有帮助时主动调用，而不是绕过它。 |
| 5 | Action Reward 为什么要同时包括 penalize 短期和 reward 长期两种信号？ | Section 3.3 + Section 4.4.3 | 如果移除 signal reward，训练会出现 reward collapse（后期 reward 掉到接近 0）；如果移除 thought richness reward，模型会输出空的 `<think></think>` 标签来 hack reward 格式。两种 reward 各司其职，共同保证训练稳定性。 |
| 6 | K=1 为什么是最优的？ | Section 4.4.2 + Appendix B | 因为 region snapshot 是将区域内所有 patch 拼接成一个 embedding，不可避免地引入了区域内的无关视觉信息。K 越大，注入的噪声累积越多，干扰推理。少量高相关 embedding 优于大量含噪 embedding。 |
| 7 | SIEVE 与 DMLR / VisMem 等潜空间推理方法的本质区别是什么？ | Section 2.2 | SIEVE 不创造新的 latent space 或 learnable latent tokens，而是**直接复用原始视觉编码的 embedding**，在文本推理空间中进行注入。这避免了额外的潜空间对齐训练成本，也使得证据是"可解释的"（可以映射回原图区域）。 |

## Citation Landscape

### Reference Grouping by Topic

**VLM Backbones & Architectures**:
- Qwen3-VL [Yang et al., 2025a], Qwen2.5-VL [Bai et al., 2025a], Kimi-VL [Team et al., 2025b], Seed1.5-VL [Guo et al., 2025]

**Tool-Augmented Visual Reasoning**:
- DyFo [Li et al., 2025e], ZoomEye [Shen et al., 2024], Zoom-Refine [Yu et al., 2025c]
- VisualToolAgent [Huang et al., 2025b], OpenThinkImg [Su et al., 2025a], GRiT [Fan et al., 2025]
- DeepEyes [Zheng et al., 2025], Omni-R1 [Zhong et al., 2025], Pixel Reasoner [Su et al., 2025b]
- Perception-R1 [Yu et al., 2025a], SifThinker [Chen et al., 2025]
- SAM-R1 [Huang et al., 2025a], UnivG-R1 [Bai et al., 2025b]

**Latent Space Reasoning / Thinking with Images**:
- Latent Visual Reasoning [Li et al., 2025c], Machine Mental Imagery [Yang et al., 2025b]
- DeepSketcher [Zhang et al., 2025a], Visual Planning [Xu et al., 2025]
- Thinking with Generated Images [Chern et al., 2025], MVoT [Li et al., 2025b]
- Visual Sketchpad [Hu et al.], Interactive Sketchpad [Lee et al., 2025]
- CAD-Assistant [Mallis et al., 2025]

**Token Efficiency / Pruning**:
- Token Merging [Bolya et al., 2022], MadTP [Cao et al., 2024]
- IVTP [Huang et al., 2024], TokenFlex [Hu et al., 2025]
- LESS [Song et al., 2024], ReGaTE [Li et al., 2025d]
- Visual Perception Token [Yu et al., 2025b], Glimpse [Zeng et al., 2025]

**Hallucination & Visual Reliance**:
- Hidden Life of Tokens [Li et al., 2025a], Grounding Language with Vision [Fang et al., 2025]

**Benchmarks**:
- V\* Bench [Wu and Xie, 2024], HR-Bench [Wang et al., 2024]
- MME-RealWorld [Zhang et al., 2024b], RealWorldQA [xAI]
- MathVista [Lu et al., 2024], LogicVista [Xiao et al., 2024], WeMath [Qiao et al., 2025]
- HallusionBench [Wu et al., 2024]

**Representation Analysis (Middle Layers etc.)**:
- Layer by Layer [Skean et al., 2025], Does Representation Matter? [Skean et al., 2024]
- Fantastic Semantics [Liu et al., 2024], From Associations to Activations [Schiekiera et al., 2026]

**Reinforcement Learning**:
- DeepSeekMath (GRPO) [Shao et al., 2024b], Kimi K1.5 [Team et al., 2025a]
- Visual CoT [Shao et al., 2024a], OpenAI Thinking with Images [OpenAI, 2025]

---

*Batch reading created on 2026-06-24*
