# Thinking with Visual Grounding

## Paper Metadata

| 项目 | 内容 |
|------|------|
| **Title** | Thinking with Visual Grounding |
| **Authors** | Junkai Zhang, Yihe Deng, Kai-Wei Chang, Wei Wang |
| **Affiliations** | University of California, Los Angeles (UCLA) |
| **Venue** | arXiv 2026 |
| **Paper Link** | https://arxiv.org/abs/26xx.xxxxx |
| **Project Page** | -- |
| **References** | 28 |

## One-Sentence Summary

本文提出 visually grounded thinking：让 VLM 在推理过程中以自然语言进行思考，同时显式地用 boxes 或 points 标注每一步所依赖的图像区域，并通过 SAM3-based 合成 pipeline 构建训练数据、设计 grounding-aware RL reward 来联合优化答案正确性和 grounding 精度，使 4B 模型在计数和空间推理上达到甚至超越 27B 模型的水平。

## Core Contributions

1. **可扩展的数据合成 pipeline** (Section 3): 以 SAM3-based agentic grounding 系统为核心，从开放数据集的视觉问答中蒸馏正确的推理轨迹、提取可 grounding 的视觉对象、获取 RLE mask，并自动生成 box-mode 和 point-mode 的对齐训练数据，同时保留用于 RL 的结构化 grounding 监督信号。

2. **Grounding-aware RL Reward** (Section 4): 设计了显式评估模型 rollout 中 grounding 质量的 reward：box mode 用生成框与 ground-truth 框的 IoU，point mode 用生成点与 ground-truth mask 的 F1 匹配分数。reward 与答案正确性 reward 联合归一化后用于 GRPO 训练。

3. **受控实验揭示 grounding 的双重收益** (Section 5): visually grounded thinking 在计数和空间推理上大幅超越纯文本 thinking baseline（后者在 RL 中出现 length collapse），4B 的 grounded 模型在空间推理上匹敌甚至超越 27B 模型。Point grounding 在计数上更优（instance-level localization 足够），box grounding 在空间推理上受益于 explicit grounding reward。

## Section Navigation

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| Abstract & Figures | [00-abstract.md](sections/00-abstract.md) | 论文概述、grounded vs non-grounded thinking 对比 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 问题动机、visually grounded thinking 定义、贡献总结 |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | Visual CoT → GCoT/Argus → GRIT/ViGoRL 的演进脉络 |
| 3. Data Synthesis Pipeline | [03-data-synthesis.md](sections/03-data-synthesis.md) | 6 阶段 pipeline、SAM3 agentic grounding、box/point 对齐 |
| 4. RL with Grounding Reward | [04-rl-grounding-reward.md](sections/04-rl-grounding-reward.md) | Grounding tag parsing、object router、box IoU / point F1、final reward |
| 5. Experiments | [05-experiments.md](sections/05-experiments.md) | 主实验、grounding reward 消融、box vs point 对比分析 |

## Key Numbers

| 指标 | 数值 |
|------|------|
| SFT 数据量 | 19,909 reasoning traces |
| Grounding 标注数 | 107,613 个 <obj> 标签 |
| 唯一 grounded objects | 72,381 个 |
| 平均每样本 grounded objects | 3.64 个 |
| 平均每样本 grounding annotations | 5.41 个 |
| 计数 benchmark | 2 (TallyBench, CountQA) |
| 空间推理 benchmark | 4 (VSR-zero, EmbSpatial, SpatialMQA, MultihopSpatial) |
| 基础模型 | Gemma3-4B-IT |
| SFT epochs | 6, lr=1e-5, batch=256 |
| RL 步数 | 100, lr=1e-6, batch=64, rollout=8/prompt |
| Answer reward weight | 1.0 |
| Grounding reward weight | 0.5 |
| Format reward weight | 0.1 |
| Point grounding 在计数上的提升 | ~6%+ pass@4 超越 box mode |
| Box mode w/ grounding reward 在 6 个 benchmark 上 | 全部优于 w/o grounding reward |
| 4B grounded 模型 vs 27B | 在 SpatialMQA 和 MultihopSpatial 上超越 27B |
| 训练+评估 GPU 消耗 | ~400 H200 GPU hours |

## Data Flow: Distillation → Grounding → Training

```
| 阶段 | 描述 |
|------|------|
| 1. Thinking with Visual Grounding Pipeline |  |
| 2. [Stage 1 | Reasoning Distillation] |
| 3. MultihopSpatial, SpatialMQA |  |

Output: Correct reasoning traces (filtered by answer match)     │
│                                                                       │
│  [Stage 2: Object Extraction]                                         │
│    ├── Input: Correct reasoning trace                                  │
│    ├── Model: LLM (DeepSeek-V4-Flash)                                  │
│    └── Output: (name, disambiguating context) pairs                    │
│                                                                       │
│  [Stage 3: Agentic Grounding]                                         │
│    ├── Core Engine: SAM3 + VLM agent (Qwen3.5-Flash)                   │
│    ├── Actions: noun-phrase query → mask candidate → verify → select   │
│    ├── Fallback: Qwen3.6-Plus → Gemini-3-Flash for retries             │
│    └── Output: RLE masks for each grounded object                      │
│                                                                       │
│  [Stage 4: Supervision Writing]                                        │
│    ├── Box mode: <obj> name | [x1,y1,x2,y2] </obj>                    │
│    ├── Point mode: <obj> name | [x,y] </obj>                          │
│    ├── Both share the same reasoning trace + SAM3 masks                │
│    └── Output: Aligned SFT data in two grounding modes                  │
│                                                                       │
│  [Stage 5: SFT Cold-Start]                                             │
│    ├── Base Model: Gemma3-4B-IT                                        │
│    ├── 3 variants: non-grounded / box-grounded / point-grounded         │
│    └── Same images, questions, answers, reasoning traces                │
│                                                                       │
│  [Stage 6: RL with Grounding Reward]                                   │
│    ├── Algorithm: GRPO (8 rollouts/prompt)                             │
│    ├── Reward = N(R_base) + w_ground * N(r_ground)                     │
│    │   ├── r_ans: answer correctness                                   │
│    │   ├── r_think: thinking format (<think>...</think> + \boxed{})    │
│    │   ├── r_gfmt: grounding tag format                                │
│    │   ├── r_trunc: truncation penalty (-1 if truncated)               │
│    │   └── r_ground: box IoU or point F1 via object router             │
│    └── Output: Visually grounded thinking model                         │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Pros/Cons & Future Work

### Strengths

1. **Scalable data pipeline**: 完全依赖现有的开源 VLM + SAM3，无需人工标注，可扩展到任意视觉问答数据集
2. **两种 grounding 模式对齐训练**: box 和 point 模式共享同一套 SAM3 mask，保证对比公平性
3. **Explicit grounding reward**: 直接监督 rollout 中的 grounding 质量，而非只依赖答案正确性，使 box mode RL 在 6 个 benchmark 上全部提升
4. **Strong empirical gains**: 4B 参数量的 grounded thinking 模型在空间推理上可达 27B 水平
5. **可控实验设计**: non-grounded、box-grounded、point-grounded 三种变体使用相同基础数据，干净地隔离 grounding 的效果

### Weaknesses / Limitations

1. **Point grounding reward 信号较粗**: point F1 在 mask 内部移动不改变分数，跨边界才有跳跃，导致 RL 优化困难，未能带来一致的性能提升
2. **SAM3 依赖**: 数据合成的质量高度依赖 SAM3 的 mask 精度，对于某些复杂场景或罕见物体可能存在 grounding 失败
3. **Object router 开销**: RL 训练中每个 rollout 都需要 VLM object router (Qwen3.5-4B) 进行匹配，增加了计算负担
4. **仅验证 Gemma3 系列**: 只在 Gemma3-4B-IT 上实验，未在其他模型家族（如 Qwen-VL、LLaVA）上验证泛化性
5. **数据范围有限**: 仅在 counting 和 spatial reasoning 任务上验证，未涵盖更广泛的视觉推理任务（如 VQA、hallucination detection）

### Future Work

1. 改进 point grounding reward 的设计，使其提供更平滑、更密集的优化信号（如 distance-to-mask 惩罚）
2. 探索将 visually grounded thinking 扩展到其他视觉任务（visual grounding、referring expression、visual entailment）
3. 减少 object router 的计算开销（如使用 embedding-based matching 替代 VLM-based routing）
4. 在更多 backbone 模型和更大规模上验证 visually grounded thinking 的泛化性
5. 结合 latent-space reasoning 方法（如 DMLR），在潜空间中进行 visually grounded thinking

## Reading Q&A Record

| # | 问题 | 答案位置 | 解答 |
|---|------|---------|------|
| 1 | 为什么纯文本 thinking (non-grounded) 在 RL 中会出现 length collapse？ | Section 5.2 | 非 grounded thinking 模型在 GRPO 训练中回答长度近似线性下降，减少了探索空间，导致性能恶化。相比之下，grounded thinking 中交错的 <obj> 标签和 grounding-format reward 提供了额外的局部结构，有助于稳定 RL 训练。 |
| 2 | Box IoU reward 和 Point F1 reward 的本质区别是什么？ | Section 4 | Box IoU 是连续的：生成框与 GT 框的重叠变化时 reward 平滑改变。Point F1 是分段常数的：点在 mask 内移动不改变分数，跨边界才有阶跃。这种离散性使 point reward 更难优化。两者在鼓励的视觉证据方向一致，但反馈密度不同。 |
| 3 | 为什么不惩罚 rollout 中未匹配的 grounding objects？ | Section 4 | 数据合成 pipeline 提取的 grounded objects 并非所有视觉线索的完整枚举。模型可能在思考中识别出额外的、合理的视觉证据。因此未匹配的 rollout grounding objects 不增减 grounding quality，只是对 ground truth objects 做匹配评估。 |
| 4 | SAM3 agent 为什么不能直接输出坐标？ | Section 3 | 设计约束：所有 geometric supervision 必须从 SAM3 的 RLE mask 中导出。Agent 只能通过 noun-phrase query → mask candidates → verify → select 的闭环操作，不能直接写坐标值。这防止了 annotation model 的坐标幻觉。 |
| 5 | Point mode 在计数上为什么优于 box mode？ | Section 5.4 | 计数任务主要需要 instance-level localization：区分哪些 object 属于被计数集合，不需要恢复每个 object 的完整 extent。Point grounding 提供了紧凑的 instance grounding，避免了生成 tight bounding box 的困难，尤其对小物体、遮挡或异形物体更友好。 |
| 6 | 为什么 grounding reward 对 box mode 有效但对 point mode 效果不明显？ | Section 5.3 | Box IoU 提供连续、平滑的优化信号，变化与区域重叠程度同步。Point F1 是离散的：点在 mask 内任意位置 reward 相同，跨边界才有突变。更粗粒度的反馈使 point reward 难以转化为一致的 accuracy 提升。 |
| 7 | Object router 为什么需要 disambiguating context？ | Section 4 | 同一图像中同一名称可能指向多个不同 object（如两个 "red car"），需要通过场景上下文（如 "near the entrance" vs "in the back row"）来区分。Object router 使用 VLM 根据 name + context 将 model 生成的 grounding objects 匹配到正确的 ground-truth objects。 |

## Citation Landscape

### Reference Grouping by Topic

**Grounding & Visual CoT**:
- Visual CoT [Shao et al., 2024a], UV-CoT [Zhao et al., 2025], GCoT [Wu et al., 2025], Argus [Man et al., 2025]

**RL for Grounded Thinking**:
- GRIT [Fan et al., 2025], ViGoRL [Sarch et al., 2025], VGR [Wang et al., 2025]

**Foundation Models & Tools**:
- Qwen3-VL-Plus [Bai et al., 2025], Qwen3.5 [QwenTeam, 2026a], Qwen3.6-Plus [QwenTeam, 2026b], Gemini-3-Flash [Google DeepMind, 2025], DeepSeek-V4-Flash [DeepSeek-AI, 2026]
- SAM3 [Carion et al., 2026], GRPO [Shao et al., 2024b]
- Gemma3 [Team et al., 2025], verl [Sheng et al., 2024], SGLang [Zheng et al., 2024]

**Reasoning & R1-style RL**:
- DeepSeek-R1 [Guo et al., 2025], OpenVLThinker [Deng et al., 2025], OpenVLThinkerV2 [Hu et al., 2026]

**Benchmarks**:
- TallyQA [Acharya et al., 2018], TallyBench [Cai et al., 2025], CountQA [Tamarapalli et al., 2025]
- VSR [Liu et al., 2023], EmbSpatial [Du et al., 2024], SpatialMQA [Liu et al., 2025], MultihopSpatial [Lee et al., 2026]

**Analysis & Cognitive Perspective**:
- MIRAGE [Asadi et al., 2026]: 揭示 VLM 的视觉理解幻觉
- Hayhoe and Ballard, 2005: 人类注视行为与视觉任务的关系
- Visual7W [Zhu et al., 2016]: grounded question answering
- Human Attention in VQA [Das et al., 2016]

---

*Batch reading created on 2026-06-24*
