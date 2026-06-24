# The Perceptual Bandwidth Bottleneck in VLMs: Active Visual Reasoning via Sequential Experimental Design (FOVEA)

## Paper Metadata

| 项目 | 内容 |
|------|------|
| **Title** | The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design |
| **Authors** | Anjie Liu\*, Ziqin Gong\*, Yan Song, Yuxiang Chen, Xiaolong Liu, Hengtong Lu, Kaike Zhang, Chen Wei, Jun Wang |
| **Affiliations** | HKUST(GZ), UCL, ShanghaiTech, AI Lab Yangtze River Delta, Li Auto |
| **Venue** | arXiv 2026 (2605.01345v3) |
| **Code** | https://github.com/iamlilAJ/active-vlm |
| **BibTeX** | Liu et al., "The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design", 2026 |

## One-Sentence Summary

将高分辨率视觉推理形式化为 Sequential Bayesian Optimal Experimental Design (S-BOED) 问题，提出 Coverage-Resolution Objective 作为信息增益的 tractable proxy，并实例化为 training-free 推理时 crop 精炼框架 FOVEA，通过可分辨性探测 (resolvability probing) 优化 VLM 的视觉证据采集策略。

## Core Contributions

1. **问题形式化** (Section 2 & 3): 识别"感知带宽瓶颈 (perceptual bandwidth bottleneck)"为 VLM 高分辨率推理的核心障碍，将主动视觉信息采集形式化为 S-BOED 问题。提出三个关键概念：perceptual bandwidth (感知带宽)、visibility event (可见性事件)、information cliff (信息悬崖)。

2. **Tractable Coverage-Resolution Objective** (Section 3.2): 在 Factorised Belief、Calibrated Visibility、Ideal Observer 三个近似假设下，推导出 Coverage-Resolution 乘积作为 task-relevant EIG 的 closed-form proxy，显式建模了覆盖范围与感知分辨率之间的 trade-off。

3. **FOVEA 框架** (Section 4): 训练自由 (training-free) 的推理时 crop 精炼框架，通过 resolvability probing 估计 crop utility，支持三种优化策略：Greedy、MCMC-style refinement、Look-ahead planning。

4. **实证验证** (Section 5): 在 4 个高分辨率 benchmark 上一致优于 Direct 和 ReAct 基线，在遥感搜索场景中提升尤为显著（+9.6% over ReAct）。展示了与 Gemini 2.5 Flash 等闭源模型竞争的性能。

## Section Navigation

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 论文提炼、"感知带宽瓶颈+主动信息采集"双核动机 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 感知带宽瓶颈定义、信息采集必要性、S-BOED 视角、贡献总结 |
| 2. Problem Formulation | [02-problem-formulation.md](sections/02-problem-formulation.md) | 概率图模型、perceptual bandwidth/resolution probability、visibility event、生成过程 |
| 3. S-BOED Derivation | [03-sboed-derivation.md](sections/03-sboed-derivation.md) | 序贯目标、Information Cliff、Coverage-Resolution Objective 推导、Bayesian belief update |
| 4. Algorithmic Realisation | [04-algorithmic-realisation.md](sections/04-algorithmic-realisation.md) | FOVEA 框架、Resolvability Probing、三种优化策略 (Greedy/MCMC/Lookahead) |
| 5. Experiments | [05-experiments.md](sections/05-experiments.md) | 多 benchmark 主结果、遥感搜索分析、Compute-Accuracy scaling、消融 |
| 6. Conclusion | [06-conclusion.md](sections/06-conclusion.md) | 总结与展望 |

## Key Numbers

| 指标 | 数值 |
|------|------|
| Benchmark 数量 | 4 (MME-RealWorld-Lite, CV-Bench, V\*Bench, HR-Bench) |
| 主 Backbone | Qwen3-VL-30B-A3B-Instruct (30B) + Qwen3-VL-8B-Instruct (8B) |
| FOVEA (30B) Mean Score | 77.7% vs ReAct 75.1% vs Direct 73.3% |
| FOVEA (8B) Mean Score | 74.9% vs ReAct 72.5% vs Direct 70.9% |
| Remote Sensing (Lookahead) | 54.7% vs ReAct 45.1% (Oracle 68.0%) |
| 候选 crop 数 (Greedy) | 3 (seed, small, large) |
| Monte Carlo probe 次数 | 3 per candidate |
| 最大交互轮次 | 10 |
| 扰动缩放因子 | {1.5, 1.0, 0.8} |
| Perturbation scaling factors | Coverage enlargement: 1.5x; Resolution enhancement: 0.8x |

## Framework Overview: S-BOED → Coverage-Resolution → FOVEA

```
| 阶段 | 描述 |
|------|------|
| 1. S-BOED Active Visual Reasoning Pipeline |  |
| 2. [Problem] Perceptual Bandwidth Bottleneck |  |

Fixed token budget B → field-of-view vs. resolution trade-off │
│                                                                       │
│  [Formulation] S-BOED (Sequential Bayesian Optimal Exp. Design)      │
│    ├── θ = {ℓ, y} : latent location + semantic target                │
│    ├── d = [u,v,w,h] : foveation action / crop design                │
│    ├── S ∈ {0,1} : visibility event = coverage × resolution         │
│    └── z : observation gated by S                                    │
│                                                                       │
│  [Derivation] Tractable Coverage-Resolution Objective                │
│    ├── Assumption 2.4: Factorised Belief (ℓ ⊥ y planning)            │
│    ├── Assumption 3.2: Calibrated Visibility (H(S|z) ≈ 0)           │
│    ├── Assumption 3.3: Ideal Observer (H(y|z,S=1) ≈ 0)              │
│    └── Result: I_t(d) = Coverage × φ(d)  (Eq. 8)                     │
│                                                                       │
│  [Instantiation] FOVEA                                               │
│    ├── Resolvability Probing: Î(d) = P(r=1 | I_d, Q) (Eq. 10)       │
│    ├── FOVEA-Greedy: sample 3 candidates, pick max Î(d)              │
│    ├── FOVEA-MCMC: Metropolis-Hastings over crop space               │
│    └── FOVEA-Lookahead: one-step look-ahead Bellman-like planning    │
│                                                                       │
│  [Execution] Tool-Integrated Agent                                    │
│    ├── VLM proposes crop d_prop                                      │
│    ├── FOVEA intercepts and refines → d*                             │
│    └── Execute refined crop → tool call → update history H_t         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Three Core Assumptions Driving Tractability

| 假设 | 内容 | 作用 | 代价 |
|------|------|------|------|
| **Factorised Belief** (Assump. 2.4) | p(ℓ,y) ≈ p(ℓ)·p(y) | 解耦空间搜索和语义识别，降维 joint posterior | 忽略空间-语义高阶相关性 |
| **Calibrated Visibility** (Assump. 3.2) | H(S\|z) ≈ 0 | 允许用 z 推断 visibility state S | 依赖 VLM 的自我校准能力 |
| **Ideal Observer** (Assump. 3.3) | H(y\|z,S=1) ≈ 0 | EIG ≈ H(y) × P(S=1)，化为几何目标 | Oracle 仅 68% 准确率，与实际存在 gap |

## Pros/Cons & Future Work

### Strengths

1. **Principled formulation**: 将启发式的 crop 选择提升为决策论框架 (S-BOED)，提供了统一的理论视角
2. **Training-free**: 完全在推理时运行，即插即用，不需要微调 backbone 模型
3. **Interpretable objective**: Coverage-Resolution product 有清晰的物理含义，显式建模了视场-分辨率 trade-off
4. **Strong empirical gains**: 在多个 benchmark 和 backbone 规模上一致有效，尤其适合搜索密集型场景
5. **Compute-Accuracy scaling**: 提供了一族不同计算预算下的操作点，而非单一固定策略

### Weaknesses / Limitations

1. **Proposal-limited (cold start)**: 如果 target 区域从未进入候选池，局部精炼无法恢复。单 seed 下 100% 的失败都是 proposal-limited
2. **Oracle gap**: 即使 perfect crop (Oracle)，准确率仅 68%，反映 backbone 的 reasoning 能力是独立瓶颈
3. **Inference overhead**: resolvability probing 增加输入/输出 token 成本（6.5x-9.5x input, 9.8x-55x output）
4. **Ideal Observer assumption**: 假设 resolved 后的 entropy collapse 在实践中不总是成立（hallucination on clear images）
5. **Single backbone evaluation**: 仅在 Qwen3-VL 系列上验证，在其他 VLM 架构上的泛化性待验证

### Future Work

1. **Uncertainty Calibration**: 改进 VLM epistemic uncertainty 的估计器，提升信息增益估计精度
2. **Amortised Inference**: 训练轻量 policy 直接预测 foveation actions，降低迭代搜索成本
3. **Adaptive Invocation**: 元策略决定何时激活 active foveation，而非每次 crop call 都触发

## Reading Q&A Record

| # | 问题 | 答案位置 | 解答 |
|---|------|---------|------|
| 1 | Perceptual Bandwidth 为什么是根本瓶颈？ | Section 2.2, Def. 2.1 | ViT encoder 将任意分辨率的图像投影到固定数量的 visual tokens。当全局压缩时，每个 token 的平均 spatial 面积过大，导致小目标/文字的像素信息在进入 LLM 之前就被 pooling 掉了。这不是语义推理问题，而是证据采集问题。 |
| 2 | Coverage-Resolution Objective 是如何从 EIG 推导出来的？ | Section 3.2, Eq. 5-8 | 在三个近似假设下：S ⊥ y (factorised belief), H(S\|z)≈0 (calibrated visibility), H(y\|z,S=1)≈0 (ideal observer)，EIG ≈ H(y) × P(S=1\|d)。P(S=1\|d) = Coverage × φ(d) 即为 coverage-resolution product。 |
| 3 | Information Cliff 是什么？为什么 greedy 不够？ | Section 3.1, Remark 3.1 | Wide view 能定位但不能读，zoom 能读但不能定位。单独看都 zero gain，但组合起来才有 high gain。这是 super-additive 的信息结构，需要 look-ahead 规划。 |
| 4 | Resolvability Probing 的本质是什么？ | Section 4.1, Eq. 10 | 不是精确的 EIG 估计，而是询问 VLM "这个 crop 包含足够回答问题的信息吗？" (Yes/No)。Monte Carlo 平滑后作为 coverage-resolution utility 的 empirical surrogate。 |
| 5 | FOVEA 为什么是 training-free 的？ | Section 4 & D.1 | 采用 tool interception 机制：VLM 正常提议 crop，FOVEA 在工具执行前拦截并精炼坐标。所有 probing 和优化都在已有的 VLM backbone 上进行，无需梯度更新。 |
| 6 | Cold start 是最主要的 failure mode 吗？ | Appendix H.2, Table 9 | 是的。在 50 例遥感子集上，single-seed 下 25 例全部失败且全部是 proposal-limited (target 不在候选池)。扩展为 multi-seed (9 seeds) 将 proposal-limited 从 25 降至 7 例。 |
| 7 | 为什么 Oracle 只达到 68% 而不是 100%？ | Appendix H.3 | Ideal Observer 假设要求 resolved 后 entropy collapse 到零，但实际 VLM 即使在完美 crop 下仍有 hallucination (如错误计数)。剩余 32% 错误反映的是 backbone 固有的 reasoning 瓶颈。 |

## Citation Landscape

### Core Theoretical Foundation
- **BOED 经典**: Lindley (1956), Chaloner & Verdinelli (1995), Rainforth et al. (2024)
- **Active Vision 经典**: Bajcsy (1988), Aloimonos et al. (1988), Najemnik & Geisler (2005)
- **Information Foraging**: Pirolli & Card (1999)

### Method Comparison
- **Discrete BOED for LLMs**: Kobalczyk et al. (2025) - 离散空间 vs 本文的连续空间
- **Visual Agents**: Thyme (Zhang et al., 2025b), LATTE (Ma et al., 2025), VisProg (Gupta & Kembhavi, 2023)
- **Retrieval-Augmented Perception**: RAP (Wang et al., 2025)
- **Latent Visual Reasoning**: Li et al. (2025a), Sun et al. (2025)

### Benchmarks
- MME-RealWorld-Lite (Zhang et al., 2024), CV-Bench (Tong et al., 2024), V\*Bench (Wu & Xie, 2023), HR-Bench (Team, 2024)

### Backbone & Tools
- Qwen3-VL (Bai et al., 2025), Grounding DINO (Liu et al., 2024b), SAM 2 (Ravi et al., 2024), Depth Anything (Yang et al., 2024), MinerU (Niu et al., 2025)

---

*Batch reading created on 2026-06-24*
