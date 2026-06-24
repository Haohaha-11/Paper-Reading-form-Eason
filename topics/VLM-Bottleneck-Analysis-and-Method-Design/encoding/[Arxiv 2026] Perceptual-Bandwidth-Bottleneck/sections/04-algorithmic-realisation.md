[← 返回 README](../README.md)

# 4. Algorithmic Realisation

## 一、Preview

理论框架提供决策论目标，但在 gigapixel 空间精确推理不可行。本节实例化 FOVEA 作为 S-BOED 的 practical surrogate，包含两个核心组件：Resolvability Probing（估计 crop utility）和三种优化策略（Greedy/MCMC/Lookahead），它们在同一个 S-BOED 模板下提供了不同 compute-accuracy 操作点。

---

## 二、原始文本

The S-BOED formulation specifies a decision-theoretic objective, but exact inference is intractable in gigapixel image spaces. In particular, Eq. 8 depends on a spatial belief $p_{t}$(ℓ) over a continuous domain, an unknown resolution function φ(d), and, for non-myopic planning, expectations over future observations. We therefore instantiate the framework with FOVEA, a training-free procedure for Foveated Observation and Visual Evidence Acquisition. FOVEA should be understood as a practical surrogate instantiation of the S-BOED view rather than an exact solver with explicit posterior maps or exact EIG computation.

FOVEA uses the interaction history $H_{t}$ as a historyconditioned search state, so later crop proposals can depend on both positive and negative evidence from earlier views. Appendix E provides empirical evidence for this history-based calibration.

> 💡 **关键澄清**: FOVEA 不是 S-BOED 的严格求解器——它不维护显式的 $p_{t}$(ℓ) posterior map，也不计算精确的 EIG。它是一个"替身实现 (surrogate instantiation)"：用 interaction history 隐式近似 Bayesian belief state，用 resolvability probing 替代 φ(d) 的精确估计，用候选池搜索替代连续空间的 argmax。

Operationally, FOVEA has two main components: it estimates crop utility with a resolvability probe, and it optimises this utility with greedy sampling, MCMC-style refinement, or look-ahead planning.

---

### 4.1. Resolvability Probing

Zero-shot visual grounding in high-resolution regimes remains prone to spatial inaccuracies and hallucinations (Xiao et al., 2025; Su et al., 2025a). We therefore treat the VLM's initial crop proposal as a noisy spatial prior rather than ground truth. Around this proposal, FOVEA samples candidate foveations and scores each crop independently.

We introduce a binary resolvability signal r ∈ {0, 1}, where r = 1 denotes that the crop contains sufficient queryrelevant visual evidence for the VLM to answer. This signal is not an exact estimator of information gain; rather, it is an empirical surrogate for crop utility under the S-BOED view. We define:

Î(d) ≜ P(r = 1 | $I_{d}$, Q) ≈ P(VLM($I_{d}$, Q) = "Yes"),  (Eq. 10)

which estimates whether a candidate achieves a favourable coverage–resolution trade-off for the current query.

> 💡 **Resolvability Probing 是什麼？**

> 本质上是一个 Yes/No 问题问 VLM：
> - **Prompt**: "这个 crop 包含足够的信息来回答原始问题吗？"
> - **期望输出**: "Yes" 或 "No"
>
> 为什么这是一个好的 surrogate？
> 1. **直接对应 Coverage-Resolution**: "包含足够信息"等价于"目标在 crop 内且分辨率足够"
> 2. **可利用 VLM 的 calibrated uncertainty**: 模型对自己"能看到什么"有较好的元认知 (Kadavath et al., 2022)
> 3. **Monte Carlo 平滑**: 每次 probe 采样 N=3 次，取 "Yes" 比例，使 stochastic VLM 输出的期望趋于稳定
>
> Appendix F 验证了 probe 的有效性：oracle crop 的 probe score 为 0.633，distractor 和 random crop 仅为 0.187 (Cohen's d = 1.22)，strong separation。

> 💡 **与 Coverage-Resolution 的理论对应**:
> - $I_{t}$(d) = Coverage × φ(d) (Equation 8, 理论)
> - Î(d) = P("Yes" | $I_{d}$, Q) (Equation 10, 实践)
>
> 当 VLM 回答 "Yes" 时，意味着它隐式判断了：(1) 目标在 crop 内 (Coverage)；(2) 分辨率足以分辨 (Resolution)。因此 Î(d) 可以作为 $I_{t}$(d) 的 empirical proxy。

---

### 4.2. Optimisation Strategies

Given Î(d), FOVEA supports different optimisers. The greedy variant selects the candidate with the largest immediate resolvability score and is used as the efficient default. MCMC-style refinement improves local search by iteratively perturbing the crop proposal. For tasks with an information cliff, where the value of a view depends on what it enables next, we use a FOVEA-Lookahead that scores a candidate by the estimated resolvability of its simulated next state:

d*_t = argmax_{d ∈ $D_{cand}$} V̂(d, H_{t-1}).

This keeps the objective fixed while allowing the optimiser to vary with the compute budget and task difficulty.

> 💡 **三种优化策略的定位**:
> 
> | 策略 | 机制 | 搜索范围 | 计算代价 | 适用场景 |
> |------|------|---------|---------|---------|
> | **FOVEA-Greedy** | 在 seed 周围生成 3 个候选（seed, small, large），选 Î(d) 最高者 | 局部 | 低 (6.5x input tokens) | 通用，计算预算有限时 |
> | **FOVEA-MCMC** | Metropolis-Hastings: 随机扰动当前 crop → 以 Î(d) 为接受概率 | 中程扩展 | 中 (7.7x input tokens) | 需要更精细的局部搜索 |
> | **FOVEA-Lookahead** | 对每个候选，模拟 VLM 的下一步动作 → 评估未来状态的 Î(d) | 前瞻 | 高 (9.5x input, 55x output) | 搜索密集型 (如 remote sensing) |

Algorithm 1 FOVEA: S-BOED-Guided Local Perceptual Refinement

```
1: Require: Global image $I_{global}$, query Q
2: Input: Initial crop proposal $d_{seed}$
3: Generate a candidate pool $D_{cand}$ around $d_{seed}$, including the seed crop and local perturbations
4: for each $d_{i}$ ∈ $D_{cand}$ do
5:     Extract crop I_{$d_{i}$}
6:     Estimate utility Î($d_{i}$) ← P(r = 1 | I_{$d_{i}$}, Q)
7: end for
8: if strategy is LOOKAHEAD then
9:     d*_t ← argmax_{d ∈ $D_{cand}$} V̂(d, H_{t-1})
10: else
11:     d*_t ← argmax_{d ∈ $D_{cand}$} Î(d)
12: end if
13: $z_{t}$ ← VLM(I_{d*_t}, Q)
14: $H_{t}$ ← H_{t-1} ∪ {(d*_t, Î(d*_t), $z_{t}$)}
15: return $H_{t}$
```

> 💡 **Algorithm 1 的关键设计**:

> 1. **候选池生成** (Line 3): 从 seed 出发，生成 small (0.8x, 高分辨率) 和 large (1.5x, 高覆盖) 两个变体。这直接对应 Coverage-Resolution trade-off：small 偏向 resolution，large 偏向 coverage。
>
> 2. **独立评分** (Line 4-7): 每个候选独立 probe 并打分，而非比较式评分。这在 Appendix F.2 中被验证优于 VLM-direct 的联合比较——因为独立 probe 用高分辨率看每个 crop，而联合比较需要将 9 个区域同时压缩到 fixed token budget 中。
>
> 3. **交互历史** (Line 14): 将 refined crop 及其 score、observation 加入历史 $H_{t}$。这使得后续搜索可以依赖之前的正/负证据进行 Bayesian-style belief update（隐式地）。
>
> 4. **策略插拔** (Line 8-12): 选择器可以是 Greedy 也可以是 Lookahead——同一个 utility function Î(d)，不同的 selection logic。

> 💡 **FOVEA 与理论框架的 gap 分析**:

> | 理论组件 (Section 2-3) | 实践近似 (Section 4) | Gap |
> |------------------------|---------------------|-----|
> | $p_{t}$(ℓ): spatial belief | Interaction history $H_{t}$ | 隐式而非显式 posterior map |
> | φ(d): resolution probability | Î(d): "Yes" probability | 不是精确 φ 估计，是综合性 utility |
> | $I_{t}$(d) = Coverage × φ(d) | Î(d) via probing | probing score 隐含了 coverage + resolution |
> | argmax over continuous [0,1]^4 | argmax over discrete $D_{cand}$ | 离散候选池替代连续搜索 |
> | Bellman value V* | Lookahead V̂ | 一步前瞻而非完整 planning |

> 这些 gap 是 deliberate 的——在 gigapixel 空间中，精确 Bayesian inference 不可行，FOVEA 用可操作的 surrogate 实现了 S-BOED 的核心思想（以信息增益为目标选择观测）。

---

## 三、Summary

- **FOVEA 定位**: S-BOED 框架的 practical surrogate，不是严格求解器。
- **Resolvability Probing**: 用 VLM 的 "Yes/No" 判断作为 coverage-resolution utility 的 empirical surrogate，Monte Carlo (N=3) 平滑。
- **Greedy**: 3 候选（seed, 0.8x small, 1.5x large）+ argmax Î(d)。
- **MCMC**: Metropolis-Hastings 在 crop 参数空间迭代搜索。
- **Lookahead**: 模拟下一步 VLM 动作 → 评估未来状态的 resolvability → 选择能打开高信息路径的当前动作。
- **核心设计原则**: 将 VLM 的初始 crop 视为 noisy spatial prior，通过 resolvability probing 做 evidence-oriented 的验证和精炼。
- **工具拦截机制**: FOVEA 不改变 VLM 的行为生成过程，而是在工具执行前"静默拦截"并精炼 crop 坐标（详见 Appendix D.1）。
