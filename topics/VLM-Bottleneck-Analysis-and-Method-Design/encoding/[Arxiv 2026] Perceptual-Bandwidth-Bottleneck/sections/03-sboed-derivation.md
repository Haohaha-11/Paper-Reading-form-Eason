[← 返回 README](../README.md)

# 3. Active Visual Reasoning as S-BOED

## 一、Preview

本节是整个论文最核心的理论推导部分，分三步：(1) 定义序贯 EIG 目标并识别 "Information Cliff" 现象 (Sec 3.1)；(2) 在三个近似假设下推导 Closed-form Coverage-Resolution Objective 作为 EIG 的 tractable proxy (Sec 3.2)；(3) 形式化 Bayesian belief update，阐释正/负视觉证据如何重塑搜索分布 (Sec 3.3)。

---

## 二、原始文本

Building on the generative process established in Section 2, we now formulate the S-BOED for active information foraging through a three-stage derivation. We first define the theoretical sequential objective and identify the "Information Cliff" that renders standard greedy strategies insufficient (Sec. 3.1). To overcome computational intractability, we then derive a closed-form Coverage-Resolution utility under specific assumptions (Sec. 3.2). Finally, we present the idealised Bayesian belief update, which clarifies how positive and negative visual evidence should reshape the search distribution.

Throughout this section, all beliefs and information quantities at step t are conditioned on the interaction history $H_{{t-1}}$. For compactness, we write $p_{t}$(·) and $H_{t}$(·) for historyconditioned beliefs and entropies, and omit the subscript t in mutual-information terms when the conditioning is clear.

---

### 3.1. The Sequential Objective

The agent's goal is to select a sequence of designs $d_{{1:T}}$ to reduce uncertainty about the latent state θ = {ℓ, y}. We quantify uncertainty using the Shannon entropy H(θ).

Expected Information Gain (EIG). For a single step, the utility of a design d is the expected reduction in entropy or, equivalently, the mutual information between the observation and the parameters:

EIG(d) ≜ I(z; θ | d) = H(θ) - $E_{{z ~ p(z|d)}}$ [H(θ | z, d)].

Sequential Planning via Bellman Equation. In the sequential setting, the agent maintains a history $H_{{t-1}}$. The optimal strategy π* maximises the cumulative information gain over a horizon T. This is formally characterised by the value function V*, which satisfies the Bellman equation:

V*($H_{{t-1}}$) = max_{$d_{t}$} ( EIG($d_{t}$ | $H_{{t-1}}$) + $E_{{$z_{t}$ ~ p(z | $H_{t-1}$}$, $d_{t}$)} [ V*($H_{{t-1}}$ ∪ {($d_{t}$, $z_{t}$)}) ] ).  (Eq. 4)

> 💡 **机制拆解 — 序贯 EIG 的 Bellman 结构**:
> - Term 1 (immediate gain): 当前观测 $z_{t}$ 带来的直接信息增益
> - Term 2 (future gain): 基于当前观测后（通过历史更新），未来可能获取的期望信息增益
> - 直接计算 Eq. 4 需要：(1) 对高维 observation z 求期望，(2) 对未来所有可能轨迹求期望——在 gigapixel 图像空间不可行。

Computing Eq. 4 requires solving a nested expectation over high-dimensional observations z, which is computationally intractable. Furthermore, the structure of visual information poses a unique theoretical challenge:

Remark 3.1 (The Information Cliff). Standard active learning often assumes submodularity (diminishing returns) to justify greedy strategies. However, constrained active vision is often super-additive. Consider reading a small text: A wide view ($d_{wide}$) locates the text but cannot read it (φ ≈ 0); a random zoom ($d_{zoom}$) can read but misses the location (ℓ ∉ d). Both yield zero gain individually. Only their sequence yields high information:

I(y; $z_{wide}$, $z_{zoom}$) ≫ I(y; $z_{wide}$) + I(y; $z_{zoom}$).

This "information cliff" requires look-ahead planning.

> 💡 **Information Cliff 的数学含义**:

> 对任意两个 action $d_{wide}$ 和 $d_{zoom}$:
> - I(y; $z_{wide}$) ≈ 0 (定位了文字但无法读 → 不知道内容)
> - I(y; $z_{zoom}$) ≈ 0 (能读但没对准目标 → 看到的不是文字)
> - I(y; $z_{wide}$, $z_{zoom}$) ≫ 0 (先定位后缩放 → 能看到并能读)
>
> 这种 super-additivity 破坏了标准 submodularity 假设（通常 greedy 在 submodular 目标下提供常数因子近似保证）。在视觉搜索中，由于 perceptual bandwidth 的存在，信息的获取是非线性的、存在 threshold 效应——就像你需要在悬崖边缘聚集足够能量才能"跳过去"一样。

> | 属性 | 标准 Active Learning | 约束 Active Vision (本文) |
> |------|---------------------|--------------------------|
> | 信息动态 | Submodular (递减收益) | Super-additive (信息悬崖) |
> | Greedy 策略 | 常数因子近似 (near-optimal) | 可能 zero gain |
> | 需要 Look-ahead | 否 | 是 |
> | 典型场景 | 选问题减少分类不确定性 | 宽视图定位 + 缩放阅读 |

---

### 3.2. Derivation of the Tractable Coverage-Resolution Objective

While the ideal agent optimises the sequential Bellman equation, the nested expectations over high-dimensional observations z render it computationally intractable. In this section, we derive a closed-form approximation for the immediate task-relevant information gain that drives our practical cropselection strategy.

The Joint Information Objective. The ultimate goal of the agent is to resolve the user's query y. However, due to the physical coupling between "seeing" and "understanding", the agent must jointly reaso about the full latent state θ = {ℓ, y}. Theoretically, the total information gain decomposes into spatial and semantic components:

I(z; ℓ, y | d) = I(z; ℓ | d) (Localisation Gain) + I(z; y | ℓ, d) (Semantic Gain).

In our active vision setting, resolving y strictly necessitates localising ℓ. Rather than optimising these terms separately, we focus on maximising the marginal mutual information regarding the semantic target y.

> 💡 **定位-语义耦合的本质**: 在视觉上下文中，你不能直接推理 y 而跳过 ℓ——必须先确定"在哪里看"，然后才能"识别是什么"。这就是为什么 coverage（覆盖 ℓ）和 resolution（分辨 y）必须同时满足。

Decomposition via the Visibility Event. Directly computing I(y; z | d) is intractable. To simplify, we introduce the auxiliary visibility variable S. We first introduce a crucial assumption regarding the VLM's self-calibration:

Assumption 3.2 (Calibrated Visibility). We assume the observation z encodes sufficient statistics to determine the visibility state S (e.g., the model can distinguish between "blurry/empty" and "resolved" content). Mathematically, this implies H(S | z, d) ≈ 0, which allows us to approximate I(y; z | d) ≈ I(y; z, S | d). This assumption is empirically supported by recent findings that large-scale foundation models exhibit high calibration regarding their own predictive uncertainty (Kadavath et al., 2022).

> 💡 **Calibrated Visibility 假设**: 这意味着 VLM 能可靠地判断一个 crop 是"有用"还是"无用"。具体来说，如果模型看到一张模糊/空白/无关的图像，它能可靠地识别出"这里没有任何与 query 相关的信息"（S=0）。这个假设的合理性在 Appendix E 中有实证支持——给出 oracle crop 时模型能锁定目标，给出 distractor crop 时模型能拒绝并重新分配概率。

By the chain rule, $I_{t}$(y; $z_{t}$, S | d) = $I_{t}$(y; $z_{t}$ | d) + $I_{t}$(y; S | $z_{t}$, d). Since $I_{t}$(y; S | $z_{t}$, d) ≤ $H_{t}$(S | $z_{t}$, d) ≈ 0, Assumption 3.2 gives $I_{t}$(y; $z_{t}$ | d) ≈ $I_{t}$(y; $z_{t}$, S | d).

We then decompose the right-hand side as:

$I_{t}$(y; $z_{t}$, S | d) = $I_{t}$(y; S | d) (Term 1) + $I_{t}$(y; $z_{t}$ | S, d) (Term 2).

> 💡 **推导核心 — 通过 S 分解 EIG**: 引入中间变量 S (visibility event) 作为 "bridge" 来分解 intractable 的 EIG。这类似于在贝叶斯推断中使用 latent variable 来简化计算——S 担当了"信息门"的角色，决定了观测是否携带语义信息。

Term 1. As illustrated in Figure 2, the visibility event S is structurally determined by the spatial parameters (ℓ, d) and sensor physics. Under the Factorised Belief Assumption (Assumption 2.4), the semantic identity y is independent of the spatial location ℓ during the planning phase (ℓ ⟂ y). Consequently, since S is a function of ℓ, it follows that y ⟂ S | d. Thus, I(y; S | d) = 0.

Term 2. We expand the second term using the definition of conditional mutual information:

$I_{t}$(y; $z_{t}$ | S, d) = $P_{t}$(S=1 | d) · $I_{t}$(y; $z_{t}$ | S=1, d) + $P_{t}$(S=0 | d) · $I_{t}$(y; $z_{t}$ | S=0, d) (=0).

The second part vanishes because an unresolved observation (S=0) yields only background noise independent of y (y ⟂ z | S=0).

Combining these results, we define the Semantic Information Gain objective:

Ĩ_t(d) ≜ $P_{t}$(S=1 | d) · $I_{t}$(y; $z_{t}$ | S=1, d).  (Eq. 5)

> 💡 **Eq. 5 的结构解读**: Ĩ_t(d) = $P_{t}$(目标被发现且分辨) × $I_{t}$(如果发现, 能获得多少语义信息)。这是一个期望——"期望获得多少语义信息"= "目标可见的概率"×"可见条件下能获取的信息量"。前者由 Coverage-Resolution 决定，后者由 backbone VLM 的语义提取能力决定。

The Perfect Perception Approximation. Eq. 5 remains difficult to compute. To proceed, we rely on the strong semantic extraction capabilities of modern VLMs.

Assumption 3.3 (Ideal Observer / Entropy Collapse). For planning tractability, we model the VLM as an ideal observer. We assume that if the target is successfully foveated and resolved (S=1), the VLM extracts semantic information with high fidelity, causing the conditional entropy of y to collapse to zero:

H(y | z, S=1, d) ≈ 0.

> 💡 **Ideal Observer 假设的实践含义**: 这个假设将 VLM 视为"完美语义提取器"——只要给对 crop，它就能给出正确答案。在实践中当然不总是成立（Appendix H.3 表明 Oracle crop 只有 68% 准确率），但它将复杂的语义消歧问题简化为几何上的可见性最大化问题。换言之：**"搜索策略负责采集高质量证据，解释证据的任务委托给 backbone VLM"**。

This implies that the information gain from a successful foveation is approximately equal to the prior uncertainty:

I(y; z | S=1, d) = H(y) - H(y | z, S=1, d) ≈ H(y).

Under Assumption 3.3, the successful-foveation information term satisfies $I_{t}$(y; $z_{t}$ | S=1, d) ≈ $H_{t}$(y). Substituting this into Eq. 5 gives:

Ĩ_t(d) ≈ $H_{t}$(y) $P_{t}$(S=1 | d).  (Eq. 6)

The remaining term is the probability that the latent target location is both covered by the crop and resolved under the fixed perceptual bandwidth. Marginalising over the current spatial belief $p_{t}$(ℓ) gives:

$P_{t}$(S=1 | d) = (∫_{x∈d} $p_{t}$(x) dx) φ(d) ≜ $I_{t}$(d).  (Eq. 7)

Thus, Ĩ_t(d) ≈ $H_{t}$(y) $I_{t}$(d). Since $H_{t}$(y) is independent of the current design d, maximising the task-relevant information gain reduces to maximising the coverage–resolution objective $I_{t}$(d).

> 💡 **推导里程碑 — 从 EIG 到 Coverage-Resolution 的关键步骤**:

> EIG(y, z | d) = H(y) - E[H(y | z, d)]
> → 引入 S:  ≈ I(y; z, S | d)  [Assump 3.2]
> → 链式展开: = I(y; S | d) + I(y; z | S, d)
> → Term 1: = 0  [Assump 2.4: y ⟂ S]
> → Term 2: = P(S=1) × I(y; z | S=1, d) + 0
> → Ideal Observer: ≈ H(y) × P(S=1 | d)  [Assump 3.3]
> → Marginalize ℓ: = H(y) × (Coverage × φ(d))

> 最终目标：argmax_d [$p_{t}$(ℓ ∈ d) × φ(d)]，其中 H(y) 被消去（独立于 d）。

Proposition 3.5 (Task-Relevant EIG Approximation). Under the Factorised Belief Approximation (Assump. 2.4), the Calibrated Visibility Assumption (Assump. 3.2), and the Ideal Observer Approximation (Assump. 3.3), the taskrelevant EIG about the answer variable y satisfies:

$U_{t}$(d) ≜ $I_{t}$(y; $z_{t}$ | d) ≈ $H_{t}$(y) $I_{t}$(d),

where $I_{t}$(d) is the coverage–resolution objective defined in Eq. 7. Since $H_{t}$(y) is independent of the current design d, maximising $U_{t}$(d) reduces to maximising $I_{t}$(d).

The Coverage–Resolution Product. The objective $I_{t}$(d) has a simple interpretation. Visibility requires the latent target location ℓ to be both spatially covered by the crop and perceptually resolved under the fixed visual-token budget:

$I_{t}$(d) = (∫_{x∈d} $p_{t}$(x) dx) × φ(d).  (Eq. 8)
          (Coverage)              (Resolution)

The greedy design is therefore selected as d*_t = argmax_d $I_{t}$(d). This objective makes the coverage–resolution trade-off explicit: larger crops cover more posterior mass but reduce effective perceptual resolution, while smaller crops increase resolution but risk missing the target.

> 💡 **Coverage-Resolution Trade-off 的形象理解**:

> 想象你在一个巨大图书馆中找一本书上的一个特定段落：
> - 宽视角（高 Coverage, 低 Resolution）: 你看到整个书架的布局，但看不清任何书的标题
> - 窄视角（低 Coverage, 高 Resolution）: 你能看清书页上的字，但不知道自己在看哪本书
> - 最优策略：先用宽视角找到正确的书架和区域（Coverage dominant），再用窄视角读具体内容（Resolution dominant）
>
> $I_{t}$(d) = Coverage × φ(d) 这个乘积形式恰好捕捉了这个 trade-off 的乘法结构性——任何一个因子为零，乘积就是零，需要两者同时满足。

---

### 3.3. Formal Bayesian Belief Update

The coverage–resolution objective depends on the current spatial belief $p_{t}$(ℓ). In the idealised Bayesian model, this belief would be updated explicitly after each observation. Although our practical implementation approximates this update implicitly through the interaction history, the formal update clarifies how positive and negative visual evidence should reshape the search distribution.

Upon executing the optimal design d*_t and receiving observation $z_{t}$, the agent updates its spatial belief map $p_{t}$(ℓ) using Bayes' rule:

$p_{{t+1}}$(ℓ) = p($z_{t}$ | ℓ, d*_t) · $p_{t}$(ℓ) / $Z_{t}$,  (Eq. 9)

where $Z_{t}$ is the normalisation constant.

The core of this update is the spatial likelihood function p($z_{t}$ | ℓ, d*_t). To derive this from the joint observation model (Definition 2.6), we marginalise over the semantic target y. Relying on the Factorised Belief Assumption (Assumption 2.4), which treats y and ℓ as independent during the inference step, the likelihood simplifies to:

p($z_{t}$ | ℓ, d*_t) ≈ $E_{{y ~ $p_{t}$(y)}}$ [p($z_{t}$ | ℓ, y, d*_t)].

Substituting the mixture model from Eq. 3 into this expectation, the likelihood bifurcates based on whether the latent location ℓ falls within the crop region d*_t:

```
p($z_{t}$ | ℓ, d*_t) = {
  φ(d*_t) · $E_{y}$[p($z_{t}$ | y, d*_t)] + (1-φ(d*_t)) · $p_{0}$($z_{t}$ | d*_t),   if ℓ ∈ d*_t
  $p_{0}$($z_{t}$ | d*_t),                                                      if ℓ ∉ d*_t
}
```

> 💡 **Belief Update 的深层逻辑**:

> 两种情况：
> 1. **ℓ 在 crop 内**: 似然 = φ × $E_{y}$[信号] + (1-φ) × 噪声。如果 φ 接近 1 且 $z_{t}$ 确实是噪声（没有目标），则似然很小 → 后验概率降低（"这里没有目标"）。
> 2. **ℓ 不在 crop 内**: 似然 = 纯噪声 p_0。对于噪声输入，p_0 有较高的基准值 → 后验概率相对升高（"目标可能在未探索区域"）。

Interpretation and Negative Evidence. The term $E_{{y ~ $p_{t}$(y)}}$ [p($z_{t}$ | y, d*_t)] represents the marginal likelihood of the observation given that the target is resolved, averaged over the agent's current semantic belief. It quantifies how well the visual observation $z_{t}$ supports the hypothesis that any valid target y is present in the crop.

Crucially, this structure enables updates via negative evidence. Consider the scenario where the agent scans a candidate region with high effective perceptual resolution (φ(d*_t) ≈ 1) but receives an uninformative observation (i.e., $z_{t}$ matches the background noise $p_{0}$). For locations inside the crop (ℓ ∈ d*_t), the likelihood collapses to the signal probability, which is vanishingly small for noise inputs ($E_{y}$[p($z_{t}$ | y)] ≪ $p_{0}$($z_{t}$)). Conversely, for unvisited locations (ℓ ∉ d*_t), the likelihood remains high at the baseline noise level $p_{0}$($z_{t}$ | d*_t), reflecting consistency with the "not seen" state. Through normalisation, this discrepancy suppresses the probability mass within the visited area (d*_t) and effectively "pushes" the belief mass to the unvisited regions, driving exploration.

> 💡 **负证据的 Bayesian 解释 — "没看到也是一种信息"**:

> 这是本节最优雅的 insight。当 agent 用高分辨率扫描一个区域但没有找到目标时，这个"空"信号不是无用的——它通过 Bayesian update 将概率质量从已扫描区域"推"到未扫描区域。正因如此，"在哪里没有看到目标"是同样有价值的证据，它驱动了探索行为。
>
> 在 Appendix E 的实证中 (Figure 5b)：当给出 distractor crop（负证据），模型对 viewed region 的概率 mass 崩塌到接近 0，而这些 mass 被重新分配到剩余的 search space，对 true target 的置信度反而提升了。

---

## 三、Summary

- **Sequential EIG**: 序贯目标的 Bellman 形式——当前增益 + 期望未来增益。
- **Information Cliff**: 视觉感知的 super-additivity 破坏了 submodularity 假设，需要 look-ahead 规划。
- **Coverage-Resolution Objective 推导链**:
  1. 引入 S (Assump 3.2) → I(y; z) ≈ I(y; z, S)
  2. 链式展开 → Term 1 (=0 by Assump 2.4) + Term 2 (= P(S=1)×I(y;z|S=1))
  3. Ideal Observer (Assump 3.3) → I(y;z|S=1) ≈ H(y)
  4. Marginalize ℓ → P(S=1|d) = Coverage × φ(d)
  5. 最终: $I_{t}$(d) = Coverage × Resolution, $U_{t}$(d) ≈ $H_{t}$(y) × $I_{t}$(d)
- **Belief Update**: 正证据修正 spatial belief 锁向目标，负证据将概率质量推进到未探索区域——两者都驱动有效搜索。
- **三个假设的递进关系**: Factorised Belief → Calibrated Visibility → Ideal Observer，层层简化，最终将复杂的 EIG 计算降解为可处理的 Coverage-Resolution 乘积。
