[← 返回 README](../README.md)

# 2. Problem Formulation: Active Vision as Experimental Design

## 一、Preview

本节建立 active visual reasoing 的概率框架，分三层构建：(1) 感知带宽的物理约束 (Sec 2.2)；(2) 从 latent state 到 observation 的生成过程 (Sec 2.3)；(3) 概率图模型 (Fig. 2) 统一上述组件。核心引入三个关键概念：Perceptual Bandwidth B、Resolution Probability φ(d)、Visibility Event S。

---

## 二、原始文本

We ground our approach in the rigorous framework of Bayesian optimal experimental design (BOED). We consider a VLM agent performing active visual reasoing over a high-resolution image I and a query Q. A comprehensive summary of notations is provided in Appendix B.

To bridge the gap between continuous visual signals and discrete token-based reasoing, we structure this formulation into three layers. First, we model the physical constraints of the VLM sensor, introducing the concept of perceptual bandwidth (Sec. 2.2). Second, we define the generative process, detailing how latent semantic states produce observable tokens through a resolution-gated mechanism (Sec. 2.3). Finally, we unify these components into a probabilistic graphical model (Fig. 2) that governs the agent's belief updates.

> 💡 **架构概述**: 三层层级结构 — Layer 1 (物理约束): 感知带宽如何限制传感器的信息采集能力；Layer 2 (生成过程): latent state (ℓ, y) 如何通过 visibility gate S 产生 observation z；Layer 3 (概率图模型): 统一所有随机变量和条件依赖关系，为后续的 Bayesian update 和 EIG 推导提供 formal basis。

![Figure 2](../images/f941b42b595ab62b58de57c02c05c7c55ea9960b675fe834a2ad99978a3b965e.jpg)

*Figure 2: Influence diagram of active visual reasoing. The foveation design d and latent target location ℓ jointly determine the visibility event S. This latent gate S modulates whether the observation z conveys information about the semantic target y. The agent's objective is to maximise the utility U, defined as the expected information gain over y, by actively managing the sensing design d.*

> 💡 **Figure 2 概率图关键路径**: d (foveation design) + ℓ (location) → S (visibility event) → z (observation). y (semantic target) → z 的路径被 S gate 控制: S=0 时 y 与 z 独立（噪声），S=1 时 z 携带 y 的信息。这精确建模了：只有当你"看见"（S=1）目标时，视觉观测才携带语义信息。

---

### 2.1. The Probabilistic System

Formally, we define the system as a tuple ⟨θ, D, Z⟩. Here, θ ∈ Θ represents the latent parameters (unknown world state), d ∈ D denotes the design (action), and z ∈ Z is the observation governed by a likelihood model p(z | θ, d).

### 2.2. Physical Constraints of Active Vision

To instantiate this framework, we first model the VLM as a stochastic sensor subject to rigorous resource limitations.

Definition 2.1 (Perceptual Bandwidth B). The fundamental bottleneck of VLM perception is the fixed encoder capacity (e.g., restricted token count (Dosovitskiy, 2020)), termed perceptual bandwidth B. This capacity induces a density-area trade-off (Najemnik & Geisler, 2005), where the information density ρ is defined as the ratio of the total bandwidth to the area A(d) of a foveation crop:

ρ(d) ≜ B / A(d)

Definition 2.2 (Resolution Probability φ). The probability that fine-grained features are resolved is governed by a saturation function f_sat (e.g., sigmoid) of information density:

φ(d) ≜ P(Resolved | d) = f_sat(ρ(d))  (Eq. 1)

This creates a physical trade-off: larger crops (high A) suffer from low density (φ → 0), while smaller crops (low A) achieve high density (φ → 1).

> 💡 **机制拆解 — Perceptual Bandwidth 的物理直觉**:

> B = 固定的 visual token 数量 (如 ViT-L/14 的 576 tokens).
> A(d) = crop 覆盖的图像面积 (归一化).
> ρ(d) = B / A(d) = 单位面积上的 token 数，代表有效信息密度.
> φ(d) = f_sat(ρ(d)) = 在给定密度下，fine-grained 特征能被分辨出来的概率.

> | 场景 | A(d) | ρ(d) | φ(d) | 能分辨什么 |
> |------|------|------|------|-----------|
> | 全局 view | 1.0 | B | ≈0 | 场景级上下文 |
> | 中等 crop | 0.25 | 4B | 适中 | 中等物体 |
> | 高分辨率 zoom | 0.01 | 100B | ≈1 | 文字、小目标 |

> 这说明 crop 操作本质上是在做 "bandwidth allocation"：通过限制空间区域，将固定 token budget 集中到更小的面积上，从而提高局部信息密度。

Remark 2.3 (Analogy: The Semantic Nyquist Rate). The saturation behavior of f_sat mirrors the classical Nyquist-Shannon Sampling Theorem (Shannon, 1949). We posit the existence of a critical density threshold τ_nyq, termed the semantic Nyquist Rate. When ρ(d) < τ_nyq, the encoder fails to distinguish between distinct local features, rendering fine-grained features indistinguishable. Conversely, once the density exceeds this threshold, the features become recoverable. In our framework, the sigmoid function serves as a differentiable approximation of this critical transition.

> 💡 **Semantic Nyquist Rate 类比**: 这个类比很深刻——就像 Shannon-Nyquist 定理要求采样率高于信号最高频率的两倍才能无失真重建，semantic Nyquist rate 要求信息密度超过阈值才能分辨 fine-grained 特征。当 ρ < τ_nyq 时，两个相邻但不同的小物体在 token 空间中"混叠"成一个模糊团块，语义区分完全丢失。

---

### 2.3. The Generative Process

Visual reasoing is not a static task but an interactive loop initiated by the agent's decisions. The generative process unfolds in three stages: action selection, physical interaction, and observation generation.

Design Space: Foveation Actions (D). Foveation actions are parameterised as spatial crops d = [u, v, w, h] ∈ [0, 1]^4. Crucially, d acts as a control variable for bandwidth allocation: by selecting a smaller region (w·h ≪ 1), the agent concentrates the fixed token budget onto a limited area, thereby boosting the local resolution density ρ(d) and increasing the resolution probability φ(d).

Latent Parameters: Semantic & Spatial State (θ). We define the unknown state space as θ ≜ {ℓ, y}, which factorises into two components: the spatial location ℓ of the relevant object and the semantic target y (e.g., the class label or text answer).

Agent's Belief State. At any time step t, the agent's knowledge about the latent parameters θ is captured by the joint posterior p_t(ℓ, y). In real-world visual reasoing, spatial location ℓ and semantic identity y are often coupled (e.g., context implies location). However, maintaining a full highdimensional joint posterior is computationally intractable for real-time inference.

Assumption 2.4 (Factorised Belief Approximation). To ensure tractability during the sequential design process, we adopt a mean-field approximation (Blei et al., 2017), assuming that the spatial search and semantic identification are momentarily decoupled during planning:

p_t(ℓ, y) ≈ p_t(ℓ) · p_t(y).

> 💡 **假设解析 — Factorised Belief**:

> 这个近似将 O(|L| × |V|) 的联合空间降为 O(|L| + |V|) 的独立空间。代价是忽略空间-语义的高阶相关性（如"船通常在水域"这种先验）。但作者明确指出这是 planning 近似，不是声称真实 posterior 真的是独立的。Sequential feedback（通过观测更新的交互历史）可以部分缓解这个 bias。

Under this assumption, we maintain two distinct belief maps: (1) A spatial belief p_t(ℓ) over the image coordinate space Ω, representing the agent's uncertainty regarding the object's location. (2) A semantic belief p_t(y), representing the uncertainty regarding the target's identity (e.g., class distribution), initialised by the linguistic priors in Q. This separation allows the agent to explicitly reaso about "where to look" (spatial uncertainty reduction) as a distinct objective from "what it is" (semantic identification), enabling the tractable EIG derivation in Section 3.

> 💡 **两层 belief 的功能分工**:
> - p_t(ℓ): "where to look" —— 指导空间搜索，告诉 agent 哪些区域还需要探索
> - p_t(y): "what it is" —— 指导语义消歧，降低候选答案的 uncertainty
> - 解耦后，EIG 的计算不再需要 joint (ℓ,y) 的积分，而是可以分别处理

The core physical constraint is that semantic information is inaccessible unless the target is physically captured. This interaction is modelled by the visibility event S, which acts as a latent bottleneck between the world state and the sensor.

Definition 2.5 (The Visibility Event). To bridge physical actions and semantic observations, we define a binary latent indicator S ∈ {0, 1}. This event represents whether the queried object is successfully captured by the encoder. Visibility occurs if and only if the object is both spatially encompassed and perceptually resolved:

P(S = 1 | ℓ, d) ≜ 1[ℓ ∈ d] (spatial coverage) × φ(d) (perceptual resolution)  (Eq. 2)

where 1[·] is the indicator function, ℓ is the latent spatial location, and φ(d) is the resolution probability (Eq. 1).

> 💡 **Visibility Event 的深层含义**:

> S=1 需要两个条件同时满足：
> 1. **空间覆盖**: 目标 ℓ 在 crop d 内部 (1[ℓ ∈ d])
> 2. **感知分辨率**: crop 的信息密度足够高，fine-grained 特征可分辨 (φ(d))
>
> 这精确建模了：即使你 crop 了正确区域（空间覆盖满足），但如果 crop 太大导致分辨率不足（φ(d) ≈ 0），目标仍然无法被"看见" (S=0)。两个条件缺一不可。
>
> 这也是 Coverage-Resolution 乘积的原始形式——S=1 的概率就是 Coverage × Resolution。

Observation Generation (z). Finally, the visibility state S gates the information flow to the VLM. The generative process concludes with the emission of the observation z, which is a mixture of signal and noise modulated by S.

Definition 2.6 (Observation Model: Resolution-Modulated Likelihood). The visual observation z is governed by a mixture model conditioned on the latent state of S. By the Law of Total Probability over the visibility event, the likelihood p(z | θ, d) is defined as:

p(z | θ, d) = P(S=1 | ℓ, d) · p(z | y, d) + (1 - P(S=1 | ℓ, d)) · p_0(z | d)  (Eq. 3)

where p(z | y, d) denotes the informative signal distribution when resolved (S=1), and p_0(z) denotes the background noise distribution (S=0). Note that p_0(z) is independent of y (y ⟂ z | S=0), representing the fact that an unresolved observation contains no semantic information about the target.

> 💡 **Observations model 的混合结构**:

> | 状态 | 概率 | 观测的统计特性 | 语义信息 |
> |------|------|---------------|---------|
> | S=1 (Gate Open) | P(S=1\|ℓ,d) | p(z\|y,d) 信号分布 | 有 (z 与 y 相关) |
> | S=0 (Gate Closed) | 1-P(S=1\|ℓ,d) | p_0(z\|d) 背景噪声 | 无 (y ⟂ z) |

> 这是一个 mixture model：agent 以概率 P(S=1) 收到一个 informative signal，以概率 1-P(S=1) 收到纯噪声。当 P(S=1) 接近 0 时（全局视图或 crop 了错误区域），无论模型多强，z 中都不包含 y 的信息——这就是 passive encoding 失败的根本原因。

The complete generative process and the resulting decisiontheoretic structure are summarised in the influence diagram in Figure 2, which serves as the basis for our sequential strategy derivation in Section 3.

---

## 三、Summary

- **概率系统**: ⟨θ, D, Z⟩ — latent state (ℓ, y), foveation action d, observation z.
- **感知带宽**: B = 固定 token budget → ρ(d) = B/A(d) → φ(d) = f_sat(ρ(d)) → φ 控制 fine-grained 特征的分辨概率。
- **Visibility Event**: S = 1 当且仅当空间覆盖 (ℓ ∈ d) 且感知分辨率 (φ(d) 高) 同时满足。
- **Observation Model**: mixture of signal (S=1) and noise (S=0), y ⟂ z when S=0.
- **Factorised Belief (Assump. 2.4)**: p(ℓ,y) ≈ p(ℓ)·p(y)，为 tractable EIG 推导提供前提。
- **核心结构**: d + ℓ → S → z ← y (gate 控制语义信息流)。
