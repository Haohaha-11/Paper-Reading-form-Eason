[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览

引言先交代扩散/score-based 模型为什么好用（单一学习对象 score 可跨任务复用），再点出本文的靶子：**canonical 任务是从 $\mu_y(x)\propto e^{R_y(x)}\rho_*(x)$ 这种 reward/measurement tilted 后验采样**；即使有完美 score，最坏情况下后验采样是计算不可行的，所以实践里都用启发式 guidance（以 DPS 为代表）。然后作者列出两个**至今悬而未决**的基本问题：(1) DPS 的偏差到底偏向谁、为什么 STSL 这类补救有效；(2) 低温下 DPS 数值不稳定，early-stopping 到底改了什么分布。最后给出三点贡献。

---

Difusion and score-based generative models [Sohl-Dickstein et al., 2015, Ho et al., 2020, Song and Ermon, 2020, Song et al., 2021] have become the workhorse of modern generative modeling, powering text-to-image systems [Rombach et al., 2022, Ramesh et al., 2021, Saharia et al., 2022, Dhariwal and Nichol, 2021] and an expanding range of scientific and medica inverse problems [Song et al., 2022]. Their flexibility hinges on a single learned object (the score ∇ log ρ of the noised data distribution) which can be repurposed across downstream tasks without retraining.

> 💡 **问题动机（Hao 批注）**: 这段是"为什么值得研究"的铺垫。关键点是 score 的**可复用性**：一个 $\nabla\log\rho_t$ 训练好后，text-to-image、逆问题、reward 对齐都能共用，不用重训。正因为大家都拿同一个 prior score 去"拼"各种下游后验，DPS 这类 plug-and-play guidance 才成了事实标准——也正因如此，"这套拼法到底采到的是不是真后验"才成了一个影响面极大的理论问题。

A canonical such task is sampling from a posterior of the form $\mu _ { y } ( x ) \propto e ^ { R _ { y } ( x ) } \rho _ { * } ( x )$ , encompassing both classical inverse problems $y = A ( x )$ +ϵ and reward-tilted generation [Daras et al., 2024]. Even granting access to a perfect score oracle, posterior sampling is computationally intractable in the worst case [Gupta et al., 2024], so practical algorithms rely on heuristic guidance that approximates the time-dependent posterior score. An early and influential such heuristic is Difusion Posterior Sampling (DPS) [Chung et al., 2023]: it replaces the intractable conditional score $\nabla _ { x _ { t } } \log p ( y \mid x _ { t } )$ by the gradient of the reward evaluated at the Tweedie posterior mean $\hat { x } _ { 0 } ( x _ { t } ) \stackrel { \cdot } { = } \mathbb { E } [ X _ { 0 } \mid X _ { t } = x _ { t } ]$ [Robbins, 1956], yielding a plug-and-play guidance compatible with any pretrained score network. Its simplicity has made DPS the de-facto baseline for inverse problems and inspired a line of research targeting its known weaknesses: manifold-constrained gradients [Chung et al., 2022], denoising restoration [Kawar et al., 2022], pseudoinverse-guided difusion [Song et al., 2023], latent-space extensions [Rout et al., 2023b], second-order Tweedie corrections [Rout et al., 2023a, Boys et al., 2024], proximal approaches to decrease the gradient computation burden [Rout et al., 2025], filtering and SMC-based reweightings [Dou and Song, 2024, Wu et al., 2024, MOUFAD et al., 2025], and recent drift-control schemes [Ren et al., 2026, Guo et al., 2026, Anil et al., 2026].

> 💡 **机制拆解：DPS 的核心近似（Hao 批注）**: 这里点破了 DPS 偏差的**源头动作**，务必记牢——真正需要的是时间相关的条件 score $\nabla_{x_t}\log p(y\mid x_t)$，但它不可算；DPS 用了两步替换：
> 1. 先用 Tweedie 公式算出后验均值 $\hat{x}_0(x_t)=\mathbb{E}[X_0\mid X_t=x_t]$（一个 denoise 到底的点估计）；
> 2. 把 reward 的梯度**在这个点估计上求值** $\nabla R_y(\hat{x}_0(x_t))$，当作 guidance。
> 
> 这就是全部偏差的种子。正规做法应是对 $X_0$ 的整个条件分布取期望 $\mathbb{E}[e^{R(X_0)}\mid X_t]$，DPS 却把期望和非线性函数交换了顺序：$e^{R(\mathbb{E}[X_0])}$ 而非 $\mathbb{E}[e^{R(X_0)}]$（Section 3 会精确写成 $h_t^{DPS}\neq h_t^{OU}$）。由 Jensen 不等式，这两者一般不等，差距正比于条件方差 $\Sigma_t$——这就预告了为什么最终偏差公式里会出现 $\Sigma_t$。**注意：SMC/filtering（Dou、Wu、MOUFAD）那类方法是用重加权去逼近真后验，而本文关心的是"不重加权的 DPS 本身偏在哪"。**
> 
> **对本课题的延伸**：在盲逆问题里 $A$ 未知，reward 变成 $R_{y,\varphi}(x)=-\|y-A_\varphi(x)\|^2/2\sigma^2$，$\hat{x}_0$ 的点估计误差会同时通过 $\varphi$ 和 $x$ 两条通道进入 guidance。本文的分析告诉我们：只要还在用"denoise 到 $\hat{x}_0$ 再评 reward"这套 plug-and-play 逻辑，偏差就无法通过调 $A_\varphi$ 消掉。

Yet despite this flurry of activity, two basic questions remain open. First, the DPS approximation is biased even for Gaussian-mixture priors with quadratic rewards; but which samples does this bias over- or under-represent, and why do correctives like STSL improve performance? Existing analyses establish convergence under restrictive assumptions on the prior or measurement operator [Xu and Chi, 2024, Parulekar et al., 2025, Moitra et al., 2026] or treat the algorithm as a black box, leaving its preferred classes unexplained. Second, in the low-temperature regime needed for hard measurement constraints in image inverse problems, standard DPS is numerically unstable. Practitioners routinely fall back on early guidance-stopping and trajectory-dependent step sizes, but the efect of these heuristics on the sampled distribution has never been quantified.

> 💡 **两个开放问题（Hao 批注）**: 这是全文的"痛点定义"，直接对应我们的 posterior-calibration 课题：
> - **Q1（偏差的方向性）**：连最简单的"高斯混合 prior + 二次 reward"都能证明 DPS 有偏——注意这是可解析的 toy case，说明偏差不是数值假象。但已有分析要么加强假设证收敛（对我们无用，因为真实图像 prior 不满足），要么把算法当黑箱。本文要给出**逐点的 over/under-sampling 图谱**。
> - **Q2（不稳定与 early-stopping）**：低温 = 强制满足硬约束 $A(x)=y$ 时需要的大 guidance，这时 forward-Euler 会炸。工程上靠 early-stopping 和轨迹相关步长救场，但**没人量化过这改变了目标分布**。
> 
> 这两个问题正是"校准"的理论前提：如果不知道偏差偏向谁、不知道 early-stopping 改了什么分布，就无法判断 coverage/SBC 的失败是采样器偏差还是模型误设。

Contributions. We close both gaps with a unified analysis based on the classical Feynman-Kac formula [Karatzas and Shreve, 1991], complementing recent stochastic-analytic perspectives on guidance [Bruna and Han, 2024, Ren et al., 2026, Guo et al., 2026].

(i) An exact bias formula for DPS. In Section 3, we derive a pointwise Radon–Nikodym weight ω(x) relating the DPS-induced distribution to the true posterior. Using trajectory reversal, this weight can be written as an expectation over Ornstein–Uhlenbeck paths. The spatially varying part of the reaction term c<sub>DPS</sub> captures the alignment between conditional covariance and reward curvature, identifying where DPS over- or under-samples.

(ii) STSL-type bias reduction. We identify the spectral structure of the DPS bias: it is amplified where the data manifold has high conditional uncertainty along rewardsensitive directions. This motivates an auxiliary potential drift ∇U that steers trajectories toward lower-uncertainty regions and flattens the spatially varying part of the DPS reaction term. The trace-of-covariance choice $U ( t , x ) = \mathrm { t r } ( \hat { \Sigma } _ { t } ( x ) )$ recovers the empirically successful STSL correction [Rout et al., 2025] and connects naturally to recent neural drift-control approaches [Ren et al., 2026, Guo et al., 2026].

(iii) Quantifying low-temperature instability and early stopping. Finally, in Section 5 we show that the standard implementation of DPS systematically violates the stability condition of the forward-Euler of the bias vector field, leading to oscillations. Practitioners have implemented early-guidance-stoppin $\mathrm { { l g } ^ { 2 } }$ as a way to mitigate them. We are the first to characterize the early-guidance-stopping heuristic as a weighted version of the prior.

> 💡 **三点贡献的证据链（Hao 批注）**: 把三点贡献映射到后面各节，方便复现路线：
> - **(i) → Section 3 + Appendix D**：核心产出是 $\mu_y(x)=\omega(x)\,\nu_y^{DPS}(x)$（Theorem 1）。$\omega$ 有两种 FK 表示（backward 沿 DPS SDE、forward 沿 OU）。**"exact bias formula"意味着：只要有 score oracle 及其 Jacobian，就能用重要性权重 $\omega$ 把 DPS 样本精确纠回真后验**——这对我们做 calibration 是一个可操作的诊断/纠偏工具（虽然 $\omega$ 本身要蒙特卡洛估计，代价高）。
> - **(ii) → Section 4**：把 STSL 从"经验 trick"提升为"让 $c_{\text{DPS}}$ 空间波动变小的 drift"。关键洞察是偏差的**谱结构**：$\tilde{c}_{\text{DPS}}=\frac{1}{(e^t-e^{-t})^2}\sum_i\lambda_i^2\gamma_R^i$，在"$\lambda_i$ 大（高不确定方向）× reward 敏感"处被放大。
> - **(iii) → Section 5 + Appendix E,G**：instability 是 forward-Euler 对 unsquared norm $\|A(x)-y\|_2$ 的标准病态（梯度在约束处不消失只归一化），必然违反稳定判据；early-stopping 被刻画成 prior 的加权 tilt（Theorem 2）。
> 
> **注意**：三点里 (i) 是理论内核，(ii)(iii) 是把内核套到两个已知现象上做解释。对我们课题最有价值的是 (i) 的框架——它是把"采样器偏差"变成可分析对象的通用机器。

---

## 🔖 Section 总结

### 核心洞察
1. **靶子定义清楚**：DPS = 用 $\hat{x}_0(x_t)$ 处的 reward 梯度替代不可算的条件 score，plug-and-play 但有偏。
2. **偏差种子**：期望与非线性交换顺序，$e^{R(\mathbb{E}[X_0])}\neq\mathbb{E}[e^{R(X_0)}]$，差距由条件协方差 $\Sigma_t$ 控制。
3. **两个开放问题**：偏差方向性（Q1）与低温不稳定/early-stopping 的分布刻画（Q2），都用 Feynman–Kac 统一解决。
4. **可复用工具**：Theorem 1 的权重 $\omega$ 原则上能重要性加权纠偏；这对做 posterior 校准是直接相关的诊断量。

### 可追问点
- 已有收敛性分析（Xu & Chi、Parulekar、Moitra）加了哪些"restrictive assumptions"？为什么在真实图像 prior 上失效？（Related Work / Appendix B.1）
- $\omega$ 的蒙特卡洛估计代价多大？在高维图像上可行吗？（Fig 2 用了 20 条轨迹估计，仅 2D toy）
