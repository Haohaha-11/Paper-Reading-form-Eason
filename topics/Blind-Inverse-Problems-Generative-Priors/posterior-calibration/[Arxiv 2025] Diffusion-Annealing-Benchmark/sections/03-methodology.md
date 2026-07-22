[← 返回 README](../README.md)

# III. PROPOSED APPROACH — BIPSDA 框架

## 📌 预览

本节是全文方法主线。BIPSDA 用「解耦噪声退火」绕开 hijacking 的 noisy-likelihood 近似难题：每个迭代分两阶段——**prediction（预测）**：从当前 latent $\mathbf{m}(t)$ 出发，采一个既符合测量又符合去噪分布的 $\mathbf{m}(0)$；**corruption（腐蚀）**：把 $\mathbf{m}(0)$ 重新加噪到 $\mathbf{m}(t-\Delta t)$。预测阶段又拆成两个正交设计维度：(1) 怎么近似去噪分布 $\pi_{0|t}$（ODE / TU / TC 三选一）；(2) 怎么从预测分布采样（Lang / MAP / RTO 三选一）。3×3 = 9 个算法，DAPS = Lang-ODE、DiffPIR = MAP-TU，其余 7 个是新算法，RTO 是原创。

---

In this section, we introduce the proposed BIPSDA framework for solving Bayesian inverse problems using difusion models. Unlike the hijacking approaches discussed in the previous section, our framework uses a recently introduced technique called decoupled noise annealing [22], [23] to avoid approximation of the noisy likelihood function and prevent samples from getting stuck in low-density regions. In what follows, we first introduce the framework, which generalizes previously introduced decoupled noise annealing approaches [22], [23] and can yield new algorithms through flexible combinations of the framework’s design choices. After discussing the relationship between our framework and previously proposed decoupled noise annealing approaches, we give examples of concrete algorithms that can be realized within our framework.

## A. THE FRAMEWORK

As in other decoupled noise annealing approaches [22], [23], the proposed BIPSDA framework generates a sequence of iterates that are approximate samples from the noisy posterior distribution at a series of decreasing noise levels. Specifically, each iteration is comprised of two stages: a prediction stage that generates a posterior sample that is consistent with the current iterate and the measurements, and a corruption stage in which noise is added back to the predicted posterior sample.

> 💡 **机制拆解 (Hao 批注)**: 这就是「解耦」的精髓。传统 hijacking 是「小步长连续去噪」，相邻步强耦合；BIPSDA 是「每步先猛地预测回 $t=0$（生成一个完整的候选 clean sample），再重新加噪回 $t-\Delta t$」。因为每步的预测只依赖当前 $\mathbf{m}(t)$ 和测量 $\mathbf{y}$，不依赖上一步的精确轨迹，所以错误不会像 hijacking 那样累积传播——这正是解决「样本卡在低密度区」的关键。

Concretely, given the current iterate m(t) and the measurements y, the goal of the prediction stage is to obtain an approximate sample m(0) from the prediction distribution $\pi _ { 0 | t , \mathbf { y } } ( \mathbf { m } ( 0 ) | \mathbf { \mu } \mathbf { m } ( t ) , \mathbf { y } )$ . In the corruption stage, $\mathbf { m } ( t - \Delta t )$ is obtained by adding noise back to m(0), i.e., by sampling from the Gaussian noising distribution $\pi _ { t - \Delta t | 0 } ( \mathbf { m } ( t - \Delta t ) \mid \mathbf { m } ( 0 ) )$ . In [22], Zhang et al prove that if m(t) is a sample from $\pi _ { t \mid \mathbf { y } } ( \mathbf { m } ( t ) \mid \mathbf { y } )$ then this procedure yields a sample from $\pi _ { t - \Delta t | \mathbf y } ( \mathbf m ( t - \Delta t ) \mid \mathbf { y } )$ . For suficiently large T , once can assume that $\pi _ { T | \mathbf { y } } ( \mathbf { m } ( T ) \mid \mathbf { y } ) \approx { \mathcal { N } } ( \mathbf { m } ( T ) ; \mathbf { 0 } , \sigma ^ { 2 } ( T ) \mathbf { I } )$ . So starting from $\mathbf { m } ( T )$ , the prediction and corruption steps can be iteratively applied with the timestep annealed from T down to 0 to yield a sample from the posterior distribution.

> 💡 **公式批读 (Hao 批注)**: 理论保证在这——Zhang et al [22] 证明：若 $\mathbf{m}(t)$ 是 noisy 后验 $\pi_{t|\mathbf{y}}$ 的样本，则「预测回 0 + 重新加噪」得到的 $\mathbf{m}(t-\Delta t)$ 是 $\pi_{t-\Delta t|\mathbf{y}}$ 的样本。所以从 $T$（近似高斯）退火到 0 就得到后验样本。**但注意这个保证有个前提：预测阶段能从 $\pi_{0|t,\mathbf{y}}$ 精确采样**。而 $\pi_{0|t,\mathbf{y}}$ 依赖去噪分布 $\pi_{0|t}$，后者必须近似——这就是所有误差的入口，也是 9 个变体差异的来源。

The prediction stage of the algorithm outlined above requires sampling from the prediction distribution $\pi _ { 0 | t , \mathbf { y } } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) , \mathbf { y } )$ . Using Bayes’ rule and the conditional independence of y from $\mathbf { m } ( t )$ given m(0), it holds that [22]:

$$
\pi _ { 0 | t , \mathbf { y } } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) , \mathbf { y } ) \propto \pi _ { \mathrm { l i k e } } ( \mathbf { y } \mid \mathbf { m } ( 0 ) ) \pi _ { 0 | t } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) )
$$

Unfortunately, while the likelihood function is known, the denoising distribution $\pi _ { 0 \mid t } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) )$ is not and requires approximation. The prediction stage can therefore be broken down into two substages: approximation of the denoising distribution and sampling from the prediction distribution using the approximate denoising distribution. Here it is worth noting that the hijacking approaches discussed in the previous section also require approximation of the denoising distribution, and approximations used in the hijacking setting can also be used in the BIPSDA setting (e.g., the Tweedie formula based Gaussian approximation of Boys et al [33]). However, approximations in the hijacking setting must be chosen so that the integral in (8) and the subsequent score operation in (7) can be evaluated eficiently. In our setting, there are no such restrictions. In what follows, we use $\pi _ { \mathrm { a p r x } } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) )$ to denote an approximation of the denoising distribution $\pi _ { 0 \mid t } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) )$ .

> 💡 **机制拆解 (Hao 批注)**: 这是 BIPSDA 相对 hijacking 的**结构性优势**。预测分布 $\pi_{0|t,\mathbf{y}}\propto\pi_{\text{like}}(\mathbf{y}|\mathbf{m}(0))\cdot\pi_{0|t}(\mathbf{m}(0)|\mathbf{m}(t))$——注意似然直接作用在 $\mathbf{m}(0)$ 上（精确、无需近似），只有去噪分布 $\pi_{0|t}$ 需要近似。关键区别：hijacking 里近似必须满足「积分 (8) + 求分数 (7) 都能高效算」的苛刻约束；BIPSDA 里**没有这个约束**，可以自由选近似（甚至用 Langevin/MCMC 从近似预测分布采样）。这个自由度就是 9 个变体的设计空间。

In summary, the BIPSDA framework requires three steps at each iteration: approximation of the denoising distribution, sampling from the prediction distribution using the approximate denoising distribution, and corruption of the predicted sample using the Gaussian noising distribution. A full outline is provided in Algorithm 1. In what follows, we discuss previously proposed methods in the literature that fall within the BIPSDA framework, with particular attention paid to the techniques used for approximating the denoising distribution and for sampling from the approximate prediction distribution. We then give examples of novel algorithms that can be realized in our framework through diferent choices of the approximation distribution and sampling scheme.

```
Algorithm 1  Bayesian Inverse Problem Solvers through Difusion Annealing (BIPSDA)
1: Input: Decreasing timesteps [t_{N_A}, t_{N_A-1}, ..., t_0], likelihood π_like, noise schedule σ(t), trained score model s_{θ*}(m(t), t)
2: Output: Approximate sample from the posterior distribution π_post(m | y)
3: Initialize m(t_{N_A}) ~ N(m(t_{N_A}); 0, σ²(t_{N_A}) I)
4: for i = N_A, N_A-1, ..., 1 do
5:     Compute π_aprx as approximation of π_{0|t} obtained using score model
6:     Sample m(0) ~ π_like(y | m(0)) π_aprx(m(0) | m(t_i))
7:     Sample m(t_{i-1}) ~ N(m(t_{i-1}); m(0), σ²(t_{i-1}) I)
8: end for
9: Return m(t_0)
```

> 💡 **机制拆解 (Hao 批注)**: Algorithm 1 就是 BIPSDA 的全部骨架，只有三步：**Line 5 近似去噪分布**（选 ODE/TU/TC）→ **Line 6 从预测分布采样**（选 Lang/MAP/RTO）→ **Line 7 重新加噪**。Line 5 和 Line 6 是两个正交插槽，任意组合出 9 个算法。这个「插槽化」正是「统一框架」的实质——它把散落的 DAPS、DiffPIR 收编成一张 3×3 表，并暴露出 7 个空格是从没被人试过的新算法。

## B. RELATIONSHIP TO PRIOR WORK

Over the last two years, several difusion-based Bayesian inverse problem solvers have been proposed that use decoupled noise annealing. Of these approaches, the work of Zhang et al [22] is the most closely related to the proposed BIPSDA framework. In their approach, referred to as Decoupled Annealing Posterior Sampling (DAPS), the approximation of the denoising distribution (line 5 in Algorithm 1) takes the form of a Gaussian distribution with mean $\mathbf { m } _ { \mathrm { a p r x } }$ and covariance $\mathbf { C } _ { \mathrm { a p r x } } = \beta ( t ) ^ { 2 } \mathbf { I }$ , where the noise level β(t) is chosen using heuristics. The mean of $\mathbf { m } _ { \mathrm { a p r x } }$ is set as an estimate of $\mathbb { E } _ { 0 \mid t } [ { \bf m } ( 0 ) | \mathrm { ~ \bf ~ m } ( t ) ]$ with the estimate obtained by solving the probability flow ODE in (5) with initial value $\mathbf { m } ( t )$ . To sample from the corresponding approximate prediction distribution (line 6 in Algorithm 1), the DAPS approach uses MCMC algorithms such as the Euler–Maruyama method for discretizing Langevin dynamics (without Metropolis adjustment), or the Hamiltonian Monte Carlo algorithm [38].

> 💡 **机制拆解 (Hao 批注)**: DAPS = **ODE（去噪均值用概率流 ODE 反解）+ Lang（用 Langevin/HMC 从预测分布采样）** = Lang-ODE。它的两个软肋在这段就埋下伏笔：(1) 协方差 $\beta(t)^2\mathbf{I}$ 靠启发式选；(2) 无 Metropolis 校正的 Langevin 在多峰/非线性下会失稳（后面 phase retrieval 里彻底失败）。

Another approach that is closely related to the present work is the DifPIR algorithm of Zhu et al [23]. Like the DAPS algorithm, this approach models the denoising distribution as a Gaussian, with the mean set as an estimate of the conditional mean $\mathbb { E } _ { 0 \mid t } [ \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) ]$ and the covariance matrix chosen to be a scalar multiple of the identity. However, unlike DAPS, DifPIR estimates the conditional mean using Tweedie’s formula, which is exact up to error in the score model. Another key diference between DAPS and DifPIR is that DifPIR does not use Langevin dynamics to sample from the corresponding prediction distribution. Instead, inspired by plug-and-play algorithms for image restoration (see, $\mathrm { e . g . }$ , [39]), it solves a maximum a posteriori (MAP) estimation problem corresponding to the prediction distri bution. This problem can be eficiently solved using fast numerical optimization methods or, in some cases, closeform proximal operators [40]. While lacking theoretical guarantees, it is empirically observed that DifPIR can produce very diverse samples (see Figure 5 in [23]).

> 💡 **机制拆解 (Hao 批注)**: DiffPIR = **TU（去噪均值用 Tweedie 公式）+ MAP（求预测分布的 MAP 点，不采样）** = MAP-TU。相比 DAPS 的两个改动：(1) 均值改用 Tweedie（只受 score 误差影响，比 ODE 离散化更干净）；(2) 采样改成解优化问题（快，但**不是真采样**——只求众数）。后面实验会揭露这个「求 MAP 而非采样」的代价：系统性低估每个峰的方差。这正是本 topic 关心的 UQ 失真——DiffPIR 看着样本多样，实则方差偏小。

Other difusion-based Bayesian inverse problem solvers that use decoupled noise annealing include those proposed in [41] and [42]. In [41], a prediction-corruption decoupled noise annealing approach, dubbed SITCOM, was proposed. As in the proposed BIPSDA framework, the corruption stage of SITCOM is implemented by sampling from the Gaussian noising distribution. However, the prediction stage difers. In BIPSDA, the prediction stage aims to sample from the prediction distribution $\pi _ { 0 | t , \mathbf { y } } ( \mathbf { m } ( 0 ) | \mathbf { m } ( t ) , \mathbf { y } )$ . In the SITCOM approach, the prediction stage requires first evaluating a proximal operator to obtain a point m(t) that is consistent with the measurement y and close to the current iterate $\mathbf { m } ( t )$ Tweedie’s formula is then applied to m(t) to obtain the prediction m(0). In [42], an approach similar to the DifPIR algorithm was proposed. Unlike DifPIR, however, which solves for the MAP point of the prediction distribution, [42] proposes the use an iterative algorithm to maximize the likelihood function, with $\mathbf { m } _ { \mathrm { a p r x } }$ used to initialize the iterative algorithm.

> 💡 **机制拆解 (Hao 批注)**: 这段把 SITCOM [41] 和 [42] 也纳入「解耦退火」谱系，但它们的预测阶段偏离了 BIPSDA 的「从 $\pi_{0|t,\mathbf{y}}$ 采样」目标——SITCOM 先算 proximal 点再套 Tweedie，[42] 直接最大化似然。作者言下之意：这些变体没有严格对应「预测分布采样」这个概率目标，所以不在本文 9 变体的系统评测里。这是在划定框架的边界。

## C. SPECIFIC VARIANTS OF THE FRAMEWORK

The proposed BIPSDA framework contains many novel algorithms for solving Bayesian inverse problems with difusion models that can be realized through diferent choices of the approximation distribution and sampling scheme. In the following, we discuss these design choices and provide an outline of the algorithms that will be tested in the numerical studies.

### 1) Approximation of the denoising distribution

In this work, we consider three diferent approaches to build a Gaussian approximation to the denoising distribution (c.f. line 5 in Algorithm 1). That is, we consider approximations to $\pi _ { 0 | t } ( \mathbf { m } ( 0 ) | \mathbf { m } ( t ) )$ of the form

$$
\pi _ { \mathrm { a p r x } } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) ) = \mathcal { N } ( \mathbf { m } ( 0 ) ; \mathbf { m } _ { \mathrm { a p r x } } , \mathbf { C } _ { \mathrm { a p r x } } )
$$

The rationale for restricting our focus to a Gaussian approximation is that, while in principle arbitrarily complex approximations of the denoising distribution can be realized by first sampling from the denoising distribution $\pi _ { 0 \mid t } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) )$ using the reverse SDE and then fitting a model to the samples, these approximations may be computationally prohibitive in practice. The approaches considered are denoted as ‘ODE’, Tweedie Uncorrelated (‘TU’), and Tweedie Correlated (‘TC’).

> 💡 **机制拆解 (Hao 批注)**: 三种去噪分布近似全都是**高斯近似**（省算力），区别只在均值和协方差怎么得来。这也是一个隐含局限——去噪分布本身多峰时，高斯近似天生抓不住多峰结构。这就预示了 phase retrieval（多峰）会全军覆没：不是采样器不行，是**去噪分布的高斯近似**这一步就丢了多峰信息。

In the ‘ODE’ approach, which is based on the DAPS algorithm, the mean $\mathbf { m } _ { \mathrm { a p r x } }$ is obtained by solving the probability flow ODE in Eq. (5), and the covariance is chosen as $\mathbf { C } _ { \mathrm { a p r x } } ~ = ~ \beta ( t ) ^ { 2 } \mathbf { I }$ , with $\beta ( t ) ~ = ~ \mathcal { O } ( \sigma ( t ) )$

User-provided hyperparameters of this approach are the marginal variance of the denoising distribution $\beta ( t ) ^ { 2 }$ and the schedule of time steps in the discretization of the probability flow ODE.

> 💡 **消融解读 (Hao 批注)**: ODE 变体（DAPS 用）的痛点：需要用户提供两个超参——方差 $\beta(t)^2$ 和 ODE 离散化时间步表。后面讨论节明确指出 ODE 对离散化超参「highly sensitive」。这是作者力推 TC/TU 的动机之一：减少超参。

In the ‘TU’ approach, which is based on the DifPIR algorithm, the mean is obtained using Tweedie’s formula, i.e.,

$$
\mathbf { m } _ { \mathrm { a p r x } } = \mathbf { m } ( t ) + \sigma ^ { 2 } ( t ) s _ { \pmb { \theta } ^ { \ast } } ( \mathbf { m } ( t ) , t )
$$

and the covariance is set as $\mathbf { C } _ { \mathrm { a p r x } } = \beta ( t ) ^ { 2 } \mathbf { I }$ , with $\beta ( t ) = \mathcal { O } ( \sigma ( t ) )$ the only user-provided hyperparameter. We note that, as shown in Appendix A, under a particular choice of parametrization and discretization, the probability flow ODE estimate used in the ‘ODE’ approach can be made to coincide with Tweedie formula.

> 💡 **公式批读 (Hao 批注)**: TU（Tweedie Uncorrelated）用 Tweedie 公式直接算去噪均值：$\mathbf{m}_{\text{aprx}}=\mathbf{m}(t)+\sigma^2(t)s_{\theta^*}(\mathbf{m}(t),t)$。这一步是**闭式的**，只受 score 误差影响，比 ODE 反解干净（ODE 还引入离散化误差）。只剩一个超参 $\beta(t)$。协方差仍是各向同性 $\beta(t)^2\mathbf{I}$——「Uncorrelated」指的就是协方差不含相关结构。Appendix A 证明：取 $\sigma(t)=t$、单步反向 Euler 时，ODE 均值恰好等于 Tweedie 均值，所以本文实现里 ODE 和 TU 的均值差异主要来自多步 ODE 的额外近似。

Finally, in the $\mathrm { { } ^ { 6 } T C } ^ { \mathrm { { * } } }$ approach, which borrows some ideas from the hijacking approach of Boys et al [33], the mean $\mathbf { m } _ { \mathrm { a p r x } }$ is set as in the $^ { 6 } \mathrm { T U } ^ { , }$ approach, while the covariance is set using the generalized version of Tweedie’s formula [37], i.e.,

$$
\mathbf { C } _ { \mathrm { a p r x } } = \sigma ^ { 2 } ( t ) \left[ \mathbf { I } + \sigma ^ { 2 } ( t ) \nabla _ { \mathbf { m } ( t ) } s _ { \pmb { \theta } ^ { * } } ( \mathbf { m } ( t ) , t ) \right]
$$

This variant is free from user-provided hyperparameters, but requires access to accurate estimates of the Jacobian of the noisy prior score.

> 💡 **公式批读 (Hao 批注)**: TC（Tweedie Correlated）是作者在「解耦退火」语境下的原创引入（技术源自 Boys et al 的 hijacking）。均值同 TU，但协方差用**广义 Tweedie 公式**：$\mathbf{C}_{\text{aprx}}=\sigma^2(t)[\mathbf{I}+\sigma^2(t)\nabla s_{\theta^*}]$，即用分数的 Jacobian（二阶分数）估协方差，能捕捉去噪分布的**相关结构**（不再是各向同性）。卖点：**零超参、理论自洽**。硬伤：需要准确的分数 Jacobian，而 learned score 的 Jacobian 极不可靠——所以后面 learned score 实验里 TC 全部被弃用，只在 analytic score 下展示（且在两个非线性问题上最优）。这是一条清晰的「理论最优但工程受限」的线索。

### 2) Sampling from the prediction distribution

The BIPSDA framework also provides considerable flexibility regarding the choice of the sampling scheme. Three variants, denoted as Langevin (‘Lang’), maximum a posteriori estimation (‘MAP’), and randomize-thenoptimize (‘RTO’), are considered here.

The ‘Lang’ variant is based on DAPS [22] and employs Markov chain Monte Carlo (MCMC) algorithms, such as Langevin dynamics or Hamiltonian Monte Carlo [38], to sample the prediction distribution. When using uncorrected Langevin dynamics the step-size and the number of time steps are key user-provided hyperparameters that control the trade-of between accuracy of the samples and computational costs.

> 💡 **机制拆解 (Hao 批注)**: Lang（DAPS 用）是**唯一真正从预测分布采样**的方案（理论上无偏，若 MCMC 收敛）。但代价：无 Metropolis 校正时 step-size 和步数难调，且在多峰/非线性下不稳。后面结果里 Lang 反而在多峰（高噪 inpainting）和 phase retrieval 上翻车——原因是它「过度采样」到似然低密度区，反而高估方差。

The ‘MAP’ variant is based on DifPIR [23] and simply computes the maximum a posteriori (MAP) point of the prediction distribution using a deterministic optimization algorithm, rather than drawing samples. While lacking solid theoretical foundation, this approach is computationally eficient. User-provided hyperparameters depends on the specific choice of the optimizer used, and, as long as the optimization problem is solved with suficient accuracy, have limited efect on the quality of the solution.

> 💡 **机制拆解 (Hao 批注)**: MAP（DiffPIR 用）根本**不采样**，只求预测分布的众数点。快、超参不敏感，但概念上有问题：用一个点冒充一个分布的样本 → 系统性低估每个峰内的方差。它靠「不同 $\mathbf{m}(t)$ 起点 + corruption 随机性」制造样本多样性，能抓对多峰权重，但抓不对峰内 spread。这是本 topic 里最典型的「点估计好看 ≠ UQ 正确」案例。

Finally, the ‘RTO’ variant is a novel contribution of our work. It is inspired by the randomize-then-optimize (RTO) [24]–[26] algorithm for approximate sampling from the prediction distribution. Here the key idea is to obtain an approximate sample from the prediction distribution by first adding noise to both the measurement and the denoising distribution mean, then solving for the MAP point of the corresponding noise-perturbed prediction distribution. Concretely, assuming the Gaussian approximation of the denoising distribution in Eq. (9) and the Gaussian additive noise model in Eq. (1), ‘RTO’ generates an approximate sample m(0) from the prediction distribution by solving

$$
\mathbf { m } ( 0 ) = \underset { \mathbf { m } } { \mathrm { a r g m a x } } \ \pi _ { \mathrm { l i k e } } ( \mathbf { y } ^ { \prime } | \mathbf { m } ) \mathcal { N } ( \mathbf { m } ; \mathbf { m } _ { \mathrm { a p r x } } ^ { \prime } , \mathbf { C } _ { \mathrm { a p r x } } )
$$

where

$$
\mathbf { m } _ { \mathrm { a p r x } } ^ { \prime } \sim \mathcal { N } ( \mathbf { m } _ { \mathrm { a p r x } } ^ { \prime } ; \mathbf { m } _ { \mathrm { a p r x } } , \mathbf { C } _ { \mathrm { a p r x } } ) , ~ \mathrm { a n d } ~ \mathbf { y } ^ { \prime } \sim \mathcal { N } ( \mathbf { y } ^ { \prime } ; \mathbf { y } , \pmb { \Sigma } _ { \mathbf { z } } )
$$

denote the noise-perturbed mean and measurement, respectively. Note that this approach can be viewed as an application of Algorithm 1 in [25] to the subproblem of sampling from the prediction distribution (line 6 in Algorithm 1). Like the ‘MAP’ variant used in DifPIR, it has the advantage of enabling eficient MAP solvers to be employed for sampling from prediction distribution. Further, in the linear-Gaussian likelihood setting, where the corresponding prediction distribution is also Gaussian, it can be proven that this approach corresponds to exact sampling from the prediction distribution [24], [43].

> 💡 **公式批读 (Hao 批注)**: RTO 是本文原创技术核心。诀窍：先给去噪均值 $\mathbf{m}_{\text{aprx}}$ 和测量 $\mathbf{y}$ **各自加一次噪声**（$\mathbf{m}'_{\text{aprx}}$、$\mathbf{y}'$），再求 noise-perturbed 预测分布的 MAP 点。这样做的魔力在于：**线性高斯似然下，这个「加噪 + 求 MAP」在概率上等价于精确采样**（Bardsley 等的经典结论）。所以 RTO 兼得二者之长——用 MAP 的现成快速优化器（继承 DiffPIR 的效率），却实现了真采样（继承 DAPS 的保真、且不低估方差）。对比记忆：MAP 求一个众数（丢方差），RTO 求「加了噪的众数」（恢复方差）。这正是它在 X-ray/inpainting 上纠正 MAP 方差低估的原因。

<table><tr><td colspan="2"></td><td colspan="3">Denoising Dist. Approximation</td></tr><tr><td colspan="2"></td><td>ODE</td><td>TU</td><td>TC</td></tr><tr><td rowspan="3">Sampling</td><td>Lang</td><td>Lang-ODE</td><td>Lang-TU</td><td>Lang-TC</td></tr><tr><td>MAP</td><td>MAP-ODE</td><td>MAP-TU</td><td>MAP-TC</td></tr><tr><td>RTO</td><td>RTO-ODE</td><td>RTO-TU</td><td>RTO-TC</td></tr></table>

*TABLE 1. 九个示例算法的可视化表示，每个对应去噪分布近似（Algorithm 1 第 5 行）与采样方案（第 6 行）的不同组合。注意 ‘Lang-ODE’ 即 DAPS 算法，‘MAP-TU’ 即 DiffPIR 算法；其余七个变体均为新算法。*

> 💡 **Table 1 批读 (Hao 批注)**: 这张 3×3 表就是全文的「元贡献」——把 DAPS（Lang-ODE，左上）和 DiffPIR（MAP-TU，中）放进同一坐标系，暴露出 7 个从未被系统探索的格子。行=预测分布采样方式（Lang/MAP/RTO），列=去噪分布近似（ODE/TU/TC）。后续所有实验表都按这 9 行组织。**读表技巧**：横向对比看「采样器选择」的影响（如 MAP vs RTO 的方差低估问题），纵向对比看「去噪近似」的影响（如 TU 普遍优于 ODE）。作者主推的组合是 **RTO-TU**（learned score 下多次夺冠）和 **RTO-TC/TC 系列**（analytic score 下非线性问题最优）。

### 3) Analyzed variants

Mixing and matching diferent variants to approximate the denoising distribution with those to sample from the prediction distribution, we obtain the 9 variants summarized in Table 1. The DAPS and DifPIR algorithms correspond to the ‘Lang-ODE’ and ‘MAP-TU’ variants of our framework, respectively.

---

## 🔖 Section 总结

### 数据流（BIPSDA 单步迭代）
1. **输入**：当前 latent $\mathbf{m}(t_i)$ + 测量 $\mathbf{y}$
2. **Line 5 去噪近似**：$\pi_{\text{aprx}}=\mathcal{N}(\mathbf{m}_{\text{aprx}},\mathbf{C}_{\text{aprx}})$，均值/协方差由 ODE/TU/TC 决定
3. **Line 6 预测采样**：从 $\propto\pi_{\text{like}}\cdot\pi_{\text{aprx}}$ 采 $\mathbf{m}(0)$，用 Lang/MAP/RTO
4. **Line 7 腐蚀**：$\mathbf{m}(t_{i-1})\sim\mathcal{N}(\mathbf{m}(0),\sigma^2(t_{i-1})\mathbf{I})$
5. **输出**：$\mathbf{m}(t_{i-1})$，退火到 $t_0$ 即后验样本

### 关键设计维度速查
| 维度 | 选项 | 均值/采样来源 | 超参 | 特点 |
|------|------|--------------|------|------|
| 去噪近似 | ODE | 概率流 ODE 反解 | $\beta$ + 时间步表 | 离散化敏感 |
| | TU | Tweedie 闭式 | $\beta$ | 干净、只受 score 误差 |
| | TC | Tweedie + 二阶分数协方差 | 无 | 零超参但需可靠 Jacobian |
| 预测采样 | Lang | Langevin/HMC | step/步数 | 真采样但多峰易失稳 |
| | MAP | 求众数点 | 优化器相关 | 快但低估峰内方差 |
| | RTO | 加噪 + 求 MAP | 优化器相关 | 快 + 线性高斯下精确采样 |

### 核心洞察
1. **解耦退火**切断相邻步耦合，从根上避开 hijacking 的 noisy-likelihood 近似。
2. **RTO = MAP 的效率 + 真采样的保真**：线性高斯下可证明为精确采样，是纠正 MAP 方差低估的关键。
3. **TC 理论最优但受限**：需要可靠的分数 Jacobian，learned score 下不可用。

### 可追问点
- 为什么高斯近似去噪分布是 phase retrieval 失败的根源？→ 去噪分布多峰时高斯天生抓不住，采样器再好也补不回来。
