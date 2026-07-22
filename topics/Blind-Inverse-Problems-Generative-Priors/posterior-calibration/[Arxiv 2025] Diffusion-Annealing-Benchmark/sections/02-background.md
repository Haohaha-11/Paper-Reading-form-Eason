[← 返回 README](../README.md)

# II. BACKGROUND

## 📌 预览

背景节铺三块地基：(A) 贝叶斯逆问题的基本符号（似然、先验、后验）；(B) 扩散模型的 VE-SDE 噪声过程、反向 SDE / 概率流 ODE、分数匹配训练；(C) 一类叫「hijacking（劫持）」的扩散逆问题采样法及其**致命缺陷**——它需要近似「noisy likelihood」，而多峰先验下这个近似在大 $t$ 处会崩，且小步长退火无法纠正。这个缺陷正是第 III 节 BIPSDA「解耦退火」要绕开的东西。

---

In this section, we provide relevant background on Bayesian inverse problems and difusion models. We also provide a brief overview of a popular class of algorithms, referred to here as hijacking algorithms, for solving Bayesian inverse problems with difusion models. This overview serves to both motivate the proposed framework and to introduce algorithmic ideas relevant to the present work.

## A. BAYESIAN INVERSE PROBLEMS

In inverse problems, the relationship between the measurements $\mathbf { \bar { y } } ~ \in ~ \mathbb { R } ^ { K }$ and the unknown variable of interest m $\in \mathbb { R } ^ { D }$ is captured by the likelihood function $\pi _ { \mathrm { l i k e } } ( \mathbf { y } \mid \mathbf { m } )$ , where here and throughout the remainder of this work we assume the measurements and the unknown variable lie in finite-dimensional Euclidean spaces. Concretely, under the assumption of additive Gaussian noise, the measurements can be written as

$$
\mathbf { y } = f ( \mathbf { m } ) + \mathbf { z } , \quad \mathbf { z } \sim \mathcal { N } ( \mathbf { z } ; \mathbf { 0 } , \Sigma _ { \mathbf { z } } )
$$

where $f : \mathbb { R } ^ { D } \to \mathbb { R } ^ { K }$ is known as the forward model, and $\pmb { \Sigma _ { z } } \in \mathbb { R } ^ { K \times K }$ is the covariance matrix of the noise distribution. The corresponding likelihood function is given as $\pi _ { \mathrm { l i k e } } ( \mathbf { y _ { \alpha } } | \mathbf { \phi } \mathbf { m } ) = \mathcal { N } ( \mathbf { y } ; f ( \mathbf { m } ) , \pmb { \Sigma } _ { \mathbf { z } } )$

By Bayes’ Theorem, the posterior distribution $\pi _ { \mathrm { p o s t } } ( \textbf { m } | \textbf { y } )$ of the unknown variable is related to the likelihood function and prior density function $\pi _ { \mathrm { p r } } ( \mathbf { m } )$ by the following expression:

$$
\pi _ { \mathrm { p o s t } } ( \mathbf { m } \mid \mathbf { y } ) \propto \pi _ { \mathrm { l i k e } } ( \mathbf { y } \mid \mathbf { m } ) \pi _ { \mathrm { p r } } ( \mathbf { m } )
$$

The goal of Bayesian inverse problems is to characterize the posterior distribution. In particular, depending on the application, various quantities, such as the mean, covariance, or higher-order moments of the posterior, may be of interest. While for certain choices of priors and likelihood functions (e.g., linear-Gaussian likelihood and Gaussian prior), these quantities can be computed in closed form, in general this is not tractable, and in practice Monte Carlo methods are often used to estimate the quantities of interest from samples.

> 💡 **公式批读 (Hao 批注)**: 这里是全文的符号契约。前向模型 $f$、加性高斯噪声 $\mathbf{z}$、似然 $\pi_{\text{like}}$、先验 $\pi_{\text{pr}}$、后验 $\pi_{\text{post}}\propto\pi_{\text{like}}\pi_{\text{pr}}$。注意作者反复强调「characterize the posterior」不等于「给点估计」——它要的是均值、协方差、高阶矩。这就是为什么后面评测不看 PSNR 而看矩差异。关键区分：只有线性高斯似然 + 高斯先验才有闭式后验，一般情况必须靠 Monte Carlo 从样本估计——这就把「采样器质量」推到了舞台中央。

## B. DIFFUSION MODELS

Broadly, the goal of generative modeling can be described as follows: given a set of samples $\mathcal { M } = \{ \mathbf { m } _ { i } \} _ { i = 1 } ^ { N _ { s } }$ from a distribution of interest with density function $\pi _ { 0 } ( \mathbf { m } )$ 2 obtain a set of samples from $\pi _ { 0 } ( \mathbf { m } )$ that are not in M. In difusion modeling, these samples are obtained by reversing a predefined noising process [14], [15]. This noising process can be written as a stochastic diferential equation (SDE). In particular, the variance exploding (VE) form of the difusion SDE can be written as follows [15]:

$$
d { \bf m } ( t ) = \sqrt { 2 \dot { \sigma } ( t ) \sigma ( t ) } d { \bf w } ( t ) , \quad t \in [ 0 , T ]
$$

where $\sigma ( t )$ is a predefined noise schedule, ${ \bf w } ( t )$ is the Wiener process, and ${ \bf m } ( 0 ) \sim \pi _ { 0 } ( { \bf m } )$ . Under this SDE, the conditional distribution of $\mathbf { m } ( t )$ given m(0) is given by the noising distribution $\pi _ { t | 0 } ( \mathbf { m } ( t ) \mid \mathbf { m } ( 0 ) ) =$ $\mathcal { N } ( \mathbf { m } ( t ) ; \mathbf { m } ( 0 ) , \sigma ^ { 2 } ( t ) \mathbf { I } )$ , where here and throughout the remainder of this work we have assumed $\sigma ( 0 ) = 0$ for simplicity. This implies that solving the forward SDE is equivalent to sampling from a Gaussian. Further, for large enough T , the distribution $\pi _ { T } ( \cdot )$ will be approximately Gaussian.

> 💡 **公式批读 (Hao 批注)**: 采用的是 VE（variance exploding）形式，噪声调度 $\sigma(t)$ 直接控制注入方差。关键性质：给定 $\mathbf{m}(0)$，加噪分布 $\pi_{t|0}$ 就是均值不变、方差 $\sigma^2(t)\mathbf{I}$ 的高斯。这个「条件高斯」性质是后面 Tweedie 公式精确成立的前提。后文实现里取 $\sigma(t)=t$、$T=10$，就是让 ODE 更新恰好和 Tweedie 公式重合（见 Appendix A）。

To sample from $\pi _ { 0 } ( \mathbf { m } )$ , we can first sample from the Gaussian approximation of $\pi _ { T } ( \mathbf { m } ( T ) )$ and then reverse the noising process in (3). Here there are two main methods to reverse the noising process [15]: an SDE-based approach and an ordinary diferential equation (ODE) based approach. The SDE-based approach leverages the remarkable fact [29] that (3) admits a time-reversal that matches the marginal distributions $\pi _ { t } ( \mathbf { m } ( t ) )$ . This SDE takes the following form:

$$
d \mathbf { m } ( t ) = - 2 \dot { \sigma } ( t ) \sigma ( t ) \nabla _ { \mathbf { m } ( t ) } \log \pi _ { t } ( \mathbf { m } ( t ) ) \ d t + \sqrt { 2 \dot { \sigma } ( t ) \sigma ( t ) } \ d \mathbf { w } ( t )
$$

The ODE based approach is based on the fact that there exists an ODE that has the same marginal distributions $\pi _ { t } ( \mathbf { m } ( t ) )$ as the forward and reverse SDEs. This ODE, known as the probability flow ODE, has the following form:

$$
d \mathbf { m } ( t ) = - \dot { \sigma } ( t ) \sigma ( t ) \nabla _ { \mathbf { m } ( t ) } \log \pi _ { t } ( \mathbf { m } ( t ) ) \ d t
$$

As the SDE and ODE approaches have the same marginal distributions, they are equivalent in probability when the score $\nabla _ { \mathbf { m } ( t ) } \log \pi _ { t } ( \mathbf { m } ( t ) )$ is known. However, the sample trajectories realized by the two formulations difer, and both empirical and theoretical results have provided evidence that the SDE based approach is more robust to score approximation error [30], [31].

> 💡 **机制拆解 (Hao 批注)**: 两条反向路线——反向 SDE（带随机项）vs 概率流 ODE（确定性）。二者在分数已知时边际分布相同，但**轨迹不同**：文献证据表明 SDE 对分数误差更鲁棒。这条洞察后面会回响：BIPSDA 的 corruption 阶段（重新加噪）本质上保留了 SDE 的随机性优势，避免像纯 ODE 那样被分数误差带偏。

Both the reverse SDE and probability flow ODE depend on $\nabla _ { \mathbf { m } ( t ) } \log \pi _ { t } ( \mathbf { m } ( t ) )$ , the score (the gradient of the log-density) of the time-t marginal distribution of $\mathbf { m } ( t )$ These vector fields are a priori unknown but can be learned from the provided samples using a parameterized model $s _ { \theta } ( \mathbf { m } ( t ) , t ) : \mathbb { R } ^ { D } \times \mathbb { R } _ { + } { \mathrm { ~ \hat { ~ } { ~ \theta ~ } ~ } } \times \mathbb { R } ^ { D }$ (the score model) with parameters $\theta ~ \in ~ \mathbb { R } ^ { P }$ . We use $\pmb { \theta } ^ { * }$ to denote the optimized parameters of the model, which are obtained by minimizing the following objective over the set of samples M [15]:

$$
L ( \pmb \theta ) = \frac { 1 } { 2 } \int _ { T _ { \mathrm { m i n } } } ^ { T } w ( t ) \sum _ { i = 1 } ^ { N _ { s } } \mathbb { E } _ { \mathbf { z } } \ \left\| s _ { \pmb \theta } ( \mathbf { m } _ { i } ( t ) ; \sigma ( t ) ) + \frac { \mathbf { z } } { \sigma ( t ) } \right\| _ { 2 } ^ { 2 } d t
$$

In the above objective, which is based on the denoising score matching objective originally introduced by Vincent [32], $w ( t )$ is a specified weighting function, $\mathbf { m } _ { i } ( t ) =$ ${ \bf m } _ { i } + \boldsymbol { \sigma } ( t ) { \bf z }$ is a sample from the noising distribution $\pi _ { t | 0 } ( \mathbf { m } ( t ) \mid \mathbf { m } ( 0 ) = \mathbf { m } _ { i } )$ , z is white noise, and the lower integral limit $T _ { \mathrm { m i n } } \geq 0$ is needed to ensure the objective is well-conditioned.

> 💡 **公式批读 (Hao 批注)**: 去噪分数匹配目标（Vincent 2011）。训练时给干净样本 $\mathbf{m}_i$ 加已知噪声 $\sigma(t)\mathbf{z}$，让分数模型 $s_\theta$ 去回归 $-\mathbf{z}/\sigma(t)$。本文的巧妙之处在于：Gaussian mixture 先验下 $\pi_t$ 是解析的，所以真实分数 $\nabla\log\pi_t$ 也是闭式的——作者既可以「真训一个网络」得到 learned score，也可以直接用解析真值当 analytic score。这两条并行，才能在实验里把「先验建模误差」单独摘出来。

## C. HIJACKING APPROACHES

The ascendance of difusion models as a state-of-theart generative modeling technique has spurred significant research on their use in the context of Bayesian inverse problems. In particular, there has been substantial interest in leveraging difusion models trained on the prior distribution of a given inverse problem to sample from the posterior distribution, i.e., setting $\pi _ { 0 } = \pi _ { \mathrm { p r } } ;$ see [21] for a survey of these approaches. This general strategy has the advantage of enabling the same difusion model to be used to sample from many posterior distributions corresponding to diferent choices of likelihood function without retraining the score model. In the remainder of this section, we discuss one algorithmic framework, which we refer to as the hijacking class of approaches,<sup>1</sup> that follows this general strategy.

The main idea of the hijacking approaches is as follows: Given a difusion model for the prior distribution, replace the $\nabla _ { \mathbf { m } ( t ) } \log \pi _ { t } ( \mathbf { m } ( t ) )$ term in (4) with $\nabla _ { \mathbf { m } ( t ) } \log \pi _ { t \mid \mathbf { y } } ( \mathbf { m } ( t ) \mid \dot { \mathbf { y } } )$ to sample from the posterior, and expand the $\pi _ { t \mid \mathbf { y } } ( \mathbf { m } ( t ) \mid \mathbf { y } )$ term using Bayes’ Theorem. This yields the following reverse SDE:

$$
d \mathbf { m } ( t ) = - 2 \dot { \sigma } ( t ) \sigma ( t ) \nabla _ { \mathbf { m } ( t ) } [ \log \pi _ { t } ( \mathbf { m } ( t ) ) + \log \pi _ { \mathbf { y } \mid t } ( \mathbf { y } \mid \mathbf { m } ( t ) ) ] ~ d t + \sqrt { 2 \dot { \sigma } ( t ) \sigma ( t ) } ~ d \mathbf { w } ( t )
$$

In the above equation, an estimate of $\nabla _ { \mathbf { m } ( t ) } \log \pi _ { t } ( \mathbf { m } ( t ) )$ is given by the score model of the difusion model pretrained on the prior distribution. The $\nabla _ { \mathbf { m } ( t ) } \log \pi _ { \mathbf { y } \mid t } ( \mathbf { y } \mid$ m(t)) term, which is known as the noisy likelihood function [22], then “hijacks” the pre-trained difusion process with information provided by the measurements. However, while the distribution of y given m(0) is given by the likelihood function, the noisy likelihood function is not known. In particular, computation of the noisy likelihood function

$$
\pi _ { \mathbf { y } \mid t } ( \mathbf { y } \mid \mathbf { m } ( t ) ) = \int \pi _ { \mathrm { l i k e } } ( \mathbf { y } \mid \mathbf { m } ( 0 ) ) \pi _ { 0 \mid t } ( \mathbf { m } ( 0 ) \mid \mathbf { m } ( t ) ) \ d \mathbf { m } ( 0 )
$$

is in general intractable. In practice this issue is resolved by using simple approximations of the denoising distribution $\pi _ { 0 | t } ( \mathbf { m } ( 0 ) | \mathbf { m } ( t ) ) [ 1 9 ] , [ 3 3 ] - [ 3 5 ]$ . For example, in [19], the denoising distribution is modeled as a Dirac delta with mass centered on $\mathbb { E } [ { \bf m } ( 0 ) | { \bf m } ( t ) ]$ and Tweedie’s formula [36] is employed to compute the expectation. In the approach of Boys et al [33], a Gaussian approximation is employed, with the mean and covariance of the Gaussian computed using a generalized version of Tweedie’s formula [37]. However, this approach is only applicable when the likelihood function is linear-Gaussian, as otherwise the integral in (8) cannot be straightforwardly computed.

> 💡 **机制拆解 (Hao 批注)**: 这段是理解「为什么要 BIPSDA」的关键。Hijacking（如 DPS [19]、Boys et al [33]）的做法是：把反向 SDE 里的先验分数换成后验分数，用贝叶斯拆成「先验分数 + noisy likelihood 分数」。但 **noisy likelihood $\pi_{\mathbf{y}|t}$ 需要对去噪分布 $\pi_{0|t}$ 积分，一般算不出来**。于是各方法只能用简单近似：DPS 用 Dirac delta（点估计 + Tweedie），Boys 用高斯近似（但只适用于线性高斯似然）。注意这里 $\pi_{0|t}$（去噪分布）是全文核心中间量，BIPSDA 和 hijacking 都要近似它，区别在于近似的**约束条件**不同。

The approaches in [19] and [33] discussed above use simple approximations of the denoising distribution to make the integral in (8) tractable and enable approximation of the noisy likelihood function. In particular, both approaches use unimodal approximations. While this may be a good approximation for many prior distributions of interest when $t \approx 0$ , if the prior is multimodal then $\pi _ { 0 \mid t } ( \mathbf { m } ( 0 ) | \mathbf { \epsilon } \mathbf { m } ( t ) )$ will be multimodal as well for large enough t. There can therefore be significant errors in the approximation of the noisy likelihood function for large t, which in turn induces errors in the distribution of m(t) in (7). These errors are dificult for subsequent hijacking iterations to correct, as discretizations of the hijacking reverse SDE use small step sizes $\Delta t \gt 0$ that ensure $\mathbf { m } ( t - \Delta t )$ will be close to m(t). In practice, it has been observed [22] that this can lead to poor performance on certain inverse problems (in particular, nonlinear inverse problems), with samples obtained that are consistent with the likelihood function but lie in low density regions with respect to the prior. These issues have motivated the development of a class of methods that address this issue by decoupling m(t) and $\mathbf { m } ( t { - } \Delta t )$ which we now introduce.

> 💡 **机制拆解 (Hao 批注)**: 这是 hijacking 的「病理诊断」，也是全文最有价值的洞察之一。病根：所有 hijacking 都用**单峰近似**去噪分布，但只要先验多峰，大 $t$ 时 $\pi_{0|t}$ 就必然多峰 → noisy likelihood 近似崩 → $\mathbf{m}(t)$ 分布出错。更糟的是这个错误**无法被后续迭代纠正**，因为反向 SDE 步长很小、$\mathbf{m}(t-\Delta t)$ 被强制贴近 $\mathbf{m}(t)$（相邻两步高度耦合）。结果就是「样本符合似然但落在先验低密度区」。解药就是**解耦 $\mathbf{m}(t)$ 与 $\mathbf{m}(t-\Delta t)$**——即 DAPS/BIPSDA 的核心思想：每步先跳回 $t=0$ 做完整预测，再重新加噪，切断相邻步的强耦合。对本 topic 的意义：这解释了为什么「诊断出先验/似然融合问题」还不够，采样器的**耦合结构**本身要修——正是增量式修复的证据。

---

## 🔖 Section 总结

### 关键变量速查
| 符号 | 含义 |
|------|------|
| $f, \Sigma_{\mathbf{z}}$ | 前向模型、噪声协方差 |
| $\pi_{\text{like}}, \pi_{\text{pr}}, \pi_{\text{post}}$ | 似然、先验、后验 |
| $\sigma(t)$ | VE 噪声调度（实现取 $\sigma(t)=t$） |
| $\nabla\log\pi_t$ | 加噪先验分数（GM 先验下解析可得） |
| $\pi_{0\mid t}$ | 去噪分布（核心中间量，需近似） |
| $\pi_{\mathbf{y}\mid t}$ | noisy likelihood（hijacking 的痛点，积分不可算） |

### 核心洞察
1. **hijacking 病根**：单峰近似去噪分布，多峰先验 + 大 $t$ 下崩坏，且相邻步强耦合无法纠错 → 样本落在先验低密度区。
2. **解药方向**：解耦 $\mathbf{m}(t)$ 与 $\mathbf{m}(t-\Delta t)$，即「预测回 0 + 重新加噪」的退火结构（下节 BIPSDA）。
3. **SDE > ODE 鲁棒性**：反向 SDE 对分数误差更鲁棒，为 corruption 阶段保留随机性提供理论支撑。

### 可追问点
- 为什么解耦就能纠错？→ 因为每步独立从 $\pi_{0\mid t,\mathbf{y}}$ 采样，不依赖上一步 latent 的精确位置，错误不会累积传播。
