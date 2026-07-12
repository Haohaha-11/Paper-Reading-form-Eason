[← 返回 README](../README.md)

# 3. Reconstruction Algorithms

## 📌 预览

这是全篇的主体：把所有"用预训练扩散先验解逆问题"的方法归入四大家族，全部收敛到同一个中心问题——如何近似/采样 measurement matching term $\nabla_{x_t}\log p_t(y|x_t)$（Eq. 2.17 那个 intractable 项）。四家族：
- **3.1 Explicit Approximations**：给 matching score 一个闭式近似（Score-ALD/Score-SDE/ILVR/DPS/ΠGDM/Moment Matching/BlindDPS/DDRM 族/DDNM 族），统一模板 $-\mathcal{L}_t\mathcal{M}_t/\mathcal{G}_t$（Eq. 3.1）。
- **3.2 Variational Inference**：用简单 $q$ 逼近后验，转成优化（RED-Diff/Blind RED-Diff/Score Prior）。
- **3.3 Asymptotically Exact**：MCMC/SMC 采真后验（PnP-DM/FPS/PMC/SMC 族）。
- **3.4 CSGM-type**：反传优化 ODE 的初始 noise（DMPlug/SHRED/Score-ILO）。
- **3.5–3.6 Latent 系**（表中归 Others）：latent diffusion 的特殊治理（Latent DPS/PSLD/Resample/MPGD/P2L/TReg/STSL）。

**本课题重点盯三处盲方法：BlindDPS (3.1.7)、GibbsDDRM (3.1.8)、Blind RED-Diff (3.2.2)**——它们都要联合估计图像 $x$ 和算子参数 $\phi/\varphi/\gamma$。

---

## 3 Reconstruction Algorithms

We summarize all the methods analyzed in this work in Table 1. The methods have been taxonomized based on the approach they use to solve the inverse problem (explicit score approximations, variational methods, CSGM-type methods and asymptotically exact methods), the type of inverse problems they can solve and the optimization techniques used to solve the problem at hand (gradient descent, sampling, projections, parameter optimization). Additionally, we provide links to the official code repositories associated with the papers included in this survey. Please note that we have not conducted a review or evaluation of these codebases to verify their consistency with the corresponding papers. These links are included for informational purposes only.

Taxonomy based on the type of the reconstruction algorithm. We identified four families of methods. Explicit Approximations for Measurement Matching: These methods approximate the measurement matching score, $\nabla \log p _ { t } ( \pmb { y } | \pmb { x } _ { t } )$ , with a closed-form expression. Variational Inference: These methods approximate the true posterior distribution, $p ( { \pmb x } | { \pmb y } )$ , with a simpler, tractable distribution. Variational formulations are then used to optimize the parameters of this simpler distribution.

> 💡 **家族总览 3.2 Variational (Hao 批注)**: 变分家族换思路：不再逐步近似 score，而是**直接找一个可算的 $q$ 去逼近后验 $p(x_0|y)$，最小化 KL**。扩散先验以 score-matching 项的形式进入 KL 上界。优点是能借成熟优化器；命门是 $q$ 的表达力——**$q$ 太简单（如各向同性高斯），后验必然 mis-calibrated**，这是本课题 SBC/coverage 最容易证伪的一类。 CSGM-type methods: The works in this category use backpropagation to change the initial noise of the deterministic diffusion sampler, essentially optimizing over a latent space for the diffusion model. Asymptotically Exact Methods: These methods aim to sample from the true posterior distribution. This is typically achieved by constructing Markov chains (MCMC) or by propagating particles through a sequence of distributions (SMC) to obtain samples that approximate the posterior. Methods that do not fall into any of these categories are classified as Others.

Taxonomy based on the type of optimization techniques used. The objective of all methods is to explain the measurements. The measurement consistency can be enforced with different opti mization techniques, e.g. through gradients (Grad), projections (Proj), sampling (Samp), or other optimization techniques (Opt). Methods that belong to the Grad-type take a single gradient step (ei ther it be deterministic or stochastic) to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ to enforce measurement consistency. Proj-type projects x<sub>t</sub> or $\mathbb { E } [ X _ { 0 } | X _ { t } ~ = ~ { \pmb x } _ { t } ]$ to the measurement subspace. Samp-type samples the next particles by defining a proposal distribution, and propagates multiple chains of particles to solve the problem. Opt-type either defines and solves an optimization problem for every timestep, or defines a global optimization problem that encompasses all timesteps. When the method belongs to more than one type, we seperate them with /. Note that the categorization of different “types” is subjective, and more often than not, the category that the method belongs to may be interpreted in multiple ways. For instance, a projection step is also a gradient descent step with a specific step size.

Taxonomy based on the type of the inverse problem. Based on the linearity of the corruption operator , the inverse problems can be classified as linear or nonlinear. The inverse problems can be further categorized based on whether there is noise in the measurements. Additionally, they are classified as non-blind or blind depending on whether full information about is available. In blind problems, the degradation operator (e.g., convolution kernel, inpainting kernel) is known, while its coefficients are unknown but parametrized. For example, we might know that we have measurements with additive Gaussian noise, but the variance of the noise might be unknown. Finally, in certain inverse problems, there is additional text-conditioning. Such inverse problems are typically solved with text-to-image latent diffusion models [134].

## 3.1 Explicit Approximations for the Measurements Matching Term

The first family of reconstruction algorithms we identify is the one were explicit approximations for the measurements matching term, $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { X } _ { t } = \pmb { x } _ { t } )$ , are made. It is important to underline that these approximations are not always clearly stated in the works that propose them, which makes it hard to understand the differences and commonalities between different methods. In what follows, we attempt to elucidate the different approximations that are being made and present different works under a common framework. To provide some insights, we often provide the explicit approximation formulas for the measurements matching term in the setting of linear inverse problems. In general, it follows the template form:


![Eq. 3.1](../images/de67f654554d2bce619ec36096d896896596a702746e4c43d9eb86dfb2d5a9bf.jpg)


Here,

$\mathcal { M } _ { t }$ represents the error vector measuring the discrepancy between the observation y and the estimated restored vector; for example, in Score $\mathrm { A L D } \left[ 1 \right] , \mathcal { M } _ { t } = \pmb { y } - A \pmb { x } _ { t }$

$\mathcal { L } _ { t }$ denotes a matrix that projects the error vector $\mathcal { M } _ { t }$ from $\mathbb { R } ^ { m }$ back into an appropriate space in $\mathbb { R } ^ { n }$ ; for instance, in Score $\mathrm { A L D } , \mathcal { L } _ { t } = A ^ { \top }$

• <sub>t</sub> is the re-scaling scalar for the guidance vector $\mathcal { L } _ { t } \mathcal { M } _ { t }$ ; for example, in Score ALD, $\begin{array} { r } { \mathcal { G } _ { t } = \sigma _ { y } ^ { 2 } + \gamma _ { t } ^ { 2 } } \end{array}$ with a hyperparameter $\gamma _ { t }$

In Figure 1, we summarize the approximation-based methods in this section using the template above. We use to omit the guidance strength terms $\mathcal { G } _ { t }$

> 💡 **公式批读 Eq. 3.1（Explicit 家族统一模板）(Hao 批注)**: 这是 3.1 全节的骨架——所有 explicit 方法的 measurement score 都写成 $-\mathcal{L}_t\mathcal{M}_t/\mathcal{G}_t$。三个部件：$\mathcal{M}_t$=测量误差（如 $y-Ax_t$）、$\mathcal{L}_t$=把误差从 $\mathbb{R}^m$ 抬回 $\mathbb{R}^n$ 的 lifting 矩阵（$A^\top$/$A^\dagger$/带协方差的逆）、$\mathcal{G}_t$=guidance 强度标量。**读 3.1 各方法只需回答两问：误差用 clean 还是 noised $y$？lifting 矩阵多复杂？** 越复杂的 lifting 越逼近真 posterior score 的二阶几何，这就是各方法拉开差距的地方。

### 3.1.0 Sampling from a Denoiser Kadkhodaie and Simoncelli [30]

Kadkhodaie and Simoncelli [30] introduce a method for solving linear inverse problems by using the implicit prior knowledge captured by a pre-trained denoiser on multiple noise levels. The method is anchored on Tweedie’s formula that connects the least-squares solution for Gaussian denoising to the gradient of the log-density of noisy images given in Equation 2.10


![Eq. 3.2](../images/c00819982a88ea9913e5c03977e2e07c1943e19cba32f634699019d04e732f7c.jpg)


where ${ \pmb y } = { \pmb x } + { \pmb n } , { \pmb n } \sim \mathcal { N } ( { \pmb 0 } , \sigma ^ { 2 } I _ { n } )$

By interpreting the denoiser’s output as an approximation of this gradient, the authors develop a stochastic gradient ascent algorithm to generate high-probability samples from the implicit prior


![Eq. 3.3](../images/e23aae734b0ebdbedcd39db0c014e02d69a115fe49ee99530d7e02b63f59dcf0.jpg)


where $\pmb { r } ( \pmb { y } ) = \hat { \pmb { x } } ( \pmb { y } ) - \pmb { y }$ is the denoiser residual, $h _ { t }$ is a step size (parameter), and $\epsilon _ { t }$ controls the amount of newly introduced Gaussian noise ${ \boldsymbol { z } } _ { t }$

To solve linear inverse problems such as deblurring, super-resolution, and compressive sensing, the generative method is extended to handle constrained sampling. Given a set of linear measurements $\bar { \mathbf { x } } _ { c } = M ^ { \top }$ x of an image x, where M is a low-rank measurement matrix, the goal is to reconstruct the original image by utilizing the following gradient:


![Eq. 3.4](../images/3e8696cd25ab7d499cea0b9bc3705fa1608eeeb81e6bf91dcc4b9a0ff7d852b4.jpg)


This approach is particularly interesting because its mathematical foundation relies solely on Tweedie’s formula, providing a simple yet powerful framework for tackling inverse problems using denoisers.

### 3.1.1 Score ALD [1]

One of the first proposed methods for solving linear inverse problems with diffusion models is the Score-Based Annealed Langevin Dynamics (Score ALD) [1] method. The approximation of this work is that:


![Eq. 3.5](../images/d14046b38f51057570d7242e7999f0afbf19dfd84b9914965758d1a52bd10ad3.jpg)


where $\gamma _ { t }$ is a parameter t

It is pretty straightforward to understand what this term is doing. The diffusion process is guided towards the opposite direction of the “lifting” (application of the ${ \bf \bar { A } } ^ { \top }$ operator) of the measurements error, i.e. $\left( { \pmb y } - { \pmb A } { \pmb x } _ { t } \right)$ ), where the denominator controls the guidance strength.

### 3.1.2 Score-SDE [2]

Score-SDE [2] is another one of the first works that discussed solving inverse problems with pretrained diffusion models. For linear inverse problems, the difference between Score-ALD and Score-SDE is that the latter noises the measurements before computing the measurements error. Specifically, for $t : \sigma _ { t } \gt \sigma _ { y }$ , the approximation becomes:


![Eq. 3.6](../images/3c71c48a49ab908404a0de0fa5a815e9a8c69cb203f2a26a556ed268b6839173.jpg)


where ǫ is sampled from $\mathcal { N } ( \mathbf { 0 } , I _ { m } )$ . Here, A is an orthogonal matrix, and taking a gradient step with Equation $3 . 6$ yields a noisy projection to $\mathbf { \mathbf { } } y _ { t } ~ = ~ A \mathbf { \mathbf { } } x _ { t }$ where ${ \pmb y } _ { t } = { \pmb y } + \sigma _ { t } { \pmb \epsilon }$ . Hence, we categorize Score-SDE as “projection”.

Disregarding the guidance strength of Equation 3.5, Equation 3.5 and Equation 3.6 look very similar. Indeed, the only difference is that the latter has stochasticity that arises from the noising of the measurements.

Special case: Inpainting (Repaint [123]) Observe that for the simplest case of inpainting, Equation 3.6 would be replacing the pixel values in the current estimate $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ with the known pixel values from the noised ${ \mathbf { } } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } } \mathbf { \Delta } _ { \mathbf { } \mathbf { } } \mathbf { \Delta } _ { \mathbf { } \mathbf { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } } \mathbf { \Delta } _ { \mathbf { } \mathcal { } \mathcal { } } \mathbf { \Delta } _ { \mathcal { } \mathcal { } } \mathbf { \Delta } _ { \mathcal { } \mathcal { } \mathcal { } } \mathbf { \Delta } _ { \mathcal { } \mathcal { } \mathcal { } } \mathbf { \Delta } _ { \mathcal { } \mathcal { } \mathcal { } \Delta } \mathbf { \Delta } _ { \mathcal \mathcal { } }$ . Coincidentally, this is exactly the Repaint Lugmayr et al. [123] algorithm that was proposed for solving the inpainting inverse problem with pre-trained diffusion models. Re-Paint++ Rout et al. [124] improves upon this approximation to run the forward-reverse diffusion processes multiple times, so that the errors arising (e.g. boundaries) can be mitigated. This can be thought of as analogous to running MCMC corrector steps as in predictor-corrector sampling [2].

### 3.1.3 ILVR [3]

ILVR is a similar approach that was initially proposed for the task of super-resolution. The approximation made here is the following:


![Eq. 3.7](../images/7c8f339d8d04191258e6b0c84cbd34b22ee39ee640b01a71df6303e1a0c83532.jpg)


where $A ^ { \dagger }$ is the Moore-Penrose pseudo-inverse of A, and similar to Score-SDE, $\mathbf { \boldsymbol { y } } _ { t } = \mathbf { \boldsymbol { y } } + \sigma _ { t } \mathbf { \boldsymbol { \epsilon } }$

ILVR can be regarded as a pre-conditioned version of score-SDE. In ILVR, the projection to the space of images happens using the Moore-Penrose pseudo-inverse of $\mathbf { A }$ , instead of the simple $A ^ { \top }$

### 3.1.4 DPS

> 💡 **机制拆解 3.1.4 DPS（非线性逆问题基石）(Hao 批注)**: DPS 是全篇被复用最多的近似，务必吃透。核心一步（Eq. 3.8）：把 intractable 的 $p(x_0|x_t)$ 用一个**点质量** $\delta(x_0-\mathbb{E}[X_0|x_t])$ 代替——即"假装干净图就是 Tweedie 去噪均值 $\hat{x}_0$，没有不确定性"。于是 matching term 变成对 $\|y-\mathcal{A}(\hat{x}_0)\|^2$ 求 $x_t$ 的梯度（Eq. 3.11–3.13），要穿过去噪器的 Jacobian。**这个 $\delta$ 近似正是"prior score 与 posterior score 差距"里最粗糙的一档：它完全丢掉了 $p(x_0|x_t)$ 的协方差**，也是后面 ΠGDM（换高斯）、Moment Matching（换真协方差）逐级改进的起点。对本课题：DPS 忽略协方差 → 后验必然过度自信（under-dispersed），SBC/coverage 会直接暴露；这解释了为何盲+校准场景不能止步于 DPS。

All of the previous algorithms were proposed for linear inverse problems. Diffusion Posterior Sampling (DPS) is one of the most well known reconstruction algorithms for solving non-linear inverse problems. The underlying approximation behind DPS is that:


![Eq. 3.8](../images/a9c3b5365212b8df8dc9498716bcffcc12022a58681c751366c02e5118d410eb.jpg)


It is easy to see that:


![Eq. 3.9](../images/de533d95f5c7fcc1b2033b14d549ace52a3b9b2987c0f581fc1e87d7b9a4f82f.jpg)


Hence, the DPS approximation can be stated as:


![Eq. 3.10](../images/2c279fbd1aef3e893ad715aa52a54bf597de37e0b8ffd28efffdd0807edf9c9b.jpg)



![Eq. 3.11](../images/98f72318e7a64bf896a85a4e26e87b8c2d8d306c566d90051822dded0e168a21.jpg)



![Eq. 3.12](../images/f7429ffadd391791006fc2f333c92afde87590cd8f7defb258c393d6f0c42cf6.jpg)


For linear inverse problems, this simplifies to:


![Eq. 3.13](../images/5e78c201ff24fe93f854a2d6e4c4864b33068fcb71ca7267d1d57d85ef62924a.jpg)



We can further use Tweedie’s formula to further write it as:


![Eq. 3.14](../images/1c18345249eb6b339fda85aa648fd84b3b706d1d4fd12afe68ad82d0825cce7c.jpg)


In practice, DPS does not use the theoretical guidance strength but instead proposes to use a reweighting with a step size inversely proportional to the norm of the measurement error.

MCG Chung et al. [31] provides a geometric interpretation of DPS by showing that the approximation used in DPS can guarantee the noisy samples stay on the manifold. DSG Yang et al. [147] showed that one can choose a theoretically “correct” step size under the geometric view of MCG, and combined with projected gradient descent, one can achieve superior sample quality. MPGD He et al. [33] showed that by constraining the gradient update step to stay on the low dimensional subspace by autoencoding, one can acquire better results.

### 3.1.5 ΠGDM Song et al. [5]

> 💡 **机制拆解 3.1.5 ΠGDM (Hao 批注)**: ΠGDM 把 DPS 的 $\delta$ 升级为**各向同性高斯** $p(x_0|x_t)\approx\mathcal{N}(\mathbb{E}[X_0|x_t], r_t^2 I)$（Eq. 3.16）。好处：线性情形 matching term 出现 $(r_t^2 AA^\top+\sigma_y^2 I)^{-1}$（Eq. 3.18），即 lifting 矩阵带上了"数据不确定性 + 观测噪声"的联合协方差。**这是往真 posterior score 靠近的第一步**，但 $r_t^2 I$ 仍是各向同性的粗糙假设。

Recall the intractable integral in Equation 1.3. According to this relation, the DPS approximation is achieved by setting


![Eq. 3.15](../images/f6eaa2665db5f8d8c7f271596464105d726ec1dc2f1e6d0233ad52f273810f9b.jpg)


In ΠGDM, the authors propose to use a Gaussian distribution for approximation


![Eq. 3.16](../images/da1203aee5fba3b903a3bf5eb0eb7a2d6081ef4ecf14727ba62d9e53b9b31319.jpg)


where $r _ { t }$ is a hyperparameter. For linear inverse problems, this leads to


![Eq. 3.17](../images/59f06d478498261d6436c9f400d728152b0063b09b7d816f43734a4cec71a1c9.jpg)


Subsequently, we have


![Eq. 3.18](../images/d9239683e1392aad6b4a3485128b599ce7ba82ec1800a651ff82e0566a8eee29.jpg)


### 3.1.6 Moment Matching [6]

In ΠGDM, the distribution $p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } )$ was assumed to be isotropic Gaussian. However, one can calculate explicitly the variance matrix, $V [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ]$ . As shown in Lemma A.4, it holds that:


![Eq. 3.19](../images/42c450fd274cbe7bf741c564cef16860eaebaf63ec50d443a43738bed0479b35.jpg)



![Eq. 3.20](../images/37d98fe744c1d2807d900179a6587bec5b52f443b577137a0bcee1b4b6c33480.jpg)


The Moment Matching [6] method approximates the distribution $p ( \pmb { x } _ { 0 } | \pmb { x } _ { t } )$ with an anisotropic Gaussian:


![Eq. 3.21](../images/107af965e7337a6d2d5b9b5c3655ac66ba4573db861ed45d8a99449bc60078d9.jpg)


For linear inverse problems, this leads to the following approximation for the measurements’ score:


![Eq. 3.22](../images/13277510936f8dab9e2f0106d0e0bb6e06b8a4cf7406a3da10ffc3546bc41211.jpg)


In high-dimensions, even materializing the matrix $\nabla _ { \pmb { x } _ { t } } \mathbb { E } [ \pmb { x } _ { 0 } | \pmb { x } _ { t } ]$ is computationally intensive. Instead, the authors of [6] use automatic differentiation to compute the Jacobian-vector products.

> 💡 **机制拆解 3.1.6 Moment Matching（协方差最忠实档）(Hao 批注)**: Moment Matching 用**各向异性高斯**——协方差直接取 Tweedie 二阶公式 $V[x_0|x_t]=\sigma_t^4 H(\log p_t)+\sigma_t^2 I=\sigma_t^2\nabla_{x_t}\mathbb{E}[x_0|x_t]$（Eq. 3.19–3.20，证明见 Lemma A.4）。这是 Explicit 家族里**对 $p(x_0|x_t)$ 协方差刻画最忠实**的方法，代价是要算 Jacobian（用 auto-diff 的 JVP 规避显式矩阵）。**对本课题极关键：这个真协方差正是"gauge-aware 校准"要盯的对象——只有把它算对，后验的 spread 才可能 well-calibrated。DPS→ΠGDM→Moment Matching 就是一条"协方差保真度递增"的谱，也是 prior/posterior score 差距逐步缩小的谱。**

### 3.1.7 BlindDPS Chung et al. [7]

> 💡 **机制拆解 3.1.7 BlindDPS（本课题核心前作 ①）(Hao 批注)**: BlindDPS 是 DPS 的盲扩展，直接对标本课题的联合后验。数据流：
> - **两条并行反向 SDE**（Eq. 3.23/3.24）：一条采图像 $x_t$，一条采算子参数 $\phi_t$（如模糊核）。二者各有自己的扩散先验 $p_t(x_t)$、$p_t(\phi_t)$。
> - **通过 likelihood 耦合**：假设 $X_t\perp\Phi_t$，Bayes 分解（Eq. 3.25/3.26）让两条 score 各自 = 自身 prior score + 共享的 $\nabla\log p_t(y|X_t,\Phi_t)$。
> - **DPS 近似套两次**（Eq. 3.27）：对 $x$ 和 $\phi$ 都用各自的 Tweedie 均值代入 likelihood，得 Eq. 3.28（对 $x$）、Eq. 3.29（对 $\phi$）。
>
> **与本课题对照**：这正是"联合估计 $(x,\phi)$"的雏形，$\phi$ 也走一条完整扩散链（需要给 $\phi$ 训练一个先验，成本高）。局限：(1) 沿用 DPS 的 $\delta$/点近似 → 忽略 $(x,\phi)$ 的联合协方差，后验校准无保证；(2) 假设 $X_t\perp\Phi_t$ 割裂了 $x$-$\phi$ 的耦合。**本课题的 gauge-aware 联合后验 + SBC/coverage/CRPS 恰好是补这两个洞：既建模 $x$-$\phi$ 耦合，又用校准诊断验证 spread。**

Methods that were considered so far were designed for non-blind inverse problems, where $A$ is fully known. BlindDPS targets the case where we have a parametrized unknown forward model $A _ { \phi } \left( \mathrm { e . g } \right.$ blurring with an unknown kernel $\phi )$ . In BlindDPS, on top of the posterior mean approximation of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ one approximates the parameter of the forward model, again, with the posterior mean. Specifically, we design two parallel generative SDEs


![Eq. 3.23](../images/488ccca423bd2ee1bd785b5398c974cc8fca7eecdf23911846df01e90159d143.jpg)



![Eq. 3.24](../images/f3ab2e6e04efb34d13a2fa3bc44ce8a5d7fa990ea926ecaa33c62ffb098dcb9d.jpg)


where the two SDEs are coupled through log $p _ { t } ( \pmb { x } _ { t } , \phi _ { t } | \pmb { y } )$ , where under the independence between $X _ { t }$ and $\Phi _ { t }$ , the Bayes rule reads


![Eq. 3.25](../images/83a8e1f9176a871b0c4e65be5356cab2aafa055a2b4a4c10f35212ad06218019.jpg)



![Eq. 3.26](../images/d0d97a4bbf6725d4eba4de26e4da18e0b83db9ea743bdf2354dc96b1894a3108.jpg)


where we see that $X _ { t }$ and $\Phi _ { t }$ are coupled through the likelihood $p ( \pmb { y } | \pmb { X } _ { t } , \pmb { \Phi } _ { t } )$ . In BlindDPS, the approximation used in DPS is applied to both the image and the operator, leading to


![Eq. 3.27](../images/d2b2644b68b7f5540c0b547d244ba7398d58dbc11beb89e8930e215591133a4f.jpg)


The gradient of the coupled likelihood with respect to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ leads to



![Eq. 3.28](../images/32eaae8cd6a789dcf90db54a0d5bb7181cbac930f71ab64ecb7b6c1f728a2288.jpg)


Similarly, for $\phi _ { t }$ , we have



![Eq. 3.29](../images/ee3703a1dc4580c1c887bd1d149801990ef321f9fb516deffca2a4aad58fabba.jpg)


### 3.1.8 DDRM Family

> 💡 **机制拆解 3.1.8 DDRM 族（SVD 谱域视角）(Hao 批注)**: DDRM 族的统一 trick：对 $A=U\Sigma V^\top$ 做 SVD，把任意线性逆问题变成**谱域里的 noisy inpainting**（Eq. 3.30）。SNIPS 在谱域采样后 $\hat{x}=V\bar{x}$；DDRM 用 Tweedie 均值 $\bar{x}_{0|t}$ 替换（Eq. 3.33），并按奇异值 $s_i$ 与噪声比 $\sigma_y/s_i$ 分档处理每个谱分量（Eq. 3.34/3.35，$\eta$ 控随机性）。**这套按奇异值分档的做法天然带"每个分量的不确定性"，比 DPS 的标量 guidance 更精细。**

The methods under the DDRM family poses all linear inverse problems to a noisy inpainting problem, by decomposing the measurement matrix with singular value decomposition (SVD), i.e. $\dot { \boldsymbol { A } } = \boldsymbol { U } \dot { \boldsymbol { \Sigma } } \dot { \boldsymbol { V } } ^ { \intercal }$ , where $U ^ { ^ { - } } \in \mathbb { R } ^ { m \times m } , V \in \mathbb { R } ^ { n \times n }$ are orthogonal matrices, and $\Sigma \in \mathbb { R } ^ { m \times n }$ is a rectangular diagonal matrix with singular values $\{ s _ { j } \} _ { j = 1 } ^ { m }$ as the elements. One can then rewrite $\pmb { y } = A \pmb { x } + \sigma _ { \pmb { y } } z , z \sim \mathcal { N } ( \mathbf { 0 } , I _ { m } )$ as


![Eq. 3.30](../images/5b9000372b0007770e229fe23d9cf055e7b82aa9f20992cd48d9b71542cdcd42.jpg)


Subsequently, Equation 3.30 becomes an inpainting problem in the spectral space.

SNIPS [8]. SNIPS proceeds by first solving the inverse problem posed as Equation 3.30 in the spectral space to achieve a sample $\bar { \pmb x } \sim p ( \bar { \pmb x } | \bar { \pmb y } )$ , then retrieving the posterior sample with ${ \hat { \mathbf { x } } } = V { \bar { \mathbf { x } } }$ The key approximation can be concisely represented as


![Eq. 3.31](../images/5a51795cffedcd21ccfd2f40ab03bbf1a9d98036e1af3bd220f4d84fb5fb9d7a.jpg)



For the simplest case of denoising where $m = n$ and $\Sigma = A = I$ , the method becomes [148]


![Eq. 3.32](../images/75c51f54539209094c697fa2be7637d9bf4bcf7865754b538bbf1ecc6bfd3f93.jpg)


which produces a vector direction that is weighted by the absolute difference between the diffusion noise level $\sigma _ { t } ^ { 2 }$ , and the measurement noise level $\sigma _ { y } ^ { 2 }$ . For the fully general case in Equation 3.31, elements in different indices are weighted according to the singular values contained in $\Sigma .$ In practice, SNIPS uses pre-conditioning with the approximate negative inverse Hessian of log $p ( \bar { \pmb x } _ { t } | \bar { \pmb y } )$ when running annealed Langevin dynamics.

DDRM [9]. DDRM extends SNIPS by leveraging the posterior mean $\bar { \pmb { x } } _ { 0 \mid t } : = V \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ]$ in the place of $\bar { \mathbf { x } } _ { t }$ used in SNIPS. i.e.,


![Eq. 3.33](../images/dd87bb1ebd2453ed6237dc99c354d7611b85b5313703eb5893bd5b175da059ee.jpg)



Expressing Equation 3.33 element-wise, we get


![Eq. 3.34](../images/b279ad6d856e47a6b251c43cb3e8d3462e38a2e893e1c616daf63224d7a1ad87.jpg)


where $\mathbf { \boldsymbol { x } } ^ { ( i ) }$ denotes the i-th element of the vector, and $s _ { i }$ its corresponding singular value. Here, DDRM introduces another hyper-parameter η to control the stochasticity of the sampling process


![Eq. 3.35](../images/d49163aa4eb78ecdc9fb0a6e4edbcf83122292449bc0b5767ae300fe8171a8e3.jpg)


with $\eta \in ( 0 , 1 ]$ such that $\eta = 1 . 0$ recovers Equation 3.34.

GibbsDDRM. GibbsDDRM Murata et al. [10] extends DDRM to the following blind linear problem $\begin{array} { r } { \pmb { y } = A _ { \varphi } \pmb { x } + \sigma _ { \pmb { y } } z } \end{array}$ , where $A _ { \varphi }$ is a linear operator parameterized by $\varphi .$ . Here, $A _ { \varphi } = U _ { \varphi } \Sigma _ { \varphi } ^ { \circ } V _ { \varphi } ^ { \intercal }$ has a ϕ dependence SVD decomposition with singular values $\{ s _ { j , \varphi } \} _ { j = 1 } ^ { m }$ as the elements of the diagonal matrix $\Sigma _ { \varphi }$ . In the spectral space, $\bar { \pmb { y } } _ { \pmb { \varphi } } : = U _ { \pmb { \varphi } } ^ { \top } \pmb { y } _ { \pmb { \varphi } } , \bar { \pmb { x } } _ { \pmb { \varphi } } : = V _ { \pmb { \varphi } } ^ { \top } \pmb { x } _ { \pmb { \varphi } } , \bar { z } _ { \pmb { \varphi } } : = U _ { \pmb { \varphi } } ^ { \top } z _ { \pmb { \varphi } }$ . Subsequently, the posterior mean in DDRM is replaced with $\bar { \pmb { x } } _ { 0 | t , \varphi } : = V _ { \pmb { \varphi } } \mathbb { E } [ \pmb { X } _ { 0 } | \pmb { X } _ { t } = \pmb { x } _ { t } ]$ , also depending on $\varphi .$ Thus, it leads to the sampling process


![Eq. 3.36](../images/163794a17973eed22bc04b12c9269d2776d7ca825fe372e0b51ebcc07c75cc14.jpg)


At time step $t , \varphi$ is sampled by using the conditional distribution $p ( \varphi | \mathbf { x } _ { t : T } , \mathbf { y } )$ and updated for several iterations in a Langevin manner:

> 💡 **机制拆解 GibbsDDRM（本课题核心前作 ②）(Hao 批注)**: GibbsDDRM 把 DDRM 扩到盲线性问题 $y=A_\varphi x+\sigma_y z$，$\varphi$ 是算子参数。做法是 **partially collapsed Gibbs 采样**：交替 (1) 给定 $\varphi$ 用 $\varphi$-依赖的 SVD $A_\varphi=U_\varphi\Sigma_\varphi V_\varphi^\top$ 更新 $x$（Eq. 3.36），(2) 给定 $x$ 用 Langevin 更新 $\varphi$（下式 + Eq. 3.37 的梯度 $-\frac{1}{2\sigma_y^2}\nabla_\varphi\|y-A_\varphi\bar{x}_{0|t,\varphi}\|^2$）。
>
> **与本课题对照**：这是"联合后验采样"最接近本课题的前作——用 Gibbs 而非点优化来交替采 $(x,\varphi)$，理论上比 BlindDPS/Blind RED-Diff 的点估计更接近真联合后验。可追问：GibbsDDRM 的 $\varphi$-Langevin 是否 well-mixed？其联合后验是否校准？综述未评（无实验），正是本课题用 SBC/coverage 可检验的缺口。


![Eq. 3.1.8-langevin](../images/240d80b182941113566ef6c5598fb6893329794159bfc372193ae135ecff3fa9.jpg)


where $\xi$ is a stepsize and $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , I _ { n } )$ . Here, $\nabla _ { \varphi } \log { p ( \varphi | x _ { t : T } , y ) } \approx \nabla _ { \varphi } \log { p ( \varphi | \bar { x } _ { 0 | t , \varphi } , y ) }$ , and the gradient can be computed as:


![Eq. 3.37](../images/0d50b2acc9f9abdacbbe508c5401879dbcf3af73289f07881e6318efd72be2ac.jpg)


### 3.1.9 DDNM Wang et al. [11] family

> 💡 **机制拆解 3.1.9 DDNM 族（range-null 分解）(Hao 批注)**: DDNM 换个视角——用**条件 Tweedie**（Eq. 3.38–3.40）把 conditional posterior mean $\mathbb{E}[X_0|x_t,y]$ 与无条件 $\mathbb{E}[X_0|x_t]$ 关联，然后对无条件均值做 range-null space 投影 $(I-A^\dagger A)\mathbb{E}[X_0|x_t]+A^\dagger y$（Eq. 3.41）：**零空间保留先验、值空间强制满足测量**。有噪时改软更新（Eq. 3.44，用 $\Sigma_t$）。DDS/DiffPIR 则把这步写成 proximal 优化（Eq. 3.46），差别在解法（DDS 用 CG、DiffPIR 用闭式 + SNR 调度 $\lambda_t$）。**这类 projection/proximal 属于 Table 1 的 Proj/Opt 型，数据一致性强但不显式建模不确定性。**

A different way to find meaningful approximations for the conditional score is to look at the conditional version of Tweedie’s formula, see Equation 2.15. Using Bayes rule and rearranging Ravula et al. [149], we have


![Eq. 3.38](../images/8e43829433aeb8f28a712110e9928a394dc08f652445baa42c485d0f0df166ac.jpg)



![Eq. 3.39](../images/1aaa31ad468dde8d3b623f755f0468523c28387a4cc496ae487de2ff1e4052d0.jpg)



![Eq. 3.40](../images/3d925dfc6e32af698086f4b4daf0a3f6828a7dc46b1d785b1781ff5185b43e1a.jpg)


The methods that belong to the DDNM family make approximations to $\mathbb { E } [ X _ { 0 } | X _ { t } ~ = ~ { \pmb { x } } _ { t } , { \pmb { y } } ]$ by making certain data consistency updates to $\mathbb { E } [ X _ { 0 } | X _ { t } = \bar { \mathbf { x } _ { t } } ]$

DDNM Wang et al. [11]. The simplest form of update when considering no noise can be obtained through range-null space decomposition, assuming that one can compute the pseudo-inverse. In DDNM, this condition is trivially met by considering operations that are SVD-decomposable. DDNM proposes to use the following projection step to the posterior mean to obtain an approximation of the conditional posterior mean


![Eq. 3.41](../images/2057f34cfc83b6b95185599697effd87f0b5e56b39bf09a97bf7f29c67c26210.jpg)


where $A ^ { \dagger }$ is the Moore-Penrose pseudo-inverse of A. One can also express Equation 3.41 as an approximation of the likelihood, consistent to other methods in the chapter. Specifically, notice that by using the relation in Equation 3.40,


![Eq. 3.42](../images/0f3e295c242320c7e83ee5f64cfd50969c08a0112cfaadc72d03f18e57b258e5.jpg)


Plugging in Equation 3.41 to Equation 3.42,


![Eq. 3.43](../images/8a8da4cf253165bfe6b197ae299e10ff7d6fa882c074ad45f7e0180e6a3faf3f.jpg)


When there is noise in the measurement, one can make soft updates


![Eq. 3.44](../images/370109f535eaad46661ce5811b848383793d6adad87ab02515a1a2e05b4e039b.jpg)


Also, similar to Equation 3.43,


![Eq. 3.45](../images/b9f2d42cb183b4ebc882470a1f572b853ce90f8ea734ebeed76b055a9845cb9e.jpg)


Here, one can choose a simple $\Sigma _ { t } = \lambda _ { t } I$ with $\lambda _ { t }$ set as a hyper-parameter, or use different scaling for each spectral component. Observe that due to the relationship between the (conditional) score function and the posterior mean established in Equation 3.40, we can also easily rewrite the approximation in terms of the score of the posterior.

DDS Chung et al. [12], DiffPIR Zhu et al. [13]. Both DDS and DiffPIR propose a proximal update to approximate the conditional posterior mean, albeit from different motivations. The resulting approximation reads


![Eq. 3.46](../images/3882e2ee22cb2668fff58c98ec79390d659be1aade69e33104e54f861f7ca169.jpg)


The difference between the two algorithms comes from how one solves the optimization problem in Equation 3.46, and how one chooses the hyperparameter $\lambda _ { t }$ . In DDS, the optimization is solved with a few-step conjugate gradient (CG) update steps, by showing that DPS gradient update steps can be effectively replaced with the CG steps under assumptions on the data manifold Chung et al. [12]. λ<sub>t</sub> is taken to be a constant value across all t. DiffPIR uses a closed-form solution for Equation 3.46, and proposes a schedule for $\lambda _ { t }$ that is proportional to the signal-to-noise (SNR) ratio of the diffusion at time t. Specifically, one chooses $\lambda _ { t } = \sigma _ { t } \zeta$ , where ζ is a constant.

## 3.2 Variational Inference

These methods approximate the true posterior distribution, $p ( { \pmb x } | { \pmb y } )$ , with a simpler, tractable distribution. Variational formulations are then used to optimize the parameters of this simpler distribution.

### 3.2.1 RED-Diff Mardani et al. [16]

Mardani et al. [16] introduce RED-diff, a new approach for solving inverse problems by leveraging stochastic optimization and diffusion models. The core idea is to use variational method by introducing a simpler distribution, $q : = \mathcal { N } ( \pmb { \mu } , \sigma ^ { 2 } I _ { n } )$ , to approximate the true posterior $p ( \pmb { x } _ { 0 } | \pmb { y } )$ by minimizing the KL divergence $\mathcal { D } _ { \mathrm { K I } }$ between them:


![Eq. 3.47](../images/3dde0a01b6a4d30ac7390b77ad6c0fadf141a60d3f89b3e1c8533af4a0416bb6.jpg)


Here, ${ \mathcal { D } } _ { \mathrm { K L } } ( q ( { \pmb x } _ { 0 } | { \pmb y } ) | | p ( { \pmb x } _ { 0 } | { \pmb y } ) )$ can be written as follows:


![Eq. 3.48](../images/a6cd16d29ef093d491aee604518302b892ae3ef80acb967fffff3827b2fe25fa.jpg)


via classic variational inference argument. The first term in VB can be simplified into reconstruction loss, and the second term can be decomposed as score-matching objective which involves matching the score function of the variational distribution with the score function of the true posterior denoisers at different timesteps:


![Eq. 3.49](../images/db32dee899b5f9813ed00b8ce443eabc19f03394354d52013d16ea361043bc7c.jpg)


where $\pmb { \mu }$ is the mean of the variational distribution, and $\sigma _ { v } ^ { 2 }$ is the noise variance in the observation, $\epsilon _ { \theta } ( x _ { t } ; t )$ is the score function of the diffusion model at timestep (t) and $\lambda _ { t }$ is a time-weighting factor.

Sampling as optimization. The goal is then to find an image $\pmb { \mu }$ that reconstructs the observation y given by $f ,$ while having a high likelihood under the denoising diffusion prior (regularizer). This score-matching objective is optimized using stochastic gradient descent, effectively turning the sampling problem into an optimization problem. The weighting factor $( \lambda _ { t } )$ is chosen based on the signal-to-noise ratio (SNR) at each timestep to balance the contribution of different denoisers in the diffusion process.

### 3.2.2 Blind RED-Diff Alkan et al. [17]

> 💡 **机制拆解 3.2.2 Blind RED-Diff（本课题核心前作 ③）(Hao 批注)**: Blind RED-Diff 把 RED-Diff 的变分框架扩到盲问题，**联合估计图像 $x_0$ 与 forward 参数 $\gamma$**。KL 分解成三项（Eq. 3.2.2-kldecomp）：$x_0$ 的先验 KL（score-matching 项）、$\gamma$ 的先验 KL（对 $\gamma$ 的正则）、以及数据一致性 $-\mathbb{E}[\log p(y|x_0,\gamma)]$。用**交替随机优化**轮流更新 $x_0,\gamma$。
>
> **与本课题对照**：这是三个盲前作里最"变分"的一个，直接对应本课题的联合估计。两条硬伤正是本课题切入点：(1) **假设 $x_0\perp\gamma\,|\,y$**（条件独立），割裂了 gauge 耦合；(2) 变分 $q$ 形式固定 + 交替优化 → 是点估计式的近似后验，**无校准保证且可能陷局部最优**。本课题用联合后验采样（而非交替优化）+ 显式校准检验来正面回应这两点。

In Alkan et al. [17] authors introduce blind RED-diff, an extension of the RED-diff framework Mardani et al. [16] to solve blind inverse problems. The main idea is to use variational inference to jointly estimate the latent image and the unknown forward model parameters.

Similar to RED-Diff, the key mathematical formulation is the minimization of the KL-divergence between the true posterior distribution $p ( \pmb { x } _ { 0 } , \gamma | \pmb { y } )$ and a variational approximation $q ( { \pmb x } _ { 0 } , \gamma | { \pmb y } ) \colon$


![Eq. 3.2.2-minkl](../images/eef0d469248ca0625d267ac00037f207195eae3e322160f96884dfa4376e3067.jpg)


If we assume the latent image and the forward model parameters are independent, the KL-divergence can be decomposed as:


![Eq. 3.2.2-kldecomp](../images/e4c06e7c6da6b1a4ba7e4c08e8a0d40c89abf07d19d25d003edb5d996ecf3c33.jpg)


The minimization with respect to $q$ involves three terms:

i. ${ \mathcal { D } } _ { \mathrm { K L } } ( q ( { \pmb x } _ { 0 } | { \pmb y } ) | | p ( { \pmb x } _ { 0 } ) )$ ) represents the KL divergence between the variational distribution of the image $\mathbf { \Gamma } ( \pmb { x } _ { 0 } )$ and its prior distribution. This term is approximated using a score-matching loss, which leverages denoising score matching with a diffusion model (as in RED-Diff).

ii. $\mathcal { D } _ { \mathrm { K L } } ( q ( \gamma | \pmb { y } ) | | p ( \gamma ) )$ is the KL divergence between the variational distribution of the forward model parameters $( \gamma )$ and their prior distribution. This term acts as a regularizer on γ.

iii. $- \mathbb { E } _ { q ( \pmb { x } _ { 0 } , \gamma | \pmb { y } ) } [ \log p ( \pmb { y } | \pmb { x } _ { 0 } , \gamma ) ]$ is the expectation of the negative log-likelihood of the observed data y given the image $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and the forward model parameters $\gamma$ . This term ensures data consistency.

The resulting optimization can be achieved using alternating stochastic optimization, where the image x<sub>0</sub> and the forward model parameters γ are updated iteratively.

The formulation assumes conditional independence between $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and $\gamma$ given the measurement $^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } ^ { \mathbf { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } ^ { \mathrm { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } ^ { \mathrm { \Lambda } } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathrm { \Lambda } ^ { \mathrm { \Lambda } }$ and it also requires a specific form for the prior distribution $p ( \gamma )$

### 3.2.3 Score Prior Feng et al. [18]

> 💡 **机制拆解 3.2.3–3.2.4 Score Prior (Hao 批注)**: Score Prior 用 normalizing flow 当 $q_\phi$，难点是算先验项 $\log p_\theta(x_0)$。3.2.3 用 PF-ODE 的 instantaneous change-of-variables 精确算（Eq. 3.53，无近似误差但要几百上千 NFE，且每个 $y$ 都要重训 flow）；3.2.4 Efficient 版改用 ELBO 代理（Eq. 3.54/3.55，单 NFE 的去噪 loss），牺牲精确性换可扩展性。**这条线提示：精确后验代价极高，实用系统必须在"精确 likelihood"与"NFE 预算"间权衡——本课题若要 exactness+校准，也得直面同一算力墙。**

We again start by introducing a variational distribution $q _ { \phi } ( { \pmb x } _ { 0 } )$ that aims to approximate the posterior distribution determined by the diffusion prior. The optimization problem becomes


![Eq. 3.50](../images/089f1b235afb8b37642f3df715218822b4ebaa24eda0b90d232e062f708a1236.jpg)



![Eq. 3.51](../images/4c362447f07f9f070db98dab49d5dc81a06d7007db4466346763a3186089441d.jpg)


One of the most expressive yet tractable proposal distributions is normalizing flows (NF) Rezende and Mohamed [150], Dinh et al. [151]. Choosing $q _ { \phi }$ to be an NF, we can transform the optimization problem to


![Eq. 3.52](../images/a40f18bf0d37490e0db2d4fa6bbb3c8830de7d98b2282204c5deecfc02339e8c.jpg)


where the expectation is over the input latent variable $z ,$ and $\pi$ is the reference Gaussian distribution. Observe that the likelihood term and the entropy can be efficiently computed with a single forward/backward pass through the NF due to the parametrization of $q _ { \phi }$ with an NF. All that is left for us is to compute the prior term log $p _ { \pmb { \theta } } ( G _ { \phi } ( \pmb { z } ) )$ . In score prior Feng et al. [18], this is solved by leveraging the instantaneous change-of-variables formula with the diffusion PF-ODE, as originally proposed in Song et al. [2]


![Eq. 3.53](../images/36df9a08dbfaea4f0ef10cbe249297bbc609b71aca568aae60065cc578981219.jpg)


where $f _ { \theta } ( x _ { t } , t )$ is the drift term of the reverse SDE in Equation 2.2 with the score replaced by the network approximation. Notice that by plugging in Equation 3.53 to Equation 3.52, we can optimize the NF model in an unsupervised fashion. Notice that while this formulation does not incur approximation errors, it is very costly as every optimization steps involve computing Equation 3.53. Moreover, observe that the training of NF is done for a specific measurement $\mathbf { \pmb { y } } .$ . One has to run Equation 3.52 for every different measurement that one wishes to recover.

### 3.2.4 Efficient Score Prior Feng and Bouman [19]

As computing Equation 3.53 is costly, Feng et al. proposed to optimize $q _ { \phi }$ with the evidence lower bound (ELBO), originally presented in the work of Score-flow Song et al. [152] $b _ { \pmb \theta } ( \pmb x _ { 0 } ) \leq$ log $p _ { \pmb { \theta } } ( \pmb { x } _ { 0 } )$


![Eq. 3.54](../images/29ae42ce36286aa93b63524f8899e5c401a976897af11a5365654b123026a5dd.jpg)


where


![Eq. 3.55](../images/9b707ebc72dbceec2da936a76b9b8e136c17e902b13191fdabb17a5fbb66e6a0.jpg)


Intuitively, the value of $b _ { \theta }$ is small when we have a small denoising loss, and large when our diffusion denoiser $h _ { \theta }$ cannot properly denoise the given image. Replacing the exact likelihood Equation 3.53 that requires hundreds to thousands of NFEs to the surrogate denoising likelihood Equation 3.54 that requires only a single NFE makes the method much more efficient and scalable to higher dimensions.

## 3.3 Asymptotically Exact Methods

These methods aim to sample from the true posterior distribution. Of course, the intractability of the posterior distribution cannot be circumvented but what these methods trade compute for ap proximation error: as the number of network evaluations increases to infinity, these methods will asymptotically converge to the true posterior (assuming no other approximation errors).

> 💡 **家族总览 3.3 Asymptotically Exact（本课题校准落点）(Hao 批注)**: 这一族用 MCMC/SMC，**NFE→∞ 时收敛到真后验**（无其他近似误差时）。这是唯一有 exactness 保证的家族，因此**也是唯一能让 SBC/coverage 检验"有意义"的落点**——因为只有当采样器目标就是真后验时，校准诊断才在测算法而非测近似偏差。命门是算力：理论保证只在无限计算下成立。本课题若追"可校准的盲后验"，最诚实的骨架应在此族，再把 $\phi$ 也纳入 MCMC/SMC 状态。

### 3.3.1 Plug and Play Diffusion Models (PnP-DM) [24]

> 💡 **机制拆解 3.3.1 PnP-DM (Hao 批注)**: PnP-DM 引入辅助变量 $z$ 和 split 分布 $\pi(x_0,z|y)\propto p(x_0)p(y|z)\exp(-\frac{1}{2\rho^2}\|x_0-z\|^2)$（Eq. 3.56，$\rho\to0$ 时收敛到真后验），再用 **Gibbs 交替采样**：likelihood 步（Eq. 3.57，$z$ 满足测量，线性时 log-concave 好采）+ prior 步（Eq. 3.58，恰好是一个去噪问题，扩散模型的主场，初始化在 $z^{(k)}$、时刻 $\sigma_t=\rho$）。**这个"likelihood/prior 解耦 + Gibbs"结构对本课题很友好：把 $\phi$ 加进 likelihood 步的采样即可扩成盲版，且天然是采样（非点估计），利于校准。**

As explained in the introduction, the end goal is to sample from the distribution $p ( \pmb { x } _ { 0 } | \pmb { y } )$ ∝ $p ( { \pmb x } _ { 0 } ) p ( { \pmb y } | { \pmb x } )$ . The authors of [24] introduce an auxiliary variable z and an auxiliary distribution:


![Eq. 3.56](../images/a51bef01dd7fb9a6ab94c581fc8c80b02e25a0a145af4e9a4f098bc46e557a8e.jpg)


It is easy to see that as $\rho  0 .$ , the auxiliary distribution converges to the target distribution $p ( \pmb { x } _ { 0 } | \pmb { y } )$ To sample from the joint distribution $\pi ( \boldsymbol { x } _ { 0 } , z | \boldsymbol { y } )$ , the authors use Gibbs Sampling, i.e. the alternate between sampling from the posteriors. Specifically, the sampling algorithm alternates between two steps:

• Likelihood term:


![Eq. 3.57](../images/2cd2b67dfa16e75d14e13f319ff4888b09d778cbcbb7bac87e515ede656be60c.jpg)


• Prior term:


![Eq. 3.58](../images/0af9b99f8b6e1c400ec3dede277f8127a1abe77c7044cf38abf34e7c3129e171.jpg)


The likelihood term samples a vector that satisfies the measurements and is close to $\pmb { x } _ { 0 } ^ { ( k ) }$ . The prior term samples a vector that is likely under $p ( \pmb { x } _ { 0 } )$ and is close to $z ^ { ( k ) }$ . For most problems of interest, sampling from Equation 3.57 is easy because the distribution is log-concave, e.g. that’s the case for linear inverse problems. The interesting observation is that sampling from Equation 3.58 corresponds to a denoising problem, for which diffusion models excel. Indeed, for any $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ at noise level $\sigma _ { t } ,$ we have that:


![Eq. 3.59](../images/b414e77ef4cfd28d6fbabd881f23f5a27a1d771ddb2db60afb167dc3ee7be795.jpg)


Hence, to sample from Equation 3.58, one initializes the reverse process at $z ^ { ( k ) }$ and time t such that: $\sigma _ { t } = \rho .$

### 3.3.2 FPS Dou and Song [25]

FPS connects posterior sampling to Bayesian filtering and uses sequential Monte Carlo methods to solve the filtering problem, avoiding the need to handcraft approximations to the posterior $p ( \pmb { y } | \pmb { x } _ { t } )$ Given an observation y, FPS proposes to first construct a sequence $\{ y _ { t } \} _ { t = 0 } ^ { N }$ from y, and then determine a tractable distribution $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { y } _ { t - 1 } )$ . Starting from $\mathbf { x } _ { N } \sim \mathcal { N } ( \mathbf { 0 } , I _ { n } )$ , FPS can then recursively sample $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ for $t = N - 1 , \ldots , 1$ , and finally obtain $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . Specifically, FPS consists of two steps:

Step 1. Generating a sequence of $\{ \boldsymbol { y } _ { t } \} _ { t = 0 } ^ { N }$ with an observation $\mathbf { \nabla } _ { \mathbf { \nabla } _ { y } } .$ This can be done either using the forward process or unconditional DDIM backward sampling.

For the construction via the forward process, we recursively construct ${ \mathbf { } } _ { \mathbf { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } } \mathbf { _ { } } \mathbf { \psi } _ { \mathbf { } \psi } \mathbf { _ { } } \textbf { } \psi _ { } \psi _ { } \left. \textbf { } \psi _ { } \mathbf { } \psi _ { } \textbf { } \right.$ as follows:


![Eq. 3.60](../images/08df7a6f8b169994252e1b9f22908160a5e8b8a6aa9ae97ea9ed719d485334eb.jpg)


This arises from ${ \pmb x } _ { t } = { \pmb x } _ { t - 1 } + \sigma _ { t } { \pmb z } _ { t }$ and applying the linear operator A to it.

For the construction via backward sampling, FPS uses methods such as unconditional DDIM as in Equation 2.9,


![Eq. 3.61](../images/a6e295b3872269ed74117390114ca23fa7a5a4ad30c5449ad75b4d258578b3a1.jpg)


Here, $u _ { t } , \ v _ { t } .$ , and $w _ { t }$ are DDIM coefficients that can be explicitly computed. Note that $\mathbf { \nabla } _ { \mathbf { \boldsymbol { y } } \mathrm { { } } N }$ is sampled from $\mathcal { N } ( \mathbf { 0 } , A A ^ { \top } )$ because the prior distribution of the diffusion model is a standard Gaussian $\pmb { x } _ { N } \sim \mathcal { N } ( \mathbf { 0 } , I )$ , and due to the linearity of the inverse problem, ${ \bf { } } _ { { \bf { } } ^ { g } N } =$ $A { \pmb x } _ { N }$

Step 2. Generating a backward sequence of $\{ \pmb { x } _ { t } \} _ { t = 0 } ^ { N }$ from Step 1’s $\{ y _ { t } \} _ { t = 0 } ^ { N } .$ First, note that $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { y } _ { t - 1 } )$ is a tractable normal distribution. This results from applying Bayes’ rule and the conditional independence of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and the random vector $Y _ { t - 1 } \mathrm { g i v e n } x _ { t - 1 } .$


![Eq. 3.62](../images/23cd24972c158ce9b78373ecd0ff21f956dd87b35bb39512a197dd1f3bc5bfa4.jpg)


Here, $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ is approximated via backward diffusion sampling with learned scores, and $p ( \dot { \mathbfcal { Y } } _ { t - 1 } | \dot { \mathbf { x } } _ { t - 1 } ) = \bar { \mathcal { N } } ( A \mathbfit { x } _ { t - 1 } , c _ { t - 1 } ^ { 2 } I )$ , where $c _ { t - 1 }$ , dependent on $\sigma _ { y } \gt 0$ , can be computed explicitly [47]. Thus, with $\{ y _ { t } \} _ { t = 0 } ^ { N }$ and initial condition x $\mathbf { \Omega } _ { N } \ \sim \ { \mathcal { N } } ( \mathbf { 0 } , I _ { n } )$ , FPS recursively samples $\pmb { x } _ { N - 1 } , \cdots \pmb { x } _ { 1 }$ using $p ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { Y } _ { t - 1 } = \pmb { y } _ { t - 1 } )$ , ultimately yielding $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$

FPS algorithm is theoretically supported to recover the oracle $p ( { \pmb x } | { \pmb y } )$ once the step size is sufficiently small.

### 3.3.3 PMC Sun et al. [26]

Plug-and-Play (PnP) Kamilov et al. [110] and RED Romano et al. [107] are two representative methods of using denoisers as priors for solving inverse problems. Let $\begin{array} { r } { g _ { \pmb { y } } ( \pmb { \bar { x } } ) = \frac { 1 } { 2 \sigma _ { \ b { u } } ^ { 2 } } \lVert \pmb { \bar { y } } - A \pmb { x } \rVert _ { 2 } ^ { 2 } } \end{array}$ be the log-likelihood function, $h _ { \theta } ^ { \sigma } ( \cdot )$ an MMSE denoiser from Equation 2.11 conditioned on the noise level $\sigma ,$ , and $R _ { \pmb \theta } ^ { \sigma } ( \cdot ) : = \mathrm { I d } - h _ { \pmb \theta } ^ { \sigma } ( \cdot )$ the residual projector. Note that conditioning on the noise level $\sigma$ is equivalent to the network being conditioned no t, since the mapping is one-to-one. A single iteration of these methods read

• PnP proximal gradient method Kamilov et al. [153]:


![Eq. 3.63](../images/1f719c581f9cc5de467a947ada6b29d22a835d4f097bb46686b8168598020243.jpg)



![Eq. 3.64](../images/b58079c3c363ae1460210a368cb74be8638fdcd8565efce66ce4a69b221a4f73.jpg)


• RED gradient descent Romano et al. [107]:


![Eq. 3.65](../images/1db7007e0c735ddfeb67c6b94b696acd474f651abd3eaadd4ce31376d556d807.jpg)



![Eq. 3.66](../images/c5646f58da25c11343a34bb6ad5fa7b30eca888f1a9dc2aa37517fbbc59a808f.jpg)


Notice that by using Tweedie’s formula, we see that $R _ { \theta } ^ { \sigma } ( { \pmb x } ) = - \sigma ^ { 2 } \nabla _ { \pmb x } \log p _ { \sigma } ( { \pmb x } )$ . Rearranging Equation 3.64 and Equation 3.66,


![Eq. 3.67](../images/3eb1b0319f6befba020b38743e1c47f21ecfe7b353634f761728d79a5258b595.jpg)


• RED:


![Eq. 3.68](../images/64f8ff5f8a4b87d2a6e5be4dbcb53d1a955575fc445c5f6c9c7d2b2e2ae02da7.jpg)


Moreover, by setting $\gamma = \sigma ^ { 2 }$ and $\tau = 1 / \sigma ^ { 2 }$ , one can show that


![Eq. 3.69](../images/160fc7a4a1338e76268b4b85dbaef935a28614da1f66c87fbd6f5d9e08db5cf9.jpg)


In other words, we see that the iteration of PnP/RED in Equation 3.64 and Equation 3.66 will converge to sampling from the posterior as $\sigma ^ { 2 } = \gamma  0$


![Eq. 3.70](../images/650be9f236fcf15f5868747fcdfeab09be3d73b3b2aa428aa47f5281de3e69fe.jpg)


where t indexes the continuous time flow of $^ { \mathbf { \delta x } , }$ as opposed to the discrete formulations in Equation 3.64 and Equation 3.66. Note that this notion of t does not match the diffusion time t, where the time index matches a specific noise level. In PMC, the authors propose to incorporate noise level annealing as done in the usual reverse diffusion process by starting from a large noise level $\sigma$ and gradually reducing the noise level. Solving Equation 3.70 with PMC then boils down to iterative application of Equation 3.64 and Equation 3.66 with the annealing strategy. Moreover, introducing Langevin diffusion yields a stochastic version


![Eq. 3.71](../images/312b394f9aab09155f12a286a7bae2a691dc86f5ced0df7fd4b540125fe810c6.jpg)


which can be solved in the same way, but with additional stochasticity.

### 3.3.4 Sequential Monte Carlo-based methods

> 💡 **机制拆解 3.3.4 SMC 族（粒子后验）(Hao 批注)**: SMCDiff/MCGDiff/TDS 用 K 个粒子沿 $X_{1:T}$ 传播（proposal → reweight → resample），逼近真后验。**粒子集天然给出后验的经验分布 → 直接可算 coverage/CRPS**，是四家族里对"不确定性量化"最原生的。代价是粒子数与维度的 scaling。对本课题：SMC + 把 $\phi$ 纳入粒子状态，是"盲联合后验 + 校准"最理论干净的路线之一，但要解决高维粒子退化。

SMCDiff Trippe et al. [27], MCGDiff Cardoso et al. [28], and TDS Wu et al. [29] belong to the category of sequential Monte Carlo (SMC)-based methods Doucet et al. [154]. SMC aims to sample from the posterior by constructing a sequence of distributions $X _ { 1 : T }$ , which terminates at the target distribution. The evolution of the distribution is approximated by K particles. In a high level, SMC can be described with three steps: 1) Transition with a proposal kernel $\{ \pmb { x } _ { t } ^ { 1 : K } \} \sim p ( \mathbf { \breve { X } } _ { t } | \mathbf { X } _ { t - 1 } ) , 2 )$ computing the weights to re-weight the importance, and 3) resampling from a reweighted multinomial distribution. Methods that belong to this category propose different ways of constructing the proposal distribution and the weighting function.

## 3.4 CSGM-Type methods

### 3.4.1 DMPlug [20], SHRED [21]

> 💡 **机制拆解 3.4 CSGM 族 (Hao 批注)**: CSGM 把整个确定性 ODE 采样器看成 $z\mapsto\hat{x}(z)$ 的映射，直接优化初始 noise $z^*=\arg\min\|y-A\hat{x}(z)\|^2$（Eq. 3.72/3.73）。DMPlug/SHRED 用 few-step 采样（3/10 步）规避 BPTT 显存爆炸。**这族本质是点估计（找一个最优 $z$），不产出后验，与本课题的后验采样目标正交**——除非改成对 $z$ 做贝叶斯采样。Score-ILO 在中间层优化并用扩散正则，缓解出流形问题。

Compressed sensing generative model (CSGM) [155, 156] is a general method for solving inverse problems with deep generative models by aiming to find the input latent vector z through


![Eq. 3.72](../images/7bbf73a61fac7ba4d09c69f00a982007501e5c4cf67296a229232795fa38f1b5.jpg)


where $G _ { \theta }$ is an arbitrary generative model. DMPlug and SHRED can be seen as extensions of CSGM to the case where one uses a diffusion model. Unlike GANs or Flows where the mapping from the latent space to the image space is done through a single NFE, diffusion models require multiple NFE to solve the generative SDE/ODE. One can rewrite Equation 3.72 as


![Eq. 3.73](../images/a84ad9beb0537b3e7a9af2b74ec241a627e1ea35670ed9454f18bb5a29138b99.jpg)


where ${ \hat { \mathbf { x } } } = { \hat { \mathbf { x } } } ( z )$ is the solution of the deterministic sampler initialized at z. Essentially, the models in this category optimize over the “latent” space of noises that are fed to the deterministic ODE sampler. One caveat of Equation 3.73 is the exploding memory required for backpropagation through time. To mitigate this, when sampling from $p _ { \pmb { \theta } } ( \pmb { x } _ { 0 } | \pmb { x } _ { T } )$ , a few-step sampling (e.g. 3 for DMPlug and 10 for SHRED) is used to approximate the true sampling process.

### 3.4.2 CSGM with consistent diffusion models [22]

Diffusion models can be distilled into one-step models, known as Consistency Models [157], that solve in one step the Probability Flow ODE. These models can be used in Equation 3.73, replacing the ODE sampling, to reduce the computational requirements [22].

### 3.4.3 Intermediate Layer Optimization [156, 23]

CSGM has been extended to perform the optimization in some intermediate latent space [156]. The problem is that the intermediate latents need to be regularized to avoid exiting the manifold of realistic images. Score-Guided Intermediate Layer Optimization (Score-ILO) [23] uses diffusion models to regularize the intermediate solutions.

## 3.5 Latent Diffusion Models

### 3.5.1 Motivation

> 💡 **机制拆解 3.5 Latent 系动机 (Hao 批注)**: latent diffusion（SD）解逆问题有四道坎：(1) **失去线性**——扩散在 latent、测量在像素，Enc/Dec 非线性使线性逆问题也变非线性；(2) 解码贵；(3) Enc∘Dec 非一一映射，guidance 会被拉向任意满足测量的 latent；(4) 文本条件。后面 Latent DPS/PSLD/Resample/MPGD/P2L/TReg/STSL 各治一坎。**本课题若用 SD 类先验做盲问题，这四坎叠加 $\phi$ 估计会更棘手；用像素域扩散先验更干净。**

In this subsection, we focus on algorithms that have been developed for solving inverse problems with latent diffusion models (see Section 2.3). There are a few additional challenges when dealing with latent diffusion models that have led to a growing literature of papers that are trying to address them.

Loss of linearity. The first challenge in solving inverse problems with latent diffusion models is that linear inverse problems become essentially non-linear. The problem stems from the fact that diffusion happens in the latent space but measurements are in the pixel-space. In order to guide the diffusion there are two potential solutions: i) either project the measurements to the latent space through the encoder, or, ii) project the latents to the pixel space as we diffuse through the decoder. Both approaches depend on non-linear functions (Enc, Dec respectively) and hence even linear inverse problems need a more general treatment.

Decoding is expensive. The other issue that arises is computational. Most of the time, we need to decode the latent to pixel-space to compare with the measurements. The motivation behind latent diffusion models is to accelerate training and sampling. Hence, we want to avoid repeated calls to the decoder as we solve inverse problems.

Decoding-encoding map is not one-to-one. Even if we ignore the computational challenges, it is not straightforward to decode the latent to the pixel-space, compare with the measurements and get meaningful guidance in the latent space since the decoding-encoding map is not an one-to-one function.

Text-conditioning. Finally, latent diffusion models typically get a textual prompt as an additional input. A lot of algorithms that have been developed in the space of using latent diffusion models to solve inverse problems innovate on how they use text conditioning.

### 3.5.2 Latent DPS

The first algorithm we review in the space of solving inverse problems with latent diffusion models is Latent DPS, i.e. the straightforward extension of DPS for latent diffusion models. The approximation made in this algorithm is:


![Eq. 3.74](../images/e2c3619b0ae0e215093086e6df4476e06abb1e5898cd3cc0dd94307e2f8b153e.jpg)


The algorithm works by performing one-step denoising in the latent space and measuring how much the decoding of the denoised latent matches the measurements y.

### 3.5.3 PSLD Rout et al. [14]

The performance of Latent DPS is hindered by the fact that the decoding-encoding map is not an one-to-one function, as discussed earlier. The approximation made above could pull $\mathbf { \bar { x } } _ { t } ^ { \mathrm { E } }$ towards any latent $\boldsymbol { x } _ { 0 } ^ { \mathrm { E } }$ that has a decoding that matches the measurements while the score function is pulling $\pmb { x } _ { t } ^ { \check { \mathrm { E } } }$ towards a specific $\boldsymbol { x } _ { 0 } ^ { \mathrm { E } }$ , i.e. towards $\mathbb { E } [ \pmb { x } _ { 0 } ^ { \mathrm { E } } | \pmb { x } _ { t } ^ { \mathrm { E } } ]$

PSLD mitigates this problem by adding an additional term that pulls towards latents that are fixed points of the decoder-encoder map. Concretely, the approximation made in PSLD is:


![Eq. 3.75](../images/5e8861dbadc5761678713ead956a258df87fa9679534c9afb93fe5fb1dbaa0cd.jpg)


where $\gamma _ { t }$ is a tunable parameter.

### 3.5.4 Resample Song et al. [32]

Resample, a concurrent work with PSLD, proposes an alternative way to improve the performance of Latent DPS. After each clean prediction $\widehat { \pmb { x } } _ { 0 } ( \pmb { x } _ { t + 1 } ^ { \mathrm { E } } )$ is obtained from the previous sample $\mathbf { \boldsymbol { x } } _ { t + } ^ { \mathrm { E } }$ 1 via Tweedie’s formula in Equation 2.10, and the unconditional reverse denoising process is updated using, say, DDIM:


![Eq. 3.76](../images/bc158f9a66f4983e767be35037941afc8358f160269035514a7fb9ec28995949.jpg)


the authors project the latent back to a point $\widehat { \pmb { x } } _ { t }$ that satisfies measurements using:


![Eq. 3.77](../images/07aa8e7b76b8364282b34a7e7c48e21ae2a0a825a38fecd6f84ac5eafb67cb6e.jpg)


Here, $\sigma _ { t } ^ { 2 }$ is a hyperparameter used to tune the alignment with measurements, $\bar { \alpha } _ { t }$ is predefined in forward process, and $\widehat { \pmb { x } } _ { 0 } ( \pmb { y } )$ is found by solving:


![Eq. 3.78](../images/8371e8ccb17fb5b52861f0c0e3dbe77b2d8d9ef446bdb74327f7b58778b25c92.jpg)


## 3.6 MPGD He et al. [158]

The MPGD authors note that some methods require expensive computations for measurement alignment during gradient updates, as they involve passing through the gradient (chain rule) of the pretrained diffusion model $\epsilon _ { \theta } ( x _ { t } ^ { \mathrm { E } } , t )$


![Eq. 3.79](../images/c7ef9d94239d0acb4f82a764a8bbbefa9ba39929e8c9ea66aa50d4fb83a7d35f.jpg)


where $\begin{array} { r } { \mathbf { \Delta x } _ { 0 \mid t } : = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \big ( \mathbf { \Delta x } _ { t } ^ { \mathrm { E } } - \sqrt { 1 - \bar { \alpha } _ { t } } \mathbf { \epsilon } _ { \theta } ( \mathbf { \Delta x } _ { t } ^ { \mathrm { E } } , t ) \big ) } \end{array}$ is a clean estimation via Tweedie’s formula in Equation 2.10. This gradient bottleneck slows down the overall inverse problem solving. MPGD proposes bypassing the direct gradient $\nabla _ { \pmb { x } _ { t } ^ { \mathrm { E } } }$ with theoretical guarantees by updating with $\nabla _ { \pmb { x } _ { 0 | t } }$


![Eq. 3.80](../images/4068f6505603a6a26cf2e348c214d505ddfe7d60e3abdd0eca072e6ebf4310d2.jpg)


with


![Eq. 3.81](../images/a28e1384fa83604d0eb6d299b0bfac37002cc8be88e6ad24eeebd96cd7d23b87.jpg)


and use the obtained $\pmb { x } _ { 0 | t } ^ { \prime }$ for unconditional reverse denoising process


![Eq. 3.82](../images/0982d668a783713020071cda8c582d056ca021406f5f5ca0678ce0dc69e0bc3f.jpg)


### 3.6.1 P2L [34]

While text conditioning is a viable option for modern latent diffusion models such as Stable diffusion, the actual use was underexplored due to ambiguities on which text to use. P2L addresses this question by proposing an algorithm that optimizes for the text embedding on the fly while solving an inverse problem.


![Eq. 3.83](../images/33eb1cd09a452daf33458dada1471c18b17f15a821aafaf7f7c4ec6efcb6a883.jpg)


where c is the text embedding, and one can approximate $\mathbb { E } [ \pmb { x } _ { 0 } ^ { \mathrm { E } } | \pmb { x } _ { t } ^ { \mathrm { E } } , \pmb { c } ]$ by using the Tweedie’s formula with the denoiser conditioned on c. Using the optimized embedding at each timestep $\boldsymbol { c } _ { t } ^ { * }$ , sampling follows the procedure of Latent DPS


![Eq. 3.84](../images/87546f305737ab8c267587a3f7c0f92d2b798e79c978cf289341d028e6c1a9c0.jpg)


In addition to the optimization of the text embedding, P2L further tries to leverage the VAE prior by decoding - running optimization in the pixel space - re-encoding


![Eq. 3.85](../images/30b61d175f974fee803f932cfbeb10b2c611d4d758c1b7587bd3a40cc7fc037c.jpg)



![Eq. 3.86](../images/05204c1dcd0c265ed89f5b7f995e0565b6b46efd39115ccb4bb32a9e476a2e88.jpg)


### 3.6.2 TReg [35], DreamSampler [36]

Instead of automatically finding a suitable text embedding to achieve maximal reconstructive performance, another advantage of text conditioning is that it can be used as an additional guiding signal to lead to a specific mode. This may seem trivial, as one has access to a conditional diffusion model. However, in practice, simply using a conditional diffusion model does not induce enough guidance as reported in [159, 160], and naively using classifier free guidance [160] (CFG) does not lead to satisfactory results. In addition to using data consistency imposing steps as in P2L, TReg proposes adaptive negation to update the null text embeddings used for CFG guidance.


![Eq. 3.87](../images/5d2a023745e9abc1e76a2d7090ee5038171221fc32312afabe2c1708e802b4dd.jpg)


where $\mathbf { \boldsymbol { x } } ^ { * }$ comes from Equation 3.85, sim denotes the CLIP similarity [161] score, and  is the CLIP image encoder. In essence, Equation 3.87 minimizes the similarity between the current estimate of the image and the null text embedding. Hence, when the optimized $c _ { \mathcal { O } }$ is used for CFG with


![Eq. 3.88](../images/10eed2d89338ecf94c57bf09d4fb66faf35065330e9bcadfd84363c6fb09e145.jpg)


the conditioning vector direction ${ \epsilon _ { \theta } } ( x _ { t } ^ { \mathrm { E } } , c ) - \epsilon _ { \theta } ( x _ { t } ^ { \mathrm { E } } , c _ { \emptyset } ^ { * } )$ is amplified. Later, TReg was further advanced by devising a way to better make use of CFG by combining score distillation sampling Poole et al. [162] into the sampling framework.

### 3.6.3 STSL Rout et al. [15]

> 💡 **机制拆解 3.6.3 STSL（二阶/协方差）(Hao 批注)**: STSL 指出多数方法只用了 $p(X_0|X_t)$ 的**均值**（单步梯度），它进一步用**协方差**——在 fidelity loss 里加 $\nabla\text{Trace}(\nabla^2\log p(x_t^E))$（Eq. 3.89），用 Hutchinson 迹估计（Eq. 3.90）近似。**这与 Moment Matching (3.1.6) 同源：都试图把二阶信息注入 guidance，缩小 prior/posterior score 差距。对本课题的意义：二阶信息正是后验 spread 的来源，是校准的物理基础。**

Most methods leverage the mean of the reverse diffusion distribution $p ( { X } _ { 0 } | { X } _ { t } )$ , and take a single gradient step with Equation 3.74. To further leverage the covariance of $p ( { X } _ { 0 } | { X } _ { t } )$ , Rout et al. Rout et al. [15] propose to use the following fidelity loss


![Eq. 3.89](../images/60c5f6f16bd2c4856eeb546e41c0e3a074828e23d75e3ace8d6e5729877a2eda.jpg)


where $\gamma$ is a constant. To effectively compute the trace, one can further use the following approximation


![Eq. 3.90](../images/422029c90e54cd372d08a3694d2708de36a0d45a76d29e6d2aa0d8f0f32ced36.jpg)


where $\pi$ can be a Gaussian or a Rademacher distribution. Using the loss in Equation 3.89 with Equation 3.90, STSL uses multiple steps of stochastic gradient updates per timestep.

---

## 🔖 Section 总结

### 关键数字速查
| 项 | 内容 |
|------|------|
| Explicit 统一模板 | Eq. 3.1：$-\mathcal{L}_t\mathcal{M}_t/\mathcal{G}_t$ |
| 协方差保真谱 | DPS($\delta$) → ΠGDM(各向同性高斯) → Moment Matching(真协方差) |
| 盲方法（3 个）| BlindDPS(3.1.7, 双 SDE)、GibbsDDRM(3.1.8, Gibbs)、Blind RED-Diff(3.2.2, 变分交替) |
| 四家族 | Explicit / Variational / Asymptotically Exact / CSGM（+Latent 系）|

### 核心洞察
1. **一切归于近似 $\nabla_{x_t}\log p_t(y|x_t)$**：Explicit 给闭式、Variational 换 $q$、Asymptotically Exact 用 MC 采、CSGM 优化 latent noise。
2. **DPS→ΠGDM→Moment Matching 是一条"$p(x_0|x_t)$ 协方差保真度递增"的谱**，直接对应 prior/posterior score 差距的收窄，也是后验校准的物理基础。
3. **三个盲方法各有近似**：BlindDPS/Blind RED-Diff 用点/独立性近似（校准无保证），GibbsDDRM 用 Gibbs 更接近真联合后验但 mixing/校准未验证。本课题的 gauge-aware 联合后验 + SBC/coverage/CRPS 正是补这一空白。
4. **Asymptotically Exact + SMC 是校准最有意义的落点**（有 exactness 保证 + 粒子原生给经验后验），代价是算力和盲扩展。

### 可追问点
- 把 $\phi$ 纳入 PnP-DM 的 Gibbs likelihood 步 / SMC 的粒子状态，能否得到"可校准的盲联合后验"？
- Moment Matching 的真协方差能否直接用于 coverage 诊断？
