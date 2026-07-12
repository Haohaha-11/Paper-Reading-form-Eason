[← 返回 README](../README.md)

# 2. Preliminarily and Background

## 📌 预览

背景节铺四块基础：(2.1) 扩散模型两种等价视角——score 视角 $s_\theta\approx\nabla\log p_t$ 与 denoiser 视角，本文把扩散模型当作"可查询的隐式分布先验"；(2.2) 扩散先验解逆问题的通用框架 $p_\theta(x\mid y)\propto p(y\mid x)p_\theta(x)$，以及施加测量一致性的四类机制（梯度/投影/采样/优化）；(2.3) 逆问题后验不确定性的 AU/EU 分解；(2.4) 现有评价指标的局限——精度不够、Wasserstein 只能在仿真、FID/LPIPS 需要双边样本。这一节是 score-KSD 的"零件仓库"。

---

## 2.1 Diffusion Models

Diffusion models (DM) have demonstrated extraordinary ability to generate high quality images [38, 16, 35]. A diffusion model defines a forward noising process that transforms clean data $x_0 \sim p_{\text{data}}$ into noisy variables $x_t$ for $t \in [0, T]$, and learns a network that enables reversing this process. In practice, the training of diffusion model can be viewed as either (i) estimating a score function $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)$ as formulated in the score-based DM [38], or (ii) learning a denoiser that predicts a clean image $\hat{x}_0 = \text{Denoise}_\theta(x_t, t)$ from the noisy image $x_t$ as formulated in Denoising Diffusion Probabilistic Model [16], where t denotes the diffusion sampling steps. Throughout, we view the diffusion model as an implicit distributional prior that can be queried via the score function or denoising operations, when the prior density log $p_\theta(x_0)$ is not available in closed form.

> 💡 **机制拆解 (Hao 批注)**: 记住最后一句——这是 score-KSD 能成立的前提：**扩散模型是"隐式先验"，密度 $\log p_\theta(x_0)$ 没有闭式，但它的 score（梯度）可以查询**。score-KSD 的巧妙之处正是绕开密度、只用 score：先验密度算不出，但先验 score 能从 $s_\theta$ 近似出来。score 视角 (i) 与 denoiser 视角 (ii) 通过 Tweedie 公式等价，本文用的是 score 视角。

## 2.2 Diffusion Priors for Inverse Problem Solving

The inverse problem aims at reconstructing an unknown signal x $\in \mathbb{R}^n$ based on the measurements $y \in \mathbb{R}^m$. Formally, y derives from a forward process determined by $y = \mathcal{A}x + \epsilon$, where $\mathcal{A}$ can be either a linear operator, such as the Radon transform in sparse-view CT reconstruction and Fourier transform in accelerated MRI, or a nonlinear operator, such as the JPEG restoration encoder. $\mathcal{A}$ can also be either given or unknown. In this work, we focus on the situation where A is given. The term $\epsilon$ denotes random measurement noise.

> 💡 **机制拆解 (Hao 批注)**: **这句"In this work, we focus on the situation where A is given"是本文与本课题的关键分界线**。本文把前向算子 $\mathcal{A}$（以及后面用到的噪声尺度 $\sigma_y$）当已知常量。而我们的盲逆问题里 $\mathcal{A}=\mathcal{A}(\phi)$ 由低维参数 $\phi$ 决定且未知，$\sigma$ 也未知。也就是说：score-KSD 的似然 score 项 $\sigma_y^{-2}\mathcal{A}^\top(y-\mathcal{A}x)$ 在盲设置下无法直接算——这是我们迁移这套评价必须先解决的第一道坎（把 $\phi,\sigma$ 也纳入被评价/被估计的量）。

Diffusion inverse solver (DIS) methods combine a pretrained diffusion model prior $p_\theta(x)$ with a known forward model to perform inference for the posterior $p_\theta(x \mid y) \propto p(y \mid x) p_\theta(x)$, where the prior term $p_\theta(x)$ comes from the diffusion model prior and the likelihood term $p(y \mid x)$ is determined by forward operator A and the noise model. The likelihood term enforces measurement consistency by favoring reconstructions that yield high $p(y \mid x)$. In practice, DIS algorithms impose measurement consistency in the diffusion sampling trajectory via different mechanisms, including gradients [5, 47, 43], projection [6, 19, 23, 41, 23], sampling [2, 10], or other optimizations [31, 36].

> 💡 **机制拆解 (Hao 批注)**: 这里给出 DIS 的统一公式 $p_\theta(x\mid y)\propto p(y\mid x)p_\theta(x)$，并把众多算法按"如何注入测量一致性"分成四类机制：
> - **梯度类**（DPS [5]、DAPS [47]、PnPDM [43]）：在采样轨迹上加似然梯度。
> - **投影类**（DDRM [23]、DDNM [41]）：往测量一致的子空间投影。
> - **采样类**（MCG-Diff [2]、FPS [10]）：粒子滤波/SMC，理论上更接近真后验。
> - **优化类**（RED-Diff [31,36]）：变分/优化视角。
> 这个分类在读实验表时极有用：**后面会看到"采样类"（MCG-Diff）score-KSD 常最小、"优化类"（RED-Diff）常精度高但 score-KSD 差**，四类机制的差异正好解释了 Accuracy Trap 的来源。

## 2.3 Posterior Uncertainty in Inverse Problems

Solutions to the ill-posed inverse problems are inherently uncertain due to incomplete measurements, measurement noise, and imperfect prior information [21, 39]. In machine learning literature, these uncertainties are commonly categorized into epistemic uncertainty (EU) arising from limited information or model uncertainty, and aleatoric uncertainty (AU) arising from intrinsic stochasticity in the measurement generation process [24, 33].

In diffusion-based inverse problems solving, AU is primarily induced by measurement noise, while EU is associated with information loss from the ill-posed forward operator, potential model specification or prior mismatch. Thus, intrinsic posterior distribution induced by the inverse problem should exhibit substantial uncertainty, particularly under ill-posed measurement settings. Since stochastic DIS aims to characterize posterior uncertainty through generated samples, posterior fidelity naturally becomes a key criterion for evaluating whether the sampled distributions reflect the underlying posterior behavior induced by the inverse problem.

> 💡 **机制拆解 (Hao 批注)**: AU/EU 分解在本文的具体映射很清楚：
> - **AU（偶然不确定性）**= 测量噪声 $\epsilon$ 带来的，噪声尺度 $\sigma_y$ 越大后验越宽。
> - **EU（认知不确定性）**= 前向算子病态导致的信息丢失 + 先验失配（例如 Section 5 的 OOD 任务：用 LIDC 训的扩散先验去重建癌症 CT）。
> 这个分解对本课题特别相关：盲设置里 $\phi$ 的不确定性（算子参数没标定准）本质是一种 EU，$\sigma$ 的不确定性又直接改变 AU。本文只把这两者当"后验应有的宽度"来评价 $x$，而我们要把 $\phi,\sigma$ 本身也当成被估计对象——这是差异也是可扩展点。

## 2.4 Limitation on Current Evaluation Metrics

Existing work mainly benchmarks the reconstruction quality by accuracy (e.g., PSNR/SSIM). While accuracy metrics remain necessary, they are insufficient for evaluating DIS methods. There are two fundamental reasons: (i) in ill-posed inverse problems the target is a posterior distribution $p(x \mid y)$ with many plausible reconstructions of the same measurement, and (ii) most DIS algorithms are inherently stochastic, producing a distribution of reconstructions rather than a single deterministic output. Together, the object of interest is a distribution over reconstructions, motivating uncertainty-aware evaluation.

Although posterior fidelity of DIS has recently received increasing attention [2, 45], existing metric such as Wasserstein distance is primarily limited to controlled simulation settings where ground-truth posterior is accessible. Common distributional metrics in real images such as FID [15] and LPIPS [48] require samples from both compared distributions, making them inapplicable to real-world inverse problems where neither true posterior samplers nor normalized posterior densities are accessible. This limitation highlights an urgent need for distributional posterior fidelity evaluation methods that do not rely on access to ground-truth posterior distribution or samples.

> 💡 **消融解读 (Hao 批注)**: 这段是 score-KSD 的"反面论证"，把竞品指标的死穴逐一点名：
> - **PSNR/SSIM**：只评单点，无法评分布（两条理由 (i)(ii)）。
> - **Wasserstein / sliced-Wasserstein**：能评分布，但要真后验样本，只活在仿真。
> - **FID/LPIPS**：双边指标，需要"目标分布"也能采样——真实逆问题里做不到。
> 于是逼出唯一的出路：一个**只需生成样本 + 目标 score field**的单边指标。KSD（核 Stein 差异）天生就是这种"只需目标分布的 score、不需目标分布的样本"的度量——这就是为什么第 3 节选它。给本课题：我们做 $x$-后验校准时，若也拿不到真后验，score-KSD 是可直接落地的补充诊断；而 SBC/coverage/CRPS 走的是另一条路（需要在联合先验下重复采样构造 rank），两者互补。
