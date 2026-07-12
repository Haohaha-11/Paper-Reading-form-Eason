[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览

引言把逆问题的贝叶斯目标（恢复后验 $p(x_0\mid y)$ 而非单点估计）说清楚，然后把已有方法分成 training-free 与 training-based 两大阵营并逐一指出其近似来源，最后引出本文的观察：**线性高斯下后验 score 有闭式解，且仍是一个去噪问题**。Figure 1 是全文的直觉图。

---

Linear inverse problems, in which an unknown signal $x_0$ must be recovered from a noisy linear measurement $y = A x_0 + \eta$ with known forward operator A and observation noise η, are pervasive across imaging and the sciences, including compressive sensing [1, 2], accelerated medical imaging [3, 4], super-resolution [5, 6], deblurring [7, 8], and inpainting in computational photography [9, 10]. The forward operator A is typically ill-conditioned or rank-deficient, so many candidate signals are consistent with the same observation, and the right object to recover is the posterior $p(x_0 | y)$ rather than any single point estimate. The posterior captures uncertainty over reconstructions, supports downstream decisions, and exposes the trade-off between data fidelity and prior plausibility.

> 💡 **问题动机 (Hao 批注)**: 这一段立住了"为什么要后验而不是点估计"。$A$ 病态或秩亏 → 零空间 $\mathcal{N}(A)$ 里的方向观测不到 → 无穷多 $x_0$ 都满足 $Ax_0\approx y$。所以正确目标是**分布** $p(x_0\mid y)$：它既量化了不确定性，又暴露了 fidelity（贴合观测）与 prior plausibility（贴合先验）之间的权衡。这正是校准研究的立足点——一个"好"的后验采样器不仅要 PSNR 高，更要**覆盖率对**（coverage/CRPS）。

Diffusion and flow-based generative models offer powerful, expressive data priors for this task, learning a denoising trajectory from noise back to clean samples [11–17]. The central question is how to turn this trajectory into a sampler from $p(x_0 | y)$. The reverse-time sampler needs the posterior score $\nabla_{x_t} \log p(x_t | y)$, not the unconditional prior score $\nabla_{x_t} \log p(x_t)$ that diffusion training provides. Replacing the former by an approximation introduces bias at every step, which compounds into oversmoothing, hallucinated structure, or poorly calibrated uncertainty. Existing methods fall into two broad camps.

> 💡 **机制拆解 (Hao 批注)**: 核心矛盾一句话——扩散训练只给你**无条件 score**，采样后验要的是**后验 score**，二者不等。用近似替代后验 score 会**每一步都注入偏差**，逐步累积成三种典型病症：oversmoothing（过平滑）、hallucination（幻觉出不符合观测的结构）、poorly calibrated uncertainty（不确定性刻画不准）。第三种正是我们校准课题最关心的失效模式。

![Figure 1](../images/baa4f58aaeb73c684a44461261846a587d516f4d1e6a30bd5c168b5095cd88a3.jpg)

*Figure 1: EPS turns posterior sampling into denoising with the right query geometry. Instead of denoising an isotropic query at $x_t$, the measurement shifts the query to the posterior pivot $\mu_\star$ and reshapes the noise into an anisotropic covariance $\Sigma_\star$. Measured directions become more certain, while unobserved directions remain uncertain. EPS trains a denoiser for this anisotropic geometry and reuses the backbone's unconditional sampler unchanged. The first step of the resulting sampler corresponds to an estimate of the posterior mean $\mathbb{E}[x_0 | y]$, which typically has higher PSNR but is over-smoothed, while the sample produced at the end (in this case, 100 steps) has more details.*

> 💡 **Figure 1 批读 (Hao 批注)**: 这张图是全文最重要的直觉，讲的是"query geometry（查询几何）"的改变：
> - **左（普通去噪）**：无条件去噪器在 $x_t$ 处、各向同性噪声球 $\beta_t^2 I$ 下去噪——所有方向不确定性相同。
> - **右（EPS）**：测量把查询点从 $x_t$ **平移**到 pivot $\mu_\star$，并把噪声球**压扁**成椭球 $\Sigma_\star$——被 $A$ 观测到的方向被压缩（更确定），零空间方向仍保持大方差（不确定）。
> - **采样轨迹的读法**：第一步（高噪声极限）等价于后验均值 $\mathbb{E}[x_0\mid y]$，PSNR 高但过平滑；跑满 100 步得到的是一个**真正的后验样本**，细节更丰富。这条"从后验均值起、到后验样本止"的路径解读是理解 1-NFE vs 100-NFE 权衡的钥匙。

Training-free methods keep a pretrained denoising backbone fixed and add a measurement-matching update at each reverse step. The prototypical example is Diffusion Posterior Sampling (DPS) [18], which differentiates a measurement loss through the unconditional denoiser, with variants using projections, denoised estimates, or task-specific correction rules [18–25]. This route is attractive because it is zero-shot and inherits the strong unconditional prior of a pretrained model. However, the added update is only an approximation to the true measurement-matching score, and even moment-matching variants that track anisotropic uncertainty in $p(x_0 | x_t)$ [26–28] only refine the unconditional denoising query. Asymptotically exact alternatives based on sequential Monte Carlo [29–31] avoid this approximation, but at the cost of running many particle trajectories per observation.

> 💡 **阵营一：Training-free 的近似来源 (Hao 批注)**: DPS 的做法是把测量 loss $\|y-A\hat{x}_0(x_t)\|^2$ 对 $x_t$ 求梯度，通过无条件去噪器反传。问题的**根**在这里：它在 $x_t$ 处评估网络，再加一个梯度修正，但真正需要的是在 pivot $\mu_\star$ 处评估。作者特意点出——即便是 moment-matching 变体（追踪 $p(x_0\mid x_t)$ 的各向异性），也只是在**融合测量之前**的去噪分布上做文章，而精确对象是**融合测量之后**的 $p(x_0\mid x_t,y)$。SMC 类方法（[29-31]）虽渐近精确，但每个观测要跑很多粒子轨迹，代价高。这段是 Section 3.3 "training-free 缺了什么"的伏笔。

Training-based methods sidestep the approximation question by training a new model specifically for the inverse problem, with the measurement y as input. This family includes conditional diffusion models that learn a measurement-conditional score [32–34], bridge-based methods that build a trajectory directly from $y$ to the data [35, 36], and methods that distill a posterior sampler from a pretrained diffusion prior [37, 38]. In all cases, the network is exposed to the raw measurement rather than to the geometry of the exact posterior denoising query, so it must learn the operator dependence end-to-end.

> 💡 **阵营二：Training-based 的结构损失 (Hao 批注)**: 这一派（Palette、conditional diffusion、bridge、distillation）不再近似 score，而是直接训一个吃 $y$ 的新模型。代价是：网络看到的是**原始测量 $y$**，而不是**精确后验去噪查询的几何**（$\mu_\star,\Sigma_\star$），所以必须端到端地从头学"算子如何影响去噪"。EPS 的差异化正在于此——它把算子依赖以**闭式解**的方式喂进去，网络只需学"在各向异性几何下怎么去噪"这一件事。

In contrast, we observe that for linear Gaussian inverse problems the exact posterior score has a closed form, with a simple structural meaning. As Figure 1 illustrates, posterior sampling is still a denoising problem, but with a measurement-aware input and an operator-dependent anisotropic noise covariance. We use this identity to define Exact Posterior Score (EPS), a denoising training objective whose target and loss match those of standard pretraining, with the input replaced by a measurement-dependent pivot. EPS can therefore be trained from scratch or fine-tuned efficiently from a pretrained denoiser. At inference, it runs the underlying backbone's sampler unchanged, with no likelihood gradients, projections, or inner optimization.

> 💡 **本文立场 (Hao 批注)**: 这是全文的 thesis statement。"后验采样仍是去噪问题，只是换了输入和噪声几何"——这句话既是理论贡献（闭式解），也是工程红利（保留预训练结构 → warm-start → 快收敛 → 复用采样器 → 推理零额外梯度）。注意"the input replaced by a measurement-dependent pivot"是唯一的结构改动，其余一切（目标 $x_0$、平方损失、噪声调度、采样器）与标准去噪训练完全一致。

Our contributions are as follows.

• We derive the exact posterior score for linear Gaussian inverse problems under general Gaussian interpolants, and show that posterior sampling reduces to denoising at an operator-dependent shifted pivot under an anisotropic covariance. We also pinpoint where existing approximate-guidance methods deviate from this exact identity.

• We turn the identity into EPS, a denoising training objective and sampling algorithm that preserves the structure of standard pretraining while incorporating the exact posterior geometry. EPS can be trained from scratch or fine-tuned from a pretrained checkpoint, and it uses the underlying backbone's sampler at inference.

• We evaluate EPS on five linear inverse problems across FFHQ and ImageNet, reporting pointwise fidelity, perceptual quality, and distributional calibration metrics, and find consistent improvements over both training-free and training-based baselines at substantially smaller sampling budgets.

> 💡 **三点贡献的证据链定位 (Hao 批注)**: 逐条对应正文：
> 1. **理论**（Theorem 1 + Prop 2 + Section 3.3）：闭式后验 score + 后验 velocity + 精确定位已有方法偏差点（Eq. 14）。
> 2. **方法**（Section 3.4 + Algorithm 1 + Section 3.5）：EPS 损失（Eq. 16）、Prop 3 让各向异性噪声可用各向同性模拟、复用 backbone 采样器。
> 3. **实验**（Section 4 + Appendix D）：五任务两数据集，同时报 fidelity（PSNR/SSIM）、perceptual（LPIPS/FID）、**distributional calibration（CRPS/MMD）**三类指标。第三类是本文相对多数 SR/restoration 论文最"校准友好"的地方，也是我们最该借鉴的评测协议。
