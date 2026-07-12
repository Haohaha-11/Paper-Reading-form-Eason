[← 返回 README](../README.md)

# Abstract

## 📌 预览

摘要一句话交代全篇：把非盲的 DDRM 扩展到"算子未知"的盲逆问题。做法是构造数据 $\mathbf{x}_0$、测量 $\mathbf{y}$、线性算子参数 $\varphi$ 的联合分布，然后用**部分塌缩 Gibbs 采样（PCGS）**从后验里交替采样图像与算子。卖点是 problem-agnostic（预训练扩散先验不需要为每个任务微调）+ 算子只需要一个通用简单先验，就能在盲去模糊和人声去混响两个任务上打赢基线。

---

Pre-trained diffusion models have been successfully used as priors in a variety of linear inverse problems, where the goal is to reconstruct a signal from noisy linear measurements. However, existing approaches require knowledge of the linear operator. In this paper, we propose GibbsDDRM, an extension of Denoising Diffusion Restoration Models (DDRM) to a blind setting in which the linear measurement operator is unknown. Gibbs-DDRM constructs a joint distribution of the data, measurements, and linear operator by using a pretrained diffusion model for the data prior, and it solves the problem by posterior sampling with an efficient variant of a Gibbs sampler. The proposed method is problem-agnostic, meaning that a pretrained diffusion model can be applied to various inverse problems without fine-tuning. In experiments, it achieved high performance on both blind image deblurring and vocal dereverberation tasks, despite the use of simple generic priors for the underlying linear operators.

> 💡 **问题动机 (Hao 批注)**: 摘要点出了本文与前作 DDRM 的关键差异——DDRM 需要**已知**线性算子 $\mathbf{H}$（比如已知模糊核），而现实里大量任务是"盲"的：模糊核未知、房间冲激响应未知。BlindDPS（同期 CVPR 2023）虽然也能盲，但它要为**算子**额外训练一个扩散先验，这在实践中很难落地（谁有大量模糊核数据集去训一个 score 网络？）。GibbsDDRM 的核心主张是：算子那一侧只用**通用简单先验**（比如 Laplace 稀疏先验），把重活全交给数据侧的预训练扩散模型 + 联合后验采样。

> 💡 **机制拆解 (Hao 批注)**: 摘要里三个词要抓住。（1）**joint distribution** $p(\mathbf{x}_0,\mathbf{y},\varphi)$——本文不是点估计 $\varphi$ 再恢复 $\mathbf{x}_0$，而是构造完整联合分布做贝叶斯后验采样。（2）**posterior sampling**——目标是从 $p(\mathbf{x}_0,\varphi\mid\mathbf{y})$ 采样，而非最大化。（3）**efficient variant of Gibbs sampler**——即 PCGS，用"部分塌缩"把朴素 Gibbs 里那个"每采一次 $\varphi$ 就要跑完整条 DDRM"的低效结构，改成在 DDRM 单个时间步内部交替更新 $\mathbf{x}_t$ 和 $\varphi$，且不改变平稳分布。这三点正是本课题（gauge-aware 联合后验采样）最直接的对照对象。

> 💡 **与本课题关系 (Hao 批注)**: 本文是我们"生成先验下参数化盲逆问题"主线的**最近核心基线**。它已经在做联合估计 $\mathbf{x}_0$ 与低维算子参数 $\varphi$（模糊核 / 声学传递函数），并用 MCMC（PCGS）逼近联合贝叶斯后验。相较之下，本课题额外强调：噪声 $\sigma$ 的联合估计、gauge/规范不变性处理、以及用 SBC / coverage / CRPS 对后验做**校准检验**。GibbsDDRM 只报点指标（PSNR/LPIPS/FID），并未检验后验是否被正确采样（附录 D.3 只对比了 Langevin vs MAP 的稳定性直方图），这正是我们可以补的缺口。
