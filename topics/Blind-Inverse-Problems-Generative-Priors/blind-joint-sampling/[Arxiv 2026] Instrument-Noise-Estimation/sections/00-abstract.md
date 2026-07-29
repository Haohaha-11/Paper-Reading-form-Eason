[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

这篇 letter 要解决的问题：**在扩散先验（diffusion prior）下的贝叶斯逆问题里，不仅要恢复图像 $x_0$，还要同时估计"观测系统的参数" $\theta$ —— 仪器响应参数 $\iota$（PSF 宽度）与噪声参数（偏置 $m_e$、方差 $v_e$）**。作者把这称作 **Hyper-G-DPS**（Hyperparameter-G-DPS），是他自己 G-DPS 采样器 [1] 的扩展。

---

Abstract — This article addresses the issue of estimating observation parameters (response and error parameters) in inverse problems. The focus is on cases where regularization is introduced in a Bayesian framework and the prior is modeled by a diffusion process. In this context, the issue of posterior sampling is known to be thorny, and a recent paper [1] proposes a notably simple and effective solution. Additionally, it opens an remarkable flexibility when it comes to estimating observation parameters. The proposed strategy enables to define an optimal estimator for both observation parameters and image of interest. Furthermore, the strategy provides a means for uncertainty quantification. In addition, MCMC algorithms allow for the computation of estimates and properties of posteriors, while offering some guarantees. The paper presents several numerical experiments that clearly confirm the computational efficiency and the quality of both estimates and uncertainty quantification.

> 💡 **问题动机**（Hao 批注）：本课题（生成先验下的盲逆问题）的核心张力是——**算子未知时，能否得到一个"真·联合贝叶斯后验"，而不是一堆看似合理的样本**。这篇文章正是把"未知算子/噪声参数"显式纳入后验：未知量 = 图像 $x_0$ + 观测参数 $\theta=[\iota,\eta]$。它的卖点不是恢复精度更高，而是**给出联合后验的 MCMC 采样 + 不确定性量化（UQ）**，且声称有收敛"保证"（Gibbs 链平稳分布 = 目标后验）。这正是本课题最看重的"校准"维度。

> 💡 **机制拆解**（Hao 批注）：为什么"扩散先验下估参数"很难？摘要点出——主流盲逆方法（如 DPS/ΠGDM，见 [18][19]）从**祖先采样（ancestral sampling，为先验设计）**改造而来，用近似的似然 score 去纠偏；这些近似**把算子 $H_\iota$ 塞进近似式内部**，一旦 $\iota$ 未知就纠缠不清、难以估计。作者的 G-DPS 走另一条路：把扩散链 $x_{0:T}$ 当作隐变量做 **block-Gibbs**，利用 Markov 结构与条件独立（Fig. 1），使得"再加一个参数块 $\theta$"变得几乎免费——这就是本文能做而别人难做的原因。

> 💡 **与竞品的定位**（Hao 批注）：和本课题已读的 [GibbsDDRM](../%5BICML%202023%5D%20GibbsDDRM/)（[17]）最像——都是 Gibbs、都估算子。但作者在 Remark 1 里划清两点差异：(1) GibbsDDRM 估仪器参数但**不估噪声参数**（偏置/功率都不估）；(2) GibbsDDRM 的 Gibbs 只在"图像↔仪器参数"间交替，**不在扩散隐变量 $x_{1:T}$ 之间交替**。本文两者都做。UQ 的直接竞品是 [PRISM](../%5BArxiv%202025%5D%20PRISM/)，但 PRISM 报的是像素级 SD/覆盖，低噪声下过自信；本文用 MCMC + 共轭，宣称 ±2 PSD 区间对参数与像素都能覆盖真值。

> 💡 **Section 概览**（Hao 批注）：全文只有 6 页，结构极简：§II 建似然/先验/后验，§III 给 Gibbs 采样器（图像块用 G-DPS，噪声参数用共轭直采，仪器参数用 MH），§IV 在 MNIST 32×32 toy 上验证，§V 结论。核心贡献全在 §III 的"conjugacy + Gibbs 分块"上。

---

Index Terms — Inverse problem, deconvolution, Bayesian, Hyperparameter estimation, Diffusion prior, Gibbs sampler.

> 💡 **关键词批读**（Hao 批注）："deconvolution" 点明实验里的 $H_\iota$ 是卷积（Lorentz PSF 去卷积）；"Hyperparameter estimation" 指把 $\theta$ 当超参一并估；"Gibbs sampler" 是全文技术骨架。注意它没有出现 "SBC/coverage/CRPS" 这类严格校准词——这也是本课题追问它的地方：它的"覆盖"只验到了 ±2 PSD 落点，尚未做 rank-histogram 级别的模型内校准。
