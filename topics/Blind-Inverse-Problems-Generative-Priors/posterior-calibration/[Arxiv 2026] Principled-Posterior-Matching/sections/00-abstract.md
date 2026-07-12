[← 返回 README](../README.md)

# 0. Abstract

## 📌 预览

摘要抛出全文的核心论断：现有 score-based 反问题方法**近似**地最小化 inversion 分布与 Bayes 后验之间的 KL 散度，这种近似会导致**严重的 mode collapse 与不可靠的不确定性量化**。作者提出 Principled Posterior Matching（PPM），回到变分推断的第一性原理——用 Fisher 散度沿扩散过程的积分来**精确**优化 KL 散度，并推导出一个可计算的等价梯度形式，从而消除以往近似带来的偏差。PPM 同时统一了两种范式：粒子化变分推断（mass-covering，提升多样性与 UQ）与摊还推断（单步重建网络）。

---

# Unbiased Diffusion Variational Inversion via Principled Posterior Matching

Weimin Bai<sup>\*</sup>, Yuxuan Gu<sup>\*</sup>, Yifei Wang, Weijian Luo, He Sun<sup>†</sup> Peking University

Abstract—Existing score-based methods for inverse problems often resort to approximate minimization of the KL divergence between the inversion distribution and the Bayesian posterior. Such an approximation leads to severe mode collapse and unreliable uncertainty quantification. In this paper, we propose Principled Posterior Matching (PPM), a framework that returns to the fundamentals of variational inference, rather than using tricky approximations. Instead of relying on heuristic approximations, we rigorously formulate the exact optimization of the KL divergence via the integration of Fisher divergence. We derive a tractable, equivalent gradient form of this integral, enabling precise optimization without the biases introduced by prior approximations. Our analysis clearly reveals that the mode collapse in previous methods stems directly from this approximation gap. Supported by our theoretical solution, PPM unifies two complementary paradigms: (1) In variational inference, PPM adopts masscovering divergences that significantly improve the inversion diversity and uncertainty quantification; (2) In amortized inference, it enables the training of an efficient reconstruction network for rapid, single-step reconstruction. Furthermore, our formulation naturally extends to a broader family of divergence measures by generalizing the integral of the Fisher divergence. We validate PPM across challenging computational imaging tasks, including inpainting, super-resolution fluorescent microscopy, and radio interferometric black-hole imaging. In all experiments, PPM achieves superior reconstruction fidelity, faithful multimodal posterior recovery, and well-calibrated uncertainty estimates, establishing a robust framework for scientific imaging.

Index Terms—Computational Imaging, Variational Inference, Amortized Inference, Diffusion Models, Uncertainty Quantification

> 💡 **问题动机 (Hao 批注)**: 这段摘要的核心论断正是本课题最关心的"样本离散 ≠ 不确定性已校准"。作者把矛头指向一个被广泛使用但很少被质疑的做法：score-based / VI 类方法**没有**在最小化真正的 $D_{\text{KL}}(q \| p)$，而是在最小化它的某种近似（Dirac 假设、IKL、启发式 repulsion）。这些近似在"逼近后验中心"上表现尚可，却在"覆盖后验支撑集"上系统性失败——于是给出的样本方差既不是真后验方差，也无法用于科学推断。PPM 的补丁不是加正则或加 repulsion，而是换掉优化目标本身：用 Fisher 散度沿扩散时间的积分**等于** KL 散度这一经典恒等式（Song et al. 2021），把不可解的 KL 转成可解的 score-matching。
>
> 💡 **机制预告 (Hao 批注)**: 摘要里三个词需要记住并在方法节验证——(1) *exact optimization of KL via integration of Fisher divergence*：这是理论主张，Fisher 积分 = KL，无近似；(2) *tractable equivalent gradient form*：这是工程主张，Theorem 1 给出绕过对变分 score 求导的等价梯度；(3) *mass-covering*：这是行为主张，精确 KL 会鼓励覆盖全部 mode 而非只抓一个。三者对应 Eq. 13 / Eq. 14 / 与 baseline 的偏差分析（第 IV 节）。
>
> 💡 **与 score-based guidance 的差别 (Hao 批注)**: 摘要把 baseline 分三类——gradient-guided MC（DPS 式 guidance）、optimization VI（RED-Diff/SDS）、amortized（DAVI）。DPS 类的问题是**推断时**用近似 likelihood 引导 reverse SDE（Jensen 不等式违背，见第 IV.B 节）；PPM 的问题定位在**优化目标**层面，它不改 guidance 的 score，而是给出一个无偏的变分目标。对本课题：这解释了为什么单纯给 guidance 加噪声/多次采样得到的"离散样本"不能当作校准后的不确定性——目标本身是有偏的 mode-seeking。
>
> 💡 **本课题关联 (Hao 批注)**: 本文只在已知前向算子 $\mathcal{A}$、已知噪声 $\sigma$ 的**非盲**设定下做 UQ 校准。我们的主线是 gauge-aware 联合后验采样（联合估计 $x$、算子参数 $\varphi$、噪声 $\sigma$）。PPM 的价值在于它给出了"如何让 $q(x|y)$ 的方差真正等于后验方差"的目标级答案——这正是把 SBC/coverage/CRPS 校准检验从"事后诊断"前推到"训练目标"的一块拼图。但要注意它的 UQ 只覆盖 $x$ 的条件后验，未触及 $\varphi,\sigma$ 的联合不确定性。
