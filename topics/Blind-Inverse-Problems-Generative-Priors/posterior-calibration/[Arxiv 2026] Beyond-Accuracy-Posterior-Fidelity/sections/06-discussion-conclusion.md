[← 返回 README](../README.md)

# 6. Discussion and Conclusion

## 📌 预览

结论重申两个贡献：命名并研究 **Accuracy Trap**，提出 theory-grounded、ground-truth-free 的 **score-KSD**。诚实地承认一个关键局限：score-KSD 需要噪声尺度 $\sigma_y$，而真实中 $\sigma_y$ 未必可得、需要额外估计——**这个局限恰好是本课题（联合估计 $\sigma$）的入口**。

---

In this work, we identify the critical "Accuracy Trap" in DIS benchmarking, and study their posterior behavior from a distributional perspective. Motivated by that, we propose the theory-grounded and ground-truth-free score-KSD as a posterior-aware diagnostic for evaluating posterior fidelity.

Through controlled simulations, and real-world inverse problems, our results suggest that score-KSD constructed from the likelihood model and learned diffusion prior, provides a practical and meaningful tool for posterior-consistency evaluation when neither true posterior samples nor posterior density are accessible. One limitation of the proposed score-KSD framework is that it requires the noise scale $\sigma_y$ in the inverse problem, which may not be directly accessible in practice and require additional estimation. How inaccuracies in such estimates affect the score-KSD evaluation remains an important direction for future work.

> 💡 **局限批注（对本课题最关键） (Hao 批注)**: 这段的最后一句是本文与本课题（gauge-aware 盲逆问题联合后验采样）的直接接口，必须重点标注：
> - **score-KSD 的阿喀琉斯之踵是 $\sigma_y$**：似然 score $\sigma_y^{-2}\mathcal{A}^\top(y-\mathcal{A}x)$ 显式依赖噪声尺度 $\sigma_y$。$\sigma_y$ 估错，似然 score 幅度整体缩放错误，后验 score field 失真，score-KSD 读数不可信。作者自己把"$\sigma_y$ 估计误差如何影响 score-KSD"列为 future work。
> - **盲设置把这个局限放大成三重**：本文假设 $\mathcal{A}$、$\sigma_y$ 都已知，只评 $x$-后验。我们的项目里 $\mathcal{A}=\mathcal{A}(\phi)$（$\phi$ 未知）、$\sigma$ 未知，还要评 $\phi$、$\sigma$ 各自的后验。也就是说，直接搬 score-KSD 会遇到"用来评价的 score field 本身依赖待估参数"的鸡生蛋问题。
> - **可行的迁移路线**：(1) 把 $(\phi,\sigma)$ 也当作后验变量，用联合后验样本 $(x,\phi,\sigma)$ 边缘化构造 score；(2) 用 SBC/coverage/CRPS 走"重复模拟-秩检验"的另一条正交路线来校准 $\phi,\sigma$，与 score-KSD 在 $x$ 维互补。本文提供的是"$x$ 维、$\sigma$ 已知"的成熟工具与失配敏感性证据，我们要补的正是它没做的 $\phi,\sigma$ 维与 $\sigma$ 估计误差分析。

## Acknowledgment

Guanyang Wang acknowledges support from the National Science Foundation through grant DMS–2210849 and an Adobe Data Science Research Award. Liyue Shen acknowledges funding support by National Science Foundation (NSF) via grant IIS-2435746, Defense Advanced Research Projects Agency (DARPA) under contract No. HR00112520042, as well as the University of Michigan MIDAS PODS Grant Award.

## References

> 💡 **参考文献导读 (Hao 批注)**: 引用结构（S2 收录 52 篇）分三类，是理解本文技术血统的地图：
> - **DIS 算法（被评价对象）**：DPS [5]、DAPS [47]、DDRM [23]、DDNM [41]、DiffPIR [50]、FPS [10]、MCG-Diff [2]、PnPDM [43]、RED-Diff [31,36]——覆盖梯度/投影/采样/优化四类机制。
> - **评价工具血统**：KSD 理论来自 Liu-Lee-Jordan [29] 与 Gorham-Mackey [13,14]；基准参照 InverseBench [49]、统计基准 [45]。
> - **不确定性理论**：AU/EU 分解 [18,24,33]，贝叶斯逆问题 [21,39]。

[1] Samuel G Armato, et al. The lung image database consortium (LIDC) and image database resource initiative (IDRI). Medical Physics, 38:915–931, 2011.

[2] Gabriel Cardoso, et al. Monte carlo guided diffusion for bayesian linear inverse problems. arXiv:2308.07983, 2023.

[3] Matthew Chan, et al. Estimating epistemic and aleatoric uncertainty with a single model. NeurIPS, 37:109845–109870, 2024.

[4] Haoxuan Chen, et al. Solving inverse problems via diffusion-based priors: An approximation-free ensemble sampling approach. arXiv:2506.03979, 2025.

[5] Hyungjin Chung, et al. Diffusion posterior sampling for general noisy inverse problems. arXiv:2209.14687, 2022.

[6] Hyungjin Chung and Jong Chul Ye. Score-based diffusion models for accelerated mri. Medical Image Analysis, 102479, 2022.

[7] Florentin Coeurdoux, et al. Plug-and-play split gibbs sampler. arXiv:2304.11134, 2023.

[8] I Craig and J Brown. Inverse problems in astronomy. Adam Hilger Ltd., 1985.

[9] Giannis Daras, et al. A survey on diffusion models for inverse problems. arXiv:2410.00083, 2024.

[10] Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A filtering perspective. ICLR, 2024.

[11] Vineet Edupuganti, et al. Uncertainty quantification in deep mri reconstruction. arXiv:1901.11228, 2020.

[12] Wenbo Gong, et al. Sliced kernelized stein discrepancy. arXiv:2006.16531, 2021.

[13] Jackson Gorham and Lester Mackey. Measuring sample quality with kernels. ICML, 1292–1301, 2017.

[14] Jackson Gorham and Lester Mackey. Measuring sample quality with stein's method. arXiv:1506.03039, 2019.

[15] Martin Heusel, et al. GANs trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS 30, 2017.

[16] Jonathan Ho, et al. Denoising diffusion probabilistic models. arXiv:2006.11239, 2020.

[17] Paul Hofman, et al. Quantifying aleatoric and epistemic uncertainty with proper scoring rules. arXiv:2404.12215, 2024.

[18] Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning. Machine Learning, 110(3):457–506, 2021.

[19] Ajil Jalal, et al. Robust compressed sensing mri with deep generative priors. arXiv:2108.01368, 2021.

[20] Zahra Kadkhodaie and Eero Simoncelli. Stochastic solutions for linear inverse problems using the prior implicit in a denoiser. NeurIPS 34, 13242–13254, 2021.

[21] Jari P Kaipio and Erkki Somersalo. Statistical and computational inverse problems. Springer, 2005.

[22] Tero Karras, et al. Elucidating the design space of diffusion-based generative models. NeurIPS, 2022.

[23] Bahjat Kawar, et al. Denoising diffusion restoration models. NeurIPS, 2022.

[24] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? NeurIPS 30, 2017.

[25] Namhoon Kim and Sara Fridovich-Keil. Towards distribution-shift uncertainty estimation for inverse problems with generative priors. arXiv:2510.10947, 2025.

[26] Preetam Kumar, et al. Multi-solution inverse design in photonics using generative modeling. JOSA B, 41(2):A152–A160, 2024.

[27] Jean-Marie Lemercier, et al. Diffusion models for audio restoration: A review. IEEE Signal Processing Magazine, 41(6):72–84, 2025.

[28] P. Li, et al. A large-scale ct and pet/ct dataset for lung cancer diagnosis (lung-pet-ct-dx), 2020.

[29] Qiang Liu, Jason D. Lee, and Michael I. Jordan. A kernelized stein discrepancy for goodness-of-fit tests and model evaluation. arXiv:1602.03253, 2016.

[30] Guanxiong Luo, et al. Bayesian mri reconstruction with joint uncertainty estimation using diffusion models. Magnetic Resonance in Medicine, 90(1):295–311, 2023.

[31] Morteza Mardani, et al. A variational perspective on solving inverse problems with diffusion models. arXiv:2305.04391, 2023.

[32] Eloi Moliner, et al. Solving audio inverse problems with a diffusion model. ICASSP, 1–5, 2023.

[33] Joseph B Nagel and Bruno Sudret. A unified framework for multilevel uncertainty quantification in bayesian inverse problems. Probabilistic Engineering Mechanics, 43:68–84, 2016.

[34] Bowen Song, et al. Solving inverse problems with latent diffusion models via hard data consistency. arXiv:2307.08123, 2024.

[35] Jiaming Song, et al. Denoising diffusion implicit models. arXiv:2010.02502, 2022.

[36] Jiaming Song, et al. Pseudoinverse-guided diffusion models for inverse problems. ICLR, 2023.

[37] Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. arXiv:2111.08005, 2022.

[38] Yang Song, et al. Score-based generative modeling through stochastic differential equations. arXiv:2011.13456, 2021.

[39] Andrew M Stuart. Inverse problems: a bayesian perspective. Acta numerica, 19:451–559, 2010.

[40] J. Virieux and S. Operto. An overview of full-waveform inversion in exploration geophysics. Geophysics, 74(6):WCC1–WCC26, 2009.

[41] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. ICLR, 2023.

[42] David Wiesner, et al. Cytopacq: a web-interface for simulating multi-dimensional cell imaging. Bioinformatics, 35(21):4531–4533, 2019.

[43] Zihui Wu, et al. Principled probabilistic imaging using diffusion models as plug-and-play priors. NeurIPS, 2024.

[44] Carl Wunsch. The Ocean Circulation Inverse Problem. Cambridge University Press, 1996.

[45] Martin Zach, Youssef Haouchat, and Michael Unser. A statistical benchmark for diffusion posterior sampling algorithms. arXiv:2509.12821, 2025.

[46] Jure Zbontar, et al. fastmri: An open dataset and benchmarks for accelerated mri. arXiv:1811.08839, 2019.

[47] Bingliang Zhang, et al. Improving diffusion inverse problem solving with decoupled noise annealing. arXiv:2407.01521, 2024.

[48] Richard Zhang, et al. The unreasonable effectiveness of deep features as a perceptual metric. CVPR, 586–595, 2018.

[49] Hongkai Zheng, et al. Inversebench: Benchmarking plug-and-play diffusion priors for inverse problems in physical sciences. ICLR, 2025.

[50] Yuanzhi Zhu, et al. Denoising diffusion models for plug-and-play image restoration. CVPRW (NTIRE), 2023.

> 💡 **6 小结 (Hao 批注)**:
> - **核心洞察**：score-KSD 是"$\sigma_y$ 已知、$\mathcal{A}$ 已知、评 $x$-后验"三前提下成熟的 ground-truth-free 诊断；它的公开局限（依赖 $\sigma_y$、$\sigma_y$ 估计误差影响未知）正是本课题联合估计 $\sigma$ 的动机与切入口。
> - **可复用点**：把 score-KSD 当作我们盲设置校准评价的 $x$-维基准工具；对 $\phi,\sigma$ 维改用 SBC/coverage/CRPS 正交检验；并把"$\sigma_y$ 估计误差 → score-KSD 失真"作为一个待补的敏感性实验。
