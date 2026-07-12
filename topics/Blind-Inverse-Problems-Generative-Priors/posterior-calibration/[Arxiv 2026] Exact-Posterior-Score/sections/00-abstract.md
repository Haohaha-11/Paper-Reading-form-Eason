[← 返回 README](../README.md)

# Abstract

## 📌 预览

这篇论文的核心是一句话：**对于线性高斯逆问题，后验 score 有闭式解**。作者证明后验采样并没有跳出"去噪"的框架——它仍然是一个去噪问题，只不过去噪器要在一个"被观测拉偏的输入"（pivot $\mu_\star$）上、在一个"各方向不等方差的噪声协方差"（$\Sigma_\star$）下工作。把这个恒等式变成训练目标，就得到 EPS（Exact Posterior Score）。

---

Diffusion and flow-based models learn powerful data priors by training a denoiser to reverse Gaussian corruption. To use this prior to solve a linear inverse problem, one needs to sample from the posterior, but the score that the prior provides is the unconditional score, not the posterior score. Existing methods either steer a fixed pretrained denoiser with approximate measurement-matching corrections, or train a conditional restoration model that abandons the denoising structure of the prior. We derive the exact posterior score in closed form for linear Gaussian inverse problems under general Gaussian interpolants, and show that posterior sampling reduces to a denoising problem at an operator-dependent shifted pivot under an anisotropic noise covariance. We turn this identity into Exact Posterior Score (EPS), a denoising training objective that preserves the input/output structure of standard pretraining and can therefore be trained from scratch or fine-tuned from a pretrained denoiser. At inference, EPS uses the same sampler as the underlying backbone, with no likelihood gradients or projections. We evaluate EPS on five linear inverse problems across FFHQ and ImageNet, where it outperforms training-free and training-based baselines on fidelity, perceptual, and distributional metrics, while using roughly an order of magnitude fewer denoiser evaluations than gradient-based posterior samplers.

> 💡 **问题动机 (Hao 批注)**: 摘要把整篇论文的张力讲清楚了。扩散/流模型训练出来的是**无条件 score** $\nabla_{x_t}\log p(x_t)$，而解逆问题需要的是**后验 score** $\nabla_{x_t}\log p(x_t\mid y)$。二者相差一个"measurement-matching score" $\nabla_{x_t}\log p(y\mid x_t)$。已有两大流派各有硬伤：
> - **Training-free（如 DPS）**：冻结预训练去噪器，每步加一个近似的测量匹配修正——问题是这个修正只是近似，逐步累积偏差。
> - **Training-based（如 Palette）**：直接训一个以 $y$ 为条件的新模型——问题是丢掉了预训练去噪器的结构，网络得从头学算子依赖。
>
> EPS 的立场是第三条路：既然线性高斯下**后验 score 有闭式解**，那就不用近似、也不用抛弃去噪结构，而是把"正确的去噪几何"直接写进训练目标。

> 💡 **机制拆解 (Hao 批注)**: 本文最需要记住的两个新变量：
> - **pivot（枢轴）$\mu_\star$**：把当前含噪状态 $x_t$ 和测量 $y$ 做**精度加权贝叶斯融合**后的输入点，去噪器该在这里被 query，而不是在 $x_t$ 处。
> - **各向异性协方差 $\Sigma_\star$**：$\mu_\star$ 上残留噪声不再是各向同性的 $\beta_t^2 I$，而是与算子 $A$ 相关的满秩协方差——**被测量的方向更确定、未观测方向仍不确定**。
>
> EPS 的训练目标 = 标准去噪回归，只是把输入从 $x_t$ 换成 $\mu_\star$。这保证了它能从预训练 checkpoint warm-start，是全文效率优势的根源。

> 💡 **与本课题的关系 (Hao 批注)**: 我们要做"生成先验下参数化盲逆问题的 gauge-aware 联合后验采样与校准"，需要一个**低维可知的参考后验**来做 SBC/coverage/CRPS 检验。EPS 的价值正在于：在**线性高斯 + 已知算子**这个受限结构下，$p(x_0\mid y)$ 的去噪核 $\mathbb{E}[x_0\mid x_t,y]$ 是**精确可写**的（Theorem 1），可以用来构造 ground-truth 参考后验，或至少提供无近似的 baseline。注意本文的算子 $A$、噪声 $\sigma_y$ 都是**已知固定**的——这与我们要联合估计 $\varphi,\sigma$ 的盲设定不同，本文正好给出了"算子已知时后验的真值形态"，是校准实验的构造依据。

> 💡 **关键数字预告 (Hao 批注)**: 五个线性逆问题（70% 随机 inpaint、box inpaint、4× 超分、高斯去模糊、运动去模糊），FFHQ + ImageNet，$\sigma_y=0.05$。核心卖点：全面超过 training-free 与 training-based baseline，且**去噪器评估次数少约一个数量级**（EPS 在 ~20 NFE 收敛，梯度类方法要 100–250 NFE）。
