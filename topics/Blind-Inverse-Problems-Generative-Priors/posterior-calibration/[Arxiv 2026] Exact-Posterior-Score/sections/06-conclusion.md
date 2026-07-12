[← 返回 README](../README.md)

# 6 Conclusion

## 📌 预览

总结全文的一条主线（闭式后验 score → EPS 训练目标 → 复用采样器 + 一步后验均值），并诚实列出两个关键局限：线性算子 + 高斯噪声的假设、以及 latent diffusion 下 decoder 把像素域线性算子变成 latent 域非线性算子的问题。

---

We derived the exact posterior score for linear Gaussian inverse problems and showed that posterior sampling reduces to a denoising problem at a measurement-induced pivot $\mu_\star$ under an operator-dependent anisotropic covariance $\Sigma_\star$. We turned this identity into EPS, a denoising training objective whose input/output structure matches standard pretraining, and which can therefore be either trained from scratch or fine-tuned efficiently from a pretrained checkpoint. EPS samples with the underlying backbone's sampler, requires no measurement gradients or projections, and admits a one-evaluation posterior-mean estimator in the high-noise limit. Empirically, EPS improves both reconstruction fidelity and distributional calibration over sampling-based and conditional-training baselines on five linear inverse problems on FFHQ and ImageNet, while exposing an explicit sampling-budget trade-off through the same sampler used by the backbone.

> 💡 **总结批读：一句话复述全文 (Hao 批注)**: "线性高斯后验 score 有闭式 → 后验采样 = 在 pivot $\mu_\star$、各向异性 $\Sigma_\star$ 下去噪 → 把它写成保留预训练结构的去噪损失 EPS → 复用采样器、无梯度、高噪声极限一步出后验均值 → 五任务两数据集全面超越 training-free 与 training-based 基线，fidelity 和分布校准双赢。"要记住的贡献层级：**理论恒等式（Theorem 1）是根，工程红利（warm-start、复用采样器、快收敛、CRPS 校准好）都是它的推论。**

Limitations. The exact derivation assumes a linear forward operator and Gaussian observation noise. Nonlinear operators can be approached by local linearization or by training against the true likelihood, but the closed form of Theorem 1 no longer applies directly. Pixel-space inverse problems with latent diffusion backbones also require care because the decoder makes a linear pixel-space operator nonlinear in latent space.

> 💡 **局限批读：对本课题最重要的边界 (Hao 批注)**: 两个局限直接决定了 EPS 能否用作我们的参考后验构造工具：
> 1. **线性 + 高斯假设**：Theorem 1 的闭式只在线性算子 + 高斯噪声下成立。非线性算子只能局部线性化或对真似然训练，闭式失效。**对我们的盲逆问题**：如果 $A(\varphi)$ 对 $x$ 仍线性（只是参数 $\varphi$ 未知），可以对**固定 $\varphi$** 用 Theorem 1 得到 $p(x_0\mid y,\varphi)$ 的精确去噪核；但要对 $\varphi$ 联合估计/边缘化，就得在 $\varphi$ 上再套一层，闭式一般不再成立。这正好说明 EPS 适合构造"$\varphi$ 已知时的条件参考后验"，作为盲设定校准的 gold-standard 分母。
> 2. **latent diffusion 的非线性化**：像素域线性算子经过 decoder 在 latent 域变非线性——所以本文全在像素空间 EDM 上做。若我们用 latent 生成先验，这个坑同样要绕。
>
> 综合看：EPS 给我们的是"**低维 / 像素域 + 已知线性算子**下真后验去噪核可精确写出"的构造依据，是参考后验实验最干净的落脚点；一旦进入非线性算子或盲 $\varphi$，需要额外近似，闭式优势消失。

> 💡 **Q&A 批注记录 (Hao 批注)**:
> - Q: EPS 能直接给出"真后验"用于校准吗？
> - A: 部分能。Theorem 1 保证 $D_{\Sigma_\star}(\mu_\star)=\mathbb{E}[x_0\mid x_t,y]$ 是**精确**的后验去噪均值，采样路径无近似偏差（不像 DPS）。但去噪器本身仍是学出来的网络，只有在"数据先验也可解析"（如高斯先验）时才是完全解析的真后验。所以最干净的参考后验用法是：**低维 + 高斯/可解析先验 + 已知线性算子** → Theorem 1 直接给闭式后验；一般数据先验下 EPS 提供的是"采样过程无近似"的强 baseline。
> - Q: 为什么 EPS 对校准（CRPS/MMD）比 DPS 好这么多？
> - A: 因为 DPS 每步的 measurement-matching score 是有偏近似（Eq. 14 在错点 $x_t$ 用错分布 $p(x_0\mid x_t)$），偏差污染整条采样链 → 后验样本分布失真；EPS 的采样链无此偏差。见 Section 3.3 + Figure 3 的 CRPS 曲线。
