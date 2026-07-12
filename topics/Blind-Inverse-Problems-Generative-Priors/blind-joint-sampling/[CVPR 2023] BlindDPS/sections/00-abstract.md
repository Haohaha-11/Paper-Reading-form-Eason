[← 返回 README](../README.md)

# Abstract

## 📌 预览

这篇论文（BlindDPS）要解决的是**盲逆问题**：不仅图像 $x$ 未知，前向算子 $\mathcal{H}$（例如模糊核 $k$、湍流 tilt 场 $\phi$）也未知，需要**联合估计**。核心 idea 一句话：既然 DPS 用图像扩散先验解非盲逆问题，那就**再为算子参数训一个扩散先验**，让图像分支和算子分支各自跑一条反向扩散，中间用同一个观测残差 $\|y - \hat{k}_0 * \hat{x}_0\|$ 的梯度做"串扰"（cross-talk）引导。

> 💡 **本课题定位（Hao 批注）**: 本文是"生成先验下参数化盲逆问题"这条线的**核心基线之一**。它把盲问题拆成"图像分支 + 算子分支"两条并行扩散，是后续所有"联合采样"方法的原型。批读时要盯死三件事：(1) 图像分支和算子分支**各自怎么更新**（score + data-consistency 梯度）；(2) **并行引导的近似在哪**（Theorem 1 的 Jensen 近似把 $p(y|x_t,k_t)$ 换成 $p(y|\hat{x}_0,\hat{k}_0)$）；(3) **联合样本的偏差来源**（两分支独立先验假设 + 尺度歧义 + 步长手调），这正是我们后续要用 SBC/coverage/CRPS 正面比较"联合后验是否校准"的靶点。

---

Diffusion model-based inverse problem solvers have demonstrated state-of-the-art performance in cases where the forward operator is known (i.e. non-blind). However, the applicability of the method to blind inverse problems has yet to be explored. In this work, we show that we can indeed solve a family of blind inverse problems by constructing another diffusion prior for the forward operator. Specifically, parallel reverse diffusion guided by gradients from the intermediate stages enables joint optimization of both the forward operator parameters as well as the image, such that both are jointly estimated at the end of the parallel reverse diffusion procedure. We show the efficacy of our method on two representative tasks — blind deblurring, and imaging through turbulence — and show that our method yields state-of-the-art performance, while also being flexible to be applicable to general blind inverse problems when we know the functional forms.

> 💡 **摘要机制拆解（Hao 批注）**:
> - **输入**：退化观测 $y$（LQ），以及**已知的前向函数形式**（如卷积、tilt-then-blur），但不知道具体参数。
> - **两个扩散先验**：一个是预训练的图像 score $s_{\theta^*}^i$，另一个是**新训练的算子 score**（模糊核 $s_{\theta^*}^k$ / tilt 场 $s_{\theta^*}^t$）。这是本文最关键的贡献——把"算子的先验"也用扩散模型隐式表达，取代传统的 sparsity/dark-channel 等手工先验。
> - **输出**：反向扩散跑到 $t=0$ 时，**同时**得到重建图像 $x_0$ 和估计算子 $k_0$（$\phi_0$）。
> - **注意"functional forms" 这个限定**：本文只是"半盲"——前向的**函数结构**（卷积/tilt）必须已知，只是参数未知。真正的 fully-blind（连函数形式都不知道）作者在 Limitation 里明确说没解决。这一点对我们课题很重要：我们做的正是**低维参数化**（模糊长度/角度、$\sigma$）的联合后验，本文的"参数化盲"设定和我们高度对齐，但它只给点估计、不谈校准。

> 💡 **与 DPS 的一句话差异（Hao 批注）**: DPS 解的是 $\nabla_{x_t}\log p(x_t|y)$（算子固定）；BlindDPS 解的是 $\nabla_{x_t,k_t}\log p(x_t,k_t|y)$（算子也是随机变量）。两者的似然近似完全同源（都靠 Tweedie 去噪估计代入），BlindDPS 只是把这套近似**复制到算子分支**并假设 $x_0 \perp k_0$ 独立。这个"独立先验 + 独立分支"的假设，就是我们后面质疑其联合后验校准的第一个入口。

> 💡 **图位说明（Hao 批注）**: 概念图 Figure 1 按批读规范归入 [01-introduction](01-introduction.md)（Abstract 不放图），那里对 (a)(b)(c) 三个子图有详细批读。

