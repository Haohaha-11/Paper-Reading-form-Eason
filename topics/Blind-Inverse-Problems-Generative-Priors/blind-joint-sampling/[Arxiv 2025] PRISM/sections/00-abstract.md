[← 返回 README](../README.md)

# Abstract

## 📌 预览

摘要把 PRISM 定位成"能处理盲逆问题的概率化、鲁棒求解器"。核心卖点有两个：(1) 把一个 **measurement-conditioned diffusion model**（观测 $y$ 作为条件的核先验）塞进一个 **理论上有依据的后验采样框架**；(2) 在盲运动去模糊任务上，图像与模糊核的恢复都优于 SOTA。注意这里已经埋下本课题最关心的两个词——probabilistic（能给不确定性）与 robust（对初始化不敏感）。

---

Diffusion models are now commonly used to solve inverse problems in computational imaging. However, most diffusion-based inverse solvers require complete knowledge of the forward operator to be used. In this work, we introduce a novel probabilistic and robust inverse solver with measurement-conditioned diffusion prior (PRISM) to effectively address blind inverse problems. PRISM offers a technical advancement over current methods by incorporating a powerful measurement-conditioned diffusion model into a theoretically principled posterior sampling scheme. Experiments on blind image deblurring validate the effectiveness of the proposed method, demonstrating the superior performance of PRISM over state-of-the-art baselines in both image and blur kernel recovery.

> 💡 **问题动机** (Hao 批注): 摘要把 baseline 的痛点浓缩成一句 "most diffusion-based inverse solvers require complete knowledge of the forward operator"。也就是说 DPS/DDRM 这类非盲求解器默认 $H_\varphi$ 已知，而现实里 $\varphi$（MRI 灵敏度图、CT 视角、去模糊核）恰恰是未知的——这就是"盲"的来源。PRISM 要同时估计图像 $x$ 与算子参数 $\varphi$。

> 💡 **机制拆解** (Hao 批注): 全文方法可以拆成一句话——"把 measurement-conditioned diffusion prior 放进 theoretically principled posterior sampling scheme"。前半句 measurement-conditioned 指核先验 $\mathsf{D}^\varphi(\cdot;y)$ 吃观测 $y$；后半句 principled scheme 指基于 split Gibbs sampling 的 PnP-DM 框架。两者的结合是本文相对 BlindDPS（无条件核先验）和 GibbsDDRM（简单 Laplace 核先验）的关键差异。

> 💡 **本课题关系** (Hao 批注): 对我们"gauge-aware 联合后验采样与校准"主线，PRISM 是最直接的竞品：它同样联合估计 $(x,\varphi)$，同样以后验采样自居并报告像素级 SD / NLL / 3-SD 覆盖。但摘要只字未提 SBC、CRPS、coverage 曲线这类严格校准检验，也没有联合噪声 $\sigma$ 的估计——这正是我们要正面比较的空档。后续实验节需重点核查它的 UQ 证据到底做到哪一步。

---

## 🔖 Section 总结

### 核心洞察
1. PRISM = **PnP-DM 后验采样框架** 从非盲扩展到盲设定 + **观测条件化的核扩散先验**。
2. 两个 claim 关键词：probabilistic（提供 UQ）与 robust（对初始化鲁棒），全文实验都围绕这两点组织证据。
3. 验证任务单一：FFHQ 上的盲运动去模糊；恢复对象是图像 $x$ 与模糊核 $\varphi$（$\varphi$ 即 gauge 参数）。

### 可追问点
- "theoretically principled" 到底 principled 在哪？→ 见 [02-method](02-method.md) 的 split Gibbs 增广分布。
- "probabilistic" 兑现到什么程度？报告了 NLL / SD / 覆盖，但校准是否严格？→ 见 [03-numerical-validation](03-numerical-validation.md) 的 Uncertainty Quantification 小节。
