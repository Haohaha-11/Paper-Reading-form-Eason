[← 返回 README](../README.md)

# Latent Metric for Denoising Network

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

The latent metric $M _ { L }$ is designed to enable the denoising network in the latent domain to perceive the “level of degradation” of the low-resolution features $\mathbf { z } _ { L }$ . Here, we define the “level of degradation” as the extent to which $\mathbf { z } _ { L }$ deviates from its HR counterpart $\mathbf { z } _ { H }$ . Thus, the definition of the latent metric $M _ { L }$ should reflect the characteristics of $\mathbf { z } _ { L }$ that indicate this degradation.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

A simple choice might be to use the difference between the mean values of $\mathbf { z } _ { L }$ and $\mathbf { z } _ { H }$ , as the forward diffusion process scales the mean of $\mathbf { z } _ { L }$ over time. However, as shown in Fig. S4(a), the mean values of $\mathbf { z } _ { L }$ and $\mathbf { z } _ { H }$ are similar across the training data. This suggests that mean difference fails to effectively capture the degradation level.

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

Instead, we use cosine similarity (CS) as $M _ { L }$ :

![Equation 5](../images/7105658d88fa77201a349eaed6683f5a29bb502be47ee5ba6136fc88134f428e.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

Cosine similarity is a widely adopted measure in representation learning (e.g., contrastive learning (Chen et al. 2020)). It can quantify the divergence between high-dimensional latent features $\mathbf { z } _ { L }$ and $\mathbf { z } _ { H }$ , which can reflect high-level changes caused by degradation. As shown in Fig. S4(b), most cosine similarity values lie between 0 and 1, providing a clear range to distinguish varying degrees of degradation for the denoising network.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Figure S4](../images/c999f6a534984d0afd72462e4febb51e25a9241b73ed00e364bbf929b8a80a53.jpg)
*Figure S4: Figure S4: Distribution of (a) mean value of LR and HR training images in the latent domain, (b) the $M _ { L }$ metric in nc y the latent domain of VAE before and after training.*

> 💡 **Figure S4 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

Value ValueDifferent distances inherently introduce a certain preference or bias, to evaluate the bias of different $M _ { L }$ , we calculate correlation coefficient (|Spearman Corr.| ↑) of CS, L1, and MSE with different metrics in Table S1 (b). The metrics are calculated between LR and HR images. The correlation reflects the degree of association between various $M _ { L }$ and different metrics. CS exhibits a higher correlation coefficient with objective (SSIM), perceptual (DISTS), and semantic (CLIPIQA) metrics compared to other distances in latent space. More visualization results can be seen in Supplementary Materials (SM).

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |
