[← 返回 README](../README.md)

# Preliminary: The Character of Denoising Network in One-step Diffusion for Real-ISR

## 📌 预览
Background 提供 diffusion/flow/trade-off 的数学基础，重点理解后面 loss 和 trajectory 约束为什么成立。

---

In SD, the latent diffusion process starts with encoding an image into a latent representation $z _ { \mathrm { 0 } }$ using a VAE encoder. The forward diffusion process then adds Gaussian noise to $z _ { \mathrm { 0 } }$ over $T$ steps via a Markov chain defined as:

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Equation 1](../images/ee2e21b86e3d1c6b9b870a094e6c535ecb264bf3658bcade62997d693eafc5a5.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

following a variance schedule $\beta _ { 1 } , \ldots , \beta _ { T }$ . Here $z _ { 0 } \sim q ( z _ { 0 } )$ . The forward process is: $z _ { t } ~ = ~ \alpha _ { t } z _ { 0 } + \sigma _ { t } \epsilon$ , where $\begin{array} { r l } { \alpha _ { t } } & { { } = } \end{array}$ $\textstyle \prod _ { s = 1 } ^ { t } { \sqrt { 1 - \beta _ { s } } }$ , $\sigma _ { t } = \sqrt { 1 - \alpha _ { t } ^ { 2 } }$ , and $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathcal { T } )$ . Here, the mean of ${ \boldsymbol { z } } _ { t }$ becomes $\alpha _ { t } z _ { 0 }$ , and the variance is $\sigma _ { t } ^ { 2 } \mathcal { T }$ . With larger $t$ , more noise is added, resulting in a greater deviation of the mean (scaled by $\alpha _ { t } < 1 $ ) and an increased variance $( \sigma _ { t } ^ { 2 } )$ , distancing ${ \boldsymbol { z } } _ { t }$ from the original distribution $z _ { \mathrm { 0 } }$ .

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会直接影响生成细节与语义一致性。

The reverse process denoises ${ \boldsymbol { z } } _ { t }$ to recover $z _ { t - 1 }$

![Equation 2](../images/0b0894b1b2c865962e8fa93c1c859c2e979b21e83055d62edbc8cbffe328bcd6.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where a time-conditional U-Net $\epsilon _ { \theta }$ predicts the noise $\epsilon$ to estimate $\pmb { \mu } _ { \theta }$ and $\Sigma _ { \theta }$ . Multi-step diffusion models iteratively refine $z _ { T }$ back to $z _ { \mathrm { 0 } }$ over $T$ steps, while one-step diffusion methods, often distilled from multi-step models, predict the clean data directly from $z _ { T }$ in a single step, significantly reducing computation.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

For Real-ISR tasks using SD-based OSD methods like $\mathrm { W u }$ et al. 2024a), the process from LR latent features $z _ { L }$ to HR latent features $\hat { z } _ { H }$ is formulated as a one-step denoising process:

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Equation 3](../images/513e9ab725ced7424ca51995a67f51feb9c108acb508093bf9cf03dbfc149b9b.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $z _ { L }$ is the LR latent representation, $c _ { y }$ is the text embedding, and $\epsilon _ { \theta }$ is the denoising network predicting noise at timestep $T$ . As shown in Fig. S2(a), vanilla OSD methods learn a direct latent feature mapping from LR to HR images with or without adding additional noise.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

In Real-ISR, images exhibit diverse degradation types and levels. SD-based methods use powerful generative image priors to recover LR images. However, OSD typically trains on all data with a fixed timestep $T$ with a constant noise level. This results in a model that generates a uniform amount of detail and converges to a confined domain, which may not suit images with varying degradation severity. Tab. S1 (a) demonstrates the results of training OSEDiff on two different degradation pipelines (DP). ‘Orig. $^ +$ New Deg.’ denotes a DP applying more degradations than the standard DP (‘Orig.’). It indicates that higher degradation in training results in a greater emphasis on realism. This exposes OSD’s flaw: by locking training to a single fixed T, the model optimizes for an “average” degradation, yielding a limited generation flexibility that struggles to adapt its output to meet the specific scenario requirements.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

In contrast, multi-step diffusion models provide greater flexibility in SR tasks. By selecting different timesteps $t$ during the inference stage, these models control the degree of noise adding and removing in diffusion, balancing fidelity and realism effectively.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

Therefore, to overcome the inherent limitation of OSD, which is incapable of adjusting its generation levels to adapt to varying scenarios, we propose a novel training strategy for real-world SR applications that can flexibly control the generation realism during the inference stage.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

![Table extracted](../images/42b99593d6abe3cd0b1e32ff72c20b2a2f003f98f2d37f52751b4f01cfa3b994.jpg)
*Table extracted: Table extracted by MinerU. ML L1 MSE Cosine Sim. SSIM ↑ 0.60 0.54 0.78 DISTS ↑ 0.15 0.11 0.42 CLIPIQA ↑ 0.06 0.05 0.27*

> 💡 **Table extracted 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

Table S1: (a) Influence of degradation degree in training data. (b) |Spearman coefficient| $\uparrow$ comparison of different $M _ { L }$ distances and image quality metrics.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

![Table S1](../images/e097b4b4897697f53304e3c3f656cf665a6c8a69607613ab0dbfba3e168d0e50.jpg)
*Table S1: Table S1: (a) Influence of degradation degree in training data. (b) |Spearman coefficient| $\uparrow$ comparison of different $M _ { L }$ distances and image quality metrics.*

> 💡 **Table S1 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

![Figure S3](../images/6ad81d2f4173cf0e8ebdaecaac99c96ef7d06de486d3551b07be36b6a40fb02d.jpg)
*Figure S3: Figure S3: Influence of different timesteps $t$ using SD-turbo.*

> 💡 **Figure S3 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

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
