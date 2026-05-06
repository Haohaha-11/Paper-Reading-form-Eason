[← 返回 README](../README.md)

# Methodology

## 📌 预览
本节是核心方法，重点看模块输入输出、训练目标、推理路径和与 baseline 的差异。

---

In this section, we first reveal the characteristic of denoising network in one-step diffusion for image super-resolution, and then propose our algorithm.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

# Preliminary: The Character of Denoising Network in One-step Diffusion for Real-ISR

In SD, the latent diffusion process starts with encoding an image into a latent representation $z _ { \mathrm { 0 } }$ using a VAE encoder. The forward diffusion process then adds Gaussian noise to $z _ { \mathrm { 0 } }$ over $T$ steps via a Markov chain defined as:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 1](../images/ee2e21b86e3d1c6b9b870a094e6c535ecb264bf3658bcade62997d693eafc5a5.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

following a variance schedule $\beta _ { 1 } , \ldots , \beta _ { T }$ . Here $z _ { 0 } \sim q ( z _ { 0 } )$ . The forward process is: $z _ { t } ~ = ~ \alpha _ { t } z _ { 0 } + \sigma _ { t } \epsilon$ , where $\begin{array} { r l } { \alpha _ { t } } & { { } = } \end{array}$ $\textstyle \prod _ { s = 1 } ^ { t } { \sqrt { 1 - \beta _ { s } } }$ , $\sigma _ { t } = \sqrt { 1 - \alpha _ { t } ^ { 2 } }$ , and $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathcal { T } )$ . Here, the mean of ${ \boldsymbol { z } } _ { t }$ becomes $\alpha _ { t } z _ { 0 }$ , and the variance is $\sigma _ { t } ^ { 2 } \mathcal { T }$ . With larger $t$ , more noise is added, resulting in a greater deviation of the mean (scaled by $\alpha _ { t } < 1 $ ) and an increased variance $( \sigma _ { t } ^ { 2 } )$ , distancing ${ \boldsymbol { z } } _ { t }$ from the original distribution $z _ { \mathrm { 0 } }$ .

> 💡 **批注**: 这段信息较密，建议拆成“问题/设定 → 方法/机制 → 结果/影响”三层读。

The reverse process denoises ${ \boldsymbol { z } } _ { t }$ to recover $z _ { t - 1 }$

![Equation 2](../images/0b0894b1b2c865962e8fa93c1c859c2e979b21e83055d62edbc8cbffe328bcd6.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where a time-conditional U-Net $\epsilon _ { \theta }$ predicts the noise $\epsilon$ to estimate $\pmb { \mu } _ { \theta }$ and $\Sigma _ { \theta }$ . Multi-step diffusion models iteratively refine $z _ { T }$ back to $z _ { \mathrm { 0 } }$ over $T$ steps, while one-step diffusion methods, often distilled from multi-step models, predict the clean data directly from $z _ { T }$ in a single step, significantly reducing computation.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

For Real-ISR tasks using SD-based OSD methods like $\mathrm { W u }$ et al. 2024a), the process from LR latent features $z _ { L }$ to HR latent features $\hat { z } _ { H }$ is formulated as a one-step denoising process:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 3](../images/513e9ab725ced7424ca51995a67f51feb9c108acb508093bf9cf03dbfc149b9b.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where $z _ { L }$ is the LR latent representation, $c _ { y }$ is the text embedding, and $\epsilon _ { \theta }$ is the denoising network predicting noise at timestep $T$ . As shown in Fig. S2(a), vanilla OSD methods learn a direct latent feature mapping from LR to HR images with or without adding additional noise.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

In Real-ISR, images exhibit diverse degradation types and levels. SD-based methods use powerful generative image priors to recover LR images. However, OSD typically trains on all data with a fixed timestep $T$ with a constant noise level. This results in a model that generates a uniform amount of detail and converges to a confined domain, which may not suit images with varying degradation severity. Tab. S1 (a) demonstrates the results of training OSEDiff on two different degradation pipelines (DP). ‘Orig. $^ +$ New Deg.’ denotes a DP applying more degradations than the standard DP (‘Orig.’). It indicates that higher degradation in training results in a greater emphasis on realism. This exposes OSD’s flaw: by locking training to a single fixed T, the model optimizes for an “average” degradation, yielding a limited generation flexibility that struggles to adapt its output to meet the specific scenario requirements.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

In contrast, multi-step diffusion models provide greater flexibility in SR tasks. By selecting different timesteps $t$ during the inference stage, these models control the degree of noise adding and removing in diffusion, balancing fidelity and realism effectively.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Therefore, to overcome the inherent limitation of OSD, which is incapable of adjusting its generation levels to adapt to varying scenarios, we propose a novel training strategy for real-world SR applications that can flexibly control the generation realism during the inference stage.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Table extracted](../images/42b99593d6abe3cd0b1e32ff72c20b2a2f003f98f2d37f52751b4f01cfa3b994.jpg)
*Table extracted: Table extracted by MinerU. ML L1 MSE Cosine Sim. SSIM ↑ 0.60 0.54 0.78 DISTS ↑ 0.15 0.11 0.42 CLIPIQA ↑ 0.06 0.05 0.27*

> 💡 **Table extracted 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

Table S1: (a) Influence of degradation degree in training data. (b) |Spearman coefficient| $\uparrow$ comparison of different $M _ { L }$ distances and image quality metrics.

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

![Table S1](../images/e097b4b4897697f53304e3c3f656cf665a6c8a69607613ab0dbfba3e168d0e50.jpg)
*Table S1: Table S1: (a) Influence of degradation degree in training data. (b) |Spearman coefficient| $\uparrow$ comparison of different $M _ { L }$ distances and image quality metrics.*

> 💡 **Table S1 批读**: 表格要看主指标、次指标与效率/鲁棒性是否一致支持论文 claim。

![Figure S3](../images/6ad81d2f4173cf0e8ebdaecaac99c96ef7d06de486d3551b07be36b6a40fb02d.jpg)
*Figure S3: Figure S3: Influence of different timesteps $t$ using SD-turbo.*

> 💡 **Figure S3 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

# Latent Domain Grouping

To achieve dynamic fidelity-realism trade-offs control in OSD generated SR results, we focus on the most basic condition in denoising network, i.e, timestep condition. Unlike prompts from a text encoder or a vision encoder, the timestep condition is an unremovable component in the diffusion process. At the same time, it controls the mean and variance of noisy latent feature $z _ { T }$ . With the larger mean and variance difference between $z _ { T }$ and $z _ { \mathrm { 0 } }$ , more contents will be generated during the denoising process. Fig. S3 shows an example of influence of different $t$ . In the foundational model SDturbo, the higher timestep value during the diffusion process usually reflects the the higher capability generating.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Therefore, to easily control the generation degree, we propose a latent domain grouping (LDG) strategy. Recall Eq. (3), we do not use a single fixed timestep $T$ , but choose a timestep $t$ according to a metrics:

![Equation 4](../images/ef6ea326c3da69d69ccda01856e9f6e1c74834cf4749e0757ec764ded760d9d6.jpg)
*Equation 4: Equation extracted by MinerU.*

> 💡 **Equation 4 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where $M _ { L }$ denotes a latent metric that can perceive the “level of degradation” of features in latent domain, $M _ { L \mathrm { - m i n } }$ is the minimum value of $M _ { L }$ in training data, $k$ is interval of timestep, $\lfloor . \rfloor$ denotes the maximum integer no larger than the entry inside, $n$ is number of groups for timestep.

To employ this strategy both on SD and distillation version of SD, i.e, SD-Turbo (SDT), which distilled a four specific steps from the original 1000-step diffusion process, we set $n$ to be $\leq 4$ and $k = 2 5 0$ .

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

By the grouping strategy, denoising network can learn different degrees of generation according to timestep. In the training stage, grouping is based on the $M _ { L }$ described in the next subsection. In the inference stage, we can easily choose a timestep to control the level of realism for SR in different scenarios. Furthermore, due to our grouping strategy, the realism level increases monotonically with the timestep.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

# Latent Metric for Denoising Network

The latent metric $M _ { L }$ is designed to enable the denoising network in the latent domain to perceive the “level of degradation” of the low-resolution features $\mathbf { z } _ { L }$ . Here, we define the “level of degradation” as the extent to which $\mathbf { z } _ { L }$ deviates from its HR counterpart $\mathbf { z } _ { H }$ . Thus, the definition of the latent metric $M _ { L }$ should reflect the characteristics of $\mathbf { z } _ { L }$ that indicate this degradation.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

A simple choice might be to use the difference between the mean values of $\mathbf { z } _ { L }$ and $\mathbf { z } _ { H }$ , as the forward diffusion process scales the mean of $\mathbf { z } _ { L }$ over time. However, as shown in Fig. S4(a), the mean values of $\mathbf { z } _ { L }$ and $\mathbf { z } _ { H }$ are similar across the training data. This suggests that mean difference fails to effectively capture the degradation level.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Instead, we use cosine similarity (CS) as $M _ { L }$ :

![Equation 5](../images/7105658d88fa77201a349eaed6683f5a29bb502be47ee5ba6136fc88134f428e.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

Cosine similarity is a widely adopted measure in representation learning (e.g., contrastive learning (Chen et al. 2020)). It can quantify the divergence between high-dimensional latent features $\mathbf { z } _ { L }$ and $\mathbf { z } _ { H }$ , which can reflect high-level changes caused by degradation. As shown in Fig. S4(b), most cosine similarity values lie between 0 and 1, providing a clear range to distinguish varying degrees of degradation for the denoising network.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Figure S4](../images/c999f6a534984d0afd72462e4febb51e25a9241b73ed00e364bbf929b8a80a53.jpg)
*Figure S4: Figure S4: Distribution of (a) mean value of LR and HR training images in the latent domain, (b) the $M _ { L }$ metric in nc y the latent domain of VAE before and after training.*

> 💡 **Figure S4 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

Value ValueDifferent distances inherently introduce a certain preference or bias, to evaluate the bias of different $M _ { L }$ , we calculate correlation coefficient (|Spearman Corr.| ↑) of CS, L1, and MSE with different metrics in Table S1 (b). The metrics are calculated between LR and HR images. The correlation reflects the degree of association between various $M _ { L }$ and different metrics. CS exhibits a higher correlation coefficient with objective (SSIM), perceptual (DISTS), and semantic (CLIPIQA) metrics compared to other distances in latent space. More visualization results can be seen in Supplementary Materials (SM).

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

# Metric Estimation

Since $\mathbf { z } _ { H }$ is introduced as a latent metric, we can only calculate it during training. In the inference stage, beyond manually choosing a timestep, we also consider an adaptive timestep selection, i.e., estimating an $M _ { L }$ for each LR image. Benefiting from the powerful representation ability of the pre-trained model, we use features from intermediate layers as input and a simple MLP as a metric estimation module (MEM), operating independently of OSD training.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

# Degradation-aware Sampling Distillation

Previous VSD method for SR (Wu et al. 2024a), utilizes the regularization network (a pre-trained SD) that sampled timesteps across a wide range (20-980). This aims to generate regularization latent features and, in turn, optimize the distribution of the OSD network. However, to better integrate the distillation process with the concept of degradation in the latent space, we propose a Degradation-Aware Sampling (DAS) strategy. DAS redefines how timesteps are sampled in the pre-trained model, adaptively aligning this process with our LDG framework to provide explicit control over regularization strength. The DAS can be written as:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 6](../images/97c2bbe0c0683ce17a9756f2d77189c87a92abfa160608b38d8bae7558c33e8a.jpg)
*Equation 6: Equation extracted by MinerU.*

> 💡 **Equation 6 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where $t _ { r }$ is the sample timestep for regularization network, $t$ is the chosen timestep in OSD network by LDG, $S ( t _ { m i n } , t _ { m a x } )$ denotes uniformly random sampling of an integer from the range $[ t _ { m i n } , t _ { m a x } ]$ .

By applying DAS, the degradation grouping information is delivered from LDG, thereby aligning this process with LDG and control over regularization strength.

# Visual Prompt Injection Module

In previous SD-based super-resolution (SR) methods like OSEDiff (Wu et al. 2024a) and SeeSR (Wu et al. 2024b), text encoders, sometimes paired with a vision-language model (VLM) as a text prompt extractor, improve non-reference (NR) metrics, which assess realism, but often constrain fullreference (FR) metrics, which assess fidelity to the ground truth. This trade-off arises because text prompts provide high-level semantic guidance to enhance realism, yet they may compromise structural accuracy.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

LR feature $\mathbf { z } _ { L }$ solely from the VAE encoder offers limited semantic information. Without additional context, the conditioned U-Net struggles to generate high-quality outputs. Text prompts attempt to bridge this gap by injecting external semantic cues, but they come with drawbacks: VLMs increase computational costs, and the prompts may not fully align with the image’s content. Some methods like S3Diff (Zhang et al. 2024) use fixed text to reduce complexity, yet still struggle to balance NR and FR metrics.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

To address these issues, we propose the Visual Prompt Injection Module (VPIM), which replaces conventional text prompts with degradation-aware visual tokens. VPIM substitutes the text encoder (typically a CLIP text model) with a CLIP vision model and an MLP layer for dimension alignment. The LR image serves as its input, i.e, visual prompt, and the output is fed into the cross-attention of the U-Net. By adopting VPIM, we eliminate the need for VLMs, reduce computational overhead, and provide the U-Net with image-specific semantic information directly from the LR input. The visual prompt is tied to the image’s pixel characteristics, leading to improvement in both fidelity and realism.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

With the combination of LDG, latent metric, DAS, and VPIM, we proposed a Realism control one-step diffusion (RCOD) framework, a realism-flexible one-step diffusion model with enhanced performance that can be applied to various recent mainstream one-step diffusion Real-ISR methods, thereby improving their capabilities. We provide a pseudo-code example of our RCOD in SM.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
