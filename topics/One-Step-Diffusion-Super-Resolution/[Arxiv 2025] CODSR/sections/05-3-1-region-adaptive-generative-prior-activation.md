[← 返回 README](../README.md)

# 3.1. Region-adaptive generative prior activation

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 CODSR 主线的关系**: CODSR 从 LQ 信息损失、区域差异化先验激活和文本-区域错配三方面补强 controllable one-step diffusion SR，在单步推理下兼顾感知质量与局部保真。

---

Different from existing one-step methods [13, 40, 50, 55] that directly feed the LQ latent into the denoising network, we introduce the deliberate addition of Gaussian noise to the LQ latent before denoising, in order to better unleash the generative priors of diffusion models for producing visually realistic outputs. To further achieve region-aware activation of diffusion priors, we propose a region-adaptive generative prior activation (RGPA) method that distinguishes between flat and textured regions via the Sobel gradient map and accordingly constructs spatially adaptive noise for the forward process.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Specifically, we first encode the LQ image $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ into the latent features $z _ { L }$ using a VAE encoder. Instead of directly denoising, we construct the noisy latent ${ \boldsymbol { z } } _ { t }$ by adding the adaptive noise $\epsilon _ { a }$ in a one-step forward process:

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Equation 1](../images/b8a82ad475e76aa14634899dfd98e74503996abcafe2c184ea4b416089b880a2.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $\bar { \alpha } _ { t }$ is the cumulative parameter controlling the noise level at timestep $t$ . During the reverse process, the clean latent $\hat { z } _ { H }$ is recovered using the noise $\boldsymbol { \epsilon } _ { \theta } ( z _ { t } , \boldsymbol { c } , t )$ predicted

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

by the U-Net, conditioned on the text embedding $^ c$

![Equation 2](../images/13d576c1deb0aea64a145308fe0514bbb2833fb1d9f3bce7f0d6683f49230860.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $\begin{array} { r } { w _ { t } = \frac { \sqrt { 1 - \bar { \alpha } _ { t } } } { \sqrt { \bar { \alpha } _ { t } } } } \end{array}$ is a time-dependent noise coefficient.

The adaptive noise $\epsilon _ { a }$ is derived by a gradient-weighted process. First, the Sobel operator is applied to compute the gradient map of $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ , denoted as ${ \mathbf { \mathit { g } } } _ { { \mathit { x } } _ { L } }$ The noise weighting coefficients are then obtained by a mapping operator $W ( \cdot )$ , which performs $1 6 \times 1 6$ patch-wise averaging followed by a piecewise transformation similar to UPSR [63]. The final adaptive noise $\epsilon _ { a }$ is thus formulated as:

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

![Equation 3](../images/9e45d1883970609d86745c8ddab2722d83beb294f73b7a1fe714805c99817023.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $\epsilon \sim \mathcal { N } ( 0 , I )$ . Additionally, to enhance the robustness of the model, a timestep $t _ { s } \in [ t _ { \operatorname* { m i n } } , t _ { \operatorname* { m a x } } ]$ is randomly sampled during training, encouraging the model to generalize across various noise levels. During inference, this design provides flexible control over the trade-off between fidelity and generative quality simply by adjusting the timestep.

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
| Core controls | region-adaptive prior activation + text-matching guidance |
| Project | https://github.com/Chanson94/CODSR |
| Benchmarks | RealSR/DrealSR and other real-world SR test sets in paper tables |
