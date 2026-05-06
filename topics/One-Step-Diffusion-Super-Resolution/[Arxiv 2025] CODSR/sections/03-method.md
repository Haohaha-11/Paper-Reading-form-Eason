[← 返回 README](../README.md)

# 3. Proposed Method

## 📌 预览
本节按数据流读：LQ image 先经 VAE 得到 z_L，同时保留 pixel-unshuffle LQ feature；RGPA 把 z_L 变成区域加权 noisy latent；U-Net denoise 时由 LQFM 补结构、TMG 约束文本区域；训练上先用重建/GAN 稳住，再用 VSD 蒸馏语义生成能力。
---

> 💡 **Q&A 批注记录**:
> - Q: 为什么还要 LQFM？
> - A: 因为 CODSR 仍建立在 latent diffusion 上，VAE encoder 会压缩掉部分低层结构；LQFM 用 pixel-unshuffle 后的 LQ feature 生成 gamma/beta，在 U-Net 中间层做 time-aware SFT，把结构线索补回 denoising 过程。


The goal of this paper is to develop an effective one-step diffusion-based SR method that is capable of balancing the reality and fidelity of restored images. We first propose a region-adaptive generative prior activation method to discriminatively add noise to regions with varying demands while preserving local fidelity. We then introduce an LQguided feature modulation module that conditions the intermediate features of the U-Net on the LQ image to enhance the reconstruction fidelity. Furthermore, we leverage Grounded-SAM2 [34] to identify text regions within the image, providing effective guidance for text interaction in the diffusion process and thereby mitigating text misalignment. The overall architecture of the proposed method is shown in Figure 2. The following subsections elaborate on the design of each component.
> 💡 **方法总览**: 这段给出三个模块的职责边界：RGPA 管“哪里需要更强生成先验”，LQFM 管“如何不丢 LQ 结构”，TMG 管“文本 token 应该影响哪个语义区域”。后面的公式都可以放回这三个问题里理解。


# 3.1. Region-adaptive generative prior activation
> 💡 **小节预览**: RGPA 讨论局部 fidelity-realism trade-off：给 z_L 加噪可以把 U-Net 拉回 pretrained diffusion 熟悉的 denoising 模式，但加噪必须按区域做，否则平坦区域会被误生成纹理。


Different from existing one-step methods [13, 40, 50, 55] that directly feed the LQ latent into the denoising network, we introduce the deliberate addition of Gaussian noise to the LQ latent before denoising, in order to better unleash the generative priors of diffusion models for producing visually realistic outputs. To further achieve region-aware activation of diffusion priors, we propose a region-adaptive generative prior activation (RGPA) method that distinguishes between flat and textured regions via the Sobel gradient map and accordingly constructs spatially adaptive noise for the forward process.
> 💡 **RGPA 批注**: 这段的关键是“deliberate addition of Gaussian noise”。CODSR 认为直接 denoise LQ latent 太像普通 restoration，不足以释放 diffusion prior；RGPA 则把噪声作为激活 prior 的入口，再用 Sobel 梯度限制它的空间范围。


Specifically, we first encode the LQ image $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ into the latent features $z _ { L }$ using a VAE encoder. Instead of directly denoising, we construct the noisy latent ${ \boldsymbol { z } } _ { t }$ by adding the adaptive noise $\epsilon _ { a }$ in a one-step forward process:
> 💡 **数据流批读**: 这里的输入是 VAE 编码后的 z_L，不是像素图；输出是 z_t。这个选择很重要，因为后续 U-Net 本来就是在 latent space 里预测噪声，RGPA 让 forward/reverse pathway 对齐。


![Equation 1](../images/b8a82ad475e76aa14634899dfd98e74503996abcafe2c184ea4b416089b880a2.jpg)
*Equation 1: Equation extracted by MinerU.*
> 💡 **公式批读**: Equation 1 是标准扩散前向形式的 CODSR 版本：z_L 乘保留系数，加上 adaptive noise epsilon_a 乘噪声系数，得到 z_t。真正的创新不在公式形态，而在 epsilon_a 是空间自适应的。


where $\bar { \alpha } _ { t }$ is the cumulative parameter controlling the noise level at timestep $t$ . During the reverse process, the clean latent $\hat { z } _ { H }$ is recovered using the noise $\boldsymbol { \epsilon } _ { \theta } ( z _ { t } , \boldsymbol { c } , t )$ predicted
> 💡 **反向过程批读**: 反向只走一步：U-Net 根据 z_t、文本条件 c 和 timestep t 预测噪声，再恢复 clean latent。单步意味着没有多轮纠错，所以前面 RGPA 的噪声位置和后面 LQFM/TMG 的条件约束都更关键。


by the U-Net, conditioned on the text embedding $^ c$

![Equation 2](../images/13d576c1deb0aea64a145308fe0514bbb2833fb1d9f3bce7f0d6683f49230860.jpg)
*Equation 2: Equation extracted by MinerU.*
> 💡 **公式批读**: Equation 2 是从 z_t 反推出 hat z_H 的一步 denoising 公式，w_t 决定预测噪声要扣回多少。它也解释了为什么后文 lambda_t=1/w_t 可以用来调节 LQFM 强度。


where $\begin{array} { r } { w _ { t } = \frac { \sqrt { 1 - \bar { \alpha } _ { t } } } { \sqrt { \bar { \alpha } _ { t } } } } \end{array}$ is a time-dependent noise coefficient.
> 💡 **timestep 批读**: w_t 越大，说明当前 z_t 的噪声成分越强，模型越依赖生成先验；这正是后文通过 t_s 控制 fidelity-realism trade-off 的数学入口。


The adaptive noise $\epsilon _ { a }$ is derived by a gradient-weighted process. First, the Sobel operator is applied to compute the gradient map of $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ , denoted as ${ \mathbf { \mathit { g } } } _ { { \mathit { x } } _ { L } }$ The noise weighting coefficients are then obtained by a mapping operator $W ( \cdot )$ , which performs $1 6 \times 1 6$ patch-wise averaging followed by a piecewise transformation similar to UPSR [63]. The final adaptive noise $\epsilon _ { a }$ is thus formulated as:
> 💡 **RGPA 机制**: Sobel 梯度只是一个低成本纹理需求 proxy：patch 平均让权重在 latent 分辨率上更稳定，piecewise mapping 则避免把所有边缘都线性放大。需要注意，梯度高不一定等于语义上应该生成，这也是潜在风险。


![Equation 3](../images/9e45d1883970609d86745c8ddab2722d83beb294f73b7a1fe714805c99817023.jpg)
*Equation 3: Equation extracted by MinerU.*
> 💡 **公式批读**: Equation 3 把标准 Gaussian noise epsilon 乘上 W(g_xL) 得到 epsilon_a。读法很直接：W 越大，该 patch 的 noisy latent 越偏 diffusion prior；W 越小，该 patch 越偏保留 LQ 结构。


where $\epsilon \sim \mathcal { N } ( 0 , I )$ . Additionally, to enhance the robustness of the model, a timestep $t _ { s } \in [ t _ { \operatorname* { m i n } } , t _ { \operatorname* { m a x } } ]$ is randomly sampled during training, encouraging the model to generalize across various noise levels. During inference, this design provides flexible control over the trade-off between fidelity and generative quality simply by adjusting the timestep.
> 💡 **控制批读**: t_s 是 CODSR 最直接的全局控制旋钮：训练时随机采样增强鲁棒性，推理时调大 t_s 会增强真实纹理但降低 PSNR，调小则更保真。RGPA 解决的是在同一个 t_s 下不同区域如何分配噪声。


# 3.2. LQ-guided feature modulation
> 💡 **小节预览**: LQFM 是 CODSR 的保真分支：它不试图让 VAE latent 完整保留一切，而是额外保留原始 LQ 的可用结构，并在 U-Net 中间特征上做温和调制。


To mitigate the LQ information loss caused by compression encoding of VAE, we propose a plug-and-play LQ-guided feature modulation (LQFM) module. In order to preserve as much information as possible while adjusting the spatial resolution of the LQ image $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ , a pixel-unshuffle operation is first applied to $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ to obtain the feature $\widetilde { \pmb { x } } _ { L }$ .
> 💡 **LQFM 批注**: pixel-unshuffle 的意义是重排空间信息而不是学习性压缩，尽量把 LQ 图里还可信的局部结构保留下来。它针对的不是生成质量，而是 VAE bottleneck 下的 fidelity 损失。


To effectively utilize the information in the LQ feature $\widetilde { \pmb { x } } _ { L }$ to guide the diffusion process and achieve faithful SISR, ewe adopt a time-aware spatial feature transform (SFT) layer that fuses $\widetilde { \pmb { x } } _ { L }$ with intermediate U-Net features to bridge the representation discrepancy. Specifically, we develop a time-aware SFT layer, conditioned on $\widetilde { \pmb { x } } _ { L }$ , which performs a spatial affine transformation on the output feature $\pmb { f } ^ { m }$ of the first convolutional layer in the U-Net:
> 💡 **SFT 批读**: time-aware SFT 不是简单相加，而是用 LQ feature 生成逐位置 gamma/beta 去仿射调制 f_m。这样能处理 LQ feature 与 U-Net feature 的 domain gap，也让调制强度随噪声强度变化。


![Equation 4](../images/8d1f4f0231cc8bd037f8705a43a688716e7a345504f1a023c798b07b04369fc3.jpg)
*Equation 4: Equation extracted by MinerU.*
> 💡 **公式批读**: Equation 4 可读成 f_m' = lambda_t * gamma * f_m + beta 一类的空间仿射调制。gamma 控制保留/放大哪些中间响应，beta 补充结构偏移，lambda_t 让 LQ 条件和 RGPA 的噪声强度匹配。


where the pair of modulation parameters $( \gamma , \beta ) = \mathcal { M } ( \widetilde { \pmb { x } } _ { L } )$ are generated by a mapping function $\mathcal { M }$ , which consists of two MLP layers, $\odot$ denotes the element-wise multiplication, and $\begin{array} { r } { \lambda _ { t } = \frac { 1 } { w _ { t } } } \end{array}$ is a time-dependent coefficient that controls the strength of the modulation, thereby making LQFM time-aware and compatible with RGPA.
> 💡 **耦合关系**: lambda_t=1/w_t 很关键：噪声越强时，LQFM 的调制策略也要随时间系数变化，否则强 LQ 条件可能压住生成先验，或者弱 LQ 条件导致结构漂移。论文把 LQFM 明确做成与 RGPA 兼容。


To preserve the integrity of the latent space, our method modulates the intermediate feature $f _ { m }$ of the U-Net, rather than the LQ latent code $z _ { L }$ . This targeted modulation mitigates the risk of information loss and maintains the stability of the original latent distribution, thereby ensuring a more robust denoising process (see Section 5).
> 💡 **插入位置批读**: 不直接调制 z_L 是一个稳健性选择。z_L 需要保持近似 Gaussian latent 分布以配合 diffusion denoising；把 LQ 条件注入 U-Net 中间层，既补结构，又少破坏 prior 的工作空间。


# 3.3. Text-matching guidance
> 💡 **小节预览**: TMG 讨论“文本先验作用到哪里”：prompt 提供语义类别，Grounded-SAM2 提供空间 mask，positive area loss 约束 cross-attention 真正看向对应区域。


To strengthen text-image alignment and effectively leverage the conditioning potential of text prompts, we propose a text-matching guidance (TMG) strategy. It employs Grounded-SAM2 [34], an open-vocabulary segmentation model, to derive region maps corresponding to text prompts. These maps provide explicit spatial guidance for modulating the interaction of text conditions with the latent features within the U-Net, ensuring that the textual conditioning is applied to semantically relevant image areas.
> 💡 **TMG 批注**: 这里的 region map 是显式空间约束。没有它时，文本 token 可能在 cross-attention 里关注到背景或邻近物体，结果就是纹理语义错位；有 mask 后，语义生成更像“在正确对象上补正确细节”。


Specifically, we employ NLTK [3] to remove adjectives from the prompt extracted by RAM [68], as they are too abstract for precise spatial guidance. The remaining nouns $\{ n ^ { 1 } , \ldots , n ^ { N } \}$ , together with $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ , are then fed into Grounded-SAM2 [34] to generate $\{ M ^ { 1 } , \ldots , M ^ { N } \}$ , where $M ^ { i }$ indicates the binary region mask corresponding to the noun $n ^ { i }$ . To ensure the effectiveness of the guidance, we perform a validity check on $M ^ { i }$ and discard it if the number of active pixels is below a predefined threshold. These masks explicitly define the regions for text interaction during the reverse process. We then achieve the text-matching interaction between $n ^ { i }$ and $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ via each cross-attention layer in the U-Net, and produce the averaged attention map $A ^ { i }$ across all layers that correspond to the noun $n ^ { i }$ , supervised by the positive area loss from CoMat [19].
> 💡 **实现细节批读**: 去掉 adjectives 是务实选择，因为“beautiful/old/clear”很难被 Grounded-SAM2 精确定位，noun 才能生成可靠 mask。风险也在这里：noun 抽取、mask 阈值、分割质量都会影响 TMG 的有效性。


# 3.4. Loss function

We train the proposed network using a two-stage training strategy. To better constrain the network training in the first stage, we use the pixel-wise content loss function and the learned perceptual image patch similarity (LPIPS) [64] loss function. In addition, the GAN-based loss function [59] is used to enhance the generation quality, particularly by improving realistic details and textures.
> 💡 **Stage 1 批读**: 第一阶段是先把模型训练成可靠的 one-step SR student：content/LPIPS 约束结构和感知相似，GAN loss 补真实纹理。这个阶段如果不稳，第二阶段 VSD 会更容易放大幻觉。


In the second stage, we adopt a dual-LoRA training strategy inspired by PiSA-SR [40]. To further enhance the semantic generation capability, semantic knowledge is distilled from a pretrained model using VSD loss [50]. The LoRA adapters trained in the first stage are frozen, while additional LoRA adapters are integrated exclusively into the cross-attention layers of the U-Net to enhance text alignment. This stage focuses on optimizing the following objective to address potential text misalignment issues:
> 💡 **Stage 2 批读**: 第二阶段冻结已有 LoRA，再只在 cross-attention 加 LoRA，说明作者把语义增强限制在文本交互通道。VSD 从 pretrained diffusion teacher 蒸馏语义生成能力，positive area loss 则防止这股能力落到错误区域。


![Equation 5](../images/582053e77467779792fc1e1e20dae7df8ceb4d56e47875e6cb04a1992da91952.jpg)
*Equation 5: Equation extracted by MinerU.*
> 💡 **公式批读**: Equation 5 的重点是把 OSEDiff 式 VSD/蒸馏目标与 L_pos 合在一起：前者提供“像真实图”的分布梯度，后者提供“在正确区域像对应语义”的空间约束。


where $\mathcal { L } _ { \mathrm { { O S E D i f f } } }$ denotes the loss function used in OSEDiff [50], $\eta _ { \mathrm { p o s } }$ is a weight parameter that is empirically set to 1 in the proposed method and ${ \mathcal { L } } _ { \mathrm { p o s } }$ is the positive area loss used in CoMat [19] to supervise text-matching interaction.
> 💡 **loss 批读**: eta_pos=1 说明作者没有把 positive area loss 当弱正则，而是作为第二阶段语义对齐的重要约束。复现时这个权重和 mask 质量会直接影响 TMG 是否稳定。


Table 1. Quantitative comparison with state-of-the-art diffusion-based SR methods on four real-world test datasets. $\uparrow$ indicates higher is better, $\downarrow$ indicates lower is better. The best and the second-best results are highlighted in red and blue, respectively.
> 💡 **表格位置提示**: Table 1 被 MinerU 抽到 Method 末尾，但内容属于主实验结果。读方法时只需记住后面会用 full-reference 与 no-reference 指标共同验证 fidelity-realism claim。


![Table 1.](../images/13f35ce3bc9936c12d26fc73ee03afdc0d54458879e3e4d0bf1bc9f9db9d45bb.jpg)
*Table 1.: Table 1. Quantitative comparison with state-of-the-art diffusion-based SR methods on four real-world test datasets. $\uparrow$ indicates higher is better, $\downarrow$ indicates lower is better. The best and the second-best results are highlighted in red and blue, respectively.*
> 💡 **表格批读**: Table 1 应按 full-step、one-step、GAN 三类 baseline 分开看。CODSR 若只赢 no-reference 而损失 full-reference，claim 就只是更会生成；若两侧都稳，才支撑“bridging fidelity-reality”。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 输入 | LQ image、prompt/noun masks、t_s |
| 核心模块 | RGPA 加区域噪声；LQFM 调制中间特征；TMG 约束 cross-attention；VSD 蒸馏语义 prior |
| 输出 | one-step SR latent，经 VAE decoder 得到高分辨率图像 |

### 核心洞察
1. RGPA 与 LQFM 是一对张力机制：前者增强生成自由度，后者补回 LQ 保真约束。
2. TMG 与 VSD 是第二阶段语义增强的配套：VSD 提供强语义梯度，TMG 约束它别错位。
3. 复现时最该盯的是 t_s、Sobel-to-noise mapping、LQFM 插入层、Grounded-SAM2 mask 质量和两阶段 LoRA 冻结策略。

### 可追问点
- RGPA 和普通加噪有什么区别？
- 为什么还要 LQFM？
- VSD 增强语义生成时，怎样避免它压过 LQ observation？
