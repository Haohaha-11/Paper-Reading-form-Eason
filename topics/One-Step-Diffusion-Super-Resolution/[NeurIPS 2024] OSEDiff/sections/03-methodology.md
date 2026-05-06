[← 返回 README](../README.md)

# 3 Methodology

## 📌 预览
本节是核心方法，按数据流读：LQ image + pretrained Stable Diffusion prior → LQ latent initialization / LoRA tuning / latent VSD / data loss → 单步 Real-ISR image。
---

> 💡 **Q&A 批注记录**:
> - Q: VSD 在这里起什么作用？
> - A: VSD 相当于用 frozen diffusion teacher 给 student 的输出打“生成先验梯度”，让结果更接近真实图像分布。


# 3.1 Problem Modeling
> 💡 **小节预览**: 本小节不是 flow 建模，而是把 Real-ISR 写成“数据保真项 + 自然图像分布正则项”。后面的 OSEDiff 就是在这个框架下用 MSE/LPIPS 做 data loss，用 latent-space VSD 近似 KL regularization。


Real-ISR is to estimate an HQ image $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ from the given LQ image $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ . This task can be conventionally modeled as an optimization problem: $\begin{array} { r } { \hat { \pmb x } _ { H } = \mathrm { a r g m i n } _ { \pmb x _ { H } } \big ( \mathcal L _ { d a t a } \left( \Phi ( \pmb x _ { H } ) , \pmb x _ { L } \right) + \lambda \mathcal L _ { r e g } \left( \pmb x _ { H } \right) \big ) } \end{array}$ , where $\Phi$ is the degradation function, $\mathcal { L } _ { d a t a }$ is the data term to measure the fidelity of the optimization output, $\mathcal { L } _ { \boldsymbol { r } e g }$ is the regularization term to exploit the prior information of natural images, and scalar $\lambda$ is the balance parameter. Many conventional ISR methods [13, 29, 65] restore the desired HQ image by assuming simple and known degradation models and employing hand-crafted natural image prior models (i.e., image sparsity based prior [54]).
> 💡 **问题建模**: 公式把传统恢复写成优化问题：data term 确保退化后的结果能解释 LQ，regularization term 注入自然图像先验。OSEDiff 后面只是把手工 prior 换成 SD/VSD 这种学习到的 latent 分布先验。


![Figure 2](../images/c870df5ccd6c086ea0660444215e562e607bc45a476d8cd01309ccbf4a8358a6.jpg)
*Figure 2: Figure 2: The training framework of OSEDiff. The LQ image is passed through a trainable encoder $E _ { \theta }$ , a LoRA finetuned diffusion network $\epsilon _ { \theta }$ and a frozen decoder $D _ { \theta }$ to obtain the desired HQ image. In addition, text prompts are extracted from the LQ image and input to the diffusion network to stimulate its generation capacity. Meanwhile, the output of the diffusion network $\epsilon _ { \theta }$ will be sent to two regularizer networks (a frozen pre-trained one and a fine-tuned one), where variational score distillation is performed in latent space to ensure that the output of $\epsilon _ { \theta }$ follows HQ natural image distribution. The regularization loss will be back-propagated to update $E _ { \theta }$ and $\epsilon _ { \theta }$ . Once training is finished, only $E _ { \theta }$ , $\epsilon _ { \theta }$ and $D _ { \theta }$ will be used in inference.*
> 💡 **Figure 2 批读**: 左侧是推理时会保留的 generator：LQ → trainable encoder → LoRA-UNet one-step denoise → frozen decoder。右侧两个 regularizer 只在训练时提供 VSD 梯度，推理时不走这条分支。


However, the performance of such optimization-based methods is largely hindered by two factors. First, the degradation function $\Phi$ is often unknown and hard to model in real-world scenarios. Second, the hand-crafted regularization terms $\mathcal { L } _ { \mathrm { r e g } }$ are hard to effectively model the complex natural image priors. With the development of deep-learning techniques, it has become prevalent to learn a neural network $G _ { \theta }$ , which is parameterized by $\theta$ , from a training dataset $S$ of $( { \pmb x } _ { L } , { \pmb x } _ { H } )$ pairs to map the LQ image to an HQ image. The network training can be described as the following learning problem:
> 💡 **从优化到学习**: 作者先指出真实退化 $\Phi$ 不可知、手工先验不足，再把问题改写成学习 $G_\theta(x_L)$。这一步为“用训练数据学 data term，用 SD prior 学 regularization”搭桥。


![Equation 1](../images/d23a28b58c61f7531d39db0b6b2724db34700d2a60c56158f5351ea85969d06b.jpg)
*Equation 1: Equation extracted by MinerU.*
> 💡 **Equation 1 批读**: 这就是训练目标的总框架：输出既要贴近 GT，又要看起来像 HQ 自然图像。后文 Eq. 5 对应 $\mathcal L_{data}$，Eq. 6-10 对应 $\mathcal L_{reg}$。


where ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and $\mathcal { L } _ { \mathrm { r e g } }$ are the loss functions. ${ \mathcal { L } } _ { \mathrm { d a t a } }$ enforces that the network output $\hat { \pmb x } _ { H } = G _ { \theta } ( { \pmb x } _ { L } )$ can approach to the ground-truth $\mathbf { \mathcal { x } } _ { H }$ as much as possible, which can be quantified by metrics such as $L _ { 1 }$ norm, $L _ { 2 }$ norm and LPIPS [64]. Using only the ${ \mathcal { L } } _ { \mathrm { d a t a } }$ loss to train the network $G _ { \theta }$ from scratch may over-fit the training dataset. In this work, we propose to finetune a pre-trained generative network, more specifically the SD [36] network, to improve the generalization capability of $G _ { \theta }$ . In addition, the regularization loss $\mathcal { L } _ { \mathrm { r e g } }$ is critical to improve the generalization capability of $G _ { \theta }$ , as well as enhance the naturalness of output HQ images $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ . Suppose that we have the distribution of real-world HQ images, denoted by $p ( { \pmb x } _ { H } )$ , the KL divergence [8] is an ideal choice to serve as the loss function of $\mathcal { L } _ { \mathrm { r e g } }$ ; that is, the distribution of restored HQ images, denoted by $q _ { \theta } ( \hat { \pmb x } _ { H } )$ , should be identical to $p ( { \pmb x } _ { H } )$ as much as possible. The regularization loss can be defined as:
> 💡 **regularization 动机**: data loss 负责成对监督，但只用它容易过拟合或过平滑；KL 正则的目标是让 restored distribution $q_\theta$ 接近真实 HQ distribution $p$，这比单张图的像素误差更接近“自然感”。


![Equation 2](../images/74cae713bc0bf8b3e39c6794b2a08856100babb8c6fafe062be5cd877da020a7.jpg)
*Equation 2: Equation extracted by MinerU.*
> 💡 **Equation 2 批读**: 这个 KL 不能直接算，因为真实 HQ 分布不可显式得到。论文的关键选择是用 SD score/VSD 来给这个 KL 提供可训练的梯度近似。


Existing works [24, 46] mostly instantiate the above objective via adversarial training [14], which involves learning a discriminator to differentiate between the generated HQ image $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ and the real HQ image $\mathbf { \mathcal { x } } _ { H }$ , and updating the generator $G _ { \theta }$ to make $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ and $\scriptstyle { \mathbf { x } } _ { H }$ indistinguishable. However, the discriminators are often trained from scratch alongside the generator. They may not be able to acquire the full distribution of HQ images and lack enough discriminative power, resulting in sub-optimal Real-ISR performance.
> 💡 **为什么不用 GAN**: GAN loss 也是分布正则，但 discriminator 要从头学，容易只学到局部或数据集偏差。OSEDiff 改用大规模预训练 SD 当 regularizer，核心假设是 SD latent space 已经编码了更完整的自然图像先验。


The recently developed T2I diffusion models such as SD [36] offer new options for us to formulate the loss $\mathcal { L } _ { \mathrm { r e g } }$ . These models, trained on billions of image-text pairs, can effectively depict the natural image distribution in latent space. Some score distillation methods have been reported to employ SD to optimize images by using the KL-divergence as the objective [49, 25, 43]. In particular, variational score distillation (VSD) [49, 58, 10] induces such a KL-divergence based objective from particlebased variational optimization to align the distributions represented by two diffusion models. Based on the above discussions, we propose to instantiate the learning objective in Eq. (1) by designing an efficient and effective one-step diffusion network. In specific, we finetune the pre-trained SD with LoRA [17] as our Real-ISR backbone network and employ VSD as our regularizer to align the distribution of network outputs with natural HQ images. The details are provided in the next section.
> 💡 **VSD 定位**: 这里的 VSD 不是单纯蒸馏采样轨迹，而是把 generator 输出分布和 SD 表示的 HQ 自然图像分布对齐。它承担 Eq. 2 的 KL regularization 角色。


# 3.2 One-Step Effective Diffusion Network

Framework Overview. As discussed in Sec. 1, the existing SD-based Real-ISR methods [42, 57, 31, 52, 40] perform multiple timesteps to estimate the HQ image with random noise as the starting point and the LQ image as control signal. These approaches are resource-intensive and will inherently introduce randomness. Based on our formulation in Sec. 3.1, we propose a one-step effective diffusion (OSEDiff) network for Real-ISR, whose training framework is shown in Fig. 2. Our generator $G _ { \theta }$ to be trained is composed of a trainable encoder $E _ { \theta }$ , a finetuned diffusion network $\epsilon _ { \theta }$ and a frozen decoder $D _ { \theta }$ . To ensure the generalization capability of $G _ { \theta }$ , the output of the diffusion network $\epsilon _ { \theta }$ will be sent to two regularizer networks, where VSD loss is performed in latent space. The regularization loss are back-propagated to update $E _ { \theta }$ and $\epsilon _ { \theta }$ . Once training is finished, only the generator $G _ { \theta }$ will be used in inference. In the following, we will delve into the detailed architecture design of OSEDiff, as well as its associated training losses.
> 💡 **框架总览**: $G_\theta=E_\theta+\epsilon_\theta+D_\theta$。训练时 data loss 直接约束解码图像，VSD loss 约束 latent 输出分布；推理时只需要一次 encoder/UNet/decoder 前向。


Network Architecture Design. Let’s denote by $E _ { \phi }$ , $\epsilon _ { \phi }$ and $D _ { \phi }$ the VAE encoder, latent diffusion network and VAE decoder of a pretrained SD model, where $\phi$ denotes the model parameters. Inspired by the recent success of LoRA [17] in finetuning SD to downstream tasks [34, 35], we adopt LoRA to fine-tune the pre-trained SD in the Real-ISR task to obtain the desired generator $G _ { \theta }$ .
> 💡 **LoRA 设计**: LoRA 的作用是小成本把 SD 从 T2I generation 迁移到 Real-ISR。它保留原 SD 权重主体，降低可训练参数，同时避免全量微调破坏底座 prior。


As shown in the left part of Fig. 2, to maintain SD’s original generation capacity, we introduce trainable LoRA [17] layers to the encoder $E _ { \phi }$ and the diffusion network $\epsilon _ { \phi }$ , finetuning them into $E _ { \theta }$ and $\epsilon _ { \theta }$ with our training data. For the decoder, we fix its parameters and directly set $D _ { \theta } = D _ { \phi }$ . This is to ensure that the output space of the diffusion network remains consistent with the regularizers.
> 💡 **冻结 decoder 的原因**: encoder 和 UNet 加 LoRA 适配退化，decoder 固定是为了让 UNet 输出仍落在原 SD VAE latent 空间里。否则右侧 frozen/fine-tuned regularizer 的 latent 对齐目标会变得不稳定。


Recall that the diffusion model diffuses the input latent feature $_ z$ through $z _ { t } = \alpha _ { t } z + \beta _ { t } \epsilon$ , where $\alpha _ { t } , \beta _ { t }$ are scalars that are dependent to diffusion timestep $t \in \{ 1 , \cdots , T \}$ [16]. With a neural network that can predict the noise in ${ \boldsymbol { z } } _ { t }$ , denoted as $\hat { \epsilon }$ , the denoised latent can be obtained as $\begin{array} { r } { \hat { z } _ { 0 } = \frac { z _ { t } - \beta _ { t } \hat { \epsilon } } { \alpha _ { t } } } \end{array}$ , which is expected to be cleaner and more photo-realistic than ${ \boldsymbol { z } } _ { t }$ . Moreover, SD is a text-conditioned generation model. By extracting the text embeddings [36], denoted by $c _ { y }$ , from the given text description $y$ , the noise prediction can be performed as $\hat { \pmb { \epsilon } } = \pmb { \epsilon } _ { \theta } ( z _ { t } ; t , c _ { y } )$ .
> 💡 **SD denoise 基础**: 这段只是回顾 latent diffusion 的噪声预测和文本条件。OSEDiff 后面会“借壳”：不从随机噪声 $z_t$ 开始，而从 LQ latent 直接做一次去噪式变换。


We adapt the above text-to-image denoising process to the Real-ISR task, and formulate the LQ-to-HQ latent transformation $F _ { \theta }$ as a text-conditioned image-to-image denoising process as:
> 💡 **任务改写**: $F_\theta$ 把 LQ-to-HQ 变成 text-conditioned image-to-image denoising。文本条件来自 DAPE 等 prompt extractor，用来激活 SD 的语义/纹理生成能力。


![Equation 3](../images/4624aa05fa9a035e1b151b5e25b992349ac809608a407854d0fba755e81e5991.jpg)
*Equation 3: Equation extracted by MinerU.*
> 💡 **Equation 3 批读**: 关键是“only one-step”和“without introducing any noise”。$\hat z_H$ 由 $z_L$ 在固定高 timestep $T$ 上一次变换得到，LQ latent 同时承担结构锚点和扩散起点。


where we conduct only one-step denoising on the LQ latent $z _ { L }$ , without introducing any noise, at the $T$ -th diffusion timestep. The denoising output $\hat { z } _ { H }$ is expected be more photo-realistic than $z _ { L }$ . As for the text embeddings, we apply some text prompt extractor, such as the DAPE [52], to LQ input $\scriptstyle { \pmb { x } } _ { L }$ , and obtain $c _ { y } = Y ( \pmb { x } _ { L } )$ . Finally, the whole LQ-to-HQ image synthesis can be written as:
> 💡 **LQ latent initialization**: $z_L=E_\theta(x_L)$ 是整篇最重要的起点选择。它把输入布局、低频颜色、主要边缘带进 latent，使一步模型主要补退化和细节，而不是从噪声重建全图。


![Equation 4](../images/4ec9bc02a7218996c38a5b97c37a0dd52c8c7d547484d8c041da650f9ca5069e.jpg)
*Equation 4: Equation extracted by MinerU.*
> 💡 **Equation 4 批读**: 完整推理链是 $x_L \rightarrow z_L \rightarrow \hat z_H \rightarrow \hat x_H$。这里没有 iterative sampler，也没有训练期 regularizer，因此 one-step inference 的速度优势来自结构本身。


As mentioned in Sec. 3.1, to improve the performance for a Real-ISR model, it is required to supervise the generator training with both the data term ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and regularization term $\mathcal { L } _ { \mathrm { r e g } }$ . As shown in the right part of Fig. 2, we propose to adapt VSD [49] as the regularization term. Apart from utilizing the SD model as the pre-trained regularizer $\epsilon _ { \phi }$ , VSD also introduces a finetuned regularizer, i.e., a latent diffusion module finetuned on the distribution $q _ { \theta } \left( \hat { \pmb x } _ { H } \right)$ of generated images with LoRA. We denote this finetuned diffusion module as $\epsilon _ { \phi ^ { \prime } }$ .
> 💡 **两个 regularizer**: VSD 需要 frozen SD $\epsilon_\phi$ 和跟随 student 输出分布训练的 $\epsilon_{\phi'}$。二者差值给 generator 提供“往真实 HQ 分布移动”的方向。


Training Loss. We train the generator $G _ { \theta }$ with the data loss ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and regularization loss $\mathcal { L } _ { \mathrm { r e g } }$ . We set ${ \mathcal { L } } _ { \mathrm { d a t a } }$ as the weighted sum of MSE loss and LPIPS loss:
> 💡 **data loss 作用**: MSE 管低频/颜色/结构，LPIPS 管感知相似度；它们防止 VSD 只追求漂亮纹理而偏离 GT。这个 data term 是 OSEDiff 区别于纯生成蒸馏的重要安全锚。


![Equation 5](../images/acd537eed44dca3f7725d7100d0539643e1ce07223dc873f4f48d454cba41537.jpg)
*Equation 5: Equation extracted by MinerU.*
> 💡 **Equation 5 批读**: $\lambda_1=2$ 时，data loss = MSE + 2*LPIPS。它不是最终目标的全部，只负责把输出拉回成对监督；真实纹理分布由后面的 VSD 补。


where $\lambda _ { 1 }$ is a weighting scalar. As for $\mathcal { L } _ { \mathrm { r e g } }$ , we adopt the VSD loss via:
> 💡 **regularization 入口**: 从这里开始读 VSD。它要解决的问题是：一步 generator 的输出分布 $q_\theta$ 如何借助 SD teacher 接近 HQ natural image distribution。


![Equation 6](../images/44c20cedce8a911e3c05fab7c6c52185f3d4aab30b62731ee3e4125e14af0773.jpg)
*Equation 6: Equation extracted by MinerU.*
> 💡 **Equation 6 批读**: 这是 VSD 正则项的符号入口，真正可用的训练梯度在 Eq. 7/Eq. 10。读公式时要区分 generator 参数 $\theta$ 和 regularizer 参数 $\phi'$。


Given any trainable image-shape feature $_ { \textbf { \em x } }$ , its latent code $z = E _ { \phi } ( { \pmb x } )$ and encoded text prompt condition $c _ { y }$ , VSD optimizes $_ { \textbf { \em x } }$ to make it consistent with the text prompt $y$ via:
> 💡 **VSD 梯度直觉**: 对任意可训练图像/特征，VSD 比较 frozen teacher 和 fine-tuned regularizer 的 score 预测差异；这个差异近似告诉输出应该如何移动，才能更像目标分布而不是 student 当前分布。


![Equation 7](../images/54b02736673efd3d527f6a25bd5050a81c085384d13fef1ec144b746435bfb6a.jpg)
*Equation 7: Equation extracted by MinerU.*
> 💡 **Equation 7 批读**: 梯度里同时采样 timestep 和噪声，说明正则不是只看干净 latent，而是在多噪声水平上校准 score。对 OSEDiff 来说，这能弥补一步输出缺少多步 refinement 的问题。


where the expectation of the gradient is conducted over all diffusion timesteps $t \in \{ 1 , \cdots , T \}$ and $\epsilon \sim \mathcal { N } ( 0 , I )$ . Therefore, the overall training objective for the generator $G _ { \theta }$ is:
> 💡 **总体训练目标**: generator 更新时同时吃两种梯度：data loss 保真，VSD regularization 逼近 HQ 分布。二者权重决定 OSEDiff 的 fidelity-realism trade-off。


![Equation 8](../images/281d0f9fb937ed0f6d27df61635515afdedd540f0fdb4eaaad01ef429172ee4c.jpg)
*Equation 8: Equation extracted by MinerU.*
> 💡 **Equation 8 批读**: 最终目标是 $\mathcal L_{data}+\lambda_2\mathcal L_{reg}$，论文实现中 $\lambda_2=1$。如果复现实验出现 hallucination 或过平滑，优先检查这两个 loss 的相对强度。


where $\lambda _ { 2 }$ is a weighting scalar. Besides, as required by VSD, the finetuned regularizer $\epsilon _ { \phi ^ { ' } }$ should also be trainable, and its training objective is:
> 💡 **regularizer 也要训练**: $\epsilon_{\phi'}$ 不是 teacher，而是拟合 generator 当前输出分布的“student-distribution score”。VSD 需要它和 frozen SD 的差值来形成分布匹配方向。


![Equation 9](../images/c9d5b86e3fce85c331d4bf64267d58a122fa74d959e98d4f8025c9f55a2e6115.jpg)
*Equation 9: Equation extracted by MinerU.*
> 💡 **Equation 9 批读**: 这个 diffusion loss 只更新 $\epsilon_{\phi'}$，不更新 generator。它让 fine-tuned regularizer 学会给 OSEDiff 当前生成分布打分。


Note that the above ${ \mathcal { L } } _ { \mathrm { d i f f } }$ loss is only applied to update $\epsilon _ { \phi ^ { ' } }$ . The whole algorithm to illustrate the training pipeline can be found in the Appendix.
> 💡 **训练分工**: generator 用 Eq. 8 更新，regularizer 用 Eq. 9 更新。推理阶段两者分离，只部署 generator，避免把 VSD 的额外网络成本带到线上。


VSD in Latent Space. The original VSD computes the gradients in the image space. When it is used to train an SD-based generator network, there will be repetitive latent decoding/encoding in computing $\mathcal { L } _ { \mathrm { r e g } }$ . This is costly and makes the regularization less effective. Considering the fact that a well-trained VAE should satisfy $E _ { \phi } ( \pmb { x } ) = E _ { \phi } ( \bar { D _ { \phi } } ( z ) ) \approx z$ , we can approximately let $\bar { \cal E } _ { \phi } ( \hat { \pmb x } _ { H } ) = \hat { \pmb z } _ { H }$ . In this case, we can eliminate the redundant latent encoding/decoding in computing the regularization loss, as we follow DMD [58] to optimize the distribution loss in the latent state space rather than in the noise space. The gradient of the regularization loss w.r.t. the network parameter $\theta$ in the latent space is:
> 💡 **latent-space VSD**: 原始 VSD 在 image space 会反复 decode/encode，成本高且梯度绕远。OSEDiff 直接在 $\hat z_H$ 上做分布正则，和 SD 原本的 latent diffusion 空间一致，也解释了为什么 decoder 要冻结。


![Equation 10](../images/87588dbf5e5b461e9b38e6bea9a011d8344b6097123a319ca7539a1165ec899b.jpg)
*Equation 10: Equation extracted by MinerU.*
> 💡 **Equation 10 批读**: 这是落地版 latent VSD 梯度：用 frozen regularizer 和 fine-tuned regularizer 的预测差推 generator latent。读实现时重点核对 stop-grad、timestep 采样范围、CFG 和归一化权重 $\omega$。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 输入 | LQ image + pretrained Stable Diffusion prior |
| 核心模块 | LQ latent initialization / LoRA tuning / latent VSD / data loss |
| 输出 | 单步 Real-ISR image |

### 核心洞察
1. Problem modeling 的核心是 data term + regularization term；OSEDiff 用 SD/VSD 实例化后者。
2. LQ latent initialization 解决随机噪声起点和结构锚定问题；LoRA 解决 SD 到 Real-ISR 的轻量适配；latent VSD 解决一步输出缺少多步先验 refinement 的问题。
3. 复现时优先确认：decoder 是否冻结、LoRA 插入 encoder/UNet/regularizer 的位置、VSD 是否在 latent space、推理是否只跑一次 generator。

### 可追问点
- 为什么从 LQ latent 而不是纯噪声开始？
- VSD 在这里起什么作用？
- 为什么固定 VAE decoder 反而有利于 latent VSD？
- data loss 和 VSD loss 的权重会怎样影响保真/真实感 trade-off？
