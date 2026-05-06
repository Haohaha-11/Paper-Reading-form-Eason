[← 返回 README](../README.md)

# 3 Methodology

## 📌 预览
本节是核心方法，重点看模块输入输出、训练目标、推理路径和与 baseline 的差异。

---

# 3.1 Problem Modeling

Real-ISR is to estimate an HQ image $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ from the given LQ image $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ . This task can be conventionally modeled as an optimization problem: $\begin{array} { r } { \hat { \pmb x } _ { H } = \mathrm { a r g m i n } _ { \pmb x _ { H } } \big ( \mathcal L _ { d a t a } \left( \Phi ( \pmb x _ { H } ) , \pmb x _ { L } \right) + \lambda \mathcal L _ { r e g } \left( \pmb x _ { H } \right) \big ) } \end{array}$ , where $\Phi$ is the degradation function, $\mathcal { L } _ { d a t a }$ is the data term to measure the fidelity of the optimization output, $\mathcal { L } _ { \boldsymbol { r } e g }$ is the regularization term to exploit the prior information of natural images, and scalar $\lambda$ is the balance parameter. Many conventional ISR methods [13, 29, 65] restore the desired HQ image by assuming simple and known degradation models and employing hand-crafted natural image prior models (i.e., image sparsity based prior [54]).

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Figure 2](../images/c870df5ccd6c086ea0660444215e562e607bc45a476d8cd01309ccbf4a8358a6.jpg)
*Figure 2: Figure 2: The training framework of OSEDiff. The LQ image is passed through a trainable encoder $E _ { \theta }$ , a LoRA finetuned diffusion network $\epsilon _ { \theta }$ and a frozen decoder $D _ { \theta }$ to obtain the desired HQ image. In addition, text prompts are extracted from the LQ image and input to the diffusion network to stimulate its generation capacity. Meanwhile, the output of the diffusion network $\epsilon _ { \theta }$ will be sent to two regularizer networks (a frozen pre-trained one and a fine-tuned one), where variational score distillation is performed in latent space to ensure that the output of $\epsilon _ { \theta }$ follows HQ natural image distribution. The regularization loss will be back-propagated to update $E _ { \theta }$ and $\epsilon _ { \theta }$ . Once training is finished, only $E _ { \theta }$ , $\epsilon _ { \theta }$ and $D _ { \theta }$ will be used in inference.*

> 💡 **Figure 2 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

However, the performance of such optimization-based methods is largely hindered by two factors. First, the degradation function $\Phi$ is often unknown and hard to model in real-world scenarios. Second, the hand-crafted regularization terms $\mathcal { L } _ { \mathrm { r e g } }$ are hard to effectively model the complex natural image priors. With the development of deep-learning techniques, it has become prevalent to learn a neural network $G _ { \theta }$ , which is parameterized by $\theta$ , from a training dataset $S$ of $( { \pmb x } _ { L } , { \pmb x } _ { H } )$ pairs to map the LQ image to an HQ image. The network training can be described as the following learning problem:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 1](../images/d23a28b58c61f7531d39db0b6b2724db34700d2a60c56158f5351ea85969d06b.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and $\mathcal { L } _ { \mathrm { r e g } }$ are the loss functions. ${ \mathcal { L } } _ { \mathrm { d a t a } }$ enforces that the network output $\hat { \pmb x } _ { H } = G _ { \theta } ( { \pmb x } _ { L } )$ can approach to the ground-truth $\mathbf { \mathcal { x } } _ { H }$ as much as possible, which can be quantified by metrics such as $L _ { 1 }$ norm, $L _ { 2 }$ norm and LPIPS [64]. Using only the ${ \mathcal { L } } _ { \mathrm { d a t a } }$ loss to train the network $G _ { \theta }$ from scratch may over-fit the training dataset. In this work, we propose to finetune a pre-trained generative network, more specifically the SD [36] network, to improve the generalization capability of $G _ { \theta }$ . In addition, the regularization loss $\mathcal { L } _ { \mathrm { r e g } }$ is critical to improve the generalization capability of $G _ { \theta }$ , as well as enhance the naturalness of output HQ images $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ . Suppose that we have the distribution of real-world HQ images, denoted by $p ( { \pmb x } _ { H } )$ , the KL divergence [8] is an ideal choice to serve as the loss function of $\mathcal { L } _ { \mathrm { r e g } }$ ; that is, the distribution of restored HQ images, denoted by $q _ { \theta } ( \hat { \pmb x } _ { H } )$ , should be identical to $p ( { \pmb x } _ { H } )$ as much as possible. The regularization loss can be defined as:

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

![Equation 2](../images/74cae713bc0bf8b3e39c6794b2a08856100babb8c6fafe062be5cd877da020a7.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

Existing works [24, 46] mostly instantiate the above objective via adversarial training [14], which involves learning a discriminator to differentiate between the generated HQ image $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ and the real HQ image $\mathbf { \mathcal { x } } _ { H }$ , and updating the generator $G _ { \theta }$ to make $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ and $\scriptstyle { \mathbf { x } } _ { H }$ indistinguishable. However, the discriminators are often trained from scratch alongside the generator. They may not be able to acquire the full distribution of HQ images and lack enough discriminative power, resulting in sub-optimal Real-ISR performance.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

The recently developed T2I diffusion models such as SD [36] offer new options for us to formulate the loss $\mathcal { L } _ { \mathrm { r e g } }$ . These models, trained on billions of image-text pairs, can effectively depict the natural image distribution in latent space. Some score distillation methods have been reported to employ SD to optimize images by using the KL-divergence as the objective [49, 25, 43]. In particular, variational score distillation (VSD) [49, 58, 10] induces such a KL-divergence based objective from particlebased variational optimization to align the distributions represented by two diffusion models. Based on the above discussions, we propose to instantiate the learning objective in Eq. (1) by designing an efficient and effective one-step diffusion network. In specific, we finetune the pre-trained SD with LoRA [17] as our Real-ISR backbone network and employ VSD as our regularizer to align the distribution of network outputs with natural HQ images. The details are provided in the next section.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

# 3.2 One-Step Effective Diffusion Network

Framework Overview. As discussed in Sec. 1, the existing SD-based Real-ISR methods [42, 57, 31, 52, 40] perform multiple timesteps to estimate the HQ image with random noise as the starting point and the LQ image as control signal. These approaches are resource-intensive and will inherently introduce randomness. Based on our formulation in Sec. 3.1, we propose a one-step effective diffusion (OSEDiff) network for Real-ISR, whose training framework is shown in Fig. 2. Our generator $G _ { \theta }$ to be trained is composed of a trainable encoder $E _ { \theta }$ , a finetuned diffusion network $\epsilon _ { \theta }$ and a frozen decoder $D _ { \theta }$ . To ensure the generalization capability of $G _ { \theta }$ , the output of the diffusion network $\epsilon _ { \theta }$ will be sent to two regularizer networks, where VSD loss is performed in latent space. The regularization loss are back-propagated to update $E _ { \theta }$ and $\epsilon _ { \theta }$ . Once training is finished, only the generator $G _ { \theta }$ will be used in inference. In the following, we will delve into the detailed architecture design of OSEDiff, as well as its associated training losses.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Network Architecture Design. Let’s denote by $E _ { \phi }$ , $\epsilon _ { \phi }$ and $D _ { \phi }$ the VAE encoder, latent diffusion network and VAE decoder of a pretrained SD model, where $\phi$ denotes the model parameters. Inspired by the recent success of LoRA [17] in finetuning SD to downstream tasks [34, 35], we adopt LoRA to fine-tune the pre-trained SD in the Real-ISR task to obtain the desired generator $G _ { \theta }$ .

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

As shown in the left part of Fig. 2, to maintain SD’s original generation capacity, we introduce trainable LoRA [17] layers to the encoder $E _ { \phi }$ and the diffusion network $\epsilon _ { \phi }$ , finetuning them into $E _ { \theta }$ and $\epsilon _ { \theta }$ with our training data. For the decoder, we fix its parameters and directly set $D _ { \theta } = D _ { \phi }$ . This is to ensure that the output space of the diffusion network remains consistent with the regularizers.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Recall that the diffusion model diffuses the input latent feature $_ z$ through $z _ { t } = \alpha _ { t } z + \beta _ { t } \epsilon$ , where $\alpha _ { t } , \beta _ { t }$ are scalars that are dependent to diffusion timestep $t \in \{ 1 , \cdots , T \}$ [16]. With a neural network that can predict the noise in ${ \boldsymbol { z } } _ { t }$ , denoted as $\hat { \epsilon }$ , the denoised latent can be obtained as $\begin{array} { r } { \hat { z } _ { 0 } = \frac { z _ { t } - \beta _ { t } \hat { \epsilon } } { \alpha _ { t } } } \end{array}$ , which is expected to be cleaner and more photo-realistic than ${ \boldsymbol { z } } _ { t }$ . Moreover, SD is a text-conditioned generation model. By extracting the text embeddings [36], denoted by $c _ { y }$ , from the given text description $y$ , the noise prediction can be performed as $\hat { \pmb { \epsilon } } = \pmb { \epsilon } _ { \theta } ( z _ { t } ; t , c _ { y } )$ .

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

We adapt the above text-to-image denoising process to the Real-ISR task, and formulate the LQ-to-HQ latent transformation $F _ { \theta }$ as a text-conditioned image-to-image denoising process as:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 3](../images/4624aa05fa9a035e1b151b5e25b992349ac809608a407854d0fba755e81e5991.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where we conduct only one-step denoising on the LQ latent $z _ { L }$ , without introducing any noise, at the $T$ -th diffusion timestep. The denoising output $\hat { z } _ { H }$ is expected be more photo-realistic than $z _ { L }$ . As for the text embeddings, we apply some text prompt extractor, such as the DAPE [52], to LQ input $\scriptstyle { \pmb { x } } _ { L }$ , and obtain $c _ { y } = Y ( \pmb { x } _ { L } )$ . Finally, the whole LQ-to-HQ image synthesis can be written as:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 4](../images/4ec9bc02a7218996c38a5b97c37a0dd52c8c7d547484d8c041da650f9ca5069e.jpg)
*Equation 4: Equation extracted by MinerU.*

> 💡 **Equation 4 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

As mentioned in Sec. 3.1, to improve the performance for a Real-ISR model, it is required to supervise the generator training with both the data term ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and regularization term $\mathcal { L } _ { \mathrm { r e g } }$ . As shown in the right part of Fig. 2, we propose to adapt VSD [49] as the regularization term. Apart from utilizing the SD model as the pre-trained regularizer $\epsilon _ { \phi }$ , VSD also introduces a finetuned regularizer, i.e., a latent diffusion module finetuned on the distribution $q _ { \theta } \left( \hat { \pmb x } _ { H } \right)$ of generated images with LoRA. We denote this finetuned diffusion module as $\epsilon _ { \phi ^ { \prime } }$ .

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Training Loss. We train the generator $G _ { \theta }$ with the data loss ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and regularization loss $\mathcal { L } _ { \mathrm { r e g } }$ . We set ${ \mathcal { L } } _ { \mathrm { d a t a } }$ as the weighted sum of MSE loss and LPIPS loss:

> 💡 **批注**: 这是实验证据段：同时看主指标、消融、效率和案例，判断 claim 是否被支撑。

![Equation 5](../images/acd537eed44dca3f7725d7100d0539643e1ce07223dc873f4f48d454cba41537.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where $\lambda _ { 1 }$ is a weighting scalar. As for $\mathcal { L } _ { \mathrm { r e g } }$ , we adopt the VSD loss via:

![Equation 6](../images/44c20cedce8a911e3c05fab7c6c52185f3d4aab30b62731ee3e4125e14af0773.jpg)
*Equation 6: Equation extracted by MinerU.*

> 💡 **Equation 6 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

Given any trainable image-shape feature $_ { \textbf { \em x } }$ , its latent code $z = E _ { \phi } ( { \pmb x } )$ and encoded text prompt condition $c _ { y }$ , VSD optimizes $_ { \textbf { \em x } }$ to make it consistent with the text prompt $y$ via:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 7](../images/54b02736673efd3d527f6a25bd5050a81c085384d13fef1ec144b746435bfb6a.jpg)
*Equation 7: Equation extracted by MinerU.*

> 💡 **Equation 7 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where the expectation of the gradient is conducted over all diffusion timesteps $t \in \{ 1 , \cdots , T \}$ and $\epsilon \sim \mathcal { N } ( 0 , I )$ . Therefore, the overall training objective for the generator $G _ { \theta }$ is:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 8](../images/281d0f9fb937ed0f6d27df61635515afdedd540f0fdb4eaaad01ef429172ee4c.jpg)
*Equation 8: Equation extracted by MinerU.*

> 💡 **Equation 8 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

where $\lambda _ { 2 }$ is a weighting scalar. Besides, as required by VSD, the finetuned regularizer $\epsilon _ { \phi ^ { ' } }$ should also be trainable, and its training objective is:

![Equation 9](../images/c9d5b86e3fce85c331d4bf64267d58a122fa74d959e98d4f8025c9f55a2e6115.jpg)
*Equation 9: Equation extracted by MinerU.*

> 💡 **Equation 9 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

Note that the above ${ \mathcal { L } } _ { \mathrm { d i f f } }$ loss is only applied to update $\epsilon _ { \phi ^ { ' } }$ . The whole algorithm to illustrate the training pipeline can be found in the Appendix.

VSD in Latent Space. The original VSD computes the gradients in the image space. When it is used to train an SD-based generator network, there will be repetitive latent decoding/encoding in computing $\mathcal { L } _ { \mathrm { r e g } }$ . This is costly and makes the regularization less effective. Considering the fact that a well-trained VAE should satisfy $E _ { \phi } ( \pmb { x } ) = E _ { \phi } ( \bar { D _ { \phi } } ( z ) ) \approx z$ , we can approximately let $\bar { \cal E } _ { \phi } ( \hat { \pmb x } _ { H } ) = \hat { \pmb z } _ { H }$ . In this case, we can eliminate the redundant latent encoding/decoding in computing the regularization loss, as we follow DMD [58] to optimize the distribution loss in the latent state space rather than in the noise space. The gradient of the regularization loss w.r.t. the network parameter $\theta$ in the latent space is:

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

![Equation 10](../images/87588dbf5e5b461e9b38e6bea9a011d8344b6097123a319ca7539a1165ec899b.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: 公式通常定义过程、loss 或更新规则；建议把符号对应到输入、模型、记忆/控制变量与输出。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
