[← 返回 README](../README.md)

# Method

## 📌 预览
本文件合并 Method/Background，重点看 LQ latent、teacher/student、flow/ODE、控制变量和训练损失。

---

# 3 Methodology

# 3.1 Problem Modeling

Real-ISR is to estimate an HQ image $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ from the given LQ image $\scriptstyle { \mathbf { \mathcal { x } } } _ { L }$ . This task can be conventionally modeled as an optimization problem: $\begin{array} { r } { \hat { \pmb x } _ { H } = \mathrm { a r g m i n } _ { \pmb x _ { H } } \big ( \mathcal L _ { d a t a } \left( \Phi ( \pmb x _ { H } ) , \pmb x _ { L } \right) + \lambda \mathcal L _ { r e g } \left( \pmb x _ { H } \right) \big ) } \end{array}$ , where $\Phi$ is the degradation function, $\mathcal { L } _ { d a t a }$ is the data term to measure the fidelity of the optimization output, $\mathcal { L } _ { \boldsymbol { r } e g }$ is the regularization term to exploit the prior information of natural images, and scalar $\lambda$ is the balance parameter. Many conventional ISR methods [13, 29, 65] restore the desired HQ image by assuming simple and known degradation models and employing hand-crafted natural image prior models (i.e., image sparsity based prior [54]).

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

![Figure 2](../images/c870df5ccd6c086ea0660444215e562e607bc45a476d8cd01309ccbf4a8358a6.jpg)
*Figure 2: Figure 2: The training framework of OSEDiff. The LQ image is passed through a trainable encoder $E _ { \theta }$ , a LoRA finetuned diffusion network $\epsilon _ { \theta }$ and a frozen decoder $D _ { \theta }$ to obtain the desired HQ image. In addition, text prompts are extracted from the LQ image and input to the diffusion network to stimulate its generation capacity. Meanwhile, the output of the diffusion network $\epsilon _ { \theta }$ will be sent to two regularizer networks (a frozen pre-trained one and a fine-tuned one), where variational score distillation is performed in latent space to ensure that the output of $\epsilon _ { \theta }$ follows HQ natural image distribution. The regularization loss will be back-propagated to update $E _ { \theta }$ and $\epsilon _ { \theta }$ . Once training is finished, only $E _ { \theta }$ , $\epsilon _ { \theta }$ and $D _ { \theta }$ will be used in inference.*

> 💡 **Figure 2 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

However, the performance of such optimization-based methods is largely hindered by two factors. First, the degradation function $\Phi$ is often unknown and hard to model in real-world scenarios. Second, the hand-crafted regularization terms $\mathcal { L } _ { \mathrm { r e g } }$ are hard to effectively model the complex natural image priors. With the development of deep-learning techniques, it has become prevalent to learn a neural network $G _ { \theta }$ , which is parameterized by $\theta$ , from a training dataset $S$ of $( { \pmb x } _ { L } , { \pmb x } _ { H } )$ pairs to map the LQ image to an HQ image. The network training can be described as the following learning problem:

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Equation 1](../images/d23a28b58c61f7531d39db0b6b2724db34700d2a60c56158f5351ea85969d06b.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and $\mathcal { L } _ { \mathrm { r e g } }$ are the loss functions. ${ \mathcal { L } } _ { \mathrm { d a t a } }$ enforces that the network output $\hat { \pmb x } _ { H } = G _ { \theta } ( { \pmb x } _ { L } )$ can approach to the ground-truth $\mathbf { \mathcal { x } } _ { H }$ as much as possible, which can be quantified by metrics such as $L _ { 1 }$ norm, $L _ { 2 }$ norm and LPIPS [64]. Using only the ${ \mathcal { L } } _ { \mathrm { d a t a } }$ loss to train the network $G _ { \theta }$ from scratch may over-fit the training dataset. In this work, we propose to finetune a pre-trained generative network, more specifically the SD [36] network, to improve the generalization capability of $G _ { \theta }$ . In addition, the regularization loss $\mathcal { L } _ { \mathrm { r e g } }$ is critical to improve the generalization capability of $G _ { \theta }$ , as well as enhance the naturalness of output HQ images $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ . Suppose that we have the distribution of real-world HQ images, denoted by $p ( { \pmb x } _ { H } )$ , the KL divergence [8] is an ideal choice to serve as the loss function of $\mathcal { L } _ { \mathrm { r e g } }$ ; that is, the distribution of restored HQ images, denoted by $q _ { \theta } ( \hat { \pmb x } _ { H } )$ , should be identical to $p ( { \pmb x } _ { H } )$ as much as possible. The regularization loss can be defined as:

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Equation 2](../images/74cae713bc0bf8b3e39c6794b2a08856100babb8c6fafe062be5cd877da020a7.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

Existing works [24, 46] mostly instantiate the above objective via adversarial training [14], which involves learning a discriminator to differentiate between the generated HQ image $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ and the real HQ image $\mathbf { \mathcal { x } } _ { H }$ , and updating the generator $G _ { \theta }$ to make $\scriptstyle { \hat { \mathbf { x } } } _ { H }$ and $\scriptstyle { \mathbf { x } } _ { H }$ indistinguishable. However, the discriminators are often trained from scratch alongside the generator. They may not be able to acquire the full distribution of HQ images and lack enough discriminative power, resulting in sub-optimal Real-ISR performance.

> 💡 **批注**: 这是一段信息密度较高的论述：建议拆成“问题 → 方法 → 证据/结论”三层来读。

The recently developed T2I diffusion models such as SD [36] offer new options for us to formulate the loss $\mathcal { L } _ { \mathrm { r e g } }$ . These models, trained on billions of image-text pairs, can effectively depict the natural image distribution in latent space. Some score distillation methods have been reported to employ SD to optimize images by using the KL-divergence as the objective [49, 25, 43]. In particular, variational score distillation (VSD) [49, 58, 10] induces such a KL-divergence based objective from particlebased variational optimization to align the distributions represented by two diffusion models. Based on the above discussions, we propose to instantiate the learning objective in Eq. (1) by designing an efficient and effective one-step diffusion network. In specific, we finetune the pre-trained SD with LoRA [17] as our Real-ISR backbone network and employ VSD as our regularizer to align the distribution of network outputs with natural HQ images. The details are provided in the next section.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

# 3.2 One-Step Effective Diffusion Network

Framework Overview. As discussed in Sec. 1, the existing SD-based Real-ISR methods [42, 57, 31, 52, 40] perform multiple timesteps to estimate the HQ image with random noise as the starting point and the LQ image as control signal. These approaches are resource-intensive and will inherently introduce randomness. Based on our formulation in Sec. 3.1, we propose a one-step effective diffusion (OSEDiff) network for Real-ISR, whose training framework is shown in Fig. 2. Our generator $G _ { \theta }$ to be trained is composed of a trainable encoder $E _ { \theta }$ , a finetuned diffusion network $\epsilon _ { \theta }$ and a frozen decoder $D _ { \theta }$ . To ensure the generalization capability of $G _ { \theta }$ , the output of the diffusion network $\epsilon _ { \theta }$ will be sent to two regularizer networks, where VSD loss is performed in latent space. The regularization loss are back-propagated to update $E _ { \theta }$ and $\epsilon _ { \theta }$ . Once training is finished, only the generator $G _ { \theta }$ will be used in inference. In the following, we will delve into the detailed architecture design of OSEDiff, as well as its associated training losses.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Network Architecture Design. Let’s denote by $E _ { \phi }$ , $\epsilon _ { \phi }$ and $D _ { \phi }$ the VAE encoder, latent diffusion network and VAE decoder of a pretrained SD model, where $\phi$ denotes the model parameters. Inspired by the recent success of LoRA [17] in finetuning SD to downstream tasks [34, 35], we adopt LoRA to fine-tune the pre-trained SD in the Real-ISR task to obtain the desired generator $G _ { \theta }$ .

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

As shown in the left part of Fig. 2, to maintain SD’s original generation capacity, we introduce trainable LoRA [17] layers to the encoder $E _ { \phi }$ and the diffusion network $\epsilon _ { \phi }$ , finetuning them into $E _ { \theta }$ and $\epsilon _ { \theta }$ with our training data. For the decoder, we fix its parameters and directly set $D _ { \theta } = D _ { \phi }$ . This is to ensure that the output space of the diffusion network remains consistent with the regularizers.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Recall that the diffusion model diffuses the input latent feature $_ z$ through $z _ { t } = \alpha _ { t } z + \beta _ { t } \epsilon$ , where $\alpha _ { t } , \beta _ { t }$ are scalars that are dependent to diffusion timestep $t \in \{ 1 , \cdots , T \}$ [16]. With a neural network that can predict the noise in ${ \boldsymbol { z } } _ { t }$ , denoted as $\hat { \epsilon }$ , the denoised latent can be obtained as $\begin{array} { r } { \hat { z } _ { 0 } = \frac { z _ { t } - \beta _ { t } \hat { \epsilon } } { \alpha _ { t } } } \end{array}$ , which is expected to be cleaner and more photo-realistic than ${ \boldsymbol { z } } _ { t }$ . Moreover, SD is a text-conditioned generation model. By extracting the text embeddings [36], denoted by $c _ { y }$ , from the given text description $y$ , the noise prediction can be performed as $\hat { \pmb { \epsilon } } = \pmb { \epsilon } _ { \theta } ( z _ { t } ; t , c _ { y } )$ .

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

We adapt the above text-to-image denoising process to the Real-ISR task, and formulate the LQ-to-HQ latent transformation $F _ { \theta }$ as a text-conditioned image-to-image denoising process as:

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

![Equation 3](../images/4624aa05fa9a035e1b151b5e25b992349ac809608a407854d0fba755e81e5991.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where we conduct only one-step denoising on the LQ latent $z _ { L }$ , without introducing any noise, at the $T$ -th diffusion timestep. The denoising output $\hat { z } _ { H }$ is expected be more photo-realistic than $z _ { L }$ . As for the text embeddings, we apply some text prompt extractor, such as the DAPE [52], to LQ input $\scriptstyle { \pmb { x } } _ { L }$ , and obtain $c _ { y } = Y ( \pmb { x } _ { L } )$ . Finally, the whole LQ-to-HQ image synthesis can be written as:

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Equation 4](../images/4ec9bc02a7218996c38a5b97c37a0dd52c8c7d547484d8c041da650f9ca5069e.jpg)
*Equation 4: Equation extracted by MinerU.*

> 💡 **Equation 4 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

As mentioned in Sec. 3.1, to improve the performance for a Real-ISR model, it is required to supervise the generator training with both the data term ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and regularization term $\mathcal { L } _ { \mathrm { r e g } }$ . As shown in the right part of Fig. 2, we propose to adapt VSD [49] as the regularization term. Apart from utilizing the SD model as the pre-trained regularizer $\epsilon _ { \phi }$ , VSD also introduces a finetuned regularizer, i.e., a latent diffusion module finetuned on the distribution $q _ { \theta } \left( \hat { \pmb x } _ { H } \right)$ of generated images with LoRA. We denote this finetuned diffusion module as $\epsilon _ { \phi ^ { \prime } }$ .

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Training Loss. We train the generator $G _ { \theta }$ with the data loss ${ \mathcal { L } } _ { \mathrm { d a t a } }$ and regularization loss $\mathcal { L } _ { \mathrm { r e g } }$ . We set ${ \mathcal { L } } _ { \mathrm { d a t a } }$ as the weighted sum of MSE loss and LPIPS loss:

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标。

![Equation 5](../images/acd537eed44dca3f7725d7100d0539643e1ce07223dc873f4f48d454cba41537.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $\lambda _ { 1 }$ is a weighting scalar. As for $\mathcal { L } _ { \mathrm { r e g } }$ , we adopt the VSD loss via:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 6](../images/44c20cedce8a911e3c05fab7c6c52185f3d4aab30b62731ee3e4125e14af0773.jpg)
*Equation 6: Equation extracted by MinerU.*

> 💡 **Equation 6 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

Given any trainable image-shape feature $_ { \textbf { \em x } }$ , its latent code $z = E _ { \phi } ( { \pmb x } )$ and encoded text prompt condition $c _ { y }$ , VSD optimizes $_ { \textbf { \em x } }$ to make it consistent with the text prompt $y$ via:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 7](../images/54b02736673efd3d527f6a25bd5050a81c085384d13fef1ec144b746435bfb6a.jpg)
*Equation 7: Equation extracted by MinerU.*

> 💡 **Equation 7 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where the expectation of the gradient is conducted over all diffusion timesteps $t \in \{ 1 , \cdots , T \}$ and $\epsilon \sim \mathcal { N } ( 0 , I )$ . Therefore, the overall training objective for the generator $G _ { \theta }$ is:

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

![Equation 8](../images/281d0f9fb937ed0f6d27df61635515afdedd540f0fdb4eaaad01ef429172ee4c.jpg)
*Equation 8: Equation extracted by MinerU.*

> 💡 **Equation 8 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

where $\lambda _ { 2 }$ is a weighting scalar. Besides, as required by VSD, the finetuned regularizer $\epsilon _ { \phi ^ { ' } }$ should also be trainable, and its training objective is:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 9](../images/c9d5b86e3fce85c331d4bf64267d58a122fa74d959e98d4f8025c9f55a2e6115.jpg)
*Equation 9: Equation extracted by MinerU.*

> 💡 **Equation 9 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

Note that the above ${ \mathcal { L } } _ { \mathrm { d i f f } }$ loss is only applied to update $\epsilon _ { \phi ^ { ' } }$ . The whole algorithm to illustrate the training pipeline can be found in the Appendix.

VSD in Latent Space. The original VSD computes the gradients in the image space. When it is used to train an SD-based generator network, there will be repetitive latent decoding/encoding in computing $\mathcal { L } _ { \mathrm { r e g } }$ . This is costly and makes the regularization less effective. Considering the fact that a well-trained VAE should satisfy $E _ { \phi } ( \pmb { x } ) = E _ { \phi } ( \bar { D _ { \phi } } ( z ) ) \approx z$ , we can approximately let $\bar { \cal E } _ { \phi } ( \hat { \pmb x } _ { H } ) = \hat { \pmb z } _ { H }$ . In this case, we can eliminate the redundant latent encoding/decoding in computing the regularization loss, as we follow DMD [58] to optimize the distribution loss in the latent state space rather than in the noise space. The gradient of the regularization loss w.r.t. the network parameter $\theta$ in the latent space is:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 10](../images/87588dbf5e5b461e9b38e6bea9a011d8344b6097123a319ca7539a1165ec899b.jpg)
*Equation 10: Equation extracted by MinerU.*

> 💡 **Equation 10 批读**: 公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把符号对应到输入、teacher/student、控制变量。

# A.4 Algorithm of OSEDiff

The pseudo-code of our OSEDiff training algorithm is summarized as Algorithm 1. We follow [49, 58] and use classifier-free guidance (cfg) when calculating $z _ { \phi }$ .The cfg value is set to 7.5, and the negative prompts we use are: "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth."

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会影响生成细节与语义一致性。

# Algorithm 1 Training Scheme of OSEDiff

Input: Training dataset $s$ , pretrained SD parameterized by $\phi$ including VAE encoder $E _ { \phi }$ , latent diffusion network $E _ { \phi }$ and VAE decoder $\epsilon _ { \phi }$ , prompt extractor $Y _ { \ l }$ , training iteration $N$   
1 Initialize $G _ { \theta }$ parameterized by $\theta$ , including $E _ { \theta }  E _ { \phi }$ with trainable LoRA $\epsilon _ { \theta } \gets \epsilon _ { \phi }$ with trainable LoRA $D _ { \theta }  D _ { \phi }$ b   
2 Initialize $\epsilon _ { \phi }  \epsilon _ { \theta }$ with trainable LoRA   
3 for $i \gets 1$ to $N$ do 4 Sample $\mathbf { \mathcal { x } } _ { L } , \mathbf { \mathcal { x } } _ { H }$ from $s$ $^ { \prime * }$ Network forward \*/ 5 $c _ { y } \gets Y ( { \pmb x } _ { L } )$ 6 $\pmb { z } _ { L } \gets E _ { \theta } ( \pmb { x } _ { L } )$ 7 $\hat { z } _ { H } \gets F _ { \theta } ( z _ { L } ; c _ { y } )$ 8 $\hat { \pmb { x } } _ { H } \gets D _ { \theta } ( \hat { \pmb { z } } _ { H } )$ $^ { \prime * }$ Compute data term objective \*/ 9 $\begin{array} { r } { \nabla _ { \theta } \mathcal { L } _ { \mathrm { d a t a } } \gets [ \mathcal { L } _ { \mathrm { M S E } } ( \hat { \pmb x } _ { H } , \pmb x _ { H } ) + \lambda _ { 1 } \mathcal { L } _ { \mathrm { L P I P S } } ( \hat { \pmb x } _ { H } , \pmb x _ { H } ) ] \frac { \partial \hat { \pmb x } _ { H } } { \partial \theta } } \end{array}$ $^ { \prime * }$ Compute regularization objective, following DMD[58] \*/   
10 Sample $\epsilon$ from $\sqrt { ( 0 , I ) }$   
11 Sample $t$ from $\{ 2 0 , \cdots , 9 8 0 \}$   
12 zˆt ← αtzˆH + σtϵ   
13 $z _ { \phi } \gets \mathrm { s t o p g r a d } ( F _ { \phi } \left( \hat { z } _ { t } ; c _ { y } \right) )$   
14 $z _ { \phi ^ { \prime } } \gets \mathrm { s t o p g r a d } ( F _ { \phi ^ { \prime } } \left( \hat { z } _ { t } ; c _ { y } \right) )$   
15 $\omega  1 / \mathrm { m e a n } ( \| z _ { \phi } - \hat { z } _ { H } \| )$   
16 $\begin{array} { r } { \nabla _ { \theta } \mathcal { L } _ { \mathrm { r e g } }  [ \omega ( { z } _ { \phi ^ { \prime } } - { z } _ { \phi } ) ] \frac { \partial \hat { { z } } _ { H } } { \partial \theta } } \end{array}$ $^ { \prime * }$ Compute regularizer funetuning objective \*/   
17 Sample $\epsilon$ from $\sqrt { ( 0 , I ) }$   
18 Sample $t$ from $\{ 1 , \cdots , T \}$   
19 $\boldsymbol { z } _ { t } \gets \boldsymbol { \alpha } _ { t }$ st $\mathrm { { o p g r a d } } ( \widehat { \pmb { z } } _ { H } ) + \sigma _ { t } \epsilon$   
20 $\mathcal { L } _ { \mathrm { d i f f } }  \mathcal { L } _ { \mathrm { M S E } } ( \epsilon _ { \phi ^ { \prime } } ( z _ { t } ; t , c _ { y } ) , \epsilon )$ $^ { \prime * }$ Network Parameter Update \*/   
21 Update $\theta$ with $\mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { r e g } }$   
22 Update $\phi ^ { \prime }$ with ${ \mathcal { L } } _ { \mathrm { d i f f } }$   
23 end Output: Generator $G _ { \theta }$ including VAE encoder $E _ { \theta }$ , latent diffusion network $E _ { \theta }$ and VAE decoder $\epsilon _ { \theta }$   
References   
[1] Stability.ai. https://stability.ai/stable-diffusion.   
[2] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 126–135, 2017.   
[3] Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3086–3095, 2019.   
[4] Chaofeng Chen, Xinyu Shi, Yipeng Qin, Xiaoming Li, Xiaoguang Han, Tao Yang, and Shihui Guo. Real-world blind super-resolution via feature matching with implicit high-resolution priors. In Proceedings of the 30th ACM International Conference on Multimedia, pages 1329–1338, 2022.   
[5] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12299–12310, 2021.   
[6] Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong. Activating more pixels in image super-resolution transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22367–22377, 2023.   
[7] Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong, Xiaokang Yang, and Fisher Yu. Dual aggregation transformer for image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12312–12321, 2023.   
[8] Imre Csiszár. I-divergence geometry of probability distributions and minimization problems. The annals of probability, pages 146–158, 1975.   
[9] Tao Dai, Jianrui Cai, Yongbing Zhang, Shu-Tao Xia, and Lei Zhang. Second-order attention network for single image super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11065–11074, 2019.   
[10] Trung Dao, Thuan Hoang Nguyen, Thanh Le, Duc Vu, Khoi Nguyen, Cuong Pham, and Anh Tran. Swiftbrush v2: Make your one-step diffusion model better than its teacher. arXiv preprint arXiv:2408.14176, 2024.   
[11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.   
[12] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5):2567–2581, 2020.   
[13] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a deep convolutional network for image super-resolution. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part IV 13, pages 184–199. Springer, 2014.   
[14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.   
[15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
[17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.   
[18] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and superresolution. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 694–711. Springer, 2016.   
[19] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.   
[20] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in Neural Information Processing Systems, 35:23593–23606, 2022.   
[21] Bahjat Kawar, Gregory Vaksman, and Michael Elad. Snips: Solving noisy inverse problems stochastically. Advances in Neural Information Processing Systems, 34:21757–21769, 2021.   
[22] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5148–5157, 2021.   
[23] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.   
[24] Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photo-realistic single image superresolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4681–4690, 2017.   
[25] Kyungmin Lee, Kihyuk Sohn, and Jinwoo Shin. Dreamflow: High-quality text-to-3d generation by approximating probability flow. arXiv preprint arXiv:2403.14966, 2024.   
[26] Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al. Lsdir: A large scale dataset for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1775–1787, 2023.   
[27] Jie Liang, Hui Zeng, and Lei Zhang. Details or artifacts: A locally discriminative learning approach to realistic image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5657–5666, 2022.   
[28] Jie Liang, Hui Zeng, and Lei Zhang. Efficient and degradation-adaptive network for real-world image super-resolution. In European Conference on Computer Vision, pages 574–591. Springer, 2022.   
[29] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833–1844, 2021.   
[30] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 136–144, 2017.   
[31] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Ben Fei, Bo Dai, Wanli Ouyang, Yu Qiao, and Chao Dong. Diffbir: Towards blind image restoration with generative diffusion prior. arXiv preprint arXiv:2308.15070, 2023.   
[32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.   
[33] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[34] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.   
[35] Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036, 2024.   
[36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
[37] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.   
[38] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.   
[39] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.   
[40] Lingchen Sun, Rongyuan Wu, Zhengqiang Zhang, Hongwei Yong, and Lei Zhang. Improving the stability of diffusion models for content consistent super-resolution. arXiv preprint arXiv:2401.00877, 2023.   
[41] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 2555–2563, 2023.   
[42] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. International Journal of Computer Vision, pages 1–21, 2024.   
[43] Peihao Wang, Dejia Xu, Zhiwen Fan, Dilin Wang, Sreyas Mohan, Forrest Iandola, Rakesh Ranjan, Yilei Li, Qiang Liu, Zhangyang Wang, et al. Taming mode collapse in score distillation for text-to-3d generation. arXiv preprint arXiv:2401.00909, 2023.   
[44] Xintao Wang, Yu Li, Honglun Zhang, and Ying Shan. Towards real-world blind face restoration with generative facial prior. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9168–9178, 2021.   
[45] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind superresolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1905–1914, 2021.   
[46] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy. Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops, pages 0–0, 2018.   
[47] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022.   
[48] Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang, Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C Kot, and Bihan Wen. Sinsr: Diffusion-based image super-resolution in a single step. arXiv preprint arXiv:2311.14760, 2023.   
[49] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024.   
[50] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.   
[51] Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image super-resolution. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16, pages 101–117. Springer, 2020.   
[52] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. Seesr: Towards semantics-aware real-world image super-resolution. arXiv preprint arXiv:2311.16518, 2023.   
[53] Liangbin Xie, Xintao Wang, Xiangyu Chen, Gen Li, Ying Shan, Jiantao Zhou, and Chao Dong. Desra: Detect and delete the artifacts of gan-based real-world super-resolution models. 2023.   
[54] Jianchao Yang, John Wright, Thomas S Huang, and Yi Ma. Image super-resolution via sparse representation. IEEE transactions on image processing, 19(11):2861–2873, 2010.   
[55] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1191–1200, 2022.   
[56] Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang. Gan prior embedded network for blind face restoration in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 672–681, 2021.   
[57] Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang. Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization. arXiv preprint arXiv:2308.14469, 2023.   
[58] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024.   
[59] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. arXiv preprint arXiv:2401.13627, 2024.   
[60] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image super-resolution by residual shifting. arXiv preprint arXiv:2307.12348, 2023.   
[61] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4791–4800, 2021.   
[62] Lin Zhang, Lei Zhang, and Alan C Bovik. A feature-enriched completely blind image quality evaluator. IEEE Transactions on Image Processing, 24(8):2579–2591, 2015.   
[63] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.   
[64] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.   
[65] Xindong Zhang, Hui Zeng, Shi Guo, and Lei Zhang. Efficient long-range attention network for image super-resolution. In European Conference on Computer Vision, pages 649–667. Springer, 2022.   
[66] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. Image super-resolution using very deep residual channel attention networks. In Proceedings of the European conference on computer vision (ECCV), pages 286–301, 2018.   
[67] Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu. Residual dense network for image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2472–2481, 2018.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Figure 5](../images/00475426de6c9eed14b18b2b72c08a769b3b2a2584d66e0ab357e4f188c8ec55.jpg)
*Figure 5: Figure 5: Qualitative comparisons between OSEDiff and GAN-based Real-ISR methods. Please zoom in for a better view.*

> 💡 **Figure 5 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

![Figure 6](../images/12950d670f4f8f65324e63bd1637848ad9eaecdbb7323627cb36c72b7ceb84ec.jpg)
*Figure 6: Figure 6: The LQ images used in user study and the voting results.*

> 💡 **Figure 6 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

![Figure 0](../images/8ac01ab7b29f8374b9659ec9254bfe1938c7da8653871484069bd534fbf2389c.jpg)
*Figure 0: (b) The voting results are from 15 volunteers. The corresponding percentages and numerical counts are displayed with the pie chart.*

> 💡 **Figure 0 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

(b) The voting results are from 15 volunteers. The corresponding percentages and numerical counts are displayed with the pie chart.

![Figure 7](../images/1b16e1c3a9a0ce5bc5149cc3eefeb7fa5c6f9b6ce87e75f73d585ef2d502d287.jpg)
*Figure 7: Figure 7: More visulization comparisons of different DM-based Real-ISR methods. Please zoom in for a better view.*

> 💡 **Figure 7 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

---

## 🔖 Section 总结

### 核心洞察
1. 明确单步路径、teacher/student 或控制变量。
2. 区分训练时机制和推理时机制。
3. 关注 loss 与模块分别解决哪个瓶颈。
