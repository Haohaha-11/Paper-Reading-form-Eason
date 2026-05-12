# Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution

Tianyi Zhang1 Zheng-Peng Duan1 Peng-Tao Jiang2 Bo Li2 Ming-Ming Cheng1 Chun-Le Guo1,3 \* Chongyi Li1,3 1VCIP, CS, Nankai University 2vivo Mobile Communication Co. Ltd 3NKIARI, Shenzhen Futian

{zty557,adamduan0211}@mail.nankai.edu.cn, {pt.jiang, librad}@vivo.com, {cmm, guochunle, lichongyi}@nankai.edu.cn

# Abstract

Diffusion-based real-world image super-resolution (Real-ISR) methods have demonstrated impressive performance. To achieve efficient Real-ISR, many works employ Variational Score Distillation (VSD) to distill pre-trained stablediffusion (SD) model for one-step SR with a fixed timestep. However, since SD will perform different generative priors at different timesteps, a fixed timestep is difficult for these methods to fully leverage the generative priors in SD, leading to suboptimal performance. To address this, we propose a Time-Aware one-step Diffusion Network for Real-ISR (TADSR). We first introduce a Time-Aware VAE Encoder, which projects the same image into different latent features based on timesteps. Through joint dynamic variation of timesteps and latent features, the student model can better align with the input pattern distribution of the pretrained SD, thereby enabling more effective utilization of SD’s generative capabilities. To better activate the generative prior of SD at different timesteps, we propose a Time-Aware VSD loss that bridges the timesteps of the student model and those of the teacher model, thereby producing more consistent generative prior guidance conditioned on timesteps. Additionally, though utilizing the generative prior in SD at different timesteps, our method can naturally achieve controllable trade-offs between fidelity and realism by changing the timestep. Experimental results demonstrate that our method achieves both state-ofthe-art performance and controllable SR results with only a single step. The source codes are released at https: //github.com/zty557/TADSR

# 1. Introduction

Real-World Image Super-Resolution (Real-ISR) aims to restore high-quality (HQ) images from low-quality (LQ) inputs degraded by complex and unknown factors in realworld scenarios, which has recently attracted increasing attention [9, 16, 25, 35, 37, 40]. To address a broader spectrum of degradation types and achieve more realistic results, many researchers have turned to generative models, particularly diffusion models [11]. Consequently, several works have explored leveraging the generative priors in pretrained Stable Diffusion (SD) models [20] to tackle Real-ISR, yielding impressive results [9, 18, 22, 24, 33, 34]. Nevertheless, the iterative denoising process inherent in diffusion models introduces significant computational overhead

![](images/c2a5170edc60d008e97b636e61f434363c573745785375a628ed0f3b1440fafb.jpg)
Figure 1. (a) Comparison between our TADSR(Ours) and PisaSR [21]. In PisaSR, increasing the semantic weight $\lambda _ { s e m }$ leads to restore more realistic images. As the timestep condition $t$ increases, our model recovers a more realistic parrot image. In contrast, PisaSR shows only an increase in sharpness as $\lambda _ { s e m }$ increases. (b) The input image and the corresponding outputs of the SD at different timesteps $t$ . The outputs vary significantly across different timesteps, reflecting distinct generative priors.

and latency.

To overcome these limitations, some researchers have focused on distilling SD into an efficient one-step model for Real-ISR [3, 8, 21, 32, 38, 39]. Specifically, OSEDiff [32] first leverages the Variational Score Distillation (VSD) loss [29] to distill the SD model, enabling realistic image reconstruction in a single step. Subsequently, S3Diff [39], PisaSR [21], AdcSR [3], and TSDSR [8] also adopt distillation-based approaches to develop SD-based one-step Real-ISR models, using either adversarial loss or modified VSD loss. They typically use a pre-trained SD with trainable LoRA modules as the student model to perform Real-ISR, while employing a fixed-weight SD as the teacher to provide generative guidance.

However, these methods generally fix the timestep (e.g. step 999) injected into the student model while randomly sampling the timestep injected into the teacher model, which prevents them from effectively leveraging the generative prior in SD. Specifically, as shown in Figure 1(b), when the timestep $t$ equals 100, most of the image information is preserved, and the teacher’s output only differs in texture details. As the $t$ grows to 300, the teacher model increasingly activates leaf-related generative priors to predict content that has been obscured by noise. However, with $t$ increasing further to 600, most of the image information is lost, and the teacher model can only recover the overall structure and color of the leaves. This observation indicates that the pretrained SD model exhibits different generative priors at different timesteps. Therefore, existing methods generally suffer from the following two problems: (1) The fixed timestep injected into the student model fails to fully leverage the generative priors at different timesteps in the pretrained SD model; (2) The randomly sampled timestep injected into the teacher model makes it difficult to provide consistent generative guidance. As a result, as shown in Figure 1(a), although we increase the semantic weight $\lambda _ { s e m }$ in PisaSR [21], it only produces enhanced sharpness without significantly enriching the semantic content. In contrast, our method gradually generates a more realistic parrot as the timestep increases.

In the end, we propose Time-Aware One Step Diffusion Network for Super-Resolution (TADSR), a framework that more effectively distills the generative prior of SD at different timesteps into a one-step diffusion model for Real-ISR. To address the first limitation and better exploit the generative priors at different timesteps, two conditions need to be satisfied: (1) the student model should receive randomly sampled timesteps; (2) the latent features fed into the student model should vary with the timestep, reflecting the noise-level changes in SD. Therefore, we incorporate a Time-Aware VAE Encoder (TAE), which introduces a time embedding layer into the VAE encoder to map the same image to different latent representations based on the timestep. To address the second limitation and ensure consistent generative guidance, we propose a Time-Aware Variational Score Distillation (TAVSD) Loss, which associates the timestep injected into the student model with the one used in the teacher model through a mapping function. When the student model is conditioned on a larger timestep, the teacher receives a latent image corrupted with stronger noise, providing guidance that emphasizes stronger semantic generation in the reconstruction results. Conversely, a smaller timestep leads to similar results with reconstruction, primarily enhancing texture details. Therefore, TAVSD can provide a more consistent generative guidance condition on the injected timestep in the student model. Our contributions are summarized as follows:

• We propose TADSR, a Time-Aware One-Step Diffusion Network for Real-ISR, which naturally leverages the generative priors of SD at different timesteps to achieve controllable trade-offs between fidelity and realism in Real-ISR. Our TADSR achieves superior performance compared with other SD-based Real-ISR methods on both real-world and synthetic datasets. • We propose a Time-Aware VAE Encoder (TAE), which maps the same image to different latent representations based on the timestep, enabling the student model to fully exploit the generative priors at various timesteps. We propose a Time-Aware Variational Score Distillation (TAVSD) Loss, which aligns the timestep of the student and teacher models, providing consistent generative guidance at different timesteps.

# 2. Related Work

# 2.1. Real-World Image Super-Resolution

Traditional Image Super-Resolution (ISR) methods [5–7, 15, 17, 42, 43] typically degrade HQ images using simple downsampling operations to construct HQ-LQ image pairs for training. However, these approaches struggle to handle images degraded by complex real-world processes. To better simulate the unknown and complex degradations in real-world scenarios, several studies [25, 40] have proposed more sophisticated degradation pipelines to synthesize LQ data. Specifically, BSRGAN [40] introduces a random combination of basic degradation operations (e.g., downsampling, blurring, noise) injection, with varying intensities to generate realistic HQ-LQ pairs. Real-ESRGAN [25] proposes a second-order degradation scheme to cover a broader range of degradation types. In addition, inspired by Generative Adversarial Networks (GANs), researchers have adopted adversarial losses to encourage the reconstruction of more realistic images. Although these GAN-based methods can produce richer texture details compared to traditional approaches, they are often unstable to train and prone to generating unnatural artifacts [32].

![](images/2427aacd722872df882a8c7bdfb4d61409ab1ca2201ee565c4c367d4cf6a5a27.jpg)
Figure 2. Overview of TADSR. We train a Student Model $G _ { \theta }$ to perform one-step Real-ISR, which consists of a Time-Aware VAE Encoder $E _ { \theta }$ and a UNet $F _ { \theta }$ . We randomly sample a timestep $t _ { s }$ and map it to $t _ { v }$ . The $t _ { s }$ and the LQ image are fed into the encoder $E _ { \theta }$ to obtain the LQ latent. Then, $t _ { s }$ and the LQ latent are fed into the UNet $F _ { \theta }$ to produce the reconstructed latent feature $\hat { z } _ { 0 }$ . After adding noise to $\hat { z } _ { 0 }$ corresponding to $t _ { v }$ , we feed it and $t _ { v }$ into the teacher model and the LoRA model to compute the TAVSD loss (orange flow). The reconstruction loss (blue flow) in pixel space and TAVSD loss is then used to jointly update the student model $G _ { \theta }$ . For the LoRA Model, we employ the diffusion loss (green flow) for training.

# 2.2. Diffusion-Based Real-ISR

Recently, many researchers have leveraged the powerful generative priors of pre-trained diffusion models for Real-ISR tasks to achieve realistic image reconstruction. For example, StableSR [24] conditions the diffusion process on LQ images by injecting them through a learnable timeaware encoder into the SD model, enabling strong detail generation capabilities. DiffBIR [18] utilizes ControlNet to extract structural information from LQ images to better guide the generative prior of SD for SR. PASD [22] and SeeSR [33] extract semantic information from LQ inputs and inject it into SD, resulting in more realistic outputs. Although these approaches yield impressive results, the multistep denoising process leads to high computational and time costs. To accelerate diffusion-based Real-ISR, OSEDiff introduces the VSD loss to distill the pre-trained SD model, enabling realistic image reconstruction in a single step. S3Diff further adopts degradation-guided LoRA adapters combined with adversarial training to achieve one-step SR. PisaSR trains two separate LoRA adapters for pixel-level and semantic-level guidance, allowing controllable tradeoffs between realism and fidelity.

However, these methods overlook the varying generative capabilities of SD in different timesteps and train with a fixed timestep. Our work aims to fully exploit these timedependent generative prior to achieve superior SR performance and a natural balance between fidelity and realism.

# 3. Methodology

# 3.1. Problem Definition

Real-ISR aims to reconstruct HQ images $x _ { H }$ from LQ images $x _ { L }$ that suffer from complex and unknown degradations. With the advancement of deep learning, researchers have commonly adopted neural networks $G _ { \theta }$ to estimate the HQ images and optimize the network through loss functions. The general form of the loss function is:

$$
\begin{array} { r l } & { \theta ^ { * } = \arg \underset { \theta } { \operatorname* { m i n } } \mathbb { E } _ { ( { x _ { L } } , { x _ { H } } ) \sim \mathcal { D } } [ \mathcal { L } _ { R e c } ( G _ { \theta } ( x _ { L } ) , x _ { H } ) } \\ & { \qquad + \lambda \mathcal { L } _ { R e g } ( G _ { \theta } ( x _ { L } ) ) ] , } \end{array}
$$

where the $\mathcal { L } _ { R e c }$ denotes the reconstruction loss to optimize the fidelity of the reconstructed results. $\mathcal { L } _ { R e g }$ is the regression loss to enhance the realism of the results, and $\lambda$ is a hyperparameter to balance $\mathcal { L } _ { R e c }$ and $\mathcal { L } _ { R e g }$ .

Recently, with the advancement of diffusion models, several studies [21, 32] have leveraged the generative prior in pre-trained SD and adopted VSD as a regression objective. VSD is designed to align the distribution of generated (fake) images with that of real images. Specifically, a pretrained teacher model trained on real images is used to estimate the score of the real image distribution. In addition, a diffusion model trained on the generator’s outputs provides an estimate of the score of the fake distribution. The generator is then optimized by minimizing the discrepancy between these two score estimates, thereby encouraging it to produce samples that are indistinguishable from real images. The formation of VSD is:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { V S D } ( \hat { z } , c ) = \mathbb { E } _ { t , \epsilon } \left[ \omega ( t ) \left( \epsilon _ { \boldsymbol { \psi } } ( \hat { z } _ { t } ; t , c ) - \epsilon _ { \boldsymbol { \phi } } ( \hat { z } _ { t } ; t , c ) \right) \frac { \partial \hat { z } } { \partial \boldsymbol { \theta } } \right] ,
$$

where $\epsilon _ { \psi }$ is the pre-trained diffusion model (teacher model) to estimate the real score, $\epsilon _ { \phi }$ represents its replica with trainable LoRA (LoRA model) to predict the fake score, and $c$ is a text embedding of a caption describing the input image. $\hat { z } = G _ { \theta } ( x _ { L } )$ is the output of the student network $G _ { \theta }$ , and $\hat { z } _ { t } = \alpha _ { t } \hat { z } + \beta _ { t } \epsilon$ is the noisy latent input. $\epsilon$ is the gaussian noise, and $\alpha _ { t }$ and $\beta _ { t }$ are the scale parameters in diffusion.

Formally, the VSD loss can be viewed as the residual between the noise outputs of the teacher model and the LoRA model, which can be transformed to the residual of their predicted latent images through $\begin{array} { r } { z = \frac { z _ { t } - \beta _ { t } \epsilon } { \alpha _ { t } } } \end{array}$ . In practice, the gradient of the VSD loss is applied only to the generator and does not propagate through the teacher model. Therefore, the loss can be expressed in the following form:

$$
\begin{array} { r l } & { \mathcal { L } _ { V S D } ( \theta ) = \mathbb { E } _ { t , \epsilon } [ \omega ^ { \prime } ( t ) \| G _ { \theta } ( x _ { L } ) - G _ { \theta ^ { - } } ( x _ { L } ) } \\ & { \qquad - z _ { \psi ^ { - } } ( \hat { z } _ { t } ; t , c ) + z _ { \phi ^ { - } } ( \hat { z } _ { t } ; t , c ) \| _ { 2 } ^ { 2 } ] , } \end{array}
$$

where $^ -$ denotes stopgrad(), $z _ { \psi }$ and $z _ { \phi }$ denote the predicted latent image of the teacher model and LoRA model, respectively. Therefore, we can decode the latent images predicted by the teacher model and the LoRA model into the image space to analyze the guidance provided by VSD.

# 3.2. Overview

As illustrated in Figure 2, we distill a Student Model $G _ { \theta }$ to perform one-step Real-ISR, which consists of a trainable Time-Aware Encoder $E _ { \theta }$ and a LoRA finetuned diffusion UNet $F _ { \theta }$ . Following the OSEDiff [32], we use a pretrained SD model as the Teacher Model $\epsilon _ { \psi }$ and its replica with trainable LoRA as the LoRA Model to obtain the fake score. First, we sample an HQ-LQ image pair from the dataset and a timestep $t _ { s }$ from a uniform distribution in the range of 0 to 999. Then, both the LQ image and $t _ { s }$ are fed into Student Model $G _ { \theta }$ to obtain the latent output $\hat { z } _ { 0 }$ . We decode $\hat { z } _ { 0 }$ into pixel space and compute the reconstruction loss with the HQ image. In the latent space, we map the $t _ { s }$ to another timestep $t _ { v }$ , and add noise corresponding to timestep $t _ { v }$ to the $\hat { z } _ { 0 }$ to obtain $\hat { z } _ { t _ { v } }$ . Then, we feed $\hat { z } _ { t _ { v } }$ and $t _ { v }$ into both the teacher model and the LoRA model to compute the Time-Aware Variational Score Distillation loss $\mathcal { L } _ { T A V S D }$ to enhance the realism. Consistent with Figure 1(b), when $t _ { v }$ is small, the gradients produced by the TAVSD loss are relatively small and mainly reflect texture details. In contrast, when $t _ { v }$ is large, the gradients become significantly larger and provide more semantic guidance. Therefore, the

![](images/e82c8f2e92b65d8ebf8b915aea5592da0644572e7d0bfe5b345bed1dc012caf6.jpg)
Figure 3. PCA visualization of latent features produced by TAE under different timesteps $t _ { s }$ , and the corresponding mean and standard deviation (Std) of latent features. TAE can encode the same image into distinct latent features conditioned on different timesteps, which aligns with the synchronized variation between timesteps and latent features in the pre-trained SD.

TAVSD loss can offer more consistent gradient guidance condition on $t _ { s }$ , enabling better distillation of the teacher model. In addition, the LoRA model is trained with a diffusion loss using the data generated by the student model.

# 3.3. Time-Aware VAE Encoder

In the original SD, as the timestep increases, the input latent features are injected with more noise, thereby activating different generative priors to produce the image. Considering the importance of the timestep in the diffusion process, multi-step SD-based Real-ISR methods often take the timestep as a condition along with the LR image to control the SR process — as seen in the time-aware encoder of StableSR and the ControlNet used in SeeSR and DiffBIR.

However, since multi-step denoising iterations are not required, existing one-step diffusion-based Real-ISR methods generally overlook the role of the timestep and train with only a fixed timestep, making it difficult to fully exploit the generative priors in SD at different timesteps. A straightforward improvement is to randomly sample timestep during training. However, since the original VAE encoder maps the same image to a single latent distribution regardless of the timestep, it is difficult for the diffusion network to effectively activate different generative priors based solely on the timestep, given the same latent input. In the original diffusion process, variations in the timestep reflect changes in the noise level of the latent distribution. We argue that a onestep SD-based Real-ISR network should exhibit a similar property in order to fully leverage the corresponding generative priors. Clearly, directly injecting noise into the latent distribution according to the timestep is inappropriate, as it would compromise reconstruction fidelity.

Therefore, we propose a Time-Aware VAE Encoder (TAE) to better utilize the generative priors in SD. By incorporating a temporal embedding layer into the VAE encoder, TAE encodes the input image into different latent distributions conditioned on the timestep $t _ { s }$ , enabling synchronized variation between $t _ { s }$ and latent distribution, thus better activating the generative priors at different timesteps within SD. This process can be formulated as:

$$
z _ { L } = E _ { \theta } ( x _ { L } , t _ { s } ) , \hat { z } = F _ { \theta } ( z _ { L } , t _ { s } ) ,
$$

where $E _ { \theta }$ is the TAE model and $F _ { \theta }$ is the Unet model.

As shown in Figure 3, TAE encodes the same input image into different latent feature condition on the timestep $t _ { s }$ . Overall, as the $t _ { s }$ increases, both the mean and variance of the latent features show a decreasing trend. After visualizing the latent feature via PCA dimensionality reduction, we can also clearly observe the changes in the latent space.

# 3.4. Time-Aware Variational Score Distillation

Following the OSEDiff [32], the VSD loss has been widely adopted in SD-based one-step Real-ISR methods to enhance the realism of reconstruction results. In one-step image generation tasks, the timestep is usually sampled randomly to distill the full generative prior of the teacher model. However, we find that in Real-ISR, this paradigm instead leads to inconsistent generative guidance, since SD exhibits different generative priors across timesteps, while the timestep sampling in the teacher model is completely random and independent of the student model.

As discussed in Section 3.1, due to the stop-gradient operation, the guidance of the VSD loss can be interpreted as the residual between the latent images predicted by the teacher model and the LoRA model. Therefore, we decode these latent images into the pixel space and analyze the guidance of the VSD loss at different timesteps. As illustrated in Figure 4, first, the mean and standard deviation of the VSD loss exhibit a clear upward trend as the timestep increases. Besides, we observe that at $t _ { v }$ equals 100, the outputs of the teacher and LoRA models are similar, and the gradients mainly reflect enhancements in texture details. In contrast, when $t _ { v }$ increases to 300, the teacher model’s output contains significantly more semantic information while the LoRA model’s output remains smooth, and the gradients reflect global semantic guidance. However, when $t _ { v }$ increases to 600, the teacher model can only recover coarse color and structural information from the noisy latent input, making it difficult to provide meaningful guidance. This implies that the VSD loss provides distinct guidance for the same image depending on $t _ { v }$ . Such opposing directional guidance creates conflicting optimization signals for the student model, leading to suboptimal convergence.

This phenomenon arises because most of the image information is preserved when $t$ is small, and both the teacher and LoRA models focus mainly on generating texture details. As $t$ increases, the injection of more noise gradually obscures the underlying image content, forcing the teacher model to rely more heavily on its generative prior. However, since the LoRA model is trained on low-quality data generated by the student model and does not employ the CFG strategy [10], its outputs tend to be overly smooth. This also explains why the VSD loss can make images more realistic while avoiding over-smoothing — since the LoRA model tends to produce smoother outputs than the teacher model, their residuals naturally provide high-frequency details that guide the generation process.

![](images/0a3b7731986f50302ad70be7c37195b93d99c41ef7a0a92e8c8d286760bbd07f.jpg)
Figure 4. (a) Mean and standard deviation (Std) of the VSD loss at different timesteps. (b) The outputs of the teacher model and the LoRA model are decoded into pixel space and gradients in latent space at different timesteps $t$ .

Considering that the guidance provided by VSD varies across different timesteps, we establish a connection between the timestep $t _ { s }$ input to the student model and the timestep $t _ { v }$ in the teacher model, so that the VSD loss can provide more consistent gradient guidance conditioned on $t _ { s }$ . Specifically, we feed the randomly sampled $t _ { s }$ and the LQ image into the student model to obtain $\hat { z } = G _ { \theta } ( x _ { L } , t _ { s } )$ . Then, $t _ { s }$ is mapped to $t _ { v }$ by:

$$
t _ { v } = \lambda t _ { s } + \gamma , t _ { s } \in [ 0 , 9 9 9 ] , t _ { v } \in [ 0 , 9 9 9 ]
$$

where the $\lambda$ and $\gamma$ are the hyperparameters.

We then add noise to $\hat { z }$ corresponding to $t _ { v }$ to obtain $\hat { z } _ { t _ { v } } = \alpha _ { t _ { v } } \hat { z } + \beta _ { t _ { v } } \epsilon$ , which is fed to the teacher model and the LoRA model with $t _ { v }$ to compute the TAVSD loss:

$$
\begin{array} { c } { { \nabla _ { \theta } \mathcal { L } _ { T A V S D } \big ( \hat { z } , c , t _ { v } \big ) = \mathbb { E } _ { \epsilon } [ \omega ( t _ { v } ) ( \epsilon _ { \psi } \big ( \hat { z } _ { t _ { v } } ; t _ { v } , c \big ) } } \\ { { - \epsilon _ { \phi } \big ( \hat { z } _ { t _ { v } } ; t _ { v } , c \big ) \big ) \displaystyle \frac { \partial \hat { z } } { \partial \theta } ] . } } \end{array}
$$

By leveraging the TAVSD loss, the model can naturally balance generation and fidelity in the Real-ISR task simply by varying the timestep condition $t _ { s }$ input to the model.

Table 1. A comprehensive evaluation against state-of-the-art methods across synthetic and real-world datasets. The top-performing and runner-up results under each metric are marked in red and blue, respectively.

<table><tr><td>Datasets</td><td>Metrics</td><td>StableSR</td><td>DiffBIR</td><td>SeeSR</td><td>SinSR</td><td>S3Diff</td><td>OSEDiff</td><td>PisaSR</td><td>TSDSR</td><td>AdcSR</td><td>TADSR</td></tr><tr><td rowspan="8">DIV2k-Val</td><td>PSNR ↑</td><td>23.261</td><td>23.409</td><td>23.679</td><td>24.417</td><td>23.530</td><td>23.723</td><td>23.867</td><td>22.17</td><td>23.743</td><td>23.815</td></tr><tr><td>SSIM ↑</td><td>0.5726</td><td>0.5732</td><td>0.6043</td><td>0.6023</td><td>0.5933</td><td>0.6109</td><td>0.6058</td><td>0.5602</td><td>0.6017</td><td>0.6028</td></tr><tr><td>LPIPS ↓</td><td>0.3113</td><td>0.3456</td><td>0.3194</td><td>0.3235</td><td>0.2581</td><td>0.2942</td><td>0.2823</td><td>0.2736</td><td>0.2853</td><td>0.3078</td></tr><tr><td>CLIPIQA ↑</td><td>0.6771</td><td>0.7082</td><td>0.6935</td><td>0.6505</td><td>0.7001</td><td>0.6682</td><td>0.6928</td><td>0.7149</td><td>0.6763</td><td>0.7353</td></tr><tr><td>MUSIQ ↑</td><td>65.918</td><td>68.396</td><td>68.665</td><td>62.838</td><td>67.923</td><td>67.971</td><td>69.681</td><td>70.65</td><td>67.995</td><td>69.649</td></tr><tr><td>MANIQA ↑</td><td>0.6174</td><td>0.6355</td><td>0.6222</td><td>0.5392</td><td>0.6311</td><td>0.6132</td><td>0.6375</td><td>0.6077</td><td>0.6073</td><td>0.6443</td></tr><tr><td>TOPIQ ↑</td><td>0.5979</td><td>0.6344</td><td>0.6856</td><td>0.5721</td><td>0.6334</td><td>0.6188</td><td>0.6619</td><td>0.6672</td><td>0.6526</td><td>0.7044</td></tr><tr><td>QALIGN ↑</td><td>3.5273</td><td>3.8774</td><td>3.9765</td><td>3.5159</td><td>3.8666</td><td>3.8357</td><td>3.8812</td><td>3.927</td><td>3.612</td><td>4.0783</td></tr><tr><td rowspan="8">DRealSR</td><td>PSNR ↑</td><td>28.030</td><td>25.929</td><td>28.073</td><td>28.345</td><td>27.539</td><td>27.915</td><td>28.318</td><td>26.20</td><td>28.099</td><td>28.387</td></tr><tr><td>SSIM ↑</td><td>0.7536</td><td>0.6526</td><td>0.7684</td><td>0.7491</td><td>0.7491</td><td>0.7833</td><td>0.7804</td><td>0.7170</td><td>0.7726</td><td>0.7758</td></tr><tr><td>LPIPS ↓</td><td>0.3284</td><td>0.4518</td><td>0.3173</td><td>0.3697</td><td>0.3109</td><td>0.2968</td><td>0.2960</td><td>0.3116</td><td>0.3046</td><td>0.3235</td></tr><tr><td>CLIPIQA ↑</td><td>0.6356</td><td>0.6863</td><td>0.6909</td><td>0.6375</td><td>0.7131</td><td>0.6974</td><td>0.6971</td><td>0.7309</td><td>0.7049</td><td>0.7398</td></tr><tr><td>MUSIQ ↑</td><td>58.512</td><td>65.667</td><td>65.080</td><td>55.009</td><td>63.941</td><td>64.691</td><td>66.113</td><td>66.12</td><td>66.266</td><td>67.016</td></tr><tr><td>MANIQA ↑</td><td>0.5594</td><td>0.6289</td><td>0.6051</td><td>0.4894</td><td>0.6124</td><td>0.5903</td><td>0.6160</td><td>0.5820</td><td>0.5915</td><td>0.6309</td></tr><tr><td>TOPIQ ↑</td><td>0.5323</td><td>0.6149</td><td>0.6575</td><td>0.5122</td><td>0.6040</td><td>0.6002</td><td>0.6333</td><td>0.6251</td><td>0.6527</td><td>0.6758</td></tr><tr><td>QALIGN ↑</td><td>3.0614</td><td>3.6011</td><td>3.5882</td><td>3.0982</td><td>3.6148</td><td>3.5450</td><td>3.5838</td><td>3.6928</td><td>3.6520</td><td>3.7491</td></tr><tr><td rowspan="8">RealSR</td><td>PSNR ↑</td><td>24.645</td><td>24.240</td><td>25.149</td><td>26.266</td><td>25.183</td><td>25.148</td><td>25.503</td><td>23.404</td><td>25.469</td><td>25.166</td></tr><tr><td>SSIM ↑</td><td>0.7080</td><td>0.6649</td><td>0.7211</td><td>0.7341</td><td>0.7269</td><td>0.7338</td><td>0.7418</td><td>0.6886</td><td>0.7301</td><td>0.7150</td></tr><tr><td>LPIPS ↓</td><td>0.3002</td><td>0.3470</td><td>0.3007</td><td>0.3241</td><td>0.2721</td><td>0.2920</td><td>0.2672</td><td>0.2805</td><td>0.2885</td><td>0.3168</td></tr><tr><td>CLIPIQA ↑</td><td>0.6234</td><td>0.6959</td><td>0.6699</td><td>0.6153</td><td>0.6731</td><td>0.6687</td><td>0.6699</td><td>0.7199</td><td>0.6731</td><td>0.7283</td></tr><tr><td>MUSIQ ↑</td><td>65.883</td><td>68.340</td><td>69.819</td><td>60.575</td><td>65.824</td><td>69.087</td><td>70.147</td><td>70.7710</td><td>69.899</td><td>71.182</td></tr><tr><td>MANIQA ↑</td><td>0.6230</td><td>0.6530</td><td>0.6450</td><td>0.5409</td><td>0.6427</td><td>0.6337</td><td>0.6551</td><td>0.6311</td><td>0.6353</td><td>0.6715</td></tr><tr><td>TOPIQ ↑</td><td>0.5748</td><td>0.6052</td><td>0.6890</td><td>0.5188</td><td>0.6162</td><td>0.6251</td><td>0.6374</td><td>0.6642</td><td>0.6793</td><td></td></tr><tr><td>QALIGN ↑</td><td>3.2767</td><td>3.6313</td><td>3.7190</td><td>3.1889</td><td>3.6638</td><td>3.6915</td><td>3.6355</td><td>3.7748</td><td>3.7749</td><td>0.7082 3.9477</td></tr><tr><td rowspan="5">RealLR200</td><td>CLIPIQA ↑</td><td>0.6036</td><td>0.7072</td><td>0.7023</td><td>0.6474</td><td>0.7122</td><td>0.6792</td><td>0.7153</td><td>0.7248</td><td>0.7048</td><td>0.7741</td></tr><tr><td>MUSIQ ↑</td><td>62.863</td><td>67.727</td><td>70.195</td><td>63.126</td><td>68.897</td><td>69.041</td><td>70.935</td><td>70.930</td><td>69.759</td><td>72.166</td></tr><tr><td>MANIQA ↑</td><td>0.5922</td><td>0.6464</td><td>0.6482</td><td>0.5522</td><td>0.6536</td><td>0.6331</td><td>0.6639</td><td>0.6363</td><td>0.6354</td><td>0.6738</td></tr><tr><td>TOPIQ ↑</td><td>0.5286</td><td>0.5905</td><td>0.6900</td><td>0.5689</td><td>0.6401</td><td>0.5990</td><td>0.6627</td><td>0.6664</td><td>0.6684</td><td>0.7249</td></tr><tr><td>QALIGN ↑</td><td>3.3409</td><td>3.7782</td><td>4.0305</td><td>3.4105</td><td>3.9228</td><td>3.8459</td><td>3.9891</td><td>3.8908</td><td>3.9312</td><td>4.2622</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

# 3.5. Training Loss

We train the student model with reconstruction and regression losses. To avoid gradient inconsistency arising from the ill-posed problem of the Real-ISR [16] while fully leveraging the teacher model knowledge, we first apply a Gaussian blur to both the reconstructed image and the HQ image $x _ { H }$ before computing the MSE loss. This ensures that the $x _ { H }$ only supervises the low-frequency content of the reconstruction, helping to preserve high-frequency details. We adopt a larger blur kernel for larger timesteps $t _ { s }$ , which enhances the trade-off between fidelity and generation:

and the LPIPS loss form the reconstruction loss:

$$
\mathcal { L } _ { R e c } = \mathcal { L } _ { M S E } ^ { b l u r } + \mathcal { L } _ { L P I P S } ( G _ { \theta } ( x _ { L } ) , x _ { H } ) .
$$

For the regression loss, we adopt the TAVSD loss in Eq. (6) to improve the realism of the generated results. The overall loss for the student model is:

$$
\mathcal { L } _ { S t u } = \mathcal { L } _ { R e c } + \lambda _ { T A V S D } \cdot \mathcal { L } _ { T A V S D } .
$$

We adopt the original diffusion loss for the LoRA model:

$$
\mathcal { L } _ { M S E } ^ { b l u r } = \mathcal { L } _ { M S E } \left( G _ { \theta } ( x _ { L } ) \ast G _ { t _ { s } } , x _ { H } \ast G _ { t _ { s } } \right) .
$$

Where $^ *$ denotes the convolution operation, $G _ { t _ { s } }$ is the convolution kernel whose size is determined by $t _ { s }$ . This loss

$$
\mathcal { L } _ { D i f f } ( \hat { \mathbf { z } } , c _ { y } ) = \mathbb { E } _ { t , \epsilon } \left[ \left\| \epsilon _ { \phi } ( \hat { \mathbf { z } } _ { t } ; t , c _ { y } ) - \boldsymbol { \epsilon } ^ { \prime } \right\| ^ { 2 } \right] ,
$$

where $\epsilon ^ { \prime }$ is the randomly sampled Gaussian noise as the training target for the denoising network.

![](images/0a4faf9277037e1927667d631e817fcb30d71eefa2b9c8ca2f845e99ac264c1c.jpg)
Figure 5. Visual comparisons between our method and other Real-ISR methods. Please zoom in for a better view.

# 4. Experiments

# 4.1. Experimental Setup

Training. We use LSDIR [14] as the training data with the $5 1 2 \times 5 1 2$ patch size. To generate paired HQ-LQ training data, we follow the degradation pipeline from Real-ESRGAN [25]. We use AdamW optimizer [19] with a learning rate $5 \times 1 0 ^ { - 5 }$ and set LoRA rank to 4 for both the student model and LoRA model. We employ the SD 2.1-base as the pre-trained model and fine-tune it for $2 \mathrm { k }$ iterations using 8 NVIDIA A40 GPUs with a batch size of 24. For the choice of hyperparameters, we set $\lambda = 0 . 5$ and $\gamma = 0$ in Eq. (5). We adopt the degradation-aware prompt extraction (DAPE) module [33] to extract text prompts.

Test Dataset. We evaluate our method in both synthetic and real-world dataset. For the synthetic dataset, we randomly crop 3K patches with a resolution of $5 1 2 \times 5 1 2$ from the DIV2K [1] validation set and synthesize LQ data using the same pipeline as that in training. For real-world data, we employ RealSR [2], DrealSR [30], and RealLR200 [33]. We center-crop RealSR [2] and DrealSR [30] datasets with size $1 2 8 \times 1 2 8$ for LQ images and $5 1 2 \times 5 1 2$ for HQ image. For RealLR200 [33] dataset, since the corresponding HQ images are unavailable, we perform only a $1 2 8 \times 1 2 8$ center-crop on the LQ images.

Evaluation Metrics. We utilize several reference and nonreference metrics to evaluate the performance of various methods on the test data. For the reference measures, we employ PSNR, SSIM [28], and LPIPS [41] to measure image fidelity. For the non-reference measures, we employ

CLIPIQA [23], MUSIQ ([13], MANIQA [36], TOPIQ [4], and QALIGN [31] to measure image quality.

Compared Methods. We compare our method with several multi-step diffusion-based methods StableSR [24], Diff-BIR [18], SeeSR [33], and one-step methods SinSR [26], OSEDiff [32], S3Diff [39], AdcSR [3], TSDSR [8], and PisaSR [21]. All comparative results are obtained using publicly released code and model weights for testing.

# 4.2. Comparisons with State-of-the-art Methods

Quantitative Comparisons. We set up the timestep condition $t _ { s } ~ = ~ 5 0 0$ in our method, and show the quantitative comparisons on the four synthetic and real-world datasets in Table 1. We have the following observations: (1) TADSR achieves the highest no-reference scores across four datasets, except for the MUSIQ on DIV2K-Val. This demonstrates that TADSR can more effectively leverage the generative priors from SD to produce more realistic results. Notably, TADSR is the only one-step method that consistently outperforms multi-step methods on all noreference metrics, achieving both efficiency and perceptual quality. (2) TADSR maintains PSNR values comparable to other SD-based one-step methods, indicating a good balance between fidelity and realism. (3) TADSR shows clear improvements over other SD-based one-step methods on CLIPIQA and TOPIQ, highlighting its superior semantic awareness and generative capability. (4) Under the same number of parameters as OSEDiff, TADSR significantly outperforms OSEDiff on no-reference metrics while maintaining comparable reference metrics, further demonstrating TADSR’s effective utilization of generative priors and the resulting performance improvements.

![](images/f5c2c19f1360fcd7cb9cde6033eaf4cc76175d99cae072035e29f5bf93a8eeb1.jpg)
Figure 6. (a) Quantitative metrics of our method under different timestep $t _ { s }$ , evaluated on the DrealSR dataset. (b) Comparison of our method under different timesteps $t _ { s }$ , PisaSR under different semantic guidance weights $\lambda _ { s e m }$ , and other one-step diffusion-based Real-ISR methods, evaluated on the DrealSR dataset.

Qualitative Comparisons. Figure 5 shows the visual comparisons between our method and the other state-of-the-art Real-ISR methods. As shown in the first row, TADSR generates significantly more natural and sharper textures from heavily degraded LQ images, especially in facial regions such as the teeth, eyes, and eyebrows, demonstrating its strong semantic generation capability. In the second row, the digits and letters produced by TADSR appear much clearer, showcasing its superior degradation removal ability while preserving fidelity. In the third row, TADSR yields more natural results around the eagle’s eyes and beak. In the fourth row, only TADSR accurately restored naturallooking facial features such as the nose, mouth, and chin. Other methods generally suffered from degradation, resulting in some distortion, and failed to reconstruct a plausible chin structure. Overall, thanks to the ability to distill generative priors from SD more effectively in TAVSD loss, TADSR can produce natural and realistic results in a single diffusion step. Compared to other methods, it achieves strong perceptual quality while maintaining high efficiency.

# 4.3. Ablation Study

Impact of Different Timestep Condition. As shown in Figure 6(a), we analyze the impact of timestep $t _ { s }$ in our method on both reference and no-reference metrics. As $t _ { s }$ increases, PSNR exhibits a decreasing trend while QALIGN shows an upward trend, indicating a trade-off where fidelity is sacrificed to enhance realism. This tradeoff between fidelity and realism aligns with the function of $t _ { s }$ , as a larger $t _ { s }$ means that TAVSD provides stronger generative guidance, while a smaller $t _ { s }$ provides more fidelitypreserving guidance. Similar visual results can be observed in supplementary materials. Furthermore, we compare the results of our method under different $t _ { s }$ , PisaSR under different $\lambda _ { s e m }$ settings, and other one-step Real-ISR methods, as shown in Figure 6(b). It can be observed that our method consistently lies in the top-right corner across different $t _ { s }$ . When $t _ { s }$ equals 200, our method achieves 26.61dB PSNR, which is more than 1dB higher than SinSR, and QALIGN is significantly higher than SinSR. In contrast, although PisaSR can also achieve a PSNR of 29.60dB by tuning the $\lambda _ { p i x } = 1 . 0$ and $\lambda _ { s e m } = 0 . 6$ , its QALIGN is only 2.91, which is similar to SinSR. This indicates that our method achieves a substantial improvement in fidelity with only a minimal compromise in realism.

![](images/1eabdadfeb1531869ead2c0cd32c8f84918991d8e56ade2720b283575d32d77b.jpg)
Figure 7. Vision Comparisons of the ablation study on TAE and TAVSD. Baseline use the original VAE encoder and VSD loss.

Table 2. Quantitative Comparison of ablation study on TAVSD and TAE. Baseline uses the original VAE encoder and VSD loss.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>PSNR↑</td><td rowspan=1 colspan=1>MUSIQ↑</td><td rowspan=1 colspan=1>CLIPIQA ↑</td><td rowspan=1 colspan=1>TOPIQ↑</td></tr><tr><td rowspan=3 colspan=1>RealSR</td><td rowspan=3 colspan=1>Baselinew/o TAEw/o TAVSDFull</td><td rowspan=3 colspan=1>24.3924.8924.8425.16</td><td rowspan=1 colspan=1>70.22</td><td rowspan=1 colspan=1>0.6751</td><td rowspan=1 colspan=1>0.6391</td></tr><tr><td rowspan=2 colspan=1>70.0870.9671.18</td><td rowspan=2 colspan=1>0.68570.69300.7283</td><td rowspan=1 colspan=1>0.64660.6553</td></tr><tr><td rowspan=1 colspan=1>0.7082</td></tr><tr><td rowspan=3 colspan=1>DrealSR</td><td rowspan=3 colspan=1>Baselinew/o TAEw/o TAVSDFull</td><td rowspan=3 colspan=1>27.4527.9528.0328.39</td><td rowspan=3 colspan=1>65.9065.9566.9567.02</td><td rowspan=1 colspan=1>0.6887</td><td rowspan=3 colspan=1>0.62750.63960.63730.6758</td></tr><tr><td rowspan=1 colspan=1>0.70300.7015</td></tr><tr><td rowspan=1 colspan=1>0.70150.7398</td></tr></table>

Impact of TAVSD and TAE. To validate the effectiveness of TAVSD and TAE, we conducted ablation studies by removing them. We employ the original VAE encoder in SD and the VSD loss as our baseline and conduct ablation studies by separately removing TAE and TAVSD. We use PSNR to evaluate fidelity and CLIPIQA, MUSIQ, and TOPIQ to assess realism. As shown in Table 2, we have the following three key observations: (1) Although the baseline also adopts randomly sampled timesteps during training, after removing TAE, both reference and no-reference metrics decline, demonstrating that timestep-adaptive latent distribution plays a crucial role in effectively utilizing the generative priors in SD. (2) When TAVSD is ablated, all metrics similarly decrease, indicating that more consistent guidance from the teacher model better activates generative priors across different timesteps. (3) Baseline shows significant degradation in PSNR and moderate decline in others, proving that both TAE and TAVSD improve fidelity and realism. Additionally, Figure 7 presents a visual comparison of our ablation studies, showing that both the absence of TAE/TAVSD leads to unrealistic parrot reconstructions, while the baseline even produces visible artifacts. In contrast, our method produces realistic and natural results by fully exploiting the generative priors in SD.

# 5. Conclusion

In this paper, we propose TADSR, a one-step SD-based Real-ISR method. TADSR introduces a variable timestep $t _ { s }$ into the student model and uses a Time-Aware VAE Encoder to fully utilize the generative priors in SD at different timesteps. To further distill the priors at different timesteps in SD to achieve varied SR effects, TADSR leverages the Time-Aware Variational Score Distillation to provide more consistent generative guidance condition on $t _ { s }$ . As a result, TADSR fully leverages the generative priors in SD and naturally achieves a controllable trade-off between fidelity and realism condition on $t _ { s }$ . Our experiments demonstrate that TADSR achieves state-of-the-art performance among all Real-ISR methods.

# References

[1] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In CVPRW, pages 126–135, 2017. 7, 13 [2] Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In ICCV, pages 3086–
3095, 2019. 7, 12, 13 [3] Bin Chen, Gehui Li, Rongyuan Wu, Xindong Zhang, Jie Chen, Jian Zhang, and Lei Zhang. Adversarial diffusion compression for real-world image super-resolution. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28208–28220, 2025. 2, 7, 15, 16, 17 [4] Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, and Weisi Lin. Topiq: A top-down approach from semantics to distortions for image quality assessment. IEEE TIP, 33:2404–2418, 2024. 7, 11 [5] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In CVPR, pages 12299–12310, 2021. 2 [6] Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong. Activating more pixels in image superresolution transformer. In CVPR, pages 22367–22377, 2023. [7] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a deep convolutional network for image super-resolution. In ECCV, pages 184–199. Springer, 2014.
2 [8] Linwei Dong, Qingnan Fan, Yihong Guo, Zhonghao Wang, Qi Zhang, Jinwei Chen, Yawei Luo, and Changqing Zou. Tsd-sr: One-step diffusion with target score distillation for real-world image super-resolution. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages
23174–23184, 2025. 2, 7, 15, 16, 17 [9] Zheng-Peng Duan, Jiawei Zhang, Xin Jin, Ziheng Zhang, Zheng Xiong, Dongqing Zou, Jimmy Ren, Chun-Le Guo, and Chongyi Li. Dit4sr: Taming diffusion transformer for real-world image super-resolution. arXiv preprint arXiv:2503.23580, 2025. 1
[10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
[11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 1
[12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 11
[13] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In ICCV, pages 5148–5157, 2021. 7, 11, 12
[14] Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al. Lsdir: A large scale dataset for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1775–1787, 2023. 7
[15] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In ICCV, pages 1833–1844, 2021. 2
[16] Jie Liang, Hui Zeng, and Lei Zhang. Details or artifacts: A locally discriminative learning approach to realistic image super-resolution. In CVPR, pages 5657–5666, 2022. 1, 6, 13
[17] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In CVPRW, pages 136–144, 2017. 2
[18] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Ben Fei, Bo Dai, Wanli Ouyang, Yu Qiao, and Chao Dong. Diffbir: Towards blind image restoration with generative diffusion prior. arXiv preprint arXiv:2308.15070, 2023. 1, 3, 7
[19] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
[20] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn- ¨ thesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 1, 11
[21] Lingchen Sun, Rongyuan Wu, Zhiyuan Ma, Shuaizheng Liu, Qiaosi Yi, and Lei Zhang. Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2333–2343, 2025. 1, 2, 3, 7, 15, 16, 17
[22] Yang Tao, Wu Rongyuan, Ren Peiran, Xie Xuansong, and Zhang Lei. Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization. In ECCV, 2023. 1, 3
[23] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, pages 2555–2563, 2023. 7, 13
[24] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. arXiv preprint arXiv:2305.07015, 2023. 1, 3, 7
[25] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In ICCV, pages 1905–1914, 2021. 1, 2, 7, 13
[26] Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang, Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C Kot, and Bihan Wen. Sinsr: Diffusion-based image superresolution in a single step. In CVPR, 2024. 7, 17
[27] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004. 11, 12
[28] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004. 7
[29] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 2, 11
[30] Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image super-resolution. In ECCV, pages 101–117. Springer, 2020. 7, 11, 13
[31] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin. Q-align: Teaching LMMs for visual scoring via discrete text-defined levels. In ICML, pages 54015–54029. PMLR, 2024. 7, 11, 12
[32] Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. One-step effective diffusion network for real-world image super-resolution. arXiv preprint arXiv:2406.08177, 2024. 2, 3, 4, 5, 7, 13, 15, 16, 17
[33] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. Seesr: Towards semantics-aware real-world image super-resolution. In CVPR, 2024. 1, 3, 7, 15, 16
[34] Zhiqiang Wu, Zhaomang Sun, Tong Zhou, Bingtao Fu, Ji Cong, Yitong Dong, Huaqi Zhang, Xuan Tang, Mingsong Chen, and Xian Wei. Omgsr: You only need one midtimestep guidance for real-world image super-resolution. arXiv preprint arXiv:2508.08227, 2025. 1
[35] Liangbin Xie, Xintao Wang, Xiangyu Chen, Gen Li, Ying Shan, Jiantao Zhou, and Chao Dong. Desra: detect and delete the artifacts of gan-based real-world super-resolution models. arXiv preprint arXiv:2307.02457, 2023. 1
[36] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In CVPR, pages 1191–1200, 2022. 7, 13
[37] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image super-resolution by residual shifting. arXiv preprint arXiv:2307.12348, 2023. 1
[38] Zongsheng Yue, Kang Liao, and Chen Change Loy. Arbitrary-steps image super-resolution via diffusion inversion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23153–23163, 2025. 2
[39] Aiping Zhang, Zongsheng Yue, Renjing Pei, Wenqi Ren, and Xiaochun Cao. Degradation-guided one-step image super-resolution with diffusion priors. arXiv preprint arXiv:2409.17058, 2024. 2, 7, 15, 16, 17
[40] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In ICCV, pages 4791–4800, 2021. 1, 2, 13
[41] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586–595, 2018. 7
[42] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. Image super-resolution using very deep residual channel attention networks. In ECCV, pages 286– 301, 2018. 2
[43] Qianqian Zhao, Chunle Guo, Tianyi Zhang, Junpei Zhang, Peiyang Jia, Tan Su, Wenjie Jiang, and Chongyi Li. A systematic investigation on deep learning-based omnidirectional image and video super-resolution. arXiv preprint arXiv:2506.06710, 2025. 2

# Time-Aware One Step Diffusion Network for Real-World Image Super-Resolution

Supplementary Material

In this supplementary material, we provide the following content:

• Detailed derivation about the Variational Score Distillation loss in Section 6
• Visual comparisons and quantitative metrics of TADSR across different timesteps in Section 7.1
• Ablation study on the blurred MSE loss in Section 7.2
• Ablation study on the hyperparameters of TAVSD loss in Section 7.3
• Efficiency comparison between TADSR and other diffusion-based Real-ISR methods in Section 8
• Comparisons with GAN-based Real-ISR methods in Section 9
• Extended visual comparisons with SD-based Real-ISR approaches in Section 10

# 6. Detailed Derivation

According to the original diffusion process in SD [20], at step $t$ , the current state $z _ { t }$ satisfies:

$$
z _ { t } = \alpha _ { t } z _ { 0 } + \beta _ { t } \epsilon , t = 1 , 2 , \ldots , T ,
$$

where $\alpha _ { t }$ and $\beta _ { t }$ are the scale parameters in diffusion, $\epsilon \sim$ $\mathcal { N } ( \mathbf { 0 } , \pmb { I } ^ { 2 } )$ and $z _ { \mathrm { 0 } }$ is HR latent in Real-ISR task. Therefore, we can express $z _ { \mathrm { 0 } }$ in terms of $z _ { t }$ and $\epsilon$ as $\begin{array} { r } { z _ { 0 } ~ = ~ \frac { z _ { t } - \beta _ { t } \epsilon } { \alpha _ { t } } } \end{array}$ Then, we can rewrite Eq. (2) in the main paper as follows:

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { V S D } = \mathbb { E } _ { t , \epsilon } \left[ \omega ( t ) \left( \epsilon _ { \psi } ( \hat { z } _ { t } ; t , c ) - \epsilon _ { \phi } ( \hat { z } _ { t } ; t , c ) \right) \frac { \partial \hat { z } } { \partial \theta } \right] , } \\ & { \quad \quad \quad \quad \quad = \mathbb { E } _ { t , \epsilon } \left[ \omega ( t ) \frac { \alpha _ { t } } { \beta _ { t } } \left( \hat { z } _ { \phi } ( \hat { z } _ { t } ; t , c ) - \hat { z } _ { \psi } ( \hat { z } _ { t } ; t , c ) \right) \frac { \partial \hat { z } } { \partial \theta } \right] , } \\ & { \quad \quad \quad \quad = \mathbb { E } _ { t , \epsilon } \left[ \omega ^ { \prime } ( t ) \left( \hat { z } _ { \phi } ( \hat { z } _ { t } ; t , c ) - \hat { z } _ { \psi } ( \hat { z } _ { t } ; t , c ) \right) \frac { \partial \hat { z } } { \partial \theta } \right] , } \end{array}
$$

where $\epsilon _ { \psi }$ is the pre-trained diffusion model (teacher model), $\epsilon _ { \phi }$ represents its replica with trainable LoRA [12] (LoRA model), $\hat { z } _ { \psi }$ and $\hat { z } _ { \phi }$ represent the latent images predicted by the teacher model and the LoRA model respectively, $c$ is a text embedding of a caption describing the input image, and $\omega _ { t }$ is a time-varying weighting function. Therefore, we can represent the VSD [29] loss using the residual between the latent images predicted by the teacher model and the LoRA model, which is then decoded into pixel space to analyze the timestep-dependent guidance.

Table 3. Quantitative comparison of ablation study on blurred MSE loss, evaluated on DrealSR [30] dataset.

<table><tr><td>Methods</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MUSIQ ↑</td><td>TOPIQ ↑</td><td>QALIGN ↑</td></tr><tr><td>ts = 100</td><td>29.576</td><td>0.8021</td><td>64.022</td><td>0.6201</td><td>3.4666</td></tr><tr><td>ts = 200</td><td>29.610</td><td>0.8059</td><td>63.735</td><td>0.6108</td><td>3.4750</td></tr><tr><td>ts = 300</td><td>29.167</td><td>0.7935</td><td>65.367</td><td>0.6463</td><td>3.6069</td></tr><tr><td>ts = 400</td><td>29.245</td><td>0.7992</td><td>64.681</td><td>0.6149</td><td>3.5635</td></tr><tr><td>ts 500 = ts =$ 600</td><td>28.387 28.473</td><td>0.7758 0.7834</td><td>67.016 66.620</td><td>0.6758 0.6597</td><td>3.7491 3.7372</td></tr></table>

Table 4. Quantitative comparison of ablation study on blurred MSE loss, evaluated on DrealSR [30] dataset.

<table><tr><td>Methods</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MUSIQ ↑</td><td>QALIGN ↑</td></tr><tr><td>w/o blurred MSE</td><td>29.074</td><td>0.7841</td><td>64.732</td><td>3.5299</td></tr><tr><td>TADSR (ts = 300)</td><td>29.167</td><td>0.794</td><td>65.367</td><td>3.6069</td></tr><tr><td>TADSR</td><td>28.387</td><td>0.7758</td><td>67.016</td><td>3.7491</td></tr></table>

# 7. More Ablation Study

# 7.1. Different Timesteps

Figure 8 presents TADSR’s results at different timesteps $t _ { s }$ , demonstrating a gradual transition from fidelity to realism reconstruction as the $t _ { s }$ increases. Specifically: (1) In the first row, TADSR progressively generates richer eyelash textures and sharper contours; (2) The second row shows how patterned shadows gradually transform into stain-like artifacts; (3) For the third row, TADSR reconstructs plausible architectural stripes not present in the low-quality input; and (4) The fourth row reveals emerging yellow pistils in flower centers. These progressive changes evidence TADSR’s enhanced utilization of the pre-trained generative priors in SD at larger $t _ { s }$ , effectively balancing the fidelity-realism trade-off condition on $t _ { s }$ . In addition, we also present the performance of our method at different timesteps. As shown in Table 3, with increasing timesteps, reference metrics (PSNR, SSIM [27]) tend to decrease while no-reference metrics (MUSIQ [13], TOPIQ [4], QALIGN [31]) tend to increase. This is consistent with the visual comparison results in Figure 8, demonstrating that our method can achieve one-step realism–fidelity controllable generation simply by adjusting the timestep.

# 7.2. Blurred MSE Loss

To avoid gradient inconsistency arising from the ill-posed problem of the Real-ISR task while fully leveraging generative prior of SD, we introduce a blurred MSE loss to replace the original MSE loss. Specifically, we first apply a Gaussian blur to both the reconstructed image $G _ { \theta } ( x _ { L } )$ and the HQ image $x _ { H }$ before computing the MSE loss. The blurred MSE loss can be formed as:

![](images/7a79e530539b5e4abeb16baefdd051d45f847b8781347fdd33fdaa1e42ba83f9.jpg)
Figure 8. Vision comparisons of TADSR at different timesteps $t _ { s }$ . Zoom in for a better view.

$$
\mathcal { L } _ { M S E } ^ { b l u r } = \mathcal { L } _ { M S E } \left( G _ { \theta } ( x _ { L } ) \ast G _ { t _ { s } } , x _ { H } \ast G _ { t _ { s } } \right) .
$$

Where $^ *$ denotes the convolution operation, $G _ { t _ { s } }$ is the Gaussian convolution kernel whose size is determined by $t _ { s }$ . Let $k _ { t _ { s } }$ as the kernel size of $G _ { t _ { s } }$ , it satisfies:

$$
k _ { t _ { s } } = 5 + 4 * \lfloor \frac { t _ { s } } { 2 0 0 } \rfloor .
$$

To validate the effectiveness of the proposed blurred MSE loss, we performed an ablation study by removing it. As shown in Tab. 4, when the blurred MSE loss is removed, the no-reference metrics degrade (MUSIQ [13], QALIGN [31]) significantly while the reference metrics (PSNR, SSIM [27]) improve, demonstrating a trade-off effect where fidelity is enhanced at the expense of realism. To better align with the reference metrics, we selected TADSR’s output at $t _ { s } = 3 0 0$ . With the blurred MSE loss incorporated, TADSR achieves improvements across all metrics, indicating that this loss function enables a more optimal balance between fidelity and realism.

Table 5. Quantitative comparison of ablation study on the hyperparameters of TAVSD loss, evaluated on RealSR [2] dataset.

<table><tr><td rowspan=1 colspan=2>Methods</td><td rowspan=2 colspan=4>PSNR ↑</td><td rowspan=2 colspan=1>SSIM↑</td><td rowspan=2 colspan=1>MUSIQ ↑</td><td rowspan=2 colspan=1>MANIQA ↑</td></tr><tr><td rowspan=1 colspan=1>λ</td><td rowspan=1 colspan=1>γ</td></tr><tr><td rowspan=3 colspan=1>0.250.500.75</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=4>25.757</td><td rowspan=1 colspan=1>0.7218</td><td rowspan=5 colspan=1>69.40871.18271.48470.74270.918</td><td rowspan=5 colspan=1>0.65410.67150.67740.67020.6727</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=3>25.166</td><td rowspan=2 colspan=2>25.16624.398</td><td rowspan=2 colspan=1>0.71500.7038</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>24</td><td></td></tr><tr><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>50</td><td></td><td></td><td rowspan=2 colspan=3>25.09024.809</td><td rowspan=2 colspan=1>0.71590.7115</td></tr><tr><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>100</td><td></td><td></td></tr></table>

# 7.3. Hyperparameters in TAVSD

To verify the sensitivity of our method to the hyperparameters in TAVSD, we conducted ablation studies by varying their values. We set the timestep $t _ { s }$ to 500 and evaluated our method under different hyperparameters on the RealSR [2] dataset. As shown in Table 5, with the increase of $\lambda$ and $\gamma$ , our method exhibits a decrease in reference metrics while no-reference metrics improve, reflecting a trade-off of fidelity for enhanced realism.

Table 6. A comprehensive evaluation against state-of-the-art GAN-based methods across synthetic and real-world datasets. The toperforming results under each metric are marked in red.

<table><tr><td>Datasets</td><td>Methods</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↑</td><td>CLIPIQA ↑</td><td>MUSIQ ↑</td><td>MAINIQA ↑</td><td>TOPIQ ↑</td><td>QALIGN ↑</td></tr><tr><td rowspan="4">DIV2k-Val</td><td>BSRGAN</td><td>24.583</td><td>0.6269</td><td>0.3351</td><td>0.5246</td><td>61.196</td><td>0.5041</td><td>0.5460</td><td>3.1708</td></tr><tr><td>RealESRGAN</td><td>24.293</td><td>0.6372</td><td>0.3112</td><td>0.5277</td><td>61.058</td><td>0.5485</td><td>0.5297</td><td>3.2768</td></tr><tr><td>LDL</td><td>23.828</td><td>0.6344</td><td>0.3256</td><td>0.5179</td><td>60.038</td><td>0.5328</td><td>0.5144</td><td>3.1797</td></tr><tr><td>TADSR</td><td>23.815</td><td>0.6028</td><td>0.3078</td><td>0.7353</td><td>69.649</td><td>0.6443</td><td>0.7044</td><td>4.0783</td></tr><tr><td rowspan="4">DrealSR</td><td>BSRGAN</td><td>28.701</td><td>0.8028</td><td>0.2858</td><td>0.5092</td><td>57.165</td><td>0.4845</td><td>0.5060</td><td>2.9580</td></tr><tr><td>RealESRGAN</td><td>28.615</td><td>0.8051</td><td>0.2819</td><td>0.4525</td><td>54.268</td><td>0.4903</td><td>0.4623</td><td>2.8645</td></tr><tr><td>LDL</td><td>28.197</td><td>0.8124</td><td>0.2792</td><td>0.4475</td><td>53.949</td><td>0.4894</td><td>0.4518</td><td>2.8564</td></tr><tr><td>TADSR</td><td>28.387</td><td>0.7758</td><td>0.3235</td><td>0.7398</td><td>67.016</td><td>0.6309</td><td>0.6758</td><td>3.7491</td></tr><tr><td rowspan="4">RealSR</td><td>BSRGAN</td><td>26.379</td><td>0.7651</td><td>0.2656</td><td>0.5116</td><td>63.287</td><td>0.5420</td><td>0.5505</td><td>3.1843</td></tr><tr><td>RealESRGAN</td><td>25.686</td><td>0.7614</td><td>0.2710</td><td>0.4494</td><td>60.370</td><td>0.5505</td><td>0.5148</td><td>3.1073</td></tr><tr><td>LDL</td><td>25.281</td><td>0.7565</td><td>0.2750</td><td>0.4555</td><td>60.928</td><td>0.5495</td><td>0.5125</td><td>3.0888</td></tr><tr><td>TADSR</td><td>25.166</td><td>0.7150</td><td>0.3168</td><td>0.7283</td><td>71.182</td><td>0.6715</td><td>0.7082</td><td>3.9477</td></tr></table>

Table 7. The inference time and the number of parameters of diffusion-based Real-ISR methods. The top-performing results under each metric are marked in red.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>StableSR</td><td rowspan=1 colspan=1>DiffBIR</td><td rowspan=1 colspan=1>SeeSR</td><td rowspan=1 colspan=1>SinSR</td><td rowspan=1 colspan=1>OSEDiff</td><td rowspan=1 colspan=1>S3Diff</td><td rowspan=1 colspan=1>AdcSR</td><td rowspan=1 colspan=1>TSDSR</td><td rowspan=1 colspan=1>PisaSR</td><td rowspan=1 colspan=1>TADSR</td></tr><tr><td rowspan=1 colspan=1>Inference Step</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Inference time(s)</td><td rowspan=1 colspan=1>10.40</td><td rowspan=1 colspan=1>9.83</td><td rowspan=1 colspan=1>5.64</td><td rowspan=1 colspan=1>0.1785</td><td rowspan=1 colspan=1>0.1463</td><td rowspan=1 colspan=1>0.4704</td><td rowspan=1 colspan=1>0.0825</td><td rowspan=1 colspan=1>0.0947</td><td rowspan=1 colspan=1>0.1675</td><td rowspan=1 colspan=1>0.1465</td></tr><tr><td rowspan=1 colspan=1>#Params(MB)</td><td rowspan=1 colspan=1>1563</td><td rowspan=1 colspan=1>1682</td><td rowspan=1 colspan=1>2514</td><td rowspan=1 colspan=1>119</td><td rowspan=1 colspan=1>1775</td><td rowspan=1 colspan=1>1327</td><td rowspan=1 colspan=1>456</td><td rowspan=1 colspan=1>2207</td><td rowspan=1 colspan=1>1302</td><td rowspan=1 colspan=1>1777</td></tr></table>

This phenomenon is consistent with the functionality of TAVSD. As discussed in Section 1, the pre-trained SD model exhibits different generative priors at different timesteps: smaller timesteps tend to favor fidelity, while larger timesteps tend to favor generation. This also means that the teacher model in VSD will provide generation guidance based on semantic priors when the time step is large. Therefor, when $\lambda$ and $\gamma$ increase, the same $t _ { s }$ is mapped to a larger $t _ { v }$ , causing the teacher model to provide guidance more biased toward generation. In terms of metrics, this is manifested as an increase in no-reference metrics and a decrease in reference-based metrics. Considering the balance between realism and fidelity, we ultimately choose $\lambda = 0 . 5$ and $\gamma = 0$ as the default setting for our model.

# 8. Comparison of Efficiency with Other Onestep Real-ISR Methods

We compare the number of parameters and inference time of one-step diffusion-based Real-ISR models in Table 7. Inference time is measured on the $\times 4$ SR task with $1 2 8 \times 1 2 8$ LQ images using a single NVIDIA 3090 24G GPU. Compared with OSEDiff [32], our method achieves roughly the same inference time and parameter count, while showing significant improvements in no-reference metrics and visual quality. PisaSR requires two inferences to achieve controllable Real-ISR due to the presence of two LoRA weights. In contrast, our method can obtain controllable Real-ISR with a single inference simply by adjusting the time step, resulting in fewer inference steps and shorter inference time.

# 9. Comparisons with GAN-based Real-ISR Methods

We compare TADSR with three GAN-based Real-ISR methods: BSRGAN [40], RealESRGAN [25], and LDL [16]. Quantitative evaluations are conducted on the DIV2K [1], RealSR [2], and DRealSR [30] datasets, with results summarized in Tab. 6. The experimental results demonstrate that TADSR, leveraging the powerful generative priors of the pre-trained SD, achieves significantly superior no-reference metrics (e.g., CLIPIQA [23], MAINIQA [36]) compared to GAN-based methods.

Additionally, Fig. 9 presents a visual comparison between TADSR and other GAN-based methods. The results show that TADSR reconstructs more photorealistic and natural outcomes, including higher fidelity in text and architectural structures (from the first to the third group), and more realistic rope textures (in the fourth group).

# 10. More Visual Comparisons with SD-based Real-ISR Methods

We provide more visual comparisons between TADSR and other SD-based SR methods in Fig. 10 and Fig. 11. Compared to other methods, TADSR consistently produces clearer, more realistic, and more natural results. Moreover, although our training is conducted at a resolution of

![](images/3db2067f6d6623ab0bffae6e8e32d9cfc6cf4f8947a441a4a8f76372c469eea9.jpg)
Figure 9. Vision comparisons between TADSR and GAN-based Real-ISR methods. Zoom in for a better view.

$5 1 2 { \times } 5 1 2$ , we provide visual comparisons of TADSR and other diffusion-based one-step Real-ISR methods on 2Kresolution images. As shown in Figure 12, TADSR is also capable of maintaining strong structural consistency and producing realistic, natural SR results on high-resolution images. Under severe synthetic degradations (first and third rows), TADSR still demonstrates powerful deblurring capability and highly realistic generative effects, whereas other single-step SR methods are noticeably affected by the degradations, exhibiting clear artifacts or color smearing. In real-world degradation scenarios (second and fourth rows), TADSR similarly recovers more authentic texture details (e.g., the cat’s fur and nose) and more natural structures (e.g., the person’s eyes and glasses). These visual comparisons consistently demonstrate that TADSR makes effective and thorough use of the SD generative prior.

![](images/b3014ed44026366983f7c830582f6f33757022ca026facd32c068c19e40c4a8b.jpg)
Figure 10. Vision comparisons between TADSR and SD-based Real-ISR methods (SeeSR [33], OSEDiff [32], S3Diff [39], PiSASR [21], AdcSR [3], TSDSR [8]). Zoom in for a better view.

![](images/bf2d5e1b30c38ed18606c800118ae1aa574295932ce37c7d0cb0c519d6847e7c.jpg)
Figure 11. Vision comparisons between TADSR and SD-based Real-ISR methods (SeeSR [33], OSEDiff [32], S3Diff [39], PiSASR [21], AdcSR [3], TSDSR [8]). Zoom in for a better view.

![](images/05747ecb59e92e1c2170909fdc9c730f184b932b1c0762081621e41251d6c4cc.jpg)
Figure 12. Vision comparisons between TADSR and Diffusion-based Real-ISR methods (SinSR [26], OSEDiff [32], S3Diff [39], PiSASR [21], AdcSR [3], TSDSR [8]). Zoom in for a better view.