[← 返回 README](../README.md)

# Introduction

## 📌 预览
本文件合并 Introduction/Related Work，重点读多步扩散慢、单步不可控/过重/不保真等动机链。

---

# 1 Introduction

Image super-resolution (ISR) [13, 66, 29, 65, 6, 24, 46, 27, 61] is a classical yet still active research problem, which aims to restore a high-quality (HQ) image from its low-quality (LQ) observation suffering from degradations of noise, blur and low-resolution, etc. While one line of ISR research [13, 66, 29, 65, 6] simplifies the degradation process from HQ to LQ images as bicubic downsampling (or downsampling after Gaussian blur) and focus on the study on network architecture design, the trained models can hardly be generalized to real-world LQ images, whose degradations are often unknown and much more complex. Therefore, another increasingly popular line of ISR research is the so-called real-world ISR (Real-ISR) [61, 45] problem, which aims to reproduce perceptually realistic HQ images from the LQ images captured in real-world applications.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

There are two major issues in training a Real-ISR model. One is how to build the LQ-HQ training image pairs, and another is how to ensure the naturalness of restored images, i.e., how to ensure that the restored images follow the distribution of HQ natural images. For the first issue, some researchers have proposed to collect real-world LQ-HQ image pairs using long-short camera focal lenses [3, 51]. However, this is very costly and can only cover certain types of real-world image degradations. Another more economical way is to simulate the real-world LQ-HQ image pairs by using complex image degradation pipelines. The representative works include BSRGAN [61] and Real-ESRGAN [45], where a random shuffling of basic degradation operators and a high-order degradation model are respectively used to generate LQ-HQ image pairs.

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

![Figure 1](../images/b7034e0624d1b43b4c76325de23f9df6bed1145a5fc5f3cb3df574a73a5bb4a6.jpg)
*Figure 1: Figure 1: Performance and efficiency comparison among SD-based Real-ISR methods. (a). Performance comparison on the DrealSR benchmark [51]. Metrics like LPIPS and NIQE, where smaller scores indicate better image quality, are inverted and normalized for display. OSEDiff achieves leading scores on most metrics with only one diffusion step. (b). Model efficiency comparison. The inference time is tested on an A100 GPU with $5 1 2 \times 5 1 2$ input image size. OSEDiff has the fewest trainable parameters and is over 100 times faster than StableSR [42].*

> 💡 **Figure 1 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

With the given training data, how to train a robust Real-ISR model to output perceptually natural images with high quality becomes a key issue. Simply learning a mapping network between LQ-HQ paired data with pixel-wise losses can lead to over-smoothed results [24, 46]. It is crucial to integrate natural image priors into the learning process to reproduce HQ images. A few methods have been proposed to this end. The perceptual loss [18] explores the texture, color, and structural priors in a pre-trained model such as VGG-16 [38] and AlexNet [23]. The generative adversarial networks (GANs) [14] alternatively train a generator and a discriminator, and they have been adopted for Real-ISR tasks [24, 46, 45, 61, 27, 53]. The generator network aims to synthesize HQ images, while the discriminator network aims to distinguish whether the synthesized image is realistic or not. While great successes have been achieved, especially in the restoration of specific classes of images such as face images [56, 44], GAN-based Real-ISR tends to generate unpleasant details due to the unstable adversarial training and the difficulties in discriminating the image space of diverse natural scenes.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

The recently developed generative diffusion models (DM) [39, 16], especially the large-scale pretrained text-to-image (T2I) models [37, 36], have demonstrated remarkable performance in various downstream tasks. Having been trained on billions of image-text pairs, the pre-trained T2I models possess powerful natural image priors, which can be well exploited to improve the naturalness and perceptual quality of Real-ISR outputs. Some methods [42, 57, 31, 52, 40, 59] have been developed to employ the pre-trained T2I model for solving the Real-ISR problem. While having shown impressive results in generating richer and more realistic image details than GAN-based methods, the existing SD-based methods have several problems to be further addressed. First, these methods typically take random Gaussian noise as the start point of the diffusion process. Though the LQ images are used as the control signal with a ControlNet module [63], these methods introduce unwanted randomness in the output HQ images [40]. Second, the restored HQ images are usually obtained by tens or even hundreds of diffusion steps, making the Real-ISR process computationally expensive. Though some one-step diffusion based Real-ISR methods [48] have been recently proposed, they fail in achieving high-quality details compared to multi-step methods.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

To address the aforementioned issues, we propose a One-Step Effective Diffusion network, OSEDiff in short, for the Real-ISR problem. The UNet backbone in pre-trained SD models has strong capability to transfer the input data into another domain, while the given LQ image actually has rich information to restore its HQ counterpart. Therefore, we propose to directly feed the LQ images into the pre-trained SD model without introducing any random noise. Meanwhile, we integrate trainable

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

LoRA layers [17] into the pre-trained UNet, and finetune the SD model to adapt it to the Real-ISR task. On the other hand, to ensure that the one-step model can still produce HQ natural images as the multi-step models, we utilize variational score distillation (VSD) [49, 58, 10] for KL-divergence regularization. This operation effectively leverages the powerful natural image priors of pre-trained SD models and aligns the distribution of generated images with natural image priors. As illustrated in Fig. 1, our extensive experiments demonstrate that OSEDiff achieves comparable or superior performance measures to state-of-the-art SD-based Real-ISR models, while it significantly reduces the number of inference steps from $N$ to 1 and has the fewest trainable parameters, leading to more than $\times 1 0 0$ speedup in inference time over previous methods such as StableSR [42].

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

# 2 Related Work

Starting from SRCNN [13], deep learning-based methods have become prevalent for ISR. A variety of methods have been proposed [30, 66, 67, 9, 5, 29, 65, 6, 7] to improve the accuracy of ISR reconstruction. However, most of these methods assume simple and known degradations such as bicubic downsampling, limiting their applications to real-world images with complex and unknown degradations. In recent years, researches have been exploring the potentials of generative models, including GAN [14] and diffusion networks [16], for solving the Real-ISR problem.

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

GAN-based Real-ISR. The use of GAN for photo-realistic ISR can be traced back to SRGAN [24], where the image degradation is assumed to be bicubic downsampling. Later on, researchers found that GAN has the potential to perform real-world image restoration with more complex degradations [61, 45]. Specifically, by using randomly shuffled degradation and high-order degradation to generate more realistic LQ-HQ training pairs, BSRGAN [61] and Real-ESRGAN [45] demonstrate promising Real-ISR results, which trigger many following works [4, 27, 28, 53]. DASR [28] designs a tiny network to predict the degradation parameters to handle degradations of various levels. SwinIR [29] switches the generator from CNNs to stronger transformers, further enhancing the performance of Real-ISR. However, the adversarial training process in GAN is unstable and its discriminator is limited in telling the quality of diverse natural image contents. Therefore, GAN-based Real-ISR methods often suffer from unnatural visual artifacts. Some works such as LDL [27] and DeSRA [53] can suppress much the artifacts, yet they are difficult to generate more natural details.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Diffusion-based Real-ISR. Some early attempts [21, 20, 47] employ the denoising diffusion probabilistic models (DDPMs) [16, 39, 11] to address the ISR problem by assuming simple and known degradations (e.g., bicubic downsampling). These methods are training-free by modifying the reverse transition of pre-trained DDPMs using gradient descent, but they cannot be applied to complex unknown degradations. Recent researches [42, 57, 31, 40, 59] have leveraged stronger pre-trained T2I models, such as Stable Diffusion (SD) [1], to tackle the Real-ISR problem. In general, they introduce an adapter [63] to fine-tune the SD model to reconstruct the HQ image with the LQ image as the control signal. StableSR [42] finetunes a time-aware encoder and employs feature warping to balance fidelity and perceptual quality. PASD [57] extracts both low-level and high-level features from the LQ image and inputs them to the pre-trained SD model with a pixel-aware cross attention module. To further enhance the semantic-aware ability of the Real-ISR model, SeeSR [52] introduces degradation-robust tag-style text prompts and utilizes soft prompts to guide the diffusion process. To mitigate the potential risks of diffusion uncertainty, CCSR [40] leverages a truncated diffusion process to recover structures and finetunes the VAE decoder by adversarial training to enhance details. SUPIR [59] leverages the powerful generation capability of SDXL model and the strong captioning capability of LLaVA [32] to synthesize rich image details.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

The above mentioned methods, however, require tens or even hundreds of steps to complete the diffusion process, resulting in unfriendly latency. SinSR shortens ResShift [60] to a single-step inference by consistency preserving distillation. Nevertheless, the non-distrbution-based distillation loss tends to obtain smooth results, and the model capacity of SinSR and ResShift are much smaller than the SD models to address Real-ISR problems.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

---

## 🔖 Section 总结

### 核心洞察
1. 把本文 gap 和 one-step SR 主线对应起来。
2. 注意作者如何区分效率、保真、真实感和可控性。
