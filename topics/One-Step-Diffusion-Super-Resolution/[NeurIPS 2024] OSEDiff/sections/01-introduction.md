[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
本节的动机链很清楚：Real-ISR 的退化未知且需要自然纹理，GAN 容易伪影，多步 SD 质量强但慢且有随机性；OSEDiff 的入口就是把 LQ 本身作为 latent 起点，再用 LoRA 和 VSD 把 SD 先验压到一步里。
---

> 💡 **Q&A 批注记录**:
> - Q: 为什么从 LQ latent 而不是纯噪声开始？
> - A: LQ latent 已经包含布局和低频结构，单步模型只需补细节；纯噪声则需要在一步内同时生成结构和纹理，难度更高且不确定性更强。


Image super-resolution (ISR) [13, 66, 29, 65, 6, 24, 46, 27, 61] is a classical yet still active research problem, which aims to restore a high-quality (HQ) image from its low-quality (LQ) observation suffering from degradations of noise, blur and low-resolution, etc. While one line of ISR research [13, 66, 29, 65, 6] simplifies the degradation process from HQ to LQ images as bicubic downsampling (or downsampling after Gaussian blur) and focus on the study on network architecture design, the trained models can hardly be generalized to real-world LQ images, whose degradations are often unknown and much more complex. Therefore, another increasingly popular line of ISR research is the so-called real-world ISR (Real-ISR) [61, 45] problem, which aims to reproduce perceptually realistic HQ images from the LQ images captured in real-world applications.
> 💡 **问题定义**: 作者先把普通 ISR 和 Real-ISR 分开。OSEDiff 不是解决 bicubic SR，而是处理真实退化下的感知恢复，所以后面评价不能只盯 PSNR/SSIM，还要看真实感和语义一致性。


There are two major issues in training a Real-ISR model. One is how to build the LQ-HQ training image pairs, and another is how to ensure the naturalness of restored images, i.e., how to ensure that the restored images follow the distribution of HQ natural images. For the first issue, some researchers have proposed to collect real-world LQ-HQ image pairs using long-short camera focal lenses [3, 51]. However, this is very costly and can only cover certain types of real-world image degradations. Another more economical way is to simulate the real-world LQ-HQ image pairs by using complex image degradation pipelines. The representative works include BSRGAN [61] and Real-ESRGAN [45], where a random shuffling of basic degradation operators and a high-order degradation model are respectively used to generate LQ-HQ image pairs.
> 💡 **训练数据批读**: 这里点出 Real-ISR 的第一个根问题：真实配对数据贵且覆盖有限，所以大家用合成退化近似真实退化。OSEDiff 后续沿用 Real-ESRGAN pipeline，本质上仍受合成退化分布约束。


![Figure 1](../images/b7034e0624d1b43b4c76325de23f9df6bed1145a5fc5f3cb3df574a73a5bb4a6.jpg)
*Figure 1: Figure 1: Performance and efficiency comparison among SD-based Real-ISR methods. (a). Performance comparison on the DrealSR benchmark [51]. Metrics like LPIPS and NIQE, where smaller scores indicate better image quality, are inverted and normalized for display. OSEDiff achieves leading scores on most metrics with only one diffusion step. (b). Model efficiency comparison. The inference time is tested on an A100 GPU with $5 1 2 \times 5 1 2$ input image size. OSEDiff has the fewest trainable parameters and is over 100 times faster than StableSR [42].*
> 💡 **Figure 1 批读**: 这张图同时服务两个 claim：质量上 OSEDiff 在 DRealSR 多个感知指标靠前；效率上一步推理、8.5M 可训练参数、A100 上 0.11s 是主要卖点。注意图里 LPIPS/NIQE 做了反向归一化，读数时别和原指标方向混淆。


With the given training data, how to train a robust Real-ISR model to output perceptually natural images with high quality becomes a key issue. Simply learning a mapping network between LQ-HQ paired data with pixel-wise losses can lead to over-smoothed results [24, 46]. It is crucial to integrate natural image priors into the learning process to reproduce HQ images. A few methods have been proposed to this end. The perceptual loss [18] explores the texture, color, and structural priors in a pre-trained model such as VGG-16 [38] and AlexNet [23]. The generative adversarial networks (GANs) [14] alternatively train a generator and a discriminator, and they have been adopted for Real-ISR tasks [24, 46, 45, 61, 27, 53]. The generator network aims to synthesize HQ images, while the discriminator network aims to distinguish whether the synthesized image is realistic or not. While great successes have been achieved, especially in the restoration of specific classes of images such as face images [56, 44], GAN-based Real-ISR tends to generate unpleasant details due to the unstable adversarial training and the difficulties in discriminating the image space of diverse natural scenes.
> 💡 **先验动机**: 单靠 paired data + pixel loss 会过平滑，GAN 又容易在复杂自然场景造伪纹理。这里为后文把 SD 当作自然图像先验和 regularizer 铺路。


The recently developed generative diffusion models (DM) [39, 16], especially the large-scale pretrained text-to-image (T2I) models [37, 36], have demonstrated remarkable performance in various downstream tasks. Having been trained on billions of image-text pairs, the pre-trained T2I models possess powerful natural image priors, which can be well exploited to improve the naturalness and perceptual quality of Real-ISR outputs. Some methods [42, 57, 31, 52, 40, 59] have been developed to employ the pre-trained T2I model for solving the Real-ISR problem. While having shown impressive results in generating richer and more realistic image details than GAN-based methods, the existing SD-based methods have several problems to be further addressed. First, these methods typically take random Gaussian noise as the start point of the diffusion process. Though the LQ images are used as the control signal with a ControlNet module [63], these methods introduce unwanted randomness in the output HQ images [40]. Second, the restored HQ images are usually obtained by tens or even hundreds of diffusion steps, making the Real-ISR process computationally expensive. Though some one-step diffusion based Real-ISR methods [48] have been recently proposed, they fail in achieving high-quality details compared to multi-step methods.
> 💡 **核心痛点**: 现有 SD-ISR 的两个痛点被压成一句：随机噪声起点带来输出不确定性，多步反推带来延迟。OSEDiff 的“LQ latent initialization + one-step”正好分别对应这两个痛点。


To address the aforementioned issues, we propose a One-Step Effective Diffusion network, OSEDiff in short, for the Real-ISR problem. The UNet backbone in pre-trained SD models has strong capability to transfer the input data into another domain, while the given LQ image actually has rich information to restore its HQ counterpart. Therefore, we propose to directly feed the LQ images into the pre-trained SD model without introducing any random noise. Meanwhile, we integrate trainable
> 💡 **方法入口**: “directly feed the LQ images”不是像 ControlNet 那样只把 LQ 当条件，而是让 LQ 经可训练 VAE encoder 后成为扩散 denoise 的起点。这样一步模型不用从纯噪声同时恢复结构和纹理。


LoRA layers [17] into the pre-trained UNet, and finetune the SD model to adapt it to the Real-ISR task. On the other hand, to ensure that the one-step model can still produce HQ natural images as the multi-step models, we utilize variational score distillation (VSD) [49, 58, 10] for KL-divergence regularization. This operation effectively leverages the powerful natural image priors of pre-trained SD models and aligns the distribution of generated images with natural image priors. As illustrated in Fig. 1, our extensive experiments demonstrate that OSEDiff achieves comparable or superior performance measures to state-of-the-art SD-based Real-ISR models, while it significantly reduces the number of inference steps from $N$ to 1 and has the fewest trainable parameters, leading to more than $\times 1 0 0$ speedup in inference time over previous methods such as StableSR [42].
> 💡 **贡献闭环**: LoRA 负责把 SD UNet 适配到退化恢复，VSD 负责把一步输出拉回 HQ natural image 分布；Fig. 1 的速度优势则来自推理时只保留 generator，不运行 regularizer。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 任务 | Real-world image super-resolution |
| 主矛盾 | 多步 SD 质量强但慢且随机；单步方法快但细节弱 |
| 本文回答 | 把 LQ latent 直接作为扩散起点，用 latent-space VSD 将 Stable Diffusion 先验压缩到单步 Real-ISR。 |

### 核心洞察
1. Introduction 的逻辑链是：真实退化难建模 → 需要自然图像先验 → SD 先验强但多步慢且随机 → 用 LQ latent 起点和 VSD 支撑一步恢复。
2. OSEDiff 的核心不是“又做了一个加速器”，而是把 restoration 的保真项和 SD 的生成先验放进同一个训练目标。
3. 风险也已埋下：越依赖 SD 先验，越要警惕文字、小结构和身份类细节被幻觉化。

### 可追问点
- 为什么从 LQ latent 而不是纯噪声开始？
- VSD 在这里起什么作用？
