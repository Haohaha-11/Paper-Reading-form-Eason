[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
本节定位相关方法谱系，重点看 OSEDiff 如何区别于 GAN SR、多步 diffusion SR 和已有 one-step SR。
---

> 💡 **Q&A 批注记录**:
> - Q: 为什么从 LQ latent 而不是纯噪声开始？
> - A: LQ latent 已经包含布局和低频结构，单步模型只需补细节；纯噪声则需要在一步内同时生成结构和纹理，难度更高且不确定性更强。


Starting from SRCNN [13], deep learning-based methods have become prevalent for ISR. A variety of methods have been proposed [30, 66, 67, 9, 5, 29, 65, 6, 7] to improve the accuracy of ISR reconstruction. However, most of these methods assume simple and known degradations such as bicubic downsampling, limiting their applications to real-world images with complex and unknown degradations. In recent years, researches have been exploring the potentials of generative models, including GAN [14] and diffusion networks [16], for solving the Real-ISR problem.
> 💡 **谱系定位**: 相关工作从确定退化的 classic ISR 转向复杂退化的 Real-ISR。OSEDiff 站在 generative prior 这一支上，不再主要拼网络结构深度。


GAN-based Real-ISR. The use of GAN for photo-realistic ISR can be traced back to SRGAN [24], where the image degradation is assumed to be bicubic downsampling. Later on, researchers found that GAN has the potential to perform real-world image restoration with more complex degradations [61, 45]. Specifically, by using randomly shuffled degradation and high-order degradation to generate more realistic LQ-HQ training pairs, BSRGAN [61] and Real-ESRGAN [45] demonstrate promising Real-ISR results, which trigger many following works [4, 27, 28, 53]. DASR [28] designs a tiny network to predict the degradation parameters to handle degradations of various levels. SwinIR [29] switches the generator from CNNs to stronger transformers, further enhancing the performance of Real-ISR. However, the adversarial training process in GAN is unstable and its discriminator is limited in telling the quality of diverse natural image contents. Therefore, GAN-based Real-ISR methods often suffer from unnatural visual artifacts. Some works such as LDL [27] and DeSRA [53] can suppress much the artifacts, yet they are difficult to generate more natural details.
> 💡 **GAN 分支批读**: GAN-based Real-ISR 的优势通常是速度和局部锐化，短板是 discriminator 从头学、覆盖不了复杂自然图像分布。OSEDiff 后面用 frozen SD prior/VSD 替代这类从零开始的判别正则。


Diffusion-based Real-ISR. Some early attempts [21, 20, 47] employ the denoising diffusion probabilistic models (DDPMs) [16, 39, 11] to address the ISR problem by assuming simple and known degradations (e.g., bicubic downsampling). These methods are training-free by modifying the reverse transition of pre-trained DDPMs using gradient descent, but they cannot be applied to complex unknown degradations. Recent researches [42, 57, 31, 40, 59] have leveraged stronger pre-trained T2I models, such as Stable Diffusion (SD) [1], to tackle the Real-ISR problem. In general, they introduce an adapter [63] to fine-tune the SD model to reconstruct the HQ image with the LQ image as the control signal. StableSR [42] finetunes a time-aware encoder and employs feature warping to balance fidelity and perceptual quality. PASD [57] extracts both low-level and high-level features from the LQ image and inputs them to the pre-trained SD model with a pixel-aware cross attention module. To further enhance the semantic-aware ability of the Real-ISR model, SeeSR [52] introduces degradation-robust tag-style text prompts and utilizes soft prompts to guide the diffusion process. To mitigate the potential risks of diffusion uncertainty, CCSR [40] leverages a truncated diffusion process to recover structures and finetunes the VAE decoder by adversarial training to enhance details. SUPIR [59] leverages the powerful generation capability of SDXL model and the strong captioning capability of LLaVA [32] to synthesize rich image details.
> 💡 **多步 SD 分支**: StableSR/PASD/DiffBIR/SeeSR 共同点是复用 SD 先验并把 LQ 当控制信号；差异在控制特征、prompt 和稳定性处理。OSEDiff 的差异点是把 LQ latent 本身作为起点，减少随机采样和迭代开销。


The above mentioned methods, however, require tens or even hundreds of steps to complete the diffusion process, resulting in unfriendly latency. SinSR shortens ResShift [60] to a single-step inference by consistency preserving distillation. Nevertheless, the non-distrbution-based distillation loss tends to obtain smooth results, and the model capacity of SinSR and ResShift are much smaller than the SD models to address Real-ISR problems.
> 💡 **one-step 对比**: SinSR 也是一步，但 teacher 来自 ResShift，模型容量和分布先验弱于 SD；OSEDiff 要证明的是“一步”不必放弃大规模 T2I 先验。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 谱系 | GAN/Restoration → multi-step SD Real-ISR → one-step SD Real-ISR |
| 本文位置 | 把 LQ latent 直接作为扩散起点，用 latent-space VSD 将 Stable Diffusion 先验压缩到单步 Real-ISR。 |
| 比较维度 | 速度、保真、真实感、可控性、部署 |

### 核心洞察
1. 相关工作不是背景罗列，而是在排除已有方法为何不能直接解决本文痛点。
2. OSEDiff 的差异化应能用一句话说清：把 LQ latent 直接作为扩散起点，用 latent-space VSD 将 Stable Diffusion 先验压缩到单步 Real-ISR。
3. 读完本节应知道最强 baseline 分两类：多步 SD 方法负责质量对照，SinSR 负责一步效率对照。

### 可追问点
- 为什么从 LQ latent 而不是纯噪声开始？
- VSD 在这里起什么作用？
