[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
Related Work 主要定位方法谱系：传统/GAN SR、多步 diffusion SR、one-step/distillation SR，以及本文区别。

---

Starting from SRCNN [13], deep learning-based methods have become prevalent for ISR. A variety of methods have been proposed [30, 66, 67, 9, 5, 29, 65, 6, 7] to improve the accuracy of ISR reconstruction. However, most of these methods assume simple and known degradations such as bicubic downsampling, limiting their applications to real-world images with complex and unknown degradations. In recent years, researches have been exploring the potentials of generative models, including GAN [14] and diffusion networks [16], for solving the Real-ISR problem.

> 💡 **批注**: 这是控制机制：作者试图把退化强度、区域差异或 timestep 映射成可操作的生成强度。

GAN-based Real-ISR. The use of GAN for photo-realistic ISR can be traced back to SRGAN [24], where the image degradation is assumed to be bicubic downsampling. Later on, researchers found that GAN has the potential to perform real-world image restoration with more complex degradations [61, 45]. Specifically, by using randomly shuffled degradation and high-order degradation to generate more realistic LQ-HQ training pairs, BSRGAN [61] and Real-ESRGAN [45] demonstrate promising Real-ISR results, which trigger many following works [4, 27, 28, 53]. DASR [28] designs a tiny network to predict the degradation parameters to handle degradations of various levels. SwinIR [29] switches the generator from CNNs to stronger transformers, further enhancing the performance of Real-ISR. However, the adversarial training process in GAN is unstable and its discriminator is limited in telling the quality of diverse natural image contents. Therefore, GAN-based Real-ISR methods often suffer from unnatural visual artifacts. Some works such as LDL [27] and DeSRA [53] can suppress much the artifacts, yet they are difficult to generate more natural details.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Diffusion-based Real-ISR. Some early attempts [21, 20, 47] employ the denoising diffusion probabilistic models (DDPMs) [16, 39, 11] to address the ISR problem by assuming simple and known degradations (e.g., bicubic downsampling). These methods are training-free by modifying the reverse transition of pre-trained DDPMs using gradient descent, but they cannot be applied to complex unknown degradations. Recent researches [42, 57, 31, 40, 59] have leveraged stronger pre-trained T2I models, such as Stable Diffusion (SD) [1], to tackle the Real-ISR problem. In general, they introduce an adapter [63] to fine-tune the SD model to reconstruct the HQ image with the LQ image as the control signal. StableSR [42] finetunes a time-aware encoder and employs feature warping to balance fidelity and perceptual quality. PASD [57] extracts both low-level and high-level features from the LQ image and inputs them to the pre-trained SD model with a pixel-aware cross attention module. To further enhance the semantic-aware ability of the Real-ISR model, SeeSR [52] introduces degradation-robust tag-style text prompts and utilizes soft prompts to guide the diffusion process. To mitigate the potential risks of diffusion uncertainty, CCSR [40] leverages a truncated diffusion process to recover structures and finetunes the VAE decoder by adversarial training to enhance details. SUPIR [59] leverages the powerful generation capability of SDXL model and the strong captioning capability of LLaVA [32] to synthesize rich image details.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

The above mentioned methods, however, require tens or even hundreds of steps to complete the diffusion process, resulting in unfriendly latency. SinSR shortens ResShift [60] to a single-step inference by consistency preserving distillation. Nevertheless, the non-distrbution-based distillation loss tends to obtain smooth results, and the model capacity of SinSR and ResShift are much smaller than the SD models to address Real-ISR problems.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Inference time | 0.11s on A100 for 512×512 input |
| Trainable parameters | 8.5M |
| Speedup claim | over 100× faster than StableSR in paper comparison |
