[← 返回 README](../README.md)

# 2 Related Work

## 📌 预览
本节定位相关方法谱系，重点看作者如何区分自己与已有工作。

---

Starting from SRCNN [13], deep learning-based methods have become prevalent for ISR. A variety of methods have been proposed [30, 66, 67, 9, 5, 29, 65, 6, 7] to improve the accuracy of ISR reconstruction. However, most of these methods assume simple and known degradations such as bicubic downsampling, limiting their applications to real-world images with complex and unknown degradations. In recent years, researches have been exploring the potentials of generative models, including GAN [14] and diffusion networks [16], for solving the Real-ISR problem.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

GAN-based Real-ISR. The use of GAN for photo-realistic ISR can be traced back to SRGAN [24], where the image degradation is assumed to be bicubic downsampling. Later on, researchers found that GAN has the potential to perform real-world image restoration with more complex degradations [61, 45]. Specifically, by using randomly shuffled degradation and high-order degradation to generate more realistic LQ-HQ training pairs, BSRGAN [61] and Real-ESRGAN [45] demonstrate promising Real-ISR results, which trigger many following works [4, 27, 28, 53]. DASR [28] designs a tiny network to predict the degradation parameters to handle degradations of various levels. SwinIR [29] switches the generator from CNNs to stronger transformers, further enhancing the performance of Real-ISR. However, the adversarial training process in GAN is unstable and its discriminator is limited in telling the quality of diverse natural image contents. Therefore, GAN-based Real-ISR methods often suffer from unnatural visual artifacts. Some works such as LDL [27] and DeSRA [53] can suppress much the artifacts, yet they are difficult to generate more natural details.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Diffusion-based Real-ISR. Some early attempts [21, 20, 47] employ the denoising diffusion probabilistic models (DDPMs) [16, 39, 11] to address the ISR problem by assuming simple and known degradations (e.g., bicubic downsampling). These methods are training-free by modifying the reverse transition of pre-trained DDPMs using gradient descent, but they cannot be applied to complex unknown degradations. Recent researches [42, 57, 31, 40, 59] have leveraged stronger pre-trained T2I models, such as Stable Diffusion (SD) [1], to tackle the Real-ISR problem. In general, they introduce an adapter [63] to fine-tune the SD model to reconstruct the HQ image with the LQ image as the control signal. StableSR [42] finetunes a time-aware encoder and employs feature warping to balance fidelity and perceptual quality. PASD [57] extracts both low-level and high-level features from the LQ image and inputs them to the pre-trained SD model with a pixel-aware cross attention module. To further enhance the semantic-aware ability of the Real-ISR model, SeeSR [52] introduces degradation-robust tag-style text prompts and utilizes soft prompts to guide the diffusion process. To mitigate the potential risks of diffusion uncertainty, CCSR [40] leverages a truncated diffusion process to recover structures and finetunes the VAE decoder by adversarial training to enhance details. SUPIR [59] leverages the powerful generation capability of SDXL model and the strong captioning capability of LLaVA [32] to synthesize rich image details.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

The above mentioned methods, however, require tens or even hundreds of steps to complete the diffusion process, resulting in unfriendly latency. SinSR shortens ResShift [60] to a single-step inference by consistency preserving distillation. Nevertheless, the non-distrbution-based distillation loss tends to obtain smooth results, and the model capacity of SinSR and ResShift are much smaller than the SD models to address Real-ISR problems.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
