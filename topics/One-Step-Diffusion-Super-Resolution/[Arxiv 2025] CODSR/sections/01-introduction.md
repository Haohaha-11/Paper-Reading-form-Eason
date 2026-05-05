[← 返回 README](../README.md)

# Introduction

## 📌 预览
本文件合并 Introduction/Related Work，重点读多步扩散慢、单步不可控/过重/不保真等动机链。

---

# 1. Introduction

The goal of the single image super-resolution (SISR) task is to recover a high-quality (HQ) image from its corresponding degraded low-quality (LQ) observation. Previous approaches, e.g., CNN-based [9, 12, 36, 38, 39, 66, 67], Transformer-based [5, 7, 8, 26, 58, 65], and GAN-based ones [24, 28, 45, 60], have markedly advanced the field of SISR by leveraging local information, capturing long-range dependencies, and incorporating adversarial learning. Nevertheless, they still fail to generate realistic and texture-rich details from severely degraded LQ inputs.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会影响生成细节与语义一致性。

Recent diffusion-based SISR methods [6, 14, 30, 43, 56] have drawn increasing attention by harnessing the strong generative priors of pretrained text-to-image diffusion models (e.g., Stable Diffusion (SD) [35]), achieving remarkable perceptual quality. However, these approaches generally depend on iterative denoising over dozens or even hundreds of steps, resulting in substantial computational overhead and limiting their practical applicability. To address this issue, recent efforts have focused on accelerating the diffusion process by simplifying it into a single-step procedure. Although current one-step methods have shown encouraging results, they still suffer from three major limitations.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Figure 1.](../images/8d7a56cfc73e54691061a0f9199e3e6d9b156a3f022583010371383bad06ba19.jpg)
*Figure 1.: Figure 1. Performance and runtime comparison among SD-based SISR methods on the DrealSR [49] benchmark. CODSR achieves high-quality reconstruction at a low latency, demonstrating a better balance between fidelity and generative quality compared with existing state-of-the-art diffusion-based methods.*

> 💡 **Figure 1. 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

A notable limitation of existing methods is the inferior fidelity performance. A primary factor contributing to this is the LQ information loss caused by the compression encoding. Existing methods [55] attempt to alleviate this by expanding the spatial resolution of features in the latent space, but the fundamental problem of information loss due to compression persists. To address this, we develop an LQguided feature modulation module that leverages original uncompressed information from LQ inputs to modulate the diffusion process for improved fidelity performance.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

In addition, existing one-step diffusion-based methods [13, 40, 50, 55] suffer from a lack of region-discriminative activation of generative priors. These approaches directly feed the LQ latent into the denoising network, deviating from the pre-trained diffusion model’s inherent denoising mode of recovering images from noisy latents. This deviation limits the model’s ability to extract and generate high-frequency information from noise, thereby inhibiting the full release of the model’s generative potential. Moreover, by treating all spatial regions equally, these methods overlook the varying demands for generative capacity across different image areas, often resulting in artifacts over smooth regions and insufficient detail in textured ones. To overcome these issues, we propose a region-adaptive generative prior activation method. This approach computes a Sobel-based gradient map from the grayscale LQ image to distinguish high-frequency regions from low-frequency ones and then adds adaptive noise to the high-frequency region, thereby enabling a region-aware activation of generative priors. The targeted approach ensures that generative priors are more effectively activated for various image areas during the reverse process, enhancing perceptual richness without compromising local structural fidelity.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Beyond the aforementioned limitations, a critical remaining challenge is the text misalignment. Text prompts offer valuable semantic guidance for SISR. Existing methods [50, 55] mostly employ DAPE [51] as a text extractor. However, they overlook the issue of text misalignment, where the interaction regions of the text embeddings in the diffusion model are not spatially aligned with corresponding semantic regions of the text. To rectify this misalignment and achieve precise semantic grounding, we propose a text-matching guidance method. This method leverages Grounded-SAM2 [34] to generate text-interactive region maps, which are then used to guide and spatially align the text–image interactions throughout the diffusion process.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会影响生成细节与语义一致性。

In this paper, we propose a controllable one-step diffusion network for super-resolution (CODSR), which activates the diffusion priors in a region-discriminative manner while improving fidelity by an LQ-guided feature modulation. Together with a text-matching guidance strategy, our CODSR shows a good capability of balancing fidelity and reality. The contributions are summarized as follows:

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

• We propose an LQ-guided feature modulation module that leverages original uncompressed information from LQ inputs to provide high-fidelity conditioning for the diffusion process. We present a region-adaptive generative prior activation method that effectively enhances perceptual richness without sacrificing local structural fidelity. • We introduce a text-matching guidance strategy that spatially aligns the text-image interactions to fully harness the conditioning potential of text prompts. • Extensive evaluations and analyses show that our CODSR achieves superior perceptual quality and competitive fidelity compared to state-of-the-art methods.

> 💡 **列表批读**: 这组条目通常是在列贡献、设置或发现；建议逐条对应到论文声称解决的 gap。

# 2. Related Work

Non-generation based methods. Early CNN-based methods [9, 36, 38, 39, 66, 67], initiated by SRCNN [12] focus on exploiting local information within LQ images to reconstruct HQ images. However, their restricted receptive field of CNN-based models hinders the integration of global image context and impedes further quality improvements. Transformer-based methods [5, 7, 8, 26, 58, 65] address this bottleneck by leveraging self-attention to capture longrange dependencies, leading to substantial performance improvements.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会影响生成细节与语义一致性。

GAN-based methods. Pioneering work such as SRGAN [24] enhances perceptual quality by combining adversarial [15] and perceptual losses. This approach is advanced by ESRGAN [44], which incorporates residual-in-residual dense blocks and a relativistic average discriminator to improve high-frequency detail reconstruction. To better model real-world degradations, BSRGAN [60] and Real-ESRGAN [45] subsequently employ complex degradation pipelines to synthesize more realistic training pairs, improving generalization ability on images with unknown degradations. Building on this progress, DASR [27] introduces a lightweight network to estimate degradation parameters for handling images with diverse degradation levels. Although these GAN-based methods significantly improve perceptual quality, their adversarial training often results in instability. Diffusion-based full-step methods. Diffusion models [10, 17, 33] have shown remarkable potential for SISR by leveraging powerful generative priors learned from large-scale text-to-image training. Early work [10, 21, 22, 30] adapts pre-trained denoising diffusion probabilistic models to restore images degraded by simple operators such as bicubic downsampling. The advent of Stable Diffusion (SD) [37] spurs a new wave of SD-based SISR methods, often incorporating controllable adapters [33, 62] for conditional guidance. StableSR [43] introduces a time-aware encoder and feature warping to balance fidelity and perceptual quality. DiffBIR [30] adopts a two-stage pipeline that first removes degradation and then refines details using SD. PASD [54] fuses pixel-level and semantic information through pixelaware cross-attention. SeeSR [51] enhances semantic robustness with degradation-aware tag-style prompts. However, these methods rely on multi-step denoising, resulting in significant computational costs and limited practicality.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近结构，又要生成自然高频细节。

Diffusion-based one-step methods. To address this, recent studies accelerate the diffusion process by compressing it into a single-step [59]. SinSR [46] derives a deterministic sampling process from ResShift [57] and adopts a consistency-preserving loss to improve the performance. AddSR [52] incorporates adversarial diffusion distillation and proposes a prediction-based self-refinement strategy to reconstruct richer high-frequency details without introducing additional modules. OSEDiff [50] directly uses the LQ image as input and applies the variational score distillation [48] to compress multi-step denoising into a single efficient generation step. TSD-SR [13] employs a target score distillation to enhance reality and a distribution-aware sampling mechanism to improve detail reconstruction. PiSA-SR [40] designs dual LoRA modules to separately learn pixel-level and semantic-level objectives, effectively balancing fidelity and perceptual quality while enabling flexible control of SISR results during inference. TVT [55] proposes a transfer VAE training strategy that adapts the $\times 8$ VAE of SD into a $\times 4$ variant through a two-stage training framework, preserving fine structural details. Although PiSA-SR and TVT improve fidelity to some extent, the issue of information loss due to compression still persists. HYPIR [31] leverages pretrained diffusion priors and lightweight adversarial LoRA fine-tuning, eliminating iterative sampling and distillation while achieving high-quality restoration. However, most aforementioned methods treat different regions in an image indiscriminately, thus failing to meet the requirements for generating images with different content regions. In contrast, we propose to explore region-discriminative activation of generative priors, in order to enhance perceptual richness without compromising local structural fidelity.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Figure 2.](../images/e37770d1438403beab89310079eef478ff63b34b91474aa306afc2912158b20a.jpg)
*Figure 2.: Figure 2. An overview of our CODSR. (a) The region-adaptive generative prior activation method introduces gradient-driven adaptive noise to achieve the region-aware activation of generative priors. (b) The LQ-guided feature modulation module exploits the uncompressed LQ information to modulate the diffusion process for restoring faithful structural details. (c) The text-matching guidance strategy harnesses the region maps generated by Grounded-SAM2 [34], which correspond to the textual descriptions, to constrain the text–image interaction regions within the cross-attention layers, thereby enabling effective textual guidance during generation.*

> 💡 **Figure 2. 批读**: 这张图通常承担方法动机、框架或视觉对比作用。重点看它证明的是质量、速度还是可控性。

---

## 🔖 Section 总结

### 核心洞察
1. 把本文 gap 和 one-step SR 主线对应起来。
2. 注意作者如何区分效率、保真、真实感和可控性。
