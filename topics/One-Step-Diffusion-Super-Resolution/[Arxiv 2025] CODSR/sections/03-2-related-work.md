[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 主要定位方法谱系：传统/GAN SR、多步 diffusion SR、one-step/distillation SR，以及本文区别。

---

Non-generation based methods. Early CNN-based methods [9, 36, 38, 39, 66, 67], initiated by SRCNN [12] focus on exploiting local information within LQ images to reconstruct HQ images. However, their restricted receptive field of CNN-based models hinders the integration of global image context and impedes further quality improvements. Transformer-based methods [5, 7, 8, 26, 58, 65] address this bottleneck by leveraging self-attention to capture longrange dependencies, leading to substantial performance improvements.

> 💡 **批注**: 这里涉及条件信号：prompt 是否准确、是否退化感知，会直接影响生成细节与语义一致性。

GAN-based methods. Pioneering work such as SRGAN [24] enhances perceptual quality by combining adversarial [15] and perceptual losses. This approach is advanced by ESRGAN [44], which incorporates residual-in-residual dense blocks and a relativistic average discriminator to improve high-frequency detail reconstruction. To better model real-world degradations, BSRGAN [60] and Real-ESRGAN [45] subsequently employ complex degradation pipelines to synthesize more realistic training pairs, improving generalization ability on images with unknown degradations. Building on this progress, DASR [27] introduces a lightweight network to estimate degradation parameters for handling images with diverse degradation levels. Although these GAN-based methods significantly improve perceptual quality, their adversarial training often results in instability. Diffusion-based full-step methods. Diffusion models [10, 17, 33] have shown remarkable potential for SISR by leveraging powerful generative priors learned from large-scale text-to-image training. Early work [10, 21, 22, 30] adapts pre-trained denoising diffusion probabilistic models to restore images degraded by simple operators such as bicubic downsampling. The advent of Stable Diffusion (SD) [37] spurs a new wave of SD-based SISR methods, often incorporating controllable adapters [33, 62] for conditional guidance. StableSR [43] introduces a time-aware encoder and feature warping to balance fidelity and perceptual quality. DiffBIR [30] adopts a two-stage pipeline that first removes degradation and then refines details using SD. PASD [54] fuses pixel-level and semantic information through pixelaware cross-attention. SeeSR [51] enhances semantic robustness with degradation-aware tag-style prompts. However, these methods rely on multi-step denoising, resulting in significant computational costs and limited practicality.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

Diffusion-based one-step methods. To address this, recent studies accelerate the diffusion process by compressing it into a single-step [59]. SinSR [46] derives a deterministic sampling process from ResShift [57] and adopts a consistency-preserving loss to improve the performance. AddSR [52] incorporates adversarial diffusion distillation and proposes a prediction-based self-refinement strategy to reconstruct richer high-frequency details without introducing additional modules. OSEDiff [50] directly uses the LQ image as input and applies the variational score distillation [48] to compress multi-step denoising into a single efficient generation step. TSD-SR [13] employs a target score distillation to enhance reality and a distribution-aware sampling mechanism to improve detail reconstruction. PiSA-SR [40] designs dual LoRA modules to separately learn pixel-level and semantic-level objectives, effectively balancing fidelity and perceptual quality while enabling flexible control of SISR results during inference. TVT [55] proposes a transfer VAE training strategy that adapts the $\times 8$ VAE of SD into a $\times 4$ variant through a two-stage training framework, preserving fine structural details. Although PiSA-SR and TVT improve fidelity to some extent, the issue of information loss due to compression still persists. HYPIR [31] leverages pretrained diffusion priors and lightweight adversarial LoRA fine-tuning, eliminating iterative sampling and distillation while achieving high-quality restoration. However, most aforementioned methods treat different regions in an image indiscriminately, thus failing to meet the requirements for generating images with different content regions. In contrast, we propose to explore region-discriminative activation of generative priors, in order to enhance perceptual richness without compromising local structural fidelity.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Figure 2.](../images/e37770d1438403beab89310079eef478ff63b34b91474aa306afc2912158b20a.jpg)
*Figure 2.: Figure 2. An overview of our CODSR. (a) The region-adaptive generative prior activation method introduces gradient-driven adaptive noise to achieve the region-aware activation of generative priors. (b) The LQ-guided feature modulation module exploits the uncompressed LQ information to modulate the diffusion process for restoring faithful structural details. (c) The text-matching guidance strategy harnesses the region maps generated by Grounded-SAM2 [34], which correspond to the textual descriptions, to constrain the text–image interaction regions within the cross-attention layers, thereby enabling effective textual guidance during generation.*

> 💡 **Figure 2. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Core controls | region-adaptive prior activation + text-matching guidance |
| Project | https://github.com/Chanson94/CODSR |
| Benchmarks | RealSR/DrealSR and other real-world SR test sets in paper tables |
