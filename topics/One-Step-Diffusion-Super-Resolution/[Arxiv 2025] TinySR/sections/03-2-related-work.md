[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
Related Work 主要定位方法谱系：传统/GAN SR、多步 diffusion SR、one-step/distillation SR，以及本文区别。

---

Real-World Image Super-Resolution. Real-world image super-resolution (Real-ISR) addresses the challenges of reconstructing high-resolution images from low-quality inputs affected by complex, unknown degradations. Early approaches like BSRGAN [54] and Real-ESRGAN [42] pioneered synthetic degradation modeling using random blur, noise, and compression patterns to enhance generalization. While these methods improved the model’s performance, they often introduced undesirable artifacts. The emergence of diffusion models, particularly Stable Diffusion [14, 36], marked a significant advancement in perceptual quality. Techniques incorporating StableSR [40] , DiffBIR [31] and SeeSR [49] demonstrated remarkable results in SR tasks, but their iterative denoising process rendered them impractical for time-sensitive applications. Recent efforts have focused on distilling multi-step diffusion processes into efficient one-step networks. Real-ISR methods such as OSEDiff [48] and TSD-SR [13] introduced specialized distillation techniques for this purpose. Nevertheless, these models still inherit the substantial computational overhead of their diffusion backbones, with parameter counts often exceeding one billion. This high complexity poses a significant challenge for their deployment on mobile and edge devices.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

![Figure 3.](../images/37ff64f4985716de8c48da36e2cd80195649dac190aaaad40bdbe6276a0f0b75.jpg)
*Figure 3.: Figure 3. Depth pruning closely aligns with the theoretical linear acceleration curve compared with width pruning.*

> 💡 **Figure 3. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

Efficient Pruning and Compression Techniques. The deployment of large diffusion models on resource-constrained hardware necessitates efficient model compression techniques. TinyFusion [15] enables real-time generation by using learnable depth pruning, which is optimized with LoRA-based fine-tuning and Gumbel-Softmax sampling [22]. Other methods, such as BK-SDM [25] and Snap-Fusion [28], reduce model size and latency through structural pruning and on-the-fly architecture modification. In the domain of super-resolution, AdcSR [5] introduces an adversarial compression methodology that achieves a $3 . 7 \times$ speedup and a $74 \%$ reduction in parameters, while preserving output quality through well-designed adversarial distillation. Collectively, these methods represent a substantial advancement in model compression, enabling resourceconstrained hardware to generate high-quality outputs with improved computational efficiency.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

---

## 🔖 Section 总结

### 核心洞察

1. 本节帮助定位论文贡献边界。
2. 读完后回到 README 的方法对比表中归纳。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
