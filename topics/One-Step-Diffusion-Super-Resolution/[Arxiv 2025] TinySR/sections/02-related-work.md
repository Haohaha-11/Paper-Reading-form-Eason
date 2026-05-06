[← 返回 README](../README.md)

# 2. Related Work

## 📌 预览
本节定位相关方法谱系，重点看 TinySR 如何区别于 GAN SR、多步 diffusion SR 和已有 one-step SR。
---

> 💡 **Q&A 批注记录**:
> - Q: TinySR 的核心不是新的生成先验吗？
> - A: 对，它更像系统压缩论文：核心贡献在剪枝和模块移除，让已有 one-step diffusion SR 更小更快。


Real-World Image Super-Resolution. Real-world image super-resolution (Real-ISR) addresses the challenges of reconstructing high-resolution images from low-quality inputs affected by complex, unknown degradations. Early approaches like BSRGAN [54] and Real-ESRGAN [42] pioneered synthetic degradation modeling using random blur, noise, and compression patterns to enhance generalization. While these methods improved the model’s performance, they often introduced undesirable artifacts. The emergence of diffusion models, particularly Stable Diffusion [14, 36], marked a significant advancement in perceptual quality. Techniques incorporating StableSR [40] , DiffBIR [31] and SeeSR [49] demonstrated remarkable results in SR tasks, but their iterative denoising process rendered them impractical for time-sensitive applications. Recent efforts have focused on distilling multi-step diffusion processes into efficient one-step networks. Real-ISR methods such as OSEDiff [48] and TSD-SR [13] introduced specialized distillation techniques for this purpose. Nevertheless, these models still inherit the substantial computational overhead of their diffusion backbones, with parameter counts often exceeding one billion. This high complexity poses a significant challenge for their deployment on mobile and edge devices.
> 💡 **问题背景**: Real-ISR 的难点在 unknown degradation：真实低质图不是简单 bicubic，下游方法必须同时处理模糊、噪声、压缩和纹理缺失。


![Figure 3.](../images/37ff64f4985716de8c48da36e2cd80195649dac190aaaad40bdbe6276a0f0b75.jpg)
*Figure 3.: Figure 3. Depth pruning closely aligns with the theoretical linear acceleration curve compared with width pruning.*
> 💡 **Figure 批读**: Fig. 3 是选择 depth pruning 的依据：删掉整层/block 通常比 width pruning 更接近理论线性加速。注意它只说明速度潜力，不自动保证剪后质量，所以后面还需要 recoverability 和蒸馏实验。


Efficient Pruning and Compression Techniques. The deployment of large diffusion models on resource-constrained hardware necessitates efficient model compression techniques. TinyFusion [15] enables real-time generation by using learnable depth pruning, which is optimized with LoRA-based fine-tuning and Gumbel-Softmax sampling [22]. Other methods, such as BK-SDM [25] and Snap-Fusion [28], reduce model size and latency through structural pruning and on-the-fly architecture modification. In the domain of super-resolution, AdcSR [5] introduces an adversarial compression methodology that achieves a $3 . 7 \times$ speedup and a $74 \%$ reduction in parameters, while preserving output quality through well-designed adversarial distillation. Collectively, these methods represent a substantial advancement in model compression, enabling resourceconstrained hardware to generate high-quality outputs with improved computational efficiency.
> 💡 **相关工作定位**: TinySR 站在两个交叉点上：借 TinyFusion/BK-SDM/SnapFusion 的 diffusion 压缩思想，又面向 AdcSR/TSD-SR 这类 Real-ISR 场景。它的不同点是同时处理 denoiser、VAE 和条件模块，而不是只做 backbone 压缩。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 谱系 | GAN/Restoration → multi-step diffusion → one-step diffusion → compressed one-step SR |
| 本文位置 | 把 one-step diffusion SR teacher 系统性压缩成更小、更快的 student，目标是实时 Real-ISR。 |
| 比较维度 | 速度、保真、真实感、参数量、部署 |

### 核心洞察
1. GAN SR 快但真实纹理能力有限，多步 diffusion 质量强但太慢，one-step diffusion 快了但模型仍大。
2. TinySR 的差异化是 compressed one-step SR：以 TSD-SR 为 teacher，把结构冗余剪掉并通过蒸馏恢复质量。
3. 相关工作为后文实验划定 baseline：OSEDiff/TSD-SR/SinSR/AdcSR 看 SR trade-off，TinyFusion/BK-SDM/SnapFusion 看剪枝策略。

### 可追问点
- TinySR 的核心不是新的生成先验吗？
- 为什么要同时压缩 VAE？
- 和 AdcSR 相比，TinySR 的 VAE 压缩是否减少了 denoiser/VAE 之间的结构耦合？
