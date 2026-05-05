[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文件保留论文题名、作者、摘要等开场信息，先抓住方法解决的 one-step SR 核心瓶颈。

---

# TinySR: Shallow Diffusion Transformers for Real-World Image Super-Resolution

Linwei Dong1 Qingnan Fan2 Yuhang $\mathrm { Y u } ^ { 2 }$ Qi Zhang2 Jinwei Chen2 Yawei Luo1† Changqing Zou1, 3 1Zhejiang University 2Vivo Mobile Communication Co. Ltd 3Zhejiang Lab

# Abstract

Real-world image super-resolution (Real-ISR) focuses on recovering high-quality images from low-resolution inputs that suffer from complex degradations like noise, blur, and compression. Recently, diffusion models (DMs) have shown great potential in this area by leveraging strong generative priors to restore fine details. However, their iterative denoising process incurs high computational overhead, posing challenges for real-time applications. Although one-step distillation methods, such as OSEDiff and TSD-SR, offer faster inference, they remain fundamentally constrained by their large, over-parameterized model architectures. In this work, we present TinySR, a compact yet effective diffusion model specifically designed for Real-ISR that achieves real-time performance while maintaining perceptual quality. We introduce Dynamic Interblock Activation and Expansion-Corrosion Strategy to facilitate more effective decision-making in depth pruning. We achieve VAE compression through channel pruning, attention blocks removal and lightweight SepConv. We eliminate time- and prompt-related modules and perform pre-caching techniques to further speed up the model. TinySR significantly reduces computational cost and model size, achieving up to $5 . 6 8 \times$ speedup and $83 \%$ parameter reduction compared to its teacher TSD-SR, while still providing high quality results. Our code is released at https://github.com/Microtreei/TinySR.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

---

## 🔖 Section 总结

### 核心洞察
1. 先记住本文的问题定义、方法名和主要 claim。
2. 后续阅读时回看摘要里的 claim 是否被实验支持。
