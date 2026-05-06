[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要快速给出本文问题、方法和结果。读完先记住一句话：TinySR 面向实时 Real-ISR，把 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存压成轻量单步模型。

---

# TinySR: Shallow Diffusion Transformers for Real-World Image Super-Resolution

Linwei Dong1 Qingnan Fan2 Yuhang $\mathrm { Y u } ^ { 2 }$ Qi Zhang2 Jinwei Chen2 Yawei Luo1† Changqing Zou1, 3 1Zhejiang University 2Vivo Mobile Communication Co. Ltd 3Zhejiang Lab

# Abstract

Real-world image super-resolution (Real-ISR) focuses on recovering high-quality images from low-resolution inputs that suffer from complex degradations like noise, blur, and compression. Recently, diffusion models (DMs) have shown great potential in this area by leveraging strong generative priors to restore fine details. However, their iterative denoising process incurs high computational overhead, posing challenges for real-time applications. Although one-step distillation methods, such as OSEDiff and TSD-SR, offer faster inference, they remain fundamentally constrained by their large, over-parameterized model architectures. In this work, we present TinySR, a compact yet effective diffusion model specifically designed for Real-ISR that achieves real-time performance while maintaining perceptual quality. We introduce Dynamic Interblock Activation and Expansion-Corrosion Strategy to facilitate more effective decision-making in depth pruning. We achieve VAE compression through channel pruning, attention blocks removal and lightweight SepConv. We eliminate time- and prompt-related modules and perform pre-caching techniques to further speed up the model. TinySR significantly reduces computational cost and model size, achieving up to $5 . 6 8 \times$ speedup and $83 \%$ parameter reduction compared to its teacher TSD-SR, while still providing high quality results. Our code is released at https://github.com/Microtreei/TinySR.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
