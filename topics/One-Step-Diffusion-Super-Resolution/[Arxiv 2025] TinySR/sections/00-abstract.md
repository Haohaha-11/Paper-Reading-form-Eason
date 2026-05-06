[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要快速定位 TinySR 的问题、方法与结果。先记住主线：把 one-step diffusion SR teacher 系统性压缩成更小、更快的 student，目标是实时 Real-ISR。
---

> 💡 **Q&A 批注记录**:
> - Q: TinySR 的核心不是新的生成先验吗？
> - A: 对，它更像系统压缩论文：核心贡献在剪枝和模块移除，让已有 one-step diffusion SR 更小更快。


# TinySR: Shallow Diffusion Transformers for Real-World Image Super-Resolution

Linwei Dong1 Qingnan Fan2 Yuhang $\mathrm { Y u } ^ { 2 }$ Qi Zhang2 Jinwei Chen2 Yawei Luo1† Changqing Zou1, 3 1Zhejiang University 2Vivo Mobile Communication Co. Ltd 3Zhejiang Lab
> 💡 **段落批读**: 这段应服务于 TinySR 的主 claim：把 one-step diffusion SR teacher 系统性压缩成更小、更快的 student，目标是实时 Real-ISR。


# Abstract

Real-world image super-resolution (Real-ISR) focuses on recovering high-quality images from low-resolution inputs that suffer from complex degradations like noise, blur, and compression. Recently, diffusion models (DMs) have shown great potential in this area by leveraging strong generative priors to restore fine details. However, their iterative denoising process incurs high computational overhead, posing challenges for real-time applications. Although one-step distillation methods, such as OSEDiff and TSD-SR, offer faster inference, they remain fundamentally constrained by their large, over-parameterized model architectures. In this work, we present TinySR, a compact yet effective diffusion model specifically designed for Real-ISR that achieves real-time performance while maintaining perceptual quality. We introduce Dynamic Interblock Activation and Expansion-Corrosion Strategy to facilitate more effective decision-making in depth pruning. We achieve VAE compression through channel pruning, attention blocks removal and lightweight SepConv. We eliminate time- and prompt-related modules and perform pre-caching techniques to further speed up the model. TinySR significantly reduces computational cost and model size, achieving up to $5 . 6 8 \times$ speedup and $83 \%$ parameter reduction compared to its teacher TSD-SR, while still providing high quality results. Our code is released at https://github.com/Microtreei/TinySR.
> 💡 **摘要批读**: 摘要把 TinySR 的四个压缩点一次性列全了：深度剪枝、VAE 压缩、time/prompt 模块移除、调制参数缓存。核心证据是相对 TSD-SR teacher 的 5.68x 加速和 83% 参数减少，所以后文要逐项核对这些速度来自哪里、质量掉了多少。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 一句话 | 把 one-step diffusion SR teacher 系统性压缩成更小、更快的 student，目标是实时 Real-ISR。 |
| 输入 | LQ image + one-step diffusion SR teacher |
| 输出 | 轻量化 one-step SR student 与实时 SR image |

### 核心洞察
1. 摘要先建立全文地图：任务、短板、核心模块、实验结论。
2. 不要把摘要当结论读，后面要逐项验证它的 claim。
3. 最重要的变量是剪枝后的 recoverability：student 是否还能通过 teacher distillation 保住 Real-ISR 的纹理和结构。

### 可追问点
- TinySR 的核心不是新的生成先验吗？
- 为什么要同时压缩 VAE？
- 5.68x 加速中，denoiser 剪枝、VAE 压缩、条件模块移除各贡献多少？
