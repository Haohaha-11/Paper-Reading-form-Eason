[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要给出问题、核心方法和主张结果；读这段时要抓住作者认为的“旧方法瓶颈”和新方法的唯一卖点。

> 💡 **与 CODSR 主线的关系**: CODSR 从 LQ 信息损失、区域差异化先验激活和文本-区域错配三方面补强 controllable one-step diffusion SR，在单步推理下兼顾感知质量与局部保真。

---

Recent diffusion-based one-step methods have shown remarkable progress in the field of image super-resolution, yet they remain constrained by three critical limitations: (1) inferior fidelity performance caused by the information loss from compression encoding of low-quality (LQ) inputs; (2) insufficient region-discriminative activation of generative priors; (3) misalignment between text prompts and their corresponding semantic regions. To address these limitations, we propose CODSR, a controllable one-step diffusion network for image super-resolution. First, we propose an LQ-guided feature modulation module that leverages original uncompressed information from LQ inputs to provide high-fidelity conditioning for the diffusion process. We then develop a region-adaptive generative prior activation method to effectively enhance perceptual richness without sacrificing local structural fidelity. Finally, we employ a text-matching guidance strategy to fully harness the conditioning potential of text prompts. Extensive experiments demonstrate that CODSR achieves superior perceptual quality and competitive fidelity compared with state-ofthe-art methods with efficient one-step inference.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

---

## 🔖 Section 总结

### 核心洞察

1. 抓住作者定义的核心 gap。
2. 把方法名和它解决的瓶颈对应起来。
3. 记录 claimed result，但等 Experiments 再验证。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Core controls | region-adaptive prior activation + text-matching guidance |
| Project | https://github.com/Chanson94/CODSR |
| Benchmarks | RealSR/DrealSR and other real-world SR test sets in paper tables |
