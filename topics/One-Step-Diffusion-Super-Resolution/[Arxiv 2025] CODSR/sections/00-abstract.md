[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要快速给出本文问题、方法和结果。读完先记住一句话：CODSR 通过 LQ-guided feature modulation、区域自适应生成先验激活和 text-matching guidance，在单步 SR 中兼顾局部保真与感知质量。

---

# Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution

Hao Chen Junyang Chen Jinshan Pan Jiangxin Dong† School of Computer Science and Engineering, Nanjing University of Science and Technology https://github.com/Chanson94/CODSR

# Abstract

Recent diffusion-based one-step methods have shown remarkable progress in the field of image super-resolution, yet they remain constrained by three critical limitations: (1) inferior fidelity performance caused by the information loss from compression encoding of low-quality (LQ) inputs; (2) insufficient region-discriminative activation of generative priors; (3) misalignment between text prompts and their corresponding semantic regions. To address these limitations, we propose CODSR, a controllable one-step diffusion network for image super-resolution. First, we propose an LQ-guided feature modulation module that leverages original uncompressed information from LQ inputs to provide high-fidelity conditioning for the diffusion process. We then develop a region-adaptive generative prior activation method to effectively enhance perceptual richness without sacrificing local structural fidelity. Finally, we employ a text-matching guidance strategy to fully harness the conditioning potential of text prompts. Extensive experiments demonstrate that CODSR achieves superior perceptual quality and competitive fidelity compared with state-ofthe-art methods with efficient one-step inference.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
