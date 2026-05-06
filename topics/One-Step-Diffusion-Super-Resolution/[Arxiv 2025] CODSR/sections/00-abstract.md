[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要先给出三条痛点：VAE 压缩导致 LQ 信息丢失、生成先验缺少区域差异、文本提示与语义区域错位。读的时候要把三条痛点分别对应到 LQFM、RGPA、TMG，而不是只记“又一个 one-step SR”。
---

> 💡 **Q&A 批注记录**:
> - Q: RGPA 和普通加噪有什么区别？
> - A: 普通 latent 加噪是全图同一强度，会同时增强纹理区和干净平坦区；RGPA 先用 Sobel 梯度估计区域纹理需求，再对噪声做空间加权，所以它是在局部调节 generative prior 的释放强度。


# Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution

Hao Chen Junyang Chen Jinshan Pan Jiangxin Dong† School of Computer Science and Engineering, Nanjing University of Science and Technology https://github.com/Chanson94/CODSR
> 💡 **标题批读**: 标题里的 controllable 不应理解成任意编辑，而是围绕 fidelity-reality 的可控折中：timestep/noise intensity 给全局强度，RGPA 给区域差异，LQFM/TMG分别拉回结构和语义边界。


# Abstract

Recent diffusion-based one-step methods have shown remarkable progress in the field of image super-resolution, yet they remain constrained by three critical limitations: (1) inferior fidelity performance caused by the information loss from compression encoding of low-quality (LQ) inputs; (2) insufficient region-discriminative activation of generative priors; (3) misalignment between text prompts and their corresponding semantic regions. To address these limitations, we propose CODSR, a controllable one-step diffusion network for image super-resolution. First, we propose an LQ-guided feature modulation module that leverages original uncompressed information from LQ inputs to provide high-fidelity conditioning for the diffusion process. We then develop a region-adaptive generative prior activation method to effectively enhance perceptual richness without sacrificing local structural fidelity. Finally, we employ a text-matching guidance strategy to fully harness the conditioning potential of text prompts. Extensive experiments demonstrate that CODSR achieves superior perceptual quality and competitive fidelity compared with state-ofthe-art methods with efficient one-step inference.
> 💡 **摘要主线**: 摘要的三段式很清楚：LQFM 解决 latent 压缩后的低层保真缺口，RGPA 解决全图统一加噪导致的局部 hallucination，TMG 解决 prompt token 没有落到对应语义区域的问题。这里的核心 claim 是单步效率不变，同时把真实感控制得更局部。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 一句话 | 在 one-step diffusion SR 中，用 LQFM 保结构、RGPA 分区域放生成先验、TMG 对齐文本区域。 |
| 输入 | LQ image、自动 prompt、timestep/noise intensity |
| 输出 | 兼顾结构保真与局部真实纹理的 SR image |

### 核心洞察
1. 摘要里的三条 limitation 就是后文消融的检查清单：LQFM、RGPA、TMG 是否分别有效。
2. “superior perceptual quality and competitive fidelity” 需要在指标上拆开看，尤其要注意 no-reference 指标提升是否伴随 PSNR/SSIM 下滑。
3. CODSR 真正想补的是 one-step 压缩后丢掉的生成自由度，但它只在需要纹理的局部释放这部分自由度。

### 可追问点
- RGPA 和普通加噪有什么区别？
- 为什么还要 LQFM？
- 这个 controllable 是用户可控，还是模型内部的区域/噪声强度可控？
