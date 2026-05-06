[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
本节建立动机链：多步 diffusion SR 质量好但慢，one-step SR 快但压缩了生成过程；CODSR 的切入点是把“快”保住，同时补上 LQ 保真、区域生成先验、文本区域对齐三类短板。
---

> 💡 **Q&A 批注记录**:
> - Q: RGPA 和普通加噪有什么区别？
> - A: 普通 latent 加噪会让整张图一起变得更生成化；RGPA 用 LQ 梯度图把噪声强度变成空间变量，让纹理区域多放先验、平坦区域少放先验。


The goal of the single image super-resolution (SISR) task is to recover a high-quality (HQ) image from its corresponding degraded low-quality (LQ) observation. Previous approaches, e.g., CNN-based [9, 12, 36, 38, 39, 66, 67], Transformer-based [5, 7, 8, 26, 58, 65], and GAN-based ones [24, 28, 45, 60], have markedly advanced the field of SISR by leveraging local information, capturing long-range dependencies, and incorporating adversarial learning. Nevertheless, they still fail to generate realistic and texture-rich details from severely degraded LQ inputs.
> 💡 **动机推进**: 这段先把传统 SR 的短板定位到“严重退化时缺真实纹理”。这给 diffusion prior 登场铺路，但也埋下后文的风险：生成纹理越强，越需要机制约束它别画错。


Recent diffusion-based SISR methods [6, 14, 30, 43, 56] have drawn increasing attention by harnessing the strong generative priors of pretrained text-to-image diffusion models (e.g., Stable Diffusion (SD) [35]), achieving remarkable perceptual quality. However, these approaches generally depend on iterative denoising over dozens or even hundreds of steps, resulting in substantial computational overhead and limiting their practical applicability. To address this issue, recent efforts have focused on accelerating the diffusion process by simplifying it into a single-step procedure. Although current one-step methods have shown encouraging results, they still suffer from three major limitations.
> 💡 **速度矛盾**: 多步 diffusion 的优势来自反复 denoising/refinement，代价是延迟；one-step 方法把过程压成一次 U-Net 前向，效率上去了，但更容易丢失迭代过程中的修正机会。CODSR 后面的模块都是在补这个压缩损失。


![Figure 1.](../images/8d7a56cfc73e54691061a0f9199e3e6d9b156a3f022583010371383bad06ba19.jpg)
*Figure 1.: Figure 1. Performance and runtime comparison among SD-based SISR methods on the DrealSR [49] benchmark. CODSR achieves high-quality reconstruction at a low latency, demonstrating a better balance between fidelity and generative quality compared with existing state-of-the-art diffusion-based methods.*
> 💡 **Figure 批读**: Figure 1 不是只展示“更快更好”，还在暗示本文的目标区间：接近 one-step 的低延迟，同时把感知质量推近甚至超过 full-step。读图时要同时看运行时间、fidelity 指标和 perceptual 指标。


A notable limitation of existing methods is the inferior fidelity performance. A primary factor contributing to this is the LQ information loss caused by the compression encoding. Existing methods [55] attempt to alleviate this by expanding the spatial resolution of features in the latent space, but the fundamental problem of information loss due to compression persists. To address this, we develop an LQguided feature modulation module that leverages original uncompressed information from LQ inputs to modulate the diffusion process for improved fidelity performance.
> 💡 **LQFM 动机**: 这里的关键不是“再加一个条件分支”，而是指出 VAE encoder 已经把原始 LQ 的部分像素级结构压掉了。LQFM 等于给 U-Net 一条绕过 latent bottleneck 的保真通道，特别影响文字、边缘、细小结构。


In addition, existing one-step diffusion-based methods [13, 40, 50, 55] suffer from a lack of region-discriminative activation of generative priors. These approaches directly feed the LQ latent into the denoising network, deviating from the pre-trained diffusion model’s inherent denoising mode of recovering images from noisy latents. This deviation limits the model’s ability to extract and generate high-frequency information from noise, thereby inhibiting the full release of the model’s generative potential. Moreover, by treating all spatial regions equally, these methods overlook the varying demands for generative capacity across different image areas, often resulting in artifacts over smooth regions and insufficient detail in textured ones. To overcome these issues, we propose a region-adaptive generative prior activation method. This approach computes a Sobel-based gradient map from the grayscale LQ image to distinguish high-frequency regions from low-frequency ones and then adds adaptive noise to the high-frequency region, thereby enabling a region-aware activation of generative priors. The targeted approach ensures that generative priors are more effectively activated for various image areas during the reverse process, enhancing perceptual richness without compromising local structural fidelity.
> 💡 **RGPA 动机**: 这段把问题说得很准：one-step 方法直接把 z_L 丢给 U-Net，和 pretrained diffusion “从 noisy latent 恢复图像”的工作模式不一致；但全图加噪又会污染平坦区。因此 RGPA 的价值在“加噪”与“保局部结构”之间做空间化折中。


Beyond the aforementioned limitations, a critical remaining challenge is the text misalignment. Text prompts offer valuable semantic guidance for SISR. Existing methods [50, 55] mostly employ DAPE [51] as a text extractor. However, they overlook the issue of text misalignment, where the interaction regions of the text embeddings in the diffusion model are not spatially aligned with corresponding semantic regions of the text. To rectify this misalignment and achieve precise semantic grounding, we propose a text-matching guidance method. This method leverages Grounded-SAM2 [34] to generate text-interactive region maps, which are then used to guide and spatially align the text–image interactions throughout the diffusion process.
> 💡 **TMG 动机**: 文本提示在 SR 里不是描述整张图越丰富越好，而是要让“bird/flower/text”等 token 的 cross-attention 落在对应区域。若 token 对错位置，生成先验会把羽毛、文字纹理之类的细节画到不该出现的地方。


In this paper, we propose a controllable one-step diffusion network for super-resolution (CODSR), which activates the diffusion priors in a region-discriminative manner while improving fidelity by an LQ-guided feature modulation. Together with a text-matching guidance strategy, our CODSR shows a good capability of balancing fidelity and reality. The contributions are summarized as follows:
> 💡 **贡献合并读**: 这句话把三个模块合成一个系统：RGPA 决定哪里释放生成能力，LQFM 把 LQ 结构拉回来，TMG 决定文本先验作用到哪里。三者分别对应 realism、fidelity、semantic grounding。


• We propose an LQ-guided feature modulation module that leverages original uncompressed information from LQ inputs to provide high-fidelity conditioning for the diffusion process. We present a region-adaptive generative prior activation method that effectively enhances perceptual richness without sacrificing local structural fidelity. • We introduce a text-matching guidance strategy that spatially aligns the text-image interactions to fully harness the conditioning potential of text prompts. • Extensive evaluations and analyses show that our CODSR achieves superior perceptual quality and competitive fidelity compared to state-of-the-art methods.
> 💡 **贡献检查**: 贡献列表后面必须能在实验里逐条找到证据：LQFM 看 Table 4，RGPA 看 Table 2/3 与 Figure 4/6，TMG 看 Table 2 与 DAAM 可视化，整体 SOTA 看 Table 1。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 任务 | Real-world image super-resolution |
| 主矛盾 | one-step 效率 vs diffusion prior 真实感 vs LQ 结构保真 |
| 本文回答 | 用 LQFM/RGPA/TMG 把 fidelity-realism trade-off 从全图平均控制改成局部机制控制。 |

### 核心洞察
1. Introduction 的逻辑链是：传统 SR 不够真实 → 多步 diffusion 真实但慢 → one-step 快但损失保真/区域性/语义对齐 → CODSR 用三个模块补短板。
2. 这篇的“controllable”主要是噪声强度与区域自适应控制，不是交互式编辑式控制。
3. 论文最有价值的问题意识是局部 fidelity-realism trade-off：同一张图里不同区域需要不同生成强度。

### 可追问点
- RGPA 和普通加噪有什么区别？
- 为什么还要 LQFM？
- t_s 增大时，哪些区域最先从真实纹理变成 hallucination？
