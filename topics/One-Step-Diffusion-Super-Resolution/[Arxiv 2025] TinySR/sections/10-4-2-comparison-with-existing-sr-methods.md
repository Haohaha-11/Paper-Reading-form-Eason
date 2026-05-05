[← 返回 README](../README.md)

# 4.2. Comparison with Existing SR Methods

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 TinySR 主线的关系**: TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

---

We categorize existing outstanding SR models into two groups, single-step and multi-step diffusion models. Singlestep models include SinSR [43], OSEDiff [48], TSD-SR [13], and AdcSR [5], and multi-step models include StableSR [41], DiffBIR [31], SeeSR [49], and ResShift [53]. Additional details of GAN-based Real-ISR methods [6, 30, 42, 54] are given in the supplementary material.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Quality Comparison. Tab. 1 shows a comparison with DMs-based baselines in Real-ISR tasks. For the first four full-reference metrics, our model achieves performance comparable to its teacher TSD-SR and outperforms most other models, securing the second rank for LPIPS and DISTS. Furthermore, our model achieves the best result for the distribution metric FID. For the latter three no-reference metrics, it demonstrates competitive performance: outperforming all other models for NIQE, and ranking second for both MUSIQ and CLIPIQA, thereby surpassing most other methods.

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

Table 1. Quantitative comparison of various methods on the DIV2K-Val dataset, with all efficiency metrics benchmarked on a NVIDIA V100 GPU. The best and second-best results are highlighted in bold, italic, respectively.

> 💡 **批注**: 这是实验证据：要同时看保真指标、感知指标和速度指标，避免被单一数字误导。

![Table 1.](../images/4caebd55230a5d11f456f4c411ee5698bac5c5d6e5606b81d03368efca4266cc.jpg)
*Table 1.: Table 1. Quantitative comparison of various methods on the DIV2K-Val dataset, with all efficiency metrics benchmarked on a NVIDIA V100 GPU. The best and second-best results are highlighted in bold, italic, respectively.*

> 💡 **Table 1. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

As illustrated in Fig. 1 and Fig. 8, TinySR achieves competitive performance in recovering high-quality, sharp, and photorealistic images. Visual artifacts, specifically the generation of fake textures, are frequently observed in outputs from multi-step models such as StableSR, SeeSR, DiffBIR, and ResShift due to their propensity for over-generation. A clear example, demonstrated in Fig. 8 (top), is the spurious generation of hair patterns in regions where they should not exist. OSEDiff and AdcSR consistently demonstrate suboptimal restoration performance, frequently yielding outputs with discernible blur. While TSD-SR exhibits excellent generation capabilities, it also tends to produce fake textures, as shown in Fig. 8 (bottom). TinySR, by contrast, demonstrates robust capabilities in reconstructing natural textures, notably encompassing structural integrity, botanical patterns, and sculpted surface details.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Efficiency Comparison. As demonstrated by the last four columns of Tab. 1, the proposed TinySR exhibits superior efficiency in terms of step number, inference time, and computational cost. By leveraging one-step inference and advanced compression, our model achieves dramatic efficiency gains over leading multi-step Real-ISR methods while maintaining comparable performance. Compared to StableSR, SeeSR, DiffBIR, and ResShift, our model delivers significant speed improvements $4 8 9 \times$ , $2 2 0 \times$ , $2 4 7 \times$ , $2 9 \times$ ) with corresponding MAC reductions $1 8 7 \times$ , $1 5 4 \times$ , $5 7 \times$ , and $1 2 \times$ ). Compared to the one-step model SinSR and OSEDiff, it achieves $8 . 1 \times$ and $6 . 4 \times$ acceleration, respectively. Compared to its teacher, TSD-SR, it achieves a $5 . 6 8 \times$ acceleration, a $84 \%$ reduction in computation, and a

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

Table 2. Performance comparison of depth pruning methods on the DIV2K-Val dataset. Our method exhibits superior recovery performance relative to other pruning strategies.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Table 2.](../images/fc649d2f3dd84d2d55fc64fa1601da1149277bb11fa0cba21f7c98ef6a50bd37.jpg)
*Table 2.: Table 2. Performance comparison of depth pruning methods on the DIV2K-Val dataset. Our method exhibits superior recovery performance relative to other pruning strategies.*

> 💡 **Table 2. 批读**: 表格要横向看 SOTA 排名，也要纵向看 fidelity 指标和 perceptual 指标是否相互牺牲。

$83 \%$ decrease in total parameters, as shown in Fig. 5. Notably, In a direct comparison with the current state-of-theart SR compression model, AdcSR, our model also demonstrates superior efficiency, with a $1 . 8 \times$ speedup and a $2 . 4 5 \times$ computation reduction.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
