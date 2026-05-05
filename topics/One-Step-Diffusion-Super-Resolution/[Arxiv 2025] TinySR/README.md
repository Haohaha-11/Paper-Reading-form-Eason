# TinySR: Pruning Diffusion for Real-World Image Super-Resolution

**作者**: Linwei Dong; Qingnan Fan; Yuhang Yu; Qi Zhang; Jinwei Chen; Yawei Luo; Changqing Zou  
**会议/年份**: Arxiv 2025  
**链接**: [arXiv](https://arxiv.org/abs/2508.17434) | [PDF](https://arxiv.org/pdf/2508.17434v2)  
**主题**: One-Step Diffusion Super-Resolution  
**阅读日期**: 2026-05-05

## 一句话总结

TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

## Why Read This Paper

- 它回答 one-step diffusion SR 的第二层问题：单步之后模型本身仍然太大，如何实时化。
- 它把架构剪枝、VAE 压缩和推理缓存组合成系统级加速方案。
- 适合与你课题中的 deployment/edge SR 方向做对比。

## 核心贡献

1. 提出动态 inter-block activation 和 expansion-corrosion 策略，为 diffusion transformer 的深度剪枝提供更稳定的 block 选择信号。
2. 通过 VAE channel pruning、attention removal 与 lightweight SepConv 压缩编码/解码侧开销。
3. 移除 time/prompt 相关模块并预缓存固定条件，进一步减少单步推理冗余。
4. 相对 TSD-SR teacher 最高给出 5.68× speedup 与 83% 参数减少，同时保持主观质量。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00-tinysr-shallow-diffusion-transformers-for-real-world-image-super-resolution](sections/00-tinysr-shallow-diffusion-transformers-for-real-world-image-super-resolution.md) | 核心方法、模块和训练目标 |
| [01-abstract](sections/01-abstract.md) | 问题定义、方法一句话和主要 claim |
| [02-1-introduction](sections/02-1-introduction.md) | 动机、gap、贡献与方法直觉 |
| [03-2-related-work](sections/03-2-related-work.md) | 相关方法谱系和本文定位 |
| [04-3-methodology](sections/04-3-methodology.md) | 核心方法、模块和训练目标 |
| [05-3-1-dynamic-depth-pruning](sections/05-3-1-dynamic-depth-pruning.md) | 核心方法、模块和训练目标 |
| [06-3-2-component-streamlining](sections/06-3-2-component-streamlining.md) | 核心方法、模块和训练目标 |
| [07-3-3-training-scheme](sections/07-3-3-training-scheme.md) | 核心方法、模块和训练目标 |
| [08-4-experiments](sections/08-4-experiments.md) | 实验设置、SOTA 对比和消融 |
| [09-4-1-experimental-settings](sections/09-4-1-experimental-settings.md) | 实验设置、SOTA 对比和消融 |
| [10-4-2-comparison-with-existing-sr-methods](sections/10-4-2-comparison-with-existing-sr-methods.md) | 核心方法、模块和训练目标 |
| [11-4-3-comparison-with-depth-pruning-methods](sections/11-4-3-comparison-with-depth-pruning-methods.md) | 核心方法、模块和训练目标 |
| [12-4-4-ablation-study](sections/12-4-4-ablation-study.md) | 分析、讨论、限制或补充实验 |
| [13-5-conclusion](sections/13-5-conclusion.md) | 总结贡献与局限 |
| [14-supplementary-material](sections/14-supplementary-material.md) | 附录实现细节、推导和更多结果 |
| [15-a-implementation-details](sections/15-a-implementation-details.md) | 附录实现细节、推导和更多结果 |
| [16-a-1-data-processing](sections/16-a-1-data-processing.md) | 核心方法、模块和训练目标 |
| [17-a-2-vae-training](sections/17-a-2-vae-training.md) | 核心方法、模块和训练目标 |
| [18-a-3-pruning-decision-training](sections/18-a-3-pruning-decision-training.md) | 核心方法、模块和训练目标 |
| [19-a-4-restoration-training](sections/19-a-4-restoration-training.md) | 核心方法、模块和训练目标 |
| [20-b-more-comparisons-on-benchmarks](sections/20-b-more-comparisons-on-benchmarks.md) | 实验设置、SOTA 对比和消融 |
| [21-b-1-more-quantitative-comparisons](sections/21-b-1-more-quantitative-comparisons.md) | 实验设置、SOTA 对比和消融 |
| [22-b-2-more-qualitative-comparisons](sections/22-b-2-more-qualitative-comparisons.md) | 实验设置、SOTA 对比和消融 |
| [23-c-more-ablation-studies](sections/23-c-more-ablation-studies.md) | 分析、讨论、限制或补充实验 |
| [24-c-1-ablation-study-on-prompt-condition](sections/24-c-1-ablation-study-on-prompt-condition.md) | 分析、讨论、限制或补充实验 |
| [25-c-2-ablation-study-on-pruning-ratio](sections/25-c-2-ablation-study-on-pruning-ratio.md) | 分析、讨论、限制或补充实验 |
| [26-c-3-ablation-study-of-knowledge-distillation](sections/26-c-3-ablation-study-of-knowledge-distillation.md) | 分析、讨论、限制或补充实验 |
| [27-c-4-ablation-study-of-losses-in-stage-2](sections/27-c-4-ablation-study-of-losses-in-stage-2.md) | 分析、讨论、限制或补充实验 |
| [28-references](sections/28-references.md) | 参考文献 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |

## Key Figures / Tables

| ID | What It Shows | Takeaway |
|----|---------------|----------|
| Figure 1. | Figure 1. A comprehensive comparison of recent Real-ISR models in terms of visual quality, | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 2. | Figure 2. Performance and efficiency comparison among DMsbased Real-ISR methods on an NVID | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 3. | Figure 3. Depth pruning closely aligns with the theoretical linear acceleration curve comp | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 4. | Figure 4. Our proposed mask learning method performs a probability-based decision for cand | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 5. | Figure 5. Comparisons of performance and efficiency for various design of efficient TinySR | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 6. | Figure 6. Left: Channel pruning effectively reduces MACs across all modules, particularly  | 看方法框架、可控性或 SOTA 对比证据 |

## Reusable Ideas

- **Problem framing**: 单步扩散不等于实时，瓶颈从采样步数转移到大模型架构。
- **Method design**: 用 teacher-guided pruning 决定深度，再补 VAE 和条件模块的系统瘦身。
- **Experiment design**: 除 PSNR/LPIPS/FID 等质量指标外，必须报告 latency、MACs、参数量。
- **Writing pattern**: 把“one-step distillation 仍 over-parameterized”作为明确 gap。

## Open Questions

- 剪枝后的模型在未见退化类型上是否比 teacher 更脆弱？
- VAE 压缩是否会损伤高频纹理与小文字/人脸细节？

## 📊 Citation Landscape

- **Semantic Scholar**: API detail 暂时被限流或未返回；本地缓存见 `semantic-scholar/detail.json`。
- **Connected Papers**: https://www.connectedpapers.com/main/2508.17434
- **TLDR**: 暂缺。
- **引用统计**: 暂缺。

### 参考文献分组

#### Super-Resolution / Restoration
- High-Resolution Image Synthesis with Latent Diffusion Models (Computer Vision and Pattern Recognition, 2021) — citations: 24028 — https://arxiv.org/abs/2112.10752
- Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (Computer Vision and Pattern Recognition, 2016) — citations: 5891 — https://arxiv.org/abs/1609.05158
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Neural Information Processing Systems, 2022) — citations: 4149 — https://arxiv.org/abs/2205.14135
- NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results (2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017) — citations: 1857
- Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data (2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021) — citations: 1830 — https://arxiv.org/abs/2107.10833

#### Diffusion / Generative Prior
- Denoising Diffusion Probabilistic Models (Neural Information Processing Systems, 2020) — citations: 29911 — https://arxiv.org/abs/2006.11239
- Scalable Diffusion Models with Transformers (IEEE International Conference on Computer Vision, 2022) — citations: 5516 — https://arxiv.org/abs/2212.09748
- Improved Denoising Diffusion Probabilistic Models (International Conference on Machine Learning, 2021) — citations: 5221 — https://arxiv.org/abs/2102.09672
- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (International Conference on Machine Learning, 2024) — citations: 3596 — https://arxiv.org/abs/2403.03206
- Image Quality Assessment: Unifying Structure and Texture Similarity (IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020) — citations: 1273 — https://arxiv.org/abs/2004.07728

#### Flow / ODE Models
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (arXiv.org, 2017) — citations: 24655 — https://arxiv.org/abs/1704.04861
- Xception: Deep Learning with Depthwise Separable Convolutions (Computer Vision and Pattern Recognition, 2016) — citations: 17515 — https://arxiv.org/abs/1610.02357
- Categorical Reparameterization with Gumbel-Softmax (International Conference on Learning Representations, 2016) — citations: 6242 — https://arxiv.org/abs/1611.01144
- MUSIQ: Multi-scale Image Quality Transformer (IEEE International Conference on Computer Vision, 2021) — citations: 1308 — https://arxiv.org/abs/2108.05997
- TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment (IEEE Transactions on Image Processing, 2023) — citations: 299 — https://arxiv.org/abs/2308.03060

#### Other Foundations
- Image quality assessment: from error visibility to structural similarity (IEEE Transactions on Image Processing, 2004) — citations: 56242
- Decoupled Weight Decay Regularization (International Conference on Learning Representations, 2017) — citations: 33244
- A Style-Based Generator Architecture for Generative Adversarial Networks (Computer Vision and Pattern Recognition, 2018) — citations: 12761 — https://arxiv.org/abs/1812.04948
- Learning both Weights and Connections for Efficient Neural Network (Neural Information Processing Systems, 2015) — citations: 7591 — https://arxiv.org/abs/1506.02626
- A Feature-Enriched Completely Blind Image Quality Evaluator (IEEE Transactions on Image Processing, 2015) — citations: 1161

#### Perceptual Quality / GAN Metrics
- GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (Neural Information Processing Systems, 2017) — citations: 17871
- The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018) — citations: 17160 — https://arxiv.org/abs/1801.03924
- Exploring CLIP for Assessing the Look and Feel of Images (AAAI Conference on Artificial Intelligence, 2022) — citations: 1197 — https://arxiv.org/abs/2207.12396

#### Distillation / Few-Step Generation
- Flux.1 lite: Distilling flux1 (dev for efficient text-to-image generation, None) — citations: None

### 推荐论文

- Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.24136
- MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.20690
- Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.11470
- GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.25457
- VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.21450
- GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.16769
- QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model (IEEE International Conference on Acoustics, Speech, and Signal Processing, 2026) — citations: 0 — https://arxiv.org/abs/2603.09125
- Allo{SR}$^2$: Rectifying One-Step Super-Resolution to Stay Real via Allomorphic Generative Flows (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.19238
- DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.13162
- Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.14112
