# One-Step Effective Diffusion Network for Real-World Image Super-Resolution

**作者**: Rongyuan Wu; Lingchen Sun; Zhiyuan Ma; Lei Zhang  
**会议/年份**: NeurIPS 2024  
**链接**: [arXiv](https://arxiv.org/abs/2406.08177) | [PDF](https://arxiv.org/pdf/2406.08177v3)  
**主题**: One-Step Diffusion Super-Resolution  
**阅读日期**: 2026-05-05

## 一句话总结

OSEDiff 将低质量图像直接作为扩散起点，用少量 LoRA/可训练层和 latent-space VSD 正则把 Stable Diffusion 的生成先验压缩到单步 Real-ISR 推理中。

## Why Read This Paper

- 这是这一组 one-step diffusion SR 的关键起点，后续 RCOD/CODSR/TinySR 都在补它的可控性、保真度或效率短板。
- 它清楚展示了单步扩散如何从随机生成任务转成确定性图像复原任务。
- 实验覆盖 RealSR/DRealSR/DIV2K-Val，可作为比较后续方法的基线。

## 核心贡献

1. 把 Real-ISR 的随机噪声起点改为 LQ latent 起点，降低 restoration 任务中不必要的随机性。
2. 用 latent-space variational score distillation 约束单步输出，使单步模型仍能获得多步扩散教师的高质量先验。
3. 只训练约 8.5M 参数，在 A100 上实现约 0.11s 单步推理，同时保持与多步 SD 方法相当或更好的感知指标。
4. 给后续 one-step diffusion SR 方法提供了“SD 先验 + 单步蒸馏 + 数据项/正则项”范式。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00-title-and-authors](sections/00-title-and-authors.md) | 核心方法、模块和训练目标 |
| [01-abstract](sections/01-abstract.md) | 问题定义、方法一句话和主要 claim |
| [02-1-introduction](sections/02-1-introduction.md) | 动机、gap、贡献与方法直觉 |
| [03-2-related-work](sections/03-2-related-work.md) | 相关方法谱系和本文定位 |
| [04-3-methodology](sections/04-3-methodology.md) | 核心方法、模块和训练目标 |
| [05-3-1-problem-modeling](sections/05-3-1-problem-modeling.md) | 核心方法、模块和训练目标 |
| [06-3-2-one-step-effective-diffusion-network](sections/06-3-2-one-step-effective-diffusion-network.md) | 核心方法、模块和训练目标 |
| [07-4-experiments](sections/07-4-experiments.md) | 实验设置、SOTA 对比和消融 |
| [08-4-1-experimental-settings](sections/08-4-1-experimental-settings.md) | 实验设置、SOTA 对比和消融 |
| [09-4-2-comparison-with-state-of-the-arts](sections/09-4-2-comparison-with-state-of-the-arts.md) | 实验设置、SOTA 对比和消融 |
| [10-4-3-ablation-study](sections/10-4-3-ablation-study.md) | 分析、讨论、限制或补充实验 |
| [11-5-conclusion-and-limitation](sections/11-5-conclusion-and-limitation.md) | 总结贡献与局限 |
| [12-a-appendix](sections/12-a-appendix.md) | 附录实现细节、推导和更多结果 |
| [13-a-1-comparison-with-gan-based-methods](sections/13-a-1-comparison-with-gan-based-methods.md) | 核心方法、模块和训练目标 |
| [14-a-2-user-study](sections/14-a-2-user-study.md) | 核心方法、模块和训练目标 |
| [15-a-3-more-visual-comparisons](sections/15-a-3-more-visual-comparisons.md) | 实验设置、SOTA 对比和消融 |
| [16-a-4-algorithm-of-osediff](sections/16-a-4-algorithm-of-osediff.md) | 核心方法、模块和训练目标 |
| [17-algorithm-1-training-scheme-of-osediff](sections/17-algorithm-1-training-scheme-of-osediff.md) | 核心方法、模块和训练目标 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Inference time | 0.11s on A100 for 512×512 input |
| Trainable parameters | 8.5M |
| Speedup claim | over 100× faster than StableSR in paper comparison |

## Key Figures / Tables

| ID | What It Shows | Takeaway |
|----|---------------|----------|
| Figure 1 | Figure 1: Performance and efficiency comparison among SD-based Real-ISR methods. (a). Perf | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 2 | Figure 2: The training framework of OSEDiff. The LQ image is passed through a trainable en | 看方法框架、可控性或 SOTA 对比证据 |
| Table 1 | Table 1: Quantitative comparison with state-of-the-art methods on both synthetic and real- | 看方法框架、可控性或 SOTA 对比证据 |
| Table 2 | Table 2: Complexity comparison among different methods. All methods are tested with an inp | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 3 | Figure 3: Qualitative comparisons of different Real-ISR methods. Please zoom in for a bett | 看方法框架、可控性或 SOTA 对比证据 |
| Table 3 | Table 3: Comparison of different losses on the RealSR benchmark. | 看方法框架、可控性或 SOTA 对比证据 |

## Reusable Ideas

- **Problem framing**: 把 one-step SR 描述为在 fidelity 与 perceptual realism 之间用生成先验补细节的 restoration 问题。
- **Method design**: LQ latent 直接作为起点，主生成器只走一次 denoising，再用 VSD/数据项共同训练。
- **Experiment design**: 同时报告 full-reference 与 no-reference 指标，避免只看 PSNR 或只看感知分数。
- **Writing pattern**: 先指出多步扩散慢与随机性，再自然引出“one-step effective diffusion”。

## Open Questions

- 单步从 LQ latent 出发时，何时会过度依赖 SD 先验而生成不忠实纹理？
- VSD 正则与数据项权重是否能自适应不同退化强度？

## 📊 Citation Landscape

- **Semantic Scholar**: API detail 暂时被限流或未返回；本地缓存见 `semantic-scholar/detail.json`。
- **Connected Papers**: https://www.connectedpapers.com/main/2406.08177
- **TLDR**: 暂缺。
- **引用统计**: 暂缺。

### 参考文献分组

#### Super-Resolution / Restoration
- High-Resolution Image Synthesis with Latent Diffusion Models (Computer Vision and Pattern Recognition, 2021) — citations: 24029 — https://arxiv.org/abs/2112.10752
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (Computer Vision and Pattern Recognition, 2016) — citations: 11988 — https://arxiv.org/abs/1609.04802
- Perceptual Losses for Real-Time Style Transfer and Super-Resolution (European Conference on Computer Vision, 2016) — citations: 11355 — https://arxiv.org/abs/1603.08155
- Enhanced Deep Residual Networks for Single Image Super-Resolution (2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017) — citations: 7085 — https://arxiv.org/abs/1707.02921
- Learning a Deep Convolutional Network for Image Super-Resolution (European Conference on Computer Vision, 2014) — citations: 5633

#### Diffusion / Generative Prior
- Denoising Diffusion Probabilistic Models (Neural Information Processing Systems, 2020) — citations: 29911 — https://arxiv.org/abs/2006.11239
- Diffusion Models Beat GANs on Image Synthesis (Neural Information Processing Systems, 2021) — citations: 11530 — https://arxiv.org/abs/2105.05233
- Score-Based Generative Modeling through Stochastic Differential Equations (International Conference on Learning Representations, 2020) — citations: 10215 — https://arxiv.org/abs/2011.13456
- Visual Instruction Tuning (Neural Information Processing Systems, 2023) — citations: 9134 — https://arxiv.org/abs/2304.08485
- Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Neural Information Processing Systems, 2022) — citations: 8153 — https://arxiv.org/abs/2205.11487

#### Other Foundations
- ImageNet classification with deep convolutional neural networks (Communications of the ACM, 2012) — citations: 128505
- Image quality assessment: from error visibility to structural similarity (IEEE Transactions on Image Processing, 2004) — citations: 56242
- GENERATIVE ADVERSARIAL NETS (N/A, 2018) — citations: 41137
- Decoupled Weight Decay Regularization (International Conference on Learning Representations, 2017) — citations: 33244
- A Style-Based Generator Architecture for Generative Adversarial Networks (Computer Vision and Pattern Recognition, 2018) — citations: 12761 — https://arxiv.org/abs/1812.04948

#### Perceptual Quality / GAN Metrics
- GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (Neural Information Processing Systems, 2017) — citations: 17871
- The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018) — citations: 17160 — https://arxiv.org/abs/1801.03924
- Exploring CLIP for Assessing the Look and Feel of Images (AAAI Conference on Artificial Intelligence, 2022) — citations: 1197 — https://arxiv.org/abs/2207.12396
- Towards Real-World Blind Face Restoration with Generative Facial Prior (Computer Vision and Pattern Recognition, 2021) — citations: 612 — https://arxiv.org/abs/2101.04061
- GAN Prior Embedded Network for Blind Face Restoration in the Wild (Computer Vision and Pattern Recognition, 2021) — citations: 358 — https://arxiv.org/abs/2105.06070

#### Flow / ODE Models
- Very Deep Convolutional Networks for Large-Scale Image Recognition (International Conference on Learning Representations, 2014) — citations: 110886 — https://arxiv.org/abs/1409.1556
- LoRA: Low-Rank Adaptation of Large Language Models (International Conference on Learning Representations, 2021) — citations: 18662 — https://arxiv.org/abs/2106.09685
- MUSIQ: Multi-scale Image Quality Transformer (IEEE International Conference on Computer Vision, 2021) — citations: 1308 — https://arxiv.org/abs/2108.05997
- Stability (N/A, 1973) — citations: 1153 — https://arxiv.org/abs/1310.0150

### 推荐论文

- QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model (IEEE International Conference on Acoustics, Speech, and Signal Processing, 2026) — citations: 0 — https://arxiv.org/abs/2603.09125
- GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.16769
- Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.24136
- GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.25457
- VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.21450
- MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.20690
- Allo{SR}$^2$: Rectifying One-Step Super-Resolution to Stay Real via Allomorphic Generative Flows (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.19238
- Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.11470
- Disentangled Textual Priors for Diffusion-based Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.07430
- Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.14112
