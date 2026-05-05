# Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution

**作者**: Hao Chen; Junyang Chen; Jinshan Pan; Jiangxin Dong  
**会议/年份**: Arxiv 2025  
**链接**: [arXiv](https://arxiv.org/abs/2512.14061) | [PDF](https://arxiv.org/pdf/2512.14061v1)  
**主题**: One-Step Diffusion Super-Resolution  
**阅读日期**: 2026-05-05

## 一句话总结

CODSR 从 LQ 信息损失、区域差异化先验激活和文本-区域错配三方面补强 controllable one-step diffusion SR，在单步推理下兼顾感知质量与局部保真。

## Why Read This Paper

- 它代表另一类可控 one-step SR：不是只调全局 realism，而是做区域级先验激活。
- 它把 LQ 原始信息重新注入 diffusion 过程，针对 latent diffusion SR 的保真损失。
- 适合和 RCOD-SR 的全局/退化感知控制做并列比较。

## 核心贡献

1. 提出 LQ-guided feature modulation，用未压缩 LQ 信息补偿 VAE/latent 编码带来的保真损失。
2. 提出 region-adaptive generative prior activation，在需要幻觉细节的区域增强先验，在结构敏感区域抑制过度生成。
3. 提出 text-matching guidance，让文本提示更准确匹配对应语义区域。
4. 在多个 real-world SR benchmark 上展示比现有 diffusion SR 更好的感知质量和有竞争力的 fidelity。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00-title-and-authors](sections/00-title-and-authors.md) | 核心方法、模块和训练目标 |
| [01-abstract](sections/01-abstract.md) | 问题定义、方法一句话和主要 claim |
| [02-1-introduction](sections/02-1-introduction.md) | 动机、gap、贡献与方法直觉 |
| [03-2-related-work](sections/03-2-related-work.md) | 相关方法谱系和本文定位 |
| [04-3-proposed-method](sections/04-3-proposed-method.md) | 核心方法、模块和训练目标 |
| [05-3-1-region-adaptive-generative-prior-activation](sections/05-3-1-region-adaptive-generative-prior-activation.md) | 核心方法、模块和训练目标 |
| [06-3-2-lq-guided-feature-modulation](sections/06-3-2-lq-guided-feature-modulation.md) | 核心方法、模块和训练目标 |
| [07-3-3-text-matching-guidance](sections/07-3-3-text-matching-guidance.md) | 核心方法、模块和训练目标 |
| [08-3-4-loss-function](sections/08-3-4-loss-function.md) | 核心方法、模块和训练目标 |
| [09-4-experimental-results](sections/09-4-experimental-results.md) | 实验设置、SOTA 对比和消融 |
| [10-4-1-experimental-settings](sections/10-4-1-experimental-settings.md) | 实验设置、SOTA 对比和消融 |
| [11-4-2-comparison-with-the-state-of-the-art](sections/11-4-2-comparison-with-the-state-of-the-art.md) | 实验设置、SOTA 对比和消融 |
| [12-5-analysis-and-discussion](sections/12-5-analysis-and-discussion.md) | 分析、讨论、限制或补充实验 |
| [13-6-conclusion](sections/13-6-conclusion.md) | 总结贡献与局限 |
| [14-references](sections/14-references.md) | 参考文献 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Core controls | region-adaptive prior activation + text-matching guidance |
| Project | https://github.com/Chanson94/CODSR |
| Benchmarks | RealSR/DrealSR and other real-world SR test sets in paper tables |

## Key Figures / Tables

| ID | What It Shows | Takeaway |
|----|---------------|----------|
| Figure 1. | Figure 1. Performance and runtime comparison among SD-based SISR methods on the DrealSR [4 | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 2. | Figure 2. An overview of our CODSR. (a) The region-adaptive generative prior activation me | 看方法框架、可控性或 SOTA 对比证据 |
| Table 1. | Table 1. Quantitative comparison with state-of-the-art diffusion-based SR methods on four  | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 3. | Figure 3. Qualitative comparisons of different methods on the RealSR [4] dataset, RealLQ25 | 看方法框架、可控性或 SOTA 对比证据 |
| Figure 4. | Figure 4. Comparisons between $B a s e _ { w / \ z _ { L } + \epsilon }$ and $B a s e _ {  | 看方法框架、可控性或 SOTA 对比证据 |
| Table 2. | Table 2. Effectiveness of each module in our proposed network. All methods are trained usi | 看方法框架、可控性或 SOTA 对比证据 |

## Reusable Ideas

- **Problem framing**: one-step SR 的保真问题部分来自 LQ 压缩编码信息损失。
- **Method design**: 用 LQ feature modulation 保结构，用区域自适应先验补纹理。
- **Experiment design**: 应同时给出区域/局部视觉案例，证明不是全图统一增强纹理。
- **Writing pattern**: 把三个 limitation 对应到三个模块，结构清晰。

## Open Questions

- 区域自适应策略是否需要额外语义/退化估计器，推理成本如何？
- 文本匹配 guidance 在无明确语义或错误 caption 场景下是否稳定？

## 📊 Citation Landscape

- **Semantic Scholar**: API detail 暂时被限流或未返回；本地缓存见 `semantic-scholar/detail.json`。
- **Connected Papers**: https://www.connectedpapers.com/main/2512.14061
- **TLDR**: 暂缺。
- **引用统计**: 暂缺。

### 参考文献分组

#### Super-Resolution / Restoration
- High-Resolution Image Synthesis with Latent Diffusion Models (Computer Vision and Pattern Recognition, 2021) — citations: 24029 — https://arxiv.org/abs/2112.10752
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (Computer Vision and Pattern Recognition, 2016) — citations: 11988 — https://arxiv.org/abs/1609.04802
- Enhanced Deep Residual Networks for Single Image Super-Resolution (2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017) — citations: 7085 — https://arxiv.org/abs/1707.02921
- Learning a Deep Convolutional Network for Image Super-Resolution (European Conference on Computer Vision, 2014) — citations: 5633
- Image Super-Resolution Using Very Deep Residual Channel Attention Networks (European Conference on Computer Vision, 2018) — citations: 5224 — https://arxiv.org/abs/1807.02758

#### Diffusion / Generative Prior
- Denoising Diffusion Probabilistic Models (Neural Information Processing Systems, 2020) — citations: 29911 — https://arxiv.org/abs/2006.11239
- Diffusion Models Beat GANs on Image Synthesis (Neural Information Processing Systems, 2021) — citations: 11530 — https://arxiv.org/abs/2105.05233
- Adding Conditional Control to Text-to-Image Diffusion Models (IEEE International Conference on Computer Vision, 2023) — citations: 6718 — https://arxiv.org/abs/2302.05543
- T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models (AAAI Conference on Artificial Intelligence, 2023) — citations: 1630 — https://arxiv.org/abs/2302.08453
- ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation (Neural Information Processing Systems, 2023) — citations: 1304 — https://arxiv.org/abs/2305.16213

#### Other Foundations
- Image quality assessment: from error visibility to structural similarity (IEEE Transactions on Image Processing, 2004) — citations: 56242
- GENERATIVE ADVERSARIAL NETS (N/A, 2018) — citations: 41137
- Decoupled Weight Decay Regularization (International Conference on Learning Representations, 2017) — citations: 33245
- A Style-Based Generator Architecture for Generative Adversarial Networks (Computer Vision and Pattern Recognition, 2018) — citations: 12761 — https://arxiv.org/abs/1812.04948
- NLTK: The Natural Language Toolkit (Annual Meeting of the Association for Computational Linguistics, 2006) — citations: 5149

#### Flow / ODE Models
- MUSIQ: Multi-scale Image Quality Transformer (IEEE International Conference on Computer Vision, 2021) — citations: 1308 — https://arxiv.org/abs/2108.05997
- Recognize Anything: A Strong Image Tagging Model (2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2023) — citations: 401 — https://arxiv.org/abs/2306.03514
- Low-rank adaptation of large language models (ICLR, None) — citations: None

#### Perceptual Quality / GAN Metrics
- The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018) — citations: 17160 — https://arxiv.org/abs/1801.03924
- Exploring CLIP for Assessing the Look and Feel of Images (AAAI Conference on Artificial Intelligence, 2022) — citations: 1197 — https://arxiv.org/abs/2207.12396

### 推荐论文

- Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.14112
- GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.25457
- QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model (IEEE International Conference on Acoustics, Speech, and Signal Processing, 2026) — citations: 0 — https://arxiv.org/abs/2603.09125
- Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.24136
- Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.11470
- Disentangled Textual Priors for Diffusion-based Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.07430
- Allo{SR}$^2$: Rectifying One-Step Super-Resolution to Stay Real via Allomorphic Generative Flows (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.19238
- DVFace: Spatio-Temporal Dual-Prior Diffusion for Video Face Restoration (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.14560
- GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.16769
- DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.13162
