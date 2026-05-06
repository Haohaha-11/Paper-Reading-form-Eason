# Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution

**作者**: Hao Chen; Junyang Chen; Jinshan Pan; Jiangxin Dong  
**会议**: Arxiv 2025  
**链接**: [arXiv 2512.14061](https://arxiv.org/abs/2512.14061) | [PDF](https://arxiv.org/pdf/2512.14061v1)

## 一句话总结

CODSR 通过 LQ-guided feature modulation、区域自适应生成先验激活和 text-matching guidance，在单步 SR 中兼顾局部保真与感知质量。

## 核心贡献

1. 用未压缩 LQ 信息补偿 latent 编码信息损失。
2. 区域自适应激活生成先验，避免全图统一幻觉细节。
3. 用 text-matching guidance 缓解 prompt 与语义区域错配。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 |
| [02 - Related Work](sections/02-related-work.md) | 相关工作谱系 |
| [03 - Proposed Method](sections/03-method.md) | 核心方法 |
| [04 - Experimental Results](sections/04-experiments.md) | 实验设置与结果 |
| [05 - Analysis and Discussion](sections/05-analysis-and-discussion.md) | 分析与消融 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与局限 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| 核心模块 | LQ modulation + region prior + text matching |
| Project | github.com/Chanson94/CODSR |
| 目标 | controllable one-step SR |

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2512.14061)

### TLDR (Semantic Scholar 自动生成)

> 暂缺。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 暂缺 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top 5 per category, by citations)

#### Super-Resolution / Restoration
- **High-Resolution Image Synthesis with Latent Diffusion Models** (2021) — 24029 citations [arXiv](https://arxiv.org/abs/2112.10752)
- **Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network** (2016) — 11988 citations [arXiv](https://arxiv.org/abs/1609.04802)
- **Enhanced Deep Residual Networks for Single Image Super-Resolution** (2017) — 7085 citations [arXiv](https://arxiv.org/abs/1707.02921)
- **Learning a Deep Convolutional Network for Image Super-Resolution** (2014) — 5633 citations
- **Image Super-Resolution Using Very Deep Residual Channel Attention Networks** (2018) — 5224 citations [arXiv](https://arxiv.org/abs/1807.02758)

#### Diffusion / Flow Foundations
- **Diffusion Models Beat GANs on Image Synthesis** (2021) — 11530 citations [arXiv](https://arxiv.org/abs/2105.05233)
- **Adding Conditional Control to Text-to-Image Diffusion Models** (2023) — 6718 citations [arXiv](https://arxiv.org/abs/2302.05543)
- **ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation** (2023) — 1304 citations [arXiv](https://arxiv.org/abs/2305.16213)
- **Image Quality Assessment: Unifying Structure and Texture Similarity** (2020) — 1273 citations [arXiv](https://arxiv.org/abs/2004.07728)
- **Exploring CLIP for Assessing the Look and Feel of Images** (2022) — 1197 citations [arXiv](https://arxiv.org/abs/2207.12396)

#### Medical VLM / Medical Imaging
- **Restormer: Efficient Transformer for High-Resolution Image Restoration** (2021) — 3757 citations [arXiv](https://arxiv.org/abs/2111.09881)
- **Second-Order Attention Network for Single Image Super-Resolution** (2019) — 1766 citations
- **Designing a Practical Degradation Model for Deep Blind Image Super-Resolution** (2021) — 894 citations [arXiv](https://arxiv.org/abs/2103.14006)
- **Exploiting Diffusion Prior for Real-World Image Super-Resolution** (2023) — 595 citations [arXiv](https://arxiv.org/abs/2305.07015)
- **Component Divide-and-Conquer for Real-World Image Super-Resolution** (2020) — 356 citations [arXiv](https://arxiv.org/abs/2008.01928)

#### Other Foundations
- **Image quality assessment: from error visibility to structural similarity** (2004) — 56242 citations
- **GENERATIVE ADVERSARIAL NETS** (2018) — 41137 citations
- **Decoupled Weight Decay Regularization** (2017) — 33245 citations
- **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric** (2018) — 17160 citations [arXiv](https://arxiv.org/abs/1801.03924)
- **NLTK: The Natural Language Toolkit** (2006) — 5149 citations

#### Latent Reasoning / Thinking
- **Denoising Diffusion Probabilistic Models** (2020) — 29911 citations [arXiv](https://arxiv.org/abs/2006.11239)
- **A Style-Based Generator Architecture for Generative Adversarial Networks** (2018) — 12761 citations [arXiv](https://arxiv.org/abs/1812.04948)
- **T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models** (2023) — 1630 citations [arXiv](https://arxiv.org/abs/2302.08453)

#### Memory / Agent Memory
- **MUSIQ: Multi-scale Image Quality Transformer** (2021) — 1308 citations [arXiv](https://arxiv.org/abs/2108.05997)

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution](https://arxiv.org/abs/2603.14112) | 2026 | 0 |
| [GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution](https://arxiv.org/abs/2604.25457) | 2026 | 0 |
| [QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model](https://arxiv.org/abs/2603.09125) | 2026 | 0 |
| [Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution](https://arxiv.org/abs/2604.24136) | 2026 | 0 |
| [Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.11470) | 2026 | 0 |
| [Disentangled Textual Priors for Diffusion-based Image Super-Resolution](https://arxiv.org/abs/2603.07430) | 2026 | 0 |
| [Allo{SR}$^2$: Rectifying One-Step Super-Resolution to Stay Real via Allomorphic Generative Flows](https://arxiv.org/abs/2604.19238) | 2026 | 0 |
| [DVFace: Spatio-Temporal Dual-Prior Diffusion for Video Face Restoration](https://arxiv.org/abs/2604.14560) | 2026 | 0 |
| [GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution](https://arxiv.org/abs/2603.16769) | 2026 | 0 |
| [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](https://arxiv.org/abs/2603.13162) | 2026 | 0 |

## BibTeX

```bibtex
@article{chen2025codsr,
  title={ Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution },
  author={ Hao Chen and Junyang Chen and Jinshan Pan and Jiangxin Dong },
  journal={arXiv preprint arXiv:2512.14061},
  year={ 2025 }
}
```
