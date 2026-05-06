# One-Step Effective Diffusion Network for Real-World Image Super-Resolution

**作者**: Rongyuan Wu; Lingchen Sun; Zhiyuan Ma; Lei Zhang  
**会议**: NeurIPS 2024  
**链接**: [arXiv 2406.08177](https://arxiv.org/abs/2406.08177) | [PDF](https://arxiv.org/pdf/2406.08177v3)

## 一句话总结

OSEDiff 将低质量图像直接作为扩散起点，用 latent-space VSD 把 Stable Diffusion 先验压缩到单步 Real-ISR 推理。

## 核心贡献

1. LQ latent 直接作为扩散起点，避免随机噪声带来的不确定性。
2. 用 latent-space VSD 正则单步输出，继承多步扩散教师的感知先验。
3. 仅训练少量参数，在单步推理下保持与多步 SD 方法相当的感知质量。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 |
| [02 - Related Work](sections/02-related-work.md) | 相关工作谱系 |
| [03 - Methodology](sections/03-methodology.md) | 核心方法 |
| [04 - Experiments](sections/04-experiments.md) | 实验设置与结果 |
| [05 - Conclusion and Limitation](sections/05-conclusion.md) | 总结与局限 |
| [06 - Appendix](sections/06-appendix.md) | 附录补充 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Inference time | 0.11s on A100 for 512×512 |
| Trainable parameters | 8.5M |
| 核心范式 | SD prior + one-step distillation + data/VSD loss |

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2406.08177)

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
- **Image Super-Resolution Via Sparse Representation** (2010) — 5229 citations

#### Diffusion / Flow Foundations
- **Very Deep Convolutional Networks for Large-Scale Image Recognition** (2014) — 110886 citations [arXiv](https://arxiv.org/abs/1409.1556)
- **Diffusion Models Beat GANs on Image Synthesis** (2021) — 11530 citations [arXiv](https://arxiv.org/abs/2105.05233)
- **Score-Based Generative Modeling through Stochastic Differential Equations** (2020) — 10215 citations [arXiv](https://arxiv.org/abs/2011.13456)
- **Visual Instruction Tuning** (2023) — 9134 citations [arXiv](https://arxiv.org/abs/2304.08485)
- **Adding Conditional Control to Text-to-Image Diffusion Models** (2023) — 6718 citations [arXiv](https://arxiv.org/abs/2302.05543)

#### Other Foundations
- **ImageNet classification with deep convolutional neural networks** (2012) — 128505 citations
- **Image quality assessment: from error visibility to structural similarity** (2004) — 56242 citations
- **GENERATIVE ADVERSARIAL NETS** (2018) — 41137 citations
- **Decoupled Weight Decay Regularization** (2017) — 33244 citations
- **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium** (2017) — 17871 citations

#### Medical VLM / Medical Imaging
- **Perceptual Losses for Real-Time Style Transfer and Super-Resolution** (2016) — 11355 citations [arXiv](https://arxiv.org/abs/1603.08155)
- **Second-Order Attention Network for Single Image Super-Resolution** (2019) — 1766 citations
- **Designing a Practical Degradation Model for Deep Blind Image Super-Resolution** (2021) — 894 citations [arXiv](https://arxiv.org/abs/2103.14006)
- **Exploiting Diffusion Prior for Real-World Image Super-Resolution** (2023) — 595 citations [arXiv](https://arxiv.org/abs/2305.07015)
- **Component Divide-and-Conquer for Real-World Image Super-Resolution** (2020) — 356 citations [arXiv](https://arxiv.org/abs/2008.01928)

#### Latent Reasoning / Thinking
- **Denoising Diffusion Probabilistic Models** (2020) — 29911 citations [arXiv](https://arxiv.org/abs/2006.11239)
- **A Style-Based Generator Architecture for Generative Adversarial Networks** (2018) — 12761 citations [arXiv](https://arxiv.org/abs/1812.04948)
- **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding** (2022) — 8153 citations [arXiv](https://arxiv.org/abs/2205.11487)
- **GAN Prior Embedded Network for Blind Face Restoration in the Wild** (2021) — 358 citations [arXiv](https://arxiv.org/abs/2105.06070)
- **One-Step Image Translation with Text-to-Image Models** (2024) — 132 citations [arXiv](https://arxiv.org/abs/2403.12036)

#### Memory / Agent Memory
- **LoRA: Low-Rank Adaptation of Large Language Models** (2021) — 18662 citations [arXiv](https://arxiv.org/abs/2106.09685)
- **MUSIQ: Multi-scale Image Quality Transformer** (2021) — 1308 citations [arXiv](https://arxiv.org/abs/2108.05997)

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model](https://arxiv.org/abs/2603.09125) | 2026 | 0 |
| [GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution](https://arxiv.org/abs/2603.16769) | 2026 | 0 |
| [Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution](https://arxiv.org/abs/2604.24136) | 2026 | 0 |
| [GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution](https://arxiv.org/abs/2604.25457) | 2026 | 0 |
| [VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.21450) | 2026 | 0 |
| [MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution](https://arxiv.org/abs/2603.20690) | 2026 | 0 |
| [Allo{SR}$^2$: Rectifying One-Step Super-Resolution to Stay Real via Allomorphic Generative Flows](https://arxiv.org/abs/2604.19238) | 2026 | 0 |
| [Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.11470) | 2026 | 0 |
| [Disentangled Textual Priors for Diffusion-based Image Super-Resolution](https://arxiv.org/abs/2603.07430) | 2026 | 0 |
| [Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution](https://arxiv.org/abs/2603.14112) | 2026 | 0 |

## BibTeX

```bibtex
@article{wu2024osediff,
  title={ One-Step Effective Diffusion Network for Real-World Image Super-Resolution },
  author={ Rongyuan Wu and Lingchen Sun and Zhiyuan Ma and Lei Zhang },
  journal={arXiv preprint arXiv:2406.08177},
  year={ 2024 }
}
```
