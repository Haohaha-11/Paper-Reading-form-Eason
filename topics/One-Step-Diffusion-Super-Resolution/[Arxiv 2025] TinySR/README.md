# TinySR: Pruning Diffusion for Real-World Image Super-Resolution

**作者**: Linwei Dong; Qingnan Fan; Yuhang Yu; Qi Zhang; Jinwei Chen; Yawei Luo; Changqing Zou  
**会议**: Arxiv 2025  
**链接**: [arXiv 2508.17434](https://arxiv.org/abs/2508.17434) | [PDF](https://arxiv.org/pdf/2508.17434v2)

## 一句话总结

TinySR 面向实时 Real-ISR，把 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存压成轻量单步模型。

## 核心贡献

1. 用 Dynamic Inter-block Activation 和 Expansion-Corrosion 策略做深度剪枝。
2. 压缩 VAE 并移除 time/prompt 相关模块，减少单步模型的系统开销。
3. 相对 teacher 最高 5.68× 加速、83% 参数减少。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 |
| [02 - Related Work](sections/02-related-work.md) | 相关工作谱系 |
| [03 - Methodology](sections/03-methodology.md) | 核心方法 |
| [04 - Experiments](sections/04-experiments.md) | 实验设置与结果 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结与局限 |
| [06 - Appendix](sections/06-appendix.md) | 附录补充 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× |
| Parameter reduction | 83% |
| 目标 | real-time one-step Real-ISR |
| 压缩对象 | backbone + VAE + prompt/time modules |

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2508.17434)

### TLDR (Semantic Scholar 自动生成)

> 暂缺。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 暂缺 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top 5 per category, by citations)

#### Diffusion / Flow Foundations
- **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** (2017) — 24655 citations [arXiv](https://arxiv.org/abs/1704.04861)
- **Improved Denoising Diffusion Probabilistic Models** (2021) — 5221 citations [arXiv](https://arxiv.org/abs/2102.09672)
- **Image Quality Assessment: Unifying Structure and Texture Similarity** (2020) — 1273 citations [arXiv](https://arxiv.org/abs/2004.07728)
- **Exploring CLIP for Assessing the Look and Feel of Images** (2022) — 1197 citations [arXiv](https://arxiv.org/abs/2207.12396)
- **MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment** (2022) — 664 citations [arXiv](https://arxiv.org/abs/2204.08958)

#### Super-Resolution / Restoration
- **High-Resolution Image Synthesis with Latent Diffusion Models** (2021) — 24028 citations [arXiv](https://arxiv.org/abs/2112.10752)
- **NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results** (2017) — 1857 citations
- **Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data** (2021) — 1830 citations [arXiv](https://arxiv.org/abs/2107.10833)
- **Toward Real-World Single Image Super-Resolution: A New Benchmark and a New Model** (2019) — 710 citations [arXiv](https://arxiv.org/abs/1904.00523)
- **ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting** (2023) — 498 citations [arXiv](https://arxiv.org/abs/2307.12348)

#### Medical VLM / Medical Imaging
- **Xception: Deep Learning with Depthwise Separable Convolutions** (2016) — 17515 citations [arXiv](https://arxiv.org/abs/1610.02357)
- **Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network** (2016) — 5891 citations [arXiv](https://arxiv.org/abs/1609.05158)
- **Designing a Practical Degradation Model for Deep Blind Image Super-Resolution** (2021) — 894 citations [arXiv](https://arxiv.org/abs/2103.14006)
- **Exploiting Diffusion Prior for Real-World Image Super-Resolution** (2023) — 595 citations [arXiv](https://arxiv.org/abs/2305.07015)
- **Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels** (2023) — 497 citations [arXiv](https://arxiv.org/abs/2312.17090)

#### Other Foundations
- **Image quality assessment: from error visibility to structural similarity** (2004) — 56242 citations
- **Decoupled Weight Decay Regularization** (2017) — 33244 citations
- **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium** (2017) — 17871 citations
- **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric** (2018) — 17160 citations [arXiv](https://arxiv.org/abs/1801.03924)
- **A Feature-Enriched Completely Blind Image Quality Evaluator** (2015) — 1161 citations

#### Latent Reasoning / Thinking
- **Denoising Diffusion Probabilistic Models** (2020) — 29911 citations [arXiv](https://arxiv.org/abs/2006.11239)
- **A Style-Based Generator Architecture for Generative Adversarial Networks** (2018) — 12761 citations [arXiv](https://arxiv.org/abs/1812.04948)
- **Categorical Reparameterization with Gumbel-Softmax** (2016) — 6242 citations [arXiv](https://arxiv.org/abs/1611.01144)
- **Scalable Diffusion Models with Transformers** (2022) — 5516 citations [arXiv](https://arxiv.org/abs/2212.09748)
- **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (2024) — 3596 citations [arXiv](https://arxiv.org/abs/2403.03206)

#### Memory / Agent Memory
- **Learning both Weights and Connections for Efficient Neural Network** (2015) — 7591 citations [arXiv](https://arxiv.org/abs/1506.02626)
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (2022) — 4149 citations [arXiv](https://arxiv.org/abs/2205.14135)
- **MUSIQ: Multi-scale Image Quality Transformer** (2021) — 1308 citations [arXiv](https://arxiv.org/abs/2108.05997)
- **Q-Diffusion: Quantizing Diffusion Models** (2023) — 277 citations [arXiv](https://arxiv.org/abs/2302.04304)
- **LD-Pruner: Efficient Pruning of Latent Diffusion Models using Task-Agnostic Insights** (2024) — 66 citations [arXiv](https://arxiv.org/abs/2404.11936)

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution](https://arxiv.org/abs/2604.24136) | 2026 | 0 |
| [MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution](https://arxiv.org/abs/2603.20690) | 2026 | 0 |
| [Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.11470) | 2026 | 0 |
| [GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution](https://arxiv.org/abs/2604.25457) | 2026 | 0 |
| [VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.21450) | 2026 | 0 |
| [GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution](https://arxiv.org/abs/2603.16769) | 2026 | 0 |
| [QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model](https://arxiv.org/abs/2603.09125) | 2026 | 0 |
| [Allo{SR}$^2$: Rectifying One-Step Super-Resolution to Stay Real via Allomorphic Generative Flows](https://arxiv.org/abs/2604.19238) | 2026 | 0 |
| [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](https://arxiv.org/abs/2603.13162) | 2026 | 0 |
| [Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution](https://arxiv.org/abs/2603.14112) | 2026 | 0 |

## BibTeX

```bibtex
@article{dong2025tinysr,
  title={ TinySR: Pruning Diffusion for Real-World Image Super-Resolution },
  author={ Linwei Dong and Qingnan Fan and Yuhang Yu and Qi Zhang and Jinwei Chen and Yawei Luo and Changqing Zou },
  journal={arXiv preprint arXiv:2508.17434},
  year={ 2025 }
}
```
