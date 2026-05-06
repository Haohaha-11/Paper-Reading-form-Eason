# OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs

**作者**: Yuanzhi Zhu; Ruiqing Wang; Shilin Lu; Junnan Li; Hanshu Yan; Kai Zhang  
**会议**: ICLR 2026  
**链接**: [arXiv 2412.09465](https://arxiv.org/abs/2412.09465) | [PDF](https://arxiv.org/pdf/2412.09465v3)

## 一句话总结

OFTSR 用 conditional flow teacher 和 ODE trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

## 核心贡献

1. 从 diffusion 转向 flow/ODE 视角训练 conditional SR teacher。
2. 用 ODE trajectory alignment 约束 student 单步预测。
3. 在 FFHQ/DIV2K/ImageNet 上展示单步 SR 与可调 trade-off。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 |
| [02 - Background](sections/02-background.md) | 背景/理论基础 |
| [03 - Method](sections/03-method.md) | 核心方法 |
| [04 - Experiments](sections/04-experiments.md) | 实验设置与结果 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结与局限 |
| [06 - Appendix](sections/06-appendix.md) | 附录补充 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Teacher | conditional flow-based SR model |
| Distillation | same ODE trajectory alignment |
| Datasets | FFHQ, DIV2K, ImageNet 256×256 |

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2412.09465)

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
- **Generative Adversarial Networks** (2021) — 30528 citations [arXiv](https://arxiv.org/abs/2203.00667)
- **Diffusion Models Beat GANs on Image Synthesis** (2021) — 11531 citations [arXiv](https://arxiv.org/abs/2105.05233)
- **Score-Based Generative Modeling through Stochastic Differential Equations** (2020) — 10215 citations [arXiv](https://arxiv.org/abs/2011.13456)
- **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** (2015) — 9727 citations [arXiv](https://arxiv.org/abs/1503.03585)
- **Generative Modeling by Estimating Gradients of the Data Distribution** (2019) — 5359 citations [arXiv](https://arxiv.org/abs/1907.05600)

#### Super-Resolution / Restoration
- **High-Resolution Image Synthesis with Latent Diffusion Models** (2021) — 24029 citations [arXiv](https://arxiv.org/abs/2112.10752)
- **ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks** (2018) — 4578 citations [arXiv](https://arxiv.org/abs/1809.00219)
- **SwinIR: Image Restoration Using Swin Transformer** (2021) — 4510 citations [arXiv](https://arxiv.org/abs/2108.10257)
- **NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study** (2017) — 3872 citations
- **Image Super-Resolution via Iterative Refinement** (2021) — 2426 citations [arXiv](https://arxiv.org/abs/2104.07636)

#### Medical VLM / Medical Imaging
- **Exploiting Diffusion Prior for Real-World Image Super-Resolution** (2023) — 595 citations [arXiv](https://arxiv.org/abs/2305.07015)
- **Super-Resolution in Medical Imaging** (2009) — 591 citations
- **Improved Distribution Matching Distillation for Fast Image Synthesis** (2024) — 415 citations [arXiv](https://arxiv.org/abs/2405.14867)
- **Mean Flows for One-step Generative Modeling** (2025) — 291 citations [arXiv](https://arxiv.org/abs/2505.13447)
- **SinSR: Diffusion-Based Image Super-Resolution in a Single Step** (2023) — 289 citations [arXiv](https://arxiv.org/abs/2311.14760)

#### Other Foundations
- **ImageNet: A large-scale hierarchical image database** (2009) — 72619 citations
- **ImageNet Large Scale Visual Recognition Challenge** (2014) — 42437 citations [arXiv](https://arxiv.org/abs/1409.0575)
- **AUTO-ENCODING VARIATIONAL BAYES** (2020) — 22778 citations
- **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric** (2018) — 17160 citations [arXiv](https://arxiv.org/abs/1801.03924)
- **Paper** (1977) — 5132 citations

#### Latent Reasoning / Thinking
- **Denoising Diffusion Probabilistic Models** (2020) — 29911 citations [arXiv](https://arxiv.org/abs/2006.11239)
- **A Style-Based Generator Architecture for Generative Adversarial Networks** (2018) — 12761 citations [arXiv](https://arxiv.org/abs/1812.04948)
- **Hierarchical Text-Conditional Image Generation with CLIP Latents** (2022) — 8856 citations [arXiv](https://arxiv.org/abs/2204.06125)
- **Density estimation using Real NVP** (2016) — 4311 citations [arXiv](https://arxiv.org/abs/1605.08803)
- **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (2024) — 3596 citations [arXiv](https://arxiv.org/abs/2403.03206)

#### Memory / Agent Memory
- **MUSIQ: Multi-scale Image Quality Transformer** (2021) — 1308 citations [arXiv](https://arxiv.org/abs/2108.05997)

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution](https://arxiv.org/abs/2603.20690) | 2026 | 0 |
| [Allo{SR}$^2$: Rectifying One-Step Super-Resolution to Stay Real via Allomorphic Generative Flows](https://arxiv.org/abs/2604.19238) | 2026 | 0 |
| Realism Control One-step Diffusion for Real-world Image Super Resolution | 2026 | 0 |
| [LPNSR: Optimal Noise-Guided Diffusion Image Super-Resolution Via Learnable Noise Prediction](https://arxiv.org/abs/2603.21045) | 2026 | 0 |
| One Step Diffusion-Based Super-Resolution With Time-Aware Distillation | 2026 | 0 |
| [IR-Flow: Bridging Discriminative and Generative Image Restoration via Rectified Flow](https://arxiv.org/abs/2604.19680) | 2026 | 0 |
| [DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](https://arxiv.org/abs/2603.22271) | 2026 | 0 |
| [GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution](https://arxiv.org/abs/2603.16769) | 2026 | 0 |
| [Tuning Real-World Image Restoration at Inference: A Test-Time Scaling Paradigm for Flow Matching Models](https://arxiv.org/abs/2603.22027) | 2026 | 0 |
| [Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution](https://arxiv.org/abs/2604.24136) | 2026 | 0 |

## BibTeX

```bibtex
@article{zhu2026oftsr,
  title={ OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs },
  author={ Yuanzhi Zhu and Ruiqing Wang and Shilin Lu and Junnan Li and Hanshu Yan and Kai Zhang },
  journal={arXiv preprint arXiv:2412.09465},
  year={ 2026 }
}
```
