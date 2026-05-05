# Realism Control One-step Diffusion for Real-World Image Super-Resolution

**作者**: Zongliang Wu; Siming Zheng; Peng-Tao Jiang; Xin Yuan  
**会议/年份**: AAAI 2026  
**链接**: [arXiv](https://arxiv.org/abs/2509.10122) | [PDF](https://arxiv.org/pdf/2509.10122v2)  
**主题**: One-Step Diffusion Super-Resolution  
**阅读日期**: 2026-05-05

## 一句话总结

RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

## Why Read This Paper

- 它直接针对 OSEDiff/S3Diff 类单步模型“只能给一个固定结果”的问题。
- 它把多步 diffusion 的 timestep 控制思想迁移到 one-step 训练和推理中。
- 适合分析 one-step SR 的 controllability 与 fidelity-realism 曲线。

## 核心贡献

1. 指出单 timestep 训练会把不同退化样本压到静态折中点，导致单步模型缺少多步采样天然具备的 realism 控制。
2. 提出 Latent Domain Grouping，把 latent 中不同退化程度映射到不同 timestep/denoising degrees。
3. 提出 degradation-aware sampling distillation，使蒸馏正则与分组策略一致，增强 fidelity-realism trade-off 控制。
4. 用 visual prompt injection 替代普通文本 prompt，提升退化感知条件与语义一致性。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00-title-and-authors](sections/00-title-and-authors.md) | 核心方法、模块和训练目标 |
| [01-abstract](sections/01-abstract.md) | 问题定义、方法一句话和主要 claim |
| [02-introduction](sections/02-introduction.md) | 动机、gap、贡献与方法直觉 |
| [03-related-work](sections/03-related-work.md) | 相关方法谱系和本文定位 |
| [04-real-world-image-super-resolution](sections/04-real-world-image-super-resolution.md) | 核心方法、模块和训练目标 |
| [05-one-step-diffusion-models-for-real-isr](sections/05-one-step-diffusion-models-for-real-isr.md) | 核心方法、模块和训练目标 |
| [06-methodology](sections/06-methodology.md) | 核心方法、模块和训练目标 |
| [07-preliminary-the-character-of-denoising-network-in-one-step-diffusion-for-real-isr](sections/07-preliminary-the-character-of-denoising-network-in-one-step-diffusion-for-real-isr.md) | 扩散/flow/数学背景 |
| [08-latent-domain-grouping](sections/08-latent-domain-grouping.md) | 核心方法、模块和训练目标 |
| [09-latent-metric-for-denoising-network](sections/09-latent-metric-for-denoising-network.md) | 核心方法、模块和训练目标 |
| [10-metric-estimation](sections/10-metric-estimation.md) | 核心方法、模块和训练目标 |
| [11-degradation-aware-sampling-distillation](sections/11-degradation-aware-sampling-distillation.md) | 核心方法、模块和训练目标 |
| [12-visual-prompt-injection-module](sections/12-visual-prompt-injection-module.md) | 核心方法、模块和训练目标 |
| [13-experiments](sections/13-experiments.md) | 实验设置、SOTA 对比和消融 |
| [14-experiments-setups](sections/14-experiments-setups.md) | 实验设置、SOTA 对比和消融 |
| [15-implement-details](sections/15-implement-details.md) | 核心方法、模块和训练目标 |
| [16-comparison-with-state-of-the-arts](sections/16-comparison-with-state-of-the-arts.md) | 实验设置、SOTA 对比和消融 |
| [17-conclusion](sections/17-conclusion.md) | 总结贡献与局限 |
| [18-supplementary-material-for-realism-control-one-step-diffusion-for-real-world-image-super-resolution](sections/18-supplementary-material-for-realism-control-one-step-diffusion-for-real-world-image-super-resolution.md) | 附录实现细节、推导和更多结果 |
| [19-in-the-supplementary-material-we-provide-the-following-contents](sections/19-in-the-supplementary-material-we-provide-the-following-contents.md) | 附录实现细节、推导和更多结果 |
| [20-implement-details-2](sections/20-implement-details-2.md) | 核心方法、模块和训练目标 |
| [21-algorithm-details](sections/21-algorithm-details.md) | 核心方法、模块和训练目标 |
| [22-experiments-2](sections/22-experiments-2.md) | 实验设置、SOTA 对比和消融 |
| [23-ablation-study](sections/23-ablation-study.md) | 分析、讨论、限制或补充实验 |
| [24-references](sections/24-references.md) | 参考文献 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |

## Key Figures / Tables

| ID | What It Shows | Takeaway |
|----|---------------|----------|
| Figure S1 | Figure S1: While previous one-step diffusion methods, such as S3Diff (Zhang et al. 2024) o | 看方法框架、可控性或 SOTA 对比证据 |
| Figure S2 | Figure S2: Realism control one-step diffusion (RCOD) training process. The left part illus | 看方法框架、可控性或 SOTA 对比证据 |
| Table extracted | Table extracted by MinerU. ML L1 MSE Cosine Sim. SSIM ↑ 0.60 0.54 0.78 DISTS ↑ 0.15 0.11 0 | 看方法框架、可控性或 SOTA 对比证据 |
| Table S1 | Table S1: (a) Influence of degradation degree in training data. (b)  Spearman coefficient  | 看方法框架、可控性或 SOTA 对比证据 |
| Figure S3 | Figure S3: Influence of different timesteps $t$ using SD-turbo. | 看方法框架、可控性或 SOTA 对比证据 |
| Figure S4 | Figure S4: Distribution of (a) mean value of LR and HR training images in the latent domai | 看方法框架、可控性或 SOTA 对比证据 |

## Reusable Ideas

- **Problem framing**: 单步方法效率高，但失去了多步方法通过采样步数/噪声强度调节真实感的自由度。
- **Method design**: 把退化程度显式离散/分组，并让模型学会不同组对应的 denoising 行为。
- **Experiment design**: 除 SOTA 对比外，应展示同一输入在不同控制强度下的视觉变化。
- **Writing pattern**: 先用图展示固定输出 vs 可控输出，再解释 latent grouping 的必要性。

## Open Questions

- latent grouping 的边界是否依赖训练退化合成策略？
- realism 控制增强后，是否可能牺牲 identity/text 等局部忠实性？

## 📊 Citation Landscape

- **Semantic Scholar**: API detail 暂时被限流或未返回；本地缓存见 `semantic-scholar/detail.json`。
- **Connected Papers**: https://www.connectedpapers.com/main/2509.10122
- **TLDR**: 暂缺。
- **引用统计**: 暂缺。

### 参考文献分组

- 参考文献 API 暂缺或限流；稍后可重新运行 Semantic Scholar 抓取脚本补齐。

### 推荐论文

- Realism Control One-step Diffusion for Real-world Image Super Resolution (AAAI Conference on Artificial Intelligence, 2026) — citations: 0
- MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.20690
- Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.24136
- Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.11470
- Selective Diffusion Distillation for Real-World High-Scale Image Super-Resolution (AAAI Conference on Artificial Intelligence, 2026) — citations: 0
- GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.16769
- GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.25457
- QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model (IEEE International Conference on Acoustics, Speech, and Signal Processing, 2026) — citations: 0 — https://arxiv.org/abs/2603.09125
- Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.14112
- VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.21450
