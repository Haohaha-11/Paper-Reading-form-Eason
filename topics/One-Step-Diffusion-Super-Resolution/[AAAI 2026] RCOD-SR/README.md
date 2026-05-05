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
| [Abstract](sections/00-abstract.md) | 题名、作者、摘要、目录或论文总体定位。 |
| [Introduction](sections/01-introduction.md) | 动机、问题定义、贡献与相关工作定位。 |
| [Method](sections/02-method.md) | 核心方法、背景、训练目标和算法细节。 |
| [Experiments](sections/03-experiments.md) | 实验设置、主结果、消融、效率和案例分析。 |
| [Discussion](sections/04-discussion.md) | 结论、局限、附录、补充材料和参考文献。 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |

## Reusable Ideas

- **Problem framing**: 单步方法效率高，但失去了多步方法通过采样步数/噪声强度调节真实感的自由度。
- **Method design**: 把退化程度显式离散/分组，并让模型学会不同组对应的 denoising 行为。
- **Experiment design**: 除 SOTA 对比外，应展示同一输入在不同控制强度下的视觉变化。
- **Writing pattern**: 先用图展示固定输出 vs 可控输出，再解释 latent grouping。

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
