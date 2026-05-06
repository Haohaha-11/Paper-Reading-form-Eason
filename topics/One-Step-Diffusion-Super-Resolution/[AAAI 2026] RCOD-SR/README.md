# Realism Control One-step Diffusion for Real-World Image Super-Resolution

**作者**: Zongliang Wu; Siming Zheng; Peng-Tao Jiang; Xin Yuan  
**会议**: AAAI 2026  
**链接**: [arXiv 2509.10122](https://arxiv.org/abs/2509.10122) | [PDF](https://arxiv.org/pdf/2509.10122v2)

## 一句话总结

RCOD-SR 用 latent domain grouping、退化感知蒸馏和视觉 prompt 注入，让 one-step diffusion SR 具备推理时 realism 控制能力。

## 核心贡献

1. 指出单 timestep 训练会导致固定 fidelity-realism 折中。
2. 提出 Latent Domain Grouping，把退化程度映射到不同 denoising degree。
3. 用 degradation-aware sampling distillation 和 visual prompt injection 增强控制与语义一致性。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Related Work](sections/01-related-work.md) | 相关工作谱系 |
| [02 - Methodology](sections/02-methodology.md) | 核心方法 |
| [03 - Experiments](sections/03-experiments.md) | 实验设置与结果 |
| [04 - Conclusion](sections/04-conclusion.md) | 总结与局限 |
| [05 - Appendix](sections/05-appendix.md) | 附录补充 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| 控制变量 | latent-domain group / denoising degree |
| 会议 | AAAI 2026 Oral |
| 核心问题 | realism controllability |

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2509.10122)

### TLDR (Semantic Scholar 自动生成)

> 暂缺。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 暂缺 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top 5 per category, by citations)

- 参考文献 API 暂缺或限流。

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| Realism Control One-step Diffusion for Real-world Image Super Resolution | 2026 | 0 |
| [MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution](https://arxiv.org/abs/2603.20690) | 2026 | 0 |
| [Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution](https://arxiv.org/abs/2604.24136) | 2026 | 0 |
| [Degradation-Aware and Structure-Preserving Diffusion for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.11470) | 2026 | 0 |
| Selective Diffusion Distillation for Real-World High-Scale Image Super-Resolution | 2026 | 0 |
| [GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution](https://arxiv.org/abs/2603.16769) | 2026 | 0 |
| [GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution](https://arxiv.org/abs/2604.25457) | 2026 | 0 |
| [QUSR: Quality-Aware and Uncertainty-Guided Image Super-Resolution Diffusion Model](https://arxiv.org/abs/2603.09125) | 2026 | 0 |
| [Revisiting the Perception-Distortion Trade-off with Spatial-Semantic Guided Super-Resolution](https://arxiv.org/abs/2603.14112) | 2026 | 0 |
| [VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution](https://arxiv.org/abs/2604.21450) | 2026 | 0 |

## BibTeX

```bibtex
@article{wu2026rcodsr,
  title={ Realism Control One-step Diffusion for Real-World Image Super-Resolution },
  author={ Zongliang Wu and Siming Zheng and Peng-Tao Jiang and Xin Yuan },
  journal={arXiv preprint arXiv:2509.10122},
  year={ 2026 }
}
```
