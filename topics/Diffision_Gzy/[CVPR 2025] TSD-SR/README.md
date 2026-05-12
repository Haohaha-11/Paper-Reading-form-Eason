# TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution

**作者**: Linwei Dong; Qingnan Fan; Yihong Guo; Zhonghao Wang; Qi Zhang; Jinwei Chen; Yawei Luo; Changqing Zou
**会议**: CVPR 2025 | **年份**: 2025
**链接**: [arXiv 2411.18263](https://arxiv.org/abs/2411.18263) | [PDF](https://arxiv.org/pdf/2411.18263) | [Code](https://github.com/Doooccer/TSD-SR) | [Semantic Scholar](https://www.semanticscholar.org/paper/7265655eb9f5dc7a00857fad3aaf86659c9ffcc3)

## 一句话总结
TSD-SR 用 Target Score Distillation 和 Distribution-Aware Sampling Module 改造 one-step diffusion SR 的蒸馏梯度，让单步模型在保持极快推理的同时恢复更自然的真实纹理。

## 核心贡献
1. 提出 Target Score Distillation，把 VSD 的 teacher guidance 与真实图像 reference 关联，缓解普通 VSD 在 suboptimal noisy samples 上方向不可靠的问题。
2. 提出 Distribution-Aware Sampling Module，用 diffusion prior 生成更贴近真实采样轨迹的 noisy latent，让细节相关梯度更容易被 student 使用。
3. 在 one-step Real-ISR 中同时保留 reconstruction loss、VSD/target score regularization 和 LoRA fake-score training，推理时只部署 student。
4. 实验显示 TSD-SR 在多数感知/分布指标上优于一步 baseline，并在 A100 512x512 输入上达到 0.1362s 推理，比 SeeSR 快 40x 以上。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机、问题定义和贡献 |
| [02 - Related Work](sections/02-related-work.md) | Real-ISR / diffusion prior / score distillation 背景 |
| [03 - Methodology](sections/03-methodology.md) | TSD、TSM、DASM 和训练目标 |
| [04 - Experiments](sections/04-experiments.md) | 主结果、速度、用户研究和消融 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结与局限 |
| [06 - Appendix](sections/06-appendix.md) | 补充实验、理论、算法和 references |

## 关键数字

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Base teacher | SD3 |
| Training cost | 96 hours, 8 NVIDIA V100, batch size 16 |
| A100 512x512 inference time | 0.1362s |
| Speed vs SeeSR | > 40x faster |
| Speed vs SUPIR | > 120x faster |
| User preference | 69.2% overall satisfaction |
| DASM setting | N=4, s=50 |
| Semantic Scholar citations | 72 citations, 14 influential citations, 70 references |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Student SR | LQ image $x_L$ | one-step student $G_\theta$ 直接生成 restored image / latent | 初始 SR 结果 |
| 2. Noisy latent construction | student output latent | DASM 构造更贴近 diffusion 真实轨迹的 $\hat z_t$ | 更适合 teacher 打分的 noisy latent |
| 3. Score guidance | $\hat z_t$、teacher SD、LoRA model、HQ reference | VSD + Target Score Matching 校正 score direction | TSD 梯度 |
| 4. Student update | reconstruction loss + TSD regularization | 更新 VAE encoder 与 denoising network；LoRA model 用 diffusion loss 更新 | one-step TSD-SR student |
| 5. Inference | LQ image + fixed/default prompt | 只运行 student 单次前向 | 真实感更强的 HR image |

## 优缺点与还能做什么

### 优点
- 直接针对 one-step SR 的梯度方向问题，而不是只减少采样步数。
- DASM 把“细节梯度不可达”变成可操作的 noisy latent 采样问题。
- 速度证据明确：在 A100 512x512 输入上比多步 diffusion SR 快一个数量级以上。
- 消融覆盖 TSM、DASM、base model 和 DASM 超参，证据链比较完整。

### 局限 / 风险
- 推理快但训练成本高，96 小时 8xV100 对普通复现并不轻。
- 感知指标提升伴随 PSNR/SSIM trade-off，结构敏感任务需要额外检查 hallucination。
- 依赖 SD3 teacher 与训练数据分布，真实退化超出覆盖时 score guidance 可能仍会偏。
- 当前没有显式可控参数，用户不能像 TADSR/OFTSR 那样调节 realism 强度。

### 还能做什么
- 与 TADSR 的 timestep control 结合，把 TSD-SR 从固定真实感映射扩展成可控 one-step SR。
- 增加结构一致性或 OCR/face identity 约束，降低真实纹理带来的 hallucination 风险。
- 做退化感知 DASM，让 N、s 或 noisy latent 构造随输入退化程度自适应。
- 将 Target Score Distillation 迁移到 flow teacher 或 DiT-based SR teacher 上验证通用性。

## 阅读 Q&A 记录

- **Q: TSD-SR 和 OSEDiff 最大差别是什么？**
  A: OSEDiff 主要把 VSD 用到 one-step Real-ISR；TSD-SR 进一步指出普通 VSD 在 SR noisy latent 上可能方向不可靠，于是用 target score 和 DASM 校正梯度。
- **Q: DASM 真正改变了什么中间表示？**
  A: 它改变送入 teacher/LoRA 的 noisy latent $\hat z_t$，让这个点更接近 diffusion prior 的真实采样轨迹，从而获得更有效的细节梯度。
- **Q: 为什么 PSNR/SSIM 不是主证据？**
  A: Real-ISR 是一对多恢复，真实纹理可能降低逐像素分数；论文用 LPIPS、DISTS、FID、NIQE、MUSIQ、CLIPIQA 和用户研究补足评价。
- **Q: 推理时是否还需要 teacher 和 LoRA regularizer？**
  A: 不需要。teacher/LoRA/VSD/TSD 都是训练分支；推理只运行 one-step student。

## 📊 Citation Landscape

> 数据来源: 本目录 `semantic-scholar/` JSON | [Connected Papers](https://www.connectedpapers.com/main/2411.18263)

### TLDR (Semantic Scholar 自动生成)

> The TSD-SR is proposed, a novel distillation framework specifically designed for real-world image super-resolution, aiming to construct an efficient and effective one-step model.

### 引用统计

| 指标 | 数值 |
|------|------|
| Semantic Scholar Paper ID | `7265655eb9f5dc7a00857fad3aaf86659c9ffcc3` |
| 参考文献数 | 70 |
| 被引次数 | 72 |
| Influential Citations | 14 |

### 参考文献分组 (Top by citations, Semantic Scholar)

#### Diffusion / Generative Foundations
- **Denoising Diffusion Probabilistic Models** (2020) - 30139 citations [arXiv](https://arxiv.org/abs/2006.11239)
- **High-Resolution Image Synthesis with Latent Diffusion Models** (2021) - 24197 citations [arXiv](https://arxiv.org/abs/2112.10752)
- **Score-Based Generative Modeling through Stochastic Differential Equations** (2020) - 10329 citations [arXiv](https://arxiv.org/abs/2011.13456)
- **Classifier-Free Diffusion Guidance** (2022) - 6135 citations [arXiv](https://arxiv.org/abs/2207.12598)
- **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (2024) - 3647 citations [arXiv](https://arxiv.org/abs/2403.03206)

#### One-Step / Distillation
- **One-Step Diffusion with Distribution Matching Distillation** (2023) - 727 citations [arXiv](https://arxiv.org/abs/2311.18828)
- **Adversarial Diffusion Distillation** (2023) - 711 citations [arXiv](https://arxiv.org/abs/2311.17042)
- **Improved Distribution Matching Distillation for Fast Image Synthesis** (2024) - 422 citations [arXiv](https://arxiv.org/abs/2405.14867)
- **SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation** (2023) - 107 citations [arXiv](https://arxiv.org/abs/2312.05239)
- **AddSR: Accelerating Diffusion-based Blind Super-Resolution with Adversarial Diffusion Distillation** (2024) - 55 citations [arXiv](https://arxiv.org/abs/2404.01717)

#### Real-ISR / Restoration
- **Real-ESRGAN** (2021) - 1856 citations [arXiv](https://arxiv.org/abs/2107.10833)
- **Exploiting Diffusion Prior for Real-World Image Super-Resolution** (2023) - 597 citations [arXiv](https://arxiv.org/abs/2305.07015)
- **ResShift** (2023) - 503 citations [arXiv](https://arxiv.org/abs/2307.12348)
- **SeeSR** (2023) - 336 citations [arXiv](https://arxiv.org/abs/2311.16518)
- **OSEDiff** (2024) - 234 citations [arXiv](https://arxiv.org/abs/2406.08177)

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution](https://arxiv.org/abs/2603.20690) | 2026 | 0 |
| [Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution](https://arxiv.org/abs/2604.24136) | 2026 | 0 |
| [GramSR: Visual Feature Conditioning for Diffusion-Based Super-Resolution](https://arxiv.org/abs/2604.25457) | 2026 | 0 |
| One Step Diffusion-Based Super-Resolution With Time-Aware Distillation | 2026 | 0 |
| Realism Control One-step Diffusion for Real-world Image Super Resolution | 2026 | 0 |

## BibTeX

```bibtex
@inproceedings{dong2025tsdsr,
  title={TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution},
  author={Dong, Linwei and Fan, Qingnan and Guo, Yihong and Wang, Zhonghao and Zhang, Qi and Chen, Jinwei and Luo, Yawei and Zou, Changqing},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={23174--23184},
  year={2025}
}
```
