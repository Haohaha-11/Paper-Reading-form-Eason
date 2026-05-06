# OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs

**作者**: Yuanzhi Zhu; Ruiqing Wang; Shilin Lu; Junnan Li; Hanshu Yan; Kai Zhang
**会议**: ICLR 2026
**链接**: [arXiv 2412.09465](https://arxiv.org/abs/2412.09465) | [PDF](https://arxiv.org/pdf/2412.09465v3)

## 一句话总结

OFTSR 先训练一个以 LR 为条件、从噪声增强 LR 走向 HR 的 rectified-flow teacher，再用 ODE trajectory alignment 把 teacher 的多步可调轨迹压进 one-step student，使单次前向仍能通过 $t$ 在 fidelity 与 realism 之间调节。

## 核心贡献

1. 从 diffusion 转向 flow/ODE 视角训练 conditional SR teacher。
2. 用噪声增强的 LR 初始分布扩大一对多 SR 支持集，避免直接 LR→HR flow 坍缩成单一解。
3. 用 ODE trajectory alignment、alignment loss 和 boundary loss 约束 student 的单步预测，使不同 $t$ 的输出落在 teacher 同一条 PF-ODE 轨迹上。
4. 在 FFHQ/DIV2K/ImageNet 与 real-world SR 上展示单步 SR、可调 trade-off 和蒸馏到 ResShift/DiT4SR teacher 的通用性。

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
| Teacher | noise-augmented conditional rectified flow |
| Distillation | ODE trajectory alignment + alignment loss + boundary loss |
| Trade-off knob | inference timestep / interpolation parameter $t$ |
| Datasets | FFHQ, DIV2K, ImageNet 256×256, RealSR, RealSet80, RealLQ250 |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. 构造 LR 条件 | HR image，经退化算子得到 LR，再上采样到 HR 尺寸 | LR 作为 condition 保留观测信息 | $\mathbf{x}_{LR}$ |
| 2. 噪声增强初态 | $\mathbf{x}_{LR}$ 与 Gaussian noise | VP noising 得到 $\mathbf{x}_0=\sqrt{1-\sigma_p^2}\mathbf{x}_{LR}+\sigma_p\epsilon$，扩大初始分布支持集 | 可产生多样 HR 的 noisy LR initial state |
| 3. Conditional flow teacher | $\mathbf{x}_0$、$\mathbf{x}_{LR}$、time $t$、HR target $\mathbf{x}_1$ | 训练 velocity field $\mathbf{v}_\theta(\mathrm{concat}(\mathbf{x}_t,\mathbf{x}_{LR}),t)$，从 $\mathbf{x}_0$ 沿 ODE 走向 HR | 多步 teacher trajectory 与可估计的 $\mathbf{x}_1^t$ |
| 4. Trajectory distillation | 同一 $\mathbf{x}_{0,LR}$ 上的两个时间 $t,s$ | student 预测 $\mathbf{x}_t,\mathbf{x}_s$，teacher 在 $\mathbf{x}_t$ 处给速度，约束二者位于同一 PF-ODE 轨迹 | one-step student $\mathbf{v}_\phi$ |
| 5. 推理可调 | $\mathbf{x}_0$、$\mathbf{x}_{LR}$、参数 $t$ | 单次前向输出 $\mathbf{x}_1^t=\mathbf{x}_0+\mathbf{v}_\phi(\mathbf{x}_{0,LR},t)$ | $t\approx0$ 偏 fidelity，$t\approx1$ 偏 realism 的 SR image |

**整体输入**: LR condition + noise-augmented initial state + trade-off parameter $t$
**整体输出**: 单步生成的 HR image；$t$ 控制从保真、平滑到更真实、更有纹理的输出移动。

## 优缺点与还能做什么

### 优点
- 从 flow/ODE 角度重新定义 one-step SR，比固定 timestep diffusion 更自然地表达连续轨迹。
- 噪声增强 LR 初态解决直接 conditional flow 的一对一坍缩问题，同时 LR condition 又补回观测信息。
- trajectory alignment 直接约束 student 的不同 $t$ 输出共享 teacher ODE 轨迹，不只是拟合最终图像或单点 score。
- 保留 tunable fidelity-realism trade-off，回应单步方法“快但不可调”的痛点；附录还说明 $t\in[0,1]$ 内较稳定，越界会失败。
- 蒸馏框架可用于自训 teacher、ResShift teacher 和 DiT4SR teacher，说明方法不只绑定一个 backbone。

### 局限 / 风险
- 需要训练/依赖强 conditional flow teacher，成本和实现门槛高于直接 one-step distillation。
- student 仍受 teacher 上限约束；附录明确把 teacher 能力列为主要 limitation，未来可能还要加 GT regression 或 adversarial supervision。
- distillation 后的 perception-distortion frontier 与 teacher 有轻微 timestep shift，说明“同一 $t$”不一定语义完全对齐。
- 真实世界复杂退化不一定能由训练退化和 teacher trajectory 完整覆盖，尤其是 blind degradation、压缩伪影和 AI 生成图增强。
- 控制参数 $t$ 对用户友好但仍需校准；论文主要用指标/视觉说明，缺少面向任务风险的自动档位选择。
- 相比强 SD prior 方法，开放域语义纹理的上限取决于 teacher；OFTSR-DiT4SR 的强表现也说明很多能力来自 teacher prior。

### 还能做什么
- 把 real-world degradation estimator 引入 flow condition，让 trajectory 同时感知退化类型。
- 做 $t$ 的自动选择：按医学、遥感、电影修复等任务分别优化保真阈值、感知指标和 hallucination 风险。
- 把 trajectory alignment 与 latent diffusion / SD teacher 更系统结合，区分蒸馏方法收益和 teacher prior 收益。
- 学习用户可解释的 fidelity-realism slider，并给出不确定性、结构一致性报警或推荐档位。
- 扩展到盲 SR、视频 SR、任意倍率 SR，重点验证时间一致性、尺度泛化和 flow 控制是否仍稳定。

## 阅读 Q&A 记录

- **Q: OFTSR 和 diffusion one-step 的差别在哪里？**
  A: OFTSR 不只是把多步 diffusion teacher 的最终输出压成一步，而是让 student 在不同 $t$ 上的隐式预测落到 teacher 的同一条 PF-ODE 轨迹上。这样单步模型仍保留“沿轨迹取不同位置”的控制语义。
- **Q: 为什么 flow 适合可调 trade-off？**
  A: rectified flow 直接建 velocity field，$\mathbf{x}_1^t$ 可以理解为从当前轨迹点估计终点。小 $t$ 更接近 LR 初态，偏 MMSE/PSNR；大 $t$ 更接近 teacher 生成终点，偏 LPIPS/FID/真实感。
- **Q: conditional flow teacher 为什么要 noise-augmented LR？**
  A: 直接从 LR 分布流到 HR 分布会把一个 LR 输入推向单一 HR 解。给 LR 加噪声扩大初始支持集，再把干净 LR 作为 condition，可以同时保留观测约束和多样性。
- **Q: ODE trajectory alignment distillation 对齐的到底是什么？**
  A: 对齐的是 student 由同一初态在 $t,s$ 产生的中间状态与 teacher PF-ODE 的局部推进关系；teacher 提供的是轨迹速度/ODE 约束，而不是单纯的最终 HR 标签。
- **Q: 主要风险是什么？**
  A: teacher trajectory 如果没有覆盖真实退化分布，student 再怎么对齐也只是在受限轨迹上可调；此外 distillation 会带来 $t$ 语义偏移，部署时需要校准。

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

#### Restoration / Evaluation / Efficient Vision
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

#### Generative Priors / Efficient Sampling
- **Denoising Diffusion Probabilistic Models** (2020) — 29911 citations [arXiv](https://arxiv.org/abs/2006.11239)
- **A Style-Based Generator Architecture for Generative Adversarial Networks** (2018) — 12761 citations [arXiv](https://arxiv.org/abs/1812.04948)
- **Hierarchical Text-Conditional Image Generation with CLIP Latents** (2022) — 8856 citations [arXiv](https://arxiv.org/abs/2204.06125)
- **Density estimation using Real NVP** (2016) — 4311 citations [arXiv](https://arxiv.org/abs/1605.08803)
- **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (2024) — 3596 citations [arXiv](https://arxiv.org/abs/2403.03206)

#### Evaluation Metrics
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
