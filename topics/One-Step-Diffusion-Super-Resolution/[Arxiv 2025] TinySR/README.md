# TinySR: Pruning Diffusion for Real-World Image Super-Resolution

**作者**: Linwei Dong; Qingnan Fan; Yuhang Yu; Qi Zhang; Jinwei Chen; Yawei Luo; Changqing Zou
**会议**: Arxiv 2025
**链接**: [arXiv 2508.17434](https://arxiv.org/abs/2508.17434) | [PDF](https://arxiv.org/pdf/2508.17434v2)

## 一句话总结

TinySR 面向实时 Real-ISR，把 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存压成轻量单步模型。

更准确地说，它不是提出新的 SR diffusion 采样过程，而是把 TSD-SR 这类 one-step diffusion SR 的推理图拆开检查：哪些 transformer block 可以删、VAE 哪些部分是瓶颈、默认 prompt 和固定 timestep 是否还有必要、adaLN 调制参数能否预计算。

## 核心贡献

1. 用 Dynamic Inter-block Activation 在 block 之间动态转移保留层预算，缓解固定分块剪枝搜索空间过窄的问题。
2. 用 Expansion-Corrosion 根据边际保留概率把 learned mask 落到具体层选择上，目标是提高剪后 recoverability。
3. 压缩 VAE、移除 time/prompt 相关模块，并预缓存 adaLN 调制参数，减少完整单步推理图的系统开销。
4. 相对 TSD-SR teacher 最高 5.68× 加速、84% MAC 减少、83% 参数减少，同时保持接近 teacher 的感知质量。

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
| MAC reduction | 84% |
| Parameter reduction | 83% |
| 目标 | real-time one-step Real-ISR |
| Teacher | TSD-SR |
| 压缩对象 | denoising transformer + VAE + prompt/time modules + modulation cache |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Teacher 准备 | TSD-SR one-step teacher、LQ/HR pairs | 生成 teacher latent/features，作为剪枝与恢复训练的锚点 | teacher denoising output / latent reference |
| 2. Mask learning | Transformer layers grouped into blocks | 用 $p(m)$ 学 block 内候选 mask，用 Gumbel-Softmax 让离散选择可训练 | 可恢复性更高的候选剪枝结构 |
| 3. Dynamic Inter-block Activation | block-wise masks | 用 $q(t)$ 在相邻 block 间转移保留层预算，突破固定分块只覆盖小搜索空间的问题 | 更灵活的跨 block sparsity profile |
| 4. Expansion-Corrosion 决策 | learned marginal probability $\pi_i$ | 高边际概率层 expansion 加回，低边际概率层 corrosion 删除，保持总激活层数预算 | 最终 shallow denoising transformer |
| 5. VAE 压缩 | teacher VAE encoder/decoder | 通道上限剪到 64，移除 attention，encoder 使用 SepConv，decoder 保守处理以保质量 | lightweight VAE |
| 6. 条件与缓存 | 默认 prompt、固定 timestep、adaLN shift/scale | 移除 text/time embedding 及相关模块，预缓存稳定的调制参数 | 更小的单步推理图 |
| 7. 恢复训练 | pruned student + tiny VAE + teacher supervision | latent distillation 后接 pixel-space LPIPS/GAN 微调 | TinySR student |
| 8. 输出 | LQ image | 单步生成 HR latent，经 tiny decoder 解码 | real-time SR result |

**整体输入**: LQ image + one-step diffusion SR teacher
**整体输出**: 轻量化 one-step SR student 与实时 SR image

## 优缺点与还能做什么

### 优点
- 问题定义非常工程化：不是提出更复杂先验，而是让 one-step diffusion SR 真正可部署。
- 剪枝覆盖 denoising transformer、VAE、prompt/time 模块和调制缓存，属于完整推理图加速而不是单点优化。
- DIA + Expansion-Corrosion 明确针对“剪完能不能恢复”的问题，比静态重要性或经验剪枝更贴合 teacher-student 压缩。
- 相对 teacher 最高 5.68x 加速、84% MAC 减少、83% 参数减少，核心指标清晰。
- VAE 消融给出了强证据：VAE 本身可达到 10x inference speed 和 22x MAC reduction，说明“只剪 denoiser”会漏掉瓶颈。

### 局限 / 风险
- 学生模型上限强依赖 TSD-SR teacher；teacher 的 hallucination、过生成纹理和 fidelity/realism trade-off 仍可能被继承。
- 剪枝策略可能对硬件、分辨率和退化分布敏感，跨设备最优结构未必一致。
- 删除 prompt/time 模块建立在固定单步 SR 场景上，若未来要做文本控制、多 timestep 或退化自适应，这些模块可能不能直接删。
- VAE decoder 对最终纹理很敏感，过度使用 SepConv 或 attention removal 可能带来细节损失。
- 论文主要报告 V100 上的 latency/MACs，端侧 GPU/NPU/CPU 上的真实瓶颈排序可能变化。

### 还能做什么
- 做硬件感知剪枝/量化联合搜索，直接优化端侧 latency 和显存峰值。
- 引入按图像退化程度动态选择 block 的 conditional computation。
- 尝试把 DIA/Expansion-Corrosion 与量化、KV/cache 或算子融合结合，验证是否能继续压端侧推理。
- 做 teacher-agnostic 版本：测试同一剪枝策略迁移到 OSEDiff、SinSR 或其他 one-step SR teacher 时是否稳定。
- 评估 video SR 或连续帧一致性，验证实时系统中是否出现 temporal flicker。

## 阅读 Q&A 记录

- **Q: TinySR 的核心不是新的生成先验吗？**
  A: 对，它更像系统压缩论文：核心贡献在剪枝和模块移除，让已有 one-step diffusion SR 更小更快。
- **Q: 为什么要同时压缩 VAE？**
  A: 在 latent diffusion SR 中，VAE encoder/decoder 也是推理瓶颈；只剪 U-Net 会留下明显系统开销。
- **Q: 剪掉 time/prompt 分支是否危险？**
  A: 单步 teacher 往往固定 timestep/prompt 使用，移除可减少冗余；但这也意味着 TinySR 更适合固定单步 SR 推理，不适合继续依赖文本控制或多步时间条件的模型。
- **Q: Dynamic Inter-block Activation 到底动态在哪里？**
  A: 动态在 block 之间转移保留层预算：一个 block 少留层，另一个 block 可以多留层，总计算预算不变，但结构不再被固定分块强约束。
- **Q: Expansion-Corrosion 和普通取最大概率 mask 有什么区别？**
  A: 它用训练得到的边际概率选择“加回”或“删掉”的具体层，避免只依赖某个局部最大 mask，目标是提高剪后恢复能力。
- **Q: 为什么 VAE decoder 不像 encoder 一样全面轻量化？**
  A: decoder 直接决定最终像素纹理，作者发现 SepConv 放到 decoder 会伤质量，所以只在 encoder 使用更激进的轻量卷积。

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

#### Diffusion / Efficient Model Foundations
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

#### Restoration / Evaluation / Efficient Vision
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

#### Diffusion Generative Backbones
- **Denoising Diffusion Probabilistic Models** (2020) — 29911 citations [arXiv](https://arxiv.org/abs/2006.11239)
- **A Style-Based Generator Architecture for Generative Adversarial Networks** (2018) — 12761 citations [arXiv](https://arxiv.org/abs/1812.04948)
- **Categorical Reparameterization with Gumbel-Softmax** (2016) — 6242 citations [arXiv](https://arxiv.org/abs/1611.01144)
- **Scalable Diffusion Models with Transformers** (2022) — 5516 citations [arXiv](https://arxiv.org/abs/2212.09748)
- **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (2024) — 3596 citations [arXiv](https://arxiv.org/abs/2403.03206)

#### Model Compression / Efficient Evaluation
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
