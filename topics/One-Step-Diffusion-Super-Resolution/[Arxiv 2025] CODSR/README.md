# Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution

**作者**: Hao Chen; Junyang Chen; Jinshan Pan; Jiangxin Dong
**会议**: Arxiv 2025
**链接**: [arXiv 2512.14061](https://arxiv.org/abs/2512.14061) | [PDF](https://arxiv.org/pdf/2512.14061v1)

## 一句话总结

CODSR 把单步 diffusion SR 的控制粒度从“全图统一真实感”细化到局部区域：LQFM 保住低质图里仍可信的结构，RGPA 只在纹理需求高的区域释放生成先验，TMG 再把文本语义约束到对应物体区域，最终用两阶段 GAN+VSD 训练把 fidelity-realism trade-off 压到可调范围内。

## 核心贡献

1. **LQFM**: 不只依赖 VAE latent，而是把 pixel-unshuffle 后的原始 LQ 特征通过 time-aware SFT 调制 U-Net 中间特征，补偿 VAE 压缩造成的低层信息损失。
2. **RGPA**: 用 Sobel 梯度图构造 spatially adaptive noise，让纹理/边缘区域获得更多 diffusion prior，自然平坦区域少加噪，避免“全图一起 hallucinate”。
3. **TMG**: 从 RAM/DAPE prompt 中抽取名词，经 Grounded-SAM2 得到区域 mask，用 positive area loss 约束 cross-attention，缓解文本 token 与图像区域错位。
4. **两阶段训练**: 第一阶段用 content/LPIPS/GAN 建立稳定 SR 能力，第二阶段冻结已有 LoRA，并在 cross-attention 加新 LoRA，用 VSD 和文本区域约束蒸馏语义生成能力。

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

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. LQ 编码 | LQ image | VAE encoder 产生 latent z_L，同时保留 pixel-unshuffle 后的 LQ feature | latent 表示 + 未压缩 LQ 细节 |
| 2. 区域先验激活 | z_L、Sobel 梯度图、Gaussian noise | RGPA 先按 16x16 patch 平均梯度估计纹理需求，再把标准噪声变成区域加权 adaptive noise | region-aware noisy latent z_t |
| 3. LQ 信息补偿 | U-Net intermediate feature f_m + pixel-unshuffle LQ feature | LQFM 生成 gamma/beta，并用 lambda_t 做 time-aware SFT；调制中间特征而不是直接改 z_L | 更保真的 denoising feature，且不破坏 latent 分布 |
| 4. 语义对齐 | RAM/DAPE prompt、名词 token、Grounded-SAM2 masks | TMG 过滤抽象形容词，只对可定位名词建立 mask，并用 CoMat positive area loss 约束 DAAM/cross-attention | 文本 token 与正确图像区域交互 |
| 5. 两阶段训练 | Stage-1 GAN/content/LPIPS + Stage-2 VSD/positive area loss | 先让 student 会稳定复原，再借 pretrained diffusion teacher 的 VSD 梯度增强语义纹理 | 兼顾 fidelity 与 perceptual quality 的 one-step student |
| 6. 输出 | denoised HR latent | VAE decoder 重建图像 | SR result |

**整体输入**: LQ image + 文本/语义提示 + timestep/noise 强度
**整体输出**: 结构更保真、局部纹理更真实的 SR image

## 优缺点与还能做什么

### 优点
- **局部 trade-off 思路清楚**: 不是用一个全局 timestep 解决所有区域，而是让高梯度区域更偏 realism、平坦区域更偏 fidelity。
- **LQFM 插入位置合理**: 论文专门比较了直接加到 feature、调制 latent、调制 U-Net 中间 feature；最后选择中间 feature，理由是既利用 LQ 结构，又不扰乱 Gaussian latent 分布。
- **TMG 有明确靶点**: 它不是泛泛“加文本提示”，而是解决 noun token 的 cross-attention 落错区域，Figure 5 的 bird DAAM 正好对应这个问题。
- **训练策略与模块职责匹配**: Stage 1 先稳定重建，Stage 2 用 VSD/LoRA 追加语义生成能力，避免一开始就让强生成先验压过保真约束。
- **证据链覆盖较全**: 主结果比较 full-step 与 one-step，消融分别验证 RGPA/LQFM/TMG/VSD，Figure 6 还显式展示 timestep 对 PSNR/MUSIQ 的 trade-off。

### 局限 / 风险
- **局部控制仍是启发式**: RGPA 用 Sobel 梯度近似“需要生成的区域”，但高梯度不等于语义上应该 hallucinate，文字、网格、噪声边缘都可能被误判。
- **timestep 控制仍偏手工**: Figure 6 说明 PSNR 与 MUSIQ 会随 t_s 此消彼长，但默认 t_s=100 是经验选择，还不是面向用户或图像内容自适应的控制器。
- **TMG 依赖外部语义链路**: RAM/DAPE、NLTK noun filtering、Grounded-SAM2 任一环节错误，都会让 cross-attention 被约束到错误区域，反而增强错误纹理。
- **VSD 带来语义生成能力，也带来幻觉风险**: 第二阶段从 pretrained teacher 蒸馏的是真实图像分布梯度，不保证每个生成细节都来自 LQ observation。
- **no-reference 指标不能单独证明真实性**: MUSIQ/CLIPIQA+ 提升说明观感更像自然图，但结构敏感场景仍应看 OCR、身份一致性或下游任务。

### 还能做什么
- 把 RGPA 的 Sobel 权重换成“退化强度 + 语义类别 + 不确定性”的联合估计，区分真实高频、噪声高频和不可恢复区域。
- 学一个 per-region timestep/noise scheduler，让不同区域自动选择 fidelity-realism 档位，而不是全图共享 t_s。
- 用 region caption、视觉 token 或 detector label 替代纯名词 prompt，降低 TMG 对 prompt extractor 和 noun parsing 的依赖。
- 增加失败案例分析：平坦墙面、重复栅格、文字、车牌、人脸、细小 logo 是最容易暴露 hallucination 的位置。
- 补充面向 OCR、人脸身份、医学/遥感等结构敏感任务的下游评估，避免只靠 no-reference IQA 判断可用性。

## 阅读 Q&A 记录

- **Q: RGPA 和普通加噪有什么区别？**
  A: 普通 latent 加噪给全图同一生成强度，RGPA 用 Sobel 梯度构造空间权重，让纹理区更依赖 diffusion prior、平坦区更保留 LQ 结构。
- **Q: 为什么还要 LQFM？**
  A: VAE latent 会丢掉部分低层结构，LQFM 用 pixel-unshuffle LQ feature 生成 time-aware SFT 参数，调制 U-Net 中间特征，补回文字笔画、边缘和局部布局。
- **Q: TMG 是否只是锦上添花？**
  A: 它不是基础重建模块，而是语义归位模块；从 DAAM 和消融看，它主要让 noun token 的纹理生成落到对应物体区域。
- **Q: 为什么 LQFM 要调制 U-Net 中间特征，而不是直接调制 z_L？**
  A: 直接改 z_L 会破坏 latent 的 Gaussian 分布假设，削弱 reverse denoising 对 diffusion prior 的利用；中间 feature 调制更像给 U-Net 补结构条件。
- **Q: CODSR 的 controllable 主要体现在哪里？**
  A: 主要在 t_s/noise intensity 对 fidelity-realism 的连续控制，以及 RGPA 的区域加权控制；它还不是完整用户接口级别的语义滑杆。
- **Q: VSD 两阶段训练解决什么？**
  A: Stage 1 先用重建和 GAN 让 one-step student 不崩；Stage 2 再用 frozen diffusion teacher 的 VSD 梯度和 TMG 约束增强语义纹理，避免语义生成能力完全依赖 GAN。

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

#### Restoration / Evaluation / Efficient Vision
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

#### Diffusion Distillation / Guidance
- **Denoising Diffusion Probabilistic Models** (2020) — 29911 citations [arXiv](https://arxiv.org/abs/2006.11239)
- **A Style-Based Generator Architecture for Generative Adversarial Networks** (2018) — 12761 citations [arXiv](https://arxiv.org/abs/1812.04948)
- **T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models** (2023) — 1630 citations [arXiv](https://arxiv.org/abs/2302.08453)

#### Evaluation / Perceptual Quality
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
