# OFTSR: One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs

**作者**: Yuanzhi Zhu; Ruiqing Wang; Shilin Lu; Junnan Li; Hanshu Yan; Kai Zhang  
**会议/年份**: ICLR 2026  
**链接**: [arXiv](https://arxiv.org/abs/2412.09465) | [PDF](https://arxiv.org/pdf/2412.09465v3)  
**主题**: One-Step Diffusion Super-Resolution  
**阅读日期**: 2026-05-05

## 一句话总结

OFTSR 用 conditional flow teacher 和 ODE-trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

## Why Read This Paper

- 它给 one-step SR 提供 diffusion 之外的 flow matching/ODE 路线。
- 它把可调 fidelity-realism trade-off 明确作为模型能力，而不是后处理选择。
- 适合与你课题中的 distillation objective 设计做重点对比。

## 核心贡献

1. 从 diffusion 转向 flow/ODE 视角，先训练 conditional flow-based SR teacher，再蒸馏单步 student。
2. 提出 trajectory alignment：同一输入的 student 初始态预测要落在 teacher 同一采样 ODE 轨迹上。
3. 结合 alignment/boundary 等约束，使单步输出既接近 teacher 轨迹又能调节 fidelity-realism。
4. 在 FFHQ、DIV2K、ImageNet 256×256 等数据集上报告 one-step SR SOTA 和可调 trade-off。

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
| Teacher type | conditional flow-based SR model |
| Distillation target | same sampling ODE trajectory alignment |
| Datasets | FFHQ 256×256, DIV2K, ImageNet 256×256 |

## Reusable Ideas

- **Problem framing**: 常规蒸馏给固定折中，flow trajectory 蒸馏尝试保留可调 trade-off。
- **Method design**: 用 teacher 的连续轨迹约束 student，而不是只匹配最终样本。
- **Experiment design**: 用 perception-distortion 曲线展示不同控制点。
- **Writing pattern**: 先讲 generative SR 高质量和慢，再讲 fixed trade-off 的 distillation 缺陷。

## Open Questions

- flow teacher 训练成本是否抵消单步 student 的部署收益？
- trajectory alignment 对真实复杂退化 Real-ISR 的泛化如何？

## 📊 Citation Landscape

- **Semantic Scholar**: API detail 暂时被限流或未返回；本地缓存见 `semantic-scholar/detail.json`。
- **Connected Papers**: https://www.connectedpapers.com/main/2412.09465
- **TLDR**: 暂缺。
- **引用统计**: 暂缺。

### 参考文献分组

#### Super-Resolution / Restoration
- High-Resolution Image Synthesis with Latent Diffusion Models (Computer Vision and Pattern Recognition, 2021) — citations: 24029 — https://arxiv.org/abs/2112.10752
- ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (ECCV Workshops, 2018) — citations: 4578 — https://arxiv.org/abs/1809.00219
- SwinIR: Image Restoration Using Swin Transformer (2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021) — citations: 4510 — https://arxiv.org/abs/2108.10257
- NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study (2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017) — citations: 3872
- Image Super-Resolution via Iterative Refinement (IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021) — citations: 2426 — https://arxiv.org/abs/2104.07636

#### Diffusion / Generative Prior
- Denoising Diffusion Probabilistic Models (Neural Information Processing Systems, 2020) — citations: 29911 — https://arxiv.org/abs/2006.11239
- Diffusion Models Beat GANs on Image Synthesis (Neural Information Processing Systems, 2021) — citations: 11531 — https://arxiv.org/abs/2105.05233
- Score-Based Generative Modeling through Stochastic Differential Equations (International Conference on Learning Representations, 2020) — citations: 10215 — https://arxiv.org/abs/2011.13456
- Deep Unsupervised Learning using Nonequilibrium Thermodynamics (International Conference on Machine Learning, 2015) — citations: 9727 — https://arxiv.org/abs/1503.03585
- Hierarchical Text-Conditional Image Generation with CLIP Latents (arXiv.org, 2022) — citations: 8856 — https://arxiv.org/abs/2204.06125

#### Other Foundations
- ImageNet: A large-scale hierarchical image database (2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009) — citations: 72619
- ImageNet Large Scale Visual Recognition Challenge (International Journal of Computer Vision, 2014) — citations: 42437 — https://arxiv.org/abs/1409.0575
- AUTO-ENCODING VARIATIONAL BAYES (N/A, 2020) — citations: 22778
- A Style-Based Generator Architecture for Generative Adversarial Networks (Computer Vision and Pattern Recognition, 2018) — citations: 12761 — https://arxiv.org/abs/1812.04948
- Paper (N/A, 1977) — citations: 5132

#### Perceptual Quality / GAN Metrics
- Generative Adversarial Networks (International Conference on Computing Communication and Networking Technologies, 2021) — citations: 30528 — https://arxiv.org/abs/2203.00667
- The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018) — citations: 17160 — https://arxiv.org/abs/1801.03924
- Prompt-to-Prompt Image Editing with Cross Attention Control (International Conference on Learning Representations, 2022) — citations: 2603 — https://arxiv.org/abs/2208.01626
- Generating Diverse High-Fidelity Images with VQ-VAE-2 (Neural Information Processing Systems, 2019) — citations: 2287 — https://arxiv.org/abs/1906.00446
- Exploring CLIP for Assessing the Look and Feel of Images (AAAI Conference on Artificial Intelligence, 2022) — citations: 1197 — https://arxiv.org/abs/2207.12396

#### Flow / ODE Models
- Density estimation using Real NVP (International Conference on Learning Representations, 2016) — citations: 4311 — https://arxiv.org/abs/1605.08803
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (International Conference on Learning Representations, 2022) — citations: 2761 — https://arxiv.org/abs/2209.03003
- MUSIQ: Multi-scale Image Quality Transformer (IEEE International Conference on Computer Vision, 2021) — citations: 1308 — https://arxiv.org/abs/2108.05997
- Rectified Flow: A Marginal Preserving Approach to Optimal Transport (arXiv.org, 2022) — citations: 241 — https://arxiv.org/abs/2209.14577
- Icml tutorial on the blessing of flow (International conference on machine learning, None) — citations: None

### 推荐论文

- MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.20690
- Allo{SR}$^2$: Rectifying One-Step Super-Resolution to Stay Real via Allomorphic Generative Flows (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.19238
- Realism Control One-step Diffusion for Real-world Image Super Resolution (AAAI Conference on Artificial Intelligence, 2026) — citations: 0
- LPNSR: Optimal Noise-Guided Diffusion Image Super-Resolution Via Learnable Noise Prediction (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.21045
- One Step Diffusion-Based Super-Resolution With Time-Aware Distillation (IEEE Transactions on Image Processing, 2026) — citations: 0
- IR-Flow: Bridging Discriminative and Generative Image Restoration via Rectified Flow (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.19680
- DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.22271
- GDPO-SR: Group Direct Preference Optimization for One-Step Generative Image Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.16769
- Tuning Real-World Image Restoration at Inference: A Test-Time Scaling Paradigm for Flow Matching Models (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.22027
- Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World Super-Resolution (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.24136
