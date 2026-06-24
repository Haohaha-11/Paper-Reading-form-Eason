# Vision-aligned Latent Reasoning for Multi-modal Large Language Model (VaLR)

## Paper Metadata

| 项目 | 内容 |
|------|------|
| **Title** | Vision-aligned Latent Reasoning for Multi-modal Large Language Model |
| **Authors** | Byungwoo Jeon<sup>1</sup>, Yoonwoo Jeong<sup>2</sup>, Hyunseok Lee<sup>1</sup>, Minsu Cho<sup>2,3,*</sup>, Jinwoo Shin<sup>1,3,*</sup> |
| **Affiliations** | <sup>1</sup>KAIST, <sup>2</sup>POSTECH, <sup>3</sup>KRAFTON |
| **Venue** | arXiv 2026 |
| **Project Page** | Code available at project page |
| **Base Model** | Qwen2.5-VL-7B |

## One-Sentence Summary

VaLR 提出了一种视觉对齐的潜空间推理框架，通过在每步 Chain-of-Thought 推理前动态生成与视觉编码器表征对齐的 latent tokens，解决了 MLLM 在长上下文推理中视觉信号逐渐衰减的问题，实现了 MLLM 领域首个真正的 test-time scaling 行为。

## Core Contributions

1. **揭示 MLLM 视觉信号衰减问题**: 在长上下文推理中，视觉信息随生成序列增长而逐渐稀释，导致现有 MLLM 无法像 LLM 那样从 test-time scaling 中受益。

2. **提出 VaLR 框架** (Section 3): 在每个文本推理步骤前插入可学习的 latent tokens 作为"视觉检查点"，通过 Representation Alignment (REPA) 将 latent tokens 与多个视觉编码器 (DINOv3, SigLIPv2, CLIP, π³) 的稠密表征对齐，使模型在推理过程中持续保持视觉 grounding。

3. **两阶段课程学习策略** (Section 3.3): Stage 1 通过标准 SFT 建立基础多模态推理能力；Stage 2 引入 latent tokens 和 REPA 损失，逐步赋予模型潜空间视觉推理能力。

4. **全面的实验验证** (Section 4): 在 VSI-Bench 上将 Qwen2.5-VL 从 33.0% 提升至 52.9%（+19.9%），并在多个感知 benchmark（BLINK, MMVP, MMStar, V*, CVBench）上一致超越 baseline。验证了首个 MLLM test-time scaling law。

5. **多编码器协同**: 混合使用 DINOv3（细粒度外观/空间）、SigLIPv2（语义理解）、π³（3D 空间结构）三种视觉编码器进行对齐，充分利用各自表征优势，达到最佳性能。

## Section Navigation

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| Abstract & Overview | [00-abstract.md](sections/00-abstract.md) | 论文概述、问题动机、方法框架总览 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 视觉信号衰减问题、与 LLM test-time scaling 对比、VAaR 解决思路 |
| 2. Related Works | [02-related-work.md](sections/02-related-work.md) | MLLM 架构、CoT/潜推理、外部视觉编码器利用 |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | 潜推理形式化、REPA 对齐目标、多编码器对齐、两阶段训练 |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | 3D 空间推理、感知任务、推理长度分析、消融实验、数据可扩展性 |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | 总结与影响声明 |

## Key Numbers

| 指标 | 数值 |
|------|------|
| VSI-Bench (base model) | 33.0% |
| VSI-Bench (VaLR-S, 单编码器 DINOv3) | 41.5% |
| VSI-Bench (VaLR-M, 多编码器) | 52.9% (+19.9%p) |
| VSI-Bench best competitor (GPT-4o) | 34.0% |
| BLINK (VaLR-M) | 64.7% |
| CVBench (VaLR-M) | 87.6% |
| V* (VaLR-M) | 86.9% |
| Training samples | 450K |
| Latent tokens per step (K) | 16 |
| Alignment weight (λ) | 0.5 |
| Stage 1 learning rate | 1e-5 |
| Stage 2 learning rate | 2e-6 (LLM), 1e-5 (MLP) |
| GPU | 4× NVIDIA Tesla A100 |
| DeepSpeed | ZeRO-2 |

## Data Flow: Input → Intermediate → Output

```
| 阶段 | 描述 |
|------|------|
| 1. VaLR Data Flow |  |
| 2. [Input] |  |

Visual features F_φ from frozen vision encoders (train)   │
│                                                                   │
│  [Stage 1: Standard CoT SFT]                                      │
│    ├── Model: Qwen2.5-VL-7B (frozen vision encoder)              │
│    ├── Data: 450K CoT VQA samples                                │
│    ├── Loss: L_CE (cross-entropy)                                │
│    └── Output: Base MLLM with text reasoning capability           │
│                                                                   │
│  [Stage 2: Latent Token Training + REPA]                          │
│    ├── Insert K=16 latent tokens before each reasoning step      │
│    ├── Special tokens: <latent> ... </latent>                     │
│    ├── Extract intermediate features F_MLLM from MLLM            │
│    ├── Project via MLP ψ: F̂_MLLM = ψ(Upsample(F_MLLM))          │
│    ├── Compute REPA loss with vision encoder features:            │
│    │     L_REPA = -cos_sim(F̂_MLLM, F_φ)                         │
│    ├── Total loss: L = L_CE + λ·L_REPA  (λ=0.5)                  │
│    └── Multi-encoder: average REPA losses across encoders        │
│                                                                   │
│  [Inference (no external encoder needed)]                         │
│    ├── Model alternates between latent mode and language mode    │
│    ├── Latent mode: K=16 steps, input = previous hidden state    │
│    ├── Language mode: input = token embedding                    │
│    └── Output: Final answer a                                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Pros/Cons & Future Work

### Strengths

1. **解决视觉衰减核心问题**: 通过在每步推理前动态注入视觉对齐的 latent tokens，从根本上解决了 MLLM 长上下文推理中视觉信号衰减的问题。
2. **首个 MLLM Test-time Scaling**: 是唯一在更长推理链上持续提升性能的方法，baseline 方法均在长推理链上性能下降（如 Ocean-R1 在 MMVP 上从 62.7% 下降到 56.5%）。
3. **编码器无关性**: VaLR 可以与多种视觉编码器（DINOv2/v3, CLIP, SigLIPv2, π³, 甚至 MLLM 原生编码器）配合使用，且编码器越强增益越大。
4. **推理时零额外开销**: 外部视觉编码器仅在训练时用于对齐监督，推理时完全不需要，无额外计算成本。
5. **训练数据高效**: VaLR-M 在 V* benchmark 上实现 >20x 更快收敛达到与 baseline 持平性能。
6. **多编码器协同**: 融合 3D（π³）+ 2D（DINOv3/SigLIPv2）编码器，在不同任务上发挥各自表征优势。

### Weaknesses / Limitations

1. **需要 CoT 标注数据**: 依赖显式的 Chain-of-Thought 推理步骤标注来插入 latent tokens，限制了可用的训练数据规模。
2. **多视图数据需要额外处理**: 对于多视图数据集，需要 GPT-4o 辅助为每个推理步骤匹配合适的目标图像，增加了数据准备成本。
3. **只在 7B 模型验证**: 所有实验基于 Qwen2.5-VL-7B，未在更大规模模型或不同架构上验证泛化性。
4. **Latent tokens 数量有限**: K=16 是实验最优值，但继续增加 K（如 25）带来的增益递减，存在分辨率上限。
5. **两阶段训练**: 需要先 SFT 再 latent token 训练，训练流程相对复杂。
6. **与非公开模型的兼容性**: 需要模型支持 latent mode（特殊 token 控制）和中间层特征提取，闭源 API 模型无法使用。

### Future Work

1. 将 VaLR 扩展到视频和具身智能场景（VLA、CUA）
2. 探索更大规模的模型（13B+）和多模态架构
3. 减少对显式 CoT 标注的依赖，探索弱监督或无监督的 latent token 定位策略
4. 结合 RL 进行 latent reasoning 的策略优化
5. 更细粒度的多尺度视觉对齐策略

## Reading Q&A Record

| # | 问题 | 答案位置 | 解答 |
|---|------|---------|------|
| 1 | 为什么 MLLM 无法像 LLM 那样从 test-time scaling 中受益？ | Section 1, Figure 2 | 随着生成序列增长，MLLM 中的视觉信号逐渐衰减。LLM 的纯文本推理没有这个问题。VaLR 通过在每步推理前注入视觉对齐的 latent tokens 来解决。 |
| 2 | VaLR 与 COCONUT 的核心区别是什么？ | Section 3.1-3.2 | COCONUT 的前一步 hidden state 直接作为下一步输入，但缺乏显式的视觉对齐监督。VaLR 的 latent tokens 通过 REPA loss 与视觉编码器特征对齐，确保 latent tokens 保持视觉信息。 |
| 3 | 为什么不用视觉编码器作为 MLLM 的额外输入？ | Section 3.2, Figure 4 (Appx C.4) | 实验证明 REPA 对齐方式优于将视觉特征作为输入 token（input token method），且 REPA 在推理时不需要外部编码器，更高效。 |
| 4 | 为什么不对齐 MLLM 的 native encoder 而要用外部编码器？ | Section 4.5, Table 3 | 对齐 MLLM 原生编码器已经有效（VaLR w/ QE），但使用更强的专用视觉编码器（如 DINOv3）能进一步提升性能。VaLR 不依赖外部编码器，但它们能提供更丰富的视觉表征。 |
| 5 | 为什么中间层（第 12 层）对齐效果最好？ | Section 4.5, Table 6 | 这与之前的研究一致（Yu et al., 2025; Kang et al., 2025; Jiang et al., 2025），视觉信息在 MLLM 的中间层表现最突出。前层（第 4 层）视觉表征不成熟，后层（第 27 层）偏向语义。 |
| 6 | Latent reasoning baselines（LVR, CoVT, Monet）为什么在 VSI-Bench 上崩溃？ | Section 4.2, Appx C.1 | 这些方法仅支持单视图场景或对多视图扩展有限。没有动态视觉重注入机制，在需要长期视觉记忆的多视图任务上视觉信号完全衰减。 |
| 7 | VaLR 中 K（latent tokens 数）和 λ（REPA 权重）如何影响性能？ | Section 4.5, Table 11-12 | K 越大性能越好但边际递减（16→25 只 +0.2%）；λ=0.5 时最佳，过大会破坏语言语义，过小则视觉对齐不足。是语言语义保持和视觉对齐之间的平衡。 |

## Citation Landscape

### Reference Statistics
- **Reference Count**: 70+
- **Code Available**: Yes (project page)

### Reference Grouping by Topic

**MLLM Backbones & Architectures**:
- Qwen2.5-VL [Bai et al., 2025], LLaVA [Liu et al., 2023; 2024], PaliGemma [Beyer et al., 2024], Flamingo [Alayrac et al., 2022], BLIP-2 [Li et al., 2023], PrismaticVLM [Karamcheti et al., 2024]

**Chain-of-Thought Reasoning**:
- CoT [Wei et al., 2022], Math-Shepherd [Wang et al., 2024], DeepSeekMath [Shao et al., 2024], MetaMath [Yu et al., 2023], Zebra-CoT [Li et al., 2025a], Visual-CoT [Shao et al., 2024a], CogCoM [Qi et al., 2025]

**Latent Reasoning**:
- COCONUT [Hao et al., 2024], LVR [Li et al., 2025c], CoVT [Qin et al., 2025], Monet [Wang et al., 2025c], Machine Mental Imagery [Yang et al., 2025d]

**Representation Alignment**:
- REPA [Yu et al., 2025], Visual Representation Alignment [Yoon et al., 2025]

**Vision Encoders**:
- DINOv2/v3 [Oquab et al., 2023; Simeoni et al., 2025], CLIP [Radford et al., 2021], SigLIP/SigLIPv2 [Zhai et al., 2023; Tschannen et al., 2025], π³ [Wang et al., 2025d], VGGT [Wang et al., 2025b]

**External Vision Encoders in MLLMs**:
- 3DRS [Huang et al., 2025], Spatial-MLLM [Wu et al., 2025], Learning from Videos for 3D World [Zheng et al., 2025a]

**Benchmarks**:
- VSI-Bench [Yang et al., 2025b], BLINK [Fu et al., 2024], MMVP [Tong et al., 2024b], MMStar [Chen et al., 2024b], V* [Wu & Xie, 2024], CVBench [Tong et al., 2024a], MathVista [Lu et al., 2023], MathVision [Wang et al., 2024a], MMhalu [Sun et al., 2024]

**Training Frameworks**:
- VLMEvalKit [Duan et al., 2024], vLLM [Kwon et al., 2023], DeepSpeed ZeRO-2

**Reasoning Search Strategies**:
- Tree-of-Thoughts [Yao et al., 2023], Self-Evaluation Beam Search [Xie et al., 2023], Stream of Search [Gandhi et al., 2024], Dualformer [Su et al., 2024]

---

*Batch reading created on 2026-06-24*
