# Latent Implicit Visual Reasoning

**作者**: Kelvin Li; Chuyi Shang; Leonid Karlinsky; Rogerio Feris; Trevor Darrell; Roei Herzig
**会议**: Arxiv 2025
**链接**: [arXiv 2512.21218](https://arxiv.org/abs/2512.21218) | [HTML](https://arxiv.org/html/2512.21218) | [PDF](https://arxiv.org/pdf/2512.21218)
**阅读顺序**: Latent-Space-Processing 课题论文
**MinerU task**: `a92548c9-06be-4679-8770-111731532760`

## 一句话总结

LIVR 通过在 LMM prompt 后追加可学习 latent visual reasoning tokens，并用视觉瓶颈 attention mask 强迫视觉信息经 latent 流动，让模型在无需显式中间监督的情况下学习任务自适应视觉抽象，从而在多类视觉中心任务上优于 Direct SFT。

## 核心贡献

1. **Task-agnostic latent visual reasoning**: 不要求 box、crop、depth map、helper image 或文本 CoT，只用最终 QA loss 学隐式视觉中间表示。
2. **Visual bottleneck attention masking**: Stage 1 阻断 answer/prompt 对 image tokens 的直接访问，强制 image 信息经 latent tokens 传递。
3. **Two-stage training recipe**: 先用 bottleneck 训练 latent 承载视觉证据，再恢复 standard mask 让原始 image tokens 与 enriched latents 协同答题。
4. **Broad perception-heavy evaluation**: 覆盖 9 个视觉任务、3 个 LMM backbone，并包含 single-task、multi-task、Mirage 对比和机制消融。
5. **Mechanism evidence**: drop-latent、bottleneck-at-test、answer-to-latent attention、t-SNE 与 attention map 共同证明 latent 不是被动占位 token。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + LIVR 问题设定 |
| [01 - Introduction](sections/01-introduction.md) | text-centric LMM 瓶颈、显式监督局限、Figure 1 |
| [02 - Related Work](sections/02-related-work.md) | 文本 CoT、视觉中间表示、latent reasoning 对比 |
| [03 - Method](sections/03-method.md) | latent tokens、visual bottleneck、两阶段训练 |
| [04 - Evaluation](sections/04-evaluation.md) | 主结果、Mirage 对比、消融、可视化 |
| [05 - Conclusion](sections/05-conclusion.md) | 方法总结与定位 |
| [06 - Limitations and Future Work](sections/06-limitations-and-future-work.md) | 不可解释性与扩展方向 |
| [07 - Appendix](sections/07-appendix.md) | 数据集、prompt、训练设置、补充可视化、References |

## 关键数字

| 指标 | 数值 |
|------|------|
| Latent tokens | 默认 $K=16$ |
| Single-task training | 每任务 1k examples；Direct SFT 10 epochs；LIVR 4 epochs Stage 1 + 6 epochs Stage 2 |
| Multi-task training | 6 tasks × 1k examples；LIVR 2 epochs Stage 1 + 3 epochs Stage 2 |
| Qwen2.5-VL-3B mean | Direct SFT 61.61 → LIVR 67.85 (+6.24) |
| Qwen3-VL-4B mean | Direct SFT 74.12 → LIVR 77.55 (+3.43) |
| LLaVA-OneVision-1.5-4B mean | Direct SFT 63.70 → LIVR 69.30 (+5.60) |
| Multi-task Qwen3-VL mean | Direct SFT 69.60 → LIVR 72.37 (+2.77) |
| Mirage comparison | Jigsaw 48.60 → LIVR 68.00；VSP 46.00 → LIVR 66.00 |
| Latent attention | LIVR 0.076 vs latents-only 0.028 |
| Bottleneck test | LIVR 70.49 vs latents-only 43.44 |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Standard LMM input | image $I$ + prompt $Q$ | vision encoder $v$ + projector $p$ + language encoder $l$ | image embeddings $p(v(I))$ + text embeddings $l(Q)$ |
| 2. Latent injection | prompt $Q$ | append $K$ new tokens $L={l_1,...,l_K}$ after prompt | augmented prompt $Q'=Q+L$ |
| 3. Stage 1 bottleneck | image tokens + prompt + latent tokens + answer | block answer/prompt → image attention; answer can use prompt + latent | latent tokens forced to carry task-relevant visual information |
| 4. Stage 1 objective | answer tokens | NLL only on answer tokens | latent embeddings and LoRA learn from final QA loss |
| 5. Stage 2 standard mask | original image tokens + enriched latent tokens | restore standard attention mask, continue answer-token NLL | model learns joint use of image tokens and latent visual summaries |
| 6. Inference | image + prompt + inserted latents | forward pass with trained latent tokens | textual answer for VQA / multiple-choice task |

**整体输入**: 图像或图像组 + VQA prompt。
**整体输出**: answer token；中间 latent visual reasoning tokens 不作为文本输出。

## 优缺点与还能做什么

### 优点
- 不需要人工定义中间视觉监督，适合视觉相似、功能对应、风格、反射率这类难以写出标准中间标签的任务。
- 训练 recipe 很轻量：冻结 vision encoder/projector，只做 LoRA + latent embedding rows。
- 机制消融完整，证明提升来自 latent + bottleneck 的组合，而不是单纯更多 token 或更多视觉输入。
- 多 backbone 和多任务结果支持一定泛化性。

### 局限 / 风险
- Latent token 不可读，attention map 和 t-SNE 只能提供间接解释，不能保证推理忠实。
- 实验规模仍集中在 3B/4B 级开源 LMM，是否随模型规模保持收益需要继续验证。
- 训练数据虽无中间监督，但仍需要每任务 QA fine-tuning 数据；不是纯 zero-shot 推理增强。
- $K=16$ 是经验最优，任务难度变化时固定 K 可能不是最优。

### 还能做什么
- 做 dynamic K / adaptive latent budget，让复杂样本使用更多 visual reasoning tokens。
- 训练 latent probing 或 latent-to-text decoder，检查 latent 是否编码对象、区域、关系、风格等可解释因素。
- 做因果干预：mask/drop/replace 单个 latent，观察不同任务的答案变化。
- 扩展到视频、3D、医学影像等更依赖非语言视觉结构的任务。

## 阅读 Q&A 记录

- **Q: LIVR 和普通 Direct SFT 最大区别是什么?**
  A: Direct SFT 让答案直接从 image tokens 和 prompt 中学习；LIVR 在 Stage 1 阻断直接 image 路径，强迫视觉信息先压入 latent tokens，再在 Stage 2 与原图 token 协作。

- **Q: 为什么不能只加 latent tokens?**
  A: 4.4.1 显示 latents-only 去掉 latent 后 accuracy 不变，说明模型会忽略没有训练压力的额外 token；LIVR 去掉 latent 会明显掉分。

- **Q: 为什么要同时阻断 answer→vision 和 prompt→vision?**
  A: 只阻断 answer→vision 时，prompt token 仍可吸收 image 信息再传给 answer，latent 不是真瓶颈。Table 4(a) 显示默认 mask 更好。

- **Q: LIVR 的 latent 是视觉 token 还是语言 token?**
  A: 它们是新增特殊 token，嵌入在语言模型词表中，但通过 attention 和 bottleneck 训练承载视觉信息。t-SNE 显示很多 latent hidden states 靠近 image token 区域。

- **Q: 这篇论文最该复现哪几个实验?**
  A: 先复现 Qwen3-VL 上 Localization/Semantic Correspondence/Functional Correspondence 的 Table 3/4 消融，因为它们最直接验证 latent、mask、K 和训练 schedule 的机制。

## 📊 Citation Landscape

> Semantic Scholar `detail` endpoint 本次返回 `429 Too Many Requests`，因此 TLDR、被引次数和 influential citation count 暂缺；`references` 与 `recommendations` 已成功缓存到 `semantic-scholar/`。
> [Semantic Scholar 查询](https://www.semanticscholar.org/search?q=Latent%20Implicit%20Visual%20Reasoning) | [Connected Papers](https://www.connectedpapers.com/main/2512.21218)

### TLDR (Semantic Scholar 自动生成)

> 暂缺，Semantic Scholar detail endpoint 当前限流。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | detail 暂缺；references 缓存 34 条 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top references by citation count)

- **基础视觉/数据集**: SSIM (2004, 56261 cites), Microsoft COCO (2014, 52161), HPatches (2017, 856)。
- **VLM / 视觉语言基础**: CLIP (2021, 47757), Visual Instruction Tuning (2023, 9148)。
- **参数高效训练**: LoRA (2021, 18683), Prompt Tuning (2021, 5430)。
- **显式推理链背景**: Chain-of-Thought Prompting (2022, 17739), Tree of Thoughts (2023, 3854)。
- **推理时计算**: Scaling LLM Test-Time Compute Optimally (2024, 1661)。

### 🔗 推荐论文 (Semantic Scholar Recommendations)

1. Language-Guided Token Compression with Reinforcement Learning in Large Vision-Language Models (2026, 8 cites, arXiv:2603.13394)
2. VLMs Need Words: Vision Language Models Ignore Visual Detail In Favor of Semantic Anchors (2026, 2 cites, arXiv:2604.02486)
3. OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams (2026, 2 cites, arXiv:2603.12265)
4. Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models (2026, 1 cite, arXiv:2603.15618)
5. DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding (2026, 1 cite, arXiv:2604.12812)
6. Insight-V++: Towards Advanced Long-Chain Visual Reasoning with Multimodal Large Language Models (2026, 1 cite, arXiv:2603.18118)
7. WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation (2026, 1 cite, arXiv:2603.10703)
8. AD-Copilot: A Vision-Language Assistant for Industrial Anomaly Detection via Visual In-context Comparison (2026, 1 cite, arXiv:2603.13779)
9. Benefit from Seen: Enhancing Open-Vocabulary Object Detection by Bridging Visual and Textual Co-Occurrence Knowledge (ICCV 2025, 1 cite)
10. LanteRn: Latent Visual Structured Reasoning (2026, arXiv:2603.25629)

## BibTeX

```bibtex
@article{li2025latentimplicitvisualreasoning,
  title={Latent Implicit Visual Reasoning},
  author={Kelvin Li and Chuyi Shang and Leonid Karlinsky and Rogerio Feris and Trevor Darrell and Roei Herzig},
  journal={arXiv preprint arXiv:2512.21218},
  year={2025}
}
```
