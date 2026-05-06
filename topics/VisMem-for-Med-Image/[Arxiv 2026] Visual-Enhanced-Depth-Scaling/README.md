# Visual Enhanced Depth Scaling for Multimodal Latent Reasoning

**作者**: Yudong Han; Yong Wang; Zaiquan Yang; Zhen Qu; Liyuan Pan; Xiangxiang Chu
**会议**: Arxiv 2026
**链接**: [arXiv 2604.10500](https://arxiv.org/abs/2604.10500) | [PDF](https://arxiv.org/pdf/2604.10500v3)

## 一句话总结

Visual Enhanced Depth Scaling 把显式多步 CoT 压缩到少量连续 latent token 中，并用 SCF-VR 反复补回关键视觉证据、用 RDS 给高复杂 token 追加层内推理深度，从而缓解 latent reasoning 中的视觉欠优化和固定深度瓶颈。

## 核心贡献

1. 从 token 级梯度动力学诊断 latent reasoning 的两个瓶颈：视觉 token 梯度范数更高且波动更大，hard samples 在固定深度网络中长期保持 QKVO/LoRA 投影梯度震荡。
2. 提出 Spatially-Coherent Finer Visual Replay (SCF-VR)：用 causal self-attention 选择当前推理步骤关注的 visual patches，并通过空间连续 crop + 重编码 self-distillation 让 replay 的 visual latents 更细粒度、更少受 scattered attention 噪声影响。
3. 提出 Routing Depth Scaling (RDS)：在每层用轻量 router 给 token 打分，只对 top-$\alpha$ 关键 token 做额外 block refinement，在不改预训练 VLM 主体的前提下把计算预算分配给复杂视觉/latent token。
4. 设计 curriculum latent training：先学显式 CoT，再逐步把 reasoning step 封装成 `<latent>` token，避免直接监督中间 latent 或大量人工标注。
5. 在 12 个多模态推理基准、多个 Qwen/Qwen2.5/Chameleon backbone 上验证准确率、生成步数和延迟的 trade-off。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + 总览 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机 + 贡献 |
| [02 - Related Work](sections/02-related-work.md) | 相关工作谱系 |
| [03 - Method](sections/03-method.md) | 核心方法 |
| [04 - Experiments](sections/04-experiments.md) | 实验设置与结果 |
| [05 - Conclusion](sections/05-conclusion.md) | 总结与局限 |

## 关键数字

| 指标 | 数值 |
|------|------|
| arXiv | 2604.10500v3, updated 2026-04-26 |
| 训练数据 | OneThinker 子集约 30K |
| 评测范围 | 3 类任务 / 12 个 benchmark |
| backbone | Qwen2-VL-2B/7B, Qwen2.5-VL-3B/7B, Chameleon-7B |
| latent reasoning tokens | $T_r=4$ |
| router token budget | 默认 $\alpha=32$ |
| 最大额外 refinement depth | $D=1$ |
| 训练 | 16 epochs, Adam lr $4\times10^{-5}$, per-GPU batch 8, DeepSpeed ZeRO-2 |
| 主效率结论 | 相比显式 CoT 至少约 $10\times$ 减少自回归 reasoning steps |
| Qwen2-VL-7B 三基准结果 | M3CoT 73.0 / ScienceQA 95.9 / GQA 67.4 |
| Qwen2.5-VL-7B 消融收益 | Base → +RDS&SCF-VR: M3CoT 71.1 → 73.9 |
| 课题作用 | 视觉证据在 latent reasoning 中的保持、重放与选择性加深 |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. 多模态输入 | 图像 $\nu$ + 问题 $\mathcal Q$ | 文本 tokenizer 得到 $\mathbf Q$，视觉 encoder 得到 visual tokens $\mathbf V$，拼成 $\mathbf X=[\mathbf Q\parallel \mathbf V]$ | prefilling 阶段的初始 hidden states |
| 2. 隐式 latent reasoning | $\mathbf X$ + 已生成 latent states $\mathbf Z_{<t}$ | 自回归 MLLM 不输出文本 CoT，而取最后 token hidden state $\tau(\cdot)$ 作为连续 latent $\mathbf z_t$ | 少量 compact reasoning latents $\mathbf Z_{1:T_r}$ |
| 3. 注意力引导视觉重放 | 当前推理步所有 layer/head 的 causal attention | 聚合 attention map，取最近 token 到历史 token 的注意分布；只保留 visual-token 分数，排除 visited token 后选 top-$K$ patch | 当前步被关注的 visual latents $\mathbf B^{(t)}$ |
| 4. 空间一致自蒸馏 | top-$K$ visual patch 坐标 | 在 $G\times G$ grid 中找 attended tokens 最密集的 $W\times W$ sub-grid，映射回原图 crop，resize 后重新过视觉 encoder | 高保真 patch reference token $\mathbf u_{ref}$ |
| 5. TokenSR 对齐 | replay 得到的粗 visual latent pool $\mathbf b^{(t)}$ + $\mathbf u_{ref}$ | 轻量 $\mathcal F_{SR}$ 预测细粒度 reference，用 MSE/self-distillation 约束视觉 latent | 更稳定的视觉 grounding signal |
| 6. RDS 路由加深 | 每层 hidden representation $\mathbf H^{(l,t)}$ | layer-specific router $\mathcal F_{router}^{(l)}$ 给 token 打分，选择 top-$\alpha$ token 进行额外 block refinement，并保留原位置编码 | 复杂 token 被选择性深推理后的 hidden states |
| 7. Curriculum 训练 | 显式 CoT 标注 + answer tokens | 先完整生成 CoT，再逐步把一个 reasoning step 替换成 `<latent>`；总 loss 为 CE + $\lambda \mathcal L_{SCF}$ | 能把视觉证据压入 latent 的模型参数 |
| 8. 推理输出 | 图像 + 问题 | 生成少量 latent token 与 replay/refined hidden state，再进入显式 answer decoding | 最终答案 tokens；中间 CoT 不再长篇输出 |

**整体输入**: image-question pair。
**关键中间表示**: visual tokens、latent reasoning tokens、attention-selected visual latents、spatial crop reference、router-selected critical tokens。
**整体输出**: answer tokens；相比显式 CoT，推理链主要留在连续 hidden/latent space 内完成。

## 优缺点与还能做什么

### 优点
- 问题诊断比较具体：不是泛泛说“视觉 grounding 不足”，而是用 visual/text token 梯度范数差异、easy/hard sample 梯度核范数演化来支撑模块设计。
- SCF-VR 把视觉证据重新插回 latent reasoning 序列，适合解释 VisMem/medical-VLM 中“关键图像区域是否被后续推理继续使用”的问题。
- 空间一致 crop + 重编码 self-distillation 避免完全依赖 scattered attention top-k，能给视觉 latent 一个更连续的局部 reference。
- RDS 的计算预算是 token-wise 而不是全模型加深，和“只让难 token 多想一步”的动机一致，延迟成本更可控。
- Curriculum latent training 避免直接强迫 latent 对齐某个手工 CoT state，更接近把显式推理逐渐内化到连续表示中。
- 实验覆盖多 backbone、多 benchmark，并同时报告准确率、AR steps、平均耗时，能够支撑“准且快”的主 claim。

### 局限 / 风险
- 论文不是医学专用实验，虽然机制对医学视觉证据保持有启发，但没有在放射、病理或临床 VQA 上验证安全性。
- attention-selected patch 不等价于因果证据；attention sink 和错误关注仍可能把错误区域 replay 进 latent。
- $D=1$ 的 RDS 说明额外深度很保守；复杂长图、多图、多帧诊断场景是否需要更深或跨层记忆还未知。
- SCF-VR 训练时需要访问 attention map、crop、重编码和 TokenSR loss；推理虽声称关闭 visual play module 无额外开销，但训练实现复杂度不低。
- Table/Figure OCR 与正文中 $\lambda$ 最优值描述存在可疑不一致，复现时应以官方代码、表格或原 PDF 图为准。
- 实验主要是 benchmark accuracy，缺少对 hallucination 类型、错误定位、反事实区域遮挡、人类专家一致性的细粒度分析。

### 还能做什么
- 把 SCF-VR 的 replay 区域变成可审计 evidence trace：在医学 VQA 中输出所用 patch、置信度和不确定性。
- 用遮挡/反事实测试验证 replay patch 是否真是因果证据，而不只是 attention 可视化。
- 将 RDS 的 router 与任务风险联动：诊断高风险 token、模糊病灶、小目标区域可分配更高深度或触发显式复核。
- 引入外部视觉记忆库或病例级 longitudinal memory，让 replay 不只来自当前图片，还能来自历史检查或多序列 MRI/CT。
- 在临床任务上增加结构化评估：finding localization、报告一致性、病灶级召回、误诊风险，而不只看多选题准确率。

## 阅读 Q&A 记录

- **Q: 这篇和 VisMem/医学图像 topic 的关系是什么？**
  A: 它不是医学专用模型，但核心问题是 latent reasoning 中视觉证据会被语言/文本 token 压制或稀释。SCF-VR 和 RDS 都是在解决“关键视觉 token 如何被保留、重放、加深处理”，可迁移到医学 VLM 的视觉记忆设计。
- **Q: 为什么显式 CoT 会被替换成 latent reasoning？**
  A: 显式 CoT 需要生成大量文本 token，慢且可能越推理越脱离图像；latent reasoning 把中间步骤变成连续 hidden vectors，减少自回归解码，同时保留多步推理能力。
- **Q: 论文说 visual tokens 梯度更大，为什么反而叫 under-optimized？**
  A: 更大且更波动的梯度说明视觉特征在文本主导的 joint training 中难以稳定对齐，不是“学得更多”，而是反复被大幅修正，无法像文本 token 那样平滑收敛。
- **Q: SCF-VR 到底 replay 什么？**
  A: replay 的不是原始图像像素，也不是文本描述，而是 attention 选中的 visual token/patch 表示；再用空间连续 crop 重新编码得到 reference，约束 replay token 保持局部细节。
- **Q: RDS 和增加更多 latent token 有什么区别？**
  A: 更多 latent token 是沿时间/step 增加推理长度；RDS 是在 transformer layer 内只让关键 token 重复 refinement，属于 token-wise depth scaling，计算更集中。
- **Q: 推理时 visual replay module 还在吗？**
  A: 论文写明推理时 only LLM component is active，visual play/replay module disabled，因此训练阶段的 self-distillation 主要改变模型表示，不作为测试时额外工具链。
- **Q: 最值得复用的想法是什么？**
  A: 对医学 VLM 来说，是“用 attention 找关键区域 + 用空间一致重编码做自蒸馏 + 对难 token 加深推理”的组合；它比单纯把更多视觉 token 塞进 prompt 更有结构。

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2604.10500)

### TLDR (Semantic Scholar 自动生成)

> 暂缺。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献数 | 暂缺 |
| 被引次数 | 暂缺 |
| Influential Citations | 暂缺 |

### 参考文献分组 (Top 5 per category, by citations)

#### Latent Reasoning / Thinking
- **Qwen2.5-VL Technical Report** (2025) — 4397 citations [arXiv](https://arxiv.org/abs/2502.13923)
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025) — 2659 citations
- **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering** (2022) — 2223 citations [arXiv](https://arxiv.org/abs/2209.09513)
- **DAPO: An Open-Source LLM Reinforcement Learning System at Scale** (2025) — 1613 citations [arXiv](https://arxiv.org/abs/2503.14476)
- **MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts** (2023) — 1464 citations [arXiv](https://arxiv.org/abs/2310.02255)

#### Medical VLM / Medical Imaging
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (2023) — 1361 citations [arXiv](https://arxiv.org/abs/2305.13245)
- **Multimodal Chain-of-Thought Reasoning in Language Models** (2023) — 829 citations [arXiv](https://arxiv.org/abs/2302.00923)
- **Training Large Language Models to Reason in a Continuous Latent Space** (2024) — 468 citations [arXiv](https://arxiv.org/abs/2412.06769)
- **BLINK: Multimodal Large Language Models Can See but Not Perceive** (2024) — 433 citations [arXiv](https://arxiv.org/abs/2404.12390)
- **Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning** (2024) — 282 citations [arXiv](https://arxiv.org/abs/2403.16999)

#### Diffusion / Flow Foundations
- **GPT-4 Technical Report** (2023) — 24033 citations [arXiv](https://arxiv.org/abs/2303.08774)
- **Scaling Laws for Neural Language Models** (2020) — 7712 citations [arXiv](https://arxiv.org/abs/2001.08361)
- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** (2024) — 3922 citations [arXiv](https://arxiv.org/abs/2409.12191)
- **LLaVA-OneVision: Easy Visual Task Transfer** (2024) — 2345 citations [arXiv](https://arxiv.org/abs/2408.03326)
- **Chameleon: Mixed-Modal Early-Fusion Foundation Models** (2024) — 803 citations [arXiv](https://arxiv.org/abs/2405.09818)

#### Other Foundations
- **Advances** (1972) — 1047 citations
- **Transformer** (2020) — 472 citations
- **The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025** (2025) — 40 citations
- **Hallucination** (2020) — 36 citations
- **Point-wise convolutional neural network** (2017) — None citations

#### Memory / Agent Memory
- **LoRA: Low-Rank Adaptation of Large Language Models** (2021) — 18665 citations [arXiv](https://arxiv.org/abs/2106.09685)
- **Efficient Streaming Language Models with Attention Sinks** (2023) — 1680 citations [arXiv](https://arxiv.org/abs/2309.17453)
- **Compositional Chain-of-Thought Prompting for Large Multimodal Models** (2023) — 203 citations [arXiv](https://arxiv.org/abs/2311.17076)
- **MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs** (2025) — 119 citations [arXiv](https://arxiv.org/abs/2502.17422)
- **Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models** (2024) — 118 citations [arXiv](https://arxiv.org/abs/2408.15556)

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs](https://arxiv.org/abs/2604.07518) | 2026 | 0 |
| [PLUME: Latent Reasoning Based Universal Multimodal Embedding](https://arxiv.org/abs/2604.02073) | 2026 | 0 |
| [From Attenuation to Attention: Variational Information Flow Manipulation for Fine-Grained Visual Perception](https://arxiv.org/abs/2604.12508) | 2026 | 0 |
| [LanteRn: Latent Visual Structured Reasoning](https://arxiv.org/abs/2603.25629) | 2026 | 0 |
| [Hybrid Latent Reasoning with Decoupled Policy Optimization](https://arxiv.org/abs/2604.20328) | 2026 | 0 |
| [LatentGeo: Learnable Auxiliary Constructions in Latent Space for Multimodal Geometric Reasoning](https://arxiv.org/abs/2603.12166) | 2026 | 0 |
| [Visually-Guided Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2604.09349) | 2026 | 0 |
| [Bridging Visual Representation and Reinforcement Learning from Verifiable Rewards in Large Vision-Language Models](https://arxiv.org/abs/2603.27375) | 2026 | 0 |
| [Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought](https://arxiv.org/abs/2603.22847) | 2026 | 0 |
| [Firebolt-VL: Efficient Vision-Language Understanding with Cross-Modality Modulation](https://arxiv.org/abs/2604.04579) | 2026 | 0 |

## BibTeX

```bibtex
@article{han2026visualenhanceddepthscaling,
  title={ Visual Enhanced Depth Scaling for Multimodal Latent Reasoning },
  author={ Yudong Han and Yong Wang and Zaiquan Yang and Zhen Qu and Liyuan Pan and Xiangxiang Chu },
  journal={arXiv preprint arXiv:2604.10500},
  year={ 2026 }
}
```
