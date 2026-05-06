# Visual Enhanced Depth Scaling for Multimodal Latent Reasoning

**作者**: Yudong Han; Yong Wang; Zaiquan Yang; Zhen Qu; Liyuan Pan; Xiangxiang Chu  
**会议**: Arxiv 2026  
**链接**: [arXiv 2604.10500](https://arxiv.org/abs/2604.10500) | [PDF](https://arxiv.org/pdf/2604.10500v3)

## 一句话总结

Visual Enhanced Depth Scaling 针对 multimodal latent reasoning 中视觉 token 梯度不稳定与语言偏置，提出视觉重放和动态深度扩展。

## 核心贡献

1. 分析 visual token 在 latent training 中更高且更波动的梯度。
2. 提出 Spatially-Coherent Finer Visual Replay 保持视觉证据。
3. 提出 Routing Depth Scaling 自适应扩展 latent reasoning 深度。

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
| 机制 | Visual Replay + Routing Depth Scaling |
| 问题 | visual token instability |
| 推理形态 | implicit latent reasoning |
| 课题作用 | 视觉证据保持 |

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
