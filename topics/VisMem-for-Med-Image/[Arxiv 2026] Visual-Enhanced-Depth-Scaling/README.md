# Visual Enhanced Depth Scaling for Multimodal Latent Reasoning

**作者**: Yudong Han; Yong Wang; Zaiquan Yang; Zhen Qu; Liyuan Pan; Xiangxiang Chu  
**会议/年份**: Arxiv 2026  
**链接**: [arXiv](https://arxiv.org/abs/2604.10500) | [PDF](https://arxiv.org/pdf/2604.10500v3)  
**主题**: VisMem for Med Image  
**阅读日期**: 2026-05-05

## 一句话总结

这篇论文针对 multimodal latent reasoning 中视觉 token 梯度不稳定和语言偏置问题，提出视觉增强的深度扩展机制来改善隐式推理。

## Why Read This Paper

- 它连接 VisMem 和 latent reasoning。
- 它提供分析 VLM 视觉瓶颈的训练动态视角。
- 对医学影像很有启发：视觉证据微弱、局部且高风险，不能被语言先验覆盖。

## 核心贡献

1. 分析 latent training 中 visual tokens 梯度更高且更波动。
2. 提出 Spatially-Coherent Finer Visual Replay。
3. 提出 Routing Depth Scaling，根据样本/模态需求自适应扩展 latent reasoning 深度。
4. 强调隐式 feature propagation 可降低显式 CoT 解码延迟。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [Abstract](sections/00-abstract.md) | 题名、作者、摘要、目录或论文总体定位。 |
| [Introduction](sections/01-introduction.md) | 动机、问题定义、贡献与相关工作定位。 |
| [Method](sections/02-method.md) | 核心方法、latent/memory/alignment 机制、理论背景和训练目标。 |
| [Experiments](sections/03-experiments.md) | 实验设置、主结果、消融、鲁棒性、效率和案例分析。 |
| [Discussion](sections/04-discussion.md) | 结论、未来工作、附录、实现细节、失败分析与参考文献。 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Core mechanism | Spatially-Coherent Finer Visual Replay + Routing Depth Scaling |
| Target problem | visual token instability and language bias |
| Inference style | implicit latent reasoning |
| Topic role | visual evidence preservation during latent reasoning |

## Reusable Ideas

- **Problem framing**: VLM 的 latent reasoning 可能被语言 token 主导。
- **Method design**: 用视觉重放和动态深度路由增强视觉 token 的持续影响。
- **Experiment design**: 分模态/分 token 观察梯度、注意力或 latent state。
- **Writing pattern**: 先给 training dynamics 观察，再提出结构改造。

## Open Questions

- 动态深度扩展在长医学报告中是否会带来不可控延迟？
- 视觉重放会不会强化伪影或非病灶区域的错误证据？

## 📊 Citation Landscape

- **Semantic Scholar**: API detail 暂时被限流或未返回；本地缓存见 `semantic-scholar/detail.json`。
- **Connected Papers**: https://www.connectedpapers.com/main/2604.10500
- **TLDR**: 暂缺。
- **引用统计**: 暂缺。

### 参考文献分组

#### Latent Reasoning / Thinking
- Qwen2.5-VL Technical Report (arXiv.org, 2025) — citations: 4397 — https://arxiv.org/abs/2502.13923
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv.org, 2025) — citations: 2659
- Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (Neural Information Processing Systems, 2022) — citations: 2223 — https://arxiv.org/abs/2209.09513
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale (arXiv.org, 2025) — citations: 1613 — https://arxiv.org/abs/2503.14476
- MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts (International Conference on Learning Representations, 2023) — citations: 1464 — https://arxiv.org/abs/2310.02255

#### Medical VLM / Medical Imaging
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Conference on Empirical Methods in Natural Language Processing, 2023) — citations: 1361 — https://arxiv.org/abs/2305.13245
- Multimodal Chain-of-Thought Reasoning in Language Models (Trans. Mach. Learn. Res., 2023) — citations: 829 — https://arxiv.org/abs/2302.00923
- Training Large Language Models to Reason in a Continuous Latent Space (arXiv.org, 2024) — citations: 468 — https://arxiv.org/abs/2412.06769
- BLINK: Multimodal Large Language Models Can See but Not Perceive (European Conference on Computer Vision, 2024) — citations: 433 — https://arxiv.org/abs/2404.12390
- Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning (Neural Information Processing Systems, 2024) — citations: 282 — https://arxiv.org/abs/2403.16999

#### Other Foundations
- Scaling Laws for Neural Language Models (arXiv.org, 2020) — citations: 7712 — https://arxiv.org/abs/2001.08361
- Advances (British Journal of Psychiatry, 1972) — citations: 1047
- Transformer (Shipboard Electrical Power Systems, 2020) — citations: 472
- Visual Hallucinations of Multi-modal Large Language Models (Annual Meeting of the Association for Computational Linguistics, 2024) — citations: 79 — https://arxiv.org/abs/2402.14683
- The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025 (International Conference on Learning Representations, 2025) — citations: 40

#### Vision-Language Alignment
- GPT-4 Technical Report (N/A, 2023) — citations: 24033 — https://arxiv.org/abs/2303.08774
- Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (arXiv.org, 2024) — citations: 3922 — https://arxiv.org/abs/2409.12191
- LLaVA-OneVision: Easy Visual Task Transfer (Trans. Mach. Learn. Res., 2024) — citations: 2345 — https://arxiv.org/abs/2408.03326
- Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv.org, 2024) — citations: 803 — https://arxiv.org/abs/2405.09818
- Are We on the Right Way for Evaluating Large Vision-Language Models? (Neural Information Processing Systems, 2024) — citations: 746 — https://arxiv.org/abs/2403.20330

#### Memory / Continual Context
- LoRA: Low-Rank Adaptation of Large Language Models (International Conference on Learning Representations, 2021) — citations: 18665 — https://arxiv.org/abs/2106.09685
- Efficient Streaming Language Models with Attention Sinks (International Conference on Learning Representations, 2023) — citations: 1680 — https://arxiv.org/abs/2309.17453
- Compositional Chain-of-Thought Prompting for Large Multimodal Models (Computer Vision and Pattern Recognition, 2023) — citations: 203 — https://arxiv.org/abs/2311.17076
- MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs (International Conference on Learning Representations, 2025) — citations: 119 — https://arxiv.org/abs/2502.17422
- Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models (arXiv.org, 2024) — citations: 118 — https://arxiv.org/abs/2408.15556

### 推荐论文

- Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.07518
- PLUME: Latent Reasoning Based Universal Multimodal Embedding (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.02073
- From Attenuation to Attention: Variational Information Flow Manipulation for Fine-Grained Visual Perception (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.12508
- LanteRn: Latent Visual Structured Reasoning (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.25629
- Hybrid Latent Reasoning with Decoupled Policy Optimization (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.20328
- LatentGeo: Learnable Auxiliary Constructions in Latent Space for Multimodal Geometric Reasoning (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.12166
- Visually-Guided Policy Optimization for Multimodal Reasoning (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.09349
- Bridging Visual Representation and Reinforcement Learning from Verifiable Rewards in Large Vision-Language Models (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.27375
- Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.22847
- Firebolt-VL: Efficient Vision-Language Understanding with Cross-Modality Modulation (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.04579
