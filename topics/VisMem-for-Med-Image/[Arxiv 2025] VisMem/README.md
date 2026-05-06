# VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models

**作者**: Xinlei Yu; Chengming Xu; Guibin Zhang; Zhangquan Chen; Yudong Zhang; Yongbo He; Peng-Tao Jiang; Jiangning Zhang; Xiaobin Hu; Shuicheng Yan  
**会议**: Arxiv 2025  
**链接**: [arXiv 2511.11007](https://arxiv.org/abs/2511.11007) | [PDF](https://arxiv.org/pdf/2511.11007v2)

## 一句话总结

VisMem 提出 latent vision memory，通过 memory invocation 和 formation 缓解 VLM 长生成中的视觉 grounding 丢失。

## 核心贡献

1. 把 VLM visual processing bottleneck 解释为视觉证据遗忘。
2. 借鉴短期视觉主导记忆与长期语义主导记忆。
3. 设计 memory invocation 和 memory formation，并系统评估泛化、遗忘和效率。

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
| 机制 | latent vision memory |
| 阶段 | memory invocation + formation |
| 瓶颈 | loss of visual grounding |
| 课题作用 | 通用 VisMem 基线 |

## 📊 Citation Landscape

> 数据来源: Semantic Scholar detail 暂缺或限流 | [Connected Papers](https://www.connectedpapers.com/main/2511.11007)

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
- **Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities** (2025) — 2741 citations [arXiv](https://arxiv.org/abs/2507.06261)
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025) — 2659 citations
- **MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts** (2023) — 1464 citations [arXiv](https://arxiv.org/abs/2310.02255)
- **MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities** (2023) — 1188 citations [arXiv](https://arxiv.org/abs/2308.02490)

#### Medical VLM / Medical Imaging
- **MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI** (2023) — 2023 citations [arXiv](https://arxiv.org/abs/2311.16502)
- **ExpeL: LLM Agents Are Experiential Learners** (2023) — 477 citations [arXiv](https://arxiv.org/abs/2308.10144)
- **Training Large Language Models to Reason in a Continuous Latent Space** (2024) — 468 citations [arXiv](https://arxiv.org/abs/2412.06769)
- **BLINK: Multimodal Large Language Models Can See but Not Perceive** (2024) — 433 citations [arXiv](https://arxiv.org/abs/2404.12390)
- **MemoryBank: Enhancing Large Language Models with Long-Term Memory** (2023) — 388 citations [arXiv](https://arxiv.org/abs/2305.10250)

#### Memory / Agent Memory
- **LoRA: Low-Rank Adaptation of Large Language Models** (2021) — 18665 citations [arXiv](https://arxiv.org/abs/2106.09685)
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (2024) — 5773 citations [arXiv](https://arxiv.org/abs/2402.03300)
- **Working Memory: Theories, Models, and Controversies** (2015) — 3024 citations
- **Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters** (2024) — 229 citations [arXiv](https://arxiv.org/abs/2403.11549)
- **Short-Term Memory and Long-Term Memory are Still Different** (2017) — 216 citations

#### Diffusion / Flow Foundations
- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** (2024) — 3923 citations [arXiv](https://arxiv.org/abs/2409.12191)
- **GPT-4o System Card** (2024) — 3903 citations [arXiv](https://arxiv.org/abs/2410.21276)
- **Language Models** (2009) — 1062 citations
- **Chameleon: Mixed-Modal Early-Fusion Foundation Models** (2024) — 803 citations [arXiv](https://arxiv.org/abs/2405.09818)
- **Are We on the Right Way for Evaluating Large Vision-Language Models?** (2024) — 746 citations [arXiv](https://arxiv.org/abs/2403.20330)

#### Other Foundations
- **Decoupled Weight Decay Regularization** (2017) — 33254 citations
- **OpenAI** (None) — None citations

#### Document Understanding
- **Visual document understanding and question answering: A multi-agent collaboration framework with test-time scaling** (None) — None citations

#### Vision-Language Alignment
- **Seeing clearly, answering incorrectly: A multimodal robustness benchmark for evaluating mllms on leading questions** (None) — None citations

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs](https://arxiv.org/abs/2605.00814) | 2026 | 0 |
| [Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs](https://arxiv.org/abs/2604.07518) | 2026 | 0 |
| [VLA-Thinker: Boosting Vision-Language-Action Models through Thinking-with-Image Reasoning](https://arxiv.org/abs/2603.14523) | 2026 | 1 |
| [LanteRn: Latent Visual Structured Reasoning](https://arxiv.org/abs/2603.25629) | 2026 | 0 |
| [CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](https://arxiv.org/abs/2603.21077) | 2026 | 0 |
| [Q-Tacit: Image Quality Assessment via Latent Visual Reasoning](https://arxiv.org/abs/2603.22641) | 2026 | 0 |
| [POINTS-Long: Adaptive Dual-Mode Visual Reasoning in MLLMs](https://arxiv.org/abs/2604.11627) | 2026 | 0 |
| [MMCORE: MultiModal COnnection with Representation Aligned Latent Embeddings](https://arxiv.org/abs/2604.19902) | 2026 | 0 |
| [LatentGeo: Learnable Auxiliary Constructions in Latent Space for Multimodal Geometric Reasoning](https://arxiv.org/abs/2603.12166) | 2026 | 0 |
| [Visually-Guided Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2604.09349) | 2026 | 0 |

## BibTeX

```bibtex
@article{yu2025vismem,
  title={ VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models },
  author={ Xinlei Yu and Chengming Xu and Guibin Zhang and Zhangquan Chen and Yudong Zhang and Yongbo He and Peng-Tao Jiang and Jiangning Zhang and Xiaobin Hu and Shuicheng Yan },
  journal={arXiv preprint arXiv:2511.11007},
  year={ 2025 }
}
```
