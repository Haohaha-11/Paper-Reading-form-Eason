# VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models

**作者**: Xinlei Yu; Chengming Xu; Guibin Zhang; Zhangquan Chen; Yudong Zhang; Yongbo He; Peng-Tao Jiang; Jiangning Zhang; Xiaobin Hu; Shuicheng Yan  
**会议/年份**: Arxiv 2025  
**链接**: [arXiv](https://arxiv.org/abs/2511.11007) | [PDF](https://arxiv.org/pdf/2511.11007v2)  
**主题**: VisMem for Med Image  
**阅读日期**: 2026-05-05

## 一句话总结

VisMem 提出 latent vision memory，通过 memory invocation 和 memory formation 缓解 VLM 长生成中的视觉证据丢失与缺少上下文化视觉经验问题。

## Why Read This Paper

- 这是本课题的命名核心。
- 它直接回答 VLM 为什么会在长回答中逐渐离开视觉证据。
- 医学影像问答常需要持续引用病灶证据，VisMem 机制很相关。

## 核心贡献

1. 把 visual processing bottleneck 解释为 grounding 丢失和 contextualized visual experience 不足。
2. 借鉴短期视觉主导记忆与长期语义主导记忆，提出 latent vision memory。
3. 设计 memory invocation 和 memory formation。
4. 系统分析视觉能力、跨域泛化、灾难性遗忘、base model 和推理效率。

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
| Core mechanism | latent vision memory |
| Main stages | memory invocation + memory formation |
| Target bottleneck | loss of visual grounding during prolonged generation |
| Topic role | general VisMem baseline for medical adaptation |

## Reusable Ideas

- **Problem framing**: VLM 的错误可能是在长生成中忘记或弱化视觉证据。
- **Method design**: 显式建模视觉记忆调用与形成，而不是单纯增加视觉 token。
- **Experiment design**: 要测跨域泛化、遗忘缓解和推理效率。
- **Writing pattern**: 用认知记忆理论组织方法模块。

## Open Questions

- latent vision memory 中存储的是可定位视觉证据，还是不可解释的经验向量？
- 在医学场景中，错误记忆形成是否会带来更高风险 hallucination？

## 📊 Citation Landscape

- **Semantic Scholar**: API detail 暂时被限流或未返回；本地缓存见 `semantic-scholar/detail.json`。
- **Connected Papers**: https://www.connectedpapers.com/main/2511.11007
- **TLDR**: 暂缺。
- **引用统计**: 暂缺。

### 参考文献分组

#### Latent Reasoning / Thinking
- Qwen2.5-VL Technical Report (arXiv.org, 2025) — citations: 4397 — https://arxiv.org/abs/2502.13923
- Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities (arXiv.org, 2025) — citations: 2741 — https://arxiv.org/abs/2507.06261
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv.org, 2025) — citations: 2659
- MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts (International Conference on Learning Representations, 2023) — citations: 1464 — https://arxiv.org/abs/2310.02255
- MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities (International Conference on Machine Learning, 2023) — citations: 1188 — https://arxiv.org/abs/2308.02490

#### Medical VLM / Medical Imaging
- MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI (Computer Vision and Pattern Recognition, 2023) — citations: 2023 — https://arxiv.org/abs/2311.16502
- ExpeL: LLM Agents Are Experiential Learners (AAAI Conference on Artificial Intelligence, 2023) — citations: 477 — https://arxiv.org/abs/2308.10144
- Training Large Language Models to Reason in a Continuous Latent Space (arXiv.org, 2024) — citations: 468 — https://arxiv.org/abs/2412.06769
- BLINK: Multimodal Large Language Models Can See but Not Perceive (European Conference on Computer Vision, 2024) — citations: 433 — https://arxiv.org/abs/2404.12390
- MemoryBank: Enhancing Large Language Models with Long-Term Memory (AAAI Conference on Artificial Intelligence, 2023) — citations: 388 — https://arxiv.org/abs/2305.10250

#### Memory / Continual Context
- LoRA: Low-Rank Adaptation of Large Language Models (International Conference on Learning Representations, 2021) — citations: 18665 — https://arxiv.org/abs/2106.09685
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (arXiv.org, 2024) — citations: 5773 — https://arxiv.org/abs/2402.03300
- Working Memory: Theories, Models, and Controversies (N/A, 2015) — citations: 3024
- Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters (Computer Vision and Pattern Recognition, 2024) — citations: 229 — https://arxiv.org/abs/2403.11549
- Short-Term Memory and Long-Term Memory are Still Different (Psychological bulletin, 2017) — citations: 216

#### Vision-Language Alignment
- Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (arXiv.org, 2024) — citations: 3923 — https://arxiv.org/abs/2409.12191
- Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv.org, 2024) — citations: 803 — https://arxiv.org/abs/2405.09818
- Are We on the Right Way for Evaluating Large Vision-Language Models? (Neural Information Processing Systems, 2024) — citations: 746 — https://arxiv.org/abs/2403.20330
- Hallusionbench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models (Computer Vision and Pattern Recognition, 2023) — citations: 496 — https://arxiv.org/abs/2310.14566
- MuirBench: A Comprehensive Benchmark for Robust Multi-image Understanding (International Conference on Learning Representations, 2024) — citations: 144 — https://arxiv.org/abs/2406.09411

#### Other Foundations
- Decoupled Weight Decay Regularization (International Conference on Learning Representations, 2017) — citations: 33254
- GPT-4o System Card (N/A, 2024) — citations: 3903 — https://arxiv.org/abs/2410.21276
- Language Models (Encyclopedia of Database Systems, 2009) — citations: 1062
- OpenAI (Gpt, None) — citations: None

#### Document Understanding
- Visual document understanding and question answering: A multi-agent collaboration framework with test-time scaling (arXiv preprint, None) — citations: None

### 推荐论文

- Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2605.00814
- Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.07518
- VLA-Thinker: Boosting Vision-Language-Action Models through Thinking-with-Image Reasoning (N/A, 2026) — citations: 1 — https://arxiv.org/abs/2603.14523
- LanteRn: Latent Visual Structured Reasoning (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.25629
- CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.21077
- Q-Tacit: Image Quality Assessment via Latent Visual Reasoning (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.22641
- POINTS-Long: Adaptive Dual-Mode Visual Reasoning in MLLMs (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.11627
- MMCORE: MultiModal COnnection with Representation Aligned Latent Embeddings (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.19902
- LatentGeo: Learnable Auxiliary Constructions in Latent Space for Multimodal Geometric Reasoning (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2603.12166
- Visually-Guided Policy Optimization for Multimodal Reasoning (N/A, 2026) — citations: 0 — https://arxiv.org/abs/2604.09349
