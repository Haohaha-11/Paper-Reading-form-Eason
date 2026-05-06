# VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models

**作者**: Xinlei Yu; Chengming Xu; Guibin Zhang; Zhangquan Chen; Yudong Zhang; Yongbo He; Peng-Tao Jiang; Jiangning Zhang; Xiaobin Hu; Shuicheng Yan
**会议**: Arxiv 2025
**链接**: [arXiv 2511.11007](https://arxiv.org/abs/2511.11007) | [PDF](https://arxiv.org/pdf/2511.11007v2)

## 一句话总结

VisMem 给 VLM 加一套可在自回归生成中按需插入的短期/长期 latent vision memory：短期记忆补当前图像的细粒度视觉证据，长期记忆补可迁移的视觉语义经验，从而缓解长链推理时“越生成越忘图”的 visual processing bottleneck。

更准确地说，它不是把图像重新画出来，也不是简单挑选已有视觉 token，而是在模型隐空间里新增可学习记忆查询、记忆生成器和调用 token。当模型生成 `<m_I^s>` 或 `<m_I^l>` 时，VisMem 根据当前视觉 token 与文本 hidden states 形成 query，再生成 latent memory tokens 插回上下文继续解码。

## 核心贡献

1. 把 VLM 的 visual processing bottleneck 具体化为长生成中的视觉证据衰减和视觉语义经验不足，而不是单纯归因于推理能力不够。
2. 借鉴认知心理学中的短期/长期记忆分工，设计视觉主导的 short-term latent memory 与语义主导的 long-term latent memory。
3. 用 4 个特殊 token 控制 memory invocation，让记忆可以在自回归序列中动态插入，且不需要重写 VLM 主干结构。
4. 用 query builder + 两个 LoRA memory former 完成 memory formation，并用两阶段 GRPO 先学“记忆内容是否有用”，再学“何时调用、调用哪类记忆”。
5. 在 12 个理解/推理/生成 benchmark、9 个 base model、跨域训练和连续学习设置中验证性能、泛化、遗忘缓解和推理效率。

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
| Base setting | Qwen2.5-VL-7B, 8x H200 141G |
| Benchmark | 12 个：5 understanding + 4 reasoning + 3 generation |
| Overall Avg. | Vanilla 54.5 → VisMem 65.5 |
| 平均提升 | +11.0% over vanilla |
| 分能力提升 | understanding +8.9%, reasoning +14.4%, generation +10.6% |
| Query / memory length | $K=8$, $N_s=8$, $N_l=16$ |
| Cross-domain | 仅用 Visual CoT + Mulberry 训练时，在 4 个未见 benchmark 上仍有 +6.9% 到 +20.2% |
| Continual learning | 四阶段后 MMVet 保持 72.1%，高于 DeepEyes 68.4% 与 Mirage 67.0% |
| 推理开销 | 相对 vanilla 增加约 8.2% 到 43.8%，显著低于多数 image-level 方法 |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 关键变换 | 输出 |
|------|------|----------|------|
| 1. 原始 VLM 解码 | instruction-vision pair $(I,V)$，当前环境状态 $s_t$，已生成 token | base VLM $\mathcal{P}$ 按 token 自回归生成 rationale/answer | 普通文本 token 序列与每步 hidden states |
| 2. 调用信号 | 当前生成上下文 | 扩展词表加入 `<m_I^s>`, `<m_E^s>`, `<m_I^l>`, `<m_E^l>`；模型生成 invocation token 表示要插入短期或长期记忆 | 明确的 memory invocation 位置与类型 |
| 3. 多模态认知状态 | 视觉 encoder tokens $\{v_i\}$ + 最近一段输出 hidden states $\{h_i\}$ | 拼成 $\mathbf{H}=[v_1,\ldots,v_y,h_1,\ldots,h_z]$，表示当前“看过什么、已经说到哪里” | query builder 的上下文输入 |
| 4. Query builder | $\mathbf{H}$ + learnable $\mathbf{Q}_{init}$ | 轻量 transformer encoder $\mathbf{B}$ 通过 masked attention 只允许 query 读 $\mathbf{H}$，取最后 $K$ 个向量 | memory query $\mathbf{Q}\in\mathbb{R}^{K\times d}$ |
| 5. Short-term formation | $\mathbf{Q}$ + learnable memory tokens + 视觉侧目标序列 | short-term memory former $\mathcal{F}_s$ 生成细粒度视觉证据；短期分支再接入视觉 token stream 并经原 projector 对齐到语言空间 | $\mathbf{M}_s\in\mathbb{R}^{N_s\times d}$ |
| 6. Long-term formation | $\mathbf{Q}$ + learnable memory tokens + 语言侧目标序列 | long-term memory former $\mathcal{F}_l$ 生成抽象视觉语义/经验，强调可迁移知识而非当前像素细节 | $\mathbf{M}_l\in\mathbb{R}^{N_l\times d}$ |
| 7. 记忆插入 | 已输出 invocation token + $\mathbf{M}_{s/l}$ | 将 latent memory tokens 插在 invocation token 后，再自动追加对应 end token | 带 latent memory 的扩展解码上下文 |
| 8. 继续生成 | 扩展上下文 + 原 VLM | 继续 token-by-token decoding，记忆 token 作为连续隐空间上下文参与后续预测 | 更稳的视觉理解、推理或生成答案 |
| 9. 训练闭环 | 轨迹集合 $\mathcal{T}$ 与可量化 score $S(\tau)$ | Stage I 冻结 policy、优化 $\mathbf{B},\mathcal{F}_s,\mathcal{F}_l$；Stage II 冻结 memory formation、优化 policy 的调用策略，并加入类型错误/负收益惩罚 | 既能生成有用记忆、又能少而准地调用记忆的 VisMem |

**整体输入**: 图像/多图 + 指令 + base VLM。
**整体中间表示**: 调用 token、视觉/文本 hidden state、memory query、短期/长期 latent memory tokens。
**整体输出**: 插入隐空间视觉记忆后的 VLM 生成结果。

## 优缺点与还能做什么

### 优点
- 介入层次选得很巧：不重绘图像、不依赖外部视觉工具、不大幅改主干参数，而是在 latent stream 里补视觉证据。
- 短期/长期记忆分工清楚：短期偏 counting、grounding、retrieval 等细粒度视觉任务，长期偏 inductive/deductive、graph/table 等视觉语义推理。
- 调用机制是动态的，模型可以决定何时调用、调用哪类记忆；消融显示随机/全量调用会带来效率下降甚至性能退化。
- 训练分两阶段，先让 memory former 学会产出有正收益的内容，再让 policy 学会低冗余调用，符合模块职责。
- 实验覆盖面较广：12 个 benchmark、15 类对比方法、9 个 base model、跨域训练、连续学习、效率和长度敏感性。

### 局限 / 风险
- 论文评估是通用 VLM benchmark，并不是医学影像专门评测；放到医学场景时，诊断安全性、病灶尺度、跨设备域偏移还需要重新验证。
- “长期记忆”主要是参数化 latent memory former 生成的语义表示，不是可审计的外部病例库或显式知识库，解释性有限。
- 调用 token 学得是否稳定依赖 GRPO reward/score 设计；如果任务 score 噪声大，可能学到错误的调用频率或记忆类型偏好。
- 推理开销虽然低于 image-level 方法，但 latent memory token 会增加上下文长度，长视频、多图长报告等场景还要看内存和 latency。
- 论文强调不损伤主干能力，但仍会更新部分 policy 参数；对高风险部署，需要更细的 forgetting、hallucination 和 calibration 测试。

### 还能做什么
- 在医学影像上做器官/病灶级细分：让短期记忆显式服务小病灶定位、计数、测量，长期记忆服务诊断模式和报告语义。
- 加入可解释的 memory probing：把 latent memory 与原图区域、OCR/结构元素或临床概念对齐，验证记忆到底补了什么。
- 研究预算自适应：按任务难度、图像数量、输出长度动态限制调用次数和 $N_s/N_l$，降低长链推理成本。
- 与检索式医学知识库结合：长期 latent memory 负责上下文压缩，外部知识库负责可追溯证据，避免黑箱语义记忆过度自信。
- 扩展到持续学习的真实流式场景，测试新任务顺序、类别不均衡和分布漂移下记忆模块是否仍能缓解遗忘。

## 阅读 Q&A 记录

- **Q: VisMem 的“memory”是不是一个外部数据库？**
  A: 不是。它更像可学习的隐空间记忆生成器：根据当前 hidden states 生成 latent memory tokens，再插回解码上下文。
- **Q: 短期记忆和长期记忆的差别在哪里？**
  A: 短期记忆接近当前视觉证据，适合 counting、grounding、retrieval；长期记忆偏抽象语义经验，适合数学/逻辑/图表推理。
- **Q: 为什么不直接让 VLM 多看一次图？**
  A: 图像级重处理通常需要外部工具或重生成视觉输入，延迟高。VisMem 在 latent space 插入连续记忆 token，成本更低且能保持原解码流程。
- **Q: 为什么需要 invocation token？**
  A: 它把“何时需要记忆”和“记忆内容是什么”解耦。模型先决定调用位置与类型，再由 query builder 和 memory former 生成具体 latent memory。
- **Q: 为什么两阶段训练？**
  A: 如果同时学内容和调用，policy 可能频繁调用无效记忆。Stage I 先优化记忆形成的正收益，Stage II 再训练少而准的调用策略。
- **Q: 全量调用记忆会不会最好？**
  A: 不会。消融里随机/全量调用会显著增加推理时间，并且 100% 调用在 MuirBench、MV-Math、MultiTrust 上反而低于动态 VisMem。
- **Q: 这篇对医学影像有什么可复用点？**
  A: 可复用的是“短期保局部视觉证据，长期保跨病例语义经验”的框架；但医学影像需要额外做病灶级可解释性和安全验证。

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
