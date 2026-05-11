# Test-Time Learning for Large Language Models

**作者**: Jinwu Hu; Zhitian Zhang; Guohao Chen; Xutao Wen; Chao Shuai; Wei Luo; Bin Xiao; Yuanqing Li; Mingkui Tan  
**会议**: ICML 2025  
**链接**: [arXiv 2505.20633](https://arxiv.org/abs/2505.20633) | [PDF](https://arxiv.org/pdf/2505.20633)  
**阅读顺序**: TTT in LLM topic paper 3  
**MinerU task**: `232d8220-5565-422b-b4c7-ca42040b6591`

## 一句话总结

TLM 把 LLM 的 test-time learning 表述为只使用无标签测试输入的 input perplexity minimization，并用高困惑度样本加权与 LoRA 更新，在 AdaptEval 上实现面向分布偏移的轻量域适应。

## 核心贡献

1. **TTL for LLMs 问题设定**: 将部署期 domain shift / linguistic diversity 下的 LLM 更新定义为只依赖 unlabeled test data 的自监督测试时学习。
2. **Input perplexity minimization**: 用输入 token 的 perplexity 作为无标签代理目标，并用经验与理论说明降低输入困惑度可帮助降低输出困惑度。
3. **High-perplexity sample efficient learning**: 观察高困惑度样本对更新贡献更大，因此用 perplexity-based score 选择并强调这些测试样本，减少低价值反向传播。
4. **LoRA-based stable update**: 只更新低秩 adapter，避免全参 test-time update 带来的高开销和 catastrophic forgetting 风险。
5. **AdaptEval benchmark 与证据**: 构建 DomainBench / InstructionBench / ReasoningBench，并报告在 domain knowledge adaptation 上相对原始 LLM 至少 $20\%$ 的性能提升证据。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要 + TLM 三件套 |
| [01 - Introduction](sections/01-introduction.md) | 问题动机、方法概览、贡献 |
| [02 - Related Work](sections/02-related-work.md) | Fine-tuning/RAG/TTA/TTT 对比 |
| [03 - Problem Formulation](sections/03-problem-formulation.md) | OOD 测试分布与 TTL 优化目标 |
| [04 - Test-Time Learning for LLMs](sections/04-test-time-learning-for-llms.md) | perplexity loss、高困惑度样本、LoRA 更新 |
| [05 - Experiments](sections/05-experiments.md) | AdaptEval、主结果、消融与讨论 |
| [06 - Conclusion](sections/06-conclusion.md) | 结论与边界 |
| [07 - Back Matter](sections/07-back-matter.md) | 致谢、影响声明、参考文献 |
| [08 - Supplementary Materials](sections/08-supplementary-materials.md) | 补充材料总览 |
| [09 - Appendix A](sections/09-supplementary-a-more-related-work.md) | 更多相关工作 |
| [10 - Appendix B](sections/10-supplementary-b-adapteval-benchmark.md) | AdaptEval 数据集构造 |
| [11 - Appendix C](sections/11-supplementary-c-experiment-settings.md) | 指标与实现细节 |
| [12 - Appendix D](sections/12-supplementary-d-more-results.md) | 更多实验结果 |
| [13 - Appendix E](sections/13-supplementary-e-discussions-and-future-works.md) | 讨论与未来工作 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 测试时监督信号 | 仅 unlabeled test samples $x \sim Q(x)$ |
| 更新参数 | LoRA adapter，而非 full-parameter update |
| 样本策略 | perplexity-based weighting，强调 high-perplexity samples |
| Benchmark | AdaptEval = DomainBench + InstructionBench + ReasoningBench |
| DomainBench domains | Geography, Agriculture, Medicine, Finance |
| InstructionBench datasets | Alpaca-GPT4, Dolly, InstructionWild |
| ReasoningBench datasets | GSM8K, SVAMP, ASDiv, StrategyQA |
| 评测模型 | Llama3.2-3B-Instruct, Llama3-8B-Instruct, Llama2-13B-chat, Qwen2.5-7B-Instruct |
| 主张性提升 | domain knowledge adaptation 上 compared to original LLMs 至少 $20\%$ |
| 额外结果示例 | Agriculture 上 Qwen2.5 BLEU 相对提升 $146.61\%$；Finance 上 Llama3.2 BLEU 相对提升 $122.50\%$ |
| 主要风险 | online update 中累积偏差、catastrophic forgetting、部署端反向传播开销 |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. 测试域数据到达 | 无标签测试样本 $\mathcal{D}_{Test}=\{x_j\}$ | 按 batch 或 online stream 送入原始 LLM | 当前测试 batch $\mathcal{X}$ |
| 2. 初始化可训练状态 | 已训练 LLM 参数 $\Theta$ | 插入 LoRA 增量 $\Delta\Theta$，形成 $\tilde{\Theta}=\Theta+\Delta\Theta$ | 只含 LoRA 的可更新模型 |
| 3. 自监督目标计算 | 输入 token 序列 $x$ | 计算 input perplexity / next-token NLL | 每个样本的困惑度与 loss |
| 4. 样本效率策略 | batch 内 perplexity 分布 | 用 score $S(x)$ 强调 high-perplexity samples | 加权后的测试时学习信号 |
| 5. 参数更新 | 加权 perplexity loss | 反向传播更新 LoRA 参数 | 适配当前测试域的 LLM |
| 6. 推理输出 | 同一批/后续测试输入 | 用更新后模型生成答案 | domain-adapted response $\tilde{y}$ |

**整体输入**: 无标签测试域文本/指令/问题。  
**整体输出**: 经过测试时 LoRA 自适应后的 LLM 回答；中间状态是 LoRA 参数和样本 perplexity，不需要人工标签。

## 优点与局限

### 优点

- 约束清晰：不需要 labeled target data、training set retrieval 或外部 knowledge base。
- 目标贴合 LLM：用 autoregressive input perplexity 替代分类 TTA 常用的 entropy minimization。
- 计算更可控：高困惑度样本策略把更新预算集中在模型不熟悉的测试输入上。
- 稳定性考虑充分：LoRA 限制更新自由度，缓解 full-parameter update 的灾难遗忘。
- 评测面较宽：AdaptEval 同时覆盖专业领域、指令分布和推理数据集。

### 局限 / 风险

- input perplexity 只是代理目标；当答案依赖输入外知识、长链推理或严格任务规则时，降低输入困惑度未必等价于更好答案。
- Online TTL 可能把近期 batch 的噪声、偏见或恶意输入写入 LoRA 状态，形成持续漂移。
- LoRA 提升稳定性但限制适应容量；full update 与 LoRA 的 trade-off 仍依赖模型规模、rank 和数据域。
- 测试时反向传播带来显存、延迟和系统复杂度，真实服务端可能无法接受。
- “至少 20%”是 domain knowledge adaptation 的论文结论，不应外推到所有任务和所有指标。

### 还能做什么

- 研究跨域连续 TTL：在多域流式输入中维护多个 adapter、路由 adapter 或做 adapter merge/reset。
- 加入安全过滤：对 high-perplexity but adversarial / low-quality 样本进行检测，避免被异常输入驱动更新。
- 做无反向传播版本：用缓存、可学习 prompt、adapter selection 或近似梯度降低部署开销。
- 细化样本选择：同时考虑 perplexity、语义多样性、重复度和不确定性，避免只追高困惑度噪声。
- 扩展评估：加入长期 online 序列、跨域切换、污染攻击和遗忘曲线。

## Q&A

- **Q: TLM 和普通 fine-tuning 最大区别是什么?**  
  A: 普通 fine-tuning 需要目标域 labeled pairs；TLM 在测试时只看 unlabeled input，用 input perplexity loss 更新 LoRA。

- **Q: 为什么选择 input perplexity，而不是直接优化答案质量?**  
  A: 测试时没有 gold answer，无法计算 supervised answer loss；论文观察到输入 perplexity 与输出 perplexity 趋势相关，因此把前者作为可计算代理目标。

- **Q: 高困惑度样本为什么更重要?**  
  A: 它们通常是当前模型最不熟悉的测试域片段，梯度信号更能推动模型贴近 $Q(x)$；低困惑度样本已经被模型较好拟合，更新收益小。

- **Q: LoRA 在这里解决什么问题?**  
  A: 它降低 test-time update 的参数量和显存，同时把更新限制在低秩空间，减少对原始模型知识的破坏。

- **Q: TLM 会不会遗忘原能力?**  
  A: 会有风险，尤其是 online 连续更新；论文用 LoRA 缓解而不是完全消除 catastrophic forgetting，未来还需要 reset、regularization 或多 adapter 管理。

- **Q: AdaptEval 为什么重要?**  
  A: TTL for LLMs 需要无标签测试域适应的评测协议；AdaptEval 把 domain、instruction、reasoning 分开，能看到 TLM 在不同 shift 类型下的收益和边界。

## 📊 Citation Landscape

> Semantic Scholar `detail.json` 本次返回 `429 Too Many Requests`，因此 TLDR、被引次数和 influential citation count 暂缺；以下只使用本目录已缓存的 `references.json` 与 `recommendations.json`，不编造 detail endpoint 数字。  
> [Semantic Scholar 查询](https://www.semanticscholar.org/search?q=Test-Time%20Learning%20for%20Large%20Language%20Models) | [Connected Papers](https://www.connectedpapers.com/main/2505.20633)

### TLDR (Semantic Scholar 自动生成)

> 暂缺，Semantic Scholar detail endpoint 当前限流 429。

### 引用统计

| 指标 | 数值 |
|------|------|
| detail endpoint | 429 Too Many Requests |
| 参考文献缓存 | 88 条 |
| 推荐论文缓存 | 100 条 |
| 被引次数 | 暂缺，detail 不可用 |
| Influential Citations | 暂缺，detail 不可用 |

### 参考文献分组 (cached references)

- **LLM 基础与评测背景**: Attention is All you Need (2017, Neural Information Processing Systems, 175574 cites); Language Models are Few-Shot Learners (2020, Neural Information Processing Systems, 57669 cites); LLaMA: Open and Efficient Foundation Language Models (2023, arXiv.org, 19748 cites)。
- **生成指标**: BLEU (2002, 32973 cites), ROUGE (2004, 20238 cites), BERTScore (2019, 8456 cites)。
- **RAG / 外部知识对比**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020, Neural Information Processing Systems, 13579 cites)。
- **TTA / TTT 近邻工作**: Tent: Fully Test-Time Adaptation by Entropy Minimization (2021, International Conference on Learning Representations, 1732 cites); TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive? (2021, Neural Information Processing Systems, 421 cites); Test-Time Training with Self-Supervision for Generalization under Distribution Shifts (2019, International Conference on Machine Learning, 1224 cites); Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs (2025, International Conference on Learning Representations, 20 cites); The Surprising Effectiveness of Test-Time Training for Abstract Reasoning (2024, arXiv.org, 50 cites)。
- **参数高效更新**: cached references 中有 QLoRA: Efficient Finetuning of Quantized LLMs (2023, Neural Information Processing Systems, 6726 cites)；主文中 LoRA 机制还需结合论文正文引用理解。

### 🔗 推荐论文 (Semantic Scholar Recommendations)

1. Training-Free Test-Time Contrastive Learning for Large Language Models (2026, venue 暂缺, 0 cites, 2604.13552)
2. Continual Learning in Large Language Models: Methods, Challenges, and Opportunities (2026, venue 暂缺, 2 cites, 2603.12658)
3. Towards Scalable Lifelong Knowledge Editing with Selective Knowledge Suppression (2026, venue 暂缺, 0 cites, 2604.19089)
4. Developing Intelligent Chatbots for Telecom Security Support: A Comparative Study of Large Language Model Utilization Strategies (2026, Proceedings of the 18th International Conference on Agents and Artificial Intelligence, 0 cites, 10.5220/0014249600004052)
5. CL-VISTA: Benchmarking Continual Learning in Video Large Language Models (2026, venue 暂缺, 0 cites, 2604.00677)
6. Embedding-based In-Context Prompt Training for Enhancing LLMs as Text Encoders (2026, venue 暂缺, 0 cites, 2605.01372)
7. Diversity in Large Language Models under Supervised Fine-Tuning (2026, venue 暂缺, 0 cites, 2605.00195)
8. Self Knowledge Re-expression: A Fully Local Method for Adapting LLMs to Tasks Using Intrinsic Knowledge (2026, venue 暂缺, 0 cites, 2604.22939)
9. Do Domain-specific Experts exist in MoE-based LLMs? (2026, venue 暂缺, 0 cites, 2604.05267)
10. Beyond Fine-Tuning: In-Context Learning and Chain-of-Thought for Reasoned Distractor Generation (2026, venue 暂缺, 0 cites, 2604.17574)

## BibTeX

```bibtex
@article{hu2025testtimelearninglargelanguage,
  title={Test-Time Learning for Large Language Models},
  author={Jinwu Hu and Zitian Zhang and Guohao Chen and Xutao Wen and Chao Shuai and Wei Luo and Bin Xiao and Yuanqing Li and Mingkui Tan},
  journal={arXiv preprint arXiv:2505.20633},
  year={2025}
}
```
