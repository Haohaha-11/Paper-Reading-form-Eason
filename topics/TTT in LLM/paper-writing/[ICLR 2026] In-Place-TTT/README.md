# In-Place Test-Time Training

**作者**: Guhao Feng; Shengjie Luo; Kai Hua; Ge Zhang; Di He; Wenhao Huang; Tianle Cai  
**会议**: ICLR Oral 2026  
**链接**: [arXiv 2604.06169](https://arxiv.org/abs/2604.06169) | [PDF](https://arxiv.org/pdf/2604.06169) | [Code](https://github.com/ByteDance-Seed/In-Place-TTT)  
**阅读顺序**: TTT in LLM topic paper 1  
**MinerU task**: `84eb7f5d-8e73-4bee-a015-442bcb03ff5c`

## 一句话总结

In-Place TTT 把标准 LLM MLP block 的 final projection matrix $W_{down}$ 当作推理时可更新的 fast weights，用 NTP-aligned objective、chunk-wise update 和 context parallelism 做无需从零预训练的可插拔 test-time training。

## 核心贡献

1. **MLP final projection fast weights**: 复用 gated MLP 的 $W_{down}$ 作为 fast weights，$W_{up}/W_{gate}$ 保持 slow weights，从而避免引入专门 TTT layer。
2. **Drop-in enhancement**: 可从 Qwen3/LLaMA 等预训练 checkpoint warm start，经 continual training 后原位增强长上下文能力，而不是从零训练新架构。
3. **NTP-aligned objective**: 用 Conv1D 和 $W_{target}$ 生成包含未来 token 信息的 value target，把 TTT 写入目标对齐到 autoregressive next-token prediction。
4. **Chunk-wise update**: 用 $\Delta W_{down}=\eta \hat V^T Z$ 做大 chunk 更新，消除 per-token TTT 的串行瓶颈；消融显示 512/1024 chunk 最优。
5. **Context parallelism**: chunk delta 可独立计算，再通过 prefix sum 得到每个 chunk 的历史 fast-weight 状态，保持因果同时支持长序列并行。
6. **完整证据链**: 覆盖 128k drop-in continual training、pretrain-from-scratch 对比、state size/chunk size/objective 消融和效率分析。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 摘要与核心 claim |
| [01 - Introduction](sections/01-introduction.md) | 动机、三重障碍、贡献概览 |
| [02 - Preliminary: Test-Time Training](sections/02-preliminary-test-time-training.md) | TTT 机制和 LLM desiderata |
| [03 - In-Place Test-Time Training](sections/03-in-place-test-time-training.md) | 方法：MLP fast weights、NTP objective、CP 实现 |
| [04 - Experiments](sections/04-experiments.md) | 128k drop-in、from-scratch、消融与效率 |
| [05 - Related Work](sections/05-related-work.md) | TTT、长上下文架构、记忆增强 |
| [06 - Conclusion](sections/06-conclusion.md) | 总结与开放问题 |
| [07 - Appendix](sections/07-appendix.md) | 证明、算法、训练细节和初始化 |

## 关键数字

| 指标 | 数值 |
|------|------|
| Drop-in base model | Qwen3-4B-Base |
| Continual training | ~20B tokens @32k + ~15B tokens @128k |
| Qwen3-4B RULER 128k | Baseline 74.8 → In-Place TTT 77.0 |
| Qwen3-4B RULER 256k extrapolation | Baseline 41.7 → In-Place TTT 43.9 |
| LLaMA-3.1-8B RULER 64k | Baseline 81.6 → In-Place TTT 83.7 |
| Qwen3-14B RULER 64k | Baseline 67.9 → In-Place TTT 70.6 |
| From-scratch 4B Full Attn. RULER-16k | 6.58 → 19.99 |
| From-scratch 4B SWA RULER-8k | 9.91 → 26.80 |
| Best chunk size in ablation | 512 / 1024 |
| Continual pretraining Conv size | 5 |
| Inference clipping threshold for Qwen3-4B | $\tau=1e-5$ |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Backbone forward | token hidden states $H$ | attention block 原样处理；MLP gate/up 得到 $Z=\phi(HW_{gate}^T)\odot(HW_{up}^T)$ | MLP intermediate activations $Z$ |
| 2. Value target | token embeddings $X_0$ | causal Conv1D + trainable $W_{target}$，生成含未来 token 信息的 $\hat V$ | NTP-aligned value target |
| 3. Chunk apply | chunk $Z_{[i]}$ + 当前 $W_{down}^{(i)}$ | $O_{[i]}=Z_{[i]}(W_{down}^{(i)})^T$ | 当前 chunk 输出 |
| 4. Chunk update | $Z_{[i]}$ + $\hat V_{[i]}$ | $W_{down}^{(i+1)}=W_{down}^{(i)}+\eta \hat V_{[i]}^T Z_{[i]}$ | 更新后的 fast weights |
| 5. Context parallel | 所有 chunk delta | 并行计算 delta，prefix sum 聚合历史更新 | 每个 chunk 的因果 fast-weight 状态 |
| 6. Boundary/reset | 文档边界 | reset 到 pretrained $W_{down}$，避免跨文档泄漏 | 独立样本的在线记忆 |

**整体输入**: 长上下文 token 序列。  
**整体输出**: 标准 LLM logits，但 MLP 输出投影在推理过程中携带由历史 chunk 写入的动态上下文记忆。

## 优点与局限

### 优点
- 工程路径现实：复用标准 MLP 参数，能 warm start 预训练 LLM。
- 机制闭环清楚：fast weights、NTP-aligned target、chunk-wise update 和 CP 实现互相支撑。
- 实验证据覆盖面较完整：drop-in、from-scratch、消融、效率都有对应结果。

### 局限 / 风险
- Drop-in 仍需要十亿级 token continual training，不是即插即用的零训练补丁。
- 长序列下需要 update clipping，说明 fast weights 累积更新有稳定性风险。
- 主要评估是 language modeling/RULER；对复杂 agent、在线学习、分布漂移任务的收益仍未验证。
- 从抽取文本看部分公式/表格有 MinerU 噪声，细读公式时应回到 PDF 校对。

### 还能做什么
- 比较不同 fast-weight optimizer，而不只用简单外积更新。
- 做 adaptive chunk size / adaptive update frequency，让容易片段少更新、需要记忆片段多更新。
- 检验在 MoE、linear attention、SSM backbone 中的 MLP fast-weight 插入策略。
- 研究更强的稳定性控制：归一化、门控写入、forgetting/decay、per-layer clipping。
- 在真实流式任务、continual QA、代码仓库级编辑任务上验证“在线吸收上下文”的实际价值。

## Q&A

- **Q: In-Place TTT 和传统 TTT layer 的本质区别是什么？**  
  A: 传统 TTT 往往引入或替换为专门 token-mixing layer；In-Place TTT 不替换 attention，而是把现有 MLP 的 $W_{down}$ 作为 fast weights 原位更新，因此能从预训练 LLM warm start。
- **Q: 为什么只更新 $W_{down}$？**  
  A: $W_{down}$ 是 gated MLP 最终投影，直接决定中间激活 $Z$ 如何写回 residual stream；保留 $W_{up}/W_{gate}$ 可减少对预训练能力的扰动。
- **Q: NTP-aligned objective 比 reconstruction 好在哪里？**  
  A: reconstruction 写入当前 token 信息，未必提升下一个 token 的 logit；NTP-aligned target 写入未来 token 信息，理论上在 induction 场景会提升正确 value 的 logit。
- **Q: chunk-wise update 会不会破坏因果性？**  
  A: 论文采用 apply-then-update、causal padding 和 prefix-sum 聚合，使 chunk $i$ 的输出只使用前面 chunk 的 fast-weight delta。
- **Q: 128k 结果应该怎么理解？**  
  A: 它证明在 Qwen3-4B 上经 32k/128k 两阶段 continual training 后有稳定增益；但不是单纯加载模块就能直接扩展到 128k。
- **Q: 消融最关键的结论是什么？**  
  A: state size 越大越好、chunk size 512/1024 最合适、Conv1D 与 $W_{target}$ 都重要。这三点分别支撑容量、效率和目标设计。

## 📊 Citation Landscape

> 数据来源: 本目录 `semantic-scholar/` JSON。`detail.json`、`references.json`、`recommendations.json` 均可用；未遇到 429。统计为 Semantic Scholar 当前返回值，可能随时间变化。
> [Semantic Scholar](https://www.semanticscholar.org/paper/3008be7598614b60f81742bc80a47d63d1c604b7) | [Connected Papers](https://www.connectedpapers.com/main/2604.06169)

### TLDR (Semantic Scholar 自动生成)

> This work introduces In-Place Test-Time Training (In-Place TTT), a framework that seamlessly endows LLMs with Test-Time Training ability, and replaces TTT's generic reconstruction objective with a tailored, theoretically-grounded objective explicitly aligned with the Next-Token-Prediction task governing autoregressive language modeling.

### 引用统计

| 指标 | 数值 |
|------|------|
| Semantic Scholar Paper ID | `3008be7598614b60f81742bc80a47d63d1c604b7` |
| 参考文献数 | 61 |
| 被引次数 | 1 |
| Influential Citations | 0 |

### 参考文献分组 (Top by citations, Semantic Scholar)

| 论文 | 年份 | Venue | Citations |
|------|------|-------|-----------|
| Attention is All you Need | 2017 | Neural Information Processing Systems | 175574 |
| Language Models are Few-Shot Learners | 2020 | Neural Information Processing Systems | 57669 |
| GPT-4 Technical Report | 2023 | N/A | 24176 |
| LLaMA: Open and Efficient Foundation Language Models | 2023 | arXiv.org | 19748 |
| Chain of Thought Prompting Elicits Reasoning in Large Language Models | 2022 | Neural Information Processing Systems | 17863 |
| The Llama 3 Herd of Models | 2024 | N/A | 14860 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 2020 | Neural Information Processing Systems | 13579 |
| Measuring Massive Multitask Language Understanding | 2020 | International Conference on Learning Representations | 7914 |

### 🔗 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | Citations | 链接 |
|------|------|-----------|------|
| Learning When to Attend: Conditional Memory Access for Long-Context LLMs | 2026 | 2 | [Semantic Scholar](https://www.semanticscholar.org/paper/290e74d22cb4cc7750f746a523e73a7b16e5dfc0) |
| Absorber LLM: Harnessing Causal Synchronization for Test-Time Training | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/5aee6464412cb5a3967a07aa93b655cde66d9383) |
| Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/80c66d2b868beede25df341ccbeb5a9e83ecd277) |
| Effective Distillation to Hybrid xLSTM Architectures | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/6d274b6db314462bb622cd402fe8f56e5e392ab9) |
| SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding | 2026 | 1 | [Semantic Scholar](https://www.semanticscholar.org/paper/d8fa68482ab639ea8e2db676c9974ecef2664ab5) |
| HiCI: Hierarchical Construction-Integration for Long-Context Attention | 2026 | 0 | [Semantic Scholar](https://www.semanticscholar.org/paper/09e2745f472ad37fbe8b53b71b4098815fdcab08) |

## BibTeX

```bibtex
@article{feng2026inplacettt,
  title={In-Place Test-Time Training},
  author={Guhao Feng and Shengjie Luo and Kai Hua and Ge Zhang and Di He and Wenhao Huang and Tianle Cai},
  journal={arXiv preprint arXiv:2604.06169},
  year={2026}
}
```
