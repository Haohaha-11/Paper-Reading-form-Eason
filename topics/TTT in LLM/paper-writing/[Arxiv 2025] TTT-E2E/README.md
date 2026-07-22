# End-to-End Test-Time Training for Long Context

**作者**: Arnuv Tandon; Karan Dalal; Xinhao Li; Daniel Koceja; Marcel Rød; Sam Buchanan; Xiaolong Wang; Jure Leskovec; Sanmi Koyejo; Tatsunori Hashimoto; Carlos Guestrin; Jed McCaleb; Yejin Choi; Yu Sun  
**会议/年份**: arXiv 2025  
**阅读顺序**: TTT in LLM topic paper 2  
**链接**: [arXiv 2512.23675](https://arxiv.org/abs/2512.23675) | [PDF](https://arxiv.org/pdf/2512.23675) | [Code](https://github.com/test-time-training/e2e)

## 一句话总结

TTT-E2E 将 long-context language modeling 视为 continual learning：用标准 sliding-window Transformer 在测试时对上下文做 next-token prediction 更新，把历史压进权重，并在训练时通过 meta-learning 学到适合这种测试时学习的初始化。

## 核心贡献

1. 把 long context 从“架构设计问题”重写为“测试时连续学习问题”，用 updated weights 做 context compression。
2. 提出 E2E TTT：inner loop 在测试时直接优化最终 next-token prediction loss，outer loop 在训练时直接优化经过 TTT 后的 loss。
3. 将 mini-batch TTT 与 sliding-window attention 组合：SWA 管局部 batch 内上下文，TTT 权重更新管跨 batch 长程信息。
4. 给出从 TTT-KVB 到 TTT-E2E 的替代推导，说明关键差异是从 layer-wise reconstruction/KV binding 转向 end-of-network next-token loss。
5. 在 3B、164B tokens、128K context 证据链上展示：TTT-E2E 的 context-length scaling 接近 full attention，同时保持近似 constant inference latency。

## 📖 批读导航

| Section | 内容 |
|---------|------|
| [00 - Abstract](sections/00-abstract.md) | 总 claim + Figure 1 |
| [01 - Introduction](sections/01-introduction.md) | 动机：long context as continual learning |
| [02 - Method](sections/02-method.md) | TTT-E2E 机制、mini-batch、SWA、KVB 推导 |
| [03 - Main Results](sections/03-main-results.md) | 消融、scaling、needle、长生成、效率 |
| [04 - Related Work](sections/04-related-work.md) | continual learning / TTT / fast weights / meta-learning |
| [05 - Conclusion](sections/05-conclusion.md) | 层级记忆观点 |
| [06 - Appendix](sections/06-appendix.md) | 作者贡献、致谢、References、toy/basic recipe、baseline 改动和 decoding 细节 |

## 关键数字

| 指标 | 数值 |
|------|------|
| 最大模型规模 | 3B parameters |
| 主结果训练 token | 164B tokens |
| 最大上下文 | 128K |
| 预训练上下文 | 8K |
| extension fine-tuning | final context length，最高 128K |
| sliding-window size | $k=8K$ |
| TTT mini-batch size | $b=1K$ |
| TTT 更新层 | 最后 1/4 Transformer blocks 的 MLP layers |
| 128K prefill latency | 比 full attention 快 2.7x，H100 |
| 训练延迟风险 | 8K 上比 full attention 慢 3.4x；128K 上快 1.2x，H200 benchmark |
| Needle-in-a-haystack | full attention 明显更强，尤其长上下文精确检索 |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 变换 | 输出 |
|------|------|------|------|
| 1. Prefill | $x_0, ..., x_T$ | sliding-window Transformer 读取局部窗口 | 每个 token 的局部表示 |
| 2. Test-time loss | mini-batch tokens | 用 next-token prediction loss 形成自监督训练信号 | batch-level gradient |
| 3. Weight update | 当前 MLP weights $W_{i-1}$ | 只更新最后 1/4 blocks 的 MLP；attention/norm/embedding 冻结 | 更新后 weights $W_i$ |
| 4. Context compression | 已读上下文 | 信息写入 updated MLP weights | 跨窗口长期记忆 |
| 5. Decode | 最近窗口 + updated weights | SWA 提供短期上下文，weights 提供长期压缩 | 下一 token 分布 |
| 6. Training-time meta-learning | 训练序列 | 把训练序列模拟成测试序列跑 inner-loop，再通过 gradients of gradients 优化初始化 | 适合 TTT 的 $W_0$ |

**整体输入**: 长上下文 token 序列  
**整体输出**: 用局部窗口和测试时更新权重共同条件化的 next-token distribution

## 优点与局限

### 优点

- 问题定义清晰：不是再造一个 long-context layer，而是把 long-context LM 变成 continual learning。
- 推理效率抓住关键瓶颈：像 SWA/RNN 一样 constant latency，同时在平均 LM loss 上接近 full attention 的 context scaling。
- E2E 对齐充分：测试时 loss 与训练时 outer objective 都围绕最终 next-token prediction，而不是代理重构目标。
- 工程上保守：基座仍是标准 Transformer + SWA，测试时更新普通 MLP weights，可复用标准训练/分片基础设施。
- 实验维度完整：有超参消融、training compute scaling、context length scaling、needle、长生成和 latency/training cost。

### 局限 / 风险

- 压缩式记忆天然会丢细节；S-NIAH 表明 full attention 在 pass-key/UUID 这类精确检索上明显更强。
- 当前训练实现需要 gradients of gradients，短上下文训练延迟仍很重，可能影响真实大规模训练成本。
- 长生成评估只是 base model + Qwen evaluator 的有限 sanity check，不能直接代表 instruction tuning 或 RL 后的长 CoT。
- 3B/164B tokens 是有说服力的中等规模证据，但距离前沿生产模型规模仍有外推风险。
- 只更新最后 1/4 blocks 是经验折中，是否适合 1M context、不同数据或更大模型还需要验证。

### 还能做什么

- 从预训练 full/SWA Transformer warm-start TTT-E2E，减少二阶训练覆盖的 token 比例。
- 做支持 gradients of gradients 的 custom attention kernel，降低训练 latency。
- 结合 retrieval/full-attention sentinel 机制，补足 needle-style 无损 recall。
- 研究 self-generated / filtered mini-batch TTT，让模型在写入权重前先清洗或总结上下文。
- 扩展到 1M context，重新评估 $k$、$b$、更新层数和权重记忆容量的 scaling law。

## Q&A

- **Q: TTT-E2E 和普通 long-context architecture 的根本区别是什么？**  
  A: 它不主要靠改 attention/RNN layer，而是把长上下文当成测试时持续到来的训练数据，用 next-token loss 更新权重。
- **Q: 为什么说 weights 是 context compression？**  
  A: 因为读过的 token 不再全部保存在 KV cache 中，而是通过梯度更新改变 MLP weights；这些权重携带有损压缩后的历史信息。
- **Q: 为什么还需要 sliding-window attention？**  
  A: mini-batch TTT 在 batch 内不会逐 token 更新权重，SWA 用来保留 batch 内和最近窗口的短期上下文；主实验用 $k=8K$、$b=1K$。
- **Q: E2E 到底 E2E 在哪里？**  
  A: 测试时 inner-loop 直接优化最终 next-token prediction loss；训练时 outer-loop 直接优化 TTT 后的最终 loss。
- **Q: 和 TTT-KVB 的差异是什么？**  
  A: TTT-KVB 用层内 key-value binding/reconstruction loss，像把 KV cache 写进层状态；TTT-E2E 用网络末端 next-token loss，目标与语言建模评价一致。
- **Q: 它是否全面替代 full attention？**  
  A: 不能。平均 LM loss 和 latency 很强，但精确检索任务仍明显弱于 full attention。
- **Q: 训练成本是否被解决？**  
  A: 没有。论文明确说当前训练 latency 是限制，主要因为二阶梯度和 kernel 支持不足。

## 📊 Citation Landscape

> 数据来源: 本目录 `semantic-scholar/`。`detail.json` 返回 HTTP 429，因此 TLDR、被引次数、influential citations 等 detail 字段不可用；以下只使用 `references.json` 和 `recommendations.json`，不编造缺失数字。

### TLDR (Semantic Scholar 自动生成)

> 暂缺：`detail.json` 因 429 不可用。

### 引用统计

| 指标 | 数值 |
|------|------|
| 参考文献条目数 | 104 |
| 被引次数 | 暂缺，detail fallback: HTTP 429 |
| Influential Citations | 暂缺，detail fallback: HTTP 429 |
| 推荐论文条目数 | 100 |

### 参考文献 Top by citations

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | 2019 | 114208 |
| [Long Short-Term Memory](https://www.semanticscholar.org/paper/2e9d221c206e9503ceb452302d68d10e293f2a10) | 1997 | 104108 |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | 2020 | 57669 |
| [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) | 2013 | 34244 |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | 2024 | 14860 |
| [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) | 2014 | 14563 |
| [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) | 2017 | 14280 |
| [Robust Locally Weighted Regression and Smoothing Scatterplots](https://www.semanticscholar.org/paper/30e7e25061b7ccd9a625548dd6836afcff85043b) | 1979 | 11456 |
| [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | 2021 | 11303 |
| [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796) | 2016 | 9693 |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | 7771 |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | 2023 | 6865 |

### 推荐论文 (Semantic Scholar Recommendations)

| 论文 | 年份 | 引用数 |
|------|------|--------|
| [In-Place Test-Time Training](https://www.semanticscholar.org/paper/3008be7598614b60f81742bc80a47d63d1c604b7) | 2026 | 1 |
| [GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent](https://www.semanticscholar.org/paper/82c0b4a4c4e9fd48077f97ece80956ae1e7d97b3) | 2026 | 0 |
| [Learning When to Attend: Conditional Memory Access for Long-Context LLMs](https://www.semanticscholar.org/paper/290e74d22cb4cc7750f746a523e73a7b16e5dfc0) | 2026 | 2 |
| [Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion](https://www.semanticscholar.org/paper/c008912e095eb4f5369c591399e3dacffa0d829c) | 2026 | 0 |
| [MemDLM: Memory-Enhanced DLM Training](https://www.semanticscholar.org/paper/a399d84184e67b21a143bd562e25e0b9edaa76ff) | 2026 | 0 |
| [Efficient Pre-Training with Token Superposition](https://www.semanticscholar.org/paper/c9d68a491a9fdeedddc0047248135365708f90fd) | 2026 | 0 |
| [Effective Distillation to Hybrid xLSTM Architectures](https://www.semanticscholar.org/paper/6d274b6db314462bb622cd402fe8f56e5e392ab9) | 2026 | 0 |
| [Rethinking Local Learning: A Cheaper and Faster Recipe for LLM Post-Training](https://www.semanticscholar.org/paper/7da3df85076586e137a432284cb7cbd4f124e0aa) | 2026 | 0 |
| [Shuffle the Context: RoPE-Perturbed Self-Distillation for Long-Context Adaptation](https://www.semanticscholar.org/paper/a4444d5937c57c3fb577b591956a5211b85c07ec) | 2026 | 0 |
| [HiCI: Hierarchical Construction-Integration for Long-Context Attention](https://www.semanticscholar.org/paper/09e2745f472ad37fbe8b53b71b4098815fdcab08) | 2026 | 0 |

## BibTeX

```bibtex
@article{tandon2025endtoend,
  title={End-to-End Test-Time Training for Long Context},
  author={Arnuv Tandon and Karan Dalal and Xinhao Li and Daniel Koceja and Marcel Rød and Sam Buchanan and Xiaolong Wang and Jure Leskovec and Sanmi Koyejo and Tatsunori Hashimoto and Carlos Guestrin and Jed McCaleb and Yejin Choi and Yu Sun},
  journal={arXiv preprint arXiv:2512.23675},
  year={2025}
}
```
