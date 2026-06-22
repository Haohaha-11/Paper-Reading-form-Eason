[← 返回 README](../README.md)

# 02 Related Work

## 📌 Preview

The paper does not have a standalone Related Work section. Instead, related work is discussed inline: DeepSeek-V4 [1] and Qwen3.5 [2] are cited in the introduction as representative approaches that use HCA or linear attention to mitigate (but not eliminate) KV cache linear scaling. The six core references span the long-context LLM architecture and evaluation landscape.

---

## Integrated References

The related work is woven throughout the text rather than presented as a dedicated section. Here we catalogue the key references and their roles:

| Reference | Role in Paper |
|-----------|--------------|
| [1] DeepSeek-V4 (DeepSeek-AI, 2026) | Primary architectural backbone; source of HCA/CSA layers, Lightning Indexer, and compressed indexer keys K^{IComp} |
| [2] Qwen3.5 (Qwen Team, 2026) | Alternative approach using linear attention for memory compression; cited as another example of "mitigating rather than eliminating" KV cache growth |
| [3] LongBench-v2 (Bai et al., 2025) | Primary benchmark for long-context understanding and reasoning |
| [4] LongMemEval (Wu et al., 2025) | Benchmark for long-term interactive memory evaluation |
| [5] RULER (Hsieh et al., 2024) | Benchmark for measuring real context size of long-context LMs |
| [6] Michelangelo/MRCR (Vodrahalli et al., 2024) | Benchmark for long-context evaluation via latent structure queries; the critical failure case for FM-DS-V4 |

> **Q&A 批注记录**: 论文的 related work 部分相对精简（没有独立 section），这与它的 technical report 性质一致。但值得注意的是，作者在 Section 3.3.2 中对 MRCR [6] 的深入失��分析实质上构成了最深刻的 related work discussion —— 它揭示了现有 long-context benchmarks 之间的 fundamental property difference：LongBench-v2/LongMemEval/RULER 是 "sparse-memory" 类任务（10-25% chunks 即恢复全部精度），而 MRCR 是 "dense-memory" 类任务（50% golden chunks 仍有精度损失）。这个分类可能对未来的 long-context evaluation 和 architecture design 有重要指导意义。

> **值得关注的缺失引用**: 论文没有详细讨论以下相关方向：(1) StreamingLLM / H2O 等基于 attention sink 的 KV cache 淘汰策略；(2) InfLLM / Quest 等基于检索增强的长上下文方案；(3) RingAttention / DistAttention 等分布式 long-context 推理方案。这些在未来的 formal paper 中可能需要补充。

## 🔖 Summary

The paper's related work is minimal but strategically deployed: DeepSeek-V4 provides the architectural foundation, Qwen3.5 represents the alternative compression paradigm, and the four benchmarks (LongBench-v2, LongMemEval, RULER, MRCR) collectively stress-test both sparse-memory and dense-memory long-context capabilities. The MRCR failure case itself serves as a de facto comparative analysis against prior benchmarks.
