[← 返回 README](../README.md)

# 02 Related Work

## 📌 Preview

本文没有独立的相关工作章节。相关工作以行内方式讨论：DeepSeek-V4 [1] 和 Qwen3.5 [2] 在引言中被引用为代表方法，它们使用 HCA 或线性注意力来缓解（但未消除）KV Cache 的线性扩展。六篇核心参考文献覆盖了长上下文 LLM 架构和评估领域。

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

> **Q&A 批注记录**: 论文的相关工作部分相对精简（没有独立章节），这与它的技术报告性质一致。但值得注意的是，作者在 Section 3.3.2 中对 MRCR [6] 的深入失败分析实质上构成了最深刻的相关工作讨论——它揭示了现有长上下文基准测试之间的根本性质差异：LongBench-v2/LongMemEval/RULER 是"稀疏记忆型"类任务（10-25% chunks 即恢复全部精度），而 MRCR 是"密集记忆型"类任务（50% golden chunks 仍有精度损失）。这个分类可能对未来的长上下文评估和架构设计有重要指导意义。

> **值得关注的缺失引用**: 论文没有详细讨论以下相关方向：(1) StreamingLLM / H2O 等基于 attention sink 的 KV Cache 淘汰策略；(2) InfLLM / Quest 等基于检索增强的长上下文方案；(3) RingAttention / DistAttention 等分布式长上下文推理方案。这些在未来的正式论文中可能需要补充。

## 🔖 Summary

论文的相关工作虽精简但布局策略明确：DeepSeek-V4 提供了架构基础，Qwen3.5 代表了替代的压缩范式，四项基准测试（LongBench-v2、LongMemEval、RULER、MRCR）共同对稀疏记忆和密集记忆的长上下文能力进行了压力测试。MRCR 失败案例本身即构成了对先前基准测试的实际对比分析。
