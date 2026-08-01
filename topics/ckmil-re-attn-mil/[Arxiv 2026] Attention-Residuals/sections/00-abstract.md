[← 返回 README](../README.md)

# 00 - 摘要

## Abstract

> Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer's contribution. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, input-dependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead.

> Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks.

> 💡 **Hao 批注 - 时间-深度对偶是论文的理论支点**: Abstract 没有提到"时间-深度对偶性"，但在正文中这是核心 intellectual contribution。论文用这个对偶性把残差连接的各种变体统一到"深度维度注意力"的框架里，AttnRes 是这个框架下最自然的下一步：softmax 注意力。

> 💡 **Hao 批注 - Block 设计是工程关键**: Abstract 明确列出了从 Full 到 Block 的退化路径：Full 全深度注意力（O(Ld) 存储），Block 分组求和+块间注意力（O(Nd)，N≈8），加上缓存流水线和两阶段计算。这是从 idea 到可部署方法的关键链路。

## Contributions

论文列出三项核心贡献：

1. **Attention Residuals**: 提出 AttnRes 将固定残差累积替换为深度维度上的可学习 softmax 注意力，以及其可扩展变体 Block AttnRes，将内存和通信从 O(Ld) 降至 O(Nd)。通过统一的结构化矩阵分析，证明标准残差和先前的递归变体对应于深度线性注意力，而 AttnRes 执行深度 softmax 注意力。

2. **大规模基础设施**: 开发系统优化使 Block AttnRes 在大规模训练和推理中实用高效，包括跨阶段缓存消除流水线并行下的冗余传输，以及两阶段推理策略通过 online softmax 摊销跨块注意力。训练开销 marginal，推理延迟开销 <2%。

3. **全面评估与分析**: 通过 Scaling Law 实验、组件消融和 48B 模型 1.4T token 预训练的下游基准测试验证 AttnRes，证明其相较标准残差的一致改进。训练动态分析揭示 AttnRes 缓解了 PreNorm 稀释，产生有界的隐藏状态幅值和更均匀的跨深度梯度分布。
