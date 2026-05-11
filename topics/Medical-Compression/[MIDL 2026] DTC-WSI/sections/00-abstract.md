[← 返回 README](../README.md)

# 00 - Abstract

## 预览

摘要把 DTC-WSI 的 claim 压缩成一条证据链：WSI token 太多 -> 纯 pruning 容易早期丢信息 -> multi-stage merging + importance pruning 保持梯度与诊断形态 -> r=0.4 附近带来速度、显存和准确率收益。

# Abstract

Whole-slide images (WSIs) contain tens of thousands of heterogeneous patches, making transformer-based multiple-instance learning (MIL) computationally expensive due to quadratic attention costs and substantial redundancy in tissue morphology. Existing token-reduction approaches for WSI analysis rely primarily on pruning, which discards information early in training and destabilizes optimization under weak supervision. We propose Dynamic Token Compression for Whole-Slide Images (DTC-WSI), a token-efficient MIL framework that performs progressive, importance-aware WSI compression. DTC-WSI integrates a lightweight saliency network with a multi-stage token compressor that combines bipartite similarity matching and soft differentiable pruning to gradually eliminate redundant or non-diagnostic patches. During training, soft gates enable stable gradient flow, while inference employs deterministic compression for substantial acceleration. This curriculum-style compression preserves discriminative morphology and dramatically reduces computational burden. Across four WSI benchmarks (TCGA-NSCLC, TCGA-BRCA, TCGA-RCC, PANDA), DTC-WSI achieves $\mathbf { 5 - 1 0 \times }$ token reduction, up to ${ \bf 5 . 3 \times }$ faster inference, and $2 0 { - } 4 0 \%$ lower memory usage, while improving MIL classification accuracy by $2 { - } 4 \%$ over state-of-the-art baselines. Our results demonstrate that dynamic token compression is a powerful and scalable alternative to pruning, enabling efficient transformer-based WSI analysis while improving accuracy.

> 💡 **摘要证据链**: 这里的关键不是“剪掉 token”，而是把 WSI token compression 拆成两个互补动作：bipartite similarity matching 用来融合冗余形态，importance-guided pruning 用来删低 saliency token。训练时 soft gates 保梯度，推理时 hard deterministic compression 才兑现加速。

> 💡 **口径提醒**: 摘要写 5-10x token reduction，但主表的默认配置是 final retention $r=0.4$，等价于保留 40% token。读结果时要区分“token reduction claim”和“表格中实际 r/FLOPs/time”的口径，避免把所有实验都理解成 10x 压缩。

Keywords: Computational pathology, Token merging, Dynamic token pruning, Weakly supervised learning

> 💡 **关键词定位**: 这篇更接近“WSI token-level compute compression”，不是像 JPEG/AE 那样压缩图像文件大小；它服务于 MIL aggregation 的 token budget 和 attention/aggregation 成本。

## Section 总结

| 关键词 | 本文含义 |
|---|---|
| WSI token compression | 在 patch feature token 层减少 MIL 输入，而不是压缩原始 WSI 文件 |
| importance network | 用 slide-level supervision 学 per-token saliency |
| dynamic multi-stage compression | 分阶段 merge/prune，缓解弱监督早期硬剪枝风险 |
| bipartite soft matching | 降低相似 token 搜索成本，并融合冗余 morphology |
| soft differentiable pruning | 训练保梯度，推理才 Top-K hard pruning |
