[← 返回 README](../README.md)

# 01. Introduction and Contributions

## 原文保留

Standard residual connections [12] are the de facto building block of modern LLMs [35, 51, 9]. The update $h_l=h_{l-1}+f_{l-1}(h_{l-1})$ is widely understood as a gradient highway that lets gradients bypass transformations via identity mappings, enabling stable training at depth. Yet residuals also play a second role that has received less attention. Unrolling the recurrence shows that every layer receives the same uniformly-weighted sum of all prior layer outputs; residuals define how information aggregates across depth. Unlike sequence mixing and expert routing, which now employ learnable input-dependent weighting [53, 20, 9], this depth-wise aggregation remains governed by fixed unit weights, with no mechanism to selectively emphasize or suppress individual layer contributions.

> 💡 **批注**: 这里是全文的概念转折。残差连接通常被解释为 gradient highway，但作者要求读者把它同时看成 depth-wise information aggregation。这个重述让“残差权重为什么必须是 1？”变成可优化问题。

In practice, PreNorm [60] has become the dominant paradigm, yet its unweighted accumulation causes hidden-state magnitudes to grow as $O(L)$ with depth, progressively diluting each layer’s relative contribution [27]. Early-layer information is buried and cannot be selectively retrieved; empirically, a significant fraction of layers can be pruned with minimal loss [11]. Recent efforts such as scaled residual paths [54] and multi-stream recurrences [72] remain bound to the additive recurrence, while methods that do introduce cross-layer access [36, 56] are difficult to scale. The situation parallels the challenges that recurrent neural networks (RNNs) faced over the sequence dimension before attention mechanism provided an alternative.

> 💡 **PreNorm dilution**: PreNorm 保留了干净的 identity path，但副作用是 residual stream 幅度随深度增长。归一化后的每层输入尺度固定，深层若想影响总状态，就要产生更大的输出；这会形成输出幅度与梯度分布的深度偏置。

We observe a formal duality between depth-wise accumulation and the sequential recurrence in RNNs. Building on this duality, we propose Attention Residuals (AttnRes), which replaces the fixed accumulation $\boldsymbol h_l=\sum_i \boldsymbol v_i$ with $\boldsymbol h_l=\sum_i \alpha_{i\to l}\boldsymbol v_i$, where $\alpha_{i\to l}$ are softmax attention weights computed from a single learned pseudo-query $\boldsymbol w_l\in\mathbb R^d$ per layer. This lightweight mechanism enables selective, content-aware retrieval across depth with only one $d$-dimensional vector per layer. Indeed, standard residual connections and prior recurrence-based variants can all be shown to perform depth-wise linear attention; AttnRes generalizes them to depth-wise softmax attention, completing for depth the same linear-to-softmax transition that proved transformative over sequences (§6.2, §6.1).

> 💡 **机制最小核**: 每层只有一个 learned pseudo-query $\boldsymbol w_l$，不是从当前 hidden state 动态投影出来的 query。这个选择牺牲了一点表达力，但换来可批处理、低推理开销和更简单的工程实现。

In standard training, Full AttnRes adds negligible overhead, since the layer outputs it requires are already retained for backpropagation. At scale, however, activation recomputation and pipeline parallelism are routinely employed, and these activations must now be explicitly preserved and communicated across pipeline stages. We introduce Block AttnRes to maintain efficiency in this regime: layers are partitioned into $N$ blocks, each reduced to a single representation via standard residuals, with cross-block attention applied only over the $N$ block-level summaries. This brings both memory and communication down to $O(Nd)$, and together with infrastructure optimizations (§4), Block AttnRes serves as a drop-in replacement for standard residual connections with marginal training cost and negligible inference latency overhead.

Scaling law experiments confirm that AttnRes consistently outperforms the baseline across compute budgets, with Block AttnRes matching the loss of a baseline trained with $1.25\times$ more compute. We further integrate AttnRes into the Kimi Linear architecture [69] (48B total / 3B activated parameters) and pre-train on $1.4\mathrm T$ tokens. Analysis of the resulting training dynamics reveals that AttnRes mitigates PreNorm dilution, with output magnitudes remaining bounded across depth and gradient norms distributing more uniformly across layers. On downstream benchmarks, our final model improves over the baseline across all evaluated tasks.

## Contributions

• Attention Residuals. We propose AttnRes, which replaces fixed residual accumulation with learned softmax attention over depth, and its scalable variant Block AttnRes that reduces memory and communication from $O(Ld)$ to $O(Nd)$. Through a unified structured-matrix analysis, we show that standard residuals and prior recurrence-based variants correspond to depth-wise linear attention, while AttnRes performs depth-wise softmax attention.

• Infrastructure for scale. We develop system optimizations that make Block AttnRes practical and efficient at scale, including cross-stage caching that eliminates redundant transfers under pipeline parallelism and a two-phase inference strategy that amortizes cross-block attention via online softmax [31]. The resulting training overhead is marginal, and the inference latency overhead is less than $2\%$ on typical inference workloads.

• Comprehensive evaluation and analysis. We validate AttnRes through scaling law experiments, component ablations, and downstream benchmarks on a 48B-parameter model pre-trained on 1.4T tokens, demonstrating consistent improvements over standard residual connections. Training dynamics analysis further reveals that AttnRes mitigates PreNorm dilution, yielding bounded hidden-state magnitudes and more uniform gradient distribution across depth.

## 中文读法

> 💡 **贡献一的含义**: “standard residuals 是 depth-wise linear attention，AttnRes 是 depth-wise softmax attention”是论文的理论包装。它把很多残差变体放进同一个 mixing matrix 视角，而不是只说自己多加了跨层连接。

> 💡 **贡献二的含义**: 这篇报告不是只提出模块，还明确处理 pipeline parallelism 和 inference I/O。没有 cross-stage caching 与 two-phase inference，Block AttnRes 很容易被大模型系统成本否决。

> 💡 **贡献三的含义**: 下游 benchmark 全面提升固然重要，但更关键的是训练动态图：如果 output magnitude 不再随深度单调膨胀、gradient magnitude 更均匀，就说明机制确实触及了 PreNorm dilution。

## 本节关键词

- **fixed unit accumulation**: 标准残差的隐含权重全是 1。
- **PreNorm dilution**: 残差流幅度随深度增长，使单层贡献相对变小。
- **pseudo-query**: 每层一个学习向量，负责深度维 attention 的 query。
- **Full vs Block AttnRes**: 完整跨层访问 vs block summary 近似。
- **drop-in replacement**: 论文声称 Block AttnRes 在系统优化后可替换标准残差，训练/推理开销很小。
