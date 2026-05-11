[← 返回 README](../README.md)

# 00. Abstract

## 原文保留

Residual connections [12] with PreNorm [60] are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer’s contribution [27]. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, inputdependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead.

Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture [69] (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks.

![Overview of Attention Residuals](../images/e81850dc0ae25b9ff97ae4a60c1a84b1511d9d11108783db88d37bf588d69281.jpg)

Figure 1: Overview of Attention Residuals. (a) Standard Residuals: standard residual connections with uniform additive accumulation. (b) Full AttnRes: each layer selectively aggregates all previous layer outputs via learned attention weights. (c) Block AttnRes: layers are grouped into blocks, reducing memory from $O(Ld)$ to $O(Nd)$.

## 批注

> 💡 **问题定位**: 摘要把标准 PreNorm residual 的缺陷说得很具体：不是“残差不好”，而是 residual stream 把所有层输出用固定单位权重累加，导致深度越大，早期层和单个层的相对贡献越被稀释。

> 💡 **核心替换**: AttnRes 的最小改动是把 $\sum_i \boldsymbol v_i$ 换成 $\sum_i \alpha_{i\to l}\boldsymbol v_i$。这里的 $\alpha$ 是深度维 softmax attention，不是序列维 attention。

> 💡 **工程折中**: Full AttnRes 最符合机制直觉，但大模型训练时需要保留/通信所有前层输出；Block AttnRes 用 block-level representation 近似它，把系统对象从 $L$ 个 layer outputs 降成 $N$ 个 block summaries。

> 💡 **证据链**: 摘要的证据不是单点 benchmark，而是三段：scaling law 证明跨规模有效，ablation 证明 content-dependent depth-wise selection 有用，大模型 Kimi Linear 1.4T tokens 证明能接进真实训练系统。

## 读者应带走的三句话

1. AttnRes 是“沿深度维的 attention”，目标是替代 PreNorm 中固定权重的 residual accumulation。
2. Block AttnRes 是让机制落地的关键，负责把 $O(Ld)$ 的存储与通信压力压到 $O(Nd)$。
3. 报告最重要的实证数字是：48B total / 3B activated Kimi Linear、1.4T tokens、典型推理延迟开销小于 2%。
