[← 返回 README](../README.md)

# 04. Infrastructure Design

## 原文保留

Block AttnRes introduces additional system challenges compared to standard residual connections. For large-scale model training, block representations must be propagated across pipeline stages, causing heavy communication in a naïve implementation. During inference, repeated access to accumulated block representations increases latency, while long-context prefilling amplifies the memory cost of caching block representations. We address these challenges with cross-stage caching in training, and with a two-phase computation strategy together with a memory-efficient prefilling scheme in inference.

![Cache-based pipeline communication example](../images/1c4b7b713d314876d9ba253dc620320690c7eb7ffaf7950361a0ca28cb64dcbb.jpg)

Figure 3: Cache-based pipeline communication example with 4 physical ranks and 2 virtual stages per rank, where hatched boxes denote end of AttnRes blocks. Numbers indicate micro-batch indices. Each rank caches previously received blocks; stage transitions only transmit incremental blocks instead of the full history.

> 💡 **系统问题不是附属品**: Block AttnRes 的数学复杂度降到 $O(Nd)$ 后，真正瓶颈变成 pipeline stage 间怎么传 block summaries，以及推理时怎么避免每层反复扫历史 block。

## 4.1 Training

For small-scale training, AttnRes adds a tiny computation overhead and no extra memory usage, as the activations need to be saved for backpropagation regardless. Under large-scale distributed training, pipeline parallelism poses the primary infrastructure challenge for AttnRes. Full AttnRes requires all $L$ layer outputs to be transmitted across stages; Block AttnRes reduces this to $N$ block representations, and the optimizations below further minimize the remaining overhead.

Pipeline communication. With standard residual connections, pipeline parallelism [18] transfers a fixed-size hidden state between adjacent stages, independent of pipeline depth. Block AttnRes requires all accumulated block representations at each stage for inter-block attention, and naïvely transmitting the full history at every transition incurs redundant communication.

Consider an interleaved pipeline schedule [33] with $P$ physical stages and $V$ virtual stages per physical stage. For simplicity, assume each physical stage produces on average $N_p$ block representations of dimension $d$ per token. With $C=PV$ total chunks, the $j$-th chunk accumulates $jN_p$ blocks. Naïvely transmitting all accumulated blocks at every transition incurs per-token communication cost:

$$
\mathrm{Comm}_{\mathrm{naive}}
=
\sum_{j=1}^{C-1}jN_p d
=
\frac{C(C-1)}{2}N_p d.
$$

Cross-stage caching. Since each physical stage processes multiple virtual stages in succession, we can eliminate this redundancy by caching blocks locally: blocks received during earlier virtual stages remain in local memory and need not be re-transmitted. The first virtual stage has no cache and accumulates normally; for $v\geq2$, each transition conveys only the $\sim PN_p$ incremental blocks accumulated since the receiver’s corresponding chunk in the previous virtual stage. Total communication reduces to:

$$
\mathrm{Comm}_{\mathrm{cached}}
=
\underbrace{\frac{P(P-1)}{2}N_p d}_{\text{first virtual stage}}
+
\underbrace{(V-1)P^2N_p d}_{\text{subsequent virtual stages}}.
$$

Caching reduces peak per-transition cost from $O(C)$ to $O(P)$, a $V\times$ improvement that enables full overlap with computation during steady-state 1F1B. The backward pass benefits from the same scheme. Fig. 3 illustrates this optimization with $P=4$ and $V=2$: for the second virtual stage, caching eliminates 6 redundant block transmissions.

Memory overhead. With cross-stage caching, each block is stored exactly once across all $V$ virtual stages, which becomes negligible relative to standard per-layer activation cache. Crucially, the per-layer activation footprint remains identical to standard architectures, as activation checkpointing eliminates all inter-block attention intermediates, and the checkpointed input matches the memory size of the hidden state $h_l$ it replaces.

In terms of wall-clock time, Block AttnRes adds negligible training overhead when pipeline parallelism is not enabled; under pipeline parallelism, the measured end-to-end overhead is less than $4\%$.

> 💡 **训练端重点**: caching 的核心不是压缩 block，而是“不要重复传已经传过的历史”。在 interleaved pipeline 中，同一个 physical rank 会连续处理多个 virtual stage，因此本地缓存可以复用。

## 4.2 Inference

The two-phase computation strategy described below applies to both Full and Block AttnRes: in either case, layers are grouped into blocks of size $S$, with Phase 1 batching the inter-block queries and Phase 2 handling sequential intra-block lookback. For Full AttnRes, this reduces per-layer I/O from $O(Ld)$ to $O((S+N)d)$; Block AttnRes further reduces the stored representations from $L$ to $N$, since each block is compressed into a single vector. In what follows, we focus on Block AttnRes and detail the two-phase computation strategy together with a sequence-sharded prefilling scheme for long-context inputs.

Two-phase computation strategy. The layer-wise attention computation of Block AttnRes resembles autoregressive decoding, where block representations serve as a shared KV cache reused across layers. A naïve implementation computes the attention residual at every layer, each requiring a full pass over all preceding blocks, resulting in $O(L\cdot N)$ memory accesses. Since the pseudo-query vectors are decoupled from the forward computation (§3), all $S=L/N$ queries within a block can be batched into a single matrix multiplication, amortizing memory access from $S$ reads to 1.

Algorithm 1 instantiates a two-phase computation strategy exploiting this property.

• Phase 1 computes inter-block attention for all $S$ layers simultaneously via a single batched query against the cached block representations, returning both outputs and softmax statistics (max and log-sum-exp). This amortizes the memory access cost, reducing reads from $S$ times to just once per block.

• Phase 2 computes intra-block attention sequentially for each layer using the evolving partial sum, then merges with Phase 1 outputs through online softmax [31]. Because the online-softmax merge is elementwise, this phase naturally admits kernel fusion with surrounding operations, further reducing I/O overhead.

With the two-phase design, Phase 2 preserves an I/O footprint similar to that of standard residual connections, whereas the main additional cost arises from Phase 1 inter-block attention. Because these inter-block reads are amortized across all layers in a block through batching, the total per-layer memory access cost remains only $(N/S+3)d$ reads and $2d$ writes in the Block setting. In practice, Phase 1 can also partially overlap with the computation of the first layer in the block, further reducing its wall-clock impact. As a result, the end-to-end inference latency overhead is less than $2\%$ on typical inference workloads.

Memory-efficient prefilling. Storing block representations during prefilling requires $N\cdot T\cdot d$ elements, which incurs 15GB of memory for a 128K-token sequence with 8 blocks. We mitigate this by sharding these representations along the sequence dimension across $P$ tensor-parallel devices, allowing Phase 1 to execute independently on local sequence shards. The Phase 2 online-softmax merge then integrates into the standard TP all-reduce communication path: the output is reduce-scattered, merged locally, and reconstructed via all-gather, naturally admitting kernel fusion with operations like RMSNorm. This reduces the per-device memory footprint to $N\cdot(T/P)\cdot d$, lowering the 128K-context example from 15GB to roughly 1.9GB per device. Combined with chunked prefill (e.g., 16K chunk size), the overhead further reduces to under 0.3GB per device.

## 中文批读

> 💡 **two-phase 的本质**: Phase 1 把“每层都读一遍历史 block”变成“每个 block 读一遍历史 block，服务 block 内所有层”；Phase 2 只处理当前 block 内不可并行的 partial sum 依赖。

> 💡 **online softmax 为什么出现**: Phase 1 和 Phase 2 分别得到一部分 softmax 分子、max 和 log-sum-exp 统计量。online softmax 允许把两部分精确合并，而不是近似相加 attention output。

> 💡 **Full 与 Block 的区别**: two-phase schedule 也能优化 Full AttnRes 的 I/O，但 Full 仍要保存 layer-level sources；Block 把 source 本身压成 block-level summaries，所以训练通信和推理缓存都更小。

> 💡 **长上下文 prefill**: $N\cdot T\cdot d$ 在 128K context 下非常大。报告用 sequence dimension sharding 把 block representations 分摊到 TP devices，再用 chunked prefill 把峰值进一步降到 <0.3GB/device。

## 机制检查表

| 场景 | 朴素问题 | 报告方案 | 数字 |
| --- | --- | --- | --- |
| Pipeline training | 每个 stage transition 重复传完整 block history | physical rank 本地缓存已收到 block，只传 incremental blocks | peak per-transition $O(C)\to O(P)$，流水线训练开销 <4%。 |
| Decoding / inference | 每层读所有 preceding blocks | Phase 1 批量 inter-block attention，Phase 2 顺序 intra-block merge | 典型推理延迟开销 <2%。 |
| Long-context prefill | block cache 为 $N\cdot T\cdot d$ | sequence-sharded block representations + TP all-reduce path fusion | 128K、8 blocks: 15GB → 1.9GB/device → <0.3GB/device。 |
