[← 返回 README](../README.md)

# 2. Motivation

## 📌 预览

Motivation 从三步搭桥：Transformer 的 KV cache 随历史增长，RNN/SSM 的固定状态容量有限，TTT 虽然用参数当记忆但 reconstruction objective 没有保证历史对未来推理的作用。

# 2.1. Memory issue of Transformers

Transformers exhibit quadratic computational complexity $\mathcal { O } ( N ^ { 2 } )$ due to their attention mechanism (Vaswani et al., 2017). Specifically, given a pretrained LLM $f$ with parameter $W$ , generating a single token $x _ { n + 1 } = f _ { W } ( x _ { 1 } , \ldots , x _ { n } )$ from an input sequence $X = ( x _ { 0 } , \ldots , x _ { n } )$ costs ${ \mathcal { O } } ( n )$ with

KV cache. Autoregressively, producing $N$ tokens results in a cumulative $\mathcal { O } ( N ^ { 2 } )$ computational overhead and $\mathcal { O } ( N )$ memory footprint (Fig. 2). Numerous variants (Child, 2019; Beltagy et al., 2020; Katharopoulos et al., 2020; Choromanski et al., 2021; Liu et al., 2024) have been proposed to resolve this memory issue, but they merely delay the inevitable exhaustion of cache memory and computing resources.

> 💡 **KV 瓶颈批注**: 这里关注的是生成阶段历史越长，每个新 token 都要面对越来越长的 KV。Absorber 后续 claim 的 O(1) decode 不是免费生成，而是把旧历史从 KV cache 移到参数更新里。

# 2.2. Superiority of parameter memory

The primary cause of the high computational cost of transformers in long contexts is their attention mechanism, which must process all historical contexts during deduction. To address this memory issue, prior works have proposed lineartime sequence models (Peng et al., 2023; Gu et al., 2021; Gu & Dao, 2024) that reduce the computational cost to $\mathcal { O } ( N )$ . Note that the linear-time models gain efficiency by sacrificing model capacity (Merrill et al., 2024), as they compress arbitrarily rich history into a fixed-size hidden state $h _ { t } \in \mathbb { R } ^ { d }$ . By contrast, the paradigm that restores contexts into the model parameters is more promising, since the parameter space $\boldsymbol { \theta } \in \mathbb { R } ^ { d \times d }$ is orders of magnitude larger than the activation space $h \in \mathbb { R } ^ { d }$ , offering a vastly superior capacity for encoding history. Consequently, computationally costly contexts can be safely omitted from long-stream inferences.

> 💡 **参数记忆批注**: 论文押注的是容量差异：固定 hidden state 是 $d$ 维，参数矩阵是 $d \times d$ 级别。这个论点支持“把历史写进参数”，但它也引出优化问题：容量大不等于写入目标正确。

Formally, let $X$ denote the historical contexts and $Y$ denote the subsequent inferences. In the parameter memory paradigm, rather than performing standard inference over the full context $f _ { W } ( X Y )$ , we compute $f _ { W ^ { * } } ( Y )$ without the explicit context $X$ , having stored the information of $X$ directly into the updated parameters $W ^ { * }$ . Such a parameter memory paradigm has been proposed by (Sun et al., 2025; Behrouz et al.).

![](../images/99313db5380dd90e701613cc6f3d783e2f18d4b6be00a0e8fbf6b410802b2714.jpg)
Figure 2. A demonstration of the necessity of inference with a limited number of contexts. Normally, transformers compute $f _ { W } ( x _ { 0 } x _ { 1 } \dots x _ { n } )$ to generate the next token $x _ { n + 1 }$ with a computational complexity of ${ \mathcal { O } } ( n )$ . If we absorb history into the parameters to get $f _ { W ^ { * } }$ and deduct with a limited context length, the deduction with discarded history has a reduced complexity of $\mathcal { O } ( 1 )$ , which suits scenarios with long sequence lengths.

> 💡 **Figure 2 批注**: 图 2 是目标计算图：旧历史不再参与 attention，而由 $W^*$ 承担。需要注意，图里的 O(1) 是相对已吸收历史的 decode complexity；吸收本身仍要做局部 full-context teacher forward 和 LoRA 更新。

# 2.3. Analysis of existing parameter memory methods

Test-Time Training (TTT) (Sun et al., 2025; Feng et al., 2026), the state-of-the-art parameter memory method, treats historical contexts as a continuous learning loop, updating model weights to memorize the incoming stream. That is:

$$
W ^ { * } = \arg \operatorname* { m i n } _ { W ^ { * } } \| f _ { W ^ { * } } ( x _ { \mathrm { k } } ) - x _ { \mathrm { v } } \|
$$

where $x _ { \mathrm { k } }$ and $x _ { \mathrm { v } }$ are conversions of input $x$ , which build target projections. By training the model on target projections, TTT effectively converts context into training data and leverages the plasticity of neural networks to extend memory capacity, and reveals optimum performance in comprehensive context streams as batched training data. Overall, TTT is more suitable for scenarios of effective training, as it relies on adequate inputs to stabilize its training and has limitations when there are insufficient contexts that cannot form batch data, which is more common in the prevalent scenario of model inference in user interactions. Specifically, in the model inference process, TTT memorizes contexts by training the model to project them, and simply memorizing such inadequate data tokens often leads to the retention of irrelevant noise while failing to capture the high-level semantic dependencies required for reasoning (Cao et al., 2025). Furthermore, when projecting contexts, TTT does not specify the model’s performance on subsequent inferences. This cannot invoke the causal mechanisms in LLM, as it does not maintain the contribution of removed historical sequences to future inferences, which is crucial for model deduction (Geiger et al., 2021; 2024). Overall, incorporating historical contexts is not just data summarization, but learning the causal representation (Scholkopf et al. ¨ , 2021) that maintains the intervention of history tokens to future ones.

> 💡 **TTT 局限批注**: 这段是 Absorber 的问题定义：TTT 的 projection loss 可能学会“像不像历史 token”，但没有问“丢掉历史后未来推理是否保持”。这就是为什么后文选择 $Y$ 上的行为同步作为监督。

Motivation summary To summarize, we aim to reduce the $\mathcal { O } ( N ^ { 2 } )$ computational cost of full-attention transformers by absorbing contexts into model parameters. Based on analyses of prior works, we propose to preserve the causal effects of historical context on subsequent sequences when building parameter memory.

> 💡 **动机小结批注**: 目标不是无损保存 $X$，而是保存 $X \rightarrow Y$ 的可用影响。这个信息瓶颈视角很适合 summary/reasoning，但对 passkey/needle 这类精确检索可能天然有风险，论文主实验也没有覆盖这类 hard retrieval。

---

## 🔖 Section 总结

- **为什么不用完整 KV**: 长流式生成时 KV cache 和 attention 成本持续增长。
- **为什么不用固定状态**: RNN/SSM 的状态容量与长尾依赖冲突。
- **为什么不用传统 TTT**: reconstruction/projection 没有直接约束未来推理行为。
