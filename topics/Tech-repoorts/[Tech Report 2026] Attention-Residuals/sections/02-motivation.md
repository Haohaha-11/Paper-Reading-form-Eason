[← 返回 README](../README.md)

# 02. Motivation

## 原文保留

Notation. Consider a batch of input sequences with shape $B\times T\times d$, where $B$ is the batch size, $T$ is the sequence length, and $d$ is the hidden dimension. For clarity, we write formulas for a single token: $\boldsymbol h_l\in\mathbb R^d$ denotes the hidden state entering layer $l$, where $l\in\{1,\ldots,L\}$ is the layer index and $L$ is the total number of layers. The token embedding is $\boldsymbol h_1$. The function $f_l$ represents the transformation applied by layer $l$. In Transformer models, we treat each self-attention or MLP as an individual layer.

> 💡 **记号提醒**: 这里的 layer 粒度不是一个 Transformer block，而是 self-attention 或 MLP 都算一个 layer。因此后文 27 Transformer blocks 会对应 54 layers。

## 2.1 Training Deep Networks via Residuals

Residual Learning. Residual learning [12] proves to be a critical technique in training deep networks as it allows gradients to bypass transformations. Specifically, each layer updates the hidden state as:

$$
\boldsymbol h_l=\boldsymbol h_{l-1}+f_{l-1}(\boldsymbol h_{l-1})
$$

Expanding this recurrence, the hidden state at layer $l$ is the sum of the embedding and all preceding layer outputs:

$$
\boldsymbol h_l=\boldsymbol h_1+\sum_{i=1}^{l-1}f_i(\boldsymbol h_i).
$$

The key insight behind residual connections is identity mapping: each layer preserves a direct path for both information and gradients to flow unchanged. During back-propagation, the gradient with respect to an intermediate hidden state is:

$$
\frac{\partial \mathcal L}{\partial \boldsymbol h_l}
=
\frac{\partial \mathcal L}{\partial \boldsymbol h_L}
\cdot
\prod_{j=l}^{L-1}
\left(\mathbf I+\frac{\partial f_j}{\partial \boldsymbol h_j}\right).
$$

Expanding this product yields $\mathbf I$ plus higher-order terms involving the layer Jacobians $\partial f_j/\partial h_j$. The identity term is always preserved, providing a direct gradient path from the loss to any layer regardless of depth.

> 💡 **关键展开式**: $\boldsymbol h_l=\boldsymbol h_1+\sum_i f_i(\boldsymbol h_i)$ 说明 residual stream 不是单纯“上一层状态”，而是一个把所有历史层输出混在一起的累加器。AttnRes 改的就是这个累加器。

Generalizing Residuals. While effective, the fixed unit coefficients in the residual update treat every layer’s contribution uniformly, offering no mechanism to adapt the mixing across depth. Highway networks [45] relax this by introducing learned element-wise gates:

$$
\boldsymbol h_l=(1-\boldsymbol g_l)\odot \boldsymbol h_{l-1}+\boldsymbol g_l\odot f_{l-1}(\boldsymbol h_{l-1})
$$

where $\boldsymbol g_l\in[0,1]^d$ interpolates between the transformation and the identity path. More generally, both are instances of a weighted recurrence

$$
\boldsymbol h_l=\alpha_l\cdot\boldsymbol h_{l-1}+\beta_l\cdot f_{l-1}(\boldsymbol h_{l-1}),
$$

with residual setting $\alpha_l=\beta_l=1$ and Highway setting $\alpha_l=1-g_l$, $\beta_l=g_l$.

> 💡 **为什么 Highway 还不够**: Highway 能决定“保留上一状态还是写入当前输出”，但它仍然只看 $\boldsymbol h_{l-1}$ 这个压缩后的单状态，不能把第 3 层、第 9 层、embedding 等 source 分开选择。

Limitations. Whether fixed or gated, both approaches share a fundamental constraint: each layer can only access its immediate input $\boldsymbol h_{l-1}$, a single compressed state that conflates all earlier layer outputs, rather than the individual outputs themselves. This entails several limitations: (1) no selective access: different layer types (e.g., attention vs. MLP) receive the same aggregated state, despite potentially benefiting from different weightings; (2) irreversible loss: information lost through aggregation cannot be selectively recovered in deeper layers; and (3) output growth: later layers learn increasingly larger outputs to gain influence over the accumulated residual, which can destabilize training. These limitations motivate a mechanism that lets each layer selectively aggregate information from all preceding layers.

## 中文批读

> 💡 **三类瓶颈**: 作者把 residual recurrence 的问题拆成 no selective access、irreversible loss、output growth。前两者是信息论/结构问题，第三个是训练稳定性问题；AttnRes 试图用同一个 depth-wise attention 同时处理。

> 💡 **和剪层现象的关系**: 如果很多深层可以被 prune 而损失很小，一种解释是这些层的输出虽然被累加进去，但没有被后续层有效区分和利用。AttnRes 的注意力权重图可以直接检查哪些层被选中。

> 💡 **从“加法”到“路由”**: 标准残差默认所有前层都应该贡献；AttnRes 默认每层应该学会路由。这个视角与 MoE/expert routing 的精神一致，只是路由对象变成了深度源。

## 本节公式速记

| 机制 | 更新式 | 仍然存在的问题 |
| --- | --- | --- |
| Standard residual | $\boldsymbol h_l=\boldsymbol h_{l-1}+f_{l-1}(\boldsymbol h_{l-1})$ | 所有历史输出固定权重为 1。 |
| Highway | $(1-\boldsymbol g_l)\odot\boldsymbol h_{l-1}+\boldsymbol g_l\odot f_{l-1}(\boldsymbol h_{l-1})$ | 动态 gate 只作用在上一压缩状态与当前输出之间。 |
| AttnRes 动机 | $\boldsymbol h_l=\sum_i\alpha_{i\to l}\boldsymbol v_i$ | 让每层直接选择历史 source，而不是被迫接收一个混合状态。 |
