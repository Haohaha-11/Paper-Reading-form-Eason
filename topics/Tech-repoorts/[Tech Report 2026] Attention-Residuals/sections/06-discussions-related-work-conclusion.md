[← 返回 README](../README.md)

# 06. Discussions, Related Work, and Conclusion

## 6. Discussions

## 6.1 Sequence-Depth Duality

Residual connections propagate information over depth via a fixed recurrence $h_l=h_{l-1}+f_{l-1}(h_{l-1})$, much as RNNs propagate information over time. Test-Time Training (TTT) [46] formalizes the sequence side of this analogy, casting each recurrent step as gradient descent on a self-supervised loss:

$$
\mathbf W_t=\mathbf W_{t-1}-\eta\nabla\ell(\mathbf W_{t-1};\mathbf x_t),
$$

where a slow network parameterizes $\ell$ and the state $\mathbf W$ is updated once per token. When $f$ is linear, this reduces to vanilla linear attention $\mathbf S_t=\mathbf S_{t-1}+k_t v_t^\top$. The standard residual exhibits the same additive form along depth, with $h_l$ serving as the state and each layer $f_l$ acting as one “gradient step.”

As noted by [4], this duality extends to richer variants. Data-dependent gates on the sequence side correspond to Highway networks on the depth side; the delta rule corresponds to DDL; and MRLA mirrors GLA’s gated linear attention. These methods all refine the recurrent update while remaining within the recurrence paradigm. AttnRes goes a step further and replaces depth-wise recurrence with direct cross-layer attention, just as Transformers replaced temporal recurrence with self-attention. Since the number of layers in current architectures remains well within the practical regime of softmax attention, we adopt vanilla depth-wise attention. Incorporating more expressive yet memory-efficient alternatives is a natural direction for future work.

> 💡 **讨论主线**: 作者不是说 residual “像” RNN 就结束了，而是把 sequence-side 的 recurrence → attention 迁移到 depth-side。这个类比支撑了为什么用 softmax attention，而不是只加门控或缩放。

## 6.2 Residual Connections as Structured Matrices

The residual variants discussed above can all be viewed as weighted aggregations over previous layer outputs. We formalize this with a depth mixing matrix $\mathbf M\in\mathbb R^{L\times L}$, where $\mathbf M_{i,l}$ is the weight that layer $l$ assigns to the output of layer $i$. The variants differ in how these weights arise (fixed, learned, or input-dependent) and whether $\mathbf M$ is constrained to low rank or allowed to be dense. The semiseparable rank of $\mathbf M$ [8] offers a unified lens for comparing them.

Concretely, the input to layer $l$ is

$$
\boldsymbol h_l=\sum_{i=0}^{l-1}\mathbf M_{i,l}\boldsymbol v_i,
$$

where $\boldsymbol v_0=\boldsymbol h_1$ (embedding) and $\boldsymbol v_i=f_i(\boldsymbol h_i)$ for $i\geq1$. Fig. 9 visualizes $\mathbf M$ for representative methods.

![Depth mixing matrix panel 1](../images/420c0487430a40289ccaeac40e07248be4cc77aa1db75549fe948918a3846d0e.jpg)

![Depth mixing matrix panel 2](../images/7cc6cc55cfdfcb714eacd195bf8092ad6138ced45b83b39c173f92a6c0891616.jpg)

Figure 9: Depth mixing matrices $\mathbf M$ for residual variants. AttnRes panels show unnormalized $\phi$ scores; background colors group entries that share the same source or source block.

• Standard residual [12]. Expanding $h_l=h_{l-1}+f_{l-1}(h_{l-1})$ gives $\boldsymbol h_l=\sum_{i=0}^{l-1}\boldsymbol v_i$, so $\mathbf M_{i\to l}=1$ for all $i<l$ and $\mathbf M$ is an all-ones lower-triangular matrix.

• Highway [45]. With scalar gates for clarity, $\boldsymbol h_l=(1-g_l)\boldsymbol h_{l-1}+g_l f_{l-1}(\boldsymbol h_{l-1})$. The weights are products of carry gates, making $\mathbf M$ 1-semiseparable and input-dependent, but still recurrence-bound.

• (m)HC [72, 59]. These methods maintain $m$ parallel streams $\mathbf H_l\in\mathbb R^{d\times m}$ and update them through learned transition matrices. The $m\times m$ transitions render $\mathbf M$ $m$-semiseparable, increasing depth-axis state capacity while keeping recurrent structure.

• Full AttnRes computes $\mathbf M_{i,l}=\alpha_{i\to l}$ via $\phi(\boldsymbol w_l,\boldsymbol k_i)=\exp(\boldsymbol w_l^\top\mathrm{RMSNorm}(\boldsymbol k_i))$ with softmax normalization, where $\boldsymbol k_i=\boldsymbol v_i$ are input-dependent layer outputs, yielding a dense, rank-$L$ mixing matrix.

• Block AttnRes partitions layers into $N$ blocks. For sources in a completed earlier block, all share the block-level key/value $b_n$, so $\mathbf M_{i,l}=\alpha_{n,l}$ for every source in that block. Within the current block, each layer additionally attends over the evolving partial sum $b_n^{i-1}$. The effective rank lies between $N$ and $N+S$, interpolating between standard residual $(N=1)$ and Full AttnRes $(N=L)$.

Practicality. The structured-matrix perspective serves two purposes. First, it enables analytical insights that are not apparent from the recurrence form alone. The input-dependent $\mathbf M$ of AttnRes reveals depth-wise attention sinks, where certain layers consistently attract high weight regardless of input, mirroring the same phenomenon in sequence-wise attention. Second, it informs new designs by exposing which properties of the kernel $\phi$ matter. When $\phi$ decomposes through a feature map, depth-wise attention collapses into a recurrence, precisely the structure underlying several linear-attention-style residual variants.

Prior Residuals as Depth-Wise Linear Attention. The structured-matrix perspective further relates to the sequence-depth duality by showing that existing residual variants are, in effect, instances of linear attention over the depth axis. The unrolled (m)HC weight admits an attention interpretation in which $\alpha_l$ plays the role of a query issued by layer $l$, $\beta_i$ serves as a key summarizing the contribution of layer $i$, and cumulative transition matrices act as depth-relative positional operators. The $m$ parallel streams correspond to state expansion along the depth axis. AttnRes acts as depth-wise softmax attention.

> 💡 **structured matrix 的价值**: 这个视角把 residual、Highway、mHC、Full/Block AttnRes 都写成 $\mathbf M$。差别不再是“有没有残差”，而是 $\mathbf M$ 的权重来源、输入依赖性、rank 和可访问 source 范围。

> 💡 **linear attention vs softmax attention**: 作者把既有 residual variants 解释为 depth-wise linear attention，把 AttnRes 解释为 depth-wise softmax attention。这个 framing 是论文理论叙事的核心。

## 7. Related Work

Normalization, Scaling, and Depth Stability. The standard residual update $\boldsymbol h_{l+1}=\boldsymbol h_l+f_l(\boldsymbol h_l)$ [12] presents a fundamental tension between normalization placement and gradient propagation. PostNorm [52] maintains bounded magnitudes but distorts gradients, as repeated normalization on the residual path compounds into gradient vanishing at depth [60]. PreNorm [34, 60] restores a clean identity path yet introduces unbounded magnitude growth: since $\|\bar{\boldsymbol h}_l\|$ grows as $O(L)$, each layer’s relative contribution shrinks, compelling deeper layers to produce ever-larger outputs and limiting effective depth [27]. Subsequent work reconciles both desiderata via scaled residual paths [54], hybrid normalization [73], amplified skip connections [4], or learned element-wise gates [45]. AttnRes sidesteps this tension by replacing the additive recurrence with selective aggregation over individual earlier-layer outputs, avoiding both the cumulative magnitude growth of PreNorm and the repeated scale contraction of PostNorm.

Multi-State Recurrence. All single-state methods above condition layer $l$ only on $\boldsymbol h_{l-1}$, from which individual earlier-layer contributions cannot be selectively retrieved. Several methods address this by widening the recurrence to multiple parallel streams: Hyper-Connections [72] and mHC [59] maintain $m$ streams with learned mixing matrices; DDL [67] maintains a matrix state updated via a delta-rule erase-and-write mechanism; SiameseNorm [27] maintains two parameter-shared streams, one PreNorm and one PostNorm, to preserve identity gradients and bounded representations. While these methods alleviate information compression, they still condition on the immediate predecessor’s state; AttnRes is orthogonal, providing selective access to individual earlier-layer outputs while remaining compatible with any normalization or gating scheme.

Cross-Layer Connectivity. A separate line of work bypasses the single-state bottleneck by giving each layer direct access to individual earlier-layer outputs. Static-weight methods include DenseNet, ELMo, DenseFormer, and ANCRe. Input-dependent methods include MUDDFormer and MRLA, though the latter is closer to linear attention than softmax-based retrieval. Other targeted designs include Value Residual Learning, LAuReL, and Dreamer. AttnRes combines softmax-normalized, input-dependent weights with selective access to all preceding layers through a single $d$-dimensional pseudo-query per layer, and introduces a block structure reducing practical cost. Cache-based pipeline communication and a two-phase computation strategy make Block AttnRes practical at scale with negligible overhead.

> 💡 **Related Work 的定位**: AttnRes 同时回应三类工作：normalization/scaling 解决稳定性，多状态 recurrence 解决状态容量，cross-layer connectivity 解决直接访问。它的卖点是把“输入相关、softmax 竞争、跨层访问、低工程开销”组合在一起。

## Conclusion

Inspired by the duality between sequence and depth, we introduce AttnRes, which replaces fixed, uniform residual accumulation with learned, input-dependent depth-wise attention. We validate the method through ablation studies and scaling law experiments, showing that its gains persist across scales. Because Full AttnRes must access all preceding layer outputs at every layer, the memory footprint of cross-layer aggregation grows as $O(Ld)$, which is prohibitive for large-scale models on current hardware. We therefore introduce Block AttnRes, which partitions layers into $N$ blocks and attends over block-level representations. Empirically, using about 8 blocks recovers most of the gains of Full AttnRes, while finer-grained blocking remains a promising direction as future hardware constraints relax. Together with cross-stage caching and a two-phase computation strategy, Block AttnRes is practical at scale, incurring only marginal training overhead and minimal inference overhead.

## 中文收束

> 💡 **结论一句话**: AttnRes 不是取消 residual，而是把 residual stream 从固定累加器升级成深度维选择器；Block AttnRes 是当前硬件条件下的可训练、可推理版本。

> 💡 **未来方向**: 如果 interconnect 和显存约束放松，Full AttnRes 或更细 block 会更有吸引力；如果继续追求低成本，可能会探索 linear-complexity 的 depth attention kernel。
