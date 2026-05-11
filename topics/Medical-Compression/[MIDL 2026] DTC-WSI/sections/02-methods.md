[← 返回 README](../README.md)

# 02 - Methods

## 预览

方法部分定义了 DTC-WSI 的完整 token 数据流：CONCH/Virchow2 patch embedding -> importance network 估计 saliency -> 多阶段 bipartite soft matching 合并冗余 token -> soft/hard pruning 控制 token budget -> attention MIL 聚合 slide-level prediction。

# 2. Methods

# 2.1. Overview

Whole-slide images (WSIs) contain tens of thousands of patches, making conventional MIL and transformer models computationally prohibitive due to quadratic attention and high memory demands. To address this, we propose Dynamic Token Compression for Whole-Slide Images (DTC-WSI), a framework that learns to compress WSI patch embeddings in a task-aware manner. DTC-WSI progressively reduces tokens across multiple stages by combining (1) similarity-guided merging to fuse redundant patches and (2) importance-guided pruning to discard low-saliency regions. Compression is applied softly during training, enabling the importance network to learn reliable saliency estimates, and deterministically at inference for fast, scalable deployment. The final compact token set is aggregated using attention-based MIL to produce slide-level predictions.

> 💡 **方法入口**: 这段把 DTC-WSI 定义成“token preprocessor + MIL aggregator”。压缩不替代 MIL，而是先把 MIL 的输入 token set 从冗余大集合变成 compact discriminative set，后端仍可用 ABMIL/TransMIL 等聚合器。

# 2.2. Patch Extraction and Feature Encoding

A whole-slide image (WSI) is denoted by $x$ , which is tiled into $N$ non-overlapping patches $\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } \}$ after tissue detection and background removal. Each patch is processed by a pretrained encoder (CONCH (Lu et al., 2024)) to obtain a semantic feature embedding:

$$
h _ { i } ^ { ( 0 ) } = f _ { \theta } ( x _ { i } ) \in \mathbb { R } ^ { D } , \qquad i = 1 , \dots , N .
$$

To preserve spatial information, optional positional embeddings $p _ { i }$ concatenated with the visual features:

$$
\tilde { h } _ { i } ^ { ( 0 ) } = [ h _ { i } ^ { ( 0 ) } \parallel p _ { i } ] .
$$

forming the initial token matrix $\begin{array} { r } { { \cal H } ^ { ( 0 ) } = [ \tilde { h } _ { 1 } ^ { ( 0 ) } , \dots , \tilde { h } _ { N } ^ { ( 0 ) } ] ^ { \top } \in \mathbb { R } ^ { N \times ( D + 1 ) } . } \end{array}$

All supervision is provided at the slide level; no patch-level labels are used during training.

> 💡 **token 层级**: Patch encoder 是 frozen 或 pretrained feature extractor 的角色，DTC-WSI 不从像素端学习 compression。弱监督约束也很关键：没有 patch label，importance network 的 saliency 只能从 slide classification loss 和 sparse regularization 间接学到。

# 2.3. Importance Network

WSIs contain large regions of redundant or clinically irrelevant tissue, making it essential to quantify which patch embeddings contribute meaningfully to the slide prediction. The Importance Network $g _ { \phi }$ assigns a saliency score to each token at stage t:

$$
s _ { i } ^ { ( t ) } = g _ { \phi } ( \tilde { h } _ { i } ^ { ( t - 1 ) } ) ,
$$

where $g _ { \phi }$ is a two-layer MLP with GELU activation. Scores are normalized into importance weights:

$$
\alpha _ { i } ^ { ( t ) } = \frac { \exp ( s _ { i } ^ { ( t ) } ) } { \sum _ { j = 1 } ^ { N ^ { ( t ) } } \exp ( s _ { j } ^ { ( t ) } ) } ,
$$

which induces soft competition among tokens. Early in training, the distribution remains diffuse; as learning progresses, high-saliency tumor regions receive larger weights.

> 💡 **importance network 的双重作用**: $\alpha_i$ 既指导 pruning，也进入 merging 的加权平均和 utility penalty。换句话说，importance 不是只在最后排序 token，而是贯穿“哪些 token 能合并”和“合并后表示如何保留高 saliency 信息”。

# 2.4. Dynamic Multi-Stage Token Compression

To prevent catastrophic loss of diagnostic evidence, we adopt a multi-stage compression schedule:

$$
N ^ { ( 0 ) } = N  N ^ { ( 1 ) }  N ^ { ( 2 ) }  N ^ { ( 3 ) } ,
$$

where each $N _ { t + 1 }$ is determined by a retention ratio $r : N ^ { ( t + 1 ) } = r \cdot N ^ { ( t ) }$

The number of token merges required in stage $t$ is: $K ^ { ( t ) } = N ^ { ( t ) } - N ^ { ( t + 1 ) }$

Each stage consists of: 1) Bipartite soft matching for token fusion, and 2) Importanceguided pruning.

> 💡 **multi-stage 的价值**: 如果一次从 N 压到 0.4N，弱监督 saliency 还没稳定就可能 irreversible 删除关键 token。多阶段 schedule 把 compression 变成 curriculum，允许模型在每次轻度压缩后重新估计 saliency。

# 2.4.1. Bipartite Soft Matching for Token Fusion

To avoid the $O ( N ^ { 2 } )$ complexity of full similarity search, tokens in stage $t$ are partitioned into alternating subsets:

$$
\begin{array} { r } { A = [ \tilde { h } _ { 1 } ^ { ( t - 1 ) } , \tilde { h } _ { 3 } ^ { ( t - 1 ) } , \tilde { h } _ { 5 } ^ { ( t - 1 ) } , \dots ] , \quad B = [ \tilde { h } _ { 2 } ^ { ( t - 1 ) } , \tilde { h } _ { 4 } ^ { ( t - 1 ) } , \dots ] . } \end{array}
$$

For each aligned pair $( i , j )$ , cosine similarity is computed:

$$
\sin ( i , j ) = \frac { \langle \widetilde { h } _ { i } ^ { ( t - 1 ) } , \widetilde { h } _ { j } ^ { ( t - 1 ) } \rangle } { \Vert \widetilde { h } _ { i } ^ { ( t - 1 ) } \Vert \Vert \widetilde { h } _ { j } ^ { ( t - 1 ) } \Vert } .
$$

A merge utility incorporating importance consistency is defined as:

$$
u _ { i j } ^ { ( t - 1 ) } = \lambda \sin ( i , j ) - ( 1 - \lambda ) | \alpha _ { i } ^ { ( t ) } - \alpha _ { j } ^ { ( t ) } | .
$$

The Top- $K ^ { ( t ) }$ pairs are fused using importance-weighted averaging:

$$
\tilde { h } _ { l } ^ { ( t ) } = \frac { \alpha _ { i } ^ { ( t ) } \tilde { h } _ { i } ^ { ( t - 1 ) } + \alpha _ { j } ^ { ( t ) } \tilde { h } _ { j } ^ { ( t - 1 ) } } { \alpha _ { i } ^ { ( t ) } + \alpha _ { j } ^ { ( t ) } } .
$$

> 💡 **bipartite soft matching 批读**: 交错分组把候选匹配规模从全量 pairwise 降下来，但也意味着匹配空间受限。utility 的第二项非常关键：只看 cosine 会把形态相似但诊断贡献不同的 token 合并，importance gap penalty 则鼓励“相似且同等重要”的 token 才融合。

# 2.4.2. Importance-Guided Token Pruning

After token merging, low-saliency tokens are suppressed. During training, pruning is differentiable:

$$
m _ { l } ^ { ( t ) } = \sigma \Bigl ( \gamma ( \alpha _ { l } ^ { ( t ) } - \tau ) \Bigr ) , \quad \tilde { h } _ { l } ^ { ( t ) } = m _ { l } ^ { ( t ) } \tilde { h } _ { l } ^ { ( t ) } .
$$

During inference, deterministic Top- $N ^ { ( t ) }$ pruning is applied:

$$
H ^ { ( t ) } = \mathrm { T o p K } \Big ( H ^ { ( t - 1 ) } , \alpha ^ { ( t - 1 ) } , N ^ { ( t - 1 ) } \Big ) .
$$

> 💡 **training/inference 分离**: 训练阶段的 sigmoid gate 是为了连续可导和稳定优化；推理阶段 hard Top-K 才真正减少 token 数、显存和延迟。若训练也硬剪，importance network 很可能在早期 noisy saliency 下形成不可恢复的错误选择。

# 2.5. MIL Aggregation and Prediction

After $t$ compression stages, the final tokens $H ^ { ( t ) }$ are passed to an attention-based MIL module. Attention weights:

$$
a _ { i } = \frac { \exp ( w ^ { \top } \operatorname { t a n h } ( W h _ { i } ^ { ( t ) } ) ) } { \sum _ { j = 1 } ^ { M } \exp ( w ^ { \top } \operatorname { t a n h } ( W h _ { j } ^ { ( t ) } ) ) } .
$$

$\begin{array} { r } { z = \sum _ { i = 1 } ^ { N ^ { ( t ) } } a _ { i } \tilde { h } _ { i } ^ { ( t ) } } \end{array}$ vel representation is computed as a weighted sum of the compressed tokens,, where the attention weights emphasize diagnostically informative regions. This embedding is then passed through a linear classifier followed by a softmax layer to produce the slide-level prediction, $\hat { y } = \mathrm { s o f t m a x } ( W _ { c } z + b _ { c } )$ .

> 💡 **MIL aggregation 关系**: DTC-WSI 没有取消 attention MIL，而是让 attention MIL 面对更少、更干净的 token。这里也解释了为什么随机采样会失败：它同样减少 token，但没有保证剩下的集合仍覆盖 MIL 需要的诊断证据。

# 2.6. Loss Function

We supervise slide-level predictions using cross-entropy, $\mathcal { L } _ { \mathrm { c l s } } = \mathrm { C E } ( y , \hat { y } )$ , and encourage the importance network to assign sparse, selective saliency through an $\ell _ { 1 }$ regularizer, $\mathcal { L } _ { \mathrm { s p a r s e } } =$ $\textstyle \beta \sum _ { t = 0 } ^ { T ^ { \prime } } \| \alpha ^ { ( t ) } \| _ { 1 }$ . The full training objective combines both terms to promote discriminative yet compact representations, the composite loss is given as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \mathcal { L } _ { \mathrm { s p a r s e } } . } \end{array}
$$

A complete step-by-step description of the algorithm is provided in Appendix D.

> 💡 **sparsity loss 的读法**: 这个 regularizer 不是装饰项，Table 7 显示去掉后 token retained 从 0.40 变 0.52，推理从 410 ms 变 610 ms，Acc 从 98.3 降到 96.9。也就是说它直接决定 importance distribution 是否足够尖锐，从而影响 pruning 可执行性。

## Section 总结

| 模块 | 输入 | 关键操作 | 输出 / 风险 |
|---|---|---|---|
| Patch encoding | WSI patches | CONCH/Virchow2 feature extraction | token matrix，仍依赖 encoder 表征质量 |
| Importance network | stage tokens | MLP + softmax | saliency weights，弱监督下可能 noisy |
| Bipartite matching | odd/even token subsets | cosine + importance consistency | 可扩展 merge，但匹配空间受限 |
| Pruning | merged tokens | train soft gate / infer Top-K | 实际 token budget 与 speedup 来源 |
| MIL aggregation | compressed tokens | attention pooling | slide prediction，验证压缩是否保住诊断信号 |
