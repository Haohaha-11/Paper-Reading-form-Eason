[← 返回 README](../README.md)

# 02 Method

> 💡 **Hao 批注 - 方法总览**: ResTopoMIL 的核心设计哲学是"让容易的部分先被解释，困难的部分作为残差"。不是设计更复杂的图网络，而是改变空间分支的训练问题。三个关键设计：(1) 统计锚点(原型直方图)先学组合并冻结；(2) 拓扑残差(浅层GCN)在冻结锚点的残差上训练；(3) shuffle loss 强制残差分支不能退化为另一个排列不变纹理编码器。

> 💡 **Hao 批注 - 与 ReadySlide 的关联**: 这个"先学 easy signal 再学 hard residual"的设计与 ReadySlide 的 budgeted progressive coding (先分配 budget 给重要 patches，再逐步细化) 有相似的方法论气质——都是把优化问题解耦为更简单和更困难的两个阶段。

---

## 原文 Section 3: Preliminaries and Motivation

> 💡 **Hao 批注 - 形式化定义**: 关键区分：composition = patch 外观的经验分布(密集信号)；topology = patch 之间的空间关系(稀疏信号)。两者优化特性不同——composition 密集(每个 patch 都贡献)，topology 稀疏(只有少量空间关系携带决定性证据)。

A WSI is written as a bag

$$
{ \cal X } = \{ ( { \bf h } _ { i } , { \bf p } _ { i } ) \} _ { i = 1 } ^ { N } ,
$$

where $\mathbf { h } _ { i } \in \mathbb { R } ^ { d }$ is a patch embedding and $\mathbf { p } _ { i } \in \mathbb { R } ^ { 2 }$ is its slide coordinate. We distinguish two sources of label information. Composition is the empirical distribution of patch appearances, for example the abundance of tumor-like or stromal patches. Topology is the spatial relation among patches: clustering, gland formation, boundaries, invasive fronts, and similar architectural cues.

The distinction matters because the two signals optimize differently. Composition is dense: every patch contributes to a histogram-like summary, and slide labels are often partly predictable from the prevalence of visual phenotypes. Topology is sparse: a small set of spatial relations may carry the decisive evidence, and supervision arrives only after whole-slide aggregation. A jointly trained model is naturally drawn to the dense signal first. Once the loss has fallen, too little error may remain to train the spatial pathway. Standard validation can then look reassuring even when the explanation ignores architecture.

To test whether a model uses topology, a coordinate-shuffling operator pi keeps {h_i} fixed and permutes {p_i}. This preserves composition but destroys adjacency and tissue architecture. A model is defined as spatially blind on a structure-dependent task if its prediction or performance is nearly invariant under this perturbation. Robustness is desirable when a perturbation preserves semantics; here the perturbation removes part of the diagnostic evidence.

> 💡 **Hao 批注 - Figure 2 (stress test)**: TransMIL 有 Transformer 上下文机制，DS-MIL 是双流 MIL，但两者在坐标打乱后 AUC 几乎不变。注意这不是说模型预测不准——而是说高 AUC 可以仅靠组合信息达到，拓扑模块形同虚设。ResTopoMIL 则在打乱后有明显下降(这正是期望的行为)。

![](../images/aadd2167d33d0a9b8901b64e75292da39bf34ea1682d3fd57b07d049bbe7391e.jpg)

Figure 2: A coordinate-shuffling stress test. Patch embeddings are fixed while coordinates are permuted. On TCGA-BRCA, several MIL models remain almost unchanged after complete spatial shuffling. ResTopoMIL is more sensitive to this topology-destroying perturbation, as expected when structure is label-relevant.

This motivates the additive view

$$
F ( X ) \approx F _ { s t a t } ( \{ \mathbf { h } _ { i } \} ) + F _ { t o p o } ( \{ ( \mathbf { h } _ { i } , \mathbf { p } _ { i } ) \} ) ,
$$

where $F_{stat}$ captures permutation-invariant composition and $F_{topo}$ captures the remaining structure-dependent signal. The difficulty is not only architectural. If both terms are learned jointly, the first term is often easier to optimize and can reduce the loss before the second term learns. ResTopoMIL therefore makes Eq. (2) operational through staged residual training.

---

## 原文 Section 4: ResTopoMIL

> 💡 **Hao 批注 - 架构概述**: ResTopoMIL 先学组合解释(原型直方图→MLP→logits)，再学拓扑残差(KNN Graph→2-layer GCN→全局平均池化→残差logits)，配合 shuffle loss 确保拓扑分支真正编码空间信息。训练分两阶段，推理时无 shuffle view。

![](../images/093b45899472d63690a48286d614aa103e04981da9c15c9450f6e2e3859ff186.jpg)

Figure 3: Overview of the ResTopoMIL Framework.

### 4.1 Statistical Anchor (统计锚点)

> 💡 **Hao 批注 - 原型直方图**: 比 mean pooling 更丰富但仍排列不变。关键设计：(1) soft assignment(非 hard clustering)避免脆性；(2) MiniBatch K-Means 初始化(非全量 K-Means)保证可扩展性。好的锚点有两个作用：计算上提供稳定的残差目标，科学上区分"能被组合解释的"和"需要拓扑的"。

The first stream ignores coordinates by construction. Let $C = \{ \mathbf { c } _ { k } \} _ { k = 1 } ^ { K }$ be a learnable codebook initialized by sampled MiniBatch K-Means. For each patch embedding h_i, the assignment is

$$
a _ { i k } = \frac { \exp ( - \| { \bf h } _ { i } - { \bf c } _ { k } \| ^ { 2 } / \tau ) } { \sum _ { j = 1 } ^ { K } \exp ( - \| { \bf h } _ { i } - { \bf c } _ { j } \| ^ { 2 } / \tau ) } , \qquad { \bf a } _ { i } = [ a _ { i 1 } , \dots , a _ { i K } ] ^ { \top } , \qquad { \bf z } _ { s t a t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf a } _ { i } .
$$

An MLP maps $\mathbf { z } _ { s t a t }$ to logits $f_{stat}$. Similar to codebook-based MIL encodings [Wei et al., 2016], this stream measures how far one can go by counting visual phenotypes alone.

### 4.2 Topological Residual Branch (拓扑残差分支)

> 💡 **Hao 批注 - 图构建**: KNN graph 从物理坐标构建(非特征相似度)是关键选择——两个形态相似的肿瘤 patch 可能属于不同腺体，但物理相邻的 patch 定义局部架构。故意用浅层 GCN (仅 2 层)，让"优化问题是否解决"变得可观测，而非被更大容量掩盖。

A KNN graph $\mathcal{G} = (V, E)$ is built from coordinates $\mathbf{p}_i$ and processed by a two-layer GCN:

$$
\mathbf { H } ^ { ( l + 1 ) } = \sigma \left( \tilde { \mathbf { D } } ^ { - \frac { 1 } { 2 } } \tilde { \mathbf { A } } \tilde { \mathbf { D } } ^ { - \frac { 1 } { 2 } } \mathbf { H } ^ { ( l ) } \mathbf { W } ^ { ( l ) } \right) ,
$$

Global mean pooling obtains the graph-level representation:

$$
{ \bf z } _ { t o p o } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf H } _ { i } ^ { ( 2 ) } , \qquad f _ { t o p o } = { \bf W } _ { t o p o } { \bf z } _ { t o p o } + { \bf b } _ { t o p o } .
$$

### 4.3 Residual Training Objective (残差训练目标)

> 💡 **Hao 批注 - 两阶段训练**: 这是 ResTopoMIL 最核心的设计。Stage 1 仅训练统计流(交叉熵)，Stage 2 冻结统计流(sg[·]停止梯度)，训练拓扑流。stop-gradient 有两个效果：(1) 统计流不能再吸收 Stage 2 的误差，(2) 拓扑流获得稳定的残差目标而非移动的联合最优。

> 💡 **Hao 批注 - Shuffle Loss 设计精髓**: 不是通用的对比学习正则项。负样本保持所有 patch 外观、所有标签、所有 bag-level 组合不变，只破坏坐标诱导的图。这意味着满足 margin 的唯一方式是编码空间排列而非另一个外观摘要。这也是为什么不需要 tumor mask 或病理学家标注——负样本自动生成。

Stage 1 trains only the statistical stream with standard cross-entropy. Stage 2 freezes this stream:

$$
f ( X ) = \mathrm { s g } [ f _ { s t a t } ( X ) ] + f _ { t o p o } ( X ) ,
$$

where sg[.] stops gradients.

The shuffle-based constraint: let $\tilde{X}$ be obtained by permuting coordinates while keeping all patch embeddings fixed. The graph representation of X is required to differ from that of $\tilde{X}$:

$$
\mathcal { L } _ { t e x t u r e } = \operatorname* { m a x } ( 0 , m - [ 1 - \mathrm { s i m } ( \mathbf { z } _ { t o p o } , \tilde { \mathbf { z } } _ { t o p o } ) ] ) .
$$

The final Stage-2 objective:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { c l s } ( \mathrm { s g } [ f _ { s t a t } ] + f _ { t o p o } , Y ) + \lambda \mathcal { L } _ { t e x t u r e } .
$$

At inference, no shuffled view is constructed. The model computes statistical logits and topological residual logits once, then sums them.

### 4.4 Why Decoupling Helps (为什么解耦有效)

> 💡 **Hao 批注 - 理论分析**: 附录 B 的两个命题是方法设计层面的分析(非独立理论贡献)。Proposition 1：拓扑更新的范数被残差误差上界控制——如果组合信号迅速减小残差，拓扑分支即使存在也收不到有效监督。Proposition 2：冻结 f_stat 后，Stage 2 的交叉熵最小化等价于最大化条件互信息 I(Z_topo; Y | Z_stat) 的变分下界。但条件互信息本身不保证 Z_topo 是空间信息——这就是为什么还需要 shuffle loss。

**Proposition 1 (Residual-error gating of the topological update):** For an additive MIL logit $f = f_{stat} + f_{topo}$ trained with cross-entropy, the topological update is

$$
\nabla _ { \theta _ { t } } \mathcal { L } = \mathbb { E } [ ( \hat { p } _ { \theta } ( X ) - Y ) \nabla _ { \theta _ { t } } f _ { t o p o } ( X ) ] .
$$

Its norm is bounded by the remaining prediction error:

$$
\| \nabla _ { \theta _ { t } } \mathcal { L } \| \leq \left( \mathbb { E } ( \hat { p } _ { \theta } ( X ) - Y ) ^ { 2 } \right) ^ { 1 / 2 } \big ( \mathbb { E } \| \nabla _ { \theta _ { t } } f _ { t o p o } ( X ) \| _ { F } ^ { 2 } \big ) ^ { 1 / 2 } .
$$

**Proposition 2 (Residual branch as conditional label information):** After $Z_{stat}$ and $f_{stat}$ are fixed, optimizing the Stage-2 decoder minimizes a variational upper bound on $H(Y | Z_{stat}, Z_{topo})$ and therefore maximizes a lower bound on $I(Z_{topo}; Y | Z_{stat})$.

The chain rule alone would still allow $Z_{topo}$ to encode another compositional statistic. ResTopoMIL therefore adds the coordinate-specific constraint $\mathcal{L}_{texture}$: the negative view preserves all patch appearances and labels but corrupts the coordinate-induced graph, so a branch that ignores topology cannot reliably satisfy the margin.

> 💡 **Hao 批注 - 为什么常见 tricks 不足以解决**: 附录 B.3 详细分析了为什么 Multi-LR (增大图学习率)、statistical dropout、curriculum scheduling、hard instance mining 都只是软化而非解决梯度竞争。Multi-LR 的更新仍被同一缩小的残差控制；dropout 注入噪声；curriculum 改变相对速度但不解耦。ResTopoMIL 的 stop-gradient 是更强的干预——将残差测量基准从移动目标改为固定锚点。

### Appendix B.3: Why Common Optimization Heuristics Are Not Equivalent

- **Multi-LR**: The update is still gated by the same shrinking residual $r_{\theta_s,\theta_t}(X,Y)$.
- **Statistical Dropout**: Can expose topology but also injects noise into useful compositional evidence.
- **Curriculum Scheduling**: Changes relative speed but the residual target remains moving.
- **Hard Instance Mining**: Reweights samples but doesn't prevent the easy stream from absorbing part of the residual.
- **ResTopoMIL's stop-gradient**: $r_{res}(X,Y) = \hat{p}(Y=1 | sg[f_{stat}(X)], f_{topo}(X)) - Y$ — residual measured against a fixed compositional anchor. The graph branch no longer competes with a moving statistical predictor.
