[← 返回 README](../README.md)

# 3.1. Dynamic Depth Pruning

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 TinySR 主线的关系**: TinySR 面向 Real-ISR 的实时落地，把已有 one-step diffusion teacher 通过深度剪枝、VAE 压缩、模块移除和缓存策略压成轻量单步模型。

---

Depth Pruning Formulation. Consider an $N$ -layers transformer parameterized by $\Phi ~ = ~ [ \phi _ { 1 } , \phi _ { 2 } , \ldots , \phi _ { N } ] ^ { T }$ , where each $\phi _ { i } ~ \in { \mathbb { R } } ^ { D }$ . Our goal is to identify an optimal binary mask $m \in \{ 0 , 1 \} ^ { N }$ that enables effective pruning while maintaining strong Super-Resolution (SR) performance. The pruning mechanism is defined by:

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Equation 1](../images/2f4b64cb94746d7bc5f9b9d82da7e732f9eb2cb917a6466e18cda1c198f29765.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

here, $x _ { i }$ is the input to the $i$ -th layer, and $\phi _ { i } ( x _ { i } )$ is its output. Traditional methods for determining the masking scheme commonly rely on heuristic-based layer importance (e.g., sensitivity analysis and metrics-based selection) or empirical manual configurations. However, these carefully designed strategies typically overlook the intricate and interdependent layer dynamics within SR diffusion transformers, leading to suboptimal optimization and potentially weak recoverability in the retained layers.

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

Inspired by TinyFusion [15], instead of pursuing models that rely on immediate high-importance feedback, we propose identifying candidate layers with strong recoverability, enabling more efficient teacher-student knowledge distillation. We formalize the pruning process as a bi-level optimization problem:

> 💡 **批注**: 这是蒸馏逻辑：用 teacher 或 score regularization 把多步/大模型能力迁移给单步模型。

![Equation 2](../images/4a35de205e78361a392505c91eef982f17c919a54183f24722c2747fab726070.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

This objective is to identify an optimal mask that minimizes the loss function $\mathcal { L }$ during the optimization process. Since discrete mask selection is non-differentiable, we reparameterize each mask option with a learnable probability parameter $p ( m )$ and then use Gumbel-Softmax [22] trick $\mathcal { G }$ to do differentiable sampling:

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Equation 3](../images/8a1314d9a7da004dcefda89a4dbf8a8d05e861918838e982e0f5a0fd4f53020f.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $g _ { i }$ is random noise drawn from the Gumbel distribution and $\tau$ refers to the temperature term. The probabilistic sampling of $m$ can be achieved by $m = \mathcal { G } ( \boldsymbol { p } ( \boldsymbol { m } ) ) ^ { T } \cdot \mathcal { M } , \mathcal { M }$ represents the complete search space. For the final pruning decision, we pick $m$ with maximum $p ( m )$ , as it reveals the strongest recovery. We call this process mask learning.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Search Space Dilemma. A significant challenge in standard mask learning arises from the combinatorial explosion. Let $M : N$ denote the selection of $M$ from $N$ layers, and $C _ { N } ^ { M }$ denote the set of combinations. As $N$ grows, pruning $50 \%$ will show an explosive trend. For instance, pruning $50 \%$ of a 24-layer transformer $C _ { 2 4 } ^ { 1 2 }$ leads to 2,704,156 possible solutions, making direct probabilistic optimization expensive. Naive approaches address this by partitioning an $N$ -layer network into $K$ non-overlapping blocks of size $B$ , enforcing uniform pruning within each. Assuming statistical independence of the pruning decisions, the total probability $p ( m )$ can be factored into the product of local probabilities $\overset { \cdot } { p } ( \overset { \cdot } { m } ^ { ( j ) } )$ :

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Equation 4](../images/26bb8065c8deecba52718352450a545420aa7eea74112ed1af3557c106b0fe13.jpg)
*Equation 4: Equation extracted by MinerU.*

> 💡 **Equation 4 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

While this simplifies the search, it drastically curtails the space of possible solutions. The cardinality of this feasible

![Figure 4.](../images/09d5f4b4fc96a0612a36980f44cca4db031248620c0c304a852f8c1a097f0ba9.jpg)
*Figure 4.: Figure 4. Our proposed mask learning method performs a probability-based decision for candidate solutions, jointly optimized with network weight updates. $p ( m )$ characterizes the probability distribution for each block’s pruning scheme. We perform a transformation probability $q ( t )$ to facilitate dynamic interaction between masks in different blocks, namely Dynamic Inter-block Activation. Leveraging the transfer scheme, we employ an Expansion-Corrosion Strategy to determine the final mask by expanding on elements with maximum marginal probability and corroding those with low marginal probability.*

> 💡 **Figure 4. 批读**: 这张图通常承担方法动机、框架或视觉对比的作用。阅读时重点看它证明的是质量、速度还是可控性，而不是只看视觉效果。

subspace, denoted $\mathcal { M } _ { v a l i d }$ is given by:

![Equation 5](../images/ecdda0943f08bad69602296cd0c7971a2084308e993f87e6c837831ab3cd30fd.jpg)
*Equation 5: Equation extracted by MinerU.*

> 💡 **Equation 5 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

The accessible fraction of the search space is $\begin{array} { r l } { \frac { | \mathcal { M } _ { \mathrm { v a l i d } } | } { | \mathcal { M } | } } & { = } \end{array}$ $\frac { 4 6 , 6 5 6 } { 2 , 7 0 4 , 1 5 6 } \approx 1 . 7 2 5 \%$ . This demonstrates that over $98 \%$ of all possible pruning masks are rendered unreachable by the imposition of this seemingly innocuous local constraint. Consequently, the true optimal pruning scheme may be inadvertently excluded from this narrowly defined search space.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Dynamic Inter-block Activation. To transcend the limitations of this static partitioning, we introduce a novel pruning framework based on dynamic inter-block activation, as shown in Fig. 4. The core motivation is to relax the rigid local constraints and empower the pruning process to discover its own optimal layer distribution across the network. Instead of confining the search to the highly-constrained $\mathcal { M } _ { v a l i d }$ , our method begins within this space yet is allowed to explore beyond it. We achieve this by defining a set of probabilistic transformation operators, $T$ , governed by distribution parameters $q ( t )$ . Specifically, $T _ { j  h } ( m , k )$ transforms $m$ into $m ^ { \prime }$ by pruning $k$ active layers from block $j$ (or $h$ ) while restoring $k$ layers in $h$ (or $j$ ). This mechanism ensures that the total number of active layers remains constant across the pair of blocks $( j , h )$ while dynamically adjusting their individual sparsity profiles. The transformation distribution parameters $q ( t )$ , are also learnable and applicable to sample a specific scheme through $\mathcal { G }$ . For example, we assume $k = 1 , h = j + 1$ and define the option space as $\mathcal { M } ^ { \prime } = \{ m ^ { - } , m , m ^ { + } \}$ , where $m$ is the pruning mask sampled by $p ( m )$ , and $m ^ { - }$ (resp. $m ^ { + }$ ) denotes $T _ { j  j + 1 } ( m , 1 )$ (resp. $T _ { j  j + 1 } ( m , 1 ) )$ ). The specific mask can be represented as $m ^ { \prime } = \mathcal { G } ( q ( t ) ) ^ { T } \cdot \mathcal { M } ^ { \prime }$ . And the total probability can be expressed as:

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

![Equation 6](../images/7c4aa72b6b6e15833e00abcc0daa43122bc76512829feaa341fd86e73cc6c05a.jpg)
*Equation 6: Equation extracted by MinerU.*

> 💡 **Equation 6 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

where $r$ is the probability of reaching $m$ through $t$ , and is related to expansion-corrosion strategies.

Expansion-Corrosion Strategy. Instead of random perturbation-based expansion or corrosion, our approach utilizes the maximum marginal probability:

![Equation 7](../images/3028007f92d90c43d48dfed7da01867aa59db360126d605805afe195a9808dcb.jpg)
*Equation 7: Equation extracted by MinerU.*

> 💡 **Equation 7 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

which is derived from training priors. Specifically, we start by sorting the values of $\pi _ { i }$ in descending order. Then we iteratively select candidate layers, corresponding to these sorted values, to form our candidate mask $\hat { m }$ , which is guaranteed to be $| m \vee \hat { m } | _ { 1 } - | m | _ { 1 } = k$ for expansion and $| m | _ { 1 } - | m \wedge \hat { m } | _ { 1 } = k$ for erosion. To ensure backpropagation, we replace the original intersection and union operations with the following simplified computations:

![Equation 8](../images/66efc45d7eee73d236b0cce01fe828877c6419a9605928906760c358a2dbedf0.jpg)
*Equation 8: Equation extracted by MinerU.*

> 💡 **Equation 8 批读**: 这类公式通常定义 forward/reverse process、loss 或 alignment 目标；建议把每个符号对应到输入、teacher/student、控制变量。

here, $\odot$ represents the element-wise product. This continuous relaxation maintains the meaning of expansion or corrosion while allowing for gradient flow, which is crucial for training our pruning models.

> 💡 **批注**: 这是效率相关段落：单步只是减少采样次数，模型结构和 VAE 仍可能是主要延迟来源。

Pruning Decision. Unlike traditional mask learning, where decisions are made directly from $p ( m )$ , our approach determines the transformations for each block based on maximum $q ( t )$ and then selects the final mask based on $\pi$ .

> 💡 **批注**: 注意 latent diffusion 架构路径：LQ/HR 往往先被 VAE 编码，再在 latent 空间完成 denoising 或调制。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Speedup | up to 5.68× over TSD-SR teacher |
| Parameter reduction | 83% reduction claimed in abstract |
| Inference regime | one-step Real-ISR |
| Compression targets | Diffusion backbone + VAE + prompt/time modules |
