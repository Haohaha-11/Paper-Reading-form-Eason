[← 返回 README](../README.md)

## 📌 预览
MLP final projection fast weights、NTP-aligned objective、chunk-wise/context-parallel 实现。

---

# 3 In-Place Test-Time Training

To satisfy the desiderata outlined in section 2, we introduce In-Place Test-Time Training (In-Place TTT), a framework designed to unlock TTT capabilities for LLMs. We first present our overall framework, which resolves architectural incompatibility via an in-place design that repurposes existing MLP blocks, while ensuring computational efficiency with a chunk-wise update mechanism (section 3.1). We then detail our novel LM-aligned objective, which is explicitly designed for LLMs by aligning with the Next-Token Prediction (NTP) goal (section 3.2). Following this, we provide a theoretical analysis of our objective’s superior properties (section 3.3) and conclude with practical implementation details (section 3.4).

# 3.1 Overall Framework

Repurposing MLP Blocks for In-Place Adaptation. Previous TTT research has largely positioned it as a potential solution to replace the attention mechanism. However, these prior studies were typically conducted at moderate scales, a regime vastly different from that of modern, billion-parameter LLMs. Consequently, replacing the core attention mechanism—whose learned properties are critical to an LLM’s capabilities—is a high-risk architectural modification. Moreover, introducing any new, randomly-initialized layer also creates a conflict with the billions of trained parameters of LLMs, necessitating costly and often impractical retraining to resolve this imbalance.

> 💡 **架构选择**: 选择 MLP 而不是 attention 是保守但关键的工程判断：attention 保留原本长程交互能力，MLP final projection 额外承担在线记忆。

Our core insight is to sidestep these challenges entirely. Instead of replacing or adding components, we repurpose a ubiquitous module–the Multi-Layer Perceptron (MLP) block–to also serve as the fast weights. Recalling the TTT formulations in section 2, there exist no constraints on the choice of fast weights, i.e., any parameters can serve as fast weights updated via the TTT mechanism. In particular, the MLP blocks in Transformers can also be viewed as a form of key-value memory [22], functioning as a “slow weights" for the vast, general knowledge acquired during pre-training. It is therefore a natural extension to leverage this same component to also function as the adaptive "fast weights", dynamically internalizing transient, in-context information at inference time.

Formally, we adapt the widely used gated MLP architecture [24, 57]. Given the hidden representation $\mathbf { H }$ , the gated MLP computes its output representation $\mathbf { O } = \left( \phi ( \mathbf { H } \mathbf { W } _ { \mathrm { g a t e } } ^ { \top } ) \odot ( \mathbf { H } \mathbf { W } _ { \mathrm { u p } } ^ { \top } ) \right) \mathbf { W } _ { \mathrm { d o w n } } ^ { \top }$ . In our framework, we treat the input projections $\mathbf { W } _ { \mathrm { u p } }$ and $\mathbf { W } _ { \mathrm { g a t e } }$ as frozen slow weights, while repurposing the final projection matrix, $\mathbf { W } _ { \mathrm { d o w n } }$ , as the adaptable fast weights. By exclusively updating $\mathbf { W } _ { \mathrm { d o w n } }$ in-place, we preserve the model’s architectural integrity, transforming TTT from a disruptive restructuring into a lightweight, “drop-in" enhancement for LLMs.

> 💡 **fast weights 定位**: $W_{up}$ 和 $W_{gate}$ 冻结为 slow weights，只有 $W_{down}$ 原位更新。由于 $W_{down}$ 是 MLP 输出投影，更新它会直接改变每个 chunk 的残差流注入信息。

Efficient Adaptation with Chunk-Wise Updates. Beyond architectural compatibility, our in-place design also unlocks significant computational efficiencies. Conventional TTT methods, by aiming to replace the attention mechanism, were bound to inefficient per-token updates to enforce strict causality and perform fine-grained token mixing. Chunk-wise updating approaches have been explored by recent works to achieve acceleration [3, 30, 36, 49]. Our framework also follows to sidestep the trade-off entirely. Since we only adapt the MLP blocks and leave the attention layers intact, we are liberated from the per-token constraint, enabling a far more efficient chunk-wise update strategy which further bypasses the small chunk constraints (our ablation study results (Section 4.3) also verify that our framework is naturally well-suited for chunk-wise—and specifically large chunk-wise—updates, achieving optimal performance with chunk sizes of 512 to 1024).

> 💡 **大 chunk 为什么可行**: standalone TTT 需要小 chunk 扮演 token mixer；这里 token mixing 仍由 attention 做，TTT 主要存储可预测信息，所以 512/1024 级别的大 chunk 可以同时保性能和吞吐。

The process operates as follows. Given the intermediate activations $\mathbf { Z } = \phi ( \mathbf { H } \mathbf { W } _ { \mathrm { g a t e } } ^ { \top } ) \odot ( \mathbf { H } \mathbf { W } _ { \mathrm { u p } } ^ { \top } ) \in \mathbb { R } ^ { n \times d _ { \mathrm { f f } } }$ and corresponding value targets and outputs $\mathbf { V } , \mathbf { O } \in \mathbb { R } ^ { n \times d _ { \mathrm { m o d e l } } }$ , we partition them into $k$ non-overlapping chunks dimen of size $C$ , denoted et $\mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) }$ $\begin{array} { r } { \ b { \Pi } _ { [ i ] } = \ b { \Pi } _ { i C + 1 : ( i + 1 ) C } \in \mathbb { R } ^ { C \times d ^ { \prime } } } \end{array}$ be the fast weights state before processing chunk for and $i$ and $d ^ { \prime }$ $\mathbf { W } _ { \mathrm { d o w n } } ^ { ( 0 ) } = \mathbf { W } _ { \mathrm { d o w n } }$ being their corresponding . For each $i \in [ k ]$ , we perform two sequential operations:

1. Apply Operation: The current state of the fast weights $\mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) }$ are used to process chunk $\mathbf { Z } _ { [ i ] }$ , i.e., $\mathbf { O } _ { [ i ] } = \mathbf { Z } _ { [ i ] } ( \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } ) ^ { \top }$ .

> 💡 **严格因果**: 对 chunk $i$ 先 apply 当前 $W_{down}^{(i)}$，再用该 chunk 更新到 $W_{down}^{(i+1)}$，避免同一 chunk 的未来 token 信息泄漏到自身输出。

2. Update Operation: The fast weight $\mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) }$ are updated using $\mathbf { Z } _ { [ i ] }$ as keys and $\mathbf { V } _ { [ i ] }$ as values, which step with a loss function $\mathcal { L }$ and a learning rate $\eta$ $\mathbf { W } _ { \mathrm { d o w n } } ^ { ( i + 1 ) } =$ $\mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } - \eta \nabla _ { \mathbf { W } } \mathcal { L } \left( \mathbf { Z } _ { [ i ] } ( \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } ) ^ { \top } , \mathbf { v } _ { [ i ] } \right) .$

This chunk-wise update strategy is designed for modern hardware, similar to prior attempts [3, 30, 36, 49].

Moreover, due to our in-place MLP adaptation, we can use a large chunk size $C$ to process large blocks of tokens at once, thereby highly leveraging parallelism and utilizing the computational power of GPUs or TPUs.

# 3.2 LM-Aligned Objective

With the efficient, in-place adaptation framework established, the performance of In-Place TTT now hinges on the design of its learning objective. In this subsection, we introduce our Language Modeling-Aligned objective, which is explicitly tailored for LLMs.

Prior TTT approaches typically use a reconstruction target, e.g., $\mathcal { L } ( f _ { \mathbf { W } } ( k ) , v )$ where both $k$ and $v$ are linear projection outputs of the same input token $x$ [47, 48, 67], which encourages the model to simply memorize the current token’s representation. We argue that this is suboptimal for language modeling tasks. Instead, we propose to align the objective with the Next-Token Prediction (NTP) goal governing LLMs.

> 💡 **从 reconstruction 到 NTP**: 传统目标把当前 token 写入记忆；这里通过 Conv1D/token embedding 生成含未来 token 信息的 $\hat V$，把写入内容对齐到 autoregressive LM 的预测目标。

To achieve this, we specify the target $v$ to include future token information. Formally, we derive our target $\hat { \mathbf { V } } = \operatorname { C o n v } 1 \mathrm { D } ( \mathbf { X } _ { 0 } ) \mathbf { W } _ { \mathrm { t a r g e t } }$ , where $\mathbf { X } _ { 0 } \in \mathbb { R } ^ { n \times d _ { \mathrm { m o d e l } } }$ denotes the token embedding, Conv1D $( \cdot )$ is the 1D Convolution operator and $\mathbf { W } _ { \mathrm { t a r g e t } } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { \mathrm { m o d e l } } }$ is a trainable projection matrix. Under this formulation, the amount of future token information can be controlled in our target $\hat { \textbf { V } }$ , e.g., the Next-Token target can be achieved by parameterizing $\mathbf { W } _ { \mathrm { t a r g e t } }$ as an identity transformation and assigning Conv1D $( \cdot )$ ’s kernel weights to be 1 for the next token and $0$ for other tokens.

With this aligned target, we use the widely used similarity measure to instantiate our loss function for simplicity, i.e., $\mathcal { L } ( \cdot , \cdot ) = - \langle \cdot , \cdot \rangle _ { F }$ . Under this loss function, the gradient with respect to the fast weights in our chunk-wise mechanism can be directly derived:

> 💡 **更新公式**: 使用负 Frobenius 内积后，更新退化成外积累加 $\eta \hat V^T Z$。这使 prefix-sum/context-parallel 实现成为可能，因为各 chunk 的 delta 可先独立计算再扫描累加。

$$
\mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } = \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i - 1 ) } + \eta \hat { \mathbf { V } } _ { [ i ] } ^ { \top } \mathbf { Z } _ { [ i ] } .
$$

# 3.3 Theoretical Analysis

Intuitively, our LM-Aligned objective explicitly encourages the fast weights to compress predictively useful information for future tokens, thereby enhancing the model’s capacity for dynamic adaptation. In this subsection, we formalize this intuition by theoretically analyzing the benefits of our objective. We ground our analysis within the canonical induction head setting [19, 38], a mechanism understood to be critical for in-context learning in LLMs.

Setup. Consider an input sequence where a key-value pair, $( x _ { t ^ { * } } , x _ { t ^ { * } + 1 } ) = ( k ^ { * } , v ^ { * } )$ , appears at an arbitrary position $t ^ { * }$ . Subsequently, at a query position $n > t ^ { * }$ , the key $k ^ { * }$ reappears, such that $x _ { n } = k ^ { * }$ . The model must then correctly predict the associated value, $x _ { n + 1 } = v ^ { * }$ .

> 💡 **理论场景**: induction head 场景直接对应“看到 key 后预测此前跟随它的 value”。这正是 long-context recall/association 任务中 TTT fast weights 应该增强的能力。

Without loss of generality, we analyze a single Transformer block enhanced by our In-Place TTT. Let $\mathbf { Z } _ { t } \in \mathbb { R } ^ { d _ { \mathrm { f f } } }$ be the intermediate activation of token $x _ { t }$ and $\mathbf { E } _ { w } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ be the token embedding for $x _ { w }$ . In our framework, the fast weights update from prior context chunks is $\begin{array} { r } { \Delta \mathbf { W } _ { \mathrm { d o w n } } = \eta \sum _ { t \in \mathrm { p r i o r ~ c h u n k s } } \mathbf { V } _ { t } \mathbf { Z } _ { t } ^ { \mathrm { ~ } } } \end{array}$ . This update then change the output logit at the query position $n$ by $\Delta \ell _ { n } [ w ] = \mathbf { E } _ { w } ^ { \top } \Delta \mathbf { W } _ { \mathrm { d o w n } } \mathbf { Z } _ { n }$ . We compare two choices for the TTT target $\mathbf { V } _ { t }$ :

• Reconstruction Target: ${ \bf V } _ { t } ^ { \mathrm { r e c } } = { \bf E } _ { x _ { t } }$ , the embedding of the current token.

• LM-Aligned Target: ${ \bf V } _ { t } ^ { \mathrm { L M } } = { \bf E } _ { x _ { t + 1 } }$ , the embedding of the next token.

Assumptions. Our analysis rests on two mild assumptions about the properties of the embeddings and intermediate activations, which are standard in theoretical analyses of Transformers:

1. Approximate Orthogonality of Embeddings: For any two distinct tokens $w _ { i } , w _ { j } \in \mathcal { V }$ , their embeddings are nearly orthogonal: $| \mathbf { e } _ { w _ { i } } ^ { \top } \mathbf { e } _ { w _ { j } } | \leq \epsilon$ for a small constant $\epsilon > 0$ . Additionally, embeddings have a non-trivial magnitude: $\| \mathbf { e } _ { w _ { i } } \| ^ { 2 } \geq c _ { \mathrm { n o r m } } ^ { 2 } > 0$ for some constant $c _ { \mathrm { n o r m } }$ .   
2. Key-Query Alignment: The intermediate activations $\mathbf { Z } _ { n }$ for the query token $x _ { n } = k ^ { * }$ is aligned with $\mathbf { Z } _ { t ^ { * } }$ of its corresponding key token $x _ { t ^ { * } } = k ^ { * }$ : $\mathbb { E } [ \mathbf { z } _ { t ^ { * } } ^ { \mathrm { ~ \tiny ~ 1 ~ } } \mathbf { z } _ { n } ] = c _ { \mathrm { a l i g n } } > 0$ . For other positions $t \neq t ^ { * }$ , the tokens are unrelated to the query, i.e, $\mathbb { E } \big [ ( \mathbf { V } _ { t } \mathbf { Z } _ { t } ^ { \top } ) \mathbf { Z } _ { n } \big ] = \mathbf { 0 }$ .

![](../images/6b2a27d1856dc5dd99cf8575d6ecdc98752777584dac608a92e8b89fbee06f97.jpg)  
Figure 1 The overall framework of our In-Place Test-Time Training. The module operates sequentially on input chunks. For each chunk, the current fast weights are first applied to the intermediate activations $\mathbf { z }$ to produce the output. Then, these weights are updated using the activations $\mathbf { z }$ and a value $\mathbf { V }$ derived from the token embeddings. This "apply-then-update" cycle allows the model to dynamically adapt to incoming context in a strictly causal manner.

With this setup, we present our main theoretical result:

Theorem 1 (Logit-wise Effect of LM-Aligned Target v.s. Reconstruction Target). Under the specified setup and assumptions, for a learning rate $\lambda _ { l r } > 0$ , the expected change in logits $\Delta \ell _ { n }$ after one update step using the LM-Aligned target satisfies:

> 💡 **理论结论**: NTP-aligned target 会提升正确 next token 的 logit，而 reconstruction target 对正确 value 的提升只有近似正交噪声量级。理论证据支撑 objective 不是随意工程选择。

$$
\begin{array} { r l } { ( C o r r e c t ~ l o g i t ~ i n c r e a s e s ) } & { \mathbb { E } \left[ \Delta \ell _ { n } [ v ^ { * } ] \right] \geq \lambda _ { l r } \cdot c _ { n o r m } ^ { 2 } \cdot c _ { a l i g n } , } \\ { ( O t h e r ~ l o g i t s ~ a l m o s t ~ u n c h a n g e d ) } & { \left| \mathbb { E } \left[ \Delta \ell _ { n } [ w ] \right] \right| \leq \lambda _ { l r } \cdot \epsilon \cdot c _ { a l i g n } , ~ \forall w \neq v ^ { * } . } \end{array}
$$

In contrast, for the reconstruction target, the expected change in logits is negligible for the correct token: $| \mathbb { E } \left[ \Delta \ell _ { n } [ v ^ { * } ] \right] | \leq \lambda _ { l r } \cdot \epsilon \cdot c _ { a l i g n }$ .

The proof is provided in Appendix A. In theorem 1, the LM-Aligned target is guaranteed in expectation to increase the logit of the correct next token $v ^ { * }$ and keep that of other tokens approximately unchanged, directly aiding the model’s prediction task. In contrast, the reconstruction target provides no such predictive benefit, failing to increase the logit of the correct token. In practice, our implementation extends this principle from a single next token to a learned, localized combination of future tokens, which also aligns with recent promising results of Multi-Token Prediction in advanced LLMs [37] as an effective extension of the NTP objective. This allows our In-Place TTT to capture a richer predictive signal, thereby compressing useful contextual information more effectively than simple reconstruction.

# 3.4 Implementation Details

Combining aforementioned designs, figure 1 illustrates our In-Place TTT framework. Here we further elaborate on practical implementation details, which are engineered for high efficiency and scalability on modern hardware. In particular, our approach is fully compatible with Context Parallelism ( $\mathit { C P }$ ), relying on a parallel scan algorithm to process sequence chunks simultaneously while preserving the strict causal semantics of an auto-regressive update. Additional discussions are further presented.

Efficient Implementation with Context Parallelism. The associative nature of our update rule in equation (1) makes In-Place TTT amenable to a context-parallel implementation, which partitions a sequence along its length and processes the chunks simultaneously. The process unfolds into three stages: (i) for all chunks $i \in \{ 1 , \ldots , T \}$ , we compute the intermediate activations $\mathbf { Z } _ { [ i ] }$ and the fast weight update $\Delta \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } = ( \hat { \mathbf { V } } _ { [ i ] } ) ^ { \top } \mathbf { Z } _ { [ i ] }$ in parallel; (ii) a single prefix sum over $[ . . . , \Delta \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i ) } , \Delta \mathbf { W } _ { \mathrm { d o w n } } ^ { ( i + 1 ) } , . . . ]$ is conducted to compute the aggregated updates for each chunk: $\Delta \mathbf { S } _ { i } = \textstyle \sum _ { j = 1 } ^ { i - 1 } \Delta \mathbf { W } _ { j }$ , which can be highly efficient on modern accelerators; (iii) the effective fast weights for each chunk, $\mathbf { Z } _ { [ i ] } ( W _ { \mathrm { d o w n } } ^ { ( i - 1 ) } ) ^ { \top }$ , are computed in parallel. $\mathbf { W } _ { \mathrm { d o w n } } ^ { ( i - 1 ) } = \mathbf { W } _ { \mathrm { d o w n } } ^ { ( 0 ) } + \eta \Delta \mathbf { S } _ { i }$ , and the corresponding output, ${ \bf O } _ { [ i ] } =$

> 💡 **Context parallelism**: 由于每个 chunk delta 可独立算，随后用 prefix sum 得到每个 chunk 前缀状态，In-Place TTT 可以沿序列维并行，保持严格因果语义。

Causality and Boundary Handling. To ensure that the update delta for chunk $_ i$ itself contains no future information, we apply causal padding to the 1D convolution when generating the value. This isolates each delta calculation to its respective chunk, making the parallel scan mathematically equivalent to a sequential update. Moreover, at document boundaries, the fast weights are reset to their pre-trained state to prevent context leakage across independent sequences. The final context parallel algorithm is presented in algorithm 1 in Appendix B.

> 💡 **边界处理**: causal padding 和 document boundary reset 是防泄漏关键。否则 NTP-aligned target 的未来信息或跨文档 fast weights 会污染评估。

Discussion. In summary, our implementation of In-Place TTT synergistically combines a simple, computationally efficient update rule with a parallel scan algorithm. This design choice makes our method not only fast and scalable but also mathematically equivalent to a strictly causal sequential process, thanks to careful boundary and padding management. The resulting module is $\mathit { C P }$ -native, fully causal, and can be seamlessly integrated as a drop-in replacement for the MLP block in standard Transformer architectures. Lastly, it is also noteworthy that the core principles of our framework are orthogonal to the specific choice of loss functions and its optimizer, which have been widely studied in the broader TTT literature, an exploration we leave as a promising direction for future work.

---

## 🔖 Section 总结

> 💡 **Section 小结**: 方法用 $W_{down}$ 做 fast weights，以 $Z$ 为 key、NTP-aligned $\hat V$ 为 value，通过外积 delta 和 prefix scan 实现严格因果的 chunk-wise context parallel TTT。
