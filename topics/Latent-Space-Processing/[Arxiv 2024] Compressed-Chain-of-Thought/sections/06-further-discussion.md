[← 返回 README](../README.md)

## 📌 预览
讨论 compression ratio、autoregressive layer 选择，以及 width/depth 的理论直觉。

---

# 6. Further Discussion

# 6.1. Hyperparameter Choices

Varying $r$ As $r$ controls how many contemplation tokens are generated, it makes sense that increasing $r$ would increase both accuracy and decode time. However, we found that accuracy plateaus after a certain threshold, about $r = 0 . 2$ . We hypothesize that this occurs because successive contemplation tokens are autoregressively decoded using the hidden state at the $l$ layer, which propagates an approximation to the next contemplation token generation. We suspect the noise from the approximation errors eventually outweighs the signal provided by the contemplation tokens.

Varying l We find that the choice of $l$ is important – we were unable to learn good weights for $\varphi$ when $l$ was set close to either 0 or the last layer $L$ . We hypothesize that hidden states at earlier layers (small l) still incorporate a lot of localized information about the token itself while hidden states at later layers (large $l$ ) instead incorporate a lot of localized information about the next token. As such, we found that $l \approx L / 2$ resulted in the best performance; we hypothesize that the hidden states at intermediate layers encode global information it them suitable for autoregressive decoding scheme we use to generate contemplation tokens. We provide results with other layer choices in Appendix A.

> 💡 **层选择批读**: 第 15 层最好，太早像局部 token embedding，太晚像 next-token 信息。中间层可能更适合承载全局 reasoning state，这和很多 representation probing 的经验一致。


Subset selection We used a learned scorer module to perform the subset selection of the gold hidden states to be emulated by $\varphi$ . In practice, we found that simply taking $k$ evenly spaced tokens resulted in a similar performance. However, we note that a module trained to decode from gold hidden states (in the setup described in Section 3.3) achieves lossless performance compared to decoding from the full reasoning chain, even for small values of $r$ . As such, we hypothesize that it is possible to learn a better scorer to identify a subset of hidden states that is easier to emulate; a better approximation of the gold hidden states could lead to lossless performance while only taking a fraction of the time to decode. The observed performance-efficiency tradeoff also likely occurs because it is easier to approximate sequences with less compression.

# 6.2. Theoretical Considerations

We explore the enhanced computational expressivity offered by contemplation tokens and crucially identify the advantage of decoding contemplation tokens autoregressively rather than in parallel. We provide a few high level intuitions that are formalized in Appendix B.

Width Suppose we have a Transformer block ATTN and an input $\bar { w } _ { 1 : n }$ . Computing $\mathrm { A T T N } \big ( \bar { w } _ { 1 : n } \big )$ results in $O ( n )$ parallel operations. If we pass in $m$ additional contemplation tokens in parallel and compute $\mathrm { A T T N } \big ( \bar { w } _ { 1 : n + m } \big )$ , we now perform $O ( n + m )$ parallel operations. The extra computations matter in tasks when the number of parallel operations required is greater than the input sequence length. This can occur when answering succinctly phrased problems that require many parallel operations: “compute all pairwise sums in the following list” or “select all combinations of dependent logical statements that can be mutually true” Computing pairwise sums of an $n$ element list requires processing $O ( n ^ { 2 } )$ parallel computations and computing the validity of all possible logical combinations of $n$ facts requires processing $O ( 2 ^ { n } )$ ones. As $n$ grows, introducing contemplation tokens during inference in these width-bottlenecked scenarios can allow models to solve additional problems.

> 💡 **理论直觉**: 并行 contemplation tokens 增加 width，适合并行操作多的问题；autoregressive contemplation tokens 增加 depth，适合多跳、递归、序列决策。CCoT 选择 autoregressive latent，正是为了获得 depth。


Depth Suppose we have a model consisting of $L$ Transformer blocks, and we generate contemplation tokens autoregressively than in parallel. Passing in $m$ additional contemplation tokens still results in the increased $O ( n { + } m )$ parallel operations, but also resuts in $O ( m L )$ sequential operations. These extra computations matter in tasks when the number of sequential operations required is greater than the depth of the model. This can occur in multi-hop question answering tasks or when determining the best move in sequential games such as go and chess. Introducing autoregressively decoded contemplation tokens in these depth-bottlenecked scenarios can allow models to solve additional problems.

To collect these observations into a formal theorem, we build from prior work that provides an analysis of the computational power of contemplation tokens decoded in parallel (Goyal et al., 2024). We restate their theorem below:

Theorem 6.1 (From Goyal et al. (2024)). Assume that the attention module has sufficiently many parameters $( K )$ that is much larger than the number of input tokens $^ { \prime N ) }$ . Then there are tasks that $M$ independent computations, where $N < M < K ,$ ), such that a 2-layer Transformer can implement the task if and only if it uses contemplation tokens.

Under the same assumptions, autoregressively decoded contemplation tokens can solve a broader class of problems. When the depth of a task $D$ exceeds the number of layers in a model $L$ , the model can only represent $L$ steps out of the required $D$ steps. Our intuition is that contemplation tokens can “save” the intermediate steps, so autoregressively the model’s representation of the Lth step as the input to the next token allows for the next forward pass to implement another $L$ steps on top of the saved work. Thus, any task of depth $D$ can be solved with an additional $D / L$ contemplation tokens. We note that in CCOT, we only pass in the representation of the $l \approx L / 2$ step, but this doesn’t detract from the asymptotic representational capacity; we simply require an additional $D / l$ tokens instead of $D / L$ We informally state our theorem below, see Theorem B.5.

Theorem 6.2. Assume the conditions in Theorem 6.1. Then, there are tasks that involve $M$ independent computations of a depth $D > 2$ such that a 2-layer Transformer can implement the task if and only if it autoregressively decodes contemplation tokens.

> 💡 **Theorem 6.2 批读**: 这个定理是非正式陈述，但提供了方法动机：对深度需求超过模型层数的任务，autoregressive latent token 可保存中间步骤，让后续 forward pass 接着算。

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - CCoT 的两个旋钮是 compression ratio $r$ 和 autoregressive layer $l$。
> - $r$ 控制 token budget，$l$ 控制 latent state 的语义层级。
> - 理论部分支持 autoregressive latent 在 depth-bottleneck 任务中的必要性。
