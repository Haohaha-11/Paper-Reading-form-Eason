[← 返回 README](../README.md)

# 5. Related Work

## 📌 预览

Related Work 把 Absorber 放在四条线交汇处：efficient attention、SSM/RNN、parametric memory/TTT、predictive information/causal alignment。

Efficient Transformers and Linear Attention. Standard self-attention computes the output as √ $O \_ { } \qquad =$ softmax $( Q K ^ { T } / \sqrt { d } ) V$ , incurring $\mathcal { O } ( N ^ { 2 } )$ computational complexity due to the $N \times N$ attention matrix. Linear attention methods (Katharopoulos et al., 2020; Choromanski et al., 2021) bypass this by employing a kernel trick. By approximating the softmax operation with feature maps $\phi ( \cdot )$ and exploiting the associativity of matrix multiplication, the attention mechanism becomes:

$$
O _ { i } = \phi ( Q _ { i } ) ^ { T } \sum _ { j = 1 } ^ { i } \left( \phi ( K _ { j } ) V _ { j } ^ { T } \right)
$$

Here, the term $\sum \phi ( K _ { j } ) V _ { j } ^ { T }$ acts as a matrix-valued hidden state $S _ { i } \in \mathbb { R } ^ { d \times d }$ , updated recurrently as $S _ { i } = S _ { i - 1 } +$ $\phi ( K _ { i } ) V _ { i } ^ { T }$ . This reduces inference complexity to $\mathcal { O } ( N )$ . However, strictly adhering to this linear recurrence degrades retrieval capabilities compared to full attention, primarily due to its deviation from the exact softmax mechanism.

> 💡 **linear attention 批注**: 线性注意力也有矩阵状态，但它是架构内的递推状态，不是通过 full-context teacher 同步出来的参数记忆。Absorber 的差异在“写入目标”，不是单纯把 attention 线性化。

State Space Models (SSMs) and Modern RNNs. Recent architectures like Mamba (Gu & Dao, 2024) and RWKV (Peng et al., 2023) discretize continuous-time state space equations into a recurrent form. The core mechanism involves compressing history into a hidden state $h _ { t } \in \mathbb { R } ^ { d }$ by:

$$
h _ { t } = \bar { A } _ { t } h _ { t - 1 } + \bar { B } _ { t } x _ { t } , \quad y _ { t } = C h _ { t }
$$

where ${ \bar { A } } _ { t }$ and ${ \bar { B } } _ { t }$ are discretized parameters (which, in the case of Mamba, are data-dependent). While efficient $\mathcal { O } ( 1 )$ inference state), these models are constrained by the Fixed-Capacity Hypothesis (Merrill et al., 2024). The state $h _ { t }$ must encode the entire context $x _ { 1 : t }$ . Since $\dim ( h _ { t } ) \ll \dim ( x _ { 1 : t } )$ for long sequences, this inevitably results in lossy compression and the catastrophic forgetting of long-tail dependencies that cannot be fit into the bounded vector space $\mathbb { R } ^ { d }$ .

> 💡 **SSM/RNN 批注**: 论文用 Fixed-Capacity Hypothesis 支撑参数记忆的必要性。不过实验里 Mamba latency 远低于 Absorber，所以两者取舍是“速度/状态简单性”对“参数容量/任务质量”。

Parametric Memory and Test-Time Training (TTT). To expand memory capacity beyond fixed vectors, TTT (Sun et al., 2025) adopts model parameters as the storage medium. TTT formulates memory retrieval as an online self-supervised learning problem. Given a history sequence or a new token $x _ { t }$ ), it updates the internal weights $W _ { t - 1 }$ to $W _ { t }$ by minimizing a reconstruction objective $\ell$ in Equation 1:

$$
W _ { t } = W _ { t - 1 } - \eta \nabla _ { W } \ell ( W _ { t - 1 } , x _ { t } )
$$

This process effectively embeds the context into the highdimensional parameter space $\mathbb { R } ^ { d \times d }$ , theoretically offering vastly superior memory capacity compared to vector states. However, this approach relies on iterative approximation through Stochastic Gradient Descent (SGD). Consequently, the quality of memory retention becomes heavily dependent on proper optimization hyperparameters and adequate training data. Furthermore, when projecting contexts, TTT does not specify the model’s performance on subsequent inferences. This cannot invoke the causal mechanisms in LLM, as it does not maintain the contribution of removed historical sequences to future inferences.

> 💡 **TTT 定位批注**: Absorber 同样用 SGD/LoRA 写参数，因此继承了优化成本和超参敏感性；它的新增点是把 loss 从 reconstruction 换成 full-context behavior alignment。

Predictive Information and Causal Alignment. Our formulation of parameter memory moves beyond simple target projection toward functional alignment, proposing a concept that bridges information theory and causal inference. From an information-theoretic perspective, the information bottleneck principle (Tishby et al., 2000; Bialek et al., 2001) suggests that an optimal representation should discard historical details irrelevant to predicting the future. More fundamentally, this aligns with recent advances in causal abstraction (Geiger et al., 2021; 2024), which posit that a simpler compressed model serves as a valid surrogate for a complex full system only if it preserves the causal mechanisms governing the output. In this view, context compression is not merely data summarization, but rather the learning of an invariant causal representation (Scholkopf et al.¨ , 2021) that maintains the intervention-effect relationship between historical and future tokens. While prior works in prompt compression (Chevalier et al., 2023; Mu et al., 2023) apply this philosophy at the token level, Absorber LLM extends causal alignment to the parameter space. By enforcing our equivalence condition, we ensure that the compressed parameters functionally substitute the full history, preserving the causal dynamics of generation without retaining the raw data.

> 💡 **causal alignment 批注**: 这段给了理论语言：information bottleneck 解释“丢掉与未来无关的历史”，causal abstraction 解释“压缩后仍能替代完整系统”。但论文实验证据主要是任务指标和 hidden-state ablation，还没有直接做 causal intervention benchmark。

---

## 🔖 Section 总结

- **相对 efficient attention**: 不改 softmax 近似，而是把旧上下文写进参数。
- **相对 SSM/RNN**: 用更大参数空间换取更强历史容量。
- **相对 TTT**: 保留参数记忆范式，但替换训练目标。
- **理论借口**: 信息瓶颈 + causal abstraction；仍需更强干预实验来坐实“causal”。
