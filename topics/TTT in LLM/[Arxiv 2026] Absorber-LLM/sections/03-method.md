[← 返回 README](../README.md)

# 3. Proposed Method

## 📌 预览

方法的核心是用 full-context oracle 生成监督轨迹，再 fine-tune contextless model，使其在未来片段 $Y$ 上同步 oracle 的 hidden states。这样历史 $X$ 的作用不再以 token/KV 形式存在，而是以参数更新形式存在。

# 3.1. Context absorption with causal relation synchronization

Inspired by the parameter memory paradigm and the necessity of invoking causal mechanisms, we propose to move beyond context memorization to the preservation of the causal relation between historical contexts and the subsequent generation process. We define the required causal relation preservation in context memorization as the following self-supervised learning target: The updated contextless model $f _ { W ^ { \ast } }$ , with historical contexts $X$ absorbed into its parameters and access only to the subsequent texts $Y$ , should behave identically to the original pretrained model $f _ { W }$ with full contexts $X Y$ . That is:

$$
f _ { W ^ { * } } ( Y ) \equiv f _ { W } ( X Y )
$$

> 💡 **函数等价批注**: 这是全文最重要的公式。Absorber 把“历史被吸收”定义成一个可检验条件：去掉 $X$ 后，只要换成 $W^*$，未来行为仍应等价于 $f_W(XY)$。

where $Y$ denotes the subsequent text following $X$ , i.e., successive input tokens or model-generated inferences, such that the distribution of $Y$ is conditionally dependent on $X$ Note that such equivalence should be kept throughout the inference process of $Y$ :

$$
f _ { W ^ { * } } ( Y [ : i ] ) = f _ { W } ( X Y [ : i ] ) \quad i = 1 , \dots , | Y |
$$

Then we get an optimization target:

$$
\boldsymbol { W } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { W } ^ { * } } \sum _ { i = 1 } ^ { | \boldsymbol { Y } | } \parallel f _ { \boldsymbol { W } ^ { * } } ( \boldsymbol { Y } [ : i ] ) - f _ { \boldsymbol { W } } ( \boldsymbol { X } \boldsymbol { Y } [ : i ] ) \parallel
$$

> 💡 **逐前缀同步批注**: 要求对每个 $Y[:i]$ 都成立，避免只在最终 token 上对齐。这对 reasoning/summary 很关键，因为中间步骤的隐藏轨迹影响后续生成。

Note that compared to prior works, our optimization is guided by the causal effect synchronization objective rather than simple reconstruction in Equation 1. Instead of trying to reconstruct the exact history $X$ token-by-token, our goal is to find a parameter configuration $W _ { * }$ that preserves the causal effect of $X$ on the subsequent generation process of $Y$ . This ensures that the model absorbs existing contextual information and preserves its effects on further deduction, rather than simply memorizing it. Moreover, by applying semantic dependencies between $X$ and $Y$ , such absorption only retains the relevant information in historical contexts $X$ for subsequent inference in $Y$ , which filters noise in contextual data. By absorbing only the relevant information, our method relies less on high-quality contexts, and is more suitable for usual scenarios of model inferences.

> 💡 **信息筛选批注**: 这里隐含一个强假设：$Y$ 能提供足够信号告诉模型 $X$ 中哪些信息有用。若 $Y$ 太短、太弱相关，或未来任务需要 $X$ 中未在局部 $Y$ 暴露的细节，absorption 可能会丢信息。

# Algorithm 1 Context Absorption

Inputs: LLM $f _ { W }$ ; Historical context $X = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { n } )$ to be absorbed into parameters $W$ ; Subsequent text segment $Y = ( x _ { n + 1 } , x _ { n + 1 } , \dots , x _ { n + m } )$ ; Learning rate $\eta$ ; Maximum fine-tuning step number $K$ ; Synchronization threshold $\epsilon$

Outputs: Updated model parameters $W ^ { * }$ with context $X$ being absorbed.

1: Forward $f _ { W } ( X Y )$ , get full-context hidden states
$H _ { 1 } , \dots , H _ { n } , H _ { n + 1 } , \dots , H _ { n + m }$
2: Initialize $W ^ { * }  W$
3: for $k \gets 1$ to $K$ do
4: Forward $f _ { W ^ { * } } ( Y )$ to get contextless hidden states:
$\tilde { H } _ { n + 1 } , \ldots , \tilde { H } _ { n + m }$
5: Compute synchronization loss:
$\begin{array} { r } { \mathcal { L }  \frac { 1 } { m } \sum _ { p = n + 1 } ^ { \cdot n + m } \parallel \tilde { H } _ { p } - H _ { p } \parallel } \end{array}$
6: if L < ϵ then
7: ReturnW ∗
8: end if
9: $\nabla _ { W ^ { * } } { \mathcal { L } } \gets \mathrm { B a c k w a r d } ( { \mathcal { L } } )$
10: $W ^ { * }  W ^ { * } - \eta \cdot \nabla _ { W ^ { * } } \mathcal { L }$
11: end for
12: ReturnW ∗

> 💡 **Algorithm 1 批注**: 这不是纯 forward-only memory。每次吸收都需要 teacher forward、student forward、backward 和参数更新；LoRA 降低了更新成本，但系统总延迟必须把 absorption time 单独算清楚。

# 3.2. Absorb contexts through hidden states synchronization

The goal of our method is to realize equations 2 through optimization in equation 4 for causal effect synchronization. To ensure a more thorough synchronization, we force the hidden states of $f _ { W ^ { * } } ( Y )$ to be identical to $f _ { W ^ { * } } ( X Y )$ , which guarantees that they conduct identical deductions in the same internal manner. Specifically, let historical contexts $X ~ = ~ ( x _ { 1 } , x _ { 2 } , . . . , x _ { n } )$ be absorbed into model parameters, and we refer to future inferences $Y = ( x _ { n + 1 } , x _ { n + 1 } , \dots , x _ { n + m } )$ to synchronize the behavior of the contextless model to the original full-context model; let $H _ { 1 } , \dots , H _ { n } , H _ { n + 1 } , \dots , H _ { n + m }$ be the hidden states in propagating $f _ { W } ( X Y ) , { \tilde { H } } _ { n + 1 } , \dots , { \tilde { H } } _ { m }$ be the hidden states in propagating $f _ { W ^ { * } } ( Y )$ , where $H _ { p } = ( h _ { p } ^ { 0 } , \ldots , h _ { p } ^ { L } ) , p =$ $1 , \ldots , n + m$ , and $\tilde { { \cal H } } _ { p } \ = \ ( \tilde { h } _ { p } ^ { 0 } , \tilde { h } _ { p } ^ { 1 } , \dots , \tilde { h } _ { p } ^ { L } ) , p \ = \ n + $ $1 , \ldots , n + m$ , and $L$ is the number of layers in LLM $f$ Then the hidden states synchronization is:

$$
\tilde { H } _ { p } = H _ { p } \quad p = n + 1 , \ldots , n + m
$$

> 💡 **hidden-state synchronization 批注**: 文字里出现 $f_{W^*}(XY)$ 的表述容易混淆，按前后文和算法应理解为对齐原模型 $f_W(XY)$ 的 full-context hidden states。核心是逐层对齐，而不是只让最终 logits 看起来像。

which ensures the output synchronization in equation 3 by synchronizing the propagation process more thoroughly. Then we get the optimization target to ensure equation 4:

$$
W ^ { * } = \arg \operatorname* { m i n } _ { W } \frac { 1 } { m } \sum _ { p = n + 1 } ^ { n + m } \parallel \tilde { H } _ { p } - H _ { p } \parallel
$$

The above process that absorbs contexts $X$ into parameters $W$ is summarized in Algorithm 1. Note that the absorption requires fine-tuning, and we employ LoRA (Hu et al., 2022) for efficiency.

> 💡 **LoRA 批注**: LoRA 是这篇工程可行性的关键，不然每次吸收都更新全量 LLaMA2-7B 权重会很重。但 LoRA rank=64 的容量、层选择、长期累积漂移仍需要更多消融。

# 3.3. Conduct absorption in the flowing sequence

In a continual workflow where new texts, i.e., model generations or subsequent user prompts, are continuously absorbed into the model parameters, the synchronization process becomes iterative: Where new incoming contexts $X$ keep being absorbed into parameters $W$ by synchronization on subsequent texts $Y$ . Specifically, we maintain a dynamic absorption window $Z = X Y$ . After the absorption of $X$ , we move the window to absorb the segment subsequent to $X$ . Such a continual absorption is detailed in Algorithm 2.

> 💡 **streaming absorption 批注**: $Z=XY$ 的滑窗设计让系统每次只处理固定长度 $n+m$，因此理论上序列越长总成本线性增长。真正的开放问题是：多轮 $W \leftarrow W^*$ 后，旧写入是否稳定、是否会污染新任务。

To summarize, we synchronize $f _ { W ^ { * } } ( Y )$ with $f _ { W } ( X Y )$ using a fine-tuning strategy. By minimizing the divergence between the updated contextless model behavior and the ideal full-context behavior within the local segment, we treat the parameter update as a continuous error-correction process, which ensures the context absorption and generalization. Please refer to Figure 1 for a brief demonstration. Note that the computational cost of context absorption in Algorithm 1 is fixed given $m$ and $n$ ; in Algorithm 2, we absorb and generate $n$ tokens in each round, so that the computational costs grow linearly with sequence length. Overall, the computational cost of our method is $\mathcal { O } ( N )$ .

> 💡 **复杂度批注**: 线性复杂度 claim 依赖固定 $n,m$。它解决的是随历史无限增长的 attention/KV 问题，不代表每个 token 的实际 wall-clock 都低；Table 1 后面会显示 Absorber 稳定但常数项不小。

---

## 🔖 Section 总结

- **训练信号**: full-context oracle hidden states。
- **参数写入**: LoRA fine-tuning 到 $W^*$。
- **流式机制**: 固定窗口吸收并滑动，理论复杂度 $\mathcal{O}(N)$。
- **核心风险**: absorption 需要 backward，且多轮参数写入的稳定性还没被充分证明。
