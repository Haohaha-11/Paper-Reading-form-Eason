[← 返回 README](../README.md)

# 2 Vanilla Compute-Scaling Strategies Fail for Long Contexts

> 💡 **📌 本章预览**: 通过两个受控sandbox任务展示static attention的失败模式，形式化score dilution现象，并证明thinking tokens无法突破其限制。这是全文的理论基础部分，论证了"为什么要用TTT替代think tokens"。

In this section, we analyze how increasing context length T affects static quadratic-attention LLMs and common inference-time compute-scaling strategies. Using controlled synthetic tasks that mirror realistic long-context retrieval, we observe sharp performance degradation as T grows, while generating intermediate "thinking" tokens yields rapidly diminishing returns. We then provide a theoretical explanation: with static, finite-precision self-attention, the target logit suffers score dilution as distractors accumulate, and avoiding this requires a logarithmic margin requirement -- the worst-case target-distractor logit gap must scale as Omega(log T). Decoding-based inference strategies do not reliably meet this requirement; in contrast, small gradient-based adaptations can increase the margin, which motivates our methodology (developed in 3). All proofs are provided in Appendix B.

> 💡 **问题动机**: Section 2的整体逻辑：empirical observation -> formalization -> impossibility result for thinking tokens -> motivation for gradient-based approach。这是一条严谨的"现象-理论-不可能性-动机"链条。

## 2.1 Empirical Analysis on Synthetic Long-Context Tasks

First, we empirically analyze the effect of context length on vanilla transformer models and current inference time compute-scaling strategies. We study two synthetic retrieval tasks that mirror realistic long-context use cases while allowing control over the context length T. For each example, the relevant evidence ("needle") is held fixed and only the surrounding "haystack" grows, isolating the effect of length on retrieval. We provide examples from our datasets in Appendix A.

> 💡 **实验设计批注**: Sandbox任务的关键设计原则——固定needle，只增长haystack。这样保证了性能变化纯粹来自上下文长度而非问题难度变化，体现了良好的受控实验设计。

### Bug Localization in a Code Repository

Starting from a large open-source repository, we inject a single-line logical bug and ask the model to identify and fix it. Examples of bugs include missing softmax temperature scaling in the attention mechanism and layernorm misplacement in the Transformer block (see Appendix for details). We vary the context length by the number of lines L exposed to the model. For a given bug instance, we sample a span of L lines around the bug, extending to other files in the directory for large L. We create splits of the dataset with L ranging from 5 to 10000. Across length conditions, the bug location and content are held fixed; only the surrounding code (the "haystack") grows to introduce realistic, semantically relevant distractors.

> 💡 **实验设计批注**: 代码bug定位任务的设计精妙之处在于：distractor是与目标代码语义相关的真实代码（同一repo中的其他文件），而非随机噪声。这比经典的Needle-in-a-Haystack（随机文档中插入不相关事实）更难，因为distractor在语义上可能是合理的、甚至看起来也像bug的代码行。L从5到10000的跨度覆盖了从短函数到完整项目目录的场景。

### Error in a Log of Transactions

We synthesize multi-account banking logs with an initial state and a sequence of operations, each line recording old->new balances and indexed with a TX_ID. Valid logs must satisfy invariants: conservation of total funds, non-negative balances, and arithmetic correctness. We inject exactly one anomaly and consider the following bug types: CALC_ERROR (incorrect arithmetic), NEGATIVE_BAL (over-debit), LOST_UPDATE (stale write overwrites a prior commit) and DUPLICATE_TXN (same payment applied twice). The model must output the bug type and offending TX_ID. Context length is controlled by the number of operations n; we sweep from 25 operations to 500 operations which varies the number of tokens from O(10^2) to O(10^4).

> 💡 **实验设计批注**: 交易日志任务相比代码任务，distractor的特征不同：代码任务的distractor是语义相关的代码片段；而交易日志的distractor是结构相同但数学上正确的交易记录。两种distractor类型分别测试了语义干扰和结构干扰，增强了结论的泛化性。四个bug类型（计算错误、负余额、丢失更新、重复交易）覆盖了不同性质的异常检测子任务。

### Findings

We evaluate Qwen3 models ranging from 1.7B to 8B parameters on these synthetic tasks. Figure 1 shows the results for the Qwen3-4B model. For both tasks, we see clear consistent trends: (i) As the context lengths increases (number of code lines/transaction logs), the standard in-context performance (i.e., without any additional inference-time compute) decreases sharply. (ii) Further, using inference-time compute via thinking tokens improves performance for shorter contexts, but shows clear diminishing returns as the context length increases, asymptotically converging close to the standard model performance for long contexts.

> 💡 **消融解读**: 两个观察构成了对thinking token策略的有力反驳：(1) in-context accuracy随T单调下降，验证了长上下文的信号检索确实在退化；(2) thinking tokens在短上下文有帮助，但在长上下文时几乎退化为无thinking状态——这正是"静态注意力机制重复劳动无效"的实证。

> 💡 **Q&A 批注记录**: Q: 为什么thinking tokens在短上下文有帮助但长上下文无效？A: 因为thinking tokens本身也要经过同一套静态注意力来读取上下文。在短上下文时，模型还能attend到needle，所以多几步推理有用；但在长上下文时，初始的注意力分配已经把needle淹没了，thinking tokens从第一步就读不到正确信息，后续的推理自然无效。这是"garbage in, garbage out"在attention层面的体现。

**Empirical Takeaway**: Across both controlled tasks, holding the needle fixed and increasing the haystack length T yields a sharp, monotonic drop in in-context accuracy. Allocating inference-time budget to "thinking" tokens offers only short-horizon gains with clear saturation at large T. These trends suggest a structural limitation of static attention in long contexts.

We now formalize this limitation as score dilution and derive the resulting logarithmic margin requirement, which explains why decoding-based scaling fails to recover retrieval (2.3).

## 2.2 Preliminaries

Recall, for a sequence of T tokens with hidden representations {h_i}_{i=1}^{T} in R^d, each Transformer layer l computes query, key, and value projections:

Equation (2.1):

```
q_i^(l) = W_Q^(l) h_i,    k_j^(l) = W_K^(l) h_j,    v_j^(l) = W_V^(l) h_j,
```

where W_Q^(l), W_K^(l) in R^{d_k x d} and W_V^(l) in R^{d_v x d} are learned projection matrices. Further, the scaled dot product between query q_i and key k_j gives the attention logits z_{i,j} that are normalized via softmax to obtain attention weights alpha_{i,j}. Finally, the output o_i is a weighted sum of value vectors:

Equation (2.2):

```
z_{i,j} := q_i^T k_j / sqrt(d_k),    alpha_{i,j} := exp(z_{i,j}) / sum_{l=1}^{T} exp(z_{i,l}),    o_i = sum_{j=1}^{T} alpha_{i,j} v_j.
```

In the autoregressive setting, causal masking enforces j <= i, so that each position i can only aggregate information from its past. Multi-head attention extends this computation across several subspaces, allowing the model to capture diverse forms of dependency.

**In-Context Learning.** This attention-based retrieval is the foundation of in-context learning (ICL; (Dong et al., 2023)). By inserting task demonstrations, instructions, or relevant passages directly into the input, LLMs can adapt their outputs without parameter updates. For applications such as analyzing codebases, synthesizing long documents, or sustaining multi-turn dialogues, the model must effectively identify and use information scattered across contexts of length 10^4-10^6 tokens.

**Thinking Tokens.** Given a prefix x_{1:i} and a target at position i+1, thinking-token methods (Wei et al., 2022a; Kojima et al., 2022; Wang et al., 2023a) append M >= 0 auxiliary tokens at indices t in {i+1, ..., i+M} before producing the final answer at a = i+M+1. Each token t is generated with static parameters and the same attention kernel as in equation (2.2), yielding logits z_{t,j}, weights alpha_{t,j}, and outputs o_t over the augmented sequence of length T' = T+M.

**Definition 2.1 (Retrieval).** When predicting token x_{i+1}, the relevant information may lie in a specific key-value pair (k_{j*}, v_{j*}) (the 'needle') at some earlier position j* < i. For a threshold tau in (0,1), we say that retrieval at position i succeeds if alpha_{i,j*} >= tau. Equivalently, in margin form define gamma_i := z_{i,j*} - log sum_{j != j*} exp(z_{i,j}), then retrieval succeeds iff

```
gamma_i >= log(tau / (1 - tau)).
```

All other positions j != j* are distractors, contributing competing logits {z_{i,j}}_{j != j*}.

> 💡 **公式批读（Definition 2.1）**: 这个定义将"检索成功"严格量化为：target token的注意力权重alpha_{i,j*}超过阈值tau。Margin形式的等价条件gamma_i >= log(tau/(1-tau))是关键——它把注意力权重的阈值转换成了logit差距的下界。例如，如果想要alpha >= 0.9（即tau=0.9），则需要margin >= log(9) ≈ 2.2。这是后续所有理论分析的基础。

## 2.3 Theoretical Limitations of Static Attention and Thinking Tokens

Informed by the empirical findings in 2.1, we now analyze a single attention layer as in equation (2.2) on the retrieval task (Definition 2.1). We formalize the fundamental challenge of score dilution, which arises when "near-tie" distractors inflate the softmax denominator, causing even a unique maximal logit to receive vanishingly small attention mass.

**Lemma 2.2 (Score dilution).** If at least m distractor keys satisfy z_{i,j} >= z_{i,j*} - Delta for some Delta >= 0, then

```
alpha_{i,j*} <= 1 / (1 + m * exp(-Delta)).
```

In particular, if m >= cT for some c > 0 and Delta = O(1), then alpha_{i,j*} -> 0 as T -> infinity.

> 💡 **公式批读（Lemma 2.2）**: 这是score dilution的核心定量描述。上界 1/(1+m*exp(-Delta)) 告诉我们：即使target是唯一最大的logit，只要有足够多的"接近平局"的distractor（m足够大），target的注意力质量也会被稀释。当m是T的常数比例时（即固定比例的token都在target的O(1) logit范围内），随着T增长，alpha必然趋近于0。

This dilution effect imposes a strict requirement on how much the target logit must stand out from all other distractors. The following corollary quantifies this necessary separation, showing that the required margin between needle and distractor must grow with the context length.

**Lemma 2.3 (Logarithmic margin requirement).** Fix epsilon in (0,1). If

```
min_{j != j*} (z_{i,j*} - z_{i,j}) >= log((T-1)(1-epsilon) / epsilon),
```

then alpha_{i,j*} >= 1 - epsilon. In particular, guaranteeing a fixed target mass against worst-case distractors requires a gap that scales as Omega(log T).

> 💡 **公式批读（Lemma 2.3）**: 这是全文最重要的理论结论之一。要保证target获得(1-epsilon)的注意力比例，需要的margin至少是log((T-1)(1-epsilon)/epsilon) = Theta(log T)。以T=10^5, epsilon=0.1为例，所需margin约为log(10^5 * 0.9 / 0.1) = log(9*10^5) ≈ 13.7。这个对数增长的要求看似温和，但对静态权重来说很难在每个query上都满足——特别是当distractor的key与needle的key在向量空间中有一定相似性时。

Achieving a margin that scales logarithmically is difficult for a model with static attention. Next, we evaluate the strategy of generating thinking tokens in satisfying the logarithmic margin requirement.

**Proposition 2.4 (Needle-signal bound for generated tokens).** For any thinking token t in {i+1, ..., i+M} and any u in R^{d_v},

```
<u, o_t> <= alpha_{t,j*} <u, v_{j*}> + (1 - alpha_{t,j*}) max_{j != j*} <u, v_j>.
```

> 💡 **公式批读（Proposition 2.4）**: 这个命题的核心信息是：任何一个thinking token能携带的needle信号的上界，受该token自身对needle的注意力权重alpha_{t,j*}限制。如果模型在生成thinking token t时本来就无法注意到needle（alpha小），那么这个token的输出主要由distractor主导。

**Corollary 2.5 (Specialization under small margin).** If the margin at token t satisfies gamma_t <= log(epsilon / (1-epsilon)) (equivalently, alpha_{t,j*} <= epsilon by Definition 2.1), then

```
<u, o_t> <= epsilon <u, v_{j*}> + (1 - epsilon) max_{j != j*} <u, v_j>.
```

Moreover, by Lemma 2.2, if at least m distractors satisfy z_{t,j} >= z_{t,j*} - Delta, then alpha_{t,j*} <= 1/(1 + m*exp(-Delta)), yielding the same bound with epsilon = 1/(1 + m*exp(-Delta)).

> 💡 **公式批读（Corollary 2.5）**: 这是对thinking tokens的"死刑判决"。当margin不够大时（即alpha_{t,j*}很小），thinking token的输出中needle信号被压缩到epsilon的比例，而(1-epsilon)的比例来自distractor。换句话说：一个"看不清"needle的模型，生成再多thinking tokens也只能在distractor信息上打转。

Proposition 2.4 shows the fraction of needle signal any generated token can carry is at most its own attention mass on the needle. Under dilution (small margin), this mass is provably tiny (Corollary 2.5), so attending to thinking tokens cannot materially increase the final answer's effective margin unless some intermediate token first assigns non-trivial attention to the needle.

**Takeaways:** (i) With fixed weights, worst-case retrieval requires a logit margin that grows like Omega(log T); failing to achieve this leads to score dilution and vanishing alpha_{i,j*}. (ii) Autoregressively generating additional tokens with the same static attention does not repair missing access to the evidence. (iii) Any successful inference-time strategy must change the similarity q_i^T k_j (e.g., by updating queries) rather than sampling more tokens with unchanged parameters.

> 💡 **机制拆解（核心结论）**: 这三条takeaway精确地回答了"为什么需要TTT"：因为解决长上下文检索失败的唯一途径是改变query-key相似度（即修改query或key的表示），而所有基于解码的策略都无法做到这一点。qTTT的设计直接针对这一理论需求——通过梯度更新修改W_Q矩阵，改变query的表示，从而在不修改K/V的前提下重新分配注意力。

> 💡 **Q&A 批注记录**: Q: Proposition 2.4和2.5是否意味着thinking tokens在任何情况下都无效？A: 不是。如果thinking token t能够获得足够的alpha_{t,j*}（即在生成t的瞬间，模型恰好能够attend到needle），那么thinking token可以有效地传播信号。问题在于：在长上下文中，这个前提条件（足够大的alpha）本身就很难满足。Thinking tokens本身无法自主突破score dilution——它们只是在已经建立好的注意力分配上做计算。

> 💡 **🔖 Section 2小结**: 本节完成了从经验观察到理论证明的完整逻辑链：(1) sandbox实验表明in-context accuracy随T单调下降，thinking tokens有边际收益递减；(2) 形式化score dilution——当固定的distractor比例与target在O(1) logit范围内时，target注意力质量随T趋向于0；(3) 证明检索成功需要Omega(log T)的logit margin；(4) 证明thinking tokens不能突破这个限制——needle信号的上界受限于该token自身的注意力质量；(5) 结论：任何有效的inference-time策略必须修改query-key相似度（如更新query权重），而非在静态参数下生成更多token。

---
