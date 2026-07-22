[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
本节建立问题动机：full attention 能近似无损 recall 但随上下文变慢，RNN/SWA 常延迟但长上下文利用不足，TTT-E2E 用权重压缩上下文。


Humans are able to improve themselves with more experience throughout their lives, despite their imperfect recall of the exact details. Consider your first lecture in machine learning: You might not recall the instructor’s first word during the lecture, but the intuition you learned is probably helping you understand this paper, even if that lecture happened years ago.

On the other hand, Transformers with self-attention still struggle to efficiently process long context equivalent to years of human experience, in part because they are designed for nearly lossless recall. Self-attention over the full context, also known as full attention, must scan through the keys and values of all previous tokens for every new token. As a consequence, it readily attends to every detail, but its cost per token grows linearly with context length and quickly becomes prohibitive.
> 💡 **问题重述**: 作者把 full attention 的强项和弱点都说清楚：它接近无损检索，所以能 recall 细节；但每个新 token 都扫描历史 KV，decode 单 token 成本随 context 线性增长。


As an alternative to Transformers, RNNs such as Mamba 2 [32] and Gated DeltaNet [104] have constant cost per token, but become less effective in longer context, as shown in Figure 1. Some modern architectures approximate full attention with a sliding window [1, 107], or stack attention and RNN layers together [91, 11]. However, these techniques are still less effective than full attention in using longer context to achieve better performance in language modeling.

How can we design an effective method for language modeling with only constant cost per token? Specifically, how can we achieve better performance in longer context without recalling every detail, as in the opening example? The key mechanism is compression. For example, humans compress a massive amount of experience into their brains, which preserve the important information while leaving out many details. For language models, training with next-token prediction also compresses a massive amount of data into their weights. So what if we just continue training the language model at test time via next-token prediction on the given context?
> 💡 **机制批注**: TTT-E2E 的核心不是“把窗口变大”，而是把读过的上下文压进可更新权重。这个观点解释了为什么它可能牺牲 needle recall，却在平均 language modeling loss 上随上下文变长继续收益。


This form of Test-Time Training (TTT), similar to an old idea known as dynamic evaluation [72, 60], still has a missing piece: At training time, we were optimizing the model for its loss out of the box, not for its loss after TTT. To resolve this mismatch, we prepare the model’s initialization for TTT via meta-learning [38, 79, 58] instead of standard pre-training. Specifically, each training sequence is first treated as if it were a test sequence, so we perform TTT on it in the inner loop. Then we average the loss after TTT over many independent training sequences, and optimize this average w.r.t. the model’s initialization for TTT through gradients of gradients in the outer loop [71, 3, 27].
> 💡 **训练-测试对齐**: 只在测试时继续 next-token training 会产生 train/test mismatch；E2E 的关键是训练时就把每条训练序列当测试序列跑 inner-loop，再用 outer-loop 优化“更新后”的损失。


In summary, our method is end-to-end in two ways. Our inner loop directly optimizes the next-token prediction loss at the end of the network, in contrast to prior work on long-context TTT [86, 110]; Subsection 2.4 explains this difference through an alternative derivation of our method. Moreover, our outer loop directly optimizes the final loss after TTT, in contrast to dynamic evaluation [72, 60], as discussed. Our key results are highlighted in Figure 1, with the rest presented in Section 3.
> 💡 **E2E 两层含义**: test-time E2E 指 inner-loop 直接用最终 next-token loss，而不是 KVB 那种层内重构 loss；training-time E2E 指 outer-loop 直接优化经过 TTT 后的最终 loss，而不是 dynamic evaluation 式的静态预训练。


The conceptual framework of TTT has a long history with many applications beyond long context, and many forms without meta-learning [85, 12, 45, 2]. Our work is also inspired by the literature on fast weights [38, 79, 77, 49], especially [17] by Clark et al., which shares our high-level approach. Section 4 discusses related work in detail.

---

## 🔖 Section 总结

- **问题**: full attention 细节 recall 强但长上下文成本高；RNN/SWA 常延迟但不能稳定从更长上下文获益。
- **答案**: 继续在测试上下文上做 next-token prediction，把历史压进权重。
- **关键区别**: TTT-E2E 同时对齐 test-time inner-loop loss 和 training-time outer-loop objective。
