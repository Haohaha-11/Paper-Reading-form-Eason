[← 返回 README](../README.md)

# 2 Background

## 📌 预览

背景节把 Transformer 放进"减少串行计算"的技术谱系里对比：先讲 Extended Neural GPU / ByteNet / ConvS2S 这类 **CNN 派**如何并行但代价是"远距离依赖需要更多层"；再介绍 **self-attention** 的既有成功应用；最后点明本文的独特性——**第一个完全靠 self-attention、不用序列对齐 RNN 或卷积**的转换模型。

---

## 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

> 💡 **机制拆解**: 这段是与 CNN 派方案的正面较量。ConvS2S、ByteNet 等用卷积实现了"所有位置并行计算"，但有个死穴：**关联两个相隔很远的位置，所需操作数随距离增长**（ConvS2S 线性、ByteNet 对数）——因为卷积核感受野有限，远距离要靠堆叠层数来覆盖。Transformer 把这个代价压到**常数**（任意两位置一步注意力直连）。但作者诚实地承认一个副作用：注意力是对多个位置做加权平均，会"模糊"分辨率；解决办法就是下一段要讲的 Multi-Head Attention（多个头分别关注不同子空间，弥补平均带来的信息损失）。

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

> 💡 **概念澄清**: 明确 self-attention（自注意力/intra-attention）的定义——**在同一个序列内部**让不同位置相互关联，从而计算该序列的表示。区别于跨序列的注意力（如 encoder-decoder attention）。作者列举它此前已在阅读理解、摘要、文本蕴含等任务上成功，说明这不是凭空发明的机制，而是把已被验证的组件推到"独当一面"。

End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

> 💡 **相关工作定位**: End-to-end memory networks 用的是"循环注意力"而非"序列对齐的循环"，在简单问答和语言建模上表现好。作者引它是为了说明"注意力可以替代 recurrence"已有先例，但这些先例仍未做到"完全不用序列对齐结构"。

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

> 💡 **贡献声明**: 明确本文的新颖性主张（novelty claim）——**据作者所知，Transformer 是第一个完全依赖 self-attention、既不用序列对齐 RNN 也不用卷积**来计算输入输出表示的转换模型。这句话把前面所有对比收束成一个定位：不是"注意力更好用"，而是"注意力可以是全部"。

---

## 🔖 Section 总结

### 核心洞察
1. **两条对手线**：RNN 派（第 1 节）和 CNN 派（本节）。RNN 败在串行，CNN 败在远距离依赖需要堆层。
2. **注意力的核心优势**：任意两位置**常数步**直连，这是后文 path length 论证（第 4 节）的理论基础。
3. **多头是对"平均模糊"的补偿**：这里已埋下伏笔——单一注意力做加权平均会损失分辨率，多头用来找回。

### 可追问点
- "reduced effective resolution due to averaging" 具体指什么？多头如何补偿？（→ 3.2.2）
- 与 ConvS2S 的复杂度到底差多少？（→ 第 4 节 Table 1 的量化对比）
