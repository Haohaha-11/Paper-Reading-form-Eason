[← 返回 README](../README.md)

# Attention Is All You Need — Abstract

Ashish Vaswani<sup>∗</sup> Google Brain avaswani@google.com

Noam Shazeer<sup>∗</sup> Google Brain noam@google.com

Niki Parmar<sup>∗</sup> Google Research nikip@google.com

Jakob Uszkoreit<sup>∗</sup> Google Research usz@google.com

Llion Jones<sup>∗</sup> Google Research llion@google.com

Aidan N. Gomez<sup>∗</sup> <sup>†</sup> University of Toronto aidan@cs.toronto.edu

Łukasz Kaiser<sup>∗</sup> Google Brain lukaszkaiser@google.com

Illia Polosukhin<sup>∗</sup> <sup>‡</sup> illia.polosukhin@gmail.com

## 📌 预览

摘要用一段话交代了本文最大的主张：**彻底扔掉 RNN 和 CNN，只用注意力机制**就能搭出序列转换模型 Transformer，而且在两个机器翻译任务上又快又好。核心卖点是三个：质量更高（BLEU 刷新 SOTA）、更易并行、训练成本极低。

---

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

> 💡 **问题动机**: 摘要开篇先立靶子——当时最强的序列转换模型（sequence transduction，指 seq→seq 任务，如翻译）全都建立在**复杂的 RNN 或 CNN + 编码器-解码器**之上，且最好的模型还额外挂了个注意力模块。作者要解决的痛点不是"效果不够好"，而是 RNN 固有的**串行计算**：隐状态 $h_t$ 依赖 $h_{t-1}$，一句话必须一个 token 接一个 token 地算，无法在序列内部并行，长句时尤其致命。

> 💡 **机制拆解**: 本文的赌注是"attention is all you need"——把注意力从"配角"提拔成**唯一的骨架**，完全抛弃 recurrence（循环）和 convolution（卷积）。这样做的直接收益是：序列内所有位置可以一次性并行计算，不再受串行链条约束。摘要用 "simple" 一词强调架构反而更简洁。

> 💡 **实验证据速读**: 摘要给了三个硬指标支撑 claim——(1) WMT2014 英德 **28.4 BLEU**，比含 ensemble 在内的旧最好结果高 2 BLEU 以上；(2) WMT2014 英法 **41.8 BLEU** 的单模型新 SOTA，只用 8 GPU 训 3.5 天，成本是旧模型的一个零头；(3) 迁移到英语句法成分分析（constituency parsing）也表现优异，说明不是只对翻译过拟合。注意"more parallelizable + significantly less time to train"是与质量提升并列的核心卖点，后文第 4 节会用复杂度表专门论证。

> 💡 **Figure 1 归属说明**: 按批读规则，Abstract 页出现的 Figure 1（Transformer 整体架构图）实际描述的是第 3 节模型结构，因此放到 [01-introduction](01-introduction.md) / [03-model-architecture](03-model-architecture.md) 中批读，此处不重复放图。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| 英德 BLEU (big) | 28.4（+2 over 旧 SOTA 含 ensemble）|
| 英法 BLEU (big) | 41.8（单模型新 SOTA）|
| 训练硬件/时长 | 8× P100 GPU，3.5 天（big）|

### 核心洞察
1. **主张极端而清晰**：只保留注意力，删掉 RNN/CNN，用"简单"换"并行 + 质量"。
2. **卖点是三元组**：质量 ↑、并行度 ↑、训练成本 ↓，三者同时成立才是真正的贡献。
3. **泛化证据**：句法分析任务的成功是防"只会翻译"质疑的关键补充。

### 可追问点
- 去掉 RNN 后，模型如何感知 token 顺序？（→ 3.5 位置编码）
- 为什么并行度会大幅提升？串行操作数如何量化？（→ 第 4 节 Table 1）
