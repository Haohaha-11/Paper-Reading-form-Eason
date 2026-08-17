[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览

引言用四段完成从"现状 → 痛点 → 已有补丁 → 本文方案"的推进：RNN 是 SOTA 但受制于**串行计算无法并行**；注意力虽已普及但总是"寄生"在 RNN 上；本文提出 Transformer，**完全靠注意力**建立全局依赖，训练可大幅并行，12 小时训练即可达到新 SOTA。

---

## 1 Introduction

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

> 💡 **问题动机**: 第一段先承认 RNN/LSTM/GRU 是当时序列建模与机器翻译的既定 SOTA，把它们立为要挑战的对象。这不是否定 RNN 的效果，而是为下一段指出其"结构性缺陷"做铺垫——先肯定再反转。

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

> 💡 **机制拆解**: 这段是全文动机的核心。RNN 沿 token 位置逐步展开：$h_t = f(h_{t-1}, x_t)$，第 $t$ 步必须等第 $t-1$ 步算完——这条**串行链条**使得单个样本内部无法并行。序列越长问题越严重，且因为显存限制不能靠增大 batch 来摊薄。作者承认已有 factorization tricks、conditional computation 等优化，但强调"串行计算这一根本约束仍在"，为"干脆去掉 recurrence"的激进方案埋下伏笔。

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

> 💡 **机制拆解**: 注意力的关键优势被点出——它能建立**任意距离**的依赖，不像 RNN 那样信息要沿时间步一格格传递（远距离依赖会衰减）。但作者指出一个空白：此前注意力几乎总是作为 RNN 的"附件"存在，很少有人让它独立支撑整个模型。这正是本文要填的空。

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

> 💡 **本文方案**: 一句话交代方案与收益：Transformer 彻底弃用 recurrence，**只靠注意力**在输入输出之间建立全局依赖。收益是"可大幅并行 + 12 小时（8×P100）即达新 SOTA"。注意这里用的是 base 模型的 12 小时数字（第 5.2 节），与摘要 big 模型的 3.5 天对应不同规模。

> 💡 **架构地图指引**: 全文的整体结构见 [03-model-architecture](03-model-architecture.md) 中的 **Figure 1**（Transformer 模型架构图）：左半编码器、右半解码器各堆叠 $N=6$ 层，底部靠 Positional Encoding 注入顺序信息，顶部经 Linear→Softmax 输出。建议先扫一眼那张图再读第 3 节。

---

## 🔖 Section 总结

### 核心洞察
1. **痛点定位精准**：RNN 的问题不是精度而是"串行不可并行"，这决定了本文优化目标是"并行度 + 训练速度"。
2. **注意力从配角转正**：本文最大创新点是让 attention 独立支撑整个模型，而非做 RNN 的补丁。
3. **顺序信息的代价**：去掉 recurrence 后必须显式注入位置信息（→ 3.5 位置编码），这是纯注意力方案的必要补偿。

### 可追问点
- 完全去掉 RNN，长距离依赖真的更好学吗？为什么？（→ 第 4 节 path length 论证）
- Figure 1 中"shifted right"和 masked attention 如何共同保证自回归？（→ 3.2.3）
