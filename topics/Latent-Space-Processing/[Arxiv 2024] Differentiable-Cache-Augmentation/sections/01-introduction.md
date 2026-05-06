[← 返回 README](../README.md)

## 📌 预览
引言解释为什么显式 CoT 慢且难优化，以及为什么把额外计算挪到 kv-cache latent space 更适合异步处理。

---

# 1. Introduction

Recent research (Kojima et al., 2022; Wei et al., 2022; Wu et al., 2024) has shown that enabling large language models (LLMs) to generate, or even search over, intermediate sequences of steps before producing a final answer can significantly improve performance on reasoning tasks. More broadly, providing LLMs with the ability to allocate compute adaptively during generation can lead to more effective generation within a fixed compute budget (Schuster et al., 2022). However, at a high level, many of these “extra thinking” approaches are similar in that their sequences of intermediate outputs are discrete, making them difficult to train in an end-to-end fashion, and in that their extra “thinking” (i.e. computation) is performed just-in-time, as part of the output generating process.

> 💡 **问题动机**: 作者抓住两个痛点：离散 token 难以端到端优化，在线生成会直接增加用户等待时间。DCA 的目标不是让回答更长，而是把额外计算变成 decoder 之前或旁路上的 latent 缓存加工。


In this work, we introduce a fundamentally different approach, inspired by the literature on kv-cache compression (Ge et al., 2024; Mu et al., 2024). Our approach takes a step towards LLMs that can deliberate on their memories (encoded in the kv-cache), and distill these deliberations into a form usable for subsequent tasks. Specifically, our method processes the transformer’s cache and augments it with a set of soft tokens produced in a single forward pass–not sequentially. This extra processing is performed by a separate model, which we refer to as a “coprocessor”, while the base transformer remains frozen. Once the kv-cache is augmented with the coprocessor’s output (which we term “latent embeddings”), decoding proceeds as normal until the coprocessor is called again. This approach offers the following key advantages:

> 💡 **机制拆解**: “single forward pass” 是本文和 CoT/Tree-of-Thought 的关键差异。coprocessor 一次性读已有 kv-cache，然后生成一批 soft tokens；这些 soft tokens 不是 vocabulary token，不需要 autoregressive decoding。


End-to-end Differentiability: Our framework enables end-to-end backpropagation during coprocessor training, facilitating efficient optimization without the need for reinforcement learning techniques.

We leverage the standard language-modeling loss on pre-training data, making the method scalable. Asynchronous Operation: Because cache augmentation improves results many tokens beyond the augmentation point, and because the base transformer remains frozen during coprocessor training, asynchronous coprocessor operation becomes feasible. This contrasts with existing methods where additional computation occurs sequentially, and online. Our approach opens the door to models that can strategically bank computation by deliberating and refining their internal memory, independent of composing a response to a query.

We evaluate our method using Gemma-2 (Team-Gemma et al., 2024) models pretrained on a diverse dataset mixture. Our experiments demonstrate that without any fine-tuning on specific downstream tasks, our approach consistently improves performance across a range of reasoningintensive tasks. We observed that increasing the number of injected latent embeddings generally leads to better performance. For example, we observe a $1 0 . 0 5 \%$ improvement on GSM8K and a $4 . 7 0 \%$ improvement on MMLU, when augmenting the Gemma-2 2B model with 64 latent embeddings. These results highlight the potential of enhancing LLMs with kv-cache coprocessing for augmenting model capabilities.

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - DCA 把 latent processing 定义为 kv-cache 级处理，而不是 token 级思维链压缩。
> - 优点是异步、可选、base LLM frozen；代价是需要额外 coprocessor 和预训练数据。
