[← 返回 README](../README.md)

## 📌 预览
引言量化 CoT 延迟问题，提出 CCoT 用 dense representations 保留部分推理收益。

---

# 1. Introduction

Chain-of-Thought (CoT) refers to the Large Language Model (LLM) technique in which the model simulates the process of thinking out loud by decomposing a complex question into parts and sequentially reasoning through each step. This behavior can be induced by finetuning on a dataset or human feedback (Liu et al., 2023; Puerto et al., 2024), demonstrating through ICL (Wei et al., 2023), or by providing tuned model instructions (Kojima et al., 2023). While CoT improves the reasoning capabilities of LLMs on a variety of tasks, the improvements come at the cost of a high generation latency. For instance, GPT-4o takes 21.37 seconds to generate a response to the question shown in Figure 1 with CoT prompting, whereas it can answer the same question without CoT prompting in 2.81 seconds, achieving the same answer with an almost 10x speedup.

> 💡 **延迟动机**: 这个例子把问题量化了：显式 CoT 可提升推理，但 latency 近 10x。CCoT 要保留一部分 CoT 的推理收益，同时避免完整 reasoning chain 的生成成本。


Past work has utilized what we term contemplation tokens as an alternative to explicit CoT reasoning traces (Pfau et al., 2024; Goyal et al., 2024). These are additional tokens used to introduce online memory, allowing for additional computations during inference. Instead of generating a reasoning chain entirely of explicit language tokens, the model conditions on a shorter sequence of contemplation tokens (Section 2). Contemplation tokens can either be contentful, grounded in semantically meaningful text, or noncontentful. There are many lines of prior work involving noncontentful contemplation tokens drawn from a set of discrete tokens; this paper introduces contentful contemplation tokens that represent reasoning chains performed in continuous space.

Our framework, called Compressed Chain of Thought (CCoT), generates contemplation tokens which are compressed representations of language-based reasoning chains. These contemplation tokens are trained through teacher forcing with respect to the gold hidden states corresponding to full reasoning traces. Our framework can be adapted to pretrained LLMs through LoRA finetuning. Moreover, the variable compression ratio during training allows for needbased adjustments to the performance-efficiency tradeoff by controlling the number of tokens generated during inference.

> 💡 **训练信号**: CCoT 的监督不是 final answer reward，而是 full reasoning trace 的 hidden states。它把自然语言 CoT 当 teacher，把 teacher 的连续中间表示压缩成短 latent sequence。


The contributions of this paper are as follows:

1. We finetune pretrained decoder-only LLMs with our new CCOT framework and empirically evaluate their performance and throughput on GSM8K; 2. We establish our framework in context of related work in filler tokens and CoT distillation in terms of performance and efficiency; 3. We extend theoretical results and demonstrate the computational capacity of CCOT contemplations tokens.

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - 本文关心的是 CoT 压缩，不是 cache augmentation。
> - 关键设计是先从 explicit reasoning chain 抽取 gold hidden states，再训练模型直接生成这些 dense contemplation tokens。
> - 阅读后续时要注意 $r$ 如何控制速度/准确率，以及 CCoT 与 PAUSE 的差异是否被实验支撑。
