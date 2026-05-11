[← 返回 README](../README.md)

# 2. Related Works

# 2.1. Large Language Models

Large language models (LLMs) have become the dominant paradigm in modern NLP. They are typically transformerbased (Vaswani et al., 2023) auto-regressive decoder models trained with next-token prediction, and scaling model size, data, and compute tends to improve generalization and unlock emergent capabilities, including zero-/few-shot in-context learning and multi-step chain-of-thought (COT) (Wei et al., 2023) reasoning. In practice, LLMs also undergo post-training to better match human intent. Instruction tuning improves instruction following, while alignment methods such as reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022) shape outputs toward helpfulness and safety.

> 💡 **批注**: 这一节只提供背景，但和本文方法的连接点是 autoregressive next-token prediction。prompt NLL/input perplexity 能作为无监督 TTA loss，正是因为 decoder-only LLM 原生支持 teacher-forced next-token loss。

Alongside these training recipes, architectural and system advances further improve performance. Mixture-of-experts (MoE) (Fedus et al., 2022) increases model capacity with sparse activation and keeps per-token compute relatively low. Retrieval-augmented generation (RAG) grounds generation in external knowledge, improving factuality and enabling maintenance through an editable retrieval index rather than weight updates. More recently, LLM agents couple an LLM with planning, memory, and tool/API execution loops, allowing it to decompose goals into actionable steps and carry out workflows beyond passive prompting.

> 💡 **对比批注**: RAG 和 prompting 改的是输入上下文，本文改的是临时 LoRA 参数。它的优势是可把 prompt 统计写进权重，潜在突破有限 context；风险是写入错误统计会破坏生成分布。

# 2.2. Test-Time Adaptation

Test-time adaptation (TTA) updates a deployed model at inference time using unlabeled test-time signals. A canonical early approach is TENT, which performs unsupervised adaptation by minimizing prediction entropy on target batches, updating only lightweight affine parameters (and normalization statistics) (Wang et al., 2021). Since na¨ıve entropy minimization can cause overconfident drift or even collapse, conservative variants such as COME stabilize adaptation by modeling uncertainty and optimizing a guarded surrogate (Zhang et al., 2024).

> 💡 **TTA 背景批注**: 视觉 TTA 常用 entropy minimization，但 LLM 生成中“低熵”不等于答案好，甚至可能过早自信坍缩。这解释了为什么本文沿用 perplexity/NLL 而不是 entropy。

For autoregressive LLMs, recent work argues that entropy is often misaligned with generation, and that perplexity (next-token negative log-likelihood) is a more suitable selfsupervised objective for test-time updates (Hu et al., 2025a). Moreover, sample-specific (per-prompt) adaptation has been shown feasible, e.g., SLOT performs a few optimization steps for each prompt under an adapt-and-reset protocol (Hu et al., 2025b).

> 💡 **和 TLM/SLOT 的关系**: 这篇基本站在 Hu et al. 2025a/b 之上：loss 还是 input perplexity / prompt NLL，protocol 还是 per-prompt adapt-and-reset。它新增的是“不要手写固定 LR，让超网络按 layer/step/prompt 控制 LoRA update strength”。

# 2.3. Low-Rank Finetuning

Parameter-efficient fine-tuning (PEFT) adapts large pretrained models by updating only a small set of additional parameters while keeping the backbone frozen. A widely used PEFT method is Low-Rank Adaptation (LoRA) (Hu et al., 2021), which represents weight updates with low-rank factors, greatly reducing trainable parameters and optimization cost. LoRA is commonly applied to transformer attention projections (e.g., query/value), and its updates can be merged into the base weights for deployment with negligible inference overhead.

> 💡 **LoRA 批注**: 常规 LoRA 是训练后合并或常驻 adapter；本文 LoRA 是 prompt-local fast state。每个 prompt 初始化、更新、丢弃，因此 LoRA 在这里更像临时工作记忆，而不是长期参数高效微调结果。
