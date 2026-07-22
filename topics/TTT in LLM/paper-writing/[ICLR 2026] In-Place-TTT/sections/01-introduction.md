[← 返回 README](../README.md)

## 📌 预览
动机、三重障碍和本文设计路线。

---

# 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across a range of complex tasks [8, 11, 39, 51]. This success is largely built on a static “train then deploy" paradigm, where a model first acquires knowledge from massive corpora and then kept fixed during inference. Yet this design imposes a fundamental limitation: once deployed, the model’s weights cannot be updated, preventing dynamic adaptation to the specific context provided by streaming input tokens. Consequently, at test time, the model is constrained in its ability to process and reason over long-horizon, evolving tasks [9, 45], and to continuously learn from unbounded streams of experience like humans [44].

> 💡 **问题设定**: 论文把长上下文问题表述为“静态权重无法吸收流式上下文”。这和单纯扩展 attention window 不同，关注点是推理时权重能否随上下文在线改变。

In-context learning [8, 56] offers a way to mitigate this problem via maintaining all past input tokens in the context. However, its effectiveness is tethered to the model’s context window, restricted by the quadratic complexity of the de facto attention mechanism [52]. This bottleneck has spurred a line of research into architectural solutions aimed at efficiently extending the context window [6, 10, 17, 42]. Differently, Test-Time Training (TTT) has emerged as a new paradigm [3, 47, 48, 53, 63]. Instead of merely making a static model more efficient, TTT enables the model to dynamically update the parameters and adapt to any specific context, directly targeting the aforementioned limitation. Specifically, TTT introduces a small subset of model parameters, called fast weights [43], which can be updated on the fly for each new input. By minimizing a self-supervised reconstruction objective, these fast weights compress and internalize contextual information, functioning as an expressive, online evolving state.

Despite its conceptual appeal, unleashing TTT’s potential within the current LLM ecosystem is hindered by critical barriers: (i) Existing TTT methods often rely on specialized layers beyond standard Transformer blocks, which usually demand costly pretraining from scratch to achieve satisfactory performance. [47, 48, 53, 67]; (ii) the canonical TTT mechanism is inherently sequential [47, 48]. While existing works explore chunk-wise acceleration [3, 30, 49, 63], TTT’s role as the primary token mixer forces a reliance on small chunks to maintain performance, thereby bottlenecking the massive parallelism required to saturate modern accelerators; and (iii) the prevalent use of a generic reconstruction objective for TTT’s fast weights updating is not explicitly tailored for the causal, Next-Token Prediction task that governs autoregressive LMs, potentially hindering their ultimate performance.

> 💡 **三重障碍**: 这里明确列出 In-Place TTT 要解决的三件事：架构兼容、计算效率、语言建模目标错配。后文每个设计都对应其中一个障碍。

To bridge this gap, we introduce In-Place Test-Time Training (In-Place TTT), a framework designed to seamlessly endow LLMs with Test-Time Training capabilities by directly addressing the aforementioned barriers. Our core insight is to repurpose existing MLP blocks with an in-place design rather than introducing a new, specialized layer (tackling barrier i). Specifically, In-Place TTT treats the final projection matrix of MLP blocks as their fast weights, updating it in-place during inference. This “drop-in" design requires no modifications to the model’s architecture, preserving the integrity of pre-trained weights and enabling on-the-fly adaptation without costly retraining from scratch.

> 💡 **In-place 设计**: final projection $W_{down}$ 是标准 MLP 已有参数，更新它不会引入一个从零开始训练的新 recurrent/TTT layer；这就是 drop-in 的技术基础。

To tackle the computational inefficiency and objective misalignment, we further design a bespoke adaptation mechanism for language modeling. Following previous works [3, 30, 49, 63, 67], we replace the inefficient per-token updates with a scalable chunk-wise update rule (tackling barrier ii). Furthermore, our in-place design operates complementarily to the attention mechanism. This synergy obviates the need for small chunks required by standalone TTT layers, thereby ensuring high throughput on modern accelerators. Concurrently, we move beyond the generic reconstruction targets of prior work [48, 67] and introduce a novel objective explicitly aligned with the Next-Token Prediction (NTP) goal (tackling barrier iii). Grounded in a rigorous theoretical analysis, we show this NTP-aligned objective encourages the fast weights to store predictively useful information for autoregressive language modeling, leading to a highly effective and scalable algorithm.

> 💡 **效率逻辑**: 因为 attention 仍然负责 token mixing，TTT 不需要像 standalone TTT layer 那样用很小 chunk 维持性能，于是 chunk-wise update 可以做大，并和 context parallelism 合拍。

Grounded in these principled design choices, our In-Place TTT provides a practical and effective framework for enhancing LLMs with dynamic, continual adaptation. We conduct extensive experiments on language modeling tasks of various compute scales, using them as a practical proxy to probe the model’s potential on long-horizon, evolving tasks. Through relatively cheap continual training, our In-Place TTT enables Qwen3-4B-Base to achieve superior performance on tasks with contexts up to 128k. Furthermore, we compare In-Place TTT with competitive TTT-related methods by conducting pretraining from scratch on up to 32k-length corpora, validating the architectural merit of our framework. Finally, ablation studies on state size, chunk size, and fast weight objectives provide deeper insights, confirming the critical role of each design choice. Collectively, our results establish In-Place TTT as a promising step towards a paradigm of continual learning in LLMs.

> 💡 **实验覆盖**: 作者没有只做 continual-training 插件实验，还做了 pretrain-from-scratch 对比。这个设计能区分“对 Qwen3 有用的工程补丁”和“架构本身有优势”。

---

## 🔖 Section 总结

> 💡 **Section 小结**: Introduction 建立了“静态权重不足以处理流式长上下文”的问题，并把解决路线拆成架构兼容、计算效率和目标对齐三条。
