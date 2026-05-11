[← 返回 README](../README.md)

# 6. Future Work

## 📌 预览

Future Work 直接点出 Absorber 的三个开放方向：continual learning、causal editability/selective forgetting、多模态长流。

In this paper, we presented Absorber LLM, a framework that internalizes historical context into model parameters by enforcing functional alignment. Our proposed framework can be extended along three main directions.

First, we will explore how context absorption can be integrated with continual learning, enabling models to incrementally internalize new data while avoiding catastrophic forgetting of previously absorbed memories. Second, we will investigate the causal editability of absorbed representations to support selective forgetting, allowing specific memories to be removed for privacy or factual updates without full retraining. Third, we aim to extend the absorption paradigm to multimodal long-context settings, such as long-form video and sensor streams, where functional alignment could offer a scalable alternative to expanding the KV cache.

> 💡 **开放问题批注**: 这三个方向都不是附加项，而是 Absorber 落地必须解决的问题。参数一旦成为记忆载体，就需要写入稳定性、删除能力、跨模态对齐和隐私边界。

These extensions will advance the development of more efficient, controllable, and broadly capable long-context models. We plan to optimize and expand the framework to make it applicable to longer texts and larger-parameter models.

> 💡 **规模化批注**: 当前实验只到 LLaMA2-7B 和 16K/20K 级别任务。要支撑 infinite-stream 叙事，后续必须报告更大模型、更长上下文和端到端吸收成本。

---

## 🔖 Section 总结

- **continual learning**: 多轮吸收后如何不忘旧记忆。
- **selective forgetting**: 参数写入后如何可控删除隐私或过期事实。
- **multimodal streams**: 视频/传感器流可能更需要压缩，但同步目标也更难定义。
