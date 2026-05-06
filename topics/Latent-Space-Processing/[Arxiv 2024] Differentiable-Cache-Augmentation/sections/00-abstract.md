[← 返回 README](../README.md)

## 📌 预览
摘要给出 DCA 的核心：frozen LLM 不变，外挂 coprocessor 读 kv-cache 并写入 latent embeddings，用 LM loss 学会改善后续 decoding。

---

# Deliberation in Latent Space via Differentiable Cache Augmentation

Luyang Liu1, Jonas Pfeiffer1, Jiaxing ${ \mathbf { W } } { \mathbf { u } } ^ { 1 }$ , $\mathbf { J u n \mathbf { X i e } ^ { 1 } }$ and Arthur Szlam1 1Google DeepMind

Techniques enabling large language models (LLMs) to “think more” by generating and attending to intermediate reasoning steps have shown promise in solving complex problems. However, the standard approaches generate sequences of discrete tokens immediately before responding, and so they can incur significant latency costs and be challenging to optimize. In this work, we demonstrate that a frozen LLM can be augmented with an offline coprocessor that operates on the model’s key-value $( \mathbf { k } \mathbf { v } )$ cache. This coprocessor augments the cache with a set of latent embeddings designed to improve the fidelity of subsequent decoding. We train this coprocessor using the language modeling loss from the decoder on standard pretraining data, while keeping the decoder itself frozen. This approach enables the model to learn, in an end-to-end differentiable fashion, how to distill additional computation into its kv-cache. Because the decoder remains unchanged, the coprocessor can operate offline and asynchronously, and the language model can function normally if the coprocessor is unavailable or if a given cache is deemed not to require extra computation. We show experimentally that when a cache is augmented, the decoder achieves lower perplexity on numerous subsequent tokens. Furthermore, even without any task-specific training, our experiments demonstrate that cache augmentation consistently reduces perplexity and improves performance across a range of reasoning-intensive tasks.

> 💡 **问题动机**: 这篇论文把“多想一会儿”从显式 CoT token 转到 kv-cache。核心矛盾是：显式思维链能提升推理，但必须在线逐 token 生成；作者希望把额外计算提前写入缓存，让 frozen decoder 后续照常解码。


Keywords: Latent reasoning, Cache augmentation, LLM

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - 核心变量: frozen LLM 参数 $\theta$、kv-cache、coprocessor $f$、latent embeddings $z$。
> - 核心洞察: latent deliberation 可以表现为“改写/扩展缓存”，不一定表现为自然语言 CoT。
> - 可追问点: 这个 latent 是否真的推理，还是只是 cache-conditioned adapter?
