[← 返回 README](../README.md)

## 📌 预览
摘要定义 CCoT：把显式 reasoning chain 压缩成 contentful continuous contemplation tokens，并用 token 数控制推理预算。

---

# Compressed Chain of Thought: Efficient Reasoning through Dense Representations

Jeffrey Cheng 1 Benjamin Van Durme 1

# Abstract

Chain-of-thought (CoT) decoding enables language models to improve reasoning performance at the cost of high generation latency in decoding. Recent proposals have explored variants of contemplation tokens, a term we introduce that refers to special tokens used during inference to allow for extra computation. Prior work has considered fixed-length sequences drawn from a discrete set of embeddings as contemplation tokens. Here we propose Compressed Chain-of-Thought (CCoT), a framework to generate contentful and continuous contemplation tokens of variable sequence length. The generated contemplation tokens are compressed representations of explicit reasoning chains, and our method can be applied to off-theshelf decoder language models. Through experiments, we illustrate how CCoT enables additional reasoning over dense contentful representations to achieve corresponding improvements in accuracy. Moreover, the reasoning improvements can be adaptively modified on demand by controlling the number of contemplation tokens generated.

> 💡 **问题动机**: CCoT 的出发点和 DCA 相同：CoT 有用但慢。区别是 CCoT 仍然围绕 reasoning chain，只是把完整自然语言链压缩成连续 dense representations。

---

## 🔖 Section 总结

> 💡 **Section 小结**:
> - 核心变量: full CoT length $m$、compression ratio $r$、compressed length $k=\lceil r m\rceil$。
> - 核心洞察: 推理可以被看成 accuracy/latency trade-off，调节 $r$ 就是在调节 latent thinking budget。
> - 可追问点: 压缩后的 latent 是否保留可解释性，能否从 dense tokens 还原 reasoning chain?
