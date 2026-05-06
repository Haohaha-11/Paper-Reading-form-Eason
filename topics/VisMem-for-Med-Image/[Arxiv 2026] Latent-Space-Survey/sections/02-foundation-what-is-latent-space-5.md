[← 返回 README](../README.md)

# 2 Foundation: What is Latent Space? 5

## 📌 预览
本节保留原文并穿插批注，重点提炼与课题主线相关的机制和证据。

---

2.1 Concept 5   
2.2 Comparison with Explicit Space 6   
2.2.1 Representational Properties 6   
2.2.2 Functional Capabilities 7   
2.3 Comparison with Generative Visual Models 8

> 💡 **导读**: 这一节不是在讲具体模型，而是在回答“latent space 到底是什么，不是什么”。如果这个边界不清，后面很容易把隐式 CoT、视觉 latent、memory slot、world model state 混成一个概念。

> 💡 **阅读重点**: 先看它如何区分 explicit space 与 latent space，再看它为什么特意拿 generative visual model 的 latent space 做对照。前者是为了说明 machine-native computation，后者是为了防止把视觉生成里的 VAE latent 直接套用到语言模型语境里。

---

## 🔖 Section 总结

### 核心洞察
1. latent space 在这篇 survey 里被定义为语言模型内部连续表示和计算的主要承载面，而不是简单的“看不见的 hidden state”。
2. 和 explicit space 相比，它的优势是连续、紧凑、可微、可并行；代价是更难审计、更难解释。
3. 和 generative visual models 的 latent 不同，语言模型 latent 没有天然空间拓扑或 reconstruction 几何，因此不能直接照搬图像 latent 的直觉。
