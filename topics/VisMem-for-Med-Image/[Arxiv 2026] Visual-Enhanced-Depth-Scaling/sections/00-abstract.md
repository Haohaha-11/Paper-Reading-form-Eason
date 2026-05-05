[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文件保留论文题名、作者、摘要等开场信息，先抓住作者定义的问题、方法名和主要 claim。

---

# Visual Enhanced Depth Scaling for Multimodal Latent Reasoning

Yudong Han1,2\*, Yong Wang2\*, Zaiquan Yang3, Zhen $\mathrm { Q u ^ { 4 } }$ , Liyuan $\mathrm { P a n } ^ { 1 , 5 }$ †, Xiangxiang Chu2 1Beijing Institute of Technology, 2AMAP, Alibaba Group 3City University of Hong Kong, 4Institute of Automation, Chinese Academy of Sciences, 5Yangtze Delta Region Academy of Beijing Institude of Technology, Jiaxing, China https://github.com/Simon98-AI/Vedas

> 💡 **批注**: 这里在讨论视觉证据是否被保留和利用；要问模型是否真的看图，而不是被语言先验带偏。

# Abstract

Multimodal latent reasoning has emerged as a promising paradigm that replaces explicit Chain-of-Thought (CoT) decoding with implicit feature propagation, simultaneously enhancing representation informativeness and reducing inference latency. By analyzing token-level gradient dynamics during latent training, we reveal two critical observations: (1) visual tokens exhibit significantly higher and more volatile gradient norms than their textual counterparts due to inherent language bias, resulting in systematic visual underoptimization; and (2) semantically simple tokens converge rapidly, whereas complex tokens exhibit persistent gradient instability constrained by fixed architectural depths. To address these limitations, we propose a visual replay module and routing depth scaling to collaboratively enhance visual perception and refine complicated latents for deeper contextual reasoning. The former module leverages causal selfattention to estimate token saliency, reinforcing fine-grained grounding through spatially-coherent constraints. Complementarily, the latter mechanism adaptively allocates additional reasoning steps to complex tokens, enabling deeper contextual refinement. Guided by a curriculum strategy that progressively internalizes explicit CoT into compact latent representations, our framework achieves state-of-the-art performance across diverse benchmarks while delivering substantial inference speedups over explicit CoT baselines.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

---

## 🔖 Section 总结

### 核心洞察
1. 先记住本文的问题定义、方法名和主要 claim。
2. 后续阅读时回看摘要里的 claim 是否被实验支持。
