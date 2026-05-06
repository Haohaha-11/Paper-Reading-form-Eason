[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要快速给出本文问题、方法和结果。读完先记住一句话：Visual Enhanced Depth Scaling 针对 multimodal latent reasoning 中视觉 token 梯度不稳定与语言偏置，提出视觉重放和动态深度扩展。

---

# Visual Enhanced Depth Scaling for Multimodal Latent Reasoning

Yudong Han1,2\*, Yong Wang2\*, Zaiquan Yang3, Zhen $\mathrm { Q u ^ { 4 } }$ , Liyuan $\mathrm { P a n } ^ { 1 , 5 }$ †, Xiangxiang Chu2 1Beijing Institute of Technology, 2AMAP, Alibaba Group 3City University of Hong Kong, 4Institute of Automation, Chinese Academy of Sciences, 5Yangtze Delta Region Academy of Beijing Institude of Technology, Jiaxing, China https://github.com/Simon98-AI/Vedas

# Abstract

Multimodal latent reasoning has emerged as a promising paradigm that replaces explicit Chain-of-Thought (CoT) decoding with implicit feature propagation, simultaneously enhancing representation informativeness and reducing inference latency. By analyzing token-level gradient dynamics during latent training, we reveal two critical observations: (1) visual tokens exhibit significantly higher and more volatile gradient norms than their textual counterparts due to inherent language bias, resulting in systematic visual underoptimization; and (2) semantically simple tokens converge rapidly, whereas complex tokens exhibit persistent gradient instability constrained by fixed architectural depths. To address these limitations, we propose a visual replay module and routing depth scaling to collaboratively enhance visual perception and refine complicated latents for deeper contextual reasoning. The former module leverages causal selfattention to estimate token saliency, reinforcing fine-grained grounding through spatially-coherent constraints. Complementarily, the latter mechanism adaptively allocates additional reasoning steps to complex tokens, enabling deeper contextual refinement. Guided by a curriculum strategy that progressively internalizes explicit CoT into compact latent representations, our framework achieves state-of-the-art performance across diverse benchmarks while delivering substantial inference speedups over explicit CoT baselines.

> 💡 **摘要批读**: 摘要的逻辑是“latent reasoning 更快，但视觉 token 在训练中不稳、复杂 token 受固定深度限制”。本文的两个模块正好一一对应：SCF-VR 负责把关键视觉区域重新带入 latent 推理，RDS 负责给复杂 token 额外 refinement。对 VisMem/医学图像阅读有用的点是：它把视觉证据保持问题转成 token 级优化和选择性计算预算问题，而不是只靠更长 CoT 或更多视觉 token。

> 💡 **批注**: 摘要还隐含一个重要判断：作者不是要否定 latent reasoning，而是指出它在视觉 token 上存在训练动力学失衡。因此 VEDS 更像“修 latent 优化”而不是“另造一套推理范式”。

---

## 🔖 Section 总结

### 核心洞察
1. Latent reasoning 的优势是少生成文本、推理更快；风险是中间推理不可见，视觉证据更容易在 hidden space 中被语言先验淹没。
2. 作者先做梯度诊断，再提出模块，叙事比单纯堆模块更可信。
3. 摘要中“visual replay”和“routing depth scaling”分别对应视觉证据保持与复杂 token 深推理，是后文读方法的两条主线。
