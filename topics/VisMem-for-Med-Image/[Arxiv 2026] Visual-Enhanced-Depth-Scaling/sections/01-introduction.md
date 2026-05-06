[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
本节建立研究动机、现有方法缺口和贡献列表，是理解论文叙事的入口。

---

Over the past few years, large language models (LLMs) have achieved remarkable progress in complex reasoning, propelled by scaling laws in data volume and model capacity [26, 49]. Advanced techniques such as Chain-of-

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

Thought (CoT) prompting [46, 77] and reinforcement learning (RL) [9, 12, 70] for trajectory optimization have proven highly effective in text-only domains. Extending these capabilities to the multimodal realm has thus become a pivotal research direction. Current approaches primarily follow three paradigms. First, Text-based Reasoning [27, 40, 77] generates explicit multi-step textual chains before producing an answer. However, these methods typically rely on static visual inputs, and recent studies [68, 71] indicate that visual grounding deteriorates significantly over extended reasoning chains. Second, Tool-augmented Reasoning manipulates visual inputs through external operations (e.g., zooming or region enhancement) and injects intermediate visual hints into the reasoning trace. While powerful, these approaches are prone to redundant or invalid tool invocations, introducing noise that degrades performance and substantially increases inference latency. Recently, emerging researches position Latent Reasoning as a viable direction for optimizing multimodal reasoning. Unlike traditional methods that rely on explicit textual chains, latent reasoning encodes intermediate reasoning steps into compact continuous vectors. This paradigm offers compelling advantages, including higher inference efficiency, reduced annotation overhead, and the ability to learn dense, high-fidelity multimodal representations [51, 77].

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

To gain deeper insights into the optimization dynamics of latent reasoning, we conducted a systematic analysis of gradient flows and parametric evolution during training. This investigation reveals two meaningful observations: (1) Visual-Text Optimization Disparity: Recent studies [67, 74] have highlighted the phenomenon of visual attention attenuation in MLLMs as the explicit Chain-of-Thought (CoT) reasoning chain extends. We discover that a similar degradation existing in latent reasoning. As depicted in Fig. 1a, visual tokens consistently exhibit significantly higher gradient norms and pronounced volatility compared to textual tokens. We reckon that this disparity stems from a fundamental mismatch between continuous visual features and discrete text tokens. During joint training, textual modality dominates the optimization process, causing visual representations to undergo large and erratic updates as they struggle to establish the fine-grained visual-text alignment. (2) Fixed-Depth Optimization Dilemma: Beyond modality imbalance, we observe an architecture bottleneck in how models handle samples with different complexity. Partitioning the training data into easy (consistently correct) and hard (persistently incorrect) subsets based on early-stage validation accuracy, we tracked the gradient nuclear norms across the QKV and output projection matrices O (Fig. 1b and Fig. 1c). The trends reveal a stark divergence: easy samples exhibit smooth gradient decay, indicating stable convergence into favorable loss basins. In contrast, hard samples maintain persistently high gradient volatility even in late training epochs. This phenomenon underscores the necessity of iterative refinement for complex reasoning, which is crucial for parsing compositional patterns in complex contexts while mitigating inherent visual ambiguities in images. Fixed-depth architectures, however, lack the flexibility to adapt to varying token complexities, thereby trapping hard examples in oscillatory optimization trends.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

![Figure 1.](../images/73a422dc1613fc717278894a2c4e3abe22f0d36970452ac37ef2b127b5ec7837.jpg)
*Figure 1.: Figure 1. Panel (a) depicts the token-wise Frobenius norm of gradients within different layer throughout the overall training process. Notably, visual tokens (i.e., those with indices within about [200, 280]) exhibit consistently larger gradient magnitudes accompanied by abrupt spikes, suggesting that they are more challenging to optimize compared to textual tokens (indices without [200, 280]). Panels (b) and (c) further reveal distinct evolution patterns of the Nuclear norm in the QKVO projection matrix across layers and training epochs: gradients for easy samples decay rapidly and converge smoothly, whereas hard samples maintain elevated gradient norms with persistent oscillations. Here, $\mathbf { W } _ { A }$ and $\mathbf { W } _ { B }$ denote the low-rank matrix in LoRA [19].*

> 💡 **Figure 1. 批读**: 这张图通常承担方法框架、动机或视觉对比作用；重点看它支撑的是机制、效果还是局限。

Motivated by these observations, we propose a unified framework that strengthens fine-grained visual engagement and achieves token-wise depth scaling strategy to enable more precise and comprehensive contextual reasoning. Firstly, to mitigate visual optimization instability, we design a visual replay module, which dynamically replays the focused visual clues interleaved with thinking latents. This mechanism iteratively exposes and propagates key visual context across reasoning steps, fostering step-wise alignment with the target answer. Complementing this, we apply self-distillation supervision to enforce spatial coherence and preserve fine-grained visual details within the visual latents. Secondly, to address the fixed-depth optimization bottleneck, we propose a per-layer token router that dynamically allocates additional reasoning steps based on token complexity or information density. This design enables highdifficulty tokens to engage in prolonged contextual reasoning by reusing layer-wise knowledge, facilitating iterative representation refinement and adaptive prioritization of salient contextual cues. Finally, in contrast to methods [18, 47] that rely on knowledge distillation for direct latent supervision, we employ a curriculum learning strategy that progressively introduces latent tokens into the training pipeline. Extensive experiments show that our method can be effortlessly combined with various widely-used MLLM backbones to further enhance reasoning performance while maintain the satisfactory inference latency. The main contributions can be summarized as follows:

> 💡 **批注**: 这是蒸馏逻辑：重点看 teacher 的什么能力被迁移给 student，以及训练/推理阶段各自保留哪些模块。

We present a unified curriculum-driven framework that progressively constructs interleaved latent representations. By integrating spatially-coherent visual constraints for fine-grained grounding and complexity-aware depth scaling, our approach enables robust and precise contextual reasoning.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

• We systematically analyze token-level gradient dynamics during latent reasoning training, revealing two critical optimization bottlenecks: visual-text optimization disparity and fixed-depth optimization dilemma.   
• Extensive experiments across twelve widely-used multimodal reasoning benchmarks demonstrate that our method achieves state-of-the-art performance while maintaining high inference efficiency.

> 💡 **列表批读**: 这组条目通常是在列贡献、设置、发现或模块；建议逐条对应到论文 claim。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
