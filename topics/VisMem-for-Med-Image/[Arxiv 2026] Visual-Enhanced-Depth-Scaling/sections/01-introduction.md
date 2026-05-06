[← 返回 README](../README.md)

# 1. Introduction

## 📌 预览
本节建立研究动机、现有方法缺口和贡献列表，是理解论文叙事的入口。

---

Over the past few years, large language models (LLMs) have achieved remarkable progress in complex reasoning, propelled by scaling laws in data volume and model capacity [26, 49]. Advanced techniques such as Chain-of-

> 💡 **动机批读**: 开篇先把多模态推理分成三条路：显式文本 CoT、工具增强视觉操作、latent reasoning。作者要解决的不是“是否需要推理”，而是显式 CoT 太慢且长链条会削弱视觉 grounding；工具调用虽然能看局部，但会引入无效操作和延迟。

Thought (CoT) prompting [46, 77] and reinforcement learning (RL) [9, 12, 70] for trajectory optimization have proven highly effective in text-only domains. Extending these capabilities to the multimodal realm has thus become a pivotal research direction. Current approaches primarily follow three paradigms. First, Text-based Reasoning [27, 40, 77] generates explicit multi-step textual chains before producing an answer. However, these methods typically rely on static visual inputs, and recent studies [68, 71] indicate that visual grounding deteriorates significantly over extended reasoning chains. Second, Tool-augmented Reasoning manipulates visual inputs through external operations (e.g., zooming or region enhancement) and injects intermediate visual hints into the reasoning trace. While powerful, these approaches are prone to redundant or invalid tool invocations, introducing noise that degrades performance and substantially increases inference latency. Recently, emerging researches position Latent Reasoning as a viable direction for optimizing multimodal reasoning. Unlike traditional methods that rely on explicit textual chains, latent reasoning encodes intermediate reasoning steps into compact continuous vectors. This paradigm offers compelling advantages, including higher inference efficiency, reduced annotation overhead, and the ability to learn dense, high-fidelity multimodal representations [51, 77].

> 💡 **范式定位**: Latent reasoning 的关键承诺是把中间推理压进连续向量，减少文本 token 解码成本。这里也埋下本文风险：如果中间步骤不可见，就必须保证视觉证据真的被 hidden states 携带，而不是被语言模式替代。

To gain deeper insights into the optimization dynamics of latent reasoning, we conducted a systematic analysis of gradient flows and parametric evolution during training. This investigation reveals two meaningful observations: (1) Visual-Text Optimization Disparity: Recent studies [67, 74] have highlighted the phenomenon of visual attention attenuation in MLLMs as the explicit Chain-of-Thought (CoT) reasoning chain extends. We discover that a similar degradation existing in latent reasoning. As depicted in Fig. 1a, visual tokens consistently exhibit significantly higher gradient norms and pronounced volatility compared to textual tokens. We reckon that this disparity stems from a fundamental mismatch between continuous visual features and discrete text tokens. During joint training, textual modality dominates the optimization process, causing visual representations to undergo large and erratic updates as they struggle to establish the fine-grained visual-text alignment. (2) Fixed-Depth Optimization Dilemma: Beyond modality imbalance, we observe an architecture bottleneck in how models handle samples with different complexity. Partitioning the training data into easy (consistently correct) and hard (persistently incorrect) subsets based on early-stage validation accuracy, we tracked the gradient nuclear norms across the QKV and output projection matrices O (Fig. 1b and Fig. 1c). The trends reveal a stark divergence: easy samples exhibit smooth gradient decay, indicating stable convergence into favorable loss basins. In contrast, hard samples maintain persistently high gradient volatility even in late training epochs. This phenomenon underscores the necessity of iterative refinement for complex reasoning, which is crucial for parsing compositional patterns in complex contexts while mitigating inherent visual ambiguities in images. Fixed-depth architectures, however, lack the flexibility to adapt to varying token complexities, thereby trapping hard examples in oscillatory optimization trends.

> 💡 **梯度诊断**: 两个观察分别对应两个设计动机。视觉 token 梯度范数更大且有尖峰，说明它们在文本主导训练中难以稳定对齐；hard samples 的 QKVO/LoRA 投影梯度长期震荡，说明统一固定深度对复杂样本不够。这里的“梯度大”不是能力强，而是优化不稳定。

![Figure 1.](../images/73a422dc1613fc717278894a2c4e3abe22f0d36970452ac37ef2b127b5ec7837.jpg)
*Figure 1.: Figure 1. Panel (a) depicts the token-wise Frobenius norm of gradients within different layer throughout the overall training process. Notably, visual tokens (i.e., those with indices within about [200, 280]) exhibit consistently larger gradient magnitudes accompanied by abrupt spikes, suggesting that they are more challenging to optimize compared to textual tokens (indices without [200, 280]). Panels (b) and (c) further reveal distinct evolution patterns of the Nuclear norm in the QKVO projection matrix across layers and training epochs: gradients for easy samples decay rapidly and converge smoothly, whereas hard samples maintain elevated gradient norms with persistent oscillations. Here, $\mathbf { W } _ { A }$ and $\mathbf { W } _ { B }$ denote the low-rank matrix in LoRA [19].*

> 💡 **Figure 1 批读**: Panel (a) 是视觉-文本优化不平衡的证据：视觉 token 大致落在索引 200-280，梯度更高、更尖。Panel (b)(c) 是固定深度瓶颈的证据：easy samples 梯度快速衰减，hard samples 到后期仍震荡。后文 SCF-VR 和 RDS 分别对应这两类现象。

Motivated by these observations, we propose a unified framework that strengthens fine-grained visual engagement and achieves token-wise depth scaling strategy to enable more precise and comprehensive contextual reasoning. Firstly, to mitigate visual optimization instability, we design a visual replay module, which dynamically replays the focused visual clues interleaved with thinking latents. This mechanism iteratively exposes and propagates key visual context across reasoning steps, fostering step-wise alignment with the target answer. Complementing this, we apply self-distillation supervision to enforce spatial coherence and preserve fine-grained visual details within the visual latents. Secondly, to address the fixed-depth optimization bottleneck, we propose a per-layer token router that dynamically allocates additional reasoning steps based on token complexity or information density. This design enables highdifficulty tokens to engage in prolonged contextual reasoning by reusing layer-wise knowledge, facilitating iterative representation refinement and adaptive prioritization of salient contextual cues. Finally, in contrast to methods [18, 47] that rely on knowledge distillation for direct latent supervision, we employ a curriculum learning strategy that progressively introduces latent tokens into the training pipeline. Extensive experiments show that our method can be effortlessly combined with various widely-used MLLM backbones to further enhance reasoning performance while maintain the satisfactory inference latency. The main contributions can be summarized as follows:

> 💡 **方法总览**: 这一段把贡献压成三个机制：visual replay 把注意到的视觉线索插入 latent steps；spatial self-distillation 约束视觉 latent 保持局部细节；token router 给难 token 追加层内推理。注意作者强调不依赖直接 latent supervision，而用 curriculum 从显式 CoT 逐步过渡。

We present a unified curriculum-driven framework that progressively constructs interleaved latent representations. By integrating spatially-coherent visual constraints for fine-grained grounding and complexity-aware depth scaling, our approach enables robust and precise contextual reasoning.

> 💡 **贡献批读**: 第一条贡献强调 curriculum-driven interleaved latents，说明 latent token 不是孤立追加，而是和视觉 replay token 交替参与上下文推理。这个设计对医学 VLM 的启发是：关键影像区域应该在推理过程中被反复引用，而不是只在 prefilling 时看一遍。

• We systematically analyze token-level gradient dynamics during latent reasoning training, revealing two critical optimization bottlenecks: visual-text optimization disparity and fixed-depth optimization dilemma.   
• Extensive experiments across twelve widely-used multimodal reasoning benchmarks demonstrate that our method achieves state-of-the-art performance while maintaining high inference efficiency.

> 💡 **贡献拆解**: 作者把 claim 限定在两个层面：训练动力学分析和 12 个 benchmark 实证。读实验时要检查这两点是否闭环：SCF-VR 是否主要提升细粒度视觉任务，RDS 是否在复杂推理任务更有收益，效率是否仍接近 latent baseline。

---

## 🔖 Section 总结

### 核心洞察
1. 论文动机不是泛化的“多模态推理更难”，而是 latent training 中视觉 token 不稳、hard token 需要更深 refinement。
2. SCF-VR 对应 visual-text optimization disparity；RDS 对应 fixed-depth optimization dilemma。
3. 对 VisMem 方向而言，本文提供了“视觉证据重放 + token 级深度调度”的可复用结构。
