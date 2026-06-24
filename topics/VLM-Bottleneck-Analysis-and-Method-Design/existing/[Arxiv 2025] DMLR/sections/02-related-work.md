[← 返回 README](../README.md)

# 2. Related Work

## 一、Preview

本节梳理了两条脉络：(1) 显式推理从纯文本 CoT 发展到 Think-with-Images；(2) 潜空间推理从需要训练的方法发展到 training-free 方法，再到注入视觉信息的潜空间方法。

---

## 二、原始文本

Explicit Reasoning. Many prior works have explored visual reasoning. Early approaches mainly relied on semantic CoT, where the model performs all inference in the text space after a one-time visual encoding [5, 4, 20, 21]. However, this separation between perception and reasoning often leads to misalignment and hallucination [6, 22, 23, 24, 22, 25]. To address these limitations, recent studies adopt a Thinking-with-Images paradigm, where the model can draw auxiliary elements [13, 26, 27], zoom or crop regions [11, 8, 28, 29], or generate intermediate visual cues [30, 10, 9], enabling it to reason directly over visual structures.

> 💡 **机制拆解 — 显式推理的两代方法**:
> - **第一代**: Semantic CoT (如 LLaVA-OneVision, Kam-CoT)。在文本空间推理，视觉编码后一次性使用。问题：感知与推理的分离导致 misalignment 和幻觉。
> - **第二代**: Thinking-with-Images (如 GRiT, Pixel Reasoner, DeepEyes, Visual Sketchpad)。允许模型在推理过程中直接操作图像 (画辅助元素、缩放/裁剪、生成中间视觉线索)。问题：工具调用不稳定、推理开销大。

Latent Reasoning. Recently, an increasing number of studies have begun to shift reasoning from the explicit token space to the model's latent representation space. Some methods introduce dedicated training frameworks that optimize latent representations to support more effective internal reasoning [31, 14, 32, 33, 34, 35], while others propose training-free approaches that manipulate latent activations during inference to refine the reasoning process [15, 36, 37, 38, 39]. In addition, several recent works explore injecting visual information into the latent space [16, 17, 40, 41, 18], enabling models to iteratively operate over both latent semantic features and latent visual cues, thereby supporting a more flexible form of interleaved multimodal reasoning.

> 💡 **机制拆解 — 潜空间推理的三条路径**:
> - **需训练的潜空间推理**: COCONUT (训练 LLM 在连续潜空间推理), Fractional Reasoning (通过潜空间 steering vectors), ThinkAct, MILR 等。需要额外训练阶段，不够灵活。
> - **Training-free 的潜空间推理**: LatentSeek (测试时策略梯度), Soft Thinking (连续概念空间), LatentEvolve (测试时 scaling) 等。无需训练但多为纯语言任务。
> - **潜空间视觉注入**: Latent Visual Reasoning, Machine Mental Imagery, ICoT, Multimodal CoCT 等。在潜空间中注入视觉信息，但大多数方法注入方式固定，缺乏自适应性。
>
> DMLR 的定位：属于 training-free + 潜空间视觉注入的交集，且视觉注入是**动态自适应**的，这是与所有 prior work 最核心的区别。

---

## 三、Summary

- **显式推理**: 从纯文本 → Think-with-Images，逐步解决 visual grounding，但效率和稳定性仍有瓶颈
- **潜空间推理**: 从需训练 → Training-free → 潜空间视觉注入，DMLR 处于最前沿
- **DMLR 的差异化**: 是首个将 test-time 置信度引导优化与动态自适应视觉注入结合在潜空间中实现的框架
