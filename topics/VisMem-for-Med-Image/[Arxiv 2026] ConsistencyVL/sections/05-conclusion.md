# 5 Conclusion

[← 返回 README](../README.md)

---

## 7 Conclusion

This study reveals that reliability and causal robustness in current VLMs are highly architecture-dependent and not well captured by attention-map structure alone. We find a stark architectural divergence: early-fusion and cyclically refining models (PaliGemma, Qwen2-VL) distribute their truth representations, remaining resilient even when ~ 50% or more of their peak informational neurons are destroyed. Conversely, late-fusion models like LLaVA rely on localized, fragile late-stage bottlenecks.

For reliability prediction, stronger signals come from generation dynamics and internal-state probes: self-consistency provides the best behavioral proxy for correctness (R = 0.429), and hidden-state probes achieve high discrimination (AUROC > 0.95 on our strongest settings). Ultimately, these findings support a practical direction for trustworthy multimodal systems: use latent-state and consistency-based monitors rather than heatmap sharpness, and favor distributed, early-fusion architectures for causally robust multimodal reasoning.

> 💡 **核心启示 — 如何构建更可信的多模态系统**:
> 1. **监控机制**: 用潜在状态探针和一致性评分替代注意力热力图的"锐度"作为可信度指标
> 2. **架构选择**: 倾向于分布式、早期融合的架构（如PaliGemma/Qwen2-VL），因为它们的可靠性路径不依赖脆弱的单点瓶颈
> 3. **部署策略**: 在实时应用中，使用单次推理的隐藏状态探针（额外开销仅一个线性层）；在允许更高延迟的场景，使用Self-Consistency作为金标准

---

## Appendix A.15: Limitations and Future Work

**Model Scale:** Our study focuses on three mid-scale open VLMs. It is possible that larger models (e.g., LLaVA-34B or GPT-4V) exhibit stronger alignment between attention and truthfulness due to better reinforcement learning from human feedback (RLHF).

> 💡 **Q&A 批注记录**:
> *Q: 这个结论对大模型（如GPT-4V、Gemini Pro Vision）有多大的可迁移性？*
> A: 作者明确承认这是研究的主要限制之一。理论上，RLHF训练可能通过偏好优化将注意力和真实性更好地对齐——因为人类反馈天然地奖励"看对的回答"。但这只是一个假设，未经实验验证。也有可能：更大的模型虽然在任务上表现更好，但注意力-可信度之间的脱钩仍然存在（甚至更大，因为更大的LLM有更强的语言先验）。

**Computational Cost:** The most reliable metric found, Self-Consistency, requires K = 10 inference passes. This is prohibitively expensive for low-latency edge applications.

> 💡 **效率-可靠性权衡**:
> Self-Consistency的10x推理代价是其主要实用障碍。但Hidden-State Probe提供了两全其美的可能：(1) AUROC可达到或超过SC；(2) 额外开销仅为一个线性层；(3) 单次推理即可输出可信度评分。

**Causal Evidence Scope:** While our ablation experiments demonstrate causal effects of probe-identified neurons (8.3% accuracy drop for top-5 vs. 0% for random), the effect requires ablating multiple neurons simultaneously, suggesting a localized circuit rather than individual "truth units." The effect is also moderate in magnitude, indicating these neurons are contributors to reliability rather than sole determinants. Future work should explore activation patching and interchange interventions to further characterize the causal mechanism.

> 💡 **消融解读 — 因果证据的范围与局限**:
> - 效应量(-2.0pp总体, -8.3pp对象识别)虽然显著但不算大，说明被识别的神经元是可信度的"贡献者"而非"决定者"
> - 需要同时消融多个神经元才有效果，确认为"局部电路"而非"真理神经元"
> - 未对PaliGemma/Qwen2-VL做同级别的小规模精细消融（因为它们的大规模消融已经显示无影响）——分布式架构中可能根本不存在可定位的"可信度神经元"

**Future Direction:** We propose that future work should focus on distillation. Since Self-Consistency provides a high-quality "silver label" for reliability (R = 0.43), we can curate a dataset of (Image, Question, Answer, SC-Score) and fine-tune a value head on top of the VLM to predict the SC-Score in a single pass. This would combine the accuracy of consistency with the efficiency of a probe.

> 💡 **自一致性蒸馏的愿景**: 
> 作者提出的蒸馏思路非常现实可行：(1) 用SC=10生成高质量"银标签"(R=0.43)；(2) 构建(图像, 问题, 答案, SC分数)数据集；(3) 微调一个value head预测SC分数；(4) 部署时单次推理即可输出可信度。这本质上是将"多路径一致性"的知识压缩到"单路径预测"中，类似于模型蒸馏的核心思想。

---

## Appendix A.14: Reliability vs. Efficiency Trade-offs

While Self-Consistency (SC) is the gold standard for reliability (R = 0.43), it comes at a high computational cost: it requires K = 10 forward passes. For real-time applications (e.g., robotics), this is often prohibitive.

Our Hidden State Probe offers a compelling alternative:

- **Self-Consistency:** High Accuracy (AUROC = 0.78), High Cost (10x inference).
- **Learned Probe:** Moderate to High Accuracy (up to AUROC = 0.96 on family-specific splits), Zero Cost (overhead of a single linear layer).
- **Visual Metrics:** Low Accuracy (AUROC = 0.50), Low Cost.

The success of the Hidden State Probe confirms that the model's reliability is encoded in the linear subspace of the final residual stream. This aligns with recent work in "Lie Detection" for LLMs, extending it to the multimodal domain. Future work should focus on distilling the signal from Self-Consistency into a single-pass value head, effectively training the model to predict its own consistency score.

> 💡 **公式批读 — 隐藏状态探针的线性性**:
> 探针只是一个线性分类器: p(reliable|h) = σ(w^T h + b)
> 这意味着可信度信号以线性可分的模式编码在残差流中。这种线性可分性是一个重要的观察——它意味着可以通过简单的线性变换提取可信度，而不需要复杂的非线性机制。这为高效部署提供了理论基础。

---

## Appendix A.13: Cross-Family Interpretation

Across all three families, the same reliability taxonomy appears with model-specific signatures. LLaVA-1.5 exhibits the strongest symbolic-detachment gap (early lock, late diffusion), which aligns with high probe separability in late layers. PaliGemma-3B integrates visual evidence earlier and more smoothly, yielding weaker late-layer separability and lower probe AUROC (0.738). Qwen2-VL-7B shows cyclical refinement and strong late-stage re-separation, consistent with high probe AUROC (0.971).

These differences suggest that reliability probing should be architecturally adaptive (e.g., layer selection and probe capacity per family), rather than assuming a one-size-fits-all late-layer template.

> 💡 **机制拆解 — 跨架构统一分类法**:
> 三个模型家族呈现出相同的可信度分类法但具有模型特定的特征：
> | 特征 | LLaVA | PaliGemma | Qwen2-VL |
> |------|-------|-----------|----------|
> | Symbolic-Detachment Gap | 最大 | 最小 | 中等 |
> | 探针层选择 | L31(最末) | L14(中间) | L25-27(后期) |
> | 探针AUROC | 0.956 | 0.738 | 0.971 |
> | 因果关系 | 局部脆弱 | 全局鲁棒 | 全局鲁棒 |
> 
> 关键启示：**可靠性探测必须是架构自适应的**——不存在"最后一层就是最佳探测位置"的通用法则。

---

**📌 Preview:** 结论部分归纳了全文的核心发现：VLM的可靠性信号存在于生成动力学和内部状态中，而非注意力图中。同时指出了三个重要限制（模型规模、计算成本和因果证据范围），并为未来的蒸馏方向提供了具体路径。附录中深入讨论了可靠性-效率权衡和跨架构分类法。

**🔖 Summary:** 本文的最终立场是：构建可信赖的多模态系统需要从三方面入手——(1) 使用潜在状态和一致性监控替代热力图锐度作为可信度指标；(2) 倾向于分布式、早期融合的架构以获得因果鲁棒性；(3) 通过蒸馏将Self-Consistency的信号压缩到单次推理中。这三点构成了一个完整的"可信度工程"路线图。
