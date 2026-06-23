# Abstract

[← 返回 README](../README.md)

---

Multimodal Foundation Models (MFMs) are rapidly evolving from simple pattern matchers into reasoning agents. As these systems are used in higher-stakes settings, reliability, or knowing when a model may hallucinate, becomes critical. A common intuition in the field, which we call the Attention-Confidence Assumption, is that reliability follows from "structural" visual perception: if a model focuses tightly on relevant image regions, its subsequent answer should be trustworthy. Conversely, scattered attention is often assumed to signal confusion.

> 💡 **问题动机**: 本文的核心问题意识在于挑战一个被VLM社区广泛默认的假设——"Attention-Confidence Assumption"（注意力-置信度假设）。这个假设认为：模型在空间上聚焦得越精准，其回答应该越可靠。这种直觉来自于人类认知中"看得清=判断准"的朴素类比，但作者指出，这个类比在VLM中可能根本不成立。核心动机是：**我们能否在不依赖注意力热力图的情况下，更可靠地判断VLM输出是否可信？**

We challenge this assumption through VLM Reliability Probe (VRP), a systematic cross-family investigation into reliability signals in contemporary Vision-Language Models (VLMs). We introduce "structural attention" metrics, including cluster counts (C_k) and spatial entropy (H_s) to quantify the coherence of the visual encoder's gaze. To capture the dynamics of this gaze, we further track attention evolution (ΔH_s) across all layers. This analysis reveals a critical "Symbolic Detachment": models often exhibit "Early Locking" of visual features only to diffuse attention in later layers, effectively severing the link between early perception and final generation.

> 💡 **机制拆解 — Symbolic Detachment（符号性脱钩）**: 这是本文最核心的理论发现。"Early Locking" 指VLM在浅层就快速确定了注意力焦点（Layer 2即形成尖锐的注意力分布），然后在中间几十层保持不变，直到最后一层突然扩散。这意味着：(1) 注意力图是"过时的"感知残留，而非当前推理状态的反映；(2) 早期视觉感知和最终语言生成之间存在一道鸿沟——模型"看"的是一回事，"说"的是另一回事。这种脱钩是注意力与正确性之间零相关的根本原因。

Contrary to the grounding hypothesis, our results demonstrate a "Cluster Failure": spatial attention patterns possess near-zero correlation (R ≈ 0.001) with model accuracy. Instead, we find that reliability is fundamentally a phenomenon of generation dynamics and internal state distributions. Self-Consistency (SC), the agreement rate across sampled reasoning paths, emerges as the dominant predictor of truth (R = 0.429). By aggressively scaling causal interventions, we further demonstrate a massive architectural divergence: LLaVA "locks" its prediction in a fragile late-stage structural bottleneck, whereas PaliGemma and Qwen2-VL distribute reliability globally, showing extreme resilience even when ~ 50% or more of their most predictive layer is destroyed. These findings suggest that for current VLMs, reliability signals are detached from visual grounding maps, and are best inferred from generation-time dynamics and hidden-state probes.

> 💡 **机制拆解 — 核心三发现概述**:
> 1. **Visuals Lie (视觉说谎)**: 注意力空间结构与正确性近零相关 (R ≈ 0.001)，即看对区域 ≠ 答对问题
> 2. **Consistency Speaks (一致性说话)**: Self-Consistency (采样多条推理路径的一致性) 是最强的行为信号 (R = 0.429)
> 3. **Causal Architectures Diverge (因果架构分化)**: LLaVA将可靠性集中在脆弱的后层瓶颈，PaliGemma/Qwen2-VL将其分布到全局，即使摧毁>50%的神经元仍保持鲁棒

> 💡 **Q&A 批注记录**:
> *Q: 为什么R值（皮尔逊相关系数）只有0.429就算"强预测器"？*
> A: 在二元分类（正确/错误）的语境下，Point-Biserial Correlation的理论最大值通常远低于1.0（受限于base rate和信噪比）。0.429意味着SC解释了约18%的方差，在可靠性预测领域已经是显著信号。作为对比，注意力的R^2 < 0.08（解释方差<8%）。此外，SC=1时精度可达90.8%，说明在高置信区间内实际效用很强。

> *Q: "50% or more of their most predictive layer" 具体是什么含义？*
> A: 指的是在模型的最具预测性的层（PaliGemma的Layer 15，Qwen2-VL的Layer 25）中随机消融50%以上的隐藏维度神经元。PaliGemma的2048维中消融1000个（~49%），Qwen2-VL的3584维中消融2000个（~56%），模型准确率几乎没有变化。这说明这些模型的可信度信息不是存储在少数关键神经元中，而是高度分布式的。

---

**📌 Preview:** 本文的摘要已经完整勾勒了研究的全貌：从质疑Attention-Confidence假设出发，通过三层递进分析（结构指标→机制探针→行为指标），揭示了VLM中信度信号的真实来源。核心理论贡献是"Symbolic Detachment"概念和"Cluster Failure"现象的发现。

**🔖 Summary:** 该研究系统性地证伪了"空间注意力质量 = 模型可靠性"的直觉假设。通过VRP框架对三族VLM的跨架构分析，发现：(1) 注意力集群数/熵与正确性近零相关；(2) Self-Consistency是预测真实性的最佳行为信号；(3) 隐藏状态探针能以AUROC>0.95区分正确与错误回答；(4) 不同架构的可信度因果路径存在根本性分化。核心启示：与其看模型"看哪里"，不如听模型"怎么说"。
