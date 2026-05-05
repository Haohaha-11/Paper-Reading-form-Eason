[← 返回 README](../README.md)

# Introduction

## 📌 预览
本文件合并 Introduction/Related Work，重点读动机链、研究 gap、贡献列表和本文在 VisMem/Med Image 课题中的定位。

---

# 1 Introduction

Seasoned diagnostic experts do not rely on stepwise logical reasoning when making clinical diagnoses; instead, they activate Implicit Diagnostic Memory, that enables near-instantaneous pattern recognition against accumulated case knowledge [4, 35, 48]. Although medical vision-language models (VLMs) have made substantial progress in diagnostic assistance [6, 24, 31, 41, 51], with reinforcement learning from verifiable rewards [19, 37, 44, 45] and chain-of-thought (CoT) [7,12,21,49,50,52] further advancing reasoning capabilities. However, their intrinsic reliance on discrete tokens engenders a profound Cognitive Misalignment with the inherently continuous nature of clinical expertise. As illustrated in Fig. 1, the limited granularity of a fixed vocabulary is inadequate for representing continuous pathological features such as gradual transitions in lesion density or textural heterogeneity, and the autoregressive decoding mechanism is prone to progressive attenuation of visual evidence over extended reasoning chains. Moreover, discrete symbols tend to encode generic linguistic priors rather than dynamic anatomical context, readily giving rise to “pseudo-logical” hallucinations that lack grounding in physical evidence.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Figure 1](../images/8b6d6e968c37c0702eb6f9b54f958ef2b5aeddfdaab94ca33ddc107c9cdfb6c8.jpg)
*Figure 1: Fig. 1: Existing medical VLMs suffer from coarse symbolic granularity and long-range information dissipation in discrete reasoning. MedSynapse-V addresses this by evolving diagnostic implicit memory in latent space via anatomical prior condensation, causal counterfactual refinement, and autonomous latent memory internalization.*

> 💡 **Figure 1 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

An intuitive remedy is to supplement models with external diagnostic knowledge. Retrieval-augmented generation (RAG) prepends retrieved text fragments or similar cases to the input context [1, 53, 61, 63, 66], while soft-prompt and prefix-tuning methods concatenate learnable vectors to the input sequence to inject domain-specific cues [14,22,47]. However, both strategies inject information that remains static and causally unverified: it has undergone neither validation of causal relevance to the current diagnostic decision nor evolution into an intrinsic model capability through gradient-based optimization, persisting as a brittle external dependency prone to context saturation and information redundancy as the differential diagnosis space expands.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

Recent latent computation paradigms [15, 25, 43, 54] offer a principled alternative by performing reasoning in continuous hidden state spaces, circumventing the expressiveness bottleneck of discrete symbols. However, their direct application to medical scenarios encounters two domain-specific obstacles. First, without structured anatomical priors, latent representations degenerate into abstract vectors decoupled from clinical semantics: they capture statistical regularities of the training distribution but fail to encode the structured spatial relationships (organ topology, lesion morphology, tissue boundaries) essential for diagnostic grounding. Second, without causal calibration, the coupling between latent representations and diagnostically critical visual features remains weak, as the model may produce correct answers by exploiting spurious correlations (e.g., datasetspecific formatting cues) rather than attending to pathologically relevant regions, undermining reliability in clinical deployment.

> 💡 **批注**: 这是医学影像相关段落：关注病灶证据、跨模态差异、临床先验和可解释风险。

These observations converge on a fundamental question: Can a VLM progressively evolve its latent memory to simulate clinical intuition, enabling the rapid synthesis of case-adaptive diagnostic patterns, while ensuring this autonomous internal reasoning stream and its continuous refinement effectively steer the model toward clinically reliable decisions?

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

This paper proposes MedSynapse-V, a framework for latent diagnostic memory evolution that addresses this question through three synergistic mechanisms operating in a progressive training paradigm. First, Meta Query for Prior Memorization (§2.2) deploys learnable meta-query probes to retrieve multi-scale spatially aware features from a frozen anatomical encoder pre-trained on large-scale segmentation tasks, condensing them into compact diagnostic implicit memory vectors that are injected into the VLM’s hidden stream. This mechanism bridges the representation gap between the encoder’s anatomical feature space and the VLM’s generation space. Second, Causal Counterfactual Refinement (CCR; §2.3) performs reinforcement learning-driven memory optimization, introducing a novel causal counterfactual reward that quantifies the causal diagnostic contribution of each memory element through region-level feature masking interventions. By contrasting model behavior under original versus intervened memory conditions, CCR systematically prunes causally irrelevant components while reinforcing those with genuine diagnostic utility. Third, Intrinsic Memory Transition (IMT; §2.4) employs a privileged-autonomous dual-branch paradigm to distill the refined diagnostic patterns from a teacher branch (with the anatomical encoder) into a lightweight student branch via full-vocabulary Jensen-Shannon divergence alignment. At inference, the anatomical encoder is entirely removed, and the model generates diagnostic memory autonomously with computational overhead nearly identical to a standard VLM.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Comprehensive evaluations across seven medical multimodal benchmarks demonstrate that MedSynapse-V consistently outperforms a broad spectrum of state-of-the-art approaches, spanning medical-specific VLMs, RL-enhanced CoT paradigms, and general-purpose latent reasoning methods, in both diagnostic accuracy and cross-domain generalization, while introducing negligible additional inference cost compared with standard VLMs. Our main contributions are:

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

(1) We propose MedSynapse-V, the first framework that evolves diagnostic implicit memory in latent space for medical diagnosis, shifting from static external knowledge injection to progressive, autonomous memory internalization.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

(2) We design Meta Query–based Prior Memorization coupled with Causal Counterfactual Refinement (CCR), which distills anatomical priors into compact latent memory and calibrates it via counterfactual interventions to retain only causally grounded diagnostic components.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

(3) We introduce Intrinsic Memory Transition (IMT), a privileged–autonomous dual-branch distillation paradigm that internalizes encoder-dependent memory into autonomously generated intrinsic memory via full-vocabulary divergence alignment, eliminating all auxiliary modules at inference.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

(4) In multimodal benchmarks, MedSynapse-V consistently surpasses mainstream CoT paradigms in diagnostic accuracy while maintaining inference efficiency on par with standard VLMs, validating latent memory evolution as a principled alternative to discrete token reasoning.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

# 2.4 Intrinsic Memory Transition

After the causal RL, the model can generate high quality diagnostic outputs under the guidance of externally derived $\mathcal { M }$ , but the persistent dependence on the encoder at inference introduces additional computational overhead. As shown in Fig. 3, IMT reformulates this problem as learning to autonomously generate equivalent latent space embeddings under the condition of removing the auxiliary encoder, employing a privileged autonomous dual-branch paradigm to accomplish memory transition from extrinsic to intrinsic.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

![Figure 3](../images/798ccd39d0cc677d034bf05869e3b0b5f2286c3cdd5cadc7623c092b65f373fe.jpg)
*Figure 3: Fig. 3: Intrinsic Memory Transition (IMT) is achieved via Jensen–Shannon divergence alignment between the teacher $\pi ^ { + }$ , conditioned on encoder-derived $\mathcal { M } _ { p r i }$ ) and student ( $\pi ^ { - }$ , conditioned on $\mathcal { M } _ { a u t o }$ ) branches. Gradients propagate solely to $\mathcal { A } _ { \psi }$ , enabling complete removal of the anatomical encoder at inference with negligible overhead.*

> 💡 **Figure 3 批读**: 这张图通常展示框架、视觉案例或 latent/memory 流程。重点看视觉证据如何进入、保留或更新 latent memory。

Privileged Branch and Autonomous Branch. The teacher branch (privileged) retains the complete pipeline with the anatomical encoder, generating reference memory $\mathcal { M } _ { p r i } \ : = \ : \mathcal { P } _ { \phi } ( \mathcal { E } _ { a n a } ( X ) ) \ : \in \ : \mathbb { R } ^ { N \times d _ { h } }$ . The student branch (autonomous) removes the encoder and predicts equivalent memory solely through a lightweight Autonomous Memory Module $\mathcal { A } _ { \psi }$ mounted on the VLM, using the VLM’s own visual encoding features:

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Equation 1](../images/28d2ffcda150ca50aa83bae8b9cf41297df3c06167ba03e554798e002f4ab7f3.jpg)
*Equation 1: Equation extracted by MinerU.*

> 💡 **Equation 1 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

Both branches share the VLM backbone parameters $\theta$ , ensuring that alignment is completed within the same function space.

> 💡 **批注**: 这是跨模态 latent alignment：视觉特征必须进入 LLM 可用的语义空间。

Training Objective: Unified Full Vocabulary Divergence. The core signal of IMT is the full vocabulary divergence objective covering all generation positions. Given a trajectory $\hat { y } \sim \pi ^ { - } ( \cdot \mid X , q , \mathcal { M } _ { a u t o } )$ sampled from the student branch, the trajectory averaged per position divergence is defined as:

![Equation 2](../images/7b826d24e26a2d7bf91a6cf2b358fecca0d263e9a1ca2da48ba37e3e62120a9b.jpg)
*Equation 2: Equation extracted by MinerU.*

> 💡 **Equation 2 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\pi ^ { + } ( \cdot  { | \hat { y } _ { < n } } ) \triangleq \pi _ { \theta } ( \cdot  { | X , q , \mathcal { M } _ { p r i } , \hat { y } _ { < n } } )$ and $\pi ^ { - } ( \cdot  { | \hat { y } _ { < n } } ) \triangleq \pi _ { \theta } ( \cdot  { | \ X , q , \mathcal { M } _ { a u t o } , \hat { y } _ { < n } } )$ are the full vocabulary next-token distributions conditioned on privileged and autonomous memory, respectively. The divergence $D$ adopts the generalized Jensen–Shannon divergence JSD $\beta$ [27]:

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

![Equation 3](../images/13a4be65b1ef3cf8ceab2e5341adb8ba2a5cd0ba53ae1b5df1e347d838f75d08.jpg)
*Equation 3: Equation extracted by MinerU.*

> 💡 **Equation 3 批读**: 公式通常定义 latent update、memory transition、loss 或 routing；建议把符号对应到视觉证据、query、memory 和输出。

where $\bar { m } = \beta \pi ^ { + } + \left( 1 { - } \beta \right) \pi ^ { - }$ . The teacher branch leverages privileged memory to expose the complete next token probability landscape to the student, driving $\mathcal { A } _ { \psi }$ to learn to generate latent space embeddings behaviorally equivalent to privileged memory. Gradients backpropagate only through the student branch to $\mathcal { A } _ { \psi }$ , while the teacher branch serves as a fixed distributional target.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

At inference, $\mathcal { A } _ { \psi }$ directly generates $\mathcal { M } _ { a u t o }$ from VLM visual encoding features and injects them into the hidden stream. The entire Meta Query pipeline and the anatomical encoder are completely removed, rendering the computational overhead of MedSynapse-V nearly identical to that of a standard VLM.

# 12 Related Works

Latent Computation. Our method is closely related to latent computation, a paradigm that leverages continuous latent states to intervene in or reshape the generation process of large language models [10, 13, 59, 62]. Existing works can be categorized into two paradigms: (I) achieving native latent reasoning at the architectural level, represented by Coconut [15], CODI [43], LatentR3 [65], and CoLaR [46], where the reasoning process is inherently completed in continuous space; (II) leveraging latent computation to regulate generation quality, represented by LaRS [56], LatentSeek [25], SoftCoT [54, 55], and Coprocessor [30], which modulate output accuracy through latent space representations. The diagnostic implicit memory proposed in this paper can be viewed as an instantiation of the latter paradigm: condensing structured domain priors into continuous memory vectors to supply the diagnostic agent with critical incremental context. Unlike the aforementioned general purpose methods, MedSynapse-V further introduces causal counterfactual refinement and intrinsic memory transition mechanisms, enabling implicit memory to undergo progressive evolution from external prior dependency to intrinsic model capability.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Medical Alignment and Hallucination. The stringent safety requirements of medical diagnosis have driven model alignment strategies from generic preference imitation toward clinically constrained causal logic alignment [3,19,19,37,42,58]. Med-R1 and MedVLM-R1 [19,37], based on Group Relative Policy Optimization (GRPO) [42], enhance the robustness of reasoning through structured diagnostic rewards and logic backtracking mechanisms [36,39,40]. MMedPO [68] introduces contrastive optimization based on lesion region perturbation and clinical significance scoring, compelling the model to prioritize learning feature mappings with causal contributions to critical anatomical details during the alignment process [3, 58]. Several additional works [7, 23, 34, 49] explore explicitly anchoring the reasoning process in visual space to strengthen the utilization of spatial evidence. The alignment signals of the aforementioned methods all operate on the model’s output layer (the generated text sequence), whereas MedSynapse-V proposes a complementary perspective: CCR directly injects causal alignment signals into the model’s latent memory representations, and IMT internalizes clinically aligned biases into the model’s internal memory weights, enabling autonomous diagnostic capability at inference without external priors.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

---

## 🔖 Section 总结

### 核心洞察
1. 把本文 gap 和 VisMem for Med Image 主线对应起来。
2. 注意作者如何区分 visual grounding、latent memory、medical prior。
