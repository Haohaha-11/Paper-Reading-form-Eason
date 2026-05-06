[← 返回 README](../README.md)

# 1 Introduction

## 📌 预览
本节把问题从“医学 VLM 是否会推理”改写为“医学 VLM 能否在连续 latent space 中调用并演化临床经验”。读的时候抓住三条线：离散 CoT 的认知错配、外部知识注入的静态/不可验证缺陷、MedSynapse-V 的三阶段 memory evolution。

---

# MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution

Chunzheng Zhu $^ { 1 }$ , Jiaqi Zeng1, Junyu Jiang1, Jianxin Lin $\bot$ ⋆, and Yijun Wang1

Hunan University, Changsha, China {zhuchzh, zjqxxl, jiangjy, linjianxin, wyjun}@hnu.edu.cn

Abstract. High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose MedSynapse-V, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model’s hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR) which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privilegedautonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets demonstrate that MedSynapse-V, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-theart methods, particularly Chain-of-Thought (CoT) paradigms, in diagnostic accuracy and multi-dataset generalization without compromising the inference efficiency of standard VLMs.

> 💡 **摘要批读**: 摘要已经给出完整数据流：医学图像和问题进入 VLM；Meta Query 从解剖先验编码器抽取结构化 prior，合成隐式诊断记忆；CCR 用区域 mask 的反事实 reward 检查每个 memory 是否真的影响诊断；IMT 再把依赖外部编码器的 teacher 行为蒸馏给 autonomous student。核心 claim 不是“多一步推理”，而是把外部专家视觉先验变成模型 hidden stream 里的内生能力。

Keywords: VLMs · Implicit Diagnostic Memory $\cdot$ Latent Space Memory · Causal Counterfactual · Memory Distillation

> 💡 **关键词批读**: 这里的 memory 是 continuous hidden-state memory，不是检索库中的显式病例文本；counterfactual 也不是语言自我反思，而是对图像区域特征做 mask intervention，观察诊断分布变化。

# 1 Introduction

Seasoned diagnostic experts do not rely on stepwise logical reasoning when making clinical diagnoses; instead, they activate Implicit Diagnostic Memory, that enables near-instantaneous pattern recognition against accumulated case knowledge [4, 35, 48]. Although medical vision-language models (VLMs) have made substantial progress in diagnostic assistance [6, 24, 31, 41, 51], with reinforcement learning from verifiable rewards [19, 37, 44, 45] and chain-of-thought (CoT) [7,12,21,49,50,52] further advancing reasoning capabilities. However, their intrinsic reliance on discrete tokens engenders a profound Cognitive Misalignment with the inherently continuous nature of clinical expertise. As illustrated in Fig. 1, the limited granularity of a fixed vocabulary is inadequate for representing continuous pathological features such as gradual transitions in lesion density or textural heterogeneity, and the autoregressive decoding mechanism is prone to progressive attenuation of visual evidence over extended reasoning chains. Moreover, discrete symbols tend to encode generic linguistic priors rather than dynamic anatomical context, readily giving rise to “pseudo-logical” hallucinations that lack grounding in physical evidence.

> 💡 **动机批读**: 作者把医学 CoT 的问题归因到 discrete tokenization：病灶密度梯度、纹理异质性、边界过渡等连续视觉线索被压成有限词表；长输出还会让早期视觉证据在自回归链里衰减。这个论证解释了为什么“更长的文本推理”不一定更临床可靠。

![Figure 1](../images/8b6d6e968c37c0702eb6f9b54f958ef2b5aeddfdaab94ca33ddc107c9cdfb6c8.jpg)
*Figure 1: Fig. 1: Existing medical VLMs suffer from coarse symbolic granularity and long-range information dissipation in discrete reasoning. MedSynapse-V addresses this by evolving diagnostic implicit memory in latent space via anatomical prior condensation, causal counterfactual refinement, and autonomous latent memory internalization.*

> 💡 **Figure 1 批读**: 图 1 是整篇的概念总图：左侧批评 coarse symbolic granularity 和 long-range dissipation，右侧给出 MedSynapse-V 的替代路径：anatomical prior condensation → causal counterfactual refinement → autonomous latent memory internalization。它把“临床直觉”落到可训练模块，而不是停留在类比。

An intuitive remedy is to supplement models with external diagnostic knowledge. Retrieval-augmented generation (RAG) prepends retrieved text fragments or similar cases to the input context [1, 53, 61, 63, 66], while soft-prompt and prefix-tuning methods concatenate learnable vectors to the input sequence to inject domain-specific cues [14,22,47]. However, both strategies inject information that remains static and causally unverified: it has undergone neither validation of causal relevance to the current diagnostic decision nor evolution into an intrinsic model capability through gradient-based optimization, persisting as a brittle external dependency prone to context saturation and information redundancy as the differential diagnosis space expands.

> 💡 **外部知识批读**: RAG、similar-case retrieval、soft prompt/prefix 都能补知识，但它们多数是静态拼接：不知道当前病灶是否真的需要这条知识，也不会通过梯度训练变成模型自身能力。这里为 IMT 的“去外部依赖”埋伏笔。

Recent latent computation paradigms [15, 25, 43, 54] offer a principled alternative by performing reasoning in continuous hidden state spaces, circumventing the expressiveness bottleneck of discrete symbols. However, their direct application to medical scenarios encounters two domain-specific obstacles. First, without structured anatomical priors, latent representations degenerate into abstract vectors decoupled from clinical semantics: they capture statistical regularities of the training distribution but fail to encode the structured spatial relationships (organ topology, lesion morphology, tissue boundaries) essential for diagnostic grounding. Second, without causal calibration, the coupling between latent representations and diagnostically critical visual features remains weak, as the model may produce correct answers by exploiting spurious correlations (e.g., datasetspecific formatting cues) rather than attending to pathologically relevant regions, undermining reliability in clinical deployment.

> 💡 **latent 方法缺口**: 普通 latent reasoning 可以绕开离散 token 瓶颈，但医学场景还要求 anatomical grounding 和 causal calibration。没有结构化解剖 prior，latent vector 只是统计压缩；没有反事实校准，模型可能仍靠数据格式、模态偏见或常见答案模式答题。

These observations converge on a fundamental question: Can a VLM progressively evolve its latent memory to simulate clinical intuition, enabling the rapid synthesis of case-adaptive diagnostic patterns, while ensuring this autonomous internal reasoning stream and its continuous refinement effectively steer the model toward clinically reliable decisions?

> 💡 **研究问题**: 这句话把论文目标说得最直接：latent memory 要同时满足 case-adaptive、progressive evolution、autonomous internal reasoning、clinical reliability。后续三个模块分别对应“合成、校准、内化”。

This paper proposes MedSynapse-V, a framework for latent diagnostic memory evolution that addresses this question through three synergistic mechanisms operating in a progressive training paradigm. First, Meta Query for Prior Memorization (§2.2) deploys learnable meta-query probes to retrieve multi-scale spatially aware features from a frozen anatomical encoder pre-trained on large-scale segmentation tasks, condensing them into compact diagnostic implicit memory vectors that are injected into the VLM’s hidden stream. This mechanism bridges the representation gap between the encoder’s anatomical feature space and the VLM’s generation space. Second, Causal Counterfactual Refinement (CCR; §2.3) performs reinforcement learning-driven memory optimization, introducing a novel causal counterfactual reward that quantifies the causal diagnostic contribution of each memory element through region-level feature masking interventions. By contrasting model behavior under original versus intervened memory conditions, CCR systematically prunes causally irrelevant components while reinforcing those with genuine diagnostic utility. Third, Intrinsic Memory Transition (IMT; §2.4) employs a privileged-autonomous dual-branch paradigm to distill the refined diagnostic patterns from a teacher branch (with the anatomical encoder) into a lightweight student branch via full-vocabulary Jensen-Shannon divergence alignment. At inference, the anatomical encoder is entirely removed, and the model generates diagnostic memory autonomously with computational overhead nearly identical to a standard VLM.

> 💡 **方法总览**: 三阶段是递进关系，不是并列插件。MQPM 解决“从哪里来”的问题，把 MedSAM3 的解剖空间特征变成 VLM 可读的 16 个 memory vectors；CCR 解决“是否真有诊断因果作用”的问题；IMT 解决“推理时是否还依赖外部专家编码器”的问题。

Comprehensive evaluations across seven medical multimodal benchmarks demonstrate that MedSynapse-V consistently outperforms a broad spectrum of state-of-the-art approaches, spanning medical-specific VLMs, RL-enhanced CoT paradigms, and general-purpose latent reasoning methods, in both diagnostic accuracy and cross-domain generalization, while introducing negligible additional inference cost compared with standard VLMs. Our main contributions are:

> 💡 **实验承诺**: 评估覆盖 closed-ended VQA、clinical reasoning、expert-level reasoning 和 multi-granularity benchmark。作者强调不牺牲标准 VLM 推理效率，后面要用 latency、token length、encoder removal gap 来检验这个承诺。

(1) We propose MedSynapse-V, the first framework that evolves diagnostic implicit memory in latent space for medical diagnosis, shifting from static external knowledge injection to progressive, autonomous memory internalization.

> 💡 **贡献 1 批读**: “first framework” 的实质是把诊断记忆放进 latent space 并做 progressive internalization。它区别于 RAG 的关键在于：memory 最终要成为模型内生路径，而不是每次推理都依赖检索文本或外部 encoder。

(2) We design Meta Query–based Prior Memorization coupled with Causal Counterfactual Refinement (CCR), which distills anatomical priors into compact latent memory and calibrates it via counterfactual interventions to retain only causally grounded diagnostic components.

> 💡 **贡献 2 批读**: MQPM + CCR 是论文最有医学味道的组合：前者把 anatomy-aware features 压缩为 memory，后者用 mask intervention 过滤“看似有用但和诊断区域无关”的 prior。可复用点是把 segmentation/region prior 变成 VLM hidden conditioning。

(3) We introduce Intrinsic Memory Transition (IMT), a privileged–autonomous dual-branch distillation paradigm that internalizes encoder-dependent memory into autonomously generated intrinsic memory via full-vocabulary divergence alignment, eliminating all auxiliary modules at inference.

> 💡 **贡献 3 批读**: IMT 的价值在部署侧：训练时可以借助 privileged anatomical encoder，推理时只保留 autonomous memory module。full-vocabulary divergence 比只蒸馏最终答案更强，因为它约束每一步 next-token distribution 的诊断倾向。

(4) In multimodal benchmarks, MedSynapse-V consistently surpasses mainstream CoT paradigms in diagnostic accuracy while maintaining inference efficiency on par with standard VLMs, validating latent memory evolution as a principled alternative to discrete token reasoning.

> 💡 **贡献 4 批读**: 作者把 MedSynapse-V 定位为 CoT 的替代而非增强：少生成文本 token，避免 hallucinated observation 在推理链中滚雪球，同时保持准确率优势。这一点需要结合 Fig. 5/13 的病例对比和 Fig. 6/Table 5 的效率结果一起读。

---

## 🔖 Section 总结

### 核心洞察
1. Introduction 的核心不是泛泛讲医学 VLM，而是提出“离散符号推理与连续临床视觉经验错配”这个问题定义。
2. MedSynapse-V 的三阶段分别对应 memory 的生成、因果校准和内化；这三步缺一不可。
3. 读后续实验时应检查：latent memory 是否真的绑定病灶区域、去掉外部 encoder 后是否仍有效、是否比 CoT 同时更准更快。
