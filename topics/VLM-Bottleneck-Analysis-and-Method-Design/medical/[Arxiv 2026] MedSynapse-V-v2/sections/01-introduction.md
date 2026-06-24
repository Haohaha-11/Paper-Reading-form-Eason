[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文的核心论证链：资深临床诊断依赖隐性记忆而非显式逻辑推理 -> 现有医学 VLM 由于离散 tokenization 产生认知错位 -> 现有补救方案(RAG/soft-prompt)注入静态外部知识但缺乏因果验证 -> 最新的潜空间推理方法缺乏解剖先验和因果校准 -> 提出 MedSynapse-V 三阶段 latent memory evolution 框架 -> 系统验证其有效性。

---

## 二、原始文本

Shaoed diagnostic experts do not rely on stepwise logical reasoning when making clinical diagnoses; instead, they activate Implicit Diagnostic Memory, that enables near-instantaneous pattern recognition against accumulated case knowledge [3, 82, 106]. Although medical vision-language models (VLMs) have made substantial progress in diagnostic assistance [7, 53, 79, 94, 117], with reinforcement learning from verifiable rewards [47, 84, 99, 100] and chain-of-thought (CoT) [8, 21, 49, 111, 113, 118] further advancing reasoning capabilities. However, their intrinsic reliance on discrete tokens engenders a profound Cognitive Misalignment with the inherently continuous nature of clinical expertise. As illustrated in Fig. 1, the limited granularity of a fixed vocabulary is inadequate for representing continuous pathological features such as gradual transitions in lesion density or textural heterogeneity, and the autoregressive decoding mechanism is prone to progressive attenuation of visual evidence over extended reasoning chains. Moreover, discrete symbols tend to encode generic linguistic priors rather than dynamic anatomical context, readily giving rise to "pseudo-logical" hallucinations that lack grounding in physical evidence.

> **机制拆解 — 离散 Tokenization 的三重认知错位**:
>
> | 维度 | 问题 | 机制 | 后果 |
> |------|------|------|------|
> | 表示粒度 | 量化损失 (Quantization Loss) | 固定词汇表粒度不足以表示连续病理特征(如病灶密度渐变、纹理异质性) | 病理细节在 tokenization 过程中被粗粒度近似，信息丢失 |
> | 序列演化 | 长程信息消散 (Long-range Info Dissipation) | 自回归解码中，早期视觉证据随 token 序列增长而逐步衰减 | 长推理链末端的决策缺乏初始视觉 grounding |
> | 语义偏差 | 通用语言先验主导 (Generic Linguistic Prior) | 离散符号编码的是统计语言规律而非动态解剖上下文 | "伪逻辑"幻觉——推理看似合理但缺乏物理证据支撑 |

An intuitive remedy is to supplement models with external diagnostic knowledge. Retrieval-augmented generation (RAG) prepends retrieved text fragments or similar cases to the input context [1, 119, 133, 136, 150], while soft-prompt and prefix-tuning methods concatenate learnable vectors to the input sequence to inject domain-specific cues [24, 51, 105]. However, both strategies inject information that remains static and causally unverified: it has undergone neither validation of causal relevance to the current diagnostic decision nor evolution into an intrinsic model capability through gradient-based optimization, persisting as a brittle external dependency prone to context saturation and information redundancy as the differential diagnosis space expands.

> **机制对比 — 现有外部知识注入方案的局限**:
>
> | 方案 | 代表方法 | 注入方式 | 核心缺陷 |
> |------|---------|---------|---------|
> | RAG | RadioRAG | 在输入 context 前拼接检索到的文本/相似病例 | 静态外部依赖，未经因果验证，随鉴别诊断空间扩大而上下文饱和 |
> | Soft-Prompt / Prefix-Tuning | LaPA | 在输入序列前拼接可学习向量注入领域线索 | 信息静态且缺乏因果校准，仅为"脆弱的"外部补丁而非内生能力 |

Recent latent computation paradigms [28, 54, 97, 124] offer a principled alternative by performing reasoning in continuous hidden state spaces, circumventing the expressiveness bottleneck of discrete symbols. However, their direct application to medical scenarios encounters two domain-specific obstacles. First, without structured anatomical priors, latent representations degenerate into abstract vectors decoupled from clinical semantics: they capture statistical regularities of the training distribution but fail to encode the structured spatial relationships (organ topology, lesion morphology, tissue boundaries) essential for diagnostic grounding. Second, without causal calibration, the coupling between latent representations and diagnostically critical visual features remains weak, as the model may produce correct answers by exploiting spurious correlations (e.g., dataset-specific formatting cues) rather than attending to pathologically relevant regions, undermining reliability in clinical deployment.

> **关键判断 — 通用潜空间推理直接移植到医学场景的两大障碍**:
>
> | 障碍 | 本质 | 后果 |
> |------|------|------|
> | 缺乏结构化先验 | 隐空间表征退化为与临床语义脱节的抽象向量 | 仅编码训练分布统计规律，无法表达器官拓扑、病灶形态、组织边界等诊断所需的空间关系 |
> | 缺乏因果校准 | 隐表征与诊断关键视觉特征耦合弱 | 模型通过 spurious correlation (如数据集格式线索) 而非病理相关区域做出判断，临床部署不可靠 |

These observations converge on a fundamental question: Can a VLM progressively evolve its latent memory to simulate clinical intuition, enabling the rapid synthesis of case-adaptive diagnostic patterns, while ensuring this autonomous internal reasoning stream and its continuous refinement effectively steer the model toward clinically reliable decisions?

> **核心研究问题**: VLM 能否通过渐进式演化其隐空间记忆来模拟临床直觉——即快速合成病例自适应的诊断模式，同时确保这种自主内部推理流及其持续精炼有效引导模型做出临床可靠的决策？

This paper proposes MedSynapse-V, a framework for latent diagnostic memory evolution that addresses this question through three synergistic mechanisms operating in a progressive training paradigm. First, Meta Query for Prior Memorization (§2.2) deploys learnable meta-query probes to retrieve multi-scale spatially aware features from a frozen anatomical encoder pre-trained on large-scale segmentation tasks, condensing them into compact diagnostic implicit memory vectors that are injected into the VLM's hidden stream. This mechanism bridges the representation gap between the encoder's anatomical feature space and the VLM's generation space. Second, Causal Counterfactual Refinement (CCR; §2.3) performs reinforcement learning-driven memory optimization, introducing a novel causal counterfactual reward that quantifies the causal diagnostic contribution of each memory element through region-level feature masking interventions. By contrasting model behavior under original versus intervened memory conditions, CCR systematically prunes causally irrelevant components while reinforcing those with genuine diagnostic utility. Third, Intrinsic Memory Transition (IMT; §2.4) employs a privileged-autonomous dual-branch paradigm to distill the refined diagnostic patterns from a teacher branch (with the anatomical encoder) into a lightweight student branch via full-vocabulary Jensen-Shannon divergence alignment. At inference, the anatomical encoder is entirely removed, and the model generates diagnostic memory autonomously with computational overhead nearly identical to a standard VLM.

> **机制拆解 — MedSynapse-V 三组件**:
>
> | 阶段 | 机制 | 核心操作 | 角色 |
> |------|------|---------|------|
> | Stage I: MQPM | Meta Query for Prior Memorization | 可学习 probe 从冻结解剖编码器检索多尺度特征，压缩为 compact memory | 为 latent space 注入结构化解剖先验，建立语义映射 |
> | Stage II: CCR | Causal Counterfactual Refinement | RL + 因果反事实 reward，通过 region masking 量化 memory 因果贡献 | 剪除因果无关成分，强化有真实诊断效用的 memory |
> | Stage III: IMT | Intrinsic Memory Transition | 特权-自主双分支 JSD 蒸馏，teacher branch(带编码器) -> student branch(仅 VLM) | 将外部依赖转化为内生能力，推理时完全移除编码器 |

Comprehensive evaluations across seven medical multimodal benchmarks demonstrate that MedSynapse-V consistently outperforms a broad spectrum of state-of-the-art approaches, spanning medical-specific VLMs, RL-enhanced CoT paradigms, and general-purpose latent reasoning methods, in both diagnostic accuracy and cross-domain generalization, while introducing negligible additional inference cost compared with standard VLMs. Our main contributions are:

(1) We propose MedSynapse-V, the first framework that evolves diagnostic implicit memory in latent space for medical diagnosis, shifting from static external knowledge injection to progressive, autonomous memory internalization.

(2) We design Meta Query-based Prior Memorization coupled with Causal Counterfactual Refinement (CCR), which distills anatomical priors into compact latent memory and calibrates it via counterfactual interventions to retain only causally grounded diagnostic components.

(3) We introduce Intrinsic Memory Transition (IMT), a privileged-autonomous dual-branch distillation paradigm that internalizes encoder-dependent memory into autonomously generated intrinsic memory via full-vocabulary divergence alignment, eliminating all auxiliary modules at inference.

(4) In multimodal benchmarks, MedSynapse-V consistently surpasses mainstream CoT paradigms in diagnostic accuracy while maintaining inference efficiency on par with standard VLMs, validating latent memory evolution as a principled alternative to discrete token reasoning.

> **贡献总结**: (1) 首次提出 medical VLM 的 latent diagnostic memory evolution 框架——从静态外部知识注入到渐进式自主 memory 内化的范式转变；(2) Meta Query + CCR 组合——将解剖先验蒸馏为紧凑 latent memory 并通过因果反事实校准；(3) IMT——特权-自主双分支蒸馏，推理时完全移除辅助模块；(4) 在 multi-benchmark 上一致超越 CoT，验证 latent memory evolution 是离散 token 推理的有原则替代方案。

---

## 三、Summary

- **问题定义**: 医学 VLM 由于离散 tokenization 产生"认知错位"——连续病理特征被粗粒度近似、视觉证据在推理中消散、离散符号缺乏动态解剖上下文。
- **现有方案不足**: RAG/prefix-tuning 注入静态未经因果验证的外部知识；通用潜空间推理缺乏解剖先验和因果校准。
- **核心研究问题**: VLM 能否通过渐进式 latent memory evolution 模拟临床直觉？
- **方案**: MedSynapse-V = MQPM (解剖先验压缩) + CCR (因果反事实精炼) + IMT (双分支蒸馏内化)
- **贡献**: 范式转变（外部注入 -> 内生演化）；因果校准的 memory 精炼；encoder-free 高效推理；全面的实验验证。
