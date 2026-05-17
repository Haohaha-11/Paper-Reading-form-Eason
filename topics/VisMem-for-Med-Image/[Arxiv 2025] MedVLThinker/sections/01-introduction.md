[← 返回 README](../README.md)

# 01 - Introduction

> 📌 **Section Preview**: 引言部分建立了医学多模态推理的研究背景。首先论述医疗实践中多模态数据整合的需求和LMMs在医学中的潜力；随后引出Large Reasoning Models (LRMs) 的"think before responding"范式及其在医学QA中的早期成功；接着指出当前医学多模态推理模型开放性不足的问题；最后概述MedVLThinker的完整方案、训练范式、核心实验结果和主要发现。

---

The practice of healthcare increasingly involves processing vast amounts of multimodal medical data (e.g., text, imaging, lab results). Clinicians must integrate information from different sources (clinical notes, radiology images, lab reports) to make diagnoses and treatment decisions. Large Multimodal Models (LMMs) have recently emerged as general-purpose foundation models that can perceive and reason about visual inputs Li et al. (2023); Liu et al. (2023b, 2024); Hurst et al. (2024); Chen et al. (2024b); Xie et al. (2024b). Given that medical data are natively multimodal (e.g., microscopy slides, CT and MRI scans, X-rays), LMMs have a natural appeal for medical AI and have begun to be adopted in modality-rich clinical settings with the potential to improve diagnosis Chen et al. (2024b); Li et al. (2023); Liu et al. (2023a), treatment planning Zhou et al. (2023), and patient monitoring Alshibli et al. (2025).

> 💡 **问题动机**: 医学数据天然是多模态的（文本病历、影像、检验报告），临床医生需要综合多种信息源进行诊断决策。LMMs作为能够同时感知和推理视觉输入的基础模型，为医学AI提供了自然的技术路径。本文的出发点是：如何让LMM不仅仅是"看到"医学图像，而是能够像临床医生一样"思考-推理-判断"。

---

Parallel to this, Large Reasoning Models (LRMs) extend large language models with a new response paradigm: the model "thinks" through a chain-ofthought before producing a final answer. This allows the model to devote more computation at inference time to reasoning, often improving performance on complex tasks Guo et al. (2025); Guha et al. (2025); Jaech et al. (2024). Early medical adaptations of text-only LRMs have demonstrated strong performance on medical QA tasks Huang et al. (2025); Chen et al. (2024a); Wu et al. (2025); Jiang et al. (2025); Xie et al. (2024a). The ability to generate detailed reasoning steps at test time appears to confer significant gains in accuracy on challenging questions Zuo et al. (2025). However, how to best combine this reasoning paradigm with multimodal understanding remains underexplored. While there have been efforts to build medical multimodal reasoning models, they are often limited in openness—being either entirely closed-source Su et al. (2025); Liu et al. (2025), releasing only model weights without data or training code Sellergren et al. (2025), or, if fully open, are confined to narrow datasets or specific domains (e.g., CT or MRI only) Lai et al. (2025); Pan et al. (2025). As a consequence, the field lacks a comprehensive analysis of how critical factors such as data modality, curation pipelines, and training strategies affect model performance.

> 💡 **机制拆解**: 这一段揭示了两个技术路线的交叉点：(1) LRMs的"先思考再回答"范式——通过test-time compute提升推理准确率；(2) 医学场景的特殊需求——纯文本LRMs已在医学QA中证明有效，但与多模态理解的结合仍是未充分探索的领域。作者指出现有工作的三个开放性问题：闭源（GMAI-VL-R1）、仅开放权重不开放数据/代码（MedGemma）、开放但局限于狭窄领域（MedVLM-R1, Med-R1）。这三个问题共同构成了本文的研究动机——需要一个全面开放的、跨领域的、系统化分析方案。

> 💡 **Q&A 批注记录**: **Q: 早期医学文本LRMs的成功（如M1, HuatuoGPT-o1）是否可以直接迁移到多模态场景？** A: 不可以直接迁移，因为涉及视觉信息的处理。本文的关键问题正是："how to best combine this reasoning paradigm with multimodal understanding"。这表明需要一个系统性的研究来回答数据模态、训练策略等因素的影响。

---

In this paper, we provide MedVLThinker, the very first fully open-source recipe for building and evaluating generalized Medical Vision-Language Reasoning Models. Our comprehensive framework provides a complete workflow, from data curation and training pipelines to a standardized evaluation protocol. This enables, for the first time, a fair and systematic comparison across diverse multimodal medical QA benchmarks. Figure 2 provides an overview of our approach. We first curate two types of training data: a text-only QA dataset and an image-text (multimodal) QA dataset. Using a general-purpose multimodal LLM (Qwen2.5-VL-Instruct) Bai et al. (2025), we probe each question with multiple trials to estimate its difficulty. Specifically, for each question, we generate multiple candidate answers and count how many times the model answers correctly (the "pass count"). Questions that are consistently answered correctly (too easy) or never answered correctly (too hard) are filtered out, yielding a focused training set of medium-difficulty questions. We then employ strong teacher models to generate detailed reasoning chains (long chains-of-thought, CoTs) for the remaining questions. For text-only questions, we use the DeepSeek Guo et al. (2025) model (a powerful text-based LRM) as the CoT teacher, and for imagebased questions, we use GPT-4o Hurst et al. (2024) (a vision-enabled GPT-4 variant).

> 💡 **机制拆解**: 数据pipeline的核心设计——"难度探针"（difficulty probing）。使用Qwen2.5-VL-Instruct对每个问题多次采样（16次），统计pass count，然后筛选中等难度问题（排除pass=0的太难题和pass>=7的太简单题）。这是一个curriculum learning的思路——中等难度的题目对训练最有益。两个teacher模型的选择也经过设计：文本题用DeepSeek（强文本推理模型），图文题用GPT-4o（视觉推理模型）。

---

Using these data, we train the base multimodal LLM under two paradigms: (1) Supervised finetuning (SFT) on the teacher-generated CoT traces, and (2) Reinforcement Learning with Verifiable Rewards (RLVR) on the question-answer pairs (without CoTs). SFT directly teaches the model to reproduce high-quality reasoning and answer traces, whereas RLVR uses only binary rewards from answer correctness to encourage the model's own reasoning. We implement RLVR via Group Relative Policy Optimization (GRPO) Shao et al. (2024), an efficient policy-gradient algorithm that requires no value estimator or critic model. In RLVR training, the model generates multiple reasoning traces for each question; each trace is verified for correct answer format and correctness of the final answer, yielding a +1 or -1 reward. These binary rewards are normalized (whitened) across the batch and fed into the GRPO update step, which applies a PPO-style clipped objective. This process gradually concentrates the model's generation probability mass on verifiably correct reasoning traces while limiting divergence from the original model output distribution.

> 💡 **机制拆解**: SFT vs RLVR的本质区别：
> - **SFT**: 输入=(question, teacher's CoT+answer)，loss=token-level cross-entropy on CoT+answer。本质是让模型模仿teacher的推理过程。
> - **RLVR**: 输入=(question)，模型自由生成N条推理链（N=8），每条链被verifier打分（+1/-1只基于最终答案正确性），reward在组内whitened后送入GRPO更新。本质是让模型自己探索并强化通向正确答案的推理路径。
>
> GRPO的关键优势：不需要critic/value network，通过组内比较（group-based advantage）替代，计算效率更高。KL penalty约束防止模型偏离原始分布太远（避免reward hacking）。

---

We conduct extensive experiments on six multimodal medical QA benchmarks to investigate the properties of our MedVLThinker. We use the Qwen2.5-VL series as the base models (in 3B, 7B, and 32B parameter sizes). Our evaluations cover both general medical visual QA and modality-specific QA (covering specialties like pathology, radiology, etc.). The results reveal several important, and at times counter-intuitive, findings: First, regarding training paradigms, models trained with RLVR consistently outperform those trained with SFT across both 3B and 7B scales. Second, in terms of data modality, text-only training outperforms image-text training. Notably, SFT on distilled text-only CoT data degrades performance relative to the base model (e.g., MedVLThinker-7B accuracy drops from 53.5% to 43.8%), whereas SFT on image-text data yields performance similar to the untrained base model. In contrast, RLVR on text-only data provides the largest performance boost, improving the 7B model from 53.5% to 54.9%. RLVR on image-text data also improves performance, but to a lesser extent. Moreover, combining text-only and image-text data—either through SFT+RL or sequential RL—does not yield additional gains beyond using text-only data alone. Third, model scale has a clear impact: 7B models consistently outperform their 3B counterparts across all configurations.

> 💡 **问题动机**: 这段概述了全文最重要的实验结果，形成了三条核心发现链条：
> 1. **训练范式链**: RLVR >> SFT（反直觉：CoT蒸馏竟然有害）
> 2. **数据模态链**: Text-only > Image-text（反直觉：多模态训练不如纯文本）
> 3. **规模效应链**: 7B > 3B（符合预期，但提升幅度值得关注）
>
> 最反直觉的是SFT on text-only CoT data导致性能大幅下降（7B: 53.5%→43.8%，降9.7%），暗示teacher-student之间的reasoning style mismatch可能比数据量更重要。

---

Among existing open-source 7B medical LMMs, MedVLThinker-7B (trained with RLVR on text-only data) achieves a new state-of-the-art average accuracy of 54.9% across six benchmarks. To evaluate the effect of model scaling, we further train a 32B variant. As shown in Figure 1, MedVLThinker-32B performs competitively with the proprietary GPT-4o, demonstrating the potential of open models to close the performance gap with commercial systems. To accelerate community-driven development and foster future innovation, we will release our complete research toolkit, including all models, code, and pipelines for data curation, training, and evaluation.

> 💡 **消融解读**: 最后一段总结了模型的scaling效果和开源意义。MedVLThinker-32B的63.12% avg与GPT-4o的63.74% avg仅差0.62%，这在医学多模态领域是一个有竞争力的结果。更关键的是这是完全开源的——包括数据、代码、模型、评估pipeline——这使得社区可以在此基础上进行公平比较和持续改进。

---

> 💡 **Q&A 批注记录**:
>
> **Q: SFT on text-only CoT data为什么会大幅降低性能？论文给出了什么解释？**
> A: 引言中未给出详细解释，但提到了关键线索："SFT on distilled text-only CoT data degrades performance" 和 "SFT on image-text data yields performance similar to the untrained base model"。Section 4.3的讨论中推测是因为text-only rationales来自文本LRM（DeepSeek），与需要同时处理图像的多模态模型的推理模式不匹配。这是一个值得深入研究的开放问题。
>
> **Q: "combining text-only and image-text data does not yield additional gains"这一发现的意义是什么？**
> A: 这暗示当前的多模态医学数据（PMC-VQA）质量不足以提供有效的训练信号，甚至可能干扰模型已经学到的推理能力。这与RLVR的成功形成对比——RLVR不依赖CoT质量，只依赖answer verifiability，因此对数据质量更鲁棒。

---

### 🔖 Section 总结

- **研究背景**: LMMs在医学中具有巨大潜力，但如何有效结合推理范式和多模态理解尚未充分探索
- **核心gap**: 缺少完全开源的、跨领域的、系统化分析的医学多模态推理方案
- **MedVLThinker方案**: 数据筛选+双训练范式（SFT/RLVR）+多规模实验（3B/7B/32B）
- **三大反直觉发现**: RLVR>SFT; Text-only>Image-text; 组合训练无额外收益
- **最佳结果**: 7B SOTA (54.9%), 32B匹敌GPT-4o (63.12% vs 63.74%)
