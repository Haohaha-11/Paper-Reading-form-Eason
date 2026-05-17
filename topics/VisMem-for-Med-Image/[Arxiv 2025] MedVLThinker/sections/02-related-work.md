[← 返回 README](../README.md)

# 02 - Related Work

> 📌 **Section Preview**: 相关工作涵盖三个子方向：(2.1) Large Reasoning Models及医学适配——LRMs的核心范式（think-then-answer）、训练方法（RLVR/GRPO vs CoT蒸馏）、以及在文本医学QA中的应用；(2.2) 多模态医学大语言模型——从Med-Flamingo到LLaVA-Med、PMC-VQA、HuatuoGPT-Vision的发展脉络；(2.3) 同期工作——HuatuoGPT-o1、MedS3、M1、MedVLM-R1、Med-R1、GMAI-VL-R1等最新医学推理模型。

---

## 2.1. Large Reasoning Models and Medical Adaptation

Large Reasoning Models (LRMs) endow large language models with the ability to articulate step-by-step reasoning before finalizing an answer Wei et al. (2022); Guo et al. (2025); Team et al. (2025); Jaech et al. (2024). This test-time "think then answer" approach allows extended reasoning and has yielded impressive gains in domains such as mathematical problem Zeng et al. (2025); Yang et al. (2025); Muennighoff et al. (2025) solving and code generation Jaech et al. (2024); Yang et al. (2024). One way to train LRMs is via Reinforcement Learning with Verifiable Rewards (RLVR), which forgoes supervised chain-of-thought data and instead uses binary feedback on answer correctness Chen et al. (2025); Yu et al. (2025). RLVR eliminates the need to curate lengthy reasoning exemplars; it directly incentivizes correct reasoning by rewarding only the final outcome. In practice, an efficient implementation of RLVR is crucial. Group Relative Policy Optimization (GRPO) Shao et al. (2024) has been adopted for its efficiency, removing the need for a separate value network (critic) during RL updates. An alternative approach is to distill the reasoning traces of stronger models via supervised fine-tuning (SFT). For example, one can use a GPT-4 level model to generate high-quality explanations (CoTs) for medical questions, and then fine-tune a smaller model on this data Chen et al. (2024a). Recent work shows that fine-tuning medical-focused LRMs (either via SFT on expert traces or via RL on answer rewards) can significantly improve medical question answering performance Wu et al. (2025); Huang et al. (2025); Jiang et al. (2025). Our work extends these ideas to the multimodal realm, examining whether similar reasoning enhancements hold when visual information is involved.

> 💡 **机制拆解**: 这一段建立了LRM训练的两条技术路线对比：
> - **RLVR + GRPO**: 无需CoT数据，只用binary answer correctness reward，通过组内相对比较（GRPO）更新策略。优势：不需要精心策划reasoning exemplars，数据成本更低。
> - **CoT Distillation + SFT**: 使用强teacher（如GPT-4）生成高质量解释链，通过SFT蒸馏到小模型。优势：提供dense supervision signal，学习速度快。
>
> 关键创新点：本文是首次将这两种路线的系统比较扩展到多模态医学场景。

---

## 2.2. Multimodal Medical Large Language Models

Given that clinical data often includes images (radiology Lau et al. (2018), pathology Ikezogwo et al. (2023), etc.), there is growing interest in extending LLMs to handle visual inputs for medical applications. Med-Flamingo Moor et al. (2023) was among the first to propose an interleaved vision-language training pipeline for a medical LLM, enabling it to handle image-text pairs in a single prompt. LLaVA-Med Li et al. (2023) introduced a two-stage approach: first, connect a vision encoder with an LLM via a learned projection (connector) and fine-tune on general images; second, fine-tune the combined model on medical image-text instruction data to specialize it. PMC-VQA Zhang et al. (2023) is one such large-scale multimodal instruction dataset, constructed from PubMed Central articles (figures and captions) using GPT-3.5 as an annotator. However, the quality of GPT-3.5-generated questions and answers in PMC-VQA is limited by the base model's capacity, and the dataset likely contains noise or insufficiently detailed questions. Other contemporaneous efforts include HuatuoGPT-Vision Chen et al. (2024b), which scales up LLaVA-Med's pipeline by generating a much larger set of QA pairs from a medical corpus and training larger models (up to 34B parameters). There are also modality-specific medical VLMs such as RadFM Wu et al. (2023) and SkinGPT Zhou et al. (2024) that follow similar pipelines but focus on particular domains (e.g., radiology, dermatology) with domain-specific image-text data. In summary, several open-source medical LMMs have been proposed, but integrating an explicit reasoning mechanism (as in LRMs) into these models has not been thoroughly studied prior to our work.

> 💡 **问题动机**: 这一段梳理了医学多模态LLM的发展脉络：
> - **先驱工作**: Med-Flamingo（交错图文训练），LLaVA-Med（两阶段：连接器+医学指令微调）
> - **数据资源**: PMC-VQA（GPT-3.5生成，大规模但有噪声），HuatuoGPT-Vision（更大规模医学语料）
> - **模态专用模型**: RadFM（放射学），SkinGPT（皮肤病学）
>
> 核心gap: 所有这些工作都是"感知+理解"的范式，未整合显式推理机制。本文正是要填补这一空白——在多模态医学模型中引入LRM的推理能力。

> 💡 **Q&A 批注记录**: **Q: PMC-VQA为什么被认为是低质量数据？这对本文的发现有什么影响？** A: PMC-VQA由GPT-3.5从PubMed Central的图表和图注自动生成，受限于GPT-3.5的生成能力，问题和答案可能过于简单、存在错误或不需深度推理。这直接解释了本文的核心发现——text-only数据（人工编写的医学考试题）优于image-text数据（自动生成的PMC-VQA）。但是，这是数据质量问题还是多模态训练本身的局限，论文没有完全区分清楚。

---

## 2.3. Concurrent Works

Very recently, a few works have begun exploring the idea of eliciting medical reasoning in LLMs. For text-only medical QA, HuatuoGPT-o1 uses a PPO-based RL approach Schulman et al. (2017) with an external reward model to train a medical reasoning LLM Chen et al. (2024a), and MedS3 leverages Process-Reward Models (PRMs) for RL to improve stepwise reasoning Jiang et al. (2025). Another approach, denoted M1 in a recent preprint, distills the reasoning traces of a GPT-4-based model (denoted R1) into a smaller model via SFT Huang et al. (2025). In the multimodal domain, MedVLM-R1 Pan et al. (2025) demonstrates the effectiveness of RLVR on a small scale of multimodal data (fewer than 1K training samples), and Med-R1 Lai et al. (2025) applies a similar RLVR scheme on separate modality-specific datasets. However, these models are trained on limited data and are not generalizable across different types of medical visual questions. GMAI-VL-R1 Su et al. (2025) is a general multimodal medical LLM trained with an RLVR paradigm, but its training data and code are not publicly available. In contrast, our work provides an open-source recipe for building multimodal medical reasoning models with both SFT and RL techniques, and we conduct a thorough experimental study across varying model scales (3B, 7B, 32B) and diverse benchmarks. To our knowledge, this is the first work to systematically compare supervised CoT distillation and RLVR for multimodal medical QA, and to benchmark the resulting models against prior open medical LMMs and closed models like GPT-4.

> 💡 **机制拆解**: 这一节系统比较了同期医学推理工作，形成清晰的差异化定位：

| 工作 | 模态 | 训练方法 | 数据规模 | 开放性 | 局限性 |
|------|------|---------|---------|--------|--------|
| HuatuoGPT-o1 | 文本 | PPO+外部reward model | - | - | 纯文本，需额外reward model |
| MedS3 | 文本 | PRM-based RL | - | - | 需process reward标注 |
| M1 | 文本 | CoT蒸馏(SFT) | - | - | 纯文本 |
| MedVLM-R1 | 多模态 | RLVR | <1K | 开放 | 小规模，不通用 |
| Med-R1 | 多模态 | RLVR | 模态专用 | 开放 | 分领域，不通用 |
| GMAI-VL-R1 | 多模态 | RLVR | 通用 | **闭源** | 不可复现 |
| **MedVLThinker** | **多模态** | **SFT+RLVR** | **大规模通用** | **完全开源** | - |

> 💡 **消融解读**: MedVLThinker区别于所有同期工作的关键优势：
> 1. **唯一同时系统比较SFT和RLVR**的医学多模态工作（其他工作只选其中一种）
> 2. **唯一覆盖3B/7B/32B三种规模的完整scaling研究**
> 3. **完全开源**（包括数据、代码、模型），而GMAI-VL-R1虽然也做通用多模态推理但是闭源的

---

### 🔖 Section 总结

- **2.1 核心**: LRM训练的两条路线——RLVR (binary reward, GRPO) vs CoT蒸馏 (SFT)，本文首次将系统比较扩展到多模态医学
- **2.2 核心**: 医学多模态LLM的发展从Med-Flamingo→LLaVA-Med→PMC-VQA→HuatuoGPT-Vision，但均未整合显式推理机制
- **2.3 核心**: 同期医学推理工作各有局限（纯文本、小规模、领域专用、闭源），MedVLThinker是首个完全开源的大规模通用多模态医学推理系统
- **关键gap**: 无论SFT还是RLVR在多模态医学中的效果差异尚未被系统研究
