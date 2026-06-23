[← 返回 README](../README.md)

---

## 📌 Preview

MedMO 是一个经过后训练的多模态医学基础模型，面向全面的医学图像理解与空间定位。论文采用多阶段训练方案：跨模态预训练、指令微调，以及带可验证奖励（包括边界框级 GIoU 奖励）的强化学习。其在 VQA、QA、报告生成和空间定位等任务上始终优于开源医学 MLLM。

---

## Abstract

Multimodal large language models (MLLMs) have rapidly advanced, yet their adoption in medicine remains limited by gaps in domain coverage, modality alignment, and grounded reasoning. In this work, we introduce MedMO, a medical foundation model built upon a generalized MLLM architecture and trained exclusively on large-scale, domainspecific data. MedMO follows a multi-stage training recipe: (i) cross-modal pretraining to align heterogeneous visual encoders with a medical language backbone; (ii) instruction tuning on multi-task supervision that spans captioning, VQA, report generation, retrieval, and grounded disease localization with bounding boxes; and (iii) reinforcement learning with verifiable rewards that combine factuality checks with a box-level GIoU reward to strengthen spatial grounding and step-by-step reasoning in complex clinical scenarios. MedMO consistently outperforms strong open-source medical MLLMs across multiple modalities and tasks. MedMO-8B-Next leads all comparisons: on VQA benchmarks, it achieves an average improvement of +6.6% over Fleming-VL-8B, with gains of +6.0% on MMMU-Med, +9.8% on PMC-VQA, and +21.3% on MedXpertQA. For text-based QA, it attains +14.4% over Fleming-VL-8B, driven by +8.4% on MMLU-Med and +30.1% on MedQA. In medical report generation, MedMO-8B-Next delivers +6.7% on MIMIC-CXR. Moreover, it exhibits strong grounding capability with a Bacteria IoU of 56.1, representing a +47.8 IoU gain over Fleming-VL-8B, underscoring its robust spatial reasoning and localization performance. MedMO-4B-Next remains highly competitive at its smaller scale, surpassing Fleming-VL-8B across VQA, QA, and report generation benchmarks. Evaluations across radiology, ophthalmology, and pathology microscopy confirm MedMO's broad cross-modality generalization.

---

## 🔖 Summary

MedMO 解决了医学 MLLM 中三个核心挑战：领域覆盖窄、模态对齐差和缺乏 grounded reasoning。其三阶段方案——跨模态对齐、多任务指令微调、带边界框级可验证奖励的强化学习——在 VQA（+6.6%）、文本 QA（+14.4%）和报告生成上均带来了稳定提升。尤其值得注意的是，grounding 结果（Bacteria IoU 56.1 vs 8.3）表明现有医学 VLM 严重缺乏空间定位能力，而这正是 MedMO 直接解决的问题。

> 💡 **问题动机**: 摘要将问题框架化为一个三重困境——现有医学 MLLM 在以下三个方面同时失败：领域覆盖（仅处理放射学或病理学等狭窄子集）、模态对齐（视觉特征与临床文本之间缺乏 grounded）、以及推理（事实性幻觉）。MedMO 的多阶段方案旨在通过渐进式课程学习同时解决这三个问题，每个阶段针对一个特定的缺口。

> 💡 **机制拆解**: 三个阶段对应三条能力轴：
> (i) 跨模态预训练 → 对齐（所有模态共享嵌入空间），
> (ii) 指令微调 → 任务泛化（在一个模型中完成图像描述/VQA/报告/检索/grounding），
> (iii) RL + 边界框级 GIoU → 事实性 + 空间精度。
> "GIoU 奖励" 值得关注，因为它将空间定位视为可验证信号（不同于仅依赖文本的事实性验证），这在医学 RL 训练中十分罕见。

> 💡 **关键数字解读**: 
> - +6.6% VQA平均提升和+14.4% QA平均提升的差距值得注意：MedMO在纯文本QA上的相对提升远大于VQA，说明医学知识融入（Stage 3的instruction tuning）比视觉理解本身（Stage 2的高分辨率训练）带来的边际提升更大。
> - Bacteria IoU从8.3到56.1的(+47.8)跳跃说明Fleming-VL等baseline基本上不具备grounding能力，MedMO是首次真正"解锁"了这一维度。

[← 返回 README](../README.md)
