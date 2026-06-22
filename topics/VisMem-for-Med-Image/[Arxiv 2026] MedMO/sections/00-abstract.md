[← 返回 README](../README.md)

---

## 📌 Preview

MedMO is a post-trained medical multimodal foundation model designed for comprehensive medical image understanding and grounding. It follows a multi-stage training recipe: cross-modal pretraining, instruction tuning, and RL with verifiable rewards (including a box-level GIoU reward). It consistently outperforms open-source medical MLLMs across VQA, QA, report generation, and spatial grounding tasks.

---

## Abstract

Multimodal large language models (MLLMs) have rapidly advanced, yet their adoption in medicine remains limited by gaps in domain coverage, modality alignment, and grounded reasoning. In this work, we introduce MedMO, a medical foundation model built upon a generalized MLLM architecture and trained exclusively on large-scale, domainspecific data. MedMO follows a multi-stage training recipe: (i) cross-modal pretraining to align heterogeneous visual encoders with a medical language backbone; (ii) instruction tuning on multi-task supervision that spans captioning, VQA, report generation, retrieval, and grounded disease localization with bounding boxes; and (iii) reinforcement learning with verifiable rewards that combine factuality checks with a box-level GIoU reward to strengthen spatial grounding and step-by-step reasoning in complex clinical scenarios. MedMO consistently outperforms strong open-source medical MLLMs across multiple modalities and tasks. MedMO-8B-Next leads all comparisons: on VQA benchmarks, it achieves an average improvement of +6.6% over Fleming-VL-8B, with gains of +6.0% on MMMU-Med, +9.8% on PMC-VQA, and +21.3% on MedXpertQA. For text-based QA, it attains +14.4% over Fleming-VL-8B, driven by +8.4% on MMLU-Med and +30.1% on MedQA. In medical report generation, MedMO-8B-Next delivers +6.7% on MIMIC-CXR. Moreover, it exhibits strong grounding capability with a Bacteria IoU of 56.1, representing a +47.8 IoU gain over Fleming-VL-8B, underscoring its robust spatial reasoning and localization performance. MedMO-4B-Next remains highly competitive at its smaller scale, surpassing Fleming-VL-8B across VQA, QA, and report generation benchmarks. Evaluations across radiology, ophthalmology, and pathology microscopy confirm MedMO's broad cross-modality generalization.

---

## 🔖 Summary

MedMO resolves three core challenges in medical MLLMs: narrow domain coverage, poor modality alignment, and lack of grounded reasoning. Its three-stage recipe -- cross-modal alignment, multi-task instruction tuning, and RL with box-level verifiable rewards -- delivers consistent gains across VQA (+6.6%), text QA (+14.4%), and report generation. Notably, grounding (Bacteria IoU 56.1 vs 8.3) shows that existing medical VLMs severely lack spatial localization, which MedMO directly addresses.

> 💡 **问题动机**: The abstract frames the problem as a trilemma -- existing medical MLLMs fail simultaneously on domain coverage (only handle narrow subsets like radiology OR pathology), modality alignment (visual features poorly grounded to clinical text), and reasoning (factual hallucination). MedMO's multi-stage recipe is designed to address all three through progressive curriculum learning, where each stage targets a specific gap.

> 💡 **机制拆解**: The three stages map to three capability axes:
> (i) cross-modal pretraining → alignment (all modalities share embedding space),
> (ii) instruction tuning → task generalization (captioning/VQA/report/retrieval/grounding in one model),
> (iii) RL + box-level GIoU → factuality + spatial precision.
> The "GIoU reward" is notable because it treats spatial grounding as a verifiable signal (unlike text-only factuality), which is rare in medical RL training.

> 💡 **关键数字解读**: 
> - +6.6% VQA平均提升和+14.4% QA平均提升的差距值得注意：MedMO在纯文本QA上的相对提升远大于VQA，说明医学知识融入（Stage 3的instruction tuning）比视觉理解本身（Stage 2的高分辨率训练）带来的边际提升更大。
> - Bacteria IoU从8.3到56.1的(+47.8)跳跃说明Fleming-VL等baseline基本上不具备grounding能力，MedMO是首次真正"解锁"了这一维度。

[← 返回 README](../README.md)
