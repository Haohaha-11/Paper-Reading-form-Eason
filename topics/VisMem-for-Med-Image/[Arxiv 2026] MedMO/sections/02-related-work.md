[← 返回 README](../README.md)

---

## 📌 Preview

Two research threads are reviewed: (1) the evolution of medical MLLMs from early efforts (LLaVA-Med, Med-Flamingo) to SOTA systems (Lingshu, Fleming-VL), and (2) grounding capabilities that extend VLMs beyond captioning/VQA to explicit spatial localization via bounding boxes or point coordinates.

---

## 2. Related Works

### 2.1. Medical Language Multi-model Models

The rapid progress of LLMs has catalyzed remarkable advances in medical images. Building upon the success of general-domain LLMs, researchers have developed domainadapted medical MLLMs that integrate visual and textual reasoning for clinical understanding [4, 84]. Early efforts such as LLaVA-Med [45], Med-Flamingo [58], Qilin-MedVL [50], and BioMedGPT [107] established the first medical vision--language models by aligning specialized visual encoders with pre-trained LLMs via linear projection layers, enabling foundational multimodal reasoning. However, these early systems were constrained by limited data diversity and suboptimal modality alignment, leading to hallucinations and factual inconsistencies [14, 45]. Subsequent studies expanded this paradigm through richer datasets [30, 33, 46], improved training strategies [60, 91], efficient finetuning [47], and reinforcement learning [40, 68]. Proprietary systems such as Med-Gemini[20] and Med-PaLM [80, 81] have further integrated multimodal and structured data for advanced reasoning, achieving strong performance across diagnostic and question-answering tasks [5, 6, 73, 95, 101]. Concurrently, specialized MLLMs targeting specific clinical contexts such as pathology [53, 76, 92, 111], radiology [19, 32, 66, 79, 83, 106], and ophthalmology [25] have emerged, highlighting the growing demand for fine-grained, modality-aware intelligence in medical. Recent SOTA frameworks, such as Lingshu [97] and Fleming-vl [78], have improved the integration of medical and natural VLM tasks. However, their capabilities remain limited to selective tasks. Building on these foundations, our work emphasizes largescale open-source post-training and progressive multimodal alignment. MedMO adopts a multi-stage design leveraging over 26M diverse multimodal samples, unifying heterogeneous medical modalities and textual data to achieve substantial gains across diverse clinical tasks.

### 2.2. Grounding using multi-model models

Unlike detection objective-based approaches such as grounding-DINO [51], recent flagship VLMs have moved beyond captioning/VQA to explicit visual grounding [18, 21, 51] as well as point grounding [24], i.e., returning spatial evidence such as bounding boxes or points aligned to textual queries. The Qwen2.5-VL [8] report highlights grounding as a built-in capability, emphasizing precise object localization and event localization in long videos through native dynamicresolution processing and absolute time encoding. Qwen2.5-VL generates grounded outputs in JSON with absolute coordinates, supporting both boxes and point clicks [24]. Although the technical report is general-domain, these grounding primitives transfer to clinical data. For instance, MedSG-Bench [105] evaluates sequential medical grounding (difference/consistency grounding across image series) and explicitly benchmarks Qwen2.5-VL alongside medical-domain MLLMs (e.g., HuatuoGPT-Vision [15]), finding that even advanced VLMs still face challenges on fine-grained, clinically realistic localization tasks-underscoring the need for domain-aligned post-training.

---

## 🔖 Summary

The related work survey is concise but strategically positioned. Sec 2.1 traces the lineage from early medical MLLMs to SOTA, highlighting a persistent gap: even the best prior systems (Lingshu, Fleming-VL) are "limited to selective tasks." Sec 2.2 establishes grounding as a distinct capability axis, noting that while general-domain VLMs like Qwen2.5-VL have built-in grounding, they fail on clinical fine-grained tasks, motivating MedMO's medical-specific grounding post-training.

> 💡 **机制拆解 - 医学MLLM演化路径**: 这段related work隐含了一条清晰的技术演进路径：（1）线性投影对齐（LLaVA-Med）→（2）更丰富的数据+更好的训练策略（HuatuoGPT, GMAI-VL）→（3）RL注入（Med-R1, MedVLM-R1）→（4）任务统一化（Lingshu, Fleming-VL）。MedMO声称自己处于第（4+）阶段：不仅统一VQA/QA/Report，还增加了Grounding维度，且全开源。

> 💡 **Grounding 批读 - 为什么是"game changer"**: 现在的医学MLLM标准评估是VQA accuracy和文本生成指标，但这些指标无法验证模型是否真正"看到"了正确的区域。Grounding通过bounding box提供了可验证的空间证据，本质上将医学AI从"黑盒回答"升级为"可解释定位"。MedMO的设计在 Stage 2 引入grounding capability，在 Stage 4 用RL精调，形成了 "先学会看 → 再看准" 的课程设计。

> 💡 **Q&A 批注记录**: Qwen2.5-VL已具备原生grounding能力（JSON格式输出boxes），但MedMO仍需要专门的医学grounding训练。这说明通用grounding → 医学grounding之间存在显著的domain gap，可能原因是（1）医学图像中异常区域边界模糊（非自然图像中的清晰object boundary），（2）医学术语到空间区域的映射需要专业知识，（3）多视角/序列图像中的grounding需要时序一致性理解。

[← 返回 README](../README.md)
