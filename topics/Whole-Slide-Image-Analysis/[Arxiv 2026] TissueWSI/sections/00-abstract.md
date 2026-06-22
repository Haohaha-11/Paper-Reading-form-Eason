[← 返回 README](../README.md)

# Abstract

## 📌 Preview

Computational pathology has advanced rapidly, yet the core problem of pathology VQA remains unsolved: gigapixel WSIs contain far more information than necessary for any given question. Pathologists naturally navigate tissue complexity by scanning broadly and zooming in selectively. The authors propose **HistoSelect**, a question-guided, tissue-aware, coarse-to-fine retrieval framework that mimics this human diagnostic workflow, reducing visual token usage by 70% while improving accuracy.

---

Computational pathology has advanced rapidly in recent years, driven by domain-specific image encoders and growing interest in using vision-language models to answer natural-language questions about diseases. Yet, the core problem behind pathology question-answering remains unsolved, considering that a gigapixel slide contains far more information than necessary for a given question. Pathologists naturally navigate tissue and morphology complexity by scanning broadly, and zooming in selectively according to the clinical questions. Current models, in contrast, rely on uniform patch sampling or broad attention maps, often attending equally to irrelevant regions while overlooking key visual evidence. In this work, we try to bring models closer to how humans actually examine slides. We propose a question-guided, tissue-aware, and coarse-to-fine retrieval framework, HistoSelect, that consists of two key components: a group sampler that identifies question-relevant tissue regions, followed by a patch selector that retrieves the most informative patches within those regions. By selecting only the most informative patches, our method becomes significantly more efficient: reducing visual token usage by 70% on average, while improving accuracy across three pathology QA tasks. Evaluated on 356,000 question-answer pairs, our approach outperforms existing methods and produces answers grounded in interpretable, pathologist-consistent regions. Our results suggest that bringing human-like search and attention patterns into WSI reasoning is a promising direction for building practical and reliable pathology VLMs. Code is available at https://github.com/winston52/HistoSelect.

> 💡 **问题动机**: 现有的病理 VQA 模型采用均匀采样或宽注意力图，对不相关区域给予了同等的关注，同时忽略了关键的视觉证据。这源于 WSI 的本质特性——一张千兆像素的切片包含远多于某个特定问题所需的信息。作者的目标是让模型像病理学家一样思考：先大范围扫描定位，再有选择地放大关键区域。

> 💡 **机制拆解**: HistoSelect 的两个核心组件：
> 1. **Group Sampler**（组采样器）：在粗粒度层面识别与问题相关的组织区域（如肿瘤区、基质区），决定每个组织的采样率
> 2. **Patch Selector**（补丁选择器）：在细粒度层面，在已识别的相关区域内检索最具信息量的补丁
>
> 这种"由粗到细"的检索框架直接模拟了病理学家的诊断工作流。

> 💡 **Q&A 批注记录**:
>
> **Q**: 为什么减少 70% 的 token 反而能提升准确率？
>
> **A**: 核心逻辑是 **信噪比**（signal-to-noise ratio）。WSI 中大量补丁是背景、良性结构或问题不相关的区域，它们对回答特定问题是噪声。均匀采样将这些噪声与信号同等对待，稀释了有效信息。HistoSelect 用问题引导的选择机制过滤掉噪声，使得输入 LLM 的 token 中信息密度大幅提升。这与 Information Bottleneck 理论一致：压缩掉与任务无关的输入信息，保留与输出最相关的信息。

## 🔖 Summary

This paper proposes HistoSelect, a hierarchical, question-guided patch selection framework for pathology VQA. It consists of a group sampler (coarse tissue-level) and a patch selector (fine-grained), together achieving 70% token reduction and improved accuracy on 356K QA pairs across three benchmarks. The key insight is that mimicking the pathologist's coarse-to-fine diagnostic workflow — first identifying relevant tissue regions, then zooming into critical evidence — is more effective than uniform patch sampling.
