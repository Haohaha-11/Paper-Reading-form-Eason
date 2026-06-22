[← 返回 README](../README.md)

# 2. Related Work

## 📌 Preview

Three strands of related work: (1) WSI analysis from MIL-based classification to slide-level VQA, (2) multimodal histopathology foundation models and agent-based reasoning frameworks, and (3) Information Bottleneck applications in computational pathology. The key positioning is that IB has been used for classification and survival analysis, but never for pathology VQA -- a gap this work fills.

---

**Whole Slide Image Analysis.** Traditional WSI analysis primarily focuses on slide-level classification [17, 19, 31, 33, 39, 48, 50] and survival analysis [8, 35, 47, 51, 52] using Multiple Instance Learning (MIL) [17, 19, 23, 30, 48, 49, 53]. More recently, the field has advanced to Pathology Visual Question Answering (VQA) [9, 22, 32], which is a more challenging task. Unlike the aggregation-focused objective of MIL, VQA demands fine-grained reasoning to answer queries ranging from global morphology to the identification of cellular features. Pathology VQA benchmarks include patch-level datasets, such as Quilt-LLaVA [32], and slide-level datasets like SlideChat [9] and WSI-LLaVA [22]. In this work, we focus on developing a hierarchical selection method for the slide-level VQA task.

> 💡 **问题动机**: 传统的 MIL 方法的核心是聚合（aggregation）——将所有补丁的信息聚合为一个 slide-level 表示用于分类。而 VQA 需要细粒度推理，从全局形态学到细胞特征识别。这个区别是关键：MIL 的聚合范式天然不适合 VQA，因为它丢失了补丁级别的细粒度信息。

**Multi-Modal Histopathology Models.** Recent advances in vision-language foundation models, such as CONCH [26], PLIP [15], MUSK [42], Gecko [18] and CPath-CLIP [36], have demonstrated significant efficacy in bridging visual morphology with clinical language for WSI analysis. Building upon these foundation models, Pathology VQA has emerged as a key task, with the most recent frameworks employing MLLMs to achieve complex reasoning. Initial efforts primarily address localized analysis at the patch or region level, as seen in models such as LLaVA-Med [20], Quilt-LLaVA [32], and PathChat [27]. More recently, the focus has shifted toward slide-level diagnostics, where frameworks such as SlideChat [9] and WSI-LLaVA [22] attempt to handle comprehensive queries by aggregating massive visual features from gigapixel images. To enhance reasoning logic, agent-based frameworks such as PathFinder [12], WSI-Agents [28] and CpathAgent [37] have been proposed to emulate a pathologist's workflow via iterative reasoning. While their dynamic navigation ensures structured evidence gathering, this sequential process may incur significant inference latency. Unlike exhaustive aggregation or iterative agents, we focus on reducing token redundancy in slide-level VQA by distilling a question-aligned subset of patches for efficient diagnostic reasoning.

> 💡 **机制拆解 - 与 Agent-based 方法的区别**: Agent-based 方法（PathFinder, WSI-Agents, CpathAgent）通过迭代替换模拟病理学家的推理流程，每次迭代只关注一个区域。虽然这种动态导航确保了结构化的证据收集，但串行过程会带来显著的推理延迟。HistoSelect 走了一条不同的路：它通过单次前向传播完成层级选择，不需要迭代替换，因此在保持推理效率的同时实现了对病理学家工作流的模拟。

> 💡 **Q&A 批注记录**:
>
> **Q**: HistoSelect 和 SlideChat/WSI-LLaVA 的核心区别是什么？
>
> **A**: SlideChat 和 WSI-LLaVA 都采用了问题无关的策略——SlideChat 使用非选择性采样，WSI-LLaVA 使用池化。它们将所有补丁 token 同等对待。HistoSelect 的独特之处在于**问题引导的选择**：在补丁进入 LLM 之前，先用问题来筛选最相关的子集。这不是一种替代架构，而是一种可以嵌入到现有 pipeline 中的选择模块。

**Information Bottleneck in Computational Pathology.** The Information Bottleneck (IB) principle [41] is an information-theoretic framework for learning, positing that an optimal model should learn a "bottleneck" representation that is maximally compressive of the input while retaining the maximum possible information about the downstream task [2]. Due to its inherent ability to mitigate redundancy, the IB principle has been increasingly adopted in computational pathology to address domain-specific challenges [11, 21, 34, 51]. For example, [21] proposed a variational IB-based fine-tuning strategy to learn task-specific features for WSI classification. Concurrently, [51] employed prototypical IB and information disentanglement to tackle the massive redundancy issues present in multimodal cancer survival prediction. Despite its demonstrated potential in classification and survival analysis, to the best of our knowledge, the IB framework has not yet been explored for pathology VQA. This represents a significant gap, as the hierarchical and token-intensive nature of LLM-based VQA models, which must process a massive number of visual tokens from WSIs, presents a critical challenge of information redundancy and computational inefficiency that the IB principle is ideally suited to address.

> 💡 **问题动机 - IB 与 VQA 的天然契合**: 这一段非常关键地阐述了为什么 IB 框架在 pathology VQA 中尚未被探索是一个"显著的空白"。LLM-based VQA 模型的层级特性和 token 密集特性（必须处理来自 WSI 的大量视觉 token）产生了一个信息冗余和计算效率的核心挑战，而 IB 原则天然适合解决这个问题。之前的 IB 工作（分类、生存预测）只需处理单一的 slide-level 表示，而 VQA 需要同时考虑问题-图像交互和层级选择，这使得 IB 的应用更具挑战性也更有价值。

## 🔖 Summary

The related work review positions HistoSelect at the intersection of three areas: (1) moving beyond MIL aggregation toward slide-level VQA with fine-grained reasoning, (2) offering a different approach from agent-based methods (single-pass selection vs. iterative reasoning) while sharing the goal of emulating pathologist workflows, and (3) pioneering the application of IB theory in pathology VQA, where its redundancy-mitigating properties are ideally suited for the token-intensive nature of LLM-based reasoning over gigapixel images.
