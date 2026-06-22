[← 返回 README](../README.md)

# 00 Abstract

## 📌 内容预览

论文摘要，阐述病理图像语义分割的临床需求、现有方法的局限，以及 PathSeg 数据集 + PathSegmentor 文本提示分割基础模型的核心方案。

---

## 原文

Pathology image segmentation plays a pivotal role in computational pathology, which enables quantitative analysis of histological features for cancer diagnosis and prognosis. However, current segmentation methods encounter significant challenges in clinical applications, primarily due to the scarcity of high-quality, large-scale annotated pathology data and the constraints of fixed, narrowly defined object categories. To address these issues, this work aims to develop a segmentation foundation model capable of segmenting anything in pathology images using natural language.

First, we establish PathSeg, the largest and most comprehensive dataset for pathology image semantic segmentation, derived from 21 publicly available datasets and comprising 275k image-mask-label triples. Our PathSeg dataset features a wide variety of 160 segmentation categories organized in a three-level hierarchy that covers 20 anatomical regions, 3 histological structures, and 61 object types.

Next, we introduce PathSegmentor, a text-prompted foundation model tailored for pathology image segmentation. With PathSegmentor, users can achieve semantic segmentation simply by providing a descriptive text prompt for the target category, thus eliminating the need to laboriously provide numerous spatial prompts like boxes or points for each instance.

Extensive experiments on both internal and external datasets demonstrate the superior segmentation performance of PathSegmentor. It outperforms the group of specialized models, effectively handling a broader range of segmentation categories while maintaining a more compact model size. As a segmentation foundation model, PathSegmentor significantly surpasses other spatial-prompted and text-prompted models by 0.145 and 0.429 improvements in overall Dice scores, respectively, showcasing its remarkable robustness in segmenting complex objects and its effective generalization ability on external evaluations.

Furthermore, we demonstrate that PathSegmentor's versatile segmentation capabilities can effectively enhance the explainability of classification models for cancer diagnosis through feature importance estimation and imaging biomarker discovery. These interpretable outputs provide pathologists with evidence-based decision support, ultimately advancing precision oncology in clinical practice.

**Keywords**: Foundation model, Text prompt, Pathology image, Semantic segmentation, Explainable cancer diagnosis

---

## 🔖 批读摘要

> 💡 **问题动机**：病理图像分割面临两个核心痛点 -- (1) 高质量标注数据稀缺，(2) 固定类别限制。传统方法为每个数据集训练一个专用模型，无法泛化到动态、多样的临床场景。SAM 等基础模型的提示范式虽有突破，但空间提示（框、点）缺乏语义信息，在病理图像中需要逐实例标注，临床不可行。因此，这篇工作的核心目标是：用自然语言描述目标类别，实现病理图像的语义分割。

> 💡 **机制拆解（数据集+模型双轮驱动）**：工作分两层递进。数据集层面，汇总 21 个公开病理分割数据集的 275k 样本，并以三层分级体系 [解剖区域]-[组织学结构]-[对象类型] 重组织 160 个语义标签，解决跨数据集的标签歧义问题（同一标签在不同数据集中可能指向不同尺度的实体，如 "tumor" 既可指肿瘤组织也可指肿瘤细胞）。模型层面，PathSegmentor 基于 Transformer 编码器-解码器架构，用文本描述的语义信息替代空间提示来实现分割，一次文本即可完成单张图像的分割。

> 💡 **关键数字**：275k 样本、160 语义类别、20 解剖区域、3 组织学结构、61 对象类型；内部验证 Dice 0.671，比空间提示模型 (MedSAM) 提升 0.145，比文本提示模型 (BiomedParse) 提升 0.429；模型大小 0.45B，比 16 个专用模型组 (1.86B) 减少 75%。

> 💡 **Q&A 批注记录**：
> - **Q: 为什么 "scarcity of annotated data" 是问题，但 PathSeg 又声称是最大数据集？** A: 这里的 scarcity 指的是语义级别的精细标注。对比 SAM-Med2D 的 19.7M 掩码（非语义）和 MedSAM 的 1.5M 掩码（仅框提示），PathSeg 的 275k 语义掩码虽然规模相对小，但包含完整的三级语义标签，是目前病理语义分割最大的数据集。这也是为什么论文将 "scaling up semantic-aware datasets" 列为未来工作方向之一。
> - **Q: 0.145 和 0.429 提升分别在什么基线上？** A: 0.145 是相对 MedSAM (best spatial-prompted) 的整体 Dice 提升；0.429 是相对 BiomedParse (best text-prompted) 的提升。基数差异巨大（0.526 vs. 0.242），说明现有文本提示模型在病理领域表现极弱，而 PathSegmentor 的领域特定设计有效弥补了这一差距。
