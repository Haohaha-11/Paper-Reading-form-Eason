[← 返回 README](../README.md)

# 01 Introduction

## 📌 内容预览

引言部分：病理图像分割的临床重要性 → 深度学习专用模型的局限 → SAM 系列分割基础模型的突破 → 当前病理分割基础模型的三大局限 → 本文 PathSegmentor 的方案和四项核心贡献。

---

## 原文

The segmentation and identification of various structures in pathology images, including tissues, cells, and nuclei, represent a critical foundation for precision medicine [1, 2]. These analyses enable quantitative assessments, such as the calculation of cellular morphology including size, shape, texture, and other imaging features [3]. Recent breakthroughs in deep learning have revolutionized the field of pathology image segmentation, leading to the development of numerous specialized models [4–12] designed for particular tasks and datasets. Although these models often achieve high accuracy, their dependence on predefined segmentation categories limits their effectiveness in real-world clinical settings, which are characterized by diverse and dynamic conditions.

The emergence of segmentation foundation models presents a transformative solution to address these challenges. Segment Anything Model (SAM) [13] pioneered a novel prompt-driven paradigm that has since been expanded in subsequent segmentation foundation models [14–16]. These models enable flexible, interactive segmentation using various prompt types, including spatial prompts (e.g., points, boxes, scribbles), text prompts, reference prompts, or combinations thereof, significantly enhancing their adaptability across diverse applications. This versatility has generated substantial interest in medical imaging analysis. Several research efforts have explored the medical adaptation of segmentation foundation models [17, 18]. In multimodal image segmentation, models such as MedSAM [19], SAM-Med2D [20], and BiomedParse [21] have demonstrated effectiveness through distinct prompting strategies, including point/box prompts and text prompts. For specialized imaging modalities, researchers have developed dedicated solutions including muSAM [22] for microscopy analysis, SegVol [23] and SAT [24] for 3D radiology imaging, and SurgicalSAM [25] for endoscopic applications. Unlike traditional specialized segmentation models, foundation models not only handle a broader range of segmentation tasks, removing the need to train individual models for each application, but also exhibit enhanced generalization capabilities, enabling them to adapt effectively to diverse data distributions in real-world clinical settings.

However, current foundation models for semantic segmentation on pathology images face three significant limitations.

First, segmentation foundation models designed for multimodal medical images [19–21] often deliver inferior performance on pathology images because they are trained on a wide range of diverse multimodal medical data, highlighting the urgent need for a domain-specific segmentation foundation model tailored for pathology.

Second, while a number of foundation models [26–30] have emerged in pathology for various downstream tasks, foundation models for segmentation tasks still remain markedly underexplored. SAM-Path [10] adapts SAM [13] for pathology segmentation by incorporating trainable class prompts, but transforms it into a specialized model for each dataset. SegAnyPath [31] is a spatial-prompted foundation model capable of segmenting pathology images across varying resolutions and stain variations. However, spatial-prompted models lack the ability to predict semantic categories and require labor-intensive localization annotations, which become impractical for pathology images with numerous objects. Thus, it is crucial to develop a foundation model that utilizes text prompts to incorporate semantic information, enabling flexible and efficient segmentation.

Moreover, existing frameworks fail to address the inherent semantic ambiguity in pathology image segmentation, where target objects can be interpreted at multiple scales. For example, the term tumor may refer to either tissue-level regions or individual cells. To overcome this challenge, it is essential to both construct datasets and develop models that explicitly handle this pathological characteristic by capturing hierarchical semantic information.

In this work, we introduce PathSegmentor, the first text-prompted segmentation foundation model for pathology images. PathSegmentor efficiently takes pathology images as visual inputs and utilizes hierarchical semantic categories as textual prompts to enable flexible segmentation of diverse objects across various anatomical regions and histological structures. Specifically, our main contributions are:

1) For dataset construction, we curate 275k image-mask-label triples from 21 publicly available pathology image segmentation datasets. Reflecting the inherent complexity of pathology, we reorganize semantic labels hierarchically, yielding 160 categories that encompass 20 anatomical regions, 3 histological structures, and 61 object types. To the best of our knowledge, this dataset, termed PathSeg, stands as the largest and most comprehensive benchmark for semantic segmentation of pathology images.

2) For model development, we present PathSegmentor, a pathology-specific segmentation foundation model that leverages textual prompts to generate semantic masks for a broad spectrum of pathological categories. Built on a Transformer encoder-decoder architecture, PathSegmentor incorporates a joint feature interaction module to effectively model the underlying relationships between pathology images and their corresponding categories. This design enables highly flexible, efficient, and semantically aware pathology image segmentation.

3) For experiment evaluation, we conduct extensive experiments in both internal and external validations, and the results demonstrate that:

- PathSegmentor demonstrates superior or comparable performance against a group of specialized models (nnUNet [9], DeepLabV3+ [32], and SAM-Path [10]) trained individually for each dataset, while supporting a wider range of segmentation categories with a compact model size.

- PathSegmentor achieves significant improvements over spatial-prompted foundation models (MedSAM [19] and SAM-Med2D [20]), particularly in segmenting intricate objects with irregular shapes, small sizes, and high density. By leveraging text prompts, PathSegmentor eliminates the need for extensive manual prompting in clinical applications, while delivering semantically aware segmentation results.

- Compared with the other text-prompted foundation model (BiomedParse [21]), PathSegmentor is trained on large-scale and diverse pathology data with hierarchical semantic labels. This tailored design enables PathSegmentor to offer a more robust solution for pathology image segmentation.

- Capitalizing on PathSegmentor's superior segmentation performance, we establish a bidirectional integration with cancer diagnosis classification models for feature importance estimation and imaging biomarker discovery. This greatly enhances the reliability of AI diagnostic models, assisting pathologists in making more informed clinical decisions.

---

## 🔖 批读摘要

> 💡 **问题动机（逐层递进）**：引言的结构非常清晰，遵循 "为何病理分割重要 → 专用模型的局限性 → 基础模型的新范式 → 为什么现有基础模型在病理上效果不好 → 我们怎么解决" 的叙事逻辑。三个局限层层递进：(1) 多模态医学基础模型在病理上泛化差（域偏移），(2) 病理分割基础模型本身稀缺且已有空间提示模型缺乏语义能力，(3) 现有框架忽略了病理中目标的多尺度语义歧义（tumor 可以是组织也可以是细胞）。

> 💡 **机制拆解（提示范式的演进）**：
> - 第一代：SAM 的点/框空间提示 → 交互式分割，但需要人工提供提示，缺乏语义识别
> - 第二代：MedSAM, SAM-Med2D → 医学领域适配，但仍然依赖空间提示
> - 第三代：BiomedParse → 引入文本提示，但多模态训练导致在病理特定任务上泛化差
> - 本文 PathSegmentor → **病理专域 + 文本提示 + 层级语义标签** = 完整的语义分割基础模型
>
> 这个演进路径的价值在于：文本提示天然携带语义信息（"这是什么"），而空间提示只提供位置信息（"在哪里"）。在病理临床场景中，医生更关心的是 "分割出肿瘤区域" 而非 "在坐标 (x,y) 处画个框"。

> 💡 **贡献拆解**：四项贡献构成完整的 "数据-模型-实验-应用" 闭环：
> 1. **PathSeg 数据集**：不只是大，更重要的是层级标签体系。这是病理多尺度特征的直接编码。
> 2. **PathSegmentor 模型**：核心创新在 joint feature interaction module（参见 03-methodology 的详细拆解）。
> 3. **实验评估**：三个对照组（专用模型 vs. 空间提示 vs. 文本提示）+ 外部验证 + 复杂对象分析，覆盖面极为全面。
> 4. **可解释诊断**：这是差异化亮点 -- 其他分割模型止步于分割性能本身，PathSegmentor 进一步将分割能力用于提升诊断模型的可解释性。

> 💡 **Q&A 批注记录**：
> - **Q: 为什么 SegAnyPath（空间提示病理基础模型）不能直接用，而要重新做文本提示？** A: SegAnyPath 支持多种分辨率、染色变异，但仍依赖空间提示（点/框），这意味着：(1) 用户需要手动为每个实例定位，在密集场景（如一张病理图有数百个细胞核）中不实用，(2) 模型输出没有语义标签，不知道分割的是什么。PathSegmentor 的文本提示同时携带 "定位 + 语义"信息，一步完成。
> - **Q: SAM-Path 也是病理分割，为什么归为 "专用模型" 而非 "基础模型"？** A: SAM-Path 使用 SAM 编码器 + 可训练类别提示，但关键问题是它为每个数据集单独训练（类提示是不可迁移的），因此本质上是 "SAM 特征提取器 + 特定数据集分类器" 的专用模型范式，而非一个统一的基础模型。PathSegmentor 的单一模型可处理全部 160 个类别，这才是基础模型的本质特征。
