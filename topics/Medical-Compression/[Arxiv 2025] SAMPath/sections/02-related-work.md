[← 返回 README](../README.md)

# 02 Related Work

## 📌 内容预览

本文没有独立的 Related Work 章节，相关工作分布在引言和正文中。本文将其整理为四条脉络：病理图像分割的演进、SAM 系列基础模型、医学分割基础模型、可解释 AI 在病理中的应用。

---

## 原文整理

### 2.1 病理图像分割的演进

The segmentation and identification of various structures in pathology images, including tissues, cells, and nuclei, represent a critical foundation for precision medicine [1, 2]. These analyses enable quantitative assessments, such as the calculation of cellular morphology including size, shape, texture, and other imaging features [3]. Recent breakthroughs in deep learning have revolutionized the field of pathology image segmentation, leading to the development of numerous specialized models [4–12] designed for particular tasks and datasets. Although these models often achieve high accuracy, their dependence on predefined segmentation categories limits their effectiveness in real-world clinical settings, which are characterized by diverse and dynamic conditions.

> 代表性专用模型包括：
> - DCAN [4]：深度轮廓感知网络，用于组织学图像实例分割
> - HoVer-Net [8]：多组织细胞核同步分割与分类
> - nnU-Net [9]：自配置医学分割框架，Nature Methods 2021
> - DoNet [5]：深度去重叠网络用于细胞学实例分割
> - CellViT [11, 12]：基于 Vision Transformer 的精确细胞分割与分类
> - SAM-Path [10]：将 SAM 适配到病理分割，但通过可训练类别提示转换为逐数据集专用模型

### 2.2 SAM 系列分割基础模型

The emergence of segmentation foundation models presents a transformative solution to address these challenges. Segment Anything Model (SAM) [13] pioneered a novel prompt-driven paradigm that has since been expanded in subsequent segmentation foundation models [14–16]. These models enable flexible, interactive segmentation using various prompt types, including spatial prompts (e.g., points, boxes, scribbles), text prompts, reference prompts, or combinations thereof, significantly enhancing their adaptability across diverse applications.

> SAM 家族关键节点：
> - **SAM [13]** (ICCV 2023)：原始提示驱动分割范式，10亿+掩码训练，支持点/框提示
> - **SAM-HQ [14]** (NeurIPS 2023)：高质量分割，保留精细边界细节
> - **Semantic-SAM [15]** (ECCV 2024)：任意粒度的分割与识别
> - **SEEM [16]** (NeurIPS 2023)：多提示类型（点、框、文本、涂鸦）统一框架，PathSegmentor 的架构基础

### 2.3 医学分割基础模型

This versatility has generated substantial interest in medical imaging analysis. Several research efforts have explored the medical adaptation of segmentation foundation models [17, 18].

**多模态医学分割**：
- **MedSAM [19]** (Nature Communications 2024)：在 1.5M 医学图像-掩码对上微调 SAM，优化框提示鲁棒性
- **SAM-Med2D [20]** (arXiv 2023)：扩展至 19.7M 医学数据，支持多种提示（点、框、掩码）
- **BiomedParse [21]** (Nature Methods 2025)：文本提示驱动的生物医学分割基础模型，覆盖 9 种模态

**特定模态分割**：
- **muSAM [22]**：显微镜分析
- **SegVol [23]** 和 **SAT [24]**：3D 放射影像
- **SurgicalSAM [25]**：内窥镜应用

**病理分割基础模型**：
- **SAM-Path [10]**：冻结 SAM 编码器 + 冻结病理编码器 + 可训练类别提示 → 每个数据集单独训练，非统一基础模型
- **SegAnyPath [31]**：空间提示（点/框）驱动的多分辨率、染色不变病理分割基础模型，但缺乏语义能力

Unlike traditional specialized segmentation models, foundation models not only handle a broader range of segmentation tasks, removing the need to train individual models for each application, but also exhibit enhanced generalization capabilities, enabling them to adapt effectively to diverse data distributions in real-world clinical settings.

### 2.4 可解释 AI 在病理中的应用

Explainable AI in medical image analysis is essential for enhancing the interpretability and trustworthiness of diagnostic models [36–39]. In pathology, accurate cancer diagnosis relies on the identification, morphological assessment, and quantitative analysis of specific pathological objects, such as characteristic tissue patterns, cell structures, and nuclear features [40, 41].

> **特征重要性估计**：
> - RISE [44]：随机输入采样，用灰色方块遮挡估计特征重要性
> - LIME [45]：超像素级别的遮挡解释
> - 本文方法：用病理学语义对象（由 PathSegmentor 分割）进行有意义的生物学扰动

> **类激活图 (CAM)**：
> - CAM [47]：通过全局平均池化 + 分类器权重生成激活图
> - Foundation model for cancer imaging biomarkers [48]：基础模型用于癌症影像生物标志物
> - 本文方法：对象感知 CAM -- 不仅定位判别区域，还关联具体病理对象类别

A common deep learning approach for cancer diagnosis involves training an end-to-end classification model that takes a WSI as input and outputs a diagnostic prediction directly. However, the black-box nature of such models raises concerns about their trustworthiness. Feature importance estimation can provide global explainability of which features are most important to each disease class.

---

## 🔖 批读摘要

> 💡 **问题动机（四条脉络的关系）**：
> 1. **病理分割演进**：从手工特征到深度学习专用模型（HoVer-Net, nnU-Net），精度不断提升但类别受限
> 2. **SAM 系列**：提示驱动范式打破了固定类别的限制，但 SAM 本身在医学/病理上泛化差，需要域适配
> 3. **医学基础模型**：MedSAM 和 SAM-Med2D 用大规模医学数据适配，但仍是空间提示（缺乏语义），BiomedParse 引入文本但多模态训练稀释了病理特定知识
> 4. **可解释 AI**：RISE/LIME 用非语义扰动，CAM 定位但不知 "是什么"，本文需要将可解释性与病理语义结合
>
> PathSegmentor 恰好填补了这四条线的交汇处：**病理专域 + 文本提示 + 层级语义 + 可解释诊断**。

> 💡 **机制拆解（为什么 BiomedParse 在病理上表现差）**：
> BiomedParse 训练数据中病理仅占约 15k 样本（vs. PathSeg 的 275k），且其文本模板 `[object type] in [anatomical region] pathology` 缺失了组织学结构信息。在病理中，"tumor" 这个词在组织级（tissue-level）表示大面积的肿瘤区域，在细胞核级（nuclei-level）表示零散的肿瘤细胞核，形态和尺度差异巨大。没有层级区分，模型难以同时处理这两种情况。PathSegmentor 的模板 `[histological structure]-level [object type] in [anatomical region] pathology` 将尺度信息显式编码为文本提示。

> 💡 **对比表格**：

| 模型 | 提示类型 | 领域 | 语义输出 | 统一模型 | 病理数据量 |
|------|---------|------|---------|---------|-----------|
| SAM | 空间(点/框) | 自然图像 | 无 | 是 | 0 |
| MedSAM | 空间(框) | 多模态医学 | 无 | 是 | ~9k |
| SAM-Med2D | 空间(点/框/掩码) | 多模态医学 | 无 | 是 | ~1k |
| BiomedParse | 文本 | 多模态医学 | 有 | 是 | ~15k |
| SAM-Path | 类提示(可训练) | 病理 | 有(类别) | 否(逐数据集) | 逐数据集 |
| SegAnyPath | 空间(点/框) | 病理 | 无 | 是 | 大量(无语义) |
| **PathSegmentor** | 文本(层级) | 病理 | 有(完整层级) | 是 | 275k |

> 💡 **Q&A 批注记录**：
> - **Q: 为什么空间提示 "缺乏语义能力" 是严重的临床缺陷？** A: 在临床流程中，病理医生需要知道 "这个区域是肿瘤还是正常组织"。空间提示只能输出 "这里有东西" 的掩码，需要额外步骤来识别类别。PathSegmentor 的文本提示直接指定目标语义，输出的掩码天然带有语义标签，更贴合临床工作流。
> - **Q: RISE/LIME 等传统可解释方法有什么不足，为什么需要 PathSegmentor？** A: RISE 用灰色方块、LIME 用超像素作为扰动单元 -- 这些都不是生物学/病理学上有意义的单元。PathSegmentor 用病理对象（如 "breast-tissue-tumor"）做扰动，这使得特征重要性估计直接关联到病理概念，对医生来说更具解释性。
