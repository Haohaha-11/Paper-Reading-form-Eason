# Segment Anything in Pathology Images with Natural Language (SAMPath)

> **Authors**: Zhixuan Chen, Junlin Hou, Liqi Lin, Yihui Wang, Yequan Bie, Xi Wang, Yanning Zhou, Ronald Cheong Kin Chan, Hao Chen
> **Affiliation**: HKUST, USTC, CUHK, Tencent
> **Venue**: arXiv 2025 (2506.20988)
> **Code**: https://anonymous.4open.science/r/PathSegmentor-3166

---

## 一句话总结

PathSegmentor 是**首个面向病理图像的文本提示分割基础模型**，基于全新的 275k 样本 PathSeg 数据集构建，覆盖 160 个层级类别（20 个解剖区域 x 3 种组织学结构 x 61 种对象类型），通过自然语言描述即可实现语义分割，无需空间提示。

---

## 核心贡献

1. **PathSeg 数据集**：最大最全的病理语义分割基准 -- 来自 21 个公开数据集的 275k 图像-掩码-标签三元组，以三级层级标签体系（解剖区域、组织学结构、对象类型）解决病理标注中的语义歧义问题。

2. **PathSegmentor 模型**：基于 Transformer 编码器-解码器架构的基础模型，配备联合特征交互模块（交叉注意力 + 自注意力），通过可学习查询（learnable queries）融合视觉特征（FocalNet）和文本特征（PubMedBERT），仅凭文本提示即可预测语义掩码。

3. **全面的实验验证**：PathSegmentor 在内部（16 个数据集）和外部（5 个数据集）评估中，全面超越专用模型（nnU-Net, DeepLabV3+, SAM-Path）、空间提示基础模型（MedSAM, SAM-Med2D）和文本提示模型（BiomedParse）。

4. **复杂对象鲁棒性**：在不规则形状、小实例和高密度对象的分割上表现出显著优势，且仅需 1 个文本提示即可完成任务，而空间提示模型平均需要约 15 个提示/掩码。

5. **可解释癌症诊断管道**：分割与分类的双向集成，实现基于对象的特征重要性估计和影像生物标志物发现，赋能可解释的诊断决策支持。

---

## 📖 批读导航

| 章节 | 文件 | 核心内容 |
|---------|------|-------------|
| Abstract | [00-abstract.md](sections/00-abstract.md) | 问题动机，PathSeg + PathSegmentor 概览 |
| 1. Introduction | [01-introduction.md](sections/01-introduction.md) | 背景、现有方法的局限性、本文贡献 |
| 2. Related Work | [02-related-work.md](sections/02-related-work.md) | SAM 家族、医学分割基础模型、病理分割 |
| 3. Methodology | [03-methodology.md](sections/03-methodology.md) | PathSeg 数据集构建、PathSegmentor 架构、可解释诊断管道 |
| 4. Experiments | [04-experiments.md](sections/04-experiments.md) | 内部/外部验证、复杂对象分析、定性结果、可解释性 |
| 5. Conclusion | [05-conclusion.md](sections/05-conclusion.md) | 关键优势讨论、局限性及未来工作 |

---

## 关键数字

| 指标 | 数值 |
|--------|-------|
| PathSeg 数据集规模 | 275k 图像-掩码-标签三元组 |
| 来源数据集数量 | 21 个公开数据集（16 个内部 + 5 个外部） |
| 层级类别数 | 160（20 AR x 3 HS x 61 OT） |
| 解剖区域数 | 20 |
| 组织学结构数 | 3（组织、细胞、细胞核） |
| 对象类型数 | 61 |
| PathSegmentor 模型参数量 | 0.45B |
| 16 个专用模型组总参数量（16 x SAM-Path） | 1.86B（减少 75%） |
| 整体 Dice（PathSegmentor） | 0.671 |
| 整体 Dice（nnU-Net, 最优专用模型） | 0.502 |
| 相对 MedSAM 的提升 | +0.145 Dice |
| 相对 BiomedParse 的提升 | +0.429 Dice |
| 提示效率（文本 vs. 实例框） | 1 个提示 vs. 平均约 15 个提示/掩码 |
| PathSeg 中单掩码最大实例数 | >800 |
| 分类 AUC（标准模型） | 0.936（宏观 AUC） |
| 分类 AUC（对象感知模型） | 0.953（宏观 AUC） |

---

## 数据流：输入 → 中间表示 → 输出

```
输入：病理图像（1024x1024, H&E 染色）
  +
输入：文本提示（"tissue-level tumor in breast pathology"）
  │
  ▼
图像编码器 (FocalNet) ──> F_image ∈ R^{m×d}（视觉特征）
  │
  ▼
文本编码器 (PubMedBERT) ──> F_text ∈ R^{L×d}（语义特征）
  │
  ▼
联合特征交互模块：
  ├── 可学习查询 q ∈ R^{n×d}
  ├── 交叉注意力 Cross-Attention(q, F_image) ──> q'（视觉增强查询）
  ├── 自注意力 Self-Attention([q' || F_text]) ──> F_joint
  ├── 前馈网络 Feed-Forward Network ──> q''（语义感知查询）
  ├── 掩码投影器 ──> E_mask（通过点积生成候选掩码）
  └── 类别投影器 ──> E_cls（类别嵌入）
  │
  ▼
掩码选择：argmax 余弦相似度(E_cls^i, F_text')
  │
  ▼
输出：二值分割掩码 ŷ ∈ {0,1}^{H×W}
```

可解释诊断扩展的数据流：

```
输入：全切片图像 (WSI)
  ├── [分类→分割管道]
  │   图像块特征 → 切片聚合 → 分类器 → 预测
  │                                          ↓
  │   PathSegmentor 掩码 → 基于对象的扰动 → 特征重要性
  │
  └── [分割→分类管道]
      图像块特征 × PathSegmentor 对象掩码 → 对象感知特征
      → 逐对象聚合 → 统一切片特征 → 分类器
      → 对象感知 CAM（类别激活图）带语义标签
```

---

## 优缺点与还能做什么

### 优点

- **统一架构**：单一模型替代 16 个专用模型，同时取得更好的整体 Dice（0.671 vs. nnU-Net 组的 0.502）。
- **语义感知**：文本提示编码层级病理知识（解剖区域 + 组织学结构 + 对象类型），解决分割任务中的语义歧义问题。
- **高效的提示机制**：单个文本提示替代每掩码约 15 个空间提示，对临床工作流至关重要（在 WSI 上标注数百万个对象是不可行的）。
- **对复杂对象的鲁棒性**：在不规则形状、微小实例和高密度区域上保持稳定性能，而这些场景下空间提示模型会出现显著性能退化。
- **可解释性集成**：与分类模型的双向耦合，同时提供特征重要性估计和对象感知 CAM，直接支持临床决策。

### 局限 / 风险

- **数据集规模仍然有限**：275k 样本 vs. 非语义数据集中的数百万样本；病理中的语义标注本身成本高昂。
- **新类别的泛化性能**：文本提示模型在语义上未见的对象类型上可能退化；空间提示可为此类情况提供互补的定位信息。
- **内皮细胞分割弱点**：PathSegmentor 和 BiomedParse 在稀疏聚集的细胞（如 CoNSeP 中的内皮细胞）上均不如空间提示模型 -- 文本提示缺乏对"少且集中"实例的精确定位信息。
- **不支持多提示**：当前架构仅使用文本提示；结合文本 + 空间提示可提升对未见类别的鲁棒性。
- **真实临床验证尚未完成**：未见多中心临床试验报告；病理医生反馈集成已规划但尚未执行。
- **基于 BiomedParse 初始化**：继承通用医学模型的权重；完全从头进行病理特定预训练可能带来进一步提升。

---

## 阅读 Q&A 记录

> **Q1：为什么不直接使用 BiomedParse 来做病理分割？PathSegmentor 到底增加了什么？**
>
> BiomedParse 在多模态生物医学数据上训练，其中病理样本仅约 15k（vs. PathSeg 的 275k）。其文本模板 `[object type] in [anatomical region] pathology` 省略了组织学结构信息（组织/细胞/细胞核），而这一信息对于解决病理中的语义歧义至关重要（例如， "tumor" 既可以指肿瘤组织，也可以指肿瘤细胞）。PathSegmentor 的三级层级模板 `[histological structure]-level [object type] in [anatomical region] pathology` 显式编码了这种多尺度上下文。

> **Q2：为什么 PathSegmentor 在高密度对象上优于空间提示模型？**
>
> 空间提示模型使用单个联合框（union box）缺乏足够的空间信息来分辨密集区域中的单个实例 -- 框只是把所有东西都包进去。实例框虽然更精确，但每个掩码需要约 15 倍数量的提示。文本提示通过大规模训练隐式学习了类别级别的形状和分布先验，使 PathSegmentor 能够在不需要显式空间定位的情况下分割密集实例。

> **Q3：联合特征交互模块中可学习查询的角色是什么？**
>
> 可学习查询作为自适应滤波器，首先通过交叉注意力聚合关键的视觉上下文，然后通过自注意力与文本语义融合。生成的语义感知查询同时编码了空间定位（"在哪里"）和语义类别（"是什么"），从而支持掩码嵌入和类别嵌入的生成。这与 DETR 的查询驱动检测范式类似，但适配为文本提示驱动的分割任务。

> **Q4：三级层级真的有必要吗，还是仅仅是标签工程？**
>
> 论文提供了证据表明它确实重要：BiomedParse 使用更简单的模板（省略组织学结构）在 PathSeg 上仅取得 0.242 Dice，而 PathSegmentor 达到 0.671。组织学结构层级（组织/细胞/细胞核）至关重要，因为同一对象类型（如 "tumor"）在不同尺度上需要根本不同的分割策略 -- 组织级的肿瘤是大面积的不规则区域，而细胞核级的肿瘤是微小密集的对象。层级体系显式编码了这一先验。

> **Q5：在训练中未见的类别上表现如何？**
>
> 论文承认了这一局限性。在外部评估中，他们只测试了 [object type] 与训练类别对齐的类别。对于真正新颖的语义类别，文本提示性能可能会退化。作者提出在未来工作中结合文本 + 空间提示来解决这一问题 -- 空间提示提供定位，文本提示为已知类别提供语义。

> **Q6：对象感知 CAM 与普通 CAM 有何不同？**
>
> 普通 CAM 高亮判别性图像区域，但无法指定这些区域包含哪些病理对象。对象感知 CAM 将分类信号分解为逐对象的贡献，将每个高亮区域与特定的病理类别关联（例如 "breast-tissue-tumor" 而不仅仅是 "暖色区域"）。这弥合了显著性图与临床解释之间的语义鸿沟。

---

## 📊 Citation Landscape

本文位于三个研究方向的交汇处：
- **病理图像分析**（nnU-Net, HoVer-Net, SAM-Path, SegAnyPath）
- **分割基础模型**（SAM, MedSAM, SAM-Med2D, BiomedParse, SEEM）
- **医学可解释 AI**（CAM, RISE, 特征重要性估计）

Connected Papers: [https://www.connectedpapers.com/main/2506.20988](https://www.connectedpapers.com/main/2506.20988)

关键引用文献：
- SAM [13]：原始提示驱动分割范式
- SEEM [16]：多提示（点、框、文本、涂鸦）统一框架 -- PathSegmentor 的架构基础
- MedSAM [19]：通过 1.5M 医学图像-掩码对将 SAM 适配到医学影像领域
- SAM-Med2D [20]：以 19.7M 掩码扩展医学 SAM
- BiomedParse [21]：覆盖 9 种模态的文本提示生物医学分割
- SAM-Path [10]：使用类别提示的病理特定 SAM 适配（专用模型，非基础模型）
- SegAnyPath [31]：空间提示驱动的病理分割基础模型（无语义感知）
- nnU-Net [9]：自配置医学分割框架（逐数据集的专用模型基线）

---

*最后更新：2025-06-22*
