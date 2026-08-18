[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

MOTCat 用**最优传输（OT）驱动的 co-attention** 融合病理 WSI + 基因组做生存预测。核心洞察：MCAT 式的 co-attention 只看两模态间**局部稠密相似**，忽略了模态内的**全局结构**（WSI 内 TME 交互、基因共表达）。OT 从全局视角匹配 patch 与基因，选出与基因共表达结构一致的信息 patch。为解决 OT 的高复杂度，提出 micro-batch 上的 unbalanced mini-batch OT（UMBOT）近似。

---

## Abstract

Survival prediction is a complicated ordinal regression task that aims to predict the ranking risk of death, which generally benefits from the integration of histology and genomic data. Despite the progress in joint learning from pathology and genomics, existing methods still suffer from challenging issues: 1) Due to the large size of pathological images, it is difficult to effectively represent the gigapixel whole slide images (WSIs). 2) Interactions within tumor microenvironment (TME) in histology are essential for survival analysis. Although current approaches attempt to model these interactions via co-attention between histology and genomic data, theyfocus on only dense local similarity across modalities, which fails to capture global consistency between potential structures, i.e. TME-related interactions of histology and co-expression of genomic data. To address these challenges, we propose a Multimodal Optimal Transport-based Co-Attention Transformerframework with global structure consistency, in which optimal transport (OT) is applied to match patches of a WSI and genes embeddings for selecting informative patches to represent the gigapixel WSI. More importantly, OT-based co-attention provides a global awareness to effectively capture structural interactions within TME for survival prediction. To overcome high computational complexity of OT, we propose a robust and efficient implementation over micro-batch of WSI patches by approximating the original OT with unbalanced mini-batch OT. Extensive experiments show the superiority ofour method onfive benchmark datasets compared to the state-of-the-art methods.

> 💡 **问题动机**（局部相似 vs 全局结构）（Hao 批注）：本文的靶子是 MCAT（[5]）——用基因当 query、对每个 (patch, gene) 对算稠密相似度做 co-attention。作者指出这是**局部视角**：它只问"这个 patch 像不像这个基因"，不问"被选中的 patch 集合整体是否与基因共表达的结构一致"。而生存相关的 TME 交互（如肿瘤细胞与浸润淋巴细胞 TIL 的共现）往往**空间分散在整张 WSI**，是长程结构。OT 的边际约束（总质量守恒）强制在模态内做权衡，从而捕获这种全局结构。

> 💡 **机制拆解**（OT co-attention 的三个优势）（Hao 批注）：作者列的三点很关键：
> 1. **全局意识**：OT 的边际约束让 patch 选择考虑"整体最优匹配"而非逐对局部相似 → 建模 WSI 内交互 + 基因共表达。
> 2. **不依赖注意力分数**：弱监督（只有 slide 级生存）下学出的 patch 注意力不是可靠度量；OT 的最优匹配流是**不需任何标签的闭式解**（解线性规划），更严格。
> 3. **降异质性**：最优匹配流提供了两模态间保结构的变换，缩小跨模态异质性 gap。

> 💡 **定位**（与本主题的联系）（Hao 批注）：这是本主题里"多模态生存预测 + 冗余/信息选择"的代表，与同目录 [PIBD](../../%5BICLR%202024%5D%20PIBD/) 是姊妹——都做病理+基因生存预测、都要从 gigapixel WSI 里选信息 patch。差异：MOTCat 用 OT 全局匹配选 patch，PIBD 用信息瓶颈 + 原型解耦压冗余。对压缩研究：**OT 匹配流 = 一种"由另一模态引导的、全局结构感知的 patch 重要性"**，比单模态注意力更鲁棒——这是"用什么信号做 retention"的一个候选。
