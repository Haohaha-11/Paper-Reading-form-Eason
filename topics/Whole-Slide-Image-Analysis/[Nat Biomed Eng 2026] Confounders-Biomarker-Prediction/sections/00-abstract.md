[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

一篇**警示性/批判性**研究：用统计分析证明"从 H&E 组织图像预测分子 biomarker"的深度学习模型，**学到的是被混杂（confounded）的信号**——训练数据里 biomarker 之间、biomarker 与临床病理特征（grade、TMB）之间存在强依赖，模型无法隔离单个 biomarker 的效应。结论：现有方法**尚不能替代分子检测**，只能谨慎用于分诊/辅助决策；要做到无混杂预测，需学习因果而非相关关系。

---

Deep learning models that infer clinically relevant biomarker status from tissue images are being explored as rapid and low-cost alternatives to molecular testing. Here we show, through statistical analysis across multiple cancer types, datasets and modelling approaches, that the datasets used to train these models contain strong dependencies between biomarkers and clinicopathological features, which prevent models from isolating the efect of a single biomarker and lead them to learn confounded signals. Consequently, their prediction accuracy varies substantially with the status of codependent biomarkers and clinicopathological variables, and for several biomarkers, the gain over what a pathologist can already infer from routine histopathological features, such as grade, remains modest. These findings indicate that current approaches are not yet suitable as substitutes for molecular testing but can support triage or complementary decision-making with caution. Unconfounded biomarker prediction will require models that learn causal rather than correlational relationships between biomarkers and tissue morphology.

> 💡 **问题动机**（这篇"打脸"文在本主题的价值）（Hao 批注）：本主题（WSI Analysis）里绝大多数论文都在"刷 AUROC"，这篇反其道而行——**质疑高 AUROC 本身的意义**。核心论点分三层：
> 1. **biomarker 之间相互依赖**（互斥/共现，如 CRC 里 MSI-H 常伴 BRAF 突变）；
> 2. 模型没建模这种依赖 → 学到的是"多个相关 biomarker 的复合表型"，而非单个 biomarker 的特异形态；
> 3. 一旦测试集里依赖结构变了（如 grade-biomarker 关联漂移），高 AUROC 就崩（Simpson 悖论）。
>
> **对 ReadySlide/压缩研究的直接冲击**：任何声称"WSI 模型学到生物学"的说法（包括压缩后仍保诊断信息）都必须排除混杂——**光看 AUROC 不掉不够，要看分层（grade/TMB/共依赖 biomarker）后是否仍稳**。这是评估压缩方法"是否真保留了因果诊断信号 vs 只是保留了 shortcut"的关键方法论。

> 💡 **机制拆解**（confounded signal 是什么）（Hao 批注）：设想训练一个"预测 ER 状态"的模型。若训练集里 ER+ 常伴低 grade、TP53 野生型，模型可以**只学 grade/TP53 的形态**就把 ER 预测得很准（因为它们共现）。但这不是"ER 的形态"，而是代理（proxy）。测试时如果 ER 与 grade 的关联变了（不同中心），模型就失效。**关键诊断工具**：分层分析——把测试集按混杂变量（如 grade）分组，若各组内 AUROC 显著低于整体 AUROC，说明模型靠的是混杂而非 biomarker 本身。

> 💡 **定位**（Hao 批注）：作者不否定 WSI biomarker 预测的价值（可做假设生成、分诊、预筛），但坚决反对"替代分子检测"。给出的解药方向：**结构化多标签学习 + 因果调整 + 反事实增强 + 分布鲁棒 + 分层校准评估**。这为本主题所有"WSI→分子/诊断"工作立了一道方法论闸门。
