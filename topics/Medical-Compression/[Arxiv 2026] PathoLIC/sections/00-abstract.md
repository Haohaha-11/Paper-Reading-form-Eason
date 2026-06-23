# 00 Abstract

[← 返回 README](../README.md)

## 📌 Preview

> 全文摘要：WSI 数据量巨大（单张可达 80,000x80,000 像素、1-4GB），现有压缩方法的两个根本缺陷——(1) 忽略相邻 patch 之间的空间冗余，(2) 不考虑内容差异进行均匀压缩。PathoLIC 通过内容感知评分 + Attention 跨 patch 冗余消除 + 差异化码率分配三个机制，实现 >8x 压缩且保持下游诊断性能。

---

## 原文

The substantial size of gigapixel whole slide images (WSIs) presents significant challenges in terms of data storage, transfer, and computational analysis. Existing image compression methods yield suboptimal compression ratios because they (1) overlook redundancy across neighboring/similar patches, and (2) apply uniform compression without considering content differences. To address these issues, we introduce PathoLIC (Pathology Learned Image Compression), a novel learning-based variable-rate compression framework tailored for WSI.

> 💡 **问题动机**：这篇论文的出发点非常清晰——两个现有方法的根本性缺陷。缺陷 (1) 是忽略了 WSI 的结构化特性：相邻 patch 往往具有相似的形态学特征（如同一肿瘤区域内的多个相邻 patch），独立压缩每个 patch 意味着同样的纹理特征被反复编码，造成严重的比特浪费；缺陷 (2) 则是"一刀切"的压缩策略不符合临床实际需求——肿瘤区域和背景区域显然不应该用同样的压缩率。

Specifically, PathoLIC initially assigns a content score to each non-overlapping patch in the WSI, which reflects its diagnostic relevance. The compression level for each patch is determined based on the content scores, prioritizing detail preservation in diagnostically important regions, e.g., tumor area, while compressing more on less informative regions, e.g., stroma and background. Furthermore, PathoLIC employs attention mechanisms to capture relationships between neighboring or similar patches, which minimize redundancy by compressing shared features.

> 💡 **机制拆解**：PathoLIC 的三个核心机制链条：(1) Content Score → 每个 patch 的诊断重要性评估（自动化、无需人工标注）；(2) Content-Aware Compression → 根据 score 差异化分配码率（高分 = 低压缩，低分 = 高压缩）；(3) Attention → 跨 patch 建模，共享特征只编码一次，消除空间冗余。这三者构成了完整的"理解内容 → 差异化处理 → 去冗余"流水线。

Experimental results demonstrate that PathoLIC achieves over 8x compression beyond the standard Aperio SVS format while preserving image details. Moreover, it maintains strong performance across various downstream tasks, such as patch-level (WSI-level) cancer subtyping and nuclei segmentation. These results demonstrate its potential for large-scale WSI data management. The source code will be released at https://github.com/wqli498/PathoLIC.

> 💡 **关键数字**：>8x 压缩比是一个很硬核的数字——意味着原来 1PB 的病理数据可以压缩到约 120TB，这对医院的实际存储成本有巨大影响。更关键的是，这个压缩以"下游任务不下降"为前提，这才是医疗场景中最有说服力的指标。代码已开源，增加了可复现性。

---

## 🔖 摘要批注总结

- **问题定义**：千兆像素 WSI 的存储/传输瓶颈，现有压缩方法的两大缺陷（跨 patch 冗余 + 均匀压缩）
- **解决方案**：PathoLIC = Content Score（自动评估诊断重要性）+ Attention（消除跨 patch 冗余）+ QCM 调制（差异化码率分配）
- **核心结果**：>8x SVS 压缩，多类下游任务性能基本保持
- **一个隐藏亮点**：代码开源地址已给出（虽然目前可能还未发布），对社区友好
