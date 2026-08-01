[← 返回 README](../README.md)

# 00 — Abstract

> **原文**:

Pathology foundation models (PFMs) have emerged as powerful tools for analyzing whole slide images (WSIs). However, adapting these pretrained PFMs for specific clinical tasks presents considerable challenges, primarily due to the availability of only weak (WSI-level) labels for gigapixel images, necessitating multiple instance learning (MIL) paradigm for effective WSI analysis. This paper proposes a novel approach for single-GPU Task Adaptation of PFMs (TAPFM) that uses vision transformer (ViT) attention for MIL aggregation while optimizing both for feature representations and attention weights. The proposed approach maintains separate computational graphs for MIL aggregator and the PFM to create stable training dynamics that align with downstream task objectives during end-to-end adaptation. Evaluated on mutation prediction tasks for bladder cancer and lung adenocarcinoma across institutional and TCGA cohorts, TAPFM consistently outperforms conventional approaches, with H-Optimus-0 (TAPFM) outperforming the benchmarks. TAPFM effectively handles multi-label classification of actionable mutations as well. Thus, TAPFM makes adaptation of powerful pre-trained PFMs practical on standard hardware for various clinical applications.

> 💡 **核心问题与方法**: Hao 批注 — 本文解决的核心矛盾是：大规模 PFM（如 H-Optimus-0, ViT-giant）参数量大，在仅有 WSI 级弱标签的 MIL 范式下做端到端微调面临两个困难：(1) 计算资源——多 GPU 需求不实际；(2) 优化稳定性——PFM 和 MIL aggregator 联合训练容易不稳定。TAPFM 的两个关键设计直接对应这两个困难：用 ViT 内部注意力做 MIL 聚合（省参数、省显存），用 detach 双图分离优化（稳定训练）。

> 💡 **与现有 task adaptation 的区别**: Hao 批注 — 已有工作要么用固定特征+外部 MIL（不微调 PFM），要么用多 GPU 端到端微调（如 Campanella et al. 2024, DEMO 2024）。TAPFM 的独特之处在于用 ViT 自带的注意力机制替代外部 MIL 聚合器，并通过分离计算图实现单 GPU 稳定训练。这是一个"做减法"的设计——不去引入复杂 MIL 模块，而是更好地利用 PFM 已有的内部结构。
