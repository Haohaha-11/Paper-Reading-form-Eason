[← 返回 README](../README.md)

# 00 - Abstract

## 预览

本节把 AdaSlide 的完整 claim 压缩成四个数字：病理 WSI 太大；专家 VTT 约 55% 接近随机；13 个下游任务多数保持性能；存储降到 10-35%。

# Abstract

Digital pathology generates gigapixel whole-slide images that require extensive storage, limiting scalability and increasing costs. Traditional compression methods apply uniform ratios, disregarding variations in diagnostic importance across regions. Here we show Adaptive compression for gigapixel whole-slide images (AdaSlide), a framework that balances compression efficiency and diagnostic integrity. AdaSlide integrates a reinforcement learning–based Compression Decision Agent (CDA) that determines the optimal compression ratio per region, and a Foundational Image Enhancer (FIE) that restores visual fidelity after compression. In evaluations, pathologists achieved near-chance accuracy (55 $\%$ ) in distinguishing original from restored images, confirming high perceptual fidelity. Across 13 downstream tasks, AdaSlide maintained diagnostic performance in most tasks while reducing storage to $1 0 { - } 3 5 \%$ of the original size. By aligning compression strategies with clinical relevance, AdaSlide enables efficient, scalable, and reliable storage for digital pathology and future AI development.

> 💡 **摘要证据链**: 摘要不是只说“压缩更强”，而是同时要求 compression efficiency 和 diagnostic integrity。CDA 决定哪里压缩，FIE 负责压缩后的视觉/结构复原；VTT 支撑“人眼难区分”，13 个任务支撑“AI/诊断任务多数不掉点”，10-35% 原始大小支撑“归档成本下降”。

> 💡 **Q&A 批注记录**:
> - Q: 为什么本文不直接用固定压缩率？
> - A: 摘要第一句后的关键假设是不同 WSI 区域诊断重要性不同。固定压缩率会把 tumor-rich、cell-dense 或难复原区域和背景/脂肪/空白区域同等处理。
> - Q: “55%”为什么是正面结果？
> - A: VTT 是让病理专家判断哪张是复原图，越接近 50% 越说明复原图难以被肉眼区分；这不是分类性能，而是感知保真度证据。

## Section 总结

| 关键对象 | 作用 |
|---|---|
| CDA | patch-level keep/compress decision |
| FIE | compressed patch restoration |
| 13 downstream tasks | 诊断性能验证 |
| VTT | pathologist perceptual fidelity 验证 |
| 10-35% | 摘要层面的存储压缩结果 |
