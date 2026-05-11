[← 返回 README](../README.md)

# Source Scope

## 📌 预览

本节记录当前 WSIR2 条目的来源边界。用户给出的 ScienceDirect PDF 是带 `X-Amz-Expires=300` 的短时 signed URL，当前会话无法复用；稳定 ScienceDirect 页面只开放摘要、Introduction 片段、若干 section snippets 和 DOI 元数据。因此这个目录不是完整 MinerU 批读，而是一个受限来源的初读占位。

> 💡 **来源边界批注**: 这里必须单独写清楚，因为 paper-reading 项目通常要求 `full.md` 原文完整保留。WSIR2 现在没有可下载 PDF，所以不能声称“已逐段阅读全文”；能可靠写入的只有摘要页公开信息、Crossref/Elsevier metadata 和由这些信息推导出的研究判断。

## 可确认信息

- 标题: Whole slide image redundancy reduction for efficient pathological diagnosis
- 作者: Shijie Li, Zhineng Chen, Feng-Jung Chen, Xieping Gao
- 期刊: Biomedical Signal Processing and Control
- 年份/卷号: 2026, Volume 114
- DOI: 10.1016/j.bspc.2025.109289
- Article number: 109289
- 核心方法名: WSIR2
- 公开摘要页 claim: 最高 24x GFLOPs 降低，保持与主流 MIL 方法相当的诊断性能。

> 💡 **可信度批注**: 标题、作者、DOI、卷号来自 Crossref/Elsevier metadata，可信度高；方法模块和关键数字来自 ScienceDirect 摘要页，适合做初读摘要；具体表格数值、baseline 设置、统计显著性和消融结论需要全文 PDF 才能验证。

## 当前不能确认的信息

- 完整方法公式和变量定义。
- 具体 MIL baselines、encoder、patch size、magnification、硬件和 batching 是否完全公平。
- 所有数据集上的逐项指标、置信区间或显著性检验。
- 可视化是否真正覆盖了稀疏病灶、边界区域和失败案例。

> 💡 **Q&A 批注记录**:
> - Q: 为什么不直接把摘要页当作 full.md?
> - A: 摘要页不是完整论文正文，且 ScienceDirect 明确需要机构访问全文；把它当成完整 MinerU 输出会污染后续阅读状态。
> - Q: 后续补 PDF 后要做什么?
> - A: 下载 PDF、跑 MinerU、生成 `full.md`/`content_list.json`/`images/`，再用全文替换当前受限 section。
