[← 返回 README](../README.md)

# Introduction 引言

## 📌 预览

引言点出现有 CPath 的低效：tile 级方法要提取分析每张 slide 上万 tile（本研究平均 ~18,000/slide @0.5MPP），且用逐任务训练的聚合器，可扩展性/可解释性差、在小活检等数据稀缺场景失效。EAGLE 用 CHIEF（全局组织表示 + 引导 tile 选择）+ Virchow2（选中 tile 精提特征）模仿病理学家聚焦 ROI 的策略。

---

Artificial intelligence (AI) has significantly advanced computational pathology (CPath) by enabling the extraction of clinically relevant information from gigapixel-scale whole-slide images (WSIs). Existing methods use resource-intensive vision transformers trained with self-supervised learning (SSL) to encode detailed morphological features. While these approaches have shown great promise, their inefficiencies and limited scalability highlight the need for solutions that better align with real-world diagnostic workflows.

Current methods predominantly operate at the tile level, requiring the extraction and analysis of thousands of tiles per WSI, with datasets in this study averaging approximately 18,000 tiles per slide at a resolution of 0.5 µm/pixel (MPP). This computationally intensive process deviates from how pathologists evaluate slides, as they selectively focus on regions of interest. Moreover, tile-wise features are aggregated into slide-level predictions using models trained separately for each task, limiting scalability and interpretability. These systems also struggle in data-scarce scenarios, where tile selection often fails to identify the most relevant regions, leading to suboptimal predictions. Such scenarios are often a clinical reality, for example during the evaluation of small biopsy specimens.

> 💡 **机制拆解**（三个痛点 → EAGLE 的三个对策）（claude 批注）：作者列的三个痛点精确对应 EAGLE 的设计：
> 1. **上万 tile 太慢**（~18,000/slide）→ EAGLE 只处理 25 个（省 >99% 算力）。
> 2. **逐任务训练聚合器、不可扩展/不可解释**→ EAGLE 产出 **task-agnostic 的统一 embedding**（一次算好，多任务下游只训小 MLP），且 25 个 tile 可审计。
> 3. **数据稀缺时 tile 选择失效**→ CHIEF 在 6 万+ slide 上预训练的 task-agnostic 显著性先验，不依赖下游任务标签，小样本更稳（few-shot 实验证明）。

To address these limitations, we developed EAGLE, a framework that emulates the diagnostic strategy of pathologists by focusing on a small, informative subset of tiles within WSIs. EAGLE combines CHIEF, a pretrained and task-agnostic model used for global tissue representation and guided tile selection, with Virchow2, for detailed feature extraction from selected tiles. This combination substantially reduces computational demands while increasing performance. By selecting a small, reproducible subset of regions, EAGLE enhances auditability and scalability, particularly in biomarker prediction tasks where subtle morphological features are critical. Unlike MLLMs, which emphasize multimodal interaction, EAGLE prioritizes efficient high-quality WSI analysis. Through comprehensive evaluation against state-of-the-art models, including multiple instance learning (MIL) and slide-encoder approaches, we demonstrate the efficacy and robustness of EAGLE across 43 tasks spanning nine cancer types.

> 💡 **机制拆解**（两个 FM 的分工 = "粗筛 + 精提"）（claude 批注）：EAGLE 的精髓是**组合两个互补的病理基础模型**：
> - **CHIEF（slide-level, task-agnostic）**：在 CTransPath 特征上运行，产出全局组织表示 + 每个 tile 的注意力分数 → 用来**选 tile**。它便宜（0.36 ms/slide）、且因在 6 万+ slide 上预训练，选择先验稳定、跨癌种通用。
> - **Virchow2（tile-level, 强特征）**：只对 CHIEF 选中的 25 个 tile 做精细特征提取。它贵，但只用在 ~2% tile 上（2MPP）。
>
> **"用便宜模型粗筛全图、用昂贵模型精提关键区"**——这正是 ReadySlide "compress once, analyze with strong FM"、以及 allocator "把预算花在高价值 patch"的思路。EAGLE 证明这个两阶段范式在 43 任务上不仅省算力还涨点。
