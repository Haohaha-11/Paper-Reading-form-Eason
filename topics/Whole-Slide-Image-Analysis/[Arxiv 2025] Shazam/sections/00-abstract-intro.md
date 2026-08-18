[← 返回 README](../README.md)

# Abstract & Introduction 摘要与引言

## 📌 预览

Shazam 是**多个病理基础模型的在线自适应融合模型**：不做离线蒸馏（如 GPFM 需专用蒸馏数据 + 每加新模型都重训），而是用轻量 MoE **在线、任务特定地**融合多个冻结 FM 的**多层（low/mid/high）特征** + 自适应专家加权 + 在线蒸馏。30 个 benchmark 平均排名 1.17（次优 Virchow2 3.20）。**这是 baseline set 里 CKMIL "多层 FM 表示" 主线的最近竞争工作**——它已占据"多层病理表示互补 + 任务自适应融合"这部分 novelty。

---

## Abstract

Foundation models have substantially advanced computational pathology, yet their performance varies widely across tasks due to differences in training data composition and reliance on proprietary datasets that cannot be cumulatively expanded. Existing efforts to combine foundation models through offline distillation partially mitigate this but require dedicated distillation data and repeated retraining to integrate new models. Here we present Shazam, an online integration model that adaptively combines multiple pretrained pathology foundation models within a unified and scalable representation learning paradigm. Fusing **multi-level features through adaptive expert weighting and online distillation** enables efficient consolidation of complementary model strengths without additional pretraining. Across spatial transcriptomics prediction, survival prognosis, tile-level classification, and visual question answering, Shazam consistently outperforms strong individual models.

> 💡 **问题动机 + 对 CKMIL 的关键定位**（Hao 批注）：**这是 baseline set 文档里标注"最高优先级（新颖性对照）"的论文**——因为它与 CKMIL/ReadySlide 的"多层 FM 表示"主线最接近。核心逻辑链：
> 1. **单 FM 不够**：benchmark（如 [PathBench](../../%5BArxiv%202025%5D%20PathBench/)）显示无普适赢家，不同 FM 各有专长。
> 2. **离线蒸馏（GPFM）的局限**：需专用蒸馏数据、加新 FM 要重训、任务无关（不能针对下游任务选知识）。
> 3. **Shazam 的解法**：**在线、任务特定**地融合多个冻结 FM 的**多层特征**（low/mid/high）+ MoE 自适应加权 + 在线蒸馏。加新 FM 无需重训。
>
> **对 CKMIL 的冲击（务必重视）**：Shazam 已经证明并占据了"**多层病理表示是互补的、可任务自适应融合**"这部分 novelty。所以 CKMIL **不能把"多层 FM 表示有用"本身当核心 novelty**。需要区分：
> - **Shazam** = 多 FM + 多层融合 + 自适应专家加权（online distillation）。
> - **CKMIL 目标** = 单 FM + depth-wise 表示 + slide/task 条件的 depth selection + MIL。
> 关键差异是 **多 FM 融合 vs 单 FM depth-selection**——CKMIL 必须把这个区分讲清楚。

> 💡 **机制拆解（online vs offline integration）**（Hao 批注）：Shazam 的"在线"是相对 GPFM"离线蒸馏"的核心卖点：
> - **离线（GPFM）**：预先用多 teacher 在固定蒸馏集上生成特征、训一个 student → 加新 teacher 要从头重训、受蒸馏集规模限制、任务无关。
> - **在线（Shazam）**：teacher 全程冻结当特征提取器，**推理/训练时按当前任务动态融合** → 加新 FM 只需插进来、无需重训、任务自适应。
>
> 这是"组合式 vs 单体式"FM 范式的转变——从"训一个大而全的 FM"转向"模块化组合现有 FM"。
