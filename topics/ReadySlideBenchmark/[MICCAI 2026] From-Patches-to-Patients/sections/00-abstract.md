[← 返回 README](../README.md)

# 摘要

> 💡 **claude 批注｜阅读预览**: 这篇工作不是再造一个病理基础模型，而是把“选哪个编码器”本身当作基准问题：先用便宜的 tile 级线性探测形成模型排序，再问这一排序能否迁移到使用 Mean Pooling 或 ABMIL 的 slide-level 任务。核心输出不是单一模型的准确率，而是两套排行榜之间的相关性、稳健性与候选集重叠。

# From Patches to Patients: A study of the tile-to-slide performance transferability in Digital Pathology

Sofiène Boutaj<sup>1,2</sup>, Leo Fillioux<sup>1,2</sup>, Maria Vakalopoulou<sup>1,2</sup>, Stergios Christodoulidis<sup>1,2†</sup>, and Pierre Marza<sup>1,2†</sup>

<sup>1</sup> Université Paris-Saclay, CentraleSupélec, Gustave Roussy, INSERM, IHU PRISM, Cancer Data Science Unit, France

<sup>2</sup> Université Paris-Saclay, CentraleSupélec, MICS Laboratory, France Corresponding author: sofiene.boutaj@centralesupelec.fr

Abstract. Foundation Models (FMs) have recently redefined the stateof-the-art in histopathology by providing robust representations for wholeslide image (WSI) analysis. However, selecting the optimal foundation model (FM) for a specific clinical cohort currently requires multiple preprocessing steps, followed by computationally expensive feature extraction and the training of a Multiple Instance Learning (MIL) aggregator for every model. In this work, we investigate whether eficient tile-level linear probing can serve as a reliable proxy for slide-level performance, reducing the need to run full slide-level pipelines for every candidate encoder. We benchmark 19 state-of-the-art FMs on 42 slide-level and 16 tile-level tasks, comparing tile probing metrics against slide-level outcomes using ABMIL and Mean Pooling aggregations. We observe a high correlation between tile and slide performance across varying task difficulties, indicating that encoder representation quality is the primary determinant of WSI success. Sensitivity analyses show that transferability is stable across models and is more influenced by cohort sizes and numbers of tiles per slide than by average task dificulty. We also measure the agreement in best performing models between tile and slide-level tasks, showing tile benchmarks reliably shortlist strong candidates. Overall, our study indicates that tile-level benchmarking provides an eficient and practical first step for narrowing down candidate models, while slide-level evaluation remains essential for final validation on clinical tasks.

> 💡 **claude 批注｜结论边界**: “proxy”在这里应读成筛选器而非替代品。论文证明的是：对同一组 19 个编码器，跨 16 个 tile 任务平均得到的排序，与跨 42 个 slide 任务平均得到的排序高度相关；它没有证明任意一张 tile 的局部预测可以直接代替 slide-level 预测，也没有证明 tile 排名能在每个具体临床任务上精确找出第一名。

Keywords: Digital pathology · Foundation models · Benchmark.

> 💡 **claude 批注｜ReadySlide 对照**: 对 ReadySlide 最可复用的研究范式是“benchmark-first”，但两端必须排同一批对象。A 类问题固定预算端 selector 与预算，在 full-bag 和 budgeted 两端比较同一组 encoder–consumer pipelines；B 类问题固定 encoder／consumer，在两个非退化预算间比较同一组 selectors，并用 retention／regret 对照共同的 full-bag 基线。full-bag 没有可辨识的 selector 排名，不能与 budgeted selector 榜直接做相关。

## 本节小结

- 输入：19 个冻结的病理基础模型及 tile／slide 两套任务集。
- 中间量：tile 汇总分数、slide 汇总分数、两端模型排序。
- 输出：相关性、敏感性与 top-5 重叠，而不是一个新的诊断器。
- 关键保留问题：总体相关能否推出具体任务、具体预算下的可替代性？正文答案是否定的，只能用于候选缩小。
