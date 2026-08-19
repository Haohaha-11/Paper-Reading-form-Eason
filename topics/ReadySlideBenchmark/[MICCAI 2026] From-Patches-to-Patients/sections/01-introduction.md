[← 返回 README](../README.md)

# 1 引言

> 💡 **claude 批注｜问题动机**: WSI 无法整图直接送入常规视觉编码器，必须经历组织分割、切 tile、逐 tile 编码、再聚合成 slide 表征。于是模型比较的成本被“每个编码器都要重跑全套 WSI 管线”放大；本文要验证的是，能否用带密集标签的 tile 基准先做一次廉价排位。

## 1 Introduction

Digital pathology plays a central role in diagnosis and prognosis by automatically extracting relevant information about the cellular environment from tissue imaging. This is particularly true now that many foundation models [7,34,41,29,12,13] [25,20,22,8,40,17,16,37] were introduced as powerful generalist feature extractors for histopathology images. However, the high-resolution of Whole Slide Images (WSI) collected in histopathology makes it impossible to process them directly. There is a need for a specific processing pipeline: in practice, each WSI is first segmented to remove background and isolate tissue, then divided into tiles; each tile is embedded with a foundation model, and all tile-level features are aggregated to perform a slide-level prediction.

![Figure 1](../images/dec08f0f6ebcdbd35c4c977cd2cba073083b61cc548ca66162cf20235f8e3d84.jpg)  
*Fig. 1: Comparison of slide-level and tile-level benchmarks. (1,2) Overview of the 2 benchmark types. (3) Storage and compute requirements: Average storage (log scale) per dataset and compute time per dataset and per model, measured on a single NVIDIA V100 GPU. Slide-level averages were computed across 42 tasks and 19 models, while tile-level averages were computed across 16 tasks and 19 models.*

> 💡 **claude 批注｜Figure 1 成本账**: 图把“更便宜”量化为同一 V100 条件下的平均数据集成本：原始数据约 300 GB 对 10 GB、特征约 21 GB 对 1 GB、特征提取约 17 小时对 1 小时（slide 对 tile）。更重要的是两端监督不同：slide 端以整张 WSI 的 slide label 训练聚合器，tile 端以人工圈定 ROI 的局部标签做冻结特征线性探测。因而这是成本—临床贴近度的交换，不是完全同任务的加速版。

Comparing such foundation models, and better understanding their diferences thus becomes important to draw a clear picture of the progress in the field but also from a practical point of view when selecting a feature extractor for a new clinical task. For these reasons, various benchmarks were proposed recently [36,19,26,15,3,14,5,21,23,2,39,6,24]. We can divide them in two categories: tile-level and slide-level benchmarks. The former evaluates how foundation model embeddings can extract relevant information from specific tiles, or regions of WSIs, while the latter targets slide-level predictions. Figure 1 provides an overview of both types of benchmarks, presents a comparison of their storage and compute eficiency, and summarizes their advantages. As presented in [24], tile-level benchmarks are more eficient, isolate the impact of foundation models from required aggregators to perform slide-level predictions, and leverage denser supervision. On the other hand, slide-level benchmarks require more pre-processing, more compute, come with sparser supervision, and necessitate training aggregators on top of foundation model features, but are closer to the final clinical tasks of interest.

> 💡 **claude 批注｜两个排行榜的对象不同**: tile 基准主要测“冻结编码器的局部表征是否线性可分”，slide 基准测“编码器 + 聚合规则能否利用一整包 tile 完成 slide-level 预测”。前者隔离了 aggregator，后者却把 aggregator 的归纳偏置和训练噪声也计入结果。本文在 ABMIL 下观察到比 Mean Pooling 更多重排，作者解释为任务特异可学习聚合带来的额外变异；仅凭这两个设置不能外推“aggregator 越强，迁移越差”的一般规律。

We are thus presented with the following dilemma: tile-level benchmarks allow a more direct and eficient evaluation of foundation models, which is appealing from a methodological point of view, while slide-level benchmarks better model clinical needs. An important question is thus: How do relative performances from tile-level benchmarks transfer to slide-level ones? A good transfer would imply that tile-level benchmarks are efective tools for evaluating new models, providing confidence that their relative ranking will be preserved on slide-level tasks. However, this question is not binary but more nuanced. This paper aims not only to address it, but also to better understand the conditions under which performance transfers.

> 💡 **claude 批注｜迁移定义**: 这里的 transferability 不是参数迁移或零样本泛化，而是“模型相对表现的迁移”：给每个编码器一个 tile 汇总分数并排序，再给同一编码器一个 slide 汇总分数并排序，检查两列分数／名次是否一致。失败可以表现为线性关系弱、整体秩次颠倒，或虽然全局相关高但目标任务的 top-k 候选重叠低。

Our contributions are fourfold: (i) to the best of our knowledge, we provide the first large-scale tile-to-slide benchmarking study across 19 open-source pathology foundation models, 16 tile-level tasks, and 42 slide-level tasks from publicly available datasets; (ii) we measure tile-to-slide rank correlation using Pearson, Spearman, and Kendall metrics for both mean-pooling and ABMIL slide-level aggregation, and show strong transferability overall; (iii) we perform sensitivity analyses showing stable correlations and highlighting the role of cohort size and number of tiles per slide for tile-to-slide transferability; and (iv) we show through a top-5 overlap analysis that tile-level benchmarks are useful for shortlisting candidate encoders before expensive slide-level training, and highlight the current limitations of tile-level datasets.

> 💡 **claude 批注｜证据链预告**: 四项贡献对应四层证据：规模保证结论不是单数据集偶然；三种相关系数区分“分数线性关系”和“名次一致”；删模型／删任务测试结论由谁驱动；top-5 重叠把统计相关转译成实际 shortlist 价值。最后一层尤其关键，因为工程决策通常不是复现完整排行榜，而是从 19 个模型缩到少数候选。

## 本节小结

- 基准先行的目标：先判断廉价基准是否能承担候选筛选，再决定昂贵临床验证投入。
- 迁移单位：同一组编码器在两个 benchmark 上的分数与名次，不是病例级预测迁移。
- ReadySlide 可追问点：不能把 encoder pipeline 与 selector 两种候选对象混成一次迁移。应分别研究固定 selector／budget 时同一批 encoder–consumer pipelines 的 full-bag→budgeted 迁移，以及固定 encoder／consumer 时同一批 selectors 在非退化预算之间的排名稳定性。
