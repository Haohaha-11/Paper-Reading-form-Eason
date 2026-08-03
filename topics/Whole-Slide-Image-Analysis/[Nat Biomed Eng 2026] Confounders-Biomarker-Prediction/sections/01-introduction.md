[← 返回 README](../README.md)

# Introduction 引言（Main）

## 📌 预览

引言建立核心假设：WSI 里的表型是**多个共依赖分子因子交互**的结果，而非单因子。现有方法只预测单个 biomarker、忽略共依赖 → 学到复合信号 + cohort 特异偏差。作者用四步验证：(1) biomarker 互斥/共现分析；(2) 训 WSI 预测模型；(3) 分层 + permutation 检验混杂；(4) 对比"病理 grade 基线"看 ML 的增量价值。

---

Fuelled by developments in computational pathology, several studies have proposed methods to predict clinically relevant biomarkers, such as gene mutations and expression levels, directly from routine haematoxylin and eosin (H&E)-stained whole-slide images (WSIs). These approaches take a WSI as input and predict the status of clinically relevant biomarkers such as microsatellite instability (MSI), hormonal receptors or mutations in TP53, BRAF, KRAS, EGFR and other genes, as their target. Such methods are typically motivated by two main objectives: first, to identify or mine histological patterns associated with specific biomarkers, and second, to rule out certain biomarkers from routine WSIs, avoiding the need for additional stains or molecular testing, which can be tissue-destructive, costly and associated with longer turnaround time.

Several methods have demonstrated that, in specific cancers, biomarker status and alterations in certain genes are predictable from WSIs using deep learning pipelines trained in a weakly supervised fashion on imaging and molecular data from The Cancer Genome Atlas (TCGA) or other similar data repositories, such as the Clinical Proteomic Tumour Analysis Consortium (CPTAC). However, for most biomarkers, the prediction accuracy of these methods remains low, with the area under the receiver operating characteristic curve (AUROC) values ranging from 0.50 to 0.90. Moreover, the true generalization of such methods to external datasets is further challenged by factors such as mutation prevalence, limited multicentric data, class imbalance between positive (mutated or high expression) and negative (wild-type or low expression) cases, quality of WSIs (such as pen markings and tissue tears) and domain shifts. In this Article, we demonstrate that even if these challenges have been handled, there are underlying fundamental issues that require addressing.

> 💡 **机制拆解**（本文比"域偏移/类不平衡"更深一层）（Hao 批注）：作者特意区分——以往讨论 WSI biomarker 预测的困难都停在"工程层"（突变率、多中心少、类不平衡、染色伪影、域偏移）。本文说：**即使这些都解决了，还有一个更根本的问题——混杂**。这是把批判从"数据质量"升级到"因果结构"。

In a WSI, disease phenotypes manifest as different visual patterns arising from the interaction of multiple codependent genes rather than from a single factor. These interactions are often characterized by patterns of mutual exclusivity or co-occurrence among molecular factors. Despite this, current approaches primarily focus on predicting the status of individual biomarker or gene mutation from WSIs, neglecting codependencies between covariates...

In this study, we show that overlooking interdependencies among biomarkers can influence the predictive performance of machine learning (ML) models. We argue that interdependencies among biomarker statuses in the training data, when ignored during model development, can lead to models capturing the aggregated influence of multiple interdependent biomarkers rather than patterns linked to a single biomarker. Moreover, this could also spuriously inflate or deflate models' apparent performance in certain subgroups when the interdependency structure among molecular factors shifts in the test cohorts. Finally, when clinicopathological variables (for example, tumour mutational burden (TMB) or tumour grade) are themselves associated with biomarker status, models may rely on phenotypes associated with these correlated variables as predictive proxies, instead of capturing the intended biological signal.

> 💡 **机制拆解**（三条混杂路径）（Hao 批注）：这段把"混杂"拆成三条具体机制，是全文的分析框架：
> 1. **biomarker 间共依赖**：模型学到"复合表型"（多个相关 biomarker 的合并影响），而非单 biomarker。
> 2. **依赖结构漂移**：测试集里 biomarker 间的关联变了 → AUROC 虚高或虚低。
> 3. **临床病理变量代理**：grade/TMB 本身与 biomarker 相关 → 模型用 grade/TMB 的形态当 proxy，而非真正的 biomarker 生物学信号。
>
> 后面 Results 就是分别验证这三条（共依赖 biomarker 分层、grade 分层、TMB 分层）。

To illustrate these effects, we first analysed interdependencies among biomarkers by assessing their patterns of mutual exclusivity and co-occurrence. We then use permutation testing and stratification analysis to demonstrate failure modes of WSI-based predictors by showing that their accuracy for a given biomarker varies substantially when conditioned on the status of other biomarkers. We also highlight the need for appropriate causal adjustments in WSI-based predictors to ensure reliable inferences necessary for informing clinical decisions, such as treatment selection and pathobiological understanding. To this end, we propose a stratification-based evaluation framework to report bias and support the development of more transparent and trustworthy ML models to advance WSI-based precision diagnostics.

> 💡 **方法论批读**（分层 + permutation = 本文的"武器"）（Hao 批注）：作者的核心工具不是新模型，而是一套**诊断协议**：
> - **分层分析**：按混杂变量把队列分组，比较"组内 AUROC" vs "整体 AUROC"。若组内显著更低 → 模型靠混杂。
> - **permutation 检验**（10,000 次）：打乱混杂变量与标签的关联，构造零分布，看观测的组内 AUROC 是否显著偏离。
>
> 这套协议**可直接迁移到评估任何 WSI 模型是否学了 shortcut**——包括压缩方法。对 ReadySlide：压缩后若 AUROC 不掉但分层后崩，说明压缩保留的是 shortcut 而非因果信号。这正是 memory 里"必须对抗 shortcut-learning、而非只看 AUROC"那条结论的方法论落地。
