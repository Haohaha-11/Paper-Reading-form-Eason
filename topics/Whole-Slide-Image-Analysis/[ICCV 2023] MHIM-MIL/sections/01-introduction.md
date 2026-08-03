[← 返回 README](../README.md)

# 1. Introduction & Related Work 引言与相关工作

## 📌 预览

引言用一张对比图（Fig. 1）立论：以往 MIL 盯显著 instance，MHIM 挖难 instance 学更好的边界。相关工作梳理两条线——WSI 里的 MIL（instance-level vs embedding-level，均偏爱显著 instance）和 CV 里的 hard sample mining（loss/相似度/可学习权重三类，均需样本标签）。MHIM 的贡献 = 把 hard mining 迁移到"无 instance 标签"的 MIL。

---

## 1. Introduction

Histopathological image analysis plays a crucial role in modern medicine, particularly in the treatment of cancer, where it serves as the gold standard for diagnosis [23, 25, 29,51]. Digitalizating pathological images into Whole Slide Images (WSIs) through digital slide scanner has opened new avenues for computer-aided analysis [9, 31]. Due to the huge size of a WSI and the lack of pixel-level annotations, histopathological image analysis is commonly formulated as a multiple instance learning (MIL) task [11, 28, 36]. In MIL, each WSI (or slide) is a bag containing thousands of unlabeled instances (patches) cropped from the slide. With at least one instance being disease positive, the bag is deemed positive, otherwise negative.

![Fig 1](../images/73e17a69fae8ff43cd06320755a30157ce863f7da297366d622987fea548b822.jpg)

*Figure 1: 左——以往 MIL 模型盯更显著的 instance；右——MHIM-MIL 挖掘大量难分 instance 来学更好的边界。*

> 💡 **Figure 1 批读**（一图立论）（Hao 批注）：这张图是全文思想的浓缩——借用 SVM 的直觉：**决定分类边界的是靠近边界的"难样本"（支持向量），而非远离边界的"易样本"**。左图 MIL 只用少数高置信 instance → 边界粗糙；右图 MHIM 遮掉易样本、逼模型用难样本 → 边界更精细。关键点：难样本不是要在测试时用，而是在**训练时**帮模型建更好的判别边界。

However, the number of slides is limited and each slide contains a mass of instances with a low positive proportion. This imbalance would hinder the inference of bag labels [20, 49]. To alleviate this issue, several WSI classification methods [6, 20, 22, 23, 31] employ an attention mechanism to aggregate salient instance features into a bag-level feature for WSI classification. Furthermore, some MIL frameworks [22, 26, 46, 49] focus on the more salient instances in the bag and leverage them to facilitate WSI classification. For instance, existing frameworks [46, 49] propose to only select the instances that correspond to the top K highest or lowest attention scores [22, 46] or patch probabilities [49] for yielding high-quality bag embedding for both training and testing.

These salient instances are actually "easy-to-classify" instances, which are not optimal for training a discriminative WSI classification model. In conventional machine learning, such as Support Vector Machines (SVM) [17], samples near the category distribution boundary are more challenging to classify, but are more useful for depicting the classification boundary, as illustrated in Figure 1. Moreover, other deep learning works [30, 33, 38, 39] also reveal that mining hard samples for training can improve the generalization abilities of models. By applying such an idea at the instance level, we can better highlight the "hard-to-classify" instances that facilitate MIL model training, and benefit the final WSI classification. However, the lack of instance labels poses a challenge to the direct application of traditional hard sample mining strategies at the instance level.

To address this issue, we present a novel MIL framework based on masked hard instance mining strategies (MHIM) named MHIM-MIL. The main idea of MHIM is to mask out the instances with high attention scores to highlight the hard instances for model training. Based on this, we incorporate two other instance masking strategies to enhance training efficiency and mitigate the over-fitting risk. Another key design of MHIM-MIL is an instance attention generator based on a Siamese structure (Teacher-Student) [3, 8]. In MHIM-MIL, the MIL-based WSI classification model is the student network, which aggregates hard instances mined by a momentum teacher with different instance masking strategies. The momentum teacher is updated using an exponential moving average (EMA) of the student model. Moreover, the framework is optimized by inducing a consistency constraint that explores more supervised information beyond the limited slide label. Unlike the conventional MIL frameworks [46, 49], which adopt complex cascade gradient-updating structures, our method is more simple and does not require additional parameters. It not only improves efficiency but also provides improved performance stability. The contribution of this paper is summarized as follows,

• We propose a simple and efficient MIL framework with masked hard instance mining named MHIM-MIL. It implicitly mines hard instances with instance attention for training a more discriminative MIL model. Extensive experiments on two WSI datasets validate that MHIM boosts different MIL models and outperforms other latest methods in terms of performance and training cost.

• We propose several hybrid instance masking strategies for indirectly mining hard instances in MIL. These strategies not only address the reliance problem of conventional methods on instance-level supervision but also enhance the training efficiency of the model and mitigate the over-fitting risk.

• With the Siamese structure, we introduce a parameterfree momentum teacher to obtain instance attention scores more efficiently and stably. Moreover, we employ a consistency-based iterative optimization to improve the discriminability of both models progressively.

> 💡 **机制拆解**（三个贡献如何环环相扣）（Hao 批注）：
> 1. **MHIM 策略**（遮高注意力 → 挖难 instance）是核心思想，但单靠它有两个风险：可能把关键信息全遮了（"error mining"）、且用 student 自身打分不稳。
> 2. **动量 teacher（Siamese）**解决"打分不稳"：teacher 由 student EMA 更新，比 student 自身（batch=1 的 SGD，噪声大）稳得多，且**无额外参数**。
> 3. **一致性损失**解决"监督信息太少"：WSI 只有 bag 标签，一致性约束让 teacher 的 bag 表征去监督 student，挖出 slide 标签之外的额外监督。
>
> 三者缺一不可（Tab. 3 消融：MHIM→+Siam→+Con 逐级提升）。

## 2. Related Work

### 2.1. Multiple Instance Learning in WSI Analysis

Multiple Instance Learning (MIL) [11] has been widely used in WSI analysis... Previous algorithms can be broadly categorized into two groups: instance-level [4, 14, 19, 46] and embeddinglevel [9, 32, 44, 45, 49]. The former obtain instance labels and aggregate them to obtain the bag label, whereas the latter aggregate all instance features into a high-level bag embedding for bag prediction. Most embedding-level methods share the basic idea of AB-MIL [20], which employs learnable weights to aggregate salient instance features into bag embedding. Furthermore, some MIL frameworks [22, 26, 46, 49] mine more salient instances making classification easier and facilitating classification... However, all these methods focused excessively on salient instances in training, which are easy instances with high confidence scores and can be easily classified. As a result, they overlook the importance of hard instances for training. In this paper, we intend to mine hard instances for improving WSI classification performance.

### 2.2. Hard Sample Mining in Computer Vision

Hard sample mining is a popular technique to speed up convergence and enhance the discriminative power of the model in many deep learning areas, such as face recognition [30], object detection [34, 42], person reidentification [1, 33, 38, 39], and deep metric learning [35, 37]. The main idea behind this technique is to select the samples which are hard to classify correctly (i.e., hard negatives and hard positives) for alleviating the imbalance between positive and negative samples and facilitating model training. There are generally three groups of approaches for evaluating sample difficulty: loss-based [18], similaritybased [7], and learnable weight-based [47]. Typically, these strategies require complete sample supervision information. Drawing on the ideas of the above works, we propose a hard instance mining approach in MIL, mining hard examples at the instance level. In this, there are no complete instance labels, only the bag label is available. Similar to our approach, Li et al. utilized attention scores to identify salient instances from false negative bags to serve as hard negative instances and used them to compose the hard bags for improving classification performance [24]. A key difference is that we indirectly mine hard instances by masking out the most salient instances rather than directly locating hard negative instances.

> 💡 **相关工作定位**（Hao 批注）：两条线交汇处就是 MHIM 的创新点。**WSI-MIL 线**：不管 instance-level 还是 embedding-level，主流（ABMIL/CLAM/DTFD）都偏爱显著 instance。**CV hard-mining 线**：三类方法（loss-based/相似度/可学习权重）都需**样本级监督**。MHIM 的独到之处 = 在"只有 bag 标签"的 MIL 里，用注意力分数**间接**挖难样本（遮显著），而非像 [24] 那样直接定位 hard negative。这个"间接"是关键——它把成熟的 hard mining 思想搬进了弱监督 WSI。
