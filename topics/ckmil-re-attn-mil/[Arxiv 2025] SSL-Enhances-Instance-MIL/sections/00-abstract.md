[← 返回 README](../README.md)

# 00 Abstract

> 💡 **Hao 批注 - 一句话概要**: 这是一个"反直觉"的基准研究——通常认为嵌入级MIL(如ABMIL/TransMIL)因为参数更多、建模能力更强，应该优于实例级MIL(如MaxMIL/MeanMIL)。但本文通过710次实验证明，配合好的SSL特征提取器后，极简单的实例级MIL方法就能匹配甚至超越复杂的嵌入级SOTA。

> 💡 **Hao 批注 - 核心洞察**: 嵌入级MIL过去之所以占优，可能是因为ImageNet预训练特征质量不够好——特征不够好时需要更强的聚合器来补偿；但SSL大幅提升了特征质量后，简单聚合就足够了。这个洞察与Paper 1(Spatial-Blindness)形成了有趣的互补：Paper 1说即使有复杂空间模块也不一定被用上，Paper 2说即使不用复杂模块效果也够好。

> 💡 **Hao 批注 - 方法亮点**: 从声音事件检测领域引入的4种池化算子(MixMIL/AutoMIL/LNPMIL/AttenMIL)是新颖的，这些算子通过可学习参数在max和mean之间自适应调整，比简单的Max/Mean更灵活但参数极少(每个算子仅1-3个参数)。

> 💡 **Hao 批注 - 实验规模**: 710次实验的规模确实令人印象深刻，覆盖了几乎所有关键组合。但需要注意：(1) 嵌入级MIL只包含4种，缺少基于Graph的方法如PatchGCN；(2) 没有K-fold交叉验证(除CAMx20外)；(3) VisioMel只有验证集结果。

---

**原文 Abstract:**

Multiple Instance Learning (MIL) has emerged as the best solution for Whole Slide Image (WSI) classification. It consists of dividing each slide into patches, which are treated as a bag of instances labeled with a global label. MIL includes two main approaches: instance-based and embedding-based. In the former, each patch is classified independently, and then the patch scores are aggregated to predict the bag label. In the latter, bag classification is performed after aggregating patch embeddings. Even if instance-based methods are naturally more interpretable, embedding-based MILs have usually been preferred in the past due to their robustness to poor feature extractors. However, recently, the quality of feature embeddings has drastically increased using self-supervised learning (SSL). Nevertheless, many authors continue to endorse the superiority of embedding-based MIL. To investigate this further, we conduct 710 experiments across 4 datasets, comparing 10 MIL strategies, 6 self-supervised methods with 4 backbones, 4 foundation models, and various pathology-adapted techniques. Furthermore, we introduce 4 instance-based MIL methods never used before in the pathology domain. Through these extensive experiments, we show that with a good SSL feature extractor, simple instance-based MILs, with very few parameters, obtain similar or better performance than complex, state-of-the-art (SOTA) embedding-based MIL methods, setting new SOTA results on the BRACS and Camelyon16 datasets. Since simple instance-based MIL methods are naturally more interpretable and explainable to clinicians, our results suggest that more effort should be put into well-adapted SSL methods for WSI rather than into complex embedding-based MIL methods.
