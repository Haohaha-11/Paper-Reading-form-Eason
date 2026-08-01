[← 返回 README](../README.md)

# 01 Introduction

> 💡 **Hao 批注 - 问题设定**: WSI分类的标准pipeline是"特征提取 + MIL聚合"。过去大多数研究集中在改进第二步(MIL聚合)，因为第一步用ImageNet预训练的特征质量不够好。但SSL的兴起改变了这一格局——特征质量大幅提升后，简单的实例级聚合可能就足够了。

> 💡 **Hao 批注 - 历史视角**: 实例级MIL(先对每个patch分类再聚合分数)是更"老"的方法，但因为过去特征质量差，研究者发现嵌入级MIL(先聚合特征再分类)更鲁棒，所以后者成了主流。这篇论文质疑的是"这种偏好现在是否还成立"。

> 💡 **Hao 批注 - 现有局限 (Section 1.3)**: 作者指出的现有benchmark的不足很到位：(1) 大多数只用1-2种SSL方法；(2) 最多用1-2种实例级MIL做baseline；(3) 数据集少且临床复杂度低(主要是Camelyon16)；(4) SSL方法和backbone的测试不系统。

---

## 1.1 Multiple-Instance Learning (MIL)

> 💡 **Hao 批注 - MIL两大范式**: 实例级(instance-based): patch classifier h(.) -> score per patch -> pooling -> slide prediction。嵌入级(embedding-based): patch encoder f(.) -> feature aggregation -> bag classifier g(.) -> slide prediction。关键区别在于"分类发生在聚合之前还是之后"。

Under MIL formulation, each WSI is treated as a "bag" containing multiple instances in the form of patches. The bag is labeled positive (i.e., diseased) if at least one of its patches is positive, or negative if all patches are negative.

Instance-based methods use an instance-level classifier, which predicts a score for each patch. Then, these scores are aggregated via a MIL-based pooling operator to make the final prediction for the entire slide. Common pooling operators include average-pooling (MeanMIL) and max-pooling (MaxMIL).

Embedding-based methods proposed to aggregate features instead of scores, moving the classification head after the pooling. The existing literature has proposed several approaches for feature aggregation based on: deep self-attention mechanism [ABMIL], graph convolutional networks, clustering, transformer [TransMIL], and additional training [DTFD-MIL].

Until recently, most researchers used ImageNet pre-trained models to extract features. However, these models might not be optimal for histopathology images due to the domain gap. This might explain why most early, ImageNet-based works reported that embedding-based MIL models outperformed instance-based ones.

## 1.2 Self Supervised Learning (SSL)

> 💡 **Hao 批注 - SSL方法概览**: 本文测试了6种SSL: SimCLR(对比学习), MoCoV3(动量对比+ViT), MAE(掩码自编码器), DINO(知识蒸馏), BYOL(自举式), Barlow Twins(冗余消除)。加上3种病理适应方法(PathAug, SRCL, CluBYOL)。覆盖了SSL的主要范式。

SSL has shown promise in improving the performance of image classification. In digital pathology, SSL has been actively used: SimCLR, MoCo, DINO, BYOL, MoCo V3, and specialized methods like SRCL, CluBYOL, and hierarchical DINO.

## 1.3 Existing Limitations

Even if recent works provided very insightful results for WSI classification using SSL methods, their experiments and comparison present some limitations:

- They either use a single SSL method, or compare few SSL methods, or multiple SSL methods but with different backbones, or a single backbone for all methods.
- Most works focus on a few embedding-based MIL methods, using at most one/two instance-based methods.
- Some works use few datasets (usually with low clinical complexity, like Camelyon16) for binary classification only.

## 1.4 Our Contributions and Main Results

> 💡 **Hao 批注 - 四个贡献**: (1) 710次实验的大规模研究；(2) 4种新实例级MIL方法；(3) 实例级MIL配合SSL匹配或超越嵌入级SOTA；(4) 新SOTA on BRACS和Camelyon16。

- Large-scale study: 4 datasets, 7 pre-training methods, 4 foundation models, 4 backbones, 10 MIL methods = 710 configurations.
- New instance-based MIL approaches based on pooling mechanisms from sound event detection.
- Simple instance-based MIL methods, with robust SSL, are on par or outperform complex embedding-based methods.
- New SOTA results in BRACS (89.4 AUC) and Camelyon16 (99.1 AUC), on par in TCGA-NSCLC.
