[← 返回 README](../README.md)

# Abstract 摘要

## 📌 预览

ACMIL 的核心命题：**MIL 注意力机制"过度集中"到少数判别性 instance，是 WSI 分类过拟合的直接推手**。作者用两组分析（UMAP + Top-K 累积注意力）定位问题，再用两个技术分别对症：**MBA（Multiple Branch Attention，多分支捕获更多判别模式）** + **STKIM（Stochastic Top-K Instance Masking，随机遮蔽 Top-K 显著 instance 并把注意力重分配）**。

---

Abstract. In the application of Multiple Instance Learning (MIL) methods for Whole Slide Image (WSI) classification, attention mechanisms often focus on a subset of discriminative instances, which are closely linked to overfitting. To mitigate overfitting, we present Attention-Challenging MIL (ACMIL). ACMIL combines two techniques based on separate analyses for attention value concentration. Firstly, UMAP of instance features reveals various patterns among discriminative instances, with existing attention mechanisms capturing only some of them. To remedy this, we introduce Multiple Branch Attention (MBA) to capture more discriminative instances using multiple attention branches. Secondly, the examination of the cumulative value of Top-K attention scores indicates that a tiny number of instances dominate the majority of attention. In response, we present Stochastic Top-K Instance Masking (STKIM), which masks out a portion of instances with Top-K attention values and allocates their attention values to the remaining instances. The extensive experimental results on three WSI datasets with two pre-trained backbones reveal that our ACMIL outperforms state-of-the-art methods. Additionally, through heatmap visualization and UMAP visualization, this paper extensively illustrates ACMIL's efectiveness in suppressing attention value concentration and overcoming the overfitting challenge. The source code is available at https://github.com/dazhangyu123/ACMIL.

> 💡 **问题动机**（Hao 批注）：本文把"过拟合"这个泛泛的病，精确归因到一个可测量的量——**注意力值的过度集中**（attention value concentration）。这是本主题（WSI Analysis）里少见的"先诊断、再开药"路线：不是又提一个新聚合器，而是先证明"注意力熵低 ↔ 验证损失高"（Fig. 1 的负相关），再针对"集中"的两种成因分别设计 MBA 和 STKIM。对压缩/保留研究的启示：**如果注意力本身就不可靠（过度集中在少数 patch），那么用注意力当 patch 重要性来做保留是危险的**——ACMIL 恰恰在说"重要的 patch 远不止 Top-10"。

> 💡 **机制拆解**（两个成因 → 两个药）（Hao 批注）：
> 1. **成因一（模式多样性不足）**：判别性 instance 在特征空间里有多个 cluster/pattern，单分支注意力只抓住一部分 → 药方 **MBA**：多分支，每支专攻一种 pattern，并用多样性正则强制各支互异。
> 2. **成因二（少数 instance 垄断注意力）**：Top-10 instance 就占了 >0.85 的注意力质量 → 药方 **STKIM**：训练时随机把 Top-K 的注意力清零并重分配给其余 instance（类比 dropout/cutout），推理时移除。
>
> 两药正交（一个增"宽度"，一个抑"尖峰"），可叠加。

Keywords: Computational pathology · Whole slide image · Multiple instance learning · Overfitting

> 💡 **定位**（Hao 批注）：关键词把 "Overfitting" 单列——这是全文的靶心。ACMIL 与 [MHIM-MIL](../../%5BICCV%202023%5D%20MHIM-MIL/)（同目录）是"表亲"：都遮蔽显著 instance 来逼模型看更多证据。但 STKIM 只遮 K=10 个、无需 teacher-student、推理时移除；MHIM 需动量 teacher 预训练 + 遮 1% instance。二者的对比是本主题"如何正确利用/挑战注意力"这条线的核心。
