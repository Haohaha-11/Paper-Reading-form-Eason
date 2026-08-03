[← 返回 README](../README.md)

# 1. Introduction & Related Work 引言与相关工作

## 📌 预览

引言提两个问题：如何从单模态去冗余抓判别信息（intra-modal）、如何从多模态重叠信息里抓紧凑而全面的知识（inter-modal）。相关工作三线：单模态生存预测（MIL / 基因 MLP-SNN，均不约束去冗余）、多模态生存预测（tensor-based / attention-based co-attention，对齐易丢模态特有信息）、信息论多模态学习（IB 压冗余、解耦抓目标知识）。

---

## 1 INTRODUCTION

Cancer survival analysis aims to estimate the death risk of patients for prognosis, in which multimodal learning by integrating both histological information and genomic molecular profiles can benefit the prognosis. Histological images give visual phenotypic information about tumor microenvironment for different grading, while genomics data provides global landscapes for various molecular subtyping. Nevertheless, a large quantity of redundancy in multimodal data poses significant challenges to effective fusion.

The primary question at hand is: **How can we capture the discriminative information from single modality by eliminating its redundancy, referred as "intra-modal redundancy" issue?** The label for a WSI is typically provided at the WSI level (weak supervision). The region of interest, e.g., tumor cells highly related to risk, only occupies a small portion of gigapixel WSIs. Although MIL provides promising solutions, they do not enforce constraints to remove redundant information. A similar redundancy issue emerges in genomic modality: pathways can yield hundreds to thousands of groups, and only a few specific pathways exhibit strong correlation with prognosis.

Another concern is: **How can we capture compact yet comprehensive knowledge from the dominant overlapping information in multimodal data, referred as "inter-modal redundancy" issue?** The knowledge can be split into modality-specific and modality-common. Existing efforts focus on integrating common information via alignment; however, common information often dominates, leading to the suppression of modality-specific information.

> 💡 **机制拆解**（两个问题定义了 PIB/PID 的分工）（Hao 批注）：作者把冗余问题精确切成两问——**问题一（intra-modal）**由 PIB 答：弱监督下 ROI 只占 WSI 一小块、基因只有少数 pathway 相关，MIL 不显式去冗余 → 需要一个"压冗余、留判别"的约束（IB）。**问题二（inter-modal）**由 PID 答：现有对齐式融合让"共有信息"主导、淹没"模态特有信息"→ 需要显式解耦并保护特有信息。这个"两冗余"框架是全文的组织骨架，也让方法与问题一一对应。

In this work, we propose PIBD, consisting of PIB for "intra-modal redundancy" and PID for "inter-modal redundancy". First, Information Bottleneck (IB) provides a promising solution to compress unnecessary redundancy while maximizing discriminative information. However, IB may suffer from high-dimensional computational challenges posed by massive patches and pathways. Instead, we propose a new IB variant, PIB, that models prototypes approximating a bunch of instances for different risk levels. Secondly, PID removes inter-modal redundancy by decomposing entangled multimodal features into modality-common and modality-specific knowledge, reusing the joint prototypical distributions modeled by PIB.

Contributions: (1) a new multimodal cancer survival framework PIBD addressing both redundancies; (2) PIB models prototypes for selecting discriminative info, PID decouples with guidance of joint prototypical distribution; (3) extensive experiments on five benchmarks.

## 2 RELATED WORKS

**Survival Prediction from Single Modality**: MIL for gigapixel WSIs (instance-level aggregation vs embedding-level: clustering, graph correlations, attention weights, transformer long-range). Genomic features via MLP/SNN. These do not provide constraints on removing redundant information.

**Survival Prediction from Multiple Modalities**: tensor-based fusion (concatenation, weighted sum, bilinear/Kronecker) — typically early/late fusion, neglecting inter-modal interactions. Attention-based fusion (co-attention): MCAT (gene-guided), MOTCat (OT global structure), SurvPath (cross-attention dense pathway-patch interactions). Though some alleviate redundancy by alignment, prone to lose modality-specific information.

**Multimodal Learning with Information Theory**: IB principle compresses raw information while retaining task-relevant knowledge (multi-view, multi-modal). Information disentanglement extracts targeted knowledge. We introduce this into multimodal cancer survival for the first time.

> 💡 **相关工作定位**（Hao 批注）：PIBD 站在三条线交汇——它继承 embedding-based MIL + attention 融合，但用**信息论**给出统一处方。关键区分：(1) 相对 MCAT/MOTCat/SurvPath（对齐式融合），PIBD 显式**保护模态特有信息**（PID 的 MI 最小化），不让共有信息淹没特有；(2) 相对普通 IB/VIB，PIBD 的 PIB 用**原型近似 bag 分布**绕开 bag 级 $p(z|\mathbf{x})$ 的高维不可算。这两点是它相对前作的真正增量。
