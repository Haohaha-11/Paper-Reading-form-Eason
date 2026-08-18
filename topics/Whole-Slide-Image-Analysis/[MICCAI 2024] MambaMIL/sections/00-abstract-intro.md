[← 返回 README](../README.md)

# Abstract & Introduction 摘要与引言

## 📌 预览

MambaMIL 首次把 **Mamba（Selective State Space Model）** 引入 WSI MIL，用**线性复杂度**建模长序列 instance。核心组件 **SR-Mamba（Sequence Reordering Mamba）**：用两个并行分支建模两种不同排序的序列（原始序 + 重排序），利用 Mamba 位置敏感的特性从散布稀疏的阳性 patch 里捕获判别特征、缓解过拟合与高算力。在 baseline set 里对应"关键只是 efficient long-sequence modeling"这一竞争解释，且是 GMMamba 的 base-Mamba control。

---

## Abstract

Multiple Instance Learning (MIL) has emerged as a dominant paradigm to extract discriminative feature representations within Whole Slide Images (WSIs). Despite driving notable progress, existing MIL approaches suffer from limitations in facilitating comprehensive and efficient interactions among instances, as well as challenges related to time-consuming computations and overfitting. In this paper, we incorporate the Selective Scan Space State Sequential Model (Mamba) in MIL for long sequence modeling with linear complexity, termed as MambaMIL. By inheriting the capability of vanilla Mamba, MambaMIL demonstrates the ability to comprehensively understand and perceive long sequences of instances. Furthermore, we propose the Sequence Reordering Mamba (SR-Mamba) aware of the order and distribution of instances, which exploits the inherent valuable information embedded within the long sequences. With the SR-Mamba as the core component, MambaMIL can effectively capture more discriminative features and mitigate the challenges associated with overfitting and high computational overhead. Extensive experiments on two public challenging tasks across nine diverse datasets demonstrate that our proposed framework performs favorably against state-of-the-art MIL methods. The code is released at https://github.com/isyangshu/MambaMIL.

> 💡 **问题动机（为什么 SSM / Mamba 适合 WSI）**（Hao 批注）：MambaMIL 的定位是"高效长序列 MIL baseline"。它针对两个痛点：(1) ABMIL 类基于 i.i.d.、忽略 instance 上下文；(2) [TransMIL](../../%5BNeurIPS%202021%5D%20TransMIL/) 类 Transformer 有相关建模但**算力大、易过拟合**（即使 Nyström 近似）。Mamba 用**选择性状态空间**——线性复杂度、无近似地建模长序列全局感受野。关键洞察：**WSI 的阳性 patch 散布稀疏、空间相关弱**，很适合 Mamba 的序列建模能力（不像自然图像那样强空间局部性）。

> 💡 **机制拆解（SR-Mamba 的核心创新）**（Hao 批注）：vanilla Mamba 有个致命缺陷——**只能与"已扫描过的"位置交互，看不到未扫描的 patch**（单向因果扫描，感受野受限）。这对无序的 WSI patch 序列是问题（patch 顺序本无意义）。SR-Mamba 的解法：**两个并行分支建模两种排序**——(1) 原始序，(2) Sequence Reordering 后的新序（把 1D 序列 reshape 成 2D 再按另一维采样）。两种排序让 Mamba 从不同"扫描视角"捕获相关性，缓解单向感受野限制。这是 MambaMIL 相对直接套 Mamba（或 S4MIL）的关键增量。

## 1 Introduction

MIL conceptualizes WSI analysis as a **long sequence modeling problem**. Attention-based methods focus on instance-level info under i.i.d. hypotheses, neglecting contextual relationships. Transformer-based methods explore correlations but face bottlenecks due to extensive computations and overfitting.

Structured State Space Sequence (S4) addresses long sequence modeling bottleneck; Mamba advances S4 with input-dependent selection + hardware-aware algorithm, achieving **linear complexity without sacrificing global receptive fields**. However, for non-sequential visual data, direct Mamba application limits receptive fields (only interacts with previously scanned positions). WSIs contain scattered, scarce positive patches with weak spatial correlation, highly suitable for Mamba's sequential modeling. S4MIL introduced S4 to WSI but directly adopted S4 without considering WSI characteristics, giving sub-optimal results.

Contributions: (1) first application of Mamba in computational pathology; (2) SR-Mamba aware of order and distribution, capturing long-range dependencies among scattered positive instances via both sequential and transpositional ordering; (3) comprehensive experiments on 2 tasks across 9 datasets.

> 💡 **相关工作定位（三条长序列聚合路线）**（Hao 批注）：引言清晰划出 WSI 长序列建模的三条路线，正是 baseline set 里的三个方法：
> - **Transformer**（[TransMIL](../../%5BNeurIPS%202021%5D%20TransMIL/)）：$O(n^2)$ → Nyström 近似，算力大易过拟合。
> - **SSM/Mamba**（MambaMIL）：线性复杂度、无近似、全局感受野。
> - （后续 [RetMIL](../../%5BMICCAI%202024%5D%20RetMIL/) 的 retention 是第三类。）
>
> MambaMIL 相对 S4MIL（直接套 S4）的创新是 **SR-Mamba 针对 WSI 特性（散布稀疏阳性）设计重排序**。对 baseline set：MambaMIL = "efficient long-sequence modeling" 竞争解释，且是 [GMMamba](../../%5BICCV%202025%5D%20GMMamba/) 的 base-Mamba control（GMMamba = Mamba + evidence selection，需与纯 MambaMIL 对比才能拆出 selection 的增益）。
