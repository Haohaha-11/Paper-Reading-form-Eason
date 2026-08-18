[← 返回 README](../README.md)

# Abstract & Introduction 摘要与引言

## 📌 预览

RetMIL 用 **retention 机制（借自 RetNet/LLM）替代非线性 self-attention**，配**层次结构**：局部层——WSI 序列切成多个子序列，每个子序列 token 用并行线性 retention 更新 + 注意力池化聚合；全局层——子序列融合成全局序列、用串行 retention 更新、全局注意力池化得 slide 表示。解决 Transformer-MIL 的高内存/慢推理/性能瓶颈。在 baseline set 里对应"更合适的 retention-style long-context aggregation 就足够"这一竞争解释。

---

## Abstract

Histopathological WSI analysis with deep learning has become a research focus. The current paradigm is mainly based on MIL, in which approaches with Transformer as the backbone are well discussed. These methods convert WSI tasks into sequence tasks by representing patches as tokens. However, the feature complexity brought by high heterogeneity and the ultra-long sequences brought by gigapixel size makes Transformer-based MIL suffer from high memory consumption, slow inference speed, and lack of performance. To this end, we propose a retentive MIL method called RetMIL, which processes WSI sequences through hierarchical feature propagation structure. At the local level, the WSI sequence is divided into multiple subsequences. Tokens of each subsequence are updated through a parallel linear retention mechanism and aggregated utilizing an attention layer. At the global level, subsequences are fused into a global sequence, then updated through a serial retention mechanism, and finally the slide-level representation is obtained through a global attention pooling. We conduct experiments on two public CAMELYON and BRACS datasets and an public-internal LUNG dataset, confirming that RetMIL not only achieves state-of-the-art performance but also significantly reduces computational overhead.

> 💡 **问题动机（retention = 长序列聚合的第三条路线）**（Hao 批注）：RetMIL 与 [TransMIL](../../%5BNeurIPS%202021%5D%20TransMIL/)、[MambaMIL](../../%5BMICCAI%202024%5D%20MambaMIL/) 构成 WSI 长序列聚合的**三条路线**：
> - **TransMIL**：self-attention + Nyström 近似（$O(n^2)$→近似 $O(n)$）；
> - **MambaMIL**：SSM 选择性扫描（线性 $O(n)$）；
> - **RetMIL**：retention 机制（借自 RetNet，线性 $O(n)$ 且训练可并行、推理可递归）。
>
> retention 的独特优势：**同时有并行（训练快）和递归（推理省内存）两种形式**，且带**显式的相对距离衰减矩阵 $D$**（近的 token 权重大）。在 baseline set 里，RetMIL 排除"更合适的 retention-style 长上下文聚合就够"这一解释。

> 💡 **机制拆解（层次结构 = local + global 两级 retention）**（Hao 批注）：RetMIL 的核心不只是"换 retention"，而是**层次化**（hierarchical）：
> - **局部层**：把超长 WSI 序列切成固定长度（512）子序列 → 每个子序列内**并行 retention** 更新 token + 注意力池化成一个子序列向量。并行处理多个子序列，避免单条超长序列的开销。
> - **全局层**：所有子序列向量组成全局序列（长度 = 子序列数）→ **串行 retention** 更新 + 全局注意力池化 → slide 表示。
>
> 这个"分块并行局部 + 串行全局"的两级设计，让内存几乎不随序列长度增长（Fig.3b：GPU 内存近乎常数）——这是 RetMIL 相对 TransMIL（内存随长度增）的关键卖点。

## 1 Introduction

MIL methods categorized into instance-level and embedding-level. Embedding-level focuses on aggregation strategies. Transformer-based MIL models patch correlation via self-attention, showing better performance, but **square complexity consumes more memory during training/inference, increasing latency and reducing speed** — not conducive to clinical deployment.

RetMIL introduces a retention mechanism to replace nonlinear self-attention, and effectively integrates subsequence information via hierarchical structure to obtain global representation with local features. Results demonstrate lower memory cost and higher throughput while competitive performance.

> 💡 **相关工作定位**（Hao 批注）：RetMIL 归到 embedding-level MIL（冻结特征 + 聚合），明确针对 Transformer-MIL 的**部署痛点**（内存/延迟/速度）。它的论证重点不是"更准"（性能只是 competitive/略优），而是**"同等性能下大幅省内存 + 提吞吐"**——这是面向临床部署的实用导向，与 [EAGLE](../../%5BNat%20Commun%202026%5D%20DL-Efficient-Pathology/)/[LitePath](../../%5BArxiv%202026%5D%20Deployment-Friendly-CPath/) 的效率主线一致。对 baseline set：RetMIL 提供 attention/Transformer/Mamba 之外的第三类高效长上下文 baseline。
