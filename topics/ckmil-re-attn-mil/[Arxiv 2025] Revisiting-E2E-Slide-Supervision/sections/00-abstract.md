[← 返回 README](../README.md)

# 00 Abstract

## 原文

Pre-trained encoders for offline feature extraction followed by multiple instance learning (MIL) aggregators have become the dominant paradigm in computational pathology (CPath), benefiting cancer diagnosis and prognosis. However, performance limitations arise from the absence of encoder fine-tuning for downstream tasks and disjoint optimization with MIL. While slide-level supervised end-to-end (E2E) learning is an intuitive solution to this issue, it faces challenges such as high computational demands and suboptimal results. These limitations motivate us to revisit E2E learning. We argue that prior work neglects inherent E2E optimization challenges, leading to performance disparities compared to traditional two-stage methods. In this paper, we pioneer the elucidation of optimization challenge caused by sparse-attention MIL and propose a novel MIL called ABMILX. ABMILX mitigates this problem through global correlation-based attention refinement and multi-head mechanisms. With the efficient multi-scale random patch sampling strategy, an E2E trained ResNet with ABMILX surpasses SOTA foundation models under the two-stage paradigm across multiple challenging benchmarks, while remaining computationally efficient (< 10 RTX3090 GPU hours). We demonstrate the potential of E2E learning in CPath and calls for greater research focus in this area. The code is here.

---

> 💡 **Hao 批注：核心创新与贡献**
>
> 这篇文章的底层逻辑是：当前病理 AI 的主流范式（FM 提取离线特征 + MIL 聚合）存在根本性问题——编码器没有针对下游任务做适配，MIL 和编码器是分离优化的。E2E 学习虽然可以直接解决这个问题，但之前的 E2E 研究把精力花在了如何更好地采样 patch 上（聚类采样、注意力采样），却忽略了一个更关键的问题：**MIL 聚合器中的稀疏注意力在 E2E 训练中会引发优化坍塌**。
>
> 具体来说，稀疏注意力（如经典的 ABMIL）在 E2E 训练中会过度聚焦于某些冗余区域，导致编码器只学到了这些区域的梯度，形成"劣质特征 → 错误注意力 → 劣质梯度 → 更劣质特征"的恶性循环。而之前的方法（TransMIL 等 transformer 方法）虽然通过全局注意力缓解了稀疏性问题，却又被大量冗余 patch 分散了注意力。
>
> ABMILX 的设计思路是：在保留稀疏性（这对病理任务重要）的同时，用两个机制降低优化风险：(1) 多头注意力——不同特征子空间独立投票，降低单头过度聚焦的风险；(2) 全局注意力增强——利用 patch 间的特征相关性，将高注意力 patch 的注意力传播给与其相似的 patch。
>
> 核心结论是：**只需要用 ImageNet 预训练的 ResNet，配合合适的 MIL 设计（ABMILX），在 E2E 训练后就能超越用 100K+ WSI 预训练的 FM**。比如在 PANDA 上，E2E ResNet-50 + ABMILX 达到 78.83% 准确率，而 UNI（ViT-L, 100M patches 预训练）+ ABMIL 只有 74.69%。

---

> 💡 **Hao 批注：与 ReadySlide 项目的关联**
>
> 这篇文章对 ReadySlide 有直接的方法论启示：
>
> 1. **编码器适配的重要性**：ReadySlide 目前使用冻结的 FM (UNI/CHIEF/Virchow2) 提取特征后再做压缩，特征质量受限于 FM 的预训练目标。如果可以在 E2E 框架下 fine-tune 编码器，压缩后的特征可能更适配下游任务。
>
> 2. **MIL 设计的杠杆效应**：本文证明在 E2E 中，MIL aggregator 的设计远比采样策略重要（换 MIL 提升 4.7pp vs 换采样策略提升 0.5pp）。ReadySlide 的 allocator 本质上也是一种实例级调度策略，其设计与 MIL 有类比关系——allocator 的"稀疏性"（选择多少 patch 给多少 bit）同样会影响训练优化。
>
> 3. **计算成本的 trade-off**：E2E Ablation 冻结编码器导致性能急剧下降（Table 4 bottom），说明 E2E 中编码器适配是必要的——这也意味着 ReadySlide 若要走 E2E 训练路线，必须考虑编码器 fine-tuning 的额外成本。

---

> 💡 **Hao 批注：Method 快速索引**
>
> | 模块 | 作用 | 对应章节 |
> |------|------|----------|
> | Multi-scale Random Sampling | 降低计算成本 + 引入多尺度信息 | Sec 3.1 |
> | Multi-Head Local Attention (MHLA) | 从不同特征子空间捕获多样化注意力，打破单头过度稀疏 | Sec 3.2 |
> | Global Attention Plus (A+) | 利用 patch 相关性传播注意力，间接抑制噪声 patch 的高注意力 | Sec 3.2 |
> | 理论分析 | 定义优化风险 R，证明 MHLA 和 A+ 均可降低 R | Appendix A |
