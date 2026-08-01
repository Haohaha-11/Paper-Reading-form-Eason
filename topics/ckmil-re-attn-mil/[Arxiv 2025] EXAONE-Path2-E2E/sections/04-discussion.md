[← 返回 README](../README.md)

# 04 Discussion

## 原文

### 4 Conclusion

We presented EXAONE Path 2.0, a pathology foundation model that learns patch-level representations under direct slide-level supervision. Our approach enables slide-level supervised signals to propagate through all hierarchical stages, allowing end-to-end learning of clinically relevant representations.

Our method addresses computational challenges through hierarchical architecture design, curriculum learning, and memory management techniques including activation checkpointing and CPU offloading. We employ multi-task learning across diverse biomarker prediction tasks and use early exit strategies to mitigate overfitting in small data regimes.

Experimental results show that EXAONE Path 2.0 achieves competitive average performance across 10 biomarker prediction tasks using only 37k WSIs for training, demonstrating improved data efficiency compared to existing foundation models. The model performs consistently across diverse cancer types and prediction targets.

These results demonstrate that direct slide-level supervision can effectively learn clinically relevant features, and our proposed methods successfully address the computational challenges of gigapixel image training, providing a practical approach for pathology foundation models.

---

> 💡 **Hao 批注：论文的整体评价**
>
> **核心贡献**:
> - 证明了 slide-level 监督信号比 SSL 更有效（数据效率维度）
> - 提出了一个可行的 gigapixel E2E 训练框架（curriculum + memory management）
> - 在 biomarker 预测任务上建立了新的 SOTA benchmark
>
> **主要局限**:
>
> 1. **缺少方法消融**: 无法判断 mult-itask、curriculum、early exit 各自的贡献。这是论文最大的学术缺陷。
>
> 2. **缺少特征空间分析**: 没有像 Revisiting-E2E 那样用 UMAP 可视化的特征空间来证明 E2E 训练确实学到了更好的特征。只有 end-task AUROC，无法了解特征的内部结构。
>
> 3. **"SSL 学不到 biomarker 特征" 缺乏直接证据**: 这个论断是论文的出发点，但只通过最终的 AUROC 对比来佐证，没有做 controlled experiment（如用 SSL 特征 + CLAM vs E2E 特征 + CLAM 在同一数据上对比）。
>
> 4. **训练数据不可复现**: 37K WSIs 来自 LG 内部数据（韩国 + 美国合作医院），公众无法复现。只有 6/10 评估 benchmark 是公开的（CPTAC）。
>
> 5. **架构与 HIPT 强耦合**: 不清楚其他架构（如 ResNet-based E2E）是否也能从多任务 slide-level 监督中获益。
>
> 6. **缺少计算成本报告**: 没有说明 37K WSIs 的 E2E 训练需要多少 GPU hours，无法与 GigaPath (3072 A100 hours) 的成本做直接对比。

---

> 💡 **Hao 批注：与 Revisiting-E2E 的交叉读后感**
>
> 将两篇文章放在一起读，可以勾勒出当前病理 E2E 学习的两个关键研究方向：
>
> **方向 A: 训练稳定性** (Revisiting-E2E)
> - 问题：MIL 优化坍塌
> - 方案：改良 MIL 聚合器
> - 适用场景：数据量中等（~1K-10K WSIs），单任务训练
>
> **方向 B: 训练信号质量** (EXAONE Path 2.0)
> - 问题：SSL 学不到生物标记特征
> - 方案：多任务 slide-level 监督 + curriculum
> - 适用场景：数据量较大（~37K WSIs），多任务训练
>
> 两个方向是正交互补的：
> - ABMILX 可以用于 EXAONE Path 2.0 的 Stage-3 聚合替代 plain ViT aggregation
> - 多任务训练的 idea 可以用于 Revisiting-E2E 来提升泛化性
> - Curriculum learning 可以用于 Revisiting-E2E 来渐进式增加采样复杂度
>
> 一个理想的组合可能是：HIPT + multi-task supervision + ABMILX aggregation + MRIS sampling。

---

> 💡 **Hao 批注：对 ReadySlide 项目的启示（详细）**
>
> **可以直接借鉴的**:
> 1. **多任务信号的思路**: 如果 ReadySlide 的目标是"压缩后对多个 FM 和多个任务通用"，那么多任务 loss（不同 FM 分类结果 + 不同下游任务）本身就是天然的多任务训练信号
> 2. **Early exit 用于下游**: 就像 EXAONE P2 用 Stage-1 + CLAM 做下游一样，ReadySlide 的压缩特征可以直接接轻量分类器而不需要重新提取
> 3. **Curricum 渐进**: 从低压缩率到高压缩率的渐进训练（而不是直接训 100× 压缩）可能更稳定
>
> **不能直接借鉴的**:
> 1. HIPT 架构太重——ReadySlide 使用冻结的 FM 提取特征，不需要三层 ViT
> 2. 37K WSIs 的数据需求——ReadySlide 目前只在 TCGA (~1K) 和 PANDA (~10K) 上实验，多任务训练可能需要更多数据
> 3. Biomarker 任务可能与 ReadySlide 的压缩目标不同（压缩关注的是特征重建或下游性能保持，而非分子预测）
>
> **最大的启发**: 两篇文章共同证明了一个方向——E2E 训练让编码器适配下游任务比单纯 scale up 预训练数据/模型更有效。对于 ReadySlide，这意味着如果在压缩框架中实现 E2E 训练（压缩 + 下游任务联合优化），可能比只做 frozen feature 压缩获得更好的性能。

---

> 💡 **Hao 批注：论文信息密度评估**
>
> EXAONE Path 2.0 是一篇典型的"工业界论文"：工程贡献大于学术贡献，结果导向强于方法分析。
>
> | 维度 | 评分 | 说明 |
> |------|------|------|
> | 新颖性 | 中等 | E2E 训练 + curriculum + MTL 的组合之前没人做，但每个单独组件都不新 |
> | 技术深度 | 较浅 | 方法描述简洁，缺少理论分析和详细消融 |
> | 实验充分性 | 中等 | 10 benchmark 覆盖面广，但缺少方法本身的消融 |
> | 实用价值 | 高 | 37K WSIs 训练成本可接受，early exit 便于部署 |
> | 可复现性 | 低 | 训练数据不可公开，架构细节不充分 |
>
> 相比之下，Revisiting-E2E 是更标准的学术论文：有理论分析、有详细消融、有特征可视化、有反例（RRTMIL 的优化坍塌）。
