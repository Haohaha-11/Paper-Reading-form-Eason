[← 返回 README](../README.md)

# 02 Method

## 原文

### 2.1 Overcoming the Prohibitive Computational Costs of Gigapixel Image Training

Training on gigapixel whole-slide images presents significant computational challenges due to memory constraints and processing requirements. To address these limitations, we employ a combination of hierarchical architecture design, curriculum learning, and efficient memory management techniques.

**Architecture Design.** We adopt a three-stage Hierarchical Image Pyramid Transformer (HIPT) [2] architecture. This hierarchical design reduces computational complexity by processing patches at progressively higher levels of abstraction rather than directly processing gigapixel images at full resolution, enabling more efficient handling of large-scale WSIs. The first-stage ViT processes individual patches, the second-stage ViT aggregates patch-level features into region-level representations, and the third-stage ViT processes the entire slide by integrating all region-level features.

**Curriculum Learning.** To manage the computational burden of end-to-end training across all stages simultaneously, we implement a two-stage curriculum learning approach with progressive resolution scaling. In the first curriculum stage, we apply 256x256 DINO loss to the first-stage ViT and 1024x1024 DINO loss to the second-stage ViT, establishing hierarchical visual representations without requiring full three-stage end-to-end computation. In the next curriculum stage, we continue applying 256x256 DINO loss to the first-stage ViT while scaling up to 4096x4096 regions for the second-stage ViT, and introduce slide-level supervised cross-entropy loss to propagate gradients into the entire three-stage model processing the full slide. This curriculum approach significantly reduces computational overhead by avoiding the need to process all stages at maximum resolution during every training iteration.

**Memory Management.** To further manage the computational demands of processing entire WSIs, we employ activation checkpointing and CPU offloading strategies. Rather than loading all patch embeddings into GPU memory at once, we dynamically compute and transfer activations as needed during supervised loss calculation. This approach significantly reduces memory requirements while maintaining training efficiency, enabling us to process gigapixel images with limited computational resources.

### 2.2 Learning Generalizable Representations across Multiple Biomarker Prediction Tasks

To learn representations that generalize across diverse biomarker prediction tasks while maintaining computational efficiency, we employ a multi-task learning framework combined with an early exit strategy for downstream task adaptation.

**Multi-Task Learning Framework.** We implement a multi-task learning approach that jointly optimizes across multiple complementary objectives. Our training encompasses three primary categories of tasks: (1) cancer subtyping across 33 cancer types, (2) tissue type classification across 12 organ systems, and (3) molecular biomarker prediction including pan-cancer and cancer-specific mutation status, microsatellite instability, and hormone receptor subtyping. This multi-task learning strategy jointly optimizes for these diverse prediction objectives, encouraging the model to learn shared representations that capture fundamental pathological patterns across different scales of biological organization. The joint optimization helps prevent overfitting to individual tasks while improving generalization across the entire spectrum of downstream applications.

**Early Exit Strategy for Downstream Adaptation.** To further mitigate overfitting in the small data and deep network regime, we adopt a shallow network approach that leverages early representations rather than the full hierarchical model [6]. Specifically, we leverage the first-stage model in combination with Clustering-constrained Attention Multiple Instance Learning (CLAM) [8] for downstream task adaptation. Rather than fine-tuning the entire hierarchical network, this early exit approach uses the robust patch-level features from the first-stage model, while CLAM efficiently aggregates these features for slide-level predictions. This strategy significantly reduces computational overhead during downstream task adaptation while avoiding the pitfalls of overfitting commonly observed in pathology applications with limited data.

---

> 💡 **Hao 批注：架构全貌**
>
> EXAONE Path 2.0 的训练分为两个阶段：
>
> ```mermaid
> flowchart TD
>     subgraph Phase1["Phase 1: Curriculum Stage 1 (SSL Warmup)"]
>         A1[256×256 Patch] --> B1[ViT Stage-1<br>DINO Loss]
>         B1 --> C1[Patch Tokens]
>         C1 --> D1[ViT Stage-2<br>1024×1024 Region<br>DINO Loss]
>     end
>     
>     subgraph Phase2["Phase 2: Curriculum Stage 2 (E2E Supervision)"]
>         A2[256×256 Patch] --> B2[ViT Stage-1<br>DINO Loss retained]
>         B2 --> C2[Patch Tokens]
>         C2 --> D2[ViT Stage-2<br>4096×4096 Region]
>         D2 --> E2[ViT Stage-3<br>Full Slide]
>         E2 --> F2[Multi-Task Heads<br>33 cancer + 12 organ + biomarkers]
>         F2 --> G2[Slide-level CE Loss]
>         G2 -.梯度回传.-> B2
>     end
>     
>     subgraph Phase3["Phase 3: Downstream (Early Exit)"]
>         B2 --> H3[CLAM Aggregator<br>on Stage-1 features]
>         H3 --> I3[Task-specific Prediction]
>     end
>     
>     style G2 fill:#f96,stroke:#333,stroke-width:2px
>     style H3 fill:#bbf,stroke:#333,stroke-width:2px
> ```
>
> **关键设计决策解读**:
>
> 1. **为什么 Stage 1 用 DINO SSL 而不是直接监督？**
>    - 渐进式训练：先用 SSL 建立基本的视觉表示（区分组织纹理/细胞形态），再引入监督信号学习临床相关特征
>    - 类似 warmup：避免随机初始化的编码器在 E2E 训练初期就产生极差的梯度
>
> 2. **为什么 Stage 2 仍保留 Stage-1 的 DINO Loss？**
>    - 防止灾难性遗忘：保留 SSL 学到的通用视觉特征
>    - 作为正则化：阻止 Stage-1 过度适配 slide-level 标签而丢失局部纹理表示能力
>
> 3. **为什么下游只用 Stage-1 特征？(Early Exit)**
>    - 避免过拟合：全三层 HIPT 参数量大，下游任务数据少（某些 biomarker 任务仅 ~100 训练样本）
>    - 推理效率：不需要运行 Stage-2 和 Stage-3
>    - 类似迁移学习中的 "feature extractor + simple classifier" 模式

---

> 💡 **Hao 批注：多任务学习的设计空间**
>
> 文章的三类训练任务有清晰的层次：
>
> | 任务类别 | 粒度 | 实例 | 作用 |
> |----------|------|------|------|
> | 癌种分型 (33 types) | 粗粒度 | LUAD vs LUSC vs BRCA... | 学习组织来源和癌种特征 |
> | 器官分类 (12 organ systems) | 粗粒度 | Lung/Breast/Colon... | 学习器官层面形态学特征 |
> | 分子 biomarker | 细粒度 | EGFR/KRAS/TP53 mutation... | 学习亚视觉形态学变化 |
>
> 这种层次化多任务设计的优势：
> - 粗粒度任务提供稳定的梯度信号（类别多，样本相对均衡）
> - 细粒度任务可能数据不平衡，但粗粒度任务的共享表示能帮助学习
> - 所有任务共享 Stage-1/Stage-2 编码器 → 鼓励学习通用的病理形态学表示
>
> 但文章没有提供多任务消融（如 ablation 不同任务类别的贡献），这是一个重要缺失。

---

> 💡 **Hao 批注：与 Revisiting-E2E 方法的关键差异**
>
> | 设计维度 | EXAONE Path 2.0 | Revisiting-E2E |
> |----------|-----------------|---------------|
> | 编码器架构 | HIPT 三层 ViT | ResNet (CNN) |
> | 训练策略 | Curriculum (SSL→Supervised) | 直接全监督 |
> | MTL | 多任务 (33+12+biomarker) | 单任务 |
> | 自监督留存 | Stage-1 保留 DINO Loss | 无 |
> | 下游适配 | Early exit (Stage-1 + CLAM) | 全模型推理 |
> | 数据处理 | 全 WSI 处理（经过分层抽象） | 随机采样 patch（MRIS） |
>
> **互补性分析**:
> - EXAONE Path 2.0 没有讨论 MIL 设计的问题（直接用已有的 CLAM），如果替换为 ABMILX 可能进一步提升
> - Revisiting-E2E 没有讨论多任务学习的收益，如果用多任务监督训练可能进一步提升泛化性
> - 两者的技术路线是正交的：一个是"用什么信号训"，一个是"怎么让训练稳定"

---

> 💡 **Hao 批注：Memory Management 的工程细节**
>
> 文章提到 activation checkpointing + CPU offloading，但没有给出具体的显存占用数据（仅在首段提到"limited computational resources"）。这是工程贡献不够量化的体现。
>
> 对于 ReadySlide 的启示：如果要做 E2E 训练（编码器 + 压缩 + 下游任务），GPU 显存将是主要瓶颈。activation checkpointing（在前向时只保留部分 activations，反向时重新计算）和 CPU offloading（将不活跃的 tensor 移到 CPU）是两个可以直接采用的技巧。
