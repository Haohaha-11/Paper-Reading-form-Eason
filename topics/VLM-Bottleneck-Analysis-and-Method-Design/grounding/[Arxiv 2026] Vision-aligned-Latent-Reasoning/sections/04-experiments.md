[← 返回 README](../README.md)

# 4. Experiments

## 一、Preview

本章通过五个研究问题组织实验：(1) VaLR 是否能提升 VQA benchmark 性能？(2) VaLR 在长上下文推理中是否保持性能？(3) Latent token 组件是否真正贡献？(4) VaLR 能否模型无关地适配多种视觉模型和任务？实验设计从粗到细，从主表到消融，从单编码器到多编码器，层层深入。

---

## 二、原始文本

We provide an empirical evaluation of Vision-aligned Latent Reasoing (VaLR) by investigating following questions:

- Does VaLR improve the performance on VQA datasets? (Table 1, Table 2)
- Does VaLR retain performance during long-context reasoing? (Figure 2)
- Does the latent token component really contribute to long-context reasoing? (Table 3)
- Can VaLR be adapted to various vision models and tasks in a model-agnostic manner? (Table 4, Table 5)

> 💡 **实验设计逻辑**: 四个研究问题构成了一个完整的验证链条：有效性（Q1）→ 长上下文特性（Q2）→ 组件贡献（Q3）→ 通用性（Q4）。这种从整体到组件、从特定到通用的实验设计非常经典。

### 4.1. Experimental Setup

**Training Setup.** For the main experiment, we trained VaLR on Qwen2.5-VL-7B (Bai et al., 2025). Unless mentioned otherwise, we use DINOv3 (Simeoni et al., 2025) as the aligning vision encoder. For the analysis and multi-encoder alignment training setting, we additionally consider alternative vision encoders, e.g., DINOv2 (Oquab et al., 2023), CLIP (Radford et al., 2021), SigLIPv2 (Tschannen et al., 2025), and π³ (Wang et al., 2025d). We perform training on 450K scale of Chain-of-Thought (CoT) dataset for both training stages. We construct the dataset with the mixture of several open-source datasets, e.g., Zebra-CoT, CogCoM, ReFocus, Visual-CoT, OneThinker-SFT, and GCoT.

**Evaluation Setup.** Following the evaluation setup of previous benchmarks, we mainly report the accuracy (%) across all benchmarks. For response generation, we apply greedy sampling. The models are evaluated on various VQA benchmarks, including VSI-Bench, BLINK, MMVP, MMStar, MathVision, and more.

**Baselines.** We compare VaLR with API, reasoing, supervised finetuned, and latent reasoing models in MLLMs, namely, GPT-4o, Claude-4-Sonnet, R1-OneVision-7B, Ocean-R1-7B, LVT, CoVT, and Monet.

### 4.2. 3D Spatial Reasoing Tasks

We examine VaLR's effectiveness in long-context reasoing by comparing performance on 3D multi-view benchmark, VSI-Bench (Yang et al., 2025b), which requires long-context reasoing ability to integrate spatial information across multiple viewpoints. We report accuracy on 8 sub-tasks and the average accuracy. We train the VaLR with two different setups: (i) VaLR-S, which aligns a single encoder using DINOv3 and (ii) VaLR-M, which aligns multiple encoders using DINOv3, SigLIPv2, and π³.

**Results.** As shown in Table 1, VaLR-S achieves an average accuracy of 41.5%, substantially outperforming the base model, Qwen2.5-VL (33.0%). In contrast, previous latent reasoing methods struggle on this benchmark requiring multi-view understanding. For example, Monet reaches 14% in average accuracy, and other models (LVR, CoVT) also collapse (see more details in Appendix C.1). This performance gap between VaLR-S and other latent reasoing methods provides strong evidence that latent reasoing without visual recall fails to maintain visual grounding during long reasoing traces, confirming the effectiveness of dynamic visual re-injection.

> 💡 **Table 1 深度分析 — VSI-Bench 结果**:
>
> | 方法组 | 代表 | Avg. Accuracy | 关键特征 |
> |--------|------|--------------|---------|
> | API Models | GPT-4o | 34.0% | 大模型但非专用 |
> | Reasoing Models | Ocean-R1-7B | 30.5% | RL 训练推理，但无视觉检查点 |
> | Base Model | Qwen2.5-VL-7B | 33.0% | 标准 MLLM |
> | Latent Reasoing | Monet / LVR / CoVT | 14.0%-18.6% | **在多视图上崩溃** |
> | VaLR-S (Ours) | 单编码器 DINOv3 | 41.5% | +8.5%p over base |
> | VaLR-M (Ours) | 多编码器 | **52.9%** | +19.9%p over base! |
>
> Monet/LVR/CoVT 在 VSI-Bench 上的崩溃是最有力的反面证据：**这些方法仅支持单视图或对多视图扩展有限，在需要长期视觉记忆的多视图任务上，缺乏动态视觉重注入机制的潜推理必然失败**。这不仅是 VaLR 的性能优势证明，更是对 latent reasoing 设计空间的重要启示：**潜推理本身不够，必须在推理过程中持续保持视觉信息**。

In addition, VaLR-M even achieves state-of-the-art performance (52.9%) over previous baselines, e.g., GPT-4o (34.0%) and Ocean-R1 (30.5%), highlighting that the combination of different vision encoders produces the synergistic effect. In particular, VaLR-M achieves remarkable performance on spatial understanding tasks, such as relative (50.0%) and absolute (40.6%) distance prediction. These results validate our hypothesis that latent reasoing with visual alignment prevents the visual information decay observed in standard reasoing approaches.

> 💡 **VaLR-M 的子任务分析**: 在 8 个子任务中，VaLR-M 在 Object Count（66.4%）、Object Size（64.2%）、Room Size（56.6%）、Relative Direction（51.8%）上表现尤为突出。这些子任务高度依赖精确的空间感知，恰好验证了 π³ 编码 3D 空间结构的价值。相比之下，Route Plan（35.1%）提升相对有限，可能是因为路径规划还需要时序推理能力，超出了纯视觉对齐的范围。

### 4.3. Perception Tasks

We further present that VaLR also improves performance on moderate-length reasoing tasks beyond long-context by evaluating it on five perception benchmarks. We report accuracy on BLINK, MMVP, MMStar, V*, and CVBench.

**Results.** As shown in Table 2, VaLR achieves substantial improvements over the base model. These results reveal that the learned visual grounding capability generalizes to improve short-context perception as well. The consistent advantages over other latent reasoing methods are particularly informative: VaLR-M outperforms CoVT by 8.7%p on BLINK and 8.9%p on V*, and significantly surpasses Monet and LVR across all benchmarks. Interestingly, reasoing models such as R1-OneVision and Ocean-R1 show inconsistent results, with Ocean-R1 achieving strong V* performance (78.0%) while underperforming on BLINK (56.8%) and MMVP (58.0%), suggesting that their reasoing enhancement overfit to specific task patterns rather than developing robust visual understanding. In contrast, VaLR's consistent improvements across diverse perception tasks validate that our visual alignment strategy during latent reasoing provides a general mechanism for maintaining high-quality visual representations throughout the reasoing process, regardless of reasoing length.

> 💡 **Table 2 深度分析 — 感知 Benchmark**:
>
> | 方法 | BLINK | MMVP | MMStar | V* | CVBench |
> |------|-------|------|--------|-----|---------|
> | GPT-4o | 63.0 | 68.7 | 65.2 | 42.9 | 79.2 |
> | Claude-4-Sonnet | 39.6 | 48.7 | 58.8 | 15.2 | 76.3 |
> | Ocean-R1-7B | 56.8 | 58.0 | 62.6 | 78.0 | 78.1 |
> | Qwen2.5-VL-7B | 55.7 | 56.0 | 67.1 | 76.4 | 74.5 |
> | VaLR-S (Ours) | 63.1 | 60.3 | 70.8 | **86.4** | 83.1 |
> | VaLR-M (Ours) | **64.7** | **60.3** | **72.3** | **86.9** | **87.6** |
>
> 关键观察：(1) VaLR 在所有 5 个 benchmark 上都超越所有开源 baseline；(2) Ocean-R1 表现不一致（V* 78% vs BLINK 56.8%），说明 RL 推理训练可能导致过拟合；(3) VaLR 的跨任务一致提升说明视觉对齐提供的是通用的视觉表征改善机制，而非针对特定任务的优化。

### 4.4. Reasoing Length Analysis

To investigate whether VaLR follows the test-time scaling law, we analyze the performance as a function of reasoing length. We consider Ocean-R1 and LVR as baselines and evaluate on MathVista, MathVision, MMhalu, and MMVP, grouping samples by generated reasoing length to observe performance trends.

**Results.** As illustrated in Figure 2, while all baseline methods peak at intermediate reasoing lengths and subsequently degrade, VaLR shows monotonic improvement across all benchmarks. In particular, on MMVP, VaLR sustains strong performance across all reasoing lengths while Ocean-R1 dramatically collapses from 62.7% to 56.5% at 300 tokens. This divergent behavior provides compelling evidence that models trained for language reasoing or naive latent reasoing progressively lose visual priors as they generate longer reasoing chains. These results validate that VaLR successfully maintain visual grounding during extended reasoing, enabling the model to benefit from longer thinking time rather than suffer from it. We thereby achieve true test-time scaling in the multi-modal domain as widely demonstrated for language models.

> 💡 **Figure 2 深入解读 — Test-time Scaling Law 的证据**:
>
> 四个 benchmark 的推理长度分析是本文最重要的实验证据：
> 1. **MMhalu（幻觉率）**: VaLR 幻觉率随推理长度下降（lower is better），Ocean-R1 和 LVR 先降后升
> 2. **MathVista**: VaLR 单调提升，Ocean-R1 在 ~150 tokens 后下降
> 3. **MathVision**: VaLR 持续上升，Ocean-R1 和 LVR 在 ~120 tokens 后衰减
> 4. **MMVP**: VaLR 平稳或微升，Ocean-R1 从 62.7% 暴跌到 56.5%
>
> 这是一个**因果性证据**：如果视觉信号衰减是问题，那么推理越长应该越差；如果 VaLR 的视觉检查点有效，那么推理越长应该不变或更好。实验结果完美符合这个因果预测。特别值得注意的是，Ocean-R1 是经过 RL 训练的推理增强模型，即便如此也无法抵御视觉衰减——说明**仅靠强化文本推理能力无法解决视觉 grounding 问题**。

### 4.5. Ablation Study and Analysis

**Effect of representation alignment.** To verify the contribution of representation alignment (REPA) to VaLR's performance, we conduct ablation studies to test if VaLR functions effectively without external vision encoders and REPA. Specifically, during training, we replaced DINOv3 with Qwen's native vision encoder (VaLR w/QE) and also trained VaLR without REPA (VaLR w/o VA). As shown in Table 3, VaLR trained with Qwen's native encoder still consistently outperforms other baselines even without external alignment. These results indicate that VaLR is not reliant on external vision encoders, while incorporating them further enhances the performance.

> 💡 **Table 3 消融分析 — REPA 的贡献**:
>
> | 方法 | VSI-Bench | BLINK | V* |
> |------|-----------|-------|-----|
> | Qwen2.5-VL-7B | 33.0 | 55.7 | 76.4 |
> | + vanilla SFT | 33.7 | 56.6 | 78.0 |
> | + VaLR w/o VA | 34.0 | 57.1 | 75.9 |
> | + VaLR w/ QE | 39.6 | 58.9 | 81.7 |
> | + VaLR (DINOv3) | **41.5** | **63.1** | **86.4** |
>
> 三层消融揭示了清晰的信号：(1) VaLR w/o VA（仅 latent tokens，无对齐）几乎无提升——说明 **latent tokens 机制本身不够，必须配合视觉对齐**；(2) VaLR w/ QE（使用 MLLM 原生编码器对齐）已有显著提升（+6.6%p）——说明 VaLR 框架不依赖外部编码器；(3) VaLR w/ DINOv3 进一步提升（+8.5%p）——说明更强大的视觉编码器确实提供了更丰富的对齐信号。

**Alignment to different vision encoders.** We further analyze whether VaLR can extend to other vision encoders for representation alignment, not limited to DINOv3. As shown in Table 5, VaLR consistently outperforms the base model regardless of the vision encoder choice. We observe that VaLR consistently improves performance in a encoder-agnostic manner, and yields larger gains when paired with stronger vision encoders such as DINOv3.

> 💡 **Table 5 编码器对比**:
>
> | 编码器 | BLINK | MMVP | MMStar | V* | CVBench |
> |--------|-------|------|--------|-----|---------|
> | Base (无对齐) | 55.7 | 56.0 | 67.1 | 76.4 | 74.5 |
> | + CLIP | 62.3 | 59.3 | 71.0 | 83.2 | 79.1 |
> | + SigLIPv2 | 62.8 | 59.7 | 71.3 | 83.2 | 81.9 |
> | + DINOv2 | 62.7 | 60.0 | 70.7 | 83.8 | 81.8 |
> | + DINOv3 | **63.1** | **60.3** | **70.8** | **86.4** | **83.1** |
>
> 所有编码器均有效（最低 +6.6%p on BLINK），DINOv3 最佳。值得注意的是 CLIP 和 SigLIPv2 性能接近，DINOv2/v3 在 V* 和 CVBench 上明显更强——这些 benchmark 更依赖精细视觉感知（如 V* 需要定位小目标），恰好符合 DINO 系列擅长细粒度特征的特点。

**Multi-encoder analysis.** As shown in Table 4, incorporating additional encoders consistently leads to performance gains. Notably, these improvements are closely aligned with the specific characteristics of each encoder's representation. In detail, integrating the 3D-specialized encoder, π³, significantly improves results on the 3D multi-view benchmark, VSI-Bench. Moreover, adding 2D encoders, such as DINOv3 or SigLIPv2, enhances the performance across several perception benchmarks. Finally, integrating all three encoders achieves the best performance across all tasks. These results indicate that VaLR successfully aligns with distinct encoder representations by effectively leveraging their domain-specific strengths.

> 💡 **Table 4 多编码器消融解读**:
>
> | π³ | DINOv3 | SigLIPv2 | VSI-Bench | BLINK | V* |
> |-----|--------|---------|-----------|-------|-----|
> | X | X | X | 33.0 | 55.7 | 76.4 |
> | V | V | X | 52.4 | 64.6 | 85.8 |
> | V | X | V | 50.5 | 63.8 | 85.3 |
> | X | V | V | 41.9 | 62.5 | 86.9 |
> | V | V | V | **52.9** | **64.7** | **86.9** |
>
> π³ 的贡献在 VSI-Bench 上非常明显：π³ + DINOv3 (52.4%) vs DINOv3 + SigLIPv2 (41.9%)，差距 10.5%p。相比之下，在 2D 感知 benchmark BLINK 上，π³ 的边际贡献较小（64.6% vs 64.7%）。这验证了各编码器的专业化特性确实被 VaLR 有效利用。

**Alignment layer analysis.** We investigate which intermediate layer of MLLMs is most effective for alignment via REPA. Specifically, we vary the layer index across three settings — Front (4th), Middle (12th), and Last (27th). As shown in Table 6, while all settings improve performance, REPA applied at the middle layer achieves the strongest results. This observation is consistent with prior studies indicating that visual information is most prominently represented in the middle layers of MLLMs.

**Data Scalability.** We investigate the data scalability of VaLR by tracking the performance of the checkpoints trained on different numbers of samples, e.g., 10K, 50K, 100K, 200K, and 450K. As shown in Figure 3, both VaLR variants consistently improve performance as the sample size grows, while Vanilla-SFT saturates beyond 200K samples. Notably, our best model VaLR-M achieves >20x faster training to reach comparable performance on V*. This suggests that aligning with encoders facilitates learning richer features from the training data and improves data scalability.

> 💡 **Figure 3 数据可扩展性分析**: VaLR-M 在 V* 上仅需约 10K（450K/20≈22.5K）样本即可达到 vanilla-SFT 450K 样本的性能——这意味着 **REPA 对齐使每个训练样本的"信息密度"大幅提升**。视觉编码器提供的结构化视觉知识作为额外的监督信号，使模型从同样的 CoT 数据中学习到更多视觉相关信息。这是一个重要的训练效率发现。

---

## 三、Summary

- **3D 空间推理 (VSI-Bench)** : VaLR-M 达到 52.9%（+19.9%p over Qwen2.5-VL），Monet/LVR/CoVT 均崩溃，证明动态视觉重注入对于长期视觉记忆是不可或缺的。
- **感知任务** : 5 个 benchmark 全面超越所有开源 baseline，Ocean-R1 等 RL 推理模型表现不一致（过拟合迹象），VaLR 提供通用视觉改善机制。
- **Test-time Scaling** : 唯一在 4 个 benchmark 上随推理长度单调提升的方法。Ocean-R1 在 MMVP 上从 62.7% 暴跌至 56.5%。
- **消融** : (1) REPA 对齐是关键，纯 latent tokens 几乎无效；(2) 所有编码器均有效，DINOv3 最佳；(3) 中间层对齐最优；(4) π³ 对 VSI-Bench 贡献最大；(5) 数据效率 >20x 提升。
