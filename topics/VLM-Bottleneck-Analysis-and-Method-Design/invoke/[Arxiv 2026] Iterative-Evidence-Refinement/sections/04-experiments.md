[← 返回 README](../README.md)

# 4. Experiments

## 一、Preview

实验部分涵盖：
- 4.1 Experiment Setup: COCO 1.5k 训练 + 8 个 benchmark + 4 组 baseline
- 4.2 Main Results: Table 1 (V\*, HR-Bench) + Table 2 (感知/推理/幻觉 6 个 benchmark) — SIEVE 一致提升
- 4.3 Visualization: 定位区域与语义一致性的定性展示
- 4.4 Ablation: 三个消融实验 — embedding 选择质量、中间层的 IHR 验证、action reward 对训练稳定性的影响

---

## 二、原始文本

### 4.1 Experimental Setup

#### 4.1.1 Benchmarks and Baselines

For training, we sample 1,500 images from COCO 2017 Lin et al. [2014] and construct the corresponding training data with pre-extracted region embeddings. For evaluation, we focus on two challenging high-resolution visual reasoning benchmarks: V\*Bench Wu and Xie [2024] and HR-Bench Wang et al. [2024], reporting results at both 4K and 8K resolutions. To assess generalization beyond high-resolution reasoning, we additionally evaluate on perception benchmarks MME-Real-Lite Zhang et al. [2024b] and RealWorldQ xAI; multimodal reasoning benchmarks MathVista Lu et al. [2024], LogicVista Xiao et al. [2024], and WeMath Qiao et al. [2025]; and the hallucination benchmark Hallusion Wu et al. [2024]. We compare SIEVE against representative zoom-and-refine baselines, including DyFo Li et al. [2025e], ZoomEye Shen et al. [2024], and Zoom-Refine Yu et al. [2025c], as well as a vanilla GRPO-trained model optimized solely with format and accuracy rewards.

> 💡 **实验配置解读 — 训练数据与评估的隔离设计**:
>
> | 维度 | 详情 |
> |------|------|
> | 训练数据 | COCO 2017 (natural images)，1500 张 |
> | 训练数据与评估数据的重叠 | 无——COCO 是自然图像，V\* 和 HR-Bench 是高分辨率细节图像 |
> | 评估 Benchmark | 8 个，覆盖 4 个维度 |
> | 高分辨率推理 | V\* Bench (Attribute + Spatial), HR-Bench (4K/8K, FSP + FCP) |
> | 感知 | MME-Real-Lite, RealWorldQA |
> | 多模态推理 | MathVista, LogicVista, WeMath |
> | 幻觉 | HallusionBench |
>
> **一个值得注意的设计**: 训练只用了 1500 张 COCO 图像，但评估跨越了多个领域。这意味着 SIEVE 学习到的"何时回访视觉证据"的能力是**task-agnostic** 的——不是记住 "对某个特定 task 要回访"，而是学习了一个通用的 evidence retrieval policy。这可能是 SIEVE 数据效率高的深层原因：它训练的是一种元能力 (meta-skill) 而非 task-specific pattern。

#### 4.1.2 Training Details

We adopt Qwen3-VL-4B-Instruct Yang et al. [2025a] and Qwen3-VL-8B-Instruct as base models. Both are trained with GRPO Shao et al. [2024b] for 60 rollout steps on two NVIDIA H200 GPUs. Each rollout batch contains 16 prompts with 8 rollouts per prompt. We set the KL divergence coefficient to 0.0 and the maximum response length is set as 8,192 tokens.

> 💡 **GRPO 训练配置 — 关键参数解读**:
> - KL coefficient = 0.0: 意味着完全不限制 policy 偏离 reference model。这是一个有趣的设置——通常 GRPO 会保留一个小 KL penalty 防止 reward hacking，但 SIEVE 将其设为 0。可能因为 (1) reward 函数已经足够约束（四维 reward 覆盖了各种行为）；(2) 模型需要相对大的策略变化来学习 evidence insertion 行为。
> - 16 prompts × 8 rollouts = 128 rollouts/batch，60 steps 意味着总共约 7680 个 rollout trajectories。
> - 训练数据仅 1500 个样本，但每步 16 prompts，意味着每个样本可能被重复采样多次。

### 4.2 Main Results

In Table 1, we report the performance of SIEVE on V\* and HRBench, comparing it with other models and methods at both the 4B and 8B scales. As shown in the table, SIEVE consistently outperforms all baselines on both benchmarks across both model sizes. Notably, the 4B variant of SIEVE achieves a 10.06% improvement over the corresponding vanilla model. This result indicates that enabling the model to reaso with hidden-state embeddings can effectively enhance performance on high-resolution tasks. We further validate our approach on additional datasets. As presented in Table 2, SIEVE demonstrates consistent improvements over both the vanilla model and the GRPO trained model across perception tasks, reasoning tasks, and hallucination benchmarks.

（Table 1 和 Table 2 见原文，数据摘录如下）

> 💡 **Table 1 批读 — V\* 和 HR-Bench 的主实验结果**:
>
> **4B 模型的关键发现**:
> | 子任务 | Vanilla | SIEVE | Δ |
> |--------|---------|-------|---|
> | V\* Attribute | 74.78 | 81.74 | +6.96 |
> | V\* Spatial | 82.89 | 92.11 | +9.22 |
> | V\* Overall | 78.01 | 85.86 | +7.85 |
> | HR-Bench 4K Overall | 77.75 | 81.25 | +3.50 |
> | HR-Bench 8K Overall | 72.38 | 76.13 | +3.75 |
>
> **8B 模型的关键发现**:
> | 子任务 | Vanilla | SIEVE | Δ |
> |--------|---------|-------|---|
> | V\* Overall | 82.72 | 87.96 | +5.24 |
> | HR-Bench 4K Overall | 79.00 | 81.50 | +2.50 |
> | HR-Bench 8K Overall | 74.25 | 78.25 | +4.00 |
>
> **关键分析**:
> 1. **4B 的提升幅度大于 8B**: 这说明 SIEVE 对小模型的边际收益更大——小模型本身的视觉能力更弱，更需要 evidence 辅助。8B 模型本身已经有更强的全局理解能力，evidence 的增量价值相对小。
> 2. **V\* Spatial 提升最大** (+9.22% for 4B): Spatial reasoning（空间关系推理）是所有子任务中受益最大的——这很符合直觉，因为空间关系需要定位特定区域并推理相对位置，region embedding 直接提供了这种定位信息。
> 3. **HR-Bench 上 4B 提升比 8B 更大**: 4B +2.0~5.0%，8B +1.0~2.75%。同样符合"小模型受益更大"的模式。
> 4. **与工具增强 baselines 对比**: SIEVE 在大部分子任务上优于或持平 ZoomEye/ZoomRefine，但注意 ZoomEye 在 4B 的 V\* 上达到了 90.05（高于 SIEVE 的 85.86）——工具增强方法在某些任务上仍有优势，但 SIEVE 以更低的推理成本实现了 close 的性能。
>
> **批判性思考**: 8B 上 ZoomEye 仅 78.35（低于 Vanilla 82.72）——这是一个异常值，说明工具增强方法的效果非常依赖具体实现和模型兼容性。SIEVE 的稳定性更好，因为它的 evidence 机制是 "内生" 的，不依赖外部工具的 API 兼容性。

> 💡 **Table 2 批读 — 感知/推理/幻觉 benchmark 的扩展验证**:
>
> **最显著的提升**:
> - MME-Real-Lite 4B: +17.58% (from 45.96 → 54.04)
> - MME-Real-Lite 8B: +19.30% (from 47.26 → 56.38)
> - WeMath 8B: +20.65% (from 54.71 → 66.01)
> - LogicVista 4B: +13.71% (from 43.08 → 48.99)
>
> **关键分析**:
> 1. **MME-Real-Lite 的巨大提升**: MME-Real-Lite 是高分辨率真实场景感知 benchmark，提升最大——这与 V\* 和 HR-Bench 的结果一致：SIEVE 对细粒度视觉感知任务帮助最大。
> 2. **WeMath 8B 的异常提升**: +20.65% 是单个 benchmark 的最大提升。WeMath 是数学推理 benchmark，其中很多问题需要从图表/几何图形中提取精确数值——region embedding 提供了精确的局部视觉信息。
> 3. **HallusionBench 的小幅但一致提升**: +2.17% (4B) / +7.96% (8B) —— SIEVE 对幻觉也有缓解效果，因为 evidence 提供了 ground-truth 视觉信号来纠正模型的错误推测。
> 4. **RealWorldQA 8B 无提升** (69.28 vs 69.28): 这是一个有趣的 null result——8B 在 RealWorldQA 上可能已经接近上限，或者 RealWorldQA 的问题类型不需要细粒度视觉回访（更多依赖常识推理）。
>
> **GRPO baseline 的贡献**: 对比 Vanilla → GRPO → SIEVE 的渐进提升，可以看出 GRPO 本身已经带来了部分提升（如 V\* 4B: 78.01 → 85.34 (+7.33)），SIEVE 在此基础上进一步提升到 85.86 (+0.52)。但 SIEVE 的最大增量来自于 evidence 机制在 GRPO 训练中的协同作用——GRPO 让模型学会更好地利用 evidence。

### 4.3 Visualization and Analysis

In Figure 4, using the V\* dataset as a case study, we demonstrate how our model retrieves bounding boxes in image coordinate space that align with the learned region embeddings. Specifically, the selected embeddings are mapped back to their corresponding spatial patch locations and aggregated to form coherent bounding regions. The resulting visualizations show that these extracted embeddings consistently correspond to semantically meaningful and task-relevant image regions, rather than arbitrary or background areas. Although minor localization drift may occur due to the patch-based segmentation mechanism in Qwen-VL where object boundaries may not perfectly align with fixed patch grids our extended patching strategy mitigates this issue. By explicitly injecting the target object's embedding as a structured guidance signal during inference, the model is encouraged to focus on spatially relevant regions and refine its reasoning accordingly. This design not only improves visual grounding fidelity but also provides an intuitive explanation for the consistent performance gains achieved by our method across benchmarks.

（Figure 4 包含多组可视化示例，每个例子展示 green box (matched region) + red box (expanded region) 及其 zoomed view）

> 💡 **Figure 4 批读 — 可视化中的关键发现**:
> - **Green box vs Red box**: Green 是 matched block (TopK=1 选中的最高分 block)，Red 是 expand 后的完整区域。两者之间的 gap 展示了 patch grid 与 object boundary 的不对齐——expand 操作确实在补全边界。
> - **对象的多样性**: 可视化覆盖了 glove, scooter, stool, bucket, bicycle, clock, motorcycle 等多种对象类别——说明 evidence discovery 机制是 object-agnostic 的，不限于特定类别。
> - **Minor drift 的坦率承认**: 论文承认由于 Qwen-VL 的 patch segmentation 机制，可能存在 minor localization drift。这是一个诚实的局限说明——但通过 zoomed view 的展示，读者可以看到 drift 的程度通常是轻微的（主要影响边缘 patch 的归属）。

### 4.4 Ablation Studies

#### 4.4.1 Role of Inserting Embedding

In this section, we empirically demonstrate that the selected region embeddings contribute positively and meaningfully to the reasoning process. To validate this claim, we construct a controlled ablation experiment in which image patch embeddings are randomly sampled and inserted following the same inference protocol as our method. Concretely, whenever the model determines that additional visual information is required to assist its reasoning, we inject randomly selected patch embeddings instead of the semantically aligned embeddings identified by our saliency-based selection mechanism. We evaluate this variant on V\* Bench, and the results are reported in Figure 5.(a) and Figure 5.(b). As shown, replacing our selected embeddings with randomly sampled ones leads to a substantial and consistent degradation in performance. This performance drop indicates that the gains observed in our method are not simply due to the act of injecting additional visual tokens into the reasoning process. Rather, they stem from incorporating semantically relevant and contextually aligned visual embeddings that meaningfully support intermediate reasoning steps. Together, these results show that our embedding selection captures task-relevant visual information and that the gains stem from informed cross-modal grounding rather than arbitrary token augmentation.

> 💡 **Figure 5(a,b) 批读 — 三组对比的核心结论**:
> - w/o Insert: 完全不注入 embedding → 基线性能
> - Random Insert: 注入随机选择的 embedding → 性能下降或持平基线
> - SIEVE Insert: 注入显著性引导选择的 embedding → 性能提升
> - **结论**: embedding 注入本身不创造价值——注入**对的内容**才创造价值。这个消融直接证明了 "evidence selection 的质量" 是 SIEVE 性能的核心驱动力。

#### 4.4.2 Select Embeddings in Different Layers

We visualize image-token hidden states from different VLM layers and analyze the corresponding image embeddings selected via saliency-based token analysis. Specifically, after identifying salient tokens that strongly influence generation, we compute their similarity with image patch representations extracted from different transformer layers and map the selected embeddings back to their spatial locations to obtain visual grounding results.

Using Qwen3-VL-4B-Instruct as an example, we compute layer-wise embedding retrieval accuracy to justify selecting middle layers. Since our goal is to extract informative key-region embeddings rather than fully reconstruct object extents, we only require the predicted region to overlap with the ground-truth object region. Accordingly, we define the Information Hit Ratio (IHR) as IHR = I(|B_pred ∩ B_gt| > 0), where a prediction is considered correct if the predicted bounding box overlaps with the ground-truth region.

> 💡 **IHR 指标的设计**: IHR 不要求预测框与真值框的 IoU 达到某个阈值——只要有任何重叠就算命中。这个宽松的指标符合 SIEVE 的设计目标：不需要精确分割对象，只需要找到包含目标对象的区域（即使覆盖了一些无关区域），然后让模型自己从中提取关键信息。这与传统 detection/segmentation 的评估标准不同——SIEVE 不追求 localization precision，而是追求 information coverage。

（Figure 5(c) 展示 IHR 随层数变化的曲线，Figure 6 展示不同层的可视化对比）

> 💡 **Figure 5(c) + Figure 6 批读 — 中间层的实证优势**:
>
> | 层范围 | IHR 表现 | 可视化表现 | 解释 |
> |--------|---------|-----------|------|
> | Early (1-10) | 较低 | 匹配到错误的、语义无关的区域 | 浅层语义抽象不够，text token 和 image patch 的对齐尚未形成 |
> | Middle | 最高 | 准确定位到目标对象区域 | 语义和空间信息达到最佳平衡 |
> | Late (30+) | 较低 | 匹配区域偏向 task-specific bias | 深层已被任务目标主导，丢失了通用的跨模态对齐能力 |
>
> **与 Skean et al. (2024, 2025) 的呼应**: 这个发现与已有的 LLM 表示分析工作高度一致——中间层捕获最丰富的语义信息。SIEVE 的创新点在于将这个理论发现**工程化**为一个具体的证据发现机制。

#### 4.4.3 Role of Action Rewards

In Section 3.3, we incorporate action-level rewards into the total reward function, including a thought richness reward and a signal reward. These additional rewards encourage the model to produce more informative reasoning traces and to issue appropriate response requests, thereby improving both stability and interpretability during training. In Figure 7, we analyze the impact of enabling or disabling these action rewards.

（Figure 7 包含四张子图：(a) reward 变化曲线，(b) entropy loss 变化曲线，(c) avg response length，(d) max response length）

> 💡 **Figure 7 批读 — Action Reward 对训练稳定性的关键影响**:
>
> **子图 (a) — Signal Reward 的作用**:
> - 移除 signal reward 时，训练 reward 在后期崩溃 (collapse to near zero)
> - 引入 signal reward 后，训练过程显著稳定
> - **解释**: 没有 signal reward 显式鼓励模型做出 "commitment"（插入或输出答案），模型可能陷入"什么都不做"的策略——既不输出答案，也不插入 evidence，从而获得极低的 reward，导致梯度信号消失
>
> **子图 (b) — Entropy Loss 的差异**:
> - 印证了 (a) 的发现：无 signal reward 时 entropy loss 波动剧烈且最终发散
>
> **子图 (c) — Thought Richness Reward 的作用**:
> - 没有 thought richness reward 时，平均响应长度显著下降——模型学会输出空推理 `<think> </think>` 来满足格式要求但不产生实质性内容
> - 有 thought richness reward 时，平均响应长度保持稳定——模型被鼓励产出有意义的长推理
>
> **子图 (d) — Max Response Length**:
> - 没有 thought richness reward 时，最大响应长度极其不稳定且出现极端值（可能是重复/循环输出）
> - 有 thought richness reward 时，最大长度稳定受控
>
> **核心教训**: RL 训练中，如果没有精心设计的 auxiliary reward 来激励"好的行为模式"，模型必然会找到 reward function 的 loophole。SIEVE 的四维 reward 设计（尤其是 R_fmt 和 R_act）正是为了防止这些 reward hacking 行为。

---

## 三、Summary

- **主实验**: SIEVE 在 V\* (+5~8%) 和 HR-Bench (+2~4%) 上一致超越 vanilla 和 GRPO baseline；在 MME-Real-Lite (+17~19%) 和 WeMath (+20.65% for 8B) 上提升最大
- **模式**: 4B 模型受益 > 8B 模型受益（小模型更需要视觉证据辅助）；空间推理受益最大
- **消融结论**: 
  1. Embedding 选择质量是关键——随机选择无效甚至有害
  2. 中间层是最佳的跨模态匹配层（IHR 最高）
  3. Action reward 是训练稳定性的必要条件——防止 reward collapse 和 empty reasoning
- **可视化**: Evidence regions 与语义目标高度一致，expand 操作补全了 patch grid 的边界不对齐
