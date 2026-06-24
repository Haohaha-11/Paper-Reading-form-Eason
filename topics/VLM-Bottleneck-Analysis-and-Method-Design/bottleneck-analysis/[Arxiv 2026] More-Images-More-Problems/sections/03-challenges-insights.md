[<- 返回 README](../README.md)

# 3. Challenges and Insights in Multi-Image LVLMs

## 一、Preview

这是本文最核心的分析章节，分两步走：(1) 构建 MIMIC 可控测试床（4 个核心任务），(2) 在 13+ LVLM 上运行 6 个维度的诊断实验，揭示 6 大 Finding。这些 Finding 从现象到根因层层递进，形成一条完整的逻辑链：**性能退化根本原因是序列长度膨胀（而非图像数量增加）-> 模型实质是"单图模型"-> 无法跨图聚合信息 -> 对干扰敏感 -> 多概念追踪有限 -> 深层 attention 从跨图转为单图主导**。前 5 个 Finding 是外部行为诊断，Finding 6 是内部机理追索，共同解释了"More Images, More Problems"的根本原因。

---

## 二、原始文本

Herein, we systematically investigate the current LVLMs limitations in multi-image scenarios across six complementary dimensions: information distribution, query complexity, reasoing patterns, robustness to visual distractors, scalability with the number of images, and multi-concept tracking. For this purpose, we introduce MIMIC, a controlled testbed synthesized from a curated subset of MS-COCO. Using the manually annotated bounding boxes and labels, MIMIC generates multi-image sequences that allow precise control over information spread, distractor presence, object-instance distributions, and sequence length. This design enables decorrelated, fine-grained analyses of the model's behavior. Beyond these dimensions, our framework scrutinizes the models' mechanisms for aggregating and reasoing over distributed visual information. Through this controlled analysis, we aim to isolate the specific limitations and offer actionable insights for the next generation of visual understanding models.

> **MIMIC 的 6 个可控维度**:
>
> | 维度 | 控制参数 | 测试目标 | 对应 Finding |
> |------|---------|---------|-------------|
> | 信息分布 (Information spread) | s: 实例分布的图像数 | 跨图聚合能力 | Finding 3 |
> | 查询复杂度 (Query complexity) | k: 目标类别数 | 多概念追踪能力 | Finding 5 |
> | 推理模式 (Reasoing patterns) | 任务类型 (Count/List/Common/Odd) | 聚合 vs 比较推理 | — |
> | 干扰鲁棒性 (Distractor robustness) | d: 干扰图像数 | 注意力聚焦能力 | Finding 4 |
> | 图像数量扩展 (Scalability) | N: 总图像数 | 序列长度处理 | Finding 1, 2 |
> | 多概念追踪 (Multi-concept tracking) | k++ + s variation | 并行概念维护 | Finding 5 |

### 3.1 Testbed Benchmark Construction

> **MIMIC 数据构建流程**:
>
> ```
> MS-COCO 全集
>     │
>     ├── 过滤: bbox < 5% 图像面积 -> 确保视觉可辨识
>     │   (LLaVA-OV 输入分辨率 384x384px 下过小目标不可辨)
>     │
>     ├── 类别池: 等概率采样 -> 消除类别不平衡偏差
>     │
>     └── 程序化生成任务:
>         ├── Counting: N 张图, k 个类别, "数一数每类有多少个"
>         ├── Listing: N 张图, "列出你能识别的所有{类别}物体"
>         ├── Common: N 张图, "找出在所有图中都出现的物体"
>         └── Odd-One: N 张图, "找出只在少数图中出现的物体"
> ```

We build the probing dataset by procedurally generating multi-image, open-ended question-answering tasks that target distinct aspects of cross-image reasoing. To this end, we sample a curated subset from MS-COCO by filtering images with object bounding boxes less than 5% of the image in order to ensure visual recognizability at common LVLM input resolutions (e.g. LLaVA-OV's 384x384px). To minimize the impact of potential class imbalance, we first select a pool of classes and then sample from this pool, ensuring that each class is chosen with an equal probability. This ensures that the distributions of classes and instances remain consistent across settings.

> **为什么用开放式问答而非多选题？** 三个原因：(1) 多选题的固定选项集可能引入 shortcut（模型通过选项排除而非真正理解图像）；(2) 干扰项选择难以校准——干扰项的好坏直接影响评测公平性；(3) 开放式问答更贴近实际使用场景。为减少 prompt bias，每个任务使用多套随机采样的 prompt 模板。

**Counting Task 的两种设置——设计精妙之处**:

- **Balanced**: 固定总实例数，仅改变实例分布在多少张不同的图像上。例如，总共 4 个实例，分布方式可以是 [4]（1 张图）、[2,2]（2 张图）、[1,1,1,1]（4 张图）。这样做的效果是：**排除了"更多图像天然包含更多实例"的混淆**。
- **Unbalanced**: 总实例数随图像数自然变化。这更贴近真实场景，但无法完全解耦实例数 vs 图像数的效果。

> **四个任务的设计意图**:
>
> | 任务 | 考察的认知能力 | 难度来源 | 评估指标 |
> |------|-------------|---------|---------|
> | Counting | 跨图信息聚合 + 多概念追踪 | 信息分散度、干扰数量、类别数 | Binary accuracy |
> | Listing | 密集信息提取 + 视觉感知 | 图像数量、类别密度 | F1-score |
> | Common | 跨图比较识别共同元素 | 图像数量（无干扰） | Binary accuracy |
> | Odd-One | 跨图比较识别独有元素 | 图像数量 + 少数类定位 | Binary accuracy |

**Counting**: Given a set of N input images, and a query containing k object classes, the model is asked to count the total number of instances of each class. With increasing difficulty, we vary the distribution of object instances across images. For example, in the easiest setting, all may be concentrated in a single image, while in more challenging cases, instances are spread across multiple images. We refer to this as the information spread. Additionally, we introduce distractors - images that do not contain any instances of the target objects - to assess the model's ability to focus on relevant information.

In summary, this task offers the following controllable dimensions: (1) number of object classes to count (k); (2) information spread across images (s); (3) number of distractor images and (4) total number of images. Each case probes different aspects and potential biases. For instance, increasing the number of object classes k tests the model's multi-concept tracking ability, while varying the information spread evaluates its capacity to aggregate information across images.

To account for potential biases caused by the long-tail distribution of object instance counts in natural images, which may lead to models favoring smaller counts, we design two distinct settings: (1) Balanced, where the total number of object instances is fixed, but distributed across a varying number of images; (2) Unbalanced, where the total number of object instances varies arbitrarily with the number of images. The metric of choice is binary accuracy, i.e. the answer is correct if it matches the ground truth count exactly.

> **Binary accuracy 的严格性**: 计数必须精确匹配才算正确。这意味着即使模型数对了 4 个中的 3 个（75% 正确），在 binary accuracy 下仍算错误。这比 MAE/MSE 等连续指标更严格，但也因此更能暴露模型的真实短板。

**Listing**: The model is presented with a set of N images and asked to list all object classes belonging to a given category (e.g.: animals, vehicles, etc.) that it can identify. This task evaluates the model's ability to exhaustively extract information in a dense manner from multiple images. As a byproduct, it also measures the model's visual perception ability to recognize and categorize multiple objects, as well as its capacity to aggregate this information into a coherent list. Similar to the Counting task, we vary the number of images and the distribution of object instances to assess the model's robustness in multi-image understanding. The model's response is evaluated on the completeness and accuracy of the list, using F1-score as metric.

**Common and Odd-One**: The two tasks are designed to assess the model's ability to identify shared or unique elements across multiple images. Importantly, while previous tasks focus on aggregating information, these tasks require comparative reasoing across images, hence the model must first implicitly identify all objects before performing cross-image analysis. In the Common task, the model has to determine which object class is present in all provided images, while in the Odd-One case, it must identify the object class that is present in a minority of images. For simplicity, we ensure by design that the answers are unique. The model's answers are evaluated based on their correctness, with binary accuracy as the metric.

> **Common vs Odd-One 的认知差异**: Common 需要跨图信息聚合——找出"共性"；Odd-One 更依赖局部差异定位——找出"特殊者"。后文跨任务泛化实验（Section 5.2）证实了这个区别：仅在 Common 上训练的模型在 Odd-One 上表现差（3.7%），反之亦然。

### 3.2 Empirical Analysis

Setup: We evaluate several state-of-the-art LVLMs: LLaVA-OV, Qwen2-VL and InternVL2. We use publicly available checkpoints and follow the official data processing pipeline. For test data, we use the MIMIC benchmark described in Section 3.1, selecting tasks and configurations that best isolate the dimensions we aim to probe.

#### Finding 1 & 2: Performance vs Sequence Length vs Number of Images

> **实验设计——解耦"序列长度"和"图像数量"**:
>
> 这是一个精妙的因果推断实验设计：
>
> (a) **直接增加图像数**: 2->35 张，不控制序列长度。结果：性能持续下降。
>
> (b) **1-D average pooling 降低序列长度**: 对 vision token 序列做 group-wise pooling，零样本操作（无额外训练）。结果：4-8x pooling 后性能**反而提升**。
>
> (c) **Control experiment**: 在像素空间下采样再上采样，保持序列长度不变但减少视觉信息。结果：**性能下降**。
>
> (a)+(b)+(c) 的逻辑链：
> - (a) 观察到性能下降；
> - (b) 降低序列长度 -> 性能上升，说明"长序列"是瓶颈；
> - (c) 降低信息量但保持长序列 -> 性能下降 vs (b) 的性能上升，排除"是因为信息减少导致 pool 有效"的替代解释。
> - 结论：性能退化的**主要原因是序列长度膨胀，而非图像数量增加或视觉信息不足**。

LLMs are known to manifest position and sequence length biases, with tokens appearing earlier and late in the sequence receiving more attention. Unlike for LLMs, we distinguish two axes of sequence length growth: (1) increasing the number of images, and (2) increasing the input image(s) resolution. We seek to understand if the performance degradation stems from the model's inability to handle long sequences, or from the inability to process many images. We disentangle these two factors with the following experiments: (a) directly increasing the number of images without explicitly controlling for sequence length; (b) reducing the vision token sequence length through 1-D average pooling applied to the original multi-image vision tokens. To ensure that the observed behavior is not an artifact of reduced information, we also perform a control experiment where we similarly reduce the amount of information by downsampling and then upsampling back the images in pixel space, prior to being passed to the vision encoder. This preserves the initial sequence length but reduces the amount of visual information available to the model.

The results are summarized in Fig. 3. On the left, we plot performance changes for different models as we decrease the number of vision tokens via 1-D pooling. Due to different processing, each model allocates different number of tokens per image, hence we mark two points - extreme right (no downsampling) and central point that maximizes performance. On the right, we show the control experiment, that decreases the information in the pixel space artificially without reducing sequence length. Surprisingly, we find that reducing the sequence length in a zero-shot manner via 1-D pooling up to 4-8x leads to significant performance improvements across all models. The control experiment confirms that gains are due to sequence length reduction rather than information reduction. This suggests that the models primarily struggle with long sequence understanding rather than with processing multiple distinct images.

> **Finding 1**: The performance degradation in multi-image scenarios stems primarily from increased sequence length rather than the increased number of images.

Moreover, we observe that for LLaVA-OV performance peaks when the vision sequence length is approximately that of one or two images (i.e., roughly the number of vision tokens for a 384x384 image/patch). This suggests the model effectively relies on a single-image context and has limited practical multi-image integration; we later evaluate how targeted finetuning can mitigate this limitation.

> **Finding 2**: Current LVLMs primarily behave as single-image models: performance peaks when the vision-token sequence length matches that produced by one or two images.

> **Finding 1 和 Finding 2 对方法设计的启示**:
> - Finding 1 -> 优化方向应该在"如何管理/压缩序列长度"而非"减少图像数" -> 注意力掩码策略（Section 4）通过限制跨图 attention 的复杂度来间接缓解序列长度问题
> - Finding 2 -> 模型的"舒适区"是 1-2 张图 -> 需要训练数据让模型适应更长/更多图的场景 -> 合成多图训练数据策略（Section 4）

#### Finding 3: Information Aggregation Across Images

Information aggregation across images: Prior benchmarks rarely control for how information is distributed across images, making it difficult to isolate whether models can effectively aggregate information across images. To this end, we vary the information spread in the counting task, which defines how object instances are distributed across images. In Fig. 1 (left and middle) we show the results of increasing the number of images containing the object instance from 1 to 7. We observe a sharp accuracy drop that approaches 0 even when very few distractors are present. This trend is consistent across all models tested and manifests both in balanced and unbalanced counting settings. This indicates that the models may rely on shortcuts, such as focusing on a single or very small subset of images, rather than effectively integrating information from all provided images.

> **Finding 3**: Current LVLMs struggle to aggregate information across multiple images.

> **实验细节——Balanced vs Unbalanced 下的信息聚合**:
>
> 在 Balanced 设置下（固定总实例数 = 4），当所有 4 个实例集中在一张图中时，准确率约 75-80%；当 4 个实例分散在 4 张图中时，准确率趋近于 0%（即使没有干扰图像！）。这说明问题不是"找不够"，而是"合不拢"——模型无法把分散在不同图中的信息有效聚合。这暗示模型可能依赖 shortcut（如只关注第一张或最后一张图），而不做真正的 cross-image integration。

#### Finding 4: Robustness to Visual Distractors

Robustness to visual distractors. In real-world scenarios, models often encounter irrelevant or distracting information. To evaluate the robustness of LVLMs, we introduce a varying numbers of irrelevant images into the input sequence. As shown in Fig. 1 (left), the accuracy decreases as the number of distractors increases (e.g: from 79.0% to 66.5% (1 vs 34 distractors) for 1 query image, from 75.0% to 12.5% for two query images, etc., for LLaVA-OV). A particularly pronounced drop occurs as the number of images containing the object of interest increases, suggesting that distractors exacerbate the models' existing difficulties in aggregating information across multiple images.

> **Finding 4**: Models are sensitive to visual distractors, especially if information is spread out.

> **干扰效应与信息分散的交互**: Finding 4 最关键的观察是交互效应——当信息已经分散在多张图中时，干扰图像的破坏性更大。这形成了一个"双重打击"：模型已经在费力聚合分散的信息，额外的干扰图像进一步稀释了注意力。这解释了为什么真实场景（天然包含信息分散 + 大量干扰）下 LVLM 的多图表现如此糟糕。

#### Finding 5: Multi-Concept Tracking

Multi-concept tracking. The ability to track and attend to multiple concepts simultaneously is critical for multi-image understanding. To probe this capability, we vary the number of object classes k that the model is required to count. As shown in Fig. 1 (right), the model performance degrades sharply as k increases, indicating a limited capacity to handle multiple concepts at once.

> **Finding 5**: LVLMs demonstrate limited capacity for multi-concept tracking, reducing their reliability on complex multi-object queries.

> **多概念追踪的退化曲线**: 当 k=1 时准确率约 80%；k=2 时降至约 50%；k=3+ 时接近随机。这个退化速度说明当前模型的多概念并行处理能力是极有限的——不是线性退化，而是崩塌式退化。这可能是 core attention bottleneck：模型无法在 attention 中同时维护多个独立概念的"追踪状态"。

#### Finding 6: Multi-Image Attention Pattern Analysis

> **Finding 6 是本文最深层的机理分析**——直接看模型内部的 attention 行为，解释前 5 个行为 Finding 的根源。

To probe how visual information is propagated and integrated across images at the token level, we analyze attention patterns among vision tokens in multi-image inputs. Concretely, we compute the normalized attention scores from each vision token to all other vision tokens in the input sequence subject to an autoregressive attention masking on a subset of 50 samples. Fig. 4 summarizes the results across a few layers of interest for a LLaVA-OV model for multi-image inputs with 4 images. We find that in earlier layers, there is a significant amount of inter-image attention, indicating that the model is attempting to integrate information across images. However, as we progress to deeper layers, the attention becomes predominantly intra-image. This inflection point occurs somewhere around the middle of the network. This shift may contribute to the observed difficulty in aggregating information across multiple images. Conceptually, the build-up of representations appears to proceed from broad semantic correlations across images to finer-grained, instance-level integration.

> **Finding 6**: Inter-image attention diminishes in deeper layers of LVLMs, indicating a shift from cross-image integration to intra-image focus.

> **Attention 模式的深层含义**:
>
> Fig.4 揭示了一条从"跨图"到"单图"的 attention 衰减曲线：
>
> - **早期层（Layer 0-5）**: 显著的跨图 attention。模型在尝试建立跨图像的语义关联——"这几张图中哪些语义上相关的？"
> - **中期层（Layer 6-12）**: 注意力逐渐从跨图转向单图内部。转折点（inflection point）约在网络中部。
> - **深层（Layer 13-23）**: 几乎完全是单图内部的 attention。模型已经放弃跨图整合，转而做单图内部的细粒度处理。
>
> 这种"从全局到局部"的模式在人类视觉认知中也有对应（先全局感知后局部聚焦），但问题在于：**深层完全失去了跨图整合的机会**。如果任务需要跨图比较（如 Odd-One），深层已经无法获取其他图像的信息了。

This has a series of consequences:

(1) The early inter-image attention may introduce noise or distractions that hinder the model's ability to focus on relevant information in later layers; hence, early mistakes in cross-image interaction are harder to correct.

(2) The cross-image interaction under a causal attention mechanism may lead to error propagation, where tokens belonging to later images cumulate higher amount of noise with incorrect information from earlier images; this may reduce the vision perception capability of the model for later images and explain some of the performance degradation as the number of images increases.

(3) The architecture and training objectives may not sufficiently encourage cross-image integration, leading to a default behavior of treating images independently.

(4) The observed attention patterns may reflect inherent biases in the training data, where the multi-image tasks don't require deep cross-image reasoing, leading the model to learn shortcuts that prioritize single-image understanding.

> **Finding 6 的四大推论——将它们与 Finding 1-5 串联起来**:
>
> | 推论 | 解释 | 关联的 Finding | 方法启示 |
> |------|------|---------------|---------|
> | (1) 早期跨图噪声 | 早期层跨图 attention 引入噪声，深层无法纠正 | Finding 3 (聚合困难) | 深层 mask 跨图 attention 以减少噪声干扰 |
> | (2) 因果注意力误差传播 | 后序图像 token 的表示累积前面图像的误差 | Finding 1 (序列长度问题) | Attention mask 阻断误差传播路径 |
> | (3) 架构/训练缺乏跨图激励 | 模型默认行为是独立处理各图 | Finding 2 (单图模型行为) | 合成多图数据提供显式跨图训练信号 |
> | (4) 训练数据偏置 | 多图训练数据中的任务不需要深度跨图推理 | Finding 4, 5 (干扰/多概念) | 程序化生成需要跨图聚合的针对性数据 |

---

## 三、Summary

### 6 大 Finding 的逻辑链条

```
Finding 1: 性能退化来自序列长度膨胀（非图像数量）
       ↓
Finding 2: 模型本质是"单图模型"（性能在 1-2 图 token 量时最优）
       ↓
Finding 3: 无法跨图聚合信息（information aggregation failure）
       ↓
Finding 4: 对干扰敏感（信息分散时干扰破坏性更大，交互效应）
       ↓
Finding 5: 多概念追踪能力有限（崩塌式退化，非线性）
       ↓
Finding 6 (根因): 深层 attention 从跨图转为单图主导
       ├── 推论 1: 早期跨图噪声无法在深层纠正
       ├── 推论 2: 因果注意力导致误差向前传播
       ├── 推论 3: 架构/训练缺乏跨图整合激励
       └── 推论 4: 训练数据偏置为单图理解
```

### 从 Finding 到 Method 的映射

| Finding | 指向的方法 | 方法类型 |
|---------|-----------|---------|
| Finding 1, 2, 6 | Attention Masking（限制深层跨图 attention） | 优化侧 |
| Finding 3, 4, 5, 6-推论4 | 合成多图训练数据（提供显式跨图推理监督） | 数据侧 |

### 实验设计亮点

1. **Balanced vs Unbalanced 设置**解耦了"实例数"和"图像数"的混淆
2. **Pooling + Control experiment** 用因果推断的思路解耦了"序列长度"和"信息量"
3. **Attention 可视化** 将行为 Finding 追索到架构层根因
4. **开放式 QA 设计** 避免了多选题的 shortcut
