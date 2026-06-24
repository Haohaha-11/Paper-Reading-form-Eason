[← 返回 README](../README.md)

# 5. Experiments

## 一、Preview

实验部分包含四个子主题：
1. **Setup**: 训练配置、评估基准、对比变体
2. **Main Results**: grounded thinking 大幅超越 non-grounded baseline 和 base model
3. **Effect of Grounding Reward**: box reward 持续正面、point reward 效果不显著
4. **Box vs Point**: 计数上 point 更优、空间推理上两者接近

所有实验基于 Gemma3-4B-IT，使用 controlled variants 干净地隔离 grounding 的效果。

---

## 二、原始文本

### 5.1 Setup

**Training**: All models are trained with verl (Sheng et al., 2024), SGLang (Zheng et al., 2024) as inference engine, FSDP2 (Zhao et al., 2023) as training backend. Base model: Gemma3-4B-IT.

Three controlled variants (same images, questions, answers, reasoing traces; differ only in grounding tags):

| Variant | Grounding | Tag Format |
|---------|-----------|------------|
| Non-grounded thinking | None | Pure text |
| Thinking with Grounding Box | Box | `<obj> name \| [x1,y1,x2,y2] </obj>` |
| Thinking with Grounding Point | Point | `<obj> name \| [x,y] </obj>` |

SFT first (cold-start), then RL with GRPO.

> 💡 **实验设计的干净性**: 三个变体使用 "parallel examples" -- 相同的图像、问题、答案和底层推理链，唯一区别是是否包含 grounding tags 以及 tag 是 box 还是 point。这确保了任何性能差异都可以归因于 grounding 的有无和类型。

**Evaluation**: 2 counting + 4 spatial reasoing benchmarks, evaluated via VLMEvalKit (Duan et al., 2025). Temperature 1.0, 4 inference passes, report both average accuracy and pass@4.

| Benchmark | Type | Metric |
|-----------|------|--------|
| TallyBench | Counting | Acc, Pass@4 |
| CountQA | Counting | Acc, Pass@4 |
| VSR-zeroshot | Spatial (yes/no) | Acc, Pass@4 |
| EmbSpatial | Spatial (embodied) | Acc, Pass@4 |
| SpatialMQA | Spatial reasoing | Acc, Pass@4 |
| MultihopSpatial | Multi-hop spatial | Acc, Pass@4 |

> 💡 **Pass@4 的意义**: Temperature 1.0 下的 stochastic decoding 引入了采样方差，4 次推理取 pass@4 能更稳定地反映模型的真实能力上限，减少 variance 对结论的干扰。

---

### 5.2 Main Results

> 💡 **阅读指南**: 以下表格数据中的关键对比关系:
> 1. **Grounded vs Non-grounded**: 看 grounding 是否有效
> 2. **Grounded vs Base (Gemma3-4B-IT)**: 看 SFT+RL 的综合提升
> 3. **4B grounded vs 27B**: 看 grounded thinking 能否弥补参数量的差距

**Table 1: Counting Benchmarks**

| Method | TallyBench Acc | TallyBench P@4 | CountQA Acc | CountQA P@4 |
|--------|:---:|:---:|:---:|:---:|
| Gemma3-4B-IT | 33.33 | 40.65 | 9.87 | 14.14 |
| Non-grounded Thinking | 21.73 | 42.00 | 4.30 | 12.24 |
| **Thinking w/ Grounding Box** | | | | |
| - w/o grounding reward | 37.24 | 64.45 | 10.73 | 27.75 |
| - w/ grounding reward | **38.81** | **64.50** | **11.19** | 28.47 |
| **Thinking w/ Grounding Point** | | | | |
| - w/o grounding reward | 39.03 | 65.50 | **12.34** | **31.48** |
| - w/ grounding reward | **39.31** | **65.75** | 11.65 | 29.77 |

> 💡 **关键发现 1 — Counting**:
>
> - **Non-grounded thinking 严重退化**: accuracy 从 33.33 降到 21.73，几乎不如直接回答。根本原因是 **length collapse**：RL 训练中回答长度线性下降，探索空间缩小。
> - **Grounded thinking 显著提升**: box mode 提升 ~5-6 个点 accuracy，pass@4 更是从 40.65 到 64.50（+24 个点）。
> - **Point mode 在计数上更强**: point w/o reward 在 CountQA 上达到最高 12.34 (acc) 和 31.48 (pass@4)。
> - **Grounding reward for box 有小幅正收益**: box w/ reward 在两个 benchmark 的 accuracy 上都超过了 w/o reward。

**Table 2: Spatial Reasoing Benchmarks**

| Method | VSR-zero Acc | EmbSpatial Acc | SpatialMQA Acc | Multihop Acc |
|--------|:---:|:---:|:---:|:---:|
| Gemma3-4B-IT | 56.65 | 49.13 | 25.35 | 22.70 |
| Non-grounded Thinking | 51.84 | 20.54 | 14.17 | 4.79 |
| Box w/o g-reward | 66.82 | 57.62 | 37.64 | 34.89 |
| Box w/ g-reward | **68.08** | **59.93** | **38.68** | **37.68** |
| Point w/o g-reward | 65.38 | **60.25** | **39.13** | 37.03 |
| Point w/ g-reward | 64.67 | 60.88 | 39.01 | 37.01 |
| *Gemma3-12B-IT* | *67.98* | *56.68* | *37.85* | *30.08* |
| *Gemma3-27B-IT* | *69.25* | *62.09* | *38.99* | *30.94* |

> 💡 **关键发现 2 — Spatial Reasoing**:
>
> - **Non-grounded thinking 灾难性退化**: 在 MultihopSpatial 上从 22.70 暴跌到 4.79，EmbSpatial 从 49.13 到 20.54。这再次验证了 length collapse 问题。
> - **4B grounded 模型超越 27B**:
>   - VSR-zero: Box w/ reward (68.08) 接近 27B (69.25)，超过 12B (67.98)
>   - SpatialMQA: Point w/o reward (39.13) 超越 27B (38.99)
>   - MultihopSpatial: Box w/ reward (37.68) 显著超越 27B (30.94)
> - **Grounding reward 对 box 更有效**: box w/ reward 在全部 4 个 benchmark 上均优于 w/o reward
> - **Pass@4 差距更大**: 所有 grounded 模型在 pass@4 上都大幅超越 27B，表明 grounding 增强了模型在多次采样中的探索能力

> 💡 **Length Collapse 解释**:
>
> Non-grounded thinking 在 RL 中的退化是一个重要发现。作者的解释：
> - Non-grounded 的 rollout 缺乏结构化约束
> - 模型逐渐学会"简化"回答来获取 format rewards，损失了思考的深度
> - Grounding tags 提供了 local structure：每个 `<obj>` tag 强制模型停下来定位一个 object
> - Grounding-format reward 进一步鼓励模型保持 structured output
> - 这种局部结构起到类似 "scaffolding" 的作用，稳定了 RL 训练

---

### 5.3 Effect of the Grounding Reward

> 💡 **核心问题**: grounding reward 是否带来了可测量的下游性能提升？

**Box mode**: grounding reward 在全部 6 个 evaluation benchmark 上均改善了平均 accuracy。

- Counting tasks: 增益相对温和
- Spatial tasks: 增益更明显

> 💡 **Box reward 在空间推理上更有效的原因**:
>
> 空间推理任务（left/right, above/below, distance, overlap）对 object extent 和相对几何关系高度敏感。Bounding box 提供了 object identity 和 object extent 的双重信息。精确的 box 意味着更准确的 spatial relation 推理。Grounding reward 通过 IoU 直接鼓励模型生成与 GT 更一致的 box，间接提升了 spatial reasoing 的质量。

**Point mode**: grounding reward 没有带来一致的提升。6 个 benchmark 中，point RL w/ 和 w/o reward 的整体表现接近，有些指标上升，有些下降。

> 💡 **Point reward 效果不显著的根本原因**:
>
> 回到 Section 4 的分析 -- point F1 是离散信号：
> - 点在 mask 内部任意移动：reward 不变
> - 点跨过 mask 边界：reward 突变
> - 这种粗粒度的反馈使梯度信号不连续，RL 优化困难
>
> 这**不意味着 point grounding 无用**（从 5.2 的 main results 看，point mode 在计数上甚至优于 box mode），而是说 **当前的 point reward 设计需要改进**才能提供有效的 RL 训练信号。

---

### 5.4 Box vs. Point Grounding

**Counting benchmarks**: Point-mode consistently outperforms box-mode.

> 💡 **Counting 上 Point 更优的原因分析**:
>
> 计数的核心需求是 **instance-level localization**：识别哪些 object 属于被计数集合，将其与干扰物区分开。不需要恢复每个 object 的完整 extent。
>
> Point grounding 的优势：
> 1. 提供紧凑的 instance grounding
> 2. 避免了生成 tight bounding box 的困难（尤其是小物体、遮挡、异形物体）
> 3. 一个点足以标记"这是一个 counted instance"
>
> Box grounding 的劣势：
> 1. 模型需要额外 effort 来确定 box 边界
> 2. 对于小物体或不规则形状物体，精确 box 更难预测
> 3. 这些 extra effort 对计数任务没有额外帮助

**Spatial reasoing benchmarks**: The two interfaces are much closer.

> 💡 **空间推理上 Box vs Point 接近的原因分析**:
>
> - Box 提供更丰富的几何信息（object extent, size, boundary），对 overlap 和 relative position 关系有帮助
> - 但 Point 仍能识别相关的 objects 和 spatial anchors，许多空间问题仅靠 instance-level grounding 加模型的 visual representation 就能回答
> - Box 的额外几何信息并非总能转化为 accuracy 优势
>
> 结论：point grounding 更适合计数，point 和 box grounding 在空间推理上大致相当。

> 💡 **整体结论的工程含义**:
>
> | 任务类型 | 推荐 Grounding 模式 | 原因 |
> |---------|-------------------|------|
> | Counting | Point | Instance-level localization 足够，避免 box 回归困难 |
> | Spatial Reasoing | Box (w/ grounding reward) | Box extent 提供几何信息，grounding reward 持续正收益 |
> | 通用场景 | Box (w/ grounding reward) | 最稳定的 RL 优化信号 |

---

## 三、Summary

- **Grounded thinking 大幅有效**: 在计数 (+24 pass@4) 和空间推理上远超 base model 和 non-grounded baseline
- **Non-grounded thinking 的 length collapse**: RL 中纯文本 thinking 会退化，grounding tags 提供结构化的 stabilization
- **4B 超越 27B**: 在 SpatialMQA 和 MultihopSpatial 上，grounded 4B 模型匹敌甚至超越 27B
- **Box reward 持续正收益**: 6/6 benchmarks 上 w/ reward > w/o reward
- **Point reward 需改进**: 离散 F1 信号不足以提供有效 RL 优化，但 point grounding 本身不差
- **Point > Box on counting, Point ≈ Box on spatial**: 任务特性决定最佳 grounding 接口
