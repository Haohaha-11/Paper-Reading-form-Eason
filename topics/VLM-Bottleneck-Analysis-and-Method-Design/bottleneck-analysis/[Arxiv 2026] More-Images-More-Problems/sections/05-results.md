[<- 返回 README](../README.md)

# 5. Results + 6. Conclusions + 7. Limitations

## 一、Preview

本节涵盖实验主结果、消融分析、结论和局限。实验从两个维度验证方法有效性：(1) 外部多图 benchmark（MuirBench, Blink, MMIU, MIRB, MMT, NLVR2）上的 SOTA 比较；(2) MIMIC 自身 benchmark 上的零样本 vs 微调对比。消融实验包括：跨任务泛化、注意力掩码效率分析、掩码层数选择。关键观察：(a) 方法在 0.5B 和 7B 两个规模上均一致有效；(b) 注意力掩码 + LoRA 在 0.5B 上甚至超过全微调；(c) Common 和 Odd-One 任务受益最大（跨图比较 vs 局部定位有各自优势）。

---

## 二、原始文本

### 5.1 Comparison with State-of-the-Art

**Existing multi-image benchmarks.** We first report results on MuirBench and its subtasks in table 2. Across all model sizes, our approach consistently outperforms the corresponding LLaVA-OV baseline. Notably, for the 7B model, our masked-attention variant improves the overall score from 41.7 to 51.3%. We observe a similar trend for the smaller 0.5B variant, indicating that the improvements are robust across model sizes. Interestingly, our method generalizes well to out-of-domain subtasks, including geographic, action and diagram understanding, suggesting that our data construction strategy teaches the model multi-image processing concepts rather than object perception, which we argues develops already in the single-image training phase.

> **MuirBench 结果解读**:
>
> | 模型 | Baseline | Ours (masked) | 提升 |
> |------|----------|---------------|------|
> | LLaVA-OV-0.5B | 26.8 | 32.5 | +5.7 |
> | LLaVA-OV-7B | 41.7 | 51.3 | +9.6 |
>
> 关键观察：
> 1. **跨模型规模一致有效**: 0.5B 和 7B 均有显著提升，7B 受益更大（+9.6 vs +5.7）
> 2. **对非目标子任务也泛化**: Geographic, Action, Diagram 等与 MIMIC 训练数据无直接关联的子任务也有提升——说明方法教的是"如何做多图处理"而非"如何识别物体"
> 3. **某些子任务提升尤为显著**: Counting (+6.2 for 7B), Grounding (+13.2 for 7B), Attribute (+19.3 for 7B)

Next, we extend the evaluation to additional multi-image benchmarks, including Blink, MMIU, MIRB, MMT, and NLVR2. Our approach achieves consistent improvements across all variants. As shown in table 3, our masked-attention fine-tuning strategy yields significant gains over the baseline even with very few trainable parameters, and in some cases outperforms full fine-tuning (e.g., LLaVA-OV 0.5B).

> **多 Benchmark 汇总 (LLaVA-OV-7B)**:
>
> | Benchmark | Baseline | Ours (masked) | Gain |
> |-----------|----------|---------------|------|
> | MuirBench | 41.7 | 51.3 | +9.6 |
> | Blink | 50.4 | 51.9 | +1.5 |
> | MMIU | 45.0 | 45.5 | +0.5 |
> | MIRB | 47.2 | 51.0 | +3.8 |
> | MMT | 56.6 | 55.3 | -1.3 |
> | NLVR2 | 84.2 | 87.3 | +3.1 |
> | **Avg** | 54.2 | 57.1 | +2.9 |
>
> 注：MMT 上略有下降 (-1.3)，其余所有 benchmark 均有提升。训练与评测数据分布存在差异时存在小幅度波动是正常的。

**MIMIC benchmark.** We report results in table 4. Unless otherwise specified, all results for Counting subtask correspond to the balanced split. Our method significantly outperforms LLaVA-OV across all four tasks. For the 0.5B model, the average score improves from 26.4 to 49.4, while for the 7B model, masked fine-tuning increases performance from 54.0 to 63.8. Gains are most pronounced on the Common and Odd-One tasks, highlighting improved information aggregation and multi-concept reasoing across images.

> **MIMIC Benchmark 结果解读**:
>
> **LLaVA-OV-0.5B**: Baseline avg 26.4 -> Full FT 45.5 (+19.1) -> Masked 49.4 (+23.0)
>
> | 任务 | Baseline | Full FT | Masked | 最佳提升 |
> |------|----------|---------|--------|---------|
> | Common | 44.7 | 68.5 | 68.9 | +24.2 |
> | Counting | 29.7 | 37.8 | 35.8 | +8.1 |
> | Odd-One | 8.3 | 41.0 | 50.9 | +42.6 |
> | Listing | 22.8 | 34.5 | 42.0 | +19.2 |
>
> **LLaVA-OV-7B**: Baseline avg 54.0 -> Masked 63.8 (+9.8)
>
> | 任务 | Baseline | Masked | 提升 |
> |------|----------|--------|------|
> | Common | 71.5 | 75.5 | +4.0 |
> | Counting | 29.7 | 51.2 | +21.5 |
> | Odd-One | 58.1 | 72.1 | +14.0 |
> | Listing | 56.6 | 55.0 | -1.6 |

> **最大的收益在 Odd-One 和 Common——为什么？** 这两个任务最需要跨图比较推理。Odd-One 需要找出"哪张图里有不同的东西"（跨图差异定位），Common 需要找出"所有图都有的东西"（跨图共性聚合）。微调后模型在这两个任务上的大幅提升，验证了方法确实增强了跨图推理能力。Counting 的提升相对较小——可能因为计数任务还需要精确的数值推理能力，仅靠更好的跨图聚合还不够。

### 5.2 Ablation Studies and Analysis

**Cross-task generalization.** In this experiment, we train models on individual subtasks (e.g., Counting, Common, Odd-One, and Listing) to analyze their complementary roles and assess cross-task generalization. Table 5 (left) shows the results. We observe that training on the Common task generalizes well to Counting and Listing, but not to Odd-One; a similar trend is observed when training on Odd-One. This behavior is expected, as the two tasks are complementary in nature: Common requires aggregating information across multiple images, whereas Odd-One emphasizes localizing distinctive evidence within a single image. Training on Listing consistently improves performance across all other tasks, while training on Counting primarily benefits Odd-One.

> **跨任务泛化矩阵**:
>
> | 训练任务 | Common | Counting | Odd-One | Listing | 泛化模式 |
> |---------|--------|----------|---------|---------|---------|
> | All tasks | 68.5 | 37.8 | 41.0 | 34.5 | (上界) |
> | Baseline | 44.7 | 29.7 | 8.3 | 22.8 | (下界) |
> | Only Common | **73.7** | 32.0 | 3.7 | 30.7 | Common -> Counting/Listing ✓, Odd-One ✗ |
> | Only Counting | 35.8 | **39.4** | 12.2 | 20.7 | Counting -> Odd-One ✓ |
> | Only Odd-One | 34.4 | 31.8 | **53.6** | 31.1 | Odd-One -> Listing ✓, Common ✗ |
> | Only Listing | 46.0 | 29.3 | 11.1 | 28.3 | Listing -> 所有任务均有正向帮助 |
>
> **核心发现**:
> 1. **Common 和 Odd-One 是互补的**: 训练其中一个对另一个帮助很小。验证了两种推理模式的本质差异——聚合（aggregation）vs 差异定位（discrimination）。
> 2. **Listing 是"全能选手"**: 训练 Listing 对所有任务都有正向帮助——可能是因为 Listing 训练的密集信息提取和分类能力是所有任务的基础。
> 3. **Counting 最"独立"**: 仅训练 Counting 对其他任务帮助有限——计数可能需要特定的数值推理能力，与其他任务共享较少。

**Efficiency analysis.** Table 5 (mid) demonstrates that our masked attention variant achieves superior performance with substantially lower computational cost compared with vanilla attention. On the 0.5B backbone, masked finetuning reduces the FLOPs by ~81%, while outperforming full finetuning. This confirms that selectively constraining inter-image attention is both effective and efficient.

> **效率-性能的 Pareto 最优**:
>
> | 方式 | FLOPs (0.5B) | MIMIC Avg | 
> |------|-------------|-----------|
> | Baseline | 58B | 26.4 |
> | Full FT | 58B | 45.5 |
> | Masked (LoRA) | 11.2B | **49.4** |
>
> 掩码版本用 81% 更少的计算量达到了比全微调更好的性能。这证明了深层跨图 attention 不仅是冗余的，而且在某种程度上是有害的（验证了 Finding 6 推论 1——早期跨图噪声干扰深层处理）。

**Attention masking strategy.** Table 5 (right) ablates the layers at which attention masking is applied. Masking only deeper layers (layers 12-23) yields the best performance, whereas masking early layers significantly degrades accuracy. These results suggest that early layers are important for effective cross-image information aggregation.

> **注意力掩码层间消融**:
>
> | 掩码层 | Common | Counting | Odd-One | Listing | Avg |
> |--------|--------|----------|---------|---------|-----|
> | 无掩码 | 70.0 | 32.0 | 37.9 | 44.5 | 46.1 |
> | 0-23 (全部) | 64.5 | 36.1 | 20.9 | 29.2 | 37.7 |
> | 0-11 (早期) | 62.5 | 27.3 | 28.8 | 33.6 | 38.1 |
> | **12-23 (深层)** | **68.9** | **35.8** | **50.9** | **42.0** | **49.4** |
>
> **核心洞察**:
> - Mask 全部层 (0-23) 性能最差——验证了早期跨图 attention 是**必要**的
> - Mask 早期层 (0-11) 性能也不好——同样说明早期跨图交互不可或缺
> - **Mask 深层 (12-23) 最佳**——既保留了早期层的跨图语义建立，又避免了深层的噪声传播
> - 这与 Fig.4 的 attention 模式完全一致：早期层有显著跨图 attention（需要保留），深层变为单图主导（可以限制）

**Qualitative analysis.** Figure 5 visualizes answer-to-image attention for a 'Counting' example. The baseline fails to attend to relevant objects in the third image, resulting in an incorrect count. In contrast, our model exhibits balanced and semantically grounded attention across all images, leading to the correct prediction. This qualitative evidence corroborates our quantitative improvements.

### 6. Conclusions

We systematically investigated the capabilities of LVLMs in multi-image contexts through MIMIC, a novel benchmark designed to isolate specific unitary behaviors. Our analysis reveals that current SOTA models fundamentally exhibit "single-image behavior," struggling to aggregate information across inputs or track multiple concepts in the presence of visual distractors. To address this, we introduced a data-centric synthetic fine-tuning strategy and an optimization-centric attention-masking mechanism. These contributions not only resolve key failure modes but also establish new state-of-the-art results, offering a robust foundation for future research in multi-image understanding.

### 7. Limitations

While our work offers a rigorous analysis and effective solutions for multi-image LVLMs, we note the following boundaries of our study:

- **Benchmark Domain**: We constructed MIMIC using MS-COCO to maintain precise control over confounding variables (e.g., object counts, occlusion levels). While this design enables exact "unit testing" of model reasoing, extending this controlled methodology to specialized domains, such as dense documents or medical imaging, remains an exciting avenue for future research.

- **Resolution Trade-offs**: Our analysis demonstrates that reducing sequence length improves multi-image reasoing by mitigating context overload. While highly effective for semantic understanding and counting, tasks requiring pixel-perfect perception of extremely small details might benefit from adaptive resolution strategies, which were outside the scope of this study.

- **Architectural Scope**: Our proposed analysis focuses on models with open weights. While we expect conclusions to hold for closed models, additional validations (which induce budget constraints) may be useful for reinforcing our findings.

> **Limitations 的深层思考**:
>
> | 局限 | 严重程度 | 潜在应对 |
> |------|---------|---------|
> | MS-COCO 域受限 | 中 | COCO 中的物体类别、场景、遮挡模式丰富，基本覆盖通用视觉理解的核心挑战；但文档、医学等域确实需要单独验证 |
> | 分辨率 trade-off | 低-中 | 论文的核心任务是诊断多图推理，而非优化视觉编码；且注意力掩码策略已经大幅降低了计算量，为未来整合自适应分辨率留出了预算 |
> | 仅开源模型 | 中 | 这是研究 logistically 的限制；Finding 的泛化性需要通过更多模型验证，但核心机制（causal attention + 序列长度问题）是架构层面通用的 |

### Appendix Highlights

**Stitching Experiment** (Table 6): 将多张图拼接成大图（grid format），保持相似 vision token 数量。结果：性能与多图输入持平或略高。进一步验证了"序列长度 > 图形格式"的结论。

**Extended Multi-Image Interaction** (Fig. 9): 将 Finding 6 的 attention 分析扩展到 6 张图场景，观察到的模式与 4 张图一致——早期跨图 attention 丰富，深层转为单图主导。说明这是模型的**固有行为**，而非特定图像数量的现象。

**Bigger and Latest Models** (Fig. 10): 在 LLaVA-OV 72B、Qwen2.5-7B、Qwen3-VL-8B 等更大/更新的模型上也验证了 Finding 1-3——即便这些更强的模型，当目标分散在多图中时性能也会显著下降。说明问题是系统性的，而非某个模型的特例。

**Counting Balanced 细粒度分析** (Fig. 8): 微调后，当 4 个实例分散在 4 张图中时准确率从 9% 提升到 45.8%，验证了方法对信息聚合能力的提升。

---

## 三、Summary

### 实验结果层次

```
Level 1: 外部 Benchmark 验证
  ├── MuirBench: +5.7 (0.5B) / +9.6 (7B) → 通用多图能力提升
  ├── Blink: +1.5 → 感知能力提升
  ├── NLVR2: +3.1 → 自然语言视觉推理提升
  └── 5/6 benchmark 提升 → 方法泛化性好

Level 2: MIMIC 自评测验证
  ├── 0.5B: 26.4 → 49.4 (+23.0, 接近翻倍)
  ├── 7B: 54.0 → 63.8 (+9.8)
  └── Common + Odd-One 提升最大 → 跨图比较能力增强

Level 3: 消融验证
  ├── 跨任务泛化: Common-Odd 互补, Listing 通用
  ├── 效率: Masked 81% FLOPs 减少且性能超越全 FT
  └── 层间消融: 深层 mask 最佳 (与 Fig.4 一致)

Level 4: 扩展验证
  ├── Stitching: 序列长度 > 图形格式
  ├── 6 图 attention: 深层单图化是固有行为
  └── 更大模型: 问题系统性存在
```

### 最终 Takeaway

1. **多图能力的本质**不是模型大小或架构先进程度的问题，而是**训练数据和注意力设计**的问题。
2. **Attention masking 是"逆向操作"的经典案例**: 不是增加跨图 attention，而是**限制**它，反而带来更好的性能和更低的成本。
3. **数据侧和优化侧是两个层次的干预**: 前者提供正确的学习目标，后者提供高效的学习路径。两者独立有效但互补。
