[← 返回 README](../README.md)

# 5. Experiments

## 一、Preview

实验部分涵盖：
- **5.1 Experiment Settings**: 基准协议、Base model、实现细节、Baselines
- **5.2 Main Results**: 检测 (Table 1) + 分割 (Table 2) 在 RegionDial-Bench 上的 7 轮表现
- **5.3 Ablation Analysis**: 消融三条信号 (Table 3/4)、单轮 vs 多轮难度对比、深度鲁棒性分析

---

## 二、原始文本

### 5.1 Experimental Settings

**Benchmark and protocol.** We evaluate under the multi-round setting in Sec. 3 on RegionDial-Bench (RefCOCO+ / RefCOCOg Multi-turn). Detailed descriptions of the dataset construction procedure, together with quantitative statistics, are provided in Appendix B. In addition, following the evaluation protocol of VisionReasoner, we also report results under the single-round setting.

**Base model.** RegionReasoner-7B is initialized from Qwen2.5-VL-7B (7B parameters). We keep the vision--language backbone intact and optimize it end-to-end with reinforcement learning; no additional task-specific heads are introduced.

**Implementation details.** RegionReasoner-7B is trained with GRPO using the rewards in Sec. 4.3. Constrained decoding enforces tag/schema validity and JSON correctness. We use the backbone's vision tokenizer and input resolution; the maximum turn depth T matches the dialogue length. Training uses a global batch size of 16 with K=8 rollout samples per prompt (per step). The initial learning rate is 1×10^{-6} with weight decay 0.01. All experiments run on 4× NVIDIA H100 GPUs; total training time is about 10 hours. Unless noted, we fix random seeds and use identical multi-turn contexts and references across methods; shared evaluation scripts ensure consistent aggregation.

> 💡 **实验配置速览**:
>
> | 参数 | 值 |
> |------|-----|
> | Base Model | Qwen2.5-VL-7B |
> | 算法 | GRPO |
> | Global Batch Size | 16 |
> | Rollout per prompt (K) | 8 |
> | Learning Rate | 1e-6 |
> | Weight Decay | 0.01 |
> | GPU | 4× NVIDIA H100 |
> | Training Time | ~10 hours |
> | 随机种子 | Fixed |

**Baselines.** We compare RegionReasoner-7B with strong VLMs and task-specialized models: Qwen2.5-VL-7B and Qwen2-VL-7B; Seg-Zero-7B (segmentation-centric); VisionReasoner-7B (structured perception--reasoning in a single-turn setting); and SegLLM (multi-round segmentation without explicit thinking or RL). All methods are evaluated under the same multi-turn protocol with reference propagation; for models without structured reasoning, we adapt prompts to accept referenced boxes.

> 💡 **Baseline 分类**:
> | 类别 | 模型 | 特点 |
> |------|------|------|
> | 通用 VLM | Qwen2-VL-7B, Qwen2.5-VL-7B | 无专门定位训练 |
> | 分割专用 | Seg-Zero-7B, SegLLM-7B | 分割导向 |
> | 结构推理 | VisionReasoner-7B | 单轮结构化推理 |
> | **本文** | **RegionReasoner-7B** | 多轮 + 显式引用 + consistency |

---

### 5.2 Main Results

**Referring detection under multi-round interaction.** Table 1 reports AP on RegionDial-Bench.

> 💡 **Table 1 批读 — 检测主结果**:
>
> **核心数字**:
> - RefCOCO+ Avg AP: 80.7 (RegionReasoner) vs 74.8 (VisionReasoner) = **+5.9 points**
> - RefCOCOg Avg AP: 78.2 vs 73.6 = **+4.6 points**
> - vs Seg-Zero: +7.6 (RefCOCO+), +7.1 (RefCOCOg)
>
> **逐轮趋势 (RefCOCO+)**:
> | Round | RegionReasoner | VisionReasoner | Gap |
> |-------|---------------|----------------|-----|
> | R1 | 89.3 | 88.3 | +1.0 |
> | R2 | 83.2 | 74.7 | +8.5 |
> | R3 | 81.6 | 75.8 | +5.8 |
> | R4 | 69.6 | 64.2 | +5.4 |
> | R5 | 61.9 | 56.3 | +5.6 |
> | R6 | 69.1 | 57.3 | +11.8 |
> | R7 | 64.7 | 47.0 | **+17.7** |
>
> **关键观察**:
> 1. **后期轮次差距远超早期**: R1 只有 +1.0, R7 达到 +17.7 → explicit citation + consistency 确实抑制了误差累积
> 2. **VisionReasoner 在 R5-R7 急剧下降**: 从 56.3 降到 47.0，而 RegionReasoner 从 61.9 降到 64.7（甚至回升）→ RegionReasoner 具有更强的深度鲁棒性
> 3. **Qwen2-VL-7B 几乎完全失效**: Avg AP 只有 6.7 → 说明多轮定位是极难的 task，未经专门训练的通用 VLM 无法胜任

**Referring segmentation under multi-round interaction.** Table 2 summarizes gIoU on RegionDial-Bench.

> 💡 **Table 2 批读 — 分割主结果**:
>
> **核心数字**:
> - RefCOCO+ Avg gIoU: 69.6 (RegionReasoner) vs 64.3 (VisionReasoner) = **+5.3 points**
> - RefCOCOg Avg gIoU: 66.5 vs 59.9 = **+6.6 points**
> - vs SegLLM: +8.9 (RefCOCO+), +9.8 (RefCOCOg)
>
> **逐轮趋势**: 与检测类似，后期轮次差距更大
> - R7 RefCOCO+: 54.6 vs 40.8 = **+13.8**
> - R7 RefCOCOg: 63.3 vs 57.6 = +5.7
>
> **SegLLM 的特殊表现**: 前几轮表现不错 (R1-R3)，但 R4 以后急剧下降 (R7 仅 30.3/25.4) → 说明有显式推理和 RL 信号对多轮鲁棒性至关重要
>
> **两条互补信号的实证体现**: 
> - R2 RefCOCO+ (73.1 vs 65.0, +8.1): 参考引用 + 一致性对"利用前轮定位信息"的帮助
> - R6-R7: 深度轮次的语义漂移被有效抑制

**Qualitative analysis.** Figure 2 shows multi-round trajectories comparing RegionReasoner vs VisionReasoner.

> 💡 **Figure 2 批读**: 定性轨迹展示了三条对比维度：
> 1. **显式引用**: RegionReasoner 在 <think> 中显式引用坐标 (e.g., "bbox=[...]")，VisionReasoner 没有
> 2. **全局-局部一致性**: RegionReasoner 的 <scene> 和 <focus> 描述与 <think> 的推理一致，VisionReasoner 可能出现语义漂移
> 3. **邻域混淆**: VisionReasoner 在后期轮次容易混淆相邻的相似物体，RegionReasoner 通过 explicit citation 保持定位稳定

---

### 5.3 Ablation Analysis

We study the contribution of each signal using Tables 3 and 4, which report single- and multi-round results on RefCOCO+ and RefCOCOg.

> 💡 **Ablation 设计**:
> | 配置 | Ref-cite | Consist. | Logic | 目的 |
> |------|----------|----------|-------|------|
> | Base only | ✗ | ✗ | ✗ | 仅 VisionReasoner 的 base rewards |
> | +Ref-cite | ✓ | ✗ | ✗ | 测试 citation 单独效果 |
> | +Ref-cite+Consist. | ✓ | ✓ | ✗ | 测试 consistency 叠加效果 |
> | +Ref-cite+Consist.+Logic (Full) | ✓ | ✓ | ✓ | 完整模型 |

**Effect of reference citation (Ref-cite).**

> 💡 **Table 3 (检测) 消融分析**:
>
> **多轮效果** (RefCOCO+):
> - Base only: 74.8
> - +Ref-cite: 78.9 (+4.1) ← citation 单独贡献
> - +Consist.: 80.2 (+5.4 total)
> - +Logic (Full): 80.7 (+5.9 total)
>
> **关键发现**:
> 1. **Ref-cite 单独贡献最大**: +4.1 points → 显式引用是多轮性能提升的核心驱动
> 2. **Consistency 叠加仍有增益**: +1.3 points → 语义一致性信号与 citation 互补
> 3. **Logic prior 小幅增益**: +0.5 points → 空间词先验有帮助但不是主力
> 4. **单轮 vs 多轮对比**: 单轮中 ref-cite 仍有提升（+0.7/+0.9）因为部分单轮 query 包含 reference region
>
> **Table 4 (分割) 消融分析**:
> - Base only: 64.3/59.9
> - +Ref-cite: 67.9/63.6 (+3.6/+3.7)
> - +Consist.: 68.3/65.8 (+4.0/+5.9 total)
> - +Logic (Full): 69.6/66.5 (+5.3/+6.6 total)
> - 趋势与检测一致，consistency 在分割上的边际增益更大 → 分割任务对语义一致性更敏感

**Effect of global--local consistency (Consist.).**

> 💡 **Consistency 的机制分析**:
> - **在 RefCOCO+ 上效果更明显**: RefCOCO+ 禁止位置词，查询以外观为中心，空间线索弱 → consistency 通过关键词锚定提供额外的语义引导
> - **核心效应: 语义锚定** — <think> 中回响的名词和物体保持了跨轮轨迹聚焦于同一实体 → 限制了注意力偏离 → 在杂乱场景中稳定分割质量
> - **与 Ref-cite 的互补**: Ref-cite 处理"在哪个位置"，Consist. 处理"是什么物体"

**Effect of the logic prior.**

> 💡 **Logic prior 的作用机制**:
> - 鼓励 <think> 中包含 inside, next to, left of 等短语
> - 增加部分正确推理的奖励密度
> - 促使模型显式表达空间关系 → 推理更容易被验证
> - 当两个候选在视觉上相似时帮助模型恢复
> - **后期轮次增益更明显**: 轮次越深，空间关系的显式表达越重要

**Depth robustness and single- vs. multi-round difficulty.** Across datasets and tasks, single-round results (Round 1) are consistently higher than their multi-round counterparts, which reflects an intrinsic difficulty gap rather than an artifact of a particular model. In the single-round setting, the policy only needs to resolve one query against the image. In contrast, later rounds must both interpret the current query and correctly reuse and propagate previously predicted boxes as references. Any localization error at an early turn is carried forward and compounds over subsequent turns, so the effective difficulty increases with turn depth. All compared methods exhibit this depth-dependent degradation, highlighting multi-turn error accumulation and robust reference propagation as central challenges for grounded dialogue. The full RegionReasoner configuration degrades more slowly with turn index than any variant without citation or without consistency: its trajectories remain parseable and self-consistent, which limits the accumulation of small localization errors over long dialogues.

> 💡 **深度鲁棒性分析 — 为什么 RegionReasoner 在多轮中退化更慢**:
>
> **误差累积机制**:
> ```
> Round 1 error (ε₁) → 传播到 Round 2 作为 B_2^ref
> → Round 2 基于错误参考推理 → 产生 ε₂ > ε₁
> → ... → Round N 误差被复合放大
> ```
>
> **RegionReasoner 的缓解策略**:
> 1. **Explicit citation**: 即使前轮定位稍有误差，<think> 中的显式引用使模型能感知到使用的是哪个参考框 → 可以显式地与当前图像进行比对校准
> 2. **Global--local consistency**: 通过关键词锚定保持跨轮语义聚焦 → 减小因参考偏移导致的"跟丢"概率
> 3. **可解析轨迹**: 结构化输出使模型自身的推理保持自洽 → 小定位误差的累积受限

---

## 三、Summary

- **检测**: RegionReasoner 在 RefCOCO+ 和 RefCOCOg 上分别以 +5.9 和 +4.6 AP 超越第二好的 baseline (VisionReasoner)
- **分割**: respectivement +5.3 和 +6.6 gIoU 优势，远超 SegLLM (+8.9/+9.8)
- **关键模式**: 后期轮次 (R5-R7) 增益远大于早期 (R1-R2)，证明 explicit citation + consistency 有效抑制误差累积
- **消融**: Ref-cite 贡献最大（+4.1 AP），Consist. 叠加仍有增益（+1.3），Logic 小幅增益（+0.5）；三信号互补
- **深度鲁棒性**: RegionReasoner 随轮深退化最慢，完整配置 > 任何部分信号配置
