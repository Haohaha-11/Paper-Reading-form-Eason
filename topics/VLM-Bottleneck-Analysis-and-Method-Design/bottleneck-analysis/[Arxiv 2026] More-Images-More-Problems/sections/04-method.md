[<- 返回 README](../README.md)

# 4. Method

## 一、Preview

本节提出两种互补的微调策略，分别对应 Section 3 中揭示的不同根因：(1) **数据侧**：用 MIMIC pipeline 在 OpenImages 上生成 ~198K 合成多图训练数据，提供显式的跨图推理监督——针对 Finding 3/4/5（信息聚合/干扰/多概念）；(2) **优化侧**：基于 Finding 6（深层跨图 attention 衰减）设计层间注意力掩码，限制深层 vision token 只看同图 token——针对 Finding 1/2/6（序列长度/单图行为/attention 模式）。两种策略可以独立使用也可以结合，注意力掩码版本使用 LoRA 实现参数高效。

核心设计哲学是：**不是从零重新设计架构，而是在现有 LVLM 框架内通过数据和优化手段修正已诊断出的 failure mode**。

---

## 二、原始文本

In the previous section, we identified key limitations of LVLMs on multi-image tasks via zero-shot evaluation using the MIMIC benchmark. Here, we investigate targeted fine-tuning strategies derived from our findings and aimed at improving multi-image reasoning capabilities. In particular, we explore two complementary approaches: a data-centric fine-tuning strategy using synthetically generated multi-image data, and an optimization-centric attention-masking strategy.

### Strategy 1: Data-Centric — Synthetic Multi-Image Training Data

Multi-Image Finetuning: We fine-tune LLaVA-OV models on a unified training dataset composed of samples procedurally generated using the MIMIC pipeline (see section 3.1) together with the original LLaVA-OV multi-image instruction-tuning data (approximately 580K samples). Unlike the MIMIC benchmark used for evaluation, our fine-tuning data is built from OpenImages and provides explicit supervision for cross-image reasoning. It contains approximately 198K samples, with sequence lengths of up to 10 images (see appendix), deliberately exposing models to substantially longer vision-token sequences. All four MIMIC tasks are included to encourage diverse multi-image reasoning behaviors.

> **数据侧策略的设计考量**:
>
> | 设计要素 | 具体选择 | 理由 |
> |---------|---------|------|
> | 数据源 | OpenImages v7（非 MS-COCO） | 防止评测数据泄漏；OpenImages 标注更丰富 |
> | 数据量 | ~198K MIMIC 样本 | 与原有 580K 多图数据比例约 1:3，平衡多图能力与通用能力 |
> | 序列长度 | 最多 10 张图 | 远超评测中常见的 2-7 张，让模型适应长序列 |
> | 任务覆盖 | 全部 4 个 MIMIC 任务 | Counting, Listing, Common, Odd-One 全覆盖，促进多样化推理行为 |
> | 数据格式 | LLaVA 风格多轮对话 + 选项式回答 | 与原有训练格式兼容，降低格式冲突 |
> | 训练起点 | LLaVA-OV Single-Image (Stage 2.1) | 已有高质量单图能力，在此基础上补充多图能力 |
> | 冻结部分 | Vision encoder | 保留视觉编码能力，仅调整 LLM + projector |

> **为什么需要显式的跨图推理监督？**
>
> Finding 6 的推论 4 指出：原有训练数据中的多图任务可能不需要深度跨图推理，模型学到了 shortcut。MIMIC 合成数据的设计恰恰反其道而行之——每个样本**必须**通过跨图聚合/比较才能正确回答。例如：
> - Counting 任务：信息分散在多张图中，必须全部查看并累加
> - Common 任务：必须逐张图比较才能找到共同元素
> - Odd-One 任务：必须先聚合再找差异
>
> 这种"必须跨图"的设计强制模型学习真正的 multi-image integration，而非 shortcut。

### Strategy 2: Optimization-Centric — Attention Masking

Attention Masking: Our analysis shows that inter-image attention diminishes in deeper layers (see fig. 4). Motivated by this, we apply layer-wise attention masking during fine-tuning, restricting vision tokens to attend only to tokens from the same image in selected layers, while leaving text-token attention unchanged. This design offers two key benefits. First, it reduces unnecessary cross-image interactions, leading to a more efficient model with lower computational cost (see table 7 and fig. 7 of appendix). Second, it encourages cleaner image-local representations in deeper layers, which empirically improves performance across several benchmarks. For this setting, we employ LoRA-based fine-tuning to further improve parameter efficiency.

> **注意力掩码的机制拆解**:
>
> ```
> Standard Attention (all-to-all):
> ┌──────┬──────┬──────┬──────┬────────┐
> │ Img1 │ Img2 │ Img3 │ Img4 │ Text   │
> ├──────┼──────┼──────┼──────┼────────┤
> │  ✓   │  ✓   │  ✓   │  ✓   │   ✓    │  ← Img1 tokens can attend everywhere
> │  ✓   │  ✓   │  ✓   │  ✓   │   ✓    │  ← Img2 tokens can attend everywhere
> └──────┴──────┴──────┴──────┴────────┘
>
> Masked Attention (vision restricted to same-image):
> ┌──────┬──────┬──────┬──────┬────────┐
> │ Img1 │ Img2 │ Img3 │ Img4 │ Text   │
> ├──────┼──────┼──────┼──────┼────────┤
> │  ✓   │  ✗   │  ✗   │  ✗   │   ✓    │  ← Img1 tokens: only self + text
> │  ✗   │  ✓   │  ✗   │  ✗   │   ✓    │  ← Img2 tokens: only self + text
> │  ✗   │  ✗   │  ✓   │  ✗   │   ✓    │  ← Img3 tokens: only self + text
> │  ✗   │  ✗   │  ✗   │  ✓   │   ✓    │  ← Img4 tokens: only self + text
> │  ✓   │  ✓   │  ✓   │  ✓   │   ✓    │  ← Text tokens: global (unchanged)
> └──────┴──────┴──────┴──────┴────────┘
> ```
>
> **为什么这样设计？**
> 1. Vision token 之间 block-diagonal 限制：减少 Finding 6 推论(1)中提到的"早期跨图噪声"和推论(2)中"误差传播"
> 2. Text token 保持全局 attention：语言理解需要跨图像信息（如"在所有图中都出现"），文本 token 作为"信息桥"仍然可以连接不同图像
> 3. 只在深层 mask：基于 Table 5(right) 的消融结果——mask 早期层（0-11）严重损害性能，mask 深层（12-23）最佳

> **注意力掩码 vs 全微调的设计取舍**:
>
> | 方式 | 优势 | 代价 |
> |------|------|------|
> | 全微调 (Full FT) | 所有参数适配多图任务，理论上限更高 | 计算量大 (~58B FLOPs for 0.5B)，存储多份 checkpoint |
> | 注意力掩码 + LoRA | 参数高效 (仅训练 LoRA adapter)，计算量减少 ~81% | 模型容量受限，可能在某些任务上不如 full FT |

> **FLOPs 减少的原理**:
>
> 标准 self-attention: O(($N_t$ + $N_v$)^2 * d)
> 掩码后的 attention: O($N_t$ * ($N_t$ + $N_v$) + M * ($N_{v}$/M)^2) * d = O($N_t^2$ + $N_t$*$N_v$ + $N_{v}^2$/M) * d
>
> M=10.4 张图时，$N_v^2$ -> $N_{v}^2$/M 意味着视觉 token 间的 attention 计算量减少约 10x。加上 MLP 开销后总体减少约 81% (0.5B model)。

### Implementation Details

> **训练配置速览**:
>
> | 参数 | Data-Centric (Full FT) | Attention Masking (LoRA) |
> |------|----------------------|--------------------------|
> | 训练起点 | LLaVA-OV Stage 2.1 | LLaVA-OV Stage 2.1 |
> | 冻结部分 | Vision encoder | Vision encoder |
> | 训练部分 | LLM + Projector | LoRA on LLM + Projector |
> | Learning rate | 2.5e-6 | 2.5e-5 |
> | Batch size | 128 | 128 |
> | Schedule | Cosine + warmup 0.03 | Cosine + warmup 0.03 |
> | LoRA rank | N/A | 128 |
> | GPU | 8x H100 80GB | 8x H100 80GB |
> | 注意力掩码层 | N/A | Layers 12-23 |

> **出发点选择——为什么从 Single-Image Stage 而非 Multi-Image Stage 开始？**
>
> 论文特意使用 LLaVA-OV 的单图阶段 (Stage 2.1) 作为起点，而非已有多图微调阶段的 checkpoint。这个选择的含义是：本文的训练策略是"从零赋予多图能力"，而不是在已有（但不够好的）多图能力上修补。这更能体现方法本身的 effectiveness。

---

## 三、Summary

### 两种策略的互补关系

| 策略 | 针对的 Problem | 针对的 Finding | 机制 | 优势 |
|------|---------------|---------------|------|------|
| Data-Centric | 训练数据中缺乏显式跨图推理监督 | Finding 3, 4, 5 + 推论4 | 程序化生成强制跨图聚合的训练样本 | 提供正确的学习信号 |
| Optimization-Centric | 深层跨图 attention 衰减 + 序列长度膨胀 | Finding 1, 2, 6 | 深层 block-diagonal attention mask | 减少噪声 + 计算量 |

### 设计哲学

1. **诊断驱动的方法设计**: 每个方法组件都有 Section 3 的 Finding 作为直接动机
2. **架构无损**: 不修改模型主干结构，不引入新模块，完全在现有框架内解决问题
3. **互补性**: 数据侧解决"学什么"，优化侧解决"怎么学更高效"
4. **可插拔**: 两种策略可独立或组合使用
