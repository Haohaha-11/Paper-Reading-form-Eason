[<- 返回 README](../README.md)

# 2. Related Work

## 一、Preview

本节组织为三条平行脉络，每条都为本文的定位服务：(1) 多图 LVLM 架构演进——说明"架构上已支持多图，但能力远未达标"；(2) LVLM 评测发展——说明"有 benchmark 但缺乏精确诊断"；(3) LVLM 内部机理分析——说明"有分析但主要针对单图"。三条脉络汇合到一个结论：多图 LVLM 的根因分析是空白领域，而本文正是填补此空白的首次系统性尝试。

---

## 二、原始文本

### Multi-Image Large Vision Language Models:

Early LVLMs such as Flamingo and PaLM-E pioneered the integration of pre-trained vision encoders with powerful LLMs for VQA and captioning. Subsequent models introduced expanded instruction tuning and multi-modal pre-training techniques. More recent advancements include MiniGPT-5, Qwen2-VL, CogVLM2 and InternVL3 further advanced the field by scaling training data and model capacity and adopting more sophisticated architectural designs. While early LVLMs primarily operated on low-resolution, single-image inputs, later research significantly expanded their scope. High-resolution images are commonly processed by splitting them into fixed-resolution patches and treating them as image sequences, while videos are represented by extracting frames to form multi-image inputs. In addition, models have begun to explicitly support multi-image context, enabling reasoing across multiple visual inputs. Multi-image capability is introduced by fine-tuning single-image LVLMs on multi-image instruction-tuning data, while largely preserving the original model architecture and attention mechanisms.

> **架构演进的关键洞察**: 多数 LVLM 的多图能力是通过"在单图模型上微调多图数据"获得的，而非从架构层面重新设计。这意味着：(1) attention 机制仍是为单图设计的 causal attention；(2) positional embedding 的调整可能不足以区分不同图像之间的边界；(3) 训练数据中的多图任务可能不需要深度跨图推理，导致模型学到 shortcut。这三点直接解释了后文 Section 3 中的 Finding。

> **LVLM 架构对比**:
>
> | 模型 | 分辨率策略 | 多图支持机制 | 特殊设计 |
> |------|-----------|-------------|---------|
> | Flamingo | 固定低分辨率 | Perceiver Resampler | 早期 pioneer |
> | LLaVA-OV | 384x384 patch 分块 | 每 patch 视为独立视觉 token，追加 image separator | AnyRes 策略 |
> | Qwen2-VL | 动态分辨率 | Naive 多图拼接 + temporal 建模 | Dynamic Resolution |
> | InternVL2/3 | 动态分辨率 | 类似多图拼接处理 | 大规模预训练 |
> | CogVLM2 | 高分辨率 | Visual Expert + 多图支持 | 保留 LLM 主干 |

### Evaluation of LVLMs:

Early evaluation efforts focused on narrower domains with benchmarks such as MS-COCO, VQA, DocVQA, GQA and AI2D, primarily assessing single-image understanding and using templatized questions with limited diversity. Later work introduced more comprehensive benchmarks to evaluate a wider range of skills, e.g. SEED-Bench, MMBench and MME, which feature diverse question types and require complex reasoing abilities. Similarly, video benchmarks such as MMVU and VideoMME require models to understand temporal dynamics and to reaso across multiple frames.

> **评测发展轨迹**: 单图 VQA (MS-COCO/VQA) -> 综合单图评测 (SEED-Bench/MMBench/MME) -> 视频评测 (MMVU/VideoMME) -> 多图评测 (MuirBench/Blink)。这条脉络揭示一个 gap：从单图到多图，评测的复杂度在提升，但分析深度在下降。本文正是要在"多图评测"这个环节补充深度分析。

Closer to our work, several benchmarks have been proposed specifically for multi-image LVLMs. MuirBench introduced 12 tasks evaluating multi-image understanding, including image comparison and multi-image reasoing. Blink included 14 tasks deemed easy for humans, highlighting LVLMs limitations in truly understanding multi-image visual content. Visual Haystack focuses on retrieval-based tasks, assessing how well models can find certain concepts within a long sequence of images. Instead, we provide a more granular analysis of model performance across various controlled dimensions, such as information distribution, query complexity and distractor presence. Moreover, while prior works often repurposed existing datasets, we design our task from scratch to allow for selective performance exploration. This allows us to pinpoint specific strengths and weaknesses in current models that prior benchmarks may have overlooked and, importantly, provide deeper actionable insights on their underlying root causes.

> **MIMIC vs 已有多图 Benchmark 对比**:
>
> | 维度 | MuirBench | Blink | Visual Haystack | MIMIC (本文) |
> |------|-----------|-------|-----------------|-------------|
> | 任务数 | 12 | 14 | 1 (检索) | 4 (+ 多子设置) |
> | 数据来源 | 复用已有数据集 | 复用已有数据集 | 程序化生成 | 程序化生成 |
> | 信息分布控制 | 无 | 无 | 无 | 精确控制 s 参数 |
> | 干扰图像控制 | 无 | 无 | 无 | 精确控制 distractor 数量 |
> | 混淆变量解耦 | 否 | 否 | 否 | 是 (Balanced vs Unbalanced) |
> | 问题格式 | 多选 | 多选 | 开放式 | 开放式 |
> | 分析深度 | 性能报告 | 性能报告 | 序列长度分析 | 6 项 Finding + attention 分析 |
> | 核心哲学 | 广度评测 | 难度评测 | 检索聚焦 | **根因诊断 (root-cause analysis)** |

### Analysis of LVLMs:

Parallel to the development of benchmarks, there is growing interest in analyzing the internal mechanisms of LVLMs to better root-cause their limitations at a data and architecture level. Current studies have investigated issues such as hallucination, modality bias, and sensitivity to input phrasing. These works often involve probing the models with carefully-designed inputs to reveal their decision-making process. Only recently have such analyses expanded to multi-image LVLMs. The closest to our work is the study by Wu et al., which examines the retrieval capabilities of multi-image LVLMs as the sequence length increases, showing limitations when operating over long sequences. However, their focus is primarily on the models' ability to locate specific items within an image set and does not control conflating factors, nor seek to identify the root causes beyond data scarcity.

Instead, we systematically probe additional dimensions of multi-image understanding, such as information aggregation and multi-concept tracking. To control for confounding factors, our evaluation is designed to isolate specific unitary aspects of multi-image understanding, leading to precise conclusions and to the identification of areas for improvement. Moreover, we analyze the internal model's behavior, and complement our analysis with proposed solutions to address the identified challenges at both data and optimization levels.

> **机理分析研究的三个层次**:
>
> | 层次 | 代表工作 | 分析对象 | 本文对应 |
> |------|---------|---------|---------|
> | 单图 behavior 分析 | CLIP-DPO, HallusionBench, Throne | 幻觉、模态偏置、prompt 敏感度 | — |
> | 多图 behavior 分析 | Visual Haystack | 检索能力 vs 序列长度 | Section 3 的更多维度 |
> | 多图 internal 分析 + 解决 | **本文** | attention 模式 + 根因 + 方案 | Section 3.2 + Section 4-5 |

> **本文与 Visual Haystack 的核心差异**:
> 1. Visual Haystack 只分析检索能力；本文多了信息聚合、多概念追踪、干扰物鲁棒性
> 2. Visual Haystack 未控制混淆因素；本文通过 Balanced/Unbalanced 设置解耦实例数 vs 图像数
> 3. Visual Haystack 的根因归到"数据不足"；本文进一步分析 attention pattern 等架构层根因
> 4. 本文不仅分析问题，还提供了数据侧和优化侧的解决方案

---

## 三、Summary

- **三条脉络**: (1) 多图 LVLM 架构演变；(2) LVLM 评测发展；(3) LVLM 内部机理分析
- **核心 gap**: 三条脉络在多图场景的交汇处——即"多图 LVLM 的根因分析 + 针对性解决方案"——是空白的
- **MIMIC 的差异化定位**: 不只是又一个 benchmark，而是"可控单元测试 + 内部机理分析 + 解决方案"的完整链路
- **与最近工作的关系**: MuirBench/Blink 提供广度/难度评测，Visual Haystack 提供检索分析，本文补上了"精确诊断 + 根因追索"的关键缺失环节
