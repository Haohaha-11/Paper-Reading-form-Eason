[<- 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文从 LVLM 研究的两个推进方向（benchmark 构建 vs. 内部机理分析）出发，指出两个方向在多图场景下都处于空白状态。通过引入 MIMIC benchmark 和后续的系统性实验，发现当前 LVLM 本质上是"单图模型行为"——即使支持多图输入，仍然倾向于将多张图当作单张图来处理。这个核心洞察驱动了后续的数据侧和优化侧两个方法设计。

---

## 二、原始文本

Current Large Vision Language Models showcase impressive vision-language understanding capabilities. Most of these models are built upon pre-trained vision encoders and large language models (LLMs). While early efforts primarily focused on single images, recent works have extended them to support multiple images and videos by incorporating temporal modeling and adjusting the positional embeddings.

Despite their success, LVLMs continue to face significant challenges. Progress towards identifying and addressing these challenges can follow two primary avenues: the development of comprehensive evaluation benchmarks and the study of the models' inner workings. To date, research in both areas has predominantly focused on the single-image setting. While early efforts have introduced benchmarks for multi-image scenarios, a comprehensive, in-depth analysis to ascertain the true efficacy of these models and identify the root causes of their limitations is notably absent.

> **研究 gap 定位**: 两个推进方向——(1) benchmark 构建和 (2) 模型内部机理分析——在多图场景下都是空白的。已有 MuirBench、Blink 等 benchmark 提供了初期的多图评测，但缺乏对 failure mode 根因的系统分析。Wu et al. (2025) 的 Visual Haystack 虽分析了多图检索能力随序列长度的退化，但**未控制混淆因素**，也未深入追索架构层根因。本文的核心差异化在于：精确控制混淆变量，做"单元测试"式的诊断。

In this work, we address this gap by conducting a systematic study of LVLMs in multi-image contexts. We first analyze and characterize common failure modes using a newly proposed benchmark, and then seek to mitigate these limitations using two novel complementary fine-tuning strategies.

> **论文的"三段式"叙事结构**:
> 1. **诊断** (Section 3): 用 MIMIC benchmark 程序化生成多图任务，精确控制信息分布、干扰存在、实例分布、序列长度等维度，系统诊断出 6 大 failure mode
> 2. **分析根因** (Section 3.2): 通过 attention pattern 可视化 + 序列长度控制实验，将根因定位到：(a) 深层跨图 attention 衰减；(b) 因果注意力机制下的误差传播；(c) 训练数据的单图偏置
> 3. **针对性解决** (Section 4-5): 数据侧合成多图训练 + 优化侧注意力掩码，直接针对根因设计

Our in-depth analysis is performed on the newly introduced MIMIC benchmark. Built from MS-COCO, using its bounding boxes and class labels, MIMIC procedurally generates multi-image sequences by leveraging per-image annotations that give fine-grained control over information spread, distractor presence, object-instance distributions, sequence length, and query complexity, while providing unambiguous ground-truth answers for robust, decorrelated analysis of the model's strengths and weaknesses.

> **MIMIC 的核心设计哲学**: 程序化控制 (procedural control)。不是从已有数据集中采样，而是根据标注"程序化生成"多图序列。这使得每个可控维度可以独立变化，做到了传统 benchmark 无法实现的"解耦分析 (decorrelated analysis)"。类比：MIMIC 对多图 LVLM 的意义，类似于单元测试对软件开发的意义。

Using both quantitative and qualitative assessments, our study reveals that current state-of-the-art LVLMs struggle to effectively aggregate information across multiple images, are unable to track/attend to multiple concepts simultaneously, while being susceptible to distractors. We attribute these shortcomings to a combination of factors, including limitations in multi-image sequence modeling, training data biases, poor inter-image communication induced by the causal attention and the inherent complexity of multi-image reasoing tasks.

> **Failure mode 的四个根因层次**:
> 1. **序列建模局限**: causal attention 机制下，后续图像的 token 可能累积来自前面图像的噪声/错误信息
> 2. **训练数据偏置**: 训练数据中的多图任务不需要深度跨图推理，模型学到 shortcut
> 3. **跨图通信不足**: 深层 attention 衰减为单图主导，跨图信息无法有效传播
> 4. **任务固有复杂度**: 多图推理天然比单图更难——需要同时维护多个图像的表示并进行比较

Finally, to address the identified problems, we propose two new finetuning strategies: (1) a data-centric approach that generates targeted multi-image training examples to provide rich, multi-image supervision derived from OpenImages; and (2) an optimization-centric approach that leverages layerwise attention analysis to derive an attention-masking scheme tailored for multi-image inputs.

In summary, our main contributions are:

- We introduce MIMIC, a comprehensive evaluation framework for multi-image LVLMs that probes various aspects of model performance through a controlled and diverse set of tasks.
- We conduct an extensive evaluation of several state-of-the-art LVLMs using MIMIC, uncovering critical insights into their capabilities and limitations in multi-image settings.
- We propose a novel data-centric finetuning approach using synthetically generated multi-image data, alongside an optimization-centric attention-masking strategy, both of which significantly enhance model performance in multi-image contexts.
- We set new state-of-the-art results on existing multi-image benchmarks, demonstrating the effectiveness of our proposed methods.

---

## 三、Summary

- **研究 gap**: 多图 LVLM 缺乏(1)系统 benchmark 诊断和(2)内部机理分析
- **核心洞察**: 当前 LVLM 本质上是"单图模型行为"——即使架构支持多图，行为仍偏向单图
- **MIMIC 的独特价值**: 程序化控制实现解耦分析，传统 benchmark 做不到
- **方法论**: 诊断 -> 分析根因 -> 针对性解决（三段式叙事）
- **两种方案**: 数据侧（合成多图训练数据）+ 优化侧（注意力掩码），互补且各自针对不同的根因
