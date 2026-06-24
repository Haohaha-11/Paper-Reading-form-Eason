[← 返回 README](../README.md)

# 2. Related Work

## 一、Preview

本节从三条脉络组织相关工作：(1) MLLM 的发展——从架构、数据、训练算法三个维度；(2) Process Reward Model 的系列研究——从人工标注到自动数据管线；(3) Reward Model 评测基准——从偏好判断到步骤级评测。每条脉络的终点都指向同一个 gap：多模态 PRM 和步骤级多模态评测仍为空白。

---

## 二、原始文本

### Multimodal Large Language Models

A wide range of efforts has been made to advance the development of MLLMs, including improvements in model architecture, data construction, and training algorithms. From an architectural perspective, several studies employ connectors to align visual embeddings from Vision Foundation Models (VFMs) with the latent space of LLMs, achieving promising performance. Another series of works extends pre-trained LLMs with additional layers to fuse visual features, reducing the number of required visual tokens while introducing extra training cost. In terms of data construction, recent studies have made significant progress. For example, Omni-Corpus offers a noisy but large-scale multimodal corpus for pre-training, while MMInstruct provides an open-source, high-quality instruction-tuning dataset. Additionally, MMPR constructs a preference dataset focusing on multimodal reasoing abilities. Regarding training algorithms, the InternVL2.5 series proposes square loss and Mix Preference Optimization to enhance MLLM capabilities. Despite these advancements, existing works primarily focus on the training process of MLLMs, leaving Test-Time Scaling (TTS) for MLLMs largely underexplored.

> **机制拆解 — MLLM 发展的三个维度**:
>
> | 维度 | 代表方法 | 核心做法 | 与本文的关系 |
> |------|---------|---------|------------|
> | 架构 | InternVL (connector), Flamingo/CogVLM (交叉注意力层) | 桥接视觉编码器与 LLM | 本文基于 InternVL2.5 架构 |
> | 数据 | OmniCorpus, MMInstruct, MMPR | 预训练→指令微调→偏好数据 | VisualPRM400K 使用 MMPR v1.1 的 question prompts |
> | 训练算法 | Square Loss, MixPO | 单模态偏好优化 | 本文关注的是**测试时优化 (TTS)**，与训练算法互补 |
>
> **核心 gap**: "TTS for MLLMs largely underexplored" —— 所有 prior work 都聚焦在训练阶段改进，而本文是测试时优化的多模态扩展。

### Process Reward Models

Reward models play a crucial role in Reinforcement Learning (RL) and TTS. Outcome Reward Models (ORMs) directly assign an overall score to the given response. In contrast, Process Reward Models (PRMs) first estimate the quality of each step in the given response and then aggregate them into a final score. PRM800K is the first open-source process supervision dataset, entirely annotated by human experts. To reduce annotation costs, Math-Shepherd and OmegaPRM introduce a Monte Carlo sampling-based data pipeline to automatically estimate the quality of each step. Despite these advancements in natural language processing, multimodal PRMs remain largely underexplored. In this work, we introduce VisualPRM400K, the first multimodal process supervision dataset, and develop VisualPRM, a multimodal PRM trained on this dataset.

> **机制拆解 — PRM 发展的两代方法**:
>
> | 代际 | 代表方法 | 标注方式 | 成本 | 局限性 |
> |------|---------|---------|------|--------|
> | 第一代 | PRM800K [39] | 全人工标注 | 极高 | 规模受限（仅 800K 步），仅限数学 |
> | 第二代 | Math-Shepherd [79], OmegaPRM [51] | Monte Carlo 自动标注 | 低 | 数据质量略低于人工标注 |
> | **本文** | **VisualPRM400K** | **Monte Carlo 自动标注** | **低** | **首个多模态 PRM 数据集** |
>
> **核心 gap**: 从 PRM800K → Math-Shepherd → VisualPRM400K，技术路线为：纯语言数学 → 纯语言数学 + 自动标注 → **多模态 + 自动标注**。VisualPRM 是将 PRM 从文本域迁移到多模态域的关键一步。
>
> **ORMs vs. PRMs 对比**:
> | 特性 | ORM | PRM |
> |------|-----|-----|
> | 评分粒度 | 整体回复一个分数 | 每个步骤一个分数，再聚合 |
> | 信号密度 | 稀疏（仅结果正确与否） | 稠密（每个步骤都有反馈） |
> | 定位错误 | 无法定位具体哪步错 | 可以精确定位错误步骤 |
> | 训练数据 | 仅需结果正确性标签 | 需要步骤级正确性标签 |

### Benchmarks for Reward Models

The evaluation of reward models (RMs) is a crucial research topic. A series of benchmarks have been proposed to assess the effectiveness of RMs, typically formulated as a binary preference judgment task. Building on this, subsequent work extends the evaluation settings and includes both pairwise and Best-of-N evaluations, providing a more comprehensive evaluation of RM performance. With the rapid advancement of PRMs, a series of benchmarks have been introduced to evaluate their step-wise judgment capabilities. Despite these developments, there remains a lack of a multimodal process benchmark. To bridge this gap and support the development of multimodal PRMs, we introduce VisualProcessBench, a benchmark designed to evaluate the ability of PRMs and MLLMs to detect erroneous steps in multimodal reasoing tasks.

> **机制拆解 — RM 评测基准的发展脉络**:
>
> | 评测层次 | 代表 Benchmark | 任务格式 | 与本文的关系 |
> |---------|---------------|---------|------------|
> | 偏好判断 | RewardBench [33], VL-RewardBench [37], RM-Bench [44] | 二选一偏好判断 | 只能评估整体偏好，无法评估步骤级别判断 |
> | 综合评测 | RMB [97] | Pairwise + BoN | 评测更全面，但仍非步骤级且缺少多模态版本 |
> | 步骤级评测（文本） | PRMBench [69], ProcessBench [96] | 寻找第一个错误步骤 | 局限于文本数学，且只找第一个错误 |
> | **步骤级评测（多模态）** | **VisualProcessBench (本文)** | **识别所有错误步骤** | **首个多模态步骤级评测基准** |
>
> **核心 gap**: "lack of a multimodal process benchmark" —— VisualProcessBench 的多重差异化：(1) 多模态；(2) 识别**所有**错误（非仅第一个）；(3) 覆盖 5 个多模态推理 benchmark。

---

## 三、Summary

- **MLLM 发展**: 架构-数据-训练三维度已有大量探索，但 TTS for MLLM 仍为空白 → 本文立足点
- **PRM 演进**: 人工标注 → 自动标注 → 多模态自动标注 → 本文填补了最后一步
- **RM 评测演进**: 偏好判断 → 综合评测 → 文本步骤级 → **多模态步骤级** → 本文填补
- **差异化定位**: VisualPRM 在三个维度上同时填补空白：多模态 PRM 模型 + 多模态过程监督数据 + 多模态步骤评测基准
