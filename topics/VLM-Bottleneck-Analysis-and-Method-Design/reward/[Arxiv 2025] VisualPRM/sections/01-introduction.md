[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

本文从 MLLM 推理能力与闭源模型的差距出发，引入 Test-Time Scaling (TTS) 作为提升推理能力的途径，分析了将 TTS 适配到 MLLM 的两大挑战，并系统性地提出解决方案：数据集构建、模型训练、基准评测三位一体。

---

## 二、原始文本

With the remarkable success of Large Language Models (LLMs) in natural language processing, Multimodal Large Language Models (MLLMs) have also achieved significant advancements across various vision-language tasks. Despite their strong performance in perception and recognition, a large gap remains in reasoning capabilities between open-source and proprietary models.

> **问题动机**: 开篇定位非常精准——开源 MLLM 在感知和识别方面表现不错，但在**推理能力**上与闭源模型（GPT-4o, Claude, Gemini）仍存在"large gap"。这是整个工作的出发点。

A series of studies have explored methods to enhance reasoning abilities, focusing on the perspectives of data collection and construction, offline preference optimization, and online reinforcement learning. Additionally, another line of research investigates utilizing Test-Time Scaling (TTS) to enhance the reasoning abilities of LLMs. This approach requires the policy model to generate multiple response candidates and select the best one, based on the quality estimation of a critic model, thereby improving the response quality at the cost of higher inference time. However, TTS for MLLMs remains largely unexplored.

> **机制拆解 — 提升推理能力的四条路线**:
>
> | 路线 | 代表方法 | 核心机制 | 局限性 |
> |------|---------|---------|--------|
> | 数据建设 | OmniCorpus, MMInstruct | 构建高质量多模态训练语料 | 数据量/质量有瓶颈 |
> | Offline 偏好优化 | Step-DPO, MMPR, MixPO | 用偏好数据做 DPO/MPO | 依赖偏好数据质量 |
> | Online RL | DeepSeek-R1, REINFORCE++ | 在训练阶段用 RL 优化 | 训练成本高 |
> | Test-Time Scaling | Self-Consistency, PRM+BoN | 测试时多采样 + critic 筛选 | **MLLM 领域基本未探索**← 本文切入点 |
>
> 本文选择第四条的 **multimodal 版本**作为研究对象，这是最大的 novelty 来源——"TTS for MLLMs remains largely unexplored."

This work investigates the application of TTS for MLLMs, focusing on the Best-of-N (BoN) evaluation strategies. The challenges of adapting TTS for MLLMs involves: (1) **Lack of effective critic models.** In BoN evaluation, a critic model is required to estimate the quality of each response candidate. However, as shown in Figure 1, existing open-source MLLMs struggle to serve as critic models, leading to marginal improvements compared to models without TTS. This limitation stems from the lack of sufficient critic data in their training corpus. (2) **Lack of evaluation benchmarks for multimodal critic models.** The effectiveness of TTS heavily depends on the performance of the critic model. However, directly evaluating critics under BoN settings poses two key issues. First, the evaluation cost of BoN is expensive. Although the focus is on the performance of critic models, the policy model is required to generate N reasoning processes, with the majority of computational costs arising from the policy model. Second, BoN performance is also affected by the policy model, making it difficult to compare different critic models when paired with varying policy models.

> **机制拆解 — TTS 适配 MLLM 的两大挑战**:
>
> **Challenge 1: 缺乏有效的 critic model**
> - 现象：开源 MLLM（如 InternVL2.5-8B）直接做 critic 几乎无提升（Figure 1）
> - 根因：训练语料中缺乏 critic 相关数据
> - 解决方案：构建 VisualPRM400K → 训练 VisualPRM
>
> **Challenge 2: 缺乏 critic model 的多模态评测基准**
> - 难点 A：BoN 评测成本高（主要开销在策略模型而非 critic）
> - 难点 B：BoN 性能与策略模型耦合，不同 critic 在不同策略模型下难以公平对比
> - 解决方案：构建 VisualProcessBench——独立于策略模型的步骤级评测基准

To solve these challenges, we first introduce VisualPRM400K, a dataset comprising approximately 400K multimodal process supervision data. Each sample includes an image, a question, a step-by-step solution, and correctness annotations for each step. Specifically, we collect question prompts from MMPR v1.1 and then generate process correctness using an automatic data pipeline. This pipeline samples multiple continuations starting from a certain step and computes the expected accuracy of that step as the average accuracy of its continuations.

> **机制拆解 — VisualPRM400K 自动数据管线**: 核心思路借鉴 Math-Shepherd [79] 的 Monte Carlo 方法：
> 1. 从 MMPR v1.1 收集 (Image, Question)
> 2. 用 InternVL2.5 采样逐步解答 s = {$s_0$, ..., $s_{n}$}
> 3. 对每个步骤 $s_i$，从前缀 $s_{≤i}$ 出发采样 16 条续写 (completions)
> 4. m$c_i$ = correc$t_completions$ / 16（expected accuracy）
> 5. 如果 m$c_i$ > 0，标记为正确步骤

To facilitate the evaluation of multimodal critic models, we introduce VisualProcessBench, a benchmark for evaluating PRMs and MLLMs in detecting erroneous steps in multimodal reasoning tasks. This benchmark includes 2,866 samples with 26,950 human-annotated step-wise correctness labels. Each sample includes a multimodal reasoning question, a step-by-step solution, and correctness labels for each step. To ensure annotation accuracy, we employ human experts with at least a university degree to manually assess the correctness of each step. Unlike prior benchmarks, which require identifying only the first erroneous step, VisualProcessBench challenges models to detect all errors within a given solution. This adjustment aligns with recent advancements in model reflection abilities, helping to reduce false negatives in evaluations. Evaluation results reveal that existing open-source MLLMs struggle to accurately assess step-wise correctness, highlighting the need for improved multimodal critic models.

> **机制拆解 — VisualProcessBench 设计要点**:
> 1. **全面错误检测**: 要求模型找出解答中的**所有**错误步骤，而非仅第一个。理由：现代 MLLM 开始具备 reflection 能力，能在推理中自我纠正。
> 2. **三标签体系**: Positive (正确) / Negative (错误) / Neural (无推理内容)，评估时排除 Neural 步骤
> 3. **人工标注质量保障**: 大学学历标注员 → 13 人 3 天 → 作者抽检 10% → 不合格批次返工
> 4. **数据来源多样**: 5 个 benchmark (MMMU, MathVision, MathVerse, DynaMath, WeMath) + 5 种 MLLM 生成的解答

Building upon the dataset and benchmark, we develop VisualPRM, an advanced multimodal Process Reward Model (PRM) with 8B parameters, to serve as the critic model in BoN evaluation. Each training sample is formulated as a multi-turn chat. The first turn includes the image, the question, and the first solution step, while each subsequent turn presents a new step. The model is trained to predict the correctness of the given step at each turn. Experimental results demonstrate that VisualPRM enhances MLLM reasoning across different model families and scales. Specifically, VisualPRM improves the overall reasoning performance of MiniCPM-V2.6, QwenVL2.5-7B, InternVL2.5-8B, and InternVL2.5-78B by 8.0, 3.7, 8.4, and 5.9 points, respectively, across seven multimodal reasoning benchmarks. Additionally, we compare PRMs with Outcome Reward Models and Self-Consistency in BoN evaluation, finding that PRMs consistently outperform both approaches.

> **机制拆解 — VisualPRM 训练与推理**:
> - **训练格式**: 多轮对话，每轮输入当前步骤，输出该步骤的正确性（+ 或 -）
> - **推理方法**: 单次前向传播即得所有步骤分数（"+"/"-" 的生成概率加权），无需 autoregressive 生成
> - **步骤聚合**: 默认取所有步骤分数的平均值作为回复总分
> - **核心指标**: 7 benchmark 平均分，4 个模型家族/规模的提升

In summary, our main contributions are as follows:

(1) We introduce VisualPRM400K, a dataset comprising approximately 400K multimodal process supervision data. Building upon this dataset, we develop VisualPRM, an advanced multimodal PRM to serve as the critic model in the BoN evaluation.

(2) We construct VisualProcessBench, a benchmark designed to measure the abilities of PRMs and MLLMs to identify erroneous steps in multimodal reasoning tasks. This benchmark comprises 2,866 samples with a total of 26,950 human-annotated step-wise correctness labels.

(3) Through extensive experiments, we demonstrate that PRMs can serve as effective critic models for test-time scaling of MLLMs. Specifically, VisualPRM enhances the overall multimodal reasoning performance of MiniCPM-V2.6, QwenVL2.5-7B, InternVL2.5-8B, and InternVL2.5-78B by 8.0, 3.7, 8.4, and 5.9 points, respectively, across seven multimodal reasoning benchmarks. Furthermore, our results show that PRMs consistently outperform both ORMs and SC in BoN evaluation. Additionally, experiments on VisualProcessBench reveal that existing open-source MLLMs struggle to accurately assess the correctness of each step.

> **贡献总结**: 三大贡献形成一个完整的"数据-模型-评测"闭环：
> - 贡献 1 = 数据 + 模型（VisualPRM400K → VisualPRM）
> - 贡献 2 = 评测基准（VisualProcessBench）
> - 贡献 3 = 实验验证（BoN 有效性 + PRM > ORM > SC + 开源 MLLM 步骤判断能力差）

---

## 三、Summary

- **问题定义**: 开源 MLLM 推理能力与闭源模型差距大，TTS for MLLM 基本未探索
- **两大挑战**: (1) 缺乏有效的多模态 critic model；(2) 缺乏 critic 评测基准
- **三条贡献**: VisualPRM400K 数据集 + VisualPRM 模型 + VisualProcessBench 基准
- **核心结果**: 跨模型家族和规模，BoN 下最高 +8.4 points，PRM 全面优于 ORM 和 SC
- **关键洞察**: 通用 MLLM 天然不善于做 critic（倾向于 positive bias），需要专门训练 PRM
