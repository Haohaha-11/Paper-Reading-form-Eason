[← 返回 README](../README.md)

# 5. Conclusion

## 一、Preview

本节总结全文工作，回顾三大贡献（数据集、模型、基准）和核心实验结果，并展望未来方向。

---

## 二、原始文本

In this work, we construct VisualPRM400K, a dataset comprising about 400K multimodal process supervision data. Building upon this dataset, we develop VisualPRM, an advanced multimodal Process Reward Model (PRM) capable of estimating the value score of each step during the reasoning process. With the Best-of-N (BoN) evaluation strategies, our model improves the reasoning abilities of existing Multimodal Large Language Models (MLLMs) across different model scales and families. Experimental results show that our model exhibits superior performance compared to Outcome Reward Models and Self-Consistency during BoN evaluation, highlighting the effectiveness of PRMs in Test-Time Scaling.

To further facilitate the development of multimodal critic models, we construct VisualProcessBench, a benchmark designed to measure the abilities of PRMs and MLLMs to detect incorrect steps in multimodal reasoning tasks. Evaluation results show that existing open-source MLLMs struggle to effectively judge the correctness of each step.

We hope that our work can inspire more future research and contribute to the development of MLLMs.

> **全文总结**:
>
> 本文工作可视为一个完整的"数据-模型-评测"系统：
>
> ```
>   VisualPRM400K           VisualPRM              VisualProcessBench
>   (自动过程监督数据)  ──→  (8B 多模态 PRM)  ──→  (人工步骤评测基准)
>        │                      │                        │
>        │ Monte Carlo 采样     │ 多轮对话 + value-based  │ 识别所有错误步骤
>        │ 400K 样本/2M 步骤    │ probability scoring     │ 2,866 样本/26,950 标注
>        │                      │                        │
>        ▼                      ▼                        ▼
>   训练 PRM 的数据基础       BoN 中筛选最优回复       评测 PRM 步骤判断能力
> ```
>
> **三个"首次"**:
> 1. 首次构建大规模多模态过程监督数据集 (VisualPRM400K)
> 2. 首次训练多模态 PRM 模型 (VisualPRM)
> 3. 首次建立多模态步骤级评测基准 (VisualProcessBench)

> **核心结论提炼**:
>
> 1. **PRM 作为 TTS 的核心 critic 是有效的**: 跨模型家族和规模，BoN 下一致提升。这打开了多模态 TTS 的研究方向——后续可以在更好的 PRM、更高效的搜索策略等方向深入。
>
> 2. **过程监督 (process supervision) 优于结果监督 (outcome supervision)**: PRM > ORM 且优势随候选池增大而扩大。这验证了"稠密信号 > 稀疏信号"在 critic 任务中的普适性。
>
> 3. **通用 MLLM 的步骤判断能力严重不足**: 开源模型存在系统性的 positive bias，即使 78B 模型也只能接近随机水平。这揭示了一个重要的研究方向——赋予 MLLM step-level critique 能力。
>
> 4. **自动标注管线是可行的规模化路径**: 400K 数据集完全由自动管线生成，成本远低于人工标注（PRM800K 的标注量），且模型效果显著。这为更大规模的过程监督数据构建提供了实践方案。

> **局限性反思**:
> - 本文聚焦 BoN 评测，未探索 PRM 在 RL 训练中的潜力（如 PPO/GRPO 使用 PRM 作为 reward）
> - PRM 仍依赖底层 MLLM 的视觉理解能力，对视觉细粒度错误（如 OCR 错误、图表细读错误）的判断能力可能有限
> - 自动数据管线的噪声问题虽然通过 value-based + threshold=0 缓解，但未根本解决
> - 步骤数限制（max 12）限制了在长链推理场景的应用

---

## 三、Summary

- **三大贡献**: VisualPRM400K (数据) + VisualPRM (模型) + VisualProcessBench (评测)
- **核心结论**: PRM > ORM > SC；PRM 跨模型泛化有效；MLLM 直接做 critic 不可行
- **关键意义**: 首个将过程奖励模型从文本域推广到多模态域的工作，开辟了多模态 Test-Time Scaling 的研究方向
- **开放问题**: PRM for RL training, 更长推理链, 细粒度错误分类, 更鲁棒的自动标注
