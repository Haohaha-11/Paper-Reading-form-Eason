[← 返回 README](../README.md)

# 1. Introduction

## 一、Preview

引入 RLVR 在 VLM 场景下的 sparse reward 困境——序列级奖励无法定位推理链内哪一步感知或推理出错。提出 perception-centric PRM 的动机：视觉推理中的中间步骤常是可直接与图像比对的 perceptual claim（物体、属性、空间关系），天然适合做自动化 hallucination 检测。

---

## 二、原始文本

Vision-language models (VLMs) [3, 7, 12] deliver strong results across tasks such as multimodal mathematics [26, 38], chart analysis [24, 27], and general VQA [50]. However, they still falter on complex visual reasoing tasks, where multi-step chains of thought can be brittle and produce perceptual or logical mistakes [6, 11, 42]. To improve the performance, reinforcement learning with verifiable rewards (RLVR) [13, 33, 35] has become a widely used post-training strategy. Built on policy-gradient methods like PPO and GRPO, RLVR assigns outcome-level rewards to explicit reasoing traces and optimizes the policy toward more consistent, robust multi-step visual reasoing.

> 💡 **背景梳理**: VLM 在复杂视觉推理任务上的根本瓶颈不是"推理能力不够"，而是多步推理链中的感知或逻辑错误难以被 RLVR 的 outcome-level reward 捕获。一句话总结现状：模型在做 chain-of-thought 时会"走神"（插入幻觉或漂移），但 reward 信号只看最后的答案对不对。

Despite these advances, outcome-level supervision in RLVR is poorly matched to the inherently multi-step nature of visual reasoing. In fact, sequence-level rewards are too coarse to identify which perception or reasoing steps went wrong, creating a hard credit-assignment problem. In practice, VLMs often insert hallucinated objects or spatial relations and drift from the image context mid-chain [1, 19, 20, 22, 53], but only the final reward offers little guidance about whether the failure arose from visual grounding or subsequent logic. Thus, the sparse-reward regime ultimately bottlenecks RLVR's gains on VLMs [48].

> 💡 **机制拆解 — Sparse Reward 瓶颈**:
>
> | 问题层面 | 具体表现 | 对 RLVR 的影响 |
> |---------|---------|---------------|
> | Credit-Assignment | 所有 token 共享同一个 sequence-level advantage | 无法区分"哪一步好/坏" |
> | Hallucination Drift | 中间步骤插入幻觉对象或空间关系并偏离图像上下文 | RLVR 无能力在中间步骤发出纠正信号 |
> | Failure Attribution | 最终错误可能来自 visual grounding 也可能来自逻辑推理 | sequence-level reward 完全无法定位根本原因 |
>
> 这就是本文的核心 gap：RLVR 给的所有东西（最终奖励）是必要的，但远不充分。需要一个更精细的监督信号来"拆解"这个最终奖励。

To overcome the sparse-reward limitation, we introduce a process reward model (PRM) that supervises intermediate steps rather than only the final outcome [39]. Prior work shows that PRMs can effectively guide both training and inference by rewarding stepwise, chain-of-thought correctness [21, 55]. However, building a high-quality PRM is difficult because step-level annotations are expensive and some steps are only verifiable after later derivations, complicating labeling and consistency [17, 54]. Fortunately, in visual reasoing many intermediate steps are perceptual claims (e.g., objects, attributes, or spatial relations) that can be grounded directly in the image, enabling automatic checks for "image-text misalignment" (hallucination). Therefore, it is promising to develop a perception-centric PRM that detects and explains such misalignments to provide fine-grained feedback, alleviating sparse-reward issue and improving learning of the reasoing ability.

> 💡 **机制拆解 — 为什么 Perception-centric 的 PRM 天然可行？**:
>
> 传统 PRM 困境：数学证明的中间步骤往往要看到后续推导才能判断对错（如 "假设 x=5"，但如果后面发现 x=2 这个步骤其实是对的——只是假设不同）。标注这样的 step-level correctness 成本极高。
>
> 视觉推理的独特优势：中间步骤是 perceptual claims（"杯子上有一个红色的盖子"），可以直接与图像比对！这在传统 PRM 场景中是无法做到的。核心洞察是：
> - 数学推理：中间步骤的可验证性取决于逻辑上下文
> - 视觉推理：中间步骤的可验证性取决于图像事实 —— 可以直接做自动化校验
>
> 这种"图可验证"的属性让 perception-centric PRM 比通用 PRM 更容易构建高质量训练数据。

To operationalize this, we first define a perception-level error-finding schema for a perception-centric PRM. We curate training queries from perception-intensive settings — such as goal-directed visual search and referring-expression grounding — and use a strong LLM to produce structured annotations that mark image-text misalignments (hallucinatory spans and their visual counter-evidence). After supervised fine-tuning on this corpus, the PRM can reliably flag hallucinations that arise within multi-step rationales and return well-structured feedback. Building on this, we integrate the PRM into RLVR by decomposing the sequence-level advantage and assigning fine-grained, token-level penalties to spans identified as hallucinatory, yielding more precise credit assignment than GRPO alone. Finally, based on PRM's structured outputs, we employ a simple Truncation-Regeneration loop at inference. In this way, suspect spans are pruned and regenerated, trading a bit more compute for stronger factual grounding.

Experimental results demonstrate that, compared to direct GRPO, our training method significantly enhances the model's perceptual capabilities, boosting performance on perception-centric tasks. Furthermore, we observe a surprising and significant generalization effect: even without applying PRM supervision during the training for complex reasoing tasks, this foundational improvement in perception nonetheless generalizes, leading to a comprehensive enhancement of the model's overall reasoing abilities.

> 💡 **核心发现 — "能力迁移"现象**:
> 这是一个非常有意思的实验观察：训练阶段只在 Visual Search 等感知密集任务上使用 Perceval 做 token-level 监督，对数学/图表等复杂推理任务照常使用 GRPO（不做 PRM 干预）。但测试时发现复杂推理任务也有显著提升！
>
> 作者的解读是：复杂推理任务（如 MathVision、ChartQA）的底层依赖于精确的细粒度感知能力——你需要先"看对"图表上的数据点，才能"算对"最终的数值。把基础感知能力做扎实了，上层复杂推理自然受益。这从另一个角度印证了"感知是 VLM 推理的公共瓶颈组件"这一观点。

Our main contributions are as follows:

- We propose a novel, perception-centric process reward model (PRM) that can explicitly identify perception errors in the reasoing process.
- We introduce a fine-grained, token-level advantage reallocation framework that integrates our PRM with GRPO, to solve the sparse reward issue.
- We design a test-time iterative refinement strategy that leverages our PRM to actively detect and correct perceptual errors from the policy model.

---

## 三、Summary

- **问题定义**: RLVR 的 sparse reward (sequence-level advantage) 导致 credit-assignment 困难
- **核心洞察**: 视觉推理的中间步骤常是 perceptual claims，可直接与图像比对 → perception-centric PRM 天然可行
- **方案**: Perceval = Error-finding PRM + Token-level Advantage Reallocation + Test-time Refinement
- **三项贡献**: (1) 感知中心 PRM；(2) Token-level advantage 重分配框架；(3) 推理时迭代精炼策略
- **关键发现**: 仅在感知任务上施加 PRM 监督，复杂推理也能受益——感知是 VLM 推理的公共基础
