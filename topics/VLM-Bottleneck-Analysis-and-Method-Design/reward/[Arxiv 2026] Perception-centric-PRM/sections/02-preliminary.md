[← 返回 README](../README.md)

# 2. Preliminary

## 一、Preview

本节铺设定理：VLM 架构、GRPO 公式（Eq.1-2）、以及将感知错误定位转化为 process reward 形式化问题定义。尤其是 GRPO 的 advantage 计算公式——全文最核心的公式——需要仔细理解，因为它就是之后要被 token-level 重分配的对象。

---

## 二、原始文本

We introduce foundational concepts and notations used throughout this paper: the architecture of vision-language models (VLMs), the reinforcement-learning framework with verifiable rewards (RLVR) which our method builds on, and our problem statement for designing a perception-centric process reward model.

### Vision-Language Models

A vision-language model (VLM) accepts multimodal input, typically an image v and a text query q, and generates the text output o, denoted as $π_{θ}$(o | q, v). For reasoning tasks, the text output is generally a chain of language reasoning steps. Typical architecture combines a visual encoder (e.g. ViT) to embed I and a large language model (LLM) to decode the output. Typically, the two modalities are linked via a connection layer.

### Reinforcement Learning with Verifiable Rewards

RL with verifiable rewards (RLVR) has become the key technique to improve the performance of VLMs in reasoning tasks [45]. It aims to train the VLM to not only generate plausible outputs but also satisfy measurable criteria (e.g. correctness, spatial consistency). One algorithm is Group Relative Policy Optimization (GRPO) [33]: given the input prompt q and image v, a reference policy $π_{θ}$(o | q, v) samples multiple responses {$o_{i}$}. Each response will be assigned with a scalar reward $R_{i}$ from the verified function or reward model. The advantage of the i-th response is calculated by normalizing its reward relative to the group:

$\hat{A}_i = \frac{ R_{i}  - \text{mean}(\{ R_{j} \}_{j=1}^G)}{\text{std}(\{ R_{j} \}_{j=1}^G)}$ (1)

> 💡 **公式批读 — GRPO 的 Advantage (Eq.1)**:
> - **输入**: G 条由同一 prompt 生成的响应，每条响应 i 被赋予一个标量奖励 $R_i$
> - **操作**: 组内标准化 — 减去组内均值，除以组内标准差
> - **直观理解**: 在同一个问题的一组回复中，"比平均好的"得到正 advantage，"比平均差的"得到负 advantage
> - **关键局限**: Eq.1 输出的 Â_i 是一个**标量**，对响应 i 中的所有 token 都是同一个值。这意味着正确和错误的 token 被"一视同仁"地对待。

Note that this advantage Â_i is a sequence-level signal, which is constant for all tokens within the i-th response. Hence, GRPO optimizes a clipped surrogate objective to update the policy $π_{θ}$ based on the advantage:

$J(\theta) = E_{(q, \{ o_{i} \}) \sim \pi_\theta} \left[ \frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{| o_{i} |} \min \Big( r_{i,t}(\theta) \hat{A}_i, \ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \Big) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right] \ \ (2)$

where ε is the clipping hyperparameter and $r_{{i,t}}$(θ) is the importance sampling ratio for token t.

> 💡 **公式批读 — GRPO 目标函数 (Eq.2)**:
> - **外层期望**: 对由当前策略 π_θ 生成的 prompt-响应组取期望
> - **内层求和**: 每条响应中的每个 token 都被纳入损失
> - **Clipping 机制**: 限制 $r_{i,t}$(θ) = π_θ(token|context) / π_$θ_old$(token|context) 在 [1-ε, 1+ε] 范围内，稳定训练
> - **KL 惩罚**: -β × $D_KL$，防止策略偏离参考策略太多
> - **关键问题**: 对于响应 i 的所有 token，Â_i 是**完全相同的**——无论 token t 是"今天的天气真好"还是"那辆车的颜色是蓝色"（但实际上车是红色的），它们接受相同的 advantage 信号。这就是本文要解决的核心痛点。

### Problem Statement

A key limitation of reinforcement learning with verifiable rewards (RLVR) is reward sparsity: conventional approaches provide a single scalar reward only at the end of the reasoning chain, so each token or step is credited equally regardless of its individual correctness or contribution. This coarse, sequence-level feedback makes it difficult to correct localized errors in perception or reasoning and undermines the model's ability to generalize robustly. To overcome this, we propose training a perception-centric process reward model (PRM) that evaluates intermediate perceptual outputs and produces step-wise feedback. Concretely, the PRM checks whether the model's perception content in response (e.g., a grounding, visual feature, or intermediate state) is correct relative to the input v, q, and generate structured outputs that can be used to provide fine-grained supervision. During inference, the PRM can be used to guide the selection of intermediate steps. During training, by designing proper learning objective with the PRM, we encourage correct intermediate perceptual reasoning, enabling more fine-grained supervision for effective learning.

> 💡 **问题定义 — 从 Sparse Reward 到 Perception-centric PRM**:
>
> **传统 RLVR 的假设**: 模型输出的好坏可以完全由最终答案的正确性来判断。
>
> **现实中的问题**: 一个"最终答案正确但推理过程充满幻觉"的响应，和一个"每一步都视觉 grounded 且最终答案正确"的响应，在 GRPO 中可能获得相同的 advantage。
>
> **本文要做的**: 引入一个 perception-centric PRM，它不关心最终答案对错，只关心"回答中的感知内容是否与图像一致"。这个 PRM 的输出是结构化的错误定位，而不是标量分数。
>
> **PRM 的输入输出**:
> - Input: (image v, query q, model response o)
> - Output: 结构化的验证结果 V，包含：
>   - 分析过程（<think>...</think>）
>   - 最终判断（<answer>...</answer>），如果发现错误，以 Python list 返回所有错误子串

---

## 三、Summary

- **VLM 架构**: Vision Encoder + Connection Layer + LLM Decoder → $π_{θ}$(o | q, v)
- **GRPO 核心**: 组内标准化 advantage (Eq.1) + clipped surrogate objective (Eq.2)
- **核心局限**: sequence-level Â_i 对一个响应中所有 token 都相同 → 无法区分正确/错误 token
- **方案方向**: 设计 perception-centric PRM 输出 structured verification，替代/增强序列级 advantage
