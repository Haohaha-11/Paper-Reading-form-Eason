[← 返回 README](../README.md)

# Abstract, Intro & Problem 摘要、引言与问题

## 📌 预览

GEPA（Genetic-Pareto）= **反射式 prompt 优化器**，把自然语言反射 + 多目标进化搜索结合。核心论点：语言的可解释性是比稀疏标量 reward 的 policy gradient **更丰富的学习媒介**。GEPA 采样 trajectory（推理/工具调用/工具输出）→ 自然语言反射诊断问题、提出并测试 prompt 更新 → 从自己尝试的 **Pareto frontier** 组合互补经验。6 个任务上平均超 GRPO 6%（最高 20%），用 up to 35× 更少 rollout；超最强 prompt 优化器 MIPROv2 10%+。

> 📌 **用户想借来改进 [Self-Harness](../%5BArxiv%202026%5D%20Self-Harness/) 的候选搜索/选择**。用户指出：Self-Harness 现在像 `h_t →K 提案→挑通过→h_{t+1}`，比较 **greedy**；GEPA 的核心是 **reflection + semantic mutation + population/archive + Pareto selection**，给 Self-Harness 明显升级方向——harness evolution 不该只有单一路径，可维护多个各有优势的 lineage 再 crossover/merge。**本篇重点读 candidate search/selection（§3.1 + Algorithm 2）。**

---

## Abstract

LLMs are increasingly adapted via RL methods like GRPO, which often require thousands of rollouts. We argue that the **interpretable nature of language** often provides a much richer learning medium than policy gradients derived from sparse, scalar rewards. We introduce **GEPA (Genetic-Pareto)**, a prompt optimizer that thoroughly incorporates natural language reflection to learn high-level rules from trial and error. Given any AI system containing one or more LLM prompts, GEPA samples trajectories and reflects on them in natural language to **diagnose problems, propose and test prompt updates, and combine complementary lessons from the Pareto frontier of its own attempts**. Across six tasks, GEPA outperforms GRPO by 6% on average and up to 20%, while using up to **35× fewer rollouts**, and outperforms the leading prompt optimizer MIPROv2 by over 10%.

> 💡 **核心论点（reflection > policy gradient）+ 对用户的定位**（Hao 批注）：GEPA 的中心主张——**从自然语言 trajectory 反射学习，比从稀疏标量 reward 估计 policy gradient 更样本高效**（因为 LLM 有强语言先验，能理解 serialized trajectory：模块指令 + 推理链 + 工具调用 + reward 函数内部如编译错误）。这对用户有两层价值：
> 1. **直接价值 = 搜索结构**：GEPA 的 Pareto-based candidate selection + genetic tree + System Aware Merge，正是 Self-Harness 缺的（Self-Harness 是 greedy 单谱系）。这是用户要借的。
> 2. **注意边界**：GEPA 是 **prompt 优化**（不是完整 harness），且**需要 validation 标签**（$D_{pareto}$ 用 μ 打分）——[RHO](../%5BArxiv%202026%5D%20RHO-Self-Preference/) 的 Table 5 把 GEPA 归为"validation-feedback + prompt-only"，[Meta-Harness](../%5BArxiv%202026%5D%20Meta-Harness/) 把它归为"Summary 反馈类"（压缩太狠不适合 harness 搜索）。**所以用户要借的是 GEPA 的搜索结构，不是它的 prompt-only 范围或标签依赖**。

## 1-2. Introduction & Problem

**动机**：RLVR（如 GRPO）把成功指标当 end-of-rollout 标量 reward 估计 policy gradient，但通常需数万到数十万 rollout。而 rollout 可序列化成自然语言 trace（模块指令、推理链、工具调用、reward 函数内部如编译错误），现代 LLM 能理解——**故意用自然语言反射学习**能更好利用 LLM 的语言先验。

**问题（sample-efficient optimization）**：compound AI system $\Phi = (M, C, \mathcal{X}, \mathcal{Y})$，可学习参数 = 各模块 prompt $\Pi$ + 权重 $\Theta$。在 rollout 预算 $B$ 下最大化 held-out 性能：

$$\langle\Pi^*, \Theta^*\rangle_\Phi = \arg\max \mathbb{E}_{(x,m)\sim\mathcal{T}}[\mu(\Phi(x; \langle\Pi,\Theta\rangle_\Phi), m)], \quad \text{s.t. \#rollouts} \leq B$$

核心挑战：*如何从每个昂贵 rollout 提取最大学习信号*。GEPA 只演化 prompt $\Pi_\Phi$，权重 $\Theta_\Phi$ 固定。

> 💡 **问题设定批读（与 harness 优化的关系）**（Hao 批注）：GEPA 的问题设定和 [Self-Harness](../%5BArxiv%202026%5D%20Self-Harness/)/[Meta-Harness](../%5BArxiv%202026%5D%20Meta-Harness/) 形式上一致（固定权重、优化非参数的 prompt/harness、rollout 预算约束），差别只在**优化对象的粒度**：
> - GEPA：优化 compound system 的**模块 prompt**（$\Pi$）。
> - Self-Harness/Meta-Harness/AHE：优化**完整 harness**（prompt + tools + middleware + memory）。
> 这意味着 GEPA 的搜索算法（Pareto illumination + genetic tree + merge）**可以直接搬到 harness 优化**——只要把"candidate = prompt"换成"candidate = harness"。这正是用户的思路。GEPA 的 sample-efficiency（35× 更少 rollout）对 harness 优化尤其宝贵，因为 harness rollout（跑完整 agent 任务）比 prompt rollout 更贵。
