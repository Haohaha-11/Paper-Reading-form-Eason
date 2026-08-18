[← 返回 README](../README.md)

# Abstract, Intro & Problem Setting 摘要、引言与问题设定

## 📌 预览

RHO（Retrospective Harness Optimization）= **只用历史 trajectory、无任何 ground-truth 标签**优化 agent harness 的自监督方法。动机：现有 harness 优化（OPRO/DSPy/TextGrad/GEPA/ADAS/Meta-Harness）全都需要 labeled validation set 打分，但部署场景难获取。RHO 用 agent 自己的 **self-preference** 代替 latent utility：从历史 trajectory 选 difficulty-diverse coreset → 每任务并行重解 G 次、提 **self-validation + self-consistency** 诊断信号 → best-of-N 采样候选 harness、用 pairwise self-preference 选最优。单轮把 SWE-Bench Pro 从 59%→78%（无外部打分）。

> 📌 **对 [Self-Harness](../%5BArxiv%202026%5D%20Self-Harness/) Validation 阶段最直接的改进来源**。用户指出：RHO 同样"从历史 trajectory 发现问题并修改完整 harness，但**不依赖 labeled validation feedback**，而是通过 self-validation + cross-trajectory consistency + pairwise self-preference 产生优化信号"。Self-Harness 的 Proposal Validation 用 labeled held-out 回归门；RHO 给出无标签替代。

---

## Abstract

AI agents rely on a harness of skills, tools, and workflows to solve complex problems. Continually improving this harness is essential for adapting to new tasks. However, existing optimization methods typically require **ground-truth validation sets**, yet such labeled data is difficult to acquire in practical deployment settings. To address this problem, we introduce **Retrospective Harness Optimization (RHO)**, a self-supervised method that optimizes the agent harness using only past trajectories. Specifically, RHO selects a diverse coreset of challenging tasks from past trajectories and re-solves them in parallel. The agent analyzes these rollouts using **self-validation and self-consistency**, then generates candidate harness updates and selects the most effective one by its own **pairwise self-preference**. We evaluate RHO across three diverse domains, spanning software engineering, technical work, and knowledge work. Notably, a single optimization round improves the pass rate on SWE-Bench Pro from **59% to 78% without any external grading**.

> 💡 **核心动机（无标签是关键差异）**（Hao 批注）：RHO 攻的是所有 harness 优化方法（含 [Self-Harness](../%5BArxiv%202026%5D%20Self-Harness/)、[Meta-Harness](../%5BArxiv%202026%5D%20Meta-Harness/)、[AHE](../%5BArxiv%202026%5D%20Agentic-Harness-Engineering/)）的共同软肋——**都需要可验证的 reward/held-out 标签来指导搜索**。但真实部署中：
> - 难收集能准确估计未来任务分布的 validation set；
> - 但 agent 持续运行天然产生大量历史 trajectory。
>
> RHO 的问题：**只有历史 trajectory（无标签）时，能否改进 harness 提升未来性能？** 这对 Self-Harness 是尖锐的——Self-Harness 的整个 Proposal Validation 建立在 held-out verifier 打分（labeled）上。RHO 证明**可以完全去掉标签**，用 self-preference 替代。这是用户改进 Self-Harness Validation 的最直接武器。

## 1. Introduction & Problem Setting

**形式化**：harness $h$ = 工具/prompt/skill 的持久集合。agent 在 $h$ 下解任务 $t$ 产生 trajectory $\tau = \text{solve}(h,t)$。目标是找最大化未来任务期望效用的 harness：

$$h^\star = \arg\max_{h'} \mathbb{E}_{t, \tau \sim \text{solve}(h',t)}[U(t,\tau)]$$

**问题**：效用函数 $U$ 是 **latent（潜在、不可直接观测）**——要评估真实效用需要代表性 validation set + 成功率机制。

**RHO 的解法**：用 **self-preference estimator** 替代 latent utility。定义排序函数 $(\text{rank}, \text{rationale}) = \text{rank}(t, \tau_1, \ldots, \tau_m)$——agent 比较同任务的多条 trajectory，产生偏好排序 + 解释为什么偏好某些执行。

> 💡 **self-preference 替代 latent utility（方法论的核心跳跃）**（Hao 批注）：RHO 的关键跳跃是**用 agent 自己的偏好排序代替不可观测的真实效用**。这依赖一个假设：模型有"部分识别自己知识边界"的能力（partial ability to recognize the limits of their own knowledge）。
> - **这正是 [Phantom-Guardrails](../%5BArxiv%202026%5D%20Phantom-Guardrails/) / [AHE](../%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) 质疑的能力**：AHE 证明 self-attribution 对 fix 可靠（5×random）对 regression 盲（2×random）；Phantom 证明 proposer 会诊断幻觉失败。RHO 押注 self-preference **足够好**（虽有噪声）。
> - **RHO 自己的证据（Table 3）**：self-preference 选择"consistently avoids the worst candidate"但"不总选到 test 最高分候选"——即**自偏好是有偏但有用的代理，能避开最差、不能保证最优**。
> - **对用户的启示**：RHO 给 Self-Harness Validation 提供无标签路径，但自偏好本身不可靠（AHE/Phantom 已证）。**理想方案 = RHO 的无标签 self-preference + Phantom 的反事实去幻觉 + AHE 的 regression 预测**——三者叠加才能既去标签依赖又保可靠性。

**贡献**：(1) 提出 retrospective harness optimization，仅从无标签 trajectory 改进**完整 harness**（memory/context/skills/tools）；(2) 三场景验证，超朴素经验累积、并在可比预算下超 validation-feedback 演化；(3) 定量分析 harness 优化如何改变 agent 行为。

## 2. Related Work

**Harness optimization**：一线针对 labeled metric 优化 prompt/pipeline（OPRO、DSPy、TextGrad、GEPA）；一线让 meta-agent 重写 agent 代码（ADAS、Meta-Harness）。**共同点：都用 labeled validation metric 指导搜索**。RHO 脱离此范式——无 validation 反馈、单次 retrospective pass。

**Agent self-improvement**：从自身经验改进（Dynamic Cheatsheet 自 curated memory、ReasoningBank 蒸馏推理策略、MemMA 多 agent 修 memory、Sleep-time Compute 离线预算 context、M⋆ 演化 memory 程序、SkillOS RL 训 skill curator）。**共同点：只丰富 memory/context/skill list，不动其余 harness**。RHO 优化**完整 harness**（含可执行 tool 和 instruction）。

> 💡 **Table 5 定位批读（RHO 的三轴独占）**（Hao 批注）：RHO 用三个轴给整个 harness 优化文献定位，这个坐标系对用户极有用：
> - **Label-free（无标签）**：validation-feedback 类（OPRO/DSPy/TextGrad/GEPA/ADAS/Meta-Harness）✗；experience-based 类 ✓。
> - **Full harness（编辑完整 harness，非仅 memory/prompt）**：ADAS/Meta-Harness ✓；experience-based 类多为 ✗（只 memory/skill）。
> - **Single pass（单次离线 retrospective）**：多数迭代搜索 ✗。
>
> **RHO 是唯一同时满足三轴的方法**。这个三轴框架直接指出 [Self-Harness](../%5BArxiv%202026%5D%20Self-Harness/) 的位置：Self-Harness = 编辑较完整 harness + **需标签（held-out verifier）** + 多轮迭代——即 Self-Harness 在"label-free"轴上是 ✗。**用户把 Self-Harness 改成 label-free（借 RHO 的 self-preference）是一个清晰的贡献点**，让 Self-Harness 也能满足三轴。
