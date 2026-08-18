[← 返回 README](../README.md)

# Abstract, Intro & Related Work 摘要、引言与相关工作

## 📌 预览

AHE = **observability 驱动的闭环 harness 演化**。核心：三个匹配的 observability 支柱把每个 harness 编辑变成**可证伪的契约**——❶ **component observability**（NexAU 把 7 类可编辑组件暴露为文件，动作空间显式可回滚）；❷ **experience observability**（Agent Debugger 把百万级原始 trajectory token 蒸馏成分层可下钻的证据语料）；❸ **decision observability**（change manifest 给每个编辑配一个自声明预测，下一轮用 task 级结果验证）。10 轮迭代把 Terminal-Bench 2 pass@1 从 69.7% 提到 77.0%，超人工 Codex（71.9%）和自演化基线 ACE/TF-GRPO；冻结 harness 跨基准/跨模型族迁移（+5.1~+10.1pp）。

> 📌 **与 [Self-Harness](../../%5BArxiv%202026%5D%20Self-Harness/) 高度相似的并行独立工作**（同为 2026）。用户指出：两者都是 `trajectory → failure diagnosis → harness edit → evaluation → rollback/retain`，且 AHE 的 **evidence ledger / change manifest**（每个修改绑定 failure evidence + root cause + expected fix + regression risk）与 Self-Harness 的 **Weakness Mining + Proposal Validation 非常接近**。

---

## Abstract

Harnesses are now central to agent performance, mediating how models interact with tools and execution environments. Yet harness engineering remains a manual craft, because automating it faces a **heterogeneous action space** across editable components, **voluminous trajectories** that bury actionable signal, and **edits whose effect is hard to attribute**. We introduce Agentic Harness Engineering (AHE), a closed loop that addresses these challenges through three matched observability pillars: ❶ **component observability** gives every editable harness component a file-level representation so the action space is explicit and revertible; ❷ **experience observability** distills millions of raw trajectory tokens into a layered, drill-down evidence corpus that an evolving agent can actually consume; and ❸ **decision observability** pairs every edit with a self-declared prediction, later verified against the next round's task-level outcomes. Together, these pillars turn every edit into a **falsifiable contract**, so harness evolution proceeds autonomously without collapsing into trial-and-error. Empirically, ten AHE iterations lift pass@1 on Terminal-Bench 2 from 69.7% to 77.0%, surpassing the human-designed harness Codex (71.9%) and the self-evolving baselines ACE and Training-Free GRPO.

> 💡 **三个 observability 支柱 = 对自动 harness 演化的系统性拆解**（Hao 批注）：AHE 把"如何自动演化 harness"归结为**三个 observability 问题**，这个拆解比 Self-Harness 更工程化、更彻底，对用户改进 Self-Harness 极有参考价值：
> 1. **component observability（动作空间可观测）**：7 类正交组件（system prompt / tool description / tool implementation / middleware / skill / sub-agent config / long-term memory）各是一个文件，每个失败模式干净映射到一个组件类，每次编辑一个 git commit（文件级 diff + 回滚粒度免费）。→ 对比 Self-Harness：Self-Harness 只编辑 DeepAgent harness 文件里声明的配置点，动作空间更窄；AHE 的 7 类组件是更完整的动作空间。
> 2. **experience observability（经验可观测）**：Agent Debugger 把原始 trajectory 蒸馏成**分层可下钻**的证据语料（per-task 分析报告 + benchmark 级总览），evolver 消费结构化 root cause 而非原始 log——**但也保留原始 trace 供验证**（progressive disclosure）。→ 这是 [Meta-Harness](../../%5BArxiv%202026%5D%20Meta-Harness/)（全 trace）和 Self-Harness（压缩 bundle）之间的**中间地带**：既蒸馏又保留原始 trace 可下钻。
> 3. **decision observability（决策可观测）**：change manifest 给每个编辑配自声明预测，下一轮验证 → **可证伪契约**。→ 这正是 Self-Harness 的 audit 记录 $a_j$ 的强化版。

## 1. Introduction — 核心洞察：瓶颈在 observability 不在 agent 能力

**核心问题**（用户可直接借用的问题表述）：*How can an evolution agent jointly and stably evolve all editable components of a coding agent's harness?* 联合演化多组件面临两个结构性障碍：(1) 长、无结构的 trajectory 提供很少可操作信号；(2) 紧耦合的 harness 框架使 prompt 之外的编辑易错。

**中心洞察**：*this question is bottlenecked by observability, not by agent capability*——一旦 evolution agent 在清晰动作空间上收到结构化 context，它就能可靠收敛到更好的 harness 设计。

**三大贡献**：
1. 提出 AHE，把 observability（跨组件、trajectory、决策）识别为设计支点，用三支柱把每个编辑变成可证伪的文件级契约。
2. 实证：Terminal-Bench 2 69.7%→77.0%，超人工和自动基线，冻结 harness 跨基准/跨模型族迁移。
3. **揭示 agent 驱动演化的两个极限**：harness 组件**非加性交互**（叠加有效编辑会 cap 总增益）；循环的 self-attribution **对 fix 可靠、但对 regression 盲**（regression foresight 是未来最清晰的方向）。

> 💡 **第三个贡献 = 对用户最关键的经验发现（regression blindness）**（Hao 批注）：AHE 的第三个贡献直接连到 [Phantom-Guardrails](../../%5BArxiv%202026%5D%20Phantom-Guardrails/) 和用户改进 Self-Harness 的核心——**LLM 对自己 harness 编辑效果的自评估是不对称地不可靠的**：
> - **对 fix 的预测可靠**（evidence-driven，见 §4.4.2：precision 33.7% / recall 51.4%，~5× random）。
> - **对 regression 的预测几乎失灵**（precision 11.8% / recall 11.1%，仅 ~2× random）——"agent 能论证为什么某编辑该 help，却无法可靠说出同一编辑将破坏哪些 task"。
>
> **这与 Phantom-Guardrails 互补地夹击同一问题**：Phantom 说"proposer 可能诊断出不存在的失败"（fix 侧的假阳性），AHE 说"proposer 预测不了会引入的 regression"（regression 侧的假阴性）。两者合起来说明：**self-improving harness 的 self-assessment 在两个方向上都不可信**。用户设计"明显更强的 Self-Harness"时，Validation 阶段既要防幻觉失败（Phantom），也要主动预测 regression（AHE 指出的空白）——一个**双向 falsification**的验证器。

## 2. Related Work

**Harness engineering & evaluation**：harness 中介模型如何感知/作用于环境（action/observation 接口、agent-computer 接口、沙箱执行编排）。评估沿 task horizon 和环境真实性两轴成熟（function 级 → repo 级可执行 patch → 多小时终端工作流）。

**Automated optimization of LLM agents**：按"优化器观察什么证据、能编辑什么"分类——修改 agent 输出（reflection/critique）、优化 prompt/instruction（playbook、semantic-advantage prior、DSPy、GEPA 的 Pareto-frontier trace 反射更新）、编辑程序结构（skill 库、scored program/agent archive、workflow graph）。

> 💡 **相关工作批读（AHE 的定位 = full harness as combinatorial whole）**（Hao 批注）：AHE 明确把自己与两条线区分：(1) 只优化单一 surface（prompt/skill/playbook）的方法——AHE 联合演化 **7 类组件作为组合整体**，让跨组件权衡对优化器可见；(2) [Meta-Harness](../../%5BArxiv%202026%5D%20Meta-Harness/)（引用 [16]）——AHE 与 Meta-Harness 都是"演化全 harness"，但 AHE 强调**保持人类先验最小**（方法论让优化器从 rollout 发现，而非手工固定）。对用户：AHE 和 Self-Harness、Meta-Harness 构成 2026 年"自动 harness 演化"的三足——Meta-Harness（外部强 proposer + 全 trace）、Self-Harness（自 proposer + 压缩 bundle）、AHE（多角色同基座 + 分层证据 + 结构化 manifest）。三者的组件粒度、证据形式、验证机制各不同，是用户设计新方法的三个参照点。
