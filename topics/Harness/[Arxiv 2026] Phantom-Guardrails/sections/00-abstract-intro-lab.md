[← 返回 README](../README.md)

# Abstract, Intro & The Fabrication Lab 摘要、引言与反事实伪造实验室

## 📌 预览

**自改进 agent 会"幻觉"出根本没发生的失败，然后给这个虚构失败加 guardrail。** 本文在自动 harness 优化中研究这个失败模式：LLM proposer 编辑 agent 周围的 scaffold（prompt/parser/filter/validator/guardrail）让观测到的失败消失，但很少问一个前置问题——**真的有失败要修吗？** 作者造了 **Counterfactual Fabrication Lab**（反事实伪造实验室）：一个确定性 micro-lab，正确动作事先已知是"do nothing"，植入一个针对**可证明永不发生**的失败类的候选 guardrail，只展示合法 episode，用 byte-exact oracle 检查每个被引用的违规。结果：proposer 在真违规时正确、在无特征合法输入上弃权，但当合法输入含一个**像熟悉游戏规则的良性模式**时会**发明失败**——15/60 次启用不存在规则的 guardrail 并引用 oracle 反驳的违规。

> 📌 **对 [Self-Harness](../../%5BArxiv%202026%5D%20Self-Harness/) 的精准反问题**（用户特别推荐）。用户指出：Self-Harness 的 `failed traces → LLM diagnosis → failure mechanism` 隐含假设"**LLM diagnosis is factually grounded**"，但未必成立。本文直接启发把 **Failure Mining** 升级成 **Failure Hypothesis → Counterfactual Verification → Validated Mechanism → Harness Proposal**，甚至加一个 **"Do Nothing"** 候选。

---

## Abstract

Self-improving AI agents are designed to learn from their mistakes. **We show that they can also hallucinate mistakes that never happened.** We study this failure mode in automated harness optimization, where an LLM-based proposer edits the scaffold around an agent (prompts, parsers, filters, validators, guardrails) to make observed failures disappear. But this process rarely asks a prior question: **was there a real failure to fix?** We introduce the **Counterfactual Fabrication Lab**, a deterministic micro-lab where the correct action is known in advance to be "do nothing." The lab plants a candidate guardrail for a failure class that provably never occurs, presents only legal episodes, and uses a byte-exact oracle to check every cited violation. The proposer behaves as expected when the violation is real and abstains on featureless legal input. Yet when the legal input contains a **harmless pattern that resembles a familiar game rule**, it invents a failure: in **15/60 runs, versus 0/60 on featureless input**, it enables the nonexistent-rule guardrail and cites a violation the oracle refutes. The effect is structured: in single-shot proposals it appears only when three conditions coincide — a **rule-shaped pattern, an open-ended rule set, and an instruction that presupposes failures**. Removing any one eliminates the fabrication. Because the invented guardrail changes no true outcome and cannot improve an already-perfect suppression score, the phenomenon is **neither reward hacking nor over-refusal**. It is a **phantom guardrail**: a fix for a failure that never happened, invisible to suppression-only acceptance.

> 💡 **核心贡献 = 命名并测量一个新失败模式（务必内化）**（Hao 批注）：这篇的价值不在数字（0.25、单个 proposer 主导），而在**首次干净地隔离并证明"幻觉失败"这个失败模式存在**，且给出可测量的工具。对整个 Harness topic 的意义：
> - 所有自改进 harness（[Self-Harness](../../%5BArxiv%202026%5D%20Self-Harness/) 的 Weakness Mining、[RHO](../../%5BArxiv%202026%5D%20RHO-Self-Preference/) 的 self-preference、[AHE](../../%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) 的 evolve）都建立在"proposer 的失败诊断是可信的"这个隐含假设上。
> - 本文证明**这个假设在特定（且常见）条件下系统性失效**：proposer 会读过可见的 legality（每个 move 标 legal），凭 genre prior 发明一个规则，然后 harden against it。
> - **关键定性**：这**不是** reward hacking（无 proxy 增益换 true-return 损失）、**不是** over-refusal（无 helpfulness 损失）、**不是** 无差别过度建造（无特征输入上弃权）。它是"在已满足、不可 hack 的 proxy 下的多余动作"——一个只有 warrant-aware 验证才能抓的盲区。

## 1-2. Introduction & Related Work

**问题**：现代 agent 裹着增长的 scaffold 层（parser 修复畸形动作、guardrail 拦非法、filter 净化答案、retrieval 路由）。一条新工作线把这层的构造交给模型自己——**自动 harness-search optimizer 读失败 episode、提出 scaffold 编辑、保留减少观测失败的**。**reward 几乎总是"观测失败的抑制"**——回答"失败停了吗？"但从不问"这个修复正当吗？"

**熟悉的担忧是 under-fixing**（找不到就修不了）。本文研究**相反的失败**：suppression-rewarded optimizer 在**根本没东西可修**时做什么？

**与近邻的区分**（本文刻意划清）：
- **vs Over-refusal**：over-refusal 的假阳性是用降低 helpfulness 换来的（安全/有用权衡 Spearman 0.89），且输入是从有毒种子对抗改写的。**本文的 guardrail 无 task-return 权衡，且 pool 全合法、触发是任务没定义规则的良性 regularity**。
- **vs Reward hacking**：reward hacking 是用 proxy 增益换 true-return 损失。**本文的 guard 在全合法 pool 上是 no-op，既不能升已满足的 proxy、也不降 true return**——在满足、不可 hack 的 proxy 下的多余动作。
- **vs Over-editing/over-action**：程序修复加"defect 不需要的 guard"、agent 过度调工具。**本文更尖锐**——一个 oracle 认证零支持的失败类的 guard，且证明弃权在同设定下可达。

> 💡 **明确点名 RHO 为脆弱对象（关键关联）**（Hao 批注）：本文在 Related Work 直接把 [RHO](../../%5BArxiv%202026%5D%20RHO-Self-Preference/) [15] 列为"最接近的系统"——"a recent optimizer makes the asymmetry concrete: it accepts a proposed edit only when a **self-preference score improves, with no separate test of whether the edit was warranted** [15]"。含义极其重要：
> - **RHO 的无标签 self-preference 验证正是本文攻击的靶子**——它只测"候选是否被偏好/抑制失败"，不测"被修的失败是否真存在"。
> - 这形成了本 topic 的一个**闭环张力**：RHO 用 self-preference 去标签（好），但 self-preference 无法证伪幻觉失败（本文证明的坏）。
> - **对用户的启示**：把 Self-Harness 改 label-free（借 RHO）时，**必须同时加 warrant-aware 验证**（本文的解法），否则去了标签就直接掉进 phantom guardrail 陷阱。二者是配套的。

## 3. The Counterfactual Fabrication Lab

**设定**：确定性 "MiniArena" 游戏。episode = 短 move 记录序列，pool = episode 多重集。episode 冻结（base agent 不重查、无方差），每个 move 带 legality tag。三个失败类 $C = \{syntax, bounds, castle\}$，每类有 oracle $O_c(e) \in \{0,1\}$ 读**不相交的记录字段**（类可分离）。

- **syntax**（畸形 token）、**bounds**（越界目标）都真实发生、各有 fixer。
- **castle = 探针**：只在非法 castling move 上触发，但 move generator **不可能产生**（只产合法 token：列 A-D + 行 1-4）。castling 记录只能通过显式 injector 进入 pool。**在其他所有 pool 上 $O_{castle}(e)=0$**，所以对应的 guard g_castle（**phantom guard**）在那里可证明永不发生。

**suppression proxy（唯一 reward 信号）**：harness 应用后无 firing 类的 episode 比例：
$$S(H,D) = \frac{1}{|D|}\sum_{e\in D} \mathbb{1}[\text{fired}(H(e)) = \emptyset]$$

**全合法 pool 上无类 firing，$S(H,D)=1$ 对每个 $H$（含空 harness）**——proxy 已达最大、任何编辑都升不了、正当 harness 是空的：fired$(e)=\emptyset\ \forall e \Rightarrow W(D)=\emptyset$。

**Fabrication 度量**（在 oracle 认证无 phantom 类的 pool 上）：
$$\text{Fab}(H,D) = \mathbb{1}[\text{g\_castle} \in H] = 1$$

**pool 条件由确定性代码决定**，所以正的 Fab 是**对 ground truth 的假阳性**：proposer 看到每个 move 都合法，仍造了个不存在规则的 guard。5 个 proposer（deepseek-v4-pro/flash、qwen3.7-max、qwen3.6-max-preview、glm-5.1）× 4 sub-pool × 3 seed。

**三个关键 pool**：
- **congruent pool**：注入 4 个非法 castle 记录（可见标 legal=ILLEGAL）→ phantom guard 正当。
- **fabrication pool**：全合法，但带良性表面 regularity（12 局中 3-5 局巧合重复某格），不违反任何规则。
- **pristine pool**：全合法、无特征（格子全不同）。

> 💡 **实验设计批读（为什么这个 lab 干净）**（Hao 批注）：这个 micro-lab 的设计极其精巧，值得学习作实验方法论：
> - **"do nothing" 是已知正确答案**——把"过度修复"从难测（需判断修复是否正当）变成可测（全合法 pool 上任何 guard 都是错的）。
> - **byte-exact oracle** 代替第二个模型的意见——避免"模型评模型"的循环，正的 Fab 是对 ground truth 的假阳性，不是对另一模型的分歧。
> - **phantom 类可分离 + 可达**（congruent pool 上 g_castle 60/60 证明 detector 工作）——排除"guard 坏了"的平凡解释。
> - **$0 可审计、SHA-256 pin pool**——完全可复现。
> - **对用户的方法论启示**：若要给 Self-Harness 加反事实验证并测其有效性，这个"植入永不发生的失败类 + oracle 检查引用违规"的 lab 设计是现成的评测工具。用户甚至可以直接用它测"升级后的 Weakness Mining 是否还会 fabricate"。
> - **⚠️ Appendix B 的方法论警告**：作者诚实报告一个 security-framed 变体一开始看起来 0.98 over-fixing，但控制混淆因素（payload 注入、homonym 效应、tool-provenance artifact）后**塌成 null**。教训：**over-fixing 极难测，naive 设计产生戏剧性假阳性**——全合法+不存在规则的设计才保持结果干净。用户做类似实验必须警惕这些混淆。
