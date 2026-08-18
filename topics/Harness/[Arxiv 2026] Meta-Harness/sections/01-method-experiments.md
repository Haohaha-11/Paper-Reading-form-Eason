[← 返回 README](../README.md)

# Method & Experiments 方法与实验

## 📌 预览

方法：形式化 $H^* = \arg\max_H \mathbb{E}_{x,\tau}[r(\tau,x)]$；搜索循环维护 population + Pareto frontier，**无 parent-selection 规则**（proposer 自由检查任意历史 harness），外层循环刻意最小化。proposer = Claude Code + Opus-4.6。实验三域：在线文本分类（+7.7 over ACE，4× 更少 context）、检索增强数学推理（+4.7 跨 5 held-out 模型）、agentic coding（TerminalBench-2，Opus 4.6 上 76.4% #2、Haiku 4.5 上 37.6% #1）。关键消融：raw trace 访问是命门（full 50.0 vs scores-only 34.6 vs scores+summary 34.9）。

---

## 3. Method — 优化 harness 的 harness

**目标**：harness $H$ 是包裹语言模型的**有状态程序**，决定模型每步看到什么 context。找使底层模型在目标任务分布上表现最好的 harness：

$$H^* = \arg\max_H \mathbb{E}_{x \sim \mathcal{X}, \tau \sim p_M(H,x)} r(\tau, x)$$

多目标时（如 accuracy + context cost）用 **Pareto dominance** 评估、报告前沿。

**搜索循环（Algorithm 1）**：单个编码 agent proposer + 一个增长的文件系统 $\mathcal{D}$ 作反馈通道。每个评估过的 harness 贡献一个目录（源码、分数、执行 trace）。文件系统通常远大于 proposer context window，所以 proposer 用 grep/cat 查询而非整体吃进 prompt。每轮：proposer 先检查历史代码/分数/trace，推理可能失败模式，再生成新 harness。**维护 population $\mathcal{H}$ 和 Pareto frontier，但不强加 parent-selection 规则**——proposer 自由检查任意历史 harness。固定轮数后在 Pareto frontier 上做最终 test-set 评估。**proposer 从不看 test-set 结果**，只从 search set 和执行 trace 得反馈。

```
Algorithm 1 Meta-Harness outer loop
1: Input: tasks X, LLM M, proposer P, iterations N
2: Initialize: population H              ▷ 初始有效 harness 集
3: Initialize: filesystem D ← ∅          ▷ 存代码、分数、trace
4: for H ∈ H do E_H ← Evaluate(H,M,X); D ← D ∪ {(H,E_H)}
5: for t=1...N do
6:   Proposer P 查询文件系统 D          ▷ 检查历史 harness 和分数
7:   Proposer P 提出 k 个新 harness {H_1,...,H_k}
8:   for H in {H_1,...,H_k}: if H 通过接口验证 then D ← D ∪ {(H, Evaluate(H,M,X))}
9: return D 中存储 harness 的 Pareto frontier
```

**代码空间搜索的优势**：harness 优化在代码空间，小改动（检索/memory/prompt 构造逻辑）可影响很多步之后的行为，local search heuristic 不适配。检查执行 trace 让 proposer 能推断**为什么**失败、哪些早期设计选择贡献了失败，而非只知道失败了。用程序表示 harness 提供**天然正则**：编码模型倾向提出连贯算法而非脆弱 hard-code 解。

**实现**：每个 harness 是单文件 Python 程序。proposer P = Claude Code + Opus-4.6，由一个最小 domain-specific skill 引导。base model M 随域变化、始终冻结。典型 run 20 轮评估约 60 个 harness。

> 💡 **方法批读（与 Self-Harness 的核心对比表 — 务必内化）**（Hao 批注）：这是本 topic 最重要的对比。Meta-Harness 和 [Self-Harness](../../%5BArxiv%202026%5D%20Self-Harness/) 是"外部优化器"vs"自改进"的两极：
>
> | 维度 | Meta-Harness | Self-Harness |
> |------|-------------|--------------|
> | **谁提案** | 外部强编码 agent（Claude Code + Opus-4.6） | target 模型自己（proposer 角色） |
> | **base model M** | 冻结，常与 proposer 不同/更弱 | 与 proposer 同一个，冻结 |
> | **反馈** | 文件系统全历史（代码+分数+trace），10M token | 聚类失败模式（evidence bundle B_t，压缩） |
> | **搜索结构** | population + Pareto frontier，无 parent-selection | 单一谱系，greedy K-提案 + 回归门 |
> | **选择** | Pareto frontier over search set | 非退化接受规则（held-in + held-out 双门） |
> | **novelty 轴** | 丰富诊断访问（trace > summary） | 内化（自改进，无外部强 agent） |
>
> **关键推论**：
> 1. **Meta-Harness 已经有 population + Pareto**——用户想给 Self-Harness 加的 [GEPA](../../%5BArxiv%202025%5D%20GEPA/) 式 population/Pareto，Meta-Harness 已验证"在外部强 proposer 下有效"。但 Self-Harness **刻意退化成单一谱系 greedy**——因为它用较弱的 target 自己当 proposer。**开放问题**：弱 proposer + population/Pareto 还成立吗？这正是用户可做的贡献点。
> 2. **反馈压缩的张力**：Meta-Harness 力证"压缩反馈丢诊断信号"，Self-Harness 却压缩成 evidence bundle。这使 Self-Harness 理论上**更易受幻觉失败影响**——它看的是聚类后的失败签名，不是原始 trace，无法像 Meta-Harness 那样"读原始 trace 去证伪假设"。[Phantom-Guardrails](../../%5BArxiv%202026%5D%20Phantom-Guardrails/) 的反事实验证正好补这个洞。
> 3. **一个融合方向**：把 Meta-Harness 的 rich-trace 访问 + Self-Harness 的自改进内化 + GEPA 的 population 结合——自己当 proposer，但保留原始 trace 访问（不压缩）、维护多 lineage、加反事实验证。可能就是"明显更强的一篇"。

## 4. Experiments

**4.1 在线文本分类**（GPT-OSS-120B classifier，3 数据集：LawBench 215 类 / Symptom2Disease 22 类 / USPTO-50k 180 类）：
- **vs 手工 harness**：Meta-Harness 48.6% acc，超 ACE 7.7 点、MCE 8.6 点，且用**更少** context（11.4K vs ACE 50.8K / MCE 28.5K）。
- **vs text optimizer**（同 proposer 配置、同预算）：匹配最佳 text optimizer（OpenEvolve/TTT-Discover）用 **0.1× 评估次数**，最终 acc 超它们 10+ 点。
- **OOD 泛化**：9 个未见数据集平均 73.1%（超 ACE 70.2%），6/9 最优——学到的是通用策略而非过拟合。

**关键消融（Table 3，raw trace 是命门）**：

| 条件 | Median | Best Acc |
|------|--------|----------|
| Scores Only | 34.6 | 41.3 |
| Scores + Summary | 34.9 | 38.7 |
| **Meta-Harness (full traces)** | **50.0** | **56.7** |

> 💡 **Table 3 消融批读（对 Self-Harness/用户最关键的一条证据）**（Hao 批注）：这是全文最重要的消融，也是**对用户改进 Self-Harness 最有指导意义的一条**：
> - **scores-only 34.6 → scores+summary 34.9 → full traces 50.0**：加摘要几乎没用（+0.3），但给**原始执行 trace** 猛涨 +15 点。甚至 full 的 median 候选都超过两个 ablation 的 best 候选。
> - **结论**："summary 恢复不了缺失的信号，甚至可能因压缩掉诊断有用的细节而有害"。
> - **⚠️ 对 Self-Harness 的警示**：Self-Harness 的 Weakness Mining 恰恰是把失败**聚类+摘要**成 evidence bundle（$\phi=(c,q,m)$ 签名 + 结构化失败模式），而非给 proposer 原始 trace。按 Meta-Harness 这个消融，**这种压缩可能显著限制了 Self-Harness 的上限**。用户改进 Weakness Mining 时，除了加反事实验证（去幻觉），还应考虑**是否给 proposer 更多原始 trace 访问**（而非只给聚类摘要）——这是 Meta-Harness 用数据证明的方向。

**4.2 检索增强数学推理**（olympiad 数学 + BM25 检索，语料 ≥500K 已解题）：单个发现的检索 harness 在 200 IMO-level 题上、跨 5 个 held-out 模型（GPT-OSS-20B、GPT-5.4-nano/mini、Gemini-3.1-Flash-Lite、Gemini-3-Flash）平均 +4.7 点 over no-retriever，超 BM25 1.3 点、避免 dense retrieval 的 regression。发现的是**4-route BM25 程序**（combinatorics/geometry/number-theory/default 词法路由）。

**4.3 Agentic Coding（TerminalBench-2，89 任务）**：从 Terminus 2 + Terminus-KIRA 初始化。
- **Opus 4.6**：76.4% pass，超手工 Terminus-KIRA（74.7%），leaderboard #2（唯一更高的 ForgeCode 81.8% 无法复现）。
- **Haiku 4.5**（更弱）：37.6%，超次佳 Goose（35.5%）2.1 点，**#1**。
- **发现的核心修改**：environment bootstrapping——agent 循环开始前跑一个复合 shell 命令收集沙箱环境快照（OS/语言/包管理器/`/app` 内容）注入初始 prompt，消除 2-4 个探索回合。仅 89 任务中 7 个受益，但都是需要 domain-specific 工具、环境不可预知的任务。

> 💡 **实验批读（proposer 的因果推理 = 反 Phantom-Guardrails 的正面案例）**（Hao 批注）：附录 A.2 的 TerminalBench-2 搜索日志是**全文最精彩的部分**，也是理解"好的 harness 搜索该长什么样"的范本：
> - proposer 前 6 轮反复 regress（改 completion flow / prompt template / observation processing 都退化）。
> - **第 3 轮识别 confound**：proposer 明确推断"两次失败的共同因素不是具体 bugfix，而是 cleanup-heavy 的 prompt 重写"——于是回退 prompt、只测结构修复。
> - **第 7 轮转向纯 additive 修改**（env bootstrap），并**论证为什么更安全**（不碰脆弱的 completion 机制，只加对难任务有用的信息）→ 成为最佳候选。
> - proposer 每轮读 median 82 个文件、引用 20+ 历史候选。
>
> **⚠️ 这正是 [Phantom-Guardrails](../../%5BArxiv%202026%5D%20Phantom-Guardrails/) 关心的反面**：Meta-Harness 的 proposer 因为**能读原始 trace**，能做"识别 confound → 隔离因果变量 → 反事实测试"的因果推理，从而**避免给虚构失败加 guardrail**。Self-Harness 把失败压成 evidence bundle 后，proposer 更难做这种原始 trace 上的因果验证——这从机制上解释了为什么 Self-Harness 更需要显式的反事实验证步骤。这个 A.2 案例应该作为"如何做 grounded 失败诊断"的正面模板，指导用户改进 Weakness Mining。

## 5. Discussion & 局限

**优势**：发现的 harness 泛化到 OOD 分类数据集和未见 base 模型；一次搜索几小时 wall-clock、产可读可迁移策略（可复用到未来更强模型）；代码空间过拟合更可检查（脆弱 if-chain 一眼可见）。**核心优势不只是搜代码，而是搜索 + 对历史诊断经验的选择性访问**。呼应 bitter lesson：一旦搜索空间可访问，更强的通用 agent 能超过手工方案。future：co-evolve harness 和模型权重。

> 💡 **总结 + 对用户研究的定位**（Hao 批注）：Meta-Harness 是 Self-Harness 的直接前作和最重要对照。用户理解 Self-Harness novelty 的坐标：
> - **Self-Harness 的 novelty = 内化**（去掉 Meta-Harness 的外部强 proposer，让 target 自己改自己）。这是"往前推一步"的正确概括。
> - **但 Self-Harness 为内化付出的代价**：(1) proposer 变弱（target 自己）；(2) 反馈被迫压缩（evidence bundle 而非全 trace）；(3) 搜索退化（单谱系 greedy 而非 population/Pareto）。
> - **用户的机会 = 在保持"内化"的前提下，把 Meta-Harness 证明有效的东西补回来**：rich-trace 访问（Table 3 证明关键）、population/Pareto（Meta-Harness 已用）、因果验证（A.2 展示的 confound 识别）——同时加 [Phantom-Guardrails](../../%5BArxiv%202026%5D%20Phantom-Guardrails/) 的反事实去幻觉。
> - **一个尖锐的研究问题**：Meta-Harness 用强 proposer + 全 trace + Pareto 拿到大增益；Self-Harness 用弱自 proposer + 压缩反馈 + greedy 也拿到 +132%。**这两个增益来源可分离吗？** 弱自 proposer 到底损失了多少、能否用更好的搜索/验证补回来——这是"明显更强的 Self-Harness"要回答的核心。

> 💡 **Q&A 批注记录**（Hao 批注）：
> - **Q：Meta-Harness 和 Self-Harness 一句话区别？**
>   A：Meta-Harness 用外部强编码 agent（Claude Code+Opus-4.6）搜索一个 target 模型的 harness；Self-Harness 让 target 模型自己改自己的 harness。后者去外部依赖但 proposer 变弱、反馈压缩、搜索退化。
> - **Q：为什么 Meta-Harness 强调文件系统全访问？**
>   A：Table 3 消融证明——raw trace 访问让 acc 从 34.6（scores-only）涨到 50.0；summary 几乎没用甚至有害。harness 作用于长 horizon，压缩反馈丢掉追溯失败的诊断信号。
> - **Q：Meta-Harness 有 population/Pareto，Self-Harness 为什么没有？**
>   A：Meta-Harness 用强外部 proposer 撑得起 population 搜索；Self-Harness 用弱自 proposer，退化成单谱系 greedy。用户借 GEPA 给 Self-Harness 加回 population 是合理方向，但要验证"弱自 proposer + population"是否成立。
> - **Q：Meta-Harness 怎么避免过拟合 benchmark？**
>   A：手工检查 + regex 审计 task-specific 字符串泄漏；代码空间过拟合可检查（脆弱 if-chain 可见）；OOD 数据集和未见模型上验证泛化。
