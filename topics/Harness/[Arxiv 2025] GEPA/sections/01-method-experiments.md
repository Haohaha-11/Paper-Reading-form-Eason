[← 返回 README](../README.md)

# Method & Experiments 方法与实验（重点：Pareto 候选选择）

## 📌 预览

GEPA 三原则：**genetic prompt evolution**（候选池 + 谱系树，reflective mutation 或 crossover 派生新候选）、**reflection with natural language feedback**（反馈函数 $\mu_f$ 返回分数 + 文本反馈如编译错误/失败 rubric，反射 LM 做隐式 credit assignment）、**Pareto-based candidate selection**（§3.1，本篇重点）。关键消融（Table 3）：Pareto 选择 +12.44% >> greedy SelectBest +6.05% / BeamSearch +5.11%——证明 Pareto illumination 逃离局部最优。System Aware Merge（crossover）合并互补谱系再 +2~5%。

---

## 3. GEPA 三原则

**Genetic Optimization Loop**：候选池 $\mathcal{P}$ 从 base system 开始。循环：(i) 选有希望的候选，(ii) 在 minibatch 上提出并评估变体，(iii) 若超父代则加入 $\mathcal{P}$（带谱系记录）并在 $D_{pareto}$ 上评估。候选经 **reflective mutation 或 crossover** 派生，**沿谱系树累积知识**（每个继承父代 + 自己 rollout 的学习信号）。预算耗尽后返回 $D_{pareto}$ 上聚合性能最好的候选。

**Reflective Prompt Mutation**：从执行 trace 提取模块的输入/输出/推理，调反馈函数 $\mu_f$（返回数值分 + 文本反馈：编译错误、失败 rubric）；round-robin 选一个模块，反射 LM 看 (当前 prompt, 程序 trajectory, 分数, 反馈) 反射式地把成功/失败归因到 prompt 元素、提出修订指令。更新后在 minibatch 重评，改进则加入候选池。

**Evaluation traces as diagnostic signals**：除执行 trace 外，**评估 trace**（环境计算 reward 前产生的文本，如编译/执行/profiling 输出）也是宝贵诊断源。$\mu_f$ 把 reward $\mu$ 扩展成反馈函数，提取评估中的文本 trace 连同最终分数返回。可以是模块特定的（如 multi-hop 每 hop 后给反馈）。

![Fig 3](../images/bc1702e6c57bac481885a2c9c9283ac4f83e6422a70b6690dea7b96c55fe0acf.jpg)

*Figure 3: GEPA 每轮用两种策略之一（Reflective Prompt Mutation 或 System Aware Merge）改进现有候选，先在 minibatch 评估、改进则在更大数据集评估。不总选最佳候选（会陷局部最优），而是引入 Pareto-based candidate sampling——从每任务最佳候选列表过滤并采样，确保足够多样性。*

## 3.1 Pareto-based Candidate Selection（★ 用户重点）

**朴素做法 = 总选最佳候选** → 常陷局部最优：一旦找到主导策略，难以超越，优化器耗尽预算而学不到新的更好策略（Fig 6a：找到一个新策略后反复精修、失败、耗尽预算）。

**GEPA 的 Pareto "illumination" 策略**（Algorithm 2，Mouret & Clune 2015 的 MAP-Elites 血统）：
1. 对每个训练实例，记录所有候选中的最高分 → 形成 **Pareto frontier**。
2. **在至少一个任务上取得最佳分的候选被保留**，严格被支配的被剪枝。
3. 从剪枝后的集合中**随机采样一个候选，概率正比于它领先的任务数**。

这让 GEPA 逃离局部最优而不膨胀搜索，把资源聚焦在体现"winning"策略的候选上。

![Fig 6](../images/42267125c7efb067f17311c2bb75bb59dda4fb4fac3284551d9e92af1747b33b.jpg)

*Figure 6: 候选选择策略对比。(左) 每轮选最佳候选 → 一轮后陷局部最优、搜索次优。(右) Pareto-based 选择 → 平衡的搜索树，同预算内找到更好的程序。*

**Algorithm 2（SelectCandidate）核心**：
```
1. 对每个实例 i：s*[i] ← max_k S_{P[k]}[i]           # 实例级最高分
2. P*[i] ← {P[k] : S_{P[k]}[i] = s*[i]}              # 在 i 上最优的候选
3. C ← ∪_i P*[i] 中的唯一候选                          # Pareto 前沿候选
4. 剪掉被支配的候选 → 得 Ĉ
5. f[Φ] ← Φ 领先的任务数
6. 按概率 ∝ f[Φ] 采样候选 Φ_k                          # 频率加权随机
```

**候选选择消融（Table 3，最关键的一张表）**：

| 选择策略 | 用于 | 聚合增益 |
|---------|------|---------|
| SelectBestCandidate（greedy） | TextGrad | +6.05% |
| BeamSearch(N=4) | APO | +5.11% |
| **GEPA Pareto-based** | GEPA | **+12.44%** |

> 💡 **§3.1 + Table 3 = 用户改进 Self-Harness 的核心武器（务必内化）**（Hao 批注）：这是全篇对用户最重要的部分——**Pareto illumination 把增益从 greedy 的 +6% 提到 +12.44%（翻倍）**，且在**相同预算**下。直接对应用户的判断"Self-Harness 现在比较 greedy"：
> - **[Self-Harness](../%5BArxiv%202026%5D%20Self-Harness/) 现状**：单一 harness 谱系 $h_0→h_1→...$，每轮 K 提案挑通过的 merge，本质是"总从当前最佳 harness 出发"——正是 GEPA 说的会陷局部最优的 greedy。
> - **GEPA 的处方**：维护候选**池**（不是单一谱系）+ 实例级 Pareto 前沿 + 频率加权随机采样。关键洞察：**一个在"整体"上不是最佳、但在"某些任务"上最佳的 harness，值得保留和探索**——因为它可能承载被全局最佳掩盖的 winning 策略。
> - **对用户的直接改造**：把 Self-Harness 的"当前 harness $h_t$"换成"harness 候选池 + 实例级（per-task）Pareto 前沿"；每轮不从"最佳 harness"而从"Pareto 前沿按领先任务数加权采样"的 harness 出发提案。这就把 Self-Harness 从 greedy 单谱系变成 GEPA 式 population 搜索。
> - **⚠️ 开放问题（弱自 proposer）**：Self-Harness 用**弱的 target 自己**当 proposer（而 GEPA 用独立 reflection LM）。Pareto illumination 在弱自 proposer 下是否还有效、是否会因自评估噪声（[AHE](../%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) regression blindness / [Phantom-Guardrails](../%5BArxiv%202026%5D%20Phantom-Guardrails/) 幻觉）而失效——这是用户可做的实验贡献。[Meta-Harness](../%5BArxiv%202026%5D%20Meta-Harness/) 已证明强外部 proposer + Pareto 有效，Self-Harness 退成 greedy，弱自 proposer + Pareto 是中间的未知地带。

## System Aware Merge（crossover）

**Merge = system-aware crossover**（Appendix D.1，Algorithm 3/4）：识别学到**互补策略**的不同优化谱系，合并——从每个谱系挑不同模块的最佳版本组成单一最优候选。**合并条件（严格）**：候选须共享公共祖先、但优化了**不相交的 prompt 集合**（互补策略）、都是 Pareto-optimal、且都超过祖先的聚合性能。因条件严格，merge 稀疏发生。

GEPA+Merge 可超 GEPA 5%（聚合 +2%）。但**最优 mutation/crossover 预算分配和何时触发 merge 需进一步研究**——GEPA+Merge 对 GPT-4.1 Mini 好，对 Qwen3 8B 反而降（超参没针对性调）。直觉：merge 在有独立且都表现好的谱系时收益最大，应在优化树演化出足够不同的谱系后才触发。

> 💡 **Merge/crossover 批读（对用户的第二个武器）**（Hao 批注）：这是用户提到的"crossover / merge complementary improvements"的具体实现。关键设计对用户改 Self-Harness 的 **MergeAccepted** 极有参考：
> - Self-Harness 现在的 MergeAccepted 是"把同轮通过的多个编辑合并"——但没有 GEPA 这种**跨谱系、互补性判据**（共享祖先 + 不相交 prompt 集 + 都 Pareto-optimal + 都超祖先）。
> - GEPA 的严格合并条件正好回应 [AHE](../%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) 的"组件非加性交互"警告——只合并**优化了不相交组件**的候选（互补而非冲突），降低非加性交互导致的增益 cap。
> - **对用户**：若给 Self-Harness 加多 lineage（Pareto），就需要 GEPA 式的 merge 判据来安全合并互补 harness——共享祖先 + 编辑不同组件（tool vs middleware vs memory）+ 都 Pareto-optimal。这与 AHE 的组件观测（7 类组件即文件）天然契合：不同 lineage 编辑不同组件文件，merge 时挑各组件最佳版本。

## 4-5. 主要结果

- **vs GRPO**（Qwen3 8B）：GEPA 平均 +9.62 vs GRPO +3.68，用 3936 vs 24000 rollout（~6× 少）；5/6 任务超 GRPO，最高 +19%，样本效率最高 78×。
- **vs MIPROv2**（prompt 优化 SOTA）：GEPA 聚合增益 +9.62~+13.33，是 MIPROv2 +5.64 的两倍多；prompt 还短 up to 9.2×。
- **跨模型泛化**：GEPA-Qwen-Opt（完全用弱 Qwen3-8B 优化）在 GPT-4.1-Mini 上 +9%，超所有直接为 GPT-4.1-Mini 优化的基线。
- **推理时搜索**：GEPA 也用作 code optimization（NPUEval/KernelBench）和 adversarial prompt search 的推理时搜索策略。

> 💡 **总结 + 对用户的三点提炼（Hao 批注）**：GEPA 对用户"明显更强的 Self-Harness"的三点贡献：
> 1. **Pareto-based candidate selection（核心）**：用实例级 Pareto 前沿 + 频率加权采样替代 Self-Harness 的 greedy 单谱系，Table 3 证明翻倍增益。
> 2. **genetic tree + population/archive**：维护候选池和谱系，沿树累积经验，而非单一 $h_t$。
> 3. **System Aware Merge（crossover）**：严格判据（共享祖先 + 不相交组件 + 都 Pareto-optimal）安全合并互补 lineage，回应 AHE 的非加性交互警告。
> **但用户要甄别**：GEPA 是 prompt-only + 需标签的；要把它的**搜索结构**（不是范围/标签依赖）移植到 Self-Harness，并验证在**弱自 proposer** 下是否成立（GEPA/Meta-Harness 都用强/独立 proposer）。**理想合成**：Self-Harness 的自改进内化 + GEPA 的 Pareto 搜索 + [RHO](../%5BArxiv%202026%5D%20RHO-Self-Preference/) 的无标签 self-preference + [Phantom-Guardrails](../%5BArxiv%202026%5D%20Phantom-Guardrails/) 的反事实去幻觉 + [AHE](../%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) 的 regression 预测/evidence ledger。

> 💡 **Q&A 批注记录**（Hao 批注）：
> - **Q：GEPA 对用户最有用的一点？**
>   A：Pareto-based candidate selection（§3.1/Algorithm 2/Table 3）——实例级 Pareto 前沿 + 频率加权随机采样，把 greedy +6% 提到 +12.44%，直接治 Self-Harness 的 greedy 单谱系。
> - **Q：为什么 greedy 会陷局部最优？**
>   A：总选全局最佳候选 → 一旦找到主导策略就反复精修、难超越、耗尽预算。Pareto illumination 保留"在某任务上最佳"的候选，探索被全局最佳掩盖的 winning 策略。
> - **Q：Merge 什么时候用？**
>   A：有独立且都表现好的谱系时。严格条件：共享祖先 + 优化不相交 prompt 集（互补）+ 都 Pareto-optimal + 都超祖先。预算分配/触发时机是 open problem。
> - **Q：GEPA 能直接当 harness 优化器吗？**
>   A：它是 prompt-only + 需标签。搜索结构可移植到 harness，但需换成完整 harness 候选、并解决弱自 proposer + 无标签（结合 RHO）的问题。
