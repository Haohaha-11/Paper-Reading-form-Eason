# Harness（Agent 自改进 harness）

这个 topic 关注 **agent harness 的自动/自改进演化**：harness = 模型与环境之间的非参数脚手架（system prompt、工具、runtime 策略、验证规则、编排逻辑、memory、middleware）。核心追问：**当把"改 harness"这件事交给 agent 自己/自动优化器时，如何让它稳定、可迁移、可信地改进——而不是陷入 greedy 局部最优、依赖昂贵标签、或幻觉出不存在的失败？**

本 topic 围绕锚论文 **[Self-Harness](./%5BArxiv%202026%5D%20Self-Harness/)** 组织：其余 5 篇按"**理解 novelty**（前 3 篇）+ **改进方法**（后 2 篇）"分层，目标是想清楚 Self-Harness 的三个阶段（Weakness Mining / Proposal Search / Validation）如何改成"明显更强的一篇"。

## 论文列表

### ⭐ 锚论文
| 论文 | 会议 | 一句话 |
|------|------|--------|
| [Self-Harness](./%5BArxiv%202026%5D%20Self-Harness/) | Arxiv 2026 | agent 改进自己运行的 harness（无人类/外部强 agent）。三阶段：Weakness Mining（聚类失败）→ Harness Proposal（多样但最小编辑）→ Proposal Validation（held-out 回归门）。9/9 组合双升，最高 +132%。 |

### 一、理解 novelty：前置工作与竞争工作
| 论文 | 会议 | 与 Self-Harness 的关系 |
|------|------|----------------------|
| [Meta-Harness](./%5BArxiv%202026%5D%20Meta-Harness/) | Arxiv 2026 | **最直接前作**。外部强编码 agent（Claude Code+Opus4.6）+ 文件系统全 trace 访问 + population/Pareto 搜索 harness。Self-Harness = 沿它"往前推一步"（外部 meta-agent → 自己）。消融证明 raw trace 访问是命门。 |
| [Agentic Harness Engineering](./%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) | Arxiv 2026 | **并行独立的相似工作**。三 observability 支柱 + **evidence ledger / change manifest**（每编辑绑 failure evidence+root cause+fix+regression risk，跨轮验证+文件级 rollback）。关键发现：self-attribution 对 fix 可靠（5×random）对 **regression 盲**（2×random）。 |
| [RHO](./%5BArxiv%202026%5D%20RHO-Self-Preference/) | Arxiv 2026 | **无标签自偏好**。不依赖 labeled validation，用 self-validation + cross-trajectory self-consistency + pairwise self-preference 产生优化信号。单轮 SWE-Bench Pro 59→78，~1/3 compute 达 Meta-Harness 10 轮天花板。 |

### 二、改进 Self-Harness 的两把武器
| 论文 | 会议 | 给 Self-Harness 的升级 |
|------|------|----------------------|
| [GEPA](./%5BArxiv%202025%5D%20GEPA/) | ICLR 2026 | **候选搜索/选择**。reflection + semantic mutation + **population/archive + Pareto illumination** + System Aware Merge（crossover）。Table 3：Pareto 选择 +12.44% vs greedy +6.05%。治 Self-Harness 的 greedy 单谱系。 |
| [Phantom Guardrails](./%5BArxiv%202026%5D%20Phantom-Guardrails/) | Arxiv 2026 | **精准反问题**。自改进 harness 会幻觉出不存在的失败并加 guardrail。诊断 Weakness Mining 的隐藏假设"LLM 诊断 grounded"；解法 = **warrant-aware acceptance**（引用可验证失败才接受）+ 三个 lever（中性 charter / 认证 taxonomy / warrant crediting）。 |

## 三阶段升级蓝图（本 topic 的核心综合）

Self-Harness 三阶段各有明确改进空间，正好对应上面 5 篇：

| Self-Harness 阶段 | 当前做法（greedy/需标签/易幻觉） | 升级 | 来源 |
|---|---|---|---|
| **Weakness Mining** | LLM 诊断失败→机制（假设诊断 grounded） | Failure Hypothesis → **Counterfactual Verification** → Validated Mechanism；加 self-consistency 信号；加 **"Do Nothing"** 候选 | [Phantom](./%5BArxiv%202026%5D%20Phantom-Guardrails/) + [RHO](./%5BArxiv%202026%5D%20RHO-Self-Preference/) |
| **Proposal Search** | K 提案→挑通过→merge（greedy 单谱系） | **population/archive + Pareto illumination + crossover**；考虑组件非加性交互 | [GEPA](./%5BArxiv%202025%5D%20GEPA/) + [AHE](./%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) |
| **Proposal Validation** | held-out 回归门（需 labeled verifier，仅 suppression） | **warrant-aware acceptance** + 无标签 self-preference + **主动 regression 预测**（evidence ledger） | [Phantom](./%5BArxiv%202026%5D%20Phantom-Guardrails/) + [RHO](./%5BArxiv%202026%5D%20RHO-Self-Preference/) + [AHE](./%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) |

**贯穿主线：self-assessment 双向不可信**——[AHE](./%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) 证明对 regression 盲（假阴性），[Phantom](./%5BArxiv%202026%5D%20Phantom-Guardrails/) 证明对 fabricated failure 假阳性。故任何"明显更强的 Self-Harness"都必须用**外部/反事实验证**替代纯 self-judgment，同时（借 [RHO](./%5BArxiv%202026%5D%20RHO-Self-Preference/)）尽量去标签依赖，（借 [GEPA](./%5BArxiv%202025%5D%20GEPA/)）广探索多 lineage。

## 推荐阅读顺序

**理解 novelty**：[Meta-Harness](./%5BArxiv%202026%5D%20Meta-Harness/) → [Self-Harness](./%5BArxiv%202026%5D%20Self-Harness/) → [AHE](./%5BArxiv%202026%5D%20Agentic-Harness-Engineering/) → [RHO](./%5BArxiv%202026%5D%20RHO-Self-Preference/)（搞清 Self-Harness 相对前作/并行工作的定位与代价）。

**改进方法**：[GEPA](./%5BArxiv%202025%5D%20GEPA/) → [Phantom Guardrails](./%5BArxiv%202026%5D%20Phantom-Guardrails/)（如何把 Weakness Mining / Proposal Search / Validation 改成明显更强的方法）。
