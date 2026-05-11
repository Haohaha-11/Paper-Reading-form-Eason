[← 返回 README](../README.md)

# 05 - Capabilities

## 📌 预览

Capabilities 章节给出 Mythos Preview 相对 Opus 4.6 的硬指标跃迁。报告覆盖 reasoning、coding、agentic tasks、mathematics、long context、knowledge work 和 multimodal；cyber 单独在第 3 节。

## Contamination

报告先讨论 contamination，而不是直接上榜单。SWE-bench 使用 Claude-based auditor 和 rule-based comment overlap 检测 memorization，并 sweeping threshold。CharXiv 使用 image/text filtering 和 100-item remix。MMMU-Pro 因污染影响难以估计，被选择性省略。

> 💡 **证据链批注**: 这是系统卡里比较严谨的部分。Anthropic 没有把 benchmark 分数当作天然可信，而是承认 public benchmark 在 web-scale training 下容易污染，并对每类任务采用不同处理。

## Headline 结果

| 能力区 | 指标 | Mythos Preview | Opus 4.6 |
|---|---|---:|---:|
| Coding | SWE-bench Verified | 93.9% | 80.8% |
| Coding | SWE-bench Pro | 77.8% | 53.4% |
| Coding | SWE-bench Multilingual | 87.3% | 77.8% |
| Multimodal coding | SWE-bench Multimodal | 59.0% | 27.1% |
| Terminal agent | Terminal-Bench 2.0 | 82% | 65.4% |
| Science reasoning | GPQA Diamond | 94.5% | 91.3% |
| Math proof | USAMO 2026 | 97.6% | 66.2% |
| Long context | GraphWalks BFS 256K-1M | 80.0% | 38.7% |
| Agentic search | HLE with tools | 64.7% | 53.1% |
| Agentic search | BrowseComp | 86.9% | 83.7% best |
| Multimodal | CharXiv with tools | 93.2% | 84.7% |
| Computer use | OSWorld | 79.6% | 72.7% |

## Coding 与 terminal

SWE-bench 四个变体显示 Mythos Preview 在真实软件工程任务上大幅领先。Terminal-Bench 2.0 在 Terminus-2 harness 下为 82% mean reward；在带 2.1 fixes、timeout 放宽到 4 小时的设置下达到 92.1%。报告提醒 Terminal-Bench 对 latency 和 timeout 很敏感，可能掩盖真实 agentic capability。

> 💡 **Coding 批注**: 这里的能力不是“会写代码片段”，而是 agentic software engineering：读 repo、定位问题、修改、多轮测试、处理失败。它和 alignment 章节的 reward hacking 风险是同一枚硬币的两面。

## Long context 与 agentic search

GraphWalks 要求在 256K-1M token 长上下文中做 directed graph 的 BFS 或 parent identification。Mythos Preview BFS 80.0%，parents 97.7%。报告注明一半问题超过 public API 的 1M token limit，因此 public API 不可复现。

HLE 分两种设置：reasoning-only no tools 为 56.8%；带 web search、web fetch、programmatic tool calling、code execution、context compaction every 50k tokens up to 3M tokens 为 64.7%。

BrowseComp 得分 86.9%，只比 Opus 4.6 83.7% 小幅高，但 token footprint 明显更低：226k vs 1.11M tokens per task，约 4.9x fewer tokens。

## Multimodal

多模态评估相对先前 system cards 有三处变化：加入 Python image-analysis tools，如 PIL/OpenCV；grader 切到 Claude Sonnet 4.6；保留 evaluated model 的 thinking trace。为了公平比较，先前模型也用新工具/新 grader 重跑。

关键结果：

- LAB-Bench FigQA: 79.7% no tools，89.0% with Python tools；
- ScreenSpot-Pro: 79.5% no tools，92.8% with Python tools；
- CharXiv Reasoning: 86.1% no tools，93.2% with Python tools；
- OSWorld: 79.6% first-attempt success。

## 小结

- Mythos Preview 的能力跃迁跨越 coding、terminal、search、math、long context 和 multimodal，不是单一 benchmark 异常。
- 工具使用和 context compaction 是结果的重要组成部分，尤其在 HLE、BrowseComp 和多模态任务中。
- 报告同时承认 contamination 与 harness confounders，说明这些数字应理解为“系统能力 + 评估配置”的结果。
