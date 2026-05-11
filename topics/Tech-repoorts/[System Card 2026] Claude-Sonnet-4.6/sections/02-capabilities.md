[← 返回 README](../README.md)

# 02 - Capabilities

## 📌 预览

Capabilities 是最长的一章，核心画像是：Sonnet 4.6 是强 coding/agentic/search/computer-use 模型，长上下文和多模态明显进步，金融与生命科学任务开始接近高端专业工作流。多数主表结果为 10 trials 平均；上下文窗口按评估不同，但不超过 1M。

## 总表读法

| 任务 | Sonnet 4.6 | 对比信号 |
|---|---:|---|
| SWE-bench Verified | 79.6% | 接近 Opus 4.6 80.8%、GPT-5.2 80.0% |
| Terminal-Bench 2.0 | 59.1% | Sonnet 4.5 为 51.0% |
| tau2-bench Retail/Telecom | 91.7% / 97.9% | 接近 Opus 4.6 91.9% / 99.3% |
| MCP-Atlas | 61.3% | 高于 Sonnet 4.5 43.8% |
| OSWorld-Verified | 72.5% | 距 Opus 4.6 72.7% 只差 0.2pp |
| ARC-AGI-2 Verified | 58.3% | Sonnet 4.5 为 13.6% |
| GPQA Diamond | 89.9% | 接近 Opus 4.6 91.3% |
| MMMLU | 89.3% | 非英语 14 语言平均 |
| GDPval-AA | 1633 ELO | 高于 Opus 4.6 1606 |
| MMMU-Pro | 74.5% / 75.6% | no tools / with tools |
| HLE | 33.2% / 49.0% | no tools / with tools |

> 💡 **表 2.1.A 批读**: 总表不是单一排行榜，而是“模型在哪些能力接近 Opus 4.6”的证据图。Sonnet 4.6 在 OSWorld、GDPval-AA、MCP-Atlas、MMMU-Pro no-tools 等处非常接近或局部超过 Opus 4.6，但在 ARC-AGI-2、HLE、long-context MRCR 1M 上仍明显低于 Opus。

## Coding / terminal / tool use

- SWE-bench Verified: 79.6%；SWE-bench Multilingual: 75.9%，覆盖 300 个问题、9 种编程语言。
- SWE-bench prompt 增强后到 80.2%，提示内容强调多用工具、先写测试、理解根因、覆盖边界情况。
- Terminal-Bench 2.0: 89 tasks，5 次运行，GKE n2-standard-32，pass rate 59.1%。
- OpenRCA: 335 software failure cases、68.5GB telemetry；Sonnet 4.6 high effort 27.9%，Sonnet 4.5 为 12.9%。
- MCP-Atlas: 61.3%，评估真实 MCP tool workflows。

> 💡 **coding 批读**: Sonnet 4.6 的 coding 能力不是只体现在 patch generation，还包括 terminal、root-cause analysis、MCP tool discovery/invocation。报告特别给出“先测试、读代码、工具超过 100 次”的 prompt，说明 scaffolding/行为规范会显著影响工程任务表现。

## Computer use / web agents

- OSWorld-Verified: 1080p Ubuntu VM，最多 100 actions/task；Sonnet 4.6 达 72.5% first-attempt success。
- WebArena: 65.6%，single policy model/general prompts；低于 WebTactix 74.3% 和 OAgent 71.6% multi-agent systems。
- WebArena-Verified: 报告称 Sonnet 4.6 在 full set 上超过 Opus 4.6，采用官方 prompt 和 grader。

> 💡 **agentic GUI 批读**: OSWorld/WebArena 数字说明模型已经能稳定操作真实软件界面，但 alignment 章节会指出同一类 GUI 能力也带来 over-eagerness 和安全不稳定。能力章和安全章必须合读。

## Long context

| Benchmark | 设置 | Sonnet 4.6 |
|---|---|---:|
| MRCR v2 256K 8-needle | Mean Match Ratio | 90.6 (64k), 90.3 (max) |
| MRCR v2 1M 8-needle | Mean Match Ratio | 65.1 (64k), 65.8 (max) |
| GraphWalks BFS 1M | F1 | 68.4 (64k), 73.8 (max) |
| GraphWalks Parents 256K subset | F1 | 96.9 (64k), 97.9 (max) |

> 💡 **long-context 批读**: MRCR 是精确顺序检索，GraphWalks 是长上下文图推理。Sonnet 4.6 在 MRCR 1M 落后 Opus 4.6，但在 GraphWalks 上被报告为 Anthropic 最强模型。这说明“长上下文能力”不是单一指标：检索、排序、多跳推理会分化。

## Multimodal

- LAB-Bench FigQA: 58.8% 无工具，77.1% 加 image cropping tool。
- MMMU-Pro: 74.5% 无工具，75.6% 加 cropping tool；Sonnet 4.5 为 67.5%/68.0%。
- CharXiv Reasoning: 72.4% 无工具，77.4% 加 cropping tool；有工具时与 Opus 4.6 77.4% 持平。

> 💡 **多模态批读**: image cropping tool 对 FigQA 提升最大，从 58.8% 到 77.1%。这表明瓶颈并非纯知识，而是视觉定位和局部细节读取；工具化视觉前处理仍是多模态 agent 的关键增益来源。

## Agentic Search

BrowseComp:

- 1,266 questions，需要 web search。
- thinking disabled 时更好。
- single-agent: 74.01%。
- compute scaling: 1M sampled tokens 64.69%，3M 为 69.67%，10M 为 74.01%。
- multi-agent orchestrator + 200k subagent context: 82.07%，高 single-agent 8.1pp。

DeepSearchQA:

- 900 prompts，17 fields，多步信息搜索。
- single-agent best: 89.2% F1。
- multi-agent: 91.1% F1，提升 1.9pp。

> 💡 **agentic search 批读**: 这里最重要的变量不是模型权重，而是计算预算和 agent 架构。BrowseComp 从 1M 到 10M tokens 提升近 9.3pp，multi-agent 再提升 8.1pp；这说明 frontier search performance 很大程度来自 inference-time compute 和任务分解。

## Finance / multilingual / life sciences

- Finance Agent: 63.3% accuracy，报告称 Max Thinking 下 state-of-the-art。
- Real-World Finance: 约 50 个 analyst workflow 任务，覆盖 investment banking、private equity、hedge funds/public investing、corporate finance；输出约 80% spreadsheets、13% slide decks、7% Word docs。
- GMMLU: all languages overall average 88.7%；average gap to English -4.4%，worst gap -16.2%，低资源非洲语言退化明显。
- MILU: average 89.6%；English-to-Indic gap -2.3%，Hindi 92.8% 反超 English 1.1pp。
- Life sciences: BioPipelineBench 52.1%，BioMysteryBench 50.4%，Structural Biology MC 85.3% / open-ended 24.7%，Organic Chemistry 48.4%，Phylogenetics 49.1%。
- MedCalc-Bench Verified: 86.24%，略高于 Opus 4.6 的 85.24%。

## 🔖 Section 总结

Sonnet 4.6 的能力提升最明显地集中在“能用工具长时间做复杂任务”的方向：coding、search、GUI、finance workflow、生命科学计算。关键追问是：这些高分有多少来自模型本体，有多少来自 max effort、context compaction、multi-agent、cropping/tooling 和 prompt engineering。
