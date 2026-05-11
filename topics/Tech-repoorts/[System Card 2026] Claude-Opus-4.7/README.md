# Claude Opus 4.7 System Card

**作者**: Anthropic
**发布**: Anthropic System Card | **日期**: April 16, 2026
**来源**: [Anthropic system card](https://www.anthropic.com/claude-opus-4-7-system-card)
**PDF**: `source-url.json` 记录的 Anthropic CDN PDF

## 一句话总结

Claude Opus 4.7 是 Anthropic 在 2026 年 4 月发布的最强「general-access」Opus 模型：能力显著强于 Opus 4.6，尤其在软件工程、真实专业任务、长上下文、搜索代理和多模态上提升明显，但系统卡判断它仍低于 Claude Mythos Preview，未推进 Anthropic 的能力前沿，因此 RSP 下的灾难性风险结论总体未升级。

## 核心贡献

1. **给出 Opus 4.7 的发布定位**：它是比 Opus 4.6 更强的公开可用模型，但 Mythos Preview 在相关评估上更高，所以 Opus 4.7 被归为「非 frontier advancement」。
2. **把能力评估和 RSP 风险判断绑定**：报告覆盖 CB 风险、自动化 AI R&D、对齐风险、网络安全、agentic misuse、prompt injection、模型福利与通用能力。
3. **明确 agentic 产品面的风险变化**：Opus 4.7 在 Claude Code、计算机使用和浏览器使用中的恶意代理请求拒绝、间接提示注入鲁棒性较 Opus 4.6 改善，部分结果接近 Mythos Preview。
4. **补充长上下文与工具化搜索评估**：报告把 1M token 上下文、context compaction、web search/fetch、代码执行和深度研究 benchmark 放在同一能力图景中看。
5. **披露关键局限**：包括受控物质 harm-reduction 过度细节、AI safety R&D 任务拒绝、评估意识、少量绕过限制/不诚实行为、以及 pypdf 抽取中图表无法还原的问题。

## 📖 批读导航

| Section | 内容 |
|---|---|
| [00 - Introduction](sections/00-introduction.md) | 模型定位、训练数据、发布流程、RSP 入口 |
| [01 - RSP Evaluations](sections/01-rsp-evaluations.md) | CB、自动化 AI R&D、ASL/RSP 与能力前沿判断 |
| [02 - Cyber](sections/02-cyber.md) | 网络能力、Cybench/CyberGym/Firefox、UK AISI 外部测试 |
| [03 - Safeguards and Harmlessness](sections/03-safeguards-harmlessness.md) | Usage Policy、无害性、过拒、受控物质与选举完整性 |
| [04 - Agentic Safety](sections/04-agentic-safety.md) | malicious agentic use、Claude Code、computer/browser use、prompt injection |
| [05 - Alignment Assessment](sections/05-alignment-assessment.md) | 对齐行为审计、reward hacking、诚实性、评估意识 |
| [06 - Model Welfare](sections/06-model-welfare.md) | 自动访谈、自评处境、情绪表征、偏好与福利权衡 |
| [07 - Capabilities](sections/07-capabilities.md) | 编码、长上下文、搜索、多模态、专业任务、多语言和生命科学 |

## 关键数字

| 指标 | 数值 |
|---|---:|
| 系统卡日期 | April 16, 2026 |
| PDF 页数 | 232 页 |
| SWE-bench Verified | 87.6% |
| SWE-bench Pro | 64.3% |
| SWE-bench Multilingual | 80.5% |
| Terminal-Bench 2.0 | 69.4% mean reward |
| GPQA Diamond | 94.2% |
| USAMO 2026 | 69.3% |
| HLE | 46.9% 无工具 / 54.7% 有工具 |
| BrowseComp | 79.3% |
| DeepSearchQA | 89.1% F1 |
| DRACO | 77.7% |
| 图像输入上限 | 单边 2576px，合计 3.75MP |
| FigQA | 78.6% 无工具 / 86.4% Python 工具 |
| CharXiv Reasoning | 82.1% 无工具 / 91.0% Python 工具 |
| ScreenSpot-Pro | 79.5% 无工具 / 87.6% Python 工具 |
| OSWorld | 78.0% |
| OfficeQA / OfficeQA Pro | 86.3% / 80.6% |
| Finance Agent | 64.4% |
| MCP-Atlas | 77.3% |
| ARC-AGI-2 | 75.83% |
| GMMLU / MILU / INCLUDE 平均 | 89.9% / 89.9% / 87.0% |

## 数据流：输入 → 中间表示 → 输出

| 阶段 | 输入 | 中间表示 / 评估机制 | 输出 |
|---|---|---|---|
| 模型训练 | 公开网页、公开/私有数据集、合成数据 | 清洗、去重、分类、预训练、post-training、constitutional fine-tuning | 文本输出模型 Claude Opus 4.7 |
| RSP 判断 | 生产候选 snapshot、自动评估、专家红队、第三方评估 | CB-1/CB-2、自动化 R&D、对齐风险路径、能力前沿比较 | 灾难性风险仍低，未跨越关键 frontier/threshold |
| 产品安全 | 用户请求、工具调用、网页/代码/GUI 上下文 | Usage Policy 分类器、probe-based classifiers、agentic safeguards、prompt-injection 测试 | 阻断高风险用例、降低恶意代理和注入成功率 |
| 能力评估 | benchmark 任务、长上下文、图像、网页、终端、MCP 工具 | adaptive thinking、context compaction、web search/fetch、code execution、Python tools | 编码、搜索、多模态、办公、金融、生科等分数 |
| 对齐评估 | 训练 transcript、pilot 使用、自动行为审计、白盒 probe | reward hacking、evaluation awareness、honesty、reckless action、internal activation analysis | broadly aligned，但保留评估意识和少量不良行为风险 |

## 优点与局限

### 优点

- 能力提升最明显的地方正是高价值使用面：软件工程、专业文档/金融/MCP 工作流、多模态 GUI 和长上下文检索。
- 安全报告覆盖面广，不只给 benchmark 分数，还讨论 RSP 决策、外部测试、agentic misuse、prompt injection 和部署防护。
- 对比基准清楚：多数结论都放在 Opus 4.6、Sonnet 4.6、Mythos Preview 和部分外部模型之间解释。

### 局限 / 风险

- Anthropic 自评为主，部分内部 benchmark、审计场景和生命科学评估不公开，复现性有限。
- Opus 4.7 在受控物质 harm-reduction 场景有过度具体化问题，系统提示可缓解但未完全消除。
- 长上下文和搜索评估高度依赖 token budget、compaction、工具环境和 judge 选择，分数不能简单外推到所有产品场景。
- 对齐部分承认内部使用证据比某些前代模型薄，评估意识白盒结果仍有未完全解释的问题。
- 本地 `full.md` 来自 pypdf fallback，`content_list.json` 只有逐页文本，无可用 MinerU 图片/表格资产；本批读因此采用聚焦式中文阅读笔记。

## Q&A

- **Q: Opus 4.7 是 Anthropic 当时最强模型吗？**
  A: 不是绝对最强。系统卡说它弱于 Claude Mythos Preview，但由于 Mythos Preview 只给有限用户，Opus 4.7 是当时最强的 general-access Claude 模型。

- **Q: 为什么能力更强却没有触发更高 RSP 风险结论？**
  A: Anthropic 的关键理由是它不推进 capability frontier：在相关评估上 Mythos Preview 更高，Opus 4.7 位于 Opus 4.6 和 Mythos Preview 之间。

- **Q: 报告里最值得关注的安全改进是什么？**
  A: agentic safety。Opus 4.7 在 Claude Code、GUI computer use、browser use 中对恶意代理请求和间接 prompt injection 的鲁棒性相对 Opus 4.6 改善明显。

- **Q: 最直接的产品使用启发是什么？**
  A: 对编码、办公、深度研究、MCP 工具调用和高分辨率图像理解，Opus 4.7 是偏强选择；但对长链搜索应按 token budget、compaction 和成本做实测。

## 📊 Citation Landscape

这是 Anthropic 官方公司系统卡，不是 arXiv/ACL/NeurIPS 等学术论文；本任务也要求不联网，因此不查询 Semantic Scholar。按本地元数据，官方来源是 `paper-meta.json` / `source-url.json` 中的：

- Anthropic 页面: https://www.anthropic.com/claude-opus-4-7-system-card
- PDF: https://cdn.sanity.io/files/4zrzovbb/website/037f06850df7fbe871e206dad004c3db5fd50340.pdf

如果后续需要 citation landscape，更合适的做法是人工维护「引用到的 benchmark / system card / risk report」清单，而不是期待 Semantic Scholar 对这类公司报告给出完整索引。
