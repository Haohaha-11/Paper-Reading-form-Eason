[← 返回 README](../README.md)

# 07 - Capabilities

## 📌 预览

Capabilities 章节展示 Opus 4.7 的主要产品价值：它相对 Opus 4.6 全面增强，最大增益集中在真实专业任务和软件工程；但总体仍弱于 Mythos Preview。标准设置多为 adaptive thinking、max effort、默认采样，部分评估使用工具、context compaction 或 Python。

## 软件工程与终端

- **SWE-bench Verified**: 87.6%
- **SWE-bench Pro**: 64.3%
- **SWE-bench Multilingual**: 80.5%
- **SWE-bench Multimodal**: 34.5%
- **Terminal-Bench 2.0**: 69.4% mean reward，89 个任务共 445 次 trial，thinking disabled。

> 💡 **编码能力批注**: SWE-bench Pro 和 Terminal-Bench 更接近真实工程难度：多文件、无公开答案泄漏、终端环境、超时和工具延迟。Opus 4.7 的优势不只是会写代码，而是在 agentic loop 里推进任务。

## 推理、数学与长上下文

- **GPQA Diamond**: 94.2%
- **MMMLU**: 91.5%
- **USAMO 2026**: 69.3%，比赛在 2026 年 3 月 21-22 日，晚于训练数据 cutoff。
- **GraphWalks**: BFS 256K-1M 为 58.6%，parents 256K-1M 为 75.1%；256K 子集 BFS 76.91%、parents 93.57%。
- **MRCR v2**: 使用 8-needle 难度，测试长上下文里区分相似信息的精确顺序推理。

> 💡 **长上下文批注**: 报告强调不是简单 needle-in-haystack，而是多轮共指和顺序定位。对真实知识库/长会话使用，关键不是“能塞 1M token”，而是能否在相似片段中取出正确实例。

## Agentic search / browsing

- **HLE**: 46.9% 无工具，54.7% 有 web search/fetch、programmatic tool calling、code execution。
- **BrowseComp**: 79.3%，thinking off、max effort、10M token limit、200k token 触发 compaction；此项 Opus 4.6 在 10M 下更高，为 83.7%。
- **DeepSearchQA**: 89.1% F1，80.7% fully correct。
- **DRACO**: 77.7%，提升集中在 breadth/depth 和 citation quality。

> 💡 **搜索批注**: 搜索代理分数高度依赖 token budget、compaction 策略、工具可用性和 judge。BrowseComp 上 Opus 4.6 反超说明“新模型更强”不等于每个搜索成本曲线都更优，实际产品应按任务和预算试跑。

## Multimodal / computer use

Opus 4.7 支持最大单边 **2576px**、总计 **3.75MP** 的图像输入；之前模型包括 Mythos Preview 和 Opus 4.6 为 **1568px / 1.15MP**。

- **LAB-Bench FigQA**: 78.6% 无工具，86.4% Python tools。
- **CharXiv Reasoning**: 82.1% 无工具，91.0% Python tools。
- **ScreenSpot-Pro**: 79.5% 无工具，87.6% Python tools。
- **OSWorld**: 78.0% first-attempt success rate。

> 💡 **多模态批注**: Opus 4.7 的多模态提升很大一部分来自更高分辨率输入，尤其是科学图表和 GUI grounding。这里的“multimodal”是理解图像与使用工具，不是图像生成。

## 真实专业任务、多语言、生命科学

- **OfficeQA / OfficeQA Pro**: 86.3% / 80.6%。
- **Finance Agent**: 64.4%。
- **MCP-Atlas**: 77.3%，扩展工具预算下约 79.5%。
- **VendingBench**: Max effort 最终余额 $10,937，高于 Opus 4.6 先前 SOTA $8,018。
- **GDPval-AA**: 领先 GPT-5.4 xhigh 约 79 ELO，约 61.2% pairwise win rate。
- **ARC-AGI-2**: 75.83%。
- **GMMLU / MILU / INCLUDE**: 89.9% / 89.9% / 87.0%。低资源非洲语言提升明显，GMMLU low-resource average 从 78.4% 到 86.2%。
- **Life sciences**: BioPipelineBench Verified 83.6%，BioMysteryBench Verified 78.9%，Structural biology MC 98.3% / open-ended 74.0%，Organic chemistry 77.2%，Phylogenetics 79.6%，Protocol troubleshooting 51.8%。

> 💡 **能力边界批注**: 生命科学能力提升和 CB 风险评估要一起读：同样的生物/化学能力既能服务科研，也可能提高误用风险。系统卡把 beneficial evaluations 和 CB threat model 分开呈现，但二者共享底层能力。

## Section 总结

Opus 4.7 的能力画像是「强编码 + 强工具 + 强专业任务 + 高分辨率多模态 + 长上下文/搜索可用」。最需要复测的是搜索和长期代理任务，因为分数对 token 上限、compaction、工具环境和 judge 设置非常敏感。
