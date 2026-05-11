[← 返回 README](../README.md)

# 01 - RSP Evaluations

## 📌 预览

RSP 部分是这张系统卡的治理骨架：它解释 Anthropic 如何把能力、安全阈值、Risk Report 和部署控制联系起来。

## RSP v3.0/v3.1 的变化

报告说 Claude Mythos Preview 是 Anthropic 在 **RSP v3.0** 框架下发布的第一张 system card；RSP v3.0 在 February 2026 采用，April 有较小 v3.1 更新。

旧版 RSP 更强调模型是否需要某个 **AI Safety Level (ASL)** 对应的 mitigations，因此系统卡常讨论 rule-in/rule-out 和二元 threshold。新版 RSP 仍然要求判断是否跨越阈值，但不再用 ASL 指代这些阈值；ASL 更多保留为“present risk mitigations clusters”的术语。

> 💡 **机制批注**: 这意味着评估叙事从“是否触发某个 ASL 档位”转成“证据如何改变总体风险判断”。对读者来说，重点不只是模型能不能做某事，还包括出现该能力的概率、部署面、监控与缓解措施。

## 风险结论

报告在 1.2.2 和第 2 节给出总体判断：Claude Mythos Preview 比最近 Risk Report 中最强的 Claude Opus 4.6 **significantly more capable**，但 Anthropic 认为 catastrophic risks overall remain low。

主要风险类别包括：

- **Non-novel chemical/biological weapons production**: 能力更强，但总体风险画像相似；mitigations 使风险 very low but not negligible。
- **Novel chemical/biological weapons production**: 即使一般发布，catastrophic risk 也被判断为 low，但有 substantial uncertainty。
- **Risks from misaligned models**: overall risk very low，但高于此前模型，因此另有 supplementary alignment risk update。
- **Automated R&D in key domains**: 能力增长超过过往趋势，但 Anthropic 判断这些增长可归因于非 AI-accelerated R&D 因素，且模型没有跨过 RSP automated AI R&D capability threshold。

## CB 与 Autonomy

CB evaluation 围绕 chemical/biological threat models，包括 expert red teaming、virology protocol uplift、catastrophic biology scenario uplift、自动化 CB-1/CB-2 相关评估。报告中 virology protocol uplift trial 显示 Mythos-assisted 组平均协议质量优于 Opus/control，但最终风险结论仍依赖 mitigation 和 threat model 解释。

Autonomy 部分把重点放在 automated R&D、research scientist/engineer 替代能力、task-based evaluations、internal survey 和外部 METR/Epoch AI 测试。报告判断 Autonomy threat model 1 applicable，但 Autonomy threat model 2 not applicable；模型尚不能以 competitive cost 替代 Anthropic 的整个 Research Scientists/Research Engineers 集合。

> 💡 **Q&A 批注记录**:
> - Q: 为什么能力强还说没有跨过 automated AI R&D threshold？
> - A: 报告把“在很多任务上非常强”和“能整体替代顶级研究团队推进关键领域 R&D”区分开。Mythos Preview 能重新发现未发表 ML research task 中 4/5 个关键 insight，但仍有研究规划、判断、长期推进和可靠性缺口。

## 小结

- RSP 章节不是单纯 safety checklist，而是 Anthropic 对“能力跃迁但有限部署”的制度解释。
- 这张系统卡弱化旧 ASL 二元叙事，强调 risk reports、thresholds、threat models、mitigations 和 release decision 的组合。
- 关键判断：风险高于此前模型但总体仍低，且不一般发布是主动部署选择，不是 RSP 自动禁止。
