[← 返回 README](../README.md)

# 01 - RSP Evaluations

## 📌 预览

RSP 章节是系统卡的主干：Anthropic 判断 Opus 4.7 没有推进能力前沿，因此灾难性风险总体仍低。这里的重点不是「没有风险」，而是「相对 Mythos Preview 和最近 Risk Report，没有出现足以改变风险等级的新能力」。

## 风险评估流程

Anthropic 的 RSP 流程把证据分成几类：自动化能力评估、uplift trials、第三方专家红队、第三方评估、训练过程中的 snapshot 趋势。系统卡和 Risk Report 的分工是：Risk Report 给全模型组合和缓解措施的总体风险，system card 说明新模型如何改变或不改变该判断。

> 💡 **机制拆解**: RSP 判断是一个「能力 × 威胁模型 × 缓解措施」的三元判断。单个 benchmark 提升不必然导致风险升级，除非它对应到明确的威胁模型，并且已有 mitigation 不再足够。

## CB 风险：CB-1 与 CB-2

报告区分两类化学/生物风险：

- **CB-1**：帮助有基础技术背景的人制造/获取/部署已知化学或生物武器，可能造成严重灾难性伤害。
- **CB-2**：帮助具备一定资源和专家支持的团队开发远超既往灾难的新型化学/生物武器。

Anthropic 承认 Opus 4.7 能提供相关信息、节省专家时间，也能做跨域综合；因此部署强实时 classifier guards、豁免访问控制、bug bounty、threat intelligence、快速响应和模型权重安全控制。结论是 CB-1 风险「very low but not negligible」。对 CB-2，Anthropic 认为 Opus 4.7 弱于 Mythos Preview，未跨阈值。

> 💡 **ASL/RSP 批注**: 文中说这些 mitigations 等于或强于历史 ASL-3 protections。这里的重点是 Anthropic 不只靠模型拒绝，还靠实时分类器、防护豁免控制和安全响应链条控制已知/非新颖 CB 风险。

## 自动化 AI R&D 与能力前沿

自动化 R&D 威胁模型关注模型是否能完全自动化或显著加速顶尖研究团队在 AI、能源、机器人、武器等关键领域的工作。Anthropic 的判断是：Opus 4.7 位于 Opus 4.6 与 Mythos Preview 之间，未推进 capability frontier，也不像能替代大量资深 Research Scientists / Research Engineers。

报告还提到 AECI trajectory：Opus 4.7 落在 pre-Mythos Preview trend 上，是「non-frontier point」。这支撑了「不用重新升级 automated R&D 风险」的结论。

> 💡 **Q&A 批注记录**:
> - Q: 为什么 Mythos Preview 可以“界定”Opus 4.7 的 RSP 风险？
> - A: 因为系统卡声称 Mythos Preview 在所有相关轴上更强；如果更强的 Mythos Preview 已经在最近风险更新中被讨论，较弱的 Opus 4.7 通常不会带来更高上界。

## Section 总结

RSP 章节的核心结论是：Opus 4.7 是能力显著提升的公开模型，但不是 Anthropic 的新能力前沿。CB 风险低但非零，自动化 R&D threshold 未跨越，对齐风险总体很低。读者应注意，这些结论依赖 Anthropic 对 Mythos Preview 的既有评估和内部阈值操作化。
