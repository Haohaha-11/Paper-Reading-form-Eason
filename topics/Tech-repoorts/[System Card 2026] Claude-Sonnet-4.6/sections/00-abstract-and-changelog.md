[← 返回 README](../README.md)

# 00 - Abstract and Changelog

## 📌 预览

这一节把 system card 的结论先压缩成三句话：Sonnet 4.6 是 Anthropic 最新 Sonnet 大模型；能力相对 Sonnet 4.5 大幅提升并在若干评估接近 Opus 4.6；安全上以 ASL-3 部署。Changelog 只更新 BrowseComp 分数，说明 Anthropic 后续用更强 cheating detection pipeline 重新标记了泄漏/非预期解法。

> 💡 **读法提示**: 这不是常规论文摘要，而是发布审计摘要。最关键不是“提出了什么方法”，而是“为什么可以发布、按什么 safety level 发布、哪些能力接近风险阈值”。

## 摘要要点

- 报告覆盖 coding、agentic、reasoning、multimodal、computer use、math、finance、cybersecurity、life sciences。
- Alignment assessment 覆盖隐藏目标、误用协作、deceptive/unfaithful scratchpad、sycophancy、sabotage safeguards、sandbagging、操纵用户等。
- 最终部署结论：Claude Sonnet 4.6 按 AI Safety Level 3 (ASL-3) Standard 部署。

## Changelog 批读

March 6, 2026 更新集中在 BrowseComp：

| 项 | 原始值 | 更新后 | 原因 |
|---|---:|---:|---|
| Single-agent BrowseComp | 74.72% | 74.01% | 改进 cheating detection pipeline 额外标记 9 个 unintended solutions |
| Multi-agent BrowseComp | 82.62% | 82.07% | 额外标记 11 个 unintended solutions；移除泄漏后 5/11 仍正确 |

> 💡 **Changelog 批读**: 这类更新说明 agentic search benchmark 容易受“在线答案泄漏/非预期解法”影响。对读者而言，BrowseComp 的数字应理解为强检索能力信号，但不是完全干净的 reasoning-only 能力证据。

## 本卡的核心判断

1. Sonnet 4.6 的能力强于 Sonnet 4.5，在部分能力上接近或超过 Opus 4.6。
2. 安全表现总体改善，misalignment 低，但 GUI/agentic 场景存在更难控制的边缘行为。
3. RSP 判定是 ASL-3；未跨过 AI R&D-4 或 CBRN-4，但报告承认干净排除这些阈值正在变难。

## 🔖 Section 总结

| 关注点 | 结论 |
|---|---|
| 发布状态 | February 2026 system card，March 6, 2026 BrowseComp 更新 |
| ASL | ASL-3 |
| 报告性质 | 公司官方 system card，偏模型发布审计 |
| 主要风险读法 | 不只看拒答率，还要看 agentic scaffolding、context compaction、GUI/browser/coding 表面 |
