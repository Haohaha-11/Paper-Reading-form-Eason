[← 返回 README](../README.md)

# 00 - 摘要与定位

## 📌 预览

这一部分回答两个问题：Mythos Preview 是什么，以及为什么这张系统卡和常规 Claude release system card 不同。

## 核心原文信息

报告首页标注为 **System Card: Claude Mythos Preview**，日期是 **April 7, 2026**。后续 changelog 在 April 8、April 13、April 14 有若干修订，包括模型名 typo、alignment/cyber/model welfare/capability 表述和 benchmark 分数修正。

Abstract 将 Claude Mythos Preview 描述为 Anthropic 的 large language model，也是当时“most capable frontier model to date”。它相对 Claude Opus 4.6 在许多 benchmark 上出现 striking leap。系统卡覆盖的范围包括 Responsible Scaling Policy、Frontier Compliance Framework、cybersecurity、alignment assessment、model welfare assessment，以及一个新的 qualitative impressions section。

最重要的部署信息是：Anthropic 决定不把 Claude Mythos Preview 一般可用化，而是把它用于一个 defensive cybersecurity program，并只开放给少量 partners。报告明确说这些发现会反馈到后续 Claude models 和 safeguards。

## 批读

> 💡 **定位批注**: “Preview”不是一个轻量 demo 的意思。这里的 Preview 更接近“受限部署的前沿模型快照”：能力足够强，需要公开系统卡解释风险，但访问边界比普通产品模型更窄。

> 💡 **和 Opus 4.6 的差异**: 报告多次把 Opus 4.6 当成最近的 frontier baseline。Mythos Preview 不只是 benchmark 小幅提升，而是在 cyber exploitation、coding agent、long context、search、多模态工具使用和数学证明上形成跨领域跃迁。

> 💡 **发布决策批注**: 1.2 节说这是 Anthropic 首次发布系统卡但不让模型一般商用。这个决定不是 RSP requirement 直接禁止，而是 Anthropic 对 dual-use cyber capability 的商业/安全判断。

## 训练与流程

报告描述模型训练使用 proprietary mix，包括 publicly available information、第三方数据、用户/worker 数据和内部生成数据。pretraining 后进行了 substantial post-training，并在训练过程中进行 iterative evaluations。报告还提到 crowd workers、usage policy/support、external testing。

早期训练信号显示模型能力很强，Anthropic 因此第一次安排了 **24-hour internal alignment review**，在模型广泛内部部署前先确认不会在内部基础设施交互中造成破坏。通过 review 后，早期版本在 **February 24, 2026** 开始内部使用。

## 小结

- Mythos Preview 是 April 2026 的 frontier model system card，不是普通产品说明。
- 模型能力显著高于 Claude Opus 4.6，但正因为能力跃迁，访问被限制到 cyber defense partners。
- 报告主线是“能力跃迁如何改变部署边界”，而不是只证明模型更聪明。
